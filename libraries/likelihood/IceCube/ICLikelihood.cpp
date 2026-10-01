#include "ICLikelihood.h"

#include "../../io/IceCube/ICParameter.h"
#include "../../io/Options.h"

#include <cmath>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ana::ic {

  ICLikelihood::ICLikelihood(std::shared_ptr<io::Options>              options,
                             std::shared_ptr<const io::ic::ICDataBase> data_base,
                             const io::ic::ICInputOptions&             input_options,
                             std::shared_ptr<GpuBackend>               gpu_backend)
    : Likelihood(std::move(options), params::ic::number_of_parameters())
    , m_DataBase(std::move(data_base))
    , m_GpuBackend(std::move(gpu_backend))
    , m_UseMultiThreading(m_Options->inputOptions().use_multi_threading()) {
    const bool use_say = input_options.likelihood_type() == io::ic::LikelihoodType::SAY;
    std::cout << "ICLikelihood: using " << (use_say ? "SAY" : "Poisson") << " likelihood\n";

    const GlobalFluxSettings settings{
        .e_ref_gev                = input_options.e_ref_gev(),
        .astro_reference_index    = input_options.astro_reference_index(),
        .conv_delta_gamma_e_ref   = input_options.conv_delta_gamma_e_ref(),
        .prompt_delta_gamma_e_ref = input_options.prompt_delta_gamma_e_ref(),
        .astro_per_type_norm      = input_options.astro_per_type_norm(),
        .veto_anchor_energy       = input_options.veto_anchor_energy(),
        .veto_rescale_energy      = input_options.veto_rescale_energy(),
        .astro_model              = input_options.astro_model(),
        .use_multi_threading      = m_UseMultiThreading,
    };

    // ICDataBase loaded the enabled configs in config order, using the same
    // enabled_sample_indices() helper, so the k-th loaded ICSample belongs to
    // the k-th enabled SampleConfig. ICDataBase::sample(k) is an unchecked
    // noexcept accessor, so verify the counts agree before indexing: a
    // divergence would otherwise be silent UB rather than an error.
    const auto enabled = io::ic::enabled_sample_indices(input_options.samples());
    if (enabled.size() != m_DataBase->n_samples())
      throw std::runtime_error(
          "ICLikelihood: enabled-sample count (" + std::to_string(enabled.size()) + ") does not match the loaded sample count (" + std::to_string(m_DataBase->n_samples()) + ")");

    for (std::size_t k = 0; k < enabled.size(); ++k) {
      const io::ic::SampleConfig& cfg = input_options.samples()[enabled[k]];
      std::cout << "ICLikelihood: sample '" << cfg.name << "' with "
                << cfg.binning.total_bins() << " bins, components:";
      for (const auto& c : cfg.components)
        std::cout << ' ' << c;
      std::cout << '\n';
      // One session per sample, not one per fit: calculate_likelihood already
      // evaluates samples concurrently, so a shared session would serialise them
      // on its single command stream. The device, the compiled kernels and the
      // uploaded MC columns stay shared behind the sessions.
      m_Samples.push_back(std::make_unique<SampleLikelihood>(
          m_DataBase->sample(k), cfg, settings,
          m_GpuBackend ? m_GpuBackend->create_session() : nullptr, use_say));
    }
    if (m_Samples.empty())
      throw std::runtime_error("ICLikelihood: no enabled IceCube samples");

    setup_pulls();
    initialize_data(input_options.use_data());
  }

  void ICLikelihood::setup_pulls() {
    const auto& input_params = m_Options->inputOptions().input_parameters();
    const auto& parameters   = input_params.parameters();
    const auto& constrained  = input_params.constrained();
    const auto& names        = input_params.names();

    for (std::size_t i = 0; i < parameters.size(); ++i) {
      if (constrained[i]) {
        std::cout << "IC pull: " << names[i]
                  << " CV=" << parameters[i].prior_value()
                  << " sigma=" << parameters[i].prior_width() << '\n';
        m_Pulls.emplace_back(static_cast<int>(i),
                             parameters[i].prior_value(),
                             parameters[i].prior_width());
      }
    }
  }

  void ICLikelihood::initialize_data(const bool use_data) {
    const auto&         parameters = m_Options->inputOptions().input_parameters().parameters();
    std::vector<double> nominal(params::ic::number_of_parameters());
    // asimov_value() is the start value unless the config separates the two
    // ("AsimovValue", NNMFit's analysis.input_params). Separating them is what
    // makes a fit start away from the truth it has to recover, and what lets a
    // fixed-point evaluation land somewhere the likelihood is not saturated.
    for (std::size_t i = 0; i < nominal.size(); ++i)
      nominal[i] = parameters[i].asimov_value();

    m_Parameter.reset_parameter(nominal.data());

    double total = 0.0;
    for (std::size_t k = 0; k < m_Samples.size(); ++k) {
      // The prediction at the nominal point is needed either way: as the Asimov
      // expectation, or to seed the SAY ssq before the measured counts replace it.
      m_Samples[k]->generate_asimov(m_Parameter);
      if (use_data) {
        const auto& counts = m_DataBase->data_histogram(k);
        if (counts.empty())
          throw std::runtime_error("ICLikelihood: UseData is true but sample " + std::to_string(k) + " has no \"data\" path in its config");
        m_Samples[k]->set_data(counts);
      }
      for (const double v : m_Samples[k]->data())
        total += v;
    }
    std::cout << "IC " << (use_data ? "data" : "Asimov") << " total events: " << total << '\n';
  }

  double ICLikelihood::calculate_pulls(const ParameterWrapper& parameter) const noexcept {
    double result = 0.0;
    for (const auto& [idx, cv, sigma] : m_Pulls) {
      const double deviation = (parameter[idx] - cv) / sigma;
      result += deviation * deviation;
    }
    return result;
  }

  double ICLikelihood::calculate_likelihood(const double* parameter) {
    m_Parameter.reset_parameter(parameter);

    // Samples are independent (own flux components, own histograms; m_Parameter
    // is only read after the reset above), so their partial_llh calls could run
    // concurrently. The shared GPU backend takes concurrent dispatches: its
    // pipeline/buffer maps are only mutated during construction and the Metal
    // command queue is thread-safe. The first sample runs inline on this thread
    // (so a single-sample fit spawns none) and the rest are joined in
    // sample-index order, keeping the result bit-identical to a sequential loop.
    //
    // DISABLED 2026-08-07 -- the cross-sample std::async path below is commented
    // out, not deleted. It is correct; it is simply not worth its cost.
    //
    //   Why: the work is not spread across samples. tracks has 267,300 analysis
    //   bins, cscd_cascade 2,646, cscd_muon 1 -- tracks alone is 99.0% of the
    //   total, so Amdahl caps cross-sample parallelism at ~1.01x no matter how
    //   many cores it gets. Measured on n25g0003 (24 dedicated cores, H100):
    //   this path fully active with OMP_NUM_THREADS=1 ran 40.09 s against a
    //   41.68 s serial baseline. It buys ~4%.
    //
    //   What it cost: it nests on top of the six `omp parallel for` regions
    //   underneath it (SampleLikelihood.cpp:166, 351, 404; PowerlawFlux.cpp:256,
    //   277; AtmosphericFlux.cpp:354), so every sample multiplied the OpenMP
    //   thread count. With `-m` and OMP_NUM_THREADS=16 the process peaked at 53
    //   OS threads on 24 cores, and libgomp's spin-wait barrier under that
    //   oversubscription produced reproducible slowdowns -- 72.5 s at
    //   OMP_NUM_THREADS=8, i.e. 0.57x, worse than running serially.
    //
    // With this path off, `-m` now means OpenMP only: one team, all cores, on
    // the 267,300-bin loop that actually holds the runtime. See section 10 of
    // OPTIMISATION_NOTES.txt for the full sweep.
    //
    // Re-enable only if the sample mix changes so that no single sample
    // dominates -- and if you do, either drop OMP_NUM_THREADS to ~cores/n_samples
    // or set OMP_WAIT_POLICY=passive, which removed the pathology in testing.
    double llh = 0.0;

    if (m_Samples.empty())
      throw std::runtime_error("ICLikelihood: No samples available");

    // if (m_UseMultiThreading) {
    //   std::vector<std::future<double>> partial;
    //   partial.reserve(m_Samples.size() - 1);
    //
    //   for (std::size_t i = 1, n = m_Samples.size(); i < n; ++i) {
    //     partial.push_back(
    //         std::async(std::launch::async, [this, &sample = m_Samples[i]] {
    //           return sample->partial_llh(m_Parameter);
    //         }));
    //   }
    //
    //   llh = m_Samples[0]->partial_llh(m_Parameter);
    //
    //   for (auto& f : partial)
    //     llh += f.get();
    // } else {
    for (const auto& sample : m_Samples)
      llh += sample->partial_llh(m_Parameter);
    // }

    // The pull term is -2 * NNMFit's log_prior: their Gaussian prior is
    // -((x - x0) / (sigma * sqrt(2)))^2 (LikelihoodBuilder.make_priors), so
    // multiplying by -2 gives exactly ((x - x0) / sigma)^2.
    llh += calculate_pulls(m_Parameter);

    // No baseline shift: both likelihood terms are absolutely normalised the
    // same way NNMFit normalises them (Poisson with the saturated term
    // subtracted, SAY unsubtracted), so this value is 2 * NNMFit's -llh --
    // PhyLiNO reports -2 log L (Minuit2 ErrorDef = 1), NNMFit reports -log L
    // (iminuit errordef = LIKELIHOOD = 0.5). Same minimum, same parameter
    // errors; only the printed scale differs.
    return std::isfinite(llh) ? llh : 1.0e25;
  }

}  // namespace ana::ic
