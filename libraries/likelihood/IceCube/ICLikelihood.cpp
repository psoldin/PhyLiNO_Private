#include "ICLikelihood.h"

#include "../../io/IceCube/ICParameter.h"
#include "../../io/Options.h"
#include "CudaBackend.h"
#include "MetalBackend.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace ana::ic {

  namespace {

    // One shared GPU backend for all samples' flux components; nullptr => CPU path. A
    // requested GPU backend falls back to CPU if no matching device is present.
    std::shared_ptr<GpuBackend> make_gpu_backend(const io::ic::BackendKind kind) {
      switch (kind) {
        case io::ic::BackendKind::Cpu:
          return nullptr;
        case io::ic::BackendKind::Metal:
          if (!MetalBackend::available()) {
            std::cout << "ICLikelihood: Metal backend requested but no device available; using CPU\n";
            return nullptr;
          }
          return std::make_shared<MetalBackend>();
        case io::ic::BackendKind::Cuda:
          if (!CudaBackend::available()) {
            std::cout << "ICLikelihood: CUDA backend requested but no device available; using CPU\n";
            return nullptr;
          }
          return std::make_shared<CudaBackend>();
      }
      return nullptr;
    }

  }  // namespace

  ICLikelihood::ICLikelihood(std::shared_ptr<io::Options>              options,
                             std::shared_ptr<const io::ic::ICDataBase> data_base,
                             const io::ic::ICInputOptions&             input_options)
    : Likelihood(std::move(options), params::ic::number_of_parameters())
    , m_DataBase(std::move(data_base))
    , m_GpuBackend(make_gpu_backend(input_options.backend_kind())) {
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
    };

    // ICDataBase loaded the enabled configs in config order, using the same
    // enabled_sample_indices() helper, so the k-th loaded ICSample belongs to
    // the k-th enabled SampleConfig. ICDataBase::sample(k) is an unchecked
    // noexcept accessor, so verify the counts agree before indexing: a
    // divergence would otherwise be silent UB rather than an error.
    const auto enabled = io::ic::enabled_sample_indices(input_options.samples());
    if (enabled.size() != m_DataBase->n_samples())
      throw std::runtime_error(
          "ICLikelihood: enabled-sample count (" + std::to_string(enabled.size()) +
          ") does not match the loaded sample count (" + std::to_string(m_DataBase->n_samples()) + ")");

    for (std::size_t k = 0; k < enabled.size(); ++k) {
      const io::ic::SampleConfig& cfg = input_options.samples()[enabled[k]];
      std::cout << "ICLikelihood: sample '" << cfg.name << "' with "
                << cfg.binning.total_bins() << " bins, components:";
      for (const auto& c : cfg.components) std::cout << ' ' << c;
      std::cout << '\n';
      m_Samples.push_back(std::make_unique<SampleLikelihood>(
          m_DataBase->sample(k), cfg, settings, m_GpuBackend, use_say));
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
    for (std::size_t i = 0; i < nominal.size(); ++i)
      nominal[i] = parameters[i].value();

    m_Parameter.reset_parameter(nominal.data());

    double total = 0.0;
    for (std::size_t k = 0; k < m_Samples.size(); ++k) {
      // The prediction at the nominal point is needed either way: as the Asimov
      // expectation, or to seed the SAY ssq before the measured counts replace it.
      m_Samples[k]->generate_asimov(m_Parameter);
      if (use_data) {
        const auto& counts = m_DataBase->data_histogram(k);
        if (counts.empty())
          throw std::runtime_error("ICLikelihood: UseData is true but sample " + std::to_string(k) +
                                   " has no \"data\" path in its config");
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

    double llh = 0.0;
    for (auto& s : m_Samples)
      llh += s->partial_llh(m_Parameter);
    llh += calculate_pulls(m_Parameter);

    if (m_FirstCall) {
      if (std::isfinite(llh)) {
        m_LLHBaseLine = llh;
        m_FirstCall = false;
      }
    }

    llh -= m_LLHBaseLine;

    return std::isfinite(llh) ? llh : 1.0e25;
  }

}  // namespace ana::ic
