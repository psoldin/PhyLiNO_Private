#include "ICLikelihood.h"

#include "../../io/IceCube/ICParameter.h"
#include "../../io/Options.h"
#include "SAYLikelihood.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace ana::ic {

  ICLikelihood::ICLikelihood(std::shared_ptr<io::Options>              options,
                             std::shared_ptr<const io::ic::ICDataBase> data_base,
                             const io::ic::ICInputOptions&             input_options)
    : Likelihood(std::move(options), params::ic::number_of_parameters())
    , m_DataBase(std::move(data_base))
    , m_Astro(m_DataBase->sample(),
              input_options.e_ref_gev(),
              input_options.astro_reference_index(),
              input_options.astro_per_type_norm(),
              input_options.use_metal_backend(),
              input_options.likelihood_type() == io::ic::LikelihoodType::SAY)
    , m_Atmo(m_DataBase->sample(),
             input_options.conv_delta_gamma_e_ref(),
             input_options.prompt_delta_gamma_e_ref())
    , m_UseSAY(input_options.likelihood_type() == io::ic::LikelihoodType::SAY) {
    if (input_options.use_oscillation())
      std::cout << "ICLikelihood: OscillationsHook configured but not yet implemented; "
                   "atmospheric baselines are used un-oscillated (no-op).\n";
    std::cout << "ICLikelihood: using " << (m_UseSAY ? "SAY" : "Poisson") << " likelihood\n";
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
                  << " CV=" << parameters[i].value()
                  << " sigma=" << parameters[i].uncertainty() << '\n';
        m_Pulls.emplace_back(static_cast<int>(i),
                             parameters[i].value(),
                             parameters[i].uncertainty());
      }
    }
  }

  bool ICLikelihood::assemble_prediction(BinArray& out) {
    const bool astro_changed = m_Astro.check_and_recalculate(m_Parameter);
    const bool atmo_changed  = m_Atmo.check_and_recalculate(m_Parameter);

    const std::span<const double> astro = m_Astro.histogram();
    const std::span<const double> atmo  = m_Atmo.histogram();

    for (int b = 0; b < io::ic::Constants::nBins; ++b) {
      // Clip at zero to avoid unphysical predictions (matches NNMFit mu clip).
      out[b] = std::max(0.0, astro[b] + atmo[b]);
    }

    return astro_changed || atmo_changed;
  }

  template <typename T>
  T square(T&& t) noexcept {
    return t * t;
  }

  void ICLikelihood::assemble_fluctuation(BinArray& out) {
    // Combined per-event weight across astro+conv+prompt, squared, then binned
    // -- matches NNMFit's rule that same-event components must be summed
    // *before* squaring (NNMFit/core/histogram_builder.py:229-329), since
    // (w1+w2)^2 != w1^2 + w2^2.
    const std::span<const double> astro = m_Astro.per_event_weight();
    const std::span<const double> atmo  = m_Atmo.per_event_weight();
    const auto&                   off   = m_DataBase->sample().bin_offsets;

#pragma omp parallel for
    for (int b = 0; b < io::ic::Constants::nBins; ++b) {
      double acc = 0.0;
#pragma omp simd reduction(+ : acc)
      for (std::size_t i = off[b]; i < off[b + 1]; ++i) {
        acc += square(astro[i] + atmo[i]);
      }
      out[b] = acc;
    }
  }

  void ICLikelihood::initialize_data(const bool use_data) {
    if (use_data)
      throw std::runtime_error("ICLikelihood: reading real data not yet implemented");
    generate_asimov_data();
  }

  void ICLikelihood::generate_asimov_data() {
    const auto&         parameters = m_Options->inputOptions().input_parameters().parameters();
    std::vector<double> nominal(params::ic::number_of_parameters());
    for (std::size_t i = 0; i < nominal.size(); ++i)
      nominal[i] = parameters[i].value();

    m_Parameter.reset_parameter(nominal.data());
    assemble_prediction(m_Data);

    double total = 0.0;
    for (int b = 0; b < io::ic::Constants::nBins; ++b)
      total += m_Data[b];
    std::cout << "IC Asimov total events: " << total << '\n';
  }

  double ICLikelihood::calculate_pulls(const ParameterWrapper& parameter) const noexcept {
    double result = 0.0;
    for (const auto& [idx, cv, sigma] : m_Pulls) {
      result += square((parameter[idx] - cv) / sigma);
    }
    return result;
  }

  static double calculate_poisson_likelihood(const std::span<const double> data,
                                             const std::span<const double> prediction) noexcept {
    double llh = 0.0;
    for (std::size_t i = 0, n = data.size(); i < n; ++i) {
      const double model = prediction[i];
      const double obs   = data[i];
      if (model <= 0.0)
        continue;
      llh += obs * std::log(model) - model;
    }
    return -2.0 * llh;
  }

  static double calculate_say_likelihood(const std::span<const double> data,
                                         const std::span<const double> prediction,
                                         const std::span<const double> ssq) noexcept {
    double llh = 0.0;
    for (std::size_t i = 0, n = data.size(); i < n; ++i) {
      llh += say_bin_log_likelihood(data[i], prediction[i], ssq[i]);
    }
    return -2.0 * llh;
  }

  double ICLikelihood::calculate_likelihood(const double* parameter) {
    m_Parameter.reset_parameter(parameter);
    const bool changed = assemble_prediction(m_TotalPredicted);

    double llh;
    if (m_UseSAY) {
      if (changed)
        assemble_fluctuation(m_Ssq);
      llh = calculate_say_likelihood(m_Data, m_TotalPredicted, m_Ssq) + calculate_pulls(m_Parameter);
    } else {
      llh = calculate_poisson_likelihood(m_Data, m_TotalPredicted) + calculate_pulls(m_Parameter);
    }

    return std::isfinite(llh) ? llh : 1.0e25;
  }

}  // namespace ana::ic
