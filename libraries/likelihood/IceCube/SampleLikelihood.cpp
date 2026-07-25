#include "SampleLikelihood.h"

#include "SAYLikelihood.h"

#include <algorithm>
#include <cmath>

namespace ana::ic {

  namespace {

    template <typename T>
    T square(T&& t) noexcept {
      return t * t;
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

  }  // namespace

  SampleLikelihood::SampleLikelihood(const io::ic::ICSample&     sample,
                                     const io::ic::SampleConfig& cfg,
                                     const GlobalFluxSettings&   settings,
                                     std::shared_ptr<GpuBackend> gpu,
                                     const bool                  use_say)
    : m_Sample(sample)
    , m_UseSAY(use_say)
    , m_Astro(sample,
              cfg.binning,
              settings.e_ref_gev,
              settings.astro_reference_index,
              settings.astro_per_type_norm,
              gpu,
              use_say)
    , m_Atmo(sample,
             cfg.binning,
             settings.conv_delta_gamma_e_ref,
             settings.prompt_delta_gamma_e_ref,
             gpu,
             use_say) {
    const int total_bins = cfg.binning.total_bins();
    m_Predicted.assign(total_bins, 0.0);
    m_Data.assign(total_bins, 0.0);
    m_Ssq.assign(total_bins, 0.0);
  }

  bool SampleLikelihood::assemble_prediction(const ParameterWrapper& parameter) {
    const bool astro_changed = m_Astro.check_and_recalculate(parameter);
    const bool atmo_changed  = m_Atmo.check_and_recalculate(parameter);

    const std::span<const double> astro = m_Astro.histogram();
    const std::span<const double> atmo  = m_Atmo.histogram();

    for (std::size_t b = 0, n = m_Predicted.size(); b < n; ++b) {
      // Clip at zero to avoid unphysical predictions (matches NNMFit mu clip).
      m_Predicted[b] = std::max(0.0, astro[b] + atmo[b]);
    }

    return astro_changed || atmo_changed;
  }

  void SampleLikelihood::assemble_fluctuation() {
    // Combined per-event weight across astro+conv+prompt, squared, then binned
    // -- matches NNMFit's rule that same-event components must be summed
    // *before* squaring (NNMFit/core/histogram_builder.py:229-329), since
    // (w1+w2)^2 != w1^2 + w2^2.
    const std::span<const double> astro = m_Astro.per_event_weight();
    const std::span<const double> atmo  = m_Atmo.per_event_weight();
    const auto&                   off   = m_Sample.bin_offsets;

    const int n_bins = static_cast<int>(m_Predicted.size());

#pragma omp parallel for
    for (int b = 0; b < n_bins; ++b) {
      double acc = 0.0;
#pragma omp simd reduction(+ : acc)
      for (std::size_t i = off[b]; i < off[b + 1]; ++i) {
        acc += square(astro[i] + atmo[i]);
      }
      m_Ssq[b] = acc;
    }
  }

  void SampleLikelihood::generate_asimov(const ParameterWrapper& nominal) {
    assemble_prediction(nominal);
    std::ranges::copy(m_Predicted, m_Data.begin());
  }

  double SampleLikelihood::partial_llh(const ParameterWrapper& parameter) {
    const bool changed = assemble_prediction(parameter);

    if (m_UseSAY) {
      if (changed)
        assemble_fluctuation();
      return calculate_say_likelihood(m_Data, m_Predicted, m_Ssq);
    }

    return calculate_poisson_likelihood(m_Data, m_Predicted);
  }

}  // namespace ana::ic
