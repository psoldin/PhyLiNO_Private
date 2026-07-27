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
    , m_Config(cfg)
    , m_UseSAY(use_say) {
    if (cfg.wants_astro())
      m_Astro.emplace(sample,
                      cfg.binning,
                      settings.e_ref_gev,
                      settings.astro_reference_index,
                      settings.astro_per_type_norm,
                      gpu,
                      use_say);

    if (cfg.wants_atmospheric())
      m_Atmo.emplace(sample,
                     cfg.binning,
                     settings.conv_delta_gamma_e_ref,
                     settings.prompt_delta_gamma_e_ref,
                     gpu,
                     use_say,
                     cfg.wants_veto(),
                     settings.veto_anchor_energy,
                     settings.veto_rescale_energy);

    if (cfg.wants_template())
      m_Template.emplace(cfg.binning, cfg.template_file, cfg.template_norm_index, cfg.livetime);

    const int total_bins = cfg.binning.total_bins();
    m_Predicted.assign(total_bins, 0.0);
    m_Data.assign(total_bins, 0.0);
    m_Ssq.assign(total_bins, 0.0);
  }

  bool SampleLikelihood::assemble_prediction(const ParameterWrapper& parameter) {
    bool changed = false;
    if (m_Astro) changed |= m_Astro->check_and_recalculate(parameter);
    if (m_Atmo) changed |= m_Atmo->check_and_recalculate(parameter);
    if (m_Template) changed |= m_Template->check_and_recalculate(parameter);

    const std::span<const double> astro = m_Astro ? m_Astro->histogram() : std::span<const double>{};
    const std::span<const double> atmo  = m_Atmo ? m_Atmo->histogram() : std::span<const double>{};
    const std::span<const double> tmpl  = m_Template ? m_Template->histogram() : std::span<const double>{};

    for (std::size_t b = 0, n = m_Predicted.size(); b < n; ++b) {
      double total = 0.0;
      if (!astro.empty()) total += astro[b];
      if (!atmo.empty()) total += atmo[b];
      if (!tmpl.empty()) total += tmpl[b];
      // Clip at zero to avoid unphysical predictions (matches NNMFit mu clip).
      m_Predicted[b] = std::max(0.0, total);
    }

    return changed;
  }

  void SampleLikelihood::assemble_fluctuation() {
    // Combined per-event weight across astro+conv+prompt, squared, then binned
    // -- matches NNMFit's rule that same-event components must be summed
    // *before* squaring (NNMFit/core/histogram_builder.py:229-329), since
    // (w1+w2)^2 != w1^2 + w2^2.
    const std::span<const double> astro = m_Astro ? m_Astro->per_event_weight() : std::span<const double>{};
    const std::span<const double> atmo  = m_Atmo ? m_Atmo->per_event_weight() : std::span<const double>{};
    const auto&                   off   = m_Sample.bin_offsets;

    const int n_bins = static_cast<int>(m_Predicted.size());

    // The component test is hoisted out of the per-event loop: which components
    // exist is fixed at construction, so each case gets its own tight loop.
    auto accumulate = [&](auto event_weight) {
#pragma omp parallel for
      for (int b = 0; b < n_bins; ++b) {
        double acc = 0.0;
#pragma omp simd reduction(+ : acc)
        for (std::size_t i = off[b]; i < off[b + 1]; ++i) {
          acc += square(event_weight(i));
        }
        m_Ssq[b] = acc;
      }
    };

    if (!astro.empty() && !atmo.empty())
      accumulate([&](const std::size_t i) { return astro[i] + atmo[i]; });
    else if (!astro.empty())
      accumulate([&](const std::size_t i) { return astro[i]; });
    else if (!atmo.empty())
      accumulate([&](const std::size_t i) { return atmo[i]; });
    else
      std::ranges::fill(m_Ssq, 0.0);

    // Histogram-level fluctuation from the template component, added after the
    // per-event sum (NNMFit: ssq += (hist_fluctuation * livetime)**2).
    if (m_Template) {
      const std::span<const double> tmpl_ssq = m_Template->fluctuation();
      for (int b = 0; b < n_bins; ++b) m_Ssq[b] += tmpl_ssq[b];
    }
  }

  void SampleLikelihood::generate_asimov(const ParameterWrapper& nominal) {
    assemble_prediction(nominal);
    std::ranges::copy(m_Predicted, m_Data.begin());

    // Seed the ssq histogram at the nominal point. partial_llh() only refreshes
    // it when a flux actually recalculated, and the minimizer's first evaluation
    // is at the start values -- which compare equal to the nominal set here, so
    // nothing would recalculate and SAY would run that evaluation with ssq == 0
    // (silently degenerating to plain Poisson).
    if (m_UseSAY)
      assemble_fluctuation();
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
