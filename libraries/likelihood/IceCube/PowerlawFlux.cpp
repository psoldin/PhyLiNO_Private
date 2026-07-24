#include "PowerlawFlux.h"

#include "../../io/IceCube/ICParameter.h"

#include <cmath>
#include <iostream>

namespace ana::ic {

  PowerlawFlux::PowerlawFlux(const io::ic::ICSample& sample,
                             const double            e_ref_gev,
                             const double            reference_index,
                             const bool              per_type_norm,
                             const bool              use_metal,
                             const bool              need_per_event)
    : m_Sample(sample)
    , m_ERef(e_ref_gev)
    , m_ReferenceIndex(reference_index)
    , m_PerTypeNorm(per_type_norm)
    , m_NeedPerEvent(need_per_event) {
    m_Histogram.fill(0.0);
    m_PerEventWeight.assign(sample.size(), 0.0);

    if (use_metal) {
      if (ICMetalPowerlaw::available()) {
        m_Metal = std::make_unique<ICMetalPowerlaw>(sample, e_ref_gev, reference_index, per_type_norm);
        std::cout << "PowerlawFlux: using Metal GPU backend\n";
      } else {
        std::cout << "PowerlawFlux: Metal backend requested but no device available; using CPU\n";
      }
    }
  }

  void PowerlawFlux::recalculate(const ParameterWrapper& parameter) noexcept {
    using namespace params::ic;

    const double norm  = parameter[AstroNorm];
    const double gamma = parameter[SpectralIndex];

    if (m_Metal) {
      m_Metal->recalculate(norm, gamma, m_NeedPerEvent);
      const float* hist = m_Metal->histogram();
      for (int bin = 0; bin < io::ic::Constants::nBins; ++bin)
        m_Histogram[bin] = static_cast<double>(hist[bin]);
      if (m_NeedPerEvent) {
        const float* pe = m_Metal->per_event_weight();
        for (std::size_t i = 0, n = m_PerEventWeight.size(); i < n; ++i)
          m_PerEventWeight[i] = static_cast<double>(pe[i]);
      }
      return;
    }

    // NNMFit Norm with per_type_norm=false halves the per-type normalization.
    const double eff_norm = m_PerTypeNorm ? norm : 0.5 * norm;
    // (E/E_ref)^(ref_index - gamma), baseline is already ~E^(-ref_index).
    const double exponent = m_ReferenceIndex - gamma;

    const auto& off      = m_Sample.bin_offsets;
    const auto& baseline = m_Sample.astro_baseline;
    const auto& e_true   = m_Sample.e_true;

    #pragma omp parallel for
    for (int bin = 0; bin < io::ic::Constants::nBins; ++bin) {
      double acc = 0.0;
      for (std::size_t i = off[bin], n = off[bin + 1]; i < n; ++i) {
        const double w = baseline[i] * eff_norm * std::pow(e_true[i] / m_ERef, exponent);
        acc += w;
        m_PerEventWeight[i] = w;
      }
      m_Histogram[bin] = acc;
    }
  }

  bool PowerlawFlux::check_and_recalculate(const ParameterWrapper& parameter) {
    using namespace params::ic;
    const bool changed = parameter.check_parameter_changed(AstroNorm) || parameter.check_parameter_changed(SpectralIndex);
    if (changed)
      recalculate(parameter);
    return changed;
  }

}  // namespace ana::ic
