#pragma once

#include "../../io/IceCube/ICConstants.h"
#include "../../io/IceCube/ICSample.h"
#include "../ParameterWrapper.h"

#include <array>
#include <span>
#include <vector>

namespace ana::ic {

  /**
   * Astrophysical single power-law flux (NNMFit Powerlaw + SpectralIndex):
   *
   *   astro_i = astro_baseline_i * eff_norm * (E_true_i / E_ref)^(ref_index - gamma)
   *
   * where gamma = parameter[SpectralIndex] is the fitted spectral index and
   * eff_norm = per_type_norm ? AstroNorm : 0.5 * AstroNorm  (NNMFit Norm with
   * per_type_norm=false halves the per-particle-type normalization).
   *
   * astro_baseline_i is the precomputed per-event "powerlaw" weight from the MC.
   * Recalculates when AstroNorm or SpectralIndex changed.
   */
  class PowerlawFlux {
   public:
    PowerlawFlux(const io::ic::ICSample& sample,
                 double                  e_ref_gev,
                 double                  reference_index,
                 bool                    per_type_norm);
    ~PowerlawFlux() = default;

    bool check_and_recalculate(const ParameterWrapper& parameter);

    [[nodiscard]] std::span<const double> histogram() const noexcept {
      return m_Histogram;
    }

    // Per-event weight, same value already summed into m_Histogram, needed by
    // ICLikelihood to build the combined ssq histogram for the SAY likelihood.
    // Indexed the same way as io::ic::ICSample (CSR bin order).
    [[nodiscard]] std::span<const double> per_event_weight() const noexcept {
      return m_PerEventWeight;
    }

   private:
    using BinArray = std::array<double, io::ic::Constants::nBins>;

    const io::ic::ICSample& m_Sample;
    double                  m_ERef;
    double                  m_ReferenceIndex;
    bool                    m_PerTypeNorm;
    BinArray                m_Histogram{};
    std::vector<double>     m_PerEventWeight;

    void recalculate(const ParameterWrapper& parameter) noexcept;
  };

}  // namespace ana::ic
