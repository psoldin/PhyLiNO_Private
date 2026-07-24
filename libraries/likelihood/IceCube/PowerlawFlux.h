#pragma once

#include "../../io/IceCube/ICConstants.h"
#include "../../io/IceCube/ICSample.h"
#include "../ParameterWrapper.h"
#include "MetalBackend.h"

#include <array>
#include <memory>
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
   * Recalculates when AstroNorm or SpectralIndex changed. When a MetalBackend is
   * supplied the per-event loop runs on the GPU; otherwise the CPU OMP+SIMD path
   * is used (and serves as the validation oracle).
   */
  class PowerlawFlux {
   public:
    PowerlawFlux(const io::ic::ICSample&        sample,
                 double                         e_ref_gev,
                 double                         reference_index,
                 bool                           per_type_norm,
                 std::shared_ptr<MetalBackend>  metal          = nullptr,
                 bool                           need_per_event = false);
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
    bool                    m_NeedPerEvent;
    BinArray                m_Histogram{};
    std::vector<double>     m_PerEventWeight;

    // Non-null when the Metal backend is selected; shared with the other flux
    // components (so e_true / bin_offsets are uploaded once).
    std::shared_ptr<MetalBackend> m_Metal;
    int                           m_hETrue    = -1;
    int                           m_hBaseline = -1;
    int                           m_hOffsets  = -1;
    int                           m_hHist     = -1;
    int                           m_hPerEvent = -1;

    void recalculate(const ParameterWrapper& parameter) noexcept;
  };

}  // namespace ana::ic
