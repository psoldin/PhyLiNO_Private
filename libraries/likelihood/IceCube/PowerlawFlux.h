#pragma once

#include "../../io/IceCube/Binning.h"
#include "../../io/IceCube/ICInputOptions.h"  // io::ic::AstroModel
#include "../../io/IceCube/ICSample.h"
#include "../ParameterWrapper.h"
#include "GpuBackend.h"

#include <memory>
#include <span>
#include <vector>

namespace ana::ic {

  // The astrophysical spectral model this component evaluates, selected by the
  // IceCube.AstroModel config setting.
  using AstroModel = io::ic::AstroModel;

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
   *
   * With AstroModel::BrokenPowerlaw the component instead evaluates NNMFit's
   * AstroBPL (parameters/astroBPL.py), driven by AstroNorm, AstroGamma1,
   * AstroGamma2 and AstroEBreak (the last given as log10(E_break / GeV)):
   *
   *   E_break = 10^AstroEBreak
   *   pivot   = (1e5 < E_break) ? (1e5 / E_break)^gamma_1 : (1e5 / E_break)^gamma_2
   *   shape   = (E_true_i < E_break) ? (E_true_i / E_break)^(-gamma_1)
   *                                  : (E_true_i / E_break)^(-gamma_2)
   *   astro_i = astro_baseline_i * eff_norm * pivot * shape * (E_true_i / 1e5)^2
   *
   * `pivot` (positive exponent, unlike `shape`) renormalises the flux so that
   * AstroNorm is its value at 100 TeV whichever side of the break that falls on.
   * The trailing (E/1e5)^2 undoes the baseline column's own spectrum: the MC's
   * "powerlaw" column is fluxless_weight * 1e-18 * (E/1e5)^-2, so AstroBPL's own
   * 1e-18 cancels and only the (E/1e5)^2 remains. That 1e5 is a property of the
   * MC column, NOT the configurable ERefGeV, and must not be replaced by it.
   *
   * Recalculates when the parameters the active model uses changed (AstroNorm
   * and SpectralIndex; AstroNorm, AstroGamma1, AstroGamma2 and AstroEBreak in
   * broken-power-law mode). When a GpuBackend is supplied the per-event loop
   * runs on the GPU; otherwise the CPU OMP+SIMD path is used (and serves as the
   * validation oracle).
   */
  class PowerlawFlux {
   public:
    PowerlawFlux(const io::ic::ICSample&        sample,
                 const io::ic::Binning&         binning,
                 double                         e_ref_gev,
                 double                         reference_index,
                 bool                           per_type_norm,
                 std::shared_ptr<GpuBackend>    gpu            = nullptr,
                 bool                           need_per_event = false,
                 io::ic::AstroModel             model          = io::ic::AstroModel::Powerlaw,
                 bool                           use_multi_threading = true);
    ~PowerlawFlux() = default;

    bool check_and_recalculate(const ParameterWrapper& parameter);

    [[nodiscard]] std::span<const double> histogram() const noexcept {
      return m_Histogram;
    }

    // Per-event weight, same value already summed into m_Histogram, needed by
    // SampleLikelihood to build the combined ssq histogram for the SAY
    // likelihood. Indexed the same way as io::ic::ICSample (CSR bin order).
    // Empty on the GPU path: there the weights stay in the GPU buffer (see
    // per_event_handle()) and the ssq reduction runs as a kernel instead.
    [[nodiscard]] std::span<const double> per_event_weight() const noexcept {
      return m_PerEventWeight;
    }

    // GPU buffer handle of the per-event weights (-1 on the CPU path or when
    // need_per_event is false), for SampleLikelihood's say_ssq kernel.
    [[nodiscard]] int per_event_handle() const noexcept { return m_hPerEvent; }

   private:
    const io::ic::ICSample& m_Sample;
    double                  m_ERef;
    double                  m_ReferenceIndex;
    bool                    m_PerTypeNorm;
    bool                    m_NeedPerEvent;
    bool                    m_UseMultiThreading;
    io::ic::AstroModel      m_Model;
    std::vector<double>     m_Histogram;
    std::vector<double>     m_PerEventWeight;

    // Non-null when a GPU backend is selected; shared with the other flux
    // components (so e_true / bin_offsets are uploaded once).
    std::shared_ptr<GpuBackend>   m_Gpu;
    int                           m_hETrue    = -1;
    int                           m_hBaseline = -1;
    int                           m_hOffsets  = -1;
    int                           m_hHist     = -1;
    int                           m_hPerEvent = -1;

    void recalculate(const ParameterWrapper& parameter) noexcept;
  };

}  // namespace ana::ic
