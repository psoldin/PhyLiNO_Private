#pragma once

#include "../../io/IceCube/Binning.h"
#include "../../io/IceCube/ICParameter.h"  // params::ic::nBarrParams
#include "../../io/IceCube/ICSample.h"
#include "../ParameterWrapper.h"
#include "GpuBackend.h"
#include "GpuBinReduce.h"

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ana::ic {

  /**
   * Conventional + prompt atmospheric flux, assembled per event exactly as the
   * multiplicative NNMFit graph (graph = baseline, then *= each parameter's
   * reweight). Both share the CRGrad and DeltaGamma nuisances; Barr applies to
   * the conventional component only (per the tracks-only config).
   *
   * Conventional (events with conv_baseline > 0):
   *   conv_i = [conv_base + CRGrad*(conv_alt - conv_base)]
   *          * prod_k (1 + barr_k * barr_conv[k]_i / conv_base)      Barr, k in {H,W,Y,Z}
   *          * ConvNorm
   *          * (E_true_i / conv_e_ref)^(-DeltaGamma)
   *
   * Prompt (events with prompt_baseline > 0):
   *   prompt_i = [prompt_base + CRGrad*(prompt_alt - prompt_base)]
   *            * PromptNorm
   *            * (E_true_i / prompt_e_ref)^(-DeltaGamma)
   *
   * When use_veto is set (the cascade samples), both conv_i and prompt_i also
   * carry NNMFit's passing-fraction reweight (parameters/veto_threshold.py):
   *   e        = veto_rescale_energy * 10^VetoThreshold - veto_anchor_energy   (both 100 GeV)
   *   PF_i     = 10^(veto_a_i + veto_b_i * e + veto_c_i * e^2)
   *   conv_i   *= PF_conv_i
   *   prompt_i *= PF_prompt_i
   * `e` is a scalar per evaluation; only the six per-event coefficients differ per
   * event. The tracks sample runs with use_veto = false and is bit-for-bit
   * unaffected (the veto columns are never read for it, see ICDataBase).
   *
   * The histogram holds conv_i + prompt_i summed per analysis bin. When a
   * GpuSession is supplied the per-event loop runs on the GPU; otherwise the
   * CPU OMP+SIMD path is used (and serves as the validation oracle).
   */
  class AtmosphericFlux {
   public:
    AtmosphericFlux(const io::ic::ICSample&       sample,
                    const io::ic::Binning&        binning,
                    double                        conv_delta_gamma_e_ref,
                    double                        prompt_delta_gamma_e_ref,
                    std::shared_ptr<GpuSession>   gpu                 = nullptr,
                    bool                          need_per_event      = false,
                    bool                          use_veto            = false,
                    double                        veto_anchor_energy  = 100.0,
                    double                        veto_rescale_energy = 100.0,
                    bool                          use_multi_threading = true);
    ~AtmosphericFlux() = default;

    bool check_and_recalculate(const ParameterWrapper& parameter);

    [[nodiscard]] std::span<const double> histogram() const noexcept {
      return m_Histogram;
    }

    // Per-event (conv_i + prompt_i) combined weight, matching NNMFit's rule
    // that same-event components are summed before squaring for ssq
    // (NNMFit/core/histogram_builder.py:229-329). Indexed like io::ic::ICSample.
    // Empty on the GPU path: the weights stay in the GPU buffer (see
    // per_event_handle()) and the ssq reduction runs as a kernel instead.
    [[nodiscard]] std::span<const double> per_event_weight() const noexcept {
      return m_PerEventWeight;
    }

    // GPU buffer handle of the per-event weights (-1 on the CPU path or when
    // need_per_event is false), for SampleLikelihood's say_ssq kernel.
    [[nodiscard]] int per_event_handle() const noexcept { return m_hPerEvent; }

   private:
    const io::ic::ICSample& m_Sample;
    double                  m_ConvDeltaGammaERef;
    double                  m_PromptDeltaGammaERef;
    bool                    m_NeedPerEvent;
    bool                    m_UseVeto;
    double                  m_VetoAnchorEnergy;
    double                  m_VetoRescaleEnergy;
    bool                    m_UseMultiThreading;
    std::vector<double>     m_Histogram;
    std::vector<double>     m_PerEventWeight;

    // Non-null when a GPU backend is selected; shared with this sample's other
    // flux components, and behind it with every other sample and fit (so
    // e_true / bin_offsets are uploaded once per process).
    std::shared_ptr<GpuSession>                      m_Gpu;
    int                                              m_hETrue      = -1;
    int                                              m_hConvBase   = -1;
    int                                              m_hConvAlt    = -1;
    int                                              m_hPromptBase = -1;
    int                                              m_hPromptAlt  = -1;
    std::array<int, params::ic::nBarrParams>         m_hBarr{};
    std::array<int, 3>                               m_hVetoConv{};
    std::array<int, 3>                               m_hVetoPrompt{};
    int                                              m_hHist       = -1;
    int                                              m_hPerEvent   = -1;
    // Owns the chunk-offset binding, the per-chunk partial buffer and the
    // gather dispatch that turns those partials into m_hHist.
    std::optional<GpuBinReduce>                      m_Reduce;

    void recalculate(const ParameterWrapper& parameter) noexcept;
    void recalculate_gpu(const ParameterWrapper& parameter) noexcept;
  };

}  // namespace ana::ic
