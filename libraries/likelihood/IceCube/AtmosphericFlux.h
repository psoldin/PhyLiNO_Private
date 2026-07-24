#pragma once

#include "../../io/IceCube/ICConstants.h"
#include "../../io/IceCube/ICSample.h"
#include "../ParameterWrapper.h"

#include <array>
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
   * The histogram holds conv_i + prompt_i summed per analysis bin.
   */
  class AtmosphericFlux {
   public:
    AtmosphericFlux(const io::ic::ICSample& sample,
                    double                  conv_delta_gamma_e_ref,
                    double                  prompt_delta_gamma_e_ref);
    ~AtmosphericFlux() = default;

    bool check_and_recalculate(const ParameterWrapper& parameter);

    [[nodiscard]] std::span<const double> histogram() const noexcept {
      return m_Histogram;
    }

    // Per-event (conv_i + prompt_i) combined weight, matching NNMFit's rule
    // that same-event components are summed before squaring for ssq
    // (NNMFit/core/histogram_builder.py:229-329). Indexed like io::ic::ICSample.
    [[nodiscard]] std::span<const double> per_event_weight() const noexcept {
      return m_PerEventWeight;
    }

   private:
    using BinArray = std::array<double, io::ic::Constants::nBins>;

    const io::ic::ICSample& m_Sample;
    double                  m_ConvDeltaGammaERef;
    double                  m_PromptDeltaGammaERef;
    BinArray                m_Histogram{};
    std::vector<double>     m_PerEventWeight;

    void recalculate(const ParameterWrapper& parameter) noexcept;
  };

}  // namespace ana::ic
