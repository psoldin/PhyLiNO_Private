#include "AtmosphericFlux.h"

#include "../../io/IceCube/ICParameter.h"

#include <cmath>

namespace ana::ic {

  AtmosphericFlux::AtmosphericFlux(const io::ic::ICSample& sample,
                                   const double            conv_delta_gamma_e_ref,
                                   const double            prompt_delta_gamma_e_ref)
    : m_Sample(sample)
    , m_ConvDeltaGammaERef(conv_delta_gamma_e_ref)
    , m_PromptDeltaGammaERef(prompt_delta_gamma_e_ref) {
    m_Histogram.fill(0.0);
    m_PerEventWeight.assign(sample.size(), 0.0);
  }

  void AtmosphericFlux::recalculate(const ParameterWrapper& parameter) noexcept {
    using namespace params::ic;

    const double cr          = parameter[CRGrad];
    const double dg          = parameter[DeltaGamma];
    const double conv_norm   = parameter[ConvNorm];
    const double prompt_norm = parameter[PromptNorm];

    double barr[nBarrParams];
    for (int k = 0; k < nBarrParams; ++k)
      barr[k] = parameter[BarrH + k];

    const auto& off    = m_Sample.bin_offsets;
    const auto& e_true = m_Sample.e_true;

    #pragma omp parallel for
    for (int bin = 0; bin < io::ic::Constants::nBins; ++bin) {
      double acc = 0.0;
      for (std::size_t i = off[bin]; i < off[bin + 1]; ++i) {
        double event_total = 0.0;

        // --- Conventional ---
        const double conv_base = m_Sample.conv_baseline[i];
        if (conv_base > 0.0) {
          // CRGrad: base + cr * (alt - base)  (== base * crgrad_reweight)
          double conv_w = conv_base + cr * (m_Sample.conv_alt[i] - conv_base);
          // Barr: product of (1 + barr_k * slope_k / base) over conventional Barr params
          for (int k = 0; k < nBarrParams; ++k)
            conv_w *= 1.0 + barr[k] * m_Sample.barr_conv[k][i] / conv_base;
          conv_w *= conv_norm * std::pow(e_true[i] / m_ConvDeltaGammaERef, -dg);
          event_total += conv_w;
        }

        // --- Prompt ---
        const double prompt_base = m_Sample.prompt_baseline[i];
        if (prompt_base > 0.0) {
          double prompt_w = prompt_base + cr * (m_Sample.prompt_alt[i] - prompt_base);
          prompt_w *= prompt_norm * std::pow(e_true[i] / m_PromptDeltaGammaERef, -dg);
          event_total += prompt_w;
        }

        acc += event_total;
        m_PerEventWeight[i] = event_total;
      }
      m_Histogram[bin] = acc;
    }
  }

  bool AtmosphericFlux::check_and_recalculate(const ParameterWrapper& parameter) {
    using namespace params::ic;
    const bool changed =
        parameter.check_parameter_changed(ConvNorm) | parameter.check_parameter_changed(PromptNorm) | parameter.check_parameter_changed(CRGrad) | parameter.check_parameter_changed(DeltaGamma) | parameter.check_parameter_changed(BarrH, BarrZ);

    if (changed)
      recalculate(parameter);
    return changed;
  }

}  // namespace ana::ic
