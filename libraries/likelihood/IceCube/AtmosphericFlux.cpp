#include "AtmosphericFlux.h"

#include "../../io/IceCube/ICParameter.h"

#include <cmath>
#include <iostream>

namespace ana::ic {

  namespace {

    static_assert(params::ic::nBarrParams == 4, "atmo kernel unrolls exactly 4 Barr params");

    // Scalars pushed to the kernel each iteration; matches AtmoParams below.
    struct AtmoParams {
      float cr;
      float dg;
      float conv_norm;
      float prompt_norm;
      float barr0;
      float barr1;
      float barr2;
      float barr3;
      float inv_eref_conv;
      float inv_eref_prompt;
      int   write_pe;
    };

    // One threadgroup per analysis bin over the CSR-sorted sample. Buffer order
    // matches the MetalBackend convention: inputs (e_true, conv/prompt
    // baseline+alt, 4 Barr gradients, bin_offsets), params, hist, per_event.
    constexpr const char* kKernel = R"METAL(
      #include <metal_stdlib>
      using namespace metal;

      struct AtmoParams {
        float cr; float dg; float conv_norm; float prompt_norm;
        float barr0; float barr1; float barr2; float barr3;
        float inv_eref_conv; float inv_eref_prompt;
        int write_pe;
      };
      constant uint kThreadsPerGroup = 256;

      kernel void atmo_hist(
          device const float*  e_true      [[buffer(0)]],
          device const float*  conv_base   [[buffer(1)]],
          device const float*  conv_alt    [[buffer(2)]],
          device const float*  prompt_base [[buffer(3)]],
          device const float*  prompt_alt  [[buffer(4)]],
          device const float*  barr0       [[buffer(5)]],
          device const float*  barr1       [[buffer(6)]],
          device const float*  barr2       [[buffer(7)]],
          device const float*  barr3       [[buffer(8)]],
          device const uint*   bin_offsets [[buffer(9)]],
          constant AtmoParams& p           [[buffer(10)]],
          device float*        hist        [[buffer(11)]],
          device float*        per_event   [[buffer(12)]],
          uint bin [[threadgroup_position_in_grid]],
          uint tid [[thread_position_in_threadgroup]])
      {
        const uint start = bin_offsets[bin];
        const uint end   = bin_offsets[bin + 1];
        float acc = 0.0f;
        for (uint i = start + tid; i < end; i += kThreadsPerGroup) {
          const float et = e_true[i];
          float event_total = 0.0f;

          const float cb = conv_base[i];
          if (cb > 0.0f) {
            float cw = cb + p.cr * (conv_alt[i] - cb);
            cw *= 1.0f + p.barr0 * barr0[i] / cb;
            cw *= 1.0f + p.barr1 * barr1[i] / cb;
            cw *= 1.0f + p.barr2 * barr2[i] / cb;
            cw *= 1.0f + p.barr3 * barr3[i] / cb;
            cw *= p.conv_norm * pow(et * p.inv_eref_conv, -p.dg);
            event_total += cw;
          }

          const float pb = prompt_base[i];
          if (pb > 0.0f) {
            float pw = pb + p.cr * (prompt_alt[i] - pb);
            pw *= p.prompt_norm * pow(et * p.inv_eref_prompt, -p.dg);
            event_total += pw;
          }

          if (p.write_pe) per_event[i] = event_total;
          acc += event_total;
        }

        threadgroup float shared[kThreadsPerGroup];
        shared[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = kThreadsPerGroup / 2; s > 0; s >>= 1) {
          if (tid < s) shared[tid] += shared[tid + s];
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) hist[bin] = shared[0];
      }
    )METAL";

  }  // namespace

  AtmosphericFlux::AtmosphericFlux(const io::ic::ICSample&       sample,
                                   const double                  conv_delta_gamma_e_ref,
                                   const double                  prompt_delta_gamma_e_ref,
                                   std::shared_ptr<MetalBackend> metal,
                                   const bool                    need_per_event)
    : m_Sample(sample)
    , m_ConvDeltaGammaERef(conv_delta_gamma_e_ref)
    , m_PromptDeltaGammaERef(prompt_delta_gamma_e_ref)
    , m_NeedPerEvent(need_per_event)
    , m_Metal(std::move(metal)) {
    m_Histogram.fill(0.0);
    m_PerEventWeight.assign(sample.size(), 0.0);

    if (m_Metal) {
      const std::size_t M = sample.size();
      m_Metal->ensure_kernel("atmo_hist", kKernel);
      m_hETrue      = m_Metal->upload_column(sample.e_true.data(), M);
      m_hConvBase   = m_Metal->upload_column(sample.conv_baseline.data(), M);
      m_hConvAlt    = m_Metal->upload_column(sample.conv_alt.data(), M);
      m_hPromptBase = m_Metal->upload_column(sample.prompt_baseline.data(), M);
      m_hPromptAlt  = m_Metal->upload_column(sample.prompt_alt.data(), M);
      for (int k = 0; k < params::ic::nBarrParams; ++k)
        m_hBarr[k] = m_Metal->upload_column(sample.barr_conv[k].data(), M);
      m_hOffsets  = m_Metal->upload_offsets(sample.bin_offsets.data(), sample.bin_offsets.size());
      m_hHist     = m_Metal->alloc_output(io::ic::Constants::nBins);
      m_hPerEvent = m_Metal->alloc_output(M);  // always bound; write gated by need
      std::cout << "AtmosphericFlux: using Metal GPU backend\n";
    }
  }

  void AtmosphericFlux::recalculate_metal(const ParameterWrapper& parameter) noexcept {
    using namespace params::ic;

    AtmoParams p;
    p.cr              = static_cast<float>(parameter[CRGrad]);
    p.dg              = static_cast<float>(parameter[DeltaGamma]);
    p.conv_norm       = static_cast<float>(parameter[ConvNorm]);
    p.prompt_norm     = static_cast<float>(parameter[PromptNorm]);
    p.barr0           = static_cast<float>(parameter[BarrH + 0]);
    p.barr1           = static_cast<float>(parameter[BarrH + 1]);
    p.barr2           = static_cast<float>(parameter[BarrH + 2]);
    p.barr3           = static_cast<float>(parameter[BarrH + 3]);
    p.inv_eref_conv   = static_cast<float>(1.0 / m_ConvDeltaGammaERef);
    p.inv_eref_prompt = static_cast<float>(1.0 / m_PromptDeltaGammaERef);
    p.write_pe        = m_NeedPerEvent ? 1 : 0;

    const int inputs[] = {m_hETrue,      m_hConvBase, m_hConvAlt,  m_hPromptBase, m_hPromptAlt,
                          m_hBarr[0],    m_hBarr[1],  m_hBarr[2],  m_hBarr[3],    m_hOffsets};
    m_Metal->dispatch("atmo_hist", inputs, 10, &p, sizeof(p), m_hHist, m_hPerEvent);

    const float* hist = m_Metal->contents(m_hHist);
    for (int bin = 0; bin < io::ic::Constants::nBins; ++bin)
      m_Histogram[bin] = static_cast<double>(hist[bin]);
    if (m_NeedPerEvent) {
      const float* pe = m_Metal->contents(m_hPerEvent);
      for (std::size_t i = 0, n = m_PerEventWeight.size(); i < n; ++i)
        m_PerEventWeight[i] = static_cast<double>(pe[i]);
    }
  }

  void AtmosphericFlux::recalculate(const ParameterWrapper& parameter) noexcept {
    if (m_Metal) {
      recalculate_metal(parameter);
      return;
    }

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
