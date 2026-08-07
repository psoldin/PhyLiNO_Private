#include "AtmosphericFlux.h"

#include "../../io/IceCube/ICParameter.h"

#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>

namespace ana::ic {

  namespace {

    static_assert(params::ic::nBarrParams == 4, "atmo kernel unrolls exactly 4 Barr params");

    // Scalars pushed to the kernel each iteration; layout shared by both the MSL
    // and CUDA kernels (same field order/size) and by the host struct. Templated
    // on the scalar type so FP32 passes floats and FP64 passes doubles, each
    // matching the `real` fields of the compiled kernel struct.
    template <class R>
    struct AtmoParamsT {
      R   cr;
      R   dg;
      R   conv_norm;
      R   prompt_norm;
      R   barr0;
      R   barr1;
      R   barr2;
      R   barr3;
      R   inv_eref_conv;
      R   inv_eref_prompt;
      int write_pe;
      int use_veto;
      R   veto_e;
    };

    // One group per *chunk* of the CSR-sorted sample (see ICSample's chunk
    // decomposition and GpuBinReduce), writing one partial per chunk that
    // bin_gather then sums per bin. Buffer order matches the GpuSession
    // convention: inputs (e_true, conv/prompt baseline+alt, 4 Barr gradients,
    // 6 veto coefficients, chunk_offsets), params, partial, per_event.
    constexpr const char* kKernelMetal = R"METAL(
      #include <metal_stdlib>
      using namespace metal;

      struct AtmoParams {
        float cr; float dg; float conv_norm; float prompt_norm;
        float barr0; float barr1; float barr2; float barr3;
        float inv_eref_conv; float inv_eref_prompt;
        int write_pe;
        int use_veto;
        float veto_e;
      };
      constant uint kThreadsPerGroup = 256;

      kernel void atmo_hist(
          device const float*  e_true       [[buffer(0)]],
          device const float*  conv_base    [[buffer(1)]],
          device const float*  conv_alt     [[buffer(2)]],
          device const float*  prompt_base  [[buffer(3)]],
          device const float*  prompt_alt   [[buffer(4)]],
          device const float*  barr0        [[buffer(5)]],
          device const float*  barr1        [[buffer(6)]],
          device const float*  barr2        [[buffer(7)]],
          device const float*  barr3        [[buffer(8)]],
          device const float*  veto_conv_a  [[buffer(9)]],
          device const float*  veto_conv_b  [[buffer(10)]],
          device const float*  veto_conv_c  [[buffer(11)]],
          device const float*  veto_pr_a    [[buffer(12)]],
          device const float*  veto_pr_b    [[buffer(13)]],
          device const float*  veto_pr_c    [[buffer(14)]],
          device const uint*   chunk_offsets [[buffer(15)]],
          constant AtmoParams& p            [[buffer(16)]],
          device float*        partial      [[buffer(17)]],
          device float*        per_event    [[buffer(18)]],
          uint chunk [[threadgroup_position_in_grid]],
          uint tid [[thread_position_in_threadgroup]])
      {
        const uint start = chunk_offsets[chunk];
        const uint end   = chunk_offsets[chunk + 1];
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
            if (p.use_veto) {
              const float log_pf = veto_conv_a[i] + veto_conv_b[i] * p.veto_e +
                                   veto_conv_c[i] * p.veto_e * p.veto_e;
              cw *= pow(10.0f, log_pf);
            }
            event_total += cw;
          }

          const float pb = prompt_base[i];
          if (pb > 0.0f) {
            float pw = pb + p.cr * (prompt_alt[i] - pb);
            pw *= p.prompt_norm * pow(et * p.inv_eref_prompt, -p.dg);
            if (p.use_veto) {
              const float log_pf = veto_pr_a[i] + veto_pr_b[i] * p.veto_e +
                                   veto_pr_c[i] * p.veto_e * p.veto_e;
              pw *= pow(10.0f, log_pf);
            }
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
        if (tid == 0) partial[chunk] = shared[0];
      }
    )METAL";

    // CUDA-C twin of the MSL kernel: one block per chunk, 256 threads, block-stride
    // sum + __shared__ tree reduction. extern "C" so cuModuleGetFunction resolves
    // the unmangled name. The params struct is passed by value at the same
    // position the Metal kernel takes its constant buffer.
    // Written against a generic scalar `real` and power macro `RPOW`;
    // cuda_kernel_source() prepends the typedef selecting float (FP32) or double
    // (FP64), so one body serves both precisions.
    constexpr const char* kKernelCudaBody = R"CUDA(
      struct AtmoParams {
        real cr; real dg; real conv_norm; real prompt_norm;
        real barr0; real barr1; real barr2; real barr3;
        real inv_eref_conv; real inv_eref_prompt;
        int write_pe;
        int use_veto;
        real veto_e;
      };

      extern "C" __global__ void atmo_hist(
          const real*         e_true,
          const real*         conv_base,
          const real*         conv_alt,
          const real*         prompt_base,
          const real*         prompt_alt,
          const real*         barr0,
          const real*         barr1,
          const real*         barr2,
          const real*         barr3,
          const real*         veto_conv_a,
          const real*         veto_conv_b,
          const real*         veto_conv_c,
          const real*         veto_pr_a,
          const real*         veto_pr_b,
          const real*         veto_pr_c,
          const unsigned int* chunk_offsets,
          AtmoParams          p,
          real*               partial,
          real*               per_event)
      {
        const unsigned int chunk    = blockIdx.x;
        const unsigned int tid      = threadIdx.x;
        const unsigned int nthreads = blockDim.x;
        const unsigned int start    = chunk_offsets[chunk];
        const unsigned int end      = chunk_offsets[chunk + 1];
        real acc = 0.0;
        for (unsigned int i = start + tid; i < end; i += nthreads) {
          const real et = e_true[i];
          real event_total = 0.0;

          const real cb = conv_base[i];
          if (cb > 0.0) {
            real cw = cb + p.cr * (conv_alt[i] - cb);
            cw *= 1.0 + p.barr0 * barr0[i] / cb;
            cw *= 1.0 + p.barr1 * barr1[i] / cb;
            cw *= 1.0 + p.barr2 * barr2[i] / cb;
            cw *= 1.0 + p.barr3 * barr3[i] / cb;
            cw *= p.conv_norm * RPOW(et * p.inv_eref_conv, -p.dg);
            if (p.use_veto) {
              const real log_pf = veto_conv_a[i] + veto_conv_b[i] * p.veto_e +
                                  veto_conv_c[i] * p.veto_e * p.veto_e;
              cw *= RPOW((real)10.0, log_pf);
            }
            event_total += cw;
          }

          const real pb = prompt_base[i];
          if (pb > 0.0) {
            real pw = pb + p.cr * (prompt_alt[i] - pb);
            pw *= p.prompt_norm * RPOW(et * p.inv_eref_prompt, -p.dg);
            if (p.use_veto) {
              const real log_pf = veto_pr_a[i] + veto_pr_b[i] * p.veto_e +
                                  veto_pr_c[i] * p.veto_e * p.veto_e;
              pw *= RPOW((real)10.0, log_pf);
            }
            event_total += pw;
          }

          if (p.write_pe) per_event[i] = event_total;
          acc += event_total;
        }

        __shared__ real sdata[256];
        sdata[tid] = acc;
        __syncthreads();
        for (unsigned int s = nthreads / 2; s > 0; s >>= 1) {
          if (tid < s) sdata[tid] += sdata[tid + s];
          __syncthreads();
        }
        if (tid == 0) partial[chunk] = sdata[0];
      }
    )CUDA";

  }  // namespace

  AtmosphericFlux::AtmosphericFlux(const io::ic::ICSample&       sample,
                                   const io::ic::Binning&        binning,
                                   const double                  conv_delta_gamma_e_ref,
                                   const double                  prompt_delta_gamma_e_ref,
                                   std::shared_ptr<GpuSession>   gpu,
                                   const bool                    need_per_event,
                                   const bool                    use_veto,
                                   const double                  veto_anchor_energy,
                                   const double                  veto_rescale_energy,
                                   const bool                    use_multi_threading)
    : m_Sample(sample)
    , m_ConvDeltaGammaERef(conv_delta_gamma_e_ref)
    , m_PromptDeltaGammaERef(prompt_delta_gamma_e_ref)
    , m_NeedPerEvent(need_per_event)
    , m_UseVeto(use_veto)
    , m_VetoAnchorEnergy(veto_anchor_energy)
    , m_VetoRescaleEnergy(veto_rescale_energy)
    , m_UseMultiThreading(use_multi_threading)
    , m_Gpu(std::move(gpu)) {
    m_Histogram.assign(binning.total_bins(), 0.0);
    // On the GPU path the per-event weights live in a GPU buffer (m_hPerEvent)
    // consumed by SampleLikelihood's say_ssq kernel; no CPU copy is kept. The
    // CPU loop always writes them, so the CPU path always allocates.
    if (!m_Gpu)
      m_PerEventWeight.assign(sample.size(), 0.0);

    if (m_Gpu) {
      const std::size_t M = sample.size();
      const std::string cuda_src =
          m_Gpu->language() == GpuLanguage::Cuda ? cuda_kernel_source(m_Gpu->is_fp64(), kKernelCudaBody) : std::string{};
      const char* src = m_Gpu->language() == GpuLanguage::Cuda ? cuda_src.c_str() : kKernelMetal;
      m_Gpu->ensure_kernel("atmo_hist", src);
      m_hETrue      = m_Gpu->upload_column(sample.e_true.data(), M);
      m_hConvBase   = m_Gpu->upload_column(sample.conv_baseline.data(), M);
      m_hConvAlt    = m_Gpu->upload_column(sample.conv_alt.data(), M);
      m_hPromptBase = m_Gpu->upload_column(sample.prompt_baseline.data(), M);
      m_hPromptAlt  = m_Gpu->upload_column(sample.prompt_alt.data(), M);
      for (int k = 0; k < params::ic::nBarrParams; ++k)
        m_hBarr[k] = m_Gpu->upload_column(sample.barr_conv[k].data(), M);
      // Every kernel argument must be bound even when unread (Metal faults on an
      // unbound buffer it might touch); a sample without veto columns binds
      // e_true's already-uploaded handle instead of allocating dead buffers.
      for (int k = 0; k < 3; ++k) {
        m_hVetoConv[k]   = m_UseVeto ? m_Gpu->upload_column(sample.veto_conv[k].data(), M) : m_hETrue;
        m_hVetoPrompt[k] = m_UseVeto ? m_Gpu->upload_column(sample.veto_prompt[k].data(), M) : m_hETrue;
      }
      m_Reduce.emplace(m_Gpu, sample, m_Histogram.size());
      m_hHist     = m_Gpu->alloc_output(m_Histogram.size());
      // The per-event weights are read only by say_ssq, never on the host.
      m_hPerEvent = m_NeedPerEvent ? m_Gpu->alloc_output(M, /*readback=*/false) : -1;
      std::cout << "AtmosphericFlux: using GPU backend (" << m_Reduce->n_chunks() << " chunks over "
                << m_Histogram.size() << " bins)\n";
    }
  }

  void AtmosphericFlux::recalculate_gpu(const ParameterWrapper& parameter) noexcept {
    using namespace params::ic;

    // Fill either a float or a double params struct from the double parameters,
    // matching the kernel's `real` precision.
    const auto fill = [&](auto& p) {
      using R           = std::decay_t<decltype(p.cr)>;
      p.cr              = static_cast<R>(parameter[CRGrad]);
      p.dg              = static_cast<R>(parameter[DeltaGamma]);
      p.conv_norm       = static_cast<R>(parameter[ConvNorm]);
      p.prompt_norm     = static_cast<R>(parameter[PromptNorm]);
      p.barr0           = static_cast<R>(parameter[BarrH + 0]);
      p.barr1           = static_cast<R>(parameter[BarrH + 1]);
      p.barr2           = static_cast<R>(parameter[BarrH + 2]);
      p.barr3           = static_cast<R>(parameter[BarrH + 3]);
      p.inv_eref_conv   = static_cast<R>(1.0 / m_ConvDeltaGammaERef);
      p.inv_eref_prompt = static_cast<R>(1.0 / m_PromptDeltaGammaERef);
      p.write_pe        = m_NeedPerEvent ? 1 : 0;
      p.use_veto        = m_UseVeto ? 1 : 0;
      p.veto_e          = static_cast<R>(
          m_UseVeto ? m_VetoRescaleEnergy * std::pow(10.0, parameter[VetoThreshold]) - m_VetoAnchorEnergy : 0.0);
    };

    // Chunk-parallel: the kernel reduces each chunk to a partial, then gather()
    // sums each bin's partials into m_hHist (see GpuBinReduce).
    const int inputs[] = {m_hETrue,        m_hConvBase,      m_hConvAlt,       m_hPromptBase,
                          m_hPromptAlt,    m_hBarr[0],       m_hBarr[1],       m_hBarr[2],
                          m_hBarr[3],      m_hVetoConv[0],   m_hVetoConv[1],   m_hVetoConv[2],
                          m_hVetoPrompt[0], m_hVetoPrompt[1], m_hVetoPrompt[2],
                          m_Reduce->chunk_offsets()};

    if (m_Gpu->is_fp64()) {
      AtmoParamsT<double> p;
      fill(p);
      m_Gpu->dispatch("atmo_hist", inputs, 16, &p, sizeof(p), m_Reduce->partial(), m_hPerEvent,
                      m_Reduce->n_chunks());
      m_Reduce->gather(m_hHist);
      const double* hist = m_Gpu->contents_f64(m_hHist);
      for (std::size_t bin = 0, n = m_Histogram.size(); bin < n; ++bin) m_Histogram[bin] = hist[bin];
    } else {
      AtmoParamsT<float> p;
      fill(p);
      m_Gpu->dispatch("atmo_hist", inputs, 16, &p, sizeof(p), m_Reduce->partial(), m_hPerEvent,
                      m_Reduce->n_chunks());
      m_Reduce->gather(m_hHist);
      const float* hist = m_Gpu->contents(m_hHist);
      for (std::size_t bin = 0, n = m_Histogram.size(); bin < n; ++bin) m_Histogram[bin] = static_cast<double>(hist[bin]);
    }
    // Per-event weights stay GPU-resident (read by the say_ssq kernel).
  }

  void AtmosphericFlux::recalculate(const ParameterWrapper& parameter) noexcept {
    if (m_Gpu) {
      recalculate_gpu(parameter);
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
    const int   n_bins = static_cast<int>(m_Histogram.size());

    // NNMFit VetoThreshold: the energy offset is scalar per evaluation; only the
    // second-order coefficients are per event.
    const double veto_e =
        m_UseVeto ? m_VetoRescaleEnergy * std::pow(10.0, parameter[VetoThreshold]) - m_VetoAnchorEnergy : 0.0;

    #pragma omp parallel for if(m_UseMultiThreading)
    for (int bin = 0; bin < n_bins; ++bin) {
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
          if (m_UseVeto) {
            const double log_pf = m_Sample.veto_conv[0][i] + m_Sample.veto_conv[1][i] * veto_e +
                                  m_Sample.veto_conv[2][i] * veto_e * veto_e;
            conv_w *= std::pow(10.0, log_pf);
          }
          event_total += conv_w;
        }

        // --- Prompt ---
        const double prompt_base = m_Sample.prompt_baseline[i];
        if (prompt_base > 0.0) {
          double prompt_w = prompt_base + cr * (m_Sample.prompt_alt[i] - prompt_base);
          prompt_w *= prompt_norm * std::pow(e_true[i] / m_PromptDeltaGammaERef, -dg);
          if (m_UseVeto) {
            const double log_pf = m_Sample.veto_prompt[0][i] + m_Sample.veto_prompt[1][i] * veto_e +
                                  m_Sample.veto_prompt[2][i] * veto_e * veto_e;
            prompt_w *= std::pow(10.0, log_pf);
          }
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
        parameter.check_parameter_changed(ConvNorm)
        | parameter.check_parameter_changed(PromptNorm)
        | parameter.check_parameter_changed(CRGrad)
        | parameter.check_parameter_changed(DeltaGamma)
        | parameter.check_parameter_changed(BarrH, BarrZ)
        | (m_UseVeto && parameter.check_parameter_changed(VetoThreshold));

    if (changed)
      recalculate(parameter);
    return changed;
  }

}  // namespace ana::ic
