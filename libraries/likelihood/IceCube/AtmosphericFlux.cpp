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
      using Scalar = R;
      R   cr;
      R   dg;
      // The two norms only reach the per-event weights; the conventional and
      // prompt histograms come back unnormalised and are combined on the host.
      R   conv_norm;
      R   prompt_norm;
      R   barr0;
      R   barr1;
      R   barr2;
      R   barr3;
      R   log_eref_conv;
      R   log_eref_prompt;
      int write_pe;
      int use_veto;
      R   veto_e;
      // Stride between the conventional and prompt halves of the partial
      // buffer (see GpuBinReduce's n_quantities).
      int n_chunks;
    };

    // One group per *chunk* of the CSR-sorted sample (see ICSample's chunk
    // decomposition and GpuBinReduce), writing one partial per chunk that
    // bin_gather then sums per bin. Buffer order matches the GpuSession
    // convention: inputs (e_true, conv/prompt baseline+alt, 4 Barr gradients,
    // 6 veto coefficients, chunk_offsets), params, partial, per_event.
    constexpr const char* kKernelMetalBody = R"METAL(
      struct AtmoParams {
        float cr; float dg; float conv_norm; float prompt_norm;
        float barr0; float barr1; float barr2; float barr3;
        float log_eref_conv; float log_eref_prompt;
        int write_pe;
        int use_veto;
        float veto_e;
        int n_chunks;
      };
      // log10(PF) -> PF as an exp2, which is one hardware instruction where
      // pow(10, x) is a log and an exp.
      constant float kLog2Of10 = 3.321928094887362f;

      kernel void atmo_hist(
          device const float*  log_e_true   [[buffer(0)]],
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
        float conv_acc   = 0.0f;
        float conv_cmp   = 0.0f;
        float prompt_acc = 0.0f;
        float prompt_cmp = 0.0f;
        for (uint i = start + tid; i < end; i += kThreadsPerGroup) {
          const float loge = log_e_true[i];
          float cw = 0.0f;
          float pw = 0.0f;

          const float cb = conv_base[i];
          if (cb > 0.0f) {
            cw = cb + p.cr * (conv_alt[i] - cb);
            cw *= 1.0f + p.barr0 * barr0[i] / cb;
            cw *= 1.0f + p.barr1 * barr1[i] / cb;
            cw *= 1.0f + p.barr2 * barr2[i] / cb;
            cw *= 1.0f + p.barr3 * barr3[i] / cb;
            cw *= exp(-p.dg * (loge - p.log_eref_conv));
            if (p.use_veto) {
              const float log_pf = veto_conv_a[i] + veto_conv_b[i] * p.veto_e +
                                   veto_conv_c[i] * p.veto_e * p.veto_e;
              cw *= exp2(kLog2Of10 * log_pf);
            }
            neumaier_add(conv_acc, conv_cmp, cw);
          }

          const float pb = prompt_base[i];
          if (pb > 0.0f) {
            pw = pb + p.cr * (prompt_alt[i] - pb);
            pw *= exp(-p.dg * (loge - p.log_eref_prompt));
            if (p.use_veto) {
              const float log_pf = veto_pr_a[i] + veto_pr_b[i] * p.veto_e +
                                   veto_pr_c[i] * p.veto_e * p.veto_e;
              pw *= exp2(kLog2Of10 * log_pf);
            }
            neumaier_add(prompt_acc, prompt_cmp, pw);
          }

          if (p.write_pe) per_event[i] = p.conv_norm * cw + p.prompt_norm * pw;
        }

        conv_acc += conv_cmp;
        prompt_acc += prompt_cmp;

        threadgroup float shared[kThreadsPerGroup];
        shared[tid] = conv_acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = kThreadsPerGroup / 2; s > 0; s >>= 1) {
          if (tid < s) shared[tid] += shared[tid + s];
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) partial[chunk] = shared[0];

        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[tid] = prompt_acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = kThreadsPerGroup / 2; s > 0; s >>= 1) {
          if (tid < s) shared[tid] += shared[tid + s];
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) partial[(uint)p.n_chunks + chunk] = shared[0];
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
        real log_eref_conv; real log_eref_prompt;
        int write_pe;
        int use_veto;
        real veto_e;
        int n_chunks;
      };
      // log10(PF) -> PF as an exp2, which is one hardware instruction where
      // pow(10, x) is a log and an exp.
      __device__ const real kLog2Of10 = (real)3.321928094887362;

      extern "C" __global__ void atmo_hist(
          const real*         log_e_true,
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
        real conv_acc   = 0.0;
        real conv_cmp   = 0.0;
        real prompt_acc = 0.0;
        real prompt_cmp = 0.0;
        for (unsigned int i = start + tid; i < end; i += nthreads) {
          const real loge = log_e_true[i];
          real cw = 0.0;
          real pw = 0.0;

          const real cb = conv_base[i];
          if (cb > 0.0) {
            cw = cb + p.cr * (conv_alt[i] - cb);
            cw *= 1.0 + p.barr0 * barr0[i] / cb;
            cw *= 1.0 + p.barr1 * barr1[i] / cb;
            cw *= 1.0 + p.barr2 * barr2[i] / cb;
            cw *= 1.0 + p.barr3 * barr3[i] / cb;
            cw *= REXP(-p.dg * (loge - p.log_eref_conv));
            if (p.use_veto) {
              const real log_pf = veto_conv_a[i] + veto_conv_b[i] * p.veto_e +
                                  veto_conv_c[i] * p.veto_e * p.veto_e;
              cw *= REXP2(kLog2Of10 * log_pf);
            }
            neumaier_add(conv_acc, conv_cmp, cw);
          }

          const real pb = prompt_base[i];
          if (pb > 0.0) {
            pw = pb + p.cr * (prompt_alt[i] - pb);
            pw *= REXP(-p.dg * (loge - p.log_eref_prompt));
            if (p.use_veto) {
              const real log_pf = veto_pr_a[i] + veto_pr_b[i] * p.veto_e +
                                  veto_pr_c[i] * p.veto_e * p.veto_e;
              pw *= REXP2(kLog2Of10 * log_pf);
            }
            neumaier_add(prompt_acc, prompt_cmp, pw);
          }

          if (p.write_pe) per_event[i] = p.conv_norm * cw + p.prompt_norm * pw;
        }

        conv_acc += conv_cmp;
        prompt_acc += prompt_cmp;

        __shared__ real sdata[256];
        sdata[tid] = conv_acc;
        __syncthreads();
        for (unsigned int s = nthreads / 2; s > 0; s >>= 1) {
          if (tid < s) sdata[tid] += sdata[tid + s];
          __syncthreads();
        }
        if (tid == 0) partial[chunk] = sdata[0];

        __syncthreads();
        sdata[tid] = prompt_acc;
        __syncthreads();
        for (unsigned int s = nthreads / 2; s > 0; s >>= 1) {
          if (tid < s) sdata[tid] += sdata[tid + s];
          __syncthreads();
        }
        if (tid == 0) partial[(unsigned int)p.n_chunks + chunk] = sdata[0];
      }
    )CUDA";

    // The scalar part of one evaluation: everything that does not vary per event.
    struct AtmoCoeffs {
      double cr;
      double dg;
      double conv_norm;
      double prompt_norm;
      double barr[params::ic::nBarrParams];
      double log_conv_e_ref;
      double log_prompt_e_ref;
      double veto_e;
      bool   use_veto;
    };

    // log10(PF) -> PF as an exp2. Same substitution as in the kernels.
    constexpr double kLog2Of10 = 3.321928094887362;

    // NNMFit VetoThreshold: the energy offset is scalar per evaluation; only the
    // second-order coefficients are per event.
    inline AtmoCoeffs make_coeffs(const ParameterWrapper& parameter,
                                  const double            conv_e_ref,
                                  const double            prompt_e_ref,
                                  const bool              use_veto,
                                  const double            veto_rescale_energy,
                                  const double            veto_anchor_energy) noexcept {
      using namespace params::ic;

      AtmoCoeffs c{};
      c.cr           = parameter[CRGrad];
      c.dg           = parameter[DeltaGamma];
      c.conv_norm    = parameter[ConvNorm];
      c.prompt_norm  = parameter[PromptNorm];
      for (int k = 0; k < nBarrParams; ++k)
        c.barr[k] = parameter[BarrH + k];
      c.log_conv_e_ref   = std::log(conv_e_ref);
      c.log_prompt_e_ref = std::log(prompt_e_ref);
      c.use_veto     = use_veto;
      c.veto_e =
          use_veto ? veto_rescale_energy * std::pow(10.0, parameter[VetoThreshold]) - veto_anchor_energy : 0.0;
      return c;
    }

    // Weight of event i, conventional and prompt kept apart. The hot loop sums
    // the two immediately; the breakdown accumulates them separately. Both go
    // through this one function so the split can never drift from the histogram
    // it decomposes. Either half is 0.0 when its baseline is not populated.
    inline std::pair<double, double> atmo_event_weights(const io::ic::ICSample& sample,
                                                        const std::size_t       i,
                                                        const AtmoCoeffs&       c) noexcept {
      using namespace params::ic;

      const double log_e = sample.log_e_true[i];

      double conv_w = 0.0;
      // --- Conventional ---
      const double conv_base = sample.conv_baseline[i];
      if (conv_base > 0.0) {
        // CRGrad: base + cr * (alt - base)  (== base * crgrad_reweight)
        conv_w = conv_base + c.cr * (sample.conv_alt[i] - conv_base);
        // Barr: product of (1 + barr_k * slope_k / base) over conventional Barr params
        for (int k = 0; k < nBarrParams; ++k)
          conv_w *= 1.0 + c.barr[k] * sample.barr_conv[k][i] / conv_base;
        // (E/E_ref)^-dg as an exp over the event's precomputed log energy.
        conv_w *= std::exp(-c.dg * (log_e - c.log_conv_e_ref));
        if (c.use_veto) {
          const double log_pf = sample.veto_conv[0][i] + sample.veto_conv[1][i] * c.veto_e +
                                sample.veto_conv[2][i] * c.veto_e * c.veto_e;
          conv_w *= std::exp2(kLog2Of10 * log_pf);
        }
      }

      double prompt_w = 0.0;
      // --- Prompt ---
      const double prompt_base = sample.prompt_baseline[i];
      if (prompt_base > 0.0) {
        prompt_w = prompt_base + c.cr * (sample.prompt_alt[i] - prompt_base);
        prompt_w *= std::exp(-c.dg * (log_e - c.log_prompt_e_ref));
        if (c.use_veto) {
          const double log_pf = sample.veto_prompt[0][i] + sample.veto_prompt[1][i] * c.veto_e +
                                sample.veto_prompt[2][i] * c.veto_e * c.veto_e;
          prompt_w *= std::exp2(kLog2Of10 * log_pf);
        }
      }

      // Both halves are returned without their normalisation: ConvNorm and
      // PromptNorm scale the conventional and prompt histograms as a whole, so
      // keeping them out of the event loop is what lets a step in either one
      // rescale a cached histogram (see check_and_recalculate).
      return {conv_w, prompt_w};
    }

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
    m_ConvShape.assign(binning.total_bins(), 0.0);
    m_PromptShape.assign(binning.total_bins(), 0.0);
    // On the GPU path the per-event weights live in a GPU buffer (m_hPerEvent)
    // consumed by SampleLikelihood's say_ssq kernel; no CPU copy is kept. The
    // CPU loop always writes them, so the CPU path always allocates.
    if (!m_Gpu)
      m_PerEventWeight.assign(sample.size(), 0.0);

    if (m_Gpu) {
      const std::size_t M = sample.size();
      const std::string src =
          gpu_kernel_source(m_Gpu->language(), m_Gpu->is_fp64(), kKernelMetalBody, kKernelCudaBody);
      m_Gpu->ensure_kernel("atmo_hist", src.c_str());
      m_hETrue      = m_Gpu->upload_column(sample.e_true.data(), M);
      m_hLogETrue   = m_Gpu->upload_column(sample.log_e_true.data(), M);
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
      // Two reduced quantities: the conventional and prompt histograms.
      m_Reduce.emplace(m_Gpu, sample, m_Histogram.size(), /*n_quantities=*/2);
      m_hConvHist   = m_Gpu->alloc_output(m_Histogram.size());
      m_hPromptHist = m_Gpu->alloc_output(m_Histogram.size());
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
      using R             = std::decay_t<decltype(p.cr)>;
      p.cr                = static_cast<R>(parameter[CRGrad]);
      p.dg                = static_cast<R>(parameter[DeltaGamma]);
      p.conv_norm         = static_cast<R>(parameter[ConvNorm]);
      p.prompt_norm       = static_cast<R>(parameter[PromptNorm]);
      p.barr0             = static_cast<R>(parameter[BarrH + 0]);
      p.barr1             = static_cast<R>(parameter[BarrH + 1]);
      p.barr2             = static_cast<R>(parameter[BarrH + 2]);
      p.barr3             = static_cast<R>(parameter[BarrH + 3]);
      p.log_eref_conv     = static_cast<R>(std::log(m_ConvDeltaGammaERef));
      p.log_eref_prompt   = static_cast<R>(std::log(m_PromptDeltaGammaERef));
      p.write_pe          = m_NeedPerEvent ? 1 : 0;
      p.use_veto          = m_UseVeto ? 1 : 0;
      p.veto_e            = static_cast<R>(
          m_UseVeto ? m_VetoRescaleEnergy * std::pow(10.0, parameter[VetoThreshold]) - m_VetoAnchorEnergy : 0.0);
      p.n_chunks          = static_cast<int>(m_Reduce->n_chunks());
    };

    // Chunk-parallel: the kernel reduces each chunk to two partials -- one
    // conventional, one prompt -- and gather() sums each bin's partials into
    // the matching histogram (see GpuBinReduce). Both come back without their
    // normalisation, which apply_norms() then folds in on the host.
    const int inputs[] = {m_hLogETrue,     m_hConvBase,      m_hConvAlt,       m_hPromptBase,
                          m_hPromptAlt,    m_hBarr[0],       m_hBarr[1],       m_hBarr[2],
                          m_hBarr[3],      m_hVetoConv[0],   m_hVetoConv[1],   m_hVetoConv[2],
                          m_hVetoPrompt[0], m_hVetoPrompt[1], m_hVetoPrompt[2],
                          m_Reduce->chunk_offsets()};

    const auto dispatch_both = [&](auto tag) {
      using ParamsT = decltype(tag);
      ParamsT p{};
      fill(p);
      m_Gpu->dispatch("atmo_hist", inputs, 16, &p, sizeof(p), m_Reduce->partial(), m_hPerEvent,
                      m_Reduce->n_chunks());
      m_Reduce->gather(m_hConvHist, /*q=*/0);
      m_Reduce->gather(m_hPromptHist, /*q=*/1);
      read_back_hist<typename ParamsT::Scalar>(*m_Gpu, m_hConvHist, m_ConvShape);
      read_back_hist<typename ParamsT::Scalar>(*m_Gpu, m_hPromptHist, m_PromptShape);
    };

    if (m_Gpu->is_fp64())
      dispatch_both(AtmoParamsT<double>{});
    else
      dispatch_both(AtmoParamsT<float>{});
    // Per-event weights stay GPU-resident (read by the say_ssq kernel).
    apply_norms(parameter);
  }

  void AtmosphericFlux::recalculate_cpu(const ParameterWrapper& parameter) noexcept {
    const AtmoCoeffs c = make_coeffs(parameter,
                                     m_ConvDeltaGammaERef,
                                     m_PromptDeltaGammaERef,
                                     m_UseVeto,
                                     m_VetoRescaleEnergy,
                                     m_VetoAnchorEnergy);

    const auto& off    = m_Sample.bin_offsets;
    const int   n_bins = static_cast<int>(m_Histogram.size());

    // guided, not the default static schedule: bin populations span orders of
    // magnitude, so an even split of the bin index range is an uneven split of
    // the work.
    #pragma omp parallel for schedule(guided) if(m_UseMultiThreading)
    for (int bin = 0; bin < n_bins; ++bin) {
      double conv_acc   = 0.0;
      double prompt_acc = 0.0;
      for (std::size_t i = off[bin]; i < off[bin + 1]; ++i) {
        const auto [conv_w, prompt_w] = atmo_event_weights(m_Sample, i, c);

        conv_acc += conv_w;
        prompt_acc += prompt_w;
        m_PerEventWeight[i] = c.conv_norm * conv_w + c.prompt_norm * prompt_w;
      }
      m_ConvShape[bin]   = conv_acc;
      m_PromptShape[bin] = prompt_acc;
    }

    apply_norms(parameter);
  }

  void AtmosphericFlux::recalculate(const ParameterWrapper& parameter) noexcept {
    if (m_Gpu) {
      recalculate_gpu(parameter);
      return;
    }
    return recalculate_cpu(parameter);
  }

  AtmoBreakdown AtmosphericFlux::breakdown(const ParameterWrapper& parameter) const {
    const AtmoCoeffs c = make_coeffs(parameter, m_ConvDeltaGammaERef, m_PromptDeltaGammaERef, m_UseVeto,
                                     m_VetoRescaleEnergy, m_VetoAnchorEnergy);
    const auto&      off    = m_Sample.bin_offsets;
    const int        n_bins = static_cast<int>(m_Histogram.size());

    AtmoBreakdown result;
    result.conv.assign(m_Histogram.size(), 0.0);
    result.prompt.assign(m_Histogram.size(), 0.0);

    #pragma omp parallel for schedule(guided) if(m_UseMultiThreading)
    for (int bin = 0; bin < n_bins; ++bin) {
      double conv_acc   = 0.0;
      double prompt_acc = 0.0;
      for (std::size_t i = off[bin]; i < off[bin + 1]; ++i) {
        const auto [conv_w, prompt_w] = atmo_event_weights(m_Sample, i, c);
        conv_acc += conv_w;
        prompt_acc += prompt_w;
      }
      // atmo_event_weights() leaves the normalisations out; the writers want the
      // two halves of the histogram, so they go back in here.
      result.conv[bin]   = c.conv_norm * conv_acc;
      result.prompt[bin] = c.prompt_norm * prompt_acc;
    }

    return result;
  }

  // The shape parameters only -- ConvNorm and PromptNorm are handled separately,
  // since a step in either needs no sweep over the events (see
  // check_and_recalculate).
  inline bool check_shape_parameters(const ParameterWrapper& parameter, const bool use_veto) noexcept {
    using namespace params::ic;
    return  parameter.check_parameter_changed(CRGrad)
    || parameter.check_parameter_changed(DeltaGamma)
    || parameter.check_parameter_changed(BarrH, BarrZ)
    || (use_veto && parameter.check_parameter_changed(VetoThreshold));
  }

  inline bool check_norm_parameters(const ParameterWrapper& parameter) noexcept {
    using namespace params::ic;
    return parameter.check_parameter_changed(ConvNorm)
    || parameter.check_parameter_changed(PromptNorm);
  }

  void AtmosphericFlux::apply_norms(const ParameterWrapper& parameter) noexcept {
    using namespace params::ic;
    const double conv_norm   = parameter[ConvNorm];
    const double prompt_norm = parameter[PromptNorm];
    for (std::size_t bin = 0, n = m_Histogram.size(); bin < n; ++bin)
      m_Histogram[bin] = conv_norm * m_ConvShape[bin] + prompt_norm * m_PromptShape[bin];
  }

  bool AtmosphericFlux::check_and_recalculate(const ParameterWrapper& parameter) {
    using namespace params::ic;

    const bool shape_changed = check_shape_parameters(parameter, m_UseVeto);
    const bool norm_changed  = check_norm_parameters(parameter);

    // This is only for the case this is the first likelihood evaluation function
    // call. Some parameters are initialized to zero, which would not trigger
    // re-calculation.
    if (!m_Seeded) {
      recalculate(parameter);
      m_Seeded = true;
      return true;
    }

    if (!shape_changed && !norm_changed)
      return false;

    // ConvNorm and PromptNorm each scale one whole half of the histogram, so
    // when neither shape parameter moved the new histogram is a recombination
    // of the two cached halves -- O(bins) instead of O(events). The per-event
    // weights carry the norms and are read by the SAY ssq reduction, so a
    // sample that needs them still has to re-run the event loop.
    if (!shape_changed && !m_NeedPerEvent) {
      apply_norms(parameter);
      return true;
    }

    recalculate(parameter);
    return true;
  }

}  // namespace ana::ic
