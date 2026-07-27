#include "PowerlawFlux.h"

#include "../../io/IceCube/ICParameter.h"

#include <cmath>
#include <iostream>

namespace ana::ic {

  namespace {

    // Scalars pushed to the kernel each iteration; layout shared by both the MSL
    // and CUDA kernels (same field order/size) and by the host struct.
    struct PowerlawParams {
      float eff_norm;
      float inv_eref;
      float exponent;
      int   write_pe;
    };

    // One group per analysis bin over the CSR-sorted sample; grid-stride sum +
    // in-group tree reduction. Buffer order matches the GpuBackend convention:
    // inputs (e_true, baseline, bin_offsets), params, hist, per_event.
    constexpr const char* kKernelMetal = R"METAL(
      #include <metal_stdlib>
      using namespace metal;

      struct PowerlawParams { float eff_norm; float inv_eref; float exponent; int write_pe; };
      constant uint kThreadsPerGroup = 256;

      kernel void powerlaw_hist(
          device const float*      e_true      [[buffer(0)]],
          device const float*      baseline    [[buffer(1)]],
          device const uint*       bin_offsets [[buffer(2)]],
          constant PowerlawParams& p           [[buffer(3)]],
          device float*            hist        [[buffer(4)]],
          device float*            per_event   [[buffer(5)]],
          uint bin [[threadgroup_position_in_grid]],
          uint tid [[thread_position_in_threadgroup]])
      {
        const uint start = bin_offsets[bin];
        const uint end   = bin_offsets[bin + 1];
        float acc = 0.0f;
        for (uint i = start + tid; i < end; i += kThreadsPerGroup) {
          const float w = baseline[i] * p.eff_norm * pow(e_true[i] * p.inv_eref, p.exponent);
          if (p.write_pe) per_event[i] = w;
          acc += w;
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

    // CUDA-C twin of the MSL kernel: one block per bin, 256 threads, block-stride
    // sum + __shared__ tree reduction. extern "C" so cuModuleGetFunction resolves
    // the unmangled name. The params struct is passed by value at the same
    // position the Metal kernel takes its constant buffer.
    constexpr const char* kKernelCuda = R"CUDA(
      struct PowerlawParams { float eff_norm; float inv_eref; float exponent; int write_pe; };

      extern "C" __global__ void powerlaw_hist(
          const float*        e_true,
          const float*        baseline,
          const unsigned int* bin_offsets,
          PowerlawParams      p,
          float*              hist,
          float*              per_event)
      {
        const unsigned int bin      = blockIdx.x;
        const unsigned int tid      = threadIdx.x;
        const unsigned int nthreads = blockDim.x;
        const unsigned int start    = bin_offsets[bin];
        const unsigned int end      = bin_offsets[bin + 1];
        float acc = 0.0f;
        for (unsigned int i = start + tid; i < end; i += nthreads) {
          const float w = baseline[i] * p.eff_norm * powf(e_true[i] * p.inv_eref, p.exponent);
          if (p.write_pe) per_event[i] = w;
          acc += w;
        }
        __shared__ float sdata[256];
        sdata[tid] = acc;
        __syncthreads();
        for (unsigned int s = nthreads / 2; s > 0; s >>= 1) {
          if (tid < s) sdata[tid] += sdata[tid + s];
          __syncthreads();
        }
        if (tid == 0) hist[bin] = sdata[0];
      }
    )CUDA";

  }  // namespace

  PowerlawFlux::PowerlawFlux(const io::ic::ICSample&       sample,
                             const io::ic::Binning&        binning,
                             const double                  e_ref_gev,
                             const double                  reference_index,
                             const bool                    per_type_norm,
                             std::shared_ptr<GpuBackend>   gpu,
                             const bool                    need_per_event)
    : m_Sample(sample)
    , m_ERef(e_ref_gev)
    , m_ReferenceIndex(reference_index)
    , m_PerTypeNorm(per_type_norm)
    , m_NeedPerEvent(need_per_event)
    , m_Gpu(std::move(gpu)) {
    m_Histogram.assign(binning.total_bins(), 0.0);
    // On the GPU path the per-event weights live in a GPU buffer (m_hPerEvent)
    // consumed by SampleLikelihood's say_ssq kernel; no CPU copy is kept. The
    // CPU loop always writes them, so the CPU path always allocates.
    if (!m_Gpu)
      m_PerEventWeight.assign(sample.size(), 0.0);

    if (m_Gpu) {
      const std::size_t M   = sample.size();
      const char*       src = m_Gpu->language() == GpuLanguage::Cuda ? kKernelCuda : kKernelMetal;
      m_Gpu->ensure_kernel("powerlaw_hist", src);
      m_hETrue    = m_Gpu->upload_column(sample.e_true.data(), M);
      m_hBaseline = m_Gpu->upload_column(sample.astro_baseline.data(), M);
      m_hOffsets  = m_Gpu->upload_offsets(sample.bin_offsets.data(), sample.bin_offsets.size());
      m_hHist     = m_Gpu->alloc_output(m_Histogram.size());
      m_hPerEvent = m_NeedPerEvent ? m_Gpu->alloc_output(M) : -1;
      std::cout << "PowerlawFlux: using GPU backend\n";
    }
  }

  void PowerlawFlux::recalculate(const ParameterWrapper& parameter) noexcept {
    using namespace params::ic;

    const double norm  = parameter[AstroNorm];
    const double gamma = parameter[SpectralIndex];

    if (m_Gpu) {
      PowerlawParams p;
      p.eff_norm = static_cast<float>(m_PerTypeNorm ? norm : 0.5 * norm);
      p.inv_eref = static_cast<float>(1.0 / m_ERef);
      p.exponent = static_cast<float>(m_ReferenceIndex - gamma);
      p.write_pe = m_NeedPerEvent ? 1 : 0;

      const int inputs[] = {m_hETrue, m_hBaseline, m_hOffsets};
      m_Gpu->dispatch("powerlaw_hist", inputs, 3, &p, sizeof(p), m_hHist, m_hPerEvent, m_Histogram.size());

      const float* hist = m_Gpu->contents(m_hHist);
      for (std::size_t bin = 0, n = m_Histogram.size(); bin < n; ++bin)
        m_Histogram[bin] = static_cast<double>(hist[bin]);
      // Per-event weights stay GPU-resident (read by the say_ssq kernel).
      return;
    }

    // NNMFit Norm with per_type_norm=false halves the per-type normalization.
    const double eff_norm = m_PerTypeNorm ? norm : 0.5 * norm;
    // (E/E_ref)^(ref_index - gamma), baseline is already ~E^(-ref_index).
    const double exponent = m_ReferenceIndex - gamma;

    const auto& off      = m_Sample.bin_offsets;
    const auto& baseline = m_Sample.astro_baseline;
    const auto& e_true   = m_Sample.e_true;
    const int   n_bins   = static_cast<int>(m_Histogram.size());

    #pragma omp parallel for
    for (int bin = 0; bin < n_bins; ++bin) {
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
