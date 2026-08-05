#include "PowerlawFlux.h"

#include "../../io/IceCube/ICParameter.h"

#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>

namespace ana::ic {

  namespace {

    // Scalars pushed to the kernel each iteration; layout shared by both the MSL
    // and CUDA kernels (same field order/size) and by the host struct. Templated
    // on the scalar type so the FP32 path passes floats and the FP64 path passes
    // doubles, each matching the `real` fields of the compiled kernel struct.
    template <class R>
    struct PowerlawParamsT {
      R   eff_norm;
      R   inv_eref;
      R   exponent;  // single PL: ref_index - gamma. Broken PL: gamma_1.
      int write_pe;
      R   gamma_2;     // broken PL only
      R   inv_ebreak;  // broken PL only: 1 / E_break
      R   pivot;       // broken PL only: the 100 TeV renormalisation
      int broken;      // 0 = single power law, 1 = broken power law
    };

    // One group per analysis bin over the CSR-sorted sample; grid-stride sum +
    // in-group tree reduction. Buffer order matches the GpuSession convention:
    // inputs (e_true, baseline, bin_offsets), params, hist, per_event.
    constexpr const char* kKernelMetal = R"METAL(
      #include <metal_stdlib>
      using namespace metal;

      struct PowerlawParams {
        float eff_norm; float inv_eref; float exponent; int write_pe;
        float gamma_2; float inv_ebreak; float pivot; int broken;
      };
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
          float w;
          if (p.broken) {
            const float e     = e_true[i];
            const float x     = e * p.inv_ebreak;
            const float shape = x < 1.0f ? pow(x, -p.exponent) : pow(x, -p.gamma_2);
            const float undo  = e * 1.0e-5f;
            w = baseline[i] * p.eff_norm * p.pivot * shape * undo * undo;
          } else {
            w = baseline[i] * p.eff_norm * pow(e_true[i] * p.inv_eref, p.exponent);
          }
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
    // position the Metal kernel takes its constant buffer. Written against a
    // generic scalar `real` and power macro `RPOW`; cuda_kernel_source() prepends
    // the typedef selecting float (FP32) or double (FP64), so one body serves
    // both precisions.
    constexpr const char* kKernelCudaBody = R"CUDA(
      struct PowerlawParams {
        real eff_norm; real inv_eref; real exponent; int write_pe;
        real gamma_2; real inv_ebreak; real pivot; int broken;
      };

      extern "C" __global__ void powerlaw_hist(
          const real*         e_true,
          const real*         baseline,
          const unsigned int* bin_offsets,
          PowerlawParams      p,
          real*               hist,
          real*               per_event)
      {
        const unsigned int bin      = blockIdx.x;
        const unsigned int tid      = threadIdx.x;
        const unsigned int nthreads = blockDim.x;
        const unsigned int start    = bin_offsets[bin];
        const unsigned int end      = bin_offsets[bin + 1];
        real acc = 0.0;
        for (unsigned int i = start + tid; i < end; i += nthreads) {
          real w;
          if (p.broken) {
            const real e     = e_true[i];
            const real x     = e * p.inv_ebreak;
            const real shape = x < 1.0 ? RPOW(x, -p.exponent) : RPOW(x, -p.gamma_2);
            const real undo  = e * (real)1.0e-5;
            w = baseline[i] * p.eff_norm * p.pivot * shape * undo * undo;
          } else {
            w = baseline[i] * p.eff_norm * RPOW(e_true[i] * p.inv_eref, p.exponent);
          }
          if (p.write_pe) per_event[i] = w;
          acc += w;
        }
        __shared__ real sdata[256];
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

  PowerlawFlux::PowerlawFlux(const io::ic::ICSample&     sample,
                             const io::ic::Binning&      binning,
                             const double                e_ref_gev,
                             const double                reference_index,
                             const bool                  per_type_norm,
                             std::shared_ptr<GpuSession> gpu,
                             const bool                  need_per_event,
                             const io::ic::AstroModel    model,
                             const bool                  use_multi_threading)
    : m_Sample(sample)
    , m_ERef(e_ref_gev)
    , m_ReferenceIndex(reference_index)
    , m_PerTypeNorm(per_type_norm)
    , m_NeedPerEvent(need_per_event)
    , m_UseMultiThreading(use_multi_threading)
    , m_Model(model)
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

    // Broken-power-law scalars, evaluated in double precision on the host and
    // narrowed only when the kernel params struct is filled. Left at their
    // single-power-law-safe defaults otherwise.
    const bool broken  = m_Model == io::ic::AstroModel::BrokenPowerlaw;
    double     g1      = 0.0;
    double     g2      = 0.0;
    double     e_break = 1.0;
    double     pivot   = 1.0;
    if (broken) {
      g1      = parameter[AstroGamma1];
      g2      = parameter[AstroGamma2];
      e_break = std::pow(10.0, parameter[AstroEBreak]);
      // NNMFit AstroBPL: the norm is the flux at 100 TeV, so it is rescaled by
      // whichever branch 100 TeV falls in. Positive exponent here, unlike the
      // per-event shape factor below.
      pivot = 1.0e5 < e_break ? std::pow(1.0e5 / e_break, g1)
                              : std::pow(1.0e5 / e_break, g2);
    }

    if (m_Gpu) {
      // Fill either a float or a double params struct from the double host
      // scalars, matching the kernel's `real` precision. The kernel reuses
      // `exponent` as gamma_1 in broken-power-law mode.
      const auto fill = [&](auto& p) {
        using R      = std::decay_t<decltype(p.eff_norm)>;
        p.eff_norm   = static_cast<R>(m_PerTypeNorm ? norm : 0.5 * norm);
        p.inv_eref   = static_cast<R>(1.0 / m_ERef);
        p.exponent   = static_cast<R>(broken ? g1 : m_ReferenceIndex - gamma);
        p.write_pe   = m_NeedPerEvent ? 1 : 0;
        p.gamma_2    = static_cast<R>(g2);
        p.inv_ebreak = static_cast<R>(1.0 / e_break);
        p.pivot      = static_cast<R>(pivot);
        p.broken     = broken ? 1 : 0;
      };

      const int inputs[] = {m_hETrue, m_hBaseline, m_hOffsets};
      if (m_Gpu->is_fp64()) {
        PowerlawParamsT<double> p;
        fill(p);
        m_Gpu->dispatch("powerlaw_hist", inputs, 3, &p, sizeof(p), m_hHist, m_hPerEvent, m_Histogram.size());
        const double* hist = m_Gpu->contents_f64(m_hHist);
        for (std::size_t bin = 0, n = m_Histogram.size(); bin < n; ++bin)
          m_Histogram[bin] = hist[bin];
      } else {
        PowerlawParamsT<float> p;
        fill(p);
        m_Gpu->dispatch("powerlaw_hist", inputs, 3, &p, sizeof(p), m_hHist, m_hPerEvent, m_Histogram.size());
        const float* hist = m_Gpu->contents(m_hHist);
        for (std::size_t bin = 0, n = m_Histogram.size(); bin < n; ++bin)
          m_Histogram[bin] = static_cast<double>(hist[bin]);
      }
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

    if (broken) {
#pragma omp parallel for if (m_UseMultiThreading)
      for (int bin = 0; bin < n_bins; ++bin) {
        double acc = 0.0;
        for (std::size_t i = off[bin], n = off[bin + 1]; i < n; ++i) {
          const double e     = e_true[i];
          const double x     = e / e_break;
          const double shape = x < 1.0 ? std::pow(x, -g1) : std::pow(x, -g2);
          // Undo the baseline column's built-in E^-2: `powerlaw` is
          // fluxless_weight * 1e-18 * (E/1e5)^-2, so the 1e-18 cancels against
          // AstroBPL's own factor and (E/1e5)^2 remains. The 1e5 is a property of
          // that column, not the configurable ERefGeV.
          const double undo = (e / 1.0e5) * (e / 1.0e5);
          const double w    = baseline[i] * eff_norm * pivot * shape * undo;
          acc += w;
          m_PerEventWeight[i] = w;
        }
        m_Histogram[bin] = acc;
      }
      return;
    }

#pragma omp parallel for if (m_UseMultiThreading)
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

  inline bool check_SPL_parameter(const ParameterWrapper& parameter) noexcept {
    using namespace params::ic;
    return parameter.check_parameter_changed(AstroNorm)
           || parameter.check_parameter_changed(SpectralIndex);
  }

  inline bool check_BPL_parameter(const ParameterWrapper& parameter) noexcept {
    using namespace params::ic;
    return parameter.check_parameter_changed(AstroNorm)
           || parameter.check_parameter_changed(AstroGamma1)
           || parameter.check_parameter_changed(AstroGamma2)
           || parameter.check_parameter_changed(AstroEBreak);
  }

  bool PowerlawFlux::check_and_recalculate(const ParameterWrapper& parameter) {
    using namespace params::ic;
    // Watch exactly the parameters the active model reads: SpectralIndex is
    // unused in broken-power-law mode, and the three AstroBPL parameters are
    // unused in single-power-law mode.

    using enum io::ic::AstroModel;

    const bool changed = (m_Model == BrokenPowerlaw)
                             ? check_BPL_parameter(parameter)
                             : check_SPL_parameter(parameter);

    if (changed)
      recalculate(parameter);

    return changed;
  }

}  // namespace ana::ic
