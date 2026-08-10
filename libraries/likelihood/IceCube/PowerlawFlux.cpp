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
      using Scalar = R;
      R   eff_norm;   // only reaches the per-event weights; the histogram is scaled on the host
      R   log_eref;
      R   exponent;   // single PL: ref_index - gamma. Broken PL: gamma_1.
      int write_pe;
      R   gamma_2;    // broken PL only
      R   log_ebreak; // broken PL only: log(E_break)
      R   pivot;      // broken PL only: the 100 TeV renormalisation
      int broken;     // 0 = single power law, 1 = broken power law
    };

    // One group per *chunk* of the CSR-sorted sample (see ICSample's chunk
    // decomposition and GpuBinReduce); grid-stride sum + in-group tree
    // reduction, writing one partial per chunk that bin_gather then sums per
    // bin. Buffer order matches the GpuSession convention: inputs (e_true,
    // baseline, chunk_offsets), params, partial, per_event.
    constexpr const char* kKernelMetalBody = R"METAL(
      struct PowerlawParams {
        float eff_norm; float log_eref; float exponent; int write_pe;
        float gamma_2; float log_ebreak; float pivot; int broken;
      };

      kernel void powerlaw_hist(
          device const float*      e_true        [[buffer(0)]],
          device const float*      log_e_true    [[buffer(1)]],
          device const float*      baseline      [[buffer(2)]],
          device const uint*       chunk_offsets [[buffer(3)]],
          constant PowerlawParams& p             [[buffer(4)]],
          device float*            partial       [[buffer(5)]],
          device float*            per_event     [[buffer(6)]],
          uint chunk [[threadgroup_position_in_grid]],
          uint tid [[thread_position_in_threadgroup]])
      {
        const uint start = chunk_offsets[chunk];
        const uint end   = chunk_offsets[chunk + 1];
        float acc = 0.0f;
        float cmp = 0.0f;
        for (uint i = start + tid; i < end; i += kThreadsPerGroup) {
          const float loge = log_e_true[i];
          float w;
          if (p.broken) {
            const float e     = e_true[i];
            const float lx    = loge - p.log_ebreak;
            const float shape = exp(-(lx < 0.0f ? p.exponent : p.gamma_2) * lx);
            const float undo  = e * 1.0e-5f;
            w = baseline[i] * p.pivot * shape * undo * undo;
          } else {
            w = baseline[i] * exp(p.exponent * (loge - p.log_eref));
          }
          if (p.write_pe) per_event[i] = p.eff_norm * w;
          neumaier_add(acc, cmp, w);
        }
        acc += cmp;
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

    // CUDA-C twin of the MSL kernel: one block per chunk, 256 threads,
    // block-stride sum + __shared__ tree reduction. extern "C" so
    // cuModuleGetFunction resolves the unmangled name. The params struct is
    // passed by value at the same position the Metal kernel takes its constant
    // buffer. Written against a generic scalar `real` and power macro `RPOW`;
    // cuda_kernel_source() prepends the typedef selecting float (FP32) or double
    // (FP64), so one body serves both precisions.
    constexpr const char* kKernelCudaBody = R"CUDA(
      struct PowerlawParams {
        real eff_norm; real log_eref; real exponent; int write_pe;
        real gamma_2; real log_ebreak; real pivot; int broken;
      };

      extern "C" __global__ void powerlaw_hist(
          const real*         e_true,
          const real*         log_e_true,
          const real*         baseline,
          const unsigned int* chunk_offsets,
          PowerlawParams      p,
          real*               partial,
          real*               per_event)
      {
        const unsigned int chunk    = blockIdx.x;
        const unsigned int tid      = threadIdx.x;
        const unsigned int nthreads = blockDim.x;
        const unsigned int start    = chunk_offsets[chunk];
        const unsigned int end      = chunk_offsets[chunk + 1];
        real acc = 0.0;
        real cmp = 0.0;
        for (unsigned int i = start + tid; i < end; i += nthreads) {
          const real loge = log_e_true[i];
          real w;
          if (p.broken) {
            const real e     = e_true[i];
            const real lx    = loge - p.log_ebreak;
            const real shape = REXP(-(lx < 0.0 ? p.exponent : p.gamma_2) * lx);
            const real undo  = e * (real)1.0e-5;
            w = baseline[i] * p.pivot * shape * undo * undo;
          } else {
            w = baseline[i] * REXP(p.exponent * (loge - p.log_eref));
          }
          if (p.write_pe) per_event[i] = p.eff_norm * w;
          neumaier_add(acc, cmp, w);
        }
        acc += cmp;
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
    m_ShapeHistogram.assign(binning.total_bins(), 0.0);
    // On the GPU path the per-event weights live in a GPU buffer (m_hPerEvent)
    // consumed by SampleLikelihood's say_ssq kernel; no CPU copy is kept. The
    // CPU loop always writes them, so the CPU path always allocates.
    if (!m_Gpu)
      m_PerEventWeight.assign(sample.size(), 0.0);

    if (m_Gpu) {
      const std::size_t M = sample.size();
      const std::string src =
          gpu_kernel_source(m_Gpu->language(), m_Gpu->is_fp64(), kKernelMetalBody, kKernelCudaBody);
      m_Gpu->ensure_kernel("powerlaw_hist", src.c_str());
      m_hETrue    = m_Gpu->upload_column(sample.e_true.data(), M);
      m_hLogETrue = m_Gpu->upload_column(sample.log_e_true.data(), M);
      m_hBaseline = m_Gpu->upload_column(sample.astro_baseline.data(), M);
      m_Reduce.emplace(m_Gpu, sample, m_Histogram.size());
      m_hHist     = m_Gpu->alloc_output(m_Histogram.size());
      // The per-event weights are read only by say_ssq, never on the host.
      m_hPerEvent = m_NeedPerEvent ? m_Gpu->alloc_output(M, /*readback=*/false) : -1;
      std::cout << "PowerlawFlux: using GPU backend (" << m_Reduce->n_chunks() << " chunks over "
                << m_Histogram.size() << " bins)\n";
    }
  }

  inline void recalculate_cpu_SPL(const io::ic::ICSample& sample,
                                  const double eff_norm,
                                  const double exponent,
                                  const double Eref,
                                  std::span<double> shape_histogram,
                                  std::span<double> PerEventWeight,
                                  const bool use_multithreading) noexcept {

    const auto& off      = sample.bin_offsets;
    const auto& baseline = sample.astro_baseline;
    const auto& log_e    = sample.log_e_true;
    const int n_bins     = static_cast<int>(shape_histogram.size());
    // (E/E_ref)^exponent == exp(exponent * (log E - log E_ref)), and log E is a
    // property of the event, taken once at load time (ICSample::log_e_true).
    // What is left per event and per evaluation is one exp instead of a pow,
    // which is a log and an exp.
    const double log_eref = std::log(Eref);

    #pragma omp parallel for schedule(guided) if (use_multithreading)
    for (int bin = 0; bin < n_bins; ++bin) {
      double acc = 0.0;
      for (std::size_t i = off[bin], n = off[bin + 1]; i < n; ++i) {
        const double w = baseline[i] * std::exp(exponent * (log_e[i] - log_eref));
        acc += w;
        PerEventWeight[i] = eff_norm * w;
      }
      shape_histogram[bin] = acc;
    }
  }

  inline void recalculate_cpu_BPL(const io::ic::ICSample& sample,
                                  const double eff_norm,
                                  const double g1,
                                  const double g2,
                                  const double e_break,
                                  const double pivot,
                                  std::span<double> shape_histogram,
                                  std::span<double> PerEventWeight,
                                  const bool use_multithreading) noexcept {

    const auto& off      = sample.bin_offsets;
    const auto& baseline = sample.astro_baseline;
    const auto& e_true   = sample.e_true;
    const auto& log_e    = sample.log_e_true;
    const int n_bins     = static_cast<int>(shape_histogram.size());
    // (E/E_break)^-g == exp(-g * (log E - log E_break)); the branch on which
    // side of the break the event falls is the sign of that same difference.
    const double log_ebreak = std::log(e_break);

    #pragma omp parallel for schedule(guided) if (use_multithreading)
    for (int bin = 0; bin < n_bins; ++bin) {
      double acc = 0.0;
      for (std::size_t i = off[bin], n = off[bin + 1]; i < n; ++i) {
        const double e     = e_true[i];
        const double lx    = log_e[i] - log_ebreak;
        const double shape = std::exp(-(lx < 0.0 ? g1 : g2) * lx);
        // Undo the baseline column's built-in E^-2: `powerlaw` is
        // fluxless_weight * 1e-18 * (E/1e5)^-2, so the 1e-18 cancels against
        // AstroBPL's own factor and (E/1e5)^2 remains. The 1e5 is a property of
        // that column, not the configurable ERefGeV.
        const double undo = (e / 1.0e5) * (e / 1.0e5);
        const double w    = baseline[i] * pivot * shape * undo;
        acc += w;
        PerEventWeight[i] = eff_norm * w;
      }
      shape_histogram[bin] = acc;
    }
    return;
  }

  double PowerlawFlux::effective_norm(const ParameterWrapper& parameter) const noexcept {
    const double norm = parameter[params::ic::AstroNorm];
    return m_PerTypeNorm ? norm : 0.5 * norm;
  }

  void PowerlawFlux::apply_norm(const ParameterWrapper& parameter) noexcept {
    const double eff_norm = effective_norm(parameter);
    for (std::size_t bin = 0, n = m_Histogram.size(); bin < n; ++bin)
      m_Histogram[bin] = eff_norm * m_ShapeHistogram[bin];
  }

  void PowerlawFlux::recalculate_cpu(const ParameterWrapper& parameter) noexcept {
    using namespace params::ic;

    const double eff_norm = effective_norm(parameter);

    switch (m_Model) {
      case io::ic::AstroModel::Powerlaw: {
        const double gamma = parameter[SpectralIndex];

        recalculate_cpu_SPL(m_Sample,
                            eff_norm,
                            m_ReferenceIndex - gamma,
                            m_ERef,
                            m_ShapeHistogram,
                            m_PerEventWeight,
                            m_UseMultiThreading);
        break;
      }
      case io::ic::AstroModel::BrokenPowerlaw: {
        const double g1      = parameter[AstroGamma1];
        const double g2      = parameter[AstroGamma2];
        const double e_break = std::pow(10, parameter[AstroEBreak]);
        const double pivot   = (1.0e5 < e_break) ? std::pow(1.0e5 / e_break, g1)
                                                 : std::pow(1.0e5 / e_break, g2);

        recalculate_cpu_BPL(m_Sample,
                            eff_norm,
                            g1,
                            g2,
                            e_break,
                            pivot,
                            m_ShapeHistogram,
                            m_PerEventWeight,
                            m_UseMultiThreading);
        break;
      }
      default:
        return;
    }

    apply_norm(parameter);
  }

  void PowerlawFlux::recalculate_gpu(const ParameterWrapper& parameter) noexcept {
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

    // Fill either a float or a double params struct from the double host
    // scalars, matching the kernel's `real` precision. The kernel reuses
    // `exponent` as gamma_1 in broken-power-law mode.
    const auto fill = [&](auto& p) {
      using R      = std::decay_t<decltype(p.eff_norm)>;
      p.eff_norm   = static_cast<R>(effective_norm(parameter));
      p.log_eref   = static_cast<R>(std::log(m_ERef));
      p.exponent   = static_cast<R>(broken ? g1 : m_ReferenceIndex - gamma);
      p.write_pe   = m_NeedPerEvent ? 1 : 0;
      p.gamma_2    = static_cast<R>(g2);
      p.log_ebreak = static_cast<R>(std::log(e_break));
      p.pivot      = static_cast<R>(pivot);
      p.broken     = broken ? 1 : 0;
    };

    // Chunk-parallel: the kernel reduces each chunk to a partial, then
    // gather() sums each bin's partials into m_hHist (see GpuBinReduce). What
    // comes back is the norm-independent shape histogram; the norm is applied
    // on the host, which is what makes a norm-only step skip this dispatch
    // entirely (see recalculate()).
    const int inputs[] = {m_hETrue, m_hLogETrue, m_hBaseline, m_Reduce->chunk_offsets()};
    if (m_Gpu->is_fp64())
      dispatch_and_gather_hist<PowerlawParamsT<double>>(*m_Gpu, "powerlaw_hist", inputs, 4, fill, *m_Reduce,
                                                         m_hPerEvent, m_hHist, m_ShapeHistogram);
    else
      dispatch_and_gather_hist<PowerlawParamsT<float>>(*m_Gpu, "powerlaw_hist", inputs, 4, fill, *m_Reduce,
                                                        m_hPerEvent, m_hHist, m_ShapeHistogram);
    // Per-event weights stay GPU-resident (read by the say_ssq kernel).
    apply_norm(parameter);
  }

  void PowerlawFlux::recalculate(const ParameterWrapper& parameter) noexcept {
    if (m_Gpu) {
      recalculate_gpu(parameter);
      return;
    }
    recalculate_cpu(parameter);
  }

  // The shape parameters only -- AstroNorm is handled separately, since a step
  // in it needs no sweep over the events (see check_and_recalculate).
  inline bool check_SPL_shape_parameter(const ParameterWrapper& parameter) noexcept {
    using namespace params::ic;
    return parameter.check_parameter_changed(SpectralIndex);
  }

  inline bool check_BPL_shape_parameter(const ParameterWrapper& parameter) noexcept {
    using namespace params::ic;
    return parameter.check_parameter_changed(AstroGamma1)
           || parameter.check_parameter_changed(AstroGamma2)
           || parameter.check_parameter_changed(AstroEBreak);
  }

  bool PowerlawFlux::check_and_recalculate(const ParameterWrapper& parameter) {
    using namespace params::ic;
    // Watch exactly the parameters the active model reads: SpectralIndex is
    // unused in broken-power-law mode, and the three AstroBPL parameters are
    // unused in single-power-law mode.

    using enum io::ic::AstroModel;

    const bool shape_changed = (m_Model == BrokenPowerlaw)
                                   ? check_BPL_shape_parameter(parameter)
                                   : check_SPL_shape_parameter(parameter);
    const bool norm_changed  = parameter.check_parameter_changed(AstroNorm);

    if (!m_Seeded) {
      recalculate(parameter);
      m_Seeded = true;
      return true;
    }

    if (!shape_changed && !norm_changed)
      return false;

    // AstroNorm multiplies every event's weight by the same factor, so when it
    // is the only parameter that moved the histogram is m_ShapeHistogram times
    // the new norm -- a pass over the bins instead of a pass over the events
    // (10^7 of them on the tracks sample). Minuit varies one parameter at a
    // time to build its numerical gradient, so this is a whole gradient
    // component that costs O(bins) rather than O(events).
    //
    // The per-event weights are the exception: the SAY ssq reduction reads them
    // and they carry the norm, so a sample that needs them has to re-run the
    // event loop even for a norm-only step.
    if (!shape_changed && !m_NeedPerEvent) {
      apply_norm(parameter);
      return true;
    }

    recalculate(parameter);
    return true;
  }

}  // namespace ana::ic
