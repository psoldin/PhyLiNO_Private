#include "SampleLikelihood.h"

#include "SAYLikelihood.h"

#include <algorithm>
#include <cmath>

namespace ana::ic {

  namespace {

    template <typename T>
    T square(T&& t) noexcept {
      return t * t;
    }

    // SAY ssq reduction on the GPU: sums (astro_i + atmo_i)^2 per analysis bin
    // over the per-event weight buffers the flux kernels already produced, so
    // the weights never leave the GPU. Same one-group-per-bin / 256-thread
    // tree-reduction layout as the flux kernels; buffer order follows the
    // GpuBackend convention: inputs (astro_pe, atmo_pe, bin_offsets), params,
    // hist (the ssq output); no per_event output.
    struct SsqParams {
      int has_astro;
      int has_atmo;
    };

    constexpr const char* kSsqKernelMetal = R"METAL(
      #include <metal_stdlib>
      using namespace metal;

      struct SsqParams { int has_astro; int has_atmo; };
      constant uint kThreadsPerGroup = 256;

      kernel void say_ssq(
          device const float* astro_pe    [[buffer(0)]],
          device const float* atmo_pe     [[buffer(1)]],
          device const uint*  bin_offsets [[buffer(2)]],
          constant SsqParams& p           [[buffer(3)]],
          device float*       ssq         [[buffer(4)]],
          uint bin [[threadgroup_position_in_grid]],
          uint tid [[thread_position_in_threadgroup]])
      {
        const uint start = bin_offsets[bin];
        const uint end   = bin_offsets[bin + 1];
        float acc = 0.0f;
        for (uint i = start + tid; i < end; i += kThreadsPerGroup) {
          float w = 0.0f;
          if (p.has_astro) w += astro_pe[i];
          if (p.has_atmo)  w += atmo_pe[i];
          acc += w * w;
        }
        threadgroup float shared[kThreadsPerGroup];
        shared[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = kThreadsPerGroup / 2; s > 0; s >>= 1) {
          if (tid < s) shared[tid] += shared[tid + s];
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) ssq[bin] = shared[0];
      }
    )METAL";

    constexpr const char* kSsqKernelCuda = R"CUDA(
      struct SsqParams { int has_astro; int has_atmo; };

      extern "C" __global__ void say_ssq(
          const float*        astro_pe,
          const float*        atmo_pe,
          const unsigned int* bin_offsets,
          SsqParams           p,
          float*              ssq)
      {
        const unsigned int bin      = blockIdx.x;
        const unsigned int tid      = threadIdx.x;
        const unsigned int nthreads = blockDim.x;
        const unsigned int start    = bin_offsets[bin];
        const unsigned int end      = bin_offsets[bin + 1];
        float acc = 0.0f;
        for (unsigned int i = start + tid; i < end; i += nthreads) {
          float w = 0.0f;
          if (p.has_astro) w += astro_pe[i];
          if (p.has_atmo)  w += atmo_pe[i];
          acc += w * w;
        }
        __shared__ float sdata[256];
        sdata[tid] = acc;
        __syncthreads();
        for (unsigned int s = nthreads / 2; s > 0; s >>= 1) {
          if (tid < s) sdata[tid] += sdata[tid + s];
          __syncthreads();
        }
        if (tid == 0) ssq[bin] = sdata[0];
      }
    )CUDA";

    static double calculate_poisson_likelihood(const std::span<const double> data,
                                               const std::span<const double> prediction) noexcept {
      double llh = 0.0;
      for (std::size_t i = 0, n = data.size(); i < n; ++i) {
        const double model = prediction[i];
        const double obs   = data[i];
        if (model <= 0.0)
          continue;
        llh += obs * std::log(model) - model;
      }
      return -2.0 * llh;
    }

    static double calculate_say_likelihood(const std::span<const double> data,
                                           const std::span<const double> prediction,
                                           const std::span<const double> ssq) noexcept {
      double llh = 0.0;
      for (std::size_t i = 0, n = data.size(); i < n; ++i) {
        llh += say_bin_log_likelihood(data[i], prediction[i], ssq[i]);
      }
      return -2.0 * llh;
    }

  }  // namespace

  SampleLikelihood::SampleLikelihood(const io::ic::ICSample&     sample,
                                     const io::ic::SampleConfig& cfg,
                                     const GlobalFluxSettings&   settings,
                                     std::shared_ptr<GpuBackend> gpu,
                                     const bool                  use_say)
    : m_Sample(sample)
    , m_Config(cfg)
    , m_UseSAY(use_say) {
    if (cfg.wants_astro())
      m_Astro.emplace(sample,
                      cfg.binning,
                      settings.e_ref_gev,
                      settings.astro_reference_index,
                      settings.astro_per_type_norm,
                      gpu,
                      use_say);

    if (cfg.wants_atmospheric())
      m_Atmo.emplace(sample,
                     cfg.binning,
                     settings.conv_delta_gamma_e_ref,
                     settings.prompt_delta_gamma_e_ref,
                     gpu,
                     use_say,
                     cfg.wants_veto(),
                     settings.veto_anchor_energy,
                     settings.veto_rescale_energy);

    if (cfg.wants_template())
      m_Template.emplace(cfg.binning, cfg.template_file, cfg.template_norm_index, cfg.livetime);

    if (!cfg.gradient_file.empty())
      m_Systematics.emplace(cfg.binning, cfg.gradient_file);

    const int total_bins = cfg.binning.total_bins();
    m_Predicted.assign(total_bins, 0.0);
    m_Data.assign(total_bins, 0.0);
    m_Ssq.assign(total_bins, 0.0);

    // SAY-on-GPU setup. Both flux kernels leave their per-event weights in GPU
    // buffers; the say_ssq kernel reduces them to the per-bin ssq without a
    // round trip through the CPU. upload_offsets deduplicates against the flux
    // components' own upload, so this returns the already-resident buffer.
    const bool gpu_per_event =
        (m_Astro && m_Astro->per_event_handle() >= 0) || (m_Atmo && m_Atmo->per_event_handle() >= 0);
    if (gpu && use_say && gpu_per_event) {
      m_Gpu = std::move(gpu);
      const char* src = m_Gpu->language() == GpuLanguage::Cuda ? kSsqKernelCuda : kSsqKernelMetal;
      m_Gpu->ensure_kernel("say_ssq", src);
      m_hSsqOffsets = m_Gpu->upload_offsets(sample.bin_offsets.data(), sample.bin_offsets.size());
      m_hSsq        = m_Gpu->alloc_output(total_bins);
    }
  }

  bool SampleLikelihood::assemble_prediction(const ParameterWrapper& parameter) {
    // Components stay sequential within a sample: a Migrad step varies one
    // parameter, so at most one component is stale per call and running the
    // checks concurrently was measured to gain nothing (Metal) / <5% (CPU).
    // Cross-sample concurrency lives in ICLikelihood::calculate_likelihood.
    bool changed = false;
    if (m_Astro) changed |= m_Astro->check_and_recalculate(parameter);
    if (m_Atmo) changed |= m_Atmo->check_and_recalculate(parameter);
    if (m_Template) changed |= m_Template->check_and_recalculate(parameter);
    if (m_Systematics) changed |= m_Systematics->check_and_recalculate(parameter);

    const std::span<const double> astro     = m_Astro ? m_Astro->histogram() : std::span<const double>{};
    const std::span<const double> atmo      = m_Atmo ? m_Atmo->histogram() : std::span<const double>{};
    const std::span<const double> tmpl      = m_Template ? m_Template->histogram() : std::span<const double>{};
    const std::span<const double> mu_delta  = m_Systematics ? m_Systematics->mu_delta() : std::span<const double>{};

    for (std::size_t b = 0, n = m_Predicted.size(); b < n; ++b) {
      double total = 0.0;
      if (!astro.empty()) total += astro[b];
      if (!atmo.empty()) total += atmo[b];
      if (!tmpl.empty()) total += tmpl[b];
      if (!mu_delta.empty()) total += mu_delta[b];
      // Clip at zero to avoid unphysical predictions (matches NNMFit mu clip),
      // applied after the gradient delta -- NNMFit clips mu_tot, not the
      // pre-gradient sum.
      m_Predicted[b] = std::max(0.0, total);
    }

    return changed;
  }

  void SampleLikelihood::assemble_fluctuation() {
    // Combined per-event weight across astro+conv+prompt, squared, then binned
    // -- matches NNMFit's rule that same-event components must be summed
    // *before* squaring (NNMFit/core/histogram_builder.py:229-329), since
    // (w1+w2)^2 != w1^2 + w2^2.
    const int n_bins = static_cast<int>(m_Predicted.size());

    if (m_hSsq >= 0) {
      // GPU path: reduce the flux kernels' per-event weight buffers in place.
      // A component that is absent binds the offsets buffer as a dummy (never
      // read, its flag is 0) -- the backend needs a valid buffer at every slot.
      const int astro_h = m_Astro ? m_Astro->per_event_handle() : -1;
      const int atmo_h  = m_Atmo ? m_Atmo->per_event_handle() : -1;

      const SsqParams p{.has_astro = astro_h >= 0 ? 1 : 0, .has_atmo = atmo_h >= 0 ? 1 : 0};
      const int       inputs[] = {astro_h >= 0 ? astro_h : m_hSsqOffsets,
                                  atmo_h >= 0 ? atmo_h : m_hSsqOffsets,
                                  m_hSsqOffsets};
      m_Gpu->dispatch("say_ssq", inputs, 3, &p, sizeof(p), m_hSsq, -1, static_cast<std::size_t>(n_bins));

      const float* ssq = m_Gpu->contents(m_hSsq);
      for (int b = 0; b < n_bins; ++b)
        m_Ssq[b] = static_cast<double>(ssq[b]);
    } else {
      const std::span<const double> astro = m_Astro ? m_Astro->per_event_weight() : std::span<const double>{};
      const std::span<const double> atmo  = m_Atmo ? m_Atmo->per_event_weight() : std::span<const double>{};
      const auto&                   off   = m_Sample.bin_offsets;

      // The component test is hoisted out of the per-event loop: which components
      // exist is fixed at construction, so each case gets its own tight loop.
      auto accumulate = [&](auto event_weight) {
#pragma omp parallel for
        for (int b = 0; b < n_bins; ++b) {
          double acc = 0.0;
#pragma omp simd reduction(+ : acc)
          for (std::size_t i = off[b]; i < off[b + 1]; ++i) {
            acc += square(event_weight(i));
          }
          m_Ssq[b] = acc;
        }
      };

      if (!astro.empty() && !atmo.empty())
        accumulate([&](const std::size_t i) { return astro[i] + atmo[i]; });
      else if (!astro.empty())
        accumulate([&](const std::size_t i) { return astro[i]; });
      else if (!atmo.empty())
        accumulate([&](const std::size_t i) { return atmo[i]; });
      else
        std::ranges::fill(m_Ssq, 0.0);
    }

    // Histogram-level fluctuation from the template component, added after the
    // per-event sum (NNMFit: ssq += (hist_fluctuation * livetime)**2).
    if (m_Template) {
      const std::span<const double> tmpl_ssq = m_Template->fluctuation();
      for (int b = 0; b < n_bins; ++b) m_Ssq[b] += tmpl_ssq[b];
    }

    // Histogram-level fluctuation from the SnowStorm detector gradients.
    if (m_Systematics) {
      const std::span<const double> sys_ssq = m_Systematics->ssq_delta();
      for (int b = 0; b < n_bins; ++b) m_Ssq[b] += sys_ssq[b];
    }
  }

  void SampleLikelihood::set_data(const std::span<const double> counts) {
    if (counts.size() != m_Data.size())
      throw std::runtime_error("SampleLikelihood: data histogram for sample '" + m_Config.name +
                               "' has " + std::to_string(counts.size()) + " bins, the binning has " +
                               std::to_string(m_Data.size()));
    std::ranges::copy(counts, m_Data.begin());

    // The SAY ssq describes MC statistics, so it still comes from the model; seed
    // it exactly as generate_asimov does.
    if (m_UseSAY) assemble_fluctuation();
  }

  void SampleLikelihood::generate_asimov(const ParameterWrapper& nominal) {
    assemble_prediction(nominal);
    std::ranges::copy(m_Predicted, m_Data.begin());

    // Seed the ssq histogram at the nominal point. partial_llh() only refreshes
    // it when a flux actually recalculated, and the minimizer's first evaluation
    // is at the start values -- which compare equal to the nominal set here, so
    // nothing would recalculate and SAY would run that evaluation with ssq == 0
    // (silently degenerating to plain Poisson).
    if (m_UseSAY)
      assemble_fluctuation();
  }

  double SampleLikelihood::partial_llh(const ParameterWrapper& parameter) {
    const bool changed = assemble_prediction(parameter);

    if (m_UseSAY) {
      if (changed)
        assemble_fluctuation();
      return calculate_say_likelihood(m_Data, m_Predicted, m_Ssq);
    }

    return calculate_poisson_likelihood(m_Data, m_Predicted);
  }

}  // namespace ana::ic
