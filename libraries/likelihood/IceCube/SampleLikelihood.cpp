#include "SampleLikelihood.h"

#include "SAYLikelihood.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ana::ic {

  namespace {

    // NNMFit's GalacticTemplate defines no fluctuation graph, so it is excluded
    // from the sigma^2 sum (histogram_builder.py:307) and the exporter writes an
    // all-zero fluctuation column. SampleLikelihood relies on that: it neither
    // adds the galactic templates to sigma^2 nor re-runs the ssq reduction when a
    // galactic norm moves. A file that carried non-zero fluctuations would
    // silently diverge from the reference, so reject it at construction.
    //
    // Checked against the file's own second column rather than
    // TemplateFlux::fluctuation(), which holds the SQUARED, norm-scaled values
    // and is still all zeros until the first check_and_recalculate().
    void require_zero_fluctuation_column(const std::string& path) {
      std::ifstream in(path);
      if (!in)
        throw std::runtime_error("SampleLikelihood: cannot open galactic template file '" + path + "'");

      std::string line;
      int         bin = 0;
      while (std::getline(in, line)) {
        if (line.empty() || line.front() == '#') continue;
        std::istringstream row(line);
        double             rate = 0.0, fluctuation = 0.0;
        if (!(row >> rate >> fluctuation)) continue;
        if (fluctuation != 0.0)
          throw std::runtime_error(
              "SampleLikelihood: galactic template '" + path + "' has a non-zero fluctuation " +
              std::to_string(fluctuation) + " in bin " + std::to_string(bin) +
              "; NNMFit's GalacticTemplate defines no fluctuation graph and is excluded from the "
              "sigma^2 sum, so a non-zero column would silently diverge from the reference");
        ++bin;
      }
    }

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
    , m_UseSAY(use_say)
    , m_UseMultiThreading(settings.use_multi_threading) {
    if (cfg.wants_astro())
      m_Astro.emplace(sample,
                      cfg.mc_binning,
                      settings.e_ref_gev,
                      settings.astro_reference_index,
                      settings.astro_per_type_norm,
                      gpu,
                      use_say,
                      settings.astro_model,
                      settings.use_multi_threading);

    if (cfg.wants_atmospheric())
      m_Atmo.emplace(sample,
                     cfg.mc_binning,
                     settings.conv_delta_gamma_e_ref,
                     settings.prompt_delta_gamma_e_ref,
                     gpu,
                     use_say,
                     cfg.wants_veto(),
                     settings.veto_anchor_energy,
                     settings.veto_rescale_energy,
                     settings.use_multi_threading);

    if (cfg.wants_template())
      m_Template.emplace(cfg.mc_binning, cfg.template_file, cfg.template_norm_index, cfg.livetime);

    if (!cfg.gradient_file.empty())
      m_Systematics.emplace(cfg.mc_binning, cfg.gradient_file);

    // Galactic templates are already 3D: NNMFit's histogram components are never
    // binned, so the exported file carries the analysis binning's RA structure.
    for (const io::ic::GalacticTemplateConfig& galactic : cfg.galactic) {
      require_zero_fluctuation_column(galactic.file);
      m_Galactic.emplace_back(cfg.binning, galactic.file, galactic.norm_index, cfg.livetime);
    }

    const int total_bins = cfg.binning.total_bins();
    const int mc_bins    = cfg.mc_binning.total_bins();
    m_RaBins             = cfg.ra_bins();

    m_Predicted.assign(total_bins, 0.0);
    m_Data.assign(total_bins, 0.0);
    m_Ssq.assign(total_bins, 0.0);
    m_GalacticTotal.assign(total_bins, 0.0);
    m_McTotal.assign(mc_bins, 0.0);
    m_McSsq.assign(mc_bins, 0.0);

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
      m_hSsq        = m_Gpu->alloc_output(mc_bins);
    }
  }

  bool SampleLikelihood::assemble_prediction(const ParameterWrapper& parameter) {
    // Components stay sequential within a sample: a Migrad step varies one
    // parameter, so at most one component is stale per call and running the
    // checks concurrently was measured to gain nothing (Metal) / <5% (CPU).
    // Cross-sample concurrency lives in ICLikelihood::calculate_likelihood.
    //
    // The returned flag says whether anything that feeds sigma^2 changed; it is
    // what partial_llh() uses to decide whether to re-run assemble_fluctuation().
    bool ssq_changed = false;
    if (m_Astro) ssq_changed |= m_Astro->check_and_recalculate(parameter);
    if (m_Atmo) ssq_changed |= m_Atmo->check_and_recalculate(parameter);
    if (m_Template) ssq_changed |= m_Template->check_and_recalculate(parameter);
    if (m_Systematics) ssq_changed |= m_Systematics->check_and_recalculate(parameter);

    // The galactic templates recalculate here too -- the prediction below reads
    // their histograms -- but deliberately do NOT set ssq_changed: NNMFit's
    // GalacticTemplate defines no fluctuation graph and is excluded from the ssq
    // sum (histogram_builder.py:307), and the constructor enforces the zero
    // fluctuation column that guarantees it. A moved galactic norm therefore
    // provably cannot change sigma^2, and re-running assemble_fluctuation for it
    // would cost a full reduction (a GPU dispatch over every bin) for nothing.
    for (TemplateFlux& galactic : m_Galactic)
      static_cast<void>(galactic.check_and_recalculate(parameter));

    const std::span<const double> astro     = m_Astro ? m_Astro->histogram() : std::span<const double>{};
    const std::span<const double> atmo      = m_Atmo ? m_Atmo->histogram() : std::span<const double>{};
    const std::span<const double> tmpl      = m_Template ? m_Template->histogram() : std::span<const double>{};
    const std::span<const double> mu_delta  = m_Systematics ? m_Systematics->mu_delta() : std::span<const double>{};

    for (std::size_t b = 0, n = m_McTotal.size(); b < n; ++b) {
      double total = 0.0;
      if (!astro.empty()) total += astro[b];
      if (!atmo.empty()) total += atmo[b];
      if (!tmpl.empty()) total += tmpl[b];
      if (!mu_delta.empty()) total += mu_delta[b];
      m_McTotal[b] = total;
    }

    // Spread the MC-binned total over the RA axis (NNMFit Binning_2D_to_3D:
    // repeat(mu, n_ra) / n_ra). n_ra == 1 is an exact copy.
    io::ic::broadcast_over_ra(m_McTotal, m_RaBins, static_cast<double>(m_RaBins), m_Predicted);

    // Galactic templates are already in the analysis binning, so they are added
    // after the broadcast and are NOT divided.
    if (!m_Galactic.empty()) {
      std::ranges::fill(m_GalacticTotal, 0.0);
      for (TemplateFlux& galactic : m_Galactic) {
        const std::span<const double> h = galactic.histogram();
        for (std::size_t b = 0, n = m_GalacticTotal.size(); b < n; ++b) m_GalacticTotal[b] += h[b];
      }
      for (std::size_t b = 0, n = m_Predicted.size(); b < n; ++b) m_Predicted[b] += m_GalacticTotal[b];
    }

    // Clip at zero to avoid unphysical predictions (matches NNMFit's mu clip),
    // applied to the final bin content -- NNMFit clips mu_tot, not the
    // pre-gradient sum.
    for (double& value : m_Predicted) value = std::max(0.0, value);

    return ssq_changed;
  }

  void SampleLikelihood::assemble_fluctuation() {
    // Combined per-event weight across astro+conv+prompt, squared, then binned
    // -- matches NNMFit's rule that same-event components must be summed
    // *before* squaring (NNMFit/core/histogram_builder.py:229-329), since
    // (w1+w2)^2 != w1^2 + w2^2.
    const int n_bins = static_cast<int>(m_McSsq.size());

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
        m_McSsq[b] = static_cast<double>(ssq[b]);
    } else {
      const std::span<const double> astro = m_Astro ? m_Astro->per_event_weight() : std::span<const double>{};
      const std::span<const double> atmo  = m_Atmo ? m_Atmo->per_event_weight() : std::span<const double>{};
      const auto&                   off   = m_Sample.bin_offsets;

      // The component test is hoisted out of the per-event loop: which components
      // exist is fixed at construction, so each case gets its own tight loop.
      auto accumulate = [&](auto event_weight) {
#pragma omp parallel for if(m_UseMultiThreading)
        for (int b = 0; b < n_bins; ++b) {
          double acc = 0.0;
#pragma omp simd reduction(+ : acc)
          for (std::size_t i = off[b]; i < off[b + 1]; ++i) {
            acc += square(event_weight(i));
          }
          m_McSsq[b] = acc;
        }
      };

      if (!astro.empty() && !atmo.empty())
        accumulate([&](const std::size_t i) { return astro[i] + atmo[i]; });
      else if (!astro.empty())
        accumulate([&](const std::size_t i) { return astro[i]; });
      else if (!atmo.empty())
        accumulate([&](const std::size_t i) { return atmo[i]; });
      else
        std::ranges::fill(m_McSsq, 0.0);
    }

    // Histogram-level fluctuation from the template component, added after the
    // per-event sum (NNMFit: ssq += (hist_fluctuation * livetime)**2).
    if (m_Template) {
      const std::span<const double> tmpl_ssq = m_Template->fluctuation();
      for (int b = 0; b < n_bins; ++b) m_McSsq[b] += tmpl_ssq[b];
    }

    // Histogram-level fluctuation from the SnowStorm detector gradients.
    if (m_Systematics) {
      const std::span<const double> sys_ssq = m_Systematics->ssq_delta();
      for (int b = 0; b < n_bins; ++b) m_McSsq[b] += sys_ssq[b];
    }

    // sigma^2 is a squared quantity, so the RA divisor is squared too (NNMFit
    // Binning_2D_to_3D.make_binned_flux with is_ssq_calc=True). The galactic
    // templates contribute nothing here: NNMFit's GalacticTemplate defines no
    // fluctuation graph and is excluded from the ssq sum.
    io::ic::broadcast_over_ra(m_McSsq, m_RaBins,
                              static_cast<double>(m_RaBins) * static_cast<double>(m_RaBins), m_Ssq);
  }

  std::vector<double> SampleLikelihood::in_analysis_bins(const std::span<const double> values) const {
    if (values.empty()) return {};

    // A component's histogram is binned either in the analysis binning or in the
    // MC binning, and the two are told apart purely by size. That is exact only
    // because every caller passes one of this sample's own component histograms:
    // size == m_Predicted.size() iff analysis-binned, size == m_McTotal.size()
    // iff MC-binned. (When the sample has no RA axis the two coincide and the
    // copy branch is taken, which is the same answer the broadcast would give.)
    if (values.size() == m_Predicted.size()) return std::vector<double>(values.begin(), values.end());
    assert(values.size() == m_McTotal.size() &&
           "in_analysis_bins: span is binned in neither the analysis nor the MC binning");

    std::vector<double> out(m_Predicted.size(), 0.0);
    io::ic::broadcast_over_ra(values, m_RaBins, static_cast<double>(m_RaBins), out);
    return out;
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
    const bool ssq_changed = assemble_prediction(parameter);

    if (m_UseSAY) {
      if (ssq_changed)
        assemble_fluctuation();
      return calculate_say_likelihood(m_Data, m_Predicted, m_Ssq);
    }

    return calculate_poisson_likelihood(m_Data, m_Predicted);
  }

}  // namespace ana::ic
