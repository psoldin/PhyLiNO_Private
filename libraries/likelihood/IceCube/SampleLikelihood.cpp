#include "SampleLikelihood.h"

#include "PoissonLikelihood.h"
#include "SAYLikelihood.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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
    // the weights never leave the GPU. Same one-group-per-chunk / 256-thread
    // tree-reduction layout as the flux kernels, gathered per bin afterwards;
    // buffer order follows the GpuSession convention: inputs (astro_pe,
    // atmo_pe, chunk_offsets), params, partial; no per_event output.
    struct SsqParams {
      int has_astro;
      int has_atmo;
    };

    constexpr const char* kSsqKernelMetalBody = R"METAL(
      struct SsqParams { int has_astro; int has_atmo; };

      kernel void say_ssq(
          device const float* astro_pe      [[buffer(0)]],
          device const float* atmo_pe       [[buffer(1)]],
          device const uint*  chunk_offsets [[buffer(2)]],
          constant SsqParams& p             [[buffer(3)]],
          device float*       partial       [[buffer(4)]],
          uint chunk [[threadgroup_position_in_grid]],
          uint tid [[thread_position_in_threadgroup]])
      {
        const uint start = chunk_offsets[chunk];
        const uint end   = chunk_offsets[chunk + 1];
        float acc = 0.0f;
        float cmp = 0.0f;
        for (uint i = start + tid; i < end; i += kThreadsPerGroup) {
          float w = 0.0f;
          if (p.has_astro) w += astro_pe[i];
          if (p.has_atmo)  w += atmo_pe[i];
          neumaier_add(acc, cmp, w * w);
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

    // Written against a generic scalar `real`; cuda_kernel_source() prepends the
    // typedef selecting float (FP32) or double (FP64). SsqParams is int-only, so
    // it needs no precision variant. The per-event input buffers (astro_pe,
    // atmo_pe) are the flux kernels' outputs, so they are already in the active
    // precision.
    constexpr const char* kSsqKernelCudaBody = R"CUDA(
      struct SsqParams { int has_astro; int has_atmo; };

      extern "C" __global__ void say_ssq(
          const real*         astro_pe,
          const real*         atmo_pe,
          const unsigned int* chunk_offsets,
          SsqParams           p,
          real*               partial)
      {
        const unsigned int chunk    = blockIdx.x;
        const unsigned int tid      = threadIdx.x;
        const unsigned int nthreads = blockDim.x;
        const unsigned int start    = chunk_offsets[chunk];
        const unsigned int end      = chunk_offsets[chunk + 1];
        real acc = 0.0;
        real cmp = 0.0;
        for (unsigned int i = start + tid; i < end; i += nthreads) {
          real w = 0.0;
          if (p.has_astro) w += astro_pe[i];
          if (p.has_atmo)  w += atmo_pe[i];
          neumaier_add(acc, cmp, w * w);
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

    // NNMFit's default Poisson likelihood: per bin, log P(k|mu) with the
    // saturated term log P(k|k) subtracted (LikelihoodBuilder.make_binwise_llh
    // with substract_saturated=True). The subtraction is not cosmetic -- without
    // it the sum carries a bin-count-sized constant that the minimiser has to
    // work against, which is what the removed hardcoded baseline in
    // ICLikelihood was papering over.
    //
    // Sum `term` over [0, n) in parallel without making the result depend on how
    // the threads were scheduled: the range is cut into fixed chunks, each summed
    // sequentially, and the chunk totals are added back in index order. OpenMP's
    // reduction clause would be shorter but its accumulation order varies with
    // the thread count, which would make two scan points incomparable and the
    // ConcurrentSessionsMatchSequential test flaky.
    //
    // The unit summed here is an RA *group* -- one MC bin's whole slice of
    // analysis bins -- not a single analysis bin, because that is the unit the
    // prediction is constant over. `partial` is the caller's scratch buffer so
    // that the loop allocates nothing per evaluation.
    template <class Term>
    double sum_over_groups(const std::size_t n_groups, const std::size_t bins_per_group,
                           const bool multi_threaded, std::vector<double>& partial,
                           Term&& term) noexcept {
      constexpr std::size_t kBinsPerChunk = 4096;
      if (n_groups == 0) return 0.0;

      // Chunks of roughly kBinsPerChunk analysis bins, whatever the group size.
      const std::size_t groups_per_chunk =
          std::max<std::size_t>(1, kBinsPerChunk / std::max<std::size_t>(1, bins_per_group));
      const int n_chunks = static_cast<int>((n_groups + groups_per_chunk - 1) / groups_per_chunk);
      partial.assign(static_cast<std::size_t>(n_chunks), 0.0);

      #pragma omp parallel for schedule(guided) if (multi_threaded)
      for (int c = 0; c < n_chunks; ++c) {
        const std::size_t lo = static_cast<std::size_t>(c) * groups_per_chunk;
        const std::size_t hi = std::min(n_groups, lo + groups_per_chunk);
        double            acc = 0.0;
        for (std::size_t b = lo; b < hi; ++b) acc += term(b);
        partial[static_cast<std::size_t>(c)] = acc;
      }

      double total = 0.0;
      for (const double v : partial) total += v;
      return total;
    }

  }  // namespace

  SampleLikelihood::SampleLikelihood(const io::ic::ICSample&     sample,
                                     const io::ic::SampleConfig& cfg,
                                     const GlobalFluxSettings&   settings,
                                     std::shared_ptr<GpuSession> gpu,
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

    // The bin scale is empty unless the sample reuses gradients exported from the
    // unfiltered sample under a topology cut, in which case ICDataBase measured
    // the surviving weight fraction per bin.
    if (!cfg.gradient_file.empty())
      m_Systematics.emplace(cfg.mc_binning, cfg.gradient_file, std::span<const double>(sample.topology_bin_fraction));

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
    m_LogGammaDataPlus1.assign(total_bins, 0.0);
    m_PoissonSaturated.assign(total_bins, 0.0);
    m_Ssq.assign(total_bins, 0.0);
    m_GalacticTotal.assign(total_bins, 0.0);
    m_McTotal.assign(mc_bins, 0.0);
    m_McSsq.assign(mc_bins, 0.0);
    m_McSsqEvent.assign(mc_bins, 0.0);
    m_GroupDataSum.assign(mc_bins, 0.0);
    m_GroupLogGammaSum.assign(mc_bins, 0.0);
    m_GroupSaturatedSum.assign(mc_bins, 0.0);
    m_GroupZeroCount.assign(mc_bins, 0);
    m_NonZeroOffsets.assign(static_cast<std::size_t>(mc_bins) + 1, 0);

    refresh_data_constants();

    // SAY-on-GPU setup. Both flux kernels leave their per-event weights in GPU
    // buffers; the say_ssq kernel reduces them to the per-bin ssq without a
    // round trip through the CPU. upload_offsets deduplicates against the flux
    // components' own upload, so this returns the already-resident buffer.
    const bool gpu_per_event =
        (m_Astro && m_Astro->per_event_handle() >= 0) || (m_Atmo && m_Atmo->per_event_handle() >= 0);
    if (gpu && use_say && gpu_per_event) {
      m_Gpu = std::move(gpu);
      const std::string src =
          gpu_kernel_source(m_Gpu->language(), m_Gpu->is_fp64(), kSsqKernelMetalBody, kSsqKernelCudaBody);
      m_Gpu->ensure_kernel("say_ssq", src.c_str());
      m_SsqReduce.emplace(m_Gpu, sample, static_cast<std::size_t>(mc_bins));
      m_hSsq = m_Gpu->alloc_output(mc_bins);
    }
  }

  SampleLikelihood::PredictionChange SampleLikelihood::assemble_prediction(const ParameterWrapper& parameter) {
    // Components stay sequential within a sample: a Migrad step varies one
    // parameter, so at most one component is stale per call and running the
    // checks concurrently was measured to gain nothing (Metal) / <5% (CPU).
    // Cross-sample concurrency lives in ICLikelihood::calculate_likelihood.
    //
    // The returned flag says whether anything that feeds sigma^2 changed; it is
    // what partial_llh() uses to decide whether to re-run assemble_fluctuation().
    PredictionChange change;
    if (m_Astro) change.per_event |= m_Astro->check_and_recalculate(parameter);
    if (m_Atmo) change.per_event |= m_Atmo->check_and_recalculate(parameter);
    change.ssq = change.per_event;
    // The template fluctuations and the detector gradients enter sigma^2 at the
    // histogram level, so they mark it stale without touching a single event
    // weight -- which is what lets assemble_fluctuation() skip its reduction.
    if (m_Template) change.ssq |= m_Template->check_and_recalculate(parameter);
    if (m_Systematics) change.ssq |= m_Systematics->check_and_recalculate(parameter);

    // The galactic templates recalculate here too -- the prediction below reads
    // their histograms -- but deliberately do NOT set ssq_changed: NNMFit's
    // GalacticTemplate defines no fluctuation graph and is excluded from the ssq
    // sum (histogram_builder.py:307), and the constructor enforces the zero
    // fluctuation column that guarantees it. A moved galactic norm therefore
    // provably cannot change sigma^2, and re-running assemble_fluctuation for it
    // would cost a full reduction (a GPU dispatch over every bin) for nothing.
    //
    // Their *sum* is cached for the same reason it is cheap to: the templates
    // are already in the analysis binning, so re-summing them is a pass over
    // n_ra times as many bins as everything else in this function, and a Migrad
    // step that moves any other parameter leaves every one of them untouched.
    bool galactic_changed = !m_GalacticSeeded;
    for (TemplateFlux& galactic : m_Galactic) {
      const bool changed = galactic.check_and_recalculate(parameter);
      galactic_changed   = galactic_changed || changed;
    }

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

    // Galactic templates are already in the analysis binning, so they are summed
    // there and added after the broadcast below, undivided.
    if (!m_Galactic.empty() && galactic_changed) {
      std::ranges::fill(m_GalacticTotal, 0.0);
      for (TemplateFlux& galactic : m_Galactic) {
        const std::span<const double> h = galactic.histogram();
        for (std::size_t b = 0, n = m_GalacticTotal.size(); b < n; ++b) m_GalacticTotal[b] += h[b];
      }
      m_GalacticSeeded = true;
    }

    // The spread over the RA axis (NNMFit Binning_2D_to_3D: repeat(mu, n_ra) /
    // n_ra, plus the galactic templates, clipped at zero) is deliberately NOT
    // done here. m_McTotal and m_GalacticTotal are everything the likelihood
    // needs, and it reads them group by group; writing the 3D array out would
    // be 267,300 stores per evaluation on the tracks sample for the benefit of
    // the results writers alone. materialize_prediction() does it for them.
    m_PredictedStale = true;

    return change;
  }

  void SampleLikelihood::materialize_prediction() const {
    const int  n_ra         = m_RaBins;
    const auto galactic_sum = std::span<const double>(m_GalacticTotal);
    const bool has_galactic = !m_Galactic.empty();
    const int  n_mc         = static_cast<int>(m_McTotal.size());

    #pragma omp parallel for schedule(guided) if (m_UseMultiThreading)
    for (int b = 0; b < n_mc; ++b) {
      const double      value = m_McTotal[b] / static_cast<double>(n_ra);
      const std::size_t base  = static_cast<std::size_t>(b) * static_cast<std::size_t>(n_ra);
      for (int r = 0; r < n_ra; ++r) {
        double bin_value = value;
        if (has_galactic) bin_value += galactic_sum[base + r];
        m_Predicted[base + r] = std::max(0.0, bin_value);
      }
    }
    m_PredictedStale = false;
  }

  void SampleLikelihood::materialize_ssq() const {
    // sigma^2 is a squared quantity, so the RA divisor is squared too (NNMFit
    // Binning_2D_to_3D.make_binned_flux with is_ssq_calc=True).
    io::ic::broadcast_over_ra(m_McSsq, m_RaBins,
                              static_cast<double>(m_RaBins) * static_cast<double>(m_RaBins), m_Ssq);
    m_SsqStale = false;
  }

  std::span<const double> SampleLikelihood::predicted() const noexcept {
    if (m_PredictedStale) materialize_prediction();
    return m_Predicted;
  }

  std::span<const double> SampleLikelihood::ssq() const noexcept {
    if (m_SsqStale) materialize_ssq();
    return m_Ssq;
  }

  void SampleLikelihood::assemble_fluctuation(const bool per_event_changed) {
    // Combined per-event weight across astro+conv+prompt, squared, then binned
    // -- matches NNMFit's rule that same-event components must be summed
    // *before* squaring (NNMFit/core/histogram_builder.py:229-329), since
    // (w1+w2)^2 != w1^2 + w2^2.
    const int n_bins = static_cast<int>(m_McSsq.size());

    // The event-level half is cached in m_McSsqEvent. A step in a detector
    // systematic or a template norm leaves every event weight where it was, and
    // re-reducing 10^7 of them to find that out is the most expensive way
    // possible to add a per-bin vector.
    if (!per_event_changed) {
      std::ranges::copy(m_McSsqEvent, m_McSsq.begin());
    } else if (m_hSsq >= 0) {
      // GPU path: reduce the flux kernels' per-event weight buffers in place.
      // A component that is absent binds the chunk-offsets buffer as a dummy
      // (never read, its flag is 0) -- the backend needs a valid buffer at every
      // slot.
      const int astro_h  = m_Astro ? m_Astro->per_event_handle() : -1;
      const int atmo_h   = m_Atmo ? m_Atmo->per_event_handle() : -1;
      const int chunks_h = m_SsqReduce->chunk_offsets();

      const SsqParams p{.has_astro = astro_h >= 0 ? 1 : 0, .has_atmo = atmo_h >= 0 ? 1 : 0};
      const int       inputs[] = {astro_h >= 0 ? astro_h : chunks_h,
                                  atmo_h >= 0 ? atmo_h : chunks_h,
                                  chunks_h};
      m_Gpu->dispatch("say_ssq", inputs, 3, &p, sizeof(p), m_SsqReduce->partial(), -1,
                      m_SsqReduce->n_chunks());
      m_SsqReduce->gather(m_hSsq);

      if (m_Gpu->is_fp64()) {
        const double* ssq = m_Gpu->contents_f64(m_hSsq);
        for (int b = 0; b < n_bins; ++b) m_McSsq[b] = ssq[b];
      } else {
        const float* ssq = m_Gpu->contents(m_hSsq);
        for (int b = 0; b < n_bins; ++b) m_McSsq[b] = static_cast<double>(ssq[b]);
      }
      std::ranges::copy(m_McSsq, m_McSsqEvent.begin());
    } else {
      const std::span<const double> astro = m_Astro ? m_Astro->per_event_weight() : std::span<const double>{};
      const std::span<const double> atmo  = m_Atmo ? m_Atmo->per_event_weight() : std::span<const double>{};
      const auto&                   off   = m_Sample.bin_offsets;

      // The component test is hoisted out of the per-event loop: which components
      // exist is fixed at construction, so each case gets its own tight loop.
      auto accumulate = [&](auto event_weight) {
        #pragma omp parallel for schedule(guided) if(m_UseMultiThreading)
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

      std::ranges::copy(m_McSsq, m_McSsqEvent.begin());
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

    // The RA broadcast that used to happen here is left to materialize_ssq():
    // sigma^2 is constant along RA, so the likelihood reads it per MC bin and
    // divides by n_ra^2 itself. The galactic templates contribute nothing
    // either way: NNMFit's GalacticTemplate defines no fluctuation graph and is
    // excluded from the ssq sum.
    m_SsqStale = true;
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
    refresh_data_constants();

    // The SAY ssq describes MC statistics, so it still comes from the model; seed
    // it exactly as generate_asimov does.
    if (m_UseSAY) assemble_fluctuation();
  }

  void SampleLikelihood::refresh_data_constants() {
    // Everything the likelihood term needs that depends only on the data, which
    // is fixed for the whole fit. Must be called whenever m_Data changes and
    // nowhere else.
    //
    // lgamma(k + 1) was already cached here. The saturated Poisson term joins
    // it: the Poisson likelihood subtracts log P(k|k) per bin, which is another
    // logarithm per analysis bin per evaluation -- 267,300 of them on the
    // tracks sample -- for a quantity that never moves.
    for (std::size_t b = 0, n = m_Data.size(); b < n; ++b) {
      const double k = m_Data[b];
      m_LogGammaDataPlus1[b] = std::lgamma(k + 1.0);
      m_PoissonSaturated[b]  = poisson_bin_log_likelihood(k, k, m_LogGammaDataPlus1[b]);
    }

    // Per-RA-group sums of the above, plus the index of every bin with k > 0.
    // These are what make the no-galactic likelihood loop cost one logarithm
    // per MC bin instead of one per analysis bin.
    const int n_ra = m_RaBins;
    m_NonZeroData.clear();
    for (std::size_t b = 0, n = m_McTotal.size(); b < n; ++b) {
      const std::size_t base = b * static_cast<std::size_t>(n_ra);
      double            data_sum = 0.0;
      double            lgamma_sum = 0.0;
      double            saturated_sum = 0.0;
      int               zeros = 0;

      m_NonZeroOffsets[b] = m_NonZeroData.size();
      for (int r = 0; r < n_ra; ++r) {
        const std::size_t bin = base + static_cast<std::size_t>(r);
        const double      k   = m_Data[bin];
        data_sum += k;
        lgamma_sum += m_LogGammaDataPlus1[bin];
        saturated_sum += m_PoissonSaturated[bin];
        if (k > 0.0)
          m_NonZeroData.push_back(static_cast<int>(bin));
        else
          ++zeros;
      }

      m_GroupDataSum[b]      = data_sum;
      m_GroupLogGammaSum[b]  = lgamma_sum;
      m_GroupSaturatedSum[b] = saturated_sum;
      m_GroupZeroCount[b]    = zeros;
    }
    m_NonZeroOffsets.back() = m_NonZeroData.size();
  }

  double SampleLikelihood::poisson_llh() const {
    const int    n_ra      = m_RaBins;
    const double ra_scale  = static_cast<double>(n_ra);
    const bool   galactic  = !m_Galactic.empty();
    const auto   galactic_sum = std::span<const double>(m_GalacticTotal);

    const double llh = sum_over_groups(
        m_McTotal.size(), static_cast<std::size_t>(n_ra), m_UseMultiThreading, m_Partial,
        [&](const std::size_t b) {
          const double      value = m_McTotal[b] / ra_scale;
          const std::size_t base  = b * static_cast<std::size_t>(n_ra);

          if (galactic) {
            // mu varies along RA, so every bin needs its own logarithm.
            double acc = 0.0;
            for (int r = 0; r < n_ra; ++r) {
              const std::size_t bin = base + static_cast<std::size_t>(r);
              const double      mu  = std::max(0.0, value + galactic_sum[bin]);
              acc += poisson_bin_log_likelihood(m_Data[bin], mu, m_LogGammaDataPlus1[bin]) -
                     m_PoissonSaturated[bin];
            }
            return acc;
          }

          // Without a galactic template mu is the same in every bin of the
          // group, so the whole slice collapses:
          //   sum_r [k_r log mu - mu - lgamma(k_r + 1) - saturated_r]
          //     = (sum_r k_r) log mu - n_ra mu - sum_r lgamma(k_r + 1)
          //       - sum_r saturated_r
          // -- one logarithm for the group instead of one per bin.
          const double mu = std::max(0.0, value);
          if (mu > 0.0)
            return m_GroupDataSum[b] * std::log(mu) - ra_scale * mu - m_GroupLogGammaSum[b] -
                   m_GroupSaturatedSum[b];

          // mu <= 0: NNMFit's finite stand-in for -inf, -690 per event, and 0
          // where there is no data (which the group sum already accounts for).
          return -690.0 * m_GroupDataSum[b] - m_GroupSaturatedSum[b];
        });

    return -2.0 * llh;
  }

  double SampleLikelihood::say_llh() const {
    const int    n_ra         = m_RaBins;
    const double ra_scale     = static_cast<double>(n_ra);
    const double ssq_divisor  = ra_scale * ra_scale;
    const bool   galactic     = !m_Galactic.empty();
    const auto   galactic_sum = std::span<const double>(m_GalacticTotal);

    const double llh = sum_over_groups(
        m_McTotal.size(), static_cast<std::size_t>(n_ra), m_UseMultiThreading, m_Partial,
        [&](const std::size_t b) {
          const double      value = m_McTotal[b] / ra_scale;
          const double      ssq   = m_McSsq[b] / ssq_divisor;
          const std::size_t base  = b * static_cast<std::size_t>(n_ra);

          if (galactic) {
            double acc = 0.0;
            for (int r = 0; r < n_ra; ++r) {
              const std::size_t bin = base + static_cast<std::size_t>(r);
              const double      mu  = std::max(0.0, value + galactic_sum[bin]);
              acc += say_bin_log_likelihood(m_Data[bin], mu, ssq, m_LogGammaDataPlus1[bin]);
            }
            return acc;
          }

          // mu and sigma^2 are both constant across the group, so alpha, beta
          // and their transcendentals are computed once for the whole slice.
          // What is left per bin is lgamma(k + alpha) -- and for an empty bin
          // that is lgamma(alpha), which is already in hand.
          const double mu = std::max(0.0, value);
          if (mu <= 0.0)
            return -690.0 * m_GroupDataSum[b];

          const double ssq_clipped = std::clamp(ssq, 0.0, mu * mu);
          if (ssq_clipped <= 0.0)
            return m_GroupDataSum[b] * std::log(mu) - ra_scale * mu - m_GroupLogGammaSum[b];

          const double alpha        = mu * mu / ssq_clipped + 1.0;
          const double beta         = mu / ssq_clipped;
          const double log_beta     = std::log(beta);
          const double log1p_beta   = std::log1p(beta);
          const double lgamma_alpha = std::lgamma(alpha);

          double acc = ra_scale * (alpha * log_beta - lgamma_alpha) -
                       (m_GroupDataSum[b] + ra_scale * alpha) * log1p_beta - m_GroupLogGammaSum[b] +
                       static_cast<double>(m_GroupZeroCount[b]) * lgamma_alpha;

          for (std::size_t j = m_NonZeroOffsets[b], end = m_NonZeroOffsets[b + 1]; j < end; ++j)
            acc += std::lgamma(m_Data[static_cast<std::size_t>(m_NonZeroData[j])] + alpha);

          return acc;
        });

    return -2.0 * llh;
  }

  void SampleLikelihood::generate_asimov(const ParameterWrapper& nominal) {
    assemble_prediction(nominal);
    std::ranges::copy(predicted(), m_Data.begin());
    refresh_data_constants();

    // Seed the ssq histogram at the nominal point. partial_llh() only refreshes
    // it when a flux actually recalculated, and the minimizer's first evaluation
    // is at the start values -- which compare equal to the nominal set here, so
    // nothing would recalculate and SAY would run that evaluation with ssq == 0
    // (silently degenerating to plain Poisson).
    if (m_UseSAY)
      assemble_fluctuation();
  }

  double SampleLikelihood::partial_llh(const ParameterWrapper& parameter) {
    const PredictionChange change = assemble_prediction(parameter);

    if (m_UseSAY) {
      if (change.ssq)
        assemble_fluctuation(change.per_event);
      return say_llh();
    }

    return poisson_llh();
  }

}  // namespace ana::ic
