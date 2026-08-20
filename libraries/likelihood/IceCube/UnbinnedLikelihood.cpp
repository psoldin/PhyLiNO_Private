#include "UnbinnedLikelihood.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ana::ic {

  namespace {

    // Scalars pushed to the kernel each dispatch; field order and sizes match the
    // device struct below.
    struct KdeParams {
      int    n_queries;
      int    n_bands;
      int    has_astro;
      int    has_atmo;
      double lo_e;
      double hi_e;
      double lo_z;
      double hi_z;
    };

    // One thread per query point. Each walks the 3x3 stencil of every bandwidth
    // band, sums the weighted product kernel over the events it finds, subtracts
    // the query's own kernel (leave-one-out) and writes ln(lambda) for that
    // query. The weighting by w^A and the reduction stay on the host so that the
    // summation order is the CPU path's, whatever the launch configuration.
    //
    // The per-event weights are the flux kernels' own device buffers, never
    // uploaded; an absent component binds a dummy buffer its flag keeps unread,
    // the same convention say_ssq uses. Index data (queries, bands,
    // cell_events) rides in as doubles because upload_column is the only
    // host-to-device entry point the facade has; the values are small integers
    // that fp64 represents exactly.
    //
    // Written against a generic scalar `real`; cuda_kernel_source() prepends the
    // typedef. The unbinned path requires fp64, so `real` is always double here.
    constexpr const char* kUnbinnedKernelCudaBody = R"CUDA(
      struct KdeParams {
        int n_queries; int n_bands; int has_astro; int has_atmo;
        real lo_e; real hi_e; real lo_z; real hi_z;
      };

      __device__ inline real reflected_kernel(const real u, const real c, const real inv_h,
                                              const real a, const real b) {
        const real kImageCutoff = (real)8.0;

        const real d   = (u - c) * inv_h;
        real       sum = exp((real)-0.5 * d * d);

        const real low_arg = (u + c - (real)2.0 * a) * inv_h;
        if (low_arg < kImageCutoff) sum += exp((real)-0.5 * low_arg * low_arg);

        const real high_arg = ((real)2.0 * b - u - c) * inv_h;
        if (high_arg < kImageCutoff) sum += exp((real)-0.5 * high_arg * high_arg);

        return sum;
      }

      extern "C" __global__ void unbinned_kde(
          const real*         x_e,
          const real*         x_z,
          const real*         inv_h_e,
          const real*         inv_h_z,
          const real*         prefactor,
          const real*         reach_e,
          const real*         reach_z,
          const real*         queries,
          const real*         bands,
          const real*         cell_events,
          const unsigned int* cell_offsets,
          const real*         astro_pe,
          const real*         atmo_pe,
          KdeParams           p,
          real*               log_lambda,
          real*               unused)
      {
        const int q = blockIdx.x * blockDim.x + threadIdx.x;
        if (q >= p.n_queries) return;

        const int  j  = (int)queries[q];
        const real qe = x_e[j];
        const real qz = x_z[j];

        real acc = (real)0.0;
        for (int b = 0; b < p.n_bands; ++b) {
          const real cell_e     = bands[5 * b + 0];
          const real cell_z     = bands[5 * b + 1];
          const int  n_e        = (int)bands[5 * b + 2];
          const int  n_z        = (int)bands[5 * b + 3];
          const int  first_cell = (int)bands[5 * b + 4];

          const int ce = (int)floor((qe - p.lo_e) / cell_e);
          const int cz = (int)floor((qz - p.lo_z) / cell_z);

          for (int de = -1; de <= 1; ++de) {
            const int ie = ce + de;
            if (ie < 0 || ie >= n_e) continue;
            for (int dz = -1; dz <= 1; ++dz) {
              const int iz = cz + dz;
              if (iz < 0 || iz >= n_z) continue;

              const int          cell  = first_cell + ie * n_z + iz;
              const unsigned int begin = cell_offsets[cell];
              const unsigned int end   = cell_offsets[cell + 1];

              for (unsigned int k = begin; k < end; ++k) {
                const int i = (int)cell_events[k];

                if (fabs(qe - x_e[i]) > reach_e[i]) continue;
                if (fabs(qz - x_z[i]) > reach_z[i]) continue;

                real w = (real)0.0;
                if (p.has_astro) w += astro_pe[i];
                if (p.has_atmo)  w += atmo_pe[i];

                acc += w * prefactor[i] * reflected_kernel(qe, x_e[i], inv_h_e[i], p.lo_e, p.hi_e)
                                        * reflected_kernel(qz, x_z[i], inv_h_z[i], p.lo_z, p.hi_z);
              }
            }
          }
        }

        real w_self = (real)0.0;
        if (p.has_astro) w_self += astro_pe[j];
        if (p.has_atmo)  w_self += atmo_pe[j];
        acc -= w_self * prefactor[j] * reflected_kernel(qe, x_e[j], inv_h_e[j], p.lo_e, p.hi_e)
                                     * reflected_kernel(qz, x_z[j], inv_h_z[j], p.lo_z, p.hi_z);

        // Same floor as the CPU path: a query with no support at all becomes a
        // large finite penalty instead of a NaN that would poison the fit.
        log_lambda[q] = log(fmax(acc, (real)2.2250738585072014e-308));
      }
    )CUDA";

    // Scalars for the freeze kernel below.
    struct FreezeParams {
      int    n_queries;
      int    has_astro;
      int    has_atmo;
      double scale;
    };

    // Gathering the Asimov quadrature weights on the device, once per fit.
    //
    // It exists because of an asymmetry: on the CPU path the flux components
    // hand out their per-event weights as spans, but on a GPU backend those
    // weights stay in device buffers allocated readback=false (deliberately --
    // copying them back cost ~220 MB per evaluation and nothing read them). So
    // the frozen weights cannot be taken from the host, and this kernel gathers
    // them into a buffer that IS read back, exactly once, when the Asimov set is
    // generated. Thereafter the host owns them and the hot path never copies.
    constexpr const char* kFreezeKernelCudaBody = R"CUDA(
      struct FreezeParams { int n_queries; int has_astro; int has_atmo; real scale; };

      extern "C" __global__ void unbinned_freeze(
          const real* queries,
          const real* astro_pe,
          const real* atmo_pe,
          FreezeParams p,
          real*        asimov,
          real*        unused)
      {
        const int q = blockIdx.x * blockDim.x + threadIdx.x;
        if (q >= p.n_queries) return;

        const int j = (int)queries[q];
        real      w = (real)0.0;
        if (p.has_astro) w += astro_pe[j];
        if (p.has_atmo)  w += atmo_pe[j];

        asimov[q] = p.scale * w;
      }
    )CUDA";

    /**
     * Sum term(q) over [0, n) in fixed chunks, adding the chunk totals back in
     * index order. The chunk size is a function of n alone -- never of the thread
     * count -- so the same fit sums identically with -m, without it, and on a
     * machine with a different core count, which is what makes two scan points
     * comparable. Aiming at ~kTargetChunks pieces keeps even a thinned run
     * (a few thousand queries) spread across the cores.
     */
    template <class Term>
    double chunked_sum(const std::size_t n, const bool multi_threaded, std::vector<double>& partial,
                       Term&& term) noexcept {
      constexpr std::size_t kTargetChunks = 512;
      if (n == 0) return 0.0;

      const std::size_t per_chunk =
          std::clamp<std::size_t>((n + kTargetChunks - 1) / kTargetChunks, 64, 4096);
      const int n_chunks = static_cast<int>((n + per_chunk - 1) / per_chunk);
      partial.assign(static_cast<std::size_t>(n_chunks), 0.0);

      #pragma omp parallel for schedule(guided) if (multi_threaded)
      for (int c = 0; c < n_chunks; ++c) {
        const std::size_t begin = static_cast<std::size_t>(c) * per_chunk;
        const std::size_t end   = std::min(n, begin + per_chunk);
        double            acc   = 0.0;
        for (std::size_t q = begin; q < end; ++q) acc += term(q);
        partial[static_cast<std::size_t>(c)] = acc;
      }

      double total = 0.0;
      for (const double v : partial) total += v;
      return total;
    }

  }  // namespace

  double KdeDensity::evaluate(const double qe, const double qz, const std::span<const double> w) const noexcept {
    double acc = 0.0;
    io::ic::for_each_neighbour(index, qe, qz, [&](const int i) {
      const auto e = static_cast<std::size_t>(i);

      // The stencil returns a box while the kernel is an ellipse, and one axis
      // alone often disqualifies a pair, so two compares save two exponentials.
      // Rejecting on the direct distance rejects the images too: their argument
      // ((u - a) + (c - a)) / h is never smaller than the direct |u - c| / h.
      const double du = qe - x_e[e];
      if (std::abs(du) > reach_e[e]) return;
      const double dz = qz - x_z[e];
      if (std::abs(dz) > reach_z[e]) return;

      acc += w[e] * prefactor[e] * reflected_kernel(qe, x_e[e], inv_h_e[e], lo[0], hi[0]) *
             reflected_kernel(qz, x_z[e], inv_h_z[e], lo[1], hi[1]);
    });
    return acc;
  }

  double KdeDensity::evaluate_loo(const std::size_t j, const std::span<const double> w) const noexcept {
    const double self = w[j] * prefactor[j] * reflected_kernel(x_e[j], x_e[j], inv_h_e[j], lo[0], hi[0]) *
                        reflected_kernel(x_z[j], x_z[j], inv_h_z[j], lo[1], hi[1]);
    return evaluate(x_e[j], x_z[j], w) - self;
  }

  UnbinnedLikelihood::UnbinnedLikelihood(const io::ic::ICSample&       sample,
                                         const io::ic::UnbinnedConfig& cfg,
                                         std::shared_ptr<GpuSession>   gpu,
                                         const bool                    use_multi_threading)
    : m_Sample(sample)
    , m_Config(cfg)
    , m_UseMultiThreading(use_multi_threading)
    , m_Gpu(std::move(gpu)) {
    if (sample.kde_log_e.size() != sample.size() || sample.kde_index.empty())
      throw std::runtime_error(
          "UnbinnedLikelihood: sample carries no KDE columns or no neighbour index; its config needs an "
          "\"Unbinned\" block so ICDataBase reads the reco coordinates and bandwidths");

    m_Weight.assign(sample.size(), 0.0);
    m_AsimovWeight.assign(sample.kde_queries.size(), 0.0);

    if (m_Gpu) setup_gpu(sample);
  }

  void UnbinnedLikelihood::setup_gpu(const io::ic::ICSample& sample) {
    if (m_Gpu->language() != GpuLanguage::Cuda)
      throw std::runtime_error(
          "UnbinnedLikelihood: the unbinned KDE runs on the CPU and CUDA backends only. Metal is FP32, and "
          "this likelihood takes the logarithm of a kernel sum, which is exactly where FP32 hurts most");
    if (!m_Gpu->is_fp64())
      throw std::runtime_error("UnbinnedLikelihood: the unbinned KDE requires an FP64 CUDA backend");

    // Static columns. Every pointer here belongs to ICSample, which the module
    // caches per process, so the backend's pointer-keyed column cache dedupes
    // them across Fits instead of aliasing a recycled address.
    m_hXe        = m_Gpu->upload_column(sample.kde_log_e.data(), sample.kde_log_e.size());
    m_hXz        = m_Gpu->upload_column(sample.kde_zenith.data(), sample.kde_zenith.size());
    m_hInvHe     = m_Gpu->upload_column(sample.kde_kernel.inv_h_e.data(), sample.kde_kernel.inv_h_e.size());
    m_hInvHz     = m_Gpu->upload_column(sample.kde_kernel.inv_h_z.data(), sample.kde_kernel.inv_h_z.size());
    m_hPrefactor = m_Gpu->upload_column(sample.kde_kernel.prefactor.data(), sample.kde_kernel.prefactor.size());
    m_hReachE    = m_Gpu->upload_column(sample.kde_kernel.reach_e.data(), sample.kde_kernel.reach_e.size());
    m_hReachZ    = m_Gpu->upload_column(sample.kde_kernel.reach_z.data(), sample.kde_kernel.reach_z.size());
    m_hCellOff   = m_Gpu->upload_offsets(sample.kde_index.cell_offsets.data(), sample.kde_index.cell_offsets.size());

    // The index's integers go up as doubles: upload_column is the facade's only
    // host-to-device path, and fp64 represents these magnitudes exactly.
    m_QueriesHost.assign(sample.kde_queries.begin(), sample.kde_queries.end());
    m_CellEventsHost.assign(sample.kde_index.cell_events.begin(), sample.kde_index.cell_events.end());

    m_BandsHost.clear();
    m_BandsHost.reserve(5 * sample.kde_index.bands.size());
    for (const io::ic::KdeIndex::Band& band : sample.kde_index.bands) {
      m_BandsHost.push_back(band.cell[0]);
      m_BandsHost.push_back(band.cell[1]);
      m_BandsHost.push_back(static_cast<double>(band.n_cells[0]));
      m_BandsHost.push_back(static_cast<double>(band.n_cells[1]));
      m_BandsHost.push_back(static_cast<double>(band.first_cell));
    }

    m_hQueries   = m_Gpu->upload_column(m_QueriesHost.data(), m_QueriesHost.size());
    m_hCellEvent = m_Gpu->upload_column(m_CellEventsHost.data(), m_CellEventsHost.size());
    m_hBands     = m_Gpu->upload_column(m_BandsHost.data(), m_BandsHost.size());

    m_hLogLambda = m_Gpu->alloc_output(m_Sample.kde_queries.size(), /*readback=*/true);
    m_hAsimov    = m_Gpu->alloc_output(m_Sample.kde_queries.size(), /*readback=*/true);

    const bool fp64 = m_Gpu->is_fp64();
    m_Gpu->ensure_kernel(
        "unbinned_kde",
        gpu_kernel_source(m_Gpu->language(), fp64, /*metal_body=*/"", kUnbinnedKernelCudaBody).c_str());
    m_Gpu->ensure_kernel(
        "unbinned_freeze",
        gpu_kernel_source(m_Gpu->language(), fp64, /*metal_body=*/"", kFreezeKernelCudaBody).c_str());
  }

  double UnbinnedLikelihood::gpu_log_sum(const int astro_handle, const int atmo_handle) {
    // A component the sample does not declare still needs a valid buffer bound;
    // its flag keeps the kernel from reading it. Same dummy-binding convention
    // as the say_ssq reduction.
    const int dummy = m_hCellOff;

    const KdeParams params{.n_queries = static_cast<int>(m_Sample.kde_queries.size()),
                           .n_bands   = static_cast<int>(m_Sample.kde_index.bands.size()),
                           .has_astro = astro_handle >= 0 ? 1 : 0,
                           .has_atmo  = atmo_handle >= 0 ? 1 : 0,
                           .lo_e      = m_Config.log_e_lo,
                           .hi_e      = m_Config.log_e_hi,
                           .lo_z      = m_Config.zenith_lo,
                           .hi_z      = m_Config.zenith_hi};

    const int inputs[] = {m_hXe,       m_hXz,        m_hInvHe,   m_hInvHz,
                          m_hPrefactor, m_hReachE,   m_hReachZ,  m_hQueries,
                          m_hBands,    m_hCellEvent, m_hCellOff,
                          astro_handle >= 0 ? astro_handle : dummy,
                          atmo_handle >= 0 ? atmo_handle : dummy};

    // One thread per query, 256 threads per block -- the block size every kernel
    // in this backend is launched with (CudaBackend's kThreadsPerGroup).
    constexpr std::size_t kThreadsPerGroup = 256;
    const std::size_t     n_groups =
        (m_Sample.kde_queries.size() + kThreadsPerGroup - 1) / kThreadsPerGroup;

    m_Gpu->dispatch("unbinned_kde", inputs, static_cast<int>(std::size(inputs)), &params, sizeof(params),
                    m_hLogLambda, /*per_event=*/-1, n_groups);

    const double* log_lambda = m_Gpu->contents_f64(m_hLogLambda);
    return chunked_sum(m_Sample.kde_queries.size(), m_UseMultiThreading, m_Partial,
                       [this, log_lambda](const std::size_t q) { return m_AsimovWeight[q] * log_lambda[q]; });
  }

  KdeDensity UnbinnedLikelihood::density() const noexcept {
    return KdeDensity{.x_e       = m_Sample.kde_log_e,
                      .x_z       = m_Sample.kde_zenith,
                      .inv_h_e   = m_Sample.kde_kernel.inv_h_e,
                      .inv_h_z   = m_Sample.kde_kernel.inv_h_z,
                      .prefactor = m_Sample.kde_kernel.prefactor,
                      .reach_e   = m_Sample.kde_kernel.reach_e,
                      .reach_z   = m_Sample.kde_kernel.reach_z,
                      .index     = m_Sample.kde_index,
                      .lo        = {m_Config.log_e_lo, m_Config.zenith_lo},
                      .hi        = {m_Config.log_e_hi, m_Config.zenith_hi}};
  }

  void UnbinnedLikelihood::combine_weights(const std::span<const double> astro, const std::span<const double> atmo) {
    // Same rule as the SAY ssq reduction: an event's components are summed
    // before anything nonlinear happens to them.
    const std::size_t n = m_Weight.size();
    if (!astro.empty() && !atmo.empty())
      for (std::size_t i = 0; i < n; ++i) m_Weight[i] = astro[i] + atmo[i];
    else if (!astro.empty())
      std::ranges::copy(astro, m_Weight.begin());
    else if (!atmo.empty())
      std::ranges::copy(atmo, m_Weight.begin());
    else
      std::ranges::fill(m_Weight, 0.0);
  }

  void UnbinnedLikelihood::freeze_asimov(const std::span<const double> astro, const std::span<const double> atmo,
                                         const int astro_handle, const int atmo_handle) {
    // Thinning drops quadrature nodes, so the survivors carry the weight of the
    // ones dropped; the total stays nu(theta_A) whatever the stride.
    const double scale = static_cast<double>(m_Config.thinning);

    if (m_Gpu) {
      const FreezeParams params{.n_queries = static_cast<int>(m_Sample.kde_queries.size()),
                                .has_astro = astro_handle >= 0 ? 1 : 0,
                                .has_atmo  = atmo_handle >= 0 ? 1 : 0,
                                .scale     = scale};
      const int          inputs[] = {m_hQueries,
                                     astro_handle >= 0 ? astro_handle : m_hQueries,
                                     atmo_handle >= 0 ? atmo_handle : m_hQueries};

      constexpr std::size_t kThreadsPerGroup = 256;
      m_Gpu->dispatch("unbinned_freeze", inputs, static_cast<int>(std::size(inputs)), &params, sizeof(params),
                      m_hAsimov, /*per_event=*/-1,
                      (m_Sample.kde_queries.size() + kThreadsPerGroup - 1) / kThreadsPerGroup);

      const double* gathered = m_Gpu->contents_f64(m_hAsimov);
      std::copy_n(gathered, m_AsimovWeight.size(), m_AsimovWeight.begin());
    } else {
      combine_weights(astro, atmo);
      for (std::size_t q = 0; q < m_Sample.kde_queries.size(); ++q)
        m_AsimovWeight[q] = scale * m_Weight[static_cast<std::size_t>(m_Sample.kde_queries[q])];
    }

    m_AsimovTotal = 0.0;
    for (const double w : m_AsimovWeight) m_AsimovTotal += w;
  }

  double UnbinnedLikelihood::llh(const std::span<const double> astro,
                                 const std::span<const double> atmo,
                                 const double nu,
                                 const int astro_handle,
                                 const int atmo_handle) {
    if (m_Gpu) return 2.0 * (nu - gpu_log_sum(astro_handle, atmo_handle));

    combine_weights(astro, atmo);
    const KdeDensity kde = density();

    const double log_sum =
        chunked_sum(m_Sample.kde_queries.size(), m_UseMultiThreading, m_Partial, [this, &kde](const std::size_t q) {
          const auto   j      = static_cast<std::size_t>(m_Sample.kde_queries[q]);
          const double lambda = kde.evaluate_loo(j, m_Weight);
          // A query point can end up with no support at all -- an isolated event
          // whose only neighbour was itself. Clamping to the smallest normal
          // double makes that a large but finite penalty (ln ~ -708) instead of a
          // NaN that would poison the fit silently.
          return m_AsimovWeight[q] * std::log(std::max(lambda, std::numeric_limits<double>::min()));
        });

    return 2.0 * (nu - log_sum);
  }

}  // namespace ana::ic
