#include "UnbinnedLikelihood.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ana::ic {

  double KdeDensity::evaluate(const double qe, const double qz, const std::span<const double> w) const noexcept {
    double acc = 0.0;
    io::ic::for_each_neighbour(index, qe, qz, [&](const int i) {
      const auto e = static_cast<std::size_t>(i);
      acc += w[e] * reflected_gauss(qe, x_e[e], h_e[e], lo[0], hi[0]) *
             reflected_gauss(qz, x_z[e], h_z[e], lo[1], hi[1]);
    });
    return acc;
  }

  double KdeDensity::evaluate_loo(const std::size_t j, const std::span<const double> w) const noexcept {
    const double self = w[j] * reflected_gauss(x_e[j], x_e[j], h_e[j], lo[0], hi[0]) *
                        reflected_gauss(x_z[j], x_z[j], h_z[j], lo[1], hi[1]);
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

    const std::size_t stride = static_cast<std::size_t>(m_Config.thinning);
    for (std::size_t i = 0; i < sample.size(); i += stride) m_Queries.push_back(static_cast<int>(i));
    m_AsimovWeight.assign(m_Queries.size(), 0.0);
  }

  KdeDensity UnbinnedLikelihood::density() const noexcept {
    return KdeDensity{.x_e   = m_Sample.kde_log_e,
                      .x_z   = m_Sample.kde_zenith,
                      .h_e   = m_Sample.kde_h_e,
                      .h_z   = m_Sample.kde_h_z,
                      .index = m_Sample.kde_index,
                      .lo    = {m_Config.log_e_lo, m_Config.zenith_lo},
                      .hi    = {m_Config.log_e_hi, m_Config.zenith_hi}};
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

  void UnbinnedLikelihood::freeze_asimov(const std::span<const double> astro, const std::span<const double> atmo) {
    combine_weights(astro, atmo);

    // Thinning drops quadrature nodes, so the survivors carry the weight of the
    // ones dropped; the total stays nu(theta_A) whatever the stride.
    const double scale = static_cast<double>(m_Config.thinning);
    m_AsimovTotal      = 0.0;
    for (std::size_t q = 0; q < m_Queries.size(); ++q) {
      m_AsimovWeight[q] = scale * m_Weight[static_cast<std::size_t>(m_Queries[q])];
      m_AsimovTotal += m_AsimovWeight[q];
    }
  }

  double UnbinnedLikelihood::llh(const std::span<const double> astro, const std::span<const double> atmo,
                                 const double nu) {
    combine_weights(astro, atmo);
    const KdeDensity kde = density();

    // Fixed chunking rather than an OpenMP reduction, for the reason the binned
    // loops use it: the accumulation order must not depend on the thread
    // schedule, or two scan points stop being comparable.
    constexpr std::size_t kQueriesPerChunk = 4096;
    const int n_chunks = static_cast<int>((m_Queries.size() + kQueriesPerChunk - 1) / kQueriesPerChunk);
    m_Partial.assign(static_cast<std::size_t>(std::max(n_chunks, 1)), 0.0);

    #pragma omp parallel for schedule(guided) if (m_UseMultiThreading)
    for (int c = 0; c < n_chunks; ++c) {
      const std::size_t begin = static_cast<std::size_t>(c) * kQueriesPerChunk;
      const std::size_t end   = std::min(m_Queries.size(), begin + kQueriesPerChunk);
      double            acc   = 0.0;
      for (std::size_t q = begin; q < end; ++q) {
        const std::size_t j      = static_cast<std::size_t>(m_Queries[q]);
        const double      lambda = kde.evaluate_loo(j, m_Weight);
        // A query point can end up with no support at all -- an isolated event
        // whose only neighbour was itself. Clamping to the smallest normal
        // double makes that a large but finite penalty (ln ~ -708) instead of a
        // NaN that would poison the fit silently.
        acc += m_AsimovWeight[q] * std::log(std::max(lambda, std::numeric_limits<double>::min()));
      }
      m_Partial[static_cast<std::size_t>(c)] = acc;
    }

    double log_sum = 0.0;
    for (const double v : m_Partial) log_sum += v;

    return 2.0 * (nu - log_sum);
  }

}  // namespace ana::ic
