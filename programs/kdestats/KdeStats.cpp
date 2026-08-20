/**
 * How dense is the unbinned KDE's neighbour structure, and would a precomputed
 * sparse kernel matrix fit in memory?
 *
 * The likelihood currently recomputes every kernel at every evaluation. The
 * alternative is to build the pair matrix once -- it is parameter-independent,
 * only the weights move -- and reduce each evaluation to a sparse
 * matrix-vector product. That trade is decided by one number: how many pairs
 * survive truncation (nnz). This tool measures it from a random sample of query
 * points, rather than paying for a full pass, and reports what the matrix would
 * cost at several thinning strides.
 *
 * It also measures how concentrated each row is, because pruning on the kernel
 * VALUE (keep the entries carrying 99.9% of a row's mass) drops far more
 * entries per byte than the distance box the fit currently truncates with.
 *
 *   ./build/programs/kdestats/KdeStats -c config.json [n_sampled_queries]
 */
#include "IceCube/ICDataBase.h"
#include "IceCube/KdeIndex.h"
#include "IceCube/SampleConfig.h"
#include "UnbinnedLikelihood.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

  struct RowStats {
    std::size_t visited = 0;  ///< events the stencil returned
    std::size_t kept    = 0;  ///< events inside the truncation on both axes
    std::size_t top999  = 0;  ///< entries carrying 99.9% of the row's kernel mass
    std::size_t top99   = 0;  ///< entries carrying 99% of it
  };

  // Entries needed to reach `fraction` of the total, largest first.
  std::size_t entries_for_mass(std::vector<double>& values, const double fraction) {
    const double total = std::accumulate(values.begin(), values.end(), 0.0);
    if (total <= 0.0) return 0;

    std::ranges::sort(values, std::ranges::greater{});
    double      running = 0.0;
    std::size_t n       = 0;
    for (const double v : values) {
      running += v;
      ++n;
      if (running >= fraction * total) break;
    }
    return n;
  }

  RowStats measure_row(const io::ic::ICSample& s, const ana::ic::KdeDensity& kde, const std::size_t j,
                       std::vector<double>& scratch) {
    RowStats stats;
    scratch.clear();

    const double qe = s.kde_log_e[j];
    const double qz = s.kde_zenith[j];

    io::ic::for_each_neighbour(s.kde_index, qe, qz, [&](const int i) {
      const auto e = static_cast<std::size_t>(i);
      ++stats.visited;
      if (std::abs(qe - s.kde_log_e[e]) > s.kde_kernel.reach_e[e]) return;
      if (std::abs(qz - s.kde_zenith[e]) > s.kde_kernel.reach_z[e]) return;
      ++stats.kept;

      scratch.push_back(s.kde_kernel.prefactor[e] *
                        ana::ic::reflected_kernel(qe, s.kde_log_e[e], s.kde_kernel.inv_h_e[e], kde.lo[0], kde.hi[0]) *
                        ana::ic::reflected_kernel(qz, s.kde_zenith[e], s.kde_kernel.inv_h_z[e], kde.lo[1], kde.hi[1]));
    });

    stats.top999 = entries_for_mass(scratch, 0.999);
    stats.top99  = entries_for_mass(scratch, 0.99);
    return stats;
  }

  double mean_of(const std::vector<std::size_t>& v) {
    return v.empty() ? 0.0 : static_cast<double>(std::accumulate(v.begin(), v.end(), std::size_t{0})) / v.size();
  }

  std::size_t quantile_of(std::vector<std::size_t> v, const double f) {
    if (v.empty()) return 0;
    std::ranges::sort(v);
    return v[static_cast<std::size_t>(f * (v.size() - 1))];
  }

}  // namespace

int main(int argc, char** argv) {
  std::string config;
  std::size_t n_sample = 4000;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-c" && i + 1 < argc) config = argv[++i];
    else if (arg[0] != '-') n_sample = std::stoul(arg);
  }
  if (config.empty()) {
    std::puts("usage: KdeStats -c config.json [n_sampled_queries]");
    return 1;
  }

  // Only the sample configs are needed, so the JSON is read directly rather than
  // through InputOptions, which also wants a command line and a fit setup.
  boost::property_tree::ptree tree;
  boost::property_tree::read_json(config, tree);
  const std::vector<io::ic::SampleConfig> samples = io::ic::parse_samples(tree.get_child("IceCube"));

  const io::ic::ICDataBase database(samples);

  for (std::size_t si = 0; si < database.n_samples(); ++si) {
    const io::ic::ICSample& s = database.sample(si);
    if (s.kde_index.empty()) continue;

    const io::ic::UnbinnedConfig& cfg = samples[si].unbinned;
    const ana::ic::KdeDensity     kde{.x_e       = s.kde_log_e,
                                      .x_z       = s.kde_zenith,
                                      .inv_h_e   = s.kde_kernel.inv_h_e,
                                      .inv_h_z   = s.kde_kernel.inv_h_z,
                                      .prefactor = s.kde_kernel.prefactor,
                                      .reach_e   = s.kde_kernel.reach_e,
                                      .reach_z   = s.kde_kernel.reach_z,
                                      .index     = s.kde_index,
                                      .lo        = {cfg.log_e_lo, cfg.zenith_lo},
                                      .hi        = {cfg.log_e_hi, cfg.zenith_hi}};

    const std::size_t n_events = s.size();
    n_sample                   = std::min(n_sample, n_events);

    std::mt19937                                 rng(20260820);
    std::uniform_int_distribution<std::size_t>   pick(0, n_events - 1);
    std::vector<std::size_t> visited, kept, top999, top99;
    std::vector<double>      scratch;
    scratch.reserve(1 << 16);

    for (std::size_t k = 0; k < n_sample; ++k) {
      const RowStats r = measure_row(s, kde, pick(rng), scratch);
      visited.push_back(r.visited);
      kept.push_back(r.kept);
      top999.push_back(r.top999);
      top99.push_back(r.top99);
    }

    std::printf("\nsample '%s': %zu events, %zu bandwidth bands, truncation %.1f sigma\n",
                samples[si].name.c_str(), n_events, s.kde_index.bands.size(), cfg.truncation);
    std::printf("  sampled %zu query points\n", n_sample);
    std::printf("  %-28s mean %10.0f  median %10zu  p90 %10zu\n", "stencil returns", mean_of(visited),
                quantile_of(visited, 0.5), quantile_of(visited, 0.9));
    std::printf("  %-28s mean %10.0f  median %10zu  p90 %10zu\n", "inside truncation (nnz/row)", mean_of(kept),
                quantile_of(kept, 0.5), quantile_of(kept, 0.9));
    std::printf("  %-28s mean %10.0f  median %10zu  p90 %10zu\n", "entries for 99.9% of mass", mean_of(top999),
                quantile_of(top999, 0.5), quantile_of(top999, 0.9));
    std::printf("  %-28s mean %10.0f  median %10zu  p90 %10zu\n", "entries for 99% of mass", mean_of(top99),
                quantile_of(top99, 0.5), quantile_of(top99, 0.9));

    // What a precomputed matrix would cost: one fp32 value plus one int32 column
    // index per entry, rows = query points at the given thinning stride.
    constexpr double kBytesPerEntry = 8.0;
    std::printf("  sparse matrix size (fp32 value + int32 index, 8 B/entry):\n");
    std::printf("    %-10s %14s %12s %14s %12s\n", "thinning", "rows", "nnz(dist)", "GB(dist)", "GB(99.9%)");
    for (const int stride : {1, 10, 50, 100, 500, 2000}) {
      const double rows = static_cast<double>(n_events) / stride;
      const double nnz  = rows * mean_of(kept);
      const double pnnz = rows * mean_of(top999);
      std::printf("    %-10d %14.0f %12.3g %14.2f %12.2f\n", stride, rows, nnz,
                  nnz * kBytesPerEntry / 1e9, pnnz * kBytesPerEntry / 1e9);
    }
  }
  return 0;
}
