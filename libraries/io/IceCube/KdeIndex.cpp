#include "KdeIndex.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace io::ic {

  namespace {

    // Geometric band index: how many doublings above the smallest bandwidth in
    // the sample. Clamped so one pathological outlier cannot create hundreds of
    // near-empty bands; the clamp only makes that band's cells larger, never its
    // search incomplete, because the cell size comes from the band's own maximum
    // bandwidth rather than from the band's nominal upper edge.
    constexpr int kMaxBandIndex = 24;

    int band_of(const double h, const double h_min) noexcept {
      return std::clamp(static_cast<int>(std::floor(std::log2(h / h_min))), 0, kMaxBandIndex);
    }

  }  // namespace

  KdeKernelConstants build_kde_kernel_constants(const std::span<const double> h_e,
                                                const std::span<const double> h_z,
                                                const double                  n_sigma) {
    // 1 / sqrt(2 pi), the Gaussian's normalisation; squared into the product
    // prefactor of the two axes so the density's inner loop multiplies once.
    constexpr double kInvSqrt2Pi = 0.39894228040143267794;

    const std::size_t  n = h_e.size();
    KdeKernelConstants constants;
    constants.inv_h_e.resize(n);
    constants.inv_h_z.resize(n);
    constants.prefactor.resize(n);
    constants.reach_e.resize(n);
    constants.reach_z.resize(n);

    for (std::size_t i = 0; i < n; ++i) {
      constants.inv_h_e[i]   = 1.0 / h_e[i];
      constants.inv_h_z[i]   = 1.0 / h_z[i];
      constants.prefactor[i] = kInvSqrt2Pi * kInvSqrt2Pi * constants.inv_h_e[i] * constants.inv_h_z[i];
      constants.reach_e[i]   = n_sigma * h_e[i];
      constants.reach_z[i]   = n_sigma * h_z[i];
    }
    return constants;
  }

  KdeIndex build_kde_index(const std::span<const double> x_e,
                           const std::span<const double> x_z,
                           const std::span<const double> h_e,
                           const std::span<const double> h_z,
                           const std::array<double, 2>   lo,
                           const std::array<double, 2>   hi,
                           const double                  n_sigma) {
    const std::size_t n = x_e.size();
    if (x_z.size() != n || h_e.size() != n || h_z.size() != n)
      throw std::runtime_error("build_kde_index: column sizes disagree");
    if (n_sigma <= 0.0)
      throw std::runtime_error("build_kde_index: n_sigma must be positive");
    if (hi[0] <= lo[0] || hi[1] <= lo[1])
      throw std::runtime_error("build_kde_index: empty domain (hi <= lo)");

    KdeIndex index;
    index.lo = lo;
    if (n == 0) return index;

    // The smallest positive bandwidth per axis is the ladder's first rung.
    double h_e_min = std::numeric_limits<double>::infinity();
    double h_z_min = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
      if (!(h_e[i] > 0.0) || !(h_z[i] > 0.0) || !std::isfinite(h_e[i]) || !std::isfinite(h_z[i]))
        throw std::runtime_error("build_kde_index: non-positive or non-finite bandwidth at event " +
                                 std::to_string(i) + "; such events carry no kernel and must be dropped at load");
      h_e_min = std::min(h_e_min, h_e[i]);
      h_z_min = std::min(h_z_min, h_z[i]);
    }

    // Assign every event to a (band_e, band_z) pair and record each pair's
    // maximum bandwidth. That maximum, not the band's nominal upper edge, is
    // what the cell size is built from, so a sparsely populated band stays tight.
    using BandKey = std::pair<int, int>;
    std::vector<BandKey>                     key(n);
    std::map<BandKey, std::array<double, 2>> band_hmax;
    for (std::size_t i = 0; i < n; ++i) {
      key[i]     = {band_of(h_e[i], h_e_min), band_of(h_z[i], h_z_min)};
      auto& hmax = band_hmax[key[i]];
      hmax[0]    = std::max(hmax[0], h_e[i]);
      hmax[1]    = std::max(hmax[1], h_z[i]);
    }

    // Dense band ordering, and each band's grid geometry.
    std::map<BandKey, std::size_t> band_slot;
    for (const auto& [band_key, hmax] : band_hmax) {
      KdeIndex::Band band;
      band.cell[0]     = n_sigma * hmax[0];
      band.cell[1]     = n_sigma * hmax[1];
      band.n_cells[0]  = std::max(1, static_cast<int>(std::ceil((hi[0] - lo[0]) / band.cell[0])));
      band.n_cells[1]  = std::max(1, static_cast<int>(std::ceil((hi[1] - lo[1]) / band.cell[1])));
      band_slot[band_key] = index.bands.size();
      index.bands.push_back(band);
    }

    // Per-band counting sort. Each band owns a self-contained CSR segment of
    // n_cells + 1 offsets, based at the running total of already placed events,
    // so one flat cell_events array serves every band and for_each_neighbour
    // needs nothing but band.first_cell to address it.
    std::vector<std::vector<int>> band_events(index.bands.size());
    for (std::size_t i = 0; i < n; ++i)
      band_events[band_slot.at(key[i])].push_back(static_cast<int>(i));

    index.cell_events.reserve(n);
    for (std::size_t slot = 0; slot < index.bands.size(); ++slot) {
      KdeIndex::Band&   band    = index.bands[slot];
      const std::size_t n_cells = static_cast<std::size_t>(band.n_cells[0]) * band.n_cells[1];
      const std::size_t base    = index.cell_events.size();
      band.first_cell           = index.cell_offsets.size();

      auto cell_of = [&band, &x_e, &x_z, lo](const int i) {
        const std::size_t e  = static_cast<std::size_t>(i);
        const int         ie = std::clamp(static_cast<int>(std::floor((x_e[e] - lo[0]) / band.cell[0])),
                                          0, band.n_cells[0] - 1);
        const int         iz = std::clamp(static_cast<int>(std::floor((x_z[e] - lo[1]) / band.cell[1])),
                                          0, band.n_cells[1] - 1);
        return static_cast<std::size_t>(ie) * band.n_cells[1] + iz;
      };

      std::vector<std::size_t> counts(n_cells + 1, 0);
      for (const int i : band_events[slot]) ++counts[cell_of(i) + 1];
      for (std::size_t c = 1; c < counts.size(); ++c) counts[c] += counts[c - 1];

      for (const std::size_t c : counts) index.cell_offsets.push_back(base + c);

      index.cell_events.resize(base + band_events[slot].size());
      std::vector<std::size_t> cursor(counts.begin(), counts.end());
      for (const int i : band_events[slot]) index.cell_events[base + cursor[cell_of(i)]++] = i;
    }

    return index;
  }

}  // namespace io::ic
