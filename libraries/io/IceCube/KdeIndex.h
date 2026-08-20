#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace io::ic {

  /**
   * Spatial index over the unbinned KDE plane (log10 E, zenith): it answers
   * "which MC events can reach this query point" without testing all of them.
   *
   * It is only a lookup structure. No event is ever binned, no kernel is ever
   * evaluated at a cell centre and nothing is interpolated -- every kernel is
   * still evaluated at the exact coordinates of the exact pair of events the
   * walk returns, so the density stays the exact (truncated) kernel sum.
   *
   * A single uniform grid does not work here: the search radius would be set by
   * the sample's single worst-resolved event, so one event with a 30x bandwidth
   * makes every query walk 30x too many cells. Events are therefore split into
   * geometric bandwidth *bands* (per axis), and each band gets its own grid
   * whose cell size is that band's own search radius, n_sigma * h_max(band).
   * The stencil is then exactly 3x3 cells in every band, whatever the spread.
   *
   * Completeness: an event in band b has h <= h_max(b) on both axes, and band b
   * is searched one cell -- i.e. n_sigma * h_max(b) -- in each direction, so
   * every query within n_sigma * h of that event visits the cell holding it. The
   * walk may return extras, since a box is a superset of the ellipse; the
   * caller's kernel evaluation handles those for free, a far kernel being ~0.
   *
   * Built once at load time from parameter-independent inputs and never rebuilt:
   * it lives on ICSample, which ICExperimentModule caches per process, so a scan
   * building one Fit per grid point pays for it once.
   */
  struct KdeIndex {
    struct Band {
      std::array<double, 2> cell{};        ///< cell size per axis == n_sigma * h_max of the band
      std::array<int, 2>    n_cells{};     ///< grid extent per axis
      std::size_t           first_cell = 0;  ///< this band's first entry in cell_offsets
    };

    std::array<double, 2>    lo{};  ///< domain lower corner, shared by all bands
    std::vector<Band>        bands;
    /// Concatenated per-band CSR arrays, each of n_cells + 1 entries: cell c of
    /// band b spans [cell_offsets[b.first_cell + c], cell_offsets[b.first_cell + c + 1])
    /// in cell_events.
    std::vector<std::size_t> cell_offsets;
    std::vector<int>         cell_events;  ///< event indices, grouped by (band, cell)

    [[nodiscard]] bool empty() const noexcept { return cell_events.empty(); }
  };

  /**
   * Per-event constants derived from the bandwidths, hoisted out of the density's
   * innermost loop: reciprocal widths, the product prefactor of the two axes, and
   * the truncation reach per axis.
   *
   * Derived at load rather than per Fit for two reasons: a scan builds one Fit
   * per grid point and would otherwise rebuild ~30 MB of them each time, and the
   * GPU column cache is keyed on the host pointer with no invalidation, so a
   * per-Fit buffer could be recycled onto an address whose device copy still
   * holds the previous fit's numbers.
   */
  struct KdeKernelConstants {
    std::vector<double> inv_h_e;
    std::vector<double> inv_h_z;
    std::vector<double> prefactor;  ///< 1 / (2 pi h_e h_z)
    std::vector<double> reach_e;    ///< n_sigma * h_e
    std::vector<double> reach_z;    ///< n_sigma * h_z

    [[nodiscard]] bool empty() const noexcept { return inv_h_e.empty(); }
  };

  [[nodiscard]] KdeKernelConstants build_kde_kernel_constants(std::span<const double> h_e,
                                                              std::span<const double> h_z,
                                                              double                  n_sigma);

  /**
   * Build the index. `x_e`/`x_z` are the KDE coordinates and `h_e`/`h_z` the
   * per-event bandwidths, all in the same (analysis-bin CSR) index space; `lo`
   * and `hi` are the domain corners; `n_sigma` is the truncation radius in units
   * of the bandwidth.
   *
   * A non-positive or non-finite bandwidth is an error rather than a skipped
   * event: such an event has no kernel at all, and dropping it here would leave
   * the weights it carries out of the density while nu still counts them. The
   * loader drops those events instead.
   */
  [[nodiscard]] KdeIndex build_kde_index(std::span<const double> x_e,
                                         std::span<const double> x_z,
                                         std::span<const double> h_e,
                                         std::span<const double> h_z,
                                         std::array<double, 2>   lo,
                                         std::array<double, 2>   hi,
                                         double                  n_sigma);

  /**
   * Call `visit(event_index)` for every event that can reach (qe, qz): the 3x3
   * cell stencil around the query point in every band. Each event is visited at
   * most once, because it lives in exactly one (band, cell).
   */
  template <class Visit>
  void for_each_neighbour(const KdeIndex& index, const double qe, const double qz, Visit&& visit) {
    for (const auto& [cell, n_cells, first_cell] : index.bands) {
      const int ce = static_cast<int>(std::floor((qe - index.lo[0]) / cell[0]));
      const int cz = static_cast<int>(std::floor((qz - index.lo[1]) / cell[1]));

      for (int de = -1; de <= 1; ++de) {
        const int ie = ce + de;
        if (ie < 0 || ie >= n_cells[0]) continue;
        for (int dz = -1; dz <= 1; ++dz) {
          const int iz = cz + dz;
          if (iz < 0 || iz >= n_cells[1]) continue;

          const std::size_t cellIdx  = first_cell + static_cast<std::size_t>(ie) * n_cells[1] + iz;
          const std::size_t begin    = index.cell_offsets[cellIdx];
          const std::size_t end      = index.cell_offsets[cellIdx + 1];

          for (std::size_t k = begin; k < end; ++k) {
            visit(index.cell_events[k]);
          }
        }
      }
    }
  }

}  // namespace io::ic
