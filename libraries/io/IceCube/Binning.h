#pragma once

#include <span>
#include <string_view>
#include <vector>

namespace io::ic {

  /**
   * One analysis axis in a derived reconstructed quantity. Either a uniform grid
   * ((lo, hi, n_bins), `edges` empty) or an explicit ascending edge list
   * (`edges.size() == n_bins + 1`), which is what NNMFit's non-uniform cascade
   * zenith binning ("cscd-cos_5up") needs.
   */
  struct Axis {
    enum class Kind { Log10Energy, CosZenith, Ra };
    Kind                kind;
    double              lo;
    double              hi;
    int                 n_bins;
    std::vector<double> edges;  // empty => uniform

    [[nodiscard]] bool uniform() const noexcept { return edges.empty(); }
    [[nodiscard]] double step() const noexcept { return (hi - lo) / n_bins; }
    [[nodiscard]] double project(double raw_value) const noexcept;
    [[nodiscard]] int index(double raw_value) const noexcept;
  };

  /**
     * Parse an axis spec for the named kind ("Log10Energy"|"CosZenith"|"Ra"):
     *   "(lo, hi, n_bins)"        uniform grid
     *   "[e0, e1, ..., eN]"       explicit ascending edges, N bins
     */
  [[nodiscard]] Axis parse_axis(std::string_view kind, std::string_view spec);

  /** Config spelling of an axis kind; inverse of parse_axis' kind argument. */
  [[nodiscard]] std::string_view axis_kind_name(Axis::Kind kind) noexcept;

  /**
   * Runtime N-dimensional analysis binning, one per sample. Row-major over the
   * axis order: flat = ((i0 * n1 + i1) * n2 + i2) ... .
   */
  class Binning {
   public:
    explicit Binning(std::vector<Axis> axes);

    [[nodiscard]] int total_bins() const noexcept { return m_TotalBins; }
    [[nodiscard]] std::size_t n_axes() const noexcept { return m_Axes.size(); }
    [[nodiscard]] std::span<const Axis> axes() const noexcept { return m_Axes; }

    [[nodiscard]] int bin_index(std::span<const double> reco) const noexcept;

   private:
    std::vector<Axis> m_Axes;
    int               m_TotalBins;
  };

  /**
   * Per-bin event counts for a data sample: bins each (reco energy, reco zenith)
   * pair with `binning` and counts, dropping out-of-range events. No weights and
   * no livetime scaling -- real data is a count.
   */
  [[nodiscard]] std::vector<double> bin_event_counts(const Binning&             binning,
                                                    const std::vector<double>& reco_energy,
                                                    const std::vector<double>& reco_zenith);

}  // namespace io::ic
