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
   * The analysis binning without its trailing Ra axis -- the binning MC events are
   * assigned to. Returns `binning` unchanged when it has no Ra axis, so a 2-axis
   * sample gets an identical binning back.
   *
   * Throws when the binning has an Ra axis that is not the last one: row-major
   * flattening only makes the 3D index equal (2D index * n_ra + ra index) -- the
   * layout the RA broadcast and NNMFit's repeat() both assume -- if RA is innermost.
   */
  [[nodiscard]] Binning drop_ra_axis(const Binning& binning);

  /**
   * Number of bins on the trailing Ra axis, or 1 when the binning has none.
   *
   * Assumes a well-formed binning: it only inspects the last axis, so a malformed
   * binning with a non-trailing Ra axis returns 1 instead of throwing, unlike
   * drop_ra_axis(). parse_samples validates every binning via drop_ra_axis() before
   * anything calls this, so that case should not reach here in practice.
   */
  [[nodiscard]] int ra_bin_count(const Binning& binning) noexcept;

  /**
   * Spread a quantity binned in the MC binning uniformly over the analysis binning's
   * RA axis:  out[b * n_ra + r] = mc_bins[b] / divisor  for every r.
   *
   * `divisor` is n_ra for mu and n_ra * n_ra for sigma^2, matching NNMFit's
   * Binning_2D_to_3D (make_binned_flux divides the repeated weights by n_ra, so their
   * square picks up n_ra^2). n_ra == 1 with divisor 1.0 is an exact copy. The caller is
   * responsible for passing a nonzero divisor.
   *
   * `out` must have exactly mc_bins.size() * n_ra entries.
   */
  void broadcast_over_ra(std::span<const double> mc_bins, int n_ra, double divisor,
                         std::span<double> out);

  /**
   * Per-bin event counts for a data sample: bins each (reco energy, reco zenith)
   * pair with `binning` and counts, dropping out-of-range events. No weights and
   * no livetime scaling -- real data is a count.
   */
  [[nodiscard]] std::vector<double> bin_event_counts(const Binning&             binning,
                                                    const std::vector<double>& reco_energy,
                                                    const std::vector<double>& reco_zenith);

  /** As above, for an analysis binning whose third axis is Ra. */
  [[nodiscard]] std::vector<double> bin_event_counts(const Binning&             binning,
                                                     const std::vector<double>& reco_energy,
                                                     const std::vector<double>& reco_zenith,
                                                     const std::vector<double>& reco_ra);

}  // namespace io::ic
