#pragma once

#include <span>
#include <string_view>
#include <vector>

namespace io::ic {

  /** One analysis axis: a uniform grid in a derived reconstructed quantity. */
  struct Axis {
    enum class Kind { Log10Energy, CosZenith, Ra };
    Kind   kind;
    double lo;
    double hi;
    int    n_bins;

    [[nodiscard]] double step() const noexcept { return (hi - lo) / n_bins; }
    [[nodiscard]] double project(double raw_value) const noexcept;
    [[nodiscard]] int index(double raw_value) const noexcept;
  };

  /** Parse "(lo, hi, n_bins)" for an axis of the named kind ("Log10Energy"|"CosZenith"|"Ra"). */
  [[nodiscard]] Axis parse_axis(std::string_view kind, std::string_view spec);

  /**
   * Runtime N-dimensional analysis binning. Row-major over the axis order:
   * flat = ((i0 * n1 + i1) * n2 + i2) ... . Replaces the old constexpr
   * Constants::nBins / bin_index for a single fixed grid.
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

}  // namespace io::ic
