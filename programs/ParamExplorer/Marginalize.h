#pragma once

#include "IceCube/Binning.h"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace explorer {

  /**
   * Project a quantity binned in `binning` onto one of its axes, summing over
   * every other axis.
   *
   * The binning is row-major over its axes (Binning's own contract:
   * flat = ((i0 * n1 + i1) * n2 + i2) ...), so the index on axis `axis` of a
   * flat bin is (flat / stride) % n, with `stride` the product of the bin counts
   * of the axes *after* it. One pass over the input, no allocation beyond the
   * result.
   *
   * `values` must hold exactly binning.total_bins() entries -- the analysis
   * binning, which is what both SampleLikelihood::data() and every entry of
   * result::ic::component_breakdown() are already expressed in.
   */
  [[nodiscard]] inline std::vector<double> project(std::span<const double>  values,
                                                   const io::ic::Binning&   binning,
                                                   std::size_t              axis) {
    const auto axes = binning.axes();
    if (axis >= axes.size())
      throw std::out_of_range("explorer::project: axis index beyond the binning");
    if (values.size() != static_cast<std::size_t>(binning.total_bins()))
      throw std::invalid_argument("explorer::project: values are not in the analysis binning");

    std::size_t stride = 1;
    for (std::size_t k = axis + 1; k < axes.size(); ++k)
      stride *= static_cast<std::size_t>(axes[k].n_bins);

    const std::size_t n = static_cast<std::size_t>(axes[axis].n_bins);

    std::vector<double> out(n, 0.0);
    for (std::size_t flat = 0; flat < values.size(); ++flat)
      out[(flat / stride) % n] += values[flat];

    return out;
  }

  /**
   * The n_bins + 1 edges of an axis: the explicit list when it has one, the
   * uniform grid otherwise. The non-uniform cascade zenith binning
   * ("cscd-cos_5up") is the reason the explicit case exists.
   */
  [[nodiscard]] inline std::vector<double> axis_edges(const io::ic::Axis& axis) {
    if (!axis.uniform())
      return axis.edges;

    std::vector<double> edges;
    edges.reserve(static_cast<std::size_t>(axis.n_bins) + 1);
    for (int i = 0; i <= axis.n_bins; ++i)
      edges.push_back(axis.lo + i * axis.step());

    return edges;
  }

}  // namespace explorer
