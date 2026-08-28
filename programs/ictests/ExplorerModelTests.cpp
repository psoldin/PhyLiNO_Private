// Projection tests for the parameter explorer's marginalization.
//
// explorer::project() is arithmetic on a std::span and a Binning, so these need
// no sample, no parquet fixture and no likelihood -- which is exactly why the
// stride walk is a free function rather than a method on ExplorerModel. The two
// checks that do need a built prediction live in ICTests.cpp with the synthetic
// sample fixtures.

#include "Marginalize.h"

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

using io::ic::Axis;
using io::ic::Binning;

namespace {

  // Log10Energy in [2, 5) with 3 bins, CosZenith in [-1, 1) with 2 bins.
  // Row-major: flat = energy_index * 2 + zenith_index, so 6 bins.
  Binning two_axis_binning() {
    return Binning({Axis{Axis::Kind::Log10Energy, 2.0, 5.0, 3},
                    Axis{Axis::Kind::CosZenith, -1.0, 1.0, 2}});
  }

}  // namespace

// Marginalizing over either axis of a 2D binning must preserve the total: the
// projection moves counts around, it does not create or destroy them.
TEST(ExplorerProjectTest, PreservesTheTotalOnEveryAxis) {
  const Binning binning = two_axis_binning();

  const std::vector<double> bins = {1.0, 2.0, 4.0, 8.0, 16.0, 32.0};
  ASSERT_EQ(bins.size(), static_cast<std::size_t>(binning.total_bins()));

  const double total = std::accumulate(bins.begin(), bins.end(), 0.0);

  for (std::size_t axis = 0; axis < binning.n_axes(); ++axis) {
    const auto projected = explorer::project(bins, binning, axis);
    EXPECT_EQ(projected.size(), static_cast<std::size_t>(binning.axes()[axis].n_bins));
    EXPECT_DOUBLE_EQ(std::accumulate(projected.begin(), projected.end(), 0.0), total);
  }
}

// The stride walk has to agree with Binning's row-major layout, not merely sum
// to the right total -- a wrong stride still conserves the sum while putting the
// counts in the wrong bins. These are the sums worked out by hand from
// flat = energy_index * 2 + zenith_index.
TEST(ExplorerProjectTest, GroupsByTheRowMajorLayout) {
  const Binning             binning = two_axis_binning();
  const std::vector<double> bins    = {1.0, 2.0, 4.0, 8.0, 16.0, 32.0};

  // Axis 0 (energy): consecutive pairs.
  const auto energy = explorer::project(bins, binning, 0);
  ASSERT_EQ(energy.size(), 3u);
  EXPECT_DOUBLE_EQ(energy[0], 1.0 + 2.0);
  EXPECT_DOUBLE_EQ(energy[1], 4.0 + 8.0);
  EXPECT_DOUBLE_EQ(energy[2], 16.0 + 32.0);

  // Axis 1 (zenith): every other entry.
  const auto zenith = explorer::project(bins, binning, 1);
  ASSERT_EQ(zenith.size(), 2u);
  EXPECT_DOUBLE_EQ(zenith[0], 1.0 + 4.0 + 16.0);
  EXPECT_DOUBLE_EQ(zenith[1], 2.0 + 8.0 + 32.0);
}

// A three-axis binning is the real case for an RA-binned sample, and it is the
// one where a stride computed from the wrong end of the axis list still passes
// the 2D tests.
TEST(ExplorerProjectTest, HandlesThreeAxes) {
  const Binning binning({Axis{Axis::Kind::Log10Energy, 2.0, 5.0, 2},
                         Axis{Axis::Kind::CosZenith, -1.0, 1.0, 2},
                         Axis{Axis::Kind::Ra, 0.0, 360.0, 3}});
  ASSERT_EQ(binning.total_bins(), 12);

  // Value = its own RA index, so the RA projection is a known non-uniform shape
  // (4 bins fall into each RA slice) and the other two are flat.
  std::vector<double> bins(12, 0.0);
  for (std::size_t flat = 0; flat < bins.size(); ++flat)
    bins[flat] = static_cast<double>(flat % 3);

  const auto ra = explorer::project(bins, binning, 2);
  ASSERT_EQ(ra.size(), 3u);
  EXPECT_DOUBLE_EQ(ra[0], 0.0);
  EXPECT_DOUBLE_EQ(ra[1], 4.0);
  EXPECT_DOUBLE_EQ(ra[2], 8.0);

  // Each of the two outer axes covers 6 flat bins holding two full RA cycles,
  // so both project flat at 0 + 1 + 2 twice.
  for (const std::size_t axis : {std::size_t{0}, std::size_t{1}}) {
    const auto projected = explorer::project(bins, binning, axis);
    ASSERT_EQ(projected.size(), 2u);
    EXPECT_DOUBLE_EQ(projected[0], 6.0);
    EXPECT_DOUBLE_EQ(projected[1], 6.0);
  }
}

// Projecting a single-bin axis is the identity on the total -- the degenerate
// case the cscd_muon sample actually is (1 analysis bin).
TEST(ExplorerProjectTest, SingleBinAxisIsTheIdentity) {
  const Binning             binning({Axis{Axis::Kind::Log10Energy, 2.0, 5.0, 1}});
  const std::vector<double> bins = {42.0};

  const auto projected = explorer::project(bins, binning, 0);
  ASSERT_EQ(projected.size(), 1u);
  EXPECT_DOUBLE_EQ(projected[0], 42.0);
}

// A span that is not in the analysis binning is a programming error the GUI
// would otherwise render as a silently wrong plot.
TEST(ExplorerProjectTest, RejectsMismatchedInput) {
  const Binning binning = two_axis_binning();

  const std::vector<double> too_short = {1.0, 2.0, 3.0};
  EXPECT_THROW(static_cast<void>(explorer::project(too_short, binning, 0)), std::invalid_argument);

  const std::vector<double> right_size(6, 1.0);
  EXPECT_THROW(static_cast<void>(explorer::project(right_size, binning, 2)), std::out_of_range);
}

// A uniform axis reports the grid its (lo, hi, n_bins) implies.
TEST(ExplorerAxisEdgesTest, UniformAxisIsTheArithmeticGrid) {
  const Axis axis{Axis::Kind::Log10Energy, 2.0, 5.0, 3};

  const auto edges = explorer::axis_edges(axis);
  ASSERT_EQ(edges.size(), 4u);
  EXPECT_DOUBLE_EQ(edges[0], 2.0);
  EXPECT_DOUBLE_EQ(edges[1], 3.0);
  EXPECT_DOUBLE_EQ(edges[2], 4.0);
  EXPECT_DOUBLE_EQ(edges[3], 5.0);
}

// An explicit edge list is returned verbatim, not re-derived from lo/hi -- this
// is NNMFit's non-uniform cascade zenith binning, where the arithmetic grid
// would be wrong.
TEST(ExplorerAxisEdgesTest, ExplicitEdgesArePreserved) {
  Axis axis{Axis::Kind::CosZenith, -1.0, 1.0, 4};
  axis.edges = {-1.0, -0.6, 0.0, 0.4, 1.0};

  EXPECT_EQ(explorer::axis_edges(axis), axis.edges);
}
