#include "AdaptiveGrid.h"

#include <gtest/gtest.h>

// STL includes
#include <cmath>
#include <vector>

/**
 * @file
 * @brief Tests for the quadtree refinement of the two-dimensional scan.
 *
 * The refinement has to place its fits along the confidence contours whatever
 * shape those take, so the surfaces used here are deliberately awkward: a
 * banana, a region running off the edge of the window, and two disconnected
 * islands. What is checked in each case is that every contour is resolved at
 * full depth and that this costs far fewer points than the uniform grid of the
 * same resolution would have.
 */
namespace {

  /// Runs the refinement over an analytic surface, counting evaluations.
  struct Harness {
    scan::Settings settings;
    int            evaluations = 0;

    /// @param surface Callable mapping (x, y) in [0,1]^2 to a likelihood.
    template <typename Function>
    scan::Surface run(Function&& surface) {
      const double spacing_x = 1.0 / settings.lattice_x();
      const double spacing_y = 1.0 / settings.lattice_y();

      auto evaluate = [&](const std::vector<scan::Node>& nodes, scan::Surface& values) {
        for (const scan::Node& node : nodes) {
          values[node] = surface(node.x * spacing_x, node.y * spacing_y);
          ++evaluations;
        }
      };

      return scan::refine(settings, std::numeric_limits<double>::infinity(), evaluate, [](int, std::size_t, std::size_t, std::size_t) {});
    }

    /// Points a uniform grid of the finest resolution would have needed.
    [[nodiscard]] int uniform_equivalent() const { return (settings.lattice_x() + 1) * (settings.lattice_y() + 1); }
  };

  /// A banana-shaped region: a curved, strongly correlated valley.
  ///
  /// Returned in -2 log L units, the same convention the fit hands the
  /// refinement, so the value is the delta chi2 itself rather than half of it.
  double banana(double x, double y) {
    const double along  = (x - 0.5) / 0.30;
    const double across = (y - 0.5 - 0.8 * (x - 0.5) * (x - 0.5) + 0.05) / 0.04;
    return along * along + across * across;
  }

  /// True if the cell of the finest depth containing (x, y) was actually evaluated.
  bool resolved_at_full_depth(const scan::Surface& surface, const scan::Settings& settings, double x, double y) {
    const int ix = static_cast<int>(std::lround(x * settings.lattice_x()));
    const int iy = static_cast<int>(std::lround(y * settings.lattice_y()));

    return surface.contains(scan::Node{ix, iy});
  }

}  // namespace

/// Every level named in the settings must be traced at the finest resolution,
/// not just the outermost one.
TEST(AdaptiveGrid, ResolvesEveryContourOfABananaAtFullDepth) {
  Harness harness;
  const auto surface = harness.run(banana);

  const double spacing = 1.0 / harness.settings.lattice_x();

  for (const double level : harness.settings.levels) {
    int found = 0;
    // Walk the ridge of the banana and check the crossing of this level is
    // sampled on the finest lattice on both sides of the valley.
    for (double x = 0.25; x <= 0.75; x += 0.05) {
      for (const int direction : {-1, 1}) {
        double crossing = 0.0;
        for (double y = 0.5 + 0.8 * (x - 0.5) * (x - 0.5) - 0.05; y > 0.0 && y < 1.0; y += direction * spacing)
          if (banana(x, y) >= level) {
            crossing = y;
            break;
          }

        if (crossing > 0.0 && resolved_at_full_depth(surface, harness.settings, x, crossing))
          ++found;
      }
    }

    EXPECT_GT(found, 15) << "level " << level << " is not sampled at full depth along the banana";
  }
}

/// The point of the refinement: the same contour resolution for a fraction of the fits.
TEST(AdaptiveGrid, CostsFarLessThanTheEquivalentUniformGrid) {
  Harness banana_harness;
  banana_harness.run(banana);
  EXPECT_LT(banana_harness.evaluations, banana_harness.uniform_equivalent() / 3);

  // A region shaped like the real one - a strongly anticorrelated ellipse - is
  // what the saving is actually claimed for. A uniform 50x50 scan is the thing
  // being replaced, and this has to beat it while resolving the contours
  // roughly twice as finely.
  Harness harness;
  harness.run([](double x, double y) {
    constexpr double sigma_x = 0.05;
    constexpr double sigma_y = 0.0867;
    constexpr double rho     = -0.8;

    const double dx = (x - 0.6727) / sigma_x;
    const double dy = (y - 0.59) / sigma_y;
    return (dx * dx - 2.0 * rho * dx * dy + dy * dy) / (1.0 - rho * rho);
  });

  EXPECT_LT(harness.evaluations, 2500);
  EXPECT_LT(harness.evaluations, harness.uniform_equivalent() / 4);
}

/// A region that runs off the edge of the window must still be traced, rather
/// than the refinement stopping where the region has no closed boundary.
TEST(AdaptiveGrid, ResolvesARegionOpenAtTheEdgeOfTheWindow) {
  Harness harness;

  // Minimum sits on the left edge; the region is cut off by the window.
  auto open_region = [](double x, double y) {
    const double along  = x / 0.25;
    const double across = (y - 0.5) / 0.10;
    return along * along + across * across;
  };

  const auto surface = harness.run(open_region);

  const double spacing = 1.0 / harness.settings.lattice_y();
  double       crossing = 0.5;
  while (open_region(0.0, crossing) < harness.settings.levels.back())
    crossing += spacing;

  EXPECT_TRUE(resolved_at_full_depth(surface, harness.settings, 0.0, crossing));
}

/// Two separate minima: the refinement follows both, because the decision to
/// split is local to a cell and carries no notion of a single region.
TEST(AdaptiveGrid, ResolvesDisconnectedIslands) {
  Harness harness;

  auto islands = [](double x, double y) {
    auto well = [&](double cx, double cy) {
      const double dx = (x - cx) / 0.06;
      const double dy = (y - cy) / 0.06;
      return dx * dx + dy * dy;
    };
    return std::min(well(0.25, 0.25), well(0.75, 0.75));
  };

  const auto surface = harness.run(islands);

  const double spacing = 1.0 / harness.settings.lattice_x();
  for (const auto& [cx, cy] : {std::pair{0.25, 0.25}, std::pair{0.75, 0.75}}) {
    double crossing = cx;
    while (islands(crossing, cy) < harness.settings.levels.back())
      crossing += spacing;

    EXPECT_TRUE(resolved_at_full_depth(surface, harness.settings, crossing, cy))
        << "island at (" << cx << ", " << cy << ") is not traced at full depth";
  }
}

/// The saving comes from *not* refining where there is nothing to resolve, so
/// the far outside of the region must stay at coarse resolution.
TEST(AdaptiveGrid, LeavesTheOutsideOfTheRegionCoarse) {
  Harness harness;
  const auto surface = harness.run(banana);

  const int step = harness.settings.coarse_step();

  // A corner of the window, far outside every contour of the banana.
  ASSERT_GT(banana(0.02, 0.95), harness.settings.levels.back());
  EXPECT_TRUE(surface.contains(scan::Node{0, harness.settings.lattice_y()})) << "the coarse grid itself must still cover the corner";

  int refined_outside = 0;
  for (const auto& [node, value] : surface) {
    if (node.x % step == 0 && node.y % step == 0)
      continue;  // a coarse-grid point, not a refinement

    const double x = static_cast<double>(node.x) / harness.settings.lattice_x();
    const double y = static_cast<double>(node.y) / harness.settings.lattice_y();
    if (banana(x, y) > 4.0 * harness.settings.levels.back())
      ++refined_outside;
  }

  // Cells straddling the outermost contour are refined and reach a little way
  // past it, but nothing far outside should have been touched.
  EXPECT_EQ(refined_outside, 0);
}
