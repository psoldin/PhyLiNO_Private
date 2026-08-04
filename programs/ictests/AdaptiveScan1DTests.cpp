#include "AdaptiveScan1D.h"

#include <gtest/gtest.h>

// STL includes
#include <cmath>
#include <limits>
#include <vector>

/**
 * @file
 * @brief Tests for the binary refinement of the one-dimensional profile scan.
 *
 * What a profile is read for is the position of its minimum and the points
 * where it cuts delta chi2 = 1, 4 and 9. The profiles used here are awkward on
 * purpose - asymmetric, cut off by the window, double-minimum - and in each
 * case the check is that those crossings land on the finest lattice while the
 * flat wings stay coarse.
 */
namespace {

  /// Runs the refinement over an analytic profile, counting evaluations.
  struct Harness {
    scan1d::Settings settings;
    int              evaluations = 0;

    /// @param curve Callable mapping x in [0,1] to a likelihood.
    template <typename Function>
    scan1d::Profile run(Function&& curve) {
      const double spacing = 1.0 / settings.lattice();

      auto evaluate = [&](const std::vector<int>& nodes, scan1d::Profile& values) {
        for (const int node : nodes) {
          values[node] = curve(node * spacing);
          ++evaluations;
        }
      };

      return scan1d::refine(settings, std::numeric_limits<double>::infinity(), evaluate, [](int, std::size_t, std::size_t, std::size_t) {});
    }

    /// Points a uniform scan of the finest resolution would have needed.
    [[nodiscard]] int uniform_equivalent() const { return settings.lattice() + 1; }
  };

  /// An asymmetric well: steep on the left, shallow on the right, so the three
  /// levels are crossed at six clearly separated places.
  ///
  /// Returned in -2 log L units, the same convention the fit hands the
  /// refinement, so the value is the delta chi2 itself rather than half of it.
  double asymmetric(double x) {
    const double d = (x - 0.4) / (x < 0.4 ? 0.04 : 0.12);
    return d * d;
  }

  /// True if the point of the finest depth nearest x was actually evaluated.
  bool resolved_at_full_depth(const scan1d::Profile& profile, const scan1d::Settings& settings, double x) {
    return profile.contains(static_cast<int>(std::lround(x * settings.lattice())));
  }

  /// Where `curve` first reaches `level`, walking from `from` in `direction`.
  /// Zero if the window ends first.
  template <typename Function>
  double crossing_of(Function&& curve, const scan1d::Settings& settings, double from, int direction, double level) {
    const double spacing = 1.0 / settings.lattice();
    for (double x = from; x >= 0.0 && x <= 1.0; x += direction * spacing)
      if (curve(x) >= level)
        return x;

    return 0.0;
  }

}  // namespace

/// Every level named in the settings must be located at the finest resolution,
/// on both sides of the minimum, not just the outermost one.
TEST(AdaptiveScan1D, ResolvesEveryCrossingAtFullDepth) {
  Harness    harness;
  const auto profile = harness.run(asymmetric);

  for (const double level : harness.settings.levels) {
    for (const int direction : {-1, 1}) {
      const double crossing = crossing_of(asymmetric, harness.settings, 0.4, direction, level);
      ASSERT_GT(crossing, 0.0) << "level " << level << " is not crossed inside the window";
      EXPECT_TRUE(resolved_at_full_depth(profile, harness.settings, crossing))
          << "level " << level << " at x = " << crossing << " is not sampled at full depth";
    }
  }
}

/// The minimum is what every reported interval is measured from, so the region
/// around it is refined as hard as the crossings themselves.
TEST(AdaptiveScan1D, ResolvesTheMinimumAtFullDepth) {
  Harness    harness;
  const auto profile = harness.run(asymmetric);

  EXPECT_TRUE(resolved_at_full_depth(profile, harness.settings, 0.4));
}

/// The point of the refinement: the same resolution for a fraction of the fits.
TEST(AdaptiveScan1D, CostsLessThanTheEquivalentUniformScan) {
  Harness harness;
  harness.run(asymmetric);

  EXPECT_LT(harness.evaluations, harness.uniform_equivalent());
}

/// A profile still falling at the edge of the window must be refined up to that
/// edge, rather than the refinement stopping where there is no crossing to find.
TEST(AdaptiveScan1D, ResolvesAProfileOpenAtTheEdgeOfTheWindow) {
  Harness harness;

  // Minimum sits on the left edge; the rise is cut off by the window.
  auto open_profile = [](double x) {
    const double d = x / 0.25;
    return d * d;
  };

  const auto profile = harness.run(open_profile);

  EXPECT_TRUE(resolved_at_full_depth(profile, harness.settings, 0.0));

  const double crossing = crossing_of(open_profile, harness.settings, 0.0, 1, harness.settings.levels.back());
  ASSERT_GT(crossing, 0.0);
  EXPECT_TRUE(resolved_at_full_depth(profile, harness.settings, crossing));
}

/// Two separate minima: the refinement follows both, because the decision to
/// split is local to an interval and carries no notion of a single region.
TEST(AdaptiveScan1D, ResolvesDisconnectedMinima) {
  Harness harness;

  auto minima = [](double x) {
    auto well = [&](double centre) {
      const double d = (x - centre) / 0.05;
      return d * d;
    };
    return std::min(well(0.25), well(0.75));
  };

  const auto profile = harness.run(minima);

  for (const double centre : {0.25, 0.75}) {
    const double crossing = crossing_of(minima, harness.settings, centre, 1, harness.settings.levels.back());
    ASSERT_GT(crossing, 0.0);
    EXPECT_TRUE(resolved_at_full_depth(profile, harness.settings, crossing))
        << "minimum at " << centre << " is not traced at full depth";
  }
}

/// The saving comes from *not* refining where there is nothing to resolve, so
/// the far wings must stay at coarse resolution.
TEST(AdaptiveScan1D, LeavesTheWingsCoarse) {
  Harness    harness;
  const auto profile = harness.run(asymmetric);

  const int step = harness.settings.coarse_step();

  ASSERT_GT(asymmetric(1.0), harness.settings.levels.back());
  EXPECT_TRUE(profile.contains(harness.settings.lattice())) << "the coarse scan itself must still cover the edge";

  int refined_outside = 0;
  for (const auto& [node, value] : profile) {
    if (node % step == 0)
      continue;  // a coarse-scan point, not a refinement

    if (asymmetric(static_cast<double>(node) / harness.settings.lattice()) > 4.0 * harness.settings.levels.back())
      ++refined_outside;
  }

  // Intervals straddling the outermost level are refined and reach a little way
  // past it, but nothing far out in the wings should have been touched.
  EXPECT_EQ(refined_outside, 0);
}
