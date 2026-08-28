#include "AdaptiveScan1D.h"

#include <gtest/gtest.h>

// STL includes
#include <cmath>
#include <limits>
#include <vector>

/**
 * @file
 * @brief Tests for the outward walk of the one-dimensional profile scan.
 *
 * What the walk has to get right is where it stops and how densely it samples on
 * the way there: far enough that the target rise is bracketed on both sides,
 * never much further, and with points spaced by the curve rather than by a step
 * fixed in advance. The profiles used here are awkward on purpose -- asymmetric,
 * far narrower or far wider than the starting step, cut off by a bound, flat --
 * and none of them is a shape the walk is allowed to assume.
 */
namespace {

  /// Runs the walk over an analytic profile, counting evaluations.
  ///
  /// One lattice unit is 0.001 of the parameter, so a settings.start_step of 64
  /// is a first step of 0.064 -- deliberately mismatched to most of the curves
  /// below, which is the case the step adaptation exists for.
  struct Harness {
    scan1d::Settings settings;
    int              evaluations = 0;
    double           unit        = 0.001;

    /// @param curve Callable mapping the parameter value to a likelihood. The
    ///              walk anchors at 0, so the curves here have their minimum there.
    template <typename Function>
    scan1d::Profile run(Function&& curve) {
      auto evaluate = [&](const std::vector<int>& nodes, scan1d::Profile& values) {
        for (const int node : nodes) {
          values[node] = curve(node * unit);
          ++evaluations;
        }
      };

      return scan1d::walk(settings, std::numeric_limits<double>::infinity(), evaluate,
                          [](int, std::size_t, std::size_t) {});
    }

    [[nodiscard]] double position(int node) const { return node * unit; }
  };

  /// An asymmetric well: steep on the left, shallow on the right, so the two
  /// directions have to stop at very different distances.
  ///
  /// Returned in -2 log L units, the same convention the fit hands the walk, so
  /// the value is the delta chi2 itself rather than half of it.
  double asymmetric(double x) {
    const double d = x / (x < 0.0 ? 0.04 : 0.12);
    return d * d;
  }

  /// Highest value reached on either side of the minimum.
  std::pair<double, double> reach(const scan1d::Profile& profile) {
    return {profile.begin()->second, profile.rbegin()->second};
  }

}  // namespace

/// The point of the whole change: the walk stops where the likelihood says to,
/// on both sides, without a window being handed to it.
TEST(AdaptiveScan1D, ReachesTheTargetOnBothSides) {
  Harness    harness;
  const auto profile = harness.run(asymmetric);

  const auto [low, high] = reach(profile);
  EXPECT_GE(low, harness.settings.target);
  EXPECT_GE(high, harness.settings.target);
}

/// Stopping at the target is only useful if it does not run far past it: the
/// points beyond the last crossing are the ones that buy nothing.
TEST(AdaptiveScan1D, StopsShortlyPastTheTarget) {
  Harness    harness;
  const auto profile = harness.run(asymmetric);

  // A round proposes a whole batch at once, so the overshoot is bounded by the
  // batch rather than by a single step.
  const auto [low, high] = reach(profile);
  EXPECT_LT(low, 10.0 * harness.settings.target);
  EXPECT_LT(high, 10.0 * harness.settings.target);
}

/// The two sides of an asymmetric profile are three times apart in width, and
/// the walk has to find that on its own rather than covering both alike.
TEST(AdaptiveScan1D, FollowsAnAsymmetricProfile) {
  Harness    harness;
  const auto profile = harness.run(asymmetric);

  const double left  = -harness.position(profile.begin()->first);
  const double right = harness.position(profile.rbegin()->first);

  EXPECT_GT(right, 1.5 * left) << "the shallow side must be scanned further than the steep one";
}

/// A profile far narrower than the starting step: the walk has to shrink its
/// step instead of jumping straight past the whole interesting region.
TEST(AdaptiveScan1D, ResolvesAProfileNarrowerThanTheStartingStep) {
  Harness harness;

  auto narrow = [](double x) {
    const double d = x / 0.002;
    return d * d;
  };

  const auto profile = harness.run(narrow);

  // Delta chi2 = 1 sits at x = 0.002, i.e. 2 lattice units out. Points inside
  // the 1 sigma interval are what a profile is read for.
  int inside = 0;
  for (const auto& [node, value] : profile)
    if (value < 1.0)
      ++inside;

  EXPECT_GE(inside, 3) << "the 1 sigma region must be sampled, not stepped over";
}

/// The mirror case: a profile far wider than the starting step must not cost
/// hundreds of points to walk out to the target.
TEST(AdaptiveScan1D, GrowsItsStepOnAWideProfile) {
  Harness harness;

  auto wide = [](double x) {
    const double d = x / 4.0;
    return d * d;
  };

  const auto profile = harness.run(wide);

  const auto [low, high] = reach(profile);
  EXPECT_GE(low, harness.settings.target);
  EXPECT_GE(high, harness.settings.target);
  EXPECT_LT(static_cast<int>(profile.size()), 100);
}

/// A configured bound cuts a direction short. The other side is unaffected and
/// still walks all the way to the target.
TEST(AdaptiveScan1D, StopsAtAConfiguredBound) {
  Harness harness;
  harness.settings.upper_limit = 20;

  const auto profile = harness.run(asymmetric);

  EXPECT_EQ(profile.rbegin()->first, 20);
  EXPECT_LT(profile.rbegin()->second, harness.settings.target) << "the bound, not the target, is what stopped this side";
  EXPECT_GE(profile.begin()->second, harness.settings.target);
}

/// A parameter the data does not constrain never reaches the target. The walk
/// has to give up on its own rather than run forever.
TEST(AdaptiveScan1D, GivesUpOnAFlatProfile) {
  Harness harness;
  harness.settings.max_points = 20;

  const auto profile = harness.run([](double) { return 0.0; });

  EXPECT_LE(static_cast<int>(profile.size()), 2 * harness.settings.max_points + 1 + 2 * harness.settings.batch);
  EXPECT_GT(static_cast<int>(profile.size()), 1);
}

/// A point without a usable likelihood must not be extrapolated through: the
/// walk backs off to a smaller step and keeps the points it can still fit.
TEST(AdaptiveScan1D, SurvivesPointsWithoutALikelihood) {
  Harness harness;

  // Everything past 0.05 fails, which is well before the target is reached.
  const auto profile = harness.run([](double x) {
    if (std::abs(x) > 0.05)
      return std::numeric_limits<double>::quiet_NaN();
    return asymmetric(x);
  });

  EXPECT_GT(static_cast<int>(profile.size()), 1);
  for (const auto& [node, value] : profile)
    if (std::abs(node * 0.001) <= 0.05)
      EXPECT_TRUE(std::isfinite(value));
}

/// The walk grows its step while the profile is flat, which can clear a whole
/// stretch -- the 1 sigma crossing included -- in one jump. Nothing below the
/// target may be left coarser than one step's worth of likelihood.
TEST(AdaptiveScan1D, LeavesNoGapWiderThanOneStepsRise) {
  Harness    harness;
  const auto profile = harness.run(asymmetric);

  for (auto point = profile.begin(); std::next(point) != profile.end(); ++point) {
    const auto [left, left_value]   = *point;
    const auto [right, right_value] = *std::next(point);

    if (right - left < 2 || std::min(left_value, right_value) >= harness.settings.target)
      continue;

    EXPECT_LE(std::abs(right_value - left_value), harness.settings.gain)
        << "gap between " << harness.position(left) << " and " << harness.position(right) << " is unsampled";
  }
}

/// The 1 sigma crossing is what a profile is read for, and on the steep side it
/// sits well inside the first step the walk takes.
TEST(AdaptiveScan1D, SamplesAroundTheOneSigmaCrossing) {
  Harness    harness;
  const auto profile = harness.run(asymmetric);

  for (const int direction : {-1, 1}) {
    int nearby = 0;
    for (const auto& [node, value] : profile)
      if (node * direction > 0 && value > 0.25 && value < 4.0)
        ++nearby;

    EXPECT_GE(nearby, 2) << "the 1 sigma crossing is not bracketed on side " << direction;
  }
}
