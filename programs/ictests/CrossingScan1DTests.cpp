#include "CrossingScan1D.h"

#include <gtest/gtest.h>

// STL includes
#include <cmath>
#include <limits>
#include <vector>

/**
 * @file
 * @brief Tests for the direct search of the 1 sigma crossings of a 1D profile.
 *
 * What the search has to get right is the crossing itself, to well below the
 * spacing a full scan would sample at, and that it gets there in a handful of
 * fits whatever the first probe step was. The curves are the same awkward kind
 * the walk is tested on: asymmetric, far narrower or wider than the probe,
 * non-parabolic, cut off by a bound, with a minimum off the anchor.
 */
namespace {

  /// Runs the search over an analytic profile, counting evaluations.
  ///
  /// One lattice unit is 0.001 of the parameter, so the default start_step of
  /// 64 is a first probe of 0.064 -- deliberately mismatched to most curves.
  struct Harness {
    scan1d::crossing::Settings settings;
    int                        evaluations = 0;
    double                     unit        = 0.001;
    double                     seed        = std::numeric_limits<double>::infinity();

    template <typename Function>
    scan1d::crossing::Result run(Function&& curve) {
      auto evaluate = [&](const std::vector<int>& nodes, scan1d::Profile& values) {
        for (const int node : nodes) {
          values[node] = curve(node * unit);
          ++evaluations;
        }
      };

      return scan1d::crossing::search(settings, seed, evaluate,
                                      [](int, std::size_t, std::size_t) {});
    }

    [[nodiscard]] double lower(const scan1d::crossing::Result& result) const { return result.sides[0].crossing * unit; }
    [[nodiscard]] double upper(const scan1d::crossing::Result& result) const { return result.sides[1].crossing * unit; }
  };

  /// Steep on the left (sigma 0.04), shallow on the right (sigma 0.12), in -2 log L.
  double asymmetric(double x) {
    const double d = x / (x < 0.0 ? 0.04 : 0.12);
    return d * d;
  }

  using scan1d::crossing::Status;

}  // namespace

/// The point of the whole search: both crossings, precisely, without a window.
TEST(CrossingScan1D, FindsBothCrossingsOfAnAsymmetricProfile) {
  Harness    harness;
  const auto result = harness.run(asymmetric);

  EXPECT_EQ(result.sides[0].status, Status::converged);
  EXPECT_EQ(result.sides[1].status, Status::converged);
  EXPECT_NEAR(harness.lower(result), -0.04, 1e-4);
  EXPECT_NEAR(harness.upper(result), 0.12, 1e-4);
}

/// The reason to have it next to the walk: a parabola is solved in a probe and
/// one secant step per side, whatever the probe got wrong about its width.
TEST(CrossingScan1D, NeedsFewFitsOnAParabolaOfAnyWidth) {
  for (const double sigma : {0.005, 0.064, 0.7}) {
    Harness    harness;
    const auto result = harness.run([sigma](double x) { return x * x / (sigma * sigma); });

    EXPECT_NEAR(harness.upper(result), sigma, 1e-3 * sigma + harness.unit) << "sigma " << sigma;
    EXPECT_NEAR(harness.lower(result), -sigma, 1e-3 * sigma + harness.unit) << "sigma " << sigma;
    EXPECT_LE(harness.evaluations, 9) << "sigma " << sigma;
  }
}

/// Far from parabolic: z is no longer linear, and the bracket phase has to do
/// the work. The quartic term triples the rise at the true crossing.
TEST(CrossingScan1D, ConvergesOnANonParabolicProfile) {
  Harness    harness;
  const auto curve  = [](double x) { const double d = x / 0.1; return d * d + 2.0 * d * d * d * d; };
  const auto result = harness.run(curve);

  // d^2 + 2 d^4 = 1  ->  d^2 = 0.5
  const double expected = 0.1 * std::sqrt(0.5);
  EXPECT_NEAR(harness.upper(result), expected, 2e-4);
  EXPECT_NEAR(harness.lower(result), -expected, 2e-4);
  EXPECT_LE(harness.evaluations, 15);
}

/// A crossing outside the configured bound is reported as the bound.
TEST(CrossingScan1D, StopsAtABoundBelowTheLevel) {
  Harness harness;
  harness.settings.lower_limit = -20;  // 0.02, well inside the 1 sigma of 0.1
  const auto result            = harness.run([](double x) { return x * x / 0.01; });

  EXPECT_EQ(result.sides[0].status, Status::at_limit);
  EXPECT_DOUBLE_EQ(harness.lower(result), -0.02);
  EXPECT_EQ(result.sides[1].status, Status::converged);
  EXPECT_NEAR(harness.upper(result), 0.1, 1e-4);
}

/// A best fit at the bound has no side to search below it at all.
TEST(CrossingScan1D, MinimumOnTheBoundIsAtLimit) {
  Harness harness;
  harness.settings.lower_limit = 0;
  const auto result            = harness.run([](double x) { return x * x / 0.01; });

  EXPECT_EQ(result.sides[0].status, Status::at_limit);
  EXPECT_DOUBLE_EQ(harness.lower(result), 0.0);
}

/// After an undercut the minimum is no longer node 0 and can sit between
/// nodes. The crossings are then only as precise as the tolerance (1 % of
/// sigma), not as the interpolation of a bracket.
TEST(CrossingScan1D, MeasuresFromTheFreeFitOffTheLattice) {
  Harness harness;
  harness.seed      = 0.0;
  const auto result = harness.run([](double x) { const double d = (x - 0.0137) / 0.05; return d * d; });

  EXPECT_NEAR(harness.lower(result), 0.0137 - 0.05, 0.01 * 0.05);
  EXPECT_NEAR(harness.upper(result), 0.0137 + 0.05, 0.01 * 0.05);
  EXPECT_FALSE(result.undercut);
}

/// A free fit that stopped short of the minimum cannot be repaired by a search
/// that never samples near the minimum. It has to say so.
TEST(CrossingScan1D, FlagsAScanPointThatUndercutsTheFreeFit) {
  Harness harness;
  harness.seed      = 0.5;  // the free fit stopped at 0, the minimum is at 0.03
  const auto result = harness.run([](double x) { const double d = (x - 0.03) / 0.05; return d * d; });

  EXPECT_TRUE(result.undercut);
}

/// A tail without a likelihood: the side backs off towards the last good point
/// instead of extrapolating past it.
TEST(CrossingScan1D, BacksOffFromFailedFits) {
  Harness    harness;
  const auto result = harness.run([](double x) {
    return x > 0.15 ? std::numeric_limits<double>::quiet_NaN() : x * x / (0.1 * 0.1);
  });

  EXPECT_EQ(result.sides[1].status, Status::converged);
  EXPECT_NEAR(harness.upper(result), 0.1, 1e-4);
}

/// A profile that never reaches the level ends at max_points, not in a loop.
TEST(CrossingScan1D, GivesUpOnAFlatProfile) {
  Harness    harness;
  const auto result = harness.run([](double) { return 0.0; });

  // Either it walks into the lattice limit or it runs out of points first.
  EXPECT_NE(result.sides[0].status, Status::converged);
  EXPECT_NE(result.sides[1].status, Status::converged);
  EXPECT_LE(harness.evaluations, 2 * harness.settings.max_points + 1);
}

/// Any level works the same way: here the 2 sigma crossing.
TEST(CrossingScan1D, FindsOtherLevels) {
  Harness harness;
  harness.settings.level = 4.0;
  const auto result      = harness.run(asymmetric);

  EXPECT_NEAR(harness.lower(result), -0.08, 1e-4);
  EXPECT_NEAR(harness.upper(result), 0.24, 1e-4);
}

/// A batch spends its extra points around the estimate and so needs fewer rounds.
TEST(CrossingScan1D, BatchesKeepTheResult) {
  Harness harness;
  harness.settings.batch = 3;
  const auto result      = harness.run(asymmetric);

  EXPECT_NEAR(harness.lower(result), -0.04, 1e-4);
  EXPECT_NEAR(harness.upper(result), 0.12, 1e-4);
}

/// Shaped after the AstroNorm profile of the IceCube tracks Asimov scan, in
/// lattice units of 1/64 of the Hesse error: the upper side is three times
/// narrower than Hesse says, the lower side about as wide. The first secant on
/// the upper side lands a hair outside the tolerance (delta chi2 0.975), which
/// used to cost a fit at every node until two neighbours bracketed the level.
/// A bracket of a few nodes already pins the crossing through the
/// interpolation in z.
TEST(CrossingScan1D, StopsOnANarrowBracket) {
  Harness harness;
  harness.unit = 1.0;
  const auto result = harness.run([](double x) {
    const double d = x / (x < 0.0 ? 65.65 : 23.29);
    return d * d;
  });

  EXPECT_EQ(result.sides[0].status, Status::converged);
  EXPECT_EQ(result.sides[1].status, Status::converged);
  EXPECT_NEAR(harness.lower(result), -65.65, 0.1);
  EXPECT_NEAR(harness.upper(result), 23.29, 0.1);
  // Node 0, then -64 and -66 below, 64, 23 and 25 above.
  EXPECT_LE(harness.evaluations, 6);
}
