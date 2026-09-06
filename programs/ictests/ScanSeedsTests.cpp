#include "AdaptiveGrid.h"
#include "ScanSeeds.h"
#include "SeedQuality.h"

#include <gtest/gtest.h>

// STL includes
#include <cmath>
#include <limits>
#include <string>
#include <map>
#include <thread>
#include <vector>

/**
 * @file
 * @brief Tests for the start values a scan point takes from its neighbours.
 *
 * The store decides which earlier fit a new scan point starts from, and getting
 * that wrong is quiet: the scan still produces a surface, only a worse-converged
 * one. What is checked here is that "nearest" means nearest in the window rather
 * than in lattice units, that a point with no fitted neighbour falls back
 * exactly once, and that the concurrent scan workers cannot tear the store.
 */
namespace {

  /// The distance the 1D scan measures with.
  auto linear_distance = [](int a, int b) { return std::abs(static_cast<double>(a - b)); };

  /// The distance the 2D scan measures with, for a window whose two axes have
  /// different lattice spacings.
  auto grid_distance(double spacing_x, double spacing_y) {
    return [spacing_x, spacing_y](const scan::Node& a, const scan::Node& b) {
      return std::hypot(static_cast<double>(a.x - b.x) * spacing_x, static_cast<double>(a.y - b.y) * spacing_y);
    };
  }

}  // namespace

TEST(ScanSeeds, EmptyStoreWithoutFallbackGivesNothing) {
  const scanseed::Store<int> seeds;

  EXPECT_TRUE(seeds.nearest(0, linear_distance).empty());
}

TEST(ScanSeeds, EmptyStoreFallsBackToTheFreeFit) {
  scanseed::Store<int> seeds;
  seeds.set_fallback({1.0, 2.0, 3.0});

  EXPECT_EQ(seeds.nearest(17, linear_distance), (std::vector<double>{1.0, 2.0, 3.0}));
}

TEST(ScanSeeds, AFittedPointBeatsTheFallback) {
  scanseed::Store<int> seeds;
  seeds.set_fallback({1.0});
  seeds.store(100, {2.0});

  // Even far away: a converged neighbour anywhere on the profile has the
  // scanned parameter's effect in it, which the free fit does not.
  EXPECT_EQ(seeds.nearest(0, linear_distance), (std::vector<double>{2.0}));
}

TEST(ScanSeeds, PicksTheNearestFittedPoint) {
  scanseed::Store<int> seeds;
  seeds.store(0, {0.0});
  seeds.store(10, {10.0});
  seeds.store(30, {30.0});

  EXPECT_EQ(seeds.nearest(12, linear_distance), (std::vector<double>{10.0}));
  EXPECT_EQ(seeds.nearest(29, linear_distance), (std::vector<double>{30.0}));
  EXPECT_EQ(seeds.nearest(-5, linear_distance), (std::vector<double>{0.0}));
}

TEST(ScanSeeds, NearestIsMeasuredInTheWindowNotInLatticeUnits) {
  // A window far wider in y than in x: the candidate four lattice steps away in
  // x is the closer one in the units the likelihood actually varies over.
  const auto distance = grid_distance(0.01, 1.0);

  scanseed::Store<scan::Node> seeds;
  seeds.store(scan::Node{4, 0}, {1.0});  // 0.04 away
  seeds.store(scan::Node{0, 1}, {2.0});  // 1.0 away

  EXPECT_EQ(seeds.nearest(scan::Node{0, 0}, distance), (std::vector<double>{1.0}));
}

TEST(ScanSeeds, StoringTwiceKeepsTheLatestResult) {
  scanseed::Store<int> seeds;
  seeds.store(5, {1.0});
  seeds.store(5, {2.0});

  EXPECT_EQ(seeds.size(), 1u);
  EXPECT_EQ(seeds.nearest(5, linear_distance), (std::vector<double>{2.0}));
}

TEST(ScanSeeds, SurvivesConcurrentWorkers) {
  constexpr int n_threads = 8;
  constexpr int per_thread = 200;

  scanseed::Store<int> seeds;
  seeds.set_fallback({-1.0});

  // What the scan does: every worker keeps inserting its own results while
  // reading the ones the others have already put in.
  auto worker = [&](int id) {
    for (int i = 0; i < per_thread; ++i) {
      const int node = id * per_thread + i;
      const auto start = seeds.nearest(node, linear_distance);
      EXPECT_FALSE(start.empty());
      seeds.store(node, {static_cast<double>(node)});
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(n_threads);
  for (int id = 0; id < n_threads; ++id)
    threads.emplace_back(worker, id);

  for (auto& thread : threads)
    thread.join();

  EXPECT_EQ(seeds.size(), static_cast<std::size_t>(n_threads * per_thread));
  EXPECT_EQ(seeds.nearest(42, linear_distance), (std::vector<double>{42.0}));
}

/**
 * @file
 * @brief Tests for the seed-quality gate (SeedQuality.h).
 *
 * Which finished fits are allowed to seed another one. The gate replaced
 * Migrad's converged/failed flag, which rejected fits that had located their
 * minimum perfectly well by the only measure that matters here.
 */

TEST(SeedQuality, ConvergedIsAlwaysUsable) {
  EXPECT_TRUE(seedquality::usable(true, 1.0e-6));

  // Even with no EDM recorded: the flag alone is enough, which is what lets a
  // result written before EDM was stored still seed a neighbour.
  EXPECT_TRUE(seedquality::usable(true, std::numeric_limits<double>::infinity()));

  // And even with an EDM that would fail on its own -- the flag wins.
  EXPECT_TRUE(seedquality::usable(true, 5.0));
}

TEST(SeedQuality, UnconvergedButCloseIsUsable) {
  // The population the old converged() gate threw away: short of Migrad's
  // 1e-4 tolerance, but far inside the 1 sigma rise the scan resolves.
  EXPECT_TRUE(seedquality::usable(false, 1.0e-3));
  EXPECT_TRUE(seedquality::usable(false, 0.5));
  EXPECT_TRUE(seedquality::usable(false, 0.0));
}

TEST(SeedQuality, UnconvergedAndFarIsRejected) {
  EXPECT_FALSE(seedquality::usable(false, seedquality::kMaxEdm));
  EXPECT_FALSE(seedquality::usable(false, 1.5));
  EXPECT_FALSE(seedquality::usable(false, 2.0e3));
}

TEST(SeedQuality, NonFiniteOrNegativeEdmIsRejected) {
  // A fit that produced no usable EDM says nothing about where it stopped, and
  // a negative one is a number that should not exist. Neither seeds anything.
  EXPECT_FALSE(seedquality::usable(false, std::numeric_limits<double>::infinity()));
  EXPECT_FALSE(seedquality::usable(false, std::numeric_limits<double>::quiet_NaN()));
  EXPECT_FALSE(seedquality::usable(false, -1.0));
}

TEST(SeedQuality, StoreTakesTheVectorsTheGateAdmits) {
  // The gate and the store together: only admitted fits become start points,
  // and a rejected one leaves its neighbour on the fallback.
  scanseed::Store<int> seeds;
  seeds.set_fallback({0.0});

  const std::vector<std::pair<double, bool>> fits{{1.0e-6, true}, {1.0e-2, false}, {7.0, false}};
  for (std::size_t i = 0; i < fits.size(); ++i) {
    const auto [edm, converged] = fits[i];
    if (seedquality::usable(converged, edm))
      seeds.store(static_cast<int>(i), {static_cast<double>(i) + 1.0});
  }

  EXPECT_EQ(seeds.size(), std::size_t{2});
  EXPECT_EQ(seeds.nearest(2, linear_distance), (std::vector<double>{2.0}));  // nearest admitted, not the rejected one
}

/**
 * @file
 * @brief Tests for the start vector built from a stored fit's parameter block.
 *
 * The rule that a partial block is filled from the configured start values
 * rather than dropped. Dropping it is what silently disabled warm start under
 * --blind, which strips the signal parameters from every result written.
 */

namespace {

  const std::vector<std::string> kNames{"AstroNorm", "SpectralIndex", "ConvNorm", "BarrH", "DOMEff"};

  /// Configured start value of parameter i, standing in for the config.
  double config_start(std::size_t i) { return 100.0 + static_cast<double>(i); }

}  // namespace

TEST(StartVector, CompleteBlockIsTakenWhole) {
  const std::map<std::string, double> stored{
    {"AstroNorm", 1.0}, {"SpectralIndex", 2.0}, {"ConvNorm", 3.0}, {"BarrH", 4.0}, {"DOMEff", 5.0}};

  std::size_t filled = 0;
  const auto  values = seedquality::start_vector(kNames, stored, config_start, filled);

  EXPECT_EQ(filled, kNames.size());
  EXPECT_EQ(values, (std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}));
}

TEST(StartVector, BlindedBlockKeepsConfiguredValuesForTheMissingNames) {
  // What --blind writes: the signal parameters are gone, the rest are there.
  const std::map<std::string, double> stored{{"ConvNorm", 3.0}, {"BarrH", 4.0}, {"DOMEff", 5.0}};

  std::size_t filled = 0;
  const auto  values = seedquality::start_vector(kNames, stored, config_start, filled);

  EXPECT_EQ(filled, std::size_t{3});
  ASSERT_EQ(values.size(), kNames.size());

  // Order is the minimizer's, not the map's: the missing names hold their slot.
  EXPECT_DOUBLE_EQ(values[0], config_start(0));
  EXPECT_DOUBLE_EQ(values[1], config_start(1));
  EXPECT_DOUBLE_EQ(values[2], 3.0);
  EXPECT_DOUBLE_EQ(values[3], 4.0);
  EXPECT_DOUBLE_EQ(values[4], 5.0);
}

TEST(StartVector, ABlockSharingNoNameIsDropped) {
  // A result from a different config, which must not be mixed into this one.
  const std::map<std::string, double> stored{{"Theta13", 0.1}, {"DeltaM2", 2.5e-3}};

  std::size_t filled = 1;
  EXPECT_TRUE(seedquality::start_vector(kNames, stored, config_start, filled).empty());
  EXPECT_EQ(filled, std::size_t{0});

  std::size_t none = 1;
  EXPECT_TRUE(seedquality::start_vector(kNames, {}, config_start, none).empty());
  EXPECT_EQ(none, std::size_t{0});
}

TEST(StartVector, ExtraNamesInTheFileAreIgnored) {
  // A newer result carrying a parameter this config does not have.
  const std::map<std::string, double> stored{
    {"AstroNorm", 1.0}, {"SpectralIndex", 2.0}, {"ConvNorm", 3.0},
    {"BarrH", 4.0},     {"DOMEff", 5.0},        {"GalacticNorm2", 9.0}};

  std::size_t filled = 0;
  const auto  values = seedquality::start_vector(kNames, stored, config_start, filled);

  EXPECT_EQ(filled, kNames.size());
  EXPECT_EQ(values.size(), kNames.size());
}
