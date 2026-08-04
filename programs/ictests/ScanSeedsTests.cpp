#include "AdaptiveGrid.h"
#include "ScanSeeds.h"

#include <gtest/gtest.h>

// STL includes
#include <cmath>
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
