#include "IceCube/ICSample.h"
#include "IceCube/KdeIndex.h"
#include "IceCube/SampleConfig.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

  boost::property_tree::ptree ic_tree(const std::string& samples_json) {
    const std::string json = R"json({
      "Binnings": { "tracks_2d": {
        "axes": "Log10Energy, CosZenith",
        "Log10Energy": "(2.0, 7.0, 50)",
        "CosZenith": "(-1.0, 0.0872, 33)" },
      "tracks_3d": {
        "axes": "Log10Energy, CosZenith, Ra",
        "Log10Energy": "(2.0, 7.0, 50)",
        "CosZenith": "(-1.0, 0.0872, 33)",
        "Ra": "(0.0, 6.28319, 4)" } },
      "Samples": )json" + samples_json + "}";
    std::istringstream          in(json);
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(in, tree);
    return tree;
  }

}  // namespace

TEST(UnbinnedConfigTest, ParsesBlockWithDefaults) {
  const auto tree = ic_tree(R"({ "tracks": {
      "binning": "tracks_2d",
      "parquet": "mc.parquet",
      "components": "astro, conventional, prompt",
      "Unbinned": { "enabled": true }
  } })");

  const auto samples = io::ic::parse_samples(tree);
  ASSERT_EQ(samples.size(), 1u);
  const io::ic::UnbinnedConfig& u = samples[0].unbinned;
  EXPECT_TRUE(u.enabled);
  EXPECT_EQ(u.energy_sigma_branch, "ELEFANTS_tg_sigma");
  EXPECT_EQ(u.zenith_sigma_branch, "L5_sigma_paraboloid");
  EXPECT_DOUBLE_EQ(u.log_e_lo, 2.0);
  EXPECT_DOUBLE_EQ(u.log_e_hi, 7.0);
  EXPECT_DOUBLE_EQ(u.truncation, 5.0);
  EXPECT_EQ(u.thinning, 1);
}

TEST(UnbinnedConfigTest, ParsesExplicitValues) {
  const auto tree = ic_tree(R"({ "tracks": {
      "binning": "tracks_2d",
      "parquet": "mc.parquet",
      "components": "astro, conventional, prompt",
      "Unbinned": {
        "enabled": true,
        "EnergySigmaBranch": "sigma_e",
        "ZenithSigmaBranch": "sigma_z",
        "Log10EnergyLo": 3.0, "Log10EnergyHi": 6.5,
        "ZenithLo": 1.4836, "ZenithHi": 3.14159265358979,
        "Truncation": 4.0,
        "Thinning": 7 }
  } })");

  const auto                    samples = io::ic::parse_samples(tree);
  const io::ic::UnbinnedConfig& u       = samples[0].unbinned;
  EXPECT_TRUE(u.enabled);
  EXPECT_EQ(u.energy_sigma_branch, "sigma_e");
  EXPECT_EQ(u.zenith_sigma_branch, "sigma_z");
  EXPECT_DOUBLE_EQ(u.log_e_lo, 3.0);
  EXPECT_DOUBLE_EQ(u.log_e_hi, 6.5);
  EXPECT_DOUBLE_EQ(u.zenith_lo, 1.4836);
  EXPECT_DOUBLE_EQ(u.zenith_hi, 3.14159265358979);
  EXPECT_DOUBLE_EQ(u.truncation, 4.0);
  EXPECT_EQ(u.thinning, 7);
}

TEST(UnbinnedConfigTest, RejectsBinnedOnlyComponents) {
  const auto with_template = ic_tree(R"({ "tracks": {
      "binning": "tracks_2d",
      "parquet": "mc.parquet",
      "components": "astro, conventional, prompt, muontemplate",
      "Template": { "File": "muon.txt", "Norm": "MuonNorm" },
      "Unbinned": { "enabled": true }
  } })");
  EXPECT_THROW(io::ic::parse_samples(with_template), std::runtime_error);

  const auto with_gradients = ic_tree(R"({ "tracks": {
      "binning": "tracks_2d",
      "parquet": "mc.parquet",
      "components": "astro, conventional, prompt",
      "Gradients": { "File": "gradients.txt" },
      "Unbinned": { "enabled": true }
  } })");
  EXPECT_THROW(io::ic::parse_samples(with_gradients), std::runtime_error);

  // tracks_3d carries a trailing Ra axis, so this Galactic block is otherwise
  // legal (parse_samples only rejects Galactic on a binning with no Ra axis) --
  // the throw here has to come from the Unbinned check, not that one.
  const auto with_galactic = ic_tree(R"({ "tracks": {
      "binning": "tracks_3d",
      "parquet": "mc.parquet",
      "components": "astro, conventional, prompt",
      "Galactic": { "gp": { "File": "galactic.txt", "Norm": "GalacticNorm0" } },
      "Unbinned": { "enabled": true }
  } })");
  EXPECT_THROW(io::ic::parse_samples(with_galactic), std::runtime_error);
}

TEST(UnbinnedConfigTest, RejectsNonPositiveTruncation) {
  const auto tree = ic_tree(R"({ "tracks": {
      "binning": "tracks_2d",
      "parquet": "mc.parquet",
      "components": "astro, conventional, prompt",
      "Unbinned": { "enabled": true, "Truncation": 0.0 }
  } })");
  EXPECT_THROW(io::ic::parse_samples(tree), std::runtime_error);
}

TEST(UnbinnedConfigTest, RejectsThinningBelowOne) {
  const auto tree = ic_tree(R"({ "tracks": {
      "binning": "tracks_2d",
      "parquet": "mc.parquet",
      "components": "astro, conventional, prompt",
      "Unbinned": { "enabled": true, "Thinning": 0 }
  } })");
  EXPECT_THROW(io::ic::parse_samples(tree), std::runtime_error);
}

TEST(UnbinnedConfigTest, RejectsEmptyDomain) {
  const auto tree = ic_tree(R"({ "tracks": {
      "binning": "tracks_2d",
      "parquet": "mc.parquet",
      "components": "astro, conventional, prompt",
      "Unbinned": { "enabled": true, "Log10EnergyLo": 6.0, "Log10EnergyHi": 5.0 }
  } })");
  EXPECT_THROW(io::ic::parse_samples(tree), std::runtime_error);
}

TEST(UnbinnedConfigTest, RejectsEmptyBranchName) {
  const auto tree = ic_tree(R"({ "tracks": {
      "binning": "tracks_2d",
      "parquet": "mc.parquet",
      "components": "astro, conventional, prompt",
      "Unbinned": { "enabled": true, "EnergySigmaBranch": "" }
  } })");
  EXPECT_THROW(io::ic::parse_samples(tree), std::runtime_error);
}

// The sample's binning axis is CosZenith (Binning.cpp applies std::cos), but
// ZenithLo/ZenithHi are radians -- pasting the CosZenith axis bounds in here,
// as this test does, is exactly the unit bug the range check exists to catch.
TEST(UnbinnedConfigTest, RejectsZenithOutOfRadianRange) {
  const auto tree = ic_tree(R"({ "tracks": {
      "binning": "tracks_2d",
      "parquet": "mc.parquet",
      "components": "astro, conventional, prompt",
      "Unbinned": { "enabled": true, "ZenithLo": -1.0, "ZenithHi": 0.0872 }
  } })");
  EXPECT_THROW(io::ic::parse_samples(tree), std::runtime_error);
}

// sort_into_bins() permutes every per-event column into bin-major order; a
// column it does not reorder keeps the pre-sort order and silently misaligns
// with the weights, which is the one way these columns can go wrong unnoticed.
TEST(UnbinnedSampleTest, SortIntoBinsReordersKdeColumns) {
  io::ic::ICSample s;
  // Three events; the middle one is out of range and must be dropped, and the
  // remaining two must be swapped into bin order (bin 1 then bin 0 on input).
  s.e_true     = {10.0, 20.0, 30.0};
  s.bin_idx    = {1, -1, 0};
  s.kde_log_e  = {4.0, 5.0, 6.0};
  s.kde_zenith = {2.0, 2.5, 3.0};
  s.kde_h_e    = {0.1, 0.2, 0.3};
  s.kde_h_z    = {0.01, 0.02, 0.03};

  s.sort_into_bins(2);

  ASSERT_EQ(s.size(), 2u);
  EXPECT_DOUBLE_EQ(s.kde_log_e[0], 6.0);
  EXPECT_DOUBLE_EQ(s.kde_log_e[1], 4.0);
  EXPECT_DOUBLE_EQ(s.kde_zenith[0], 3.0);
  EXPECT_DOUBLE_EQ(s.kde_h_e[1], 0.1);
  EXPECT_DOUBLE_EQ(s.kde_h_z[0], 0.03);
}

namespace {

  // A synthetic sample with a deliberately wide bandwidth spread: one event in
  // 20 gets a 30x kernel, which is the case a single-grid index would either
  // miss or pay for on every query.
  struct SyntheticKde {
    std::vector<double> x_e, x_z, h_e, h_z, w;
  };

  SyntheticKde make_synthetic(const std::size_t n, const unsigned seed) {
    std::mt19937                           rng(seed);
    std::uniform_real_distribution<double> energy(2.0, 7.0);
    std::uniform_real_distribution<double> zenith(1.5, 3.1);
    std::uniform_real_distribution<double> weight(0.1, 2.0);
    SyntheticKde                           s;
    for (std::size_t i = 0; i < n; ++i) {
      s.x_e.push_back(energy(rng));
      s.x_z.push_back(zenith(rng));
      const double scale = (i % 20 == 0) ? 30.0 : 1.0;
      s.h_e.push_back(0.05 * scale);
      s.h_z.push_back(0.01 * scale);
      s.w.push_back(weight(rng));
    }
    return s;
  }

}  // namespace

// The property the whole banded construction exists for: a wide-kernel event
// must still be found by every query it can reach, without the search radius of
// every other query growing to match it.
TEST(UnbinnedIndexTest, FindsEveryEventWithinTruncation) {
  const SyntheticKde s       = make_synthetic(2000, 12345);
  constexpr double   kNSigma = 5.0;

  const io::ic::KdeIndex index =
      io::ic::build_kde_index(s.x_e, s.x_z, s.h_e, s.h_z, {2.0, 1.5}, {7.0, 3.1}, kNSigma);

  std::mt19937                           rng(999);
  std::uniform_real_distribution<double> energy(2.0, 7.0);
  std::uniform_real_distribution<double> zenith(1.5, 3.1);

  for (int q = 0; q < 200; ++q) {
    const double qe = energy(rng);
    const double qz = zenith(rng);

    std::vector<int> expected;
    for (std::size_t i = 0; i < s.x_e.size(); ++i)
      if (std::abs(qe - s.x_e[i]) <= kNSigma * s.h_e[i] && std::abs(qz - s.x_z[i]) <= kNSigma * s.h_z[i])
        expected.push_back(static_cast<int>(i));

    std::vector<int> found;
    io::ic::for_each_neighbour(index, qe, qz, [&found](const int i) { found.push_back(i); });

    // The index may return extras -- a box is a superset of the ellipse -- but
    // it may never miss one.
    std::ranges::sort(found);
    for (const int i : expected)
      EXPECT_TRUE(std::ranges::binary_search(found, i)) << "missed event " << i << " for query " << q;
  }
}

// Each event lives in exactly one (band, cell), so a walk that returned it twice
// would double its weight in the density.
TEST(UnbinnedIndexTest, VisitsEachEventAtMostOnce) {
  const SyntheticKde     s = make_synthetic(500, 7);
  const io::ic::KdeIndex index =
      io::ic::build_kde_index(s.x_e, s.x_z, s.h_e, s.h_z, {2.0, 1.5}, {7.0, 3.1}, 5.0);

  std::vector<int> found;
  io::ic::for_each_neighbour(index, 4.5, 2.5, [&found](const int i) { found.push_back(i); });

  std::vector<int> unique = found;
  std::ranges::sort(unique);
  EXPECT_EQ(std::ranges::unique(unique).begin() - unique.begin(), static_cast<long>(found.size()));
}

// An event with no kernel cannot be indexed, and silently skipping it would drop
// its weight from the density while nu still counted it.
TEST(UnbinnedIndexTest, RejectsUnusableBandwidths) {
  const std::vector<double> x_e{3.0, 4.0};
  const std::vector<double> x_z{2.0, 2.5};
  const std::vector<double> good{0.1, 0.1};

  for (const std::vector<double> bad :
       {std::vector<double>{0.1, 0.0}, std::vector<double>{0.1, -1.0},
        std::vector<double>{0.1, std::numeric_limits<double>::quiet_NaN()}}) {
    EXPECT_THROW(io::ic::build_kde_index(x_e, x_z, bad, good, {2.0, 1.5}, {7.0, 3.1}, 5.0),
                 std::runtime_error);
    EXPECT_THROW(io::ic::build_kde_index(x_e, x_z, good, bad, {2.0, 1.5}, {7.0, 3.1}, 5.0),
                 std::runtime_error);
  }
}
