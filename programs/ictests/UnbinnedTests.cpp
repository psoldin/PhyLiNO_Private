#include "IceCube/ICSample.h"
#include "IceCube/SampleConfig.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <gtest/gtest.h>

#include <sstream>
#include <string>

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
