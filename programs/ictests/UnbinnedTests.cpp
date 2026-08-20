#include "IceCube/ICDataBase.h"
#include "IceCube/ICSample.h"
#include "IceCube/KdeIndex.h"
#include "IceCube/SampleConfig.h"
#include "IceCube/ICParameter.h"
#include "ParameterWrapper.h"
#include "SampleLikelihood.h"
#include "UnbinnedLikelihood.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <sstream>
#include <string>
#include <utility>
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
  EXPECT_EQ(u.energy_sigma_branch, "ELEFANTS_tg_sigma_log10");
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

namespace {

  void write_double_parquet(const std::string&                                              path,
                            const std::vector<std::pair<std::string, std::vector<double>>>& columns) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    for (const auto& [name, values] : columns) {
      arrow::DoubleBuilder          builder;
      std::shared_ptr<arrow::Array> array;
      EXPECT_TRUE(builder.AppendValues(values).ok());
      EXPECT_TRUE(builder.Finish(&array).ok());
      fields.push_back(arrow::field(name, arrow::float64()));
      arrays.push_back(array);
    }
    const auto table = arrow::Table::Make(arrow::schema(fields), arrays);
    auto       sink  = arrow::io::FileOutputStream::Open(path).ValueOrDie();
    ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink,
                                           static_cast<int64_t>(table->num_rows()))
                    .ok());
    ASSERT_TRUE(sink->Close().ok());
  }

  io::ic::Binning tracks_binning_2d() {
    return io::ic::Binning({io::ic::Axis{io::ic::Axis::Kind::Log10Energy, 2.0, 7.0, 50},
                            io::ic::Axis{io::ic::Axis::Kind::CosZenith, -1.0, 0.0872, 33}});
  }

}  // namespace

// The KDE coordinates and bandwidths are derived at load, and the three kinds of
// event that have no usable kernel -- NaN reco (an unmatched ELEFANTS merge),
// a degenerate zero bandwidth, and a coordinate outside the KDE domain -- must
// leave the sample entirely rather than reach the density.
TEST(UnbinnedLoadTest, DerivesCoordinatesAndDropsUnusableEvents) {
  const std::string path = "ictests_unbinned_mc.parquet";
  const double      nan  = std::numeric_limits<double>::quiet_NaN();

  // sigma = ln10 * mu * 0.1 makes every usable event's h_e exactly 0.1 dex.
  const double ln10 = std::log(10.0);
  write_double_parquet(path, {{"energy_truncated", {1.0e4, 1.0e5, 1.0e4, 1.0e4, 1.0e4}},
                              {"zenith_MPEFit", {2.0, 2.5, 2.0, 2.0, 0.2}},
                              {"MCPrimaryEnergy", {1.0e4, 1.0e5, 1.0e4, 1.0e4, 1.0e4}},
                              {"powerlaw", {1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8}},
                              {"ELEFANTS_tg_sigma",
                               {1.0e4 * 0.1 * ln10, 1.0e5 * 0.1 * ln10, nan, 0.0, 1.0e4 * 0.1 * ln10}},
                              {"L5_sigma_paraboloid", {0.02, 0.03, 0.02, 0.02, 0.02}}});

  io::ic::SampleConfig cfg{
      .name = "tracks", .binning = tracks_binning_2d(), .mc_binning = tracks_binning_2d()};
  cfg.parquet            = path;
  cfg.components         = {"astro"};
  cfg.unbinned.enabled                = true;
  cfg.unbinned.energy_sigma_branch    = "ELEFANTS_tg_sigma";
  cfg.unbinned.energy_sigma_transform = io::ic::SigmaTransform::LinearToDex;
  cfg.unbinned.zenith_sigma_transform = io::ic::SigmaTransform::None;
  cfg.unbinned.zenith_lo              = 1.4836;
  cfg.unbinned.zenith_hi = 3.14159265358979;

  const io::ic::ICDataBase db({cfg});
  const io::ic::ICSample&  s = db.sample(0);

  ASSERT_EQ(s.size(), 2u);
  // h_e inverts the delta method: sigma / (mu * ln10) == sigma_log10 == 0.1.
  EXPECT_NEAR(s.kde_h_e[0], 0.1, 1e-12);
  EXPECT_NEAR(s.kde_h_e[1], 0.1, 1e-12);
  EXPECT_NEAR(s.kde_log_e[0], 4.0, 1e-12);
  EXPECT_NEAR(s.kde_log_e[1], 5.0, 1e-12);
  EXPECT_NEAR(s.kde_zenith[0], 2.0, 1e-12);
  EXPECT_NEAR(s.kde_h_z[1], 0.03, 1e-12);
  EXPECT_FALSE(s.kde_index.empty());

  std::remove(path.c_str());
}

// A sample without an "Unbinned" block must not pay for any of it: no columns,
// no index, and no requirement that the sigma branches even exist in the file.
TEST(UnbinnedLoadTest, BinnedSampleKeepsNoKdeColumns) {
  const std::string path = "ictests_binned_mc.parquet";
  write_double_parquet(path, {{"energy_truncated", {1.0e4, 1.0e5}},
                              {"zenith_MPEFit", {2.0, 2.5}},
                              {"MCPrimaryEnergy", {1.0e4, 1.0e5}},
                              {"powerlaw", {1.0e-8, 1.0e-8}}});

  io::ic::SampleConfig cfg{
      .name = "tracks", .binning = tracks_binning_2d(), .mc_binning = tracks_binning_2d()};
  cfg.parquet    = path;
  cfg.components = {"astro"};

  const io::ic::ICDataBase db({cfg});
  const io::ic::ICSample&  s = db.sample(0);

  ASSERT_EQ(s.size(), 2u);
  EXPECT_TRUE(s.kde_log_e.empty());
  EXPECT_TRUE(s.kde_h_e.empty());
  EXPECT_TRUE(s.kde_index.empty());

  std::remove(path.c_str());
}

// Reflection is what keeps the extended term honest: nu = sum_i w_i is only the
// integral of the density over the domain if no kernel mass escapes it. This is
// the worst case, a kernel sitting right on the lower edge.
TEST(UnbinnedKernelTest, ReflectedKernelIntegratesToOne) {
  constexpr double a = 1.4836, b = 3.14159265358979;
  constexpr double c = 1.4936, h = 0.05;

  constexpr int n     = 200000;
  const double  step  = (b - a) / n;
  double        total = 0.0;
  for (int i = 0; i < n; ++i)
    total += ana::ic::reflected_gauss(a + (i + 0.5) * step, c, h, a, b) * step;

  EXPECT_NEAR(total, 1.0, 1e-6);
}

// The truncation is an optimisation, not an approximation the result depends on:
// at a wide enough radius the indexed walk must reproduce the full double sum.
TEST(UnbinnedDensityTest, TruncatedMatchesBruteForce) {
  const SyntheticKde s = make_synthetic(3000, 4242);
  constexpr double   lo_e = 2.0, hi_e = 7.0, lo_z = 1.5, hi_z = 3.1;
  constexpr double   kNSigma = 8.0;

  const io::ic::KdeIndex index =
      io::ic::build_kde_index(s.x_e, s.x_z, s.h_e, s.h_z, {lo_e, lo_z}, {hi_e, hi_z}, kNSigma);

  auto brute = [&s](const double qe, const double qz) {
    double acc = 0.0;
    for (std::size_t i = 0; i < s.x_e.size(); ++i)
      acc += s.w[i] * ana::ic::reflected_gauss(qe, s.x_e[i], s.h_e[i], lo_e, hi_e) *
             ana::ic::reflected_gauss(qz, s.x_z[i], s.h_z[i], lo_z, hi_z);
    return acc;
  };

  const ana::ic::KdeDensity density{.x_e   = s.x_e,
                                    .x_z   = s.x_z,
                                    .h_e   = s.h_e,
                                    .h_z   = s.h_z,
                                    .index = index,
                                    .lo    = {lo_e, lo_z},
                                    .hi    = {hi_e, hi_z}};

  for (const auto& [qe, qz] :
       std::vector<std::pair<double, double>>{{4.0, 2.0}, {2.05, 1.52}, {6.9, 3.05}, {5.5, 2.7}}) {
    const double exact = brute(qe, qz);
    EXPECT_NEAR(density.evaluate(qe, qz, s.w), exact, 1e-8 * exact) << "at (" << qe << ", " << qz << ")";
  }
}

// Using the MC events as quadrature nodes for a density built from those same
// events inflates lambda at every node by the node's own kernel; leave-one-out
// removes exactly that term and nothing else.
TEST(UnbinnedDensityTest, LeaveOneOutRemovesSelfTerm) {
  const SyntheticKde s = make_synthetic(400, 8);
  constexpr double   lo_e = 2.0, hi_e = 7.0, lo_z = 1.5, hi_z = 3.1;

  const io::ic::KdeIndex index =
      io::ic::build_kde_index(s.x_e, s.x_z, s.h_e, s.h_z, {lo_e, lo_z}, {hi_e, hi_z}, 8.0);
  const ana::ic::KdeDensity density{.x_e   = s.x_e,
                                    .x_z   = s.x_z,
                                    .h_e   = s.h_e,
                                    .h_z   = s.h_z,
                                    .index = index,
                                    .lo    = {lo_e, lo_z},
                                    .hi    = {hi_e, hi_z}};

  constexpr std::size_t j    = 17;
  const double          full = density.evaluate(s.x_e[j], s.x_z[j], s.w);
  const double          self = s.w[j] * ana::ic::reflected_gauss(s.x_e[j], s.x_e[j], s.h_e[j], lo_e, hi_e) *
                      ana::ic::reflected_gauss(s.x_z[j], s.x_z[j], s.h_z[j], lo_z, hi_z);

  EXPECT_NEAR(density.evaluate_loo(j, s.w), full - self, 1e-9 * full);
}

// The two invariants that say the wiring is right: the frozen quadrature weight
// is the same nu as the binned Asimov total (not N_MC), and -2lnL sits at a
// minimum where the Asimov weights were frozen.
TEST(UnbinnedSampleLikelihoodTest, AsimovTotalMatchesBinnedAndMinimisesAtTruth) {
  using ana::ParameterWrapper;

  const std::string path = "ictests_unbinned_fit.parquet";
  const double      ln10 = std::log(10.0);

  std::vector<double> reco_e, reco_z, e_true, powerlaw, sigma_e, sigma_z;
  std::mt19937                           rng(2026);
  std::uniform_real_distribution<double> log_e(2.5, 6.5);
  std::uniform_real_distribution<double> zen(1.6, 3.0);
  for (int i = 0; i < 2000; ++i) {
    const double le = log_e(rng);
    const double e  = std::pow(10.0, le);
    reco_e.push_back(e);
    reco_z.push_back(zen(rng));
    e_true.push_back(e);
    powerlaw.push_back(1.0e-6 * std::pow(e / 1.0e5, -2.0));
    sigma_e.push_back(e * 0.1 * ln10);  // 0.1 dex
    sigma_z.push_back(0.02);
  }
  write_double_parquet(path, {{"energy_truncated", reco_e},
                              {"zenith_MPEFit", reco_z},
                              {"MCPrimaryEnergy", e_true},
                              {"powerlaw", powerlaw},
                              {"ELEFANTS_tg_sigma", sigma_e},
                              {"L5_sigma_paraboloid", sigma_z}});

  io::ic::SampleConfig cfg{
      .name = "tracks", .binning = tracks_binning_2d(), .mc_binning = tracks_binning_2d()};
  cfg.parquet            = path;
  cfg.components         = {"astro"};
  cfg.livetime           = 1.0e7;
  cfg.unbinned.enabled                = true;
  cfg.unbinned.energy_sigma_branch    = "ELEFANTS_tg_sigma";
  cfg.unbinned.energy_sigma_transform = io::ic::SigmaTransform::LinearToDex;
  cfg.unbinned.zenith_sigma_transform = io::ic::SigmaTransform::None;
  cfg.unbinned.zenith_lo              = 1.5;
  cfg.unbinned.zenith_hi = 3.05;

  const io::ic::ICDataBase db({cfg});

  const ana::ic::GlobalFluxSettings settings{.e_ref_gev                = 1.0e5,
                                             .astro_reference_index    = 2.0,
                                             .conv_delta_gamma_e_ref   = 1.0e3,
                                             .prompt_delta_gamma_e_ref = 3.8e3,
                                             .astro_per_type_norm      = false,
                                             .veto_anchor_energy       = 100.0,
                                             .veto_rescale_energy      = 100.0};
  ana::ic::SampleLikelihood         sample(db.sample(0), cfg, settings, /*gpu=*/nullptr, /*use_say=*/false);

  std::vector<double> values(params::ic::number_of_parameters(), 0.0);
  values[params::ic::AstroNorm]     = 1.0;
  values[params::ic::SpectralIndex] = 2.5;
  ParameterWrapper parameter(params::ic::number_of_parameters());
  parameter.reset_parameter(values.data());
  sample.generate_asimov(parameter);

  const double at_truth = sample.partial_llh(parameter);

  double binned_total = 0.0;
  for (const double v : sample.data()) binned_total += v;
  EXPECT_NEAR(sample.unbinned_asimov_total(), binned_total, 1e-6 * binned_total);

  for (const double gamma : {2.4, 2.6}) {
    values[params::ic::SpectralIndex] = gamma;
    parameter.reset_parameter(values.data());
    EXPECT_GT(sample.partial_llh(parameter), at_truth) << "at gamma = " << gamma;
  }

  std::remove(path.c_str());
}

// The per-bin inputs have no per-event representation, so a config that pairs
// them with the unbinned path must fail loudly rather than silently fit a
// different model. parse_samples catches this too, but a SampleConfig built in
// code (as here, and as in every test) bypasses it.
TEST(UnbinnedSampleLikelihoodTest, RejectsSayLikelihood) {
  const std::string path = "ictests_unbinned_say.parquet";
  const double      ln10 = std::log(10.0);
  write_double_parquet(path, {{"energy_truncated", {1.0e4, 1.0e5}},
                              {"zenith_MPEFit", {2.0, 2.5}},
                              {"MCPrimaryEnergy", {1.0e4, 1.0e5}},
                              {"powerlaw", {1.0e-8, 1.0e-8}},
                              {"ELEFANTS_tg_sigma", {1.0e4 * 0.1 * ln10, 1.0e5 * 0.1 * ln10}},
                              {"L5_sigma_paraboloid", {0.02, 0.03}}});

  io::ic::SampleConfig cfg{
      .name = "tracks", .binning = tracks_binning_2d(), .mc_binning = tracks_binning_2d()};
  cfg.parquet            = path;
  cfg.components         = {"astro"};
  cfg.unbinned.enabled                = true;
  cfg.unbinned.energy_sigma_branch    = "ELEFANTS_tg_sigma";
  cfg.unbinned.energy_sigma_transform = io::ic::SigmaTransform::LinearToDex;
  cfg.unbinned.zenith_sigma_transform = io::ic::SigmaTransform::None;
  cfg.unbinned.zenith_lo              = 1.5;
  cfg.unbinned.zenith_hi = 3.05;

  const io::ic::ICDataBase          db({cfg});
  const ana::ic::GlobalFluxSettings settings{.e_ref_gev                = 1.0e5,
                                             .astro_reference_index    = 2.0,
                                             .conv_delta_gamma_e_ref   = 1.0e3,
                                             .prompt_delta_gamma_e_ref = 3.8e3,
                                             .astro_per_type_norm      = false,
                                             .veto_anchor_energy       = 100.0,
                                             .veto_rescale_energy      = 100.0};

  EXPECT_THROW(
      ana::ic::SampleLikelihood(db.sample(0), cfg, settings, /*gpu=*/nullptr, /*use_say=*/true),
      std::runtime_error);

  std::remove(path.c_str());
}

// The uncertainty columns are not self-describing -- ELEFANTS emits log(sigma)
// and the paraboloid sigma is in degrees -- so the transform is config, and a
// wrong one must change the bandwidth rather than pass unnoticed.
TEST(UnbinnedLoadTest, AppliesConfiguredSigmaTransforms) {
  const std::string path = "ictests_unbinned_transform.parquet";
  // log(0.3) and 0.61 degrees: the medians of the real ELEFANTS/paraboloid columns.
  const double log_sigma = std::log(0.3);
  write_double_parquet(path, {{"energy_truncated", {1.0e4, 1.0e5}},
                              {"zenith_MPEFit", {2.0, 2.5}},
                              {"MCPrimaryEnergy", {1.0e4, 1.0e5}},
                              {"powerlaw", {1.0e-8, 1.0e-8}},
                              {"sigma_e", {log_sigma, log_sigma}},
                              {"sigma_z", {0.61, 0.61}}});

  io::ic::SampleConfig cfg{
      .name = "tracks", .binning = tracks_binning_2d(), .mc_binning = tracks_binning_2d()};
  cfg.parquet                         = path;
  cfg.components                      = {"astro"};
  cfg.unbinned.enabled                = true;
  cfg.unbinned.energy_sigma_branch    = "sigma_e";
  cfg.unbinned.zenith_sigma_branch    = "sigma_z";
  cfg.unbinned.energy_sigma_transform = io::ic::SigmaTransform::Exp;
  cfg.unbinned.zenith_sigma_transform = io::ic::SigmaTransform::DegToRad;
  cfg.unbinned.zenith_lo              = 1.5;
  cfg.unbinned.zenith_hi              = 3.05;

  // Name the database: binding a reference through sample() into a temporary
  // would dangle the moment the full expression ends.
  const io::ic::ICDataBase db({cfg});
  const io::ic::ICSample&  s = db.sample(0);

  ASSERT_EQ(s.size(), 2u);
  EXPECT_NEAR(s.kde_h_e[0], 0.3, 1e-12);
  EXPECT_NEAR(s.kde_h_z[0], 0.61 * std::numbers::pi / 180.0, 1e-15);

  std::remove(path.c_str());
}
