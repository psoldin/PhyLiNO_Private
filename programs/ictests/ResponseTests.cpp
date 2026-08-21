#include "../../libraries/io/IceCube/ResponseMatrix.h"
#include "../../libraries/io/IceCube/SampleConfig.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <sstream>
#include <vector>

namespace {

  constexpr double kInvSqrt2 = 0.70710678118654752440;

  double normal_cdf(const double x, const double c, const double h) {
    return 0.5 * std::erfc(-(x - c) * kInvSqrt2 / h);
  }

  /** 50 log10-energy bins over [2, 7] and 4 cos-zenith bins over [-1, 0]. */
  io::ic::Binning test_binning() {
    return io::ic::Binning({io::ic::Axis{io::ic::Axis::Kind::Log10Energy, 2.0, 7.0, 50},
                            io::ic::Axis{io::ic::Axis::Kind::CosZenith, -1.0, 0.0, 4}});
  }

  /** Per-event sums of the matrix's fractions, in event order. */
  std::vector<double> row_sums(const io::ic::ResponseMatrix& m, const std::size_t n_events) {
    std::vector<double> sums(n_events, 0.0);
    for (std::size_t k = 0; k < m.nnz(); ++k) sums[static_cast<std::size_t>(m.events[k])] += m.fractions[k];
    return sums;
  }

  boost::property_tree::ptree ic_tree(const std::string& samples) {
    std::ostringstream json;
    json << R"JSON({"IceCube": {"Binnings": {"tracks_2d": {"axes": "Log10Energy, CosZenith",)JSON"
         << R"JSON("Log10Energy": "(2.0, 7.0, 50)", "CosZenith": "(-1.0, 0.0, 4)"}},)JSON"
         << R"JSON("Samples": )JSON" << samples << "}}";
    std::istringstream          in(json.str());
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(in, tree);
    return tree.get_child("IceCube");
  }

}  // namespace

// The one invariant everything else rests on: each event's response is
// renormalised over the bins that survive, so folding moves weight between bins
// and never creates or destroys it. Without this a folded fit would be compared
// against an unfolded one at a different total and the comparison would be
// meaningless.
TEST(ResponseMatrixTest, EveryEventFractionSumsToOne) {
  const io::ic::Binning     binning = test_binning();
  const std::vector<int>    bins{0, 137, 42, 199};
  const std::vector<double> truth_e{2.4, 3.9, 5.1, 6.8};
  const std::vector<double> truth_z{2.0, 2.6, 1.9, 3.0};
  const std::vector<double> sigma_e{0.30, 0.12, 0.45, 0.30};
  const std::vector<double> sigma_z{0.02, 0.05, 0.01, 0.10};

  std::size_t                  unusable = 0;
  const io::ic::ResponseMatrix m = io::ic::build_response_matrix(
      binning, bins, truth_e, truth_z, sigma_e, sigma_z, 5.0, 1.0e-4, unusable);

  EXPECT_EQ(unusable, 0u);
  for (const double s : row_sums(m, bins.size())) EXPECT_NEAR(s, 1.0, 1e-6);
}

// A response far narrower than a bin must reproduce what the plain histogram
// already does: all of the weight in the bin the event was assigned to. This is
// the limit in which forward folding is a no-op, and it is the check that the
// bin indexing in the fold agrees with Binning::bin_index.
TEST(ResponseMatrixTest, NarrowResponseReproducesTheUnfoldedHistogram) {
  const io::ic::Binning binning = test_binning();

  // Bin centres: energy bin 20 spans [4.0, 4.1), zenith bin 1 spans
  // cos in [-0.75, -0.5), i.e. the middle of that is cos = -0.625.
  const double        centre_e = 4.05;
  const double        centre_z = std::acos(-0.625);
  const std::array<double, 2> reco{std::pow(10.0, centre_e), centre_z};
  const int           expected = binning.bin_index(reco);
  ASSERT_GE(expected, 0);

  const std::vector<int>    bins{expected};
  const std::vector<double> truth_e{centre_e}, truth_z{centre_z};
  const std::vector<double> sigma_e{1.0e-4}, sigma_z{1.0e-5};

  std::size_t                  unusable = 0;
  const io::ic::ResponseMatrix m = io::ic::build_response_matrix(
      binning, bins, truth_e, truth_z, sigma_e, sigma_z, 5.0, 1.0e-4, unusable);

  ASSERT_EQ(m.nnz(), 1u);
  EXPECT_EQ(m.events[0], 0);
  EXPECT_NEAR(m.fractions[0], 1.0f, 1e-6);
  EXPECT_GE(m.bin_offsets[static_cast<std::size_t>(expected) + 1],
            m.bin_offsets[static_cast<std::size_t>(expected)] + 1);
}

// The energy fractions must be the Gaussian's actual bin masses, not something
// proportional to them.
TEST(ResponseMatrixTest, EnergyFractionsMatchTheGaussianBinMasses) {
  const io::ic::Binning binning = test_binning();

  // One zenith bin takes everything, so the product factorises and the energy
  // fractions are directly comparable.
  const double              centre_z = std::acos(-0.625);
  const std::vector<int>    bins{0};
  const std::vector<double> truth_e{4.0}, truth_z{centre_z};
  const std::vector<double> sigma_e{0.30}, sigma_z{1.0e-5};

  std::size_t                  unusable = 0;
  const io::ic::ResponseMatrix m = io::ic::build_response_matrix(
      binning, bins, truth_e, truth_z, sigma_e, sigma_z, 5.0, 0.0, unusable);

  // Total mass inside the domain, which the renormalisation divides out.
  const double inside = normal_cdf(7.0, 4.0, 0.30) - normal_cdf(2.0, 4.0, 0.30);

  for (int b = 0; b < binning.total_bins(); ++b) {
    const std::size_t begin = m.bin_offsets[static_cast<std::size_t>(b)];
    const std::size_t end   = m.bin_offsets[static_cast<std::size_t>(b) + 1];
    if (begin == end) continue;

    const int    energy_bin = b / 4;
    const double lo         = 2.0 + 0.1 * energy_bin;
    const double expected   = (normal_cdf(lo + 0.1, 4.0, 0.30) - normal_cdf(lo, 4.0, 0.30)) / inside;

    double got = 0.0;
    for (std::size_t k = begin; k < end; ++k) got += m.fractions[k];
    EXPECT_NEAR(got, expected, 1e-5) << "energy bin " << energy_bin;
  }
}

// The response is Gaussian in the zenith ANGLE while the axis is its cosine, so
// the bin's angular interval has to be taken through acos. Treating the axis as
// the response coordinate would put a symmetric response into asymmetric bins
// and is exactly what this catches: at cos = -0.9 the two neighbouring cos bins
// subtend very different angles.
TEST(ResponseMatrixTest, ZenithIsFoldedInAngleNotInCosine) {
  const io::ic::Binning binning = test_binning();

  // Sit on a cos-zenith bin edge (cos = -0.75) with a wide angular response, so
  // the split between the two bins is entirely decided by the mapping.
  const double              centre_z = std::acos(-0.75);
  const std::vector<int>    bins{0};
  const std::vector<double> truth_e{4.05}, truth_z{centre_z};
  const std::vector<double> sigma_e{1.0e-4}, sigma_z{0.20};

  std::size_t                  unusable = 0;
  const io::ic::ResponseMatrix m = io::ic::build_response_matrix(
      binning, bins, truth_e, truth_z, sigma_e, sigma_z, 5.0, 0.0, unusable);

  // Zenith bin 1 is cos in [-0.75, -0.5) -> angle in (acos(-0.5), acos(-0.75)],
  // bin 0 is cos in [-1, -0.75) -> angle in (acos(-0.75), pi].
  double got_bin0 = 0.0, got_bin1 = 0.0;
  for (int b = 0; b < binning.total_bins(); ++b) {
    const std::size_t begin = m.bin_offsets[static_cast<std::size_t>(b)];
    const std::size_t end   = m.bin_offsets[static_cast<std::size_t>(b) + 1];
    double            sum   = 0.0;
    for (std::size_t k = begin; k < end; ++k) sum += m.fractions[k];
    if (b % 4 == 0) got_bin0 += sum;
    if (b % 4 == 1) got_bin1 += sum;
  }

  const double a_edge = std::acos(-0.75);
  const double a_up   = std::acos(-0.5);
  const double a_pi   = 3.14159265358979323846;
  const double expect_bin0 = normal_cdf(a_pi, centre_z, 0.20) - normal_cdf(a_edge, centre_z, 0.20);
  const double expect_bin1 = normal_cdf(a_edge, centre_z, 0.20) - normal_cdf(a_up, centre_z, 0.20);

  // Renormalised over what stayed inside the domain.
  const double inside = expect_bin0 + expect_bin1 +
                        (normal_cdf(a_up, centre_z, 0.20) - normal_cdf(std::acos(0.0), centre_z, 0.20));

  EXPECT_NEAR(got_bin0, expect_bin0 / inside, 1e-4);
  EXPECT_NEAR(got_bin1, expect_bin1 / inside, 1e-4);

  // The cosine-space split would be 50/50 by construction, since the centre sits
  // on the edge. The angular one must not be.
  EXPECT_GT(std::abs(got_bin0 - got_bin1), 0.02) << "this looks like a fold in cosine, not in angle";
}

// An event with no usable response must keep the weight it already had, in the
// bin it already had. Dropping it would quietly change the predicted total.
TEST(ResponseMatrixTest, UnusableEventsKeepTheirUnfoldedBin) {
  const io::ic::Binning     binning = test_binning();
  const double              nan     = std::numeric_limits<double>::quiet_NaN();
  const std::vector<int>    bins{17, 33, 45};
  const std::vector<double> truth_e{4.0, nan, 4.0};
  const std::vector<double> truth_z{2.0, 2.0, 2.0};
  const std::vector<double> sigma_e{0.3, 0.3, 0.0};
  const std::vector<double> sigma_z{0.02, 0.02, 0.02};

  std::size_t                  unusable = 0;
  const io::ic::ResponseMatrix m = io::ic::build_response_matrix(
      binning, bins, truth_e, truth_z, sigma_e, sigma_z, 5.0, 1.0e-4, unusable);

  EXPECT_EQ(unusable, 2u);
  for (const double s : row_sums(m, bins.size())) EXPECT_NEAR(s, 1.0, 1e-6);

  for (const int fallback : {33, 45}) {
    bool found = false;
    for (std::size_t k = m.bin_offsets[static_cast<std::size_t>(fallback)];
         k < m.bin_offsets[static_cast<std::size_t>(fallback) + 1]; ++k)
      if (m.fractions[k] == 1.0f) found = true;
    EXPECT_TRUE(found) << "no whole-weight entry in the fallback bin " << fallback;
  }
}

// Events out of the analysis range are not in the sample and must not appear.
TEST(ResponseMatrixTest, OutOfRangeEventsAreSkipped) {
  const io::ic::Binning     binning = test_binning();
  const std::vector<int>    bins{-1, 20, -1};
  const std::vector<double> truth_e{4.0, 4.0, 4.0};
  const std::vector<double> truth_z{2.0, 2.0, 2.0};
  const std::vector<double> sigma_e{0.3, 0.3, 0.3};
  const std::vector<double> sigma_z{0.02, 0.02, 0.02};

  std::size_t                  unusable = 0;
  const io::ic::ResponseMatrix m = io::ic::build_response_matrix(
      binning, bins, truth_e, truth_z, sigma_e, sigma_z, 5.0, 1.0e-4, unusable);

  for (std::size_t k = 0; k < m.nnz(); ++k) EXPECT_EQ(m.events[k], 1);
}

// Truncation and the minimum-fraction cut are optimisations: they bound the
// matrix, and what survives is renormalised, so neither may move the total.
TEST(ResponseMatrixTest, TruncationAndMinFractionPreserveTheTotal) {
  const io::ic::Binning     binning = test_binning();
  const std::vector<int>    bins{100};
  const std::vector<double> truth_e{4.5}, truth_z{2.2};
  const std::vector<double> sigma_e{0.4}, sigma_z{0.08};

  std::size_t generous = 0, tight = 0;
  const io::ic::ResponseMatrix wide = io::ic::build_response_matrix(
      binning, bins, truth_e, truth_z, sigma_e, sigma_z, 8.0, 0.0, generous);
  const io::ic::ResponseMatrix narrow = io::ic::build_response_matrix(
      binning, bins, truth_e, truth_z, sigma_e, sigma_z, 2.0, 1.0e-2, tight);

  EXPECT_NEAR(row_sums(wide, 1)[0], 1.0, 1e-6);
  EXPECT_NEAR(row_sums(narrow, 1)[0], 1.0, 1e-6);
  EXPECT_LT(narrow.nnz(), wide.nnz());
}

TEST(ResponseConfigTest, ParsesTheBlock) {
  const auto tree = ic_tree(R"({ "tracks": {
      "binning": "tracks_2d",
      "parquet": "mc.parquet",
      "components": "astro, conventional, prompt",
      "Response": { "enabled": true, "TruthEnergyBranch": "my_e", "TruthZenithBranch": "my_z",
                    "EnergySigmaTransform": "exp", "Truncation": 4.0, "MinFraction": 1e-3 }
  } })");
  const auto samples = io::ic::parse_samples(tree);
  EXPECT_TRUE(samples[0].response.enabled);
  EXPECT_EQ(samples[0].response.truth_energy_branch, "my_e");
  EXPECT_EQ(samples[0].response.truth_zenith_branch, "my_z");
  EXPECT_EQ(samples[0].response.energy_sigma_transform, io::ic::SigmaTransform::Exp);
  EXPECT_EQ(samples[0].response.truncation, 4.0);
}

TEST(ResponseConfigTest, DefaultsToOff) {
  const auto tree = ic_tree(R"({ "tracks": {
      "binning": "tracks_2d",
      "parquet": "mc.parquet",
      "components": "astro, conventional, prompt"
  } })");
  EXPECT_FALSE(io::ic::parse_samples(tree)[0].response.enabled);
}

TEST(ResponseConfigTest, RejectsBadTransformAndTruncation) {
  EXPECT_THROW(io::ic::parse_samples(ic_tree(R"({ "tracks": {
      "binning": "tracks_2d", "parquet": "mc.parquet", "components": "astro",
      "Response": { "enabled": true, "ZenithSigmaTransform": "radians" } } })")),
               std::runtime_error);
  EXPECT_THROW(io::ic::parse_samples(ic_tree(R"({ "tracks": {
      "binning": "tracks_2d", "parquet": "mc.parquet", "components": "astro",
      "Response": { "enabled": true, "Truncation": 0.0 } } })")),
               std::runtime_error);
  EXPECT_THROW(io::ic::parse_samples(ic_tree(R"({ "tracks": {
      "binning": "tracks_2d", "parquet": "mc.parquet", "components": "astro",
      "Response": { "enabled": true, "TruthZenithBranch": "" } } })")),
               std::runtime_error);
}

// --- Category axes -------------------------------------------------------

// The axis carries its own branch name, since unlike every other kind there is
// no fixed set of columns it could read.
TEST(CategoryAxisTest, ParsesBranchAndEdges) {
  const io::ic::Axis axis = io::ic::parse_axis("Category:bdt_score", "[0.9, 0.99, 1.0]");
  EXPECT_EQ(axis.kind, io::ic::Axis::Kind::Category);
  EXPECT_EQ(axis.branch, "bdt_score");
  EXPECT_EQ(axis.n_bins, 2);
  EXPECT_FALSE(axis.uniform());
  // Identity projection: a classifier score is not a derived kinematic quantity.
  EXPECT_DOUBLE_EQ(axis.project(0.95), 0.95);
  EXPECT_EQ(axis.index(0.93), 0);
  EXPECT_EQ(axis.index(0.995), 1);
  EXPECT_EQ(axis.index(0.5), -1);
  EXPECT_EQ(axis.index(1.5), -1);
}

TEST(CategoryAxisTest, RejectsAnEmptyBranchName) {
  EXPECT_THROW(io::ic::parse_axis("Category:", "[0.0, 1.0]"), std::runtime_error);
}

// A category axis multiplies the bin count, and the flux components index on
// (energy, zenith) being the first two axes.
TEST(CategoryAxisTest, MultipliesTheBinCountAndIsListed) {
  const io::ic::Binning binning({io::ic::Axis{io::ic::Axis::Kind::Log10Energy, 2.0, 7.0, 50},
                                 io::ic::Axis{io::ic::Axis::Kind::CosZenith, -1.0, 0.0872, 33},
                                 io::ic::parse_axis("Category:bdt_score", "[0.9, 0.99, 1.0]"),
                                 io::ic::parse_axis("Category:L5_ldir_c", "[0, 700, 1600]")});
  EXPECT_EQ(binning.total_bins(), 50 * 33 * 2 * 2);

  const std::vector<io::ic::Axis> cats = io::ic::category_axes(binning);
  ASSERT_EQ(cats.size(), 2u);
  EXPECT_EQ(cats[0].branch, "bdt_score");
  EXPECT_EQ(cats[1].branch, "L5_ldir_c");

  // Row-major: the category indices are innermost, so an event in the last
  // category of both sits at the end of its (energy, zenith) block.
  const std::vector<double> reco{std::pow(10.0, 2.05), std::acos(-0.99), 0.995, 1000.0};
  const int                 bin = binning.bin_index(reco);
  ASSERT_GE(bin, 0);
  EXPECT_EQ(bin % 4, 3);
}

TEST(CategoryAxisTest, AnOrdinaryBinningHasNoCategoryAxes) {
  EXPECT_TRUE(io::ic::category_axes(test_binning()).empty());
}

// Every per-bin input -- gradients, muon template, galactic maps -- is delivered
// as a histogram in the 2D binning and cannot follow a finer one. Refusing at
// parse time is what makes "categories without gradients" a checked precondition
// rather than a silent binning mismatch.
TEST(CategoryAxisTest, RejectsPerBinInputsAlongsideACategoryAxis) {
  auto tree_with = [](const std::string& extra) {
    std::ostringstream json;
    json << R"JSON({"IceCube": {"Binnings": {"cat": {)JSON"
         << R"JSON("axes": "Log10Energy, CosZenith, Category:bdt_score",)JSON"
         << R"JSON("Log10Energy": "(2.0, 7.0, 50)", "CosZenith": "(-1.0, 0.0, 4)",)JSON"
         << R"JSON("Category:bdt_score": "[0.9, 0.99, 1.0]"}},)JSON"
         << R"JSON("Samples": {"tracks": {"binning": "cat", "parquet": "mc.parquet",)JSON"
         << R"JSON("components": "astro")JSON" << extra << "}}}}";
    std::istringstream          in(json.str());
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(in, tree);
    return tree.get_child("IceCube");
  };

  // Without any per-bin input the same binning parses fine.
  EXPECT_NO_THROW(io::ic::parse_samples(tree_with("")));

  EXPECT_THROW(io::ic::parse_samples(tree_with(R"JSON(, "Gradients": {"File": "g.txt"})JSON")),
               std::runtime_error);
  EXPECT_THROW(io::ic::parse_samples(tree_with(R"JSON(, "Template": {"File": "t.txt", "Norm": "MuonNorm"})JSON")),
               std::runtime_error);
}
