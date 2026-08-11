// Plain-C++, GPU-free validation of the double-float (df32 hi+lo) primitives
// used by the Metal df64 kernels. two_sum/two_prod/df_add/df_mul are built
// only from +,-,*,fma -- IEEE-754-conformant single-precision ops that behave
// identically on CPU float and on Metal float (fastMathEnabled = NO), so
// correctness proven here transfers to the MSL transcription without needing
// a GPU to test against.
#include "Df64.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>

using ana::ic::df64::Df32;
using ana::ic::df64::df_add;
using ana::ic::df64::df_exp;
using ana::ic::df64::df_from_double;
using ana::ic::df64::df_mul;
using ana::ic::df64::df_sub;
using ana::ic::df64::df_to_double;
using ana::ic::df64::two_prod;
using ana::ic::df64::two_sum;

namespace {
  // ~1 part in 2^46: two float32 mantissas (24 bits each) overlap a little
  // once normalised, so df32 lands a few bits short of the naive 48-bit sum.
  constexpr double kDf64Rel = 1.0e-13;
}  // namespace

TEST(Df64Test, TwoSumIsExactRoundingError) {
  std::mt19937                         rng(42);
  std::uniform_real_distribution<float> dist(-1.0e6f, 1.0e6f);
  for (int i = 0; i < 10000; ++i) {
    const float a = dist(rng);
    const float b = dist(rng);
    const Df32  s = two_sum(a, b);
    // s.hi + s.lo must reproduce the exact mathematical sum a+b to full
    // double precision -- the defining property of TwoSum.
    const double exact = static_cast<double>(a) + static_cast<double>(b);
    const double got    = static_cast<double>(s.hi) + static_cast<double>(s.lo);
    EXPECT_DOUBLE_EQ(exact, got);
  }
}

TEST(Df64Test, TwoProdIsExactRoundingError) {
  std::mt19937                         rng(43);
  std::uniform_real_distribution<float> dist(-1.0e3f, 1.0e3f);
  for (int i = 0; i < 10000; ++i) {
    const float a = dist(rng);
    const float b = dist(rng);
    const Df32  p = two_prod(a, b);
    const double exact = static_cast<double>(a) * static_cast<double>(b);
    const double got    = static_cast<double>(p.hi) + static_cast<double>(p.lo);
    EXPECT_DOUBLE_EQ(exact, got);
  }
}

TEST(Df64Test, DfFromDoubleRoundTripsNearDoublePrecision) {
  std::mt19937                    rng(44);
  std::uniform_real_distribution<double> dist(1.0e-3, 1.0e6);
  for (int i = 0; i < 1000; ++i) {
    const double x  = dist(rng);
    const Df32   df = df_from_double(x);
    const double back = df_to_double(df);
    EXPECT_NEAR(back, x, std::fabs(x) * kDf64Rel);
  }
}

TEST(Df64Test, DfAddMatchesDoubleAddition) {
  // Looser than kDf64Rel: each df_from_double() input already carries its own
  // ~1.5e-13-relative conversion error (a df32 pair cannot represent an
  // arbitrary double exactly), and df_add combines two of them -- the
  // addition itself is exact to df32's own precision (see DfExp, which hits
  // kDf64Rel with no input-conversion error to absorb).
  constexpr double kInputConversionRel = 3.0e-13;
  std::mt19937                    rng(45);
  std::uniform_real_distribution<double> dist(-1.0e4, 1.0e4);
  for (int i = 0; i < 2000; ++i) {
    const double a = dist(rng);
    const double b = dist(rng);
    const Df32   r = df_add(df_from_double(a), df_from_double(b));
    const double expected = a + b;
    const double scale    = std::max({std::fabs(expected), 1.0});
    EXPECT_NEAR(df_to_double(r), expected, scale * kInputConversionRel);
  }
}

TEST(Df64Test, DfMulMatchesDoubleMultiplication) {
  std::mt19937                    rng(46);
  std::uniform_real_distribution<double> dist(-1.0e3, 1.0e3);
  for (int i = 0; i < 2000; ++i) {
    const double a = dist(rng);
    const double b = dist(rng);
    const Df32   r = df_mul(df_from_double(a), df_from_double(b));
    const double expected = a * b;
    const double scale    = std::max({std::fabs(expected), 1.0});
    EXPECT_NEAR(df_to_double(r), expected, scale * kDf64Rel);
  }
}

// The workload this exists for: exponent * (logE - logEref), typically in
// [-15, 15] across the IceCube astro sample's energy range.
TEST(Df64Test, DfExpMatchesStdExpAcrossFluxArgumentRange) {
  std::mt19937                    rng(47);
  std::uniform_real_distribution<double> dist(-15.0, 15.0);
  double max_rel = 0.0;
  for (int i = 0; i < 5000; ++i) {
    const double x        = dist(rng);
    const Df32   r        = df_exp(df_from_double(x));
    const double expected = std::exp(x);
    const double rel      = std::fabs(df_to_double(r) - expected) / expected;
    max_rel               = std::max(max_rel, rel);
  }
  EXPECT_LT(max_rel, kDf64Rel) << "max_rel=" << max_rel;
}

TEST(Df64Test, DfExpHandlesZeroAndSmallArguments) {
  EXPECT_NEAR(df_to_double(df_exp(df_from_double(0.0))), 1.0, 1.0e-14);
  const double x = 1.0e-6;
  EXPECT_NEAR(df_to_double(df_exp(df_from_double(x))), std::exp(x), 1.0e-14);
}

TEST(Df64Test, DfSubMatchesDoubleSubtraction) {
  const Df32   r = df_sub(df_from_double(3.0000001), df_from_double(3.0));
  EXPECT_NEAR(df_to_double(r), 0.0000001, 1.0e-14);
}
