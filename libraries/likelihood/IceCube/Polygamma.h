#pragma once

#include <cmath>

namespace ana::ic::polygamma {

  /**
   * Digamma and trigamma for positive arguments, the derivatives of lgamma the
   * analytic SAY gradient and Hessian need (see SampleLikelihood's
   * accumulate_gradient()).
   *
   * Both shift the argument up by recurrence until the asymptotic series is
   * accurate to ~1e-15, then sum that. Written here rather than taken from
   * Boost.Math because the likelihood library does not otherwise depend on it,
   * and because the gradient needs a *difference* of digammas at large argument
   * that neither library provides in a cancellation-free form (see
   * digamma_difference()).
   */

  namespace detail {
    /// Recurrence threshold: past 10 the next omitted asymptotic term is ~2e-14.
    constexpr double kShift = 10.0;

    /// The Bernoulli tail of the digamma series, sum_k B_2k / (2k x^2k).
    inline double digamma_tail(const double x) noexcept {
      const double f = 1.0 / (x * x);
      return f * (1.0 / 12.0 -
                  f * (1.0 / 120.0 -
                       f * (1.0 / 252.0 - f * (1.0 / 240.0 - f * (1.0 / 132.0 - f * (691.0 / 32760.0 - f / 12.0))))));
    }

    /// The Bernoulli series of the trigamma, sum_k B_2k / x^(2k+1), over r * f.
    inline double trigamma_series(const double f) noexcept {
      return 1.0 / 6.0 -
             f * (1.0 / 30.0 -
                  f * (1.0 / 42.0 - f * (1.0 / 30.0 - f * (5.0 / 66.0 - f * (691.0 / 2730.0 - f * (7.0 / 6.0))))));
    }
  }  // namespace detail

  /// psi(x) = d lgamma(x) / dx, x > 0.
  inline double digamma(double x) noexcept {
    double result = 0.0;
    while (x < detail::kShift) {
      result -= 1.0 / x;
      x += 1.0;
    }
    return result + std::log(x) - 0.5 / x - detail::digamma_tail(x);
  }

  /// psi_1(x) = d^2 lgamma(x) / dx^2, x > 0.
  inline double trigamma(double x) noexcept {
    double result = 0.0;
    while (x < detail::kShift) {
      result += 1.0 / (x * x);
      x += 1.0;
    }
    const double r = 1.0 / x;
    const double f = r * r;
    return result + r + 0.5 * f + r * f * detail::trigamma_series(f);
  }

  /**
   * psi(a + k) - psi(a) for a > 0, k >= 0.
   *
   * The SAY term's derivative with respect to its gamma shape alpha carries
   * exactly this difference, and alpha grows like mu^2 / sigma^2 -- 1e6 and
   * beyond in a bin with plenty of MC. There the two digammas agree to a few
   * parts in alpha and subtracting them directly loses that many digits. Past
   * the recurrence threshold both share one asymptotic expansion, so the
   * difference is taken term by term instead: log1p(k / a) for the logarithms,
   * which is exact to rounding however small k / a is.
   */
  inline double digamma_difference(const double k, const double a) noexcept {
    if (a < detail::kShift)
      return digamma(a + k) - digamma(a);
    const double b = a + k;
    // -0.5 * (1/b - 1/a) written as 0.5 * k / (a b), which needs no subtraction.
    return std::log1p(k / a) + 0.5 * k / (a * b) - (detail::digamma_tail(b) - detail::digamma_tail(a));
  }

  /**
   * psi_1(a + k) - psi_1(a) for a > 0, k >= 0, the second-derivative
   * counterpart of digamma_difference(): the leading terms of the asymptotic
   * series are differenced algebraically, so nothing of size 1/a is subtracted
   * from something of the same size.
   */
  inline double trigamma_difference(const double k, const double a) noexcept {
    if (a < detail::kShift)
      return trigamma(a + k) - trigamma(a);
    const double b  = a + k;
    const double ab = a * b;
    // 1/b - 1/a, 1/(2b^2) - 1/(2a^2) and 1/(6b^3) - 1/(6a^3), each as -k * (...).
    const double d1 = -k / ab;
    const double d2 = -0.5 * k * (a + b) / (ab * ab);
    const double d3 = -k * (a * a + ab + b * b) / (6.0 * ab * ab * ab);
    // The rest of the series is O(1/a^5) on each side; plain subtraction is
    // exact enough there.
    auto tail = [](const double x) {
      const double r = 1.0 / x;
      const double f = r * r;
      return r * f * (detail::trigamma_series(f) - 1.0 / 6.0);
    };
    return d1 + d2 + d3 + (tail(b) - tail(a));
  }

}  // namespace ana::ic::polygamma
