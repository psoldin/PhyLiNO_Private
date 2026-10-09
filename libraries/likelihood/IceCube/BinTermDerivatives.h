#pragma once

#include "Polygamma.h"

#include <cmath>

/**
 * @file
 * @brief Derivatives of the per-bin likelihood terms with respect to the bin's
 * prediction mu and MC variance s, in exactly the branches poisson_llh() and
 * say_llh() evaluate. Used by SampleLikelihood::accumulate_gradient(); kept in
 * a header of their own so the tests can check every branch against finite
 * differences of say_bin_log_likelihood().
 */

namespace ana::ic::bin_terms {

  /// Derivatives of a bin's (or an RA group's) log-likelihood -- not -2 lnL --
  /// with respect to its prediction mu and variance s.
  struct TermDerivatives {
    double mu   = 0.0;
    double s    = 0.0;
    double mumu = 0.0;
    double mus  = 0.0;
    double ss   = 0.0;
  };

  /// Poisson term K log mu - n mu + const of n bins with total count K.
  inline TermDerivatives poisson_derivatives(const double mu, const double n, const double k_total) {
    TermDerivatives d;
    d.mu   = k_total / mu - n;
    d.mumu = -k_total / (mu * mu);
    return d;
  }

  /**
   * The SAY term of n bins sharing mu and s (unclipped, s > 0), with total
   * count K; `each_count` calls its argument with the count and weight of
   * every bin whose count is not zero (zero counts contribute nothing to the
   * alpha derivatives below). The weight is 1 for data; an expected-likelihood
   * Asimov passes each bin's quadrature nodes with their weights, and K is
   * then the sum of the bins' mean counts.
   *
   *   l = n (alpha log beta - lgamma alpha) - (K + n alpha) log1p beta
   *       + sum_r w_r lgamma(k_r + alpha) + const,
   *   alpha = mu^2 / s + a0,  beta = mu / s.
   *
   * Every difference that cancels in the Poisson limit (alpha, beta -> inf) is
   * rewritten so it is never formed by subtraction.
   */
  template <class EachCount>
  TermDerivatives say_derivatives(const double mu, const double s, const double n, const double k_total,
                                  const double alpha_offset, EachCount&& each_count, const bool second) {
    const double alpha = mu * mu / s + alpha_offset;
    const double beta  = mu / s;
    const double opb   = 1.0 + beta;

    // dl/dalpha = -n log1p(1/beta) + sum_r w_r [psi(k_r + alpha) - psi(alpha)]
    // (the weights of each bin sum to 1, so the psi(alpha) still come n times)
    double l_a  = -n * std::log1p(1.0 / beta);
    double l_aa = 0.0;
    each_count([&](const double k, const double w) {
      l_a += w * polygamma::digamma_difference(k, alpha);
      if (second) l_aa += w * polygamma::trigamma_difference(k, alpha);
    });
    // dl/dbeta = n alpha / beta - (K + n alpha) / (1 + beta)
    const double l_b = n * alpha / (beta * opb) - k_total / opb;

    const double a_mu = 2.0 * mu / s;
    const double a_s  = -mu * mu / (s * s);
    const double b_mu = 1.0 / s;
    const double b_s  = -mu / (s * s);

    TermDerivatives d;
    d.mu = l_a * a_mu + l_b * b_mu;
    d.s  = l_a * a_s + l_b * b_s;
    if (!second)
      return d;

    const double l_ab = n / (beta * opb);
    const double l_bb = -n * alpha * (1.0 + 2.0 * beta) / (beta * beta * opb * opb) + k_total / (opb * opb);

    const double a_mumu = 2.0 / s;
    const double a_mus  = -2.0 * mu / (s * s);
    const double a_ss   = 2.0 * mu * mu / (s * s * s);
    const double b_mus  = -1.0 / (s * s);
    const double b_ss   = 2.0 * mu / (s * s * s);

    d.mumu = l_aa * a_mu * a_mu + 2.0 * l_ab * a_mu * b_mu + l_bb * b_mu * b_mu + l_a * a_mumu;
    d.mus  = l_aa * a_mu * a_s + l_ab * (a_mu * b_s + a_s * b_mu) + l_bb * b_mu * b_s + l_a * a_mus + l_b * b_mus;
    d.ss   = l_aa * a_s * a_s + 2.0 * l_ab * a_s * b_s + l_bb * b_s * b_s + l_a * a_ss + l_b * b_ss;
    return d;
  }

  /**
   * The SAY term in the branches say_llh() takes: prediction clamped at 0
   * (no derivative below it), variance clipped into [0, mu^2] -- the Poisson
   * term at s <= 0, and at s >= mu^2 the term at s = mu^2, which then depends
   * on mu through the clip and not on s at all.
   */
  template <class EachCount>
  TermDerivatives say_term(const double value, const double s, const double n, const double k_total,
                           const double alpha_offset, EachCount&& each_count, const bool second) {
    if (value <= 0.0)
      return {};
    const double mu = value;
    if (s <= 0.0)
      return poisson_derivatives(mu, n, k_total);
    if (s < mu * mu)
      return say_derivatives(mu, s, n, k_total, alpha_offset, each_count, second);

    const TermDerivatives c = say_derivatives(mu, mu * mu, n, k_total, alpha_offset, each_count, second);
    TermDerivatives       d;
    d.mu   = c.mu + 2.0 * mu * c.s;
    d.mumu = c.mumu + 4.0 * mu * c.mus + 4.0 * mu * mu * c.ss + 2.0 * c.s;
    return d;
  }

}  // namespace ana::ic::bin_terms
