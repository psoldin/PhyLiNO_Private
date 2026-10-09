#pragma once

#include <vector>

/**
 * @file
 * @brief The counts an expected-likelihood Asimov bin is averaged over.
 *
 * Plain Asimov data sets k = mu at the truth. That is the maximum of a Poisson
 * term, whose derivative is linear in k, but not of a SAY term: its derivative
 * carries psi(k + alpha), which is curved in k, so the derivative at the mean
 * count is not the mean derivative. Wherever sigma^2 is large compared with mu
 * (small alpha) the fit then moves away from the truth even on Asimov data.
 *
 * The expected-likelihood Asimov replaces each bin's term l(k) by its average
 * over the counts the SAY model expects at the truth,
 *
 *   E[l] = sum_j w_j l(k_j),
 *
 * whose derivative at the truth is the expected score, zero for any properly
 * normalised likelihood. The distribution is the gamma-Poisson mixture of mean
 * mu and variance mu + sigma^2 -- a negative binomial with shape mu^2/sigma^2
 * (L_Mean's), or Poisson at sigma^2 = 0 -- whatever SAY variant is fitted, so
 * L_Eff's own offset shows up as the bias it is.
 *
 * Every term of l except lgamma(k + alpha) is linear in k, so only that one
 * needs the nodes; the rest takes the mean count.
 */

namespace ana::ic {

  /// Quadrature order: counts below it are kept exactly, and each band of the
  /// distribution above it gets a Gauss rule of this many nodes. 8 reproduces
  /// the exact expectation of psi(k + alpha) to ~1e-8 of its Jensen gap from
  /// psi(mu + alpha) -- the effect this exists to capture -- from Poisson bins
  /// to heavy-tailed ones (alpha ~ 1, mean 1e4).
  inline constexpr int kExpectedAsimovNodes = 8;

  /**
   * Append the nodes `k` and weights `w` (summing to 1) of the count
   * distribution of mean `mean` and variance mean + clip(ssq, 0, mean^2).
   *
   * Counts whose probability is below 1e-17 of the mode's are dropped. Of the
   * rest, those below `max_nodes` are nodes themselves, with their
   * probabilities as weights: lgamma(k + alpha) is least polynomial there when
   * alpha is small. Above, the support is cut into bands a factor 4 wide in k,
   * and each band with more than `max_nodes` counts is replaced by its Gauss
   * rule of that order (non-integer nodes inside the band, moments up to
   * 2 max_nodes - 1 exact). A bin of ordinary width costs one or two bands;
   * only a heavy-tailed one spans more. A non-positive mean gives a single
   * node at 0.
   */
  void append_expected_counts(double mean, double ssq, int max_nodes, std::vector<double>& k,
                              std::vector<double>& w);

}  // namespace ana::ic
