#pragma once

#include <span>

namespace ana::ic {

  /**
   * SAY (Saturated Asimov Yield) effective likelihood: a gamma-Poisson mixture
   * accounting for finite MC statistics per bin, given the per-bin sum of
   * event weights (mu) and sum of squared event weights (ssq).
   * Port of NNMFit's SAYLLH.compute_log_L (arXiv:1901.04645),
   * NNMFit/likelihoods/impl/say_llh.py:71-124.
   *
   * Returns log L for one bin (not -2 log L).
   */
  [[nodiscard]] double say_bin_log_likelihood(double k, double mu, double ssq) noexcept;

  /**
   * The same term with lgamma(k + 1) supplied by the caller.
   *
   * k is fixed for the whole fit while mu and ssq move every evaluation, so this
   * hoists one of the three lgamma calls out of the per-analysis-bin loop. The
   * value must be exactly std::lgamma(k + 1.0) for the result to match the
   * three-argument overload bit for bit.
   */
  [[nodiscard]] double say_bin_log_likelihood(double k, double mu, double ssq,
                                              double lgamma_k_plus_1) noexcept;

  /**
   * The same term with the gamma shape alpha = mu^2/ssq + alpha_offset.
   *
   * alpha_offset = 1 is L_Eff (the overloads above): the gamma's mode sits at mu
   * and its mean at mu + ssq/mu. alpha_offset = 0 is L_Mean of arXiv:1901.04645:
   * mean mu, variance mu + ssq. beta = mu/ssq in both.
   */
  [[nodiscard]] double say_bin_log_likelihood(double k, double mu, double ssq,
                                              double lgamma_k_plus_1,
                                              double alpha_offset) noexcept;

  /**
   * The term averaged over a distribution of counts, for the expected-likelihood
   * Asimov (see ExpectedAsimov.h): sum_j w_j l(k_j) with sum_j w_j = 1, in the
   * same branches as say_bin_log_likelihood(). Every part of l but
   * lgamma(k + alpha) is linear in k, so the caller passes the mean count
   * `k_mean` = sum_j w_j k_j and `expected_lgamma_k_plus_1` =
   * sum_j w_j lgamma(k_j + 1), and only lgamma(k + alpha) walks the nodes.
   */
  [[nodiscard]] double say_bin_expected_log_likelihood(std::span<const double> k, std::span<const double> w,
                                                       double k_mean, double mu, double ssq,
                                                       double expected_lgamma_k_plus_1,
                                                       double alpha_offset) noexcept;

}  // namespace ana::ic
