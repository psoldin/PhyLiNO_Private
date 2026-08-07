#pragma once

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

}  // namespace ana::ic
