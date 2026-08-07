#pragma once

namespace ana::ic {

  /**
   * Poisson log-PMF for one bin, given the data count (k) and the sum of event
   * weights (mu). Port of NNMFit's PoissonLLH.compute_log_L
   * (NNMFit/likelihoods/impl/poisson_llh.py:76-104), including its mu <= 0
   * branch: -690 * k for k > 0 (their finite stand-in for -inf) and 0 for
   * k == 0.
   *
   * Returns log L for one bin (not -2 log L).
   */
  [[nodiscard]] double poisson_bin_log_likelihood(double k, double mu) noexcept;

  /**
   * The same term with lgamma(k + 1) supplied by the caller.
   *
   * k is the data count, which is fixed for the whole fit while mu moves every
   * evaluation, so its lgamma is a per-bin constant. Recomputing it inside the
   * loop costs one lgamma per analysis bin per evaluation -- 267k of them per
   * evaluation on the 3D tracks binning, where it was the single most expensive
   * term. The value must be exactly std::lgamma(k + 1.0) for the result to match
   * the two-argument overload bit for bit.
   */
  [[nodiscard]] double poisson_bin_log_likelihood(double k, double mu, double lgamma_k_plus_1) noexcept;

  /**
   * The same term with the saturated Poisson likelihood (mu = k) subtracted,
   * which is what NNMFit's LikelihoodBuilder builds for llh == "PoissonLLH"
   * (make_llh(substract_saturated=True) is the default, and the only path
   * run_fit takes unless minimizer_settings.subtract_saturated is set false).
   *
   * The subtraction removes the -lgamma(k+1) constant, so the sum over bins is
   * the saturated-likelihood test statistic (up to the factor of 2 the caller
   * applies) instead of a large offset that swamps the parameter dependence.
   */
  [[nodiscard]] double poisson_bin_log_likelihood_saturated(double k, double mu) noexcept;

  /** As above, with lgamma(k + 1) supplied by the caller (see the overload of
      poisson_bin_log_likelihood). The term cancels between the two halves of the
      subtraction, but is deliberately still applied to both rather than dropped:
      (a - L) - (b - L) is not exactly a - b in floating point, and this path is
      pinned against NNMFit golden values. */
  [[nodiscard]] double poisson_bin_log_likelihood_saturated(double k, double mu,
                                                            double lgamma_k_plus_1) noexcept;

}  // namespace ana::ic
