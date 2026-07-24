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

}  // namespace ana::ic
