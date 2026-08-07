#include "PoissonLikelihood.h"

#include <cmath>

namespace ana::ic {

  double poisson_bin_log_likelihood(const double k, const double mu,
                                    const double lgamma_k_plus_1) noexcept {
    if (mu > 0.0)
      return k * std::log(mu) - mu - lgamma_k_plus_1;

    // mu <= 0: NNMFit's workaround for -inf (poisson_llh.py:89-102).
    return (k > 0.0) ? (-690.0 * k) : 0.0;
  }

  double poisson_bin_log_likelihood(const double k, const double mu) noexcept {
    return poisson_bin_log_likelihood(k, mu, std::lgamma(k + 1.0));
  }

  double poisson_bin_log_likelihood_saturated(const double k, const double mu,
                                              const double lgamma_k_plus_1) noexcept {
    // The saturated term is the same expression evaluated at mu = k, so k <= 0
    // takes the mu <= 0 branch and contributes 0 -- matching the nested switch
    // NNMFit's graph produces for compute_log_L(k=k, mu=k).
    return poisson_bin_log_likelihood(k, mu, lgamma_k_plus_1)
           - poisson_bin_log_likelihood(k, k, lgamma_k_plus_1);
  }

  double poisson_bin_log_likelihood_saturated(const double k, const double mu) noexcept {
    return poisson_bin_log_likelihood_saturated(k, mu, std::lgamma(k + 1.0));
  }

}  // namespace ana::ic
