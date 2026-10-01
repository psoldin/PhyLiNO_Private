#include "SAYLikelihood.h"

#include "PoissonLikelihood.h"

#include <algorithm>
#include <cmath>

namespace ana::ic {

  double say_bin_log_likelihood(const double k, const double mu, const double ssq) noexcept {
    return say_bin_log_likelihood(k, mu, ssq, std::lgamma(k + 1.0));
  }

  template <typename T>
  [[nodiscard]] auto square(T&& t) noexcept {
    return t * t;
  }

  inline double calculate_effective_likelihood(const double mu,
                                               const double ssq_clipped,
                                               const double k,
                                               const double
                                               lgamma_k_plus_1) noexcept {

    const double alpha = mu * mu / ssq_clipped + 1.0;
    const double beta  = mu / ssq_clipped;

    return alpha * std::log(beta)
            + std::lgamma(k + alpha)
            - lgamma_k_plus_1
            - (k + alpha) * std::log1p(beta)
            - std::lgamma(alpha);
  }

  double say_bin_log_likelihood(const double k, const double mu, const double ssq,
                                const double lgamma_k_plus_1) noexcept {

    if (mu <= 0.0)
      return (k > 0.0) ? (-690.0 * k) : 0.0;

    // Clip ssq into [0, mu^2] before computing alpha/beta (matches NNMFit's
    // T.clip(ssq, 0, mu**2) in the graph version of compute_log_L).
    const double ssq_clipped = std::clamp(ssq, 0.0, mu * mu);

    return (ssq_clipped > 0.0) ? calculate_effective_likelihood(mu, ssq_clipped, k, lgamma_k_plus_1)
                               : poisson_bin_log_likelihood(k, mu, lgamma_k_plus_1);
  }

}  // namespace ana::ic
