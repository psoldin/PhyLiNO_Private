#include "SAYLikelihood.h"

#include "PoissonLikelihood.h"

#include <algorithm>
#include <cmath>

namespace ana::ic {

  double say_bin_log_likelihood(const double k, const double mu, const double ssq) noexcept {
    return say_bin_log_likelihood(k, mu, ssq, std::lgamma(k + 1.0));
  }

  double say_bin_log_likelihood(const double k, const double mu, const double ssq,
                                const double lgamma_k_plus_1) noexcept {
    // Clip ssq into [0, mu^2] before computing alpha/beta (matches NNMFit's
    // T.clip(ssq, 0, mu**2) in the graph version of compute_log_L).
    const double ssq_clipped = std::clamp(ssq, 0.0, mu * mu);

    double llh_eff;
    if (ssq_clipped > 0.0) {
      const double alpha = mu * mu / ssq_clipped + 1.0;
      const double beta  = mu / ssq_clipped;
      llh_eff            = alpha * std::log(beta)
                           + std::lgamma(k + alpha)
                           - lgamma_k_plus_1
                           - (k + alpha) * std::log1p(beta)
                           - std::lgamma(alpha);
    } else {
      // ssq == 0 (after clipping): the *unsubtracted* Poisson log-PMF, which is
      // the fallback NNMFit's SAY graph switches to (say_llh.py:100-106 calls
      // PoissonLLH.compute_log_L, never the saturated-subtracted builder path).
      // Keeping -lgamma(k+1) here is what makes this branch continuous with the
      // general formula above as ssq -> 0.
      llh_eff = poisson_bin_log_likelihood(k, mu, lgamma_k_plus_1);
    }

    // mu <= 0 always overrides (matches NNMFit's final switch; workaround for -inf).
    if (mu <= 0.0) {
      llh_eff = (k > 0.0) ? (-690.0 * k) : 0.0;
    }

    return llh_eff;
  }

}  // namespace ana::ic
