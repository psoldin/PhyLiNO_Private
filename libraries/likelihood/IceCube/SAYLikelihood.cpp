#include "SAYLikelihood.h"

#include <algorithm>
#include <cmath>

namespace ana::ic {

  double say_bin_log_likelihood(const double k, const double mu, const double ssq) noexcept {
    // Clip ssq into [0, mu^2] before computing alpha/beta (matches NNMFit's
    // T.clip(ssq, 0, mu**2) in the graph version of compute_log_L).
    const double ssq_clipped = std::clamp(ssq, 0.0, mu * mu);

    double llh_eff;
    if (ssq_clipped > 0.0) {
      const double alpha = mu * mu / ssq_clipped + 1.0;
      const double beta  = mu / ssq_clipped;
      llh_eff            = alpha * std::log(beta)
                           + std::lgamma(k + alpha)
                           - std::lgamma(k + 1.0)
                           - (k + alpha) * std::log1p(beta)
                           - std::lgamma(alpha);
    } else {
      // ssq == 0 (after clipping): full Poisson log-PMF, matching NNMFit's
      // PoissonLLH.compute_log_L fallback (includes -lgamma(k+1) so this
      // branch is continuous with the general formula above as ssq -> 0;
      // PhyLiNO's existing plain-Poisson branch in ICLikelihood.cpp drops
      // that constant, which is fine there since it doesn't depend on
      // fit parameters, but would break continuity here).
      llh_eff = (mu > 0.0) ? (k * std::log(mu) - mu - std::lgamma(k + 1.0)) : 0.0;
    }

    // mu <= 0 always overrides (matches NNMFit's final switch; workaround for -inf).
    if (mu <= 0.0) {
      llh_eff = (k > 0.0) ? (-690.0 * k) : 0.0;
    }

    return llh_eff;
  }

}  // namespace ana::ic
