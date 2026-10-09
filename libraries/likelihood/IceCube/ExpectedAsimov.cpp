#include "ExpectedAsimov.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

namespace ana::ic {

  namespace {

    /// Counts below this log-probability relative to the mode are dropped (1e-17).
    constexpr double kLogCut = -39.1;
    /// Guard against a support no bin of a real analysis would have.
    constexpr std::size_t kMaxSupport = 50'000'000;

    /**
     * The gamma-Poisson count distribution of mean mu and variance mu + s: a
     * negative binomial NB(alpha = mu^2/s, q = s/(mu + s)), or Poisson(mu) at
     * s = 0. Log-probabilities up to a constant, which the normalisation drops.
     */
    struct CountDistribution {
      double mean;
      double alpha = 0.0;  ///< 0 selects Poisson
      double q     = 0.0;
      double log_q = 0.0;
      double log_mean;

      CountDistribution(const double mu, const double s) : mean(mu), log_mean(std::log(mu)) {
        if (s > 0.0) {
          alpha = mu * mu / s;
          q     = s / (mu + s);
          log_q = std::log(q);
        }
      }

      [[nodiscard]] bool poisson() const noexcept { return alpha == 0.0; }

      [[nodiscard]] double log_pmf(const double k) const {
        if (poisson())
          return k * log_mean - std::lgamma(k + 1.0);
        return std::lgamma(k + alpha) - std::lgamma(k + 1.0) + k * log_q;
      }

      [[nodiscard]] double mode() const {
        if (poisson())
          return std::floor(mean);
        // floor((alpha - 1) q / (1 - q)), and (alpha - 1) s / mu = mu - s / mu.
        return alpha > 1.0 ? std::floor(mean - mean / alpha) : 0.0;
      }

      /// A bound on P(j + 1) / P(j) for every j >= k past the mode.
      [[nodiscard]] double upward_ratio_bound(const double k) const {
        if (poisson())
          return mean / (k + 1.0);
        const double ratio = q * (k + alpha) / (k + 1.0);
        // The ratio falls towards q for alpha > 1 and rises towards it below.
        return alpha >= 1.0 ? ratio : q;
      }
    };

    /**
     * Eigenvalues `d` and the first components `z0` of the eigenvectors of the
     * symmetric tridiagonal matrix with diagonal `d` and sub-diagonal `e`
     * (e[i] couples i and i + 1), by implicit QL with Wilkinson shifts. The
     * Golub-Welsch weights are z0^2; only the first row of the eigenvector
     * matrix is carried, since every rotation acts on each row on its own.
     */
    void tridiagonal_eigen(std::vector<double>& d, std::vector<double> e, std::vector<double>& z0) {
      const int n = static_cast<int>(d.size());
      e.resize(static_cast<std::size_t>(n), 0.0);
      e[static_cast<std::size_t>(n - 1)] = 0.0;
      z0.assign(static_cast<std::size_t>(n), 0.0);
      z0[0] = 1.0;

      for (int l = 0; l < n; ++l) {
        int iterations = 0;
        int m          = l;
        do {
          for (m = l; m < n - 1; ++m) {
            const double dd = std::fabs(d[m]) + std::fabs(d[m + 1]);
            if (std::fabs(e[m]) <= 1.0e-16 * dd)
              break;
          }
          if (m == l)
            break;
          if (++iterations > 100)
            throw std::runtime_error("append_expected_counts: tridiagonal eigenvalues did not converge");

          double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
          double r = std::hypot(g, 1.0);
          g        = d[m] - d[l] + e[l] / (g + std::copysign(r, g));
          double s = 1.0, c = 1.0, p = 0.0;
          int    i = m - 1;
          for (; i >= l; --i) {
            const double f = s * e[i];
            const double b = c * e[i];
            r              = std::hypot(f, g);
            e[i + 1]       = r;
            if (r == 0.0) {
              d[i + 1] -= p;
              e[m] = 0.0;
              break;
            }
            s = f / r;
            c = g / r;
            g = d[i + 1] - p;
            r = (d[i] - g) * s + 2.0 * c * b;
            p = s * r;
            d[i + 1] = g + p;
            g        = c * r - b;

            const double z = z0[i + 1];
            z0[i + 1]      = s * z0[i] + c * z;
            z0[i]          = c * z0[i] - s * z;
          }
          if (r == 0.0 && i >= l)
            continue;
          d[l] -= p;
          e[l] = g;
          e[m] = 0.0;
        } while (m != l);
      }
    }

    /**
     * Append the Gauss rule of order at most `max_nodes` of the discrete
     * measure (x, p) (p need not be normalised; the weights sum to its mass).
     * Lanczos with full reorthogonalisation builds its Jacobi matrix in a
     * standardised variable; the matrix's eigenvalues are the nodes and its
     * eigenvectors' squared first components the weights (Golub-Welsch). A
     * measure on at most `max_nodes` points is copied as it is.
     */
    void append_gauss_rule(const std::span<const double> x, const std::span<const double> p_in, const int max_nodes,
                           std::vector<double>& k, std::vector<double>& w) {
      const std::size_t n = x.size();
      if (n <= static_cast<std::size_t>(max_nodes)) {
        k.insert(k.end(), x.begin(), x.end());
        w.insert(w.end(), p_in.begin(), p_in.end());
        return;
      }
      double mass = 0.0;
      for (const double v : p_in) mass += v;
      std::vector<double> p(n);
      for (std::size_t i = 0; i < n; ++i) p[i] = p_in[i] / mass;

      double m = 0.0;
      for (std::size_t i = 0; i < n; ++i) m += p[i] * x[i];
      double var = 0.0;
      for (std::size_t i = 0; i < n; ++i) var += p[i] * (x[i] - m) * (x[i] - m);
      const double        sd = std::sqrt(var);
      std::vector<double> t(n);
      for (std::size_t i = 0; i < n; ++i) t[i] = (x[i] - m) / sd;

      auto inner = [&](const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += p[i] * a[i] * b[i];
        return s;
      };
      std::vector<std::vector<double>> basis;
      basis.emplace_back(n, 1.0);
      std::vector<double> diag, off, r(n);
      for (int j = 0;; ++j) {
        const std::vector<double>& qj = basis.back();
        double                     a  = 0.0;
        for (std::size_t i = 0; i < n; ++i) a += p[i] * t[i] * qj[i] * qj[i];
        diag.push_back(a);
        if (j + 1 == max_nodes)
          break;

        for (std::size_t i = 0; i < n; ++i) r[i] = (t[i] - a) * qj[i];
        if (j > 0)
          for (std::size_t i = 0; i < n; ++i) r[i] -= off.back() * basis[basis.size() - 2][i];
        for (int pass = 0; pass < 2; ++pass)
          for (const std::vector<double>& qi : basis) {
            const double c = inner(r, qi);
            for (std::size_t i = 0; i < n; ++i) r[i] -= c * qi[i];
          }
        const double b = std::sqrt(inner(r, r));
        // The measure has (numerically) fewer points than max_nodes: the rule
        // built so far already integrates it to rounding.
        if (b < 1.0e-12)
          break;
        off.push_back(b);
        for (double& v : r) v /= b;
        basis.push_back(r);
      }

      std::vector<double> z0;
      tridiagonal_eigen(diag, off, z0);
      for (std::size_t j = 0; j < diag.size(); ++j) {
        k.push_back(std::max(0.0, m + sd * diag[j]));
        w.push_back(mass * z0[j] * z0[j]);
      }
    }

  }  // namespace

  void append_expected_counts(const double mean, const double ssq, const int max_nodes, std::vector<double>& k,
                              std::vector<double>& w) {
    if (!(mean > 0.0)) {
      k.push_back(0.0);
      w.push_back(1.0);
      return;
    }
    if (max_nodes < 1)
      throw std::invalid_argument("append_expected_counts: max_nodes must be positive");

    const CountDistribution dist(mean, std::clamp(ssq, 0.0, mean * mean));
    const double            mode    = dist.mode();
    const double            lp_mode = dist.log_pmf(mode);

    // The support: walk out from the mode until the probability -- and, above
    // it, a geometric bound on everything further out -- is below the cut.
    double lo = mode;
    while (lo > 0.0 && dist.log_pmf(lo - 1.0) - lp_mode > kLogCut) lo -= 1.0;
    double hi = mode;
    for (;;) {
      const double lp = dist.log_pmf(hi) - lp_mode;
      const double r  = dist.upward_ratio_bound(hi);
      if (hi > mode && lp < kLogCut && r < 1.0 && lp + std::log(r / (1.0 - r)) < kLogCut)
        break;
      hi += 1.0;
      if (hi - lo > static_cast<double>(kMaxSupport))
        throw std::runtime_error("append_expected_counts: support of a bin with mean " + std::to_string(mean) +
                                 " and variance " + std::to_string(ssq) + " is too wide");
    }
    // hi is the first count past the cut; it is not part of the support.
    const std::size_t   n_support = static_cast<std::size_t>(hi - lo);
    std::vector<double> x(n_support), p(n_support);
    double              norm = 0.0;
    for (std::size_t i = 0; i < n_support; ++i) {
      x[i] = lo + static_cast<double>(i);
      p[i] = std::exp(dist.log_pmf(x[i]) - lp_mode);
      norm += p[i];
    }
    for (double& v : p) v /= norm;

    if (n_support <= static_cast<std::size_t>(max_nodes)) {
      k.insert(k.end(), x.begin(), x.end());
      w.insert(w.end(), p.begin(), p.end());
      return;
    }

    // lgamma(k + alpha) is far from polynomial in k only next to k = 0, where
    // alpha can be small, and it varies like log k: keep the counts below
    // max_nodes exactly, and above them take a Gauss rule per band of a factor
    // 4 in k, inside which it is smooth. A bin of ordinary width falls into one
    // or two bands; only a heavy-tailed one (alpha ~ 1, large mean) spans more.
    std::size_t i = 0;
    while (i < n_support && x[i] < static_cast<double>(max_nodes)) {
      k.push_back(x[i]);
      w.push_back(p[i]);
      ++i;
    }
    double band_end = static_cast<double>(max_nodes);
    while (i < n_support) {
      while (band_end <= x[i]) band_end *= 4.0;
      std::size_t end = i;
      while (end < n_support && x[end] < band_end) ++end;
      append_gauss_rule(std::span<const double>(x).subspan(i, end - i), std::span<const double>(p).subspan(i, end - i),
                        max_nodes, k, w);
      i = end;
    }
  }

}  // namespace ana::ic
