#include "NewtonMinimizer.h"

#include <Eigen/Dense>
#include <Fit/ParameterSettings.h>
#include <Math/IFunction.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>

namespace ana {

  namespace {

    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    /**
     * Solve A d = b for a symmetric A that is meant to be positive definite but
     * may be singular or slightly indefinite (a Gauss-Newton matrix with a flat
     * direction, an exact Hessian at a bound): eigenvalues below `floor` times
     * the largest are raised to it. Returns whether any had to be.
     */
    bool solve_floored(const Matrix& a, const Vector& b, Vector& d, Matrix* inverse = nullptr, const double floor = 1e-12) {
      Eigen::SelfAdjointEigenSolver<Matrix> eig(a);
      Vector       lambda  = eig.eigenvalues();
      const double largest = std::max(lambda.cwiseAbs().maxCoeff(), std::numeric_limits<double>::min());
      bool         floored = false;
      for (Eigen::Index i = 0; i < lambda.size(); ++i)
        if (!(lambda(i) > floor * largest)) {
          lambda(i) = floor * largest;
          floored   = true;
        }
      const Matrix& v = eig.eigenvectors();
      d               = v * (lambda.cwiseInverse().asDiagonal() * (v.transpose() * b));
      if (inverse) *inverse = v * lambda.cwiseInverse().asDiagonal() * v.transpose();
      return floored;
    }

    struct Box {
      bool   fixed = false;
      double lower = -std::numeric_limits<double>::infinity();
      double upper = std::numeric_limits<double>::infinity();
    };

  }  // namespace

  void NewtonMinimizer::SetHessianFunction(std::function<bool(std::span<const double>, double*)> hessian) {
    const unsigned int n = NDim();
    m_Hessian            = [hessian = std::move(hessian), n](const double* x, double* out) {
      return hessian(std::span<const double>(x, n), out);
    };
  }

  double NewtonMinimizer::CovMatrix(const unsigned int i, const unsigned int j) const {
    const std::size_t n = NDim();
    return m_Covariance.size() == n * n ? m_Covariance[i * n + j] : 0.0;
  }

  bool NewtonMinimizer::Minimize() {
    const ROOT::Math::IMultiGenFunction*  function = ObjFunction();
    const ROOT::Math::IMultiGradFunction* gradient = GradObjFunction();
    if (function == nullptr || gradient == nullptr || !m_Hessian) {
      std::cerr << "NewtonMinimizer: needs a gradient function and a Hessian\n";
      fStatus = 5;
      return false;
    }

    const std::size_t n = NDim();
    std::vector<Box>  box(n);
    for (std::size_t i = 0; i < n; ++i) {
      ROOT::Fit::ParameterSettings settings;
      GetVariableSettings(static_cast<unsigned int>(i), settings);
      box[i].fixed = settings.IsFixed();
      if (settings.HasLowerLimit()) box[i].lower = settings.LowerLimit();
      if (settings.HasUpperLimit()) box[i].upper = settings.UpperLimit();
    }

    std::vector<double> x(X(), X() + n);
    for (std::size_t i = 0; i < n; ++i) x[i] = std::clamp(x[i], box[i].lower, box[i].upper);

    const double       edm_target = 0.002 * Tolerance() * ErrorDef();
    const unsigned int max_calls  = MaxFunctionCalls() > 0 ? MaxFunctionCalls() : 10000;
    const double       eps        = std::numeric_limits<double>::epsilon();

    m_NCalls      = 0;
    m_NIterations = 0;
    std::vector<double> g(n), h(n * n);
    double              fx = (*function)(x.data());
    ++m_NCalls;
    // Hessian before gradient: a likelihood that computes both in one pass
    // (ICLikelihood does) then serves the gradient from its cache.
    m_Hessian(x.data(), h.data());
    gradient->Gradient(x.data(), g.data());

    auto model = [&](const std::size_t i, const std::size_t j) { return h[i * n + j]; };

    // The parameter is on a bound and the gradient pushes it outward: hold it.
    auto held = [&](const std::size_t i) {
      const double slack = 1e-12 * (1.0 + std::fabs(x[i]));
      return (x[i] <= box[i].lower + slack && g[i] > 0.0) || (x[i] >= box[i].upper - slack && g[i] < 0.0);
    };

    double lambda    = 0.0;
    double growth    = 2.0;
    bool   converged = false;
    double edm       = std::numeric_limits<double>::infinity();

    // The free set of the next step: not fixed, not pushed outward by the
    // gradient at a bound, and -- the binding-set rule of projected Newton --
    // not pushed outward by the Newton step itself. Without the last condition
    // a parameter on its bound is released whenever the gradient turns by a
    // hair, the coupled step drives it out of the box, the projection cuts the
    // step and the step is rejected, over and over.
    std::vector<std::size_t> free;
    Matrix                   hf;
    Vector                   gf, newton;
    auto build = [&] {
      std::vector<char> hold(n, 0);
      for (std::size_t i = 0; i < n; ++i) hold[i] = box[i].fixed || held(i);
      for (int pass = 0; pass <= static_cast<int>(n); ++pass) {
        free.clear();
        for (std::size_t i = 0; i < n; ++i)
          if (!hold[i]) free.push_back(i);
        const auto k = static_cast<Eigen::Index>(free.size());
        hf.resize(k, k);
        gf.resize(k);
        for (Eigen::Index a = 0; a < k; ++a) {
          gf(a) = g[free[a]];
          for (Eigen::Index b = 0; b < k; ++b) hf(a, b) = model(free[a], free[b]);
        }
        if (k == 0) return;
        solve_floored(hf, -gf, newton);
        bool changed = false;
        for (Eigen::Index a = 0; a < k; ++a) {
          const std::size_t i     = free[a];
          const double      slack = 1e-12 * (1.0 + std::fabs(x[i]));
          if ((x[i] <= box[i].lower + slack && newton(a) < 0.0) || (x[i] >= box[i].upper - slack && newton(a) > 0.0)) {
            hold[i] = 1;
            changed = true;
          }
        }
        if (!changed) return;
      }
    };

    while (m_NCalls < max_calls) {
      build();
      const auto k = static_cast<Eigen::Index>(free.size());
      if (k == 0) {
        edm       = 0.0;
        converged = true;
        break;
      }
      edm = -0.5 * gf.dot(newton);
      if (PrintLevel() > 0)
        std::printf("NewtonMinimizer %3u: f = %.10f  edm = %.3e  lambda = %.1e  free = %td\n", m_NIterations, fx, edm,
                    lambda, k);
      if (edm < edm_target) {
        converged = true;
        break;
      }

      // Levenberg-Marquardt: damp along the diagonal until the projected step
      // decreases the likelihood by a reasonable fraction of its prediction.
      bool accepted = false;
      for (int attempt = 0; attempt < 40 && m_NCalls < max_calls; ++attempt) {
        Matrix damped = hf;
        for (Eigen::Index a = 0; a < k; ++a) damped(a, a) += lambda * std::max(hf(a, a), 1e-12);
        Vector step;
        solve_floored(damped, -gf, step);

        std::vector<double> trial = x;
        for (Eigen::Index a = 0; a < k; ++a) {
          const std::size_t i = free[a];
          trial[i]            = std::clamp(x[i] + step(a), box[i].lower, box[i].upper);
        }

        // Model decrease of the step actually taken, i.e. after projection.
        Vector dp(k);
        for (Eigen::Index a = 0; a < k; ++a) dp(a) = trial[free[a]] - x[free[a]];
        const double predicted = -(gf.dot(dp) + 0.5 * dp.dot(hf * dp));

        const double ft = (*function)(trial.data());
        ++m_NCalls;

        // Past the point where the two likelihood values differ by more than
        // their rounding, the ratio below is noise: accept anything that did
        // not get worse beyond it.
        const double noise = 100.0 * eps * std::max(1.0, std::fabs(fx));
        const double rho   = predicted > noise ? (fx - ft) / predicted : (ft <= fx + noise ? 1.0 : -1.0);

        if (std::isfinite(ft) && rho > 1e-4) {
          x  = std::move(trial);
          fx = ft;
          m_Hessian(x.data(), h.data());
          gradient->Gradient(x.data(), g.data());

          // Any acceptable step lets the damping fall. With diagonal scaling a
          // lambda of a few percent already throttles the poorly constrained,
          // strongly correlated directions -- exactly those a fit still has
          // to move along late in the minimization -- so holding it there
          // after merely decent steps turns the end game linear.
          lambda = rho < 0.25 ? std::max(2.0 * lambda, 1e-3) : lambda / 3.0;
          if (lambda < 1e-7) lambda = 0.0;
          growth = 2.0;
          accepted = true;
          break;
        }
        lambda = lambda == 0.0 ? 1e-3 : growth * lambda;
        growth *= 2.0;
        if (lambda > 1e12) break;
      }
      ++m_NIterations;
      if (!accepted) break;
    }

    SetFinalValues(x.data());
    SetMinValue(fx);
    m_Edm      = edm;
    m_Gradient = g;

    // Errors from the Hessian the minimization ended on, until Hesse() replaces
    // them with the exact ones.
    set_errors(h, /*exact=*/false);

    fStatus = converged ? 0 : 3;
    return converged;
  }

  bool NewtonMinimizer::Hesse() {
    if (!m_Hessian) return false;
    std::vector<double> h(static_cast<std::size_t>(NDim()) * NDim());
    if (!m_Hessian(X(), h.data())) return false;
    set_errors(h, /*exact=*/true);
    return true;
  }

  void NewtonMinimizer::set_errors(const std::vector<double>& h, const bool exact) {
    const std::size_t n = NDim();

    // A parameter the minimization ended holding on a bound is treated as
    // fixed for the others' errors: the curvature beyond the bound is not part
    // of the problem, and letting the others trade against it would report the
    // errors of a likelihood the parameter is not allowed to explore. Its own
    // error is its conditional one, from its diagonal alone.
    std::vector<std::size_t> free, held;
    for (std::size_t i = 0; i < n; ++i) {
      if (IsFixedVariable(static_cast<unsigned int>(i))) continue;
      ROOT::Fit::ParameterSettings settings;
      GetVariableSettings(static_cast<unsigned int>(i), settings);
      const double x     = X()[i];
      const double slack = 1e-12 * (1.0 + std::fabs(x));
      const double g     = i < m_Gradient.size() ? m_Gradient[i] : 0.0;
      const bool   on    = (settings.HasLowerLimit() && x <= settings.LowerLimit() + slack && g > 0.0) ||
                      (settings.HasUpperLimit() && x >= settings.UpperLimit() - slack && g < 0.0);
      (on ? held : free).push_back(i);
    }
    const auto k = static_cast<Eigen::Index>(free.size());

    m_Errors.assign(n, 0.0);
    m_Covariance.assign(n * n, 0.0);
    for (const std::size_t i : held)
      if (h[i * n + i] > 0.0) {
        m_Covariance[i * n + i] = 2.0 * ErrorDef() / h[i * n + i];
        m_Errors[i]             = std::sqrt(m_Covariance[i * n + i]);
      }
    if (k == 0) return;

    Matrix hf(k, k);
    for (Eigen::Index a = 0; a < k; ++a)
      for (Eigen::Index b = 0; b < k; ++b) hf(a, b) = h[free[a] * n + free[b]];
    Vector       unused, zero = Vector::Zero(k);
    Matrix       inverse;
    const bool   forced = solve_floored(hf, zero, unused, &inverse);
    for (Eigen::Index a = 0; a < k; ++a) {
      for (Eigen::Index b = 0; b < k; ++b) m_Covariance[free[a] * n + free[b]] = 2.0 * ErrorDef() * inverse(a, b);
      m_Errors[free[a]] = std::sqrt(std::max(0.0, m_Covariance[free[a] * n + free[a]]));
    }
    // Minuit2's convention: 3 = accurate, 2 = forced positive definite, 1 = approximate.
    m_CovStatus = !exact ? 1 : (forced ? 2 : 3);
  }

}  // namespace ana
