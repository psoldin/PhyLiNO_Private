#pragma once

#include <Math/BasicMinimizer.h>

#include <functional>
#include <span>
#include <vector>

namespace ana {

  /**
   * Projected trust-region Newton (Levenberg-Marquardt) minimizer for
   * likelihoods with an analytic gradient and a positive semi-definite Hessian
   * approximation (--minimizer Newton).
   *
   * Each iteration solves (H_F + lambda diag(H_F)) d = -g_F over the free
   * parameters F that are not held at a bound, projects x + d back into the
   * box, and accepts the step if the likelihood fell by a reasonable fraction
   * of what the quadratic model predicted; lambda shrinks after good steps and
   * grows after bad ones. With the Gauss-Newton Hessian of a binned likelihood
   * this needs a handful of iterations, each one likelihood evaluation, one
   * gradient and one Hessian, where Migrad's quasi-Newton updates need dozens.
   *
   * Bounds are handled in the parameter's own coordinate: a parameter sits on
   * its bound when the gradient pushes it outward, and is then held there
   * (active set) until the gradient turns. There is no internal
   * transformation as in Minuit2, so a minimum on a bound -- a norm at 0 -- is
   * an ordinary case rather than a singular point of arcsin or sqrt.
   *
   * Converged when the estimated distance to the minimum,
   * EDM = g_F^T H_F^-1 g_F / 2, is below 0.002 * tolerance * ErrorDef: the
   * same criterion and the same tolerance semantics as Migrad, so --tolerance
   * and --scanTolerance mean the same thing for both minimizers.
   *
   * Hesse() inverts whatever the Hessian callback returns at the minimum over
   * the free parameters; Fit switches that callback to the exact Hessian
   * before calling it, so the errors are the exact curvature's.
   *
   * The variable bookkeeping (values, bounds, fixing, names) is
   * ROOT::Math::BasicMinimizer's, so to the rest of the code this is just
   * another ROOT::Math::Minimizer.
   */
  class NewtonMinimizer final : public ROOT::Math::BasicMinimizer {
   public:
    using HessianFn = std::function<bool(const double* x, double* hessian)>;

    NewtonMinimizer() = default;

    /** Positive semi-definite Hessian approximation, row-major n x n over all parameters. */
    void set_hessian(HessianFn hessian) { m_Hessian = std::move(hessian); }

    bool Minimize() override;
    bool Hesse() override;

    [[nodiscard]] double        Edm() const override { return m_Edm; }
    [[nodiscard]] const double* MinGradient() const override { return m_Gradient.empty() ? nullptr : m_Gradient.data(); }
    [[nodiscard]] unsigned int  NCalls() const override { return m_NCalls; }
    [[nodiscard]] unsigned int  NIterations() const override { return m_NIterations; }
    [[nodiscard]] bool          ProvidesError() const override { return true; }
    [[nodiscard]] const double* Errors() const override { return m_Errors.data(); }
    [[nodiscard]] double        CovMatrix(unsigned int i, unsigned int j) const override;
    [[nodiscard]] int           CovMatrixStatus() const override { return m_CovStatus; }

    // ROOT::Math::Minimizer's span-taking virtuals, overridden so this class's
    // vtable does not reference the base versions, which libMathCore may
    // export under a different std::span than this build sees (see Fit.cpp).
    // The Hessian goes through set_hessian().
    void SetHessianFunction(std::function<bool(std::span<const double>, double*)> hessian) override;
    bool SetCovarianceDiag(std::span<const double>, unsigned int) override { return false; }
    bool SetCovariance(std::span<const double>, unsigned int) override { return false; }

   private:
    /** Errors and covariance from Hessian `h` at X() (see Hesse()). */
    void set_errors(const std::vector<double>& h, bool exact);

    HessianFn           m_Hessian;
    double              m_Edm         = -1.0;
    unsigned int        m_NCalls      = 0;
    unsigned int        m_NIterations = 0;
    int                 m_CovStatus   = 0;
    std::vector<double> m_Gradient;
    std::vector<double> m_Errors;
    std::vector<double> m_Covariance;  // n x n, row-major; zero rows for fixed parameters
  };

}  // namespace ana
