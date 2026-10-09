#pragma once

#include <Math/BasicMinimizer.h>
#include <RVersion.h>

#include <cmath>
#include <functional>
#include <span>
#include <vector>

namespace ana {

  /**
   * A minimizer that does not minimize: Minimize() evaluates the function once
   * at the variables' current values and leaves them there (--evaluateAt).
   *
   * Behind Fit::evaluate(), which fixes every parameter at the point it is
   * handed. Minuit2 cannot stand in for this: with no free variable its seed
   * generator crashes instead of returning the function value. Being a
   * ROOT::Math::Minimizer, it lets the result writers read X(), MinValue() and
   * Errors() off it exactly as they do after a fit.
   *
   * Errors are zero and EDM is zero: nothing was fitted, so there is no
   * curvature and no distance to a minimum to report.
   */
  class PointEvaluator final : public ROOT::Math::BasicMinimizer {
   public:
    PointEvaluator() = default;

    bool Minimize() override {
      const ROOT::Math::IMultiGenFunction* function = ObjFunction();
      if (function == nullptr) {
        fStatus = 5;
        return false;
      }
      SetMinValue((*function)(X()));
      m_Errors.assign(NDim(), 0.0);
      m_NCalls = 1;
      fStatus  = std::isfinite(MinValue()) ? 0 : 1;
      return fStatus == 0;
    }

    [[nodiscard]] double        Edm() const override { return 0.0; }
    [[nodiscard]] unsigned int  NCalls() const override { return m_NCalls; }
    [[nodiscard]] bool          ProvidesError() const override { return true; }
    [[nodiscard]] const double* Errors() const override { return m_Errors.empty() ? nullptr : m_Errors.data(); }
    [[nodiscard]] double        CovMatrix(unsigned int, unsigned int) const override { return 0.0; }

    // ROOT::Math::Minimizer's span-taking virtuals, overridden for the same
    // reason as in NewtonMinimizer.h: the base versions may be exported under a
    // different std::span than this build sees. There is no Hessian to take.
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 34, 0)
    void SetHessianFunction(std::function<bool(std::span<const double>, double*)>) override {}
    bool SetCovarianceDiag(std::span<const double>, unsigned int) override { return false; }
    bool SetCovariance(std::span<const double>, unsigned int) override { return false; }
#else
    void SetHessianFunction(std::function<bool(const std::vector<double>&, double*)>) override {}
#endif

   private:
    unsigned int        m_NCalls = 0;
    std::vector<double> m_Errors;
  };

}  // namespace ana
