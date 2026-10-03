// Tests of ana::NewtonMinimizer, the projected trust-region Newton behind
// --minimizer Newton, on functions whose minima and curvatures are known.

#include "NewtonMinimizer.h"

#include <Math/Functor.h>
#include <Minuit2/Minuit2Minimizer.h>

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <vector>

namespace {

  /// Sets a function, its gradient and Hessian on a fresh minimizer.
  struct Problem {
    std::function<double(const double*)>         f;
    std::function<void(const double*, double*)>  g;
    std::function<bool(const double*, double*)>  h;
    unsigned int                                 n;
    std::unique_ptr<ROOT::Math::GradFunctor>     functor;

    ana::NewtonMinimizer& attach(ana::NewtonMinimizer& m) {
      functor = std::make_unique<ROOT::Math::GradFunctor>(f, n, g);
      m.SetFunction(*functor);
      m.set_hessian(h);
      m.SetTolerance(1e-6);
      m.SetMaxFunctionCalls(10000);
      return m;
    }
  };

  /// (x - a)^T A (x - a) with a correlated A: one Newton step.
  Problem correlated_quadratic() {
    static const double a[2]    = {1.0, -2.0};
    static const double A[2][2] = {{4.0, 1.5}, {1.5, 1.0}};
    Problem             p;
    p.n = 2;
    p.f = [](const double* x) {
      const double d0 = x[0] - a[0], d1 = x[1] - a[1];
      return A[0][0] * d0 * d0 + 2 * A[0][1] * d0 * d1 + A[1][1] * d1 * d1;
    };
    p.g = [](const double* x, double* g) {
      const double d0 = x[0] - a[0], d1 = x[1] - a[1];
      g[0]            = 2 * (A[0][0] * d0 + A[0][1] * d1);
      g[1]            = 2 * (A[1][0] * d0 + A[1][1] * d1);
    };
    p.h = [](const double*, double* h) {
      h[0] = 2 * A[0][0];
      h[1] = h[2] = 2 * A[0][1];
      h[3]        = 2 * A[1][1];
      return true;
    };
    return p;
  }

}  // namespace

TEST(NewtonMinimizerTest, SolvesAQuadraticInOneStepWithExactErrors) {
  Problem              p = correlated_quadratic();
  ana::NewtonMinimizer m;
  p.attach(m);
  m.SetVariable(0, "x", 5.0, 0.1);
  m.SetVariable(1, "y", 5.0, 0.1);
  ASSERT_TRUE(m.Minimize());
  EXPECT_NEAR(m.X()[0], 1.0, 1e-10);
  EXPECT_NEAR(m.X()[1], -2.0, 1e-10);
  EXPECT_LE(m.NIterations(), 2u);
  ASSERT_TRUE(m.Hesse());
  // Covariance = 2 * ErrorDef * H^-1 = A^-1 for this f (ErrorDef 1).
  const double det = 4.0 * 1.0 - 1.5 * 1.5;
  EXPECT_NEAR(m.Errors()[0], std::sqrt(1.0 / det), 1e-12);
  EXPECT_NEAR(m.Errors()[1], std::sqrt(4.0 / det), 1e-12);
  EXPECT_NEAR(m.CovMatrix(0, 1), -1.5 / det, 1e-12);
}

TEST(NewtonMinimizerTest, Rosenbrock) {
  Problem p;
  p.n = 2;
  p.f = [](const double* x) { return 100 * std::pow(x[1] - x[0] * x[0], 2) + std::pow(1 - x[0], 2); };
  p.g = [](const double* x, double* g) {
    g[0] = -400 * x[0] * (x[1] - x[0] * x[0]) - 2 * (1 - x[0]);
    g[1] = 200 * (x[1] - x[0] * x[0]);
  };
  p.h = [](const double* x, double* h) {
    h[0] = 1200 * x[0] * x[0] - 400 * x[1] + 2;
    h[1] = h[2] = -400 * x[0];
    h[3]        = 200;
    return true;
  };
  ana::NewtonMinimizer m;
  p.attach(m);
  m.SetVariable(0, "x", -1.2, 0.1);
  m.SetVariable(1, "y", 1.0, 0.1);
  ASSERT_TRUE(m.Minimize());
  EXPECT_NEAR(m.X()[0], 1.0, 1e-5);
  EXPECT_NEAR(m.X()[1], 1.0, 1e-5);
}

/// A minimum outside the box ends on the bound, held there by the active set,
/// and still counts as converged -- the case a norm at 0 is.
TEST(NewtonMinimizerTest, HoldsAParameterOnItsBound) {
  Problem              p = correlated_quadratic();
  ana::NewtonMinimizer m;
  p.attach(m);
  m.SetLowerLimitedVariable(0, "x", 3.0, 0.1, 2.0);  // the minimum wants x = 1
  m.SetVariable(1, "y", 0.0, 0.1);
  ASSERT_TRUE(m.Minimize());
  EXPECT_DOUBLE_EQ(m.X()[0], 2.0);
  // y minimises the quadratic at x = 2: d f / d y = 0 -> y = a1 - A01 / A11 (x - a0).
  EXPECT_NEAR(m.X()[1], -2.0 - 1.5 * (2.0 - 1.0), 1e-9);
}

TEST(NewtonMinimizerTest, LeavesAFixedParameterAlone) {
  Problem              p = correlated_quadratic();
  ana::NewtonMinimizer m;
  p.attach(m);
  m.SetVariable(0, "x", 0.5, 0.1);
  m.SetVariable(1, "y", 0.0, 0.1);
  m.FixVariable(0);
  ASSERT_TRUE(m.Minimize());
  EXPECT_DOUBLE_EQ(m.X()[0], 0.5);
  EXPECT_NEAR(m.X()[1], -2.0 - 1.5 * (0.5 - 1.0), 1e-9);
  ASSERT_TRUE(m.Hesse());
  EXPECT_EQ(m.Errors()[0], 0.0);
}

/// A binned Poisson fit of a falling spectrum with its Gauss-Newton Hessian:
/// same minimum as Minuit2's Migrad.
TEST(NewtonMinimizerTest, AgreesWithMigradOnAPoissonFit) {
  const int           bins = 20;
  std::vector<double> data(bins);
  auto mu = [](const double* x, int b) { return x[0] * std::exp(-x[1] * (b + 0.5) / 20.0); };
  const double truth[2] = {500.0, 2.0};
  for (int b = 0; b < bins; ++b) data[b] = std::round(mu(truth, b) * (1.0 + 0.05 * std::sin(3.0 * b)));

  Problem p;
  p.n = 2;
  p.f = [&](const double* x) {
    double s = 0;
    for (int b = 0; b < bins; ++b) s += 2 * (mu(x, b) - data[b] * std::log(mu(x, b)));
    return s;
  };
  auto jac = [&](const double* x, int b, double* j) {
    const double t = (b + 0.5) / 20.0, e = std::exp(-x[1] * t);
    j[0]           = e;
    j[1]           = -x[0] * t * e;
  };
  p.g = [&](const double* x, double* g) {
    g[0] = g[1] = 0;
    for (int b = 0; b < bins; ++b) {
      double j[2];
      jac(x, b, j);
      const double r = 2 * (1 - data[b] / mu(x, b));
      g[0] += r * j[0];
      g[1] += r * j[1];
    }
  };
  p.h = [&](const double* x, double* h) {
    h[0] = h[1] = h[2] = h[3] = 0;
    for (int b = 0; b < bins; ++b) {
      double j[2];
      jac(x, b, j);
      const double w = 2 * data[b] / (mu(x, b) * mu(x, b));  // Gauss-Newton weight
      h[0] += w * j[0] * j[0];
      h[1] += w * j[0] * j[1];
      h[3] += w * j[1] * j[1];
    }
    h[2] = h[1];
    return true;
  };

  ana::NewtonMinimizer newton;
  p.attach(newton);
  newton.SetLowerLimitedVariable(0, "norm", 300.0, 10.0, 0.0);
  newton.SetVariable(1, "slope", 1.0, 0.1);
  ASSERT_TRUE(newton.Minimize());

  ROOT::Minuit2::Minuit2Minimizer migrad("Migrad");
  ROOT::Math::Functor             f(p.f, 2);
  migrad.SetFunction(f);
  migrad.SetPrintLevel(0);
  migrad.SetTolerance(1e-6);
  migrad.SetLowerLimitedVariable(0, "norm", 300.0, 10.0, 0.0);
  migrad.SetVariable(1, "slope", 1.0, 0.1);
  ASSERT_TRUE(migrad.Minimize());

  EXPECT_NEAR(newton.X()[0], migrad.X()[0], 1e-4 * migrad.Errors()[0]);
  EXPECT_NEAR(newton.X()[1], migrad.X()[1], 1e-4 * migrad.Errors()[1]);
  EXPECT_LE(newton.NCalls(), 20u);
}
