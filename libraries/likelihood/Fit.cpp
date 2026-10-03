#include "Fit.h"

#include "NewtonMinimizer.h"
#include "ParameterSeeding.h"

// STL includes
#include <algorithm>
#include <cmath>
#include <random>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>

#include <Minuit2/FCNGradAdapter.h>
#include <Minuit2/Minuit2Minimizer.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ana {

  namespace {

    /**
     * Minuit2Minimizer with a way to attach a Hessian that does not go through
     * ROOT::Math::Minimizer::SetHessianFunction(). That one takes a
     * std::function over std::span, and a ROOT built in C++17 mode compiles
     * its library against its own std::__ROOT::span while this project, built
     * as C++23, sees the real std::span -- the same declaration naming two
     * different std::function types on the two sides of the library boundary,
     * which crashes on the first call. The FCN adapter's own setter stores a
     * std::function over std::vector, identical on both sides.
     */
    //
    // The same mismatch is why the three span-taking virtuals are overridden
    // here: a subclass's vtable would otherwise reference the base versions
    // under the C++23 std::span mangling, which libMinuit2 does not export.
    class Minuit2 final : public ROOT::Minuit2::Minuit2Minimizer {
     public:
      using ROOT::Minuit2::Minuit2Minimizer::Minuit2Minimizer;

      /** Attach `hessian` to the gradient adapter SetFunction() built. */
      template <class Hessian>
      void set_hessian(Hessian hessian) {
        using Adapter = ROOT::Minuit2::FCNGradAdapter<ROOT::Math::IMultiGradFunction>;
        auto* fcn     = dynamic_cast<const Adapter*>(GetFCN());
        if (fcn == nullptr)
          throw std::logic_error("Fit: the Hessian needs the minimizer's function to be a gradient function");
        // SetFunction() allocated the adapter itself; it is only handed out const.
        const_cast<Adapter*>(fcn)->SetHessianFunction(std::move(hessian));
      }

      void SetHessianFunction(std::function<bool(std::span<const double>, double*)> hessian) override {
        set_hessian([hessian = std::move(hessian)](const std::vector<double>& x, double* out) {
          return hessian(std::span<const double>(x.data(), x.size()), out);
        });
      }

      bool SetCovarianceDiag(std::span<const double>, unsigned int) override { return unsupported(); }
      bool SetCovariance(std::span<const double>, unsigned int) override { return unsupported(); }

     private:
      static bool unsupported() {
        std::cerr << "Fit: setting Minuit2's initial covariance is not supported in this build\n";
        return false;
      }
    };

  }  // namespace

  Fit::Fit(std::shared_ptr<io::Options> options, std::shared_ptr<ExperimentModule> module, int worker_index)
    : m_Options(std::move(options))
    , m_Module(std::move(module))
    , m_FitDuration(0)
    , m_Converged(false)
    , m_FitPerformed(false)
    , m_Tolerance(m_Options->inputOptions().tolerance()) {
#ifdef _OPENMP
    // Process-wide: OpenMP has no per-object thread pool, unlike the
    // std::async sample-level concurrency in ICLikelihood, which is gated
    // per-instance instead. -m/-1 (default) leaves the OpenMP/environment
    // default team size alone.
    if (m_Options->inputOptions().use_multi_threading())
      omp_set_num_threads(std::max(1, m_Options->inputOptions().multi_threading_cores()));
#endif

    const int         n_parameter = m_Module->number_of_parameters();
    const std::size_t n_config    = m_Options->inputOptions().input_parameters().size();

    if (static_cast<std::size_t>(n_parameter) != n_config) {
      throw std::invalid_argument("Experiment " + m_Module->name() + " expects " + std::to_string(n_parameter) + " parameters, but the config file provides " + std::to_string(n_config));
    }

    // Initialize the likelihood of the selected experiment. Each Fit builds its
    // own, so this is safe to run concurrently across scan workers -- unlike
    // the factory call in setup_minimizer(), it touches no shared ROOT state.
    m_Likelihood = m_Module->create_likelihood(m_Options, worker_index);

    // The function the minimizer sees. With an analytic gradient Migrad needs
    // about one evaluation per iteration for its derivatives instead of two per
    // free parameter, and its convergence is no longer limited by the rounding
    // of finite differences (Minuit2's "machine accuracy" stops).
    Likelihood* const likelihood = m_Likelihood.get();
    auto              value      = [this, likelihood](const double* x) {
      ++m_NCalls;
      return likelihood->calculate_likelihood(x);
    };
    m_UseGradient = m_Options->inputOptions().analytic_gradient() && likelihood->has_gradient();
    m_UseHessian  = m_UseGradient && m_Options->inputOptions().hessian() == "gn";
    if (m_UseGradient) {
      m_Functor = std::make_shared<ROOT::Math::GradFunctor>(value, static_cast<unsigned int>(n_parameter),
                                                            [this, likelihood](const double* x, double* gradient) {
                                                              ++m_NGradCalls;
                                                              likelihood->calculate_gradient(x, gradient);
                                                            });
    } else {
      m_Functor = std::make_shared<ROOT::Math::Functor>(value, static_cast<unsigned int>(n_parameter));
    }

    // Builds the minimizer and declares the parameters on it.
    setup_minimizer();
  }

  void Fit::setup_minimizer(const std::vector<double>* start_override, const std::vector<bool>* fixed_override) {
    const auto& input_options    = m_Options->inputOptions();
    const bool  silent           = input_options.silent();
    const bool  randomize        = input_options.randomize_seeds();
    const auto& input_parameters = input_options.input_parameters();

    // ROOT::Math::Factory::CreateMinimizer goes through ROOT's plugin manager,
    // which is not safe to enter from multiple threads at once. This is the
    // only shared ROOT state the whole setup touches, and it is reached both
    // when a Fit is built and on every restart, so the lock lives here rather
    // than around the constructor: scan workers build and restart their fits
    // concurrently, and everything else below is this Fit's own state.
    //
    // Built directly rather than through the factory: set_hessian() below needs
    // the concrete Minuit2 type, and the factory's plugin lookup ends in this
    // same constructor anyway. The lock is kept for whatever global state the
    // constructor touches (default options).
    std::shared_ptr<Minuit2>         minuit;
    std::shared_ptr<NewtonMinimizer> newton;
    if (input_options.minimizer() == "Newton") {
      if (!m_UseHessian)
        throw std::runtime_error("Fit: --minimizer Newton needs a likelihood with an analytic gradient and Hessian");
      newton      = std::make_shared<NewtonMinimizer>();
      m_Minimizer = newton;
    } else {
      static std::mutex mutex;
      std::unique_lock  lock{mutex};
      minuit      = std::make_shared<Minuit2>(input_options.minimizer_algo().c_str());
      m_Minimizer = minuit;
    }
    m_MinuitTransform = minuit != nullptr;

    // Set here rather than in minimize(): a restart builds a *fresh* minimizer,
    // and one left at the default level 0 makes Minuit2Minimizer::Minimize()
    // call TurnOffPrintInfoLevel(), which raises the process-wide
    // gErrorIgnoreLevel to 1001 for the whole call. Minuit2 prints through
    // ::Info(), so that silences every *other* scan worker's Minuit output
    // until the restart finishes -- and with two restarts overlapping, the
    // save/restore of that global races and can leave it muted for good.
    //
    // --blind mutes it the same way: level 2 makes Minuit2 print the final
    // parameter table, signal parameters included, which is exactly what the
    // blinding keeps out of the written result.
    m_Minimizer->SetPrintLevel(silent || input_options.blind() ? 0 : 2);

    m_Minimizer->SetFunction(*m_Functor);

    // The Gauss-Newton Hessian is what Migrad seeds its metric with (strategy
    // 2 takes the whole matrix, strategy 1 its diagonal) and what Hesse
    // returns, in place of the numerical second derivatives that cost
    // n(n+1)/2 evaluations each time. Only valid on a gradient function:
    // Minuit2 attaches it to the gradient adapter.
    if (m_UseHessian) {
      const std::size_t n_parameters = m_Options->inputOptions().input_parameters().size();
      auto              hessian      = [this, n_parameters](const double* x, double* out) {
        return m_ExactHessianMode ? exact_hessian(std::vector<double>(x, x + n_parameters), out)
                                  : m_Likelihood->calculate_hessian(x, out);
      };
      if (minuit)
        minuit->set_hessian([hessian](const std::vector<double>& x, double* out) { return hessian(x.data(), out); });
      else
        newton->set_hessian(hessian);
    }
    m_Minimizer->SetTolerance(m_Tolerance);
    // Strategy 2 is what makes Migrad seed its metric with the full Hessian
    // instead of its diagonal; with the Gauss-Newton Hessian that is one cheap
    // call, and on the IceCube tracks Asimov strategy 1 does not converge from
    // the diagonal alone. So --hessian gn raises the strategy to 2.
    m_Minimizer->SetStrategy(m_UseHessian ? std::max(2, input_options.minuit_strategy()) : input_options.minuit_strategy());

    // Minuit2's own default budget is 200 + 100*n + 5*n^2 calls, which a fit
    // with every nuisance free can exhaust before it is anywhere near the
    // minimum. Give it room; a fit that converges never reaches the cap.
    m_Minimizer->SetMaxFunctionCalls(100000);

    // A restart reports only what it does differently -- the parameter table
    // was already printed by the first attempt.
    const bool print_parameters = silent ? false : start_override == nullptr;

    const auto& names      = input_parameters.names();
    const auto& fixed      = input_parameters.fixed();
    const auto& parameters = input_parameters.parameters();

    // Randomized start values are drawn from the run's own --seed, so a draw is
    // reproducible (NNMFit's equivalent uses numpy's global RNG and is not).
    // The likelihood was already constructed above, so anything built from the
    // configured values -- the Asimov data in particular -- is unaffected;
    // only where the minimizer starts moves.
    std::mt19937_64 rng(static_cast<std::mt19937_64::result_type>(input_options.seed()));

    for (std::size_t i = 0; i < parameters.size(); ++i) {
      // A fixed parameter is a constraint, not a start point: randomizing it
      // would silently change the point being evaluated. NNMFit likewise
      // overwrites its fixed parameters after drawing the seeds.
      const bool randomize_this = randomize && !fixed[i];
      double     start          = randomize_this
                                      ? randomized_start_value(parameters[i].value(),
                                                               parameters[i].uncertainty(),
                                                               input_options.randomize_width(), rng)
                                      : parameters[i].value();

      // A restart resumes from where the previous attempt stopped, so its point
      // wins over both the configured and the randomized start value.
      if (start_override != nullptr)
        start = (*start_override)[i];

      const std::optional<double>& lower = parameters[i].lower_bound();
      const std::optional<double>& upper = parameters[i].upper_bound();

      // A parameter the caller fixed is a constraint, not a start point: its
      // value is the point being evaluated and must be reseeded exactly.
      const bool fixed_here = fixed_override != nullptr && (*fixed_override)[i];

      // Migrad can leave a parameter sitting exactly on a limit, where Minuit2's
      // internal arcsin transformation is singular -- reseeding a restart there
      // would hand the next attempt an immovable parameter.
      if (start_override != nullptr && !fixed_here && (lower || upper)) {
        const double margin = 1.0e-3 * parameters[i].uncertainty();
        if (lower) start = std::max(start, *lower + margin);
        if (upper) start = std::min(start, *upper - margin);
      }

      // A randomized draw can land outside the bounds -- the draw knows nothing
      // about them -- and Minuit2 rejects a seed outside its own limits. Pull it
      // back inside, a hair off the boundary: seeding exactly ON a limit makes
      // Minuit2's internal arcsin transformation singular. The configured start
      // value is never clamped; InputParameter rejects that at parse time.
      if (randomize_this && (lower || upper)) {
        const double margin = 1.0e-3 * parameters[i].uncertainty();
        if (lower) start = std::max(start, *lower + margin);
        if (upper) start = std::min(start, *upper - margin);
      }

      if (print_parameters && !input_options.blind()) {
        std::cout << "Set up parameter " << std::setw(5) << i << ": " << std::setw(18) << names[i]
                  << " with value " << std::setw(10) << start;
        if (randomize_this)
          std::cout << " (randomized from " << parameters[i].value() << ')';
        std::cout << " and uncertainty " << parameters[i].uncertainty();
        if (lower || upper) {
          std::cout << ", bounds [" << (lower ? std::to_string(*lower) : "-inf") << ", "
                    << (upper ? std::to_string(*upper) : "+inf") << ']';
        }
        std::cout << '\n';
      }

      // Minuit2 handles limits by an internal variable transformation, so a
      // bounded parameter is minimised in a different coordinate than an
      // unbounded one. Only the variants a parameter actually needs are used:
      // declaring a huge artificial limit instead of leaving a side open would
      // apply that transformation for nothing.
      const double step = i < m_StepSizes.size() && std::isfinite(m_StepSizes[i]) && m_StepSizes[i] > 0.0
                              ? m_StepSizes[i]
                              : parameters[i].uncertainty();
      if (lower && upper)
        m_Minimizer->SetLimitedVariable(static_cast<unsigned int>(i), names[i], start, step, *lower, *upper);
      else if (lower)
        m_Minimizer->SetLowerLimitedVariable(static_cast<unsigned int>(i), names[i], start, step, *lower);
      else if (upper)
        m_Minimizer->SetUpperLimitedVariable(static_cast<unsigned int>(i), names[i], start, step, *upper);
      else
        m_Minimizer->SetVariable(static_cast<unsigned int>(i), names[i], start, step);
    }

    if (print_parameters) {
      std::cout << "-----\n";
    }

    for (std::size_t i = 0; i < parameters.size(); ++i) {
      // What the caller fixed on the previous minimizer outranks the module's
      // opinion: a scan point fixes its scanned parameters exactly this way,
      // and freeing one here would turn that point into a free fit.
      if (fixed_override != nullptr && (*fixed_override)[i]) {
        m_Minimizer->FixVariable(static_cast<unsigned int>(i));
        continue;
      }

      if (m_Module->keep_parameter_free(i))
        continue;

      if (fixed[i]) {
        if (print_parameters && !input_options.blind()) {
          std::cout << "Fixing parameter " << std::setw(5) << i << " " << names[i] << '\n';
        }
        m_Minimizer->FixVariable(static_cast<unsigned int>(i));
      }
    }
  }

  bool Fit::minimize() {
    using namespace std::chrono;

    const auto begin = high_resolution_clock::now();
    m_Converged      = m_Minimizer->Minimize();

    // Migrad still gives up on a minority of points once the call budget above
    // is generous: it stalls in a line search, well short of the minimum and
    // holding a covariance it cannot recover from. Calling Minimize() again on
    // the same minimizer inherits that state and barely helps; building a fresh
    // one seeded where the last attempt stopped does.
    //
    // Measured on nine hard points of the IceCube tracks grid (the ones a cold
    // start gets wrong), against NNMFit's own values for the same points:
    // without restarts 7/9 converge and the sum of the differences is +0.139;
    // with them 9/9 converge at -0.011, for 4% more wall time -- the restarts
    // only ever run on the points that failed. A converged fit is left alone.
    //
    // Two Minuit2 knobs were tried instead and rejected: strategy 2 was worse
    // at 4 of 9 points and 17% slower, and "Combined" (Migrad -> Simplex ->
    // Migrad) was byte-identical to plain Migrad everywhere, including where it
    // fails. Both remain reachable via --minuitStrategy / --minimizerAlgo.
    const int          retries      = m_Options->inputOptions().fit_retries();
    const std::size_t  n_parameters = m_Options->inputOptions().input_parameters().size();
    for (int attempt = 1; !m_Converged && attempt <= retries; ++attempt) {
      const std::vector<double> resume(m_Minimizer->X(), m_Minimizer->X() + n_parameters);

      // Which variables are fixed is read off the minimizer rather than the
      // config: the scans fix their scanned parameters on it directly, and a
      // rebuild from the config alone would free them.
      std::vector<bool> was_fixed(n_parameters);
      for (std::size_t i = 0; i < n_parameters; ++i)
        was_fixed[i] = m_Minimizer->IsFixedVariable(static_cast<unsigned int>(i));

      if (!m_Options->inputOptions().silent())
        std::cout << "Migrad did not converge (EDM " << m_Minimizer->Edm() << "); restart " << attempt << " of " << retries
                  << " from its last point\n";

      setup_minimizer(&resume, &was_fixed);
      m_Converged = m_Minimizer->Minimize();
    }

    // Migrad ran on the Gauss-Newton Hessian; the reported errors come from the
    // exact one. Hesse() reaches it through the same callback.
    if (m_UseHessian && m_ExactErrors) {
      m_ExactHessianMode = true;
      m_Minimizer->Hesse();
      m_ExactHessianMode = false;
    }

    const auto end   = high_resolution_clock::now();

    m_FitDuration = end - begin;

    std::stringstream ss;
    ss << "Fit finished: " << std::boolalpha << m_Converged << '\n';
    ss << "It took: " << m_FitDuration.count() << " seconds\n";
    ss << "Likelihood: " << m_Minimizer->MinValue() << '\n';
    ss << "EDM: " << m_Minimizer->Edm() << '\n';
    ss << "Evaluations: " << m_NCalls;
    if (m_UseGradient)
      ss << " (+ " << m_NGradCalls << " analytic gradients" << (m_UseHessian ? ", Gauss-Newton Hessian" : "") << ')';
    ss << '\n';

    std::cout << ss.rdbuf() << std::endl;

    m_FitPerformed = true;

    return m_Converged;
  }

  void Fit::set_step_sizes(const std::span<const double> steps) {
    m_StepSizes.assign(steps.begin(), steps.end());
    for (std::size_t i = 0; i < m_StepSizes.size(); ++i)
      if (std::isfinite(m_StepSizes[i]) && m_StepSizes[i] > 0.0)
        m_Minimizer->SetVariableStepSize(static_cast<unsigned int>(i), m_StepSizes[i]);
  }

  void Fit::set_tolerance(const double tolerance) {
    m_Tolerance = tolerance;
    m_Minimizer->SetTolerance(tolerance);
  }

  bool Fit::exact_hessian(const std::vector<double>& x, double* hessian) {
    const std::size_t n          = x.size();
    const auto&       parameters = m_Options->inputOptions().input_parameters().parameters();
    const double*     errors     = m_Minimizer->Errors();

    std::fill(hessian, hessian + n * n, 0.0);
    std::vector<double> up(n), down(n), point(x);

    for (std::size_t i = 0; i < n; ++i) {
      if (m_Minimizer->IsFixedVariable(static_cast<unsigned int>(i)))
        continue;

      // A thousandth of the parameter's current error: far below where the
      // likelihood stops being quadratic, far above the gradient's rounding.
      const double error = errors != nullptr && std::isfinite(errors[i]) && errors[i] > 0.0 ? errors[i] : parameters[i].uncertainty();
      const double h     = 1.0e-3 * error;

      // One-sided next to a bound the step would cross.
      const auto& lower    = parameters[i].lower_bound();
      const auto& upper    = parameters[i].upper_bound();
      const bool  can_up   = !upper || x[i] + h <= *upper;
      const bool  can_down = !lower || x[i] - h >= *lower;

      point[i] = can_up ? x[i] + h : x[i];
      m_Likelihood->calculate_gradient(point.data(), up.data());
      point[i] = can_down ? x[i] - h : x[i];
      m_Likelihood->calculate_gradient(point.data(), down.data());
      point[i] = x[i];

      const double span = (can_up ? h : 0.0) + (can_down ? h : 0.0);
      if (span == 0.0)
        continue;
      for (std::size_t j = 0; j < n; ++j)
        hessian[i * n + j] = (up[j] - down[j]) / span;
      m_NGradCalls += 2;
    }

    // Symmetrize: the two differences of each pair agree to O(h^2).
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < i; ++j) {
        const double mean   = 0.5 * (hessian[i * n + j] + hessian[j * n + i]);
        hessian[i * n + j] = hessian[j * n + i] = mean;
      }

    if (!m_MinuitTransform)
      return true;

    // Minuit2 works in internal coordinates, int -> ext(int) for a bounded
    // parameter, and its numerical Hesse measures d^2 f / d int^2 =
    // f'' ext'^2 + f' ext''. When it is handed an external Hessian it only
    // applies the first term (MnHesse's analytical path transforms with ext'
    // alone), so the second is folded in here, as f' ext'' / ext'^2 on the
    // diagonal. For a parameter well inside its bounds it is negligible; for
    // one at its bound -- where f' does not vanish -- it is what Minuit's
    // numerical errors have always contained, and leaving it out would change
    // what the reported errors mean.
    std::vector<double> gradient(n);
    m_Likelihood->calculate_gradient(x.data(), gradient.data());
    ++m_NGradCalls;
    for (std::size_t i = 0; i < n; ++i) {
      if (m_Minimizer->IsFixedVariable(static_cast<unsigned int>(i)))
        continue;
      const auto& lower = parameters[i].lower_bound();
      const auto& upper = parameters[i].upper_bound();
      if (!lower && !upper)
        continue;

      // Minuit2's transformations (Sin-, SqrtLow-, SqrtUpParameterTransformation):
      // the internal value of x[i] and the first two derivatives of ext(int).
      double d1 = 1.0, d2 = 0.0;
      if (lower && upper) {
        const double half = 0.5 * (*upper - *lower);
        const double yy   = std::clamp((x[i] - *lower) / half - 1.0, -1.0, 1.0);
        const double in   = std::asin(yy);
        d1                = half * std::cos(in);
        d2                = -half * std::sin(in);
      } else {
        const double yy = lower ? x[i] - *lower + 1.0 : *upper - x[i] + 1.0;
        const double in = yy * yy > 1.0 ? std::sqrt(yy * yy - 1.0) : 0.0;
        const double r  = std::sqrt(in * in + 1.0);
        d1              = (lower ? 1.0 : -1.0) * in / r;
        d2              = (lower ? 1.0 : -1.0) / (r * r * r);
      }
      if (std::fabs(d1) < 1e-150)
        continue;
      hessian[i * n + i] += gradient[i] * d2 / (d1 * d1);
    }
    return true;
  }

  double Fit::time_duration() const {
    if (!m_FitPerformed)
      std::cout << "Fit not performed yet\n";

    return m_FitDuration.count();
  }

  bool Fit::converged() const {
    if (!m_FitPerformed)
      std::cout << "Fit not performed yet\n";

    return m_Converged;
  }

  const std::shared_ptr<io::Options>& Fit::options() const {
    return m_Options;
  }

}  // namespace ana
