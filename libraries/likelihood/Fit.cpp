#include "Fit.h"

#include "ParameterSeeding.h"

// STL includes
#include <algorithm>
#include <random>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ana {

  Fit::Fit(std::shared_ptr<io::Options> options, std::shared_ptr<ExperimentModule> module)
    : m_Options(std::move(options))
    , m_Module(std::move(module))
    , m_FitDuration(0)
    , m_Converged(false)
    , m_FitPerformed(false) {
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
    // the factory call below, it touches no shared ROOT state.
    m_Likelihood = m_Module->create_likelihood(m_Options);

    // ROOT::Math::Factory::CreateMinimizer goes through ROOT's plugin manager,
    // which is not safe to enter from multiple threads at once; that is the
    // only part of this constructor that needs serializing; everything else
    // here operates on this Fit's own state and may run in parallel with other
    // scan workers building their Fits at the same time.
    {
      static std::mutex mutex;
      std::unique_lock  lock{mutex};
      m_Minimizer = std::shared_ptr<ROOT::Math::Minimizer>(ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad"));
    }

    // Initialize the functor object
    m_Functor = std::make_shared<ROOT::Math::Functor>(m_Likelihood.get(),
                                                      &Likelihood::calculate_likelihood,
                                                      n_parameter);

    // Builds the minimizer and declares the parameters on it.
    setup_minimizer();
  }

  void Fit::setup_minimizer(const std::vector<double>* start_override, const std::vector<bool>* fixed_override) {
    const auto& input_options    = m_Options->inputOptions();
    const bool  silent           = input_options.silent();
    const bool  randomize        = input_options.randomize_seeds();
    const auto& input_parameters = input_options.input_parameters();

    m_Minimizer = std::shared_ptr<ROOT::Math::Minimizer>(
        ROOT::Math::Factory::CreateMinimizer("Minuit2", input_options.minimizer_algo()));
    if (!m_Minimizer)
      throw std::runtime_error("Fit: ROOT has no Minuit2 minimizer called '" + input_options.minimizer_algo() + "'");

    m_Minimizer->SetFunction(*m_Functor);
    m_Minimizer->SetTolerance(input_options.tolerance());
    m_Minimizer->SetStrategy(input_options.minuit_strategy());

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

      if (print_parameters) {
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
      const double step = parameters[i].uncertainty();
      if (lower && upper)
        m_Minimizer->SetLimitedVariable(i, names[i], start, step, *lower, *upper);
      else if (lower)
        m_Minimizer->SetLowerLimitedVariable(i, names[i], start, step, *lower);
      else if (upper)
        m_Minimizer->SetUpperLimitedVariable(i, names[i], start, step, *upper);
      else
        m_Minimizer->SetVariable(i, names[i], start, step);
    }

    if (print_parameters) {
      std::cout << "-----\n";
    }

    for (std::size_t i = 0; i < parameters.size(); ++i) {
      // What the caller fixed on the previous minimizer outranks the module's
      // opinion: a scan point fixes its scanned parameters exactly this way,
      // and freeing one here would turn that point into a free fit.
      if (fixed_override != nullptr && (*fixed_override)[i]) {
        m_Minimizer->FixVariable(i);
        continue;
      }

      if (m_Module->keep_parameter_free(i))
        continue;

      if (fixed[i]) {
        if (print_parameters) {
          std::cout << "Fixing parameter " << std::setw(5) << i << " " << names[i] << '\n';
        }
        m_Minimizer->FixVariable(i);
      }
    }
  }

  bool Fit::minimize() {
    using namespace std::chrono;

    m_Minimizer->SetPrintLevel(m_Options->inputOptions().silent() ? 0 : 2);

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

    const auto end   = high_resolution_clock::now();

    m_FitDuration = end - begin;

    std::stringstream ss;
    ss << "Fit finished: " << std::boolalpha << m_Converged << '\n';
    ss << "It took: " << m_FitDuration.count() << " seconds\n";
    ss << "Likelihood: " << m_Minimizer->MinValue() << '\n';
    ss << "EDM: " << m_Minimizer->Edm() << '\n';

    std::cout << ss.rdbuf() << std::endl;

    m_FitPerformed = true;

    return m_Converged;
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
