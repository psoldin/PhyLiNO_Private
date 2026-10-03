#pragma once

#include "ExperimentModule.h"
#include "Likelihood.h"
#include "Options.h"

// STL includes
#include <chrono>
#include <memory>
#include <span>
#include <vector>

// ROOT includes
#include <Math/Factory.h>
#include <Math/Functor.h>
#include <Math/Minimizer.h>

namespace ana {

  class Fit {
   public:
    // worker_index identifies the scan worker thread building this Fit (see
    // ExperimentModule::create_likelihood); 0 for a lone --fitOnly fit.
    Fit(std::shared_ptr<io::Options> options, std::shared_ptr<ExperimentModule> module, int worker_index = 0);

    ~Fit() = default;

    [[nodiscard]] const std::shared_ptr<Likelihood>& likelihood() const noexcept { return m_Likelihood; }

    [[nodiscard]] const std::shared_ptr<ExperimentModule>& module() const noexcept { return m_Module; }

    bool minimize();

    /**
     * Override the --tolerance this fit converges to. The scans set their
     * per-point fits to --scanTolerance this way. Kept on the Fit rather than
     * set on the minimizer alone because a restart builds a fresh minimizer,
     * which would otherwise fall back to --tolerance.
     */
    void set_tolerance(double tolerance);

    /**
     * Whether minimize() ends with a Hesse on the exact Hessian (default
     * true). Only matters with --hessian gn: Migrad then runs on the
     * Gauss-Newton Hessian, which is the right metric but not the right
     * curvature for the reported errors, so the errors are recomputed from
     * central differences of the analytic gradient -- 2 gradients per free
     * parameter. Fits whose errors nobody reads (scan points) switch it off.
     */
    void set_exact_errors(bool exact) noexcept { m_ExactErrors = exact; }

    /**
     * Step sizes for the free parameters in place of the config's StepWidths,
     * one per parameter in minimizer order; entries that are not positive and
     * finite keep the configured width. Kept across restarts, which rebuild
     * the minimizer from the configuration otherwise.
     */
    void set_step_sizes(std::span<const double> steps);

    [[nodiscard]] double time_duration() const;

    [[nodiscard]] bool converged() const;

    [[nodiscard]] const std::shared_ptr<io::Options>& options() const;

    auto get_minimizer() const { return m_Minimizer; }

    /** Likelihood evaluations this fit made, restarts included. */
    [[nodiscard]] std::size_t n_calls() const noexcept { return m_NCalls; }

    /** Analytic gradient evaluations this fit made (0 without --gradient). */
    [[nodiscard]] std::size_t n_gradient_calls() const noexcept { return m_NGradCalls; }

   private:
    std::shared_ptr<io::Options>      m_Options;
    std::shared_ptr<ExperimentModule> m_Module;

    std::chrono::duration<double, std::ratio<1>> m_FitDuration;

    bool m_Converged;
    bool m_FitPerformed;

    double m_Tolerance;

    std::vector<double> m_StepSizes;  ///< Overrides of the configured StepWidths (see set_step_sizes()).

    std::shared_ptr<ROOT::Math::Minimizer> m_Minimizer;

    // A GradFunctor when the analytic gradient is used, a plain Functor otherwise.
    std::shared_ptr<ROOT::Math::IMultiGenFunction> m_Functor;

    bool m_UseGradient = false;  ///< --gradient and the likelihood provides one.
    bool m_UseHessian  = false;  ///< --hessian gn on top of the gradient.
    bool m_ExactErrors = true;   ///< Final Hesse on the exact Hessian (see set_exact_errors()).
    bool m_ExactHessianMode = false;  ///< The Hessian callback differences the gradient instead of returning Gauss-Newton.
    bool m_MinuitTransform  = true;   ///< The minimizer works in Minuit2's internal coordinates (see exact_hessian()).

    /** Central differences of the analytic gradient over the free parameters, row-major n x n. */
    bool exact_hessian(const std::vector<double>& x, double* hessian);

    // Likelihood and gradient evaluations of this fit, restarts included.
    std::size_t m_NCalls     = 0;
    std::size_t m_NGradCalls = 0;

    std::shared_ptr<Likelihood> m_Likelihood;

    /**
     * Build m_Minimizer and declare every parameter on it.
     *
     * `start_override`, when given, replaces the configured start values --
     * that is how a restart resumes from where the previous Migrad stopped.
     * The minimizer object itself is always new: reusing one that stalled
     * carries its broken covariance into the next attempt, which measurably
     * ends worse than starting a fresh one at the same point.
     *
     * `fixed_override` carries over which variables were fixed on the previous
     * minimizer, which is not the same set the config declares: the scans fix
     * the two scanned parameters on the minimizer itself, and a rebuild that
     * only replayed the config would quietly free them and turn a scan point
     * into a free fit.
     */
    void setup_minimizer(const std::vector<double>* start_override = nullptr,
                         const std::vector<bool>*   fixed_override = nullptr);
  };

}  // namespace ana
