#pragma once

#include "ExperimentModule.h"
#include "Likelihood.h"
#include "Options.h"

// STL includes
#include <chrono>
#include <memory>
#include <vector>

// ROOT includes
#include <Math/Factory.h>
#include <Math/Functor.h>
#include <Math/Minimizer.h>

namespace ana {

  class Fit {
   public:
    Fit(std::shared_ptr<io::Options> options, std::shared_ptr<ExperimentModule> module);

    ~Fit() = default;

    [[nodiscard]] const std::shared_ptr<Likelihood>& likelihood() const noexcept { return m_Likelihood; }

    [[nodiscard]] const std::shared_ptr<ExperimentModule>& module() const noexcept { return m_Module; }

    bool minimize();

    [[nodiscard]] double time_duration() const;

    [[nodiscard]] bool converged() const;

    [[nodiscard]] const std::shared_ptr<io::Options>& options() const;

    auto get_minimizer() const { return m_Minimizer; }

   private:
    std::shared_ptr<io::Options>      m_Options;
    std::shared_ptr<ExperimentModule> m_Module;

    std::chrono::duration<double, std::ratio<1>> m_FitDuration;

    bool m_Converged;
    bool m_FitPerformed;

    std::shared_ptr<ROOT::Math::Minimizer> m_Minimizer;

    std::shared_ptr<ROOT::Math::Functor> m_Functor;

    std::shared_ptr<Likelihood> m_Likelihood;

    /**
     * Build m_Minimizer and declare every parameter on it.
     *
     * `start_override`, when given, replaces the configured start values --
     * that is how a restart resumes from where the previous Migrad stopped.
     * The minimizer object itself is always new: reusing one that stalled
     * carries its broken covariance into the next attempt, which measurably
     * ends worse than starting a fresh one at the same point.
     */
    void setup_minimizer(const std::vector<double>* start_override = nullptr);
  };

}  // namespace ana
