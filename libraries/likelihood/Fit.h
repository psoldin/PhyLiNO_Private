#pragma once

#include "ExperimentModule.h"
#include "Likelihood.h"
#include "Options.h"

// STL includes
#include <chrono>
#include <memory>

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

    void setup_minimizer();
  };

}  // namespace ana
