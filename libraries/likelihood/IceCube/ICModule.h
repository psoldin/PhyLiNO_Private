#pragma once

#include "../ExperimentModule.h"
#include "ICLikelihood.h"

#include "IceCube/ICInputOptions.h"
#include "IceCube/ICParameter.h"

namespace ana::ic {

  /**
   * @brief Experiment module of the IceCube tracks-only diffuse-flux analysis.
   *
   * Owns the IceCube input options and, when IceCube is the selected experiment,
   * loads the MC baseline parquet (ICDataBase) and creates the ICLikelihood.
   */
  class ICExperimentModule : public ExperimentModule {
   public:
    ICExperimentModule()
      : m_InputOptions(std::make_shared<io::ic::ICInputOptions>()) {}

    ~ICExperimentModule() override = default;

    [[nodiscard]] std::string name() const override { return "IceCube"; }

    [[nodiscard]] std::shared_ptr<io::InputOptionBase> input_options() override { return m_InputOptions; }

    [[nodiscard]] int number_of_parameters() const override { return params::ic::number_of_parameters(); }

    [[nodiscard]] std::shared_ptr<Likelihood> create_likelihood(std::shared_ptr<io::Options> options) override;

    void write_results(Fit& fit, std::string_view name) override;

    [[nodiscard]] const std::shared_ptr<ICLikelihood>& likelihood() const noexcept { return m_Likelihood; }

    [[nodiscard]] const io::ic::ICInputOptions& ic_input_options() const noexcept { return *m_InputOptions; }

   private:
    std::shared_ptr<io::ic::ICInputOptions> m_InputOptions;
    std::shared_ptr<ICLikelihood>           m_Likelihood;
  };

}  // namespace ana::ic
