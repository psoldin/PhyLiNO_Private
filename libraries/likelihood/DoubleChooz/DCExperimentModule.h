#pragma once

#include "../ExperimentModule.h"
#include "DCLikelihood.h"
#include "DoubleChooz/DCInputOptions.h"

namespace ana::dc {

  /**
   * @brief Experiment module of the Double Chooz experiment.
   *
   * Owns the Double Chooz input options and creates the DCLikelihood together with the
   * DCOptions data base when Double Chooz is the selected experiment.
   */
  class DCExperimentModule : public ExperimentModule {
   public:
    DCExperimentModule()
      : m_InputOptions(std::make_shared<io::dc::DCInputOptions>()) {}

    ~DCExperimentModule() override = default;

    [[nodiscard]] std::string name() const override { return "DoubleChooz"; }

    [[nodiscard]] std::shared_ptr<io::InputOptionBase> input_options() override { return m_InputOptions; }

    [[nodiscard]] int number_of_parameters() const override { return params::number_of_parameters(); }

    [[nodiscard]] std::shared_ptr<Likelihood> create_likelihood(std::shared_ptr<io::Options> options) override;

    [[nodiscard]] bool keep_parameter_free(const io::Options& options, std::size_t i) const override {
      static_cast<void>(options);
      // With the sterile hypothesis enabled the sterile oscillation parameters stay free even if
      // the config marks them as fixed.
      return m_InputOptions->use_sterile() && (i == params::DeltaM41 || i == params::SinSqT14);
    }

    [[nodiscard]] const std::shared_ptr<DCLikelihood>& likelihood() const noexcept { return m_Likelihood; }

    [[nodiscard]] const io::dc::DCInputOptions& dc_input_options() const noexcept { return *m_InputOptions; }

   private:
    std::shared_ptr<io::dc::DCInputOptions> m_InputOptions;
    std::shared_ptr<DCLikelihood>           m_Likelihood;
  };

}  // namespace ana::dc
