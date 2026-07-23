#pragma once

#include "../ExperimentModule.h"
#include "LinRegLikelihood.h"

#include "LinearRegression/LinRegParameter.h"

namespace ana::linreg {

  /**
   * @brief Experiment module of the linear-regression example.
   *
   * The minimal template for adding an experiment to the framework: provide input options, the
   * number of parameters, a likelihood factory, and a result writer.
   */
  class LinRegModule : public ExperimentModule {
   public:
    LinRegModule()
      : m_InputOptions(std::make_shared<io::linreg::LinRegInputOptions>()) {}

    ~LinRegModule() override = default;

    [[nodiscard]] std::string name() const override { return "LinearRegression"; }

    [[nodiscard]] std::shared_ptr<io::InputOptionBase> input_options() override { return m_InputOptions; }

    [[nodiscard]] int number_of_parameters() const override { return params::linreg::number_of_parameters(); }

    [[nodiscard]] std::shared_ptr<Likelihood> create_likelihood(std::shared_ptr<io::Options> options) override {
      m_Likelihood = std::make_shared<LinRegLikelihood>(std::move(options), *m_InputOptions);
      return m_Likelihood;
    }

    void write_results(Fit& fit, std::string_view name) override;

   private:
    std::shared_ptr<io::linreg::LinRegInputOptions> m_InputOptions;
    std::shared_ptr<LinRegLikelihood>               m_Likelihood;
  };

}  // namespace ana::linreg
