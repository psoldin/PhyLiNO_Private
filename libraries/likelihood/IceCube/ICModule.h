#pragma once

#include "../ExperimentModule.h"
#include "ICLikelihood.h"

#include "IceCube/ICDataBase.h"
#include "IceCube/ICInputOptions.h"
#include "IceCube/ICParameter.h"

#include <mutex>

namespace ana::ic {

  /**
   * @brief Experiment module of the IceCube diffuse-flux analysis.
   *
   * Owns the IceCube input options and, when IceCube is the selected experiment,
   * loads the MC baseline parquet of every enabled sample (ICDataBase) and
   * creates the composite ICLikelihood over them.
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

    [[nodiscard]] const io::ic::ICInputOptions& ic_input_options() const noexcept { return *m_InputOptions; }

   private:
    std::shared_ptr<io::ic::ICInputOptions>   m_InputOptions;
    // Cached immutable MC sample; loaded once, reused across Fit constructions.
    // The only state the module keeps: the likelihood itself belongs to the Fit
    // that created it, so several Fits can run concurrently on one module.
    std::shared_ptr<const io::ic::ICDataBase> m_DataBase;
    std::mutex                                m_DataBaseMutex;
  };

}  // namespace ana::ic
