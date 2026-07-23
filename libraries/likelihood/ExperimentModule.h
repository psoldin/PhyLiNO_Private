#pragma once

#include "InputOptionBase.h"
#include "Likelihood.h"

#include <map>
#include <memory>
#include <string>

namespace ana {

  /**
   * @brief Interface every experiment implements to plug into the framework.
   *
   * The constructor of an implementation must be cheap: modules for all experiments are created
   * before the command line and config file are parsed. Heavy input loading belongs into
   * create_likelihood(), which is only invoked for the experiment selected in the config file.
   */
  class ExperimentModule {
   public:
    virtual ~ExperimentModule() = default;

    /** Name used for the "Experiment" config key and the registry. */
    [[nodiscard]] virtual std::string name() const = 0;

    /** Option parser hook registered with io::InputOptions before parsing. */
    [[nodiscard]] virtual std::shared_ptr<io::InputOptionBase> input_options() = 0;

    /** Number of fit parameters this experiment expects. */
    [[nodiscard]] virtual int number_of_parameters() const = 0;

    /** Create the likelihood. This is where experiment data is loaded. */
    [[nodiscard]] virtual std::shared_ptr<Likelihood> create_likelihood(std::shared_ptr<io::Options> options) = 0;

    /**
     * Keep parameter i free even if the config marks it as fixed. Default: never.
     * (Double Chooz uses this for the sterile oscillation parameters.)
     */
    [[nodiscard]] virtual bool keep_parameter_free(const io::Options& options, std::size_t i) const {
      static_cast<void>(options);
      static_cast<void>(i);
      return false;
    }
  };

  using module_map_t = std::map<std::string, std::shared_ptr<ExperimentModule>>;

  /** Collect the InputOptionBase handles of all registered modules for io::InputOptions. */
  inline std::map<std::string, std::shared_ptr<io::InputOptionBase>> collect_input_options(const module_map_t& modules) {
    std::map<std::string, std::shared_ptr<io::InputOptionBase>> result;
    for (const auto& [name, module] : modules) {
      result[name] = module->input_options();
    }
    return result;
  }

}  // namespace ana
