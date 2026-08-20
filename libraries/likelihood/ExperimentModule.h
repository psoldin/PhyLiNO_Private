#pragma once

#include "InputOptionBase.h"
#include "Likelihood.h"

#include <map>
#include <memory>
#include <string>
#include <string_view>

namespace ana {

  class Fit;

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

    /**
     * Create the likelihood. This is where experiment data is loaded.
     *
     * worker_index identifies the scan worker thread doing the building (0
     * for --fitOnly and the first scan worker); GPU-backed modules that
     * support --gpuDevices use it to pick which device this worker's Fit
     * runs on (see InputOptions::gpu_device_for_worker()). Modules that don't
     * support multiple devices simply ignore it.
     */
    [[nodiscard]] virtual std::shared_ptr<Likelihood> create_likelihood(std::shared_ptr<io::Options> options, int worker_index = 0) = 0;

    /**
     * Keep parameter i free even if the config marks it as fixed. Default: never.
     * (Double Chooz uses this for the sterile oscillation parameters.)
     */
    [[nodiscard]] virtual bool keep_parameter_free(std::size_t i) const {
      static_cast<void>(i);
      return false;
    }

    /** Write the experiment-specific results of a finished fit to "<name>.json". */
    virtual void write_results(Fit& fit, std::string_view name) = 0;
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
