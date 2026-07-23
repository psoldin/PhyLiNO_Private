#pragma once

#include "InputOptions.h"

namespace io {

  /**
   * Main class for the options that are handled in the Likelihood fit.
   * Here the command line arguments as well as the measurement data are accessible.
   */
  class Options {
   public:
    /**
     * Constructor
     * @param argc Command line argc
     * @param argv Command line argv
     */
    Options(int argc, char** argv, InputOptions::experiment_options_t experiment_options)
      : m_InputOptions(argc, argv, std::move(experiment_options)) {}

    /** Default constructor */
    Options()
      : Options(1, nullptr, {}) {}

    /** Default destructor */
    ~Options() = default;

    /**
     * Accessor for the Input Options handed over at program start
     * @return InputOptions
     */
    [[nodiscard]] const InputOptions& inputOptions() const noexcept { return m_InputOptions; }

   private:
    InputOptions m_InputOptions;
  };

}  // namespace io
