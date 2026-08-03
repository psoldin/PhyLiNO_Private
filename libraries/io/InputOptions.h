#pragma once

// STL includes
#include <map>
#include <memory>
#include <string>

// includes
#include "InputOptionBase.h"
#include "InputParameter.h"

namespace io {

  /**
   * @brief Class representing input options for the program.
   *
   * This class provides methods to access and manipulate the input options
   * for the program. It stores information such as the command line arguments,
   * random seed, silent mode flag, and other related options.
   */
  class InputOptions {
   public:
    using experiment_options_t = std::map<std::string, std::shared_ptr<InputOptionBase>>;

    /**
     * @brief Constructor that initializes the input options.
     *
     * @param argc The number of command line arguments.
     * @param argv The array of command line arguments.
     * @param experiment_options Option parsers of all registered experiments.
     */
    InputOptions(int argc, char** argv, experiment_options_t experiment_options);

    /** Default destructor */
    ~InputOptions() = default;

    /**
     * @brief Get the global random seed.
     *
     * @return The global random seed for number generation.
     */
    [[nodiscard]] long seed() const noexcept { return m_Seed; }

    /**
     * @brief Check if the program should run in silent mode.
     *
     * @return True if the program should run in silent mode, false otherwise.
     */
    [[nodiscard]] bool silent() const noexcept { return m_Silent; }

    [[nodiscard]] const auto& input_parameters() const noexcept { return *m_InputParameter; }

    /** Name of the experiment selected via the "Experiment" config key. */
    [[nodiscard]] const std::string& experiment() const noexcept { return m_Experiment; }

    [[nodiscard]] bool use_multi_threading() const noexcept { return m_UseMultiThreading; }

    [[nodiscard]] const boost::property_tree::ptree& config_tree() const noexcept { return m_ConfigTree; }

    [[nodiscard]] double tolerance() const noexcept { return m_Tolerance; }

    /** Result output format ("json" or "protobuf"), as passed via --output-format. */
    [[nodiscard]] const std::string& output_format() const noexcept { return m_OutputFormat; }

    /**
     * Run one fit and write its result ("Output.json") instead of the 2D scan.
     * What LLHFit did before the scan became its default entry point; the
     * NNMFit likelihood-parity harness needs it (tools/nnmfit_oracle).
     */
    [[nodiscard]] bool fit_only() const noexcept { return m_FitOnly; }

    /**
     * Randomize the minimizer start values around the configured ones, the
     * counterpart of NNMFit's default `randomize_param_seeds` (see
     * ana::randomized_start_value). Only the start point moves -- the data,
     * Asimov included, is built from the configured values either way.
     */
    [[nodiscard]] bool randomize_seeds() const noexcept { return m_RandomizeSeeds; }

    /** Relative width of the randomized start values, as passed via --randomizeWidth. */
    [[nodiscard]] double randomize_width() const noexcept { return m_RandomizeWidth; }

   private:
    long   m_Seed;              /**< The global random seed. */
    bool   m_Silent;            /**< Flag indicating if the program should run in silent mode. */
    bool   m_UseMultiThreading; /**< The number of cores to use for multi-threading. */
    bool   m_FitOnly{false};    /**< Run a single fit instead of the 2D scan. */
    bool   m_RandomizeSeeds{false}; /**< Randomize the minimizer start values. */
    double m_RandomizeWidth{0.08};  /**< Relative width of the randomized start values. */
    double m_Tolerance;         /**< The tolerance for the minimizer. */
    std::string m_OutputFormat; /**< Result output format ("json" or "protobuf"). */

    std::string m_ConfigFile; /**< The configuration file path. */

    boost::property_tree::ptree m_ConfigTree;  // < The configuration tree

    std::shared_ptr<InputParameter> m_InputParameter; /**< The input parameter object. */

    std::string          m_Experiment;        /**< The selected experiment. */
    experiment_options_t m_ExperimentOptions; /**< Option parsers of all registered experiments. */
  };
}  // namespace io
