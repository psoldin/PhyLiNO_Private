#pragma once

// STL includes
#include <map>
#include <memory>
#include <string>
#include <vector>

// includes
#include "InputOptionBase.h"
#include "InputParameter.h"

namespace io {

  // Expands --gpuDevices ("device:count[,device:count...]", e.g. "0:6,1:10")
  // into one device ordinal per worker slot. "" (the default) returns {},
  // meaning "no override, every worker uses device 0". A free function so it
  // is testable without building a full InputOptions (which needs argv and a
  // config file on disk).
  [[nodiscard]] std::vector<int> parse_gpu_devices(const std::string& spec);

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

    /**
     * OpenMP team size requested via --threads. -1 (the default) leaves the
     * OpenMP/environment default alone. Only consulted when -m is given.
     */
    [[nodiscard]] int multi_threading_cores() const noexcept { return m_MultiThreadingCores; }

    /**
     * Number of grid points the 2D scan fits concurrently (--scanWorkers).
     * Each worker runs a whole fit with its own likelihood, so on the IceCube
     * GPU backends every worker also holds its own copy of the MC columns on
     * the device. Defaults to 1, i.e. the sequential scan.
     */
    [[nodiscard]] int scan_workers() const noexcept { return m_ScanWorkers; }

    /**
     * Which GPU ordinal scan worker `worker_index` (0-based, as handed to
     * ExperimentModule::create_likelihood) should run on, from --gpuDevices.
     * That flag is "device:count[,device:count...]", e.g. "0:6,1:10" fills
     * device 0 with the first 6 workers and device 1 with the next 10, so
     * bumping a count and rerunning is how you saturate one card before
     * spilling onto the next. Empty (the default) means every worker uses
     * device 0, i.e. today's single-GPU behaviour. Consulted only by GPU
     * backends that support more than one device (currently IceCube/CUDA);
     * wraps around if worker_index exceeds the expanded list.
     */
    [[nodiscard]] int gpu_device_for_worker(int worker_index) const noexcept {
      return m_GpuDeviceOfWorker.empty() ? 0 : m_GpuDeviceOfWorker[worker_index % static_cast<int>(m_GpuDeviceOfWorker.size())];
    }

    /**
     * Start each scan fit from the converged parameters of the nearest scan
     * point already fitted, instead of from the configured start values
     * (--scanWarmStart, on by default). Neighbouring scan points differ only in
     * the scanned parameter, so the previous point's nuisance parameters leave
     * the minimizer close to the answer and it spends its iterations polishing
     * rather than travelling.
     *
     * Ignored when --randomizeSeeds is given: that option exists to spread the
     * start points on purpose.
     */
    [[nodiscard]] bool scan_warm_start() const noexcept { return m_ScanWarmStart; }

    /**
     * Which scan LLHFit runs when --fitOnly is not given (--scanMode, default
     * "2d"): "2d" (adaptive, perform_2d_scan), "2d-regular"
     * (perform_2d_scan_regular), "1d" or "1d-regular" (perform_1d_scan_all,
     * regular selecting its grid flavour). Validated in LLHFit.C, the only
     * place that knows the scan functions.
     */
    [[nodiscard]] const std::string& scan_mode() const noexcept { return m_ScanMode; }

    /**
     * Restricts --scanMode 1d/1d-regular to this single named parameter
     * (perform_1d_scan/perform_1d_scan_regular) instead of every non-fixed one
     * (perform_1d_scan_all). Empty (the default) means "every parameter".
     */
    [[nodiscard]] const std::string& scan_parameter() const noexcept { return m_ScanParameter; }

    /** Grid points per axis for --scanMode 2d-regular/1d-regular (--scanPoints, default 30). */
    [[nodiscard]] int scan_points() const noexcept { return m_ScanPoints; }

    [[nodiscard]] const boost::property_tree::ptree& config_tree() const noexcept { return m_ConfigTree; }

    [[nodiscard]] double tolerance() const noexcept { return m_Tolerance; }

    /**
     * How often a Migrad that reported failure is restarted, each time on a
     * freshly built minimizer seeded where the previous attempt stopped
     * (--fitRetries, default 3). See Fit::minimize() for the measurements.
     *
     * Each restart costs a full Migrad, so this is capped rather than looped
     * until convergence; 0 disables the loop.
     */
    [[nodiscard]] int fit_retries() const noexcept { return m_FitRetries; }

    /**
     * Minuit2 strategy (--minuitStrategy, default 1). 2 buys more accurate
     * numerical derivatives and more frequent Hessian refreshes for roughly
     * twice the function calls, which is the other standard answer to a Migrad
     * that stalls in a line search.
     */
    [[nodiscard]] int minuit_strategy() const noexcept { return m_MinuitStrategy; }

    /**
     * Which Minuit2 algorithm the fit runs (--minimizerAlgo, default "Migrad").
     * "Combined" is MnCombinedMinimizer: Migrad, and on failure Simplex followed
     * by Migrad again. Simplex needs no derivatives, so it leaves a stalled
     * point instead of rebuilding a Hessian at it.
     */
    [[nodiscard]] const std::string& minimizer_algo() const noexcept { return m_MinimizerAlgo; }

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

    /**
     * Blind the written results (--blind): the signal parameters are left out of
     * the parameter block and every bin above 1e4 GeV is written as zero, in the
     * data, the prediction and every component alike. The fit is unaffected --
     * all parameters float and all bins enter the likelihood, so a blinded run
     * and an unblinded one minimize exactly the same thing. See ICBlinding.h for
     * what counts as signal.
     */
    [[nodiscard]] bool blind() const noexcept { return m_Blind; }

   private:
    long   m_Seed;              /**< The global random seed. */
    bool   m_Silent;            /**< Flag indicating if the program should run in silent mode. */
    bool   m_UseMultiThreading;      /**< Flag indicating if the fit may use multiple threads. */
    int    m_MultiThreadingCores{-1}; /**< OpenMP team size; -1 keeps the environment default. */
    int    m_ScanWorkers{1};          /**< Grid points the 2D scan fits concurrently. */
    std::string      m_GpuDevices;         /**< Raw --gpuDevices value, e.g. "0:6,1:10". */
    std::vector<int> m_GpuDeviceOfWorker;  /**< --gpuDevices expanded to one device ordinal per worker slot. */
    bool   m_ScanWarmStart{true};     /**< Seed each scan fit from the nearest point already fitted. */
    std::string m_ScanMode{"2d"};     /**< Which scan LLHFit runs when --fitOnly is not given. */
    std::string m_ScanParameter;      /**< Single parameter for --scanMode 1d/1d-regular; empty means every parameter. */
    int    m_ScanPoints{30};          /**< Grid points per axis for --scanMode 2d-regular/1d-regular. */
    bool   m_FitOnly{false};    /**< Run a single fit instead of the 2D scan. */
    bool   m_RandomizeSeeds{false}; /**< Randomize the minimizer start values. */
    double m_RandomizeWidth{0.08};  /**< Relative width of the randomized start values. */
    bool   m_Blind{false};          /**< Keep the signal out of the written results. */
    double m_Tolerance;         /**< The tolerance for the minimizer. */
    int    m_FitRetries{3};     /**< Restarts granted to a Migrad that stalled. */
    int    m_MinuitStrategy{1}; /**< Minuit2 strategy passed to the minimizer. */
    std::string m_MinimizerAlgo{"Migrad"}; /**< Minuit2 algorithm the fit runs. */
    std::string m_OutputFormat; /**< Result output format ("json" or "protobuf"). */

    std::string m_ConfigFile; /**< The configuration file path. */

    boost::property_tree::ptree m_ConfigTree;  // < The configuration tree

    std::shared_ptr<InputParameter> m_InputParameter; /**< The input parameter object. */

    std::string          m_Experiment;        /**< The selected experiment. */
    experiment_options_t m_ExperimentOptions; /**< Option parsers of all registered experiments. */
  };
}  // namespace io
