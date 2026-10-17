#pragma once

#include "../ExperimentModule.h"
#include "GpuBackend.h"
#include "ICLikelihood.h"

#include "IceCube/ICDataBase.h"
#include "IceCube/ICInputOptions.h"
#include "IceCube/ICParameter.h"

#include <map>
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

    [[nodiscard]] std::shared_ptr<Likelihood> create_likelihood(std::shared_ptr<io::Options> options, int worker_index) override;

    void write_results(Fit& fit, std::string_view name) override;

    [[nodiscard]] const io::ic::ICInputOptions& ic_input_options() const noexcept { return *m_InputOptions; }

    /**
     * Use `other`'s MC and GPU cache from now on (LLHBatch). This module's config
     * must describe the same MC samples, binning, systematics inputs and backend
     * as the one that fills the cache -- only its data, template, likelihood and
     * parameters may differ, which the caller checks. Its data histograms are
     * still read from its own config (ICDataBase::with_data).
     */
    void share_cache_with(const ICExperimentModule& other) { m_Cache = other.m_Cache; }

   private:
    struct Cache {
      // Immutable MC sample; loaded once, reused across Fit constructions. The
      // likelihood itself belongs to the Fit that created it, so several Fits can
      // run concurrently on one module.
      std::shared_ptr<const io::ic::ICDataBase> database;
      // The options whose config built `database`; a module with other options
      // binds its own data histograms to the shared MC.
      std::shared_ptr<const io::ic::ICInputOptions> owner;
      // GPU backend(s) (empty on the CPU path), one per CUDA device ordinal in
      // use, so each device's context, compiled kernels and uploaded MC columns
      // are paid for once rather than once per scan point. Keyed by 0 for the
      // CPU/Metal paths, which only ever run one device. Declared after
      // `database` so it is destroyed first: its column cache is keyed on the raw
      // ICSample column pointers.
      std::map<int, std::shared_ptr<GpuBackend>> gpu_backends;
      std::mutex                                 mutex;
    };

    std::shared_ptr<io::ic::ICInputOptions> m_InputOptions;
    std::shared_ptr<Cache>                  m_Cache = std::make_shared<Cache>();
  };

}  // namespace ana::ic
