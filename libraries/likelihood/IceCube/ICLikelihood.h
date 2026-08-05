#pragma once

#include "../../io/IceCube/ICDataBase.h"
#include "../../io/IceCube/ICInputOptions.h"
#include "../Likelihood.h"
#include "GpuBackend.h"
#include "SampleLikelihood.h"

#include <cstddef>
#include <memory>
#include <tuple>
#include <vector>

namespace ana::ic {

  /**
   * IceCube composite binned profile likelihood over one or more analysis
   * samples (e.g. tracks, cascades), each enabled/disabled independently via
   * its SampleConfig. Sums each sample's partial -2lnL (Poisson or SAY, no
   * pulls) and adds the Gaussian pulls on constrained parameters once at the
   * meta level.
   *
   * Owns the ICDataBase whose ICSamples the per-sample flux components read;
   * it and the GPU backend are handed over by
   * ICExperimentModule::create_likelihood, which caches both, so the parquet
   * load, the device context, the kernel compiles and the column uploads happen
   * once per process rather than once per Fit -- a scan builds one Fit per grid
   * point. A null backend selects the CPU path.
   */
  class ICLikelihood : public Likelihood {
   public:
    ICLikelihood(std::shared_ptr<io::Options>              options,
                 std::shared_ptr<const io::ic::ICDataBase> data_base,
                 const io::ic::ICInputOptions&             input_options,
                 std::shared_ptr<GpuBackend>               gpu_backend = nullptr);
    ~ICLikelihood() override = default;

    [[nodiscard]] double calculate_likelihood(const double* parameter) override;

    /** Number of enabled samples this composite sums over. */
    [[nodiscard]] std::size_t n_samples() const noexcept { return m_Samples.size(); }

    /**
     * The i-th enabled sample (config order), for the results writer: each one
     * carries its own binning, data and prediction.
     */
    [[nodiscard]] const SampleLikelihood& sample(std::size_t i) const noexcept { return *m_Samples[i]; }

   private:
    // Declared first so it (and its ICSamples) outlives the sample likelihoods below.
    std::shared_ptr<const io::ic::ICDataBase> m_DataBase;

    // The module's GPU backend (Metal or CUDA) behind every sample's session,
    // null on the CPU path. Held so it outlives the sessions below -- which the
    // sessions also enforce themselves, since each keeps the backend alive.
    std::shared_ptr<GpuBackend> m_GpuBackend;

    std::vector<std::unique_ptr<SampleLikelihood>> m_Samples;

    // Gaussian pulls: (param_index, central_value, sigma)
    std::vector<std::tuple<int, double, double>> m_Pulls;

    // Gates both concurrency mechanisms in this likelihood: the std::async
    // per-sample partial_llh spawn below and (via each pragma's if() clause)
    // the OpenMP loops in the flux components. Defaults on; later CPU/GPU
    // backend selection can drive this per-instance instead of only from the
    // global -m option.
    bool m_UseMultiThreading{true};

    void                 initialize_data(bool use_data);
    void                 setup_pulls();
    [[nodiscard]] double calculate_pulls(const ParameterWrapper& parameter) const noexcept;
  };

}  // namespace ana::ic
