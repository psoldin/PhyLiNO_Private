#pragma once

#include "../../io/IceCube/ICConstants.h"
#include "../../io/IceCube/ICDataBase.h"
#include "../../io/IceCube/ICInputOptions.h"
#include "../Likelihood.h"
#include "GpuBackend.h"
#include "SampleLikelihood.h"

#include <memory>
#include <span>
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
   * both are handed over by ICExperimentModule::create_likelihood so heavy
   * parquet loading only happens when IceCube is the selected experiment.
   */
  class ICLikelihood : public Likelihood {
   public:
    ICLikelihood(std::shared_ptr<io::Options>              options,
                 std::shared_ptr<const io::ic::ICDataBase> data_base,
                 const io::ic::ICInputOptions&             input_options);
    ~ICLikelihood() override = default;

    [[nodiscard]] double calculate_likelihood(const double* parameter) override;

    /** Predicted counts per analysis bin at the last evaluated parameter set (first sample). */
    [[nodiscard]] std::span<const double> predicted() const noexcept { return m_Samples.front()->predicted(); }

    /** Asimov (or measured) counts per analysis bin the fit runs against (first sample). */
    [[nodiscard]] std::span<const double> data() const noexcept { return m_Samples.front()->data(); }

   private:
    // Declared first so it (and its ICSamples) outlives the sample likelihoods below.
    std::shared_ptr<const io::ic::ICDataBase> m_DataBase;

    // One GPU backend (Metal or CUDA) shared by every sample's flux components
    // (null on the CPU path), so per-event columns like e_true / bin_offsets are
    // uploaded only once. Declared before the sample members so it outlives them.
    std::shared_ptr<GpuBackend> m_GpuBackend;

    std::vector<std::unique_ptr<SampleLikelihood>> m_Samples;

    // Gaussian pulls: (param_index, central_value, sigma)
    std::vector<std::tuple<int, double, double>> m_Pulls;

    void                 initialize_data(bool use_data);
    void                 setup_pulls();
    [[nodiscard]] double calculate_pulls(const ParameterWrapper& parameter) const noexcept;
  };

}  // namespace ana::ic
