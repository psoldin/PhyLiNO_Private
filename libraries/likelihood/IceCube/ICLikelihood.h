#pragma once

#include "../../io/IceCube/ICConstants.h"
#include "../../io/IceCube/ICDataBase.h"
#include "../../io/IceCube/ICInputOptions.h"
#include "../Likelihood.h"
#include "AtmosphericFlux.h"
#include "MetalBackend.h"
#include "PowerlawFlux.h"

#include <array>
#include <memory>
#include <span>
#include <tuple>
#include <vector>

namespace ana::ic {

  /**
   * IceCube tracks-only binned profile likelihood.
   * Prediction = astro (Powerlaw) + conv+prompt (AtmosphericFlux)
   *              (+ muon template + detector-systematics delta, scaffolded).
   * -2 ln L = Poisson (or SAY) term + Gaussian pulls on constrained parameters.
   *
   * Owns the ICDataBase whose ICSample the flux components read; both are handed
   * over by ICExperimentModule::create_likelihood so heavy parquet loading only
   * happens when IceCube is the selected experiment.
   */
  class ICLikelihood : public Likelihood {
   public:
    ICLikelihood(std::shared_ptr<io::Options>              options,
                 std::shared_ptr<const io::ic::ICDataBase> data_base,
                 const io::ic::ICInputOptions&             input_options);
    ~ICLikelihood() override = default;

    [[nodiscard]] double calculate_likelihood(const double* parameter) override;

    /** Predicted counts per analysis bin at the last evaluated parameter set. */
    [[nodiscard]] std::span<const double> predicted() const noexcept { return m_TotalPredicted; }

    /** Asimov (or measured) counts per analysis bin the fit runs against. */
    [[nodiscard]] std::span<const double> data() const noexcept { return m_Data; }

   private:
    using BinArray = std::array<double, io::ic::Constants::nBins>;

    // Declared first so it (and its ICSample) outlives the flux components below.
    std::shared_ptr<const io::ic::ICDataBase> m_DataBase;

    // One Metal backend shared by every flux component (null on the CPU path),
    // so per-event columns like e_true / bin_offsets are uploaded only once.
    // Declared before the flux members so it outlives them.
    std::shared_ptr<MetalBackend> m_MetalBackend;

    PowerlawFlux    m_Astro;
    AtmosphericFlux m_Atmo;
    bool            m_UseSAY;

    BinArray m_TotalPredicted{};
    BinArray m_Data{};
    BinArray m_Ssq{};  // only populated/used when m_UseSAY is true

    // Gaussian pulls: (param_index, central_value, sigma)
    std::vector<std::tuple<int, double, double>> m_Pulls;

    bool                 assemble_prediction(BinArray& out);
    void                 assemble_fluctuation(BinArray& out);
    void                 initialize_data(bool use_data);
    void                 generate_asimov_data();
    void                 setup_pulls();
    [[nodiscard]] double calculate_pulls(const ParameterWrapper& parameter) const noexcept;
  };

}  // namespace ana::ic
