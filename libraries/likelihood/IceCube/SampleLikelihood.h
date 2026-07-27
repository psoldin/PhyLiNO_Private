#pragma once

#include "../../io/IceCube/ICSample.h"
#include "../../io/IceCube/SampleConfig.h"
#include "../ParameterWrapper.h"
#include "AtmosphericFlux.h"
#include "GpuBackend.h"
#include "PowerlawFlux.h"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ana::ic {

  /** Global (shared-across-samples) flux settings, built once from ICInputOptions. */
  struct GlobalFluxSettings {
    double e_ref_gev;
    double astro_reference_index;
    double conv_delta_gamma_e_ref;
    double prompt_delta_gamma_e_ref;
    bool   astro_per_type_norm;
    double veto_anchor_energy;
    double veto_rescale_energy;
  };

  /**
   * One IceCube sample's prediction and its partial -2lnL contribution. Owns the
   * flux components its SampleConfig declares (see io::ic::component) and its
   * runtime-sized histograms; knows nothing about other samples. The meta
   * ICLikelihood sums partial_llh() across samples and adds the Gaussian pulls
   * once.
   *
   * `cfg` must outlive the SampleLikelihood: it is held by reference, and its
   * name/binning/components describe this sample in the results output.
   * ICExperimentModule owns the ICInputOptions the configs live in.
   */
  class SampleLikelihood {
   public:
    SampleLikelihood(const io::ic::ICSample&     sample,
                     const io::ic::SampleConfig& cfg,
                     const GlobalFluxSettings&   settings,
                     std::shared_ptr<GpuBackend> gpu,
                     bool                        use_say);

    /** Recompute prediction for the current parameters; return this sample's -2lnL (no pulls). */
    [[nodiscard]] double partial_llh(const ParameterWrapper& parameter);

    /** Fill this sample's Asimov data from the nominal parameters. */
    void generate_asimov(const ParameterWrapper& nominal);

    [[nodiscard]] std::span<const double> predicted() const noexcept { return m_Predicted; }
    [[nodiscard]] std::span<const double> data() const noexcept { return m_Data; }

    /** This sample's config: name, binning and component list (for the results writer). */
    [[nodiscard]] const io::ic::SampleConfig& config() const noexcept { return m_Config; }

   private:
    const io::ic::ICSample&     m_Sample;
    const io::ic::SampleConfig& m_Config;
    bool                        m_UseSAY;

    // Only the components the config declares are constructed; the parquet
    // columns of an absent component were never read (see ICDataBase).
    std::optional<PowerlawFlux>    m_Astro;
    std::optional<AtmosphericFlux> m_Atmo;

    std::vector<double> m_Predicted;
    std::vector<double> m_Data;
    std::vector<double> m_Ssq;

    // NOTE: SampleLikelihood does not own a ParameterWrapper member (unlike
    // ICLikelihood's m_Parameter) -- the caller resets one shared ParameterWrapper
    // and passes it into partial_llh()/generate_asimov(). assemble_prediction()
    // therefore takes it explicitly so it can call check_and_recalculate() and
    // report whether any flux changed. assemble_fluctuation() needs no parameter:
    // it only re-sums the per-event weights the flux components already
    // recalculated.
    bool assemble_prediction(const ParameterWrapper& parameter);
    void assemble_fluctuation();
  };

}  // namespace ana::ic
