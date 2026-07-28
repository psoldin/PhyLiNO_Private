#pragma once

#include "../../io/IceCube/ICSample.h"
#include "../../io/IceCube/SampleConfig.h"
#include "../ParameterWrapper.h"
#include "AtmosphericFlux.h"
#include "DetectorSystematics.h"
#include "GpuBackend.h"
#include "PowerlawFlux.h"
#include "TemplateFlux.h"

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
    // Astrophysical spectral model: single power law or NNMFit's AstroBPL.
    io::ic::AstroModel astro_model = io::ic::AstroModel::Powerlaw;
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

    /** Per-bin sigma^2 in the analysis binning (SAY only; zero under Poisson). */
    [[nodiscard]] std::span<const double> ssq() const noexcept { return m_Ssq; }

    /** This sample's config: name, binning and component list (for the results writer). */
    [[nodiscard]] const io::ic::SampleConfig& config() const noexcept { return m_Config; }

    /** Per-bin prediction of one named part of this sample, for the results writer. */
    [[nodiscard]] std::span<const double> astro_histogram() const noexcept {
      return m_Astro ? m_Astro->histogram() : std::span<const double>{};
    }
    [[nodiscard]] std::span<const double> atmospheric_histogram() const noexcept {
      return m_Atmo ? m_Atmo->histogram() : std::span<const double>{};
    }
    [[nodiscard]] std::span<const double> template_histogram() const noexcept {
      return m_Template ? m_Template->histogram() : std::span<const double>{};
    }
    /** Summed per-bin prediction of this sample's galactic templates (analysis binning). */
    [[nodiscard]] std::span<const double> galactic_histogram() const noexcept { return m_GalacticTotal; }
    [[nodiscard]] std::span<const double> systematics_mu_delta() const noexcept {
      return m_Systematics ? m_Systematics->mu_delta() : std::span<const double>{};
    }

    /**
     * Re-express a component's per-bin histogram in the analysis binning. Components
     * binned in mc_binning are spread over the RA axis exactly as the prediction is;
     * a span already in the analysis binning (or an empty one) is returned as a copy.
     */
    [[nodiscard]] std::vector<double> in_analysis_bins(std::span<const double> values) const;

    /** Replace the Asimov expectation with measured counts (UseData). */
    void set_data(std::span<const double> counts);

   private:
    const io::ic::ICSample&     m_Sample;
    const io::ic::SampleConfig& m_Config;
    bool                        m_UseSAY;

    // Only the components the config declares are constructed; the parquet
    // columns of an absent component were never read (see ICDataBase).
    std::optional<PowerlawFlux>         m_Astro;
    std::optional<AtmosphericFlux>      m_Atmo;
    std::optional<TemplateFlux>         m_Template;
    std::optional<DetectorSystematics>  m_Systematics;

    // Galactic-plane templates (NNMFit GalacticTemplate). Structurally identical to
    // the muon TemplateFlux -- a per-bin rate times a norm times the livetime -- but
    // stored in the analysis binning and exported with a zero fluctuation column,
    // since NNMFit's GalacticTemplate defines no fluctuation graph and is excluded
    // from sigma^2 (histogram_builder.py:307).
    std::vector<TemplateFlux> m_Galactic;
    std::vector<double>       m_GalacticTotal;  // sum over m_Galactic, analysis binning

    // Analysis-binning buffers (RA axis included when the sample has one).
    std::vector<double> m_Predicted;
    std::vector<double> m_Data;
    std::vector<double> m_Ssq;

    // MC-binning scratch: the per-event components and the 2D templates/gradients
    // are summed here, then spread over the RA axis (mu / n_ra, sigma^2 / n_ra^2 --
    // NNMFit Binning_2D_to_3D). n_ra == 1 makes both broadcasts exact copies.
    std::vector<double> m_McTotal;
    std::vector<double> m_McSsq;
    int                 m_RaBins = 1;

    // SAY-on-GPU: the per-event ssq reduction runs as the say_ssq kernel over
    // the flux components' GPU-resident per-event weight buffers. Set up in the
    // constructor when a backend is present, SAY is active and at least one
    // per-event component exists; m_hSsq == -1 selects the CPU fallback.
    std::shared_ptr<GpuBackend> m_Gpu;
    int                         m_hSsqOffsets = -1;
    int                         m_hSsq        = -1;

    // NOTE: SampleLikelihood does not own a ParameterWrapper member (unlike
    // ICLikelihood's m_Parameter) -- the caller resets one shared ParameterWrapper
    // and passes it into partial_llh()/generate_asimov(). assemble_prediction()
    // therefore takes it explicitly so it can call check_and_recalculate() and
    // report whether any flux that feeds sigma^2 changed -- which excludes the
    // galactic templates, see the comment at its definition.
    // assemble_fluctuation() needs no parameter:
    // it only re-sums the per-event weights the flux components already
    // recalculated.
    bool assemble_prediction(const ParameterWrapper& parameter);
    void assemble_fluctuation();
  };

}  // namespace ana::ic
