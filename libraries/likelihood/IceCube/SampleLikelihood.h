#pragma once

#include "../../io/IceCube/ICSample.h"
#include "../../io/IceCube/SampleConfig.h"
#include "../ParameterWrapper.h"
#include "AtmosphericFlux.h"
#include "DetectorSystematics.h"
#include "GpuBackend.h"
#include "GpuBinReduce.h"
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
    // Gates the OpenMP loops in the flux components and SampleLikelihood's own
    // per-bin ssq reduction, via each pragma's if() clause. Set once from
    // io::InputOptions::use_multi_threading() in ICLikelihood's constructor.
    bool                use_multi_threading = true;
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
                     std::shared_ptr<GpuSession> gpu,
                     bool                        use_say);

    /** Recompute prediction for the current parameters; return this sample's -2lnL (no pulls). */
    [[nodiscard]] double partial_llh(const ParameterWrapper& parameter);

    /** Fill this sample's Asimov data from the nominal parameters. */
    void generate_asimov(const ParameterWrapper& nominal);

    /**
     * The prediction in the analysis binning.
     *
     * Materialised on demand rather than by every evaluation: the likelihood
     * loop reads the MC-binned total and the galactic templates directly and
     * spreads them over the RA axis as it goes, so writing the 3D array out
     * would be 2 MB of stores and loads per evaluation that only the results
     * writers and the tests ever look at.
     */
    [[nodiscard]] std::span<const double> predicted() const noexcept;
    [[nodiscard]] std::span<const double> data() const noexcept { return m_Data; }

    /** Per-bin sigma^2 in the analysis binning (SAY only; zero under Poisson).
        Materialised on demand, like predicted(). */
    [[nodiscard]] std::span<const double> ssq() const noexcept;

    /** This sample's config: name, binning and component list (for the results writer). */
    [[nodiscard]] const io::ic::SampleConfig& config() const noexcept { return m_Config; }

    /** Per-bin prediction of one named part of this sample, for the results writer. */
    [[nodiscard]] std::span<const double> astro_histogram() const noexcept {
      return m_Astro ? m_Astro->histogram() : std::span<const double>{};
    }
    [[nodiscard]] std::span<const double> atmospheric_histogram() const noexcept {
      return m_Atmo ? m_Atmo->histogram() : std::span<const double>{};
    }
    /**
     * atmospheric_histogram() split into its conventional and prompt halves, for
     * the results writers. Both vectors are empty when this sample declares no
     * atmospheric component, matching the empty span the accessors above return.
     * Recomputed on the CPU from the given parameters -- pass the parameters the
     * prediction was last built with, or the split will not add up to it.
     */
    [[nodiscard]] AtmoBreakdown atmospheric_breakdown(const ParameterWrapper& parameter) const {
      return m_Atmo ? m_Atmo->breakdown(parameter) : AtmoBreakdown{};
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
    bool                        m_UseMultiThreading;

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
    // m_GalacticTotal is only re-summed when a galactic norm actually moved;
    // this makes the very first assemble_prediction fill it regardless.
    bool                      m_GalacticSeeded = false;

    // Analysis-binning buffers (RA axis included when the sample has one).
    // m_Predicted and m_Ssq are outputs of the accessors above, not of the hot
    // path: mutable because they are filled lazily from const accessors, and
    // stale until the flags below say otherwise.
    mutable std::vector<double> m_Predicted;
    std::vector<double>         m_Data;
    mutable std::vector<double> m_Ssq;
    mutable bool                m_PredictedStale = true;
    mutable bool                m_SsqStale       = true;

    // lgamma(m_Data[b] + 1), refreshed by refresh_data_constants() whenever
    // m_Data changes. A per-bin constant of the fit that the likelihood term
    // would otherwise recompute on every evaluation.
    std::vector<double> m_LogGammaDataPlus1;
    // The saturated Poisson term log P(k|k), which the Poisson likelihood
    // subtracts per bin. It depends only on the data, so it is a constant of
    // the fit -- and one that costs a log per analysis bin per evaluation when
    // it is not treated as one.
    std::vector<double> m_PoissonSaturated;

    // Per-RA-group constants, one entry per MC bin. Everything the prediction
    // contributes to a likelihood bin is identical across that bin's RA slice
    // (mu is the MC total divided by n_ra, sigma^2 the MC value divided by
    // n_ra^2), so with no galactic template in the sample the only thing that
    // varies along RA is the data. Summing the data-side constants per group
    // once turns the whole slice into a handful of scalar operations plus one
    // logarithm -- see poisson_llh()/say_llh().
    std::vector<double> m_GroupDataSum;       ///< sum of k over the group
    std::vector<double> m_GroupLogGammaSum;   ///< sum of lgamma(k + 1) over the group
    std::vector<double> m_GroupSaturatedSum;  ///< sum of the saturated Poisson term
    std::vector<int>    m_GroupZeroCount;     ///< bins with k == 0, which need no lgamma(k + alpha)
    // Analysis-bin indices with k > 0, grouped by MC bin (CSR, offsets sized
    // n_mc + 1). Only the SAY term walks them.
    std::vector<int>          m_NonZeroData;
    std::vector<std::size_t>  m_NonZeroOffsets;

    // MC-binning scratch: the per-event components and the 2D templates/gradients
    // are summed here, then spread over the RA axis (mu / n_ra, sigma^2 / n_ra^2 --
    // NNMFit Binning_2D_to_3D). n_ra == 1 makes both broadcasts exact copies.
    // Folded flux histograms, filled from the per-event weights through the
    // sample's response matrix. Empty unless the sample declares one.
    std::vector<double> m_FoldedAstro;
    std::vector<double> m_FoldedAtmo;

    /** hist[b] = sum_k f_bk * per_event[event_bk] over the response matrix. */
    void fold(std::span<const double> per_event, std::vector<double>& out) const noexcept;

    std::vector<double> m_McTotal;
    std::vector<double> m_McSsq;
    // The per-event half of m_McSsq, kept apart from the histogram-level terms
    // (template fluctuations, detector gradients) that are added on top of it.
    // Only a flux recalculation can change it, so a step in a parameter that
    // only moves those histogram-level terms re-adds them to this cache instead
    // of re-reducing every event in the sample.
    std::vector<double> m_McSsqEvent;
    int                 m_RaBins = 1;

    // Scratch for the deterministic chunked sum in the likelihood loops, a
    // member so the loop allocates nothing per evaluation.
    mutable std::vector<double> m_Partial;

    // SAY-on-GPU: the per-event ssq reduction runs as the say_ssq kernel over
    // the flux components' GPU-resident per-event weight buffers. Set up in the
    // constructor when a backend is present, SAY is active and at least one
    // per-event component exists; m_hSsq == -1 selects the CPU fallback.
    std::shared_ptr<GpuSession>  m_Gpu;
    std::optional<GpuBinReduce>  m_SsqReduce;
    int                          m_hSsq = -1;

    // NOTE: SampleLikelihood does not own a ParameterWrapper member (unlike
    // ICLikelihood's m_Parameter) -- the caller resets one shared ParameterWrapper
    // and passes it into partial_llh()/generate_asimov(). assemble_prediction()
    // therefore takes it explicitly so it can call check_and_recalculate() and
    // report whether any flux that feeds sigma^2 changed -- which excludes the
    // galactic templates, see the comment at its definition.
    // assemble_fluctuation() needs no parameter:
    // it only re-sums the per-event weights the flux components already
    // recalculated.
    /**
     * @return {anything that feeds sigma^2 changed, the per-event weights changed}.
     * The second flag is what lets assemble_fluctuation() skip the event-level
     * reduction and only re-add the histogram-level terms.
     */
    struct PredictionChange {
      bool ssq       = false;
      bool per_event = false;
    };
    PredictionChange assemble_prediction(const ParameterWrapper& parameter);
    void assemble_fluctuation(bool per_event_changed = true);

    /** Fill m_Predicted / m_Ssq from the MC-binned state (see predicted()). */
    void materialize_prediction() const;
    void materialize_ssq() const;

    /** -2 lnL over the analysis bins, reading the prediction group by group. */
    [[nodiscard]] double poisson_llh() const;
    [[nodiscard]] double say_llh() const;

    /** Recompute the per-bin constants derived from m_Data. Call after every
        write to m_Data and nowhere else. */
    void refresh_data_constants();
  };

}  // namespace ana::ic
