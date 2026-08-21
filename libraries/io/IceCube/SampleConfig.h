#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <boost/property_tree/ptree_fwd.hpp>

#include "Binning.h"
#include "BranchNames.h"

namespace io::ic {

  /**
   * Flux components a sample can declare in its config "components" list. The
   * names match NNMFit's component keys so an NNMFit analysis config translates
   * directly. Only the components listed here are implemented; parse_samples()
   * rejects anything else rather than silently ignoring it.
   *
   * NNMFit gives each sample one atmospheric variant: the plain
   * `conventional`/`prompt` pair (tracks) or the veto-reweighted
   * `conventional_veto`/`prompt_veto` pair (the cascade samples), which is the same
   * flux times a per-event passing fraction. Muon backgrounds come from a binned
   * template: Corsika `muontemplate` for tracks, MuonGun `muon` for the cascades.
   */
  namespace component {
    inline constexpr std::string_view Astro            = "astro";
    inline constexpr std::string_view Conventional     = "conventional";
    inline constexpr std::string_view Prompt           = "prompt";
    inline constexpr std::string_view ConventionalVeto = "conventional_veto";
    inline constexpr std::string_view PromptVeto       = "prompt_veto";
    inline constexpr std::string_view MuonTemplate     = "muontemplate";  // Corsika, tracks
    inline constexpr std::string_view MuonGun          = "muon";          // MuonGun, cascades
  }  // namespace component

  /**
   * One galactic-plane template a sample declares (NNMFit GalacticTemplate): an
   * exported per-bin rate file in the sample's *analysis* binning and the norm
   * parameter scaling it. Unlike the muon template these are already 3D -- NNMFit's
   * histogram components are never binned, so the file carries the RA structure.
   */
  struct GalacticTemplateConfig {
    std::string name;        // config key, for diagnostics and the results output
    std::string file;
    int         norm_index = -1;
  };

  /**
   * Config for one analysis sample: which parquet file backs it, its analysis
   * binning, which flux components it includes, its livetime, and per-sample
   * branch-name overrides. Populated by parse_samples() from the
   * "IceCube.Binnings" / "IceCube.Samples" config subtrees; see
   * ICInputOptions::samples().
   *
   * Binning has no default constructor, so SampleConfig is aggregate/
   * designated-init constructed with `.binning` always provided, e.g.:
   *   SampleConfig sc{.name = sname, .binning = resolved};
   */
  /**
   * How a per-event uncertainty column is turned into a response width. The
   * conventions differ per column and are not self-describing, so they are
   * config rather than folklore:
   *
   *   None        the column already is the width in the axis's own units
   *   Exp         exp(x): a network emitting log(sigma) to keep it positive,
   *               which is what ELEFANTS does
   *   Pow10       10^x, for a column carrying log10(sigma)
   *   DegToRad    x * pi/180, for an angular sigma in degrees
   */
  enum class SigmaTransform { None, Exp, Pow10, DegToRad };

  /**
   * Forward folding of the per-event detector response into the analysis
   * histogram (see io::ic::ResponseMatrix).
   *
   * Off by default: with it disabled every MC event drops its full weight into
   * the single bin its reconstructed value falls in, which is what this code has
   * always done.
   */
  struct ResponseConfig {
    bool enabled = false;

    // Centres. These must be the truth the uncertainty column was trained
    // against, not the primary neutrino energy: ELEFANTS reconstructs the muon
    // energy at closest approach, so its sigma is the width around THAT, and the
    // neutrino-to-muon spread is separate physics the MC already carries event
    // by event.
    std::string truth_energy_branch = "ELEFANTS_tg_truth_log10";  // log10(E_true / GeV)
    std::string truth_zenith_branch = "MCPrimaryZenith";          // radians

    std::string    energy_sigma_branch    = "ELEFANTS_tg_sigma_log10";
    std::string    zenith_sigma_branch    = "L5_sigma_paraboloid";
    SigmaTransform energy_sigma_transform = SigmaTransform::None;
    SigmaTransform zenith_sigma_transform = SigmaTransform::DegToRad;

    // Bins beyond this many widths are not considered, and entries below
    // min_fraction of an event's total are dropped. Both bound the matrix; what
    // survives is renormalised, so neither changes the predicted total.
    double truncation   = 5.0;
    double min_fraction = 1.0e-4;
  };

  struct SampleConfig {
    std::string name;
    bool        enabled = true;
    Binning     binning;

    // The binning MC events are assigned to: `binning` without its trailing Ra axis
    // (identical to `binning` when there is none). Per-event fluxes, the muon
    // template and the SnowStorm gradients all live here; SampleLikelihood spreads
    // them over the RA axis. Set by parse_samples() via drop_ra_axis().
    Binning     mc_binning;

    std::string parquet;
    std::string data_path;

    // Pre-binned data counts ("DataCounts"), one value per analysis bin, read
    // instead of binning a data parquet. This is how a pseudo-experiment is fed
    // to both frameworks from the same numbers: NNMFit takes the identical
    // array through analysis_type "custom_data" + "custom_dataset". Takes
    // precedence over data_path when both are set.
    std::string data_counts_path;
    double      livetime = 1.0;
    std::vector<std::string> components;
    BranchNames branches;

    // Muon template, when the sample declares "muontemplate" or "muon": the
    // exported per-bin template file and which norm parameter scales it.
    std::string template_file;
    int         template_norm_index = -1;

    // Exported SnowStorm gradient file for this sample ("" = no detector systematics).
    std::string gradient_file;

    // "Gradients": { "ScaleToTopology": true }: reuse a gradient file exported
    // from the unfiltered sample with a topology cut active, rescaling each bin's
    // gradient by the fraction of that bin's nominal weight the cut kept. The
    // gradients are absolute per-bin count deltas, so applying them unscaled to a
    // subsample would overstate the detector systematics by 1/fraction. This is
    // an approximation -- it assumes every topology class responds to the
    // detector parameters the same way -- and is no substitute for re-exporting
    // the gradients per class. Only legal together with a topology cut.
    bool scale_gradients_to_topology = false;

    // Galactic-plane templates, in config order. Empty unless the analysis binning
    // has an Ra axis (parse_samples rejects the combination otherwise).
    std::vector<GalacticTemplateConfig> galactic;

    ResponseConfig response;

    // Per-event nu_mu survival factor (NNMFit OscillationsHook), exported by
    // tools/export_oscillation_factors.py. Row-aligned with `parquet`; applied to
    // the atmospheric baselines at load time. "" = no oscillation reweight.
    std::string oscillation_file;
    std::string oscillation_branch = "osc_survival";

    // Per-event topology cut ("Topology": { "Branch": ..., "Values": "1, 2" }):
    // keep only the events whose `topology_branch` column equals one of
    // `topology_values` (or, with "Exclude": true, keep everything except
    // those -- for the sample that should get "the rest" without enumerating
    // every other sample's labels). Applied to the MC parquet and to a data
    // parquet alike, before binning, so both sides of the likelihood see the
    // same selection. Empty `topology_values` = no cut. Pre-binned inputs
    // (muon/galactic templates, SnowStorm gradients, "DataCounts") were
    // exported from the unfiltered sample, so parse_samples() rejects
    // combining them with a cut.
    //
    // A NaN in `topology_branch` is not a class label, so it passes the cut
    // (and Exclude) by default. List "NaN" among `Values` to opt into
    // dropping NaN-labelled events instead.
    std::string      topology_branch;
    std::vector<int> topology_values;
    bool             topology_exclude  = false;
    bool             topology_drop_nan = false;

    [[nodiscard]] bool filters_topology() const noexcept { return !topology_values.empty() || topology_drop_nan; }

    [[nodiscard]] bool has_component(std::string_view component) const noexcept {
      return std::ranges::any_of(components,
        [component](std::string_view component_name) {
          return component_name == component;
        });
    }

    /** Astrophysical power law requested (drives PowerlawFlux + its parquet columns). */
    [[nodiscard]] bool wants_astro() const noexcept { return has_component(component::Astro); }

    /**
     * Conventional + prompt atmospheric flux requested, plain or veto-reweighted.
     * AtmosphericFlux computes conv and prompt in one per-event pass, so each
     * variant's pair is enabled together; parse_samples() rejects a half pair and
     * rejects declaring both variants at once.
     */
    [[nodiscard]] bool wants_atmospheric() const noexcept {
      return has_component(component::Conventional) || has_component(component::Prompt) ||
             wants_veto();
    }

    /** The atmospheric components carry NNMFit's passing-fraction reweight. */
    [[nodiscard]] bool wants_veto() const noexcept {
      return has_component(component::ConventionalVeto) || has_component(component::PromptVeto);
    }

    /** A muon template was declared (see template_file / template_norm_index). */
    [[nodiscard]] bool wants_template() const noexcept {
      return has_component(component::MuonTemplate) || has_component(component::MuonGun);
    }

    /** Bins on the analysis binning's trailing Ra axis, or 1 when it has none. */
    [[nodiscard]] int ra_bins() const noexcept { return ra_bin_count(binning); }
  };

  /**
   * Positions of the enabled entries in `samples`, in config order. ICDataBase
   * loads samples in exactly this order and ICLikelihood pairs its
   * SampleLikelihoods with it, so both call this instead of filtering on
   * `enabled` themselves -- the two filters cannot drift apart.
   */
  [[nodiscard]] std::vector<std::size_t> enabled_sample_indices(const std::vector<SampleConfig>& samples);

  /**
   * Tolerant parser for the "Binnings" / "Samples" subtrees of the "IceCube"
   * config node. `ic` is the "IceCube" node itself (i.e. what
   * config.get_child("IceCube") returns), not the top-level config.
   *
   * Each entry under "Binnings" has an "axes" string (comma-separated axis
   * kinds, e.g. "Log10Energy, CosZenith") plus one same-named key per axis
   * holding its "(lo, hi, n_bins)" spec, parsed via parse_axis().
   *
   * Each entry under "Samples" references a binning by name (throws if
   * unknown), and reads "enabled" (default true), "parquet" (required),
   * "data" (default ""), "livetime" (default 1.0), "components" (comma-split,
   * required and validated against io::ic::component), an optional per-sample
   * "Branches" subtree, an optional "Template" subtree ("File" required, "Norm"
   * one of MuonNorm|MuonGunNorm, default MuonNorm), an optional "Gradients"
   * subtree ("File") and an optional "Galactic" subtree (one child per template,
   * each with "File" and "Norm" in GalacticNorm0|GalacticNorm1).
   */
  [[nodiscard]] std::vector<SampleConfig> parse_samples(const boost::property_tree::ptree& ic);

}  // namespace io::ic
