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
   * Unbinned extended-KDE likelihood settings for one sample (the "Unbinned"
   * config block). Absent block, or enabled == false, leaves the sample on the
   * binned Poisson/SAY path.
   *
   * The KDE energy coordinate is log10 of the sample's ordinary reco energy
   * branch (Branches.RecoEnergy), so the binned and unbinned paths cannot end
   * up scoring different observables. Only the two per-event uncertainty
   * columns are named here, since nothing but this path reads them.
   */
  /**
   * How a per-event uncertainty column is turned into a kernel bandwidth. The
   * conventions differ per column and are not self-describing, so they are
   * config rather than folklore:
   *
   *   None        the column already is the bandwidth in the KDE's own units
   *   Exp         exp(x): a network that emits log(sigma) to keep it positive,
   *               which is what ELEFANTS does (its sigma column is negative for
   *               99.99% of rows, so it cannot be a width)
   *   Pow10       10^x, for a column carrying log10(sigma)
   *   LinearToDex x / (E_reco * ln10): a sigma in linear GeV, converted to the
   *               log10 energy axis by the delta method
   *   DegToRad    x * pi/180, for an angular sigma in degrees
   */
  enum class SigmaTransform { None, Exp, Pow10, LinearToDex, DegToRad };

  struct UnbinnedConfig {
    bool        enabled             = false;
    std::string energy_sigma_branch = "ELEFANTS_tg_sigma_log10";  // log(sigma / dex)
    std::string zenith_sigma_branch = "L5_sigma_paraboloid";      // degrees

    // ELEFANTS is trained on log10(E) with a Gaussian head that emits log(sigma),
    // and the paraboloid sigma is in degrees (median 0.61, i.e. 0.61 deg; as
    // radians it would be 35 deg, wider than any real MPEFit error).
    SigmaTransform energy_sigma_transform = SigmaTransform::Exp;
    SigmaTransform zenith_sigma_transform = SigmaTransform::DegToRad;

    // KDE domain in KDE coordinates: log10(E_reco / GeV) and zenith in radians.
    // Events outside are dropped at load, as out-of-binning events already are.
    double log_e_lo  = 2.0;
    double log_e_hi  = 7.0;
    double zenith_lo = 0.0;
    double zenith_hi = 3.14159265358979323846;

    // Kernels beyond truncation * h are skipped. Large values recover the exact
    // double sum (at the corresponding cost) and are how the truncation is validated.
    double truncation = 5.0;

    // Precompute the kernel matrix at load and reduce each evaluation to a
    // sparse matrix-vector product (see KdeMatrix). Worth ~50x on the CPU, at
    // the price of holding nnz entries in RAM -- which only fits once the
    // quadrature is thinned, so the two knobs are chosen together.
    bool   matrix           = false;
    double matrix_budget_gb = 8.0;

    // Use every thinning-th MC event as a quadrature node, with its weight
    // scaled by thinning. Cuts runtime linearly, adds MC-integration noise,
    // leaves the statistical content unchanged. 1 = every event.
    int thinning = 1;
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

    // Unbinned extended-KDE likelihood ("Unbinned" block); disabled by default.
    UnbinnedConfig unbinned;

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
