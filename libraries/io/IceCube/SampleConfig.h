#pragma once

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
   * NNMFit's cascade samples use `conventional_veto`/`prompt_veto` (the same
   * atmospheric flux times a passing-fraction reweight) and the `muon`/
   * `muontemplate` templates; those arrive with the cascade samples.
   */
  namespace component {
    inline constexpr std::string_view Astro        = "astro";
    inline constexpr std::string_view Conventional = "conventional";
    inline constexpr std::string_view Prompt       = "prompt";
  }  // namespace component

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
    std::string parquet;
    std::string data_path;
    double      livetime = 1.0;
    std::vector<std::string> components;
    BranchNames branches;

    [[nodiscard]] bool has_component(std::string_view component) const noexcept {
      for (const auto& c : components)
        if (c == component) return true;
      return false;
    }

    /** Astrophysical power law requested (drives PowerlawFlux + its parquet columns). */
    [[nodiscard]] bool wants_astro() const noexcept { return has_component(component::Astro); }

    /**
     * Conventional + prompt atmospheric flux requested. AtmosphericFlux computes
     * both in one per-event pass, so the two are enabled together; parse_samples()
     * rejects a sample that declares only one of them.
     */
    [[nodiscard]] bool wants_atmospheric() const noexcept {
      return has_component(component::Conventional) || has_component(component::Prompt);
    }
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
   * required and validated against io::ic::component) and an optional
   * per-sample "Branches" subtree.
   */
  [[nodiscard]] std::vector<SampleConfig> parse_samples(const boost::property_tree::ptree& ic);

}  // namespace io::ic
