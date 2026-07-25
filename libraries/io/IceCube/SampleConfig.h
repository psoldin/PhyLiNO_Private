#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <boost/property_tree/ptree_fwd.hpp>

#include "Binning.h"
#include "BranchNames.h"

namespace io::ic {

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
  };

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
   * "data" (default ""), "livetime" (default 1.0), "components" (comma-split
   * string, default "") and an optional per-sample "Branches" subtree.
   */
  [[nodiscard]] std::vector<SampleConfig> parse_samples(const boost::property_tree::ptree& ic);

}  // namespace io::ic
