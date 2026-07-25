#include "SampleConfig.h"

#include <boost/property_tree/ptree.hpp>

#include <map>
#include <sstream>
#include <stdexcept>

#include "ICParameter.h"  // params::ic::nBarrParams

namespace io::ic {

  namespace {

    // Split on `delim`, trimming surrounding whitespace from each piece and
    // dropping empty pieces (so "" splits to an empty vector, not {""}).
    std::vector<std::string> split_trim(const std::string& s, char delim) {
      std::vector<std::string> out;
      std::istringstream       iss(s);
      std::string              item;
      while (std::getline(iss, item, delim)) {
        const auto start = item.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        const auto end = item.find_last_not_of(" \t");
        out.push_back(item.substr(start, end - start + 1));
      }
      return out;
    }

    // Factored out of the old flat "IceCube.Branches" parsing so it can be
    // reused per-sample; `node` is the node that may contain a "Branches"
    // child (either the top-level "IceCube" node or a "Samples.<name>" node).
    BranchNames parse_branches(const boost::property_tree::ptree& node) {
      BranchNames branches;
      if (auto branches_node = node.get_child_optional("Branches")) {
        const auto& br          = *branches_node;
        branches.reco_energy    = br.get<std::string>("RecoEnergy", branches.reco_energy);
        branches.reco_zenith    = br.get<std::string>("RecoZenith", branches.reco_zenith);
        branches.true_energy    = br.get<std::string>("TrueEnergy", branches.true_energy);
        branches.astro_baseline = br.get<std::string>("AstroBaseline", branches.astro_baseline);
        branches.conv_baseline  = br.get<std::string>("ConvBaseline", branches.conv_baseline);
        branches.conv_alt       = br.get<std::string>("ConvAlt", branches.conv_alt);
        branches.prompt_baseline = br.get<std::string>("PromptBaseline", branches.prompt_baseline);
        branches.prompt_alt      = br.get<std::string>("PromptAlt", branches.prompt_alt);

        if (auto barr = br.get_child_optional("BarrConv")) {
          int k = 0;
          for (const auto& [_, child] : *barr) {
            if (k >= params::ic::nBarrParams)
              throw std::runtime_error("parse_branches: too many BarrConv branch names (expected 4)");
            branches.barr_conv[k++] = child.get_value<std::string>();
          }
          if (k != params::ic::nBarrParams)
            throw std::runtime_error("parse_branches: expected exactly 4 BarrConv branch names");
        }
      }
      return branches;
    }

  }  // namespace

  std::vector<SampleConfig> parse_samples(const boost::property_tree::ptree& ic) {
    std::map<std::string, Binning> binnings;
    for (const auto& [binning_name, binning_node] : ic.get_child("Binnings")) {
      const std::string axes_str = binning_node.get<std::string>("axes");
      std::vector<Axis> axes;
      for (const auto& kind : split_trim(axes_str, ',')) {
        const std::string spec = binning_node.get<std::string>(kind);
        axes.push_back(parse_axis(kind, spec));
      }
      binnings.emplace(binning_name, Binning(std::move(axes)));
    }

    std::vector<SampleConfig> samples;
    for (const auto& [sample_name, sample_node] : ic.get_child("Samples")) {
      const std::string binning_name = sample_node.get<std::string>("binning");
      const auto        it           = binnings.find(binning_name);
      if (it == binnings.end())
        throw std::runtime_error(
            "parse_samples: sample '" + sample_name + "' references unknown binning '" + binning_name + "'");

      samples.push_back(SampleConfig{
          .name       = sample_name,
          .enabled    = sample_node.get<bool>("enabled", true),
          .binning    = it->second,
          .parquet    = sample_node.get<std::string>("parquet"),
          .data_path  = sample_node.get<std::string>("data", ""),
          .livetime   = sample_node.get<double>("livetime", 1.0),
          .components = split_trim(sample_node.get<std::string>("components", ""), ','),
          .branches   = parse_branches(sample_node),
      });
    }
    return samples;
  }

}  // namespace io::ic
