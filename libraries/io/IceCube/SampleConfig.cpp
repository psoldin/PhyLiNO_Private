#include "SampleConfig.h"

#include <boost/algorithm/string/predicate.hpp>
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
        const auto& br           = *branches_node;
        branches.reco_energy     = br.get<std::string>("RecoEnergy", branches.reco_energy);
        branches.reco_zenith     = br.get<std::string>("RecoZenith", branches.reco_zenith);
        branches.reco_ra         = br.get<std::string>("RecoRa", branches.reco_ra);
        branches.true_energy     = br.get<std::string>("TrueEnergy", branches.true_energy);
        branches.astro_baseline  = br.get<std::string>("AstroBaseline", branches.astro_baseline);
        branches.conv_baseline   = br.get<std::string>("ConvBaseline", branches.conv_baseline);
        branches.conv_alt        = br.get<std::string>("ConvAlt", branches.conv_alt);
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

    bool is_known_component(const std::string& name) noexcept {
      return name == component::Astro || name == component::Conventional || name == component::Prompt ||
             name == component::ConventionalVeto || name == component::PromptVeto ||
             name == component::MuonTemplate || name == component::MuonGun;
    }

    // Reject unknown component names and combinations the flux components
    // cannot express, so a config typo (or a component that is not implemented
    // yet) fails at startup instead of quietly producing a smaller prediction.
    void validate_components(const SampleConfig& sample) {
      if (sample.components.empty())
        throw std::runtime_error("parse_samples: sample '" + sample.name +
                                 "' declares no components (expected a \"components\" list, e.g. "
                                 "\"astro, conventional, prompt\")");

      for (const std::string& c : sample.components) {
        if (!is_known_component(c))
          throw std::runtime_error("parse_samples: sample '" + sample.name + "' declares unknown component '" + c +
                                   "' (supported: astro, conventional, prompt, conventional_veto, prompt_veto, "
                                   "muontemplate, muon)");
      }

      const bool plain = sample.has_component(component::Conventional);
      const bool veto  = sample.has_component(component::ConventionalVeto);

      if (plain != sample.has_component(component::Prompt))
        throw std::runtime_error("parse_samples: sample '" + sample.name +
                                 "' declares only one of 'conventional'/'prompt'; AtmosphericFlux computes both in "
                                 "one pass, so they must be enabled together");
      if (veto != sample.has_component(component::PromptVeto))
        throw std::runtime_error("parse_samples: sample '" + sample.name +
                                 "' declares only one of 'conventional_veto'/'prompt_veto'; they must be enabled "
                                 "together");
      if (plain && veto)
        throw std::runtime_error("parse_samples: sample '" + sample.name +
                                 "' declares both the plain and the veto atmospheric components; NNMFit excludes one "
                                 "variant per sample and enabling both would double-count");

      if (sample.has_component(component::MuonTemplate) && sample.has_component(component::MuonGun))
        throw std::runtime_error("parse_samples: sample '" + sample.name +
                                 "' declares two muon templates ('muontemplate' and 'muon'); pick one");

      if (sample.wants_template() && sample.template_file.empty())
        throw std::runtime_error("parse_samples: sample '" + sample.name +
                                 "' declares a muon template but has no \"Template\": { \"File\": ... } entry");
      if (!sample.wants_template() && !sample.template_file.empty())
        throw std::runtime_error("parse_samples: sample '" + sample.name +
                                 "' has a \"Template\" entry but declares neither 'muontemplate' nor 'muon'");

      if (sample.scale_gradients_to_topology && !sample.filters_topology())
        throw std::runtime_error("parse_samples: sample '" + sample.name +
                                 "' sets Gradients.ScaleToTopology but has no topology cut; there is nothing to "
                                 "rescale to and the flag would hide a later config mistake");
      if (sample.scale_gradients_to_topology && sample.gradient_file.empty())
        throw std::runtime_error("parse_samples: sample '" + sample.name +
                                 "' sets Gradients.ScaleToTopology but no Gradients.File");

      if (sample.filters_topology() && sample.topology_branch.empty())
        throw std::runtime_error("parse_samples: sample '" + sample.name +
                                 "' has Topology.Values but an empty Topology.Branch");
      if (!sample.topology_branch.empty() && !sample.filters_topology())
        throw std::runtime_error("parse_samples: sample '" + sample.name +
                                 "' has a Topology.Branch but no Topology.Values; the cut would keep nothing");

      // Every pre-binned input was exported from the full sample. Cutting events
      // out of the parquet without re-exporting them would leave the templates,
      // the gradients and the counts describing a selection the prediction no
      // longer has, which is a normalisation error no gate downstream can see.
      if (sample.filters_topology()) {
        if (!sample.template_file.empty())
          throw std::runtime_error("parse_samples: sample '" + sample.name +
                                   "' combines a topology cut with a muon template; the template was exported from "
                                   "the unfiltered sample and would no longer match. Re-export it, then drop this "
                                   "check for that file");
        if (!sample.gradient_file.empty() && !sample.scale_gradients_to_topology)
          throw std::runtime_error("parse_samples: sample '" + sample.name +
                                   "' combines a topology cut with a SnowStorm gradient file; the gradients were "
                                   "exported from the unfiltered sample and would no longer match. Re-export them, "
                                   "or set Gradients.ScaleToTopology to rescale them by each bin's surviving "
                                   "weight fraction (an approximation, see SampleConfig)");
        if (!sample.galactic.empty())
          throw std::runtime_error("parse_samples: sample '" + sample.name +
                                   "' combines a topology cut with a galactic template; the template was exported "
                                   "from the unfiltered sample and would no longer match. Re-export it, then drop "
                                   "this check for that file");
        if (!sample.data_counts_path.empty())
          throw std::runtime_error("parse_samples: sample '" + sample.name +
                                   "' combines a topology cut with pre-binned \"DataCounts\"; the counts are not "
                                   "filtered by this framework, so the cut would only apply to the prediction");
      }

      if (!sample.galactic.empty() && !has_ra_axis(sample.binning))
        throw std::runtime_error("parse_samples: sample '" + sample.name +
                                 "' declares a galactic template but its binning has no Ra axis; the "
                                 "template is stored in the analysis binning and would not fit");

      for (std::size_t i = 0; i < sample.galactic.size(); ++i)
        for (std::size_t j = i + 1; j < sample.galactic.size(); ++j)
          if (sample.galactic[i].norm_index == sample.galactic[j].norm_index)
            throw std::runtime_error("parse_samples: sample '" + sample.name + "' galactic templates '" +
                                     sample.galactic[i].name + "' and '" + sample.galactic[j].name +
                                     "' share one norm parameter; the fit could not tell them apart");

      // The unbinned density is a sum over per-event weights, and these three
      // inputs have no per-event representation at all: the muon and galactic
      // templates are per-bin rate files, and the SnowStorm gradients come from
      // independent MC, so there is no event to attach them to even in
      // principle. Silently dropping them would fit a different model than the
      // config asks for.
      if (sample.unbinned.enabled) {
        const char* offender = nullptr;
        if (!sample.template_file.empty()) offender = "a muon template (\"Template\")";
        else if (!sample.gradient_file.empty()) offender = "SnowStorm gradients (\"Gradients\")";
        else if (!sample.galactic.empty()) offender = "galactic templates (\"Galactic\")";
        if (offender != nullptr)
          throw std::runtime_error("parse_samples: sample '" + sample.name + "' combines Unbinned with " +
                                   std::string(offender) +
                                   ", which is a per-bin input with no per-event representation; fix the "
                                   "corresponding parameters and remove the block");
      }
    }

    // "Template": { "File": ..., "Norm": "MuonNorm"|"MuonGunNorm" },
    // "Gradients": { "File": ... }, "Oscillations": { "File": ..., "Branch": ... },
    // "Topology": { "Branch": ..., "Values": "1, 2" } and "Galactic": { ... },
    // all optional.
    void parse_component_files(const boost::property_tree::ptree& node, SampleConfig& sample) {
      if (const auto tmpl = node.get_child_optional("Template")) {
        sample.template_file = tmpl->get<std::string>("File");

        const std::string norm = tmpl->get<std::string>("Norm", "MuonNorm");
        if (norm == "MuonNorm")
          sample.template_norm_index = params::ic::MuonNorm;
        else if (norm == "MuonGunNorm")
          sample.template_norm_index = params::ic::MuonGunNorm;
        else
          throw std::runtime_error("parse_samples: sample '" + sample.name + "' Template.Norm '" + norm +
                                   "' is not a known template norm (expected MuonNorm or MuonGunNorm)");
      }

      if (const auto gradients = node.get_child_optional("Gradients")) {
        sample.gradient_file                = gradients->get<std::string>("File");
        sample.scale_gradients_to_topology  = gradients->get<bool>("ScaleToTopology", false);
      }

      if (const auto osc = node.get_child_optional("Oscillations")) {
        sample.oscillation_file   = osc->get<std::string>("File");
        sample.oscillation_branch = osc->get<std::string>("Branch", sample.oscillation_branch);
      }

      if (const auto topology = node.get_child_optional("Topology")) {
        sample.topology_branch  = topology->get<std::string>("Branch");
        sample.topology_exclude = topology->get<bool>("Exclude", false);

        for (const std::string& value : split_trim(topology->get<std::string>("Values"), ',')) {
          if (boost::iequals(value, "NaN")) {
            sample.topology_drop_nan = true;
            continue;
          }
          std::size_t consumed = 0;
          int         parsed   = 0;
          try {
            parsed = std::stoi(value, &consumed);
          } catch (const std::exception&) {
            consumed = 0;
          }
          if (consumed != value.size())
            throw std::runtime_error("parse_samples: sample '" + sample.name + "' Topology.Values entry '" + value +
                                     "' is not an integer class label or \"NaN\"");
          sample.topology_values.push_back(parsed);
        }
      }

      if (const auto unbinned = node.get_child_optional("Unbinned")) {
        auto parse_transform = [&sample](const std::string& text, const std::string& key) {
          if (text == "none") return SigmaTransform::None;
          if (text == "exp") return SigmaTransform::Exp;
          if (text == "pow10") return SigmaTransform::Pow10;
          if (text == "linear_to_dex") return SigmaTransform::LinearToDex;
          if (text == "deg2rad") return SigmaTransform::DegToRad;
          throw std::runtime_error("parse_samples: sample '" + sample.name + "' has Unbinned." + key + " '" +
                                   text + "' (expected none, exp, pow10, linear_to_dex or deg2rad)");
        };

        UnbinnedConfig& u   = sample.unbinned;
        u.enabled           = unbinned->get<bool>("enabled", true);
        u.energy_sigma_branch = unbinned->get<std::string>("EnergySigmaBranch", u.energy_sigma_branch);
        u.zenith_sigma_branch = unbinned->get<std::string>("ZenithSigmaBranch", u.zenith_sigma_branch);
        u.log_e_lo          = unbinned->get<double>("Log10EnergyLo", u.log_e_lo);
        u.log_e_hi          = unbinned->get<double>("Log10EnergyHi", u.log_e_hi);
        u.zenith_lo         = unbinned->get<double>("ZenithLo", u.zenith_lo);
        u.zenith_hi         = unbinned->get<double>("ZenithHi", u.zenith_hi);
        u.energy_sigma_transform =
            parse_transform(unbinned->get<std::string>("EnergySigmaTransform", "exp"), "EnergySigmaTransform");
        u.zenith_sigma_transform =
            parse_transform(unbinned->get<std::string>("ZenithSigmaTransform", "deg2rad"), "ZenithSigmaTransform");
        u.truncation        = unbinned->get<double>("Truncation", u.truncation);
        u.thinning          = unbinned->get<int>("Thinning", u.thinning);

        if (u.truncation <= 0.0)
          throw std::runtime_error("parse_samples: sample '" + sample.name +
                                   "' has Unbinned.Truncation <= 0; nothing would contribute to the density");
        if (u.thinning < 1)
          throw std::runtime_error("parse_samples: sample '" + sample.name + "' has Unbinned.Thinning < 1");

        if (u.energy_sigma_branch.empty())
          throw std::runtime_error("parse_samples: sample '" + sample.name +
                                   "' has an empty Unbinned.EnergySigmaBranch");
        if (u.zenith_sigma_branch.empty())
          throw std::runtime_error("parse_samples: sample '" + sample.name +
                                   "' has an empty Unbinned.ZenithSigmaBranch");

        // The binning axis this feeds is CosZenith (Binning.cpp applies std::cos to
        // the raw angle), but the KDE's zenith coordinate is the raw angle itself, in
        // radians. Pasting the CosZenith axis bounds in here (e.g. -1.0, 0.0872) would
        // pass the hi > lo check below while silently building the wrong-unit domain.
        constexpr double kPi = 3.14159265358979323846;
        if (u.zenith_lo < 0.0 || u.zenith_lo > kPi || u.zenith_hi < 0.0 || u.zenith_hi > kPi)
          throw std::runtime_error("parse_samples: sample '" + sample.name +
                                   "' has Unbinned.ZenithLo/ZenithHi outside [0, pi]; these are radians, not "
                                   "the sample's CosZenith binning bounds");

        if (u.log_e_hi <= u.log_e_lo)
          throw std::runtime_error("parse_samples: sample '" + sample.name +
                                   "' has an empty Unbinned energy domain (Log10EnergyHi <= Log10EnergyLo)");
        if (u.zenith_hi <= u.zenith_lo)
          throw std::runtime_error("parse_samples: sample '" + sample.name +
                                   "' has an empty Unbinned zenith domain (ZenithHi <= ZenithLo)");
      }

      if (const auto galactic = node.get_child_optional("Galactic")) {
        for (const auto& [template_name, template_node] : *galactic) {
          GalacticTemplateConfig entry{.name = template_name,
                                       .file = template_node.get<std::string>("File")};

          const std::string norm = template_node.get<std::string>("Norm");
          if (norm == "GalacticNorm0")
            entry.norm_index = params::ic::GalacticNorm0;
          else if (norm == "GalacticNorm1")
            entry.norm_index = params::ic::GalacticNorm1;
          else
            throw std::runtime_error("parse_samples: sample '" + sample.name + "' galactic template '" +
                                     template_name + "' has Norm '" + norm +
                                     "' (expected GalacticNorm0 or GalacticNorm1)");

          sample.galactic.push_back(std::move(entry));
        }
      }
    }

  }  // namespace

  std::vector<std::size_t> enabled_sample_indices(const std::vector<SampleConfig>& samples) {
    std::vector<std::size_t> enabled;
    enabled.reserve(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i)
      if (samples[i].enabled) enabled.push_back(i);
    return enabled;
  }

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
      const auto binning_name = sample_node.get<std::string>("binning");
      const auto it           = binnings.find(binning_name);
      if (it == binnings.end())
        throw std::runtime_error(
            "parse_samples: sample '" + sample_name + "' references unknown binning '" + binning_name + "'");

      samples.push_back(SampleConfig {
          .name             = sample_name,
          .enabled          = sample_node.get<bool>("enabled", true),
          .binning          = it->second,
          .mc_binning       = drop_ra_axis(it->second),
          .parquet          = sample_node.get<std::string>("parquet"),
          .data_path        = sample_node.get<std::string>("data", ""),
          .data_counts_path = sample_node.get<std::string>("DataCounts", ""),
          .livetime         = sample_node.get<double>("livetime", 1.0),
          .components       = split_trim(sample_node.get<std::string>("components", ""), ','),
          .branches         = parse_branches(sample_node),
      });

      SampleConfig& sample = samples.back();
      parse_component_files(sample_node, sample);
      validate_components(sample);
    }
    return samples;
  }

}  // namespace io::ic
