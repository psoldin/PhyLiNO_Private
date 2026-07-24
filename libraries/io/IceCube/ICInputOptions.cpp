#include "ICInputOptions.h"

#include <stdexcept>

namespace io::ic {

  void ICInputOptions::read(const boost::program_options::variables_map& /*vm*/, const boost::property_tree::ptree& config) {
    const auto& ic = config.get_child("IceCube");

    m_TrackBaselineFilePath = ic.get<std::string>("TrackBaselineFilePath");
    m_UseData               = ic.get<bool>("UseData", false);

    const std::string likelihood_str = ic.get<std::string>("Likelihood", "Poisson");
    if (likelihood_str == "Poisson") {
      m_LikelihoodType = LikelihoodType::Poisson;
    } else if (likelihood_str == "SAY") {
      m_LikelihoodType = LikelihoodType::SAY;
    } else {
      throw std::runtime_error(
          "ICInputOptions: unknown Likelihood '" + likelihood_str + "' (expected 'Poisson' or 'SAY')");
    }

    m_Livetime = ic.get<double>("Livetime", m_Livetime);

    m_ERefGeV             = ic.get<double>("ERefGeV", m_ERefGeV);
    m_AstroReferenceIndex = ic.get<double>("AstroReferenceIndex", m_AstroReferenceIndex);
    m_AstroPerTypeNorm    = ic.get<bool>("AstroPerTypeNorm", m_AstroPerTypeNorm);

    m_ConvDeltaGammaERef   = ic.get<double>("ConvDeltaGammaERef", m_ConvDeltaGammaERef);
    m_PromptDeltaGammaERef = ic.get<double>("PromptDeltaGammaERef", m_PromptDeltaGammaERef);

    m_UseMuonTemplate        = ic.get<bool>("UseMuonTemplate", false);
    m_MuonTemplateFile       = ic.get<std::string>("MuonTemplateFile", "");
    m_UseDetectorSystematics = ic.get<bool>("UseDetectorSystematics", false);
    m_DetectorGradientFile   = ic.get<std::string>("DetectorGradientFile", "");
    m_UseOscillation         = ic.get<bool>("UseOscillation", false);
    m_OscillationSplineFile  = ic.get<std::string>("OscillationSplineFile", "");

    // Optional per-column branch-name overrides. Defaults (set in BranchNames)
    // match the tracks-only baseline parquet, so this subtree may be omitted.
    if (auto branches = config.get_child_optional("IceCube.Branches")) {
      const auto& br             = *branches;
      m_Branches.reco_energy     = br.get<std::string>("RecoEnergy", m_Branches.reco_energy);
      m_Branches.reco_zenith     = br.get<std::string>("RecoZenith", m_Branches.reco_zenith);
      m_Branches.true_energy     = br.get<std::string>("TrueEnergy", m_Branches.true_energy);
      m_Branches.astro_baseline  = br.get<std::string>("AstroBaseline", m_Branches.astro_baseline);
      m_Branches.conv_baseline   = br.get<std::string>("ConvBaseline", m_Branches.conv_baseline);
      m_Branches.conv_alt        = br.get<std::string>("ConvAlt", m_Branches.conv_alt);
      m_Branches.prompt_baseline = br.get<std::string>("PromptBaseline", m_Branches.prompt_baseline);
      m_Branches.prompt_alt      = br.get<std::string>("PromptAlt", m_Branches.prompt_alt);

      if (auto barr = br.get_child_optional("BarrConv")) {
        int k = 0;
        for (const auto& [_, child] : *barr) {
          if (k >= params::ic::nBarrParams)
            throw std::runtime_error("ICInputOptions: too many BarrConv branch names (expected 4)");
          m_Branches.barr_conv[k++] = child.get_value<std::string>();
        }
        if (k != params::ic::nBarrParams)
          throw std::runtime_error("ICInputOptions: expected exactly 4 BarrConv branch names");
      }
    }
  }

}  // namespace io::ic
