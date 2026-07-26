#include "ICInputOptions.h"

#include <stdexcept>

#include "SampleConfig.h"  // parse_samples()

namespace io::ic {

  void ICInputOptions::read(const boost::program_options::variables_map& /*vm*/, const boost::property_tree::ptree& config) {
    const auto& ic = config.get_child("IceCube");

    m_UseData = ic.get<bool>("UseData", false);

    const std::string likelihood_str = ic.get<std::string>("Likelihood", "Poisson");
    if (likelihood_str == "Poisson") {
      m_LikelihoodType = LikelihoodType::Poisson;
    } else if (likelihood_str == "SAY") {
      m_LikelihoodType = LikelihoodType::SAY;
    } else {
      throw std::runtime_error(
          "ICInputOptions: unknown Likelihood '" + likelihood_str + "' (expected 'Poisson' or 'SAY')");
    }

    const std::string backend_str = ic.get<std::string>("Backend", "cpu");
    if (backend_str == "cpu") {
      m_BackendKind = BackendKind::Cpu;
    } else if (backend_str == "metal") {
      m_BackendKind = BackendKind::Metal;
    } else if (backend_str == "cuda") {
      m_BackendKind = BackendKind::Cuda;
    } else {
      throw std::runtime_error(
          "ICInputOptions: unknown Backend '" + backend_str + "' (expected 'cpu', 'metal' or 'cuda')");
    }

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

    // The samples (with their binnings, branch names and components) drive the
    // whole fit path: ICModule hands samples() to ICDataBase, which loads the
    // enabled ones. A config without them describes no analysis at all.
    if (!ic.get_child_optional("Samples"))
      throw std::runtime_error(
          "ICInputOptions: config has no \"IceCube.Samples\" section (see SampleConfig.h for the layout)");

    m_Samples = parse_samples(ic);
  }

}  // namespace io::ic
