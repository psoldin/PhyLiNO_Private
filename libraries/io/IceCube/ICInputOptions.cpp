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

    const std::string precision_str = ic.get<std::string>("GpuPrecision", "fp32");
    if (precision_str == "fp32") {
      m_GpuPrecision = GpuPrecision::Fp32;
    } else if (precision_str == "fp64") {
      m_GpuPrecision = GpuPrecision::Fp64;
    } else {
      throw std::runtime_error(
          "ICInputOptions: unknown GpuPrecision '" + precision_str + "' (expected 'fp32' or 'fp64')");
    }
    if (m_GpuPrecision == GpuPrecision::Fp64 && m_BackendKind == BackendKind::Metal)
      throw std::runtime_error(
          "ICInputOptions: GpuPrecision 'fp64' is not supported by the Metal backend (Apple GPUs have no double precision)");

    const std::string astro_model_str = ic.get<std::string>("AstroModel", "Powerlaw");
    if (astro_model_str == "Powerlaw") {
      m_AstroModel = AstroModel::Powerlaw;
    } else if (astro_model_str == "BrokenPowerlaw") {
      m_AstroModel = AstroModel::BrokenPowerlaw;
    } else {
      throw std::runtime_error(
          "ICInputOptions: unknown AstroModel '" + astro_model_str + "' (expected 'Powerlaw' or 'BrokenPowerlaw')");
    }

    m_ERefGeV             = ic.get<double>("ERefGeV", m_ERefGeV);
    m_AstroReferenceIndex = ic.get<double>("AstroReferenceIndex", m_AstroReferenceIndex);
    m_AstroPerTypeNorm    = ic.get<bool>("AstroPerTypeNorm", m_AstroPerTypeNorm);

    m_ConvDeltaGammaERef   = ic.get<double>("ConvDeltaGammaERef", m_ConvDeltaGammaERef);
    m_PromptDeltaGammaERef = ic.get<double>("PromptDeltaGammaERef", m_PromptDeltaGammaERef);

    m_VetoAnchorEnergy  = ic.get<double>("VetoAnchorEnergy", m_VetoAnchorEnergy);
    m_VetoRescaleEnergy = ic.get<double>("VetoRescaleEnergy", m_VetoRescaleEnergy);

    m_UseDetectorSystematics = ic.get<bool>("UseDetectorSystematics", false);
    m_DetectorGradientFile   = ic.get<std::string>("DetectorGradientFile", "");

    // The samples (with their binnings, branch names and components) drive the
    // whole fit path: ICModule hands samples() to ICDataBase, which loads the
    // enabled ones. A config without them describes no analysis at all.
    if (!ic.get_child_optional("Samples"))
      throw std::runtime_error(
          "ICInputOptions: config has no \"IceCube.Samples\" section (see SampleConfig.h for the layout)");

    m_Samples = parse_samples(ic);
  }

}  // namespace io::ic
