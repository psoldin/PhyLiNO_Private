#pragma once

#include <string>
#include <vector>

#include "../InputOptionBase.h"
#include "SampleConfig.h"  // io::ic::SampleConfig; must be a complete type for the m_Samples vector member below

namespace io::ic {

  enum class LikelihoodType { Poisson,
                              SAY };

  // Compute backend for the per-event flux histograms.
  //   Cpu   - OMP+SIMD reference path, available everywhere.
  //   Metal - Apple GPU (Apple builds only).
  //   Cuda  - NVIDIA GPU (builds with the CUDA toolkit only).
  // A requested GPU backend falls back to Cpu at runtime if no device is present.
  enum class BackendKind { Cpu,
                           Metal,
                           Cuda };

  /**
   * @brief Input options of the IceCube diffuse-flux experiment.
   *
   * Follows the modular convention: default-constructed before the config is
   * parsed, then populated from the "IceCube" config section in read() --
   * global flux settings plus the list of analysis samples. All heavy input
   * loading happens later in ICDataBase, created by ICModule.
   */
  class ICInputOptions : public InputOptionBase {
   public:
    ICInputOptions()
      : InputOptionBase("IceCube Options") {}

    ~ICInputOptions() final = default;

    void read(const boost::program_options::variables_map& vm, const boost::property_tree::ptree& config) final;

    [[nodiscard]] bool           use_data() const noexcept { return m_UseData; }
    [[nodiscard]] LikelihoodType likelihood_type() const noexcept { return m_LikelihoodType; }
    // Selected compute backend for the flux histograms (see BackendKind).
    [[nodiscard]] BackendKind    backend_kind() const noexcept { return m_BackendKind; }

    // --- Astro (Powerlaw) ---
    [[nodiscard]] double e_ref_gev() const noexcept { return m_ERefGeV; }
    [[nodiscard]] double astro_reference_index() const noexcept { return m_AstroReferenceIndex; }
    [[nodiscard]] bool   astro_per_type_norm() const noexcept { return m_AstroPerTypeNorm; }

    // --- Atmospheric (conv + prompt) ---
    [[nodiscard]] double conv_delta_gamma_e_ref() const noexcept { return m_ConvDeltaGammaERef; }
    [[nodiscard]] double prompt_delta_gamma_e_ref() const noexcept { return m_PromptDeltaGammaERef; }

    // NNMFit effective_veto "additional" block: the passing-fraction expansion
    // point and the scale the fit parameter is exponentiated against, in GeV.
    // Shared by every veto-reweighted sample.
    [[nodiscard]] double veto_anchor_energy() const noexcept { return m_VetoAnchorEnergy; }
    [[nodiscard]] double veto_rescale_energy() const noexcept { return m_VetoRescaleEnergy; }

    // --- Scaffolded components (no-op until enabled AND a file is provided) ---
    [[nodiscard]] bool               use_detector_systematics() const noexcept { return m_UseDetectorSystematics; }
    [[nodiscard]] const std::string& detector_gradient_file() const noexcept { return m_DetectorGradientFile; }

    // Analysis samples in config order, each with its own binning, parquet,
    // livetime, branch names and component list (see parse_samples() in
    // SampleConfig.h). Drives the whole fit path: ICModule feeds these to
    // ICDataBase, which loads the enabled ones.
    [[nodiscard]] const std::vector<SampleConfig>& samples() const noexcept { return m_Samples; }

   private:
    bool           m_UseData        = false;
    LikelihoodType m_LikelihoodType = LikelihoodType::Poisson;
    BackendKind    m_BackendKind    = BackendKind::Cpu;

    double m_ERefGeV             = 1.0e5;
    double m_AstroReferenceIndex = 2.0;
    bool   m_AstroPerTypeNorm    = false;

    double m_ConvDeltaGammaERef   = 1000.0;
    double m_PromptDeltaGammaERef = 3800.0;

    double m_VetoAnchorEnergy  = 100.0;
    double m_VetoRescaleEnergy = 100.0;

    bool        m_UseDetectorSystematics = false;
    std::string m_DetectorGradientFile;

    std::vector<SampleConfig> m_Samples;
  };

}  // namespace io::ic
