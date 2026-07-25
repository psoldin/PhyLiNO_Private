#pragma once

#include <array>
#include <string>

#include "../InputOptionBase.h"
#include "ICParameter.h"  // params::ic::nBarrParams

namespace io::ic {

  /**
   * Names of the parquet columns read into ICSample. Defaults match the
   * tracks-only baseline dataset (dataset_tracks_baseline.parquet); override
   * any of them via the "IceCube.Branches" config subtree if the schema differs.
   */
  struct BranchNames {
    std::string reco_energy     = "energy_truncated";
    std::string reco_zenith     = "zenith_MPEFit";
    std::string true_energy     = "MCPrimaryEnergy";
    std::string astro_baseline  = "powerlaw";
    std::string conv_baseline   = "mceq_conv_H4a_SIBYLL23c";
    std::string conv_alt        = "mceq_conv_GST4_SIBYLL23c";
    std::string prompt_baseline = "mceq_pr_H4a_SIBYLL23c";
    std::string prompt_alt      = "mceq_pr_GST4_SIBYLL23c";
    // Conventional Barr gradients, order matches params::ic {BarrH, BarrW, BarrY, BarrZ}.
    std::array<std::string, params::ic::nBarrParams> barr_conv = {
        "barr_h_mceq_H4a_SIBYLL23c",
        "barr_w_mceq_H4a_SIBYLL23c",
        "barr_y_mceq_H4a_SIBYLL23c",
        "barr_z_mceq_H4a_SIBYLL23c"};
  };

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
   * @brief Input options of the IceCube tracks-only diffuse-flux experiment.
   *
   * Follows the modular convention: default-constructed before the config is
   * parsed, then populated from the "IceCube" config section in read(). All
   * heavy input loading happens later in ICDataBase, created by ICModule.
   */
  class ICInputOptions : public InputOptionBase {
   public:
    ICInputOptions()
      : InputOptionBase("IceCube Options") {}

    ~ICInputOptions() final = default;

    void read(const boost::program_options::variables_map& vm, const boost::property_tree::ptree& config) final;

    [[nodiscard]] const std::string& track_baseline_file_path() const noexcept { return m_TrackBaselineFilePath; }
    [[nodiscard]] const BranchNames& branch_names() const noexcept { return m_Branches; }
    [[nodiscard]] bool               use_data() const noexcept { return m_UseData; }
    [[nodiscard]] LikelihoodType     likelihood_type() const noexcept { return m_LikelihoodType; }
    // Selected compute backend for the flux histograms (see BackendKind).
    [[nodiscard]] BackendKind        backend_kind() const noexcept { return m_BackendKind; }
    // Detector livetime in seconds. The per-event MC weights are rates (Hz);
    // multiplying by the livetime turns the binned prediction into event counts.
    [[nodiscard]] double livetime() const noexcept { return m_Livetime; }

    // --- Astro (Powerlaw) ---
    [[nodiscard]] double e_ref_gev() const noexcept { return m_ERefGeV; }
    [[nodiscard]] double astro_reference_index() const noexcept { return m_AstroReferenceIndex; }
    [[nodiscard]] bool   astro_per_type_norm() const noexcept { return m_AstroPerTypeNorm; }

    // --- Atmospheric (conv + prompt) ---
    [[nodiscard]] double conv_delta_gamma_e_ref() const noexcept { return m_ConvDeltaGammaERef; }
    [[nodiscard]] double prompt_delta_gamma_e_ref() const noexcept { return m_PromptDeltaGammaERef; }

    // --- Scaffolded components (no-op until enabled AND a file is provided) ---
    [[nodiscard]] bool               use_muon_template() const noexcept { return m_UseMuonTemplate; }
    [[nodiscard]] const std::string& muon_template_file() const noexcept { return m_MuonTemplateFile; }
    [[nodiscard]] bool               use_detector_systematics() const noexcept { return m_UseDetectorSystematics; }
    [[nodiscard]] const std::string& detector_gradient_file() const noexcept { return m_DetectorGradientFile; }
    [[nodiscard]] bool               use_oscillation() const noexcept { return m_UseOscillation; }
    [[nodiscard]] const std::string& oscillation_spline_file() const noexcept { return m_OscillationSplineFile; }

   private:
    std::string    m_TrackBaselineFilePath;
    BranchNames    m_Branches;
    bool           m_UseData          = false;
    LikelihoodType m_LikelihoodType   = LikelihoodType::Poisson;
    BackendKind    m_BackendKind      = BackendKind::Cpu;
    double         m_Livetime         = 1.0;

    double m_ERefGeV             = 1.0e5;
    double m_AstroReferenceIndex = 2.0;
    bool   m_AstroPerTypeNorm    = false;

    double m_ConvDeltaGammaERef   = 1000.0;
    double m_PromptDeltaGammaERef = 3800.0;

    bool        m_UseMuonTemplate = false;
    std::string m_MuonTemplateFile;
    bool        m_UseDetectorSystematics = false;
    std::string m_DetectorGradientFile;
    bool        m_UseOscillation = false;
    std::string m_OscillationSplineFile;
  };

}  // namespace io::ic
