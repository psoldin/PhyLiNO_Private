#pragma once

#include "../../io/IceCube/Binning.h"
#include "../../io/IceCube/ICParameter.h"
#include "../ParameterWrapper.h"

#include <array>
#include <span>
#include <string>
#include <vector>

namespace ana::ic {

  /**
   * SnowStorm detector-gradient systematics (NNMFit SnowStormGradient with
   * hist_parameter_overall: True), applied to one sample's summed prediction as
   * an additive perturbation of mu and sigma^2:
   *
   *   D_k       = (parameter_k - split_k) * lt_scale   k in {DOMEff .. HoleIceP1}
   *   mu_add_b  = sum_k D_k * gradient_k_b
   *   ssq_add_b = sum_k (D_k * gradient_error_k_b)^2 + 2 * sum_{i<j} D_i * D_j * cov_ij_b
   *
   * lt_scale is the analysis / gradient livetime ratio. The histogram-gradient
   * covariance term of NNMFit's formula is correctly absent: the FTP gradient
   * configs set external_gradients: True (the gradients come from independent MC).
   *
   * Gradients are per sample (each detector config has its own pickle; the
   * cascade samples share the _5up one) and are read from the text file produced
   * by tools/export_nnmfit_inputs.py, whose systematics order matches
   * params::ic {DOMEff .. HoleIceP1}. O(nBins) work: CPU only.
   *
   * `bin_scale` (empty = all ones) multiplies every gradient at load time. Its
   * only use is a sample carrying a topology cut whose gradients were exported
   * from the unfiltered sample: the gradients are absolute per-bin count deltas,
   * so they must be scaled to the surviving fraction of each bin. Errors scale
   * with it and the covariances with its square, so that mu and sigma^2 keep the
   * relation they had before. See SampleConfig::scale_gradients_to_topology.
   */
  class DetectorSystematics {
   public:
    DetectorSystematics(const io::ic::Binning& binning, const std::string& gradient_file,
                        std::span<const double> bin_scale = {});
    ~DetectorSystematics() = default;

    bool check_and_recalculate(const ParameterWrapper& parameter);

    /** Additive contribution to the predicted counts per bin. */
    [[nodiscard]] std::span<const double> mu_delta() const noexcept { return m_MuDelta; }

    /** Additive contribution to sigma^2 per bin (SAY only). */
    [[nodiscard]] std::span<const double> ssq_delta() const noexcept { return m_SsqDelta; }

   private:
    static constexpr int nPairs = params::ic::nDetSysParams * (params::ic::nDetSysParams - 1) / 2;

    double                                        m_LivetimeScale = 1.0;
    std::array<double, params::ic::nDetSysParams> m_Split{};

    std::array<std::vector<double>, params::ic::nDetSysParams> m_Gradient;
    std::array<std::vector<double>, params::ic::nDetSysParams> m_GradientError;
    std::array<std::vector<double>, nPairs>                    m_Covariance;

    std::vector<double> m_MuDelta;
    std::vector<double> m_SsqDelta;

    void load(const std::string& path, int total_bins);
  };

}  // namespace ana::ic
