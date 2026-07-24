#pragma once

#include "../../io/IceCube/ICConstants.h"
#include "../../io/IceCube/ICParameter.h"
#include "../ParameterWrapper.h"

#include <array>
#include <span>
#include <string>

namespace ana::ic {

  /**
   * SnowStorm detector-gradient systematics (NNMFit SnowStormGradient), applied
   * at histogram level as an additive perturbation of the predicted counts:
   *
   *   mu_delta_b = sum_k (param_k - split_k) * gradient_k_b
   *
   * with k in {DOMEff, IceAbs, IceScat}. Because the tracks-only fit uses a
   * plain PoissonLLH, the sigma^2 fluctuation term of the gradient is ignored.
   *
   * SCAFFOLD: the pre-computed gradients live in a Python pickle not present in
   * this repo. When disabled (or no file given) the delta is identically zero.
   * When enabled with a file, a plain-text file of nDetSysParams*nBins values
   * (systematic-major: all bins for DOMEff, then IceAbs, then IceScat) is read;
   * the pickle reader is left for future work.
   */
  class DetectorSystematics {
   public:
    DetectorSystematics(bool enabled, const std::string& gradient_file);
    ~DetectorSystematics() = default;

    bool check_and_recalculate(const ParameterWrapper& parameter);

    [[nodiscard]] std::span<const double> delta() const noexcept { return m_Delta; }
    [[nodiscard]] bool enabled() const noexcept { return m_Enabled; }

   private:
    using BinArray = std::array<double, io::ic::Constants::nBins>;

    bool m_Enabled = false;
    // Split (reference) values the gradients were computed at; NNMFit FTP config
    // uses 1.0 for DOM efficiency, absorption and scattering.
    std::array<double, params::ic::nDetSysParams> m_Split{{1.0, 1.0, 1.0}};
    std::array<BinArray, params::ic::nDetSysParams> m_Gradients{};  // per-systematic, per-bin
    BinArray m_Delta{};  // additive contribution to the total prediction

    bool load_gradients(const std::string& path);
  };

}  // namespace ana::ic
