#pragma once

#include "../../io/IceCube/ICConstants.h"
#include "../ParameterWrapper.h"

#include <array>
#include <span>
#include <string>

namespace ana::ic {

  /**
   * Atmospheric-muon template component (NNMFit TemplateFlux, skip_syst).
   *
   *   muon_i = MuonNorm * template_i
   *
   * SCAFFOLD: the NNMFit muon template ships as a Python pickle that is not in
   * this repo. When disabled (or no file given) the component is a no-op and
   * contributes nothing. When enabled with a file, a plain-text template of
   * exactly Constants::nBins whitespace-separated values (row-major, E outer)
   * is read; the pickle reader is left for future work.
   */
  class MuonTemplate {
   public:
    MuonTemplate(bool enabled, const std::string& template_file);
    ~MuonTemplate() = default;

    bool check_and_recalculate(const ParameterWrapper& parameter);

    [[nodiscard]] std::span<const double> histogram() const noexcept { return m_Histogram; }
    [[nodiscard]] bool enabled() const noexcept { return m_Enabled; }

   private:
    using BinArray = std::array<double, io::ic::Constants::nBins>;

    bool     m_Enabled = false;
    BinArray m_Template{};   // fixed per-bin template counts
    BinArray m_Histogram{};  // MuonNorm * m_Template

    bool load_template(const std::string& path);
  };

}  // namespace ana::ic
