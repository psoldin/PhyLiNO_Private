#pragma once

#include "../../io/IceCube/Binning.h"
#include "../ParameterWrapper.h"

#include <span>
#include <string>
#include <vector>

namespace ana::ic {

  /**
   * Binned template flux (NNMFit TemplateFlux): a fixed per-bin rate scaled by
   * one norm parameter, used for the atmospheric-muon templates -- Corsika
   * ("muontemplate", MuonNorm) for tracks and MuonGun ("muon", MuonGunNorm) for
   * the cascade samples.
   *
   *   mu_b  = norm * template_b * livetime
   *   ssq_b = (norm * fluctuation_b * livetime)^2
   *
   * matching NNMFit's histogram_builder, which multiplies the fluctuation graph
   * by the same parameters and adds its square to sigma^2. Template files are
   * produced by tools/export_nnmfit_inputs.py; both columns are rates (per
   * second), so the livetime scaling mirrors what ICDataBase does to the
   * per-event weights.
   *
   * O(nBins) work: CPU only, no GPU path.
   */
  class TemplateFlux {
   public:
    /**
     * `file_bins` maps the sample's bins onto the grid the file was exported in
     * (io::ic::make_bin_map). The default identity map is the usual case: the
     * file is already in the sample's own binning.
     */
    TemplateFlux(const io::ic::Binning& binning,
                 const std::string&     template_file,
                 int                    norm_index,
                 double                 livetime,
                 const io::ic::BinMap&  file_bins = io::ic::BinMap{});
    ~TemplateFlux() = default;

    /** Rescale for the current parameters; false when the norm did not change. */
    bool check_and_recalculate(const ParameterWrapper& parameter);

    /** Predicted counts per analysis bin from this template. */
    [[nodiscard]] std::span<const double> histogram() const noexcept { return m_Histogram; }

    /** sigma^2 contribution per analysis bin (zero if the file carried no fluctuations). */
    [[nodiscard]] std::span<const double> fluctuation() const noexcept { return m_Fluctuation; }

   private:
    int                 m_NormIndex;
    std::vector<double> m_Template;     // rate * livetime, per bin
    std::vector<double> m_Sigma;        // fluctuation rate * livetime, per bin
    std::vector<double> m_Histogram;    // norm * m_Template
    std::vector<double> m_Fluctuation;  // (norm * m_Sigma)^2

    void load(const std::string& path, int total_bins, double livetime, const io::ic::BinMap& file_bins);
  };

}  // namespace ana::ic
