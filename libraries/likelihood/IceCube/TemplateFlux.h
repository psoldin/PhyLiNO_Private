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
   * Shape modes (PhyLiNO only, no NNMFit counterpart). A file whose header has
   * "# template modes <M>" carries M more columns d_j, rates like the others,
   * each the 1-sigma shift of the template along one direction of its own
   * correlated uncertainty. The template then is
   *
   *   template_b(theta) = max(template_b + sum_j theta_j * d_jb, 0),
   *   theta_j = parameter MuonTmplMode1 + j  (Gaussian-constrained in the config),
   *
   * and mu_b = norm * template_b(theta) * livetime; the fluctuation column is not
   * affected.
   *
   * A header line "# template mode_forms <form_1> ... <form_M>" (each "additive" or
   * "multiplicative", default all additive) makes a mode act on the logarithm
   * instead: with r_jb = ln((template_b + d_jb) / template_b),
   *
   *   template_b(theta) = max(template_b + sum_additive theta_j d_jb, 0)
   *                       * exp(sum_multiplicative theta_k r_kb).
   *
   * Both forms give template + d at theta = 1, but a multiplicative mode never
   * drives a bin negative (theta = -1 gives template^2 / (template + d)), which
   * matters where d is several times the template. It needs template_b > 0 and
   * template_b + d_jb > 0 in every bin; the file is rejected otherwise. Only MuonNorm's template may carry modes: the mode parameters are
   * global, and a second template reading them would tie two unrelated shapes.
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

    /**
     * The norm-free template at the current mode values, rate * livetime per bin:
     * d histogram() / d norm. Without modes, the template as read.
     */
    [[nodiscard]] std::span<const double> rates() const noexcept { return m_Shaped; }

    /** Number of mode columns the file carried (0 for a plain template). */
    [[nodiscard]] int mode_count() const noexcept { return static_cast<int>(m_Modes.size()); }

    /** Parameter index of mode j. */
    [[nodiscard]] static int mode_index(int j) noexcept;

    /** True if mode j acts multiplicatively (header "mode_forms"). */
    [[nodiscard]] bool mode_multiplicative(const int j) const noexcept {
      return m_Multiplicative[static_cast<std::size_t>(j)] != 0;
    }

    /**
     * d histogram() / d theta_j per bin, added to `out`. Additive mode: norm * d_jb *
     * livetime * F_b where the additive part is positive, 0 where the clip at 0 holds
     * it (F_b the multiplicative factor). Multiplicative mode: norm * shaped_b * r_jb.
     */
    void add_mode_derivative(int j, double norm, std::span<double> out) const;

    /** The norm-free fluctuation, sigma * livetime per bin: fluctuation() = (norm * sigmas())^2. */
    [[nodiscard]] std::span<const double> sigmas() const noexcept { return m_Sigma; }

    /** Index of the parameter this template's norm is. */
    [[nodiscard]] int norm_index() const noexcept { return m_NormIndex; }

   private:
    int                 m_NormIndex;
    std::vector<double> m_Template;     // rate * livetime, per bin
    std::vector<std::vector<double>> m_Modes;       // d_j * livetime (additive) or r_j (multiplicative)
    std::vector<char>                m_Multiplicative;  // per mode: 1 if it acts on the logarithm
    std::vector<double>              m_ModeValues;  // theta_j the shaped template was built at
    std::vector<double> m_Linear;       // max(m_Template + sum_additive theta_j m_Modes[j], 0)
    std::vector<double> m_Factor;       // exp(sum_multiplicative theta_k m_Modes[k])
    std::vector<double> m_Shaped;       // m_Linear * m_Factor
    std::vector<double> m_Sigma;        // fluctuation rate * livetime, per bin
    std::vector<double> m_Histogram;    // norm * m_Template
    std::vector<double> m_Fluctuation;  // (norm * m_Sigma)^2

    void load(const std::string& path, int total_bins, double livetime, const io::ic::BinMap& file_bins);
  };

}  // namespace ana::ic
