#include "DetectorSystematics.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace ana::ic {

  namespace {

    // The order tools/export_nnmfit_inputs.py writes, matching params::ic.
    constexpr const char* kSystematicNames[params::ic::nDetSysParams] = {
        "DOMEfficiency", "IceAbsorption", "IceScattering", "HoleIceForward_p0", "HoleIceForward_p1"};

    // Next non-empty line, comments included (the parser needs the section markers).
    bool next_line(std::istream& in, std::string& line) {
      while (std::getline(in, line))
        if (!line.empty()) return true;
      return false;
    }

  }  // namespace

  DetectorSystematics::DetectorSystematics(const io::ic::Binning&        binning,
                                           const std::string&            gradient_file,
                                           const std::span<const double> bin_scale) {
    const int total_bins = binning.total_bins();
    m_MuDelta.assign(total_bins, 0.0);
    m_SsqDelta.assign(total_bins, 0.0);
    load(gradient_file, total_bins);

    if (bin_scale.empty()) return;
    if (bin_scale.size() != static_cast<std::size_t>(total_bins))
      throw std::runtime_error("DetectorSystematics: bin scale has " + std::to_string(bin_scale.size()) +
                               " entries, the binning has " + std::to_string(total_bins));

    // Scaled once here rather than per call: the factors come from the MC and do
    // not move during the fit.
    for (int b = 0; b < total_bins; ++b) {
      const double f = bin_scale[b];
      for (int k = 0; k < params::ic::nDetSysParams; ++k) {
        m_Gradient[k][b] *= f;
        m_GradientError[k][b] *= f;
      }
      for (int p = 0; p < nPairs; ++p)
        m_Covariance[p][b] *= f * f;
    }

    double mean = 0.0;
    for (const double f : bin_scale) mean += f;
    std::cout << "DetectorSystematics: rescaled gradients to the topology-filtered sample (mean bin factor "
              << mean / static_cast<double>(bin_scale.size()) << ")\n";
  }

  void DetectorSystematics::load(const std::string& path, const int total_bins) {
    std::ifstream in(path);
    if (!in)
      throw std::runtime_error("DetectorSystematics: cannot open gradient file '" + path + "'");

    std::string line;
    if (!next_line(in, line))
      throw std::runtime_error("DetectorSystematics: '" + path + "' is empty");

    // "# gradients bins <N> params <K> lt_scale <s>"
    {
      std::istringstream header(line);
      std::string        hash, word, bins_key, params_key, scale_key;
      int                bins = -1, params = -1;
      if (!(header >> hash >> word >> bins_key >> bins >> params_key >> params >> scale_key >>
            m_LivetimeScale))
        throw std::runtime_error("DetectorSystematics: '" + path +
                                 "' has an incomplete '# gradients bins <N> params <K> lt_scale <s>' header");
      if (word != "gradients" || bins_key != "bins" || params_key != "params" || scale_key != "lt_scale")
        throw std::runtime_error("DetectorSystematics: '" + path +
                                 "' has no '# gradients bins <N> params <K> lt_scale <s>' header");
      if (bins != total_bins)
        throw std::runtime_error("DetectorSystematics: '" + path + "' declares " + std::to_string(bins) +
                                 " bins, the sample's binning has " + std::to_string(total_bins));
      if (params != params::ic::nDetSysParams)
        throw std::runtime_error("DetectorSystematics: '" + path + "' declares " + std::to_string(params) +
                                 " systematics, expected " + std::to_string(params::ic::nDetSysParams));
    }

    auto read_marker = [&](const std::string& expected_kind) {
      while (next_line(in, line)) {
        if (line.front() != '#') continue;
        std::istringstream marker(line);
        std::string        hash, kind;
        marker >> hash >> kind;
        if (kind == expected_kind) return marker;
      }
      throw std::runtime_error("DetectorSystematics: '" + path + "' has no further '# " + expected_kind +
                               "' section");
    };

    for (int k = 0; k < params::ic::nDetSysParams; ++k) {
      std::istringstream marker = read_marker("param");
      std::string        name, split_key;
      marker >> name >> split_key >> m_Split[k];
      if (name != kSystematicNames[k])
        throw std::runtime_error("DetectorSystematics: '" + path + "' systematic " + std::to_string(k) +
                                 " is '" + name + "', expected '" + kSystematicNames[k] +
                                 "' (the export script's order must match params::ic)");
      m_Gradient[k].assign(total_bins, 0.0);
      m_GradientError[k].assign(total_bins, 0.0);
      for (int b = 0; b < total_bins; ++b)
        if (!(in >> m_Gradient[k][b] >> m_GradientError[k][b]))
          throw std::runtime_error("DetectorSystematics: '" + path + "' ran out of values in '" + name + "'");
      std::getline(in, line);  // consume the rest of the last data line
    }

    for (int p = 0; p < nPairs; ++p) {
      read_marker("cov");
      m_Covariance[p].assign(total_bins, 0.0);
      for (int b = 0; b < total_bins; ++b)
        if (!(in >> m_Covariance[p][b]))
          throw std::runtime_error("DetectorSystematics: '" + path + "' ran out of covariance values");
      std::getline(in, line);
    }

    std::cout << "DetectorSystematics: loaded " << params::ic::nDetSysParams << " gradients x " << total_bins
              << " bins (+ " << nPairs << " covariance pairs, livetime scale " << m_LivetimeScale << ") from "
              << path << '\n';
  }

  bool DetectorSystematics::check_and_recalculate(const ParameterWrapper& parameter) {
    using namespace params::ic;
    if (!parameter.check_parameter_changed(DOMEff, HoleIceP1)) return false;

    // The livetime scale is folded into the deviations, so the covariance term
    // picks up lt_scale^2 as the product of two of them (NNMFit's convention).
    double deviation[nDetSysParams];
    for (int k = 0; k < nDetSysParams; ++k)
      deviation[k] = (parameter[DOMEff + k] - m_Split[k]) * m_LivetimeScale;

    for (std::size_t b = 0, n = m_MuDelta.size(); b < n; ++b) {
      double mu  = 0.0;
      double ssq = 0.0;
      for (int k = 0; k < nDetSysParams; ++k) {
        mu += deviation[k] * m_Gradient[k][b];
        const double term = deviation[k] * m_GradientError[k][b];
        ssq += term * term;
      }
      int pair = 0;
      for (int i = 0; i < nDetSysParams; ++i)
        for (int j = i + 1; j < nDetSysParams; ++j, ++pair)
          ssq += 2.0 * deviation[i] * deviation[j] * m_Covariance[pair][b];
      m_MuDelta[b]  = mu;
      m_SsqDelta[b] = ssq;
    }
    return true;
  }

}  // namespace ana::ic
