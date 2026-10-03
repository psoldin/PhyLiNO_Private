#include "TemplateFlux.h"

#include "../../io/IceCube/ICParameter.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace ana::ic {

  TemplateFlux::TemplateFlux(const io::ic::Binning& binning,
                             const std::string&     template_file,
                             const int              norm_index,
                             const double           livetime,
                             const io::ic::BinMap&  file_bins)
    : m_NormIndex(norm_index) {
    const int total_bins = binning.total_bins();
    m_Histogram.assign(total_bins, 0.0);
    m_Fluctuation.assign(total_bins, 0.0);
    load(template_file, total_bins, livetime, file_bins);
    if (!m_Modes.empty() && m_NormIndex != params::ic::MuonNorm)
      throw std::runtime_error("TemplateFlux: template '" + template_file +
                               "' carries shape modes, but only MuonNorm's template may (its norm here is "
                               "parameter " + std::to_string(m_NormIndex) + ")");
    m_Shaped = m_Template;
    m_ModeValues.assign(m_Modes.size(), 0.0);
  }

  int TemplateFlux::mode_index(const int j) noexcept { return params::ic::MuonTmplMode1 + j; }

  void TemplateFlux::add_mode_derivative(const int j, const double norm, const std::span<double> out) const {
    const std::vector<double>& d = m_Modes[static_cast<std::size_t>(j)];
    for (std::size_t b = 0, n = out.size(); b < n; ++b)
      if (m_Shaped[b] > 0.0) out[b] += norm * d[b];
  }

  void TemplateFlux::load(const std::string& path, const int total_bins, const double livetime,
                          const io::ic::BinMap& file_bins) {
    std::ifstream in(path);
    if (!in)
      throw std::runtime_error("TemplateFlux: cannot open template file '" + path + "'");

    // Rows the file itself carries: the sample's own bin count, unless the sample
    // fits a sub-grid of the exported binning, in which case the file is read whole
    // and gathered down (see io::ic::make_bin_map).
    const int file_rows = file_bins.identity() ? total_bins : file_bins.source_bins;

    // Header: "# template bins <N>", optionally "# template modes <M>"; the
    // remaining comments (bin edges) are informational. The bin count is a hard
    // check -- a template binned differently from the sample would mis-assign
    // every bin while still summing to a plausible total.
    int         declared_bins = -1;
    int         modes         = 0;
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      if (line.front() != '#') break;
      std::istringstream header(line);
      std::string        hash, word, key;
      int                value = -1;
      if (header >> hash >> word && word == "template" && header >> key >> value) {
        if (key == "bins") declared_bins = value;
        if (key == "modes") modes = value;
      }
    }
    if (modes < 0 || modes > params::ic::nTemplateModes)
      throw std::runtime_error("TemplateFlux: template '" + path + "' declares " + std::to_string(modes) +
                               " modes; the parameter layout has " + std::to_string(params::ic::nTemplateModes));
    if (declared_bins != file_rows)
      throw std::runtime_error("TemplateFlux: template '" + path + "' declares " +
                               std::to_string(declared_bins) + " bins, the sample's binning has " +
                               std::to_string(total_bins) +
                               (file_bins.identity() ? "" : " and its file binning " + std::to_string(file_rows)));

    std::vector<double>              rates(file_rows, 0.0);
    std::vector<double>              sigmas(file_rows, 0.0);
    std::vector<std::vector<double>> mode_rates(modes, std::vector<double>(file_rows, 0.0));

    // `line` holds the first non-comment line read above; parse it, then the rest.
    // Rows are read line by line so a row with too few or too many columns for the
    // declared mode count is an error rather than a silent shift of every later bin.
    for (int b = 0; b < file_rows; ++b) {
      if (b > 0) {
        do {
          if (!std::getline(in, line)) {
            line.clear();
            break;
          }
        } while (line.find_first_not_of(" \t\r") == std::string::npos);
      }
      std::istringstream row(line);
      bool ok = static_cast<bool>(row >> rates[b] >> sigmas[b]);
      for (int j = 0; ok && j < modes; ++j) ok = static_cast<bool>(row >> mode_rates[j][b]);
      std::string extra;
      if (!ok || row >> extra)
        throw std::runtime_error("TemplateFlux: template '" + path + "' row " + std::to_string(b) + " of " +
                                 std::to_string(file_rows) + " does not hold " + std::to_string(2 + modes) +
                                 " numbers (rate, fluctuation" + (modes ? ", " + std::to_string(modes) + " modes" : "") +
                                 "): '" + line + "'");
    }

    auto to_sample = [&](const std::vector<double>& column) {
      std::vector<double> out(total_bins, 0.0);
      io::ic::gather_bins(file_bins, column, out);
      for (double& v : out) v *= livetime;
      return out;
    };
    m_Template = to_sample(rates);
    m_Sigma    = to_sample(sigmas);
    for (const auto& column : mode_rates) m_Modes.push_back(to_sample(column));

    double total = 0.0;
    for (const double v : m_Template) total += v;
    std::cout << "TemplateFlux: loaded " << total_bins << "-bin template from " << path;
    if (!file_bins.identity())
      std::cout << " (" << file_rows << " bins in the file, " << (file_rows - total_bins) << " dropped)";
    std::cout << " (" << total << " events at norm 1)";
    if (modes > 0) std::cout << " with " << modes << " shape mode" << (modes > 1 ? "s" : "");
    std::cout << "\n";
  }

  bool TemplateFlux::check_and_recalculate(const ParameterWrapper& parameter) {
    // The mode values are compared against what the shaped template was built at,
    // not via check_parameter_changed: that would leave it at theta = 0 if the
    // first evaluation already sat at a non-zero mode value.
    bool shape_changed = false;
    for (std::size_t j = 0; j < m_Modes.size(); ++j)
      shape_changed |= parameter[mode_index(static_cast<int>(j))] != m_ModeValues[j];
    if (!shape_changed && !parameter.check_parameter_changed(m_NormIndex)) return false;

    if (shape_changed) {
      for (std::size_t j = 0; j < m_Modes.size(); ++j) m_ModeValues[j] = parameter[mode_index(static_cast<int>(j))];
      for (std::size_t b = 0, n = m_Shaped.size(); b < n; ++b) {
        double value = m_Template[b];
        for (std::size_t j = 0; j < m_Modes.size(); ++j) value += m_ModeValues[j] * m_Modes[j][b];
        m_Shaped[b] = std::max(value, 0.0);
      }
    }

    const double norm = parameter[m_NormIndex];
    for (std::size_t b = 0, n = m_Histogram.size(); b < n; ++b) {
      m_Histogram[b]   = norm * m_Shaped[b];
      const double sig = norm * m_Sigma[b];
      m_Fluctuation[b] = sig * sig;
    }
    return true;
  }

}  // namespace ana::ic
