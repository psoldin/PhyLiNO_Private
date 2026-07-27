#include "TemplateFlux.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace ana::ic {

  TemplateFlux::TemplateFlux(const io::ic::Binning& binning,
                             const std::string&     template_file,
                             const int              norm_index,
                             const double           livetime)
    : m_NormIndex(norm_index) {
    const int total_bins = binning.total_bins();
    m_Histogram.assign(total_bins, 0.0);
    m_Fluctuation.assign(total_bins, 0.0);
    load(template_file, total_bins, livetime);
  }

  void TemplateFlux::load(const std::string& path, const int total_bins, const double livetime) {
    std::ifstream in(path);
    if (!in)
      throw std::runtime_error("TemplateFlux: cannot open template file '" + path + "'");

    // Header: "# template bins <N>"; the remaining comments (bin edges) are
    // informational. The bin count is a hard check -- a template binned
    // differently from the sample would mis-assign every bin while still
    // summing to a plausible total.
    int         declared_bins = -1;
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      if (line.front() != '#') break;
      std::istringstream header(line);
      std::string        hash, word, bins_key;
      if (header >> hash >> word && word == "template") {
        if (!(header >> bins_key >> declared_bins) || bins_key != "bins") declared_bins = -1;
      }
    }
    if (declared_bins != total_bins)
      throw std::runtime_error("TemplateFlux: template '" + path + "' declares " +
                               std::to_string(declared_bins) + " bins, the sample's binning has " +
                               std::to_string(total_bins));

    m_Template.assign(total_bins, 0.0);
    m_Sigma.assign(total_bins, 0.0);

    // `line` holds the first non-comment line read above; parse it, then the rest.
    std::istringstream first(line);
    double             rate = 0.0, sigma = 0.0;
    if (!(first >> rate >> sigma))
      throw std::runtime_error("TemplateFlux: template '" + path + "' has no data rows");
    m_Template[0] = rate * livetime;
    m_Sigma[0]    = sigma * livetime;

    for (int b = 1; b < total_bins; ++b) {
      if (!(in >> rate >> sigma))
        throw std::runtime_error("TemplateFlux: template '" + path + "' ended after " +
                                 std::to_string(b) + " of " + std::to_string(total_bins) + " bins");
      m_Template[b] = rate * livetime;
      m_Sigma[b]    = sigma * livetime;
    }

    double total = 0.0;
    for (const double v : m_Template) total += v;
    std::cout << "TemplateFlux: loaded " << total_bins << "-bin template from " << path << " (" << total
              << " events at norm 1)\n";
  }

  bool TemplateFlux::check_and_recalculate(const ParameterWrapper& parameter) {
    if (!parameter.check_parameter_changed(m_NormIndex)) return false;

    const double norm = parameter[m_NormIndex];
    for (std::size_t b = 0, n = m_Histogram.size(); b < n; ++b) {
      m_Histogram[b]   = norm * m_Template[b];
      const double sig = norm * m_Sigma[b];
      m_Fluctuation[b] = sig * sig;
    }
    return true;
  }

}  // namespace ana::ic
