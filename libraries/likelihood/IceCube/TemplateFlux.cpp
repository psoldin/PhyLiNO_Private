#include "TemplateFlux.h"

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
    if (declared_bins != file_rows)
      throw std::runtime_error("TemplateFlux: template '" + path + "' declares " +
                               std::to_string(declared_bins) + " bins, the sample's binning has " +
                               std::to_string(total_bins) +
                               (file_bins.identity() ? "" : " and its file binning " + std::to_string(file_rows)));

    std::vector<double> rates(file_rows, 0.0);
    std::vector<double> sigmas(file_rows, 0.0);

    // `line` holds the first non-comment line read above; parse it, then the rest.
    std::istringstream first(line);
    if (!(first >> rates[0] >> sigmas[0]))
      throw std::runtime_error("TemplateFlux: template '" + path + "' has no data rows");

    for (int b = 1; b < file_rows; ++b)
      if (!(in >> rates[b] >> sigmas[b]))
        throw std::runtime_error("TemplateFlux: template '" + path + "' ended after " +
                                 std::to_string(b) + " of " + std::to_string(file_rows) + " bins");

    m_Template.assign(total_bins, 0.0);
    m_Sigma.assign(total_bins, 0.0);
    io::ic::gather_bins(file_bins, rates, m_Template);
    io::ic::gather_bins(file_bins, sigmas, m_Sigma);
    for (int b = 0; b < total_bins; ++b) {
      m_Template[b] *= livetime;
      m_Sigma[b] *= livetime;
    }

    double total = 0.0;
    for (const double v : m_Template) total += v;
    std::cout << "TemplateFlux: loaded " << total_bins << "-bin template from " << path;
    if (!file_bins.identity())
      std::cout << " (" << file_rows << " bins in the file, " << (file_rows - total_bins) << " dropped)";
    std::cout << " (" << total << " events at norm 1)\n";
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
