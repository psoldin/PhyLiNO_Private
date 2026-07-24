#include "MuonTemplate.h"

#include "../../io/IceCube/ICParameter.h"

#include <fstream>
#include <iostream>

namespace ana::ic {

  MuonTemplate::MuonTemplate(const bool enabled, const std::string& template_file)
    : m_Enabled(enabled) {
    m_Template.fill(0.0);
    m_Histogram.fill(0.0);

    if (!m_Enabled) return;

    if (template_file.empty() || !load_template(template_file)) {
      std::cout << "MuonTemplate: enabled but no usable template file ('"
                << template_file << "'); disabling (component contributes nothing).\n";
      m_Enabled = false;
    }
  }

  bool MuonTemplate::load_template(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
      std::cout << "MuonTemplate: cannot open template file '" << path << "'\n";
      return false;
    }
    for (int b = 0; b < io::ic::Constants::nBins; ++b) {
      if (!(in >> m_Template[b])) {
        std::cout << "MuonTemplate: file '" << path << "' has fewer than "
                  << io::ic::Constants::nBins << " values\n";
        return false;
      }
    }
    std::cout << "MuonTemplate: loaded " << io::ic::Constants::nBins
              << "-bin template from " << path << '\n';
    return true;
  }

  bool MuonTemplate::check_and_recalculate(const ParameterWrapper& parameter) {
    if (!m_Enabled) return false;
    if (!parameter.check_parameter_changed(params::ic::MuonNorm)) return false;

    const double norm = parameter[params::ic::MuonNorm];
    for (int b = 0; b < io::ic::Constants::nBins; ++b)
      m_Histogram[b] = norm * m_Template[b];
    return true;
  }

}  // namespace ana::ic
