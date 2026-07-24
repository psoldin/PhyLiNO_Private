#include "DetectorSystematics.h"

#include <fstream>
#include <iostream>

namespace ana::ic {

  DetectorSystematics::DetectorSystematics(const bool enabled, const std::string& gradient_file)
    : m_Enabled(enabled) {
    for (auto& g : m_Gradients) g.fill(0.0);
    m_Delta.fill(0.0);

    if (!m_Enabled) return;

    if (gradient_file.empty() || !load_gradients(gradient_file)) {
      std::cout << "DetectorSystematics: enabled but no usable gradient file ('"
                << gradient_file << "'); disabling (delta is zero).\n";
      m_Enabled = false;
    }
  }

  bool DetectorSystematics::load_gradients(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
      std::cout << "DetectorSystematics: cannot open gradient file '" << path << "'\n";
      return false;
    }
    for (int k = 0; k < params::ic::nDetSysParams; ++k) {
      for (int b = 0; b < io::ic::Constants::nBins; ++b) {
        if (!(in >> m_Gradients[k][b])) {
          std::cout << "DetectorSystematics: file '" << path << "' has fewer than "
                    << params::ic::nDetSysParams * io::ic::Constants::nBins << " values\n";
          return false;
        }
      }
    }
    std::cout << "DetectorSystematics: loaded " << params::ic::nDetSysParams
              << " gradients x " << io::ic::Constants::nBins << " bins from " << path << '\n';
    return true;
  }

  bool DetectorSystematics::check_and_recalculate(const ParameterWrapper& parameter) {
    using namespace params::ic;
    if (!m_Enabled) return false;
    if (!parameter.check_parameter_changed(DOMEff, IceScat)) return false;

    double variation[nDetSysParams];
    for (int k = 0; k < nDetSysParams; ++k)
      variation[k] = parameter[DOMEff + k] - m_Split[k];

    for (int b = 0; b < io::ic::Constants::nBins; ++b) {
      double d = 0.0;
      for (int k = 0; k < nDetSysParams; ++k)
        d += variation[k] * m_Gradients[k][b];
      m_Delta[b] = d;
    }
    return true;
  }

}  // namespace ana::ic
