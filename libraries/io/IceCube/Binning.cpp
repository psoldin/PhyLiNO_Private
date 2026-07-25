#include "Binning.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace io::ic {

  double Axis::project(const double raw_value) const noexcept {
    switch (kind) {
      case Kind::Log10Energy: return std::log10(raw_value);
      case Kind::CosZenith:   return std::cos(raw_value);
      case Kind::Ra:          return raw_value;
    }
    return raw_value;
  }

  int Axis::index(const double raw_value) const noexcept {
    const double v = project(raw_value);
    if (v < lo || v >= hi) return -1;
    return static_cast<int>((v - lo) / step());
  }

  Axis parse_axis(const std::string_view kind, const std::string_view spec) {
    Axis::Kind k;
    if (kind == "Log10Energy")   k = Axis::Kind::Log10Energy;
    else if (kind == "CosZenith") k = Axis::Kind::CosZenith;
    else if (kind == "Ra")        k = Axis::Kind::Ra;
    else throw std::runtime_error("parse_axis: unknown axis kind '" + std::string(kind) + "'");

    std::string s(spec);
    for (char& c : s) if (c == '(' || c == ')' || c == ',') c = ' ';
    double lo = 0, hi = 0; int n = 0;
    if (std::sscanf(s.c_str(), "%lf %lf %d", &lo, &hi, &n) != 3 || n <= 0)
      throw std::runtime_error("parse_axis: bad spec '" + std::string(spec) + "' (want '(lo, hi, n_bins)')");
    return Axis{k, lo, hi, n};
  }

  Binning::Binning(std::vector<Axis> axes) : m_Axes(std::move(axes)) {
    if (m_Axes.empty()) throw std::runtime_error("Binning: needs at least one axis");
    m_TotalBins = 1;
    for (const Axis& a : m_Axes) m_TotalBins *= a.n_bins;
  }

  int Binning::bin_index(const std::span<const double> reco) const noexcept {
    int flat = 0;
    for (std::size_t d = 0; d < m_Axes.size(); ++d) {
      const int i = m_Axes[d].index(reco[d]);
      if (i < 0) return -1;
      flat = flat * m_Axes[d].n_bins + i;
    }
    return flat;
  }

}  // namespace io::ic
