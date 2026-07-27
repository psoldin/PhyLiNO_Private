#include "Binning.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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
    if (uniform()) return static_cast<int>((v - lo) / step());
    // First edge strictly greater than v. Since v >= lo == edges.front(), that
    // iterator is at least edges.begin() + 1, and the bin below it is the one
    // containing v -- hence the -1.
    const auto upper = std::ranges::upper_bound(edges, v);
    return static_cast<int>(std::distance(edges.begin(), upper)) - 1;
  }

  namespace {

    Axis::Kind parse_axis_kind(const std::string_view kind) {
      if (kind == "Log10Energy") return Axis::Kind::Log10Energy;
      if (kind == "CosZenith") return Axis::Kind::CosZenith;
      if (kind == "Ra") return Axis::Kind::Ra;
      throw std::runtime_error("parse_axis: unknown axis kind '" + std::string(kind) + "'");
    }

    // "(a, b, c)" / "[a, b, c]" -> the numbers, punctuation blanked out
    std::vector<double> parse_numbers(const std::string_view spec) {
      std::string s(spec);
      for (char& c : s) {
        if (c == '(' || c == ')' || c == '[' || c == ']' || c == ',')
          c = ' ';
      }
      std::istringstream in(s);
      std::vector<double> values;
      double v = 0.0;

      while (in >> v)
        values.push_back(v);

      return values;
    }
  }

  Axis parse_axis(const std::string_view kind, const std::string_view spec) {
    const Axis::Kind k = parse_axis_kind(kind);
    const auto trimmed = spec.substr(spec.find_first_not_of(" \t"));
    const auto numbers = parse_numbers(trimmed);

    if (trimmed.starts_with('[')) {
      if (numbers.size() < 2)
        throw std::runtime_error("parse_axis: bad spec '" + std::string(spec) + "' needs at least two edges");

      if (!std::ranges::is_sorted(numbers))
        throw std::runtime_error("parse_axis: edge list '" + std::string(spec) + "' is not ascending");

      return Axis{k, numbers.front(), numbers.back(), static_cast<int>(numbers.size() - 1), numbers};
    }

    if (numbers.size() != 3 || numbers[2] <= 0.0)
      throw std::runtime_error("parse_axis: bad spec '" + std::string(spec) + "' (want '(lo, hi, n_bins)') or [e0, e1, ...]");

    return Axis{k, numbers[0], numbers[1], static_cast<int>(numbers[2]), {}};
  }

  std::string_view axis_kind_name(const Axis::Kind kind) noexcept {
    switch (kind) {
      case Axis::Kind::Log10Energy: return "Log10Energy";
      case Axis::Kind::CosZenith:   return "CosZenith";
      case Axis::Kind::Ra:          return "Ra";
    }
    return "Unknown";
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
