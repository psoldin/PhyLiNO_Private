#include "Binning.h"

#include <algorithm>
#include <array>
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

  std::vector<double> bin_event_counts(const Binning&             binning,
                                       const std::vector<double>& reco_energy,
                                       const std::vector<double>& reco_zenith) {
    std::vector<double> counts(binning.total_bins(), 0.0);
    for (std::size_t i = 0, n = reco_energy.size(); i < n; ++i) {
      const std::array<double, 2> reco{reco_energy[i], reco_zenith[i]};
      const int                   bin = binning.bin_index(reco);
      if (bin >= 0) counts[bin] += 1.0;
    }
    return counts;
  }

  Binning drop_ra_axis(const Binning& binning) {
    const std::span<const Axis> axes = binning.axes();

    for (std::size_t d = 0; d + 1 < axes.size(); ++d)
      if (axes[d].kind == Axis::Kind::Ra)
        throw std::runtime_error("drop_ra_axis: the Ra axis must be the last one (found it at index " +
                                 std::to_string(d) + " of " + std::to_string(axes.size()) + ")");

    if (axes.back().kind != Axis::Kind::Ra)
      return binning;

    if (axes.size() < 2)
      throw std::runtime_error("drop_ra_axis: a binning cannot consist of the Ra axis alone");

    return Binning(std::vector<Axis>(axes.begin(), axes.end() - 1));
  }

  int ra_bin_count(const Binning& binning) noexcept {
    const std::span<const Axis> axes = binning.axes();
    return axes.back().kind == Axis::Kind::Ra ? axes.back().n_bins : 1;
  }

  void broadcast_over_ra(const std::span<const double> mc_bins, const int n_ra, const double divisor,
                         const std::span<double> out) {
    if (n_ra <= 0)
      throw std::runtime_error("broadcast_over_ra: n_ra must be positive, got " + std::to_string(n_ra));

    if (out.size() != mc_bins.size() * static_cast<std::size_t>(n_ra))
      throw std::runtime_error("broadcast_over_ra: output has " + std::to_string(out.size()) +
                               " bins, expected " + std::to_string(mc_bins.size()) + " * " +
                               std::to_string(n_ra));

    for (std::size_t b = 0, n = mc_bins.size(); b < n; ++b) {
      const double value = mc_bins[b] / divisor;
      const std::size_t base = b * static_cast<std::size_t>(n_ra);
      for (int r = 0; r < n_ra; ++r) out[base + r] = value;
    }
  }

  std::vector<double> bin_event_counts(const Binning&             binning,
                                       const std::vector<double>& reco_energy,
                                       const std::vector<double>& reco_zenith,
                                       const std::vector<double>& reco_ra) {
    std::vector<double> counts(binning.total_bins(), 0.0);
    for (std::size_t i = 0, n = reco_energy.size(); i < n; ++i) {
      const std::array<double, 3> reco{reco_energy[i], reco_zenith[i], reco_ra[i]};
      const int                   bin = binning.bin_index(reco);
      if (bin >= 0) counts[bin] += 1.0;
    }
    return counts;
  }

}  // namespace io::ic
