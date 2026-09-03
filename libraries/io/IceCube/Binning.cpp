#include "Binning.h"

#include <algorithm>
#include <array>
#include <cassert>
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
      case Kind::Category:    return raw_value;
    }
    return raw_value;
  }

  int Axis::index(const double raw_value) const noexcept {
    const double v = project(raw_value);
    // Negated so a NaN lands here too: a failed reco is stored as -1 in the
    // parquet, log10(-1) is NaN, and "NaN < lo || NaN >= hi" is false -- the
    // event would fall through to the cast below, which is UB and in practice
    // piles every such event into bin 0.
    if (!(v >= lo && v < hi)) return -1;
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
      if (kind.starts_with("Category:")) return Axis::Kind::Category;
      throw std::runtime_error("parse_axis: unknown axis kind '" + std::string(kind) +
                               "' (expected Log10Energy, CosZenith, Ra or Category:<branch>)");
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

    // "Category:bdt_score" carries its own branch name; every other kind reads a
    // column fixed by BranchNames.
    std::string branch;
    if (k == Axis::Kind::Category) {
      branch = std::string(kind.substr(std::string_view("Category:").size()));
      if (branch.empty())
        throw std::runtime_error("parse_axis: 'Category:' needs a branch name, e.g. Category:bdt_score");
    }

    const auto trimmed = spec.substr(spec.find_first_not_of(" \t"));
    const auto numbers = parse_numbers(trimmed);

    if (trimmed.starts_with('[')) {
      if (numbers.size() < 2)
        throw std::runtime_error("parse_axis: bad spec '" + std::string(spec) + "' needs at least two edges");

      if (!std::ranges::is_sorted(numbers))
        throw std::runtime_error("parse_axis: edge list '" + std::string(spec) + "' is not ascending");

      return Axis{k, numbers.front(), numbers.back(), static_cast<int>(numbers.size() - 1), numbers,
                  std::move(branch)};
    }

    if (numbers.size() != 3 || numbers[2] <= 0.0)
      throw std::runtime_error("parse_axis: bad spec '" + std::string(spec) + "' (want '(lo, hi, n_bins)') or [e0, e1, ...]");

    return Axis{k, numbers[0], numbers[1], static_cast<int>(numbers[2]), {}, std::move(branch)};
  }

  std::string_view axis_kind_name(const Axis::Kind kind) noexcept {
    switch (kind) {
      case Axis::Kind::Log10Energy: return "Log10Energy";
      case Axis::Kind::CosZenith:   return "CosZenith";
      case Axis::Kind::Ra:          return "Ra";
      case Axis::Kind::Category:    return "Category";
    }
    return "Unknown";
  }

  std::vector<double> axis_edges(const Axis& axis) {
    if (!axis.uniform()) return axis.edges;
    std::vector<double> edges(static_cast<std::size_t>(axis.n_bins) + 1);
    for (int i = 0; i <= axis.n_bins; ++i) edges[i] = axis.lo + i * axis.step();
    return edges;
  }

  namespace {

    // Source bin of every target bin on one axis. Both edge lists ascend, so a
    // single sweep of the source edges finds them all.
    std::vector<int> axis_bin_map(const Axis& source, const Axis& target, const std::size_t axis) {
      const std::string where = "make_bin_map: axis " + std::to_string(axis) + " (" +
                                std::string(axis_kind_name(target.kind)) + ")";
      if (source.kind != target.kind || source.branch != target.branch)
        throw std::runtime_error(where + " is '" + std::string(axis_kind_name(source.kind)) +
                                 "' in the source binning; the two must agree axis by axis");

      const std::vector<double> s = axis_edges(source);
      const std::vector<double> t = axis_edges(target);
      // Relative to the source span: the same grid written out twice in decimal
      // agrees to ~1e-16 of its own scale, and nothing physical is this close.
      const double tol = 1.0e-9 * std::max(1.0, std::abs(s.back() - s.front()));

      std::vector<int> map(static_cast<std::size_t>(target.n_bins));
      std::size_t      j = 0;
      for (int i = 0; i < target.n_bins; ++i) {
        while (j + 1 < s.size() && std::abs(s[j] - t[i]) > tol) ++j;
        if (j + 1 >= s.size() || std::abs(s[j + 1] - t[i + 1]) > tol) {
          char lo[32], hi[32];
          std::snprintf(lo, sizeof(lo), "%.10g", t[i]);
          std::snprintf(hi, sizeof(hi), "%.10g", t[i + 1]);
          throw std::runtime_error(where + ": bin " + std::to_string(i) + " [" + lo + ", " + hi +
                                   "] is not a bin of the source binning. A sub-grid may drop source "
                                   "bins but not split or merge them");
        }
        map[static_cast<std::size_t>(i)] = static_cast<int>(j);
      }
      return map;
    }

  }  // namespace

  BinMap make_bin_map(const Binning& source, const Binning& target) {
    if (source.n_axes() != target.n_axes())
      throw std::runtime_error("make_bin_map: the source binning has " + std::to_string(source.n_axes()) +
                               " axes, the sample's has " + std::to_string(target.n_axes()));

    std::vector<std::vector<int>> per_axis;
    bool                          same_shape = true;
    for (std::size_t d = 0; d < target.n_axes(); ++d) {
      per_axis.push_back(axis_bin_map(source.axes()[d], target.axes()[d], d));
      same_shape &= source.axes()[d].n_bins == target.axes()[d].n_bins;
    }

    // Equal bin counts on every axis, with each target bin matched to a distinct
    // ascending source bin, leaves only the identity -- so there is nothing to
    // gather and the loaders can read the file straight into place.
    BinMap map{.source_bins = source.total_bins(), .index = {}};
    if (same_shape) return map;

    map.index.assign(static_cast<std::size_t>(target.total_bins()), 0);
    std::vector<int> at(target.n_axes(), 0);  // row-major odometer over target bins
    for (int b = 0; b < target.total_bins(); ++b) {
      int flat = 0;
      for (std::size_t d = 0; d < target.n_axes(); ++d)
        flat = flat * source.axes()[d].n_bins + per_axis[d][static_cast<std::size_t>(at[d])];
      map.index[static_cast<std::size_t>(b)] = flat;

      for (int d = static_cast<int>(target.n_axes()) - 1; d >= 0; --d) {
        if (++at[static_cast<std::size_t>(d)] < target.axes()[static_cast<std::size_t>(d)].n_bins) break;
        at[static_cast<std::size_t>(d)] = 0;
      }
    }
    return map;
  }

  void gather_bins(const BinMap& map, const std::span<const double> values, const std::span<double> out) {
    if (map.identity()) {
      std::ranges::copy(values, out.begin());
      return;
    }
    if (out.size() != map.index.size())
      throw std::runtime_error("gather_bins: " + std::to_string(out.size()) +
                               " output bins for a map covering " + std::to_string(map.index.size()));
    for (std::size_t b = 0; b < map.index.size(); ++b)
      out[b] = values[static_cast<std::size_t>(map.index[b])];
  }

  std::vector<Axis> category_axes(const Binning& binning) {
    std::vector<Axis> out;
    for (const Axis& a : binning.axes())
      if (a.kind == Axis::Kind::Category) out.push_back(a);
    return out;
  }

  Binning::Binning(std::vector<Axis> axes) : m_Axes(std::move(axes)) {
    if (m_Axes.empty()) throw std::runtime_error("Binning: needs at least one axis");
    m_TotalBins = 1;
    for (const Axis& a : m_Axes) m_TotalBins *= a.n_bins;
  }

  int Binning::bin_index(const std::span<const double> reco) const noexcept {
    assert(reco.size() >= m_Axes.size() && "bin_index: one reco value per axis is required");
    int flat = 0;
    for (std::size_t d = 0; d < m_Axes.size(); ++d) {
      const int i = m_Axes[d].index(reco[d]);
      if (i < 0) return -1;
      flat = flat * m_Axes[d].n_bins + i;
    }
    return flat;
  }

  std::vector<double> bin_event_counts(const Binning&                       binning,
                                       const std::span<const std::vector<double>> columns) {
    if (columns.size() != binning.n_axes())
      throw std::runtime_error("bin_event_counts: got " + std::to_string(columns.size()) +
                               " columns for a binning with " + std::to_string(binning.n_axes()) + " axes");

    std::vector<double> counts(static_cast<std::size_t>(binning.total_bins()), 0.0);
    if (columns.empty()) return counts;

    const std::size_t   n = columns.front().size();
    std::vector<double> reco(columns.size(), 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t c = 0; c < columns.size(); ++c) reco[c] = columns[c][i];
      const int bin = binning.bin_index(reco);
      if (bin >= 0) counts[static_cast<std::size_t>(bin)] += 1.0;
    }
    return counts;
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

    if (!has_ra_axis(binning))
      return binning;

    if (axes.size() < 2)
      throw std::runtime_error("drop_ra_axis: a binning cannot consist of the Ra axis alone");

    return Binning(std::vector<Axis>(axes.begin(), axes.end() - 1));
  }

  bool has_ra_axis(const Binning& binning) noexcept {
    return binning.axes().back().kind == Axis::Kind::Ra;
  }

  int ra_bin_count(const Binning& binning) noexcept {
    return has_ra_axis(binning) ? binning.axes().back().n_bins : 1;
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
