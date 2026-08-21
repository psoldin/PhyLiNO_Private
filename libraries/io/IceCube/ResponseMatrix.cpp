#include "ResponseMatrix.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace io::ic {

  namespace {

    constexpr double kInvSqrt2 = 0.70710678118654752440;

    /** Mass of N(c, h) below x. */
    double normal_cdf(const double x, const double c, const double h) noexcept {
      return 0.5 * std::erfc(-(x - c) * kInvSqrt2 / h);
    }

    /** Lower and upper edge of bin `i` on a uniform-or-explicit axis. */
    std::pair<double, double> edges_of(const Axis& axis, const int i) noexcept {
      if (axis.uniform()) {
        const double step = axis.step();
        return {axis.lo + i * step, axis.lo + (i + 1) * step};
      }
      return {axis.edges[static_cast<std::size_t>(i)], axis.edges[static_cast<std::size_t>(i) + 1]};
    }

    /**
     * Per-bin mass of N(c, h) on one axis, over the bins the truncation reaches.
     * `to_axis` maps an axis edge to the coordinate the response lives in, which
     * is the identity for log10 energy and acos for a cos-zenith axis.
     *
     * Returns the first bin filled; `out` holds that bin onwards.
     */
    template <class ToAxis>
    int axis_masses(const Axis& axis, const double c, const double h, const double truncation,
                    ToAxis&& to_axis, std::vector<double>& out) {
      out.clear();

      // Walk every bin the kernel can reach. The axis may be reversed in the
      // response coordinate (cos zenith is), so the interval is taken as an
      // unordered pair and the CDF difference by absolute value.
      int first = -1;
      for (int i = 0; i < axis.n_bins; ++i) {
        const auto [e_lo, e_hi] = edges_of(axis, i);
        const double a          = to_axis(e_lo);
        const double b          = to_axis(e_hi);
        const double lo         = std::min(a, b);
        const double hi         = std::max(a, b);

        // Cheap rejection before two erfc calls: the whole bin beyond the reach.
        if (lo > c + truncation * h || hi < c - truncation * h) {
          if (first >= 0) break;  // bins are ordered, so the reach is contiguous
          continue;
        }
        if (first < 0) first = i;
        out.push_back(normal_cdf(hi, c, h) - normal_cdf(lo, c, h));
      }
      return first;
    }

  }  // namespace

  ResponseMatrix build_response_matrix(const Binning&                binning,
                                       const std::span<const int>    bin_idx,
                                       const std::span<const double> truth_log_e,
                                       const std::span<const double> truth_zenith,
                                       const std::span<const double> sigma_log_e,
                                       const std::span<const double> sigma_zenith,
                                       const double truncation, const double min_fraction,
                                       std::size_t& unusable) {
    const std::span<const Axis> axes = binning.axes();
    if (axes.size() != 2 || axes[0].kind != Axis::Kind::Log10Energy || axes[1].kind != Axis::Kind::CosZenith)
      throw std::runtime_error(
          "build_response_matrix: expects an MC binning of (Log10Energy, CosZenith); the response is defined "
          "in those two coordinates and nothing else has been derived");
    if (truncation <= 0.0) throw std::runtime_error("build_response_matrix: truncation must be positive");

    const std::size_t n_events = bin_idx.size();
    const auto        n_bins   = static_cast<std::size_t>(binning.total_bins());
    const int         n_zenith = axes[1].n_bins;

    unusable = 0;

    // Event-major first, because that is the order the fractions are produced
    // and normalised in; transposed to bin-major below.
    std::vector<int>    entry_bin;
    std::vector<float>  entry_value;
    std::vector<int>    entry_event;
    entry_bin.reserve(n_events * 64);
    entry_value.reserve(n_events * 64);
    entry_event.reserve(n_events * 64);

    std::vector<double> mass_e, mass_z;
    std::vector<int>    bins_of_event;
    std::vector<double> frac_of_event;

    for (std::size_t i = 0; i < n_events; ++i) {
      if (bin_idx[i] < 0) continue;  // not in the sample

      const double t_e = truth_log_e[i];
      const double t_z = truth_zenith[i];
      const double h_e = sigma_log_e[i];
      const double h_z = sigma_zenith[i];

      const bool usable = std::isfinite(t_e) && std::isfinite(t_z) && std::isfinite(h_e) &&
                          std::isfinite(h_z) && h_e > 0.0 && h_z > 0.0;
      if (!usable) {
        // No response to fold: keep the event exactly where the histogram put it.
        ++unusable;
        entry_bin.push_back(bin_idx[i]);
        entry_value.push_back(1.0f);
        entry_event.push_back(static_cast<int>(i));
        continue;
      }

      const int first_e = axis_masses(axes[0], t_e, h_e, truncation, [](const double x) { return x; }, mass_e);
      const int first_z = axis_masses(
          axes[1], t_z, h_z, truncation,
          [](const double x) { return std::acos(std::clamp(x, -1.0, 1.0)); }, mass_z);

      if (first_e < 0 || first_z < 0) {
        // The whole response sits outside the analysis range; fall back rather
        // than drop, so the event's weight stays in the prediction.
        ++unusable;
        entry_bin.push_back(bin_idx[i]);
        entry_value.push_back(1.0f);
        entry_event.push_back(static_cast<int>(i));
        continue;
      }

      bins_of_event.clear();
      frac_of_event.clear();
      double total = 0.0;
      for (std::size_t je = 0; je < mass_e.size(); ++je) {
        if (mass_e[je] <= 0.0) continue;
        for (std::size_t jz = 0; jz < mass_z.size(); ++jz) {
          const double f = mass_e[je] * mass_z[jz];
          if (f <= 0.0) continue;
          bins_of_event.push_back((first_e + static_cast<int>(je)) * n_zenith + first_z +
                                  static_cast<int>(jz));
          frac_of_event.push_back(f);
          total += f;
        }
      }

      if (!(total > 0.0)) {
        ++unusable;
        entry_bin.push_back(bin_idx[i]);
        entry_value.push_back(1.0f);
        entry_event.push_back(static_cast<int>(i));
        continue;
      }

      // Drop the negligible tail, then normalise what is left to one. Doing it
      // in this order is what keeps the folded total equal to the unfolded one
      // whatever min_fraction is set to.
      double kept = 0.0;
      for (std::size_t k = 0; k < frac_of_event.size(); ++k)
        if (frac_of_event[k] >= min_fraction * total) kept += frac_of_event[k];
      if (!(kept > 0.0)) kept = total;

      for (std::size_t k = 0; k < frac_of_event.size(); ++k) {
        if (frac_of_event[k] < min_fraction * total) continue;
        entry_bin.push_back(bins_of_event[k]);
        entry_value.push_back(static_cast<float>(frac_of_event[k] / kept));
        entry_event.push_back(static_cast<int>(i));
      }
    }

    // Transpose to bin-major by counting sort.
    ResponseMatrix matrix;
    matrix.bin_offsets.assign(n_bins + 1, 0);
    for (const int b : entry_bin) ++matrix.bin_offsets[static_cast<std::size_t>(b) + 1];
    for (std::size_t b = 1; b < matrix.bin_offsets.size(); ++b)
      matrix.bin_offsets[b] += matrix.bin_offsets[b - 1];

    matrix.events.resize(entry_bin.size());
    matrix.fractions.resize(entry_bin.size());
    std::vector<std::size_t> cursor(matrix.bin_offsets.begin(), matrix.bin_offsets.end() - 1);
    for (std::size_t k = 0; k < entry_bin.size(); ++k) {
      const std::size_t at = cursor[static_cast<std::size_t>(entry_bin[k])]++;
      matrix.events[at]    = entry_event[k];
      matrix.fractions[at] = entry_value[k];
    }

    return matrix;
  }

}  // namespace io::ic
