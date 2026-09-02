#pragma once

#include "IceCube/Binning.h"

// STL includes
#include <algorithm>
#include <cstddef>
#include <span>
#include <string_view>

namespace result::ic {

  /**
   * What "--blind" keeps out of a written result. The fit itself is untouched:
   * every parameter still floats and every bin still enters the likelihood --
   * only the two writers (JSON and protobuf) drop what is listed here, so a
   * blind run on real data produces a file that can be inspected without
   * unblinding the signal.
   */

  /// Highest reconstructed energy a blinded result may report, as log10(E / GeV):
  /// 1e4 GeV. Bins above it are written as zero.
  inline constexpr double kBlindMaxLog10Energy = 4.0;

  /**
   * The signal parameters a blinded result omits: the astrophysical flux (the
   * single power law and the broken-power-law parameters that replace it) plus
   * the prompt normalization, which the analysis treats as signal-like because it
   * is what the astrophysical flux is measured against.
   */
  [[nodiscard]] inline bool is_blinded_parameter(std::string_view name) noexcept {
    return name == "AstroNorm" || name == "SpectralIndex" || name == "PromptNorm" ||
           name == "AstroGamma1" || name == "AstroGamma2" || name == "AstroEBreak";
  }

  /**
   * Component-breakdown entries a blinded result leaves out entirely: the
   * astrophysical flux, which is what the signal parameters above describe.
   *
   * Dropped rather than zeroed, unlike the hidden energy bins: a zeroed astro
   * curve reads as "the fit found no astrophysical flux", which is a wrong
   * statement rather than a withheld one.
   *
   * Note what this does NOT hide: the prediction is written in full below 1e4
   * GeV and the components sum to it exactly (see the explorer stack test in
   * ICTests.cpp), so prediction minus the remaining components is still the
   * astrophysical contribution plus the systematics delta. Hiding that too means
   * withholding the prediction itself, which would take the data/MC comparison
   * the file exists for with it.
   */
  [[nodiscard]] inline bool is_blinded_component(std::string_view name) noexcept {
    return name == "astro";
  }

  /**
   * Number of leading bins of `binning` that stay at or below kBlindMaxLog10Energy.
   *
   * The analysis binning is row-major with Log10Energy outermost (enforced by
   * ICDataBase::read_sample), so the visible bins are a prefix of every bin array
   * and one count describes them all. A bin straddling the threshold counts as
   * hidden. A binning that does not start with an energy axis returns 0 -- an
   * unrecognised layout blinds everything rather than guessing wrong.
   */
  [[nodiscard]] inline std::size_t visible_bins(const io::ic::Binning& binning) noexcept {
    const auto axes = binning.axes();
    if (axes.empty() || axes[0].kind != io::ic::Axis::Kind::Log10Energy)
      return 0;

    const io::ic::Axis& energy = axes[0];

    int visible_energy_bins = 0;
    while (visible_energy_bins < energy.n_bins) {
      const double upper_edge = energy.uniform()
                                    ? energy.lo + (visible_energy_bins + 1) * energy.step()
                                    : energy.edges[visible_energy_bins + 1];
      if (upper_edge > kBlindMaxLog10Energy + 1e-9)
        break;
      ++visible_energy_bins;
    }

    const int bins_per_energy_bin = binning.total_bins() / std::max(energy.n_bins, 1);
    return static_cast<std::size_t>(visible_energy_bins) * static_cast<std::size_t>(bins_per_energy_bin);
  }

  /**
   * Zeroes every bin the blinding hides, in place. Pass a copy -- never the
   * likelihood's own histogram, which the fit still needs intact.
   */
  inline void blind_bins(const io::ic::Binning& binning, std::span<double> bins) noexcept {
    const std::size_t visible = std::min(visible_bins(binning), bins.size());
    std::fill(bins.begin() + static_cast<std::ptrdiff_t>(visible), bins.end(), 0.0);
  }

}  // namespace result::ic
