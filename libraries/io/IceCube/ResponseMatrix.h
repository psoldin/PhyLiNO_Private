#pragma once

#include "Binning.h"

#include <cstddef>
#include <span>
#include <vector>

namespace io::ic {

  /**
   * The per-event detector response, projected onto the analysis binning.
   *
   * The ordinary histogram gives each MC event's whole weight to the single bin
   * its RECONSTRUCTED value falls into. That is a delta function at a value we
   * know to be uncertain: the reco is one draw from a response whose width the
   * reconstruction reports per event. This matrix replaces the delta with the
   * response,
   *
   *     prediction[b] = sum_i w_i(theta) * f_ib,
   *     f_ib          = integral over bin b of N(truth_i, sigma_i),
   *
   * which is the Rao-Blackwellised form of the same estimator: identical
   * expectation, lower variance, because each event contributes its full
   * conditional distribution instead of one realised draw. What it removes is
   * exactly the reconstruction-draw variance of the bin contents, and that is
   * the term that dominates in the sparse high-energy bins where the spectral
   * index is measured.
   *
   * The kernel must be centred on TRUTH. Smearing the reconstructed value by its
   * own sigma instead would fold the response in twice -- the reco is already
   * smeared by it -- and give a width of sqrt(2) sigma.
   *
   * Stored bin-major (CSR over bins) rather than event-major so the product is a
   * gather: each bin sums over its own entries with no write conflicts, matching
   * how every other per-bin reduction in this codebase is parallelised.
   *
   * Parameter-independent by construction -- truth, sigma and bin edges -- so it
   * is built once at load and never rebuilt, like the sample's bin assignment.
   */
  struct ResponseMatrix {
    std::vector<std::size_t> bin_offsets;  ///< size n_bins + 1
    std::vector<int>         events;       ///< event index per entry
    std::vector<float>       fractions;    ///< that event's response fraction in this bin

    [[nodiscard]] bool        empty() const noexcept { return fractions.empty(); }
    [[nodiscard]] std::size_t nnz() const noexcept { return fractions.size(); }
    [[nodiscard]] std::size_t bytes() const noexcept {
      return fractions.size() * (sizeof(float) + sizeof(int)) + bin_offsets.size() * sizeof(std::size_t);
    }
  };

  /**
   * Build the response matrix over `binning` for events at (`truth_log_e`,
   * `truth_zenith`) with per-event widths (`sigma_log_e`, `sigma_zenith`).
   *
   * `binning` must be the MC binning: a Log10Energy axis and a CosZenith axis,
   * in that order. The zenith response is Gaussian in the ANGLE while the axis
   * is in its cosine, so each bin's angular interval is taken through acos
   * rather than the response being pretended Gaussian in cos.
   *
   * Only bins within `truncation` widths are considered, and entries below
   * `min_fraction` of the event's total are dropped; both bound the matrix
   * without changing it materially, and what survives is renormalised so every
   * event's fractions sum to one.
   *
   * That renormalisation is deliberate. The sample is defined by a cut on the
   * reconstructed value, so every event in it did land inside the analysis
   * range; conditioned on that, its response is the truncated one. It also keeps
   * the folded prediction's total identical to the unfolded histogram's, so a
   * comparison between them is a comparison of shape and variance alone.
   *
   * An event whose response is unusable -- a non-finite truth or a non-positive
   * width -- keeps its unfolded behaviour: a single entry of 1.0 in the bin it
   * was assigned to. `bin_idx` supplies that fallback and is the sample's own
   * assignment; an event with bin_idx < 0 is not in the sample at all and is
   * skipped.
   */
  [[nodiscard]] ResponseMatrix build_response_matrix(const Binning&          binning,
                                                     std::span<const int>    bin_idx,
                                                     std::span<const double> truth_log_e,
                                                     std::span<const double> truth_zenith,
                                                     std::span<const double> sigma_log_e,
                                                     std::span<const double> sigma_zenith,
                                                     double truncation, double min_fraction,
                                                     std::size_t& unusable);

}  // namespace io::ic
