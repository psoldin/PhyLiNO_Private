#pragma once

// STL includes
#include <cmath>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

/**
 * @file
 * @brief When a fit's parameters are good enough to start another fit from.
 *
 * A scan point is the same fit with one parameter moved a little, so the
 * converged nuisance parameters of a neighbour are the best start values
 * available. The question is which neighbours count.
 *
 * Migrad's own converged/failed flag is the wrong test. It compares EDM against
 * `0.002 * tolerance`, i.e. 1e-4 by default, which is a statement about how
 * precisely the minimum was located and not about whether the fit ended up
 * anywhere near it. A point that stops at EDM 1e-3 is four orders of magnitude
 * short of that threshold and still sits a thousandth of a unit above its own
 * minimum -- a start point indistinguishable from a perfect one. Gating on the
 * flag threw all of those away, and when a run's convergence rate is low it
 * empties the seed store altogether: nearly every fit then cold-starts from the
 * global best fit, which far from the minimum is a poor enough guess to drop
 * into a different local minimum of the nuisance parameters.
 *
 * EDM is the honest test because it is measured in the same -2 log L units as
 * the profile itself: it bounds how far above its own minimum the fit stopped.
 * Below 1 -- the rise that defines the 1 sigma interval -- the parameter vector
 * is inside the region the scan is trying to resolve, and seeding from it can
 * only save Migrad iterations. Above it the fit has not found its minimum in any
 * useful sense and its parameters must not spread to its neighbours.
 */
namespace seedquality {

  /// Largest EDM whose parameter vector is still worth starting a fit from, in
  /// -2 log L units. One is the 1 sigma rise of a one-dimensional profile.
  constexpr double kMaxEdm = 1.0;

  /**
   * @brief Whether a finished fit's parameters may seed another fit.
   *
   * @param converged Migrad's own verdict. A converged fit is always usable:
   *                  its EDM is below the tolerance by construction, and the
   *                  flag also covers a stored result whose EDM was not
   *                  recorded (pass infinity for `edm` in that case).
   * @param edm       Estimated distance to the minimum at the end of the fit.
   */
  [[nodiscard]] inline bool usable(bool converged, double edm) noexcept {
    return converged || (std::isfinite(edm) && edm >= 0.0 && edm < kMaxEdm);
  }

  /**
   * @brief Start vector, in minimizer order, from a stored fit's parameter block.
   *
   * Names the file does not carry keep their configured start value instead of
   * invalidating the whole vector. That case is not a corrupt file: --blind
   * drops the six signal parameters from every result it writes, so under
   * blinding no stored fit carries a complete block, and rejecting them all
   * silently disabled warm start for the whole scan -- every point then
   * cold-started from the free fit, which is the condition that lets a distant
   * point settle into a different local minimum.
   *
   * A file sharing no name at all with the config is a different fit, not a
   * partial one, and is dropped -- which is the case the strict rule was really
   * guarding against.
   *
   * @param names      Parameter names in minimizer order.
   * @param stored     Fitted value by name, as read back from the result file.
   * @param default_of `double(std::size_t)` giving the configured start value of
   *                   a parameter, used for the names `stored` does not carry.
   * @param filled     Set to the number of names taken from `stored`.
   * @return The start vector, or empty if nothing matched.
   */
  template <typename DefaultOf>
  [[nodiscard]] std::vector<double> start_vector(const std::vector<std::string>& names, const std::map<std::string, double>& stored, DefaultOf&& default_of, std::size_t& filled) {
    std::vector<double> values;
    values.reserve(names.size());

    filled = 0;
    for (std::size_t i = 0; i < names.size(); ++i) {
      const auto it = stored.find(names[i]);
      if (it == stored.end()) {
        values.push_back(default_of(i));
        continue;
      }
      values.push_back(it->second);
      ++filled;
    }

    if (filled == 0)
      return {};

    return values;
  }

}  // namespace seedquality
