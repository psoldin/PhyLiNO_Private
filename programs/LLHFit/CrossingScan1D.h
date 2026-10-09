#pragma once

#include "AdaptiveScan1D.h"

// STL includes
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

/**
 * @file
 * @brief Direct search for the confidence crossings of a one-dimensional profile.
 *
 * The outward walk of AdaptiveScan1D.h maps the whole profile out to 4 sigma,
 * which is what a plot needs and far more than an interval does. This search
 * only looks for the two points where the profile has risen by `level` above
 * its minimum -- 1 for the 1 sigma interval -- and puts its fits there.
 *
 * It does so as a root search in z = sqrt(delta chi2) rather than in delta chi2
 * itself. On a parabolic profile z is exactly linear in the parameter, so a
 * secant step lands on the crossing at once whatever the starting step was; on
 * a real profile z is still close to linear around the crossing, which is all
 * the search relies on. Nothing about the width is taken from Hesse: the first
 * step is only a probe, and every later one comes from the fitted points.
 *
 * Each side is solved on its own, in two phases:
 *
 *  - Unbracketed: every point so far is below the level. The next point is the
 *    secant through the two outermost points, extrapolated to z = sqrt(level),
 *    with its reach capped so one nearly flat pair cannot send the search far
 *    past the crossing.
 *  - Bracketed: a point at or above the level exists. The next point is the
 *    secant interpolated between the outermost point below and the innermost
 *    point above, kept off the ends of the bracket so the bracket keeps
 *    shrinking even where the profile bends.
 *
 * A side is done once a fitted point lies within `tolerance` of the level, once
 * the bracket is down to `bracket_width` lattice nodes, or once it runs into a
 * configured bound below the level. The reported crossing is the interpolation
 * in z between the bracketing points, which is far more precise than the
 * tolerance on the point itself.
 *
 * The state of a side is re-derived from the profile every round instead of
 * being carried along. A scan point that undercuts the free fit moves the
 * minimum and with it every delta chi2; deriving the brackets afresh is what
 * keeps the search measuring from the lowest point known, just like the walk.
 * Unlike the walk, though, the search never samples the neighbourhood of the
 * minimum, so it cannot find a better one there: an undercut is flagged in the
 * result rather than silently absorbed.
 *
 * Points use the same integer lattice as the walk (see AdaptiveScan1D.h), so a
 * search is resumable in the same way.
 */
namespace scan1d::crossing {

  struct Settings {
    /// Rise above the minimum whose crossings are searched for: 1 for the 1
    /// sigma interval of one parameter, 4 for 2 sigma, 2.71 for 90 %.
    double level = 1.0;

    /// A side is done once a fitted point lies this close to the level, in
    /// delta chi2. The crossing itself comes from interpolating the bracketing
    /// points, so it is known much better than this.
    double tolerance = 0.02;

    /// First probe step in lattice units, used only while a side has no point
    /// of its own besides the minimum.
    int start_step = 64;

    /// How far beyond the last point a side may extrapolate, as a multiple of
    /// that point's distance from the minimum. Large enough that a probe far
    /// too short reaches the crossing within a round or two, small enough that
    /// a pair of points sitting in noise cannot jump across the whole profile.
    double max_growth = 8.0;

    /// Points proposed per side per round. The first is the secant estimate,
    /// the others sit around it so a round has a good chance to bracket the
    /// crossing tightly on its own.
    int batch = 1;

    /// Spacing of the extra points of a batch, as a fraction of the distance
    /// from the minimum (unbracketed) or of the bracket width (bracketed).
    double spread = 0.1;

    /// Interpolated points are kept at least this fraction of the bracket away
    /// from either end, so a bracket on a bending profile still shrinks.
    double margin = 0.05;
    /// A side is also done once its bracket is down to this many lattice units.
    /// The crossing is the interpolation in z between the two bracketing
    /// points, and z is close to linear over a bracket this narrow (with the
    /// default lattice of 64 units per Hesse error, 4 units are 1/16 of a sigma),
    /// so tightening it further buys nothing but fits. Before this, a point that
    /// missed `tolerance` by a hair was followed by a fit at every lattice node
    /// until two adjacent ones bracketed the level. 1 restores that behaviour.
    int bracket_width = 4;

    /// Safety cap on the points of one side.
    int max_points = 30;

    /// Hard limits in lattice units, from the parameter's configured bounds.
    int lower_limit = -(1 << 24);
    int upper_limit = 1 << 24;
  };

  enum class Status {
    converged,  ///< A point lies within tolerance of the level, or the bracket is down to bracket_width lattice units.
    at_limit,   ///< The profile stays below the level up to the configured bound; the crossing is the bound.
    failed,     ///< Fits next to the last usable point produced no likelihood.
    exhausted,  ///< max_points used without convergence.
  };

  struct Side {
    double crossing = std::numeric_limits<double>::quiet_NaN();  ///< Offset from node 0 in lattice units, fractional.
    Status status   = Status::exhausted;
    int    points   = 0;  ///< Points of the profile on this side of the minimum.
  };

  struct Result {
    Profile             profile;
    int                 minimum   = 0;  ///< Lowest node of the profile.
    double              reference = std::numeric_limits<double>::infinity();
    std::array<Side, 2> sides;  ///< [0] below the minimum, [1] above it.

    /// A scan point came out below the free fit by more than the tolerance. The
    /// crossings are then measured from that point, but the search samples too
    /// sparsely to find the true minimum: the free fit needs redoing (e.g. via
    /// --seedFrom that point) before the interval can be trusted.
    bool undercut = false;
  };

  namespace detail {

    /// What a side knows about its crossing, read off the profile.
    struct Bracket {
      int                direction = 1;
      std::vector<int>   inner;  ///< Points below the level, from the minimum outward; starts with the minimum.
      std::optional<int> outer;  ///< First point at or above the level beyond inner.
      std::optional<int> cap;    ///< First point without a likelihood beyond inner, if it comes before outer.
    };

    inline double z_of(const Profile& profile, double reference, int node) {
      return std::sqrt(std::max(0.0, profile.at(node) - reference));
    }

    /// Walks outward from the minimum and stops at the first point that is at
    /// or above the level or has no likelihood: the first crossing is the one
    /// the interval ends at, whatever the profile does further out.
    inline Bracket bracket(const Profile& profile, double reference, int minimum, int direction, double level) {
      Bracket result;
      result.direction = direction;
      result.inner.push_back(minimum);

      auto visit = [&](int node, double value) {
        if (!std::isfinite(value)) {
          result.cap = node;
          return false;
        }
        if (value - reference >= level) {
          result.outer = node;
          return false;
        }
        result.inner.push_back(node);
        return true;
      };

      if (direction > 0) {
        for (auto it = profile.upper_bound(minimum); it != profile.end(); ++it)
          if (!visit(it->first, it->second))
            break;
      } else {
        for (auto it = std::make_reverse_iterator(profile.lower_bound(minimum)); it != profile.rend(); ++it)
          if (!visit(it->first, it->second))
            break;
      }

      return result;
    }

    /// Secant estimate of the crossing in z, NaN if the points do not rise.
    inline double estimate(const Profile& profile, double reference, const Bracket& side, double target_z) {
      const int    b  = side.inner.back();
      const double zb = z_of(profile, reference, b);

      if (side.outer) {
        const double zo = z_of(profile, reference, *side.outer);
        if (!(zo > zb))
          return static_cast<double>(*side.outer);
        return b + (target_z - zb) * (*side.outer - b) / (zo - zb);
      }

      if (side.inner.size() < 2)
        return std::numeric_limits<double>::quiet_NaN();

      const int    a  = side.inner[side.inner.size() - 2];
      const double za = z_of(profile, reference, a);
      if (!(zb > za))
        return std::numeric_limits<double>::quiet_NaN();
      return b + (target_z - zb) * (b - a) / (zb - za);
    }

    inline int count_points(const Profile& profile, int minimum, int direction) {
      int n = 0;
      for (const auto& [node, value] : profile)
        n += direction > 0 ? node > minimum : node < minimum;
      return n;
    }

    inline int limit_of(const Settings& settings, int direction) {
      return direction > 0 ? settings.upper_limit : settings.lower_limit;
    }

    /// Whether a side is done, and if so with what result. Empty while the side
    /// still wants points.
    inline std::optional<Side> finished(const Settings& settings, const Profile& profile, double reference, int minimum, const Bracket& side) {
      const int    direction = side.direction;
      const int    b         = side.inner.back();
      const double target_z  = std::sqrt(settings.level);

      Side result;
      result.points = count_points(profile, minimum, direction);

      auto near_level = [&](int node) { return std::abs(profile.at(node) - reference - settings.level) <= settings.tolerance; };

      if (side.outer) {
        if (near_level(b) || near_level(*side.outer) || std::abs(*side.outer - b) <= std::max(1, settings.bracket_width)) {
          result.crossing = estimate(profile, reference, side, target_z);
          result.status   = Status::converged;
          return result;
        }
      } else {
        if (b != minimum && near_level(b)) {
          // Just below the level: a short extrapolation, or the point itself
          // when the last pair does not rise.
          const double x  = estimate(profile, reference, side, target_z);
          result.crossing = std::isfinite(x) ? x : static_cast<double>(b);
          result.status   = Status::converged;
          return result;
        }

        if (b == limit_of(settings, direction)) {
          result.crossing = static_cast<double>(b);
          result.status   = Status::at_limit;
          return result;
        }

        if (side.cap && std::abs(*side.cap - b) <= 1) {
          result.crossing = std::numeric_limits<double>::quiet_NaN();
          result.status   = Status::failed;
          return result;
        }
      }

      if (result.points >= settings.max_points) {
        result.crossing = estimate(profile, reference, side, target_z);
        result.status   = Status::exhausted;
        return result;
      }

      return std::nullopt;
    }

    /// The nodes a side wants fitted next.
    inline std::vector<int> proposals(const Settings& settings, const Profile& profile, double reference, int minimum, const Bracket& side) {
      const int    direction = side.direction;
      const int    b         = side.inner.back();
      const double target_z  = std::sqrt(settings.level);

      // Strictly between `b` and `end` is where the next point may go.
      std::optional<int> end = side.outer ? side.outer : side.cap;

      const int limit = limit_of(settings, direction);

      double guess = estimate(profile, reference, side, target_z);
      double scale = 0.0;  // Spacing unit of the extra batch points.

      if (side.outer) {
        const double width = std::abs(*side.outer - b);
        const double low   = b + direction * settings.margin * width;
        const double high  = *side.outer - direction * settings.margin * width;
        guess              = direction > 0 ? std::clamp(guess, low, high) : std::clamp(guess, high, low);
        scale              = width;
      } else {
        const double reach = std::max(1.0, settings.max_growth * std::abs(b - minimum));

        if (!std::isfinite(guess)) {
          // Nothing to extrapolate from yet: the probe on the first round, and
          // a step as long as allowed when the profile has not started rising.
          guess = b == minimum ? static_cast<double>(minimum + direction * std::max(1, settings.start_step)) : b + direction * reach;
        } else {
          const double distance = std::clamp(direction * (guess - b), 1.0, reach);
          guess                 = b + direction * distance;
        }

        // Beyond a failed fit nothing is known; go back halfway towards it.
        if (side.cap && direction * (guess - *side.cap) >= 0.0)
          guess = 0.5 * (b + *side.cap);

        scale = std::abs(guess - minimum);
      }

      auto inside = [&](int node) {
        if (direction * (node - b) <= 0)
          return false;
        if (end && direction * (node - *end) >= 0)
          return false;
        return direction > 0 ? node <= limit : node >= limit;
      };

      std::vector<int> nodes;
      auto             add = [&](double x) {
        int node = static_cast<int>(std::lround(x));
        if (!end)
          node = direction > 0 ? std::min(node, limit) : std::max(node, limit);
        if (inside(node) && !profile.contains(node) && std::ranges::find(nodes, node) == nodes.end())
          nodes.push_back(node);
      };

      add(guess);

      // A guess that rounded onto a point already fitted, or onto the edge of
      // the bracket, still has a free node next to it more often than not.
      if (nodes.empty()) {
        add(guess + direction);
        add(guess - direction);
      }

      // Extra points around the guess: outward first, since an underestimate is
      // what leaves a side unbracketed.
      for (int k = 1; static_cast<int>(nodes.size()) < std::max(1, settings.batch) && k <= 4 * settings.batch; ++k) {
        const int    step   = (k + 1) / 2;
        const double offset = (k % 2 == 1 ? 1.0 : -1.0) * direction * step * settings.spread * scale;
        if (std::abs(offset) < 1.0)
          continue;
        add(guess + offset);
      }

      return nodes;
    }

    /// Lowest node, the one nearest the anchor among equals: on a flat stretch
    /// the minimum must not wander to whichever point happens to be outermost.
    inline int lowest_node(const Profile& profile) {
      int    best  = 0;
      double value = std::numeric_limits<double>::infinity();
      for (const auto& [node, v] : profile) {
        if (std::isfinite(v) && (v < value || (v == value && std::abs(node) < std::abs(best)))) {
          value = v;
          best  = node;
        }
      }
      return best;
    }

  }  // namespace detail

  /**
   * @brief Searches both crossings of the profile with the level.
   *
   * @param settings Search settings.
   * @param seed     Likelihood of the free fit, used as the reference until a
   *                 scan point undercuts it. Pass infinity if there is none.
   * @param evaluate Callable `void(const std::vector<int>&, Profile&)` that fills
   *                 in every node it is handed, as for scan1d::walk.
   * @param report   Callable `void(int round, std::size_t proposed, std::size_t points)`
   *                 invoked once per round, for progress output.
   */
  template <typename Evaluator, typename Reporter>
  Result search(const Settings& settings, double seed, Evaluator&& evaluate, Reporter&& report) {
    Result result;
    auto&  profile = result.profile;

    evaluate(std::vector<int>{0}, profile);
    report(0, std::size_t{1}, profile.size());

    for (int round = 1;; ++round) {
      result.reference = std::min(seed, reference_value(profile));
      result.minimum   = detail::lowest_node(profile);

      std::vector<int> nodes;
      for (std::size_t i = 0; i < result.sides.size(); ++i) {
        const int  direction = i == 0 ? -1 : +1;
        const auto side      = detail::bracket(profile, result.reference, result.minimum, direction, settings.level);

        if (const auto done = detail::finished(settings, profile, result.reference, result.minimum, side)) {
          result.sides[i] = *done;
          continue;
        }

        const auto proposed = detail::proposals(settings, profile, result.reference, result.minimum, side);
        if (proposed.empty()) {
          // Nowhere left to put a point: the bracket is as tight as the lattice
          // allows, so take what the points there say.
          Side stuck;
          stuck.points    = detail::count_points(profile, result.minimum, direction);
          stuck.crossing  = detail::estimate(profile, result.reference, side, std::sqrt(settings.level));
          stuck.status    = side.outer ? Status::converged : Status::failed;
          result.sides[i] = stuck;
          continue;
        }

        nodes.insert(nodes.end(), proposed.begin(), proposed.end());
      }

      if (nodes.empty())
        break;

      report(round, nodes.size(), profile.size());
      evaluate(nodes, profile);
    }

    result.undercut = std::isfinite(seed) && result.reference < seed - settings.tolerance;
    return result;
  }

}  // namespace scan1d::crossing
