#pragma once

// STL includes
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <vector>

/**
 * @file
 * @brief Outward walk of a one-dimensional profile likelihood scan.
 *
 * The scan starts at the best fit and steps away from it along the parameter
 * axis until the profile has risen by a given delta chi2 -- 16 by default, well
 * past the 3 sigma crossing -- then stops. Nothing about the window is guessed
 * in advance: how far the walk goes is decided by the likelihood itself, so a
 * parameter that is tightly constrained gets a short scan and a loose one a long
 * one without either being configured.
 *
 * Step size is chosen the same way. After every round the local slope of the
 * profile is measured and the next step is sized to raise the likelihood by
 * about `gain`, so points come out dense where the curve bends and sparse where
 * it is flat, with no assumption that the profile is parabolic, symmetric, or
 * even single-minimum.
 *
 * Points sit on integer offsets from the best fit, in units of `unit` (chosen by
 * the caller, normally a small fraction of the parameter's step width). Integer
 * keys keep every point exactly representable, give it a stable name across
 * runs -- which is what makes a scan resumable -- and let two points of
 * different rounds coincide instead of drifting apart by a rounding error.
 *
 * A round proposes several points per direction at once so a worker pool has
 * something to chew on: the walk is sequential in its decisions but not in its
 * fits.
 */
namespace scan1d {

  /// Values of the fit objective, keyed by offset from the best fit in lattice
  /// units. Missing entries are points never evaluated.
  ///
  /// The objective is expected in -2 log L units, which is what the fit
  /// minimises, so a difference of two values is already a delta chi2 and must
  /// not be scaled again. Same convention as scan::Surface.
  using Profile = std::map<int, double>;

  struct Settings {
    /// Rise above the best fit at which a direction stops. 16 is 4 sigma for a
    /// one-dimensional profile, so the 1, 2 and 3 sigma crossings all fall
    /// strictly inside the scanned range instead of on its edge.
    double target = 16.0;

    /// Rise the next step aims for. Sets the point density: the whole walk out
    /// to `target` costs roughly target/gain points per direction wherever the
    /// slope estimate holds.
    double gain = 1.0;

    /// Step of the first round, in lattice units. Only a starting guess; from
    /// the second round on the step comes from the measured slope.
    int start_step = 64;

    /// Smallest and largest step, in lattice units. The lower bound keeps a
    /// near-vertical profile from stalling on ever smaller steps; the upper one
    /// keeps a flat one from jumping over the rise.
    int min_step = 1;
    int max_step = 1 << 14;

    /// Points proposed per direction per round. Larger batches keep more
    /// workers busy at the cost of overshooting the target by up to a batch.
    int batch = 4;

    /// Safety cap on the points of one direction, so a profile that never rises
    /// (an unconstrained parameter) ends instead of running forever.
    int max_points = 200;

    /// Hard limits in lattice units, from the parameter's configured bounds.
    /// A direction that reaches one stops there: outside them the minimiser
    /// cannot go, so neither can an honest profile.
    int lower_limit = -(1 << 24);
    int upper_limit = 1 << 24;
  };

  /// Lowest finite value on the profile; infinity if there is none.
  inline double reference_value(const Profile& profile) {
    double lowest = std::numeric_limits<double>::infinity();
    for (const auto& [node, value] : profile)
      if (std::isfinite(value))
        lowest = std::min(lowest, value);

    return lowest;
  }

  namespace detail {

    /// One direction of the walk.
    struct Arm {
      int  direction = 1;
      int  last      = 0;  ///< Outermost node evaluated so far that is finite and below the target.
      int  step      = 1;  ///< Step to the next point, in lattice units.
      int  points    = 0;
      bool active    = true;

      /// The first point found at or past the target, if there is one. The walk
      /// does not stop the moment it steps over the target: a step sized for a
      /// wider profile can clear the whole rise in one jump and leave nothing
      /// sampled in between. Instead the crossing point becomes a wall and the
      /// direction keeps filling the gap behind it until the last step is worth
      /// no more than `gain`, which is what makes the walk independent of how
      /// well the configured step width happens to match the true width.
      bool bounded = false;
      int  wall    = 0;
    };

    inline double value_at(const Profile& profile, int node) {
      const auto found = profile.find(node);
      return found == profile.end() ? std::numeric_limits<double>::quiet_NaN() : found->second;
    }

    /// The nodes a direction wants fitted next. Stops early at a configured
    /// bound -- past it there is nothing to profile -- and at its own wall,
    /// beyond which the profile is already known to be above the target.
    inline std::vector<int> proposals(const Settings& settings, const Arm& arm) {
      std::vector<int> nodes;
      if (!arm.active)
        return nodes;

      nodes.reserve(static_cast<std::size_t>(std::max(1, settings.batch)));

      for (int i = 1; i <= std::max(1, settings.batch); ++i) {
        const int raw = arm.last + arm.direction * arm.step * i;
        if (arm.bounded && (arm.direction > 0 ? raw >= arm.wall : raw <= arm.wall))
          break;

        const int node = std::clamp(raw, settings.lower_limit, settings.upper_limit);
        if (node == arm.last)
          break;

        nodes.push_back(node);
        if (node != raw)
          break;
      }

      return nodes;
    }

    /// Reads back one direction's round and decides where it goes next.
    inline void advance(const Settings& settings, Arm& arm, const Profile& profile, double reference, const std::vector<int>& nodes) {
      int    previous       = arm.last;
      double previous_delta = value_at(profile, arm.last) - reference;
      if (!std::isfinite(previous_delta))
        previous_delta = 0.0;

      bool failed = false;

      for (const int node : nodes) {
        const double value = value_at(profile, node);

        // A point without a usable likelihood says nothing about the slope, and
        // stepping past it would extrapolate from the last point that did. The
        // direction backs off to a smaller step and tries again from where it
        // still has a value.
        if (!std::isfinite(value)) {
          failed = true;
          break;
        }

        ++arm.points;

        // A point at or past the target is a wall, not a place to walk on from:
        // what is still missing lies behind it.
        if (value - reference >= settings.target) {
          arm.bounded = true;
          arm.wall    = node;
          break;
        }

        previous       = arm.last;
        previous_delta = value_at(profile, arm.last) - reference;
        if (!std::isfinite(previous_delta))
          previous_delta = 0.0;

        arm.last = node;
      }

      if (arm.points >= settings.max_points || arm.last == settings.lower_limit || arm.last == settings.upper_limit) {
        arm.active = false;
        return;
      }

      if (failed) {
        if (arm.step <= settings.min_step)
          arm.active = false;
        arm.step = std::max(settings.min_step, arm.step / 2);
        return;
      }

      const double last_delta = value_at(profile, arm.last) - reference;

      // Size the next step from the slope the last two points measured, so the
      // profile rises by about `gain` per step. Behind a wall the slope of the
      // gap itself is the better estimate: it is the stretch still to be filled.
      int    span  = std::abs(arm.last - previous);
      double delta = last_delta - previous_delta;
      if (arm.bounded) {
        span  = std::abs(arm.wall - arm.last);
        delta = (value_at(profile, arm.wall) - reference) - last_delta;
      }

      const double slope = span > 0 && std::isfinite(delta) ? delta / span : 0.0;

      // Clamped as a double first: a slope of 1e-12 would otherwise ask for a
      // step no int can hold.
      const double wanted_step = slope > 0.0 ? std::clamp(settings.gain / slope, 1.0, static_cast<double>(settings.max_step))
                                             : static_cast<double>(settings.max_step);

      // Growing the step by more than a factor of four in one round would let a
      // single flat pair of points decide the rest of the walk. Shrinking is not
      // damped: that is the walk correcting a step that was too big, and the
      // sooner it does the fewer points it wastes past the target.
      arm.step = std::clamp(static_cast<int>(std::lround(wanted_step)), settings.min_step, std::min(settings.max_step, arm.step * 4));

      if (arm.bounded) {
        // The direction is done once the gap behind the wall is worth less than
        // one step's rise, or is too narrow to put another point in.
        if (span <= settings.min_step || delta <= settings.gain) {
          arm.active = false;
          return;
        }

        // Whatever the slope suggests, the next point has to land inside the gap.
        arm.step = std::clamp(arm.step, settings.min_step, std::max(settings.min_step, span / 2));
      }
    }

  }  // namespace detail

  /**
   * @brief Walks outward from the best fit until the profile has risen by `target`.
   *
   * @param settings Walk settings.
   * @param seed     Likelihood of the free fit, used as the reference until the
   *                 walk produces a lower one. Pass infinity if there is none.
   * @param evaluate Callable `void(const std::vector<int>&, Profile&)` that fills
   *                 in every node it is handed. It may be called with nodes that
   *                 already have a value; those are filtered out before the call.
   * @param report   Callable `void(int round, std::size_t proposed, std::size_t points)`
   *                 invoked once per round, for progress output.
   */
  template <typename Evaluator, typename Reporter>
  Profile walk(const Settings& settings, double seed, Evaluator&& evaluate, Reporter&& report) {
    Profile profile;

    auto evaluate_missing = [&](std::vector<int> nodes) {
      std::ranges::sort(nodes);
      const auto repeated = std::ranges::unique(nodes);
      nodes.erase(repeated.begin(), repeated.end());
      std::erase_if(nodes, [&](int node) { return profile.contains(node); });

      if (!nodes.empty())
        evaluate(nodes, profile);
    };

    // The best fit itself. Every rise is measured from here, and it is the point
    // the two directions start from.
    evaluate_missing({0});
    report(0, std::size_t{1}, profile.size());

    std::array<detail::Arm, 2> arms{detail::Arm{-1, 0, std::max(1, settings.start_step)},
                                    detail::Arm{+1, 0, std::max(1, settings.start_step)}};

    int round = 1;
    for (; arms[0].active || arms[1].active; ++round) {
      std::array<std::vector<int>, 2> proposed{detail::proposals(settings, arms[0]), detail::proposals(settings, arms[1])};

      std::vector<int> nodes;
      for (const auto& arm_nodes : proposed)
        nodes.insert(nodes.end(), arm_nodes.begin(), arm_nodes.end());

      // Both directions out of moves at once: nothing left to fit.
      if (nodes.empty())
        break;

      report(round, nodes.size(), profile.size());
      evaluate_missing(nodes);

      // Taken fresh each round: a scan point may undercut the free fit, and
      // every rise is measured from the lowest point known.
      const double reference = std::min(seed, reference_value(profile));

      for (std::size_t i = 0; i < arms.size(); ++i) {
        if (!arms[i].active || proposed[i].empty())
          continue;
        detail::advance(settings, arms[i], profile, reference, proposed[i]);
      }
    }

    // The walk only guarantees its density behind the wall it stopped at: a
    // round whose step was still growing can clear a whole stretch of the
    // profile in one jump, and on a steep parameter that stretch is wide enough
    // to hide the 1 sigma crossing entirely. Every neighbouring pair that rises
    // by more than `gain` is halved until it does not, which gives the finished
    // curve one density everywhere instead of one near the target and another
    // in the middle.
    for (;; ++round) {
      const double reference = std::min(seed, reference_value(profile));

      std::vector<int> nodes;
      for (auto point = profile.begin(); point != profile.end() && std::next(point) != profile.end(); ++point) {
        const auto [left, left_value]   = *point;
        const auto [right, right_value] = *std::next(point);

        if (!std::isfinite(left_value) || !std::isfinite(right_value))
          continue;

        // Nothing to gain past the target, and nothing to split when the two
        // points are already neighbours on the lattice.
        if (right - left < 2 || std::min(left_value, right_value) - reference >= settings.target)
          continue;

        if (std::abs(right_value - left_value) > settings.gain)
          nodes.push_back(left + (right - left) / 2);
      }

      if (nodes.empty())
        break;

      report(round, nodes.size(), profile.size());
      evaluate_missing(std::move(nodes));
    }

    return profile;
  }

}  // namespace scan1d
