#pragma once

// STL includes
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <vector>

/**
 * @file
 * @brief Binary refinement of a one-dimensional profile likelihood scan.
 *
 * The one-dimensional counterpart of AdaptiveGrid.h: a coarse uniform set of
 * points along a single parameter, then repeated halving of only those
 * intervals a confidence level crosses. What matters in a profile is where the
 * curve cuts delta chi2 = 1, 4, 9 and where its minimum sits; the flat outer
 * wings carry no information and stay cheap.
 *
 * As in the two-dimensional case nothing here assumes a shape. An interval is
 * split from the values at its own two ends, so a double minimum, a curve that
 * leaves the window still rising and a plateau are all handled the same way.
 */
namespace scan1d {

  /// Values of the fit objective, keyed by lattice index at the finest depth.
  /// Missing entries are points never evaluated.
  ///
  /// The objective is expected in -2 log L units, which is what the fit
  /// minimises, so a difference of two values is already a delta chi2 and must
  /// not be scaled again. Same convention as scan::Surface.
  using Profile = std::map<int, double>;

  /// An interval of the refinement: left lattice index plus its length in
  /// lattice units.
  struct Segment {
    int origin = 0;
    int size   = 0;
  };

  struct Settings {
    /// Intervals of the initial uniform scan. Coarse, but fine enough that no
    /// plausible minimum hides between two of its points.
    int coarse_cells = 16;

    /// Refinement depth applied to intervals a level crosses.
    int max_depth = 4;

    /// Refinement depth applied to intervals inside the outermost level but not
    /// crossed by one themselves.
    ///
    /// This is where the minimum lives, and the minimum is what every reported
    /// interval is measured from, so it is refined as hard as the crossings.
    /// One dimension makes that cheap: full depth over the whole region costs
    /// coarse_cells * 2^max_depth points at worst, a few hundred fits.
    int region_depth = 4;

    /// Levels to resolve, as delta chi2. The largest one bounds the refined
    /// region.
    ///
    /// The first three are the 1, 2 and 3 sigma points of a one-dimensional
    /// profile, which is what the crossings are read off at. The fourth is not
    /// reported: it is there so the 3 sigma crossing falls inside the refined
    /// region instead of sitting on its edge, the same trick AdaptiveGrid uses
    /// with 11.83.
    std::vector<double> levels{1.0, 4.0, 9.0, 12.0};

    /// Length of a coarse interval in lattice units.
    [[nodiscard]] int coarse_step() const { return 1 << max_depth; }

    /// Number of lattice intervals spanning the scanned range.
    [[nodiscard]] int lattice() const { return coarse_cells << max_depth; }
  };

  /**
   * @brief Decides whether an interval has to be halved.
   *
   * @param profile   Values known so far.
   * @param settings  Refinement settings.
   * @param segment   Interval under consideration.
   * @param reference Lowest likelihood seen, which the levels are measured from.
   * @param depth     Current refinement depth of the interval.
   */
  inline bool needs_split(const Profile& profile, const Settings& settings, const Segment& segment, double reference, int depth) {
    double lowest  = std::numeric_limits<double>::infinity();
    double highest = -std::numeric_limits<double>::infinity();

    for (const int dx : {0, segment.size}) {
      const auto end = profile.find(segment.origin + dx);
      // An end that is missing or unusable leaves the interval undecidable, so
      // it is refined rather than silently trusted.
      if (end == profile.end() || !std::isfinite(end->second))
        return depth < settings.max_depth;

      const double delta = end->second - reference;
      lowest             = std::min(lowest, delta);
      highest            = std::max(highest, delta);
    }

    const bool on_level = std::ranges::any_of(settings.levels, [&](double level) { return lowest <= level && level <= highest; });
    if (on_level)
      return depth < settings.max_depth;

    const bool inside = !settings.levels.empty() && lowest <= std::ranges::max(settings.levels);
    return inside && depth < settings.region_depth;
  }

  /// Lattice indices of the initial uniform scan.
  inline std::vector<int> coarse_nodes(const Settings& settings) {
    const int step = settings.coarse_step();

    std::vector<int> nodes;
    nodes.reserve(static_cast<std::size_t>(settings.coarse_cells) + 1);
    for (int i = 0; i <= settings.coarse_cells; ++i)
      nodes.push_back(i * step);

    return nodes;
  }

  /// Intervals of the initial uniform scan.
  inline std::vector<Segment> coarse_segments(const Settings& settings) {
    const int step = settings.coarse_step();

    std::vector<Segment> segments;
    segments.reserve(static_cast<std::size_t>(settings.coarse_cells));
    for (int i = 0; i < settings.coarse_cells; ++i)
      segments.push_back(Segment{i * step, step});

    return segments;
  }

  /// The node a split adds: the midpoint. The ends are already there, and
  /// neighbouring intervals share them.
  inline void split_nodes(const Segment& segment, std::vector<int>& nodes) {
    nodes.push_back(segment.origin + segment.size / 2);
  }

  /// The two intervals a split produces.
  inline void split_segments(const Segment& segment, std::vector<Segment>& segments) {
    const int half = segment.size / 2;

    segments.push_back(Segment{segment.origin, half});
    segments.push_back(Segment{segment.origin + half, half});
  }

  /// Lowest finite value on the profile; infinity if there is none.
  inline double reference_value(const Profile& profile) {
    double lowest = std::numeric_limits<double>::infinity();
    for (const auto& [node, value] : profile)
      if (std::isfinite(value))
        lowest = std::min(lowest, value);

    return lowest;
  }

  /**
   * @brief Runs the coarse scan and all refinement rounds.
   *
   * @param settings  Refinement settings.
   * @param seed      Likelihood of a free fit, used as the reference until the
   *                  scan produces a lower one. Pass infinity if there is none.
   * @param evaluate  Callable `void(const std::vector<int>&, Profile&)` that
   *                  fills in every node it is handed. It may be called with
   *                  nodes that already have a value; those are filtered out
   *                  before the call.
   * @param report    Callable `void(int round, std::size_t split, std::size_t segments, std::size_t points)`
   *                  invoked once per round, for progress output.
   */
  template <typename Evaluator, typename Reporter>
  Profile refine(const Settings& settings, double seed, Evaluator&& evaluate, Reporter&& report) {
    Profile profile;

    auto evaluate_missing = [&](std::vector<int> nodes) {
      std::ranges::sort(nodes);
      const auto repeated = std::ranges::unique(nodes);
      nodes.erase(repeated.begin(), repeated.end());
      std::erase_if(nodes, [&](int node) { return profile.contains(node); });

      evaluate(nodes, profile);
    };

    evaluate_missing(coarse_nodes(settings));
    report(0, std::size_t{0}, std::size_t{0}, profile.size());

    std::vector<Segment> segments = coarse_segments(settings);

    for (int depth = 0; depth < settings.max_depth; ++depth) {
      const double reference = std::min(seed, reference_value(profile));

      std::vector<Segment> split;
      for (const Segment& segment : segments)
        if (needs_split(profile, settings, segment, reference, depth))
          split.push_back(segment);

      report(depth + 1, split.size(), segments.size(), profile.size());

      if (split.empty())
        break;

      std::vector<int>     nodes;
      std::vector<Segment> children;
      for (const Segment& segment : split) {
        split_nodes(segment, nodes);
        split_segments(segment, children);
      }

      evaluate_missing(std::move(nodes));
      segments = std::move(children);
    }

    return profile;
  }

}  // namespace scan1d
