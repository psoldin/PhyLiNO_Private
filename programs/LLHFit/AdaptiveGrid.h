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
 * @brief Quadtree refinement of a two-dimensional likelihood scan.
 *
 * A plain grid spends the same number of fits everywhere, while the confidence
 * contours occupy a vanishing fraction of the window. This refines a coarse grid
 * only where a contour actually passes, so the cost follows the length of the
 * contours instead of the area of the window.
 *
 * Nothing here assumes a shape: the decision to subdivide a cell is taken from
 * the values at that cell's own corners. A banana, a region running off the edge
 * of the window and several disconnected islands are all handled the same way.
 */
namespace scan {

  /// A point of the refinement lattice, in integer coordinates at the finest depth.
  struct Node {
    int x = 0;
    int y = 0;

    friend auto operator<=>(const Node&, const Node&) = default;
  };

  /// A square of the quadtree: lower-left node plus its side length in lattice units.
  struct Cell {
    Node origin;
    int  size = 0;
  };

  /// Values of the fit objective, keyed by lattice node. Missing entries are
  /// points never evaluated.
  ///
  /// The objective is expected in -2 log L units, which is what the fit
  /// minimises: both IceCube likelihoods return -2 * sum(log L) and the Gaussian
  /// pulls are summed as squared deviations. The difference between two values
  /// is therefore already a delta chi2 and must not be scaled again.
  using Surface = std::map<Node, double>;

  struct Settings {
    /// Cells of the initial uniform grid. Coarse, but fine enough that no plausible
    /// confidence region fits between two of its points unnoticed.
    int coarse_cells_x = 12;
    int coarse_cells_y = 12;

    /// Refinement depth applied to cells a contour passes through.
    int max_depth = 3;

    /// Refinement depth applied to cells inside the outermost contour but not on
    /// a contour themselves.
    ///
    /// Tempting to leave shallow, since the interior only has to look smooth
    /// rather than resolve a line - but the ridge a 1D profile follows runs
    /// through the interior, not along a contour. At depth 2 the profile
    /// minimum in each column was taken over too few points and came out too
    /// high, narrowing the 1D intervals by around 15% on a test surface with
    /// known answers. At full depth they agree to within one lattice step, for
    /// about 12% more fits.
    int region_depth = 3;

    /// Levels to resolve, as delta chi2. The largest one bounds the refined region.
    ///
    /// The first three are the 68/95/99% contours for two degrees of freedom,
    /// which are what the 2D plot draws. The fourth is not drawn: it is there so
    /// the 3-sigma point of a 1D profile, at delta chi2 = 9, falls comfortably
    /// inside the refined region instead of sitting on its edge at 9.21. Costs
    /// about 23% more fits.
    ///
    /// Adding the 1D levels 1, 4 and 9 explicitly would change nothing while
    /// region_depth equals max_depth, since everything inside the outermost
    /// level is already refined to full depth.
    std::vector<double> levels{2.28, 5.99, 9.21, 11.83};

    /// Side length of a coarse cell in lattice units.
    [[nodiscard]] int coarse_step() const { return 1 << max_depth; }

    /// Number of lattice intervals spanning the scanned range.
    [[nodiscard]] int lattice_x() const { return coarse_cells_x << max_depth; }
    [[nodiscard]] int lattice_y() const { return coarse_cells_y << max_depth; }
  };

  /**
   * @brief Decides whether a cell has to be subdivided.
   *
   * @param surface   Values known so far.
   * @param settings  Refinement settings.
   * @param cell      Cell under consideration.
   * @param reference Lowest likelihood seen, which the levels are measured from.
   * @param depth     Current refinement depth of the cell.
   */
  inline bool needs_split(const Surface& surface, const Settings& settings, const Cell& cell, double reference, int depth) {
    double lowest  = std::numeric_limits<double>::infinity();
    double highest = -std::numeric_limits<double>::infinity();

    for (const int dx : {0, cell.size})
      for (const int dy : {0, cell.size}) {
        const auto corner = surface.find(Node{cell.origin.x + dx, cell.origin.y + dy});
        // A corner that is missing or unusable leaves the cell undecidable, so it
        // is refined rather than silently trusted.
        if (corner == surface.end() || !std::isfinite(corner->second))
          return depth < settings.max_depth;

        const double delta = corner->second - reference;
        lowest             = std::min(lowest, delta);
        highest            = std::max(highest, delta);
      }

    const bool on_contour = std::ranges::any_of(settings.levels, [&](double level) { return lowest <= level && level <= highest; });
    if (on_contour)
      return depth < settings.max_depth;

    const bool inside = !settings.levels.empty() && lowest <= std::ranges::max(settings.levels);
    return inside && depth < settings.region_depth;
  }

  /// Lattice nodes of the initial uniform grid.
  inline std::vector<Node> coarse_nodes(const Settings& settings) {
    const int step = settings.coarse_step();

    std::vector<Node> nodes;
    nodes.reserve(static_cast<std::size_t>(settings.coarse_cells_x + 1) * (settings.coarse_cells_y + 1));
    for (int ix = 0; ix <= settings.coarse_cells_x; ++ix)
      for (int iy = 0; iy <= settings.coarse_cells_y; ++iy)
        nodes.push_back(Node{ix * step, iy * step});

    return nodes;
  }

  /// Cells of the initial uniform grid.
  inline std::vector<Cell> coarse_cells(const Settings& settings) {
    const int step = settings.coarse_step();

    std::vector<Cell> cells;
    cells.reserve(static_cast<std::size_t>(settings.coarse_cells_x) * settings.coarse_cells_y);
    for (int ix = 0; ix < settings.coarse_cells_x; ++ix)
      for (int iy = 0; iy < settings.coarse_cells_y; ++iy)
        cells.push_back(Cell{Node{ix * step, iy * step}, step});

    return cells;
  }

  /// The five nodes a split adds: the edge midpoints and the centre. The corners
  /// are already there, and neighbouring cells share the edge midpoints.
  inline void split_nodes(const Cell& cell, std::vector<Node>& nodes) {
    const int half = cell.size / 2;

    for (const int dx : {0, half, cell.size})
      for (const int dy : {0, half, cell.size})
        nodes.push_back(Node{cell.origin.x + dx, cell.origin.y + dy});
  }

  /// The four cells a split produces.
  inline void split_cells(const Cell& cell, std::vector<Cell>& cells) {
    const int half = cell.size / 2;

    for (const int dx : {0, half})
      for (const int dy : {0, half})
        cells.push_back(Cell{Node{cell.origin.x + dx, cell.origin.y + dy}, half});
  }

  /// Lowest finite value on the surface; infinity if there is none.
  inline double reference_value(const Surface& surface) {
    double lowest = std::numeric_limits<double>::infinity();
    for (const auto& [node, value] : surface)
      if (std::isfinite(value))
        lowest = std::min(lowest, value);

    return lowest;
  }

  /**
   * @brief Runs the coarse grid and all refinement rounds.
   *
   * @param settings  Refinement settings.
   * @param seed      Likelihood of a free fit, used as the reference until the
   *                  grid produces a lower one. Pass infinity if there is none.
   * @param evaluate  Callable `void(const std::vector<Node>&, Surface&)` that
   *                  fills in every node it is handed. It may be called with
   *                  nodes that already have a value; those are filtered out
   *                  before the call.
   * @param report    Callable `void(int round, std::size_t split, std::size_t cells, std::size_t points)`
   *                  invoked once per round, for progress output.
   */
  template <typename Evaluator, typename Reporter>
  Surface refine(const Settings& settings, double seed, Evaluator&& evaluate, Reporter&& report) {
    Surface surface;

    auto evaluate_missing = [&](std::vector<Node> nodes) {
      std::ranges::sort(nodes);
      const auto repeated = std::ranges::unique(nodes);
      nodes.erase(repeated.begin(), repeated.end());
      std::erase_if(nodes, [&](const Node& node) { return surface.contains(node); });

      evaluate(nodes, surface);
    };

    evaluate_missing(coarse_nodes(settings));
    report(0, std::size_t{0}, std::size_t{0}, surface.size());

    std::vector<Cell> cells = coarse_cells(settings);

    for (int depth = 0; depth < settings.max_depth; ++depth) {
      const double reference = std::min(seed, reference_value(surface));

      std::vector<Cell> split;
      for (const Cell& cell : cells)
        if (needs_split(surface, settings, cell, reference, depth))
          split.push_back(cell);

      report(depth + 1, split.size(), cells.size(), surface.size());

      if (split.empty())
        break;

      std::vector<Node> nodes;
      std::vector<Cell> children;
      for (const Cell& cell : split) {
        split_nodes(cell, nodes);
        split_cells(cell, children);
      }

      evaluate_missing(std::move(nodes));
      cells = std::move(children);
    }

    return surface;
  }

}  // namespace scan
