#include "Scans.h"

#include "ScanCommon.h"

// STL includes
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// includes
#include "AdaptiveGrid.h"
#include "IceCube/ICModule.h"

namespace llhfit {


  /**
   * @brief Maps the confidence region of AstroNorm against SpectralIndex.
   *
   * The scan starts from a coarse uniform grid and then repeatedly subdivides only
   * the cells a confidence contour crosses, so the fits end up where the lines are
   * and the featureless parts of the window stay cheap. See AdaptiveGrid.h.
   *
   * Points are named by their integer lattice coordinates, so a resumed run picks
   * up every fit an earlier one completed.
   */
  void perform_2d_scan(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module) {
    constexpr double low_x  = 1.7;
    constexpr double high_x = 2.8;
    constexpr double low_y  = 0.0;
    constexpr double high_y = 3.0;

    using namespace ana::ic;
    using enum params::ic::General;

    const scan::Settings settings;

    const int scan_workers = std::max(1, options->inputOptions().scan_workers());

    const auto& input_parameters = options->inputOptions().input_parameters();
    const auto& names            = input_parameters.names();
    const auto& format           = options->inputOptions().output_format();

    const bool warm_start = resolve_warm_start(options);

    scanseed::Store<scan::Node> seeds;
    seeds.set_fallback(seed_from_values(options));

    // Working in integer lattice coordinates keeps every point of every depth
    // exactly representable and gives each one a stable name.
    const double spacing_x = (high_x - low_x) / settings.lattice_x();
    const double spacing_y = (high_y - low_y) / settings.lattice_y();

    auto position = [&](const scan::Node& node) { return std::pair{low_x + node.x * spacing_x, low_y + node.y * spacing_y}; };
    auto distance = [&](const scan::Node& a, const scan::Node& b) {
      return std::hypot(static_cast<double>(a.x - b.x) * spacing_x, static_cast<double>(a.y - b.y) * spacing_y);
    };
    auto apply_fixed = [&](ROOT::Math::Minimizer& min, const scan::Node& node) {
      const auto [x, y] = position(node);
      min.SetVariableValue(SpectralIndex, x);
      min.SetVariableValue(AstroNorm, y);
      min.FixVariable(AstroNorm);
      min.FixVariable(SpectralIndex);
    };

    {
      nlohmann::json grid;
      grid["SpectralIndex"] = {{"low", low_x}, {"high", high_x}, {"lattice", settings.lattice_x()}};
      grid["AstroNorm"]     = {{"low", low_y}, {"high", high_y}, {"lattice", settings.lattice_y()}};
      grid["levels"]        = settings.levels;
      write_or_check_grid("scan_grid.json", grid, "lattice", "Output_i_j");
    }

    // The free fit gives the reference the delta likelihood is measured against
    // before any grid point exists. It is only a starting value: any scan point
    // may undercut it, and refine() takes the lower of the two. Saved to disk as
    // BestFit so analysis can reference the true unfixed minimum instead of the
    // lowest sampled grid point.
    const auto seed = bootstrap_seed_fit(options, module, "BestFit", input_parameters, format, warm_start, seeds, no_extra_fixed);
    if (!seed.from_store && !seed.parameters.empty())
      std::cout << "Seed fit: SpectralIndex = " << seed.parameters[static_cast<int>(SpectralIndex)]
                << ", AstroNorm = " << seed.parameters[static_cast<int>(AstroNorm)] << (seed.converged ? "" : " (did not converge)") << '\n';

    int fits_performed = 0;

    auto evaluate = [&](const std::vector<scan::Node>& nodes, scan::Surface& surface) {
      std::vector<scan::Node> ordered = nodes;

      // Within a round the points nearest the current minimum come first, so an
      // interrupted run still leaves a filled-in neighbourhood of the best fit.
      if (!surface.empty()) {
        const scan::Node best = std::ranges::min_element(surface, {}, [](const auto& entry) { return entry.second; })->first;
        std::ranges::sort(ordered, {}, [&](const scan::Node& node) { return distance(node, best); });
      }

      run_scan_batch(std::move(ordered), surface, options, module, input_parameters, format, warm_start, seeds, scan_workers, [](const scan::Node& node) { return node_name(node); }, apply_fixed, distance, fits_performed, seed.errors);
    };

    auto report = [scan_workers](int round, std::size_t split, std::size_t cells, std::size_t points) {
      if (round == 0)
        std::cout << "Coarse grid: " << points << " points"
                  << (scan_workers > 1 ? " (" + std::to_string(scan_workers) + " workers)" : "") << '\n';
      else
        std::cout << "Refinement " << round << ": splitting " << split << " of " << cells << " cells (" << points << " points so far)\n";
    };

    const scan::Surface surface = scan::refine(settings, seed.llh, evaluate, report);

    std::cout << "Scan finished: " << surface.size() << " points, " << fits_performed << " fitted in this run\n";
  }

  /**
   * @brief Maps AstroNorm against AstroGamma2 on a plain regular grid.
   *
   * Unlike perform_2d_scan, every point of the grid is fitted once instead of
   * only where a contour needs resolving. The window matches the reference scan
   * in galactic/ScanResults.hdf, index "astro_norm-gamma_2" (gamma_2 =
   * AstroGamma2 from 2.4 to 3.0, astro_norm = AstroNorm from 1.4 to 2.6, both on
   * 20 evenly spaced points), so with --scanPoints 20 results line up point for
   * point with that grid. Note the reference file carries 25 extra off-grid
   * points from a coarser 5 x 5 scan on top of the 20 x 20 grid; those are not
   * reproduced here.
   *
   * Points are named "OutputRegular_i_j" -- distinct from perform_2d_scan's
   * "Output_i_j" -- so the two scans can share a directory without one's resume
   * logic picking up the other's fit at the same integer coordinates but a
   * different physical point.
   *
   * Selected via --scanMode 2d-regular.
   *
   * @param points_x Number of grid points along AstroGamma2.
   * @param points_y Number of grid points along AstroNorm.
   */
  void perform_2d_scan_regular(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, int points_x, int points_y) {
    constexpr double low_x  = 2.4;
    constexpr double high_x = 3.0;
    constexpr double low_y  = 1.4;
    constexpr double high_y = 2.6;

    using namespace ana::ic;
    using enum params::ic::General;

    if (points_x < 2 || points_y < 2)
      throw std::runtime_error("perform_2d_scan_regular needs at least 2 points along each axis");

    const int scan_workers = std::max(1, options->inputOptions().scan_workers());

    const auto& input_parameters = options->inputOptions().input_parameters();
    const auto& names            = input_parameters.names();
    const auto& format           = options->inputOptions().output_format();

    const bool warm_start = resolve_warm_start(options);

    scanseed::Store<scan::Node> seeds;
    seeds.set_fallback(seed_from_values(options));

    const double spacing_x = (high_x - low_x) / (points_x - 1);
    const double spacing_y = (high_y - low_y) / (points_y - 1);

    auto position = [&](const scan::Node& node) { return std::pair{low_x + node.x * spacing_x, low_y + node.y * spacing_y}; };
    auto distance = [&](const scan::Node& a, const scan::Node& b) {
      return std::hypot(static_cast<double>(a.x - b.x) * spacing_x, static_cast<double>(a.y - b.y) * spacing_y);
    };
    auto apply_fixed = [&](ROOT::Math::Minimizer& min, const scan::Node& node) {
      const auto [x, y] = position(node);
      min.SetVariableValue(AstroGamma2, x);
      min.SetVariableValue(AstroNorm, y);
      min.FixVariable(AstroNorm);
      min.FixVariable(AstroGamma2);
    };

    {
      nlohmann::json grid;
      grid["AstroGamma2"] = {{"low", low_x}, {"high", high_x}, {"points", points_x}};
      grid["AstroNorm"]   = {{"low", low_y}, {"high", high_y}, {"points", points_y}};
      write_or_check_grid("scan_grid_regular.json", grid, "grid", "OutputRegular_i_j");
    }

    // Also the source of the true (unfixed) best fit, saved to disk as
    // BestFitRegular so analysis can reference it instead of the lowest sampled
    // grid point. There is no adaptive reference level here, so unlike
    // perform_2d_scan the free fit's likelihood itself is otherwise unused --
    // it only seeds the very first fits, before any grid point has a value of
    // its own to seed from.
    const auto seed = bootstrap_seed_fit(options, module, "BestFitRegular", input_parameters, format, warm_start, seeds, no_extra_fixed);
    if (!seed.from_store && !seed.parameters.empty())
      std::cout << "Seed fit: AstroGamma2 = " << seed.parameters[static_cast<int>(AstroGamma2)]
                << ", AstroNorm = " << seed.parameters[static_cast<int>(AstroNorm)] << (seed.converged ? "" : " (did not converge)") << '\n';

    std::vector<scan::Node> nodes;
    nodes.reserve(static_cast<std::size_t>(points_x) * points_y);
    for (int ix = 0; ix < points_x; ++ix)
      for (int iy = 0; iy < points_y; ++iy)
        nodes.push_back(scan::Node{ix, iy});

    scan::Surface surface;
    int           fits_performed = 0;

    std::cout << "Regular grid: " << nodes.size() << " points (" << points_x << " x " << points_y << ")"
              << (scan_workers > 1 ? " (" + std::to_string(scan_workers) + " workers)" : "") << '\n';

    run_scan_batch(std::move(nodes), surface, options, module, input_parameters, format, warm_start, seeds, scan_workers, [](const scan::Node& node) { return node_name_regular(node); }, apply_fixed, distance, fits_performed, seed.errors);

    std::cout << "Regular grid scan finished: " << surface.size() << " points, " << fits_performed << " fitted in this run\n";
  }

}  // namespace llhfit
