// STL includes
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

// includes
#include "DoubleChooz/DCExperimentModule.h"
#include "ExperimentModule.h"
#include "Fit.h"
#include "IceCube/ICModule.h"
#include "LinearRegression/LinRegModule.h"
#include "Options.h"
#include "write_results.h"

#include "AdaptiveGrid.h"
#include "AdaptiveScan1D.h"
#include "ScanSeeds.h"

#include <TROOT.h>



namespace {

  /// A point an earlier run already fitted.
  struct StoredFit {
    double              llh = 0.0;
    std::vector<double> parameters;  ///< Empty unless the fit converged and every parameter was found.
  };

  /**
   * @brief Reads back a point an earlier run already fitted.
   *
   * Only the JSON output is read back, so a scan written with
   * --output-format protobuf refits its points on resume, as it always did.
   *
   * @param name  Base name of the point's output file.
   * @param names Parameter names in minimizer order, used to turn the file's
   *              name-keyed parameter block back into a start vector.
   * @return The result, or nullopt if the file is absent or holds no usable likelihood.
   */
  std::optional<StoredFit> stored_fit(const std::string& name, const std::vector<std::string>& names) {
    const std::filesystem::path path = name + ".json";
    if (!std::filesystem::exists(path))
      return std::nullopt;

    std::ifstream  file(path);
    nlohmann::json stored = nlohmann::json::parse(file, nullptr, false);
    if (stored.is_discarded() || !stored.contains("LLH"))
      return std::nullopt;

    const double llh = stored["LLH"].get<double>();
    if (!std::isfinite(llh))
      return std::nullopt;

    StoredFit result;
    result.llh = llh;

    // Start values are only carried over from a fit that converged: a point that
    // stalled is exactly the one whose parameters must not spread to its
    // neighbours. A partial or renamed parameter block is dropped whole rather
    // than filled in from the defaults, which would mix two different fits.
    if (stored.value("converged", false) && stored.contains("parameters")) {
      const auto&         block = stored["parameters"];
      std::vector<double> values;
      values.reserve(names.size());
      for (const std::string& parameter : names) {
        if (!block.contains(parameter) || !block[parameter].contains("value"))
          break;
        values.push_back(block[parameter]["value"].get<double>());
      }

      if (values.size() == names.size())
        result.parameters = std::move(values);
    }

    return result;
  }

  /**
   * @brief Uses the parameters of an earlier fit as the start point of the next one.
   *
   * Only variables that are still free are touched, so a parameter the config
   * fixed keeps its configured value -- there it is a constraint, not a start
   * point, and moving it would silently change which point is being evaluated.
   * The scanned parameter has to be fixed before this is called for the same
   * reason.
   *
   * @param minimizer        Minimizer to seed, already set up by ana::Fit.
   * @param input_parameters Configured parameters, for their bounds and step widths.
   * @param values           Parameters of the earlier fit, in minimizer order.
   */
  void apply_start_values(ROOT::Math::Minimizer& minimizer, const io::InputParameter& input_parameters, const std::vector<double>& values) {
    const auto& parameters = input_parameters.parameters();
    if (values.size() != parameters.size())
      return;

    for (std::size_t i = 0; i < parameters.size(); ++i) {
      if (minimizer.IsFixedVariable(i))
        continue;

      double start = values[i];

      // Minuit2's internal transformation of a bounded parameter is singular on
      // the limit itself, and a converged fit can come arbitrarily close to one.
      // Same margin the configured start values are pulled back by in Fit.cpp.
      const std::optional<double>& lower = parameters[i].lower_bound();
      const std::optional<double>& upper = parameters[i].upper_bound();
      if (lower || upper) {
        const double margin = 1.0e-3 * parameters[i].uncertainty();
        if (lower) start = std::max(start, *lower + margin);
        if (upper) start = std::min(start, *upper - margin);
      }

      minimizer.SetVariableValue(i, start);
    }
  }

  std::string node_name(const scan::Node& node) {
    std::stringstream name;
    name << "Output_" << node.x << '_' << node.y;
    return name.str();
  }

  std::string node_name(const std::string& parameter, int node) {
    std::stringstream name;
    name << "Output_" << parameter << '_' << node;
    return name.str();
  }

  /// Distinct from node_name(const scan::Node&) above: a regular-grid node
  /// (i, j) sits at a different physical point than an adaptive-grid node of
  /// the same integer coordinates, so the two must never share a resume file.
  std::string node_name_regular(const scan::Node& node) {
    std::stringstream name;
    name << "OutputRegular_" << node.x << '_' << node.y;
    return name.str();
  }

}  // namespace

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

  // Randomized start values exist to spread the start points on purpose, which
  // seeding every fit from its neighbour would undo. The two do not combine.
  const bool warm_start = options->inputOptions().scan_warm_start() && !options->inputOptions().randomize_seeds();
  if (options->inputOptions().scan_warm_start() && !warm_start)
    std::cout << "Warm start off: --randomizeSeeds already sets the start point of every scan fit\n";

  scanseed::Store<scan::Node> seeds;

  // Working in integer lattice coordinates keeps every point of every depth
  // exactly representable and gives each one a stable name.
  const double spacing_x = (high_x - low_x) / settings.lattice_x();
  const double spacing_y = (high_y - low_y) / settings.lattice_y();

  auto position = [&](const scan::Node& node) { return std::pair{low_x + node.x * spacing_x, low_y + node.y * spacing_y}; };

  {
    nlohmann::json grid;
    grid["SpectralIndex"] = {{"low", low_x}, {"high", high_x}, {"lattice", settings.lattice_x()}};
    grid["AstroNorm"]     = {{"low", low_y}, {"high", high_y}, {"lattice", settings.lattice_y()}};
    grid["levels"]        = settings.levels;

    if (std::filesystem::exists("scan_grid.json")) {
      std::ifstream  existing_file("scan_grid.json");
      nlohmann::json existing = nlohmann::json::parse(existing_file);
      if (existing != grid)
        throw std::runtime_error("scan_grid.json in this directory describes a different lattice; the Output_i_j files here belong to that one. Scan into an empty directory instead.");
    } else {
      std::ofstream("scan_grid.json") << grid.dump(2) << '\n';
    }
  }

  // A free fit gives the reference the delta likelihood is measured against
  // before any grid point exists. It is only a starting value: any scan point
  // may undercut it, and refine() takes the lower of the two.
  double seed_llh = std::numeric_limits<double>::infinity();
  {
    ana::Fit seed_fit(options, module);
    seed_fit.minimize();

    const auto seed_min = seed_fit.get_minimizer();
    if (std::isfinite(seed_min->MinValue())) {
      seed_llh = seed_min->MinValue();

      // The free fit is the best guess available for the coarse grid, whose
      // points have no fitted neighbour to take one from yet.
      if (warm_start && seed_fit.converged())
        seeds.set_fallback(std::vector<double>(seed_min->X(), seed_min->X() + input_parameters.size()));

      std::cout << "Seed fit: SpectralIndex = " << seed_min->X()[static_cast<int>(SpectralIndex)]
                << ", AstroNorm = " << seed_min->X()[static_cast<int>(AstroNorm)]
                << (seed_fit.converged() ? "" : " (did not converge)") << '\n';
    } else {
      std::cout << "Seed fit produced no likelihood; the reference will come from the coarse grid\n";
    }
  }

  int fits_performed = 0;

  auto evaluate = [&](const std::vector<scan::Node>& nodes, scan::Surface& surface) {
    std::vector<scan::Node> ordered = nodes;

    // Within a round the points nearest the current minimum come first, so an
    // interrupted run still leaves a filled-in neighbourhood of the best fit.
    if (!surface.empty()) {
      const scan::Node best = std::ranges::min_element(surface, {}, [](const auto& entry) { return entry.second; })->first;
      std::ranges::sort(ordered, {}, [&](const scan::Node& node) {
        return std::hypot(static_cast<double>(node.x - best.x) * spacing_x, static_cast<double>(node.y - best.y) * spacing_y);
      });
    }

    // Each worker builds its own Fit -- and therefore its own likelihood -- over
    // the shared module, whose only state is the immutable MC sample. Points are
    // handed out one at a time, so a slow fit does not stall the others.
    const int        n_nodes   = static_cast<int>(ordered.size());
    const int        n_workers = std::clamp(scan_workers, 1, std::max(n_nodes, 1));
    std::atomic<int> next_index{0};
    std::mutex       surface_mutex;

    auto worker = [&]() {
      for (int pos = next_index.fetch_add(1); pos < n_nodes; pos = next_index.fetch_add(1)) {
        const scan::Node  node = ordered[pos];
        const std::string name = node_name(node);

        if (const auto known = stored_fit(name, names)) {
          // A resumed run puts the points it reads back into the store, so the
          // fits it still has to do start from a neighbour just as they would
          // have in the run that was interrupted.
          if (warm_start && !known->parameters.empty())
            seeds.store(node, known->parameters);

          const std::scoped_lock lock(surface_mutex);
          surface[node] = known->llh;
          continue;
        }

        const auto [x, y] = position(node);

        ana::Fit fit(options, module);
        auto     min = fit.get_minimizer();
        min->SetVariableValue(SpectralIndex, x);
        min->SetVariableValue(AstroNorm, y);
        min->FixVariable(AstroNorm);
        min->FixVariable(SpectralIndex);

        // The scanned pair is fixed above, so what is carried over is the
        // nuisance parameters, and those barely move between adjacent points.
        // The ordering above hands the points out from the current minimum
        // outwards, so the nearest known point is usually an immediate
        // neighbour rather than something across the window.
        if (warm_start) {
          const auto start = seeds.nearest(node, [&](const scan::Node& a, const scan::Node& b) {
            return std::hypot(static_cast<double>(a.x - b.x) * spacing_x, static_cast<double>(a.y - b.y) * spacing_y);
          });
          if (!start.empty())
            apply_start_values(*min, input_parameters, start);
        }

        fit.minimize();

        result::write_results(fit, name);

        if (warm_start && fit.converged() && std::isfinite(min->MinValue()))
          seeds.store(node, std::vector<double>(min->X(), min->X() + input_parameters.size()));

        // Minuit reporting "Edm is above max" is common at the edges of the
        // window and still leaves a usable likelihood, so the value is taken
        // whenever it is finite rather than only when the fit converged.
        const std::scoped_lock lock(surface_mutex);
        surface[node] = min->MinValue();
        ++fits_performed;
      }
    };

    std::vector<std::thread> workers;
    workers.reserve(n_workers);
    for (int w = 0; w < n_workers; ++w)
      workers.emplace_back(worker);

    for (auto& t : workers)
      t.join();
  };

  auto report = [scan_workers](int round, std::size_t split, std::size_t cells, std::size_t points) {
    if (round == 0)
      std::cout << "Coarse grid: " << points << " points"
                << (scan_workers > 1 ? " (" + std::to_string(scan_workers) + " workers)" : "") << '\n';
    else
      std::cout << "Refinement " << round << ": splitting " << split << " of " << cells << " cells (" << points << " points so far)\n";
  };

  const scan::Surface surface = scan::refine(settings, seed_llh, evaluate, report);

  std::cout << "Scan finished: " << surface.size() << " points, " << fits_performed << " fitted in this run\n";
}

/**
 * @brief Maps AstroNorm against SpectralIndex on a plain regular grid.
 *
 * Unlike perform_2d_scan, every point of the grid is fitted once instead of
 * only where a contour needs resolving. The window and the default 30 x 30
 * point count match the reference scan in
 * shuyang/NT_roundtrip_fullE_numu (gamma_astro = SpectralIndex from 1.7 to
 * 2.8, astro_norm = AstroNorm from 0 to 3, both on 30 evenly spaced points),
 * so results line up point for point with those pickles.
 *
 * Points are named "OutputRegular_i_j" -- distinct from perform_2d_scan's
 * "Output_i_j" -- so the two scans can share a directory without one's resume
 * logic picking up the other's fit at the same integer coordinates but a
 * different physical point.
 *
 * Not wired into the command line: call it in place of perform_2d_scan.
 *
 * @param points_x Number of grid points along SpectralIndex.
 * @param points_y Number of grid points along AstroNorm.
 */
void perform_2d_scan_regular(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, int points_x = 30,
                              int points_y = 30) {
  constexpr double low_x  = 1.7;
  constexpr double high_x = 2.8;
  constexpr double low_y  = 0.0;
  constexpr double high_y = 3.0;

  using namespace ana::ic;
  using enum params::ic::General;

  if (points_x < 2 || points_y < 2)
    throw std::runtime_error("perform_2d_scan_regular needs at least 2 points along each axis");

  const int scan_workers = std::max(1, options->inputOptions().scan_workers());

  const auto& input_parameters = options->inputOptions().input_parameters();
  const auto& names            = input_parameters.names();

  // Randomized start values exist to spread the start points on purpose, which
  // seeding every fit from its neighbour would undo. The two do not combine.
  const bool warm_start = options->inputOptions().scan_warm_start() && !options->inputOptions().randomize_seeds();
  if (options->inputOptions().scan_warm_start() && !warm_start)
    std::cout << "Warm start off: --randomizeSeeds already sets the start point of every scan fit\n";

  scanseed::Store<scan::Node> seeds;

  const double spacing_x = (high_x - low_x) / (points_x - 1);
  const double spacing_y = (high_y - low_y) / (points_y - 1);

  auto position = [&](const scan::Node& node) { return std::pair{low_x + node.x * spacing_x, low_y + node.y * spacing_y}; };

  {
    nlohmann::json grid;
    grid["SpectralIndex"] = {{"low", low_x}, {"high", high_x}, {"points", points_x}};
    grid["AstroNorm"]     = {{"low", low_y}, {"high", high_y}, {"points", points_y}};

    const std::string grid_path = "scan_grid_regular.json";
    if (std::filesystem::exists(grid_path)) {
      std::ifstream  existing_file(grid_path);
      nlohmann::json existing = nlohmann::json::parse(existing_file);
      if (existing != grid)
        throw std::runtime_error(grid_path +
                                 " in this directory describes a different grid; the OutputRegular_i_j files here belong to that "
                                 "one. Scan into an empty directory instead.");
    } else {
      std::ofstream(grid_path) << grid.dump(2) << '\n';
    }
  }

  // Only needed to seed the very first fits, before any grid point has a
  // value of its own to seed from; there is no adaptive reference level here,
  // so unlike perform_2d_scan the free fit's likelihood itself goes unused.
  if (warm_start) {
    ana::Fit seed_fit(options, module);
    seed_fit.minimize();

    const auto seed_min = seed_fit.get_minimizer();
    if (seed_fit.converged() && std::isfinite(seed_min->MinValue()))
      seeds.set_fallback(std::vector<double>(seed_min->X(), seed_min->X() + input_parameters.size()));

    std::cout << "Seed fit: SpectralIndex = " << seed_min->X()[static_cast<int>(SpectralIndex)]
               << ", AstroNorm = " << seed_min->X()[static_cast<int>(AstroNorm)] << (seed_fit.converged() ? "" : " (did not converge)")
               << '\n';
  }

  std::vector<scan::Node> nodes;
  nodes.reserve(static_cast<std::size_t>(points_x) * points_y);
  for (int ix = 0; ix < points_x; ++ix)
    for (int iy = 0; iy < points_y; ++iy)
      nodes.push_back(scan::Node{ix, iy});

  // Each worker builds its own Fit -- and therefore its own likelihood -- over
  // the shared module, whose only state is the immutable MC sample. Points are
  // handed out one at a time, so a slow fit does not stall the others.
  const int        n_nodes   = static_cast<int>(nodes.size());
  const int        n_workers = std::clamp(scan_workers, 1, std::max(n_nodes, 1));
  std::atomic<int> next_index{0};
  std::mutex       surface_mutex;

  scan::Surface surface;
  int           fits_performed = 0;

  auto worker = [&]() {
    for (int pos = next_index.fetch_add(1); pos < n_nodes; pos = next_index.fetch_add(1)) {
      const scan::Node  node = nodes[pos];
      const std::string name = node_name_regular(node);

      if (const auto known = stored_fit(name, names)) {
        // A resumed run puts the points it reads back into the store, so the
        // fits it still has to do start from a neighbour just as they would
        // have in the run that was interrupted.
        if (warm_start && !known->parameters.empty())
          seeds.store(node, known->parameters);

        const std::scoped_lock lock(surface_mutex);
        surface[node] = known->llh;
        continue;
      }

      const auto [x, y] = position(node);

      ana::Fit fit(options, module);
      auto     min = fit.get_minimizer();
      min->SetVariableValue(SpectralIndex, x);
      min->SetVariableValue(AstroNorm, y);
      min->FixVariable(AstroNorm);
      min->FixVariable(SpectralIndex);

      // The scanned pair is fixed above, so what is carried over is the
      // nuisance parameters, and those barely move between adjacent grid points.
      if (warm_start) {
        const auto start = seeds.nearest(node, [&](const scan::Node& a, const scan::Node& b) {
          return std::hypot(static_cast<double>(a.x - b.x) * spacing_x, static_cast<double>(a.y - b.y) * spacing_y);
        });
        if (!start.empty())
          apply_start_values(*min, input_parameters, start);
      }

      fit.minimize();

      result::write_results(fit, name);

      if (warm_start && fit.converged() && std::isfinite(min->MinValue()))
        seeds.store(node, std::vector<double>(min->X(), min->X() + input_parameters.size()));

      // Minuit reporting "Edm is above max" is common at the edges of the
      // window and still leaves a usable likelihood, so the value is taken
      // whenever it is finite rather than only when the fit converged.
      const std::scoped_lock lock(surface_mutex);
      surface[node] = min->MinValue();
      ++fits_performed;
    }
  };

  std::cout << "Regular grid: " << n_nodes << " points (" << points_x << " x " << points_y << ")"
            << (scan_workers > 1 ? " (" + std::to_string(scan_workers) + " workers)" : "") << '\n';

  std::vector<std::thread> workers;
  workers.reserve(n_workers);
  for (int w = 0; w < n_workers; ++w)
    workers.emplace_back(worker);

  for (auto& t : workers)
    t.join();

  std::cout << "Regular grid scan finished: " << surface.size() << " points, " << fits_performed << " fitted in this run\n";
}

/**
 * @brief Profiles the likelihood along a single configured parameter.
 *
 * The one-dimensional counterpart of perform_2d_scan: the scanned parameter is
 * fixed point by point and every other parameter is minimised over, so what
 * comes out is a profile likelihood rather than a slice. The window is taken
 * from the parameter's own "LowerBound"/"UpperBound" in the config, which is
 * the range the fit is allowed to explore anyway - scanning outside it would
 * profile over a region the minimiser can never reach.
 *
 * The scan starts uniform and then halves only the intervals a confidence
 * level crosses, so the fits end up around the minimum and the 1/2/3 sigma
 * crossings instead of spread over the flat wings. See AdaptiveScan1D.h.
 *
 * Points are named by their integer lattice index, so a resumed run picks up
 * every fit an earlier one completed.
 *
 * Not wired into the command line: call it in place of perform_2d_scan.
 *
 * @param parameter_name Name of the parameter to profile, as it appears in the
 *                       config's parameter list.
 */
void perform_1d_scan(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, const std::string& parameter_name) {
  const auto& input_parameters = options->inputOptions().input_parameters();
  const auto& names            = input_parameters.names();

  const auto found = std::ranges::find(names, parameter_name);
  if (found == names.end())
    throw std::runtime_error("No parameter named '" + parameter_name + "' in the config");

  const int   index     = static_cast<int>(std::ranges::distance(names.begin(), found));
  const auto& parameter = input_parameters.parameters()[index];

  // The window has to come from somewhere, and a parameter with no bounds gives
  // no honest one: start value +- a few step widths would be an invention, and
  // one that quietly decides the answer for a parameter whose profile is wide.
  const auto& lower = parameter.lower_bound();
  const auto& upper = parameter.upper_bound();
  if (!lower || !upper)
    throw std::runtime_error("Parameter '" + parameter_name +
                             "' has no LowerBound/UpperBound in the config; the 1D scan takes its window from those");

  const double low  = *lower;
  const double high = *upper;

  const scan1d::Settings settings;

  const int scan_workers = std::max(1, options->inputOptions().scan_workers());

  // Randomized start values exist to spread the start points on purpose, which
  // seeding every fit from its neighbour would undo. The two do not combine.
  const bool warm_start = options->inputOptions().scan_warm_start() && !options->inputOptions().randomize_seeds();
  if (options->inputOptions().scan_warm_start() && !warm_start)
    std::cout << "Warm start off: --randomizeSeeds already sets the start point of every scan fit\n";

  scanseed::Store<int> seeds;

  // Working in integer lattice coordinates keeps every point of every depth
  // exactly representable and gives each one a stable name.
  const double spacing = (high - low) / settings.lattice();

  auto position = [&](int node) { return low + node * spacing; };

  {
    nlohmann::json grid;
    grid[parameter_name] = {{"low", low}, {"high", high}, {"lattice", settings.lattice()}};
    grid["levels"]       = settings.levels;

    const std::string grid_path = "scan_grid_" + parameter_name + ".json";
    if (std::filesystem::exists(grid_path)) {
      std::ifstream  existing_file(grid_path);
      nlohmann::json existing = nlohmann::json::parse(existing_file);
      if (existing != grid)
        throw std::runtime_error(grid_path + " in this directory describes a different lattice; the Output_" + parameter_name +
                                 "_i files here belong to that one. Scan into an empty directory instead.");
    } else {
      std::ofstream(grid_path) << grid.dump(2) << '\n';
    }
  }

  // A free fit gives the reference the delta likelihood is measured against
  // before any scan point exists. It is only a starting value: any scan point
  // may undercut it, and refine() takes the lower of the two.
  double seed_llh = std::numeric_limits<double>::infinity();
  {
    ana::Fit seed_fit(options, module);
    seed_fit.minimize();

    const auto seed_min = seed_fit.get_minimizer();
    if (std::isfinite(seed_min->MinValue())) {
      seed_llh = seed_min->MinValue();

      // The free fit is the best guess available for the coarse scan, whose
      // points have no fitted neighbour to take one from yet.
      if (warm_start && seed_fit.converged())
        seeds.set_fallback(std::vector<double>(seed_min->X(), seed_min->X() + input_parameters.size()));

      std::cout << "Seed fit: " << parameter_name << " = " << seed_min->X()[index]
                << (seed_fit.converged() ? "" : " (did not converge)") << '\n';
    } else {
      std::cout << "Seed fit produced no likelihood; the reference will come from the coarse scan\n";
    }
  }

  int fits_performed = 0;

  auto evaluate = [&](const std::vector<int>& nodes, scan1d::Profile& profile) {
    std::vector<int> ordered = nodes;

    // Within a round the points nearest the current minimum come first, so an
    // interrupted run still leaves a filled-in neighbourhood of the best fit.
    if (!profile.empty()) {
      const int best = std::ranges::min_element(profile, {}, [](const auto& entry) { return entry.second; })->first;
      std::ranges::sort(ordered, {}, [&](int node) { return std::abs(node - best); });
    }

    // Each worker builds its own Fit -- and therefore its own likelihood -- over
    // the shared module, whose only state is the immutable MC sample. Points are
    // handed out one at a time, so a slow fit does not stall the others.
    const int        n_nodes   = static_cast<int>(ordered.size());
    const int        n_workers = std::clamp(scan_workers, 1, std::max(n_nodes, 1));
    std::atomic<int> next_index{0};
    std::mutex       profile_mutex;

    auto worker = [&]() {
      for (int pos = next_index.fetch_add(1); pos < n_nodes; pos = next_index.fetch_add(1)) {
        const int         node = ordered[pos];
        const std::string name = node_name(parameter_name, node);

        if (const auto known = stored_fit(name, names)) {
          // A resumed run puts the points it reads back into the store, so the
          // fits it still has to do start from a neighbour just as they would
          // have in the run that was interrupted.
          if (warm_start && !known->parameters.empty())
            seeds.store(node, known->parameters);

          const std::scoped_lock lock(profile_mutex);
          profile[node] = known->llh;
          continue;
        }

        ana::Fit fit(options, module);
        auto     min = fit.get_minimizer();
        min->SetVariableValue(index, position(node));
        min->FixVariable(index);

        // The scanned parameter is fixed above, so what is carried over is the
        // profiled parameters, and those barely move between adjacent points.
        // The ordering above hands the points out from the current minimum
        // outwards, so the nearest known point is usually an immediate
        // neighbour rather than something across the window.
        if (warm_start) {
          const auto start = seeds.nearest(node, [](int a, int b) { return std::abs(static_cast<double>(a - b)); });
          if (!start.empty())
            apply_start_values(*min, input_parameters, start);
        }

        fit.minimize();

        result::write_results(fit, name);

        if (warm_start && fit.converged() && std::isfinite(min->MinValue()))
          seeds.store(node, std::vector<double>(min->X(), min->X() + input_parameters.size()));

        // Minuit reporting "Edm is above max" is common at the edges of the
        // window and still leaves a usable likelihood, so the value is taken
        // whenever it is finite rather than only when the fit converged.
        const std::scoped_lock lock(profile_mutex);
        profile[node] = min->MinValue();
        ++fits_performed;
      }
    };

    std::vector<std::thread> workers;
    workers.reserve(n_workers);
    for (int w = 0; w < n_workers; ++w)
      workers.emplace_back(worker);

    for (auto& t : workers)
      t.join();
  };

  auto report = [scan_workers](int round, std::size_t split, std::size_t segments, std::size_t points) {
    if (round == 0)
      std::cout << "Coarse scan: " << points << " points"
                << (scan_workers > 1 ? " (" + std::to_string(scan_workers) + " workers)" : "") << '\n';
    else
      std::cout << "Refinement " << round << ": splitting " << split << " of " << segments << " intervals (" << points << " points so far)\n";
  };

  const scan1d::Profile profile = scan1d::refine(settings, seed_llh, evaluate, report);

  std::cout << "Scan of " << parameter_name << " finished: " << profile.size() << " points, " << fits_performed << " fitted in this run\n";
}

int main(int argc, char** argv) {
  ROOT::EnableThreadSafety();

  try {
    // Register all available experiments. Only the one selected via the "Experiment" config key
    // is initialized and used for the fit.
    ana::module_map_t modules;
    {
      auto dc_module             = std::make_shared<ana::dc::DCExperimentModule>();
      modules[dc_module->name()] = dc_module;
    }
    {
      auto linreg_module             = std::make_shared<ana::linreg::LinRegModule>();
      modules[linreg_module->name()] = linreg_module;
    }
    {
      auto ic_module             = std::make_shared<ana::ic::ICExperimentModule>();
      modules[ic_module->name()] = ic_module;
    }

    const auto options = std::make_shared<io::Options>(argc, argv, ana::collect_input_options(modules));

    const auto module = modules.at(options->inputOptions().experiment());

    // --fitOnly runs the plain fit and writes "Output.json"; without it the 2D
    // scan is the entry point. The single fit is what the NNMFit parity harness
    // drives (tools/nnmfit_oracle/compare_llh_value.py, run_fit_parity.sh),
    // which needs one converged result rather than a surface.
    if (options->inputOptions().fit_only()) {
      ana::Fit fit(options, module);
      fit.minimize();
      result::write_results(fit, "Output");
    } else {
      // perform_1d_scan(options, module, "AstroNorm");
      perform_2d_scan(options, module);
      // perform_2d_scan_regular(options, module);
    }
  } catch (const std::exception& e) {
    std::cout << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
