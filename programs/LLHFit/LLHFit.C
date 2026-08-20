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
#include "IceCube/ICWriteResultsProto.h"
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
   * Reads the format the run is configured to write, so a resume never looks
   * for a file the current run would not itself produce.
   *
   * @param name   Base name of the point's output file.
   * @param names  Parameter names in minimizer order, used to turn the file's
   *               name-keyed parameter block back into a start vector.
   * @param format --output-format, either "json" or "protobuf".
   * @return The result, or nullopt if the file is absent or holds no usable likelihood.
   */
  std::optional<StoredFit> stored_fit(const std::string& name, const std::vector<std::string>& names, const std::string& format) {
    if (format == "protobuf") {
      const auto stored = result::ic::read_ice_cube_results_protobuf(name);
      if (!stored)
        return std::nullopt;

      StoredFit result;
      result.llh = stored->llh;

      // Same rule as the JSON branch below: only a converged fit's parameters
      // are trusted as a neighbour's start point.
      if (stored->converged) {
        std::vector<double> values;
        values.reserve(names.size());
        for (const std::string& parameter : names) {
          const auto it = stored->parameter_values.find(parameter);
          if (it == stored->parameter_values.end())
            break;
          values.push_back(it->second);
        }

        if (values.size() == names.size())
          result.parameters = std::move(values);
      }

      return result;
    }

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
      if (minimizer.IsFixedVariable(static_cast<unsigned int>(i)))
        continue;

      double start = values[i];

      // Minuit2's internal transformation of a bounded parameter is singular on
      // the limit itself, and a converged fit can come arbitrarily close to one.
      // Same margin the configured start values are pulled back by in Fit.cpp.
      const std::optional<double>& lower = parameters[i].lower_bound();
      const std::optional<double>& upper = parameters[i].upper_bound();
      if (lower || upper) {
        const double margin = 1.0e-3 * parameters[i].uncertainty();
        if (lower)
          start = std::max(start, *lower + margin);
        if (upper)
          start = std::min(start, *upper - margin);
      }

      minimizer.SetVariableValue(static_cast<unsigned int>(i), start);
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

  /// 1D counterpart of node_name_regular(const scan::Node&) above -- distinct
  /// from node_name(parameter, node) so a regular-grid point never shares a
  /// resume file with an adaptive-grid point at the same integer index.
  std::string node_name_regular(const std::string& parameter, int node) {
    std::stringstream name;
    name << "OutputRegular_" << parameter << '_' << node;
    return name.str();
  }

  /// Randomized start values exist to spread the start points on purpose, which
  /// seeding every fit from its neighbour would undo. The two do not combine.
  bool resolve_warm_start(const std::shared_ptr<io::Options>& options) {
    const auto& input      = options->inputOptions();
    const bool  warm_start = input.scan_warm_start() && !input.randomize_seeds();
    if (input.scan_warm_start() && !warm_start)
      std::cout << "Warm start off: --randomizeSeeds already sets the start point of every scan fit\n";
    return warm_start;
  }

  /// Writes `path` the first time a scan runs in this directory; on every later
  /// run checks the stored descriptor still matches, so Output_* files from a
  /// differently configured scan are never silently reused.
  void write_or_check_grid(const std::string& path, const nlohmann::json& grid, const std::string& kind, const std::string& file_pattern) {
    if (std::filesystem::exists(path)) {
      std::ifstream  existing_file(path);
      nlohmann::json existing = nlohmann::json::parse(existing_file);
      if (existing != grid)
        throw std::runtime_error(path + " in this directory describes a different " + kind + "; the " + file_pattern +
                                  " files here belong to that one. Scan into an empty directory instead.");
    } else {
      std::ofstream(path) << grid.dump(2) << '\n';
    }
  }

  /// Result of the free (unfixed) fit that bootstraps a scan: its likelihood as
  /// the reference the adaptive grids measure delta chi2 from, and its
  /// converged parameters as the fallback warm-start point for cells that have
  /// no fitted neighbour yet.
  struct SeedFit {
    double              llh        = std::numeric_limits<double>::infinity();
    std::vector<double> parameters;  ///< Fitted X; empty if not run, non-finite, or read back from disk.
    bool                converged  = false;
    bool                from_store = false;  ///< True if read back from best_fit_name instead of freshly fitted.
  };

  /**
   * @brief Loads or runs the free fit a scan bootstraps from.
   *
   * Read back from best_fit_name if a previous run already wrote it, so a
   * resumed scan does not redo its own free fit. Otherwise runs one, saves it,
   * and -- for a converged, finite result -- seeds the warm-start fallback used
   * before any scan point has a fitted neighbour of its own.
   */
  template <typename Key>
  SeedFit bootstrap_seed_fit(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module,
                              const std::string& best_fit_name, const std::vector<std::string>& names, const std::string& format,
                              bool warm_start, scanseed::Store<Key>& seeds, std::size_t n_parameters) {
    if (const auto known = stored_fit(best_fit_name, names, format)) {
      std::cout << "Best fit already stored: " << known->llh << '\n';
      return {known->llh, {}, false, true};
    }

    ana::Fit seed_fit(options, module);
    seed_fit.minimize();

    const auto seed_min = seed_fit.get_minimizer();

    SeedFit result;
    result.converged = seed_fit.converged();

    if (std::isfinite(seed_min->MinValue())) {
      result.llh = seed_min->MinValue();
      result.parameters.assign(seed_min->X(), seed_min->X() + n_parameters);

      result::write_results(seed_fit, best_fit_name);

      // The free fit is the best guess available before any scan point has a
      // fitted neighbour of its own to take one from yet.
      if (warm_start && result.converged)
        seeds.set_fallback(result.parameters);
    } else {
      std::cout << "Seed fit produced no likelihood; the reference will come from the coarse scan\n";
    }

    return result;
  }

  /**
   * @brief Runs one batch of scan points across a worker pool, filling in `surface`.
   *
   * Shared by all four scan flavours (2D/1D, adaptive/regular): the only things
   * that differ between them are how a node is named, how it fixes the
   * minimizer's scanned variable(s), and the metric used to find a warm-start
   * neighbour -- all three come in as callables.
   *
   * Each worker builds its own Fit -- and therefore its own likelihood -- over
   * the shared module, whose only state is the immutable MC sample. Points are
   * handed out one at a time, so a slow fit does not stall the others. Worker
   * threads are numbered 0..n_workers-1 and keep that number for the pool's
   * whole lifetime, which is what lets a GPU-backed module pin worker N to a
   * fixed device via --gpuDevices (see InputOptions::gpu_device_for_worker()).
   *
   * @param nodes          Points to fill in, in the order they should be handed out.
   * @param surface        Filled in as points complete.
   * @param name_fn        Key -> output file base name.
   * @param apply_fixed    Sets and fixes the minimizer's scanned variable(s) for a node.
   * @param distance       Metric between two keys, used for the warm-start lookup.
   * @param fits_performed Accumulates the count of fits actually run (as opposed
   *                       to read back from a resumed point).
   */
  template <typename Key, typename NameFn, typename ApplyFixed, typename Distance>
  void run_scan_batch(std::vector<Key> nodes, std::map<Key, double>& surface, std::shared_ptr<io::Options> options,
                       std::shared_ptr<ana::ExperimentModule> module, const io::InputParameter& input_parameters,
                       const std::vector<std::string>& names, const std::string& format, bool warm_start, scanseed::Store<Key>& seeds,
                       int scan_workers, NameFn&& name_fn, ApplyFixed&& apply_fixed, Distance&& distance, int& fits_performed) {
    const int        n_nodes   = static_cast<int>(nodes.size());
    const int        n_workers = std::clamp(scan_workers, 1, std::max(n_nodes, 1));
    std::atomic<int> next_index{0};
    std::mutex       surface_mutex;

    auto worker = [&](int worker_index) {
      for (int pos = next_index.fetch_add(1); pos < n_nodes; pos = next_index.fetch_add(1)) {
        const Key          node = nodes[pos];
        const std::string  name = name_fn(node);

        if (const auto known = stored_fit(name, names, format)) {
          // A resumed run puts the points it reads back into the store, so the
          // fits it still has to do start from a neighbour just as they would
          // have in the run that was interrupted.
          if (warm_start && !known->parameters.empty())
            seeds.store(node, known->parameters);

          const std::scoped_lock lock(surface_mutex);
          surface[node] = known->llh;
          continue;
        }

        ana::Fit fit(options, module, worker_index);
        auto     min = fit.get_minimizer();
        apply_fixed(*min, node);

        // The scanned variable(s) are fixed above, so what warm start carries
        // over is the nuisance parameters, and those barely move between
        // adjacent points.
        if (warm_start) {
          const auto start = seeds.nearest(node, distance);
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
      workers.emplace_back(worker, w);

    for (auto& t : workers)
      t.join();
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
  const auto& format           = options->inputOptions().output_format();

  const bool warm_start = resolve_warm_start(options);

  scanseed::Store<scan::Node> seeds;

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
  const auto seed = bootstrap_seed_fit(options, module, "BestFit", names, format, warm_start, seeds, input_parameters.size());
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

    run_scan_batch(std::move(ordered), surface, options, module, input_parameters, names, format, warm_start, seeds, scan_workers,
                   [](const scan::Node& node) { return node_name(node); }, apply_fixed, distance, fits_performed);
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
void perform_2d_scan_regular(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, int points_x = 30, int points_y = 30) {
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
  const auto seed = bootstrap_seed_fit(options, module, "BestFitRegular", names, format, warm_start, seeds, input_parameters.size());
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

  run_scan_batch(std::move(nodes), surface, options, module, input_parameters, names, format, warm_start, seeds, scan_workers,
                 [](const scan::Node& node) { return node_name_regular(node); }, apply_fixed, distance, fits_performed);

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
 * Selected via --scanMode 1d, either for every non-fixed parameter
 * (perform_1d_scan_all) or, with --scanParameter, just this one
 * (perform_1d_scan).
 *
 * @param parameter_name Name of the parameter to profile, as it appears in the
 *                       config's parameter list.
 */
void perform_1d_scan_window(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module,
                             const std::string& parameter_name, int index, double low, double high) {
  const auto& input_parameters = options->inputOptions().input_parameters();
  const auto& names            = input_parameters.names();
  const auto& format           = options->inputOptions().output_format();

  const scan1d::Settings settings;

  const int scan_workers = std::max(1, options->inputOptions().scan_workers());

  const bool warm_start = resolve_warm_start(options);

  scanseed::Store<int> seeds;

  // Working in integer lattice coordinates keeps every point of every depth
  // exactly representable and gives each one a stable name.
  const double spacing = (high - low) / settings.lattice();

  auto position     = [&](int node) { return low + node * spacing; };
  auto distance      = [](int a, int b) { return std::abs(static_cast<double>(a - b)); };
  auto apply_fixed = [&](ROOT::Math::Minimizer& min, int node) {
    min.SetVariableValue(index, position(node));
    min.FixVariable(index);
  };

  {
    nlohmann::json grid;
    grid[parameter_name] = {{"low", low}, {"high", high}, {"lattice", settings.lattice()}};
    grid["levels"]       = settings.levels;
    write_or_check_grid("scan_grid_" + parameter_name + ".json", grid, "lattice", "Output_" + parameter_name + "_i");
  }

  // The free fit gives the reference the delta likelihood is measured against
  // before any scan point exists. It is only a starting value: any scan point
  // may undercut it, and refine() takes the lower of the two. Saved to disk
  // as BestFit_<parameter> so analysis can reference the true unfixed
  // minimum instead of the lowest sampled scan point.
  const auto seed = bootstrap_seed_fit(options, module, "BestFit_" + parameter_name, names, format, warm_start, seeds, input_parameters.size());
  if (!seed.from_store && !seed.parameters.empty())
    std::cout << "Seed fit: " << parameter_name << " = " << seed.parameters[index] << (seed.converged ? "" : " (did not converge)") << '\n';

  int fits_performed = 0;

  auto evaluate = [&](const std::vector<int>& nodes, scan1d::Profile& profile) {
    std::vector<int> ordered = nodes;

    // Within a round the points nearest the current minimum come first, so an
    // interrupted run still leaves a filled-in neighbourhood of the best fit.
    if (!profile.empty()) {
      const int best = std::ranges::min_element(profile, {}, [](const auto& entry) { return entry.second; })->first;
      std::ranges::sort(ordered, {}, [&](int node) { return distance(node, best); });
    }

    run_scan_batch(std::move(ordered), profile, options, module, input_parameters, names, format, warm_start, seeds, scan_workers,
                   [&](int node) { return node_name(parameter_name, node); }, apply_fixed, distance, fits_performed);
  };

  auto report = [scan_workers](int round, std::size_t split, std::size_t segments, std::size_t points) {
    if (round == 0)
      std::cout << "Coarse scan: " << points << " points"
                << (scan_workers > 1 ? " (" + std::to_string(scan_workers) + " workers)" : "") << '\n';
    else
      std::cout << "Refinement " << round << ": splitting " << split << " of " << segments << " intervals (" << points << " points so far)\n";
  };

  const scan1d::Profile profile = scan1d::refine(settings, seed.llh, evaluate, report);

  std::cout << "Scan of " << parameter_name << " finished: " << profile.size() << " points, " << fits_performed << " fitted in this run\n";
}

/**
 * @brief Profiles a single configured parameter on a regular grid instead of
 * the adaptive lattice of perform_1d_scan_window.
 *
 * Same window convention as perform_1d_scan_window (taken from the caller),
 * same warm-start/resume machinery, but every point in [low, high] is fitted
 * up front rather than refined around confidence crossings.
 *
 * @param points Number of grid points, including both endpoints.
 */
void perform_1d_scan_regular_window(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module,
                                     const std::string& parameter_name, int index, double low, double high, int points) {
  if (points < 2)
    throw std::runtime_error("perform_1d_scan_regular_window needs at least 2 points");

  const auto& input_parameters = options->inputOptions().input_parameters();
  const auto& names            = input_parameters.names();
  const auto& format           = options->inputOptions().output_format();

  const int scan_workers = std::max(1, options->inputOptions().scan_workers());

  const bool warm_start = resolve_warm_start(options);

  scanseed::Store<int> seeds;

  const double spacing = (high - low) / (points - 1);

  auto position     = [&](int node) { return low + node * spacing; };
  auto distance      = [](int a, int b) { return std::abs(static_cast<double>(a - b)); };
  auto apply_fixed = [&](ROOT::Math::Minimizer& min, int node) {
    min.SetVariableValue(index, position(node));
    min.FixVariable(index);
  };

  {
    nlohmann::json grid;
    grid[parameter_name] = {{"low", low}, {"high", high}, {"points", points}};
    write_or_check_grid("scan_grid_regular_" + parameter_name + ".json", grid, "grid", "OutputRegular_" + parameter_name + "_i");
  }

  // Also the source of the true (unfixed) best fit, saved to disk as
  // BestFit_<parameter> so analysis can reference it instead of the lowest
  // sampled grid point.
  const auto seed = bootstrap_seed_fit(options, module, "BestFitRegular_" + parameter_name, names, format, warm_start, seeds, input_parameters.size());
  if (!seed.from_store && !seed.parameters.empty())
    std::cout << "Seed fit: " << parameter_name << " = " << seed.parameters[index] << (seed.converged ? "" : " (did not converge)") << '\n';

  std::vector<int> nodes;
  nodes.reserve(static_cast<std::size_t>(points));
  for (int i = 0; i < points; ++i)
    nodes.push_back(i);

  std::map<int, double> profile;
  int                   fits_performed = 0;

  std::cout << "Regular scan of " << parameter_name << ": " << points << " points"
            << (scan_workers > 1 ? " (" + std::to_string(scan_workers) + " workers)" : "") << '\n';

  run_scan_batch(std::move(nodes), profile, options, module, input_parameters, names, format, warm_start, seeds, scan_workers,
                 [&](int node) { return node_name_regular(parameter_name, node); }, apply_fixed, distance, fits_performed);

  std::cout << "Regular scan of " << parameter_name << " finished: " << profile.size() << " points, " << fits_performed
            << " fitted in this run\n";
}

/// Looks up parameter_name and requires the window to come from its own
/// config LowerBound/UpperBound, same convention as perform_1d_scan, but
/// walks a regular grid of `points` values instead of the adaptive lattice.
void perform_1d_scan_regular(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module,
                              const std::string& parameter_name, int points) {
  const auto& input_parameters = options->inputOptions().input_parameters();
  const auto& names            = input_parameters.names();

  const auto found = std::ranges::find(names, parameter_name);
  if (found == names.end())
    throw std::runtime_error("No parameter named '" + parameter_name + "' in the config");

  const int   index     = static_cast<int>(std::ranges::distance(names.begin(), found));
  const auto& parameter = input_parameters.parameters()[index];

  const auto& lower = parameter.lower_bound();
  const auto& upper = parameter.upper_bound();
  if (!lower || !upper)
    throw std::runtime_error("Parameter '" + parameter_name + "' has no LowerBound/UpperBound in the config; the 1D scan takes its window from those");

  perform_1d_scan_regular_window(options, module, parameter_name, index, *lower, *upper, points);
}

/// Looks up parameter_name and requires the window to come from its own
/// config LowerBound/UpperBound: a parameter with no bounds gives no honest
/// window, and start value +- a few step widths would be an invention that
/// quietly decides the answer for a parameter whose profile is wide.
void perform_1d_scan(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, const std::string& parameter_name) {
  const auto& input_parameters = options->inputOptions().input_parameters();
  const auto& names            = input_parameters.names();

  const auto found = std::ranges::find(names, parameter_name);
  if (found == names.end())
    throw std::runtime_error("No parameter named '" + parameter_name + "' in the config");

  const int   index     = static_cast<int>(std::ranges::distance(names.begin(), found));
  const auto& parameter = input_parameters.parameters()[index];

  const auto& lower = parameter.lower_bound();
  const auto& upper = parameter.upper_bound();
  if (!lower || !upper)
    throw std::runtime_error("Parameter '" + parameter_name + "' has no LowerBound/UpperBound in the config; the 1D scan takes its window from those");

  perform_1d_scan_window(options, module, parameter_name, index, *lower, *upper);
}

/**
 * Runs a 1D scan over every non-fixed parameter in the config.
 *
 * A parameter with both LowerBound and UpperBound set scans that window, same
 * as perform_1d_scan. A parameter missing one or both is not skipped: a
 * single shared free fit (all non-fixed parameters floating) supplies a
 * Migrad/Hesse error per parameter, and the missing side of the window is
 * approximated as fitted_value +- 3 * error. Doubling or tripling the error
 * this way is only ever a coarse proxy for the true profile width -- it is
 * what a parabolic error extrapolates to, not what the (possibly asymmetric,
 * possibly flat) likelihood actually does out there -- so a side that does
 * have a config bound always keeps that bound instead of the approximation:
 * the config value is the real physical edge, and takes precedence.
 *
 * @param regular When true, every parameter is scanned on a regular grid of
 *                `points` points (perform_1d_scan_regular_window) instead of
 *                the adaptive lattice (perform_1d_scan_window).
 * @param points  Grid points per parameter; only used when regular is true.
 */
void perform_1d_scan_all(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, bool regular = false,
                          int points = 30) {
  const auto& input_parameters = options->inputOptions().input_parameters();
  const auto& names            = input_parameters.names();

  std::vector<std::size_t> unbounded_indices;
  for (std::size_t i = 0; i < input_parameters.size(); ++i) {
    if (input_parameters.fixed(static_cast<int>(i)))
      continue;
    const auto& parameter = input_parameters.parameters()[i];
    if (!parameter.lower_bound() || !parameter.upper_bound())
      unbounded_indices.push_back(i);
  }

  // The fallback window needs a fitted value and an error per parameter, and
  // Migrad already computes both for every floating parameter in one pass --
  // no reason to pay for a second fit just to profile one of them.
  std::vector<double> fallback_values;
  std::vector<double> fallback_errors;
  bool                 have_fallback_fit = false;
  if (!unbounded_indices.empty()) {
    ana::Fit reference_fit(options, module);
    reference_fit.minimize();
    const auto min = reference_fit.get_minimizer();
    if (reference_fit.converged() && min->Errors() != nullptr) {
      fallback_values.assign(min->X(), min->X() + input_parameters.size());
      fallback_errors.assign(min->Errors(), min->Errors() + input_parameters.size());
      have_fallback_fit = true;
    } else {
      std::cout << "Reference fit for approximated windows did not converge or has no errors; "
                   "parameters without config bounds will be skipped\n";
    }
  }

  constexpr double kErrorMultiplier = 3.0;

  for (std::size_t i = 0; i < input_parameters.size(); ++i) {
    if (input_parameters.fixed(static_cast<int>(i)))
      continue;

    const auto& parameter = input_parameters.parameters()[i];
    const auto& lower     = parameter.lower_bound();
    const auto& upper     = parameter.upper_bound();

    double low;
    double high;
    if (lower && upper) {
      low  = *lower;
      high = *upper;
    } else {
      if (!have_fallback_fit)
        continue;

      const double value = fallback_values[i];
      const double error = fallback_errors[i];
      low                = lower ? *lower : value - kErrorMultiplier * error;
      high               = upper ? *upper : value + kErrorMultiplier * error;

      std::cout << names[i] << ": approximated window [" << low << ", " << high
                << "] from fitted value " << value << " +- " << error << '\n';
    }

    std::cout << "Scanning " << names[i] << "...\n";
    if (regular)
      perform_1d_scan_regular_window(options, module, names[i], static_cast<int>(i), low, high, points);
    else
      perform_1d_scan_window(options, module, names[i], static_cast<int>(i), low, high);
  }
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

    // --fitOnly runs the plain fit and writes "Output.json"; without it the scan
    // picked by --scanMode (2d by default) is the entry point. The single fit
    // is what the NNMFit parity harness drives
    // (tools/nnmfit_oracle/compare_llh_value.py, run_fit_parity.sh), which needs
    // one converged result rather than a surface.
    if (options->inputOptions().fit_only()) {
      ana::Fit fit(options, module);
      fit.minimize();
      result::write_results(fit, "Output");
    } else {
      const std::string& scan_mode      = options->inputOptions().scan_mode();
      const std::string& scan_parameter = options->inputOptions().scan_parameter();
      const int           scan_points    = options->inputOptions().scan_points();

      if (scan_mode == "2d") {
        perform_2d_scan(options, module);
      } else if (scan_mode == "2d-regular") {
        perform_2d_scan_regular(options, module, scan_points, scan_points);
      } else if (scan_mode == "1d") {
        if (scan_parameter.empty())
          perform_1d_scan_all(options, module, false, scan_points);
        else
          perform_1d_scan(options, module, scan_parameter);
      } else if (scan_mode == "1d-regular") {
        if (scan_parameter.empty())
          perform_1d_scan_all(options, module, true, scan_points);
        else
          perform_1d_scan_regular(options, module, scan_parameter, scan_points);
      } else {
        throw std::invalid_argument("Unknown --scanMode \"" + scan_mode + "\". Valid values: 2d, 2d-regular, 1d, 1d-regular");
      }
    }
  } catch (const std::exception& e) {
    std::cout << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
