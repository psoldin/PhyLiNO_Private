#include "Scans.h"

#include "ScanCommon.h"

// STL includes
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// includes
#include "AdaptiveScan1D.h"
#include "CrossingScan1D.h"
#include "IceCube/ICBlinding.h"

namespace llhfit {

  namespace {


    /// Lattice units per sigma of the best fit. The walk sizes its own steps from
    /// the measured slope, so this only sets how finely a step can be resolved:
    /// fine enough that the whole 4 sigma range has room for the points the walk
    /// wants, coarse enough that node indices stay small and readable in file names.
    constexpr int kUnitsPerStepWidth = 64;

    /// Where a walk that already ran in this directory put its lattice.
    struct Lattice {
      double anchor = 0.0;  ///< Physical value of node 0, the best fit.
      double unit   = 0.0;  ///< Physical distance between two neighbouring nodes.
    };

    /// Lattice of an earlier run. A resumed run reuses it rather than the one it
    /// would derive now: both anchor and unit come from a fit, the two runs agree to
    /// many digits but not exactly, and a lattice that moved would put every node at
    /// a slightly different physical value than the file of the same name holds.
    std::optional<Lattice> stored_lattice(const std::string& path) {
      if (!std::filesystem::exists(path))
        return std::nullopt;

      std::ifstream  file(path);
      nlohmann::json stored = nlohmann::json::parse(file, nullptr, false);
      if (stored.is_discarded() || !stored.contains("anchor") || !stored.contains("unit"))
        return std::nullopt;

      return Lattice{stored["anchor"].get<double>(), stored["unit"].get<double>()};
    }

    /// Where a 1D scan of one parameter puts its points, and the free fit it is
    /// anchored on.
    struct Axis1D {
      SeedFit seed;
      double  anchor      = 0.0;  ///< Physical value of node 0, the best fit.
      double  unit        = 0.0;  ///< Physical distance between two neighbouring nodes.
      int     lower_limit = -(1 << 24);  ///< Configured LowerBound in lattice units.
      int     upper_limit = 1 << 24;     ///< Configured UpperBound in lattice units.
    };

    /**
     * @brief Runs or reads back the free fit of a 1D scan and lays its lattice.
     *
     * Shared by the walk (perform_1d_scan_walk) and the crossing search
     * (perform_1d_crossing): both anchor node 0 on the best fit and space their
     * nodes by the fitted error, so a parameter gets the same lattice whichever
     * of them runs.
     *
     * @param grid_path Descriptor of an earlier run in this directory; its lattice
     *                  is reused so resumed point files keep their meaning.
     */
    Axis1D anchor_1d_axis(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, const std::string& parameter_name, int index, const FixedParameters& extra_fixed, const std::string& label, const std::string& grid_path, bool warm_start, scanseed::Store<int>& seeds) {
      const auto& input_parameters = options->inputOptions().input_parameters();
      const auto& format           = options->inputOptions().output_format();
      const auto& parameter        = input_parameters.parameters()[index];

      Axis1D axis;

      // The free fit is both the reference the rise is measured from and the point
      // the scan starts at. Saved to disk as BestFit_<label> so analysis can
      // reference the true unfixed minimum instead of the lowest sampled point.
      axis.seed = bootstrap_seed_fit(options, module, "BestFit_" + label, input_parameters, format, warm_start, seeds,
                                     [&](ROOT::Math::Minimizer& min) { apply_extra_fixed(min, extra_fixed); });
      const auto& seed = axis.seed;

      double& anchor = axis.anchor;
      double& unit   = axis.unit;

      if (const auto known = stored_lattice(grid_path)) {
        anchor = known->anchor;
        unit   = known->unit;
      } else if (!seed.parameters.empty()) {
        anchor = seed.parameters[index];

        // The lattice has to resolve the profile, and the profile's own width is
        // the only honest statement of that scale. StepWidth is the fallback for a
        // fit that produced no error: it is the config's guess at the same thing,
        // and can be off by the factor the walk exists to stop mattering -- so it
        // decides the spacing only, never how far the walk goes.
        const double scale = index < static_cast<int>(seed.errors.size()) && seed.errors[index] > 0.0 ? seed.errors[index]
                                                                                                      : parameter.uncertainty();
        unit               = scale / kUnitsPerStepWidth;
      } else {
        // No fitted position to start from: the walk would otherwise anchor on
        // whatever the config happens to start at and quietly scan around a point
        // that is not the best fit.
        throw std::runtime_error("Best fit of '" + label + "' produced no parameter values; the 1D scan has nothing to start from");
      }

      if (!(unit > 0.0))
        throw std::runtime_error("Parameter '" + parameter_name + "' has neither a fitted error nor a positive StepWidth; the 1D scan has no scale to space its points on");

      if (!seed.from_store)
        std::cout << "Seed fit: " << parameter_name << " = " << anchor << (seed.converged ? "" : " (did not converge)")
                  << ", scanning in steps of " << unit << '\n';

      // Configured bounds become hard stops. A bound the scan never reaches costs
      // nothing.
      if (const auto& lower = parameter.lower_bound())
        axis.lower_limit = std::max(axis.lower_limit, static_cast<int>(std::ceil((*lower - anchor) / unit)));
      if (const auto& upper = parameter.upper_bound())
        axis.upper_limit = std::min(axis.upper_limit, static_cast<int>(std::floor((*upper - anchor) / unit)));

      return axis;
    }

    /**
     * @brief Profiles the likelihood along a single configured parameter.
     *
     * The one-dimensional counterpart of perform_2d_scan: the scanned parameter is
     * fixed point by point and every other parameter is minimised over, so what
     * comes out is a profile likelihood rather than a slice.
     *
     * There is no window. The scan starts at the best fit and walks outward along
     * the axis until the profile has risen by scan1d::Settings::target (16, i.e. 4
     * sigma) in each direction, so how far it goes is decided by the likelihood
     * instead of by a range guessed in advance. Configured LowerBound/UpperBound,
     * where present, only stop a direction early: outside them the minimiser cannot
     * go, so neither can an honest profile. See AdaptiveScan1D.h.
     *
     * Points are named by their integer offset from the best fit, so a resumed run
     * picks up every fit an earlier one completed.
     *
     * Selected via --scanMode 1d, either for every non-fixed parameter
     * (perform_1d_scan_all) or, with --scanParameter, just this one
     * (perform_1d_scan).
     *
     * @param parameter_name Name of the parameter to profile, as it appears in the
     *                       config's parameter list.
     * @param index          Its position in minimizer order.
     * @param extra_fixed    Parameters held fixed on top of the scanned one, which
     *                       is how an N-1 scan is run (perform_nm1_scan). They are
     *                       fixed for the best fit this walk is anchored on as well,
     *                       so the profile and its reference describe the same fit.
     * @param tag            Prefix distinguishing this scan's files from the plain
     *                       scan of the same parameter. Empty for the plain scan.
     */
    void perform_1d_scan_walk(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, const std::string& parameter_name, int index, const FixedParameters& extra_fixed = {}, const std::string& tag = "") {
      const auto& input_parameters = options->inputOptions().input_parameters();
      const auto& names            = input_parameters.names();
      const auto& format           = options->inputOptions().output_format();

      const std::string label = tag + parameter_name;

      const int scan_workers = std::max(1, options->inputOptions().scan_workers());

      const bool warm_start = resolve_warm_start(options);

      scanseed::Store<int> seeds;
      seeds.set_fallback(seed_from_values(options));

      const std::string grid_path = "scan_grid_" + label + ".json";

      const Axis1D axis   = anchor_1d_axis(options, module, parameter_name, index, extra_fixed, label, grid_path, warm_start, seeds);
      const auto&  seed   = axis.seed;
      const double anchor = axis.anchor;
      const double unit   = axis.unit;

      scan1d::Settings settings;
      settings.start_step = kUnitsPerStepWidth;

      // One point per worker per direction, so a round keeps the whole pool busy.
      settings.batch = scan_workers;

      settings.lower_limit = axis.lower_limit;
      settings.upper_limit = axis.upper_limit;

      auto position    = [&](int node) { return anchor + node * unit; };
      auto distance    = [](int a, int b) { return std::abs(static_cast<double>(a - b)); };
      auto apply_fixed = [&](ROOT::Math::Minimizer& min, int node) {
        min.SetVariableValue(index, position(node));
        min.FixVariable(index);
        apply_extra_fixed(min, extra_fixed);
      };

      {
        nlohmann::json grid;
        grid["anchor"]       = anchor;
        grid["unit"]         = unit;
        grid[parameter_name] = {{"lower_limit", settings.lower_limit}, {"upper_limit", settings.upper_limit}};
        grid["target"]       = settings.target;
        for (const auto& [fixed_index, value] : extra_fixed)
          grid["fixed"][names[fixed_index]] = value;

        write_or_check_grid(grid_path, grid, "walk", "Output_" + label + "_i");
      }

      int fits_performed = 0;

      auto evaluate = [&](const std::vector<int>& nodes, scan1d::Profile& profile) {
        std::vector<int> ordered = nodes;

        node_zero_from_free_fit(ordered, profile, seed, "BestFit_" + label, node_name(label, 0), format, warm_start, seeds);
        if (ordered.empty())
          return;

        // Within a round the points nearest the current minimum come first, so an
        // interrupted run still leaves a filled-in neighbourhood of the best fit.
        if (!profile.empty()) {
          const int best = std::ranges::min_element(profile, {}, [](const auto& entry) { return entry.second; })->first;
          std::ranges::sort(ordered, {}, [&](int node) { return distance(node, best); });
        }

        run_scan_batch(std::move(ordered), profile, options, module, input_parameters, format, warm_start, seeds, scan_workers, [&](int node) { return node_name(label, node); }, apply_fixed, distance, fits_performed, seed.errors);
      };

      auto report = [scan_workers](int round, std::size_t proposed, std::size_t points) {
        if (round == 0)
          std::cout << "Walk starts at the best fit" << (scan_workers > 1 ? " (" + std::to_string(scan_workers) + " workers)" : "") << '\n';
        else
          std::cout << "Round " << round << ": " << proposed << " points proposed (" << points << " points so far)\n";
      };

      const scan1d::Profile profile = scan1d::walk(settings, seed.llh, evaluate, report);

      std::cout << "Scan of " << label << " finished: " << profile.size() << " points, " << fits_performed << " fitted in this run";
      if (!profile.empty())
        std::cout << ", covering " << parameter_name << " in [" << position(profile.begin()->first) << ", " << position(profile.rbegin()->first) << ']';
      std::cout << '\n';
    }

    /**
     * @brief Profiles a single configured parameter on a regular grid instead of
     * the outward walk of perform_1d_scan_walk.
     *
     * Same window convention as its caller (taken from the caller),
     * same warm-start/resume machinery, but every point in [low, high] is fitted
     * up front rather than refined around confidence crossings.
     *
     * @param points Number of grid points, including both endpoints.
     */
    void perform_1d_scan_regular_window(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, const std::string& parameter_name, int index, double low, double high, int points) {
      if (points < 2)
        throw std::runtime_error("perform_1d_scan_regular_window needs at least 2 points");

      const auto& input_parameters = options->inputOptions().input_parameters();
      const auto& names            = input_parameters.names();
      const auto& format           = options->inputOptions().output_format();

      const int scan_workers = std::max(1, options->inputOptions().scan_workers());

      const bool warm_start = resolve_warm_start(options);

      scanseed::Store<int> seeds;
      seeds.set_fallback(seed_from_values(options));

      const double spacing = (high - low) / (points - 1);

      auto position    = [&](int node) { return low + node * spacing; };
      auto distance    = [](int a, int b) { return std::abs(static_cast<double>(a - b)); };
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
      const auto seed = bootstrap_seed_fit(options, module, "BestFitRegular_" + parameter_name, input_parameters, format, warm_start, seeds, no_extra_fixed);
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

      run_scan_batch(std::move(nodes), profile, options, module, input_parameters, format, warm_start, seeds, scan_workers, [&](int node) { return node_name_regular(parameter_name, node); }, apply_fixed, distance, fits_performed, seed.errors);

      std::cout << "Regular scan of " << parameter_name << " finished: " << profile.size() << " points, " << fits_performed
                << " fitted in this run\n";
    }

  }  // namespace

  /// Looks up parameter_name and requires the window to come from its own
  /// config LowerBound/UpperBound, same convention as perform_1d_scan, but
  /// walks a regular grid of `points` values instead of the adaptive lattice.
  void perform_1d_scan_regular(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, const std::string& parameter_name, int points) {
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

  /// Looks up parameter_name and profiles it. Unlike the regular-grid scan this
  /// needs no window: the walk decides for itself how far it has to go
  /// (perform_1d_scan_walk), so a parameter without config bounds is scanned like
  /// any other.
  void perform_1d_scan(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, const std::string& parameter_name) {
    const auto& input_parameters = options->inputOptions().input_parameters();

    perform_1d_scan_walk(options, module, parameter_name, parameter_index(input_parameters, parameter_name));
  }

  /**
   * @brief Runs the N-1 scans of one parameter.
   *
   * One profile of `target` per other free parameter, each with that parameter
   * held at its configured AsimovValue -- the point the data was injected at,
   * which is where a sensitivity study wants a nuisance pinned rather than at a
   * fitted value that already absorbed part of the signal. Holding a parameter
   * removes whatever degeneracy it has with the target, so the width of each
   * profile says what that one parameter costs the target's sensitivity.
   *
   * The plain profile (nothing held) is run first as the reference the N-1 curves
   * are read against. It writes the same files as --scanMode 1d, so the two share
   * their fits instead of repeating them.
   *
   * Every scan gets its own best fit, itself run with the same parameter held, so
   * a profile and the reference it is measured from always describe the same fit.
   *
   * Selected via --scanMode nm1, which needs --scanParameter to say what the
   * target is.
   */
  void perform_nm1_scan(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, const std::string& target) {
    const auto& input_parameters = options->inputOptions().input_parameters();
    const auto& names            = input_parameters.names();
    const auto& parameters       = input_parameters.parameters();

    const int index = parameter_index(input_parameters, target);

    std::cout << "N-1 scan of " << target << ": reference profile with every parameter free\n";
    perform_1d_scan_walk(options, module, target, index);

    for (std::size_t i = 0; i < input_parameters.size(); ++i) {
      if (static_cast<int>(i) == index || input_parameters.fixed(static_cast<int>(i)))
        continue;

      const FixedParameters held{{static_cast<int>(i), parameters[i].asimov_value()}};

      std::cout << "N-1 scan of " << target << ": holding " << names[i] << " at its injected value " << parameters[i].asimov_value() << '\n';
      perform_1d_scan_walk(options, module, target, index, held, "nm1_" + names[i] + "_");
    }
  }

  namespace {

    const char* status_name(scan1d::crossing::Status status) {
      using enum scan1d::crossing::Status;
      switch (status) {
        case converged: return "converged";
        case at_limit: return "at_limit";
        case failed: return "failed";
        case exhausted: return "exhausted";
      }
      return "unknown";
    }

  }  // namespace

  /**
   * @brief Finds the 1 sigma interval of a single parameter without mapping its profile.
   *
   * The cheap counterpart of perform_1d_scan_walk for when only the interval is
   * wanted: same free fit, same lattice, same per-point profile fits, but the
   * points go only where the profile crosses --crossingLevel (1 by default) on
   * either side of the best fit -- a handful of fits instead of the walk's
   * several dozen. See CrossingScan1D.h for the search.
   *
   * Its point files are Crossing_<label>_<node> and its descriptor
   * crossing_grid_<label>.json, kept apart from the walk's so a directory can
   * hold both without either reading the other's points. BestFit_<label> is
   * shared: it is the same free fit.
   *
   * Writes Crossing_<label>.json with the interval, the status of each side and
   * the points it fitted.
   *
   * Selected via --scanMode 1d-crossing, for every non-fixed parameter
   * (perform_1d_crossing_all) or, with --scanParameter, just this one.
   *
   * @return The content written to Crossing_<label>.json.
   */
  nlohmann::json perform_1d_crossing(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, const std::string& parameter_name, int index) {
    const auto& input_parameters = options->inputOptions().input_parameters();
    const auto& format           = options->inputOptions().output_format();

    const std::string& label = parameter_name;

    const int scan_workers = std::max(1, options->inputOptions().scan_workers());

    const bool warm_start = resolve_warm_start(options);

    scanseed::Store<int> seeds;
    seeds.set_fallback(seed_from_values(options));

    const std::string grid_path = "crossing_grid_" + label + ".json";

    const Axis1D axis   = anchor_1d_axis(options, module, parameter_name, index, {}, label, grid_path, warm_start, seeds);
    const double anchor = axis.anchor;
    const double unit   = axis.unit;

    scan1d::crossing::Settings settings;
    settings.level       = options->inputOptions().crossing_level();
    settings.start_step  = kUnitsPerStepWidth;
    settings.lower_limit = axis.lower_limit;
    settings.upper_limit = axis.upper_limit;

    // The two sides share the pool: half the workers each.
    settings.batch = std::max(1, scan_workers / 2);

    auto position    = [&](double node) { return anchor + node * unit; };
    auto distance    = [](int a, int b) { return std::abs(static_cast<double>(a - b)); };
    auto apply_fixed = [&](ROOT::Math::Minimizer& min, int node) {
      min.SetVariableValue(index, position(node));
      min.FixVariable(index);
    };

    {
      nlohmann::json grid;
      grid["anchor"]       = anchor;
      grid["unit"]         = unit;
      grid[parameter_name] = {{"lower_limit", settings.lower_limit}, {"upper_limit", settings.upper_limit}};
      grid["level"]        = settings.level;

      write_or_check_grid(grid_path, grid, "crossing search", "Crossing_" + label + "_i");
    }

    int fits_performed = 0;

    auto evaluate = [&](const std::vector<int>& nodes, scan1d::Profile& profile) {
      std::vector<int> remaining = nodes;
      node_zero_from_free_fit(remaining, profile, axis.seed, "BestFit_" + label, "Crossing_" + label + "_0", format, warm_start, seeds);
      if (remaining.empty())
        return;
      run_scan_batch(std::move(remaining), profile, options, module, input_parameters, format, warm_start, seeds, scan_workers,
                     [&](int node) { return "Crossing_" + label + "_" + std::to_string(node); }, apply_fixed, distance, fits_performed, axis.seed.errors);
    };

    auto report = [](int round, std::size_t proposed, std::size_t points) {
      if (round > 0)
        std::cout << "Round " << round << ": " << proposed << " points proposed (" << points << " points so far)\n";
    };

    const auto result = scan1d::crossing::search(settings, axis.seed.llh, evaluate, report);

    const double best  = result.undercut ? position(result.minimum) : anchor;
    const double lower = position(result.sides[0].crossing);
    const double upper = position(result.sides[1].crossing);

    nlohmann::json summary;
    summary["parameter"]      = parameter_name;
    summary["level"]          = settings.level;
    summary["best_fit"]       = best;
    summary["lower"]          = lower;
    summary["upper"]          = upper;
    summary["minus"]          = best - lower;
    summary["plus"]           = upper - best;
    summary["lower_status"]   = status_name(result.sides[0].status);
    summary["upper_status"]   = status_name(result.sides[1].status);
    summary["undercut"]       = result.undercut;
    summary["fits"]           = result.profile.size();
    summary["fits_performed"] = fits_performed;
    for (const auto& [node, value] : result.profile)
      summary["points"].push_back({{"value", position(node)}, {"delta_chi2", value - result.reference}});

    std::ofstream("Crossing_" + label + ".json") << summary.dump(2) << '\n';

    std::cout << label << " = " << best << " -" << best - lower << " +" << upper - best << " (delta chi2 = " << settings.level << ", "
              << result.profile.size() << " points, " << fits_performed << " fitted in this run; sides "
              << status_name(result.sides[0].status) << '/' << status_name(result.sides[1].status) << ")\n";

    if (result.undercut)
      std::cout << "Warning: a scan point of " << label << " came out " << axis.seed.llh - result.reference
                << " below the free fit; the free fit missed its minimum and this interval is not trustworthy."
                   " Rerun with --seedFrom that point.\n";

    return summary;
  }

  /// --scanMode 1d-crossing without --scanParameter: the interval of every
  /// non-fixed parameter, collected into crossings.json.
  void perform_1d_crossing_all(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module) {
    const auto& input_parameters = options->inputOptions().input_parameters();
    const auto& names            = input_parameters.names();
    const bool  blind            = options->inputOptions().blind();

    const std::vector<std::string> scanned = free_parameter_names(options);

    nlohmann::json all;
    for (std::size_t i = 0; i < input_parameters.size(); ++i) {
      if (input_parameters.fixed(static_cast<int>(i)) || (blind && result::ic::is_blinded_parameter(names[i])))
        continue;

      reuse_sibling_free_fit("BestFit_", names[i], scanned, input_parameters, options->inputOptions().output_format());
      std::cout << "Searching the crossings of " << names[i] << "...\n";
      all[names[i]] = perform_1d_crossing(options, module, names[i], static_cast<int>(i));
    }

    std::ofstream("crossings.json") << all.dump(2) << '\n';
  }

  /**
   * Runs a 1D scan over every non-fixed parameter in the config.
   *
   * The adaptive scan needs no window: each parameter is walked outward from the
   * best fit until its profile has risen far enough (perform_1d_scan_walk).
   *
   * The regular grid does need one, and takes it from the parameter's own
   * LowerBound/UpperBound. A parameter missing one or both is not skipped: a
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
   *                being walked outward (perform_1d_scan_walk).
   * @param points  Grid points per parameter; only used when regular is true.
   */
  void perform_1d_scan_all(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, bool regular, int points) {
    const auto& input_parameters = options->inputOptions().input_parameters();
    const auto& names            = input_parameters.names();
    const bool  blind            = options->inputOptions().blind();

    // Under --blind the signal parameters are not scanned at all: their profile
    // is exactly what the blinding withholds from the written results.
    auto skipped = [&](std::size_t i) {
      return input_parameters.fixed(static_cast<int>(i)) || (blind && result::ic::is_blinded_parameter(names[i]));
    };

    const std::vector<std::string> scanned = free_parameter_names(options);

    const auto& format = options->inputOptions().output_format();

    if (!regular) {
      for (std::size_t i = 0; i < input_parameters.size(); ++i) {
        if (skipped(i))
          continue;

        reuse_sibling_free_fit("BestFit_", names[i], scanned, input_parameters, format);
        std::cout << "Scanning " << names[i] << "...\n";
        perform_1d_scan_walk(options, module, names[i], static_cast<int>(i));
      }

      return;
    }

    std::vector<std::size_t> unbounded_indices;
    for (std::size_t i = 0; i < input_parameters.size(); ++i) {
      if (skipped(i))
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
    bool                have_fallback_fit = false;
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
      if (skipped(i))
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

      reuse_sibling_free_fit("BestFitRegular_", names[i], scanned, input_parameters, format);
      std::cout << "Scanning " << names[i] << "...\n";
      perform_1d_scan_regular_window(options, module, names[i], static_cast<int>(i), low, high, points);
    }
  }

}  // namespace llhfit
