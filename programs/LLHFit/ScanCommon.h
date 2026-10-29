#pragma once

// Helpers shared by every scan of LLHFit: reading back earlier results, start
// values and seeding, point names, grid descriptors, the free fit a scan
// bootstraps from, and the worker pool that fits a batch of points.

// STL includes
#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

// includes
#include "ExperimentModule.h"
#include "Fit.h"
#include "Options.h"
#include "write_results.h"

#include "AdaptiveGrid.h"
#include "ScanSeeds.h"
#include "SeedQuality.h"

namespace llhfit {

  /// A point an earlier run already fitted.
  struct StoredFit {
    double              llh = 0.0;
    std::vector<double> parameters;  ///< Empty unless the fit is usable as a seed; see seedquality::usable().
    /// Fitted errors in minimizer order, 0 where the file has none (e.g. a
    /// parameter --blind dropped). Empty exactly when `parameters` is: a 1D scan
    /// spaces its lattice by the best fit's error, so a free fit read back from
    /// disk has to bring its errors along or the lattice would change.
    std::vector<double> errors;
  };

  /**
   * @brief Reads back a point an earlier run already fitted.
   *
   * Reads the format the run is configured to write, so a resume never looks
   * for a file the current run would not itself produce.
   *
   * @param name             Base name of the point's output file.
   * @param input_parameters Configured parameters, for the minimizer order and
   *                         for the start values of anything the file omits.
   * @param format           --output-format, either "json" or "protobuf".
   * @return The result, or nullopt if the file is absent or holds no usable likelihood.
   */
  std::optional<StoredFit> stored_fit(const std::string& name, const io::InputParameter& input_parameters, const std::string& format);

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
  void apply_start_values(ROOT::Math::Minimizer& minimizer, const io::InputParameter& input_parameters, const std::vector<double>& values);

  /// Parameters an N-1 scan holds fixed on top of the scanned one, as
  /// (minimizer index, value) pairs.
  using FixedParameters = std::vector<std::pair<int, double>>;

  void apply_extra_fixed(ROOT::Math::Minimizer& min, const FixedParameters& fixed);

  /// The fixer of a scan that holds nothing beyond its own scanned parameter.
  inline constexpr auto no_extra_fixed = [](ROOT::Math::Minimizer&) {};

  std::string node_name(const scan::Node& node);
  std::string node_name(const std::string& parameter, int node);

  /// Distinct from node_name(const scan::Node&) above: a regular-grid node
  /// (i, j) sits at a different physical point than an adaptive-grid node of
  /// the same integer coordinates, so the two must never share a resume file.
  std::string node_name_regular(const scan::Node& node);

  /// 1D counterpart of node_name_regular(const scan::Node&) above -- distinct
  /// from node_name(parameter, node) so a regular-grid point never shares a
  /// resume file with an adaptive-grid point at the same integer index.
  std::string node_name_regular(const std::string& parameter, int node);

  /// Randomized start values exist to spread the start points on purpose, which
  /// seeding every fit from its neighbour would undo. The two do not combine.
  bool resolve_warm_start(const std::shared_ptr<io::Options>& options);

  /**
   * @brief Start values named by --seedFrom, empty if the option was not given.
   *
   * The free fit that opens a scan is a plain minimisation from the config's
   * StartValues, and on a likelihood with more than one minimum it can converge
   * cleanly into the wrong one -- which then anchors the whole lattice, sets
   * every Hesse error, and seeds every scan point that has no fitted neighbour.
   * This is the way out: point the run at a result that is known to sit at the
   * right minimum and every fit starts from there.
   *
   * Read with the same reader a resume uses, so the file may be in either
   * output format regardless of what this run writes -- a scan switched to
   * protobuf can still be seeded from the JSON fit that found the minimum. The
   * reader adds the extension, so the option takes a base name.
   *
   * Throws rather than warning: the option is an explicit instruction, and a
   * silently ignored one would leave the run doing exactly what it was told not
   * to do.
   *
   * The file is read once per run: the scans ask for it from several places,
   * and reading the file again each time would repeat both the work and the
   * message. Calling this before the module is built is also what turns a
   * mistyped --seedFrom into an error in the first second of a run instead of
   * after the MC has been loaded.
   */
  const std::vector<double>& seed_from_values(const std::shared_ptr<io::Options>& options);

  /**
   * @brief The point --evaluateAt names, in minimizer order; empty without the option.
   *
   * Unlike --seedFrom, the result is taken whatever its convergence: the point
   * is what the caller wants evaluated, not a start the minimizer could walk
   * away from. A result that did not converge is reported, not refused.
   * Parameters the result does not carry (a different config, or --blind
   * output) keep their configured StartValue, and are listed; so are names in
   * the result this config does not have. Read once per run and before the
   * module is built, like seed_from_values(), so a bad path fails at once.
   */
  const std::vector<double>& evaluate_at_values(const std::shared_ptr<io::Options>& options);

  /// Writes `path` the first time a scan runs in this directory; on every later
  /// run checks the stored descriptor still matches, so Output_* files from a
  /// differently configured scan are never silently reused.
  void write_or_check_grid(const std::string& path, const nlohmann::json& grid, const std::string& kind, const std::string& file_pattern);

  /// The parameters a 1D scan of every parameter would visit: the non-fixed
  /// ones, minus the signal parameters under --blind.
  std::vector<std::string> free_parameter_names(const std::shared_ptr<io::Options>& options);

  /// Minimizer index of the parameter called parameter_name; throws if the config has none.
  int parameter_index(const io::InputParameter& input_parameters, const std::string& parameter_name);

  /// Extension the result writers give a file of the given --output-format.
  std::string result_extension(const std::string& format);

  /**
   * @brief Lets a 1D scan start from the free fit a sibling scan already did.
   *
   * Every 1D scan of a run without extra fixed parameters starts from the same
   * free fit -- all non-fixed parameters floating -- and only names it after
   * the parameter it scans (`prefix` + label). Running it again per parameter
   * repeated the most expensive fit of the whole run once per parameter. If
   * this label has no free fit yet but a sibling's is on disk and usable, it is
   * copied under this label's name, and the scan reads it back exactly as a
   * resumed run would -- errors included, since they space the lattice.
   */
  void reuse_sibling_free_fit(const std::string& prefix, const std::string& label, const std::vector<std::string>& siblings, const io::InputParameter& input_parameters, const std::string& format);

  /// Result of the free (unfixed) fit that bootstraps a scan: its likelihood as
  /// the reference the adaptive grids measure delta chi2 from, and its
  /// converged parameters as the fallback warm-start point for cells that have
  /// no fitted neighbour yet.
  struct SeedFit {
    double              llh = std::numeric_limits<double>::infinity();
    std::vector<double> parameters;  ///< Fitted X; empty if not run, non-finite, or read back from disk.
    std::vector<double> errors;      ///< Migrad/Hesse errors, fitted in this run or read back with the parameters.
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
  template <typename Key, typename ApplyFixed>
  SeedFit bootstrap_seed_fit(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, const std::string& best_fit_name, const io::InputParameter& input_parameters, const std::string& format, bool warm_start, scanseed::Store<Key>& seeds, ApplyFixed&& apply_fixed) {
    const std::size_t n_parameters = input_parameters.size();

    if (const auto known = stored_fit(best_fit_name, input_parameters, format)) {
      std::cout << "Best fit already stored: " << known->llh << '\n';

      // The parameters come back too, unlike the likelihood alone that used to
      // be read here: the 1D walk anchors its lattice on the best fit's own
      // position, and a resumed run has to land on exactly the same anchor or
      // its point names would refer to different physical values.
      SeedFit result;
      result.llh        = known->llh;
      result.parameters = known->parameters;
      result.errors     = known->errors;
      result.from_store = true;
      if (warm_start && !known->parameters.empty())
        seeds.set_fallback(known->parameters);
      return result;
    }

    ana::Fit seed_fit(options, module);
    apply_fixed(*seed_fit.get_minimizer());

    // The store holds nothing yet, so this is the --seedFrom vector when one
    // was given and empty otherwise. Applied here because the free fit is the
    // one that most needs it: it is the only fit of a scan with no neighbour to
    // fall back on, and it is the fit the whole lattice is anchored on.
    if (const auto start = seeds.fallback(); !start.empty()) {
      std::cout << "Seeding the free fit from --seedFrom\n";
      apply_start_values(*seed_fit.get_minimizer(), input_parameters, start);
    }

    seed_fit.minimize();

    const auto seed_min = seed_fit.get_minimizer();

    SeedFit result;
    result.converged = seed_fit.converged();

    if (std::isfinite(seed_min->MinValue())) {
      result.llh = seed_min->MinValue();
      result.parameters.assign(seed_min->X(), seed_min->X() + n_parameters);
      if (seed_min->Errors() != nullptr)
        result.errors.assign(seed_min->Errors(), seed_min->Errors() + n_parameters);

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
   * @brief Takes node 0 of a 1D profile from the free fit instead of fitting it.
   *
   * Node 0 of the walk and of the crossing search sits exactly on the free
   * fit's value of the scanned parameter, so fixing it there and minimising the
   * rest reproduces the free fit -- at the cost of a whole fit. The free fit's
   * own result file is copied to the node's name instead, which keeps the point
   * on disk for plotting and for resuming, and the free fit's parameters seed
   * the neighbours exactly as the fitted node used to.
   *
   * @return True if node 0 was in `nodes` and has been filled in (and removed).
   */
  template <typename Key>
  bool node_zero_from_free_fit(std::vector<int>& nodes, std::map<int, double>& profile, const SeedFit& seed, const std::string& best_fit_name, const std::string& node_file, const std::string& format, bool warm_start, scanseed::Store<Key>& seeds) {
    const auto zero = std::ranges::find(nodes, 0);
    if (zero == nodes.end() || !std::isfinite(seed.llh))
      return false;

    const std::string           extension = result_extension(format);
    const std::filesystem::path source    = best_fit_name + extension;
    const std::filesystem::path target    = node_file + extension;
    if (!std::filesystem::exists(source))
      return false;
    if (!std::filesystem::exists(target))
      std::filesystem::copy_file(source, target);

    profile[0] = seed.llh;
    if (warm_start && !seed.parameters.empty())
      seeds.store(0, seed.parameters);
    nodes.erase(zero);
    return true;
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
  void run_scan_batch(std::vector<Key> nodes, std::map<Key, double>& surface, std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, const io::InputParameter& input_parameters, const std::string& format, bool warm_start, scanseed::Store<Key>& seeds, int scan_workers, NameFn&& name_fn, ApplyFixed&& apply_fixed, Distance&& distance, int& fits_performed, std::span<const double> step_sizes = {}) {
    const int        n_nodes   = static_cast<int>(nodes.size());
    const int        n_workers = std::clamp(scan_workers, 1, std::max(n_nodes, 1));
    std::atomic<int> next_index{0};
    std::mutex       surface_mutex;

    auto worker = [&](int worker_index) {
      for (int pos = next_index.fetch_add(1); pos < n_nodes; pos = next_index.fetch_add(1)) {
        const Key         node = nodes[pos];
        const std::string name = name_fn(node);

        if (const auto known = stored_fit(name, input_parameters, format)) {
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
        // A scan point only has to resolve the profile to well within the delta
        // chi2 the scan works at, not to the free fit's --tolerance.
        fit.set_tolerance(options->inputOptions().scan_tolerance());
        // Nothing reads a scan point's errors.
        fit.set_exact_errors(false);
        // The free fit's errors are the scale every scan point's parameters
        // move on, which the config's StepWidths can miss by an order of
        // magnitude; Migrad's numerical derivatives and first steps start
        // from them. (With --hessian gn Migrad seeds from the Hessian instead.)
        if (!step_sizes.empty())
          fit.set_step_sizes(step_sizes);
        auto min = fit.get_minimizer();
        apply_fixed(*min, node);

        // The scanned variable(s) are fixed above, so what warm start carries
        // over is the nuisance parameters, and those barely move between
        // adjacent points. Consulted even with warm start off: nothing is ever
        // stored then, so what comes back is the --seedFrom vector if there is
        // one and nothing at all if there is not.
        if (const auto start = seeds.nearest(node, distance); !start.empty())
          apply_start_values(*min, input_parameters, start);

        fit.minimize();

        result::write_results(fit, name);

        // Gated on EDM rather than on Migrad's verdict: a point that stopped
        // just short of the tolerance is still at its minimum for seeding
        // purposes, and dropping those empties the store on a run whose
        // convergence rate is low -- which leaves every point cold-starting
        // from the free fit. See SeedQuality.h.
        if (warm_start && seedquality::usable(fit.converged(), min->Edm()) && std::isfinite(min->MinValue()))
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

}  // namespace llhfit
