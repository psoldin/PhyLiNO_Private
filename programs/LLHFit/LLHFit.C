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

#include <TROOT.h>



namespace {

  /**
   * @brief Reads back the likelihood of a point an earlier run already fitted.
   *
   * @return The likelihood, or nullopt if the file is absent or holds no usable result.
   */
  std::optional<double> stored_llh(const std::string& name) {
    const std::filesystem::path path = name + ".json";
    if (!std::filesystem::exists(path))
      return std::nullopt;

    std::ifstream  file(path);
    nlohmann::json stored = nlohmann::json::parse(file, nullptr, false);
    if (stored.is_discarded() || !stored.contains("LLH"))
      return std::nullopt;

    const double llh = stored["LLH"].get<double>();
    return std::isfinite(llh) ? std::optional{llh} : std::nullopt;
  }

  std::string node_name(const scan::Node& node) {
    std::stringstream name;
    name << "Output_" << node.x << '_' << node.y;
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

        if (const auto known = stored_llh(name)) {
          const std::scoped_lock lock(surface_mutex);
          surface[node] = *known;
          continue;
        }

        const auto [x, y] = position(node);

        ana::Fit fit(options, module);
        auto     min = fit.get_minimizer();
        min->SetVariableValue(SpectralIndex, x);
        min->SetVariableValue(AstroNorm, y);
        min->FixVariable(AstroNorm);
        min->FixVariable(SpectralIndex);

        fit.minimize();

        result::write_results(fit, name);

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
      perform_2d_scan(options, module);
    }
  } catch (const std::exception& e) {
    std::cout << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
