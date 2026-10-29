#include "Scans.h"

#include "ScanCommon.h"

// STL includes
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// includes
#include "AdaptiveGrid.h"

namespace llhfit {

  namespace {


    /// The free parameters named by --scanParameter as a comma-separated list, or
    /// all of them when it is empty (--scanMode 2d-stencil and 2d-pairs).
    std::vector<std::string> selected_free_parameters(const std::shared_ptr<io::Options>& options) {
      std::vector<std::string> free = free_parameter_names(options);
      const auto&              list = options->inputOptions().scan_parameter();
      if (list.empty())
        return free;

      std::vector<std::string> requested;
      std::stringstream        stream(list);
      for (std::string name; std::getline(stream, name, ',');) {
        if (name.empty())
          continue;
        if (std::ranges::find(free, name) == free.end())
          throw std::invalid_argument("--scanParameter " + name + " is not a free parameter of this config");
        requested.push_back(name);
      }
      return requested;
    }

    /// A point of the covariance stencil around the free fit: parameter i moved by
    /// si steps and, for a pair point, parameter j by sj steps (j < 0 for a point
    /// of the 1D profile of i). Indices are minimizer indices.
    struct StencilPoint {
      int i  = 0;
      int j  = -1;
      int si = 0;
      int sj = 0;

      friend auto operator<=>(const StencilPoint&, const StencilPoint&) = default;
    };

  }  // namespace

  /**
   * @brief Measures the covariance of the free parameters from profile fits around the best fit.
   *
   * The cross-check of Hesse's V = 2 H^-1 that stays a profile likelihood: no
   * second derivative is taken, every number is a Delta(-2 ln L) of a fit with
   * one or two parameters fixed and all others minimised. It uses the fewest
   * such fits that determine the covariance:
   *
   * - per parameter i, the 1D profile at x_i +- h_i, which gives the width
   *   sigma_i = h_i / sqrt(Delta) -- what a 1D scan's Delta = 1 crossing
   *   measures -- per side and from the mean curvature of both;
   * - per pair (i, j), the 2D profile at the four corners (x_i +- h_i,
   *   x_j +- h_j). For a Gaussian likelihood Delta(s_i h_i, s_j h_j) =
   *   (k_i^2 + k_j^2 - 2 s_i s_j rho k_i k_j) / (1 - rho^2) with k = h / sigma,
   *   so the antisymmetric combination of the corners over their sum fixes rho
   *   without the overall scale entering.
   *
   * That is 2 n + 2 n (n - 1) fits for n parameters, against 8 per pair for a
   * 3 x 3 grid. Steps are --stencilStep times the free fit's error, shrunk to
   * stay inside the configured bounds so both sides keep the same step; a
   * parameter on its bound has no symmetric step and is left out. Everything
   * assumes the likelihood is close to quadratic within one step: where it is
   * not, the two sides of sigma_i disagree, and so does the stencil from Hesse.
   *
   * Writes stencil_covariance.json with the widths, correlations and the
   * covariance assembled from them, which -- built pair by pair -- need not be
   * positive definite. --scanParameter takes a comma-separated subset of the
   * free parameters (default: all of them). Selected via --scanMode 2d-stencil.
   */
  void perform_2d_stencil(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module) {
    const auto& input_parameters = options->inputOptions().input_parameters();
    const auto& names            = input_parameters.names();
    const auto& format           = options->inputOptions().output_format();
    const int   scan_workers     = std::max(1, options->inputOptions().scan_workers());
    const double step            = options->inputOptions().stencil_step();

    if (!(step > 0.0))
      throw std::invalid_argument("--stencilStep must be positive");

    const std::vector<std::string> selected = selected_free_parameters(options);

    const bool warm_start = resolve_warm_start(options);

    scanseed::Store<StencilPoint> seeds;
    seeds.set_fallback(seed_from_values(options));

    const auto seed = bootstrap_seed_fit(options, module, "BestFit", input_parameters, format, warm_start, seeds, no_extra_fixed);
    if (seed.parameters.empty() || seed.errors.empty())
      throw std::runtime_error("The free fit produced no parameters or errors; the stencil has nothing to centre on");

    // Minimizer index and step of every parameter the stencil moves.
    std::vector<int>    index;
    std::vector<double> h;
    for (const auto& name : selected) {
      const int    i         = parameter_index(input_parameters, name);
      const auto&  parameter = input_parameters.parameters()[i];
      const double centre    = seed.parameters[i];
      double       hi        = step * seed.errors[i];
      if (const auto& lower = parameter.lower_bound())
        hi = std::min(hi, 0.9 * (centre - *lower));
      if (const auto& upper = parameter.upper_bound())
        hi = std::min(hi, 0.9 * (*upper - centre));
      if (!(hi > 0.0)) {
        std::cout << name << " sits on a bound or has no error; left out of the stencil\n";
        continue;
      }
      if (hi < step * seed.errors[i])
        std::cout << name << ": step shrunk to " << hi / seed.errors[i] << " sigma by its bounds\n";
      index.push_back(i);
      h.push_back(hi);
    }

    const std::size_t n = index.size();
    if (n < 2)
      throw std::runtime_error("The stencil needs at least two free parameters with room inside their bounds");

    std::map<int, double> step_of;
    for (std::size_t a = 0; a < n; ++a)
      step_of[index[a]] = h[a];

    {
      nlohmann::json grid;
      for (std::size_t a = 0; a < n; ++a)
        grid[names[index[a]]] = {{"centre", seed.parameters[index[a]]}, {"step", h[a]}};
      write_or_check_grid("stencil_grid.json", grid, "stencil", "Stencil_*");
    }

    // The 1D points first: they are what the pair points warm-start from.
    std::vector<StencilPoint> nodes;
    nodes.reserve(2 * n * n);
    for (std::size_t a = 0; a < n; ++a)
      for (const int s : {1, -1})
        nodes.push_back({index[a], -1, s, 0});
    for (std::size_t a = 0; a < n; ++a)
      for (std::size_t b = a + 1; b < n; ++b)
        for (const int si : {1, -1})
          for (const int sj : {1, -1})
            nodes.push_back({index[a], index[b], si, sj});

    auto apply_fixed = [&](ROOT::Math::Minimizer& min, const StencilPoint& p) {
      min.SetVariableValue(p.i, seed.parameters[p.i] + p.si * step_of.at(p.i));
      min.FixVariable(p.i);
      if (p.j >= 0) {
        min.SetVariableValue(p.j, seed.parameters[p.j] + p.sj * step_of.at(p.j));
        min.FixVariable(p.j);
      }
    };

    // In steps: every stencil point is one step from the free fit along one or
    // two axes, so this counts the axes two points differ along.
    auto offset   = [](const StencilPoint& p, int k) { return (p.i == k ? p.si : 0) + (p.j == k ? p.sj : 0); };
    auto distance = [&](const StencilPoint& a, const StencilPoint& b) {
      double sum = 0.0;
      for (const int k : index)
        sum += std::pow(offset(a, k) - offset(b, k), 2);
      return std::sqrt(sum);
    };

    auto name_fn = [&](const StencilPoint& p) {
      auto        sign = [](int s) { return s > 0 ? "p" : "m"; };
      std::string name = "Stencil_" + names[p.i] + '_' + sign(p.si);
      if (p.j >= 0)
        name += '_' + names[p.j] + '_' + sign(p.sj);
      return name;
    };

    std::cout << "Stencil: " << nodes.size() << " points for " << n << " parameters"
              << (scan_workers > 1 ? " (" + std::to_string(scan_workers) + " workers)" : "") << '\n';

    std::map<StencilPoint, double> surface;
    int                            fits_performed = 0;
    run_scan_batch(std::move(nodes), surface, options, module, input_parameters, format, warm_start, seeds, scan_workers, name_fn, apply_fixed, distance, fits_performed, seed.errors);

    std::cout << "Stencil finished: " << surface.size() << " points, " << fits_performed << " fitted in this run\n";

    // Delta(-2 ln L) of a point above the free fit; NaN, written as null, for a
    // point that produced no likelihood.
    const double nan  = std::numeric_limits<double>::quiet_NaN();
    auto         rise = [&](const StencilPoint& p) {
      const auto it = surface.find(p);
      return it != surface.end() && std::isfinite(it->second) ? it->second - seed.llh : nan;
    };

    std::vector<double> sigma(n), sigma_up(n), sigma_down(n);
    for (std::size_t a = 0; a < n; ++a) {
      const double up   = rise({index[a], -1, 1, 0});
      const double down = rise({index[a], -1, -1, 0});
      sigma_up[a]       = h[a] / std::sqrt(up);
      sigma_down[a]     = h[a] / std::sqrt(down);
      sigma[a]          = h[a] * std::sqrt(2.0 / (up + down));
    }

    std::vector<std::vector<double>> correlation(n, std::vector<double>(n, 1.0));
    std::vector<std::vector<double>> covariance(n, std::vector<double>(n));
    for (std::size_t a = 0; a < n; ++a) {
      covariance[a][a] = sigma[a] * sigma[a];
      for (std::size_t b = a + 1; b < n; ++b) {
        const double pp  = rise({index[a], index[b], 1, 1});
        const double pm  = rise({index[a], index[b], 1, -1});
        const double mp  = rise({index[a], index[b], -1, 1});
        const double mm  = rise({index[a], index[b], -1, -1});
        const double ka  = h[a] / sigma[a];
        const double kb  = h[b] / sigma[b];
        const double rho = (pm + mp - pp - mm) * (ka * ka + kb * kb) / (2.0 * (pp + pm + mp + mm) * ka * kb);

        correlation[a][b] = correlation[b][a] = rho;
        covariance[a][b] = covariance[b][a] = rho * sigma[a] * sigma[b];
      }
    }

    nlohmann::json result;
    result["reference"] = seed.llh;
    for (std::size_t a = 0; a < n; ++a)
      result["parameters"].push_back(names[index[a]]);
    result["steps"]       = h;
    result["sigma"]       = sigma;
    result["sigmaUp"]     = sigma_up;
    result["sigmaDown"]   = sigma_down;
    result["correlation"] = correlation;
    result["covariance"]  = covariance;
    std::ofstream("stencil_covariance.json") << result.dump(2) << '\n';
  }

  /**
   * @brief Traces the Delta(-2 ln L) = 1 contour of the 2D profile of every pair of free parameters.
   *
   * The 2D profile of (i, j) -- both fixed, all other parameters minimised --
   * makes no assumption about the shape of the likelihood, and the bounding box
   * of its Delta = 1 region is exactly the 1D profile interval of i and of j, so
   * every pair measures both 1D intervals besides the correlation between them.
   * Each pair is the adaptive quadtree of perform_2d_scan resolving only the
   * Delta = 1 level: cells the contour does not cross are not refined, and the
   * interior only to depth 1.
   *
   * The lattice is laid out in coordinates decorrelated by the free fit's Hesse
   * block of the pair, (x_i, x_j) = centre + L (u, v) with L L^T the 2 x 2
   * covariance: a pair with |rho| near 1 would otherwise be a thin diagonal
   * valley the coarse grid can step over, and here it is a round region as long
   * as Hesse is roughly right. Where it is not, the contour is only distorted --
   * the coarse grid's centre node is the free fit itself, so the cells around it
   * are always refined -- and a contour that runs into the window edge is
   * reported. The window is +-(--scanWindow) in (u, v). Points outside a
   * parameter's bounds are not fitted and are written as null: the minimiser
   * cannot reach them, so neither can the profile.
   *
   * One free fit centres every pair (pairs_reference.json, with its Hesse
   * covariance; BestFit is the full result). Each pair scans in its own
   * directory pairs/<i>__<j>/ with the usual resumable Output_x_y points and a
   * surface.json of every point in physical coordinates for analysis.
   * --scanParameter takes a comma-separated subset of the free parameters
   * (default: all of them). Selected via --scanMode 2d-pairs.
   */
  void perform_2d_pairs(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module) {
    const auto&       input_parameters = options->inputOptions().input_parameters();
    const auto&       format           = options->inputOptions().output_format();
    const int         scan_workers     = std::max(1, options->inputOptions().scan_workers());
    const double      window           = options->inputOptions().scan_window();
    const std::size_t n_parameters     = input_parameters.size();

    if (!(window > 0.0))
      throw std::invalid_argument("--scanWindow must be positive");

    const std::vector<std::string> selected = selected_free_parameters(options);
    if (selected.size() < 2)
      throw std::invalid_argument("--scanMode 2d-pairs needs at least two free parameters");

    const bool warm_start = resolve_warm_start(options);

    // The free fit every pair is centred on, with the Hesse covariance that
    // bootstrap_seed_fit() does not carry. Kept in a file of its own so a resumed
    // run centres every pair on exactly the same point.
    const std::string reference_path = "pairs_reference.json";
    nlohmann::json    reference;
    if (std::filesystem::exists(reference_path)) {
      std::ifstream file(reference_path);
      reference = nlohmann::json::parse(file);
      std::cout << "Best fit already stored: " << reference["llh"].get<double>() << '\n';
    } else {
      ana::Fit fit(options, module);
      if (const auto start = seed_from_values(options); !start.empty())
        apply_start_values(*fit.get_minimizer(), input_parameters, start);
      fit.minimize();
      result::write_results(fit, "BestFit");

      const auto min = fit.get_minimizer();
      if (!fit.converged() || !std::isfinite(min->MinValue()))
        throw std::runtime_error("The free fit did not converge; the pairs have no centre to scan around");

      std::vector<std::vector<double>> covariance(n_parameters, std::vector<double>(n_parameters));
      for (std::size_t i = 0; i < n_parameters; ++i)
        for (std::size_t j = 0; j < n_parameters; ++j)
          covariance[i][j] = min->CovMatrix(static_cast<unsigned int>(i), static_cast<unsigned int>(j));

      reference["llh"]        = min->MinValue();
      reference["parameters"] = std::vector<double>(min->X(), min->X() + n_parameters);
      reference["covariance"] = covariance;
      std::ofstream(reference_path) << reference.dump(2) << '\n';
    }

    const double llh        = reference["llh"].get<double>();
    const auto   centre     = reference["parameters"].get<std::vector<double>>();
    const auto   covariance = reference["covariance"].get<std::vector<std::vector<double>>>();
    if (centre.size() != n_parameters || covariance.size() != n_parameters)
      throw std::runtime_error(reference_path + " belongs to a config with a different number of parameters");

    // The Hesse errors scale every point's first steps, as in the other scans.
    std::vector<double> errors(n_parameters);
    for (std::size_t i = 0; i < n_parameters; ++i)
      errors[i] = std::sqrt(std::max(covariance[i][i], 0.0));

    scan::Settings settings;
    settings.coarse_cells_x = 8;
    settings.coarse_cells_y = 8;
    settings.max_depth      = 3;
    settings.region_depth   = 1;
    settings.levels         = {1.0};

    const int    half    = settings.lattice_x() / 2;
    const int    last    = settings.lattice_x();
    const double spacing = window / half;

    const scan::Node centre_node{half, half};

    // Delta given to a point outside the bounds: far above every level, so a cell
    // with such a corner is only refined where the contour runs into the bound.
    constexpr double kExcluded = 1.0e6;

    const std::filesystem::path top   = std::filesystem::current_path();
    const std::size_t           total = selected.size() * (selected.size() - 1) / 2;
    std::size_t                 done  = 0;

    for (std::size_t a = 0; a < selected.size(); ++a) {
      for (std::size_t b = a + 1; b < selected.size(); ++b) {
        const int         ia    = parameter_index(input_parameters, selected[a]);
        const int         ib    = parameter_index(input_parameters, selected[b]);
        const std::string label = selected[a] + "__" + selected[b];
        ++done;

        // Cholesky factor of the pair's Hesse block.
        const double l11 = std::sqrt(covariance[ia][ia]);
        const double l21 = covariance[ia][ib] / l11;
        const double l22 = std::sqrt(covariance[ib][ib] - l21 * l21);
        if (!(l11 > 0.0) || !(l22 > 0.0)) {
          std::cout << '[' << done << '/' << total << "] " << label << ": Hesse block not positive definite, skipped\n";
          continue;
        }

        auto whitened = [&](const scan::Node& node) { return std::pair{(node.x - half) * spacing, (node.y - half) * spacing}; };
        auto position = [&](const scan::Node& node) {
          const auto [u, v] = whitened(node);
          return std::pair{centre[ia] + l11 * u, centre[ib] + l21 * u + l22 * v};
        };
        auto within = [&](int i, double value) {
          const auto& parameter = input_parameters.parameters()[i];
          return (!parameter.lower_bound() || value >= *parameter.lower_bound()) && (!parameter.upper_bound() || value <= *parameter.upper_bound());
        };
        auto inside = [&](const scan::Node& node) {
          const auto [x, y] = position(node);
          return within(ia, x) && within(ib, y);
        };
        auto distance    = [](const scan::Node& p, const scan::Node& q) { return std::hypot(static_cast<double>(p.x - q.x), static_cast<double>(p.y - q.y)); };
        auto apply_fixed = [&](ROOT::Math::Minimizer& min, const scan::Node& node) {
          const auto [x, y] = position(node);
          min.SetVariableValue(ia, x);
          min.SetVariableValue(ib, y);
          min.FixVariable(ia);
          min.FixVariable(ib);
        };

        std::filesystem::create_directories(top / "pairs" / label);
        std::filesystem::current_path(top / "pairs" / label);

        nlohmann::json grid;
        grid["parameters"] = {selected[a], selected[b]};
        grid["centre"]     = {centre[ia], centre[ib]};
        grid["cholesky"]   = {{l11, 0.0}, {l21, l22}};
        grid["window"]     = window;
        grid["lattice"]    = last;
        grid["levels"]     = settings.levels;
        write_or_check_grid("scan_grid.json", grid, "pair scan", "Output_i_j");

        scanseed::Store<scan::Node> seeds;
        seeds.set_fallback(warm_start ? centre : seed_from_values(options));

        int  fits_performed = 0;
        auto evaluate       = [&](const std::vector<scan::Node>& nodes, scan::Surface& surface) {
          std::vector<scan::Node> to_fit;
          for (const auto& node : nodes) {
            if (node == centre_node)
              surface[node] = llh;  // The free fit itself.
            else if (!inside(node))
              surface[node] = llh + kExcluded;
            else
              to_fit.push_back(node);
          }
          // Outward from the free fit, so every point has a fitted neighbour to start from.
          std::ranges::sort(to_fit, {}, [&](const scan::Node& node) { return distance(node, centre_node); });
          run_scan_batch(std::move(to_fit), surface, options, module, input_parameters, format, warm_start, seeds, scan_workers, [](const scan::Node& node) { return node_name(node); }, apply_fixed, distance, fits_performed, errors);
        };

        const scan::Surface surface = scan::refine(settings, llh, evaluate, [](int, std::size_t, std::size_t, std::size_t) {});

        nlohmann::json points  = nlohmann::json::array();
        bool           at_edge = false;
        for (const auto& [node, value] : surface) {
          const auto [x, y]  = position(node);
          const bool usable  = inside(node) && std::isfinite(value);
          points.push_back({node.x, node.y, x, y, usable ? nlohmann::json(value - llh) : nlohmann::json(nullptr)});
          if (usable && value - llh < settings.levels.back() && (node.x == 0 || node.y == 0 || node.x == last || node.y == last))
            at_edge = true;
        }

        nlohmann::json out = grid;
        out["reference"]   = llh;
        out["atEdge"]      = at_edge;
        out["points"]      = points;  // [x node, y node, value of i, value of j, Delta or null]
        std::ofstream("surface.json") << out.dump() << '\n';

        std::filesystem::current_path(top);

        std::cout << '[' << done << '/' << total << "] " << label << ": " << surface.size() << " points, " << fits_performed << " fitted"
                  << (at_edge ? " -- Delta = 1 region reaches the window edge, rerun with a larger --scanWindow" : "") << '\n';
      }
    }
  }

}  // namespace llhfit
