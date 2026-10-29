// STL includes
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

// includes
#include "DoubleChooz/DCExperimentModule.h"
#include "ExperimentModule.h"
#include "Fit.h"
#include "IceCube/ICModule.h"
#include "LinearRegression/LinRegModule.h"
#include "Options.h"
#include "write_results.h"

#include "IceCube/ICBlinding.h"
#include "ScanCommon.h"
#include "Scans.h"

#include <TROOT.h>

int main(int argc, char** argv) {
  ROOT::EnableThreadSafety();

  using namespace llhfit;

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

    // Read here rather than where it is used: a bad --seedFrom should fail now,
    // not once the MC is in memory. The result is cached for the real callers.
    seed_from_values(options);
    evaluate_at_values(options);

    const auto module = modules.at(options->inputOptions().experiment());

    // --evaluateAt writes "Output" for this config's model at another result's
    // parameters, with nothing minimized: how a best fit is carried over to a
    // differently built model to compare that model with its data.
    if (const auto& point = evaluate_at_values(options); !point.empty()) {
      ana::Fit fit(options, module);
      fit.evaluate(point);
      result::write_results(fit, "Output");
      return EXIT_SUCCESS;
    }

    // --fitOnly runs the plain fit and writes "Output.json"; without it the scan
    // picked by --scanMode (2d by default) is the entry point. The single fit
    // is what the NNMFit parity harness drives
    // (tools/nnmfit_oracle/compare_llh_value.py, run_fit_parity.sh), which needs
    // one converged result rather than a surface.
    if (options->inputOptions().fit_only()) {
      ana::Fit fit(options, module);
      auto     min = fit.get_minimizer();

      if (const auto start = seed_from_values(options); !start.empty())
        apply_start_values(*min, options->inputOptions().input_parameters(), start);

      fit.minimize();
      result::write_results(fit, "Output");
    } else {
      const std::string& scan_mode      = options->inputOptions().scan_mode();
      const std::string& scan_parameter = options->inputOptions().scan_parameter();
      const int          scan_points    = options->inputOptions().scan_points();

      if (options->inputOptions().blind() && (scan_mode == "1d" || scan_mode == "1d-regular" || scan_mode == "1d-crossing") &&
          result::ic::is_blinded_parameter(scan_parameter))
        throw std::invalid_argument("--blind does not scan the signal parameters; --scanParameter " + scan_parameter + " is one of them");

      if (scan_mode == "2d") {
        perform_2d_scan(options, module);
      } else if (scan_mode == "2d-regular") {
        perform_2d_scan_regular(options, module, scan_points, scan_points);
      } else if (scan_mode == "2d-stencil") {
        perform_2d_stencil(options, module);
      } else if (scan_mode == "2d-pairs") {
        perform_2d_pairs(options, module);
      } else if (scan_mode == "1d") {
        if (scan_parameter.empty()) {
          perform_1d_scan_all(options, module, false, scan_points);
        } else {
          reuse_sibling_free_fit("BestFit_", scan_parameter, free_parameter_names(options), options->inputOptions().input_parameters(), options->inputOptions().output_format());
          perform_1d_scan(options, module, scan_parameter);
        }
      } else if (scan_mode == "1d-crossing") {
        if (scan_parameter.empty()) {
          perform_1d_crossing_all(options, module);
        } else {
          reuse_sibling_free_fit("BestFit_", scan_parameter, free_parameter_names(options), options->inputOptions().input_parameters(), options->inputOptions().output_format());
          perform_1d_crossing(options, module, scan_parameter, parameter_index(options->inputOptions().input_parameters(), scan_parameter));
        }
      } else if (scan_mode == "nm1") {
        if (scan_parameter.empty())
          throw std::invalid_argument("--scanMode nm1 needs --scanParameter to say which parameter's sensitivity is being measured");
        perform_nm1_scan(options, module, scan_parameter);
      } else if (scan_mode == "1d-regular") {
        if (scan_parameter.empty())
          perform_1d_scan_all(options, module, true, scan_points);
        else
          perform_1d_scan_regular(options, module, scan_parameter, scan_points);
      } else {
        throw std::invalid_argument("Unknown --scanMode \"" + scan_mode + "\". Valid values: 2d, 2d-regular, 2d-stencil, 2d-pairs, 1d, 1d-regular, 1d-crossing, nm1");
      }
    }
  } catch (const std::exception& e) {
    std::cout << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
