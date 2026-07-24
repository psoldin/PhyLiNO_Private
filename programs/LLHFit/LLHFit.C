// STL includes
#include <iostream>

// includes
#include "DoubleChooz/DCExperimentModule.h"
#include "ExperimentModule.h"
#include "IceCube/ICModule.h"
#include "LinearRegression/LinRegModule.h"
#include "Fit.h"
#include "Options.h"
#include "write_results.h"

#include <TROOT.h>

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

    auto options = std::make_shared<io::Options>(argc, argv, ana::collect_input_options(modules));

    const auto module = modules.at(options->inputOptions().experiment());

    ana::Fit fit(options, module);

    fit.minimize();
    result::write_results(fit, "Output");

    std::cout << "####\t" << fit.get_minimizer()->X()[0] << '\n';
  } catch (const std::exception& e) {
    std::cout << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
