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

void perform_2d_scan(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module) {

  constexpr int x_bins = 50;
  constexpr int y_bins = 50;
  constexpr double low_x = 1.0;
  constexpr double high_x = 3.0;
  constexpr double low_y = 2.0;
  constexpr double high_y = 3.0;

  using namespace ana::ic;
  using enum params::ic::General;

  for (int i = 0; i < x_bins; ++i) {
    const double x = low_x + static_cast<double>(i) * (high_x - low_x) / x_bins;
    for (int j = 0; j < y_bins; ++j) {
      const double y = low_y + static_cast<double>(j) * (high_y - low_y) / y_bins;

      ana::Fit fit(options, module);
      auto min = fit.get_minimizer();
      min->SetVariableValue(AstroNorm, x);
      min->SetVariableValue(SpectralIndex, y);
      min->FixVariable(AstroNorm);
      min->FixVariable(SpectralIndex);

      fit.minimize();

      std::stringstream ss;

      ss << "Output_" << i << '_' << j;

      result::write_results(fit, ss.str());
    }
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

    ana::Fit fit(options, module);

    fit.minimize();

    result::write_results(fit, "Output");

    // perform_2d_scan(options, module);

    std::cout << "####\t" << fit.get_minimizer()->X()[0] << '\n';
  } catch (const std::exception& e) {
    std::cout << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
