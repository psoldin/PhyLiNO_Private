// Compares a likelihood's analytic gradient and Hessian with finite differences
// of the likelihood itself, on the real configuration: same config file, same
// backend, same options as LLHFit. Meant for validating a new backend or a new
// configuration before trusting fits that use the gradient (--gradient).
//
//   GradientCheck -c config.json [--seedFrom Output.json] [--hessianCheck]
//
// The point is the config's start values, or the parameters of the result named
// by --seedFrom. Fixed parameters are skipped.

#include "DoubleChooz/DCExperimentModule.h"
#include "ExperimentModule.h"
#include "Fit.h"
#include "IceCube/ICModule.h"
#include "LinearRegression/LinRegModule.h"
#include "Options.h"

#include <nlohmann/json.hpp>

#include <TROOT.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

  /// Start values: the config's, overwritten by a result file's where it has them.
  std::vector<double> start_point(const io::InputParameter& input, const std::string& result) {
    std::vector<double> x(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) x[i] = input.value(static_cast<int>(i));
    if (result.empty()) return x;

    std::string path = result;
    if (!path.ends_with(".json")) path += ".json";
    std::ifstream  file(path);
    nlohmann::json stored = nlohmann::json::parse(file, nullptr, false);
    if (stored.is_discarded() || !stored.contains("parameters"))
      throw std::runtime_error("GradientCheck: cannot read parameters from " + path);
    const auto& names = input.names();
    for (std::size_t i = 0; i < names.size(); ++i)
      if (stored["parameters"].contains(names[i])) x[i] = stored["parameters"][names[i]]["value"].get<double>();
    std::cout << "Point: " << path << '\n';
    return x;
  }

}  // namespace

int main(int argc, char** argv) {
  ROOT::EnableThreadSafety();
  try {
    ana::module_map_t modules;
    for (const std::shared_ptr<ana::ExperimentModule> m :
         {std::shared_ptr<ana::ExperimentModule>(std::make_shared<ana::dc::DCExperimentModule>()),
          std::shared_ptr<ana::ExperimentModule>(std::make_shared<ana::linreg::LinRegModule>()),
          std::shared_ptr<ana::ExperimentModule>(std::make_shared<ana::ic::ICExperimentModule>())})
      modules[m->name()] = m;

    const bool hessian_check = std::ranges::any_of(std::vector<std::string>(argv + 1, argv + argc),
                                                   [](const std::string& a) { return a == "--hessianCheck"; });
    std::vector<char*> args;
    for (int i = 0; i < argc; ++i)
      if (std::string(argv[i]) != "--hessianCheck") args.push_back(argv[i]);

    const auto options = std::make_shared<io::Options>(static_cast<int>(args.size()), args.data(),
                                                       ana::collect_input_options(modules));
    const auto module  = modules.at(options->inputOptions().experiment());

    ana::Fit    fit(options, module);
    const auto& likelihood = fit.likelihood();
    if (!likelihood->has_gradient()) {
      std::cout << "The " << module->name() << " likelihood has no analytic gradient\n";
      return EXIT_FAILURE;
    }

    const auto&               input = options->inputOptions().input_parameters();
    const auto&               names = input.names();
    const std::size_t         n     = input.size();
    const std::vector<double> x     = start_point(input, options->inputOptions().seed_from());

    std::vector<double> gradient(n);
    likelihood->calculate_gradient(x.data(), gradient.data());
    const double value = likelihood->calculate_likelihood(x.data());
    std::printf("-2lnL = %.10f\n\n", value);

    // Central differences at h and h/2: their disagreement estimates the
    // finite-difference error itself, so a mismatch can be told apart from it.
    auto central = [&](std::size_t i, double h) {
      std::vector<double> p = x;
      p[i]                  = x[i] + h;
      const double up       = likelihood->calculate_likelihood(p.data());
      p[i]                  = x[i] - h;
      const double down     = likelihood->calculate_likelihood(p.data());
      return (up - down) / (2.0 * h);
    };

    std::printf("%-16s %16s %16s %10s %10s\n", "parameter", "analytic", "numerical", "rel.diff", "fd.noise");
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      if (input.fixed(static_cast<int>(i))) continue;
      const double h      = 1e-3 * input.parameters()[i].uncertainty();
      const double fd     = central(i, h);
      const double fd2    = central(i, 0.5 * h);
      const double richer = (4.0 * fd2 - fd) / 3.0;  // Richardson: O(h^4)
      const double scale  = std::max({std::fabs(richer), std::fabs(gradient[i]), 1e-12});
      const double rel    = std::fabs(gradient[i] - richer) / scale;
      const double noise  = std::fabs(fd - fd2) / scale;
      worst               = std::max(worst, rel);
      std::printf("%-16s %16.8e %16.8e %10.2e %10.2e\n", names[i].c_str(), gradient[i], richer, rel, noise);
    }
    std::printf("\nlargest relative difference: %.2e\n", worst);

    if (hessian_check) {
      std::vector<double> gn(n * n);
      likelihood->calculate_hessian(x.data(), gn.data());
      std::printf("\n%-16s %16s %16s %8s\n", "parameter", "Gauss-Newton", "exact (fd grad)", "ratio");
      std::vector<double> up(n), down(n), p = x;
      for (std::size_t i = 0; i < n; ++i) {
        if (input.fixed(static_cast<int>(i))) continue;
        const double h = 1e-3 * input.parameters()[i].uncertainty();
        p[i]           = x[i] + h;
        likelihood->calculate_gradient(p.data(), up.data());
        p[i] = x[i] - h;
        likelihood->calculate_gradient(p.data(), down.data());
        p[i]               = x[i];
        const double exact = (up[i] - down[i]) / (2.0 * h);
        std::printf("%-16s %16.8e %16.8e %8.4f\n", names[i].c_str(), gn[i * n + i], exact, gn[i * n + i] / exact);
      }
    }
  } catch (const std::exception& e) {
    std::cout << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
