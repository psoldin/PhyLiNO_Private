/**
 * How long one likelihood evaluation takes, which is what decides whether the
 * explorer's GUI can evaluate synchronously in the Qt event loop or needs a
 * worker thread behind it.
 *
 * Usage: PhyLiNOExplorerBench <config.json> [repeats]
 */

#include "ExplorerModel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <config.json> [repeats]\n";
    return EXIT_FAILURE;
  }

  try {
    const int repeats = argc > 2 ? std::stoi(argv[2]) : 20;

    const auto load_start = std::chrono::steady_clock::now();
    explorer::ExplorerModel model(argv[1]);
    const auto load_end = std::chrono::steady_clock::now();

    std::cout << "load: " << std::chrono::duration<double>(load_end - load_start).count() << " s\n"
              << "samples: " << model.sample_names().size()
              << ", parameters: " << model.parameters().size()
              << ", -2lnL: " << model.reference_likelihood() << "\n";

    // With UseData: false the data IS the Asimov expectation, so at the point it
    // was generated at the stack has to reproduce it bin for bin. This is the
    // check the design doc describes as reading off the plot -- the data points
    // sitting on top of the stack -- done as arithmetic instead, so it can
    // actually be run.
    //
    // At the ASIMOV point, not the start point: a config may seed the fit off
    // truth, and config_icecube_combined.json does (SpectralIndex starts at 2.4
    // with the data generated at 2.44), which leaves a residual of ~14% in the
    // highest bins and is not a bug.
    for (const auto& info : model.parameters())
      model.set(info.index, info.asimov);

    for (std::size_t sample = 0; sample < model.sample_names().size(); ++sample) {
      const auto marginalized = model.marginalize(sample, 0);

      double worst = 0.0;
      for (std::size_t b = 0; b < marginalized.data.size(); ++b) {
        double stacked = 0.0;
        for (const auto& component : marginalized.components)
          stacked += component.values[b];

        // systematicsDelta is not in the stack, so the residual is only zero at
        // the nominal detector parameters -- which is where the start point is.
        const double scale = std::max(std::abs(marginalized.data[b]), 1.0);
        worst = std::max(worst, std::abs(stacked - marginalized.data[b]) / scale);
      }

      std::cout << "asimov check, sample " << model.sample_names()[sample]
                << ": worst relative stack-vs-data residual " << worst << '\n';
    }

    std::cout << "-2lnL at the asimov point: " << model.evaluate() << '\n';

    for (const auto& info : model.parameters())
      model.set(info.index, info.value);

    // Per parameter, not just one: a norm only rescales a cached histogram
    // while a spectral index forces every event to be reweighted, and the
    // slider that matters for responsiveness is the slowest one.
    std::cout << "\nper-parameter evaluate + marginalize, median of " << repeats << ":\n";

    double worst_total = 0.0;
    std::string worst_name;

    for (const auto& info : model.parameters()) {
      std::vector<double> eval_ms;
      std::vector<double> marg_ms;
      eval_ms.reserve(static_cast<std::size_t>(repeats));
      marg_ms.reserve(static_cast<std::size_t>(repeats));

      for (int i = 0; i < repeats; ++i) {
        // A step that stays inside the slider range, and is never zero even for
        // a parameter whose start value is.
        const double span = info.hi - info.lo;
        model.set(info.index, info.lo + span * (0.4 + 0.2 * i / std::max(1, repeats - 1)));

        const auto t0 = std::chrono::steady_clock::now();
        static_cast<void>(model.evaluate());
        const auto t1 = std::chrono::steady_clock::now();
        static_cast<void>(model.marginalize(0, 0));
        const auto t2 = std::chrono::steady_clock::now();

        eval_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        marg_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
      }

      model.set(info.index, info.value);

      std::sort(eval_ms.begin(), eval_ms.end());
      std::sort(marg_ms.begin(), marg_ms.end());

      const double eval = eval_ms[eval_ms.size() / 2];
      const double marg = marg_ms[marg_ms.size() / 2];

      std::cout << "  " << info.name << (info.fixed ? " [fixed]" : "")
                << "  eval " << eval << " ms, marginalize " << marg << " ms\n";

      if (eval + marg > worst_total) {
        worst_total = eval + marg;
        worst_name  = info.name;
      }
    }

    std::cout << "\nworst slider: " << worst_name << " at " << worst_total << " ms per move\n";

  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
