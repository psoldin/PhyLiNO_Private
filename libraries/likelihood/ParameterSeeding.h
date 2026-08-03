#pragma once

#include <random>

namespace ana {

  /**
   * Randomized minimizer start value for one parameter, the counterpart of
   * NNMFit's `NNMFitter._setup_minimizer_seeds(randomize_param_seeds=True)`
   * (NNMFit/core/nnm_fitter.py:270-286), which `run_fit.py` applies by default.
   *
   * NNMFit draws a multiplicative factor, `np.random.normal(1, w) * value`, so a
   * start value of exactly 0 can never move -- in the IceCube configs that
   * freezes BarrH/W/Y/Z, CRGrad, DeltaGamma and VetoThreshold, the very
   * parameters a randomized start is meant to probe. This keeps the
   * multiplicative draw where it is meaningful and falls back to an additive
   * one of width `step` (the parameter's configured StepWidth) at zero.
   *
   * NNMFit's other branch (sigma = 0.25 * (hi - lo) for bounded parameters) has
   * no counterpart here: this framework has no parameter bounds.
   *
   * @param value  the parameter's configured start value
   * @param step   the parameter's configured step width, used only when value == 0
   * @param width  relative width of the multiplicative draw (NNMFit's 0.08)
   */
  template <typename Rng>
  [[nodiscard]] double randomized_start_value(const double value, const double step,
                                              const double width, Rng& rng) {
    if (value == 0.0) {
      std::normal_distribution<double> additive(0.0, step);
      return additive(rng);
    }

    std::normal_distribution<double> factor(1.0, width);
    return value * factor(rng);
  }

}  // namespace ana
