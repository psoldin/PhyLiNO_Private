#pragma once

namespace params::linreg {

  /**
   * @brief Fit parameters of the linear-regression example, in the order the minimizer sees them.
   *
   * The equivalent of params::dc::Detector for this experiment: the enum is the single source of
   * truth for parameter indices, so the likelihood, the module and the result writer agree.
   */
  enum Parameter : int {
    slope = 0,
    offset,
    _last_of_parameter_
  };

  /** Number of fit parameters of the linear-regression experiment. */
  constexpr int number_of_parameters() noexcept {
    return static_cast<int>(_last_of_parameter_);
  }

}  // namespace params::linreg
