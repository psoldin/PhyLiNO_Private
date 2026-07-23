#pragma once

#include "../Likelihood.h"

#include "LinearRegression/LinRegInputOptions.h"

// STL includes
#include <span>
#include <vector>

namespace ana::linreg {

  /**
   * @brief Chi-square likelihood of a straight line y = a*x + b fitted to Asimov data.
   *
   * The data points are generated on construction from the truth values in the config file
   * ("LinearRegression" section) without noise, so the fit has to recover the truth exactly and
   * the chi-square at the minimum has to vanish. See params::linreg::Parameter for the parameter
   * order (slope, offset).
   */
  class LinRegLikelihood : public Likelihood {
   public:
    LinRegLikelihood(std::shared_ptr<io::Options> options, const io::linreg::LinRegInputOptions& input_options);

    ~LinRegLikelihood() override = default;

    [[nodiscard]] double calculate_likelihood(const double* parameter) override;

    /** The Asimov x positions the likelihood was built from. */
    [[nodiscard]] std::span<const double> x() const noexcept { return m_X; }

    /** The Asimov measurement values the likelihood was built from. */
    [[nodiscard]] std::span<const double> y() const noexcept { return m_Y; }

   private:
    std::vector<double> m_X;     /**< x positions of the generated data points. */
    std::vector<double> m_Y;     /**< Asimov measurement values. */
    double              m_Sigma; /**< Per-point Gaussian uncertainty. */
  };

}  // namespace ana::linreg
