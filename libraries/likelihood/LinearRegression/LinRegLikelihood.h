#pragma once

#include "../Likelihood.h"
#include "LinRegInputOptions.h"

// STL includes
#include <vector>

namespace ana::linreg {

  /**
   * @brief Chi-square likelihood of a straight line y = a*x + b fitted to Asimov data.
   *
   * The data points are generated on construction from the truth values in the config file
   * ("LinearRegression" section) without noise, so the fit has to recover the truth exactly and
   * the chi-square at the minimum has to vanish. Parameter 0 is the slope a, parameter 1 the
   * offset b.
   */
  class LinRegLikelihood : public Likelihood {
   public:
    LinRegLikelihood(std::shared_ptr<io::Options> options, const io::linreg::LinRegInputOptions& input_options);

    ~LinRegLikelihood() override = default;

    [[nodiscard]] double calculate_likelihood(const double* parameter) override;

   private:
    std::vector<double> m_X;     /**< x positions of the generated data points. */
    std::vector<double> m_Y;     /**< Asimov measurement values. */
    double              m_Sigma; /**< Per-point Gaussian uncertainty. */
  };

}  // namespace ana::linreg
