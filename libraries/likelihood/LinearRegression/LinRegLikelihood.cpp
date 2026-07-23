#include "LinRegLikelihood.h"

#include <stdexcept>

namespace ana::linreg {

  LinRegLikelihood::LinRegLikelihood(std::shared_ptr<io::Options> options, const io::linreg::LinRegInputOptions& input_options)
    : Likelihood(std::move(options), 2)
    , m_X()
    , m_Y()
    , m_Sigma(input_options.sigma()) {
    if (input_options.n_points() < 2) {
      throw std::invalid_argument("LinearRegression: NPoints has to be at least 2");
    }

    if (m_Sigma <= 0.0) {
      throw std::invalid_argument("LinearRegression: Sigma has to be positive");
    }

    m_X.resize(input_options.n_points());
    m_Y.resize(input_options.n_points());

    const double dx = (input_options.x_max() - input_options.x_min()) / static_cast<double>(m_X.size() - 1);

    // Asimov data: the measurement is exactly the model prediction at the truth values.
    for (std::size_t i = 0; i < m_X.size(); ++i) {
      m_X[i] = input_options.x_min() + static_cast<double>(i) * dx;
      m_Y[i] = input_options.truth_a() * m_X[i] + input_options.truth_b();
    }
  }

  double LinRegLikelihood::calculate_likelihood(const double* parameter) {
    m_Parameter.reset_parameter(parameter);

    const double a = m_Parameter[0];
    const double b = m_Parameter[1];

    double chi2 = 0.0;
    for (std::size_t i = 0; i < m_X.size(); ++i) {
      const double residual = (m_Y[i] - (a * m_X[i] + b)) / m_Sigma;
      chi2 += residual * residual;
    }
    return chi2;
  }

}  // namespace ana::linreg
