#include "LinRegModule.h"

#include "../Fit.h"

#include "LinearRegression/LinRegWriteResults.h"

#include <stdexcept>

namespace ana::linreg {

  void LinRegModule::write_results(Fit& fit, std::string_view name) {
    if (m_Likelihood == nullptr) {
      throw std::logic_error("LinRegModule::write_results called before create_likelihood");
    }
    result::linreg::write_linear_regression_results(fit, *m_Likelihood, *m_InputOptions, name);
  }

}  // namespace ana::linreg
