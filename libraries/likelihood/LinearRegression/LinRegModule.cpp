#include "LinRegModule.h"

#include "../Fit.h"

#include "LinearRegression/LinRegWriteResults.h"

#include <stdexcept>

namespace ana::linreg {

  void LinRegModule::write_results(Fit& fit, std::string_view name) {
    const auto likelihood = std::dynamic_pointer_cast<LinRegLikelihood>(fit.likelihood());
    if (likelihood == nullptr) {
      throw std::logic_error("LinRegModule::write_results called with a fit that holds no LinRegLikelihood");
    }
    result::linreg::write_linear_regression_results(fit, *likelihood, *m_InputOptions, name);
  }

}  // namespace ana::linreg
