#pragma once

#include "Fit.h"

// STL includes
#include <string_view>

namespace result {

  inline void write_results(ana::Fit& fit, std::string_view name) {
    fit.module()->write_results(fit, name);
  }

}  // namespace result
