#include "LinRegModule.h"

#include "../Fit.h"

#include <nlohmann/json.hpp>

// STL includes
#include <fstream>
#include <string>

namespace ana::linreg {

  void LinRegModule::write_results(Fit& fit, std::string_view name) {
    const auto min = fit.get_minimizer();

    nlohmann::json j;
    j["converged"] = fit.converged();
    j["chi2"]      = min->MinValue();
    j["EDM"]       = min->Edm();
    j["a"]         = min->X()[0];
    j["a_error"]   = min->Errors()[0];
    j["b"]         = min->X()[1];
    j["b_error"]   = min->Errors()[1];

    std::ofstream file(std::string(name) + ".json");
    file << j.dump(2) << '\n';
  }

}  // namespace ana::linreg
