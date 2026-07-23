#pragma once

#include "Fit.h"

#include "LinearRegression/LinRegInputOptions.h"
#include "LinearRegression/LinRegLikelihood.h"
#include "LinearRegression/LinRegParameter.h"

#include <nlohmann/json.hpp>

// STL includes
#include <fstream>
#include <sstream>
#include <string_view>

namespace result::linreg {

  inline nlohmann::json get_json_file(ana::Fit& fit, const ana::linreg::LinRegLikelihood& llh, const io::linreg::LinRegInputOptions& info) {
    const auto min = fit.get_minimizer();

    nlohmann::json j;

    j["converged"]   = fit.converged();
    j["chi2"]        = min->MinValue();
    j["EDM"]         = min->Edm();
    j["fitDuration"] = fit.time_duration();

    j["a"]       = min->X()[params::linreg::slope];
    j["a_error"] = min->Errors()[params::linreg::slope];
    j["b"]       = min->X()[params::linreg::offset];
    j["b_error"] = min->Errors()[params::linreg::offset];

    // The Asimov data the fit was run against, so the output is self-contained.
    j["truth_a"] = info.truth_a();
    j["truth_b"] = info.truth_b();
    j["sigma"]   = info.sigma();
    j["x"]       = llh.x();
    j["y"]       = llh.y();

    return j;
  }

  inline void write_linear_regression_results(ana::Fit& fit, const ana::linreg::LinRegLikelihood& llh, const io::linreg::LinRegInputOptions& info, std::string_view name) {
    auto j = get_json_file(fit, llh, info);

    std::stringstream ss;
    ss << name << ".json";
    std::ofstream file;
    file.open(ss.str());
    file << j.dump(2) << '\n';
    file.close();
  }

}  // namespace result::linreg
