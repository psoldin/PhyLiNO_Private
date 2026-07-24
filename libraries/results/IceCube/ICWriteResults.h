#pragma once

#include "Fit.h"

#include "IceCube/ICInputOptions.h"
#include "IceCube/ICLikelihood.h"
#include "IceCube/ICParameter.h"

#include <nlohmann/json.hpp>

// STL includes
#include <fstream>
#include <sstream>
#include <string_view>
#include <vector>

namespace result::ic {

  inline nlohmann::json get_json_file(ana::Fit& fit, const ana::ic::ICLikelihood& llh, const io::ic::ICInputOptions& info) {
    const auto  min   = fit.get_minimizer();
    const auto& names = fit.options()->inputOptions().input_parameters().names();

    nlohmann::json j;

    j["converged"]   = fit.converged();
    j["chi2"]        = min->MinValue();
    j["EDM"]         = min->Edm();
    j["fitDuration"] = fit.time_duration();

    // Fitted value and error of every parameter, keyed by its config name.
    const double* x   = min->X();
    const double* err = min->Errors();
    for (std::size_t i = 0; i < names.size(); ++i) {
      j["parameters"][names[i]] = {{"value", x[i]}, {"error", err[i]}};
    }

    // The binned Asimov data and the prediction at the minimum, so the output
    // is self-contained (row-major, reco-energy outer, cos-zenith inner).
    const auto data      = llh.data();
    const auto predicted = llh.predicted();

    double data_total = 0.0;
    double pred_total = 0.0;
    for (std::size_t b = 0; b < data.size(); ++b) {
      data_total += data[b];
      pred_total += predicted[b];
    }

    j["nEnergyBins"] = io::ic::Constants::nEnergyBins;
    j["nZenithBins"] = io::ic::Constants::nZenithBins;
    j["data"]        = std::vector<double>(data.begin(), data.end());
    j["prediction"]  = std::vector<double>(predicted.begin(), predicted.end());
    j["dataTotal"]   = data_total;
    j["predTotal"]   = pred_total;
    j["likelihood"]  = (info.likelihood_type() == io::ic::LikelihoodType::SAY) ? "SAY" : "Poisson";

    return j;
  }

  inline void write_ice_cube_results(ana::Fit& fit, const ana::ic::ICLikelihood& llh, const io::ic::ICInputOptions& info, std::string_view name) {
    auto j = get_json_file(fit, llh, info);

    std::stringstream ss;
    ss << name << ".json";
    std::ofstream file;
    file.open(ss.str());
    file << j.dump(2) << '\n';
    file.close();
  }

}  // namespace result::ic
