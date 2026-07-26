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

    // One block per enabled sample: its own binning, its binned data and the
    // prediction at the minimum, so the output is self-contained. Bin arrays are
    // row-major over the sample's axes (first axis outer, last axis inner).
    double data_total = 0.0;
    double pred_total = 0.0;

    j["samples"] = nlohmann::json::array();
    for (std::size_t s = 0; s < llh.n_samples(); ++s) {
      const auto& sample    = llh.sample(s);
      const auto& config    = sample.config();
      const auto  data      = sample.data();
      const auto  predicted = sample.predicted();

      double sample_data_total = 0.0;
      double sample_pred_total = 0.0;
      for (std::size_t b = 0; b < data.size(); ++b) {
        sample_data_total += data[b];
        sample_pred_total += predicted[b];
      }
      data_total += sample_data_total;
      pred_total += sample_pred_total;

      nlohmann::json axes = nlohmann::json::array();
      for (const io::ic::Axis& axis : config.binning.axes()) {
        axes.push_back({{"kind", io::ic::axis_kind_name(axis.kind)},
                        {"low", axis.lo},
                        {"high", axis.hi},
                        {"nBins", axis.n_bins}});
      }

      j["samples"].push_back({
          {"name", config.name},
          {"components", config.components},
          {"livetime", config.livetime},
          {"totalBins", config.binning.total_bins()},
          {"axes", std::move(axes)},
          {"data", std::vector<double>(data.begin(), data.end())},
          {"prediction", std::vector<double>(predicted.begin(), predicted.end())},
          {"dataTotal", sample_data_total},
          {"predTotal", sample_pred_total},
      });
    }

    // Summed over all samples (what the single-sample output used to report).
    j["dataTotal"]  = data_total;
    j["predTotal"]  = pred_total;
    j["likelihood"] = (info.likelihood_type() == io::ic::LikelihoodType::SAY) ? "SAY" : "Poisson";

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
