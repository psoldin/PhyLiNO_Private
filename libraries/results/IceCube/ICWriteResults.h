#pragma once

#include "Fit.h"

#include "IceCube/ICInputOptions.h"
#include "IceCube/ICLikelihood.h"
#include "IceCube/ICParameter.h"

#include "ICBlinding.h"
#include "ICComponentBreakdown.h"
#include "ICWriteResultsProto.h"

#include <nlohmann/json.hpp>

// STL includes
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace result::ic {

  inline nlohmann::json get_json_file(ana::Fit& fit, ana::ic::ICLikelihood& llh, const io::ic::ICInputOptions& info) {
    const auto  min   = fit.get_minimizer();
    const auto& names = fit.options()->inputOptions().input_parameters().names();
    const bool  blind = fit.options()->inputOptions().blind();

    // Every histogram read below -- prediction and per-component breakdown alike --
    // is whatever the last likelihood call left behind, so put the likelihood back
    // on the minimum first.
    evaluate_at_minimum(llh, min->X());

    nlohmann::json j;

    j["converged"]   = fit.converged();
    j["LLH"]         = min->MinValue();
    j["EDM"]         = min->Edm();
    j["fitDuration"] = fit.time_duration();

    // Fitted value and error of every parameter, keyed by its config name.
    const double* x   = min->X();
    const double* err = min->Errors();
    for (std::size_t i = 0; i < names.size(); ++i) {
      if (blind && is_blinded_parameter(names[i]))
        continue;
      j["parameters"][names[i]] = {{"value", x[i]}, {"error", err[i]}};
    }

    // One block per enabled sample: its own binning, its binned data and the
    // prediction at the minimum, so the output is self-contained. Bin arrays are
    // row-major over the sample's axes (first axis outer, last axis inner).
    double data_total = 0.0;
    double pred_total = 0.0;

    j["samples"] = nlohmann::json::array();
    for (std::size_t s = 0; s < llh.n_samples(); ++s) {
      const auto& sample = llh.sample(s);
      const auto& config = sample.config();

      // Copied out of the likelihood before blinding: the histograms below are the
      // ones the fit is still holding, so they are zeroed here and not in place.
      std::vector<double> data(sample.data().begin(), sample.data().end());
      std::vector<double> predicted(sample.predicted().begin(), sample.predicted().end());
      if (blind) {
        blind_bins(config.binning, data);
        blind_bins(config.binning, predicted);
      }

      // Summed after blinding, so a blinded total cannot be differenced against an
      // unblinded one to recover the hidden bins.
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

      // Per-component breakdown, both summed and per-bin (see component_breakdown()
      // in ICComponentBreakdown.h for the component list and what each key means).
      nlohmann::json component_totals = nlohmann::json::object();
      nlohmann::json component_bins   = nlohmann::json::object();
      for (auto& [key, bins] : component_breakdown(sample, llh.parameter())) {
        if (blind) {
          if (is_blinded_component(key))
            continue;
          blind_bins(config.binning, bins);
        }
        component_totals[key] = sum_of(bins);
        component_bins[key]   = std::move(bins);
      }

      j["samples"].push_back({
          {"name", config.name},
          {"components", config.components},
          {"livetime", config.livetime},
          {"totalBins", config.binning.total_bins()},
          {"axes", std::move(axes)},
          {"data", std::move(data)},
          {"prediction", std::move(predicted)},
          {"dataTotal", sample_data_total},
          {"predTotal", sample_pred_total},
          {"componentTotals", std::move(component_totals)},
          {"componentBins", std::move(component_bins)},
      });
    }

    // Summed over all samples (what the single-sample output used to report).
    j["dataTotal"]  = data_total;
    j["predTotal"]  = pred_total;
    j["likelihood"] = (info.likelihood_type() == io::ic::LikelihoodType::SAY) ? "SAY" : "Poisson";

    return j;
  }

  inline void write_ice_cube_results_json(ana::Fit& fit, ana::ic::ICLikelihood& llh, const io::ic::ICInputOptions& info, std::string_view name) {
    auto j = get_json_file(fit, llh, info);

    std::stringstream ss;
    ss << name << ".json";
    std::ofstream file;
    file.open(ss.str());
    file << j.dump(2) << '\n';
    file.close();
  }

  // Dispatches on the global "--output-format" option (default "json"). The
  // protobuf format is the same content, binary-encoded and gzip-compressed,
  // for the multi-thousand-file production runs where the pretty-printed JSON
  // (one 3D-binned sample alone can be tens of MB) is not practical to store.
  inline void write_ice_cube_results(ana::Fit& fit, ana::ic::ICLikelihood& llh, const io::ic::ICInputOptions& info, std::string_view name) {
    const auto& format = fit.options()->inputOptions().output_format();

    if (format == "json") {
      write_ice_cube_results_json(fit, llh, info, name);
    } else if (format == "protobuf") {
      write_ice_cube_results_protobuf(fit, llh, info, name);
    } else {
      throw std::invalid_argument("Unknown --output-format \"" + format + "\" (expected \"json\" or \"protobuf\")");
    }
  }

}  // namespace result::ic
