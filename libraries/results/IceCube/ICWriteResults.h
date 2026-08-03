#pragma once

#include "Fit.h"

#include "IceCube/ICInputOptions.h"
#include "IceCube/ICLikelihood.h"
#include "IceCube/ICParameter.h"

#include "ICWriteResultsProto.h"

#include <nlohmann/json.hpp>

// STL includes
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace result::ic {

  inline nlohmann::json get_json_file(ana::Fit& fit, const ana::ic::ICLikelihood& llh, const io::ic::ICInputOptions& info) {
    const auto  min   = fit.get_minimizer();
    const auto& names = fit.options()->inputOptions().input_parameters().names();

    nlohmann::json j;

    j["converged"]   = fit.converged();
    j["LLH"]         = min->MinValue();
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

      // Per-component breakdown, both summed and per-bin: a mis-scaled template or
      // gradient is visible here instead of hidden inside the sample total. Every
      // component is reported in the sample's ANALYSIS binning -- the per-event and
      // 2D-template components are spread over the RA axis exactly as the prediction
      // is -- so an external per-bin diff (tools/nnmfit_oracle/compare_to_nnmfit.py)
      // needs no reshaping. The atmospheric component is keyed by whichever variant
      // the sample declared, so the diff can tell a veto-reweighted sample from a
      // plain one.
      auto sum_of = [](const std::vector<double>& values) {
        double total = 0.0;
        for (const double v : values)
          total += v;
        return total;
      };
      const std::string atmo_key = config.wants_veto() ? "atmospheric_veto" : "atmospheric";

      const std::vector<double> astro_bins    = sample.in_analysis_bins(sample.astro_histogram());
      const std::vector<double> atmo_bins     = sample.in_analysis_bins(sample.atmospheric_histogram());
      const std::vector<double> template_bins = sample.in_analysis_bins(sample.template_histogram());
      const std::vector<double> systematics_bins =
          sample.in_analysis_bins(sample.systematics_mu_delta());
      const std::vector<double> galactic_bins = sample.in_analysis_bins(sample.galactic_histogram());

      nlohmann::json component_totals = {
          {"astro", sum_of(astro_bins)},
          {atmo_key, sum_of(atmo_bins)},
          {"template", sum_of(template_bins)},
          {"systematicsDelta", sum_of(systematics_bins)},
          {"galactic", sum_of(galactic_bins)},
      };

      nlohmann::json component_bins = {
          {"astro", astro_bins},
          {atmo_key, atmo_bins},
          {"template", template_bins},
          {"systematicsDelta", systematics_bins},
          {"galactic", galactic_bins},
      };

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

  inline void write_ice_cube_results_json(ana::Fit& fit, const ana::ic::ICLikelihood& llh, const io::ic::ICInputOptions& info, std::string_view name) {
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
  inline void write_ice_cube_results(ana::Fit& fit, const ana::ic::ICLikelihood& llh, const io::ic::ICInputOptions& info, std::string_view name) {
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
