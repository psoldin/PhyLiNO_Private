#include "ICWriteResultsProto.h"

#include "ICBlinding.h"
#include "ICComponentBreakdown.h"
#include "ic_result.pb.h"

// STL includes
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

// boost includes
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

namespace result::ic {

  namespace {

    result::ic::proto::FitResult build_proto_result(ana::Fit& fit, ana::ic::ICLikelihood& llh, const io::ic::ICInputOptions& info) {
      const auto  min   = fit.get_minimizer();
      const auto& names = fit.options()->inputOptions().input_parameters().names();
      const bool  blind = fit.options()->inputOptions().blind();

      // Every histogram read below -- prediction and per-component breakdown alike --
      // is whatever the last likelihood call left behind, so put the likelihood back
      // on the minimum first.
      evaluate_at_minimum(llh, min->X());

      result::ic::proto::FitResult msg;

      msg.set_converged(fit.converged());
      msg.set_chi2(min->MinValue());
      msg.set_edm(min->Edm());
      msg.set_fit_duration(fit.time_duration());

      const double* x   = min->X();
      const double* err = min->Errors();
      auto&         parameters = *msg.mutable_parameters();
      for (std::size_t i = 0; i < names.size(); ++i) {
        if (blind && is_blinded_parameter(names[i]))
          continue;
        auto& p = parameters[names[i]];
        p.set_value(x[i]);
        p.set_error(err[i]);
      }

      double data_total = 0.0;
      double pred_total = 0.0;

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

        auto* sample_msg = msg.add_samples();
        sample_msg->set_name(config.name);
        for (const auto& c : config.components) sample_msg->add_components(c);
        sample_msg->set_livetime(config.livetime);
        sample_msg->set_total_bins(config.binning.total_bins());
        sample_msg->set_data_total(sample_data_total);
        sample_msg->set_pred_total(sample_pred_total);

        for (const io::ic::Axis& axis : config.binning.axes()) {
          auto* axis_msg = sample_msg->add_axes();
          axis_msg->set_kind(std::string(io::ic::axis_kind_name(axis.kind)));
          axis_msg->set_low(axis.lo);
          axis_msg->set_high(axis.hi);
          axis_msg->set_n_bins(axis.n_bins);
        }

        *sample_msg->mutable_data()       = {data.begin(), data.end()};
        *sample_msg->mutable_prediction() = {predicted.begin(), predicted.end()};

        // Same component list and semantics as the JSON writer (see
        // component_breakdown() in ICComponentBreakdown.h).
        for (auto& [key, bins] : component_breakdown(sample, llh.parameter())) {
          if (blind) {
            if (is_blinded_component(key))
              continue;
            blind_bins(config.binning, bins);
          }
          auto* component_msg = sample_msg->add_components_breakdown();
          component_msg->set_name(key);
          component_msg->set_total(sum_of(bins));
          *component_msg->mutable_bins() = {bins.begin(), bins.end()};
        }
      }

      msg.set_data_total(data_total);
      msg.set_pred_total(pred_total);
      msg.set_likelihood((info.likelihood_type() == io::ic::LikelihoodType::SAY) ? "SAY" : "Poisson");

      return msg;
    }

  }  // namespace

  void write_ice_cube_results_protobuf(ana::Fit& fit, ana::ic::ICLikelihood& llh, const io::ic::ICInputOptions& info, std::string_view name) {
    const auto msg = build_proto_result(fit, llh, info);

    std::stringstream ss;
    ss << name << ".pb.gz";

    std::ofstream file(ss.str(), std::ios::binary);

    namespace bio = boost::iostreams;
    bio::filtering_ostream gzip_stream;
    gzip_stream.push(bio::gzip_compressor());
    gzip_stream.push(file);

    if (!msg.SerializeToOstream(&gzip_stream)) {
      throw std::runtime_error("Failed to serialize protobuf result to \"" + ss.str() + "\"");
    }
  }

  std::optional<ProtoStoredFit> read_ice_cube_results_protobuf(std::string_view name) {
    const std::filesystem::path path = std::string(name) + ".pb.gz";
    if (!std::filesystem::exists(path))
      return std::nullopt;

    std::ifstream file(path, std::ios::binary);

    namespace bio = boost::iostreams;
    bio::filtering_istream gzip_stream;
    gzip_stream.push(bio::gzip_decompressor());
    gzip_stream.push(file);

    result::ic::proto::FitResult msg;
    if (!msg.ParseFromIstream(&gzip_stream) || !std::isfinite(msg.chi2()))
      return std::nullopt;

    ProtoStoredFit result;
    result.converged = msg.converged();
    result.llh       = msg.chi2();
    for (const auto& [key, parameter] : msg.parameters())
      result.parameter_values[key] = parameter.value();

    return result;
  }

}  // namespace result::ic
