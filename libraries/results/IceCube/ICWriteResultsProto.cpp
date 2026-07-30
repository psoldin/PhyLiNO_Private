#include "ICWriteResultsProto.h"

#include "ic_result.pb.h"

// STL includes
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

// boost includes
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

namespace result::ic {

  namespace {

    result::ic::proto::FitResult build_proto_result(ana::Fit& fit, const ana::ic::ICLikelihood& llh, const io::ic::ICInputOptions& info) {
      const auto  min   = fit.get_minimizer();
      const auto& names = fit.options()->inputOptions().input_parameters().names();

      result::ic::proto::FitResult msg;

      msg.set_converged(fit.converged());
      msg.set_chi2(min->MinValue());
      msg.set_edm(min->Edm());
      msg.set_fit_duration(fit.time_duration());

      const double* x   = min->X();
      const double* err = min->Errors();
      auto&         parameters = *msg.mutable_parameters();
      for (std::size_t i = 0; i < names.size(); ++i) {
        auto& p = parameters[names[i]];
        p.set_value(x[i]);
        p.set_error(err[i]);
      }

      double data_total = 0.0;
      double pred_total = 0.0;

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

        auto sum_of = [](const std::vector<double>& values) {
          double total = 0.0;
          for (const double v : values) total += v;
          return total;
        };
        const std::string atmo_key = config.wants_veto() ? "atmospheric_veto" : "atmospheric";

        const std::vector<double> astro_bins       = sample.in_analysis_bins(sample.astro_histogram());
        const std::vector<double> atmo_bins        = sample.in_analysis_bins(sample.atmospheric_histogram());
        const std::vector<double> template_bins    = sample.in_analysis_bins(sample.template_histogram());
        const std::vector<double> systematics_bins = sample.in_analysis_bins(sample.systematics_mu_delta());
        const std::vector<double> galactic_bins    = sample.in_analysis_bins(sample.galactic_histogram());

        const auto add_component = [&](const std::string& name, const std::vector<double>& bins) {
          auto* component_msg = sample_msg->add_components_breakdown();
          component_msg->set_name(name);
          component_msg->set_total(sum_of(bins));
          *component_msg->mutable_bins() = {bins.begin(), bins.end()};
        };
        add_component("astro", astro_bins);
        add_component(atmo_key, atmo_bins);
        add_component("template", template_bins);
        add_component("systematicsDelta", systematics_bins);
        add_component("galactic", galactic_bins);
      }

      msg.set_data_total(data_total);
      msg.set_pred_total(pred_total);
      msg.set_likelihood((info.likelihood_type() == io::ic::LikelihoodType::SAY) ? "SAY" : "Poisson");

      return msg;
    }

  }  // namespace

  void write_ice_cube_results_protobuf(ana::Fit& fit, const ana::ic::ICLikelihood& llh, const io::ic::ICInputOptions& info, std::string_view name) {
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

}  // namespace result::ic
