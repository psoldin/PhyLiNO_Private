/**
 * How much of the fitted parameter's uncertainty comes from finite MC, with and
 * without forward folding -- including the bin-to-bin correlations that folding
 * creates and that SAY's per-bin treatment cannot see.
 *
 * Propagate MC noise to the fitted parameter to first order. With the data held
 * at the nominal expectation and the model's bin contents perturbed by delta_b,
 * the stationarity condition of the Poisson likelihood gives
 *
 *     delta_gamma = -( sum_b g_b delta_b ) / I,
 *     g_b         = (d mu_b / d gamma) / mu_b,
 *     I           = sum_b (d mu_b / d gamma)^2 / mu_b   ( = 1 / sigma_stat^2 )
 *
 * so that Var(gamma)_MC = sum_bb' g_b g_b' Cov(delta_b, delta_b') / I^2.
 *
 * Resampling the MC (event multiplicities Poisson(1)) makes
 * Cov(delta_b, delta_b') = sum_i w_i^2 f_ib f_ib', and the double sum over bins
 * collapses to one pass over the response matrix:
 *
 *     sum_bb' g_b g_b' sum_i w_i^2 f_ib f_ib'  =  sum_i w_i^2 ( sum_b g_b f_ib )^2
 *
 * No bootstrap ensemble is needed -- this IS the ensemble variance, every
 * correlation included, computed exactly rather than sampled.
 *
 * Reported as V / I = (sigma_MC / sigma_stat)^2:
 *
 *   unfolded      f_ib is a delta at the event's own bin, so the covariance is
 *                 diagonal and SAY's assumption is exact there.
 *   folded, full  the honest folded answer, off-diagonal terms and all.
 *   folded, diag  what SAY actually uses. The gap between it and "full" is by
 *                 how much a folded fit overstates its own improvement.
 */
#include "ICDataBase.h"
#include "ICSample.h"
#include "SampleConfig.h"
#include "SampleLikelihood.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace {

  /** Config parameter names to their fit indices, for the ones a tracks fit moves. */
  const std::map<std::string, std::size_t>& parameter_indices() {
    using namespace params::ic;
    static const std::map<std::string, std::size_t> map{
        {"AstroNorm", AstroNorm},         {"SpectralIndex", SpectralIndex},
        {"ConvNorm", ConvNorm},           {"PromptNorm", PromptNorm},
        {"BarrH", BarrH},                 {"BarrW", BarrW},
        {"BarrY", BarrY},                 {"BarrZ", BarrZ},
        {"CRGrad", CRGrad},               {"DeltaGamma", DeltaGamma},
        {"MuonNorm", MuonNorm},           {"MuonGunNorm", MuonGunNorm},
        {"VetoThreshold", VetoThreshold}, {"DOMEff", DOMEff},
        {"IceAbs", IceAbs},               {"IceScat", IceScat},
        {"HoleIceP0", HoleIceP0},         {"HoleIceP1", HoleIceP1},
        {"GalacticNorm0", GalacticNorm0}, {"GalacticNorm1", GalacticNorm1},
        {"AstroGamma1", AstroGamma1},     {"AstroGamma2", AstroGamma2},
        {"AstroEBreak", AstroEBreak},
    };
    return map;
  }

  std::vector<double> prediction_of(const ana::ic::SampleLikelihood& sample, const std::size_t n_bins) {
    const std::span<const double> astro = sample.astro_histogram();
    const std::span<const double> atmo  = sample.atmospheric_histogram();

    std::vector<double> mu(n_bins, 0.0);
    for (std::size_t b = 0; b < n_bins; ++b) {
      if (!astro.empty()) mu[b] += astro[b];
      if (!atmo.empty()) mu[b] += atmo[b];
    }
    return mu;
  }

}  // namespace

int main(int argc, char** argv) {
  std::string config;
  std::string scanned = "SpectralIndex";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-c" && i + 1 < argc) config = argv[++i];
    else if (arg == "-p" && i + 1 < argc) scanned = argv[++i];
  }
  if (config.empty()) {
    std::puts("usage: McVariance -c config.json [-p ParameterName]");
    return 1;
  }

  boost::property_tree::ptree tree;
  boost::property_tree::read_json(config, tree);
  const auto                              ic      = tree.get_child("IceCube");
  const std::vector<io::ic::SampleConfig> samples = io::ic::parse_samples(ic);
  const io::ic::ICDataBase                database(samples);

  const ana::ic::GlobalFluxSettings settings{
      .e_ref_gev                = ic.get<double>("ERefGeV", 1.0e5),
      .astro_reference_index    = ic.get<double>("AstroReferenceIndex", 2.0),
      .conv_delta_gamma_e_ref   = ic.get<double>("ConvDeltaGammaERef", 1.0e3),
      .prompt_delta_gamma_e_ref = ic.get<double>("PromptDeltaGammaERef", 3.8e3),
      .astro_per_type_norm      = ic.get<bool>("AstroPerTypeNorm", false),
      .veto_anchor_energy       = ic.get<double>("VetoAnchorEnergy", 100.0),
      .veto_rescale_energy      = ic.get<double>("VetoRescaleEnergy", 100.0),
      .astro_model              = io::ic::AstroModel::Powerlaw,
      .use_multi_threading      = true,
  };

  // Evaluate at the Asimov point, which is where the reported errors are taken.
  std::vector<double> values(params::ic::number_of_parameters(), 0.0);
  std::size_t         scanned_index = params::ic::SpectralIndex;
  for (const auto& [ignored, node] : tree.get_child("Parameter")) {
    (void)ignored;
    const std::string name = node.get<std::string>("Name");
    const auto        it   = parameter_indices().find(name);
    if (it == parameter_indices().end()) continue;
    values[it->second] = node.get<double>("AsimovValue", node.get<double>("StartValue"));
    if (name == scanned) scanned_index = it->second;
  }

  const auto enabled = io::ic::enabled_sample_indices(samples);
  for (std::size_t k = 0; k < database.n_samples(); ++k) {
    const io::ic::SampleConfig& cfg    = samples[enabled[k]];
    const io::ic::ICSample&     sample = database.sample(k);
    if (sample.response.empty()) {
      std::printf("sample '%s': no response matrix; set Response.enabled\n", cfg.name.c_str());
      continue;
    }

    ana::ic::SampleLikelihood likelihood(sample, cfg, settings, /*gpu=*/nullptr, /*use_say=*/false);

    const auto n_bins = static_cast<std::size_t>(cfg.mc_binning.total_bins());

    ana::ParameterWrapper parameter(params::ic::number_of_parameters());
    parameter.reset_parameter(values.data());
    likelihood.generate_asimov(parameter);

    const std::vector<double> mu = prediction_of(likelihood, n_bins);
    const std::vector<double> w  = likelihood.per_event_weight();

    // d mu / d gamma by a central difference: the flux models are smooth in it
    // and this avoids hand-differentiating four of them.
    constexpr double step = 1.0e-3;
    std::vector<double> shifted = values;

    shifted[scanned_index] = values[scanned_index] + step;
    parameter.reset_parameter(shifted.data());
    likelihood.partial_llh(parameter);
    const std::vector<double> mu_up = prediction_of(likelihood, n_bins);

    shifted[scanned_index] = values[scanned_index] - step;
    parameter.reset_parameter(shifted.data());
    likelihood.partial_llh(parameter);
    const std::vector<double> mu_down = prediction_of(likelihood, n_bins);

    std::vector<double> g(n_bins, 0.0);
    double              fisher = 0.0;
    for (std::size_t b = 0; b < n_bins; ++b) {
      if (!(mu[b] > 0.0)) continue;
      const double d = (mu_up[b] - mu_down[b]) / (2.0 * step);
      g[b]           = d / mu[b];
      fisher += d * d / mu[b];
    }

    const io::ic::ResponseMatrix& response = sample.response;

    // s_i = sum_b g_b f_ib, accumulated by walking the bin-major matrix once.
    std::vector<double> s(sample.size(), 0.0);
    double              v_folded_diag = 0.0;
    for (std::size_t b = 0; b < n_bins; ++b)
      for (std::size_t k = response.bin_offsets[b]; k < response.bin_offsets[b + 1]; ++k) {
        const auto   i = static_cast<std::size_t>(response.events[k]);
        const double f = response.fractions[k];
        s[i] += g[b] * f;
        v_folded_diag += g[b] * g[b] * w[i] * w[i] * f * f;
      }

    double v_folded_full = 0.0;
    double v_unfolded    = 0.0;
    for (std::size_t i = 0; i < sample.size(); ++i) {
      v_folded_full += w[i] * w[i] * s[i] * s[i];
      const double g_own = g[static_cast<std::size_t>(sample.bin_idx[i])];
      v_unfolded += w[i] * w[i] * g_own * g_own;
    }

    auto ratio = [fisher](const double v) { return v / fisher; };
    std::printf("\nsample '%s', parameter %s\n", cfg.name.c_str(), scanned.c_str());
    std::printf("  sigma_stat                     %.5f\n", 1.0 / std::sqrt(fisher));
    std::printf("  (sigma_MC/sigma_stat)^2\n");
    std::printf("    unfolded                     %.5f   -> total error x %.4f\n", ratio(v_unfolded),
                std::sqrt(1.0 + ratio(v_unfolded)));
    std::printf("    folded, diagonal only (SAY)  %.5f   -> total error x %.4f\n", ratio(v_folded_diag),
                std::sqrt(1.0 + ratio(v_folded_diag)));
    std::printf("    folded, full covariance      %.5f   -> total error x %.4f\n", ratio(v_folded_full),
                std::sqrt(1.0 + ratio(v_folded_full)));
    std::printf("  correlations inflate the folded MC variance by x%.2f\n",
                v_folded_diag > 0.0 ? v_folded_full / v_folded_diag : 0.0);
  }

  return 0;
}
