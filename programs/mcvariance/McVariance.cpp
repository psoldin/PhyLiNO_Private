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
#include <algorithm>
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

namespace {

  /**
   * sigma_stat and sigma_MC for an arbitrary per-event bin assignment.
   *
   * Both come out of the same two sums, so any candidate binning can be scored
   * without touching the fit: the Fisher information
   *
   *     I = sum_b (d mu_b / d gamma)^2 / mu_b,    sigma_stat = 1 / sqrt(I)
   *
   * and the MC-induced variance, which for an unsmeared histogram (each event in
   * exactly one bin) is diagonal and exact,
   *
   *     V = sum_i w_i^2 g_bin(i)^2,               sigma_MC^2 = V / I^2.
   *
   * Refining a binning can only raise I -- information is never lost by
   * splitting a bin -- but it also spreads the same events over more bins, which
   * raises V. Whether a new axis is worth having is that trade, and this
   * computes both halves of it.
   */
  struct BinnedInfo {
    double fisher = 0.0;
    double v_mc   = 0.0;

    [[nodiscard]] double sigma_stat() const { return 1.0 / std::sqrt(fisher); }
    [[nodiscard]] double mc_ratio() const { return v_mc / fisher; }
    [[nodiscard]] double sigma_total() const { return sigma_stat() * std::sqrt(1.0 + mc_ratio()); }
  };

  BinnedInfo score_binning(const std::vector<int>& bin_of_event, const std::size_t n_bins,
                           const std::vector<double>& w, const std::vector<double>& dw) {
    std::vector<double> mu(n_bins, 0.0), dmu(n_bins, 0.0);
    for (std::size_t i = 0; i < w.size(); ++i) {
      const auto b = static_cast<std::size_t>(bin_of_event[i]);
      mu[b] += w[i];
      dmu[b] += dw[i];
    }

    BinnedInfo          info;
    std::vector<double> g(n_bins, 0.0);
    for (std::size_t b = 0; b < n_bins; ++b) {
      if (!(mu[b] > 0.0)) continue;
      g[b] = dmu[b] / mu[b];
      info.fisher += dmu[b] * dmu[b] / mu[b];
    }
    for (std::size_t i = 0; i < w.size(); ++i) {
      const double gb = g[static_cast<std::size_t>(bin_of_event[i])];
      info.v_mc += w[i] * w[i] * gb * gb;
    }
    return info;
  }

  /** Equal-occupancy edges, so no resolution category is starved of events. */
  std::vector<double> quantile_edges(std::vector<double> values, const int n) {
    std::ranges::sort(values);
    std::vector<double> edges;
    for (int k = 1; k < n; ++k)
      edges.push_back(values[static_cast<std::size_t>(static_cast<double>(k) / n * (values.size() - 1))]);
    return edges;
  }

}  // namespace

int main(int argc, char** argv) {
  std::string config;
  std::string scanned      = "SpectralIndex";
  int         n_sigma_bins = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-c" && i + 1 < argc) config = argv[++i];
    else if (arg == "-p" && i + 1 < argc) scanned = argv[++i];
    else if (arg == "--sigma-bins" && i + 1 < argc) n_sigma_bins = std::stoi(argv[++i]);
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

    // A resolution axis: the same events, the same reco coordinates, split by the
    // per-event energy uncertainty. Nothing is smeared -- sigma is used as an
    // observable, not as a width -- so this asks a different question from the
    // fold above: does knowing HOW WELL an event was measured tell you anything
    // the reconstructed value alone does not?
    if (n_sigma_bins > 1) {
      std::vector<double> dw(w.size(), 0.0);
      {
        std::vector<double> up = values, down = values;
        up[scanned_index] += step;
        down[scanned_index] -= step;
        parameter.reset_parameter(up.data());
        likelihood.partial_llh(parameter);
        const std::vector<double> w_up = likelihood.per_event_weight();
        parameter.reset_parameter(down.data());
        likelihood.partial_llh(parameter);
        const std::vector<double> w_down = likelihood.per_event_weight();
        for (std::size_t i = 0; i < dw.size(); ++i) dw[i] = (w_up[i] - w_down[i]) / (2.0 * step);
      }

      const std::vector<double> edges =
          quantile_edges(sample.response_sigma_log_e, n_sigma_bins);

      std::vector<int> flat_2d(sample.size()), flat_3d(sample.size());
      for (std::size_t i = 0; i < sample.size(); ++i) {
        const double s_i = sample.response_sigma_log_e[i];
        int          k   = 0;
        while (k < static_cast<int>(edges.size()) && s_i >= edges[static_cast<std::size_t>(k)]) ++k;
        flat_2d[i] = sample.bin_idx[i];
        flat_3d[i] = sample.bin_idx[i] * n_sigma_bins + k;
      }

      const BinnedInfo flat  = score_binning(flat_2d, n_bins, w, dw);
      const BinnedInfo split = score_binning(flat_3d, n_bins * static_cast<std::size_t>(n_sigma_bins), w, dw);

      std::printf("\nsample '%s', parameter %s -- resolution as a third axis (%d quantile bins)\n",
                  cfg.name.c_str(), scanned.c_str(), n_sigma_bins);
      std::printf("  sigma_E quantile edges [dex]:");
      for (const double e : edges) std::printf(" %.3f", e);
      std::printf("\n");
      std::printf("                     bins    sigma_stat   (sigMC/sigStat)^2   sigma_total\n");
      std::printf("  2D                %6zu      %.5f          %.4f          %.5f\n", n_bins,
                  flat.sigma_stat(), flat.mc_ratio(), flat.sigma_total());
      std::printf("  2D x sigma_E      %6zu      %.5f          %.4f          %.5f\n",
                  n_bins * static_cast<std::size_t>(n_sigma_bins), split.sigma_stat(), split.mc_ratio(),
                  split.sigma_total());
      std::printf("  change                            %+.2f%%                        %+.2f%%\n",
                  100.0 * (split.sigma_stat() / flat.sigma_stat() - 1.0),
                  100.0 * (split.sigma_total() / flat.sigma_total() - 1.0));

      // The ceiling on every method that tries to exploit resolution: bin on the
      // TRUE energy instead of the reconstructed one, i.e. a perfect energy
      // measurement. No unbinned kernel, no fold and no resolution axis can beat
      // this, because none of them recovers information the reconstruction did
      // not keep. If the ceiling is low, the whole line of attack is bounded and
      // the question stops being which method to use.
      {
        const io::ic::Axis& energy   = cfg.mc_binning.axes()[0];
        const int           n_zenith = cfg.mc_binning.axes()[1].n_bins;

        std::vector<int>    truth_bin;
        std::vector<double> w_in, dw_in;
        truth_bin.reserve(sample.size());
        for (std::size_t i = 0; i < sample.size(); ++i) {
          // Axis::index() projects with log10, and the truth column is already
          // in log10; hand it the linear energy so the axis does its own mapping.
          const int ie = energy.index(std::pow(10.0, sample.response_truth_log_e[i]));
          if (ie < 0) continue;  // true energy outside the analysis range
          truth_bin.push_back(ie * n_zenith + sample.bin_idx[i] % n_zenith);
          w_in.push_back(w[i]);
          dw_in.push_back(dw[i]);
        }

        const BinnedInfo perfect = score_binning(truth_bin, n_bins, w_in, dw_in);
        std::printf("  perfect E         %6zu      %.5f          %.4f          %.5f\n", n_bins,
                    perfect.sigma_stat(), perfect.mc_ratio(), perfect.sigma_total());
        std::printf("  ceiling                           %+.2f%%                        %+.2f%%   "
                    "(%zu of %zu events kept)\n",
                    100.0 * (perfect.sigma_stat() / flat.sigma_stat() - 1.0),
                    100.0 * (perfect.sigma_total() / flat.sigma_total() - 1.0), w_in.size(),
                    sample.size());
      }

      parameter.reset_parameter(values.data());
      likelihood.partial_llh(parameter);
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
