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
#include <limits>
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

  /**
   * Profiled uncertainty on one parameter: sqrt of the diagonal of the inverted
   * Fisher matrix, rather than 1/sqrt of its diagonal element.
   *
   * With the nuisances fixed, an axis that separates populations looks purely
   * beneficial. Freeing them is where such gains usually go: the same
   * separation that distinguishes signal from background often also lets a
   * nuisance absorb the difference, and only the profiled number says which
   * happened.
   *
   * Gaussian priors enter as 1/width^2 on the diagonal, which is what a
   * constrained parameter contributes to the second derivative of -2lnL.
   */
  struct FisherMatrix {
    std::size_t         n = 0;
    std::vector<double> m;  ///< row-major n x n

    double&       at(const std::size_t a, const std::size_t b) { return m[a * n + b]; }
    const double& at(const std::size_t a, const std::size_t b) const { return m[a * n + b]; }
  };

  /** Gauss-Jordan with partial pivoting; the matrix is small and symmetric. */
  std::vector<double> invert(FisherMatrix f) {
    const std::size_t   n = f.n;
    std::vector<double> inv(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) inv[i * n + i] = 1.0;

    for (std::size_t col = 0; col < n; ++col) {
      std::size_t pivot = col;
      for (std::size_t r = col + 1; r < n; ++r)
        if (std::abs(f.at(r, col)) > std::abs(f.at(pivot, col))) pivot = r;
      if (std::abs(f.at(pivot, col)) < 1e-300) return {};  // singular

      if (pivot != col)
        for (std::size_t c = 0; c < n; ++c) {
          std::swap(f.at(col, c), f.at(pivot, c));
          std::swap(inv[col * n + c], inv[pivot * n + c]);
        }

      const double d = f.at(col, col);
      for (std::size_t c = 0; c < n; ++c) {
        f.at(col, c) /= d;
        inv[col * n + c] /= d;
      }
      for (std::size_t r = 0; r < n; ++r) {
        if (r == col) continue;
        const double factor = f.at(r, col);
        if (factor == 0.0) continue;
        for (std::size_t c = 0; c < n; ++c) {
          f.at(r, c) -= factor * f.at(col, c);
          inv[r * n + c] -= factor * inv[col * n + c];
        }
      }
    }
    return inv;
  }

  /**
   * Profiled sigma on parameter `target`, given per-event derivatives for every
   * free parameter and a bin assignment. `prior_width` is 0 for unconstrained.
   */
  double profiled_sigma(const std::vector<int>& bin_of_event, const std::size_t n_bins,
                        const std::vector<double>& w, const std::vector<std::vector<double>>& dw,
                        const std::vector<double>& prior_width, const std::size_t target) {
    const std::size_t n_par = dw.size();

    std::vector<double>              mu(n_bins, 0.0);
    std::vector<std::vector<double>> dmu(n_par, std::vector<double>(n_bins, 0.0));
    for (std::size_t i = 0; i < w.size(); ++i) {
      const auto b = static_cast<std::size_t>(bin_of_event[i]);
      mu[b] += w[i];
      for (std::size_t a = 0; a < n_par; ++a) dmu[a][b] += dw[a][i];
    }

    FisherMatrix f{.n = n_par, .m = std::vector<double>(n_par * n_par, 0.0)};
    for (std::size_t b = 0; b < n_bins; ++b) {
      if (!(mu[b] > 0.0)) continue;
      const double inv_mu = 1.0 / mu[b];
      for (std::size_t a = 0; a < n_par; ++a) {
        if (dmu[a][b] == 0.0) continue;
        for (std::size_t c = a; c < n_par; ++c) f.at(a, c) += dmu[a][b] * dmu[c][b] * inv_mu;
      }
    }
    for (std::size_t a = 0; a < n_par; ++a) {
      for (std::size_t c = 0; c < a; ++c) f.at(a, c) = f.at(c, a);
      if (prior_width[a] > 0.0) f.at(a, a) += 1.0 / (prior_width[a] * prior_width[a]);
    }

    const std::vector<double> inv = invert(f);
    if (inv.empty()) return std::numeric_limits<double>::quiet_NaN();
    return std::sqrt(inv[target * n_par + target]);
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
  std::string dump_pair, dump_file;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-c" && i + 1 < argc) config = argv[++i];
    else if (arg == "-p" && i + 1 < argc) scanned = argv[++i];
    else if (arg == "--sigma-bins" && i + 1 < argc) n_sigma_bins = std::stoi(argv[++i]);
    else if (arg == "--dump-pair" && i + 1 < argc) dump_pair = argv[++i];
    else if (arg == "--dump-file" && i + 1 < argc) dump_file = argv[++i];
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

  // Free parameters, in config order, with their Gaussian prior widths. These
  // are the ones the profiled error marginalises over. Detector systematics are
  // absent here by construction: they enter as per-BIN histogram gradients tied
  // to the sample's own binning, so a category axis leaves them undefined -- see
  // the note printed below.
  std::vector<std::size_t> free_index;
  std::vector<std::string> free_name;
  std::vector<double>      prior_width;
  for (const auto& [ignored, node] : tree.get_child("Parameter")) {
    (void)ignored;
    const std::string name = node.get<std::string>("Name");
    const auto        it   = parameter_indices().find(name);
    if (it == parameter_indices().end() || node.get<bool>("Fixed", true)) continue;
    free_index.push_back(it->second);
    free_name.push_back(name);
    prior_width.push_back(node.get<bool>("Constrained", false) ? node.get<double>("PriorWidth", 0.0) : 0.0);
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

    // Candidate binnings, all scored on the SAME events so the comparison is
    // like for like: an event whose truth falls outside the analysis range
    // cannot be binned on truth, and letting the baseline keep it while the
    // ceiling drops it would flatter the baseline.
    //
    // Two distinct uses of the per-event uncertainties appear here. A resolution
    // AXIS does not smear anything: it partitions the sample, so
    // p(reco, sigma | theta) replaces p(reco | theta) and carries strictly more
    // information whenever sigma says how sharply reco tracks truth. The
    // "perfect" rows bin on truth itself and are the ceiling: nothing that
    // exploits resolution can beat having measured the quantity exactly.
    if (n_sigma_bins > 1) {
      // One central difference per free parameter, giving dw_i/dtheta_a for all
      // of them; the profiled error needs the whole Fisher matrix, not just the
      // target's diagonal.
      std::vector<std::vector<double>> dw_all;
      std::size_t                      target_slot = 0;
      for (std::size_t a = 0; a < free_index.size(); ++a) {
        if (free_index[a] == scanned_index) target_slot = a;
        std::vector<double> up = values, down = values;
        up[free_index[a]] += step;
        down[free_index[a]] -= step;
        parameter.reset_parameter(up.data());
        likelihood.partial_llh(parameter);
        const std::vector<double> w_up = likelihood.per_event_weight();
        parameter.reset_parameter(down.data());
        likelihood.partial_llh(parameter);
        const std::vector<double> w_down = likelihood.per_event_weight();

        std::vector<double> d(w.size(), 0.0);
        for (std::size_t i = 0; i < d.size(); ++i) d[i] = (w_up[i] - w_down[i]) / (2.0 * step);
        dw_all.push_back(std::move(d));
      }

      const io::ic::Axis& energy_axis = cfg.mc_binning.axes()[0];
      const io::ic::Axis& zenith_axis = cfg.mc_binning.axes()[1];
      const int           n_zenith    = zenith_axis.n_bins;

      const std::vector<double> edges_e = quantile_edges(sample.response_sigma_log_e, n_sigma_bins);
      const std::vector<double> edges_z = quantile_edges(sample.response_sigma_zenith, n_sigma_bins);
      auto category = [](const std::vector<double>& edges, const double x) {
        int k = 0;
        while (k < static_cast<int>(edges.size()) && x >= edges[static_cast<std::size_t>(k)]) ++k;
        return k;
      };

      // Per-event indices on every axis a candidate might use.
      std::vector<std::size_t> keep;
      std::vector<int>    ie_reco, iz_reco, ie_true, iz_true, ke, kz;
      std::vector<double> w_in;
      for (std::size_t i = 0; i < sample.size(); ++i) {
        // Axis::index() projects with log10 / cos, so it is handed the linear
        // energy and the raw angle rather than the already-projected columns.
        const int e_true = energy_axis.index(std::pow(10.0, sample.response_truth_log_e[i]));
        const int z_true = zenith_axis.index(sample.response_truth_zenith[i]);
        if (e_true < 0 || z_true < 0) continue;

        keep.push_back(i);
        ie_reco.push_back(sample.bin_idx[i] / n_zenith);
        iz_reco.push_back(sample.bin_idx[i] % n_zenith);
        ie_true.push_back(e_true);
        iz_true.push_back(z_true);
        ke.push_back(category(edges_e, sample.response_sigma_log_e[i]));
        kz.push_back(category(edges_z, sample.response_sigma_zenith[i]));
        w_in.push_back(w[i]);
      }

      const auto n_kept = ie_reco.size();
      const int  N      = n_sigma_bins;

      std::vector<std::vector<double>> dw_in(dw_all.size());
      for (std::size_t a = 0; a < dw_all.size(); ++a) {
        dw_in[a].reserve(n_kept);
        for (const std::size_t i : keep) dw_in[a].push_back(dw_all[a][i]);
      }

      std::printf("  profiling over %zu free parameters:", free_name.size());
      for (const std::string& n : free_name) std::printf(" %s", n.c_str());
      std::printf("\n  (detector systematics are per-bin gradients tied to the 2D binning and cannot\n"
                  "   follow a category axis; they are absent here, see the report)\n");

      // Flatten (energy, zenith, cat_e, cat_z) row-major over whichever axes a
      // candidate uses; unused categories collapse to a single bin.
      auto compose = [&](const std::vector<int>& ie, const std::vector<int>& iz, const bool use_e,
                         const bool use_z, std::size_t& n_bins_out) {
        const int ne = use_e ? N : 1;
        const int nz = use_z ? N : 1;
        n_bins_out   = static_cast<std::size_t>(energy_axis.n_bins) * n_zenith * ne * nz;
        std::vector<int> flat(n_kept);
        for (std::size_t i = 0; i < n_kept; ++i)
          flat[i] = ((ie[i] * n_zenith + iz[i]) * ne + (use_e ? ke[i] : 0)) * nz + (use_z ? kz[i] : 0);
        return flat;
      };

      std::printf("\nsample '%s', parameter %s -- candidate binnings (%d quantile categories each)\n",
                  cfg.name.c_str(), scanned.c_str(), N);
      std::printf("  sigma_E   edges [dex]:");
      for (const double e : edges_e) std::printf(" %.3f", e);
      std::printf("\n  sigma_dir edges [deg]:");
      for (const double e : edges_z) std::printf(" %.3f", e * 180.0 / 3.14159265358979323846);
      std::printf("\n  %zu of %zu events have both truths in range\n\n", n_kept, sample.size());

      std::printf("\n  %-40s %8s  %10s  %8s  %10s  %8s\n", "binning", "bins", "sigma_fixed", "d%",
                  "sigma_prof", "d%");

      BinnedInfo baseline;
      double     baseline_prof = 0.0;
      auto       row = [&](const char* label, const std::vector<int>& flat, const std::size_t nb,
                     const bool is_baseline) {
        const BinnedInfo info = score_binning(flat, nb, w_in, dw_in[target_slot]);
        const double     prof =
            profiled_sigma(flat, nb, w_in, dw_in, prior_width, target_slot);
        if (is_baseline) {
          baseline      = info;
          baseline_prof = prof;
        }
        std::printf("  %-40s %8zu  %10.5f  %+7.2f%%  %10.5f  %+7.2f%%\n", label, nb, info.sigma_stat(),
                    100.0 * (info.sigma_stat() / baseline.sigma_stat() - 1.0), prof,
                    100.0 * (prof / baseline_prof - 1.0));
      };

      std::size_t nb = 0;
      const std::vector<int> flat_2d = compose(ie_reco, iz_reco, false, false, nb);
      row("2D (baseline)", flat_2d, nb, true);

      const std::vector<int> flat_e = compose(ie_reco, iz_reco, true, false, nb);
      row("2D x sigma_E", flat_e, nb, false);

      const std::vector<int> flat_z = compose(ie_reco, iz_reco, false, true, nb);
      row("2D x sigma_dir", flat_z, nb, false);

      const std::vector<int> flat_ez = compose(ie_reco, iz_reco, true, true, nb);
      row("2D x sigma_E x sigma_dir", flat_ez, nb, false);

      // Any other per-event reco column the config named, on the same footing.
      // The two sigmas above are only special in having a physical story; what
      // an axis actually has to do is separate populations the fit is sensitive
      // to, and nothing says a reported uncertainty is best at that.
      std::vector<std::vector<int>> extra_category;
      for (std::size_t c = 0; c < sample.categories.size(); ++c) {
        std::vector<double> values_in;
        values_in.reserve(n_kept);
        for (std::size_t i = 0, k = 0; i < sample.size(); ++i) {
          const int e_true = energy_axis.index(std::pow(10.0, sample.response_truth_log_e[i]));
          const int z_true = zenith_axis.index(sample.response_truth_zenith[i]);
          if (e_true < 0 || z_true < 0) continue;
          values_in.push_back(sample.categories[c][i]);
          ++k;
        }

        const std::vector<double> edges = quantile_edges(values_in, N);
        std::vector<int>          cat(n_kept);
        for (std::size_t i = 0; i < n_kept; ++i) cat[i] = category(edges, values_in[i]);
        extra_category.push_back(cat);

        std::vector<int>  flat(n_kept);
        const std::size_t bins = static_cast<std::size_t>(energy_axis.n_bins) * n_zenith * N;
        for (std::size_t i = 0; i < n_kept; ++i)
          flat[i] = (ie_reco[i] * n_zenith + iz_reco[i]) * N + cat[i];
        row(("2D x " + sample.category_names[c]).c_str(), flat, bins, false);
      }

      // Pairs, since the two sigmas were markedly super-additive.
      for (std::size_t c = 0; c < extra_category.size(); ++c)
        for (std::size_t d = c + 1; d < extra_category.size(); ++d) {
          std::vector<int>  flat(n_kept);
          const std::size_t bins =
              static_cast<std::size_t>(energy_axis.n_bins) * n_zenith * N * N;
          for (std::size_t i = 0; i < n_kept; ++i)
            flat[i] = ((ie_reco[i] * n_zenith + iz_reco[i]) * N + extra_category[c][i]) * N +
                      extra_category[d][i];
          row(("2D x " + sample.category_names[c] + " x " + sample.category_names[d]).c_str(), flat, bins,
              false);
        }

      // Dump one candidate binning for the external SnowStorm gradient producer:
      // the predicted histogram, its derivative with respect to every free flux
      // parameter, and the exact category edges, so the ensemble can be
      // histogrammed into the identical bins.
      if (!dump_file.empty() && !dump_pair.empty()) {
        const auto comma = dump_pair.find(',');
        const std::string name_a = dump_pair.substr(0, comma);
        const std::string name_b = comma == std::string::npos ? std::string{} : dump_pair.substr(comma + 1);

        std::size_t ia = sample.category_names.size(), ib = sample.category_names.size();
        for (std::size_t c = 0; c < sample.category_names.size(); ++c) {
          if (sample.category_names[c] == name_a) ia = c;
          if (sample.category_names[c] == name_b) ib = c;
        }
        if (ia == sample.category_names.size())
          throw std::runtime_error("--dump-pair: column '" + name_a + "' is not in the sample's Categories");
        const bool have_b = ib < sample.category_names.size();

        const int         nb_cat = have_b ? N : 1;
        const std::size_t n_dump =
            static_cast<std::size_t>(energy_axis.n_bins) * n_zenith * N * nb_cat;
        std::vector<int> flat(n_kept);
        for (std::size_t i = 0; i < n_kept; ++i)
          flat[i] = ((ie_reco[i] * n_zenith + iz_reco[i]) * N + extra_category[ia][i]) * nb_cat +
                    (have_b ? extra_category[ib][i] : 0);

        std::vector<double>              mu_d(n_dump, 0.0);
        std::vector<std::vector<double>> dmu_d(dw_in.size(), std::vector<double>(n_dump, 0.0));
        for (std::size_t i = 0; i < n_kept; ++i) {
          const auto b2 = static_cast<std::size_t>(flat[i]);
          mu_d[b2] += w_in[i];
          for (std::size_t a2 = 0; a2 < dw_in.size(); ++a2) dmu_d[a2][b2] += dw_in[a2][i];
        }

        std::FILE* f = std::fopen(dump_file.c_str(), "w");
        if (!f) throw std::runtime_error("--dump-file: cannot open " + dump_file);
        std::fprintf(f, "# energy %.10g %.10g %d\n", energy_axis.lo, energy_axis.hi, energy_axis.n_bins);
        std::fprintf(f, "# zenith %.10g %.10g %d\n", zenith_axis.lo, zenith_axis.hi, n_zenith);
        auto write_edges = [f, N](const char* name, const std::vector<double>& e) {
          std::fprintf(f, "# category %s %d", name, N);
          for (const double v : e) std::fprintf(f, " %.10g", v);
          std::fprintf(f, "\n");
        };
        write_edges(name_a.c_str(), quantile_edges(
            [&] { std::vector<double> v; for (const std::size_t i : keep) v.push_back(sample.categories[ia][i]); return v; }(), N));
        if (have_b)
          write_edges(name_b.c_str(), quantile_edges(
              [&] { std::vector<double> v; for (const std::size_t i : keep) v.push_back(sample.categories[ib][i]); return v; }(), N));
        std::fprintf(f, "# params %zu", dw_in.size());
        for (const std::string& n : free_name) std::fprintf(f, " %s", n.c_str());
        std::fprintf(f, "\n# bins %zu\n# per bin: mu then d(mu)/d(param) for each param above\n", n_dump);
        for (std::size_t b2 = 0; b2 < n_dump; ++b2) {
          std::fprintf(f, "%.10g", mu_d[b2]);
          for (std::size_t a2 = 0; a2 < dw_in.size(); ++a2) std::fprintf(f, " %.10g", dmu_d[a2][b2]);
          std::fprintf(f, "\n");
        }
        std::fclose(f);
        std::printf("\n  dumped %zu bins x %zu parameters to %s\n", n_dump, dw_in.size(), dump_file.c_str());
      }

      const std::vector<int> perfect_e = compose(ie_true, iz_reco, false, false, nb);
      row("perfect E      (ceiling)", perfect_e, nb, false);

      const std::vector<int> perfect_z = compose(ie_reco, iz_true, false, false, nb);
      row("perfect dir    (ceiling)", perfect_z, nb, false);

      const std::vector<int> perfect_both = compose(ie_true, iz_true, false, false, nb);
      row("perfect E+dir  (ceiling)", perfect_both, nb, false);

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
