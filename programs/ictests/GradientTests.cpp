// Tests of the analytic gradient and Gauss-Newton Hessian of the IceCube
// likelihood (FluxGradient, SampleLikelihood::accumulate_gradient()).
//
// Every derivative is checked against a central finite difference of the value
// the fit actually minimises -- SampleLikelihood::partial_llh() -- so the tests
// pin the gradient to the likelihood as implemented, branches and all, not to a
// second transcription of its formulas.

#include "BinTermDerivatives.h"
#include "CudaBackend.h"
#include "FluxGradient.h"
#include "IceCube/Binning.h"
#include "IceCube/ICParameter.h"
#include "IceCube/ICSample.h"
#include "IceCube/SampleConfig.h"
#include "MetalBackend.h"
#include "ParameterWrapper.h"
#include "PoissonLikelihood.h"
#include "Polygamma.h"
#include "SAYLikelihood.h"
#include "SampleLikelihood.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

  using ana::ParameterWrapper;
  using ana::ic::FluxGradient;
  using ana::ic::SampleLikelihood;
  using io::ic::AstroModel;
  using io::ic::Binning;
  namespace P = params::ic;

  constexpr int kNPar = P::number_of_parameters();

  Binning gradient_binning() {
    return Binning({io::ic::parse_axis("Log10Energy", "(2.0, 5.0, 3)"),
                    io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 2)")});
  }

  /**
   * A sample with enough events per bin, weights spread enough within a bin,
   * and bin contents of O(10..1000), that the SAY term sits in its ordinary
   * branch (0 < s < mu^2) with alpha anywhere from a few to a few thousand.
   * Every per-event column the flux components read is populated and varies
   * with energy, so no derivative vanishes by accident.
   */
  io::ic::ICSample gradient_sample(const Binning& binning, const int events_per_bin, const bool veto) {
    io::ic::ICSample sample;
    const double     zeniths[2] = {2.0, 0.5};
    for (int eb = 0; eb < 3; ++eb)
      for (int zb = 0; zb < 2; ++zb)
        for (int k = 0; k < events_per_bin; ++k) {
          const double frac   = (k + 0.5) / events_per_bin;
          const double jitter = 0.5 + frac + 0.1 * zb;
          const double e      = std::pow(10.0, 2.0 + eb + frac);
          const double n      = events_per_bin;
          sample.e_true.push_back(e);
          sample.astro_baseline.push_back(3.0 * jitter / n);
          const double conv = 200.0 * jitter / n;
          sample.conv_baseline.push_back(conv);
          sample.conv_alt.push_back(0.9 * conv * (1.0 + 0.2 * frac));
          const double prompt = 8.0 * jitter / n;
          sample.prompt_baseline.push_back(prompt);
          sample.prompt_alt.push_back(1.15 * prompt);
          for (int b = 0; b < P::nBarrParams; ++b)
            sample.barr_conv[b].push_back(0.05 * (b + 1) * conv * (frac - 0.3 + 0.1 * eb));
          if (veto) {
            sample.veto_conv[0].push_back(-0.10 * jitter);
            sample.veto_conv[1].push_back(-1.0e-3 * jitter);
            sample.veto_conv[2].push_back(2.0e-6 * jitter);
            sample.veto_prompt[0].push_back(-0.05 * jitter);
            sample.veto_prompt[1].push_back(-2.0e-3 * jitter);
            sample.veto_prompt[2].push_back(1.0e-6 * jitter);
          }
          const double reco[2] = {e, zeniths[zb]};
          sample.bin_idx.push_back(binning.bin_index(reco));
        }
    sample.sort_into_bins(binning.total_bins());
    return sample;
  }

  ana::ic::GlobalFluxSettings gradient_settings(const AstroModel model) {
    return ana::ic::GlobalFluxSettings{.e_ref_gev                = 1.0e5,
                                       .astro_reference_index    = 2.0,
                                       .conv_delta_gamma_e_ref   = 1.0e3,
                                       .prompt_delta_gamma_e_ref = 3.8e3,
                                       .astro_per_type_norm      = false,
                                       .veto_anchor_energy       = 100.0,
                                       .veto_rescale_energy      = 100.0,
                                       .astro_model              = model,
                                       .use_multi_threading      = false};
  }

  /// Writes a template file over `bins` bins; removed by the caller.
  std::string write_template(const std::string& path, const int bins, const double scale) {
    std::ofstream out(path);
    out << "# template bins " << bins << "\n";
    for (int b = 0; b < bins; ++b) out << scale * (1.0 + 0.3 * b) << ' ' << 0.1 * scale * (1.0 + 0.2 * b) << '\n';
    return path;
  }

  /// Writes a SnowStorm gradient file over `bins` bins; removed by the caller.
  std::string write_gradients(const std::string& path, const int bins) {
    static const char* kNames[5] = {"DOMEfficiency", "IceAbsorption", "IceScattering", "HoleIceForward_p0",
                                    "HoleIceForward_p1"};
    const double       split[5]  = {1.0, 1.0, 1.0, 0.25, -0.05};
    std::ofstream      out(path);
    out << "# gradients bins " << bins << " params 5 lt_scale 1.3\n";
    for (int k = 0; k < 5; ++k) {
      out << "# param " << kNames[k] << " split " << split[k] << "\n";
      for (int b = 0; b < bins; ++b) out << 40.0 * (k + 1) * (1.0 + 0.1 * b) << ' ' << 3.0 * (k + 1) + b << '\n';
    }
    int pair = 0;
    for (int i = 0; i < 5; ++i)
      for (int j = i + 1; j < 5; ++j, ++pair) {
        out << "# cov " << kNames[i] << " " << kNames[j] << "\n";
        for (int b = 0; b < bins; ++b) out << 0.5 * (pair + 1) + 0.1 * b << "\n";
      }
    return path;
  }

  /// A point away from the Asimov truth in every parameter the likelihood
  /// reads, so no gradient component is zero by symmetry.
  std::vector<double> perturbed_values() {
    std::vector<double> v(kNPar, 0.0);
    v[P::AstroNorm]      = 1.7;
    v[P::SpectralIndex]  = 2.55;
    v[P::ConvNorm]       = 1.08;
    v[P::PromptNorm]     = 0.7;
    v[P::BarrH]          = 0.12;
    v[P::BarrW]          = -0.2;
    v[P::BarrY]          = 0.07;
    v[P::BarrZ]          = -0.05;
    v[P::CRGrad]         = 0.3;
    v[P::DeltaGamma]     = 0.04;
    v[P::MuonNorm]       = 1.2;
    v[P::MuonGunNorm]    = 1.1;
    v[P::VetoThreshold]  = 0.15;
    v[P::DOMEff]         = 1.02;
    v[P::IceAbs]         = 0.98;
    v[P::IceScat]        = 1.01;
    v[P::HoleIceP0]      = 0.27;
    v[P::HoleIceP1]      = -0.06;
    v[P::GalacticNorm0]  = 1.3;
    v[P::GalacticNorm1]  = 0.8;
    v[P::AstroGamma1]    = 2.1;
    v[P::AstroGamma2]    = 2.9;
    v[P::AstroEBreak]    = 4.3;
    v[P::AstroParabolaB] = 0.15;
    v[P::AstroLogECut]   = 5.5;
    return v;
  }

  /// The truth the Asimov data is generated at: a little off the evaluation
  /// point in every parameter.
  std::vector<double> truth_values() {
    std::vector<double> v = perturbed_values();
    for (int i = 0; i < kNPar; ++i) v[i] = (i % 2 == 0 ? 0.97 : 1.03) * v[i] + (i % 3 == 0 ? 0.01 : -0.01);
    return v;
  }

  /// Sets the Asimov data at the truth, then scatters it by a smooth pattern so
  /// the per-bin residuals do not all share a sign.
  void prepare_data(SampleLikelihood& likelihood) {
    const std::vector<double> truth = truth_values();
    ParameterWrapper          at_truth(kNPar);
    at_truth.reset_parameter(truth.data());
    likelihood.generate_asimov(at_truth);
    std::vector<double> data(likelihood.data().begin(), likelihood.data().end());
    for (std::size_t b = 0; b < data.size(); ++b) data[b] *= 1.0 + 0.08 * std::sin(1.7 * static_cast<double>(b) + 0.3);
    likelihood.set_data(data);
  }

  std::vector<double> analytic_gradient(SampleLikelihood& likelihood, const std::vector<double>& values,
                                        std::vector<double>* hessian = nullptr) {
    ParameterWrapper parameter(kNPar);
    parameter.reset_parameter(values.data());
    std::vector<double> gradient(kNPar, 0.0);
    if (hessian) hessian->assign(kNPar * kNPar, 0.0);
    likelihood.accumulate_gradient(parameter, gradient, hessian ? hessian->data() : nullptr);
    return gradient;
  }

  double llh_at(SampleLikelihood& likelihood, ParameterWrapper& parameter, const std::vector<double>& values) {
    parameter.reset_parameter(values.data());
    return likelihood.partial_llh(parameter);
  }

  /// Central differences of partial_llh() in every parameter.
  std::vector<double> numerical_gradient(SampleLikelihood& likelihood, const std::vector<double>& values) {
    ParameterWrapper    parameter(kNPar);
    std::vector<double> gradient(kNPar, 0.0);
    for (int i = 0; i < kNPar; ++i) {
      const double        h  = 1.0e-5 * std::max(1.0, std::fabs(values[i]));
      std::vector<double> up = values, down = values;
      up[i] += h;
      down[i] -= h;
      gradient[i] = (llh_at(likelihood, parameter, up) - llh_at(likelihood, parameter, down)) / (2.0 * h);
    }
    return gradient;
  }

  /// Every component agrees to `rel` of the gradient's own scale, and every
  /// parameter the configuration reads has a non-zero derivative.
  void expect_gradients_agree(const std::vector<double>& analytic, const std::vector<double>& numerical,
                              const std::vector<int>& must_be_nonzero, const double rel) {
    double scale = 0.0;
    for (const double g : numerical) scale = std::max(scale, std::fabs(g));
    ASSERT_GT(scale, 0.0);
    for (int i = 0; i < kNPar; ++i) {
      const double tolerance = rel * std::max(std::fabs(numerical[i]), 1.0e-3 * scale);
      EXPECT_NEAR(analytic[i], numerical[i], tolerance) << "parameter " << i;
    }
    for (const int i : must_be_nonzero) EXPECT_NE(numerical[i], 0.0) << "parameter " << i << " does not enter";
  }

  struct LikelihoodCase {
    const char* name;
    bool        say;
    double      alpha_offset;
  };

  constexpr LikelihoodCase kLikelihoods[] = {{"poisson", false, 1.0}, {"say", true, 1.0}, {"saymean", true, 0.0}};

  std::vector<int> flux_parameters(const AstroModel model, const bool veto) {
    std::vector<int> p = {P::AstroNorm, P::ConvNorm, P::PromptNorm, P::BarrH, P::BarrW,
                          P::BarrY,     P::BarrZ,    P::CRGrad,     P::DeltaGamma};
    for (const int k : FluxGradient::astro_shape_parameters(model))
      if (k >= 0) p.push_back(k);
    if (veto) p.push_back(P::VetoThreshold);
    return p;
  }

}  // namespace

TEST(PolygammaTest, MatchesDerivativesOfLgamma) {
  using namespace ana::ic::polygamma;
  for (const double x : {0.05, 0.3, 1.0, 1.5, 7.0, 9.99, 10.0, 12.5, 1.0e3, 1.0e6}) {
    const double h  = 1.0e-5 * x;
    const double fd = (std::lgamma(x + h) - std::lgamma(x - h)) / (2.0 * h);
    EXPECT_NEAR(digamma(x), fd, 1.0e-7 * std::max(1.0, std::fabs(fd))) << x;
    const double fd2 = (digamma(x + h) - digamma(x - h)) / (2.0 * h);
    EXPECT_NEAR(trigamma(x), fd2, 1.0e-6 * std::fabs(fd2)) << x;
  }
  // Reference values (Abramowitz & Stegun): psi(1) = -gamma_E, psi_1(1) = pi^2 / 6.
  EXPECT_NEAR(digamma(1.0), -0.5772156649015329, 1e-14);
  EXPECT_NEAR(trigamma(1.0), 1.6449340668482264, 1e-14);
}

/// The differences are what the SAY derivative actually uses, at alpha far past
/// where plain subtraction of the two polygammas loses digits.
TEST(PolygammaTest, DifferencesStayAccurateAtLargeArgument) {
  using namespace ana::ic::polygamma;
  // psi(a + k) - psi(a) = sum_{j<k} 1 / (a + j) for integer k.
  for (const double a : {0.4, 3.0, 25.0, 1.0e4, 1.0e9}) {
    for (const int k : {1, 5, 40}) {
      long double sum1 = 0.0L, sum2 = 0.0L;
      for (int j = 0; j < k; ++j) {
        sum1 += 1.0L / (static_cast<long double>(a) + j);
        sum2 -= 1.0L / ((static_cast<long double>(a) + j) * (static_cast<long double>(a) + j));
      }
      EXPECT_NEAR(digamma_difference(k, a), static_cast<double>(sum1), 1e-12 * static_cast<double>(sum1)) << a << ' ' << k;
      EXPECT_NEAR(trigamma_difference(k, a), static_cast<double>(sum2), 1e-10 * std::fabs(static_cast<double>(sum2)))
          << a << ' ' << k;
    }
  }
}

/// The central test: astro, conventional, prompt, a muon template and the
/// detector gradients in one sample, under every likelihood, for every
/// astrophysical model.
TEST(AnalyticGradientTest, MatchesFiniteDifferencesOfThePartialLlh) {
  const Binning          binning = gradient_binning();
  const io::ic::ICSample sample  = gradient_sample(binning, 12, false);
  const std::string      tmpl    = write_template("ictests_grad_template.txt", binning.total_bins(), 15.0);
  const std::string      grads   = write_gradients("ictests_grad_gradients.txt", binning.total_bins());

  for (const AstroModel model : {AstroModel::Powerlaw, AstroModel::BrokenPowerlaw, AstroModel::LogParabola,
                                 AstroModel::PowerlawCutoff}) {
    for (const LikelihoodCase& lc : kLikelihoods) {
      SCOPED_TRACE(std::string(lc.name) + " model " + std::to_string(static_cast<int>(model)));
      io::ic::SampleConfig cfg{.name = "grad", .binning = binning, .mc_binning = binning};
      cfg.components          = {"astro", "conventional", "prompt", "muontemplate"};
      cfg.template_file       = tmpl;
      cfg.template_norm_index = P::MuonNorm;
      cfg.gradient_file       = grads;

      SampleLikelihood likelihood(sample, cfg, gradient_settings(model), nullptr, lc.say, lc.alpha_offset);
      prepare_data(likelihood);

      const std::vector<double> x         = perturbed_values();
      const std::vector<double> analytic  = analytic_gradient(likelihood, x);
      const std::vector<double> numerical = numerical_gradient(likelihood, x);

      std::vector<int> used = flux_parameters(model, false);
      used.push_back(P::MuonNorm);
      for (int k = 0; k < P::nDetSysParams; ++k) used.push_back(P::DOMEff + k);
      expect_gradients_agree(analytic, numerical, used, 2.0e-5);
    }
  }
  std::remove(tmpl.c_str());
  std::remove(grads.c_str());
}

/// The cascade samples' veto reweight, with its own parameter.
TEST(AnalyticGradientTest, VetoThreshold) {
  const Binning          binning = gradient_binning();
  const io::ic::ICSample sample  = gradient_sample(binning, 10, true);
  for (const LikelihoodCase& lc : kLikelihoods) {
    SCOPED_TRACE(lc.name);
    io::ic::SampleConfig cfg{.name = "veto", .binning = binning, .mc_binning = binning};
    cfg.components = {"astro", "conventional_veto", "prompt_veto"};
    SampleLikelihood likelihood(sample, cfg, gradient_settings(AstroModel::Powerlaw), nullptr, lc.say, lc.alpha_offset);
    prepare_data(likelihood);
    const std::vector<double> x = perturbed_values();
    expect_gradients_agree(analytic_gradient(likelihood, x), numerical_gradient(likelihood, x),
                           flux_parameters(AstroModel::Powerlaw, true), 2.0e-5);
  }
}

/// An RA axis with galactic templates: one term per analysis bin, the galactic
/// norms as columns of their own.
TEST(AnalyticGradientTest, RaBinsAndGalacticTemplates) {
  const Binning mc_binning = gradient_binning();
  const Binning binning({io::ic::parse_axis("Log10Energy", "(2.0, 5.0, 3)"),
                         io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 2)"),
                         io::ic::parse_axis("Ra", "(0.0, 6.28319, 3)")});
  const io::ic::ICSample sample = gradient_sample(mc_binning, 10, false);
  const std::string      gal0   = "ictests_grad_gal0.txt";
  const std::string      gal1   = "ictests_grad_gal1.txt";
  for (const auto& [path, scale] : {std::pair{gal0, 4.0}, std::pair{gal1, 9.0}}) {
    std::ofstream out(path);
    out << "# template bins " << binning.total_bins() << "\n";
    for (int b = 0; b < binning.total_bins(); ++b) out << scale * (1.0 + 0.1 * b) << " 0\n";
  }

  for (const LikelihoodCase& lc : kLikelihoods) {
    for (const bool galactic : {false, true}) {
      SCOPED_TRACE(std::string(lc.name) + (galactic ? " galactic" : " ra only"));
      io::ic::SampleConfig cfg{.name = "ra", .binning = binning, .mc_binning = mc_binning};
      cfg.components = {"astro", "conventional", "prompt"};
      if (galactic) {
        cfg.galactic.push_back({.name = "a", .file = gal0, .norm_index = P::GalacticNorm0});
        cfg.galactic.push_back({.name = "b", .file = gal1, .norm_index = P::GalacticNorm1});
      }
      SampleLikelihood likelihood(sample, cfg, gradient_settings(AstroModel::Powerlaw), nullptr, lc.say, lc.alpha_offset);
      prepare_data(likelihood);
      const std::vector<double> x    = perturbed_values();
      std::vector<int>          used = flux_parameters(AstroModel::Powerlaw, false);
      if (galactic) {
        used.push_back(P::GalacticNorm0);
        used.push_back(P::GalacticNorm1);
      }
      expect_gradients_agree(analytic_gradient(likelihood, x), numerical_gradient(likelihood, x), used, 2.0e-5);
    }
  }
  std::remove(gal0.c_str());
  std::remove(gal1.c_str());
}

/// A forward-folded sample: the derivatives are summed over the response
/// matrix, weighted like the fold and the folded variance.
TEST(AnalyticGradientTest, ForwardFoldedSample) {
  const Binning    binning = gradient_binning();
  io::ic::ICSample sample  = gradient_sample(binning, 10, false);
  // Each event lands 70 % in its own bin and 30 % in the next one (cyclically).
  const int n_bins = binning.total_bins();
  sample.response.bin_offsets.assign(static_cast<std::size_t>(n_bins) + 1, 0);
  std::vector<std::vector<std::pair<int, float>>> entries(n_bins);
  for (int b = 0; b < n_bins; ++b)
    for (std::size_t i = sample.bin_offsets[b]; i < sample.bin_offsets[b + 1]; ++i) {
      entries[b].push_back({static_cast<int>(i), 0.7f});
      entries[(b + 1) % n_bins].push_back({static_cast<int>(i), 0.3f});
    }
  for (int b = 0; b < n_bins; ++b) {
    for (const auto& [event, fraction] : entries[b]) {
      sample.response.events.push_back(event);
      sample.response.fractions.push_back(fraction);
    }
    sample.response.bin_offsets[b + 1] = sample.response.events.size();
  }

  for (const LikelihoodCase& lc : kLikelihoods) {
    SCOPED_TRACE(lc.name);
    io::ic::SampleConfig cfg{.name = "folded", .binning = binning, .mc_binning = binning};
    cfg.components = {"astro", "conventional", "prompt"};
    SampleLikelihood likelihood(sample, cfg, gradient_settings(AstroModel::Powerlaw), nullptr, lc.say, lc.alpha_offset);
    prepare_data(likelihood);
    const std::vector<double> x = perturbed_values();
    expect_gradients_agree(analytic_gradient(likelihood, x), numerical_gradient(likelihood, x),
                           flux_parameters(AstroModel::Powerlaw, false), 2.0e-5);
  }
}

/// The GPU kernels compute the CPU's derivative table, in the backend's
/// precision, for every astrophysical model and with the veto columns bound.
TEST(AnalyticGradientTest, GpuMatchesCpu) {
  struct Backend {
    const char*                                          name;
    bool                                                 fp64;
    std::function<std::shared_ptr<ana::ic::GpuBackend>()> make;
  };
  std::vector<Backend> backends;
  if (ana::ic::MetalBackend::available())
    backends.push_back({"metal", false, [] { return std::make_shared<ana::ic::MetalBackend>(); }});
  if (ana::ic::CudaBackend::available()) {
    backends.push_back({"cuda-fp32", false, [] { return std::make_shared<ana::ic::CudaBackend>(false); }});
    backends.push_back({"cuda-fp64", true, [] { return std::make_shared<ana::ic::CudaBackend>(true); }});
  }
  if (backends.empty()) GTEST_SKIP() << "no GPU backend is available";

  const Binning binning = gradient_binning();
  // Enough events that bins span several chunks and the gather does real work.
  const io::ic::ICSample sample = gradient_sample(binning, 20000, true);

  for (const Backend& backend_case : backends) {
    const auto backend = backend_case.make();
    for (const AstroModel model : {AstroModel::Powerlaw, AstroModel::BrokenPowerlaw, AstroModel::LogParabola,
                                   AstroModel::PowerlawCutoff}) {
      SCOPED_TRACE(std::string(backend_case.name) + " model " + std::to_string(static_cast<int>(model)));
      io::ic::SampleConfig cfg{.name = "gpu", .binning = binning, .mc_binning = binning};
      cfg.components = {"astro", "conventional_veto", "prompt_veto"};

      SampleLikelihood cpu(sample, cfg, gradient_settings(model), nullptr, true, 0.0);
      SampleLikelihood gpu(sample, cfg, gradient_settings(model), backend->create_session(), true, 0.0);
      prepare_data(cpu);
      prepare_data(gpu);

      const std::vector<double> x      = perturbed_values();
      const std::vector<double> g_cpu  = analytic_gradient(cpu, x);
      const std::vector<double> g_gpu  = analytic_gradient(gpu, x);
      double                    scale  = 0.0;
      for (const double g : g_cpu) scale = std::max(scale, std::fabs(g));
      const double rel = backend_case.fp64 ? 1.0e-9 : 2.0e-4;
      for (int i = 0; i < kNPar; ++i)
        EXPECT_NEAR(g_gpu[i], g_cpu[i], rel * std::max(std::fabs(g_cpu[i]), 1.0e-3 * scale)) << "parameter " << i;
    }
  }
}

/// The Gauss-Newton Hessian is symmetric and positive semi-definite everywhere,
/// and under Poisson at a point where the prediction meets the data it is the
/// exact Hessian (the dropped term is weighted by the per-bin residual).
TEST(AnalyticGradientTest, GaussNewtonHessian) {
  const Binning          binning = gradient_binning();
  const io::ic::ICSample sample  = gradient_sample(binning, 12, false);
  const std::string      tmpl    = write_template("ictests_grad_template_h.txt", binning.total_bins(), 15.0);
  const std::string      grads   = write_gradients("ictests_grad_gradients_h.txt", binning.total_bins());

  for (const LikelihoodCase& lc : kLikelihoods) {
    SCOPED_TRACE(lc.name);
    io::ic::SampleConfig cfg{.name = "hess", .binning = binning, .mc_binning = binning};
    cfg.components          = {"astro", "conventional", "prompt", "muontemplate"};
    cfg.template_file       = tmpl;
    cfg.template_norm_index = P::MuonNorm;
    cfg.gradient_file       = grads;
    SampleLikelihood likelihood(sample, cfg, gradient_settings(AstroModel::Powerlaw), nullptr, lc.say, lc.alpha_offset);
    prepare_data(likelihood);

    std::vector<double> hessian;
    analytic_gradient(likelihood, perturbed_values(), &hessian);
    for (int i = 0; i < kNPar; ++i)
      for (int j = 0; j < i; ++j)
        EXPECT_NEAR(hessian[i * kNPar + j], hessian[j * kNPar + i], 1e-9 * std::fabs(hessian[i * kNPar + j]) + 1e-12);
    // PSD: no direction of negative curvature (checked along every parameter
    // pair, which a sum of clipped PSD blocks satisfies exactly).
    for (int i = 0; i < kNPar; ++i) {
      EXPECT_GE(hessian[i * kNPar + i], -1e-9);
      for (int j = 0; j < i; ++j) {
        const double a = hessian[i * kNPar + i], c = hessian[j * kNPar + j], b = hessian[i * kNPar + j];
        EXPECT_GE(a * c - b * b, -1e-9 * std::max(1.0, a * c)) << i << ' ' << j;
      }
    }
  }

  // Poisson, data equal to the prediction at the evaluation point.
  io::ic::SampleConfig cfg{.name = "hess_exact", .binning = binning, .mc_binning = binning};
  cfg.components = {"astro", "conventional", "prompt"};
  SampleLikelihood likelihood(sample, cfg, gradient_settings(AstroModel::Powerlaw), nullptr, false);
  const std::vector<double> x = perturbed_values();
  ParameterWrapper          at_x(kNPar);
  at_x.reset_parameter(x.data());
  likelihood.generate_asimov(at_x);

  std::vector<double> hessian;
  analytic_gradient(likelihood, x, &hessian);
  for (const int i : flux_parameters(AstroModel::Powerlaw, false)) {
    const double        h  = 1.0e-5 * std::max(1.0, std::fabs(x[i]));
    std::vector<double> up = x, down = x;
    up[i] += h;
    down[i] -= h;
    const std::vector<double> g_up   = analytic_gradient(likelihood, up);
    const std::vector<double> g_down = analytic_gradient(likelihood, down);
    for (const int j : flux_parameters(AstroModel::Powerlaw, false)) {
      const double fd = (g_up[j] - g_down[j]) / (2.0 * h);
      EXPECT_NEAR(hessian[i * kNPar + j], fd, 1e-5 * std::max(1.0, std::fabs(fd))) << i << ' ' << j;
    }
  }
  std::remove(tmpl.c_str());
  std::remove(grads.c_str());
}

/// The per-bin derivatives in every branch say_llh() takes -- ordinary,
/// variance clipped at mu^2, variance <= 0 (Poisson), large alpha -- against
/// finite differences of the bin term itself.
TEST(BinTermDerivativesTest, EveryBranchMatchesFiniteDifferences) {
  using namespace ana::ic::bin_terms;
  struct Case {
    double k, mu, s, a0, rel;
  };
  const Case cases[] = {
      {37.0, 40.0, 50.0, 1.0, 1e-6},    // ordinary, L_Eff
      {37.0, 40.0, 50.0, 0.0, 1e-6},    // ordinary, L_Mean
      {0.0, 3.0, 2.0, 1.0, 1e-6},       // empty bin
      {5.5, 2.0, 0.5, 0.0, 1e-6},       // non-integer (Asimov) count
      {600.0, 610.0, 2.0, 1.0, 1e-5},   // alpha ~ 2e5
      {3.0, 2.0, 5.0, 1.0, 1e-6},       // s > mu^2: clipped
      {3.0, 2.0, 5.0, 0.0, 1e-6},       // clipped, L_Mean
      {4.0, 3.0, -1.0, 1.0, 1e-6},      // s < 0: Poisson
  };
  for (const Case& c : cases) {
    SCOPED_TRACE("k " + std::to_string(c.k) + " mu " + std::to_string(c.mu) + " s " + std::to_string(c.s) +
                 " a0 " + std::to_string(c.a0));
    const double lg  = std::lgamma(c.k + 1.0);
    auto         llh = [&](double mu, double s) { return ana::ic::say_bin_log_likelihood(c.k, mu, s, lg, c.a0); };
    auto         der = [&](double mu, double s, bool second) {
      return say_term(mu, s, 1.0, c.k, c.a0, [&](auto&& f) { if (c.k > 0.0) f(c.k); }, second);
    };

    const TermDerivatives d  = der(c.mu, c.s, true);
    const double          hm = 1e-5 * c.mu;
    const double          hs = 1e-5 * std::fabs(c.s);
    const double          fd_mu = (llh(c.mu + hm, c.s) - llh(c.mu - hm, c.s)) / (2.0 * hm);
    const double          fd_s  = (llh(c.mu, c.s + hs) - llh(c.mu, c.s - hs)) / (2.0 * hs);
    EXPECT_NEAR(d.mu, fd_mu, c.rel * std::max(1.0, std::fabs(fd_mu)));
    EXPECT_NEAR(d.s, fd_s, c.rel * std::max(1.0, std::fabs(fd_s)));

    // Second derivatives against differences of the (now validated) first ones.
    const TermDerivatives up_m = der(c.mu + hm, c.s, false), down_m = der(c.mu - hm, c.s, false);
    const TermDerivatives up_s = der(c.mu, c.s + hs, false), down_s = der(c.mu, c.s - hs, false);
    const double          fd_mumu = (up_m.mu - down_m.mu) / (2.0 * hm);
    const double          fd_mus  = (up_s.mu - down_s.mu) / (2.0 * hs);
    const double          fd_ss   = (up_s.s - down_s.s) / (2.0 * hs);
    EXPECT_NEAR(d.mumu, fd_mumu, 1e-5 * std::max(1.0, std::fabs(fd_mumu)));
    EXPECT_NEAR(d.mus, fd_mus, 1e-5 * std::max(1.0, std::fabs(fd_mus)));
    EXPECT_NEAR(d.ss, fd_ss, 1e-5 * std::max(1.0, std::fabs(fd_ss)));
  }

  // A group of bins sharing mu and s is the sum of its bins.
  const double     counts[3] = {0.0, 4.0, 9.0};
  TermDerivatives  sum;
  for (const double k : counts) {
    const TermDerivatives d = say_term(5.0, 3.0, 1.0, k, 1.0, [&](auto&& f) { if (k > 0.0) f(k); }, true);
    sum.mu += d.mu;
    sum.s += d.s;
    sum.mumu += d.mumu;
    sum.mus += d.mus;
    sum.ss += d.ss;
  }
  const TermDerivatives group = say_term(5.0, 3.0, 3.0, 13.0, 1.0, [&](auto&& f) { f(4.0); f(9.0); }, true);
  EXPECT_NEAR(group.mu, sum.mu, 1e-12 * std::fabs(sum.mu));
  EXPECT_NEAR(group.s, sum.s, 1e-12 * std::fabs(sum.s));
  EXPECT_NEAR(group.mumu, sum.mumu, 1e-12 * std::fabs(sum.mumu));
  EXPECT_NEAR(group.mus, sum.mus, 1e-12 * std::fabs(sum.mus));
  EXPECT_NEAR(group.ss, sum.ss, 1e-12 * std::fabs(sum.ss));

  // Poisson.
  const double k = 7.0, mu = 5.5, lg = std::lgamma(8.0), h = 1e-5;
  const double fd = (ana::ic::poisson_bin_log_likelihood(k, mu + h, lg) - ana::ic::poisson_bin_log_likelihood(k, mu - h, lg)) /
                    (2.0 * h);
  EXPECT_NEAR(poisson_derivatives(mu, 1.0, k).mu, fd, 1e-7);
}
