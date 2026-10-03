#include "SampleLikelihood.h"

#include "../../io/IceCube/ICParameter.h"
#include "BinTermDerivatives.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

/**
 * @file
 * @brief SampleLikelihood::accumulate_gradient(): the analytic gradient and
 * Gauss-Newton Hessian of one sample's -2 lnL.
 *
 * The likelihood of MC bin b reads two numbers built from the parameters: the
 * MC total T_b (the prediction before the RA broadcast) and, under SAY, its MC
 * variance S_b. So
 *
 *   d(-2lnL)/dtheta  = sum_b  gT_b dT_b/dtheta + gS_b dS_b/dtheta
 *   H_ij            ~= sum_b  J_bi^T W_b J_bj,     J_b = (dT_b, dS_b)
 *
 * with gT, gS and the 2x2 W_b the first and second derivatives of bin b's term
 * with respect to (T_b, S_b). The columns dT/dtheta and dS/dtheta come from
 * FluxGradient for the flux parameters and directly from the template and
 * detector components; the bin terms are differentiated here, in exactly the
 * branches say_llh() and poisson_llh() evaluate.
 */

namespace ana::ic {

  namespace {

    using namespace bin_terms;

    /**
     * Second derivatives of -2 lnL in (T, S) clipped to their positive
     * semi-definite part. The clip is done in the basis (T, S / scale), with
     * scale ~ T, so that a direction is not judged by the units it happens to
     * carry; a bin whose block is already PSD is returned unchanged.
     */
    void clip_to_psd(double& tt, double& ts, double& ss, const double scale) {
      const double b    = ts * scale;
      const double c    = ss * scale * scale;
      const double half = 0.5 * (tt - c);
      const double r    = std::hypot(half, b);
      const double mean = 0.5 * (tt + c);
      const double low  = mean - r;
      if (low >= 0.0)
        return;
      const double high = mean + r;
      if (high <= 0.0) {
        tt = ts = ss = 0.0;
        return;
      }
      // Eigenvector of `high`: (b, high - tt) or (high - c, b), whichever is larger.
      double vx = b, vy = high - tt;
      if (std::hypot(vx, vy) < std::hypot(high - c, b)) {
        vx = high - c;
        vy = b;
      }
      const double norm = std::hypot(vx, vy);
      if (norm == 0.0) {
        // Diagonal with one negative entry: keep the positive one.
        if (tt < 0.0) tt = 0.0;
        if (ss < 0.0) ss = 0.0;
        ts = 0.0;
        return;
      }
      vx /= norm;
      vy /= norm;
      tt = high * vx * vx;
      ts = high * vx * vy / scale;
      ss = high * vy * vy / (scale * scale);
    }

  }  // namespace

  void SampleLikelihood::accumulate_gradient(const ParameterWrapper& parameter, const std::span<double> gradient,
                                             double* hessian) {
    using namespace params::ic;

    // Prediction and variance at `parameter`; the value itself is not needed.
    static_cast<void>(partial_llh(parameter));

    const std::size_t n_mc   = m_McTotal.size();
    const std::size_t n_par  = gradient.size();
    const bool        second = hessian != nullptr;

    // --- dT/dtheta and dS/dtheta, MC binning ----------------------------------
    std::size_t n_columns = 0;
    auto column = [&](const int index) -> GradientColumn& {
      for (std::size_t c = 0; c < n_columns; ++c)
        if (m_GradColumns[c].parameter == index) return m_GradColumns[c];
      if (m_GradColumns.size() == n_columns) m_GradColumns.emplace_back();
      GradientColumn& col = m_GradColumns[n_columns++];
      col.parameter       = index;
      col.dmu.assign(n_mc, 0.0);
      col.dssq.assign(n_mc, 0.0);
      return col;
    };

    if (m_FluxGradient) {
      m_FluxGradient->compute(parameter, m_GradTable);
      auto q = [&](const int quantity) {
        return std::span<const double>(m_GradTable).subspan(static_cast<std::size_t>(quantity) * n_mc, n_mc);
      };
      // Adds factor * q(quantity) to dT and 2 * q(ssq_quantity) to dS.
      auto add = [&](GradientColumn& col, const double factor, const int quantity) {
        const auto values = q(quantity);
        for (std::size_t b = 0; b < n_mc; ++b) col.dmu[b] += factor * values[b];
      };
      auto add_ssq = [&](GradientColumn& col, const int quantity) {
        const auto values = q(quantity);
        for (std::size_t b = 0; b < n_mc; ++b) col.dssq[b] += 2.0 * values[b];
      };

      using Q              = FluxGradient::Quantity;
      const double a_eff   = m_AstroNormFactor * parameter[AstroNorm];
      const double c_norm  = parameter[ConvNorm];
      const double p_norm  = parameter[PromptNorm];

      if (m_Astro) {
        GradientColumn& norm = column(AstroNorm);
        add(norm, m_AstroNormFactor, Q::SumAstro);
        add_ssq(norm, Q::SsqAstroNorm);
        for (int k = 0; k < 3; ++k) {
          if (m_AstroShapeParameters[k] < 0) continue;
          GradientColumn& shape = column(m_AstroShapeParameters[k]);
          add(shape, a_eff, Q::DAstro0 + k);
          add_ssq(shape, Q::SsqAstro0 + k);
        }
      }
      if (m_Atmo) {
        GradientColumn& conv = column(ConvNorm);
        add(conv, 1.0, Q::SumConv);
        add_ssq(conv, Q::SsqConvNorm);
        GradientColumn& prompt = column(PromptNorm);
        add(prompt, 1.0, Q::SumPrompt);
        add_ssq(prompt, Q::SsqPromptNorm);

        GradientColumn& cr = column(CRGrad);
        add(cr, c_norm, Q::DConvCR);
        add(cr, p_norm, Q::DPromptCR);
        add_ssq(cr, Q::SsqCR);
        GradientColumn& dg = column(DeltaGamma);
        add(dg, c_norm, Q::DConvDG);
        add(dg, p_norm, Q::DPromptDG);
        add_ssq(dg, Q::SsqDG);
        for (int k = 0; k < nBarrParams; ++k) {
          GradientColumn& barr = column(BarrH + k);
          add(barr, c_norm, Q::DConvB0 + k);
          add_ssq(barr, Q::SsqB0 + k);
        }
        if (m_UseVeto) {
          GradientColumn& veto = column(VetoThreshold);
          add(veto, c_norm, Q::DConvVeto);
          add(veto, p_norm, Q::DPromptVeto);
          add_ssq(veto, Q::SsqVeto);
        }
      }
    }

    if (m_Template) {
      GradientColumn& col    = column(m_Template->norm_index());
      const double    norm   = parameter[m_Template->norm_index()];
      const auto      rates  = m_Template->rates();
      const auto      sigmas = m_Template->sigmas();
      for (std::size_t b = 0; b < n_mc; ++b) {
        col.dmu[b] += rates[b];
        col.dssq[b] += 2.0 * norm * sigmas[b] * sigmas[b];
      }
    }

    if (m_Systematics) {
      std::vector<double> dmu(n_mc), dssq(n_mc);
      for (int k = 0; k < nDetSysParams; ++k) {
        m_Systematics->derivatives(parameter, k, dmu, dssq);
        GradientColumn& col = column(DOMEff + k);
        for (std::size_t b = 0; b < n_mc; ++b) {
          col.dmu[b] += dmu[b];
          col.dssq[b] += dssq[b];
        }
      }
    }

    // Under Poisson the variance does not enter the likelihood.
    if (!m_UseSAY)
      for (std::size_t c = 0; c < n_columns; ++c) std::ranges::fill(m_GradColumns[c].dssq, 0.0);

    // --- the bin terms --------------------------------------------------------
    const double n_ra        = static_cast<double>(m_RaBins);
    const double ssq_divisor = n_ra * n_ra;

    // H += J^T W J for one bin: `jm`/`js` are the bin's d mu / d s per column
    // (indexed like `params`), `w*` its clipped -2 lnL second derivatives.
    std::vector<double> u, v;
    auto add_outer = [&](const std::vector<int>& params, const double* jm, const double* js, const double wtt,
                         const double wts, const double wss) {
      const std::size_t n = params.size();
      u.resize(n);
      v.resize(n);
      for (std::size_t i = 0; i < n; ++i) {
        u[i] = wtt * jm[i] + wts * js[i];
        v[i] = wts * jm[i] + wss * js[i];
      }
      for (std::size_t i = 0; i < n; ++i) {
        if (u[i] == 0.0 && v[i] == 0.0) continue;
        double* row = hessian + static_cast<std::size_t>(params[i]) * n_par;
        for (std::size_t j = 0; j < n; ++j) row[params[j]] += u[i] * jm[j] + v[i] * js[j];
      }
    };

    std::vector<int> params(n_columns);
    for (std::size_t c = 0; c < n_columns; ++c) params[c] = m_GradColumns[c].parameter;

    if (m_Galactic.empty()) {
      // One term per RA group: every analysis bin of MC bin b shares mu and s.
      std::vector<double> jm(n_columns), js(n_columns);
      for (std::size_t b = 0; b < n_mc; ++b) {
        const double value   = m_McTotal[b] / n_ra;
        const double k_total = m_GroupDataSum[b];
        TermDerivatives d;
        if (m_UseSAY) {
          const double s = m_McSsq[b] / ssq_divisor;
          if (m_ExpectedAsimov) {
            // Every bin of the group averages over the same counts.
            const WeightedCounts c = expected_set_counts(b);
            d = say_term(value, s, n_ra, k_total, m_SayAlphaOffset,
                         [&](auto&& f) {
                           for (std::size_t j = 0; j < c.k.size(); ++j)
                             if (c.k[j] > 0.0) f(c.k[j], n_ra * c.w[j]);
                         },
                         second);
          } else {
            d = say_term(value, s, n_ra, k_total, m_SayAlphaOffset,
                         [&](auto&& f) {
                           for (std::size_t j = m_NonZeroOffsets[b]; j < m_NonZeroOffsets[b + 1]; ++j)
                             f(m_Data[static_cast<std::size_t>(m_NonZeroData[j])], 1.0);
                         },
                         second);
          }
        } else if (value > 0.0) {
          d = poisson_derivatives(value, n_ra, k_total);
        }

        // d/dT = (d/dmu) / n_ra, d/dS = (d/ds) / n_ra^2, and -2 for -2 lnL.
        const double g_t = -2.0 * d.mu / n_ra;
        const double g_s = -2.0 * d.s / ssq_divisor;
        for (std::size_t c = 0; c < n_columns; ++c)
          gradient[static_cast<std::size_t>(params[c])] += g_t * m_GradColumns[c].dmu[b] + g_s * m_GradColumns[c].dssq[b];

        if (second) {
          double wtt = -2.0 * d.mumu / (n_ra * n_ra);
          double wts = -2.0 * d.mus / (n_ra * ssq_divisor);
          double wss = -2.0 * d.ss / (ssq_divisor * ssq_divisor);
          clip_to_psd(wtt, wts, wss, std::max(m_McTotal[b], 1.0e-12));
          for (std::size_t c = 0; c < n_columns; ++c) {
            jm[c] = m_GradColumns[c].dmu[b];
            js[c] = m_GradColumns[c].dssq[b];
          }
          add_outer(params, jm.data(), js.data(), wtt, wts, wss);
        }
      }
    } else {
      // With galactic templates every analysis bin has its own mu: one term per
      // bin, and the galactic norms are columns of their own (analysis binning,
      // not in the variance).
      std::vector<int> all_params = params;
      for (const TemplateFlux& galactic : m_Galactic) all_params.push_back(galactic.norm_index());
      const std::size_t   n_all = all_params.size();
      std::vector<double> jm(n_all), js(n_all, 0.0);
      const auto          galactic_sum = std::span<const double>(m_GalacticTotal);

      for (std::size_t b = 0; b < n_mc; ++b) {
        const double      value = m_McTotal[b] / n_ra;
        const double      s     = m_McSsq[b] / ssq_divisor;
        const std::size_t base  = b * static_cast<std::size_t>(m_RaBins);
        for (int r = 0; r < m_RaBins; ++r) {
          const std::size_t bin = base + static_cast<std::size_t>(r);
          const double      mu  = value + galactic_sum[bin];
          const double      k   = m_Data[bin];
          TermDerivatives   d;
          if (m_UseSAY) {
            if (m_ExpectedAsimov) {
              const WeightedCounts c = expected_counts(bin);
              d = say_term(mu, s, 1.0, k, m_SayAlphaOffset,
                           [&](auto&& f) {
                             for (std::size_t j = 0; j < c.k.size(); ++j)
                               if (c.k[j] > 0.0) f(c.k[j], c.w[j]);
                           },
                           second);
            } else {
              d = say_term(mu, s, 1.0, k, m_SayAlphaOffset, [&](auto&& f) { if (k > 0.0) f(k, 1.0); }, second);
            }
          } else if (mu > 0.0) {
            d = poisson_derivatives(mu, 1.0, k);
          }

          const double g_mu = -2.0 * d.mu;
          const double g_s  = -2.0 * d.s / ssq_divisor;
          for (std::size_t c = 0; c < n_columns; ++c) {
            jm[c] = m_GradColumns[c].dmu[b] / n_ra;
            js[c] = m_GradColumns[c].dssq[b];
            gradient[static_cast<std::size_t>(params[c])] += g_mu * jm[c] + g_s * js[c];
          }
          for (std::size_t g = 0; g < m_Galactic.size(); ++g) {
            jm[n_columns + g] = m_Galactic[g].rates()[bin];
            gradient[static_cast<std::size_t>(m_Galactic[g].norm_index())] += g_mu * jm[n_columns + g];
          }

          if (second) {
            // js holds dS here, so W's s-entries carry the 1/n_ra^2 of s = S / n_ra^2.
            double wtt = -2.0 * d.mumu;
            double wts = -2.0 * d.mus / ssq_divisor;
            double wss = -2.0 * d.ss / (ssq_divisor * ssq_divisor);
            clip_to_psd(wtt, wts, wss, std::max(m_McTotal[b], 1.0e-12));
            add_outer(all_params, jm.data(), js.data(), wtt, wts, wss);
          }
        }
      }
    }
  }

}  // namespace ana::ic
