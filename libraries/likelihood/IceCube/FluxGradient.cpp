#include "FluxGradient.h"

#include "../../io/IceCube/ICParameter.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace ana::ic {

  namespace {

    static_assert(params::ic::nBarrParams == 4, "flux_grad unrolls exactly 4 Barr params");

    constexpr double kLn10 = 2.302585092994046;

    // Model codes shared by the host struct and both kernel dialects.
    constexpr int kModelPowerlaw    = 0;
    constexpr int kModelBroken      = 1;
    constexpr int kModelLogParabola = 2;
    constexpr int kModelCutoff      = 3;

    /**
     * Everything about one evaluation that does not vary per event, evaluated
     * once in double on the host. The per-event weights are exactly those of
     * PowerlawFlux and AtmosphericFlux; the extra fields are the scalars their
     * derivatives need on top:
     *
     *   broken power law  pivot = (1e5 / E_break)^pivot_exponent, with
     *                     pivot_exponent gamma_1 above the break and gamma_2
     *                     below it; d ln(pivot) / d gamma_{1,2} is pivot_g1/g2.
     *   veto              d veto_e / d VetoThreshold = dveto.
     */
    struct Coeffs {
      double a_eff;  // the astro norm every astro weight carries
      double ae;     // d a_eff / d AstroNorm: 1, or 0.5 without per-type norms
      double exponent;
      double log_eref;
      double gamma_2;
      double log_ebreak;
      double pivot;
      double pivot_g1;
      double pivot_g2;
      double pivot_exponent;
      double curvature;
      double log_epivot;
      double inv_ecut;
      double conv_norm;
      double prompt_norm;
      double cr;
      double dg;
      double barr[params::ic::nBarrParams];
      double log_eref_conv;
      double log_eref_prompt;
      double veto_e;
      double dveto;
      int    model;
    };

    Coeffs make_coeffs(const ParameterWrapper& parameter, const FluxGradient::Settings& s) {
      using namespace params::ic;
      using enum io::ic::AstroModel;

      Coeffs c{};
      c.ae         = s.per_type_norm ? 1.0 : 0.5;
      c.a_eff      = c.ae * parameter[AstroNorm];
      c.log_eref   = std::log(s.e_ref_gev);
      c.log_epivot = std::log(1.0e5);
      c.pivot      = 1.0;

      switch (s.astro_model) {
        case BrokenPowerlaw: {
          const double g1      = parameter[AstroGamma1];
          const double g2      = parameter[AstroGamma2];
          const double e_break = std::pow(10.0, parameter[AstroEBreak]);
          const bool   above   = 1.0e5 < e_break;
          const double log_r   = std::log(1.0e5 / e_break);
          c.model              = kModelBroken;
          c.exponent           = g1;
          c.gamma_2            = g2;
          c.log_ebreak         = std::log(e_break);
          c.pivot_exponent     = above ? g1 : g2;
          c.pivot              = std::exp(c.pivot_exponent * log_r);
          c.pivot_g1           = above ? log_r : 0.0;
          c.pivot_g2           = above ? 0.0 : log_r;
          break;
        }
        case LogParabola:
          c.model     = kModelLogParabola;
          c.exponent  = s.reference_index - parameter[SpectralIndex];
          c.curvature = parameter[AstroParabolaB] / kLn10;
          break;
        case PowerlawCutoff:
          c.model    = kModelCutoff;
          c.exponent = s.reference_index - parameter[SpectralIndex];
          c.inv_ecut = std::pow(10.0, -parameter[AstroLogECut]);
          break;
        default:
          c.model    = kModelPowerlaw;
          c.exponent = s.reference_index - parameter[SpectralIndex];
      }

      c.conv_norm   = parameter[ConvNorm];
      c.prompt_norm = parameter[PromptNorm];
      c.cr          = parameter[CRGrad];
      c.dg          = parameter[DeltaGamma];
      for (int k = 0; k < nBarrParams; ++k)
        c.barr[k] = parameter[BarrH + k];
      c.log_eref_conv   = std::log(s.conv_e_ref);
      c.log_eref_prompt = std::log(s.prompt_e_ref);
      if (s.use_veto) {
        const double scale = s.veto_rescale_energy * std::pow(10.0, parameter[VetoThreshold]);
        c.veto_e           = scale - s.veto_anchor_energy;
        c.dveto            = scale * kLn10;
      }
      return c;
    }

    // ---------------------------------------------------------------- CPU --

    /// One event's weights and their derivatives, norm-free (see FluxGradient).
    struct EventTerms {
      double wa = 0.0, ka[3] = {0.0, 0.0, 0.0};  // dwa/dtheta_k = ka[k] * wa
      double cw = 0.0, dc[7] = {};               // CR, DG, B0..B3, Veto
      double pw = 0.0, dp[3] = {};               // CR, DG, Veto
    };

    void astro_terms(const io::ic::ICSample& sample, const std::size_t i, const Coeffs& c, EventTerms& t) {
      const double log_e = sample.log_e_true[i];
      const double base  = sample.astro_baseline[i];
      if (c.model == kModelBroken) {
        const double lx    = log_e - c.log_ebreak;
        const bool   below = lx < 0.0;
        const double g     = below ? c.exponent : c.gamma_2;
        const double undo  = sample.e_true[i] * 1.0e-5;
        t.wa               = base * c.pivot * std::exp(-g * lx) * undo * undo;
        t.ka[0]            = (below ? -lx : 0.0) + c.pivot_g1;
        t.ka[1]            = (below ? 0.0 : -lx) + c.pivot_g2;
        t.ka[2]            = (g - c.pivot_exponent) * kLn10;
        return;
      }
      const double lr  = log_e - c.log_eref;
      double       arg = c.exponent * lr;
      if (c.model == kModelLogParabola) {
        const double lp = log_e - c.log_epivot;
        arg -= c.curvature * lp * lp;
        t.ka[1] = -lp * lp / kLn10;
      } else if (c.model == kModelCutoff) {
        const double et = sample.e_true[i] * c.inv_ecut;
        arg -= et;
        t.ka[1] = et * kLn10;
      }
      t.wa    = base * std::exp(arg);
      t.ka[0] = -lr;
    }

    void atmo_terms(const io::ic::ICSample& sample, const std::size_t i, const Coeffs& c, const bool use_veto,
                    EventTerms& t) {
      const double log_e = sample.log_e_true[i];

      const double cb = sample.conv_baseline[i];
      if (cb > 0.0) {
        const double alt = sample.conv_alt[i];
        const double lin = cb + c.cr * (alt - cb);
        double       x[4], B[4];
        for (int k = 0; k < 4; ++k) {
          x[k] = sample.barr_conv[k][i] / cb;
          B[k] = 1.0 + c.barr[k] * x[k];
        }
        double ev    = std::exp(-c.dg * (log_e - c.log_eref_conv));
        double slope = 0.0;
        if (use_veto) {
          const double va = sample.veto_conv[0][i], vb = sample.veto_conv[1][i], vc = sample.veto_conv[2][i];
          ev *= std::exp2(3.321928094887362 * (va + vb * c.veto_e + vc * c.veto_e * c.veto_e));
          slope = kLn10 * (vb + 2.0 * vc * c.veto_e) * c.dveto;
        }
        const double b01  = B[0] * B[1];
        const double b23  = B[2] * B[3];
        const double rest = b01 * b23 * ev;
        const double le   = lin * ev;
        t.cw              = lin * rest;
        t.dc[0]           = (alt - cb) * rest;
        t.dc[1]           = -(log_e - c.log_eref_conv) * t.cw;
        t.dc[2]           = le * x[0] * B[1] * b23;
        t.dc[3]           = le * x[1] * B[0] * b23;
        t.dc[4]           = le * x[2] * b01 * B[3];
        t.dc[5]           = le * x[3] * b01 * B[2];
        t.dc[6]           = t.cw * slope;
      }

      const double pb = sample.prompt_baseline[i];
      if (pb > 0.0) {
        const double alt   = sample.prompt_alt[i];
        const double lin   = pb + c.cr * (alt - pb);
        double       ev    = std::exp(-c.dg * (log_e - c.log_eref_prompt));
        double       slope = 0.0;
        if (use_veto) {
          const double va = sample.veto_prompt[0][i], vb = sample.veto_prompt[1][i], vc = sample.veto_prompt[2][i];
          ev *= std::exp2(3.321928094887362 * (va + vb * c.veto_e + vc * c.veto_e * c.veto_e));
          slope = kLn10 * (vb + 2.0 * vc * c.veto_e) * c.dveto;
        }
        t.pw    = lin * ev;
        t.dp[0] = (alt - pb) * ev;
        t.dp[1] = -(log_e - c.log_eref_prompt) * t.pw;
        t.dp[2] = t.pw * slope;
      }
    }

    /// Adds one event to a bin's accumulators; f weights the mu-side terms and
    /// f^2 the ssq terms (f = 1 unless the sample is forward folded).
    void accumulate(double* acc, const EventTerms& t, const Coeffs& c, const double f, const double f2) {
      using Q = FluxGradient::Quantity;
      const double awa = c.a_eff * t.wa;
      const double w   = awa + c.conv_norm * t.cw + c.prompt_norm * t.pw;

      acc[Q::SumAstro] += f * t.wa;
      for (int k = 0; k < 3; ++k)
        acc[Q::DAstro0 + k] += f * t.ka[k] * t.wa;
      acc[Q::SumConv] += f * t.cw;
      acc[Q::SumPrompt] += f * t.pw;
      for (int k = 0; k < 7; ++k)
        acc[Q::DConvCR + k] += f * t.dc[k];
      for (int k = 0; k < 3; ++k)
        acc[Q::DPromptCR + k] += f * t.dp[k];

      const double fw = f2 * w;
      acc[Q::SsqAstroNorm] += fw * c.ae * t.wa;
      for (int k = 0; k < 3; ++k)
        acc[Q::SsqAstro0 + k] += fw * t.ka[k] * awa;
      acc[Q::SsqConvNorm] += fw * t.cw;
      acc[Q::SsqPromptNorm] += fw * t.pw;
      acc[Q::SsqCR] += fw * (c.conv_norm * t.dc[0] + c.prompt_norm * t.dp[0]);
      acc[Q::SsqDG] += fw * (c.conv_norm * t.dc[1] + c.prompt_norm * t.dp[1]);
      for (int k = 0; k < 4; ++k)
        acc[Q::SsqB0 + k] += fw * c.conv_norm * t.dc[2 + k];
      acc[Q::SsqVeto] += fw * (c.conv_norm * t.dc[6] + c.prompt_norm * t.dp[2]);
    }

    // ---------------------------------------------------------------- GPU --

    // Host mirror of the kernels' GradParams: every real first, then the ints,
    // so the FP64 layout has no padding between fields that a hand-written
    // kernel struct could disagree about.
    template <class R>
    struct GradParamsT {
      using Scalar = R;
      R   a_eff, ae, exponent, log_eref, gamma_2, log_ebreak, pivot, pivot_g1, pivot_g2, pivot_exponent;
      R   curvature, log_epivot, inv_ecut;
      R   conv_norm, prompt_norm, cr, dg, barr0, barr1, barr2, barr3;
      R   log_eref_conv, log_eref_prompt, veto_e, dveto;
      int model, has_astro, has_atmo, use_veto, n_chunks;
    };

    template <class R>
    GradParamsT<R> make_params(const Coeffs& c, const FluxGradient::Settings& s, const int n_chunks) {
      GradParamsT<R> p{};
      p.a_eff           = static_cast<R>(c.a_eff);
      p.ae              = static_cast<R>(c.ae);
      p.exponent        = static_cast<R>(c.exponent);
      p.log_eref        = static_cast<R>(c.log_eref);
      p.gamma_2         = static_cast<R>(c.gamma_2);
      p.log_ebreak      = static_cast<R>(c.log_ebreak);
      p.pivot           = static_cast<R>(c.pivot);
      p.pivot_g1        = static_cast<R>(c.pivot_g1);
      p.pivot_g2        = static_cast<R>(c.pivot_g2);
      p.pivot_exponent  = static_cast<R>(c.pivot_exponent);
      p.curvature       = static_cast<R>(c.curvature);
      p.log_epivot      = static_cast<R>(c.log_epivot);
      p.inv_ecut        = static_cast<R>(c.inv_ecut);
      p.conv_norm       = static_cast<R>(c.conv_norm);
      p.prompt_norm     = static_cast<R>(c.prompt_norm);
      p.cr              = static_cast<R>(c.cr);
      p.dg              = static_cast<R>(c.dg);
      p.barr0           = static_cast<R>(c.barr[0]);
      p.barr1           = static_cast<R>(c.barr[1]);
      p.barr2           = static_cast<R>(c.barr[2]);
      p.barr3           = static_cast<R>(c.barr[3]);
      p.log_eref_conv   = static_cast<R>(c.log_eref_conv);
      p.log_eref_prompt = static_cast<R>(c.log_eref_prompt);
      p.veto_e          = static_cast<R>(c.veto_e);
      p.dveto           = static_cast<R>(c.dveto);
      p.model           = c.model;
      p.has_astro       = s.has_astro ? 1 : 0;
      p.has_atmo        = s.has_atmo ? 1 : 0;
      p.use_veto        = s.use_veto ? 1 : 0;
      p.n_chunks        = n_chunks;
      return p;
    }

    // The quantity indices, spelled out for the kernel sources so both dialects
    // index the accumulators exactly as the host reads the table back.
    std::string quantity_defines() {
      using Q = FluxGradient::Quantity;
      const std::pair<const char*, int> names[] = {
          {"SumAstro", Q::SumAstro},       {"DAstro0", Q::DAstro0},       {"DAstro1", Q::DAstro1},
          {"DAstro2", Q::DAstro2},         {"SumConv", Q::SumConv},       {"SumPrompt", Q::SumPrompt},
          {"DConvCR", Q::DConvCR},         {"DConvDG", Q::DConvDG},       {"DConvB0", Q::DConvB0},
          {"DConvB1", Q::DConvB1},         {"DConvB2", Q::DConvB2},       {"DConvB3", Q::DConvB3},
          {"DConvVeto", Q::DConvVeto},     {"DPromptCR", Q::DPromptCR},   {"DPromptDG", Q::DPromptDG},
          {"DPromptVeto", Q::DPromptVeto}, {"SsqAstroNorm", Q::SsqAstroNorm}, {"SsqAstro0", Q::SsqAstro0},
          {"SsqAstro1", Q::SsqAstro1},     {"SsqAstro2", Q::SsqAstro2},   {"SsqConvNorm", Q::SsqConvNorm},
          {"SsqPromptNorm", Q::SsqPromptNorm}, {"SsqCR", Q::SsqCR},       {"SsqDG", Q::SsqDG},
          {"SsqB0", Q::SsqB0},             {"SsqB1", Q::SsqB1},           {"SsqB2", Q::SsqB2},
          {"SsqB3", Q::SsqB3},             {"SsqVeto", Q::SsqVeto},
      };
      std::string out = "#define NQ " + std::to_string(Q::kQuantities) + "\n";
      for (const auto& [name, index] : names)
        out += std::string("#define Q_") + name + ' ' + std::to_string(index) + '\n';
      return out;
    }

    // The per-event body, written once against `real` and the REXP/REXP2
    // macros: the CUDA prelude defines them for float or double, the Metal one
    // below for float. Reads the event's columns, fills the norm-free weights
    // and derivatives, and adds them to acc[] exactly as accumulate() does on
    // the host. Absent components are skipped on the uniform has_* flags, so
    // their (dummy) buffers are never read.
    constexpr const char* kEventBody = R"BODY(
          const real lnE = log_e_true[i];
          real wa = 0, ka0 = 0, ka1 = 0, ka2 = 0;
          if (p.has_astro) {
            const real base = astro_base[i];
            if (p.model == 1) {
              const real lx    = lnE - p.log_ebreak;
              const bool below = lx < (real)0;
              const real g     = below ? p.exponent : p.gamma_2;
              const real undo  = e_true[i] * (real)1.0e-5;
              wa  = base * p.pivot * REXP(-g * lx) * undo * undo;
              ka0 = (below ? -lx : (real)0) + p.pivot_g1;
              ka1 = (below ? (real)0 : -lx) + p.pivot_g2;
              ka2 = (g - p.pivot_exponent) * kGradLn10;
            } else {
              const real lr = lnE - p.log_eref;
              real arg = p.exponent * lr;
              if (p.model == 2) {
                const real lp = lnE - p.log_epivot;
                arg -= p.curvature * lp * lp;
                ka1 = -lp * lp / kGradLn10;
              } else if (p.model == 3) {
                const real et = e_true[i] * p.inv_ecut;
                arg -= et;
                ka1 = et * kGradLn10;
              }
              wa  = base * REXP(arg);
              ka0 = -lr;
            }
          }

          real cw = 0, dccr = 0, dcdg = 0, dcb0 = 0, dcb1 = 0, dcb2 = 0, dcb3 = 0, dcv = 0;
          real pw = 0, dpcr = 0, dpdg = 0, dpv = 0;
          if (p.has_atmo) {
            const real cb = conv_base[i];
            if (cb > (real)0) {
              const real alt = conv_alt[i];
              const real lin = cb + p.cr * (alt - cb);
              const real inv = (real)1 / cb;  // one division instead of four
              const real x0 = barr0[i] * inv, x1 = barr1[i] * inv, x2 = barr2[i] * inv, x3 = barr3[i] * inv;
              const real B0 = (real)1 + p.barr0 * x0, B1 = (real)1 + p.barr1 * x1;
              const real B2 = (real)1 + p.barr2 * x2, B3 = (real)1 + p.barr3 * x3;
              real ev = REXP(-p.dg * (lnE - p.log_eref_conv));
              real slope = 0;
              if (p.use_veto) {
                const real vb = veto_conv_b[i], vc = veto_conv_c[i];
                ev *= REXP2(kGradLog2Of10 * (veto_conv_a[i] + vb * p.veto_e + vc * p.veto_e * p.veto_e));
                slope = kGradLn10 * (vb + (real)2 * vc * p.veto_e) * p.dveto;
              }
              const real b01 = B0 * B1, b23 = B2 * B3;
              const real rest = b01 * b23 * ev;
              const real le = lin * ev;
              cw   = lin * rest;
              dccr = (alt - cb) * rest;
              dcdg = -(lnE - p.log_eref_conv) * cw;
              dcb0 = le * x0 * B1 * b23;
              dcb1 = le * x1 * B0 * b23;
              dcb2 = le * x2 * b01 * B3;
              dcb3 = le * x3 * b01 * B2;
              dcv  = cw * slope;
            }
            const real pb = prompt_base[i];
            if (pb > (real)0) {
              const real alt = prompt_alt[i];
              const real lin = pb + p.cr * (alt - pb);
              real ev = REXP(-p.dg * (lnE - p.log_eref_prompt));
              real slope = 0;
              if (p.use_veto) {
                const real vb = veto_pr_b[i], vc = veto_pr_c[i];
                ev *= REXP2(kGradLog2Of10 * (veto_pr_a[i] + vb * p.veto_e + vc * p.veto_e * p.veto_e));
                slope = kGradLn10 * (vb + (real)2 * vc * p.veto_e) * p.dveto;
              }
              pw   = lin * ev;
              dpcr = (alt - pb) * ev;
              dpdg = -(lnE - p.log_eref_prompt) * pw;
              dpv  = pw * slope;
            }
          }

          const real awa = p.a_eff * wa;
          const real w   = awa + p.conv_norm * cw + p.prompt_norm * pw;
          acc[Q_SumAstro]  += wa;
          acc[Q_DAstro0]   += ka0 * wa;
          acc[Q_DAstro1]   += ka1 * wa;
          acc[Q_DAstro2]   += ka2 * wa;
          acc[Q_SumConv]   += cw;
          acc[Q_SumPrompt] += pw;
          acc[Q_DConvCR]   += dccr;
          acc[Q_DConvDG]   += dcdg;
          acc[Q_DConvB0]   += dcb0;
          acc[Q_DConvB1]   += dcb1;
          acc[Q_DConvB2]   += dcb2;
          acc[Q_DConvB3]   += dcb3;
          acc[Q_DConvVeto] += dcv;
          acc[Q_DPromptCR] += dpcr;
          acc[Q_DPromptDG] += dpdg;
          acc[Q_DPromptVeto] += dpv;
          acc[Q_SsqAstroNorm]  += w * p.ae * wa;
          acc[Q_SsqAstro0]     += w * ka0 * awa;
          acc[Q_SsqAstro1]     += w * ka1 * awa;
          acc[Q_SsqAstro2]     += w * ka2 * awa;
          acc[Q_SsqConvNorm]   += w * cw;
          acc[Q_SsqPromptNorm] += w * pw;
          acc[Q_SsqCR] += w * (p.conv_norm * dccr + p.prompt_norm * dpcr);
          acc[Q_SsqDG] += w * (p.conv_norm * dcdg + p.prompt_norm * dpdg);
          acc[Q_SsqB0] += w * p.conv_norm * dcb0;
          acc[Q_SsqB1] += w * p.conv_norm * dcb1;
          acc[Q_SsqB2] += w * p.conv_norm * dcb2;
          acc[Q_SsqB3] += w * p.conv_norm * dcb3;
          acc[Q_SsqVeto] += w * (p.conv_norm * dcv + p.prompt_norm * dpv);
    )BODY";

    // Field list of GradParams in both dialects, matching GradParamsT.
    constexpr const char* kParamsFields = R"FIELDS(
        real a_eff; real ae; real exponent; real log_eref; real gamma_2; real log_ebreak; real pivot;
        real pivot_g1; real pivot_g2; real pivot_exponent; real curvature; real log_epivot; real inv_ecut;
        real conv_norm; real prompt_norm; real cr; real dg; real barr0; real barr1; real barr2; real barr3;
        real log_eref_conv; real log_eref_prompt; real veto_e; real dveto;
        int model; int has_astro; int has_atmo; int use_veto; int n_chunks;
    )FIELDS";

    // CUDA: one block per chunk, the event loop above, then a warp-shuffle
    // reduction of all NQ accumulators and one partial per quantity per chunk.
    std::string cuda_body() {
      return quantity_defines() + "struct GradParams {" + kParamsFields + R"CUDA(};
      __device__ const real kGradLn10     = (real)2.302585092994046;
      __device__ const real kGradLog2Of10 = (real)3.321928094887362;

      extern "C" __global__ void flux_grad(
          const real* e_true, const real* log_e_true, const real* astro_base,
          const real* conv_base, const real* conv_alt, const real* prompt_base, const real* prompt_alt,
          const real* barr0, const real* barr1, const real* barr2, const real* barr3,
          const real* veto_conv_a, const real* veto_conv_b, const real* veto_conv_c,
          const real* veto_pr_a, const real* veto_pr_b, const real* veto_pr_c,
          const unsigned int* chunk_offsets,
          GradParams p,
          real* partial)
      {
        const unsigned int chunk    = blockIdx.x;
        const unsigned int tid      = threadIdx.x;
        const unsigned int nthreads = blockDim.x;
        const unsigned int start    = chunk_offsets[chunk];
        const unsigned int end      = chunk_offsets[chunk + 1];

        real acc[NQ];
        #pragma unroll
        for (int q = 0; q < NQ; ++q) acc[q] = 0;

        for (unsigned int i = start + tid; i < end; i += nthreads) {
)CUDA" + std::string(kEventBody) +
             R"CUDA(
        }

        __shared__ real sdata[(256 / 32) * NQ];
        const unsigned int lane = tid & 31u;
        const unsigned int warp = tid >> 5;
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
          real v = acc[q];
          for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, offset);
          if (lane == 0) sdata[warp * NQ + q] = v;
        }
        __syncthreads();
        if (tid < NQ) {
          real v = 0;
          for (unsigned int k = 0; k < (nthreads >> 5); ++k) v += sdata[k * NQ + tid];
          partial[tid * (unsigned int)p.n_chunks + chunk] = v;
        }
      }
    )CUDA";
    }

    // Metal: the same event loop in float, reduced with the threadgroup tree
    // the other kernels use, one quantity after the other.
    std::string metal_body() {
      std::string fields = kParamsFields;
      // MSL has no `real`: spell the fields out as float.
      for (std::size_t pos = 0; (pos = fields.find("real ", pos)) != std::string::npos;)
        fields.replace(pos, 5, "float ");
      return quantity_defines() + "#define real float\n#define REXP exp\n#define REXP2 exp2\n" +
             "struct GradParams {" + fields + R"METAL(};
      constant float kGradLn10     = 2.302585092994046f;
      constant float kGradLog2Of10 = 3.321928094887362f;

      kernel void flux_grad(
          device const float* e_true        [[buffer(0)]],
          device const float* log_e_true    [[buffer(1)]],
          device const float* astro_base    [[buffer(2)]],
          device const float* conv_base     [[buffer(3)]],
          device const float* conv_alt      [[buffer(4)]],
          device const float* prompt_base   [[buffer(5)]],
          device const float* prompt_alt    [[buffer(6)]],
          device const float* barr0         [[buffer(7)]],
          device const float* barr1         [[buffer(8)]],
          device const float* barr2         [[buffer(9)]],
          device const float* barr3         [[buffer(10)]],
          device const float* veto_conv_a   [[buffer(11)]],
          device const float* veto_conv_b   [[buffer(12)]],
          device const float* veto_conv_c   [[buffer(13)]],
          device const float* veto_pr_a     [[buffer(14)]],
          device const float* veto_pr_b     [[buffer(15)]],
          device const float* veto_pr_c     [[buffer(16)]],
          device const uint*  chunk_offsets [[buffer(17)]],
          constant GradParams& p            [[buffer(18)]],
          device float*       partial       [[buffer(19)]],
          uint chunk [[threadgroup_position_in_grid]],
          uint tid [[thread_position_in_threadgroup]])
      {
        const uint start = chunk_offsets[chunk];
        const uint end   = chunk_offsets[chunk + 1];

        float acc[NQ];
        for (int q = 0; q < NQ; ++q) acc[q] = 0.0f;

        for (uint i = start + tid; i < end; i += kThreadsPerGroup) {
)METAL" + std::string(kEventBody) +
             R"METAL(
        }

        threadgroup float shared[kThreadsPerGroup];
        for (int q = 0; q < NQ; ++q) {
          shared[tid] = acc[q];
          threadgroup_barrier(mem_flags::mem_threadgroup);
          for (uint s = kThreadsPerGroup / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] += shared[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }
          if (tid == 0) partial[(uint)q * (uint)p.n_chunks + chunk] = shared[0];
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
    )METAL";
    }

  }  // namespace

  std::array<int, 3> FluxGradient::astro_shape_parameters(const io::ic::AstroModel model) noexcept {
    using namespace params::ic;
    using enum io::ic::AstroModel;
    switch (model) {
      case BrokenPowerlaw:
        return {AstroGamma1, AstroGamma2, AstroEBreak};
      case LogParabola:
        return {SpectralIndex, AstroParabolaB, -1};
      case PowerlawCutoff:
        return {SpectralIndex, AstroLogECut, -1};
      default:
        return {SpectralIndex, -1, -1};
    }
  }

  FluxGradient::FluxGradient(const io::ic::ICSample& sample, const std::size_t n_bins, const Settings& settings,
                             std::shared_ptr<GpuSession> gpu)
    : m_Sample(sample)
    , m_NBins(n_bins)
    , m_Settings(settings)
    , m_Gpu(std::move(gpu)) {
    if (sample.bin_offsets.size() != n_bins + 1)
      throw std::runtime_error("FluxGradient: the sample's bin layout covers " +
                               std::to_string(sample.bin_offsets.size()) + " - 1 bins, expected " +
                               std::to_string(n_bins));
    if (m_Gpu && !sample.response.empty())
      throw std::runtime_error("FluxGradient: forward-folded samples are differentiated on the CPU only");
  }

  void FluxGradient::prepare_gpu() {
    // Done on the first compute() rather than at construction, so a fit that
    // never asks for the gradient (--gradient false) pays for no kernel
    // compile. The column uploads all deduplicate against the flux components'.
    const io::ic::ICSample& sample   = m_Sample;
    const Settings&         settings = m_Settings;
    const std::size_t       n_bins   = m_NBins;

    const std::string src = m_Gpu->language() == GpuLanguage::Cuda
                                ? cuda_kernel_source(m_Gpu->is_fp64(), cuda_body().c_str())
                                : metal_kernel_source(metal_body().c_str());
    m_Gpu->ensure_kernel("flux_grad", src.c_str());

    // Every input must be bound, read or not (Metal faults on an unbound buffer
    // a kernel might touch); an absent column is bound to log E, which every
    // sample has and which the kernel never reads in its place.
    const std::size_t M     = sample.size();
    auto              col   = [&](const std::vector<double>& c) { return m_Gpu->upload_column(c.data(), M); };
    const int         log_e = col(sample.log_e_true);
    auto              opt   = [&](bool present, const std::vector<double>& c) { return present ? col(c) : log_e; };

    const bool astro = settings.has_astro;
    const bool atmo  = settings.has_atmo;
    const bool veto  = atmo && settings.use_veto;
    const bool e_col = astro && (settings.astro_model == io::ic::AstroModel::BrokenPowerlaw ||
                                settings.astro_model == io::ic::AstroModel::PowerlawCutoff);

    m_Reduce.emplace(m_Gpu, sample, n_bins, static_cast<std::size_t>(kQuantities));
    m_hInputs = {opt(e_col, sample.e_true),
                 log_e,
                 opt(astro, sample.astro_baseline),
                 opt(atmo, sample.conv_baseline),
                 opt(atmo, sample.conv_alt),
                 opt(atmo, sample.prompt_baseline),
                 opt(atmo, sample.prompt_alt),
                 opt(atmo, sample.barr_conv[0]),
                 opt(atmo, sample.barr_conv[1]),
                 opt(atmo, sample.barr_conv[2]),
                 opt(atmo, sample.barr_conv[3]),
                 opt(veto, sample.veto_conv[0]),
                 opt(veto, sample.veto_conv[1]),
                 opt(veto, sample.veto_conv[2]),
                 opt(veto, sample.veto_prompt[0]),
                 opt(veto, sample.veto_prompt[1]),
                 opt(veto, sample.veto_prompt[2]),
                 m_Reduce->chunk_offsets()};
    m_hTable = m_Gpu->alloc_output(static_cast<std::size_t>(kQuantities) * n_bins);
  }

  void FluxGradient::compute(const ParameterWrapper& parameter, const std::span<double> table) {
    if (table.size() != static_cast<std::size_t>(kQuantities) * m_NBins)
      throw std::invalid_argument("FluxGradient::compute: table has " + std::to_string(table.size()) +
                                  " entries, expected " + std::to_string(kQuantities * m_NBins));
    if (m_Gpu)
      compute_gpu(parameter, table);
    else
      compute_cpu(parameter, table);
  }

  void FluxGradient::compute_cpu(const ParameterWrapper& parameter, const std::span<double> table) const {
    const Coeffs                  c        = make_coeffs(parameter, m_Settings);
    const io::ic::ResponseMatrix& response = m_Sample.response;
    const bool                    folded   = !response.empty();
    const auto&                   off      = folded ? response.bin_offsets : m_Sample.bin_offsets;
    const int                     n_bins   = static_cast<int>(m_NBins);
    const bool                    astro    = m_Settings.has_astro;
    const bool                    atmo     = m_Settings.has_atmo;
    const bool                    veto     = m_Settings.use_veto;

    #pragma omp parallel for schedule(guided) if (m_Settings.use_multi_threading)
    for (int b = 0; b < n_bins; ++b) {
      double acc[kQuantities] = {};
      for (std::size_t k = off[static_cast<std::size_t>(b)]; k < off[static_cast<std::size_t>(b) + 1]; ++k) {
        const std::size_t i = folded ? static_cast<std::size_t>(response.events[k]) : k;
        const double      f = folded ? static_cast<double>(response.fractions[k]) : 1.0;
        EventTerms        t;
        if (astro) astro_terms(m_Sample, i, c, t);
        if (atmo) atmo_terms(m_Sample, i, c, veto, t);
        accumulate(acc, t, c, f, f * f);
      }
      for (int q = 0; q < kQuantities; ++q)
        table[static_cast<std::size_t>(q) * m_NBins + static_cast<std::size_t>(b)] = acc[q];
    }
  }

  void FluxGradient::compute_gpu(const ParameterWrapper& parameter, const std::span<double> table) {
    if (!m_Reduce)
      prepare_gpu();
    const Coeffs c        = make_coeffs(parameter, m_Settings);
    const int    n_chunks = static_cast<int>(m_Reduce->n_chunks());

    auto run = [&](auto params) {
      m_Gpu->dispatch("flux_grad", m_hInputs.data(), static_cast<int>(m_hInputs.size()), &params, sizeof(params),
                      m_Reduce->partial(), /*per_event=*/-1, m_Reduce->n_chunks());
      m_Reduce->gather_all(m_hTable);
    };

    if (m_Gpu->is_fp64()) {
      run(make_params<double>(c, m_Settings, n_chunks));
      const double* out = m_Gpu->contents_f64(m_hTable);
      for (std::size_t j = 0; j < table.size(); ++j) table[j] = out[j];
    } else {
      run(make_params<float>(c, m_Settings, n_chunks));
      const float* out = m_Gpu->contents(m_hTable);
      for (std::size_t j = 0; j < table.size(); ++j) table[j] = static_cast<double>(out[j]);
    }
  }

}  // namespace ana::ic
