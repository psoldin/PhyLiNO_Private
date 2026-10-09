#pragma once

#include "../../io/IceCube/ICInputOptions.h"  // io::ic::AstroModel
#include "../../io/IceCube/ICSample.h"
#include "../ParameterWrapper.h"
#include "GpuBackend.h"
#include "GpuBinReduce.h"

#include <array>
#include <memory>
#include <optional>
#include <span>

namespace ana::ic {

  /**
   * Per-bin derivatives of one sample's per-event flux sums, the raw material
   * of the analytic gradient (see SampleLikelihood::accumulate_gradient()).
   *
   * The likelihood sees the per-event flux components through two per-bin sums:
   * the prediction mu_b = sum_i w_i and, under SAY, ssq_b = sum_i w_i^2, where
   *
   *   w_i = a_eff * wa_i + ConvNorm * cw_i + PromptNorm * pw_i
   *
   * and wa_i, cw_i, pw_i are the norm-free astrophysical, conventional and
   * prompt weights PowerlawFlux and AtmosphericFlux reduce. Every derivative of
   * mu_b and ssq_b with respect to a flux parameter is therefore a per-bin sum
   * of per-event quantities, and all of them come out of one sweep over the
   * events -- the same memory traffic as one likelihood evaluation, which is
   * what makes the gradient cost about one evaluation instead of the 2n of a
   * numerical one. The sweep fills `kQuantities` histograms:
   *
   *   SumAstro, DAstro0..2      sum wa_i and sum dwa_i/dtheta_k over the active
   *                             model's shape parameters (astro_shape_parameters())
   *   SumConv, SumPrompt        sum cw_i, sum pw_i
   *   DConv{CR,DG,B0..B3,Veto}  sum dcw_i/dtheta
   *   DPrompt{CR,DG,Veto}       sum dpw_i/dtheta
   *   Ssq*                      sum w_i dw_i/dtheta for every flux parameter, i.e.
   *                             half of d ssq_b / dtheta
   *
   * Forward-folded samples sum over the response matrix instead of the bin's own
   * events, the mu-side terms weighted by the response fraction f and the ssq
   * terms by f^2 -- the same weighting the fold and the folded ssq use.
   *
   * The CPU loop is the reference; the GPU kernels (CUDA in the backend's
   * precision, Metal in FP32) compute the same table and are tested against it.
   * The derivatives only steer the minimizer, so FP32 is enough for them even
   * where the likelihood value itself needs more.
   */
  class FluxGradient {
   public:
    enum Quantity : int {
      SumAstro = 0,
      DAstro0,
      DAstro1,
      DAstro2,
      SumConv,
      SumPrompt,
      DConvCR,
      DConvDG,
      DConvB0,
      DConvB1,
      DConvB2,
      DConvB3,
      DConvVeto,
      DPromptCR,
      DPromptDG,
      DPromptVeto,
      SsqAstroNorm,
      SsqAstro0,
      SsqAstro1,
      SsqAstro2,
      SsqConvNorm,
      SsqPromptNorm,
      SsqCR,
      SsqDG,
      SsqB0,
      SsqB1,
      SsqB2,
      SsqB3,
      SsqVeto,
      kQuantities
    };

    /// Shape parameters the active astrophysical model differentiates in the
    /// DAstro0..2 / SsqAstro0..2 slots, as params::ic indices (-1 = unused slot).
    static std::array<int, 3> astro_shape_parameters(io::ic::AstroModel model) noexcept;

    struct Settings {
      bool               has_astro           = false;
      bool               has_atmo            = false;
      bool               use_veto            = false;
      io::ic::AstroModel astro_model         = io::ic::AstroModel::Powerlaw;
      double             e_ref_gev           = 1.0e5;
      double             reference_index     = 2.0;
      bool               per_type_norm       = false;
      double             conv_e_ref          = 1.0e3;
      double             prompt_e_ref        = 3.8e3;
      double             veto_anchor_energy  = 100.0;
      double             veto_rescale_energy = 100.0;
      bool               use_multi_threading = true;
    };

    /**
     * `n_bins` is the sample's MC bin count. A null `gpu` selects the CPU loop;
     * a forward-folded sample must use it (as its value path does).
     */
    FluxGradient(const io::ic::ICSample& sample, std::size_t n_bins, const Settings& settings,
                 std::shared_ptr<GpuSession> gpu = nullptr);

    /**
     * Fill the table at `parameter`: quantity q of MC bin b at q * n_bins + b.
     * Reads only the parameter values, not their change flags, so it can be
     * called at any point regardless of what the value path cached.
     */
    void compute(const ParameterWrapper& parameter, std::span<double> table);

    [[nodiscard]] std::size_t n_bins() const noexcept { return m_NBins; }

   private:
    const io::ic::ICSample&     m_Sample;
    std::size_t                 m_NBins;
    Settings                    m_Settings;
    std::shared_ptr<GpuSession> m_Gpu;

    std::array<int, 18>         m_hInputs{};  // GPU column handles, kernel input order
    int                         m_hTable = -1;
    std::optional<GpuBinReduce> m_Reduce;

    void prepare_gpu();
    void compute_cpu(const ParameterWrapper& parameter, std::span<double> table) const;
    void compute_gpu(const ParameterWrapper& parameter, std::span<double> table);
  };

}  // namespace ana::ic
