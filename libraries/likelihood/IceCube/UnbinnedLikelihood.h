#pragma once

#include "../../io/IceCube/ICSample.h"
#include "../../io/IceCube/KdeIndex.h"
#include "../../io/IceCube/SampleConfig.h"
#include "GpuBackend.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace ana::ic {

  inline constexpr double kInvSqrt2Pi = 0.39894228040143267794;

  namespace {

    template <typename T>
    [[nodiscard]] inline auto square(T&& t) noexcept {
        return t * t;
    }

  }

  /**
   * Gaussian kernel of width h centred at c, reflected at both ends of [a, b].
   *
   * Reflection is what keeps the extended likelihood self-consistent: the
   * normalisation term is nu = sum_i w_i, which is the integral of the density
   * over the domain only if no kernel mass escapes it. A kernel sitting on a
   * boundary would otherwise leak up to half its mass and bias the fitted
   * normalisation. Second-order images are omitted -- they are below 1e-6 for
   * any bandwidth smaller than the domain.
   */
  [[nodiscard]] inline double reflected_gauss(const double u, const double c, const double h, const double a,
                                              const double b) noexcept {
    const double inv = 1.0 / h;
    auto         g   = [inv](const double d) noexcept {
      return std::exp(-0.5 * square(d * inv));
    };
    return kInvSqrt2Pi * inv * (g(u - c) + g(u + c - 2.0 * a) + g(u - 2.0 * b + c));
  }

  /**
   * The weighted KDE rate density over the MC events, evaluated exactly (up to
   * the index's truncation) at arbitrary query points.
   *
   * Holds only views: coordinates, bandwidths and index live on ICSample and
   * outlive every Fit. The weights are passed per call, because they are the one
   * thing that moves during the fit.
   */
  struct KdeDensity {
    std::span<const double> x_e;
    std::span<const double> x_z;
    std::span<const double> h_e;
    std::span<const double> h_z;
    const io::ic::KdeIndex& index;
    std::array<double, 2>   lo;
    std::array<double, 2>   hi;

    /** lambda(qe, qz) = sum_i w_i K(qe - x_e,i) K(qz - x_z,i). */
    [[nodiscard]] double evaluate(double qe, double qz, std::span<const double> w) const noexcept;

    /** evaluate() at event j's own coordinates, minus event j's own kernel: the
        leave-one-out density that removes the self-density bias of using the MC
        events as quadrature nodes for a density built from those same events. */
    [[nodiscard]] double evaluate_loo(std::size_t j, std::span<const double> w) const noexcept;
  };

  /**
   * Extended unbinned likelihood for one sample:
   *
   *   -2 lnL = 2 * [ nu(theta) - sum_j w_j^A ln lambda_j(theta) ]
   *
   * The query points are the MC events themselves, weighted by their frozen
   * Asimov weights -- the unbinned analogue of copying predicted() into m_Data.
   * The sum has N_MC terms but total weight nu(theta_A), because it is a Monte
   * Carlo estimate of integral(lambda_A ln lambda) with the MC events as
   * quadrature nodes and w_j^A as quadrature weights. Parameter errors therefore
   * scale with the expected event count, not with the MC statistics.
   *
   * No saturated term is subtracted: a constant offset moves neither the minimum
   * nor a delta-chi2 scan, and there is no reference implementation whose
   * convention would have to be matched.
   */
  class UnbinnedLikelihood {
   public:
    UnbinnedLikelihood(const io::ic::ICSample& sample, const io::ic::UnbinnedConfig& cfg,
                       std::shared_ptr<GpuSession> gpu, bool use_multi_threading);

    /** Freeze the Asimov quadrature weights from the nominal per-event weights. */
    void freeze_asimov(std::span<const double> astro, std::span<const double> atmo);

    /** -2 lnL for the current per-event weights. `nu` is sum_i w_i(theta). */
    [[nodiscard]] double llh(std::span<const double> astro, std::span<const double> atmo, double nu);

    /** Sum of the frozen Asimov weights; equals the binned Asimov total. */
    [[nodiscard]] double asimov_total() const noexcept { return m_AsimovTotal; }

    /** Query points actually used (all events, or every thinning-th one). */
    [[nodiscard]] std::size_t n_queries() const noexcept { return m_Queries.size(); }

   private:
    const io::ic::ICSample& m_Sample;
    io::ic::UnbinnedConfig  m_Config;
    bool                    m_UseMultiThreading;

    // Combined per-event weight (astro + atmo), refilled per evaluation so the
    // density reads one contiguous column instead of two spans.
    std::vector<double> m_Weight;

    // Frozen Asimov quadrature weights, one per query point, already scaled by
    // the thinning factor.
    std::vector<double> m_AsimovWeight;
    std::vector<int>    m_Queries;  ///< event indices used as query points
    double              m_AsimovTotal = 0.0;

    std::vector<double> m_Partial;  ///< deterministic chunked-sum scratch

    std::shared_ptr<GpuSession> m_Gpu;  ///< CUDA only; null on the CPU path

    void                     combine_weights(std::span<const double> astro, std::span<const double> atmo);
    [[nodiscard]] KdeDensity density() const noexcept;
  };

}  // namespace ana::ic
