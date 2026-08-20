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
   * One axis of the product kernel, without the 1/(h sqrt(2pi)) prefactor, which
   * the caller carries per event instead of recomputing it per pair.
   *
   * Same value as reflected_gauss() up to that prefactor, but each image term is
   * skipped when its own exponent is already past 8 sigma (a factor 1e-14). An
   * image's argument is ((u - a) + (c - a)) / h, i.e. the query's distance to the
   * wall plus the event's, so it exceeds the direct term's |u - c| / h always:
   * the images matter only when query and event are BOTH near the same
   * boundary, which is a small minority of pairs. Two compares replace four
   * exponentials in the common case.
   */
  [[nodiscard]] inline double reflected_kernel(const double u, const double c, const double inv_h,
                                               const double a, const double b) noexcept {
    constexpr double kImageCutoff = 8.0;

    double       sum     = std::exp(-0.5 * square((u - c) * inv_h));
    const double low_arg = (u + c - 2.0 * a) * inv_h;
    if (low_arg < kImageCutoff) sum += std::exp(-0.5 * square(low_arg));
    const double high_arg = (2.0 * b - u - c) * inv_h;
    if (high_arg < kImageCutoff) sum += std::exp(-0.5 * square(high_arg));

    return sum;
  }

  /**
   * Per-event constants derived from the bandwidths, hoisted out of the innermost
   * loop: the reciprocal widths, the product prefactor of the two axes, and the
   * truncation reach on each axis.
   *
   * Built once per SampleLikelihood rather than stored on ICSample, because they
   * depend on the sample's configured truncation as well as on its columns.
   */
  struct KdeScratch {
    std::vector<double> inv_h_e;
    std::vector<double> inv_h_z;
    std::vector<double> prefactor;  ///< 1 / (2 pi h_e h_z)
    std::vector<double> reach_e;    ///< n_sigma * h_e
    std::vector<double> reach_z;    ///< n_sigma * h_z

    [[nodiscard]] static KdeScratch build(std::span<const double> h_e, std::span<const double> h_z,
                                          double n_sigma);
  };

  /**
   * The weighted KDE rate density over the MC events, evaluated exactly (up to
   * the index's truncation) at arbitrary query points.
   *
   * Holds only views: coordinates, bandwidths and index live on ICSample and
   * outlive every Fit; the derived constants live in a KdeScratch the caller
   * owns. The weights are passed per call, because they are the one thing that
   * moves during the fit.
   */
  struct KdeDensity {
    std::span<const double> x_e;
    std::span<const double> x_z;
    std::span<const double> inv_h_e;
    std::span<const double> inv_h_z;
    std::span<const double> prefactor;
    std::span<const double> reach_e;
    std::span<const double> reach_z;
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

    KdeScratch          m_Scratch;  ///< per-event kernel constants, built once here
    std::vector<double> m_Partial;  ///< deterministic chunked-sum scratch

    std::shared_ptr<GpuSession> m_Gpu;  ///< CUDA only; null on the CPU path

    void                     combine_weights(std::span<const double> astro, std::span<const double> atmo);
    [[nodiscard]] KdeDensity density() const noexcept;
  };

}  // namespace ana::ic
