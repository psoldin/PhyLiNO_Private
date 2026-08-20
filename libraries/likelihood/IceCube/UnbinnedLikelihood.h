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

  using io::ic::reflected_kernel;

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

    /** Freeze the Asimov quadrature weights from the nominal per-event weights.
        Takes the GPU buffer handles for the same reason llh() does: on CUDA the
        weights never leave the device, so they are gathered there once. */
    void freeze_asimov(std::span<const double> astro, std::span<const double> atmo, int astro_handle = -1,
                       int atmo_handle = -1);

    /**
     * -2 lnL for the current per-event weights. `nu` is sum_i w_i(theta).
     *
     * On the CPU path the weights arrive as spans. On CUDA they never leave the
     * device -- the flux kernels wrote them there -- so the caller passes the
     * buffer handles instead and the spans are empty, exactly as the SAY ssq
     * reduction takes them.
     */
    [[nodiscard]] double llh(std::span<const double> astro, std::span<const double> atmo, double nu,
                             int astro_handle = -1, int atmo_handle = -1);

    /** Sum of the frozen Asimov weights; equals the binned Asimov total. */
    [[nodiscard]] double asimov_total() const noexcept { return m_AsimovTotal; }

    /** Query points actually used (all events, or every thinning-th one). */
    [[nodiscard]] std::size_t n_queries() const noexcept { return m_Sample.kde_queries.size(); }

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
    double              m_AsimovTotal = 0.0;

    std::vector<double> m_Partial;  ///< deterministic chunked-sum scratch

    // CUDA only; null on the CPU path (Metal is rejected in the constructor).
    // Every buffer below holds parameter-independent data owned by ICSample, so
    // the backend's pointer-keyed column cache can dedupe them safely; the
    // per-event weights are the flux kernels' own device buffers and are bound
    // per dispatch instead.
    std::shared_ptr<GpuSession> m_Gpu;
    int                         m_hXe        = -1;
    int                         m_hXz        = -1;
    int                         m_hInvHe     = -1;
    int                         m_hInvHz     = -1;
    int                         m_hPrefactor = -1;
    int                         m_hReachE    = -1;
    int                         m_hReachZ    = -1;
    int                         m_hQueries   = -1;
    int                         m_hBands     = -1;
    int                         m_hCellEvent = -1;
    int                         m_hCellOff   = -1;
    int                         m_hLogLambda = -1;  ///< kernel output, one ln(lambda) per query
    int                         m_hAsimov    = -1;  ///< frozen quadrature weights, gathered once

    // Index data widened to double for upload_column(), which is the only
    // host-to-device entry point the facade offers. Members rather than locals
    // because the column cache keys on the source pointer, so it must stay alive
    // and stable for as long as the session might reuse the upload.
    std::vector<double> m_BandsHost;
    std::vector<double> m_CellEventsHost;
    std::vector<double> m_QueriesHost;

    void                     combine_weights(std::span<const double> astro, std::span<const double> atmo);
    [[nodiscard]] KdeDensity density() const noexcept;

    /** Set up the CUDA kernel and upload every static column. */
    void setup_gpu(const io::ic::ICSample& sample);

    /** sum_j w_j^A ln lambda_j on the device; the reduction itself stays on the
        host so it keeps the CPU path's chunk order. */
    [[nodiscard]] double gpu_log_sum(int astro_handle, int atmo_handle);
  };

}  // namespace ana::ic
