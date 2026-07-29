#pragma once

#include "Definitions.h"

// Eigen include
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

// STL include
#include <algorithm>
#include <functional>
#include <span>
#include <utility>

namespace ana::dc {

  namespace {
    template <typename T>
    inline auto pow_2(T&& t) noexcept {
      return t * t;
    }

    inline auto make_spectrum(std::span<const double> shape) noexcept {
      return Eigen::Map<const Eigen::VectorXd>(shape.data(), shape.size());
    }
  }  // namespace

  /**
   * @brief Caches the eigendecomposition of a de-fractionalised covariance matrix so that
   *        calculate_spectrum() does not have to repeat the expensive eigensolve on every call.
   *
   * The bin shifts are the result of de-fractionalising the covariance matrix as
   * `defracCovMatrix = covMatrix * rate^2 * spectrum_i * spectrum_j`, eigen-decomposing it, clipping
   * negative eigenvalues and Cholesky-decomposing the reconstructed matrix. The eigensolve
   * (iterative, O(n^3)) only depends on `spectrum` and `covMatrix`, never on the nuisance parameters
   * that end up multiplied onto the Cholesky factor, so it does not need to be repeated whenever only
   * those change.
   *
   * For the accidental, lithium and fast neutron backgrounds `spectrum` is a fixed background
   * template and only `rate` varies call to call, so the cache is built once at construction: the
   * de-fractionalised matrix at rate=1 is a fixed matrix M0 and `defracCovMatrix = rate^2 * M0`;
   * scaling a symmetric matrix by a positive scalar scales its eigenvalues by the same factor and
   * leaves its eigenvectors unchanged, so `shifts()` only has to rescale the cached eigenvalues.
   *
   * For the reactor shape correction `spectrum` is the oscillated flux, which does change (whenever
   * SinSqT13/DeltaM41 change), so there the cache is instead rebuilt only on those steps and reused
   * on steps that merely touch a NuShape nuisance parameter.
   */
  class ShapeShiftCache {
   public:
    ShapeShiftCache() = default;

    ShapeShiftCache(std::span<const double>        spectrum,
                    const Eigen::MatrixXd&         covMatrix,
                    const std::function<int(int)>& row_of         = nullptr,
                    int                             spectrum_offset = 0) {
      const auto nBins = static_cast<int>(covMatrix.rows());

      auto spectrum_map = make_spectrum(spectrum);

      Eigen::MatrixXd defracCovMatrix(nBins, nBins);

      for (int i = 0; i < nBins; ++i) {
        const int    row     = row_of ? row_of(i) : i;
        const double value_i = spectrum_map[spectrum_offset + i];

        for (int j = 0; j < nBins; ++j) {
          const int    column  = row_of ? row_of(j) : j;
          const double value_j = spectrum_map[spectrum_offset + j];

          defracCovMatrix(i, j) = covMatrix(row, column) * value_i * value_j;
        }
      }

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(0.5 * (defracCovMatrix + defracCovMatrix.transpose()));

      m_EigenValues  = eigen_solver.eigenvalues();
      m_EigenVectors = eigen_solver.eigenvectors();
    }

    /**
     * @brief Computes the bin shifts for a given rate and set of nuisance parameters, reusing the
     *        cached eigendecomposition.
     */
    [[nodiscard]] Eigen::VectorXd shifts(double rate, std::span<const double> shape_parameter) const {
      const Eigen::VectorXd scaledEigenValues = pow_2(rate) * m_EigenValues;

      // The correction value for the eigenvalue that is iteratively increased until success
      double eigenvalueCorrection = 4e-14;

      using matrix_t = Eigen::MatrixXd;

      matrix_t             corrected_matrix;
      Eigen::LLT<matrix_t> llt_solver;

      // Iteratively increase the correction value until the matrix is positive definite
      // Compute the Cholesky decomposition of the corrected matrix
      while (true) {
        const auto clipped = scaledEigenValues.unaryExpr([eigenvalueCorrection](double v) { return std::max(v, eigenvalueCorrection); }).asDiagonal();

        corrected_matrix = m_EigenVectors * clipped * m_EigenVectors.transpose();
        llt_solver.compute(corrected_matrix);
        if (llt_solver.info() == Eigen::Success) {
          break;
        }
        eigenvalueCorrection *= 1.1;
      }

      return llt_solver.matrixL() * make_spectrum(shape_parameter);
    }

   private:
    Eigen::VectorXd m_EigenValues;
    Eigen::MatrixXd m_EigenVectors;
  };

  /**
   * @brief Scales a spectrum with a rate and applies the covariance driven bin shifts on top.
   *
   * @param rate Rate parameter the spectrum is scaled with.
   * @param shape The spectrum template.
   * @param shape_parameter The nuisance parameters, one per covariance matrix row.
   * @param cache The cached eigendecomposition of the de-fractionalised covariance matrix at rate=1,
   *              built from the same `shape` template with a ShapeShiftCache.
   * @param result The scaled and shifted spectrum. Has to be at least as long as shape.
   * @param clip_result If true the shifted spectrum is clipped at zero. This has to match the
   *                    convention of the background the spectrum belongs to.
   */
  inline void calculate_spectrum(double                  rate,
                                 std::span<const double> shape,
                                 std::span<const double> shape_parameter,
                                 const ShapeShiftCache&  cache,
                                 std::span<double>       result,
                                 bool                    clip_result = false) {
    auto backgroundSpectrum = make_spectrum(shape);

    const Eigen::VectorXd shifts = cache.shifts(rate, shape_parameter);

    // Copy the background spectrum to the result, scaled by the rate
    std::ranges::transform(std::as_const(backgroundSpectrum), result.begin(),
                           [rate](double x) { return std::max(x * rate, 0.0); });

    // Add the shifts to the result
    if (clip_result) {
      std::transform(shifts.cbegin(), shifts.cend(), result.begin(), result.begin(),
                     [](double shift, double value) { return std::max(0.0, value + shift); });
    } else {
      std::transform(shifts.cbegin(), shifts.cend(), result.begin(), result.begin(), std::plus<>());
    }
  }
}  // namespace ana::dc
