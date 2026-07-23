#pragma once

#include "Definitions.h"

// Eigen include
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

// STL include
#include <functional>
#include <span>

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
   * @brief Turns a fractional covariance matrix and a set of nuisance parameters into bin shifts.
   *
   * The fractional covariance matrix is de-fractionalised with the (rate scaled) spectrum, made
   * symmetric and then Cholesky decomposed. Since the de-fractionalised matrix is in general only
   * positive semi definite, the eigenvalues are clipped from below and the clipping value is
   * increased until the decomposition succeeds. The lower triangular factor finally maps the
   * uncorrelated nuisance parameters onto correlated bin shifts.
   *
   * @param rate Rate parameter the spectrum is scaled with. Use 1.0 if the rate is applied elsewhere.
   * @param spectrum The spectrum the covariance matrix refers to. Only the first covMatrix.rows()
   *                 entries are used, offset by spectrum_offset.
   * @param shape_parameter The nuisance parameters, one per covariance matrix row.
   * @param covMatrix The fractional covariance matrix.
   * @param row_of Maps a covariance matrix index onto the row/column that has to be read from
   *               covMatrix. Pass nullptr for the identity mapping.
   * @param spectrum_offset Index of the spectrum entry that corresponds to the first covariance
   *                        matrix row.
   * @return The bin shifts, one per covariance matrix row.
   */
  inline Eigen::VectorXd calculate_shifts(double                        rate,
                                          std::span<const double>       spectrum,
                                          std::span<const double>       shape_parameter,
                                          const Eigen::MatrixXd&        covMatrix,
                                          const std::function<int(int)>& row_of        = nullptr,
                                          int                           spectrum_offset = 0) {
    const auto nBins = static_cast<int>(covMatrix.rows());

    auto spectrum_map = make_spectrum(spectrum);

    Eigen::MatrixXd defracCovMatrix(nBins, nBins);

    for (int i = 0; i < nBins; ++i) {
      const int    row     = row_of ? row_of(i) : i;
      const double value_i = spectrum_map[spectrum_offset + i];

      for (int j = 0; j < nBins; ++j) {
        const int    column  = row_of ? row_of(j) : j;
        const double value_j = spectrum_map[spectrum_offset + j];

        defracCovMatrix(i, j) = covMatrix(row, column) * pow_2(rate) * value_i * value_j;
      }
    }

    using matrix_t = Eigen::MatrixXd;

    Eigen::SelfAdjointEigenSolver<matrix_t> eigen_solver(0.5 * (defracCovMatrix + defracCovMatrix.transpose()));

    // The correction value for the eigenvalue that is iteratively increased until success
    double eigenvalueCorrection = 4e-14;

    // Function to compute the corrected eigenvalues
    auto compute_corrected_eigenvalues = [&eigen_solver](double correction) noexcept {
      return eigen_solver.eigenvalues().unaryExpr([correction](double v) { return std::max(v, correction); }).asDiagonal();
    };

    // Get the eigenvectors of the rescaled covariance matrix
    const auto& eigenvectors = eigen_solver.eigenvectors();

    matrix_t             corrected_matrix;
    Eigen::LLT<matrix_t> llt_solver;

    // Iteratively increase the correction value until the matrix is positive definite
    // Compute the Cholesky decomposition of the corrected matrix
    while (true) {
      corrected_matrix = eigenvectors * compute_corrected_eigenvalues(eigenvalueCorrection) * eigenvectors.transpose();
      llt_solver.compute(corrected_matrix);
      if (llt_solver.info() == Eigen::Success) {
        break;
      }
      eigenvalueCorrection *= 1.1;
    }

    return llt_solver.matrixL() * make_spectrum(shape_parameter);
  }

  /**
   * @brief Scales a spectrum with a rate and applies the covariance driven bin shifts on top.
   *
   * @param rate Rate parameter the spectrum is scaled with.
   * @param shape The spectrum template.
   * @param shape_parameter The nuisance parameters, one per covariance matrix row.
   * @param covMatrix The fractional covariance matrix.
   * @param result The scaled and shifted spectrum. Has to be at least as long as shape.
   * @param clip_result If true the shifted spectrum is clipped at zero. This has to match the
   *                    convention of the background the spectrum belongs to.
   */
  inline void calculate_spectrum(double                  rate,
                                 std::span<const double> shape,
                                 std::span<const double> shape_parameter,
                                 const Eigen::MatrixXd&  covMatrix,
                                 std::span<double>       result,
                                 bool                    clip_result = false) {
    auto backgroundSpectrum = make_spectrum(shape);

    const Eigen::VectorXd shifts = calculate_shifts(rate, shape, shape_parameter, covMatrix);

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
