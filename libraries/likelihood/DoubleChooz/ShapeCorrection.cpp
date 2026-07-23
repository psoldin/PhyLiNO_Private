#include "ShapeCorrection.h"

#include <numeric>

#include "Calculate_Spectrum.h"
#include "Oscillator.h"

namespace ana::dc {

  /**
   * @brief Maps an equidistant spline bin onto the official Double Chooz analysis bin.
   *
   * The reactor spectrum is calculated on an equidistant 0.25 MeV grid, while the reactor
   * covariance matrix is given in the official, non equidistant analysis binning. Above 8 MeV
   * several equidistant bins fall into the same official bin, so the covariance matrix entry has
   * to be looked up through this mapping.
   *
   * @param idx The equidistant bin index.
   * @return The official analysis bin index.
   */
  [[nodiscard]] inline int equidistant_bin_to_official_bin(int idx) noexcept {
    switch (idx) {
      case 0:
        return 0;
      case 29:
      case 30:
        return 28;
      case 31:
      case 32:
        return 29;
      case 33:
      case 34:
        return 30;
      case 35:
      case 36:
        return 31;
      case 37:
      case 38:
      case 39:
      case 40:
        return 32;
      case 41:
      case 42:
      case 43:
      case 44:
        return 33;
      case 45:
      case 46:
      case 47:
      case 48:
        return 34;
      case 49:
      case 50:
      case 51:
      case 52:
        return 35;
      default:
        return idx - 1;
    }
  }

  [[nodiscard]] bool parameter_changed(const ParameterWrapper& parameter) noexcept {
    using enum params::dc::DetectorType;
    using namespace params;
    using namespace params::dc;

    const bool fd1_changed = parameter.check_parameter_changed(index(FDI, NuShape01),
                                                               index(FDI, NuShape43));

    const bool fd2_changed = parameter.check_parameter_changed(index(FDII, NuShape01),
                                                               index(FDII, NuShape43));

    const bool nd_changed = parameter.check_parameter_changed(index(ND, NuShape01),
                                                              index(ND, NuShape43));

    const bool parameter_changed = fd1_changed | fd2_changed | nd_changed;

    return parameter_changed;
  }

  ShapeCorrection::ShapeCorrection(std::shared_ptr<io::Options> options, std::shared_ptr<Oscillator> oscillator)
    : SpectrumBase(std::move(options))
    , m_Oscillator(std::move(oscillator)) {
    using enum params::dc::DetectorType;

    const auto& db = m_Options->double_chooz().dataBase();

    for (auto detector : {ND, FDI, FDII}) {
      const auto& cov       = db.covariance_matrix(detector, params::dc::SpectrumType::reactor);
      m_CovMatrix[detector] = cov;
    }
  }

  bool ShapeCorrection::check_and_recalculate(const ParameterWrapper& parameter) noexcept {
    const bool previous_step = m_Oscillator->check_and_recalculate(parameter);
    const bool this_step     = parameter_changed(parameter);
    const bool recalculate   = previous_step | this_step;

    if (recalculate) {
      recalculate_spectra(parameter);
    }

    return recalculate;
  }

  void ShapeCorrection::recalculate_spectra(const ParameterWrapper& parameter) noexcept {
    using enum params::dc::DetectorType;
    using namespace params::dc;

    // The reactor spectrum does not need an explicit rate
    // This is done later with other parameters, this here is just a placeholder for the function call
    const double rate = 1.0;

    for (const auto detector : {ND, FDI, FDII}) {
      const std::span<const double> oscillated_spectrum = m_Oscillator->get_spectrum(detector);

      const auto shape_parameter = parameter.sub_range(params::index(detector, NuShape01),
                                                       params::index(detector, NuShape43) + 1);

      assert(m_CovMatrix[detector] != nullptr);

      const Eigen::MatrixXd& covMatrix = *m_CovMatrix[detector];

      std::array<double, 80>& result = m_Cache[detector];

      // The covariance matrix is de-fractionalised with the spectrum starting at bin zero, ...
      const Eigen::VectorXd shifts = calculate_shifts(rate,
                                                      oscillated_spectrum,
                                                      shape_parameter,
                                                      covMatrix,
                                                      &equidistant_bin_to_official_bin,
                                                      0);

      // ... but the resulting shifts are applied starting at the first bin, because the
      // normalisation shifts do not cover the whole spline support.
      constexpr int shift_offset = 1;

      std::ranges::copy(oscillated_spectrum, result.begin());

      for (Eigen::Index i = 0; i < shifts.size(); ++i) {
        result[shift_offset + i] += shifts[i];
      }
    }
  }

}  // namespace ana::dc
