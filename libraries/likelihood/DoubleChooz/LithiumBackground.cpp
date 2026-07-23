#include "LithiumBackground.h"
#include <DoubleChooz/Constants.h>
#include "Calculate_Spectrum.h"

// STL includes
#include <numeric>

// ROOT includes
#include <TH1D.h>

namespace ana::dc {

  [[nodiscard]] inline bool check_parameters(const ParameterWrapper& parameter) noexcept {
    using enum params::dc::DetectorType;
    using namespace params;
    using namespace params::dc;

    bool recalculate = parameter.check_parameter_changed(General::LiShape01, General::LiShape38);

    recalculate |= parameter.check_parameter_changed(index(ND, BkgRLi));
    recalculate |= parameter.check_parameter_changed(index(FDI, BkgRLi));
    recalculate |= parameter.check_parameter_changed(index(FDII, BkgRLi));

    return recalculate;
  }

  LithiumBackground::LithiumBackground(std::shared_ptr<io::Options> options)
    : SpectrumBase(std::move(options)) {
    using enum params::dc::DetectorType;
    for (const auto detector : {ND, FDI, FDII}) {
      fill_data(detector);
    }
  }

  bool LithiumBackground::check_and_recalculate(const ParameterWrapper& parameter) {
    bool has_changed = check_parameters(parameter);
    if (has_changed) {
      recalculate_spectra(parameter);
    }
    return has_changed;
  }

  void LithiumBackground::recalculate_spectra(const ParameterWrapper& parameter) {
    using enum params::dc::DetectorType;
    using namespace params::dc;

    using span_t = std::span<const double>;

    // Lithium shape is fully correlated between all detectors
    span_t shape_parameter = parameter.sub_range(params::LiShape01, params::LiShape38 + 1);

    for (const auto detector : {ND, FDI, FDII}) {
      span_t background_template = get_background_template(detector);

      const double rate = parameter[params::index(detector, BkgRLi)];

      assert(m_CovMatrix[detector] != nullptr);

      const auto& covMatrix = *m_CovMatrix[detector];

      auto& result = m_LiSpectrum[detector];

      // The lithium spectrum is clipped at zero after the shifts have been applied
      calculate_spectrum(rate,
                         background_template,
                         shape_parameter,
                         covMatrix,
                         result,
                         /* clip_result = */ true);
    }
  }

  void LithiumBackground::fill_data(params::dc::DetectorType type) {
    const auto& db = m_Options->double_chooz().dataBase();

    const auto li_data = db.background_data(type, params::dc::SpectrumType::lithium);

    // The lithium background is only defined within the analysis range, so the histogram uses the
    // corresponding number of bins. Everything above ends up in the overflow bin and is therefore
    // not part of the template.
    constexpr int nBins = io::dc::Constants::number_of_energy_bins;

    const auto& binning = io::dc::Constants::EnergyBinXaxis;

    auto h = std::make_unique<TH1D>("h", "", nBins, binning.data());

    for (auto E : li_data) {
      h->Fill(E);
    }

    std::array<double, 44> background_template{};

    for (int i = 0; i < nBins; ++i) {
      background_template[i] = h->GetBinContent(i + 1);
    }

    const double sum = std::accumulate(background_template.begin(), background_template.end(), 0.0);

    const double lifeTime = db.on_lifetime(type);

    const double scaling_factor = lifeTime / sum;

    std::array<double, 44> background_spectrum{};

    for (int i = 0; i < 44; ++i) {
      background_spectrum[i] = scaling_factor * background_template[i];
    }

    m_BackgroundTemplate[type] = background_spectrum;
    m_LiSpectrum[type]         = std::array<double, 44>{};
    m_CovMatrix[type]          = db.covariance_matrix(type, params::dc::SpectrumType::lithium);
  }
}  // namespace ana::dc
