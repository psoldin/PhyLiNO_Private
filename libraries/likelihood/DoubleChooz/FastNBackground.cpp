#include "FastNBackground.h"
#include <DoubleChooz/Constants.h>
#include "Calculate_Spectrum.h"
#include "Parameter.h"

// STL includes
#include <numeric>

// ROOT includes
#include <TH1D.h>

namespace ana::dc {

  inline bool parameter_changed(const ParameterWrapper& parameter) noexcept {
    using enum params::dc::DetectorType;
    using enum params::dc::Detector;
    using namespace params;

    bool has_changed = false;

    for (auto detector : {ND, FDI, FDII}) {
      has_changed |= parameter.check_parameter_changed(index(detector, FNSMShape01), index(detector, FNSMShape44));
      has_changed |= parameter.check_parameter_changed(index(detector, BkgRFNSM));
    }

    return has_changed;
  }

  FastNBackground::FastNBackground(std::shared_ptr<const io::dc::DCOptions> dc_options)
    : SpectrumBase(std::move(dc_options)) {
    using enum params::dc::DetectorType;

    const auto& db = this->dc_options().dataBase();

    for (const auto detector : {ND, FDI, FDII}) {
      auto cov              = db.covariance_matrix(detector, params::dc::SpectrumType::fastN);
      m_CovMatrix[detector] = std::move(cov);
      m_FastNSpectrum[detector].fill(0.0);
      fill_data(detector);
    }
  }

  bool FastNBackground::check_and_recalculate(const ParameterWrapper& parameter) {
    bool has_changed = parameter_changed(parameter);
    if (has_changed) {
      recalculate_spectra(parameter);
    }
    return has_changed;
  }

  void FastNBackground::recalculate_spectra(const ParameterWrapper& parameter) noexcept {
    using enum params::dc::DetectorType;
    using enum params::dc::Detector;
    using namespace params;

    for (const auto detector : {ND, FDI, FDII}) {
      using span_t = std::span<const double>;

      span_t background_template = get_background_template(detector);
      span_t shape_parameter     = parameter.sub_range(index(detector, FNSMShape01), index(detector, FNSMShape44) + 1);

      const double rate = parameter[index(detector, BkgRFNSM)];

      assert(m_CovMatrix[detector] != nullptr);

      const ShapeShiftCache& cache = m_ShapeShiftCache.at(detector);

      std::array<double, 44>& result = m_FastNSpectrum[detector];

      // The fast neutron spectrum is clipped at zero after the shifts have been applied
      calculate_spectrum(rate,
                         background_template,
                         shape_parameter,
                         cache,
                         result,
                         /* clip_result = */ true);
    }
  }

  void FastNBackground::fill_data(params::dc::DetectorType type) {
    const auto& db = dc_options().dataBase();

    auto fastN_data = db.background_data(type, params::dc::SpectrumType::fastN);

    // Unlike the other backgrounds the fast neutron spectrum extends up to 50 MeV, so the full
    // binning with all 44 bins is used here.
    const auto& binning = io::dc::Constants::EnergyBinXaxis;

    const int nBins = static_cast<int>(binning.size()) - 1;

    auto h = std::make_unique<TH1D>("h", "", nBins, binning.data());

    for (auto E : fastN_data) {
      h->Fill(E);
    }

    std::array<double, 44> background_template{};

    for (int i = 0; i < 44; ++i) {
      background_template[i] = h->GetBinContent(i + 1);
    }

    const double sum = std::accumulate(background_template.begin(), background_template.end(), 0.0);

    const double lifeTime = db.on_lifetime(type);

    std::array<double, 44> background_spectrum{};

    for (int i = 0; i < 44; ++i) {
      background_spectrum[i] = (lifeTime / sum) * background_template[i];
    }

    m_BackgroundTemplate[type] = background_spectrum;

    m_ShapeShiftCache[type] = ShapeShiftCache(background_spectrum, *m_CovMatrix.at(type));
  }

}  // namespace ana::dc
