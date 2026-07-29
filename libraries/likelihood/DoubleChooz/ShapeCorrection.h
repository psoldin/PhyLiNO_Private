#pragma once

#include "Calculate_Spectrum.h"
#include "Oscillator.h"

namespace ana::dc {

  class ShapeCorrection : public SpectrumBase {
   public:
    explicit ShapeCorrection(std::shared_ptr<const io::dc::DCOptions> dc_options, std::shared_ptr<Oscillator> oscillator);

    ~ShapeCorrection() override = default;

    bool check_and_recalculate(const ParameterWrapper& parameter) noexcept override;

    [[nodiscard]] std::span<const double> get_spectrum(params::dc::DetectorType type) const noexcept override {
      return m_Cache.at(type);
    }

   private:
    std::shared_ptr<Oscillator> m_Oscillator;

    template <typename T>
    using uo_map = std::unordered_map<params::dc::DetectorType, T>;

    uo_map<std::array<double, 80>>           m_Cache;
    uo_map<std::shared_ptr<Eigen::MatrixXd>> m_CovMatrix;

    // Caches the eigendecomposition of the de-fractionalised covariance matrix per detector. Only
    // valid for the oscillated spectrum it was built from, so it is rebuilt whenever the oscillator
    // recalculates (i.e. SinSqT13/DeltaM41 changed); it can be reused as-is when only the NuShape
    // nuisance parameters changed, since those do not affect the spectrum that is de-fractionalised.
    uo_map<ShapeShiftCache> m_ShapeShiftCache;

    void recalculate_spectra(const ParameterWrapper& parameter, bool spectrum_changed) noexcept;
  };

}  // namespace ana::dc