#include "DCLikelihood.h"

namespace ana::dc {

  /**
   * @brief Calculates the Poisson likelihood for given data, signal, and background.
   *
   * This function computes the Poisson likelihood for a given set of observed data,
   * expected signal, and background. The likelihood is calculated using the formula:
   *
   * \f[
   * \mathcal{L} = -2 \sum_{i} \left( data_i \log(model_i) - model_i \right)
   * \f]
   *
   * where \f$ model_i = signal_i + bkg_i \f$.
   *
   * This assumes that the sum over background and signal prediction is larger than zero.
   *
   * @param data A span of observed data values.
   * @param signal A span of expected signal values.
   * @param bkg A span of expected background values.
   * @return The calculated Poisson likelihood.
   */
  inline double calculate_poisson_likelihood(std::span<const double> data, std::span<const double> signal, std::span<const double> bkg) noexcept {
    double return_value = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
      const double model_i = signal[i] + bkg[i];
      return_value += data[i] * std::log(model_i) - model_i;
    }

    return -2.0 * return_value;
  }

  bool DCLikelihood::recalculate_spectra(const ParameterWrapper& parameter) noexcept {
    bool recalculate = false;

    std::array<SpectrumBase*, 5> components = {&m_Accidental, &m_Lithium, &m_FastN, &m_DNC, &m_Reactor};
    for (auto* component : components) {
      recalculate |= component->check_and_recalculate(parameter);
    }

    return recalculate;
  }

  void DCLikelihood::initialize_measurement_data() {
    if (m_DCInputOptions->use_data()) {
      read_measurement_data();
    } else {
      generate_measurement_data();
    }
  }

  void DCLikelihood::read_measurement_data() {
    throw std::runtime_error("Reading measurement data is not yet implemented");
  }

  inline void print_array(std::span<const double> data, std::string_view name) {
    std::cout << name << ":\n";
    for (const auto& d : data) {
      std::cout << d << ' ';
    }
    std::cout << '\n';
  }

  void DCLikelihood::generate_measurement_data() {
    std::vector<double> parameter(params::number_of_parameters(), 0.0);

    const auto& pv = m_Options->inputOptions().input_parameters().parameters();

    std::cout << "Setting starting parameters as set in config file for Asimov data set generation!\n";
    for (std::size_t i = 0; i < parameter.size(); ++i) {
      parameter[i] = pv[i].value();
    }

    std::cout << "Set SinSqT13 = 0.1 in the parameter array for Asimov data set generation!\n";
    parameter[params::SinSqT13] = 0.1;

    m_Parameter.reset_parameter(parameter.data());

    std::cout << "Calculate the spectrum components for Asimov data set generation!\n";
    for (auto* component : m_Components) {
      component->check_and_recalculate(m_Parameter);
    }

    using enum params::dc::DetectorType;

    constexpr int nBins = 44;

    for (auto detector : {ND, FDI, FDII}) {
      using map_t   = Eigen::Map<const Eigen::Array<double, nBins, 1>>;
      using array_t = Eigen::Array<double, nBins, 1>;

      // Get all spectrum components as Eigen::Map
      map_t acc(m_Accidental.get_spectrum(detector).data(), nBins);
      map_t li(m_Lithium.get_spectrum(detector).data(), nBins);
      map_t fastN(m_FastN.get_spectrum(detector).data(), nBins);
      map_t dnc(m_DNC.get_spectrum(detector).data(), nBins);
      map_t reactor(m_Reactor.get_spectrum(detector).data(), nBins);

      // Add all background components to the full background contribution
      const array_t bkg = acc + li + fastN + dnc;

      // Get the MC normalization parameter
      const double mcNorm = calculate_mcNorm(m_Parameter, detector);

      // Calculate the full spectrum prediction
      array_t prediction = (bkg + (mcNorm * reactor));

      std::array<double, 44> array{};
      std::ranges::copy(prediction, array.begin());
      m_MeasurementData[detector] = array;

      if (detector == ND || detector == FDII) {
        std::array<double, 44> off_off_data{};

        const array_t off_off_bkg = off_off_scaling(detector) * bkg;

        std::ranges::copy(off_off_bkg, off_off_data.begin());
        m_OffOffData[detector] = off_off_data;
      }
    }
  }

  double DCLikelihood::off_off_scaling(params::dc::DetectorType detector) const noexcept {
    const auto& db = m_DCOptions->dataBase();

    const double on_lifetime = db.on_lifetime(detector);

    double off_lifetime = db.off_lifetime(detector);

    // FD-I and FD-II are the same physical detector, so the FD-I off-off lifetime also applies to
    // the FD-II background prediction.
    if (detector == params::dc::DetectorType::FDII) {
      off_lifetime += db.off_lifetime(params::dc::DetectorType::FDI);
    }

    return off_lifetime / on_lifetime;
  }

  void correlate_parameters(const io::dc::DCOptions& options, std::span<double> parameters) {
    using enum params::dc::DetectorType;
    using enum params::dc::Detector;
    using namespace params;

    // FDI and FDII lithium background rates are fully correlated
    parameters[index(FDI, BkgRLi)] = parameters[index(FDII, BkgRLi)];

    // The fast neutron shape is fully correlated among all detectors. FD-II carries the free
    // parameters, ND and FD-I simply copy them.
    {
      constexpr int nShape = (FNSMShape44 - FNSMShape01) + 1;

      const auto source = parameters.subspan(index(FDII, FNSMShape01), nShape);

      for (const auto detector : {ND, FDI}) {
        std::ranges::copy(source, parameters.begin() + index(detector, FNSMShape01));
      }
    }

    const auto& dco = options.dataBase();

    {  // Correlate Energy Parameters
      // EnergyA is fully correlated among all detectors
      constexpr std::array<int, 7> energy_indices = {EnergyA,
                                                     index(FDI, EnergyB),
                                                     index(ND, EnergyB),
                                                     index(FDII, EnergyB),
                                                     index(FDI, EnergyC),
                                                     index(ND, EnergyC),
                                                     index(FDII, EnergyC)};

      TVectorD energy_correlations(7);

      for (std::size_t i = 0; i < energy_indices.size(); ++i) {
        energy_correlations[i] = parameters[energy_indices[i]];
      }

      energy_correlations *= dco.energy_correlation_matrix();

      for (std::size_t i = 0; i < energy_indices.size(); ++i) {
        parameters[energy_indices[i]] = energy_correlations[i];
      }
    }

    {
      constexpr std::array mcNorm_indices = {index(FDI, MCNorm),
                                             index(ND, MCNorm),
                                             index(FDII, MCNorm)};

      TVectorD mcNorm_correlations(3);
      for (std::size_t i = 0; i < mcNorm_indices.size(); ++i) {
        mcNorm_correlations[i] = parameters[mcNorm_indices[i]];
      }

      mcNorm_correlations *= dco.mcNorm_correlation_matrix();

      for (std::size_t i = 0; i < mcNorm_indices.size(); ++i) {
        parameters[mcNorm_indices[i]] = mcNorm_correlations[i];
      }
    }
    {
      const auto& covMatrix = dco.interDetector_correlation_matrix();
      TVectorD    reactor_correlations(3);
      for (int i = NuShape01; i <= NuShape43; ++i) {
        reactor_correlations[0] = parameters[index(FDI, i)];
        reactor_correlations[1] = parameters[index(ND, i)];
        reactor_correlations[2] = parameters[index(FDII, i)];

        reactor_correlations *= covMatrix;

        parameters[index(FDI, i)]  = reactor_correlations[0];
        parameters[index(ND, i)]   = reactor_correlations[1];
        parameters[index(FDII, i)] = reactor_correlations[2];
      }
    }
  }

  DCLikelihood::DCLikelihood(std::shared_ptr<io::Options>                  options,
                             int                                           nParameter,
                             std::shared_ptr<const io::dc::DCOptions>      dc_options,
                             std::shared_ptr<const io::dc::DCInputOptions> dc_input_options)
    : Likelihood(options, nParameter,
                 [dc_options](std::span<double> parameter) { correlate_parameters(*dc_options, parameter); })
    , m_DCInputOptions(std::move(dc_input_options))
    , m_DCOptions(dc_options)
    , m_Accidental(options, dc_options)
    , m_Lithium(options, dc_options)
    , m_FastN(options, dc_options)
    , m_DNC(options, dc_options)
    , m_Reactor(options, dc_options) {
    m_Components = {&m_Accidental, &m_Lithium, &m_FastN, &m_DNC, &m_Reactor};
    initialize_measurement_data();
    setup_pulls();
  }

  void DCLikelihood::setup_pulls() {
    const auto& input_parameters = m_Options->inputOptions().input_parameters();

    const auto& names       = input_parameters.names();
    const auto& parameters  = input_parameters.parameters();
    const auto& constrained = input_parameters.constrained();

    using enum params::dc::DetectorType;
    using enum params::dc::Detector;
    using namespace params;

    // The energy scale, MC normalisation and reactor shape parameters are constrained among the
    // detectors and are handled by the correlated pull terms further below. Giving them an
    // additional uncorrelated pull here would count their constraint twice.
    auto has_correlated_pull = [](std::size_t i) {
      if (i == static_cast<std::size_t>(EnergyA)) {
        return true;
      }

      for (const auto detector : {ND, FDI, FDII}) {
        if (i == static_cast<std::size_t>(index(detector, EnergyB))
            || i == static_cast<std::size_t>(index(detector, EnergyC))
            || i == static_cast<std::size_t>(index(detector, MCNorm))) {
          return true;
        }

        const auto first = static_cast<std::size_t>(index(detector, NuShape01));
        const auto last  = static_cast<std::size_t>(index(detector, NuShape43));

        if (i >= first && i <= last) {
          return true;
        }
      }

      return false;
    };

    // Every other parameter that is flagged as constrained in the configuration gets an
    // uncorrelated Gaussian pull term.
    for (std::size_t i = 0, end = constrained.size(); i < end; ++i) {
      if (constrained[i] && !has_correlated_pull(i)) {
        std::cout << "Setup pull for parameter " << std::setw(8) << i << ":\t" << names[i] << '\n';
        m_Pulls.emplace_back(i, parameters[i].value(), parameters[i].uncertainty());
      }
    }

    for (int i = 0, end = (NuShape43 - NuShape01) + 1; i < end; ++i) {
      m_ShapeCV.emplace_back(parameters[index(ND, NuShape01 + i)].value(),
                             parameters[index(FDI, NuShape01 + i)].value(),
                             parameters[index(FDII, NuShape01 + i)].value());
    }

    // Central values of the correlated energy scale pull.
    // The order has to match the one used in correlate_parameters and the energy correlation matrix.
    const std::array<int, 7> energy_indices = {EnergyA,
                                               index(FDI, EnergyB),
                                               index(ND, EnergyB),
                                               index(FDII, EnergyB),
                                               index(FDI, EnergyC),
                                               index(ND, EnergyC),
                                               index(FDII, EnergyC)};

    for (std::size_t i = 0; i < energy_indices.size(); ++i) {
      m_EnergyCV[i] = parameters[energy_indices[i]].value();
    }

    // Central values of the correlated MC normalisation pull, same ordering rules apply.
    const std::array<int, 3> mcNorm_indices = {index(FDI, MCNorm),
                                               index(ND, MCNorm),
                                               index(FDII, MCNorm)};

    for (std::size_t i = 0; i < mcNorm_indices.size(); ++i) {
      m_MCNormCV[i] = parameters[mcNorm_indices[i]].value();
    }
  }

  double DCLikelihood::calculate_correlated_pulls(const ParameterWrapper& parameter) const noexcept {
    using enum params::dc::DetectorType;
    using enum params::dc::Detector;
    using namespace params;

    const auto& dco = m_DCOptions->dataBase();

    double result = 0.0;

    {  // Energy scale pull
      const std::array<int, 7> energy_indices = {EnergyA,
                                                 index(FDI, EnergyB),
                                                 index(ND, EnergyB),
                                                 index(FDII, EnergyB),
                                                 index(FDI, EnergyC),
                                                 index(ND, EnergyC),
                                                 index(FDII, EnergyC)};

      TVectorD difference(7);
      for (std::size_t i = 0; i < energy_indices.size(); ++i) {
        difference[i] = parameter[energy_indices[i]] - m_EnergyCV[i];
      }

      result += dco.energy_inverse_correlation_matrix().Similarity(difference);
    }

    {  // MC normalisation pull
      const std::array<int, 3> mcNorm_indices = {index(FDI, MCNorm),
                                                 index(ND, MCNorm),
                                                 index(FDII, MCNorm)};

      TVectorD difference(3);
      for (std::size_t i = 0; i < mcNorm_indices.size(); ++i) {
        difference[i] = parameter[mcNorm_indices[i]] - m_MCNormCV[i];
      }

      result += dco.mcNorm_inverse_correlation_matrix().Similarity(difference);
    }

    return result;
  }

  double DCLikelihood::calculate_pulls(const ParameterWrapper& parameter) const noexcept {
    double result = 0.0;
    for (const auto [idx, CV, sig] : m_Pulls) {
      result += pow_2((parameter[idx] - CV) / sig);
    }

    using span_t = std::span<const double>;

    span_t rawP = parameter.raw_parameters();

    using enum params::dc::DetectorType;
    using enum params::dc::Detector;

    constexpr size_t nShape = (NuShape43 - NuShape01) + 1;

    span_t nd_shape  = rawP.subspan(params::index(ND, NuShape01), nShape);
    span_t fd1_shape = rawP.subspan(params::index(FDI, NuShape01), nShape);
    span_t fd2_shape = rawP.subspan(params::index(FDII, NuShape01), nShape);

    constexpr double scale = 1.0;

    for (std::size_t i = 0; i < nShape; ++i) {
      const auto [nd_CV, fd1_CV, fd2_CV] = m_ShapeCV[i];

      const double nd_result  = pow_2((nd_shape[i] - nd_CV) / scale);
      const double fd1_result = pow_2((fd1_shape[i] - fd1_CV) / scale);
      const double fd2_result = pow_2((fd2_shape[i] - fd2_CV) / scale);

      result += nd_result + fd1_result + fd2_result;
    }

    // Penalise unphysical negative values of sin^2(theta13) so that the minimizer is pushed back
    // into the physical region instead of wandering off.
    result += std::abs(std::min(parameter[params::SinSqT13], 0.0));

    return result;
  }

  void DCLikelihood::check_and_recalculate(const double* parameter) noexcept {
    m_Parameter.reset_parameter(parameter);

    for (auto* component : m_Components) {
      component->check_and_recalculate(m_Parameter);
    }
  }

  double DCLikelihood::calculate_likelihood(const double* parameter) {
    check_and_recalculate(parameter);
    if (m_DCInputOptions->reactor_split()) {
      return calculate_reactor_split_likelihood(m_Parameter);
    }
    return calculate_default_likelihood(m_Parameter);
  }

  double DCLikelihood::calculate_off_off_likelihood(const Eigen::Array<double, 44, 1>& bkg, params::dc::DetectorType detector) const {
    constexpr int nBins = 44;
    using map_t         = Eigen::Map<const Eigen::Array<double, 44, 1>>;

    // Get the off-off data
    map_t off_off_data(get_off_off_data(detector).data(), nBins);

    // Rescale the background to the off time
    const Eigen::Array<double, 44, 1> off_off_bkg = off_off_scaling(detector) * bkg;

    // Exclude the low energy bins due to residual neutrinos.
    //
    // The cut is expressed as "the last (number_of_analysis_edges - idx) bins", where idx is the
    // first bin edge at or above 3 MeV and number_of_analysis_edges is the number of edges of the
    // analysis range only, i.e. without the six extended bins that reach up to 50 MeV. Counting
    // from the back over the full 44 bin spectrum therefore drops everything below roughly 4.25 MeV
    // rather than below 3 MeV. This is what the reference implementation does and it is kept here
    // so both give identical likelihood values.
    constexpr std::size_t idx = std::distance(io::dc::Constants::EnergyBinXaxis.cbegin(),
                                              std::ranges::lower_bound(io::dc::Constants::EnergyBinXaxis, 3.0));

    constexpr std::size_t number_of_analysis_edges = io::dc::Constants::number_of_energy_bins + 1;

    static_assert(number_of_analysis_edges > idx);

    const Eigen::Array<double, 44, 1> off_off_llh = off_off_data * off_off_bkg.log() - off_off_bkg;

    // Calculate Poisson Likelihood
    return -2.0 * off_off_llh.tail(number_of_analysis_edges - idx).sum();
  }

  double DCLikelihood::calculate_mcNorm(const ParameterWrapper& parameter, params::dc::DetectorType type) const noexcept {
    using namespace params::dc;

    const auto [value, error] = m_DCOptions->dataBase().mcNorm_central_values(type);

    const double norm = parameter[params::index(type, Detector::MCNorm)];

    const double result = value + error * norm;

    const double bugey4 = parameter[params::Bugey4];

    return result * bugey4;
  }

  double DCLikelihood::calculate_default_likelihood(const ParameterWrapper& parameter) const noexcept {
    using enum params::dc::DetectorType;

    double likelihood = 0.0;

    constexpr int nBins = 44;

    for (const auto detector : {ND, FDI, FDII}) {
      using map_t   = Eigen::Map<const Eigen::Array<double, nBins, 1>>;
      using array_t = Eigen::Array<double, nBins, 1>;

      // Get all spectrum components as Eigen::Map
      map_t acc(m_Accidental.get_spectrum(detector).data(), nBins);
      map_t li(m_Lithium.get_spectrum(detector).data(), nBins);
      map_t fastN(m_FastN.get_spectrum(detector).data(), nBins);
      map_t dnc(m_DNC.get_spectrum(detector).data(), nBins);
      map_t reactor(m_Reactor.get_spectrum(detector).data(), nBins);

      // Add all background components to the full background contribution
      const array_t bkg = acc + li + fastN + dnc;

      // Get the measurement data as Eigen::Map
      map_t data(get_measurement_data(detector).data(), nBins);

      // Get the MC normalization parameter
      const double mcNorm = calculate_mcNorm(parameter, detector);

      // Calculate the full spectrum prediction
      array_t prediction = (bkg + (mcNorm * reactor));

      // Calculate Poisson Likelihood
      likelihood += -2.0 * (data * prediction.log() - prediction).sum();

      // Calculate the off-off component of the likelihood.
      // Only ND and FD-II have reactor-off data taking periods.
      if (detector == ND || detector == FDII) {
        likelihood += calculate_off_off_likelihood(bkg, detector);
      }
    }

    likelihood += calculate_pulls(parameter) + calculate_correlated_pulls(parameter);

    // Return the likelihood parameter if it is finite, otherwise return a large number. This is to prevent the minimizer from crashing.
    return std::isfinite(likelihood) ? likelihood : 1.0e25;
  }
}  // namespace ana::dc
