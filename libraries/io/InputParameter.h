#pragma once

// STL includes
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

// boost includes
#include <boost/property_tree/ptree.hpp>

namespace io {

  /**
   * @class InputParameter
   * @brief Represents a set of input parameters.
   *
   * The InputParameter class provides methods to access input parameters from the config file.
   * Each input parameter has a name, value, uncertainty, fixed status, and constrained status.
   * The class also provides methods to retrieve information about the parameters, such as the number of parameters,
   * the names of all parameters, the parameters themselves, and the fixed and constrained status of all parameters.
   */
  class InputParameter {
    class Parameter;

   public:
    /**
     * @brief Constructor for InputParameter.
     *
     * @param tree The boost property tree containing the input parameters.
     */
    explicit InputParameter(const boost::property_tree::ptree& tree) {
      try {
        for (const auto& [_, parameter] : tree) {
          m_Parameters.emplace_back(parameter);
          m_Fixed.emplace_back(parameter.get<bool>("Fixed"));
          m_Constrained.emplace_back(parameter.get<bool>("Constrained"));
          m_Names.emplace_back(parameter.get<std::string>("Name"));
        }
      } catch (std::exception& e) {
        std::cout << "A problem occurred in InputParameters class: " << e.what() << '\n';
        throw;
      }
    }

    /**
     * @brief Get the name of the parameter at the specified index.
     *
     * @param i The index of the parameter.
     * @return The name of the parameter.
     */
    [[nodiscard]] const std::string& name(int i) const noexcept { return m_Names[i]; }

    /**
     * @brief Get the value of the parameter at the specified index.
     *
     * @param i The index of the parameter.
     * @return The value of the parameter.
     */
    [[nodiscard]] double value(int i) const noexcept { return m_Parameters[i].value(); }

    /**
     * @brief Get the uncertainty of the parameter at the specified index.
     *
     * @param i The index of the parameter.
     * @return The uncertainty of the parameter.
     */
    [[nodiscard]] double uncertainty(int i) const noexcept { return m_Parameters[i].uncertainty(); }

    /**
     * @brief Check if the parameter at the specified index is fixed.
     *
     * @param i The index of the parameter.
     * @return True if the parameter is fixed, false otherwise.
     */
    [[nodiscard]] bool fixed(int i) const noexcept { return m_Fixed[i]; }

    /**
     * @brief Check if the parameter at the specified index is constrained.
     *
     * @param i The index of the parameter.
     * @return True if the parameter is constrained, false otherwise.
     */
    [[nodiscard]] bool constrained(int i) const noexcept { return m_Constrained[i]; }

    /**
     * @brief Get the number of parameters.
     *
     * @return The number of parameters.
     */
    [[nodiscard]] std::size_t size() const noexcept { return m_Parameters.size(); }

    /**
     * @brief Get the names of all parameters.
     *
     * @return The names of all parameters.
     */
    [[nodiscard]] const std::vector<std::string>& names() const noexcept { return m_Names; }

    /**
     * @brief Get the parameters.
     *
     * @return The parameters.
     */
    [[nodiscard]] const std::vector<Parameter>& parameters() const noexcept { return m_Parameters; }

    /**
     * @brief Get the fixed status of all parameters.
     *
     * @return The fixed status of all parameters.
     */
    [[nodiscard]] const std::vector<bool>& fixed() const noexcept { return m_Fixed; }

    /**
     * @brief Get the constrained status of all parameters.
     *
     * @return The constrained status of all parameters.
     */
    [[nodiscard]] const std::vector<bool>& constrained() const noexcept { return m_Constrained; }

   private:
    std::vector<Parameter>   m_Parameters;   ///< The parameters.
    std::vector<bool>        m_Fixed;        ///< The fixed status of the parameters.
    std::vector<bool>        m_Constrained;  ///< The constrained status of the parameters.
    std::vector<std::string> m_Names;        ///< The names of the parameters.

    /**
     * @brief Constructs an InputParameter object.
     *
     * @param parameter The boost::property_tree::ptree object containing the parameter information.
     */
    class Parameter {
     public:
      /**
       * @brief Constructs a Parameter object.
       *
       * @param parameter The boost::property_tree::ptree object containing the parameter information.
       */
      /** ptree::get_optional yields a boost::optional; this is the std one. */
      static std::optional<double> optional_double(const boost::property_tree::ptree& parameter,
                                                   const std::string&                 key) {
        const auto value = parameter.get_optional<double>(key);
        return value ? std::optional<double>(*value) : std::nullopt;
      }

      explicit Parameter(const boost::property_tree::ptree& parameter)
        : m_Value(parameter.get<double>("StartValue"))
        , m_Uncertainty(parameter.get<double>("StepWidth"))
        // The Gaussian pull's central value and width. Optional: they default to
        // the start value and the step width, which is what every config meant
        // before the two were separable, so existing configs keep their pulls.
        , m_PriorValue(parameter.get<double>("PriorValue", m_Value))
        , m_PriorWidth(parameter.get<double>("PriorWidth", m_Uncertainty))
        // The point the Asimov data is generated at, NNMFit's
        // analysis.input_params / run_fit.py --inject. Optional: it defaults to
        // the start value, which is what the Asimov set was always built from,
        // so existing configs are unaffected. Separating the two is what lets a
        // fit (or a fixed-point evaluation) sit somewhere other than the truth.
        , m_AsimovValue(parameter.get<double>("AsimovValue", m_Value))
        // Optional minimiser bounds, NNMFit's per-parameter "range". Either
        // side may be omitted for a one-sided bound; omitting both (the
        // default) leaves the parameter unbounded, which is what every config
        // meant before bounds existed.
        , m_LowerBound(optional_double(parameter, "LowerBound"))
        , m_UpperBound(optional_double(parameter, "UpperBound")) {
        if (m_LowerBound && m_UpperBound && *m_LowerBound >= *m_UpperBound)
          throw std::invalid_argument(
              "Parameter '" + parameter.get<std::string>("Name", "<unnamed>") + "': LowerBound " +
              std::to_string(*m_LowerBound) + " must be below UpperBound " +
              std::to_string(*m_UpperBound));

        // A start value outside its own bounds is a config error, not something
        // to silently clamp: Minuit2 would take the clamped point as the seed
        // and the fit would quietly start somewhere the config never asked for.
        if ((m_LowerBound && m_Value < *m_LowerBound) || (m_UpperBound && m_Value > *m_UpperBound))
          throw std::invalid_argument(
              "Parameter '" + parameter.get<std::string>("Name", "<unnamed>") + "': StartValue " +
              std::to_string(m_Value) + " lies outside its bounds");
      }

      /**
       * @brief Gets the value of the parameter.
       *
       * @return The value of the parameter.
       */
      [[nodiscard]] double value() const noexcept { return m_Value; }

      /**
       * @brief Gets the minimiser step width (Minuit's initial step) of the parameter.
       *
       * @return The step width of the parameter.
       */
      [[nodiscard]] double uncertainty() const noexcept { return m_Uncertainty; }

      /** Central value of the Gaussian pull on a constrained parameter. */
      [[nodiscard]] double prior_value() const noexcept { return m_PriorValue; }

      /** Width (sigma) of the Gaussian pull on a constrained parameter. */
      [[nodiscard]] double prior_width() const noexcept { return m_PriorWidth; }

      /**
       * Value used when generating the Asimov data set ("AsimovValue"),
       * defaulting to the start value. Only the data generation reads it; the
       * minimizer always starts from value().
       */
      [[nodiscard]] double asimov_value() const noexcept { return m_AsimovValue; }

      /** Minimiser bounds ("LowerBound"/"UpperBound"); empty means unbounded on that side. */
      [[nodiscard]] const std::optional<double>& lower_bound() const noexcept { return m_LowerBound; }
      [[nodiscard]] const std::optional<double>& upper_bound() const noexcept { return m_UpperBound; }

     private:
      double m_Value;        ///< The start value of the parameter.
      double m_Uncertainty;  ///< The minimiser step width.
      double m_PriorValue;   ///< Central value of the Gaussian pull.
      double m_PriorWidth;   ///< Width of the Gaussian pull.
      double m_AsimovValue;  ///< Value the Asimov data is generated at.

      std::optional<double> m_LowerBound;  ///< Optional minimiser lower bound.
      std::optional<double> m_UpperBound;  ///< Optional minimiser upper bound.
    };
  };

}  // namespace io
