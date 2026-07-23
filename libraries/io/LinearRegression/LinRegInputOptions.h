#pragma once

#include "InputOptionBase.h"

namespace io::linreg {

  /**
   * @brief Input options of the linear-regression example experiment.
   *
   * Reads the truth values used to generate the Asimov data from the "LinearRegression" section
   * of the config file. This class doubles as the minimal template for experiment input options:
   * inherit from InputOptionBase, parse your config section in read().
   */
  class LinRegInputOptions : public InputOptionBase {
   public:
    LinRegInputOptions()
      : InputOptionBase("Linear Regression Options") {}

    ~LinRegInputOptions() final = default;

    void read(const boost::program_options::variables_map& /*vm*/, const boost::property_tree::ptree& config) final {
      const auto& section = config.get_child("LinearRegression");
      m_TruthA  = section.get<double>("TruthA");
      m_TruthB  = section.get<double>("TruthB");
      m_NPoints = section.get<int>("NPoints");
      m_Sigma   = section.get<double>("Sigma");
      m_XMin    = section.get<double>("XMin");
      m_XMax    = section.get<double>("XMax");
    }

    [[nodiscard]] double truth_a() const noexcept { return m_TruthA; }
    [[nodiscard]] double truth_b() const noexcept { return m_TruthB; }
    [[nodiscard]] int    n_points() const noexcept { return m_NPoints; }
    [[nodiscard]] double sigma() const noexcept { return m_Sigma; }
    [[nodiscard]] double x_min() const noexcept { return m_XMin; }
    [[nodiscard]] double x_max() const noexcept { return m_XMax; }

   private:
    double m_TruthA  = 0.0; /**< True slope used for Asimov data generation. */
    double m_TruthB  = 0.0; /**< True offset used for Asimov data generation. */
    int    m_NPoints = 0;   /**< Number of generated data points. */
    double m_Sigma   = 1.0; /**< Gaussian uncertainty assigned to every point. */
    double m_XMin    = 0.0; /**< Lower edge of the x range. */
    double m_XMax    = 1.0; /**< Upper edge of the x range. */
  };

}  // namespace io::linreg
