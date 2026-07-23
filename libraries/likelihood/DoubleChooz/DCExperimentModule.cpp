#include "DCExperimentModule.h"

#include "../Fit.h"

#include "DoubleChooz/DCWriteResults.h"

#include <stdexcept>

namespace ana::dc {

  std::shared_ptr<Likelihood> DCExperimentModule::create_likelihood(std::shared_ptr<io::Options> options) {
    auto dc_options = std::make_shared<const io::dc::DCOptions>(options->inputOptions(), *m_InputOptions);

    m_Likelihood = std::make_shared<DCLikelihood>(std::move(options),
                                                  number_of_parameters(),
                                                  std::move(dc_options),
                                                  m_InputOptions);
    return m_Likelihood;
  }

  void DCExperimentModule::write_results(Fit& fit, std::string_view name) {
    if (m_Likelihood == nullptr) {
      throw std::logic_error("DCExperimentModule::write_results called before create_likelihood");
    }
    result::dc::write_double_chooz_results(fit, *m_Likelihood, *m_InputOptions, name);
  }

}  // namespace ana::dc
