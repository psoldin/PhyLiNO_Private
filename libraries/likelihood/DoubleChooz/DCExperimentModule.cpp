#include "DCExperimentModule.h"

namespace ana::dc {

  std::shared_ptr<Likelihood> DCExperimentModule::create_likelihood(std::shared_ptr<io::Options> options) {
    auto dc_options = std::make_shared<const io::dc::DCOptions>(options->inputOptions(), *m_InputOptions);

    m_Likelihood = std::make_shared<DCLikelihood>(std::move(options),
                                                  params::number_of_parameters(),
                                                  std::move(dc_options),
                                                  m_InputOptions);
    return m_Likelihood;
  }

}  // namespace ana::dc
