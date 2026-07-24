#include "ICModule.h"

#include "../Fit.h"

#include "IceCube/ICWriteResults.h"

#include <stdexcept>

namespace ana::ic {

  std::shared_ptr<Likelihood> ICExperimentModule::create_likelihood(std::shared_ptr<io::Options> options) {
    // Heavy parquet loading happens here, only for the selected experiment.
    auto data_base = std::make_shared<const io::ic::ICDataBase>(*m_InputOptions);

    m_Likelihood = std::make_shared<ICLikelihood>(std::move(options),
                                                  std::move(data_base),
                                                  *m_InputOptions);
    return m_Likelihood;
  }

  void ICExperimentModule::write_results(Fit& fit, std::string_view name) {
    if (m_Likelihood == nullptr) {
      throw std::logic_error("ICExperimentModule::write_results called before create_likelihood");
    }
    result::ic::write_ice_cube_results(fit, *m_Likelihood, *m_InputOptions, name);
  }

}  // namespace ana::ic
