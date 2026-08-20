#include "DCExperimentModule.h"

#include "../Fit.h"

#include "DoubleChooz/DCWriteResults.h"

#include <stdexcept>

namespace ana::dc {

  std::shared_ptr<Likelihood> DCExperimentModule::create_likelihood(std::shared_ptr<io::Options> options, int /*worker_index*/) {
    auto dc_options = std::make_shared<const io::dc::DCOptions>(options->inputOptions(), *m_InputOptions);

    return std::make_shared<DCLikelihood>(std::move(options),
                                          number_of_parameters(),
                                          std::move(dc_options),
                                          m_InputOptions);
  }

  void DCExperimentModule::write_results(Fit& fit, std::string_view name) {
    const auto likelihood = std::dynamic_pointer_cast<DCLikelihood>(fit.likelihood());
    if (likelihood == nullptr) {
      throw std::logic_error("DCExperimentModule::write_results called with a fit that holds no DCLikelihood");
    }
    result::dc::write_double_chooz_results(fit, *likelihood, *m_InputOptions, name);
  }

}  // namespace ana::dc
