#include "ICModule.h"

#include "../Fit.h"

#include "IceCube/ICWriteResults.h"

#include <memory>
#include <stdexcept>

namespace ana::ic {

  std::shared_ptr<Likelihood> ICExperimentModule::create_likelihood(std::shared_ptr<io::Options> options) {
    // Heavy parquet loading happens once, only for the selected experiment. The
    // ICDataBase is cached on the module so repeated Fit constructions (e.g. the
    // 2D scan) reuse the same immutable sample instead of re-reading the file.
    // The lock makes the first construction safe when several scan workers build
    // their Fits at the same time; the sample itself is const afterwards.
    std::shared_ptr<const io::ic::ICDataBase> database;
    {
      const std::scoped_lock lock(m_DataBaseMutex);
      if (m_DataBase == nullptr) {
        m_DataBase = std::make_shared<const io::ic::ICDataBase>(m_InputOptions->samples());
      }
      database = m_DataBase;
    }

    return std::make_shared<ICLikelihood>(std::move(options),
                                          std::move(database),
                                          *m_InputOptions);
  }

  void ICExperimentModule::write_results(Fit& fit, std::string_view name) {
    // The likelihood comes from the Fit being written, not from the module:
    // concurrent scan workers each own one, and a module-wide handle would let
    // them write each other's results.
    const auto likelihood = std::dynamic_pointer_cast<ICLikelihood>(fit.likelihood());
    if (likelihood == nullptr) {
      throw std::logic_error("ICExperimentModule::write_results called with a fit that holds no ICLikelihood");
    }
    result::ic::write_ice_cube_results(fit, *likelihood, *m_InputOptions, name);
  }

}  // namespace ana::ic
