#pragma once

#include "Options.h"
#include "ParameterWrapper.h"

namespace ana {

  // TODO Documentation
  class Likelihood {
   public:
    using transform_fn_t = ParameterWrapper::transform_fn_t;

    /**
     * @param options The options object used for likelihood calculation.
     * @param nParameter The number of fit parameters.
     * @param transform_fn Applied to every incoming parameter set before it is used, e.g. to
     *                     correlate parameters among each other. May be nullptr.
     */
    explicit Likelihood(std::shared_ptr<io::Options> options, int nParameter, transform_fn_t transform_fn = nullptr)
      : m_Options(std::move(options))
      , m_Parameter(nParameter, std::move(transform_fn)) {}

    virtual ~Likelihood() = default;

    [[nodiscard]] const std::shared_ptr<io::Options>& options() const noexcept { return m_Options; }

    [[nodiscard]] virtual double calculate_likelihood(const double* parameter) = 0;

    [[nodiscard]] ParameterWrapper& parameter() noexcept { return m_Parameter; }

   protected:
    std::shared_ptr<io::Options> m_Options;  ///< The options object used for likelihood calculation.

    ParameterWrapper m_Parameter;  ///< The parameter wrapper object used for likelihood calculation.
  };

}  // namespace ana
