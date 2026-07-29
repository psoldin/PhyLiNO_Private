#pragma once

// includes
#include "DCInputOptions.h"
#include "DataBase.h"
#include "StartingParameter.h"

namespace io::dc {

  class DCOptions {
  public:
    explicit DCOptions(const InputOptions& inputOptions, const DCInputOptions& dc_input_options)
      : m_DataBase(inputOptions, dc_input_options)
      , m_StartingParameter(inputOptions)
      , m_UseMultiThreading(inputOptions.use_multi_threading()) { }

    ~DCOptions() = default;

    [[nodiscard]] const DataBase& dataBase() const noexcept {
      return m_DataBase;
    }

    [[nodiscard]] const StartingParameter& starting_parameters() const noexcept {
      return m_StartingParameter;
    }

    [[nodiscard]] bool use_multi_threading() const noexcept {
      return m_UseMultiThreading;
    }

  private:
    DataBase m_DataBase;
    StartingParameter m_StartingParameter;
    bool m_UseMultiThreading;
  };

} // namespace io::dc