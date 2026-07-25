#pragma once

#include "ICSample.h"
#include "SampleConfig.h"

#include <cstddef>
#include <vector>

#include <arrow/type_fwd.h>  // forward decls: arrow::Status, arrow::Table, arrow::Result

namespace io::ic {

  /**
   * Loads a config-driven set of IceCube analysis samples, each backed by its
   * own parquet file and binning. All per-event flux weights, Barr gradients
   * and the CR-model alternative weights are read once at construction;
   * reconstructed energy/zenith are used to assign analysis bins and then
   * discarded.
   */
  class ICDataBase {
   public:
    explicit ICDataBase(const std::vector<SampleConfig>& samples);
    ~ICDataBase() = default;

    [[nodiscard]] const ICSample& sample(std::size_t i) const noexcept { return m_Samples[i]; }
    [[nodiscard]] std::size_t     n_samples() const noexcept { return m_Samples.size(); }

   private:
    std::vector<ICSample> m_Samples;

    arrow::Status read_sample(const SampleConfig& cfg, ICSample& out);
  };

}  // namespace io::ic
