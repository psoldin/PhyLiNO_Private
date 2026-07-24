#pragma once

#include "ICSample.h"
#include "ICInputOptions.h"

#include <string>

#include <arrow/type_fwd.h>  // forward decls: arrow::Status, arrow::Table, arrow::Result

namespace io::ic {

  /**
   * Loads the IceCube tracks-only MC baseline parquet into an ICSample (SoA).
   * All per-event flux weights, Barr gradients and the CR-model alternative
   * weights are read once at construction; reconstructed energy/zenith are used
   * to assign analysis bins and then discarded.
   */
  class ICDataBase {
   public:
    explicit ICDataBase(const ICInputOptions& options);
    ~ICDataBase() = default;

    [[nodiscard]] const ICSample& sample() const noexcept { return m_Sample; }

   private:
    ICSample m_Sample;

    arrow::Status read_track_baseline(const ICInputOptions& options);
  };

}  // namespace io::ic
