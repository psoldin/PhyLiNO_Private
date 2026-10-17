#pragma once

#include "ICSample.h"
#include "SampleConfig.h"

#include <cstddef>
#include <memory>
#include <vector>

#include <arrow/type_fwd.h>  // forward decls: arrow::Status, arrow::Table, arrow::Result

namespace io::ic {

  /**
   * Loads a config-driven set of IceCube analysis samples, each backed by its
   * own parquet file and binning. All per-event flux weights, Barr gradients
   * and the CR-model alternative weights are read once at construction;
   * reconstructed energy/zenith are used to assign analysis bins and then
   * discarded.
   *
   * The MC samples are held behind a shared pointer: with_data() hands out a
   * database that shares them but carries the data histograms of another config,
   * so a process fitting many pseudo-experiments (LLHBatch) reads the parquet once.
   * The GPU column cache is keyed on the sample column pointers, which stay the
   * same across such databases.
   */
  class ICDataBase {
   public:
    explicit ICDataBase(const std::vector<SampleConfig>& samples);
    ~ICDataBase() = default;

    [[nodiscard]] const ICSample& sample(std::size_t i) const noexcept { return (*m_Samples)[i]; }
    [[nodiscard]] std::size_t     n_samples() const noexcept { return m_Samples->size(); }

    /**
     * A database with this one's MC samples (shared, not copied) and the data
     * histograms `samples` configures ("DataCounts" or "data"). The caller vouches
     * that `samples` describes the same MC as the config this database was built
     * from; only the enabled-sample count is checked here.
     */
    [[nodiscard]] ICDataBase with_data(const std::vector<SampleConfig>& samples) const;

    /** Per-bin measured counts for sample i; empty when the config gave no data path. */
    [[nodiscard]] const std::vector<double>& data_histogram(std::size_t i) const noexcept {
      return m_DataHistograms[i];
    }

   private:
    std::shared_ptr<const std::vector<ICSample>> m_Samples;
    std::vector<std::vector<double>>             m_DataHistograms;

    ICDataBase() = default;
    void read_data_histograms(const std::vector<SampleConfig>& samples);

    arrow::Status read_sample(const SampleConfig& cfg, ICSample& out);
    arrow::Status read_data_histogram(const SampleConfig& cfg, std::vector<double>& out);
  };

}  // namespace io::ic
