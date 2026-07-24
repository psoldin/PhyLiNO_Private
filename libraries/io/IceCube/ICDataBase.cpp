#include "ICDataBase.h"

#include "ICConstants.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>

#include <parquet/arrow/reader.h>

namespace io::ic {

  namespace {

    arrow::Result<std::shared_ptr<arrow::Table>> read_parquet_file(const std::string& filename) {
      ARROW_ASSIGN_OR_RAISE(auto input, arrow::io::ReadableFile::Open(filename));
      ARROW_ASSIGN_OR_RAISE(auto reader,
                            parquet::arrow::OpenFile(input, arrow::default_memory_pool()));

      std::shared_ptr<arrow::Table> table;
      ARROW_RETURN_NOT_OK(reader->ReadTable(&table));
      ARROW_ASSIGN_OR_RAISE(auto combined, table->CombineChunks());
      return combined;
    }

    // Fetch one CombineChunks'd double column by name into a dense vector; null -> 0.
    arrow::Result<std::vector<double>> get_double_column(const arrow::Table& table,
                                                         const std::string&  name) {
      auto col = table.GetColumnByName(name);
      if (!col)
        return arrow::Status::Invalid("ICDataBase: missing required column '" + name + "'");

      auto array = std::static_pointer_cast<arrow::DoubleArray>(col->chunk(0));
      std::vector<double> out;
      out.reserve(array->length());
      for (int64_t i = 0, n = array->length(); i < n; ++i)
        out.push_back(array->IsNull(i) ? 0.0 : array->Value(i));
      return out;
    }

  }  // namespace

  arrow::Status ICDataBase::read_track_baseline(const ICInputOptions& options) {
    const std::string& path = options.track_baseline_file_path();
    std::cout << "Reading IceCube track baseline: " << path << '\n';
    ARROW_ASSIGN_OR_RAISE(auto table, read_parquet_file(path));

    const auto& b = options.branch_names();

    // Reconstructed variables: only needed to assign analysis bins.
    ARROW_ASSIGN_OR_RAISE(auto e_reco,       get_double_column(*table, b.reco_energy));
    ARROW_ASSIGN_OR_RAISE(auto reco_zenith,  get_double_column(*table, b.reco_zenith));

    // Per-event fit-time columns.
    ARROW_ASSIGN_OR_RAISE(m_Sample.e_true,          get_double_column(*table, b.true_energy));
    ARROW_ASSIGN_OR_RAISE(m_Sample.astro_baseline,  get_double_column(*table, b.astro_baseline));
    ARROW_ASSIGN_OR_RAISE(m_Sample.conv_baseline,   get_double_column(*table, b.conv_baseline));
    ARROW_ASSIGN_OR_RAISE(m_Sample.conv_alt,        get_double_column(*table, b.conv_alt));
    ARROW_ASSIGN_OR_RAISE(m_Sample.prompt_baseline, get_double_column(*table, b.prompt_baseline));
    ARROW_ASSIGN_OR_RAISE(m_Sample.prompt_alt,      get_double_column(*table, b.prompt_alt));

    for (int k = 0; k < params::ic::nBarrParams; ++k) {
      ARROW_ASSIGN_OR_RAISE(m_Sample.barr_conv[k], get_double_column(*table, b.barr_conv[k]));
    }

    // The MC weights are per-event rates (Hz); scale by the livetime so the
    // binned prediction is in event counts. Baseline weights and their Barr
    // slopes are scaled together, which leaves the (slope/base) reweight ratios
    // and the CR-gradient ratios exactly invariant.
    const double livetime = options.livetime();
    if (livetime != 1.0) {
      auto scale = [livetime](std::vector<double>& col) {
        for (double& v : col) v *= livetime;
      };
      scale(m_Sample.astro_baseline);
      scale(m_Sample.conv_baseline);
      scale(m_Sample.conv_alt);
      scale(m_Sample.prompt_baseline);
      scale(m_Sample.prompt_alt);
      for (auto& g : m_Sample.barr_conv) scale(g);
    }

    // Assign each event to an analysis bin from its reco energy and zenith.
    const std::size_t N = m_Sample.e_true.size();
    m_Sample.bin_idx.resize(N);
    for (std::size_t i = 0; i < N; ++i)
      m_Sample.bin_idx[i] = Constants::bin_index(e_reco[i], reco_zenith[i]);

    // Compact to in-range events, group by bin, build the CSR index.
    m_Sample.sort_into_bins();

    std::cout << "IceCube MC loaded: " << N << " rows, "
              << m_Sample.size() << " in analysis range ("
              << Constants::nBins << " bins)\n";

    return arrow::Status::OK();
  }

  ICDataBase::ICDataBase(const ICInputOptions& options) {
    auto status = read_track_baseline(options);
    if (!status.ok())
      throw std::runtime_error("Failed to read IceCube track baseline: " + status.ToString());
  }

}  // namespace io::ic
