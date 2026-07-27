#include "ICDataBase.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
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

  arrow::Status ICDataBase::read_sample(const SampleConfig& cfg, ICSample& out) {
    // read_sample builds a fixed 2-element {e_reco, reco_zenith} reco array
    // below; Binning::bin_index reads reco[d] for d < n_axes(), so a binning
    // with more axes (e.g. a future RA axis) would read past the array.
    // Phase 3 will extend this deliberately; guard against it until then.
    if (cfg.binning.n_axes() != 2)
      return arrow::Status::Invalid(
          "ICDataBase::read_sample: only 2-axis binnings supported in Phase 1 (sample '" + cfg.name + "')");

    std::cout << "Reading IceCube sample '" << cfg.name << "': " << cfg.parquet << '\n';
    ARROW_ASSIGN_OR_RAISE(auto table, read_parquet_file(cfg.parquet));

    const auto& b = cfg.branches;

    // Reconstructed variables: only needed to assign analysis bins.
    ARROW_ASSIGN_OR_RAISE(auto e_reco,       get_double_column(*table, b.reco_energy));
    ARROW_ASSIGN_OR_RAISE(auto reco_zenith,  get_double_column(*table, b.reco_zenith));

    // Per-event fit-time columns. Only the components this sample declares are
    // read: a parquet that carries no atmospheric weights (or no astrophysical
    // baseline) still loads for a sample that does not ask for them.
    ARROW_ASSIGN_OR_RAISE(out.e_true, get_double_column(*table, b.true_energy));

    if (cfg.wants_astro()) {
      ARROW_ASSIGN_OR_RAISE(out.astro_baseline, get_double_column(*table, b.astro_baseline));
    }

    if (cfg.wants_atmospheric()) {
      ARROW_ASSIGN_OR_RAISE(out.conv_baseline,   get_double_column(*table, b.conv_baseline));
      ARROW_ASSIGN_OR_RAISE(out.conv_alt,        get_double_column(*table, b.conv_alt));
      ARROW_ASSIGN_OR_RAISE(out.prompt_baseline, get_double_column(*table, b.prompt_baseline));
      ARROW_ASSIGN_OR_RAISE(out.prompt_alt,      get_double_column(*table, b.prompt_alt));

      for (int k = 0; k < params::ic::nBarrParams; ++k) {
        ARROW_ASSIGN_OR_RAISE(out.barr_conv[k], get_double_column(*table, b.barr_conv[k]));
      }

      // Veto passing-fraction coefficients {a, b, c} per component. Dimensionless,
      // so deliberately left out of the livetime scaling below.
      if (cfg.wants_veto()) {
        for (int k = 0; k < 3; ++k) {
          ARROW_ASSIGN_OR_RAISE(out.veto_conv[k], get_double_column(*table, b.veto_conv[k]));
          ARROW_ASSIGN_OR_RAISE(out.veto_prompt[k], get_double_column(*table, b.veto_prompt[k]));
        }
      }

      // NNMFit's OscillationsHook multiplies the atmospheric baseline weights by a
      // per-event nu_mu survival probability, once at load time. The factor lives
      // in a sidecar parquet, row-aligned with this sample's baseline file, so the
      // hot loop is untouched. Barr slopes are scaled too: they are derivatives of
      // the same conventional weight, so leaving them unscaled would change the
      // (slope / baseline) reweight ratios.
      if (!cfg.oscillation_file.empty()) {
        ARROW_ASSIGN_OR_RAISE(auto osc_table, read_parquet_file(cfg.oscillation_file));
        ARROW_ASSIGN_OR_RAISE(auto survival, get_double_column(*osc_table, cfg.oscillation_branch));
        if (survival.size() != out.conv_baseline.size())
          return arrow::Status::Invalid(
              "ICDataBase: oscillation sidecar '" + cfg.oscillation_file + "' has " +
              std::to_string(survival.size()) + " rows, the baseline parquet has " +
              std::to_string(out.conv_baseline.size()) + " (they must be row-aligned)");

        auto apply_survival = [&survival](std::vector<double>& column) {
          for (std::size_t i = 0; i < column.size(); ++i) column[i] *= survival[i];
        };
        apply_survival(out.conv_baseline);
        apply_survival(out.conv_alt);
        apply_survival(out.prompt_baseline);
        apply_survival(out.prompt_alt);
        for (auto& slope : out.barr_conv) apply_survival(slope);

        double mean = 0.0;
        for (const double v : survival) mean += v;
        std::cout << "IceCube sample '" << cfg.name << "': applied oscillation survival factors (mean "
                  << mean / static_cast<double>(survival.size()) << ")\n";
      }
    }

    // The MC weights are per-event rates (Hz); scale by the livetime so the
    // binned prediction is in event counts. Baseline weights and their Barr
    // slopes are scaled together, which leaves the (slope/base) reweight ratios
    // and the CR-gradient ratios exactly invariant.
    const double livetime = cfg.livetime;
    if (livetime != 1.0) {
      auto scale = [livetime](std::vector<double>& col) {
        for (double& v : col) v *= livetime;
      };
      scale(out.astro_baseline);
      scale(out.conv_baseline);
      scale(out.conv_alt);
      scale(out.prompt_baseline);
      scale(out.prompt_alt);
      for (auto& g : out.barr_conv) scale(g);
    }

    // Assign each event to an analysis bin from its reco energy and zenith.
    const std::size_t N = out.e_true.size();
    out.bin_idx.resize(N);
    for (std::size_t i = 0; i < N; ++i) {
      const std::array<double, 2> reco{e_reco[i], reco_zenith[i]};
      out.bin_idx[i] = cfg.binning.bin_index(reco);
    }

    // Compact to in-range events, group by bin, build the CSR index.
    out.sort_into_bins(cfg.binning.total_bins());

    std::cout << "IceCube sample '" << cfg.name << "' loaded: " << N << " rows, "
              << out.size() << " in analysis range ("
              << cfg.binning.total_bins() << " bins)\n";

    return arrow::Status::OK();
  }

  ICDataBase::ICDataBase(const std::vector<SampleConfig>& samples) {
    for (const std::size_t i : enabled_sample_indices(samples)) {
      const SampleConfig& cfg = samples[i];
      ICSample            sample;
      const auto          status = read_sample(cfg, sample);
      if (!status.ok())
        throw std::runtime_error("Failed to read IceCube sample '" + cfg.name + "': " + status.ToString());
      m_Samples.push_back(std::move(sample));
    }
    if (m_Samples.empty())
      throw std::runtime_error("ICDataBase: no enabled IceCube samples");
  }

}  // namespace io::ic
