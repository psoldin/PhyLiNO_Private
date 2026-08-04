#include "ICDataBase.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/util/config.h>  // ARROW_VERSION_MAJOR, for the OpenFile signature below

#include <parquet/arrow/reader.h>

namespace io::ic {

  namespace {

    arrow::Result<std::shared_ptr<arrow::Table>> read_parquet_file(const std::string& filename) {
      ARROW_ASSIGN_OR_RAISE(auto input, arrow::io::ReadableFile::Open(filename));

      // Arrow 19 replaced the out-parameter form of parquet::arrow::OpenFile
      // with one returning a Result. Both are in use here: the HPC cluster
      // builds against an older Arrow than a current Homebrew install.
#if ARROW_VERSION_MAJOR >= 19
      ARROW_ASSIGN_OR_RAISE(auto reader, parquet::arrow::OpenFile(input, arrow::default_memory_pool()));
#else
      std::unique_ptr<parquet::arrow::FileReader> reader;
      ARROW_RETURN_NOT_OK(parquet::arrow::OpenFile(input, arrow::default_memory_pool(), &reader));
#endif

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

      auto                array = std::static_pointer_cast<arrow::DoubleArray>(col->chunk(0));
      std::vector<double> out;
      out.reserve(array->length());
      for (int64_t i = 0, n = array->length(); i < n; ++i)
        out.push_back(array->IsNull(i) ? 0.0 : array->Value(i));
      return out;
    }

    // NNMFit's standard mask (MaskHandler._make_standard_mask): an event passes
    // if its reco-energy and reco-direction fits both exist and succeeded.
    // Measured 2026-07-27: 0% dropped on both cascade MC baselines and both
    // cascade data files, but 0.846% dropped on the tracks data file, so this is
    // applied unconditionally to real data (never to the MC baselines, which
    // ICDataBase::read_sample does not call this from).
    arrow::Result<std::vector<bool>> standard_mask(const arrow::Table& table, const std::string& reco_energy_branch) {
      auto get               = [&table](const std::string& name) { return table.GetColumnByName(name); };
      auto energy_exists     = get(reco_energy_branch + "_exists");
      auto energy_fit_status = get(reco_energy_branch + "_fit_status");
      auto dir_exists        = get("reco_dir_exists");
      auto dir_fit_status    = get("reco_dir_fit_status");
      if (!energy_exists || !energy_fit_status || !dir_exists || !dir_fit_status)
        return arrow::Status::Invalid("ICDataBase: missing standard-mask columns ('" + reco_energy_branch + "_exists', '" + reco_energy_branch + "_fit_status', 'reco_dir_exists', 'reco_dir_fit_status')");

      auto ee = std::static_pointer_cast<arrow::UInt8Array>(energy_exists->chunk(0));
      auto es = std::static_pointer_cast<arrow::Int32Array>(energy_fit_status->chunk(0));
      auto de = std::static_pointer_cast<arrow::UInt8Array>(dir_exists->chunk(0));
      auto ds = std::static_pointer_cast<arrow::Int32Array>(dir_fit_status->chunk(0));

      const int64_t     n = table.num_rows();
      std::vector<bool> pass(n, false);
      for (int64_t i = 0; i < n; ++i)
        pass[i] = ee->Value(i) == 1 && es->Value(i) == 0 && de->Value(i) == 1 && ds->Value(i) == 0;
      return pass;
    }

  }  // namespace

  arrow::Status ICDataBase::read_sample(const SampleConfig& cfg, ICSample& out) {
    // read_sample builds a fixed 2-element {e_reco, reco_zenith} reco array below, and
    // MC is always binned in the RA-free mc_binning: NNMFit's Binning_2D_to_3D bins the
    // events in 2D and spreads the result over RA (see SampleLikelihood), so the
    // per-event path never sees an RA axis.
    if (cfg.mc_binning.n_axes() != 2)
      return arrow::Status::Invalid(
          "ICDataBase::read_sample: only 2-axis MC binnings are supported (sample '" + cfg.name + "')");

    std::cout << "Reading IceCube sample '" << cfg.name << "': " << cfg.parquet << '\n';
    ARROW_ASSIGN_OR_RAISE(auto table, read_parquet_file(cfg.parquet));

    const auto& b = cfg.branches;

    // Reconstructed variables: only needed to assign analysis bins.
    ARROW_ASSIGN_OR_RAISE(auto e_reco, get_double_column(*table, b.reco_energy));
    ARROW_ASSIGN_OR_RAISE(auto reco_zenith, get_double_column(*table, b.reco_zenith));

    // Per-event fit-time columns. Only the components this sample declares are
    // read: a parquet that carries no atmospheric weights (or no astrophysical
    // baseline) still loads for a sample that does not ask for them.
    ARROW_ASSIGN_OR_RAISE(out.e_true, get_double_column(*table, b.true_energy));

    if (cfg.wants_astro()) {
      ARROW_ASSIGN_OR_RAISE(out.astro_baseline, get_double_column(*table, b.astro_baseline));
    }

    if (cfg.wants_atmospheric()) {
      ARROW_ASSIGN_OR_RAISE(out.conv_baseline, get_double_column(*table, b.conv_baseline));
      ARROW_ASSIGN_OR_RAISE(out.conv_alt, get_double_column(*table, b.conv_alt));
      ARROW_ASSIGN_OR_RAISE(out.prompt_baseline, get_double_column(*table, b.prompt_baseline));
      ARROW_ASSIGN_OR_RAISE(out.prompt_alt, get_double_column(*table, b.prompt_alt));

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
              "ICDataBase: oscillation sidecar '" + cfg.oscillation_file + "' has " + std::to_string(survival.size()) + " rows, the baseline parquet has " + std::to_string(out.conv_baseline.size()) + " (they must be row-aligned)");

        // NNMFit's OscillationsHook is constructed per-flux with the flux's own
        // self._baseline_weight as its target column (Flux.py::apply_hooks_for_flux:
        // hook_obj = hook_cls(self._baseline_weight, **self._hooks[hook])), so it
        // multiplies ONLY the primary baseline weight (conv_baseline / prompt_baseline).
        // The CR-gradient alternative (conv_alt/prompt_alt) and the Barr slope columns
        // are separate flux graphs the hook never touches.
        auto apply_survival = [&survival](std::vector<double>& column) {
          for (std::size_t i = 0; i < column.size(); ++i)
            column[i] *= survival[i];
        };
        apply_survival(out.conv_baseline);
        apply_survival(out.prompt_baseline);

        double mean = 0.0;
        for (const double v : survival)
          mean += v;
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
        for (double& v : col)
          v *= livetime;
      };
      scale(out.astro_baseline);
      scale(out.conv_baseline);
      scale(out.conv_alt);
      scale(out.prompt_baseline);
      scale(out.prompt_alt);
      for (auto& g : out.barr_conv)
        scale(g);
    }

    // Assign each event to an analysis bin from its reco energy and zenith.
    const std::size_t N = out.e_true.size();
    out.bin_idx.resize(N);
    for (std::size_t i = 0; i < N; ++i) {
      const std::array<double, 2> reco{e_reco[i], reco_zenith[i]};
      out.bin_idx[i] = cfg.mc_binning.bin_index(reco);
    }

    // Compact to in-range events, group by bin, build the CSR index.
    out.sort_into_bins(cfg.mc_binning.total_bins());

    std::cout << "IceCube sample '" << cfg.name << "' loaded: " << N << " rows, "
              << out.size() << " in analysis range ("
              << cfg.mc_binning.total_bins() << " MC bins, "
              << cfg.binning.total_bins() << " analysis bins)\n";

    return arrow::Status::OK();
  }

  arrow::Status ICDataBase::read_data_histogram(const SampleConfig& cfg, std::vector<double>& out) {
    // Pre-binned counts win over the parquet: the point of "DataCounts" is to
    // fit numbers that came from somewhere else (a pseudo-experiment shared
    // with NNMFit), so silently preferring a configured parquet would defeat it.
    if (!cfg.data_counts_path.empty()) {
      std::cout << "Reading IceCube data counts for sample '" << cfg.name << "': "
                << cfg.data_counts_path << '\n';
      std::ifstream in(cfg.data_counts_path);
      if (!in)
        return arrow::Status::IOError("cannot open DataCounts file '" + cfg.data_counts_path + "'");

      const int   total_bins = cfg.binning.total_bins();
      std::string line;
      out.clear();
      out.reserve(total_bins);
      while (std::getline(in, line)) {
        if (line.empty() || line.front() == '#') continue;
        std::istringstream row(line);
        double             count = 0.0;
        if (!(row >> count)) continue;
        out.push_back(count);
      }
      if (out.size() != static_cast<std::size_t>(total_bins))
        return arrow::Status::Invalid("DataCounts file '" + cfg.data_counts_path + "' has " +
                                      std::to_string(out.size()) + " values, sample '" + cfg.name +
                                      "' has " + std::to_string(total_bins) + " bins");

      double total = 0.0;
      for (const double v : out) total += v;
      std::cout << "IceCube data counts '" << cfg.name << "': " << total << " events in "
                << out.size() << " bins\n";
      return arrow::Status::OK();
    }

    if (cfg.data_path.empty()) {
      out.clear();
      return arrow::Status::OK();
    }

    std::cout << "Reading IceCube data for sample '" << cfg.name << "': " << cfg.data_path << '\n';
    ARROW_ASSIGN_OR_RAISE(auto table, read_parquet_file(cfg.data_path));

    const auto& b = cfg.branches;
    ARROW_ASSIGN_OR_RAISE(auto energy, get_double_column(*table, b.reco_energy));
    ARROW_ASSIGN_OR_RAISE(auto zenith, get_double_column(*table, b.reco_zenith));
    const std::size_t n_rows = energy.size();

    // Data is binned in the full analysis binning (NNMFit Binning_2D_to_3D bins data
    // truly 3D), so a sample with an RA axis needs its reco RA column too. The test
    // is on the axis structure, not on the RA bin count: a one-bin RA axis is still
    // an RA axis, and feeding bin_index only two of three reco values would read
    // past the end of the array it is handed.
    const bool          needs_ra = io::ic::has_ra_axis(cfg.binning);
    std::vector<double> ra;
    if (needs_ra) {
      ARROW_ASSIGN_OR_RAISE(ra, get_double_column(*table, b.reco_ra));
    }

    // NNMFit's standard mask: apply it to real data (the MC baselines are
    // measured pre-cut and read_sample does not apply it there). Compact to the
    // passing rows rather than poisoning failing ones with a sentinel: NaN would
    // NOT be rejected by Axis::index's `v < lo || v >= hi` (both comparisons are
    // false for NaN), so it would fall through to an undefined-behaviour cast.
    ARROW_ASSIGN_OR_RAISE(auto passes, standard_mask(*table, b.reco_energy));
    std::vector<double> masked_energy;
    std::vector<double> masked_zenith;
    std::vector<double> masked_ra;
    masked_energy.reserve(n_rows);
    masked_zenith.reserve(n_rows);
    if (needs_ra)
      masked_ra.reserve(n_rows);
    for (std::size_t i = 0; i < n_rows; ++i) {
      if (passes[i]) {
        masked_energy.push_back(energy[i]);
        masked_zenith.push_back(zenith[i]);
        if (needs_ra)
          masked_ra.push_back(ra[i]);
      }
    }
    if (masked_energy.size() != n_rows)
      std::cout << "IceCube data '" << cfg.name << "': standard mask dropped " << (n_rows - masked_energy.size())
                << " of " << n_rows << " rows ("
                << (100.0 * static_cast<double>(n_rows - masked_energy.size()) / static_cast<double>(n_rows))
                << "%)\n";

    out          = needs_ra ? bin_event_counts(cfg.binning, masked_energy, masked_zenith, masked_ra)
                            : bin_event_counts(cfg.binning, masked_energy, masked_zenith);
    double total = 0.0;
    for (const double v : out)
      total += v;
    std::cout << "IceCube data '" << cfg.name << "': " << n_rows << " rows, " << total
              << " in analysis range\n";
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

      std::vector<double> data_histogram;
      const auto          data_status = read_data_histogram(cfg, data_histogram);
      if (!data_status.ok())
        throw std::runtime_error("Failed to read IceCube data for sample '" + cfg.name + "': " + data_status.ToString());
      m_DataHistograms.push_back(std::move(data_histogram));
    }
    if (m_Samples.empty())
      throw std::runtime_error("ICDataBase: no enabled IceCube samples");
  }

}  // namespace io::ic
