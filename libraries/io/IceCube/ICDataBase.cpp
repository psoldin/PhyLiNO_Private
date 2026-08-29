#include "ICDataBase.h"

#include <algorithm>
#include <array>
#include <cmath>
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

      // The cast below is unchecked, so an int-typed column would be reinterpreted
      // as doubles and read as garbage rather than failing. Every weight branch is
      // a double, but config-named columns (topology) can be anything.
      if (col->type()->id() != arrow::Type::DOUBLE)
        return arrow::Status::Invalid("ICDataBase: column '" + name + "' has type " + col->type()->ToString() +
                                      ", expected double");

      auto                array = std::static_pointer_cast<arrow::DoubleArray>(col->chunk(0));
      std::vector<double> out;
      out.reserve(array->length());
      for (int64_t i = 0, n = array->length(); i < n; ++i)
        out.push_back(array->IsNull(i) ? 0.0 : array->Value(i));
      return out;
    }

    // Same, but tolerant of the numeric type: the diagnostic category columns are
    // whatever the file happens to carry (bdt_score is float where the weight
    // branches are double), and reading one as garbage would be worse than a
    // slower branch here. Never used for anything the fit reads.
    arrow::Result<std::vector<double>> get_numeric_column(const arrow::Table& table,
                                                          const std::string&  name) {
      auto col = table.GetColumnByName(name);
      if (!col)
        return arrow::Status::Invalid("ICDataBase: missing category column '" + name + "'");

      const auto          chunk = col->chunk(0);
      std::vector<double> out;
      out.reserve(chunk->length());

      if (chunk->type_id() == arrow::Type::DOUBLE) {
        const auto array = std::static_pointer_cast<arrow::DoubleArray>(chunk);
        for (int64_t i = 0, n = array->length(); i < n; ++i)
          out.push_back(array->IsNull(i) ? 0.0 : array->Value(i));
      } else if (chunk->type_id() == arrow::Type::FLOAT) {
        const auto array = std::static_pointer_cast<arrow::FloatArray>(chunk);
        for (int64_t i = 0, n = array->length(); i < n; ++i)
          out.push_back(array->IsNull(i) ? 0.0 : static_cast<double>(array->Value(i)));
      } else if (chunk->type_id() == arrow::Type::INT64) {
        const auto array = std::static_pointer_cast<arrow::Int64Array>(chunk);
        for (int64_t i = 0, n = array->length(); i < n; ++i)
          out.push_back(array->IsNull(i) ? 0.0 : static_cast<double>(array->Value(i)));
      } else if (chunk->type_id() == arrow::Type::INT32) {
        const auto array = std::static_pointer_cast<arrow::Int32Array>(chunk);
        for (int64_t i = 0, n = array->length(); i < n; ++i)
          out.push_back(array->IsNull(i) ? 0.0 : static_cast<double>(array->Value(i)));
      } else {
        return arrow::Status::Invalid("ICDataBase: category column '" + name + "' has type " +
                                      chunk->type()->ToString() + ", expected a numeric type");
      }
      return out;
    }

    // The sample's configured topology cut, as a dense keep-mask over the table's
    // rows: an event is kept if its topology column equals one of the configured
    // class labels, or (Topology.Exclude) kept unless it does. Returns an empty
    // mask when no cut is configured, which callers read as "keep everything".
    // The labels are compared as doubles rather than casting the column to int:
    // a non-integral entry then compares false instead of hitting the
    // undefined-behaviour cast. A NaN entry passes the cut by default (it is not
    // a class label, so it is not "excluded" by any Values list either) --
    // Topology.Values listing "NaN" opts into dropping it instead, for the one
    // sample that specifically wants NaNs gone.
    arrow::Result<std::vector<bool>> topology_mask(const arrow::Table& table, const SampleConfig& cfg) {
      if (!cfg.filters_topology())
        return std::vector<bool>{};

      ARROW_ASSIGN_OR_RAISE(auto column, get_double_column(table, cfg.topology_branch));

      std::vector<bool> keep(column.size(), false);
      for (std::size_t i = 0; i < column.size(); ++i) {
        const double value = column[i];
        if (std::isnan(value)) {
          keep[i] = !cfg.topology_drop_nan;
          continue;
        }
        const bool matches = std::ranges::any_of(
            cfg.topology_values, [value](const int label) { return value == static_cast<double>(label); });
        keep[i] = cfg.topology_exclude ? !matches : matches;
      }
      return keep;
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
    // MC is always binned in the RA-free mc_binning: NNMFit's Binning_2D_to_3D bins
    // the events in 2D and spreads the result over RA (see SampleLikelihood), so the
    // per-event path never sees an RA axis. Beyond energy and zenith the binning may
    // carry any number of Category axes, which ARE per-event and are read below.
    const std::vector<Axis> mc_categories = category_axes(cfg.mc_binning);
    if (cfg.mc_binning.n_axes() != 2 + mc_categories.size())
      return arrow::Status::Invalid(
          "ICDataBase::read_sample: the MC binning of sample '" + cfg.name +
          "' must be (Log10Energy, CosZenith) followed by Category axes only");
    if (cfg.mc_binning.axes()[0].kind != Axis::Kind::Log10Energy ||
        cfg.mc_binning.axes()[1].kind != Axis::Kind::CosZenith)
      return arrow::Status::Invalid(
          "ICDataBase::read_sample: sample '" + cfg.name +
          "' must bin energy first and zenith second; the flux components index on that order");

    std::cout << "Reading IceCube sample '" << cfg.name << "': " << cfg.parquet << '\n';
    ARROW_ASSIGN_OR_RAISE(auto table, read_parquet_file(cfg.parquet));

    const auto& b = cfg.branches;

    // Reconstructed variables: only needed to assign analysis bins.
    ARROW_ASSIGN_OR_RAISE(auto e_reco, get_double_column(*table, b.reco_energy));
    ARROW_ASSIGN_OR_RAISE(auto reco_zenith, get_double_column(*table, b.reco_zenith));

    for (const std::string& branch : cfg.category_branches) {
      ARROW_ASSIGN_OR_RAISE(auto column, get_numeric_column(*table, branch));
      out.category_names.push_back(branch);
      out.categories.push_back(std::move(column));
    }

    // Forward-folding inputs: the response centres and the per-event widths.
    // Read here rather than in the likelihood because they are
    // parameter-independent and ICDataBase is what is cached for the process.
    if (cfg.response.enabled) {
      ARROW_ASSIGN_OR_RAISE(out.response_truth_log_e,
                            get_double_column(*table, cfg.response.truth_energy_branch));
      ARROW_ASSIGN_OR_RAISE(out.response_truth_zenith,
                            get_double_column(*table, cfg.response.truth_zenith_branch));
      ARROW_ASSIGN_OR_RAISE(auto sigma_e, get_double_column(*table, cfg.response.energy_sigma_branch));
      ARROW_ASSIGN_OR_RAISE(auto sigma_z, get_double_column(*table, cfg.response.zenith_sigma_branch));

      auto apply = [](const double x, const SigmaTransform transform) noexcept {
        constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
        switch (transform) {
          case SigmaTransform::None: return x;
          case SigmaTransform::Exp: return std::exp(x);
          case SigmaTransform::Pow10: return std::pow(10.0, x);
          case SigmaTransform::DegToRad: return x * kDegToRad;
        }
        return x;
      };
      out.response_sigma_log_e.resize(sigma_e.size());
      out.response_sigma_zenith.resize(sigma_z.size());
      for (std::size_t i = 0; i < sigma_e.size(); ++i)
        out.response_sigma_log_e[i] = apply(sigma_e[i], cfg.response.energy_sigma_transform);
      for (std::size_t i = 0; i < sigma_z.size(); ++i)
        out.response_sigma_zenith[i] = apply(sigma_z[i], cfg.response.zenith_sigma_transform);
    }

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

    // Events the configured topology cut rejects are marked out-of-range rather
    // than compacted here: sort_into_bins() already drops bin_idx < 0 from every
    // per-event column at once, so the cut needs no per-column bookkeeping of its
    // own and cannot go stale when a column is added.
    ARROW_ASSIGN_OR_RAISE(const auto topology_keeps, topology_mask(*table, cfg));
    if (!topology_keeps.empty() && topology_keeps.size() != out.e_true.size())
      return arrow::Status::Invalid("ICDataBase: topology column '" + cfg.topology_branch + "' has " +
                                    std::to_string(topology_keeps.size()) + " rows, the weight columns have " +
                                    std::to_string(out.e_true.size()));

    // Assign each event to an analysis bin from its reco energy and zenith.
    const std::size_t N = out.e_true.size();
    out.bin_idx.resize(N);

    // Per-bin surviving weight fraction, for gradients exported from the
    // unfiltered sample (see ICSample::topology_bin_fraction). The weight used is
    // the sum of the baselines the sample declares, i.e. the nominal composition
    // before any parameter is applied; the ratio is livetime-independent, so it
    // does not matter that this runs before the livetime scaling above.
    const bool          want_fraction = cfg.filters_topology() && cfg.scale_gradients_to_topology;
    std::vector<double> kept_weight;
    std::vector<double> full_weight;
    if (want_fraction) {
      kept_weight.assign(cfg.mc_binning.total_bins(), 0.0);
      full_weight.assign(cfg.mc_binning.total_bins(), 0.0);
    }
    auto nominal_weight = [&out](const std::size_t i) {
      double w = 0.0;
      if (!out.astro_baseline.empty()) w += out.astro_baseline[i];
      if (!out.conv_baseline.empty()) w += out.conv_baseline[i];
      if (!out.prompt_baseline.empty()) w += out.prompt_baseline[i];
      return w;
    };

    // Category axes read raw reco columns, one per axis, in axis order.
    std::vector<std::vector<double>> category_values;
    for (const Axis& axis : mc_categories) {
      ARROW_ASSIGN_OR_RAISE(auto column, get_numeric_column(*table, axis.branch));
      category_values.push_back(std::move(column));
    }

    std::size_t         topology_dropped = 0;
    std::vector<double> reco(2 + mc_categories.size(), 0.0);
    for (std::size_t i = 0; i < N; ++i) {
      reco[0] = e_reco[i];
      reco[1] = reco_zenith[i];
      for (std::size_t c = 0; c < category_values.size(); ++c) reco[2 + c] = category_values[c][i];
      const int bin = cfg.mc_binning.bin_index(reco);
      const bool                  kept = topology_keeps.empty() || topology_keeps[i];

      if (want_fraction && bin >= 0) {
        const double w = nominal_weight(i);
        full_weight[bin] += w;
        if (kept) kept_weight[bin] += w;
      }

      out.bin_idx[i] = kept ? bin : -1;
      if (!kept) ++topology_dropped;
    }

    if (want_fraction) {
      out.topology_bin_fraction.assign(full_weight.size(), 0.0);
      for (std::size_t b = 0; b < full_weight.size(); ++b)
        // An empty bin has no gradient worth keeping either, so 0 is the right
        // fill: it cannot be a division and must not default to 1.
        out.topology_bin_fraction[b] = full_weight[b] > 0.0 ? kept_weight[b] / full_weight[b] : 0.0;
    }

    // Compact to in-range events, group by bin, build the CSR index.
    out.sort_into_bins(cfg.mc_binning.total_bins());

    // The response matrix indexes the sorted events, so it is built after the
    // sort and never again: truth, widths and bin edges do not move during a fit.
    if (cfg.response.enabled) {
      std::size_t unusable = 0;
      out.response = build_response_matrix(cfg.mc_binning, out.bin_idx, out.response_truth_log_e,
                                           out.response_truth_zenith, out.response_sigma_log_e,
                                           out.response_sigma_zenith, cfg.response.truncation,
                                           cfg.response.min_fraction, unusable);

      const double per_event = out.size() > 0 ? static_cast<double>(out.response.nnz()) /
                                                    static_cast<double>(out.size())
                                              : 0.0;
      std::cout << "IceCube sample '" << cfg.name << "': response matrix " << out.response.nnz()
                << " entries over " << out.size() << " events (" << per_event << " bins per event), "
                << (static_cast<double>(out.response.bytes()) / 1e9) << " GB";
      if (unusable > 0)
        std::cout << "; " << unusable << " events had no usable response and kept their unfolded bin";
      std::cout << '\n';
    }

    if (cfg.filters_topology())
      std::cout << "IceCube sample '" << cfg.name << "': topology cut on '" << cfg.topology_branch << "' dropped "
                << topology_dropped << " of " << N << " rows ("
                << (100.0 * static_cast<double>(topology_dropped) / static_cast<double>(N)) << "%)\n";

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

    // Data must be binned on the same category axes as the MC, from the same
    // columns; a data file lacking one of them fails here rather than silently
    // producing a differently-shaped histogram.
    std::vector<std::vector<double>> data_categories;
    for (const Axis& axis : io::ic::category_axes(cfg.binning)) {
      ARROW_ASSIGN_OR_RAISE(auto column, get_numeric_column(*table, axis.branch));
      data_categories.push_back(std::move(column));
    }

    // The same topology cut the MC path applies, so both sides of the likelihood
    // see one selection.
    ARROW_ASSIGN_OR_RAISE(const auto topology_keeps, topology_mask(*table, cfg));
    if (!topology_keeps.empty() && topology_keeps.size() != n_rows)
      return arrow::Status::Invalid("ICDataBase: topology column '" + cfg.topology_branch + "' has " +
                                    std::to_string(topology_keeps.size()) + " rows, the data file has " +
                                    std::to_string(n_rows));

    // NNMFit's standard mask: apply it to real data (the MC baselines are
    // measured pre-cut and read_sample does not apply it there). Compact to the
    // passing rows rather than poisoning failing ones with a sentinel: NaN would
    // NOT be rejected by Axis::index's `v < lo || v >= hi` (both comparisons are
    // false for NaN), so it would fall through to an undefined-behaviour cast.
    ARROW_ASSIGN_OR_RAISE(auto passes, standard_mask(*table, b.reco_energy));
    std::vector<double> masked_energy;
    std::vector<double> masked_zenith;
    std::vector<double> masked_ra;
    std::vector<std::vector<double>> masked_categories(data_categories.size());
    masked_energy.reserve(n_rows);
    masked_zenith.reserve(n_rows);
    if (needs_ra)
      masked_ra.reserve(n_rows);
    std::size_t topology_dropped = 0;
    for (std::size_t i = 0; i < n_rows; ++i) {
      if (!topology_keeps.empty() && !topology_keeps[i]) {
        ++topology_dropped;
        continue;
      }
      if (passes[i]) {
        masked_energy.push_back(energy[i]);
        masked_zenith.push_back(zenith[i]);
        if (needs_ra)
          masked_ra.push_back(ra[i]);
        for (std::size_t c = 0; c < data_categories.size(); ++c)
          masked_categories[c].push_back(data_categories[c][i]);
      }
    }

    if (cfg.filters_topology())
      std::cout << "IceCube data '" << cfg.name << "': topology cut on '" << cfg.topology_branch << "' dropped "
                << topology_dropped << " of " << n_rows << " rows ("
                << (100.0 * static_cast<double>(topology_dropped) / static_cast<double>(n_rows)) << "%)\n";

    // The standard-mask report counts only the rows the mask itself rejected, so a
    // topology cut does not inflate it.
    const std::size_t after_topology = n_rows - topology_dropped;
    if (masked_energy.size() != after_topology)
      std::cout << "IceCube data '" << cfg.name << "': standard mask dropped "
                << (after_topology - masked_energy.size()) << " of " << after_topology << " rows ("
                << (100.0 * static_cast<double>(after_topology - masked_energy.size()) /
                    static_cast<double>(after_topology))
                << "%)\n";

    if (!masked_categories.empty()) {
      std::vector<std::vector<double>> columns{masked_energy, masked_zenith};
      for (auto& column : masked_categories) columns.push_back(std::move(column));
      out = bin_event_counts(cfg.binning, columns);
    } else {
      out = needs_ra ? bin_event_counts(cfg.binning, masked_energy, masked_zenith, masked_ra)
                     : bin_event_counts(cfg.binning, masked_energy, masked_zenith);
    }
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
