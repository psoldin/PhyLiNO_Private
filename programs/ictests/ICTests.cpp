#include "IceCube/Binning.h"
#include "IceCube/ICParameter.h"
#include "IceCube/ICSample.h"
#include "IceCube/SampleConfig.h"
#include "SampleLikelihood.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <vector>

using io::ic::Axis;
using io::ic::Binning;

// The current tracks grid, expressed the new way.
static Binning tracks_binning() {
  return Binning({Axis{Axis::Kind::Log10Energy, 2.5, 7.0, 45},
                  Axis{Axis::Kind::CosZenith, -1.0, 0.0872, 33}});
}

static void test_total_bins() {
  assert(tracks_binning().total_bins() == 45 * 33);
}

static void test_bin_index_matches_legacy() {
  const Binning b = tracks_binning();
  auto legacy = [](double e_gev, double zen_rad) -> int {
    const double log_e = std::log10(e_gev);
    if (log_e < 2.5 || log_e >= 7.0) return -1;
    const double cz = std::cos(zen_rad);
    if (cz < -1.0 || cz >= 0.0872) return -1;
    const int eb = static_cast<int>((log_e - 2.5) / ((7.0 - 2.5) / 45));
    const int zb = static_cast<int>((cz - (-1.0)) / ((0.0872 - (-1.0)) / 33));
    return eb * 33 + zb;
  };
  for (double e : {50.0, 316.0, 1000.0, 1e4, 1e5, 5e6, 2e7})
    for (double z : {0.0, 1.0, 1.57, 2.0, 2.6, 3.14}) {
      const double reco[2] = {e, z};
      assert(b.bin_index(reco) == legacy(e, z));
    }
}

static void test_parse_axis_spec() {
  const Axis a = io::ic::parse_axis("Log10Energy", "(2.5, 7.0, 45)");
  assert(a.n_bins == 45);
  assert(std::abs(a.lo - 2.5) < 1e-12);
  assert(std::abs(a.hi - 7.0) < 1e-12);
}

// Exercises the real io::ic::parse_samples() (declared in SampleConfig.h,
// implemented in SampleConfig.cpp) against an in-memory JSON config, tolerant
// "Binnings" + "Samples" parser added in Task 3. One binning shared by two
// samples, one of which is disabled, to check both binning resolution and
// per-sample field parsing (including the comma-split "components" list).
static void test_parse_samples() {
  static constexpr char kJson[] = R"JSON(
{
  "IceCube": {
    "Binnings": {
      "tracks_2d": {
        "axes": "Log10Energy, CosZenith",
        "Log10Energy": "(2.5, 7.0, 45)",
        "CosZenith": "(-1.0, 0.0872, 33)"
      }
    },
    "Samples": {
      "tracks_baseline": {
        "binning": "tracks_2d",
        "parquet": "dataset_tracks_baseline.parquet",
        "livetime": 3.0e8,
        "components": "astro, conv, prompt"
      },
      "tracks_alt": {
        "binning": "tracks_2d",
        "parquet": "dataset_tracks_alt.parquet",
        "enabled": false,
        "livetime": 1.0e8
      }
    }
  }
}
)JSON";

  boost::property_tree::ptree pt;
  std::istringstream          iss(kJson);
  boost::property_tree::read_json(iss, pt);

  const auto samples = io::ic::parse_samples(pt.get_child("IceCube"));
  assert(samples.size() == 2);

  assert(samples[0].name == "tracks_baseline");
  assert(samples[0].enabled == true);
  assert(std::abs(samples[0].livetime - 3.0e8) < 1.0);
  assert(samples[0].binning.total_bins() == 1485);
  assert(samples[0].components.size() == 3);
  assert(samples[0].components[0] == "astro");
  assert(samples[0].components[1] == "conv");
  assert(samples[0].components[2] == "prompt");
  assert(samples[0].has_component("conv"));
  assert(!samples[0].has_component("muon"));

  assert(samples[1].name == "tracks_alt");
  assert(samples[1].enabled == false);
  assert(std::abs(samples[1].livetime - 1.0e8) < 1.0);
  assert(samples[1].binning.total_bins() == 1485);
}

// Proves the CSR invariant that ICSample::sort_into_bins() is supposed to
// maintain: out-of-range events dropped, per-event columns reordered in
// lockstep (grouped by bin, stable within a bin), and bin_offsets forming a
// valid CSR index into the reordered columns.
//
// 6 hand-built events against total_bins = 4:
//   original index : 0  1  2  3  4  5
//   bin_idx         : 2 -1  0  2  1  0
// Event 1 is out of range (bin_idx = -1) and must be dropped. The other five
// group into bin 0 = {2, 5}, bin 1 = {4}, bin 2 = {0, 3} (stable_sort keeps
// each bin's original relative order).
//
// Every per-event column is seeded as `offset + original_index` (a distinct
// offset per column, and 1000*(k+1) per Barr slope), so after the permutation
// is applied we can check that e_true (offset 0) and every other column moved
// together: col[i] - offset == e_true[i] for every surviving event i.
static void test_sort_into_bins_csr_invariant() {
  using io::ic::ICSample;

  ICSample s;
  const std::vector<int> bins = {2, -1, 0, 2, 1, 0};
  const std::size_t      N    = bins.size();

  s.e_true.resize(N);
  s.astro_baseline.resize(N);
  s.conv_baseline.resize(N);
  s.conv_alt.resize(N);
  s.prompt_baseline.resize(N);
  s.prompt_alt.resize(N);
  for (auto& g : s.barr_conv) g.resize(N);
  s.bin_idx = bins;

  for (std::size_t i = 0; i < N; ++i) {
    const double idx        = static_cast<double>(i);
    s.e_true[i]              = idx;            // offset 0
    s.astro_baseline[i]      = 100.0 + idx;     // offset 100
    s.conv_baseline[i]       = 200.0 + idx;     // offset 200
    s.conv_alt[i]            = 300.0 + idx;     // offset 300
    s.prompt_baseline[i]     = 400.0 + idx;     // offset 400
    s.prompt_alt[i]          = 500.0 + idx;     // offset 500
    for (int k = 0; k < params::ic::nBarrParams; ++k)
      s.barr_conv[k][i] = 1000.0 * (k + 1) + idx;  // offset 1000*(k+1)
  }

  s.sort_into_bins(/*total_bins=*/4);

  // The out-of-range event (original index 1) must be dropped: 6 -> 5.
  assert(s.size() == 5);

  // CSR shape: total_bins + 1 offsets, monotonic non-decreasing, last == size().
  assert(s.bin_offsets.size() == 5);
  for (std::size_t b = 0; b + 1 < s.bin_offsets.size(); ++b)
    assert(s.bin_offsets[b] <= s.bin_offsets[b + 1]);
  assert(s.bin_offsets.back() == s.size());

  // Expected per-bin counts: bin0={2,5}(2), bin1={4}(1), bin2={0,3}(2), bin3={}(0).
  assert(s.bin_offsets[0] == 0);
  assert(s.bin_offsets[1] == 2);
  assert(s.bin_offsets[2] == 3);
  assert(s.bin_offsets[3] == 5);
  assert(s.bin_offsets[4] == 5);

  // Events are grouped by bin: bin_idx is non-decreasing across the sorted array.
  for (std::size_t i = 0; i + 1 < s.size(); ++i)
    assert(s.bin_idx[i] <= s.bin_idx[i + 1]);

  // Original index 1 (dropped) must not survive in any column.
  for (std::size_t i = 0; i < s.size(); ++i)
    assert(s.e_true[i] != 1.0);

  // Every column was permuted in lockstep with e_true (same permutation
  // applied to every per-event column, including all four Barr slopes).
  for (std::size_t i = 0; i < s.size(); ++i) {
    assert(std::abs(s.astro_baseline[i]  - s.e_true[i] - 100.0) < 1e-9);
    assert(std::abs(s.conv_baseline[i]   - s.e_true[i] - 200.0) < 1e-9);
    assert(std::abs(s.conv_alt[i]        - s.e_true[i] - 300.0) < 1e-9);
    assert(std::abs(s.prompt_baseline[i] - s.e_true[i] - 400.0) < 1e-9);
    assert(std::abs(s.prompt_alt[i]      - s.e_true[i] - 500.0) < 1e-9);
    for (int k = 0; k < params::ic::nBarrParams; ++k)
      assert(std::abs(s.barr_conv[k][i] - s.e_true[i] - 1000.0 * (k + 1)) < 1e-9);
  }

  // Explicit grouping check via the recorded permutation: sorted original
  // indices should be [2, 5, 4, 0, 3] (stable within each bin).
  const double expected_original_index[5] = {2.0, 5.0, 4.0, 0.0, 3.0};
  for (std::size_t i = 0; i < 5; ++i)
    assert(std::abs(s.e_true[i] - expected_original_index[i]) < 1e-9);
}

// Builds a tiny synthetic tracks-like sample (3 log10-energy bins x 2
// cos-zenith bins = 6 analysis bins, 2 events per bin, every per-event column
// populated with nonzero physically plausible values) and checks that the
// Asimov point (predicted == data by construction) minimizes SampleLikelihood's
// partial -2lnL: perturbing AstroNorm away from its nominal value must not
// improve (must strictly worsen) the likelihood. Exercises the CPU (gpu =
// nullptr) path of both PowerlawFlux and AtmosphericFlux end to end, including
// the SAY ssq path (assemble_fluctuation).
static void test_sample_likelihood_asimov_is_minimum() {
  using ana::ic::GlobalFluxSettings;
  using ana::ic::SampleLikelihood;
  using ana::ParameterWrapper;

  // 2D binning: Log10Energy in [2, 5) with 3 bins, CosZenith in [-1, 1) with
  // 2 bins -> 6 analysis bins total.
  Binning binning({Axis{Axis::Kind::Log10Energy, 2.0, 5.0, 3},
                   Axis{Axis::Kind::CosZenith, -1.0, 1.0, 2}});
  assert(binning.total_bins() == 6);

  // log10-energy bin centers: 10^2.5, 10^3.5, 10^4.5 GeV.
  const double energies[3] = {316.227766, 3162.27766, 31622.7766};
  // zenith (radians) chosen so cos(zenith) lands in cos-zenith bin 0 ([-1,0))
  // or bin 1 ([0,1)) respectively.
  const double zeniths[2] = {2.0, 0.5};

  io::ic::ICSample sample;
  const int        events_per_bin = 2;
  const int        n_events       = 3 * 2 * events_per_bin;
  sample.e_true.reserve(n_events);
  sample.astro_baseline.reserve(n_events);
  sample.conv_baseline.reserve(n_events);
  sample.conv_alt.reserve(n_events);
  sample.prompt_baseline.reserve(n_events);
  sample.prompt_alt.reserve(n_events);
  for (auto& g : sample.barr_conv) g.reserve(n_events);
  sample.bin_idx.reserve(n_events);

  int event_index = 0;
  for (int eb = 0; eb < 3; ++eb) {
    for (int zb = 0; zb < 2; ++zb) {
      for (int k = 0; k < events_per_bin; ++k, ++event_index) {
        // Small per-event jitter so events within a bin aren't identical.
        const double jitter    = 1.0 + 0.01 * static_cast<double>(k);
        const double e_true    = energies[eb] * jitter;
        const double reco_e    = energies[eb] * jitter;
        const double reco_z    = zeniths[zb];
        const double conv_base = 5.0e-3 * jitter;
        const double prompt_base = 1.0e-4 * jitter;

        sample.e_true.push_back(e_true);
        sample.astro_baseline.push_back(1.0e-3 * jitter);
        sample.conv_baseline.push_back(conv_base);
        sample.conv_alt.push_back(0.9 * conv_base);
        sample.prompt_baseline.push_back(prompt_base);
        sample.prompt_alt.push_back(0.9 * prompt_base);
        for (int b = 0; b < params::ic::nBarrParams; ++b)
          sample.barr_conv[b].push_back(1.0e-4 * (b + 1) * jitter);

        const double reco[2] = {reco_e, reco_z};
        sample.bin_idx.push_back(binning.bin_index(reco));
      }
    }
  }
  assert(static_cast<int>(sample.size()) == n_events);

  sample.sort_into_bins(binning.total_bins());
  // All 12 synthetic events were built in-range; none should be dropped.
  assert(static_cast<int>(sample.size()) == n_events);
  assert(sample.bin_offsets.back() == sample.size());

  io::ic::SampleConfig cfg{.name = "unit_test_sample", .binning = binning};

  const GlobalFluxSettings settings{.e_ref_gev                = 1.0e5,
                                    .astro_reference_index    = 2.0,
                                    .conv_delta_gamma_e_ref   = 1.0e3,
                                    .prompt_delta_gamma_e_ref = 3.8e3,
                                    .astro_per_type_norm      = false};

  SampleLikelihood likelihood(sample, cfg, settings, /*gpu=*/nullptr, /*use_say=*/true);

  // Nominal parameter values mirror configs/config_icecube_tracks_cpu.json's
  // "StartValue" entries (physically reasonable defaults for this layout).
  std::vector<double> nominal_values(params::ic::number_of_parameters(), 0.0);
  nominal_values[params::ic::AstroNorm]    = 1.5;
  nominal_values[params::ic::SpectralIndex] = 2.4;
  nominal_values[params::ic::ConvNorm]     = 1.0;
  nominal_values[params::ic::PromptNorm]   = 1.0;
  // BarrH/W/Y/Z, CRGrad, DeltaGamma stay 0.0 (nominal).
  nominal_values[params::ic::MuonNorm] = 1.0;
  nominal_values[params::ic::DOMEff]   = 1.0;
  nominal_values[params::ic::IceAbs]   = 1.0;
  nominal_values[params::ic::IceScat]  = 1.0;

  ParameterWrapper nominal(params::ic::number_of_parameters());
  nominal.reset_parameter(nominal_values.data());

  likelihood.generate_asimov(nominal);

  const double llh_nominal = likelihood.partial_llh(nominal);
  assert(std::isfinite(llh_nominal));

  // Perturb AstroNorm x1.5 away from its Asimov (nominal) value; everything
  // else stays at the nominal/config default.
  std::vector<double> perturbed_values = nominal_values;
  perturbed_values[params::ic::AstroNorm] *= 1.5;

  ParameterWrapper perturbed(params::ic::number_of_parameters());
  perturbed.reset_parameter(perturbed_values.data());

  const double llh_perturbed = likelihood.partial_llh(perturbed);
  assert(std::isfinite(llh_perturbed));

  // The Asimov point (data == prediction at nominal) must minimize -2lnL:
  // do NOT assert llh_nominal ~= 0 -- SAY/Poisson here keep the saturated
  // constant, so the minimum is a nonzero value.
  assert(llh_nominal < llh_perturbed);
}

int main() {
  test_total_bins();
  test_bin_index_matches_legacy();
  test_parse_axis_spec();
  test_parse_samples();
  test_sort_into_bins_csr_invariant();
  test_sample_likelihood_asimov_is_minimum();
  std::puts("ICTests: all passed");
  return 0;
}
