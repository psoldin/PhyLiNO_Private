#include "IceCube/Binning.h"
#include "IceCube/ICSample.h"
#include "IceCube/SampleConfig.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <sstream>

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

int main() {
  test_total_bins();
  test_bin_index_matches_legacy();
  test_parse_axis_spec();
  test_parse_samples();
  test_sort_into_bins_csr_invariant();
  std::puts("ICTests: all passed");
  return 0;
}
