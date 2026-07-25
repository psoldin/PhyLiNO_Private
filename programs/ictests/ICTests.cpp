#include "IceCube/Binning.h"
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

int main() {
  test_total_bins();
  test_bin_index_matches_legacy();
  test_parse_axis_spec();
  test_parse_samples();
  std::puts("ICTests: all passed");
  return 0;
}
