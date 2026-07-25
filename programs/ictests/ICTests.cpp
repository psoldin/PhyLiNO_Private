#include "IceCube/Binning.h"

#include <cassert>
#include <cmath>
#include <cstdio>

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

int main() {
  test_total_bins();
  test_bin_index_matches_legacy();
  test_parse_axis_spec();
  std::puts("ICTests: all passed");
  return 0;
}
