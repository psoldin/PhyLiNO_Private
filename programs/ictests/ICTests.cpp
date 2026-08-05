#include "DetectorSystematics.h"
#include "IceCube/Binning.h"
#include "IceCube/ICParameter.h"
#include "IceCube/ICSample.h"
#include "IceCube/SampleConfig.h"
#include "InputParameter.h"
#include "MetalBackend.h"
#include "ParameterSeeding.h"
#include "PoissonLikelihood.h"
#include "SAYLikelihood.h"
#include "SampleLikelihood.h"
#include "TemplateFlux.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using io::ic::Axis;
using io::ic::Binning;

// The current tracks grid, expressed the new way.
static Binning tracks_binning() {
  return Binning({Axis{Axis::Kind::Log10Energy, 2.5, 7.0, 45},
                  Axis{Axis::Kind::CosZenith, -1.0, 0.0872, 33}});
}

TEST(BinningTest, TotalBins) {
  ASSERT_TRUE(tracks_binning().total_bins() == 45 * 33);
}

TEST(BinningTest, BinIndexMatchesLegacy) {
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
      ASSERT_TRUE(b.bin_index(reco) == legacy(e, z));
    }
}

TEST(BinningTest, ParseAxisSpec) {
  const Axis a = io::ic::parse_axis("Log10Energy", "(2.5, 7.0, 45)");
  ASSERT_TRUE(a.n_bins == 45);
  ASSERT_TRUE(std::abs(a.lo - 2.5) < 1e-12);
  ASSERT_TRUE(std::abs(a.hi - 7.0) < 1e-12);
  ASSERT_TRUE(a.uniform());
}

// The cascade zenith axis is a hardcoded non-uniform cos-zenith edge list in
// NNMFit (binning/rectangular_binning.py, spacing "cscd-cos_5up"). Expressed here
// as explicit edges, Axis::index must bin by upper_bound over those edges.
TEST(BinningTest, NonUniformAxisIndex) {
  const std::vector<double> edges{-1.0, -0.76, -0.52, -0.28, -0.04, 0.2, 0.6, 1.0};
  const Axis a = io::ic::parse_axis("CosZenith", "[-1.0, -0.76, -0.52, -0.28, -0.04, 0.2, 0.6, 1.0]");
  ASSERT_TRUE(!a.uniform());
  ASSERT_TRUE(a.n_bins == 7);
  ASSERT_TRUE(a.edges.size() == 8);
  ASSERT_TRUE(std::abs(a.lo + 1.0) < 1e-12);
  ASSERT_TRUE(std::abs(a.hi - 1.0) < 1e-12);

  // Reference: the bin containing cos(zenith), -1 outside [lo, hi).
  auto reference = [&edges](const double zenith_rad) -> int {
    const double cos_zenith = std::cos(zenith_rad);
    if (cos_zenith < edges.front() || cos_zenith >= edges.back()) return -1;
    for (std::size_t i = 0; i + 1 < edges.size(); ++i)
      if (cos_zenith >= edges[i] && cos_zenith < edges[i + 1]) return static_cast<int>(i);
    return -1;
  };

  for (double zenith : {0.0, 0.3, 0.8, 1.0, 1.2, 1.5708, 1.9, 2.4, 2.9, 3.14159, 3.2})
    ASSERT_TRUE(a.index(zenith) == reference(zenith));

  // Lower edge inclusive, upper edge exclusive, in cos(zenith).
  ASSERT_TRUE(a.index(std::acos(-1.0)) == 0);      // cos = -1 -> first bin
  ASSERT_TRUE(a.index(std::acos(-0.76)) == 1);     // exactly an interior edge
  ASSERT_TRUE(a.index(std::acos(0.999999)) == 6);  // last bin
  ASSERT_TRUE(a.index(std::acos(1.0)) == -1);      // cos = +1 == hi -> out of range

  // A non-ascending edge list is a config error, not a silently wrong binning.
  bool threw = false;
  try {
    const Axis descending = io::ic::parse_axis("CosZenith", "[-1.0, 0.5, 0.2]");
    ASSERT_TRUE(descending.n_bins == 2);  // unreachable; keeps the parse result used
  } catch (const std::runtime_error&) {
    threw = true;
  }
  ASSERT_TRUE(threw);
}

// A binning may mix a uniform energy axis with an explicit-edge zenith axis: that
// is exactly the cscd_cascade grid (21 x 7 = 147 bins). cscd_muon is one bin.
TEST(BinningTest, MixedCascadeGrid) {
  const Binning cascade({io::ic::parse_axis("Log10Energy", "(2.8, 7.0, 21)"),
                         io::ic::parse_axis("CosZenith",
                                            "[-1.0, -0.76, -0.52, -0.28, -0.04, 0.2, 0.6, 1.0]")});
  ASSERT_TRUE(cascade.total_bins() == 147);

  // 10^3 GeV is energy bin 1 ((3.0 - 2.8) / 0.2); cos(zenith) = 0 is zenith bin 4.
  const double reco[2] = {1000.0, 1.5707963267948966};
  ASSERT_TRUE(cascade.bin_index(reco) == 1 * 7 + 4);

  const Binning muon({io::ic::parse_axis("Log10Energy", "(2.6, 4.8, 1)"),
                      io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 1)")});
  ASSERT_TRUE(muon.total_bins() == 1);
  const double inside[2] = {1000.0, 1.5};
  ASSERT_TRUE(muon.bin_index(inside) == 0);
  const double too_soft[2] = {100.0, 1.5};
  ASSERT_TRUE(muon.bin_index(too_soft) == -1);
}

// The parameter layout is the contract between the config's "Parameter" array, the
// Minuit index array and every component that reads a fixed index, so pin down the
// count and the two contiguous blocks components index into.
TEST(ICParameterTest, Layout) {
  using namespace params::ic;
  ASSERT_TRUE(number_of_parameters() == 23);
  ASSERT_TRUE(nBarrParams == 4);
  ASSERT_TRUE(nDetSysParams == 5);
  // Barr block, contiguous in {H, W, Y, Z} order (AtmosphericFlux reads BarrH + k).
  ASSERT_TRUE(BarrW == BarrH + 1 && BarrY == BarrH + 2 && BarrZ == BarrH + 3);
  // Detector block, contiguous in the order the exported gradient file uses.
  ASSERT_TRUE(IceAbs == DOMEff + 1 && IceScat == DOMEff + 2);
  ASSERT_TRUE(HoleIceP0 == DOMEff + 3 && HoleIceP1 == DOMEff + 4);
  // The two template norms are distinct: tracks Corsika vs cascade MuonGun.
  ASSERT_TRUE(MuonNorm != MuonGunNorm);
  // Galactic norms are the last block, one per galactic template a sample can declare.
  ASSERT_TRUE(GalacticNorm1 == GalacticNorm0 + 1);
  // Broken-power-law block (NNMFit AstroBPL), contiguous after the galactic norms.
  ASSERT_TRUE(AstroGamma2 == AstroGamma1 + 1);
  ASSERT_TRUE(AstroEBreak == AstroGamma2 + 1);
}

// The Gaussian pull width must be separable from the minimiser step, while a
// config that specifies neither prior key keeps today's meaning -- the
// compatibility guarantee the Double Chooz configs rely on.
TEST(InputParameterTest, PriorDefaultsAndOverrides) {
  static constexpr char kJson[] = R"JSON(
{
  "Parameter": [
    { "Name": "legacy",   "StartValue": 1.0, "StepWidth": 0.4, "Fixed": false, "Constrained": true },
    { "Name": "explicit", "StartValue": 1.0, "StepWidth": 0.1, "PriorValue": 1.2,
      "PriorWidth": 0.5, "Fixed": false, "Constrained": true }
  ]
}
)JSON";

  boost::property_tree::ptree pt;
  std::istringstream          iss(kJson);
  boost::property_tree::read_json(iss, pt);

  const io::InputParameter parameters(pt.get_child("Parameter"));
  ASSERT_TRUE(parameters.size() == 2);

  // Legacy entry: prior falls back to StartValue / StepWidth exactly as before.
  ASSERT_TRUE(parameters.parameters()[0].value() == 1.0);
  ASSERT_TRUE(parameters.parameters()[0].uncertainty() == 0.4);
  ASSERT_TRUE(parameters.parameters()[0].prior_value() == 1.0);
  ASSERT_TRUE(parameters.parameters()[0].prior_width() == 0.4);

  // Explicit entry: step and prior are independent.
  ASSERT_TRUE(parameters.parameters()[1].uncertainty() == 0.1);
  ASSERT_TRUE(parameters.parameters()[1].prior_value() == 1.2);
  ASSERT_TRUE(parameters.parameters()[1].prior_width() == 0.5);
}

// Exercises the real io::ic::parse_samples() (declared in SampleConfig.h,
// implemented in SampleConfig.cpp) against an in-memory JSON config, tolerant
// "Binnings" + "Samples" parser added in Task 3. One binning shared by two
// samples, one of which is disabled, to check both binning resolution and
// per-sample field parsing (including the comma-split "components" list).
TEST(SampleConfigTest, ParseSamples) {
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
        "components": "astro, conventional, prompt"
      },
      "tracks_alt": {
        "binning": "tracks_2d",
        "parquet": "dataset_tracks_alt.parquet",
        "enabled": false,
        "livetime": 1.0e8,
        "components": "astro"
      }
    }
  }
}
)JSON";

  boost::property_tree::ptree pt;
  std::istringstream          iss(kJson);
  boost::property_tree::read_json(iss, pt);

  const auto samples = io::ic::parse_samples(pt.get_child("IceCube"));
  ASSERT_TRUE(samples.size() == 2);

  ASSERT_TRUE(samples[0].name == "tracks_baseline");
  ASSERT_TRUE(samples[0].enabled == true);
  ASSERT_TRUE(std::abs(samples[0].livetime - 3.0e8) < 1.0);
  ASSERT_TRUE(samples[0].binning.total_bins() == 1485);
  ASSERT_TRUE(samples[0].components.size() == 3);
  ASSERT_TRUE(samples[0].components[0] == "astro");
  ASSERT_TRUE(samples[0].components[1] == "conventional");
  ASSERT_TRUE(samples[0].components[2] == "prompt");
  ASSERT_TRUE(samples[0].has_component("conventional"));
  ASSERT_TRUE(!samples[0].has_component("muon"));
  ASSERT_TRUE(samples[0].wants_astro());
  ASSERT_TRUE(samples[0].wants_atmospheric());

  ASSERT_TRUE(samples[1].name == "tracks_alt");
  ASSERT_TRUE(samples[1].enabled == false);
  ASSERT_TRUE(std::abs(samples[1].livetime - 1.0e8) < 1.0);
  ASSERT_TRUE(samples[1].binning.total_bins() == 1485);
  // Component masking: an astro-only sample must not pull in the atmospheric
  // flux (nor its parquet columns, see ICDataBase::read_sample).
  ASSERT_TRUE(samples[1].wants_astro());
  ASSERT_TRUE(!samples[1].wants_atmospheric());
}

// parse_samples() must reject component lists the flux components cannot
// honour, instead of silently predicting fewer events than the config asks for.
TEST(SampleConfigTest, RejectsBadComponents) {
  static constexpr char kTemplate[] = R"JSON(
{
  "IceCube": {
    "Binnings": {
      "grid": {
        "axes": "Log10Energy, CosZenith",
        "Log10Energy": "(2.5, 7.0, 45)",
        "CosZenith": "(-1.0, 0.0872, 33)"
      }
    },
    "Samples": {
      "s": { "binning": "grid", "parquet": "s.parquet"COMPONENTS }
    }
  }
}
)JSON";

  auto parse_with = [](const std::string& components_entry) {
    std::string json(kTemplate);
    json.replace(json.find("COMPONENTS"), std::strlen("COMPONENTS"), components_entry);
    boost::property_tree::ptree pt;
    std::istringstream          iss(json);
    boost::property_tree::read_json(iss, pt);
    const auto parsed = io::ic::parse_samples(pt.get_child("IceCube"));
    ASSERT_TRUE(parsed.size() == 1);
  };

  auto throws = [&parse_with](const std::string& components_entry) {
    try {
      parse_with(components_entry);
    } catch (const std::runtime_error&) {
      return true;
    }
    return false;
  };

  ASSERT_TRUE(throws(""));                                        // no components at all
  ASSERT_TRUE(throws(R"(, "components": "astro, cascades")"));     // unknown component name
  ASSERT_TRUE(throws(R"(, "components": "astro, conventional")")); // conventional without prompt
  ASSERT_TRUE(throws(R"(, "components": "prompt")"));              // prompt without conventional
  ASSERT_TRUE(!throws(R"(, "components": "astro")"));              // astro alone is fine
  ASSERT_TRUE(!throws(R"(, "components": "conventional, prompt")"));

  // Veto pair, variant mixing and duplicate templates.
  ASSERT_TRUE(throws(R"(, "components": "astro, conventional_veto")"));  // veto pair incomplete
  ASSERT_TRUE(throws(R"(, "components": "astro, prompt_veto")"));
  ASSERT_TRUE(throws(R"(, "components": "conventional, prompt, conventional_veto, prompt_veto")"));
  ASSERT_TRUE(throws(R"(, "components": "astro, muon, muontemplate", "Template": { "File": "t.txt" })"));
  ASSERT_TRUE(!throws(R"(, "components": "astro, conventional_veto, prompt_veto")"));

  // A template component needs its file, and a file needs its component.
  ASSERT_TRUE(throws(R"(, "components": "astro, muon")"));  // template declared, no Template block
  ASSERT_TRUE(throws(R"(, "components": "astro", "Template": { "File": "t.txt" })"));
  ASSERT_TRUE(throws(R"(, "components": "astro, muon", "Template": { "File": "t.txt", "Norm": "Nope" })"));
  ASSERT_TRUE(!throws(R"(, "components": "astro, muon", "Template": { "File": "t.txt", "Norm": "MuonGunNorm" })"));
}

// The cascade samples declare the veto variants plus the MuonGun template, and
// carry their template / gradient file paths and reco branch overrides per sample.
TEST(SampleConfigTest, ParsesCascadeEntry) {
  static constexpr char kJson[] = R"JSON(
{
  "IceCube": {
    "Binnings": {
      "cscd_cascade_2d": {
        "axes": "Log10Energy, CosZenith",
        "Log10Energy": "(2.8, 7.0, 21)",
        "CosZenith": "[-1.0, -0.76, -0.52, -0.28, -0.04, 0.2, 0.6, 1.0]"
      }
    },
    "Samples": {
      "cscd_cascade": {
        "binning": "cscd_cascade_2d",
        "parquet": "cscd_cascade.parquet",
        "data": "data_cscd_cascade.parquet",
        "livetime": 330315015.11,
        "components": "astro, conventional_veto, prompt_veto, muon",
        "Template": { "File": "muongun_cascade.txt", "Norm": "MuonGunNorm" },
        "Gradients": { "File": "gradients_cscd_cascade.txt" },
        "Branches": { "RecoEnergy": "energy_monopod", "RecoZenith": "zenith_monopod" }
      }
    }
  }
}
)JSON";

  boost::property_tree::ptree pt;
  std::istringstream          iss(kJson);
  boost::property_tree::read_json(iss, pt);

  const auto samples = io::ic::parse_samples(pt.get_child("IceCube"));
  ASSERT_TRUE(samples.size() == 1);
  const io::ic::SampleConfig& cascade = samples[0];

  ASSERT_TRUE(cascade.binning.total_bins() == 147);
  ASSERT_TRUE(std::abs(cascade.livetime - 330315015.11) < 1e-6);
  ASSERT_TRUE(cascade.wants_astro());
  ASSERT_TRUE(cascade.wants_atmospheric());
  ASSERT_TRUE(cascade.wants_veto());
  ASSERT_TRUE(cascade.wants_template());
  ASSERT_TRUE(cascade.template_file == "muongun_cascade.txt");
  ASSERT_TRUE(cascade.template_norm_index == params::ic::MuonGunNorm);
  ASSERT_TRUE(cascade.gradient_file == "gradients_cscd_cascade.txt");
  ASSERT_TRUE(cascade.data_path == "data_cscd_cascade.parquet");
  ASSERT_TRUE(cascade.branches.reco_energy == "energy_monopod");
  ASSERT_TRUE(cascade.branches.reco_zenith == "zenith_monopod");
  // Defaults still apply to the columns the sample did not override.
  ASSERT_TRUE(cascade.branches.true_energy == "MCPrimaryEnergy");

  // The tracks-style sample: plain atmospheric pair, Corsika template norm, and
  // no gradient file, so detector systematics stay off for it.
  static constexpr char kTracksJson[] = R"JSON(
{
  "IceCube": {
    "Binnings": { "grid": { "axes": "Log10Energy, CosZenith",
                            "Log10Energy": "(2.5, 7.0, 45)",
                            "CosZenith": "(-1.0, 0.0872, 33)" } },
    "Samples": {
      "tracks": {
        "binning": "grid",
        "parquet": "tracks.parquet",
        "components": "astro, conventional, prompt, muontemplate",
        "Template": { "File": "corsika_tracks.txt" }
      }
    }
  }
}
)JSON";

  boost::property_tree::ptree tracks_pt;
  std::istringstream          tracks_iss(kTracksJson);
  boost::property_tree::read_json(tracks_iss, tracks_pt);

  const auto tracks = io::ic::parse_samples(tracks_pt.get_child("IceCube"));
  ASSERT_TRUE(tracks.size() == 1);
  ASSERT_TRUE(tracks[0].wants_atmospheric());
  ASSERT_TRUE(!tracks[0].wants_veto());
  ASSERT_TRUE(tracks[0].wants_template());
  ASSERT_TRUE(tracks[0].template_norm_index == params::ic::MuonNorm);  // Norm defaults to MuonNorm
  ASSERT_TRUE(tracks[0].gradient_file.empty());
}

// The oscillation sidecar is a per-event multiplicative factor on the atmospheric
// baselines only (NNMFit's OscillationsHook: nu_mu disappearance, applied to the
// conventional and prompt baseline weights at load time).
TEST(SampleConfigTest, ParsesOscillationEntry) {
  static constexpr char kJson[] = R"JSON(
{
  "IceCube": {
    "Binnings": {
      "grid": {
        "axes": "Log10Energy, CosZenith",
        "Log10Energy": "(2.5, 7.0, 45)",
        "CosZenith": "(-1.0, 0.0872, 33)"
      }
    },
    "Samples": {
      "tracks": {
        "binning": "grid",
        "parquet": "tracks.parquet",
        "components": "astro, conventional, prompt",
        "Oscillations": { "File": "osc_tracks.parquet", "Branch": "osc_survival" }
      },
      "no_osc": {
        "binning": "grid",
        "parquet": "other.parquet",
        "components": "astro, conventional, prompt"
      }
    }
  }
}
)JSON";

  boost::property_tree::ptree pt;
  std::istringstream          iss(kJson);
  boost::property_tree::read_json(iss, pt);

  const auto samples = io::ic::parse_samples(pt.get_child("IceCube"));
  ASSERT_TRUE(samples.size() == 2);
  ASSERT_TRUE(samples[0].oscillation_file == "osc_tracks.parquet");
  ASSERT_TRUE(samples[0].oscillation_branch == "osc_survival");
  ASSERT_TRUE(samples[1].oscillation_file.empty());
  // Default branch name applies even without an explicit "Branch" key.
  ASSERT_TRUE(samples[1].oscillation_branch == "osc_survival");
}

// A sample whose analysis binning carries an RA axis keeps a second, RA-free
// binning for the MC: per-event weights, the muon template and the SnowStorm
// gradients all stay 2D and are broadcast over RA at prediction time.
TEST(SampleConfigTest, ParsesRaBinning) {
  const std::string json = R"JSON({
    "Binnings": {
      "b3d": {
        "axes": "Log10Energy, CosZenith, Ra",
        "Log10Energy": "(2.0, 5.0, 3)",
        "CosZenith": "(-1.0, 1.0, 2)",
        "Ra": "(0.0, 6.28319, 4)"
      }
    },
    "Samples": {
      "s": {
        "binning": "b3d",
        "parquet": "mc.parquet",
        "components": "astro",
        "Galactic": {
          "fermi": { "File": "fermi.txt", "Norm": "GalacticNorm0" },
          "unresolved": { "File": "unresolved.txt", "Norm": "GalacticNorm1" }
        }
      }
    }
  })JSON";

  std::istringstream            in(json);
  boost::property_tree::ptree   tree;
  boost::property_tree::read_json(in, tree);
  const auto samples = io::ic::parse_samples(tree);

  ASSERT_TRUE(samples.size() == 1);
  const io::ic::SampleConfig& s = samples[0];
  ASSERT_TRUE(s.binning.total_bins() == 24);
  ASSERT_TRUE(s.mc_binning.total_bins() == 6);
  ASSERT_TRUE(s.mc_binning.n_axes() == 2);
  ASSERT_TRUE(s.ra_bins() == 4);

  ASSERT_TRUE(s.galactic.size() == 2);
  ASSERT_TRUE(s.galactic[0].name == "fermi");
  ASSERT_TRUE(s.galactic[0].file == "fermi.txt");
  ASSERT_TRUE(s.galactic[0].norm_index == params::ic::GalacticNorm0);
  ASSERT_TRUE(s.galactic[1].norm_index == params::ic::GalacticNorm1);
}

// A 2-axis sample gets mc_binning == binning and ra_bins() == 1, so nothing about
// the existing configs changes.
TEST(SampleConfigTest, ParsesWithoutRa) {
  const std::string json = R"JSON({
    "Binnings": {
      "b2d": {
        "axes": "Log10Energy, CosZenith",
        "Log10Energy": "(2.0, 5.0, 3)",
        "CosZenith": "(-1.0, 1.0, 2)"
      }
    },
    "Samples": {
      "s": { "binning": "b2d", "parquet": "mc.parquet", "components": "astro" }
    }
  })JSON";

  std::istringstream            in(json);
  boost::property_tree::ptree   tree;
  boost::property_tree::read_json(in, tree);
  const auto samples = io::ic::parse_samples(tree);

  ASSERT_TRUE(samples[0].mc_binning.total_bins() == samples[0].binning.total_bins());
  ASSERT_TRUE(samples[0].ra_bins() == 1);
  ASSERT_TRUE(samples[0].galactic.empty());
}

// Configurations the prediction path cannot express must fail at startup.
TEST(SampleConfigTest, RejectsBadGalactic) {
  auto parse = [](const std::string& binnings, const std::string& sample_body) {
    const std::string json = "{ \"Binnings\": " + binnings + ", \"Samples\": { \"s\": { " +
                             sample_body + " } } }";
    std::istringstream          in(json);
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(in, tree);
    static_cast<void>(io::ic::parse_samples(tree));
  };

  const std::string b3d = R"JSON({ "b3d": { "axes": "Log10Energy, CosZenith, Ra",
      "Log10Energy": "(2.0, 5.0, 3)", "CosZenith": "(-1.0, 1.0, 2)", "Ra": "(0.0, 6.28319, 4)" } })JSON";
  const std::string b2d = R"JSON({ "b2d": { "axes": "Log10Energy, CosZenith",
      "Log10Energy": "(2.0, 5.0, 3)", "CosZenith": "(-1.0, 1.0, 2)" } })JSON";
  const std::string ra_first = R"JSON({ "bad": { "axes": "Ra, Log10Energy",
      "Ra": "(0.0, 6.28319, 4)", "Log10Energy": "(2.0, 5.0, 3)" } })JSON";

  auto throws = [&](const std::string& binnings, const std::string& body) {
    try {
      parse(binnings, body);
    } catch (const std::runtime_error&) {
      return true;
    }
    return false;
  };

  // The RA axis must be last.
  ASSERT_TRUE(throws(ra_first, R"("binning": "bad", "parquet": "m.parquet", "components": "astro")"));

  // A galactic template needs an RA axis to be binned against.
  ASSERT_TRUE(throws(b2d, R"("binning": "b2d", "parquet": "m.parquet", "components": "astro",
      "Galactic": { "fermi": { "File": "f.txt", "Norm": "GalacticNorm0" } })"));

  // Unknown norm name.
  ASSERT_TRUE(throws(b3d, R"("binning": "b3d", "parquet": "m.parquet", "components": "astro",
      "Galactic": { "fermi": { "File": "f.txt", "Norm": "MuonNorm" } })"));

  // Two templates sharing one norm would be indistinguishable in the fit.
  ASSERT_TRUE(throws(b3d, R"("binning": "b3d", "parquet": "m.parquet", "components": "astro",
      "Galactic": { "a": { "File": "a.txt", "Norm": "GalacticNorm0" },
                    "b": { "File": "b.txt", "Norm": "GalacticNorm0" } })"));

  // A valid 3D sample with one galactic template parses.
  ASSERT_TRUE(!throws(b3d, R"("binning": "b3d", "parquet": "m.parquet", "components": "astro",
      "Galactic": { "fermi": { "File": "f.txt", "Norm": "GalacticNorm0" } })"));
}

// The composite pairs its SampleLikelihoods with the ICSamples that ICDataBase
// loaded, relying on both walking the enabled configs in the same order --
// the riskiest line in the multi-sample refactor. Both sides call
// enabled_sample_indices(), so pin its behaviour down here.
TEST(SampleConfigTest, EnabledSampleIndices) {
  using io::ic::SampleConfig;

  const Binning grid = tracks_binning();

  auto make = [&grid](const std::string& name, const bool enabled) {
    return SampleConfig{.name = name, .enabled = enabled, .binning = grid, .mc_binning = grid};
  };

  // Middle sample disabled: the loaded sample at load-index 1 must pair with
  // config 2 ("cscd_muon"), not config 1.
  const std::vector<SampleConfig> middle_disabled{
      make("tracks", true), make("cscd_cascade", false), make("cscd_muon", true)};
  const auto enabled = io::ic::enabled_sample_indices(middle_disabled);
  ASSERT_TRUE(enabled.size() == 2);
  ASSERT_TRUE(enabled[0] == 0);
  ASSERT_TRUE(enabled[1] == 2);
  ASSERT_TRUE(middle_disabled[enabled[1]].name == "cscd_muon");

  // First disabled, and order is config order, not "enabled first".
  const std::vector<SampleConfig> first_disabled{
      make("tracks", false), make("cscd_cascade", true), make("cscd_muon", true)};
  const auto enabled_tail = io::ic::enabled_sample_indices(first_disabled);
  ASSERT_TRUE(enabled_tail.size() == 2);
  ASSERT_TRUE(enabled_tail[0] == 1);
  ASSERT_TRUE(enabled_tail[1] == 2);

  // All enabled: identity. All disabled: empty (ICDataBase/ICLikelihood throw).
  const std::vector<SampleConfig> all_enabled{make("a", true), make("b", true)};
  const auto                      identity = io::ic::enabled_sample_indices(all_enabled);
  ASSERT_TRUE(identity.size() == 2);
  ASSERT_TRUE(identity[0] == 0 && identity[1] == 1);
  ASSERT_TRUE(io::ic::enabled_sample_indices({make("a", false)}).empty());
  ASSERT_TRUE(io::ic::enabled_sample_indices({}).empty());
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
TEST(ICSampleTest, SortIntoBinsMaintainsCsrInvariant) {
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
  for (auto& v : s.veto_conv) v.resize(N);
  for (auto& v : s.veto_prompt) v.resize(N);
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
    for (int k = 0; k < 3; ++k) {
      s.veto_conv[k][i]   = 5000.0 * (k + 1) + idx;  // offset 5000*(k+1)
      s.veto_prompt[k][i] = 9000.0 * (k + 1) + idx;  // offset 9000*(k+1)
    }
  }

  s.sort_into_bins(/*total_bins=*/4);

  // The out-of-range event (original index 1) must be dropped: 6 -> 5.
  ASSERT_TRUE(s.size() == 5);

  // CSR shape: total_bins + 1 offsets, monotonic non-decreasing, last == size().
  ASSERT_TRUE(s.bin_offsets.size() == 5);
  for (std::size_t b = 0; b + 1 < s.bin_offsets.size(); ++b)
    ASSERT_TRUE(s.bin_offsets[b] <= s.bin_offsets[b + 1]);
  ASSERT_TRUE(s.bin_offsets.back() == s.size());

  // Expected per-bin counts: bin0={2,5}(2), bin1={4}(1), bin2={0,3}(2), bin3={}(0).
  ASSERT_TRUE(s.bin_offsets[0] == 0);
  ASSERT_TRUE(s.bin_offsets[1] == 2);
  ASSERT_TRUE(s.bin_offsets[2] == 3);
  ASSERT_TRUE(s.bin_offsets[3] == 5);
  ASSERT_TRUE(s.bin_offsets[4] == 5);

  // Events are grouped by bin: bin_idx is non-decreasing across the sorted array.
  for (std::size_t i = 0; i + 1 < s.size(); ++i)
    ASSERT_TRUE(s.bin_idx[i] <= s.bin_idx[i + 1]);

  // Original index 1 (dropped) must not survive in any column.
  for (std::size_t i = 0; i < s.size(); ++i)
    ASSERT_TRUE(s.e_true[i] != 1.0);

  // Every column was permuted in lockstep with e_true (same permutation
  // applied to every per-event column, including all four Barr slopes).
  for (std::size_t i = 0; i < s.size(); ++i) {
    ASSERT_TRUE(std::abs(s.astro_baseline[i]  - s.e_true[i] - 100.0) < 1e-9);
    ASSERT_TRUE(std::abs(s.conv_baseline[i]   - s.e_true[i] - 200.0) < 1e-9);
    ASSERT_TRUE(std::abs(s.conv_alt[i]        - s.e_true[i] - 300.0) < 1e-9);
    ASSERT_TRUE(std::abs(s.prompt_baseline[i] - s.e_true[i] - 400.0) < 1e-9);
    ASSERT_TRUE(std::abs(s.prompt_alt[i]      - s.e_true[i] - 500.0) < 1e-9);
    for (int k = 0; k < params::ic::nBarrParams; ++k)
      ASSERT_TRUE(std::abs(s.barr_conv[k][i] - s.e_true[i] - 1000.0 * (k + 1)) < 1e-9);
    for (int k = 0; k < 3; ++k) {
      ASSERT_TRUE(std::abs(s.veto_conv[k][i]   - s.e_true[i] - 5000.0 * (k + 1)) < 1e-9);
      ASSERT_TRUE(std::abs(s.veto_prompt[k][i] - s.e_true[i] - 9000.0 * (k + 1)) < 1e-9);
    }
  }

  // Explicit grouping check via the recorded permutation: sorted original
  // indices should be [2, 5, 4, 0, 3] (stable within each bin).
  const double expected_original_index[5] = {2.0, 5.0, 4.0, 0.0, 3.0};
  for (std::size_t i = 0; i < 5; ++i)
    ASSERT_TRUE(std::abs(s.e_true[i] - expected_original_index[i]) < 1e-9);
}

// 2D test binning: Log10Energy in [2, 5) with 3 bins, CosZenith in [-1, 1) with
// 2 bins -> 6 analysis bins total.
static Binning synthetic_binning() {
  return Binning({Axis{Axis::Kind::Log10Energy, 2.0, 5.0, 3},
                  Axis{Axis::Kind::CosZenith, -1.0, 1.0, 2}});
}

// Tiny synthetic tracks-like sample over synthetic_binning(): 2 events per bin,
// every per-event column populated with nonzero physically plausible values,
// already compacted into the CSR bin layout. With `with_atmospheric = false`
// the conventional/prompt/Barr columns are left empty, exactly as
// ICDataBase::read_sample leaves them for a sample that declares only "astro".
static io::ic::ICSample synthetic_sample(const Binning& binning, const bool with_atmospheric) {
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
  sample.bin_idx.reserve(n_events);

  for (int eb = 0; eb < 3; ++eb) {
    for (int zb = 0; zb < 2; ++zb) {
      for (int k = 0; k < events_per_bin; ++k) {
        // Small per-event jitter so events within a bin aren't identical.
        const double jitter      = 1.0 + 0.01 * static_cast<double>(k);
        const double e_true      = energies[eb] * jitter;
        const double reco_e      = energies[eb] * jitter;
        const double reco_z      = zeniths[zb];
        const double conv_base   = 5.0e-3 * jitter;
        const double prompt_base = 1.0e-4 * jitter;

        sample.e_true.push_back(e_true);
        sample.astro_baseline.push_back(1.0e-3 * jitter);
        if (with_atmospheric) {
          sample.conv_baseline.push_back(conv_base);
          sample.conv_alt.push_back(0.9 * conv_base);
          sample.prompt_baseline.push_back(prompt_base);
          sample.prompt_alt.push_back(0.9 * prompt_base);
          for (int b = 0; b < params::ic::nBarrParams; ++b)
            sample.barr_conv[b].push_back(1.0e-4 * (b + 1) * jitter);
        }

        const double reco[2] = {reco_e, reco_z};
        sample.bin_idx.push_back(binning.bin_index(reco));
      }
    }
  }
  if (static_cast<int>(sample.size()) != n_events)
    throw std::logic_error("failed to construct the complete synthetic sample");

  sample.sort_into_bins(binning.total_bins());
  // All 12 synthetic events were built in-range; none should be dropped.
  if (static_cast<int>(sample.size()) != n_events || sample.bin_offsets.empty()
      || sample.bin_offsets.back() != sample.size())
    throw std::logic_error("synthetic sample does not have a valid CSR layout");
  return sample;
}

static ana::ic::GlobalFluxSettings synthetic_settings() {
  return ana::ic::GlobalFluxSettings{.e_ref_gev                = 1.0e5,
                                     .astro_reference_index    = 2.0,
                                     .conv_delta_gamma_e_ref   = 1.0e3,
                                     .prompt_delta_gamma_e_ref = 3.8e3,
                                     .astro_per_type_norm      = false,
                                     .veto_anchor_energy       = 100.0,
                                     .veto_rescale_energy      = 100.0};
}

// Nominal parameter values mirror configs/config_icecube_tracks_cpu.json's
// "StartValue" entries (physically reasonable defaults for this layout).
static std::vector<double> nominal_parameter_values() {
  std::vector<double> values(params::ic::number_of_parameters(), 0.0);
  values[params::ic::AstroNorm]     = 1.5;
  values[params::ic::SpectralIndex] = 2.4;
  values[params::ic::ConvNorm]      = 1.0;
  values[params::ic::PromptNorm]    = 1.0;
  // BarrH/W/Y/Z, CRGrad, DeltaGamma stay 0.0 (nominal).
  values[params::ic::MuonNorm] = 1.0;
  values[params::ic::DOMEff]   = 1.0;
  values[params::ic::IceAbs]   = 1.0;
  values[params::ic::IceScat]  = 1.0;
  return values;
}

// Checks that the Asimov point (predicted == data by construction) minimizes
// SampleLikelihood's partial -2lnL: perturbing AstroNorm away from its nominal
// value must not improve (must strictly worsen) the likelihood. Exercises the
// CPU (gpu = nullptr) path of both PowerlawFlux and AtmosphericFlux end to end,
// including the SAY ssq path (assemble_fluctuation).
TEST(SampleLikelihoodTest, AsimovIsMinimum) {
  using ana::ic::SampleLikelihood;
  using ana::ParameterWrapper;

  const Binning binning = synthetic_binning();
  ASSERT_TRUE(binning.total_bins() == 6);
  const io::ic::ICSample sample = synthetic_sample(binning, /*with_atmospheric=*/true);

  const io::ic::SampleConfig cfg{.name       = "unit_test_sample",
                                 .binning    = binning,
                                 .mc_binning = binning,
                                 .components = {"astro", "conventional", "prompt"}};

  SampleLikelihood likelihood(sample, cfg, synthetic_settings(), /*gpu=*/nullptr, /*use_say=*/true);

  const std::vector<double> nominal_values = nominal_parameter_values();

  ParameterWrapper nominal(params::ic::number_of_parameters());
  nominal.reset_parameter(nominal_values.data());

  likelihood.generate_asimov(nominal);

  const double llh_nominal = likelihood.partial_llh(nominal);
  ASSERT_TRUE(std::isfinite(llh_nominal));

  // Perturb AstroNorm x1.5 away from its Asimov (nominal) value; everything
  // else stays at the nominal/config default.
  std::vector<double> perturbed_values = nominal_values;
  perturbed_values[params::ic::AstroNorm] *= 1.5;

  ParameterWrapper perturbed(params::ic::number_of_parameters());
  perturbed.reset_parameter(perturbed_values.data());

  const double llh_perturbed = likelihood.partial_llh(perturbed);
  ASSERT_TRUE(std::isfinite(llh_perturbed));

  // The Asimov point (data == prediction at nominal) must minimize -2lnL:
  // do NOT assert llh_nominal ~= 0 -- this instance is SAY, which (like
  // NNMFit's) does not subtract the saturated term, so the minimum is a
  // nonzero value. The Poisson term does subtract it; that zero is asserted
  // by LikelihoodParityTest.PoissonAsimovIsExactlyZero.
  ASSERT_TRUE(llh_nominal < llh_perturbed);
}

// The GPU SAY path (flux kernels leaving per-event weights on the GPU + the
// say_ssq reduction kernel) must reproduce the CPU path within FP32 tolerance,
// at the Asimov point and away from it. Skipped when no Metal device exists.
TEST(SampleLikelihoodTest, MetalSaySsqMatchesCpu) {
  using ana::ic::MetalBackend;
  using ana::ic::SampleLikelihood;
  using ana::ParameterWrapper;

  if (!MetalBackend::available()) {
    GTEST_SKIP() << "Metal backend is unavailable";
  }

  const Binning          binning = synthetic_binning();
  const io::ic::ICSample sample  = synthetic_sample(binning, /*with_atmospheric=*/true);

  const io::ic::SampleConfig cfg{.name       = "metal_ssq_sample",
                                 .binning    = binning,
                                 .mc_binning = binning,
                                 .components = {"astro", "conventional", "prompt"}};

  const auto backend = std::make_shared<MetalBackend>();

  SampleLikelihood cpu(sample, cfg, synthetic_settings(), /*gpu=*/nullptr, /*use_say=*/true);
  SampleLikelihood gpu(sample, cfg, synthetic_settings(), backend->create_session(), /*use_say=*/true);

  const std::vector<double> nominal_values = nominal_parameter_values();
  ParameterWrapper          nominal(params::ic::number_of_parameters());
  nominal.reset_parameter(nominal_values.data());

  cpu.generate_asimov(nominal);
  gpu.generate_asimov(nominal);

  // Move every ssq-relevant parameter off its nominal value so the SAY term
  // (data fixed at the nominal Asimov, prediction and ssq recomputed) is
  // sensitive to a wrong ssq, then compare the partial -2lnL.
  std::vector<double> perturbed_values     = nominal_values;
  perturbed_values[params::ic::AstroNorm]  = 1.9;
  perturbed_values[params::ic::ConvNorm]   = 1.2;
  perturbed_values[params::ic::PromptNorm] = 0.6;

  for (const auto& values : {nominal_values, perturbed_values}) {
    ParameterWrapper parameter(params::ic::number_of_parameters());
    parameter.reset_parameter(values.data());

    const double llh_cpu = cpu.partial_llh(parameter);
    const double llh_gpu = gpu.partial_llh(parameter);
    ASSERT_TRUE(std::isfinite(llh_cpu) && std::isfinite(llh_gpu));

    // FP32 flux weights + FP32 ssq reduction vs FP64: a loose relative bound
    // still catches a structurally wrong ssq (missing component, wrong bin).
    const double scale = std::max({std::fabs(llh_cpu), std::fabs(llh_gpu), 1.0});
    ASSERT_TRUE(std::fabs(llh_cpu - llh_gpu) / scale < 1.0e-4);

    // The per-bin predictions must agree too (flux kernels unchanged by the
    // ssq work; this pins the readback-skip refactor).
    const auto pred_cpu = cpu.predicted();
    const auto pred_gpu = gpu.predicted();
    for (std::size_t b = 0; b < pred_cpu.size(); ++b) {
      const double bin_scale = std::max({std::fabs(pred_cpu[b]), std::fabs(pred_gpu[b]), 1.0e-30});
      ASSERT_TRUE(std::fabs(pred_cpu[b] - pred_gpu[b]) / bin_scale < 1.0e-4);
    }
  }
}

// The point of splitting the backend from the session is that the expensive
// half is paid once per process rather than once per Fit. A scan builds one Fit
// per grid point, so a second sample over the same backend must upload no new
// column and compile no new kernel -- and a session must give its output buffers
// back when it dies, or the saving is traded for a leak that grows per point.
// None of that is observable from the fit results, hence the counters.
TEST(GpuBackendTest, SharedBackendUploadsAndCompilesOnce) {
  using ana::ic::MetalBackend;
  using ana::ic::SampleLikelihood;

  if (!MetalBackend::available()) {
    GTEST_SKIP() << "Metal backend is unavailable";
  }

  const Binning          binning = synthetic_binning();
  const io::ic::ICSample sample  = synthetic_sample(binning, /*with_atmospheric=*/true);

  const io::ic::SampleConfig cfg{.name       = "shared_backend_sample",
                                 .binning    = binning,
                                 .mc_binning = binning,
                                 .components = {"astro", "conventional", "prompt"}};

  const auto backend = std::make_shared<MetalBackend>();

  SampleLikelihood first(sample, cfg, synthetic_settings(), backend->create_session(),
                         /*use_say=*/true);
  const std::size_t columns_after_first = backend->column_count();
  const std::size_t kernels_after_first = backend->kernel_compile_count();

  ASSERT_GT(columns_after_first, 0u);
  // powerlaw_hist, atmo_hist, say_ssq.
  ASSERT_EQ(kernels_after_first, 3u);

  SampleLikelihood second(sample, cfg, synthetic_settings(), backend->create_session(),
                          /*use_say=*/true);
  EXPECT_EQ(backend->column_count(), columns_after_first);
  EXPECT_EQ(backend->kernel_compile_count(), kernels_after_first);
}

TEST(GpuBackendTest, SessionOutputsFreedOnDestruction) {
  using ana::ic::MetalBackend;
  using ana::ic::SampleLikelihood;

  if (!MetalBackend::available()) {
    GTEST_SKIP() << "Metal backend is unavailable";
  }

  const Binning          binning = synthetic_binning();
  const io::ic::ICSample sample  = synthetic_sample(binning, /*with_atmospheric=*/true);

  const io::ic::SampleConfig cfg{.name       = "session_lifetime_sample",
                                 .binning    = binning,
                                 .mc_binning = binning,
                                 .components = {"astro", "conventional", "prompt"}};

  const auto backend = std::make_shared<MetalBackend>();
  ASSERT_EQ(backend->live_output_count(), 0u);

  std::size_t live_while_alive = 0;
  for (int i = 0; i < 5; ++i) {
    SampleLikelihood likelihood(sample, cfg, synthetic_settings(), backend->create_session(),
                                /*use_say=*/true);
    if (i == 0) {
      live_while_alive = backend->live_output_count();
      ASSERT_GT(live_while_alive, 0u);
    }
    // Every iteration must cost the same: outputs are per session, never pooled
    // across them, so a session that failed to free would show up as growth.
    EXPECT_EQ(backend->live_output_count(), live_while_alive);
  }

  EXPECT_EQ(backend->live_output_count(), 0u);
}

// An "astro"-only sample must run on a sample whose atmospheric columns were
// never read: no atmospheric flux is constructed, the prediction is the
// astrophysical component alone, and the atmospheric parameters have no effect
// on it. This is what lets a cascade parquet without conv/prompt/Barr columns
// (or a tracks parquet used astro-only) be fitted.
TEST(SampleLikelihoodTest, ComponentMasking) {
  using ana::ic::SampleLikelihood;
  using ana::ParameterWrapper;

  const Binning binning = synthetic_binning();

  const io::ic::ICSample full       = synthetic_sample(binning, /*with_atmospheric=*/true);
  const io::ic::ICSample astro_only = synthetic_sample(binning, /*with_atmospheric=*/false);
  ASSERT_TRUE(astro_only.conv_baseline.empty());
  ASSERT_TRUE(astro_only.prompt_baseline.empty());
  ASSERT_TRUE(astro_only.barr_conv[0].empty());
  ASSERT_TRUE(astro_only.size() == full.size());

  const io::ic::SampleConfig astro_cfg{
      .name = "astro_only", .binning = binning, .mc_binning = binning, .components = {"astro"}};
  const io::ic::SampleConfig full_cfg{.name       = "all_components",
                                      .binning    = binning,
                                      .mc_binning = binning,
                                      .components = {"astro", "conventional", "prompt"}};

  SampleLikelihood astro_llh(astro_only, astro_cfg, synthetic_settings(), nullptr, /*use_say=*/true);
  SampleLikelihood full_llh(full, full_cfg, synthetic_settings(), nullptr, /*use_say=*/true);

  const std::vector<double> nominal_values = nominal_parameter_values();
  ParameterWrapper          nominal(params::ic::number_of_parameters());
  nominal.reset_parameter(nominal_values.data());

  astro_llh.generate_asimov(nominal);
  full_llh.generate_asimov(nominal);

  const auto astro_pred = astro_llh.predicted();
  const auto full_pred  = full_llh.predicted();
  ASSERT_TRUE(astro_pred.size() == static_cast<std::size_t>(binning.total_bins()));

  // Every bin has atmospheric events in the full sample, so masking them out
  // must lower the prediction everywhere -- and leave it strictly positive.
  for (std::size_t b = 0; b < astro_pred.size(); ++b) {
    ASSERT_TRUE(astro_pred[b] > 0.0);
    ASSERT_TRUE(astro_pred[b] < full_pred[b]);
  }

  const double astro_at_nominal = astro_llh.partial_llh(nominal);
  ASSERT_TRUE(std::isfinite(astro_at_nominal));

  // Moving the atmospheric parameters cannot touch an astro-only sample.
  std::vector<double> atmo_shifted = nominal_values;
  atmo_shifted[params::ic::ConvNorm] *= 1.7;
  atmo_shifted[params::ic::PromptNorm] *= 0.3;
  atmo_shifted[params::ic::CRGrad]     = 0.5;
  atmo_shifted[params::ic::DeltaGamma] = 0.1;
  atmo_shifted[params::ic::BarrH]      = 0.2;

  ParameterWrapper shifted(params::ic::number_of_parameters());
  shifted.reset_parameter(atmo_shifted.data());
  ASSERT_TRUE(astro_llh.partial_llh(shifted) == astro_at_nominal);

  // ... while the same shift does move the sample that includes them.
  const double full_at_nominal = full_llh.partial_llh(nominal);
  ASSERT_TRUE(full_llh.partial_llh(shifted) != full_at_nominal);

  // AstroNorm still moves the astro-only sample away from its Asimov minimum.
  std::vector<double> astro_shifted = nominal_values;
  astro_shifted[params::ic::AstroNorm] *= 1.5;
  ParameterWrapper astro_perturbed(params::ic::number_of_parameters());
  astro_perturbed.reset_parameter(astro_shifted.data());
  ASSERT_TRUE(astro_llh.partial_llh(astro_perturbed) > astro_at_nominal);
}

// A sample fitted with an RA axis must predict exactly the 2D prediction spread
// evenly over the RA bins (NNMFit Binning_2D_to_3D: repeat(mu, n_ra) / n_ra), and
// its sigma^2 the 2D sigma^2 divided by n_ra^2. A galactic template is the one
// piece that is genuinely 3D: it is added to mu after the broadcast, undivided,
// and stays out of sigma^2 entirely.
TEST(SampleLikelihoodTest, RaBroadcastMatches2d) {
  using ana::ic::GlobalFluxSettings;
  using ana::ic::SampleLikelihood;
  using ana::ParameterWrapper;

  io::ic::ICSample sample;
  sample.e_true         = {1.0e5, 2.0e5};
  sample.astro_baseline = {1.0e-8, 2.0e-8};
  sample.bin_idx        = {0, 0};
  sample.sort_into_bins(1);

  const GlobalFluxSettings settings{.e_ref_gev                = 1.0e5,
                                    .astro_reference_index    = 2.0,
                                    .conv_delta_gamma_e_ref   = 1000.0,
                                    .prompt_delta_gamma_e_ref = 3800.0,
                                    .astro_per_type_norm      = false,
                                    .veto_anchor_energy       = 100.0,
                                    .veto_rescale_energy      = 100.0};

  const Binning binning_2d({io::ic::parse_axis("Log10Energy", "(2.0, 7.0, 1)"),
                            io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 1)")});
  const Binning binning_3d({io::ic::parse_axis("Log10Energy", "(2.0, 7.0, 1)"),
                            io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 1)"),
                            io::ic::parse_axis("Ra", "(0.0, 6.28319, 4)")});

  io::ic::SampleConfig cfg_2d{.name = "flat", .binning = binning_2d, .mc_binning = binning_2d};
  cfg_2d.livetime   = 1.0e8;
  cfg_2d.components = {"astro"};

  io::ic::SampleConfig cfg_3d{.name = "with_ra", .binning = binning_3d, .mc_binning = binning_2d};
  cfg_3d.livetime   = 1.0e8;
  cfg_3d.components = {"astro"};

  std::vector<double> values(params::ic::number_of_parameters(), 0.0);
  values[params::ic::AstroNorm]     = 1.5;
  values[params::ic::SpectralIndex] = 2.4;
  ParameterWrapper parameter(params::ic::number_of_parameters());
  parameter.reset_parameter(values.data());

  SampleLikelihood flat(sample, cfg_2d, settings, nullptr, true);
  SampleLikelihood with_ra(sample, cfg_3d, settings, nullptr, true);
  flat.generate_asimov(parameter);
  with_ra.generate_asimov(parameter);

  ASSERT_TRUE(flat.predicted().size() == 1);
  ASSERT_TRUE(with_ra.predicted().size() == 4);
  for (int r = 0; r < 4; ++r)
    ASSERT_TRUE(with_ra.predicted()[r] == flat.predicted()[0] / 4.0);

  // sigma^2 is a squared quantity, so its divisor is n_ra^2, not n_ra. Getting
  // this divisor wrong at the call site would leave mu right and silently scale
  // every SAY variance by n_ra -- exactly the kind of error that survives to a
  // published number, so assert it rather than only asserting mu.
  ASSERT_TRUE(flat.ssq().size() == 1);
  ASSERT_TRUE(with_ra.ssq().size() == 4);
  ASSERT_TRUE(flat.ssq()[0] > 0.0);
  for (int r = 0; r < 4; ++r)
    ASSERT_TRUE(with_ra.ssq()[r] == flat.ssq()[0] / 16.0);

  // A galactic template is already in the analysis binning: its per-bin rate is
  // added to mu UNDIVIDED (no RA broadcast), and it contributes nothing to
  // sigma^2 -- NNMFit's GalacticTemplate has no fluctuation graph and is left out
  // of the ssq sum (histogram_builder.py:307).
  const std::string galactic_path = "ictests_galactic.txt";
  const double      galactic_rates[4] = {1.0e-8, 2.0e-8, 3.0e-8, 4.0e-8};
  {
    std::ofstream out(galactic_path);
    out << "# template bins 4\n";
    out << "# columns: template_rate fluctuation_rate (both per second)\n";
    for (const double rate : galactic_rates) out << rate << " 0\n";
  }

  io::ic::SampleConfig cfg_gal{.name = "with_galactic", .binning = binning_3d, .mc_binning = binning_2d};
  cfg_gal.livetime   = 1.0e8;
  cfg_gal.components = {"astro"};
  cfg_gal.galactic.push_back({.name = "fermi", .file = galactic_path, .norm_index = params::ic::GalacticNorm0});

  const double galactic_norm = 2.0;
  values[params::ic::AstroNorm]        = 1.5;
  values[params::ic::GalacticNorm0]    = galactic_norm;
  ParameterWrapper with_galactic_parameter(params::ic::number_of_parameters());
  with_galactic_parameter.reset_parameter(values.data());

  SampleLikelihood with_galactic(sample, cfg_gal, settings, nullptr, true);
  std::remove(galactic_path.c_str());
  with_galactic.generate_asimov(with_galactic_parameter);

  for (int r = 0; r < 4; ++r) {
    const double expected =
        with_ra.predicted()[r] + galactic_norm * (galactic_rates[r] * cfg_gal.livetime);
    ASSERT_TRUE(std::abs(with_galactic.predicted()[r] - expected) < 1e-12 * expected);
    // Undivided: the galactic part alone must be the full per-bin rate, not a
    // quarter of it.
    ASSERT_TRUE(std::abs(with_galactic.galactic_histogram()[r] -
                    galactic_norm * (galactic_rates[r] * cfg_gal.livetime)) < 1e-12);
    // sigma^2 is untouched by the galactic template.
    ASSERT_TRUE(with_galactic.ssq()[r] == with_ra.ssq()[r]);
  }

  // A galactic file with a non-zero fluctuation column contradicts that exclusion
  // and must be rejected rather than silently ignored.
  const std::string bad_galactic = "ictests_galactic_bad.txt";
  {
    std::ofstream out(bad_galactic);
    out << "# template bins 4\n";
    for (int b = 0; b < 4; ++b) out << galactic_rates[b] << " 1.0e-9\n";
  }
  io::ic::SampleConfig cfg_bad = cfg_gal;
  cfg_bad.galactic[0].file     = bad_galactic;
  bool threw = false;
  try {
    SampleLikelihood rejected(sample, cfg_bad, settings, nullptr, true);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  std::remove(bad_galactic.c_str());
  ASSERT_TRUE(threw);

  // Asimov data is the prediction, so the partial -2lnL of the two must agree to
  // the extent SAY allows: the 3D bins are the 2D bin split four ways, with sigma^2
  // split by 4^2, which is the split SAY's per-bin term is invariant under only
  // approximately -- so compare the predictions, not the likelihood, and just
  // require the 3D likelihood to be finite and at its minimum.
  const double at_nominal = with_ra.partial_llh(parameter);
  ASSERT_TRUE(std::isfinite(at_nominal));

  values[params::ic::AstroNorm] = 1.8;
  ParameterWrapper perturbed(params::ic::number_of_parameters());
  perturbed.reset_parameter(values.data());
  ASSERT_TRUE(with_ra.partial_llh(perturbed) > at_nominal);
}

// NNMFit AstroBPL (parameters/astroBPL.py). e_break is log10(E/GeV); `pivot`
// renormalises so the norm is the flux at 100 TeV whichever side of the break
// that falls on. Checked here against a hand-evaluated reference rather than
// against a second implementation of the same formula.
TEST(PowerlawFluxTest, BrokenPowerLaw) {
  using ana::ic::AstroModel;
  using ana::ic::PowerlawFlux;
  using ana::ParameterWrapper;

  // Two events, one either side of the break at 10^4.4 GeV.
  io::ic::ICSample sample;
  sample.e_true         = {1.0e3, 1.0e6};
  sample.astro_baseline = {2.0e-9, 3.0e-9};
  sample.bin_idx        = {0, 0};
  sample.sort_into_bins(1);

  const Binning binning({io::ic::parse_axis("Log10Energy", "(2.0, 7.0, 1)"),
                         io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 1)")});

  const double norm = 1.77, g1 = 1.31, g2 = 2.74, log_ebreak = 4.4;

  std::vector<double> values(params::ic::number_of_parameters(), 0.0);
  values[params::ic::AstroNorm]    = norm;
  values[params::ic::AstroGamma1]  = g1;
  values[params::ic::AstroGamma2]  = g2;
  values[params::ic::AstroEBreak]  = log_ebreak;
  ParameterWrapper parameter(params::ic::number_of_parameters());
  parameter.reset_parameter(values.data());

  // per_type_norm = false, so the 0.5 factor applies.
  PowerlawFlux flux(sample, binning, 1.0e5, 2.0, false, nullptr, false,
                    AstroModel::BrokenPowerlaw);
  ASSERT_TRUE(flux.check_and_recalculate(parameter));

  const double e_break = std::pow(10.0, log_ebreak);
  const double pivot   = 1.0e5 < e_break ? std::pow(1.0e5 / e_break, g1)
                                         : std::pow(1.0e5 / e_break, g2);
  double expected = 0.0;
  for (int i = 0; i < 2; ++i) {
    const double e     = sample.e_true[i];
    const double shape = e < e_break ? std::pow(e / e_break, -g1) : std::pow(e / e_break, -g2);
    expected += sample.astro_baseline[i] * norm * pivot * shape *
                (e / 1.0e5) * (e / 1.0e5) * 0.5;
  }
  ASSERT_TRUE(std::abs(flux.histogram()[0] - expected) < 1e-12 * expected);

  // The break must actually bite: moving it above both events changes the result.
  values[params::ic::AstroEBreak] = 7.0;
  ParameterWrapper moved(params::ic::number_of_parameters());
  moved.reset_parameter(values.data());
  ASSERT_TRUE(flux.check_and_recalculate(moved));
  ASSERT_TRUE(std::abs(flux.histogram()[0] - expected) > 1e-6 * expected);
}

// The single-power-law mode must be untouched by the broken-power-law work:
// same inputs, same numbers as the formula documented on PowerlawFlux.
TEST(PowerlawFluxTest, SinglePowerLawUnchanged) {
  using ana::ic::AstroModel;
  using ana::ic::PowerlawFlux;
  using ana::ParameterWrapper;

  io::ic::ICSample sample;
  sample.e_true         = {1.0e3, 1.0e6};
  sample.astro_baseline = {2.0e-9, 3.0e-9};
  sample.bin_idx        = {0, 0};
  sample.sort_into_bins(1);

  const Binning binning({io::ic::parse_axis("Log10Energy", "(2.0, 7.0, 1)"),
                         io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 1)")});

  std::vector<double> values(params::ic::number_of_parameters(), 0.0);
  values[params::ic::AstroNorm]     = 1.5;
  values[params::ic::SpectralIndex] = 2.4;
  ParameterWrapper parameter(params::ic::number_of_parameters());
  parameter.reset_parameter(values.data());

  PowerlawFlux flux(sample, binning, 1.0e5, 2.0, false, nullptr, false, AstroModel::Powerlaw);
  ASSERT_TRUE(flux.check_and_recalculate(parameter));

  double expected = 0.0;
  for (int i = 0; i < 2; ++i)
    expected += sample.astro_baseline[i] * 0.5 * 1.5 *
                std::pow(sample.e_true[i] / 1.0e5, 2.0 - 2.4);
  ASSERT_TRUE(std::abs(flux.histogram()[0] - expected) < 1e-12 * expected);
}

// The veto reweight is NNMFit's VetoThreshold parameter (parameters/veto_threshold.py):
//   e  = rescale * 10^p - anchor        (both 100 GeV in the combined config)
//   PF = 10^(a + b*e + c*e^2)           per event, per component
// applied multiplicatively to the conventional and prompt weights. Checked here
// against the same formula evaluated by hand on a one-bin sample, and against the
// invariant that veto-off reproduces the un-vetoed prediction exactly.
TEST(AtmosphericFluxTest, VetoReweight) {
  using ana::ic::AtmosphericFlux;
  using ana::ParameterWrapper;

  const Binning binning({io::ic::parse_axis("Log10Energy", "(2.0, 5.0, 1)"),
                         io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 1)")});
  ASSERT_TRUE(binning.total_bins() == 1);

  // Two events in the single bin, no Barr slopes and conv_alt == conv_base, so the
  // un-vetoed weight is exactly conv_base * ConvNorm + prompt_base * PromptNorm at
  // DeltaGamma = 0 and e_true == the delta-gamma reference energy.
  io::ic::ICSample sample;
  const double     conv[2]   = {2.0, 3.0};
  const double     prompt[2] = {0.5, 0.25};
  const double     a_conv[2] = {-0.30, -0.20};
  const double     b_conv[2] = {-1.0e-3, -2.0e-3};
  const double     c_conv[2] = {1.0e-6, 2.0e-6};
  const double     a_pr[2]   = {-0.10, -0.05};
  const double     b_pr[2]   = {-5.0e-4, -1.0e-3};
  const double     c_pr[2]   = {5.0e-7, 1.0e-6};

  for (int i = 0; i < 2; ++i) {
    sample.e_true.push_back(1000.0);
    sample.astro_baseline.push_back(0.0);
    sample.conv_baseline.push_back(conv[i]);
    sample.conv_alt.push_back(conv[i]);
    sample.prompt_baseline.push_back(prompt[i]);
    sample.prompt_alt.push_back(prompt[i]);
    for (int k = 0; k < params::ic::nBarrParams; ++k) sample.barr_conv[k].push_back(0.0);
    sample.veto_conv[0].push_back(a_conv[i]);
    sample.veto_conv[1].push_back(b_conv[i]);
    sample.veto_conv[2].push_back(c_conv[i]);
    sample.veto_prompt[0].push_back(a_pr[i]);
    sample.veto_prompt[1].push_back(b_pr[i]);
    sample.veto_prompt[2].push_back(c_pr[i]);
    sample.bin_idx.push_back(0);
  }
  sample.sort_into_bins(binning.total_bins());

  std::vector<double> values(params::ic::number_of_parameters(), 0.0);
  values[params::ic::ConvNorm]   = 1.0;
  values[params::ic::PromptNorm] = 1.0;

  auto histogram_at = [&](const bool use_veto, const double veto_threshold) {
    AtmosphericFlux flux(sample, binning, /*conv_e_ref=*/1000.0, /*prompt_e_ref=*/1000.0,
                         /*gpu=*/nullptr, /*need_per_event=*/false,
                         /*use_veto=*/use_veto, /*veto_anchor_energy=*/100.0,
                         /*veto_rescale_energy=*/100.0);
    std::vector<double> v = values;
    v[params::ic::VetoThreshold] = veto_threshold;
    ParameterWrapper p(params::ic::number_of_parameters());
    p.reset_parameter(v.data());
    flux.check_and_recalculate(p);
    return flux.histogram()[0];
  };

  // Reference: the NNMFit formula, evaluated in double precision here.
  auto reference = [&](const bool use_veto, const double veto_threshold) {
    const double e     = 100.0 * std::pow(10.0, veto_threshold) - 100.0;
    double       total = 0.0;
    for (int i = 0; i < 2; ++i) {
      double conv_w   = conv[i];
      double prompt_w = prompt[i];
      if (use_veto) {
        conv_w *= std::pow(10.0, a_conv[i] + b_conv[i] * e + c_conv[i] * e * e);
        prompt_w *= std::pow(10.0, a_pr[i] + b_pr[i] * e + c_pr[i] * e * e);
      }
      total += conv_w + prompt_w;
    }
    return total;
  };

  const double unvetoed = reference(false, 0.0);
  ASSERT_TRUE(std::abs(histogram_at(false, 0.0) - unvetoed) < 1e-12 * unvetoed);
  for (double p : {0.0, 0.5, -0.5, 1.301, -1.301})
    ASSERT_TRUE(std::abs(histogram_at(true, p) - reference(true, p)) < 1e-12 * unvetoed);
  // The veto suppresses the flux, and its parameter actually triggers a recalculation.
  ASSERT_TRUE(histogram_at(true, 0.0) < histogram_at(false, 0.0));
  ASSERT_TRUE(histogram_at(true, 1.0) != histogram_at(true, 0.0));
}

// A muon template is a per-bin rate plus a per-bin fluctuation; the component
// scales both by its norm parameter and the sample livetime, matching NNMFit
// (histogram_builder: ssq += (hist_fluctuation * livetime)**2).
TEST(TemplateFluxTest, RatesAndFluctuations) {
  using ana::ic::TemplateFlux;
  using ana::ParameterWrapper;

  const Binning binning({io::ic::parse_axis("Log10Energy", "(2.0, 5.0, 3)"),
                         io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 1)")});
  ASSERT_TRUE(binning.total_bins() == 3);

  const std::string path = "ictests_template.txt";
  {
    std::ofstream out(path);
    out << "# template bins 3\n";
    out << "# columns: template_rate fluctuation_rate (both per second)\n";
    out << "1.0e-6 2.0e-7\n";
    out << "2.0e-6 3.0e-7\n";
    out << "4.0e-6 5.0e-7\n";
  }

  const double livetime = 3.0e8;
  TemplateFlux flux(binning, path, params::ic::MuonGunNorm, livetime);
  std::remove(path.c_str());

  std::vector<double> values(params::ic::number_of_parameters(), 0.0);
  values[params::ic::MuonGunNorm] = 2.0;
  ParameterWrapper parameter(params::ic::number_of_parameters());
  parameter.reset_parameter(values.data());

  ASSERT_TRUE(flux.check_and_recalculate(parameter));

  const double rates[3]  = {1.0e-6, 2.0e-6, 4.0e-6};
  const double sigmas[3] = {2.0e-7, 3.0e-7, 5.0e-7};
  for (int b = 0; b < 3; ++b) {
    const double mu  = 2.0 * rates[b] * livetime;
    const double ssq = (2.0 * sigmas[b] * livetime) * (2.0 * sigmas[b] * livetime);
    ASSERT_TRUE(std::abs(flux.histogram()[b] - mu) < 1e-9 * mu);
    ASSERT_TRUE(std::abs(flux.fluctuation()[b] - ssq) < 1e-9 * ssq);
  }

  // Unchanged parameters must not trigger a recalculation. check_parameter_changed
  // compares against the set from the last reset_parameter() call, not against
  // what check_and_recalculate last saw -- so "unchanged" has to be established by
  // resetting to the same values again, exactly as ICLikelihood::calculate_likelihood
  // resets the shared ParameterWrapper once per evaluation.
  parameter.reset_parameter(values.data());
  ASSERT_TRUE(!flux.check_and_recalculate(parameter));

  // A template whose bin count disagrees with the binning is a hard error: it
  // would otherwise silently mis-assign every bin.
  const std::string bad = "ictests_template_bad.txt";
  {
    std::ofstream out(bad);
    out << "# template bins 2\n1.0 0.1\n2.0 0.2\n";
  }
  bool threw = false;
  try {
    TemplateFlux mismatched(binning, bad, params::ic::MuonGunNorm, livetime);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  std::remove(bad.c_str());
  ASSERT_TRUE(threw);
}

// SnowStorm gradients are a histogram-level additive perturbation of mu and
// sigma^2 (NNMFit snowstorm_gradient.make_graph, external_gradients: True so the
// histogram-gradient covariance is excluded):
//   D_k       = p_k - split_k
//   mu_add_b  = sum_k D_k * gradient_k_b * lt_scale
//   ssq_add_b = sum_k (D_k * error_k_b * lt_scale)^2 + 2 * sum_{i<j} D_i D_j cov_ij_b
TEST(DetectorSystematicsTest, AppliesGradientsAndCovariance) {
  using ana::ic::DetectorSystematics;
  using ana::ParameterWrapper;

  const Binning binning({io::ic::parse_axis("Log10Energy", "(2.0, 5.0, 2)"),
                         io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 1)")});
  ASSERT_TRUE(binning.total_bins() == 2);

  const int    n_sys    = params::ic::nDetSysParams;  // 5
  const double lt_scale = 2.0;
  const double split[5] = {1.0, 1.0, 1.0, 0.25, -0.05};
  // gradient[k][b], error[k][b]: distinct values so a transposed read is caught.
  double gradient[5][2];
  double error[5][2];
  for (int k = 0; k < n_sys; ++k)
    for (int b = 0; b < 2; ++b) {
      gradient[k][b] = 10.0 * (k + 1) + b;
      error[k][b]    = 0.5 * (k + 1) + 0.1 * b;
    }
  // covariance for each of the 10 unordered pairs, in the file's pair order.
  double cov[10][2];
  for (int p = 0; p < 10; ++p)
    for (int b = 0; b < 2; ++b) cov[p][b] = 0.01 * (p + 1) + 0.001 * b;

  static const char* kNames[5] = {"DOMEfficiency", "IceAbsorption", "IceScattering",
                                  "HoleIceForward_p0", "HoleIceForward_p1"};

  const std::string path = "ictests_gradients.txt";
  {
    std::ofstream out(path);
    out << "# gradients bins 2 params 5 lt_scale " << lt_scale << "\n";
    for (int k = 0; k < n_sys; ++k) {
      out << "# param " << kNames[k] << " split " << split[k] << "\n";
      for (int b = 0; b < 2; ++b) out << gradient[k][b] << " " << error[k][b] << "\n";
    }
    int pair = 0;
    for (int i = 0; i < n_sys; ++i)
      for (int j = i + 1; j < n_sys; ++j, ++pair) {
        out << "# cov " << kNames[i] << " " << kNames[j] << "\n";
        for (int b = 0; b < 2; ++b) out << cov[pair][b] << "\n";
      }
  }

  DetectorSystematics systematics(binning, path);
  std::remove(path.c_str());

  std::vector<double> values(params::ic::number_of_parameters(), 0.0);
  for (int k = 0; k < n_sys; ++k) values[params::ic::DOMEff + k] = split[k];
  ParameterWrapper at_split(params::ic::number_of_parameters());
  at_split.reset_parameter(values.data());
  systematics.check_and_recalculate(at_split);
  for (int b = 0; b < 2; ++b) {
    ASSERT_TRUE(systematics.mu_delta()[b] == 0.0);
    ASSERT_TRUE(systematics.ssq_delta()[b] == 0.0);
  }

  // Perturb two systematics.
  values[params::ic::DOMEff] = split[0] + 0.05;
  values[params::ic::IceAbs] = split[1] - 0.02;
  ParameterWrapper shifted(params::ic::number_of_parameters());
  shifted.reset_parameter(values.data());
  ASSERT_TRUE(systematics.check_and_recalculate(shifted));

  const double d[5] = {0.05, -0.02, 0.0, 0.0, 0.0};
  for (int b = 0; b < 2; ++b) {
    double mu_add = 0.0;
    for (int k = 0; k < n_sys; ++k) mu_add += d[k] * gradient[k][b] * lt_scale;

    double ssq_add = 0.0;
    for (int k = 0; k < n_sys; ++k) {
      const double term = d[k] * error[k][b] * lt_scale;
      ssq_add += term * term;
    }
    int pair = 0;
    for (int i = 0; i < n_sys; ++i)
      for (int j = i + 1; j < n_sys; ++j, ++pair)
        ssq_add += 2.0 * (d[i] * lt_scale) * (d[j] * lt_scale) * cov[pair][b];

    ASSERT_TRUE(std::abs(systematics.mu_delta()[b] - mu_add) < 1e-12 * std::abs(mu_add));
    ASSERT_TRUE(std::abs(systematics.ssq_delta()[b] - ssq_add) < 1e-12 * std::abs(ssq_add));
  }
}

// Real data is a plain per-bin count in the sample's own binning: no weights, no
// livetime scaling. This tests the counting helper the loader uses.
TEST(BinningTest, DataHistogramCounts2d) {
  const Binning binning({io::ic::parse_axis("Log10Energy", "(2.0, 5.0, 3)"),
                         io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 2)")});
  ASSERT_TRUE(binning.total_bins() == 6);

  // Three events in bin (energy 0, zenith 0), one out of range.
  const std::vector<double> energies{316.0, 316.0, 316.0, 10.0};
  const std::vector<double> zeniths{2.0, 2.0, 2.0, 2.0};

  const std::vector<double> counts = io::ic::bin_event_counts(binning, energies, zeniths);
  ASSERT_TRUE(counts.size() == 6);
  ASSERT_TRUE(counts[0] == 3.0);
  for (std::size_t b = 1; b < counts.size(); ++b) ASSERT_TRUE(counts[b] == 0.0);
}

// The analysis binning may carry a trailing RA axis; the MC binning is the same
// binning without it. Row-major flattening makes the 3D index the 2D index times
// n_ra plus the RA index, which is what makes the broadcast a block write.
TEST(BinningTest, DropRaAxis) {
  const Binning analysis({io::ic::parse_axis("Log10Energy", "(2.0, 5.0, 3)"),
                          io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 2)"),
                          io::ic::parse_axis("Ra", "(0.0, 6.28319, 4)")});
  ASSERT_TRUE(analysis.total_bins() == 24);

  const Binning mc = io::ic::drop_ra_axis(analysis);
  ASSERT_TRUE(mc.n_axes() == 2);
  ASSERT_TRUE(mc.total_bins() == 6);

  // A binning with no RA axis is returned unchanged.
  const Binning two_d({io::ic::parse_axis("Log10Energy", "(2.0, 5.0, 3)"),
                       io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 2)")});
  ASSERT_TRUE(io::ic::drop_ra_axis(two_d).total_bins() == 6);

  // The flat index of (e, zen, ra) is the 2D index * n_ra + ra index.
  const std::array<double, 3> reco{316.0, 2.0, 0.1};
  const std::array<double, 2> reco_2d{316.0, 2.0};
  const int flat_3d = analysis.bin_index(reco);
  const int flat_2d = mc.bin_index(reco_2d);
  ASSERT_TRUE(flat_3d == flat_2d * 4 + analysis.axes()[2].index(0.1));

  auto throws = [](auto&& f) {
    try {
      f();
    } catch (const std::runtime_error&) {
      return true;
    }
    return false;
  };

  // Ra axis present but not last: row-major flattening would not give the
  // (2D index * n_ra + ra index) layout the broadcast assumes.
  const Binning ra_first({io::ic::parse_axis("Ra", "(0.0, 6.28319, 4)"),
                          io::ic::parse_axis("Log10Energy", "(2.0, 5.0, 3)")});
  ASSERT_TRUE(throws([&] { (void)io::ic::drop_ra_axis(ra_first); }));

  // A binning cannot consist of the Ra axis alone.
  const Binning ra_only({io::ic::parse_axis("Ra", "(0.0, 6.28319, 4)")});
  ASSERT_TRUE(throws([&] { (void)io::ic::drop_ra_axis(ra_only); }));
}

// "Has an Ra axis" is a property of the axis list, not of the RA bin count: a
// one-bin Ra axis is a legitimate binning (the 2D cross-check of a 3D fit) and
// must still be treated as three-dimensional everywhere -- reading its data with
// only two reco columns would run off the end of the reco array.
TEST(BinningTest, HasRaAxis) {
  const Binning three_d({io::ic::parse_axis("Log10Energy", "(2.0, 5.0, 3)"),
                         io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 2)"),
                         io::ic::parse_axis("Ra", "(0.0, 6.28319, 4)")});
  const Binning two_d({io::ic::parse_axis("Log10Energy", "(2.0, 5.0, 3)"),
                       io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 2)")});
  ASSERT_TRUE(io::ic::has_ra_axis(three_d));
  ASSERT_TRUE(!io::ic::has_ra_axis(two_d));

  // One RA bin: ra_bin_count() is 1 exactly as for the 2D binning, so only
  // has_ra_axis() tells the two apart.
  const Binning one_ra_bin({io::ic::parse_axis("Log10Energy", "(2.0, 5.0, 3)"),
                            io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 2)"),
                            io::ic::parse_axis("Ra", "(0.0, 6.28319, 1)")});
  ASSERT_TRUE(io::ic::has_ra_axis(one_ra_bin));
  ASSERT_TRUE(io::ic::ra_bin_count(one_ra_bin) == 1);
  ASSERT_TRUE(io::ic::ra_bin_count(two_d) == 1);
  ASSERT_TRUE(one_ra_bin.n_axes() == 3);
  ASSERT_TRUE(one_ra_bin.total_bins() == 6);

  const Binning mc = io::ic::drop_ra_axis(one_ra_bin);
  ASSERT_TRUE(mc.n_axes() == 2);
  ASSERT_TRUE(mc.total_bins() == 6);
  ASSERT_TRUE(!io::ic::has_ra_axis(mc));
}

// A one-bin RA axis plus a galactic template is a valid configuration: the
// template is stored in the (3-axis) analysis binning and fits it. The old
// `ra_bins() == 1` test rejected it with a "has no Ra axis" message.
TEST(SampleConfigTest, ParsesOneRaBinGalactic) {
  const std::string json = R"JSON({
    "Binnings": {
      "b3d1": {
        "axes": "Log10Energy, CosZenith, Ra",
        "Log10Energy": "(2.0, 5.0, 3)",
        "CosZenith": "(-1.0, 1.0, 2)",
        "Ra": "(0.0, 6.28319, 1)"
      }
    },
    "Samples": {
      "s": {
        "binning": "b3d1",
        "parquet": "mc.parquet",
        "components": "astro",
        "Galactic": { "fermi": { "File": "fermi.txt", "Norm": "GalacticNorm0" } }
      }
    }
  })JSON";

  std::istringstream          in(json);
  boost::property_tree::ptree tree;
  boost::property_tree::read_json(in, tree);
  const auto samples = io::ic::parse_samples(tree);

  ASSERT_TRUE(samples.size() == 1);
  const io::ic::SampleConfig& s = samples[0];
  ASSERT_TRUE(io::ic::has_ra_axis(s.binning));
  ASSERT_TRUE(s.binning.n_axes() == 3);
  ASSERT_TRUE(s.mc_binning.n_axes() == 2);
  ASSERT_TRUE(s.ra_bins() == 1);
  ASSERT_TRUE(s.binning.total_bins() == s.mc_binning.total_bins());
  ASSERT_TRUE(s.galactic.size() == 1);
}

// mu is spread uniformly over RA (divisor n_ra), sigma^2 with divisor n_ra^2 --
// NNMFit Binning_2D_to_3D.make_binned_flux divides the repeated weights, so the
// square of those weights picks up the square of the divisor.
TEST(BinningTest, BroadcastOverRa) {
  const std::vector<double> mc_bins{10.0, 20.0, 30.0};

  std::vector<double> mu(9, -1.0);
  io::ic::broadcast_over_ra(mc_bins, 3, 3.0, mu);
  for (int b = 0; b < 3; ++b)
    for (int r = 0; r < 3; ++r)
      ASSERT_TRUE(mu[b * 3 + r] == mc_bins[b] / 3.0);

  std::vector<double> ssq(9, -1.0);
  io::ic::broadcast_over_ra(mc_bins, 3, 9.0, ssq);
  for (int b = 0; b < 3; ++b)
    for (int r = 0; r < 3; ++r)
      ASSERT_TRUE(ssq[b * 3 + r] == mc_bins[b] / 9.0);

  // n_ra == 1 with divisor 1.0 is an exact copy: x / 1.0 == x in IEEE-754, so a
  // 2-axis sample going through this path is bit-for-bit unchanged.
  std::vector<double> identity(3, -1.0);
  io::ic::broadcast_over_ra(mc_bins, 1, 1.0, identity);
  for (int b = 0; b < 3; ++b) ASSERT_TRUE(identity[b] == mc_bins[b]);

  auto throws = [](auto&& f) {
    try {
      f();
    } catch (const std::runtime_error&) {
      return true;
    }
    return false;
  };

  // A wrong-size out span must throw rather than write out of bounds.
  std::vector<double> wrong_size(8, -1.0);
  ASSERT_TRUE(throws([&] { io::ic::broadcast_over_ra(mc_bins, 3, 3.0, wrong_size); }));

  // n_ra <= 0 must throw with an actionable message, not underflow to a huge
  // size_t and produce a nonsense "expected 3 * 18446744073709551614".
  std::vector<double> negative_n_ra(3, -1.0);
  ASSERT_TRUE(throws([&] { io::ic::broadcast_over_ra(mc_bins, -1, 3.0, negative_n_ra); }));
  ASSERT_TRUE(throws([&] { io::ic::broadcast_over_ra(mc_bins, 0, 3.0, negative_n_ra); }));
}

// Data is binned truly 3D (NNMFit make_binned_data), so the counting helper needs
// a three-column overload.
TEST(BinningTest, DataHistogramCounts3d) {
  const Binning binning({io::ic::parse_axis("Log10Energy", "(2.0, 5.0, 3)"),
                         io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 1)"),
                         io::ic::parse_axis("Ra", "(0.0, 4.0, 2)")});
  ASSERT_TRUE(binning.total_bins() == 6);

  // Two events in RA bin 0 of energy bin 0, one in RA bin 1, one out of RA range.
  const std::vector<double> energies{316.0, 316.0, 316.0, 316.0};
  const std::vector<double> zeniths{2.0, 2.0, 2.0, 2.0};
  const std::vector<double> ras{0.5, 1.5, 3.0, 4.5};

  const std::vector<double> counts = io::ic::bin_event_counts(binning, energies, zeniths, ras);
  ASSERT_TRUE(counts.size() == 6);
  ASSERT_TRUE(counts[0] == 2.0);
  ASSERT_TRUE(counts[1] == 1.0);
  for (std::size_t b = 2; b < counts.size(); ++b) ASSERT_TRUE(counts[b] == 0.0);
}

// ---------------------------------------------------------------------------
// Likelihood parity with NNMFit
// ---------------------------------------------------------------------------

// Per-bin golden values dumped from NNMFit's own aesara graphs by
// tools/nnmfit_oracle/gen_llh_golden.py. Regenerate with:
//   /Users/soldin/Projects/IceCube/NNMFit/.venv/bin/python \
//       tools/nnmfit_oracle/gen_llh_golden.py -o programs/ictests/LLHGolden.inc
#include "LLHGolden.inc"

// Every bin-level likelihood term must reproduce NNMFit bit-for-bit up to
// double rounding: the SAY term, the plain Poisson term (the branch SAY falls
// back to at ssq == 0), and the saturated-subtracted Poisson term that
// NNMFit's LikelihoodBuilder actually minimises for llh == "PoissonLLH".
TEST(LikelihoodParityTest, BinTermsMatchNNMFitGolden) {
  using ana::ic::poisson_bin_log_likelihood;
  using ana::ic::poisson_bin_log_likelihood_saturated;
  using ana::ic::say_bin_log_likelihood;

  for (const LLHGoldenCase& c : kLLHGolden) {
    const double tolerance = 1e-12 * std::max(1.0, std::abs(c.say));
    EXPECT_NEAR(say_bin_log_likelihood(c.k, c.mu, c.ssq), c.say, tolerance)
        << "SAY mismatch at k=" << c.k << " mu=" << c.mu << " ssq=" << c.ssq;
    EXPECT_NEAR(poisson_bin_log_likelihood(c.k, c.mu), c.poisson,
                1e-12 * std::max(1.0, std::abs(c.poisson)))
        << "Poisson mismatch at k=" << c.k << " mu=" << c.mu;
    EXPECT_NEAR(poisson_bin_log_likelihood_saturated(c.k, c.mu), c.poisson_saturated,
                1e-12 * std::max(1.0, std::abs(c.poisson_saturated)))
        << "Saturated Poisson mismatch at k=" << c.k << " mu=" << c.mu;
  }
}

// The saturated subtraction is what makes the Poisson likelihood absolutely
// normalised: at the Asimov point every bin has mu == k, so each bin term is
// exactly zero and the total is zero without any baseline offset. This is the
// property the removed hardcoded baseline in ICLikelihood was faking, so guard
// it directly.
TEST(LikelihoodParityTest, PoissonAsimovIsExactlyZero) {
  using ana::ic::SampleLikelihood;
  using ana::ParameterWrapper;

  const Binning          binning = synthetic_binning();
  const io::ic::ICSample sample  = synthetic_sample(binning, /*with_atmospheric=*/true);

  const io::ic::SampleConfig cfg{.name       = "unit_test_sample",
                                 .binning    = binning,
                                 .mc_binning = binning,
                                 .components = {"astro", "conventional", "prompt"}};

  SampleLikelihood likelihood(sample, cfg, synthetic_settings(), /*gpu=*/nullptr, /*use_say=*/false);

  const std::vector<double> nominal_values = nominal_parameter_values();
  ParameterWrapper          nominal(params::ic::number_of_parameters());
  nominal.reset_parameter(nominal_values.data());

  likelihood.generate_asimov(nominal);

  EXPECT_NEAR(likelihood.partial_llh(nominal), 0.0, 1e-9);

  // ... and any move away from it costs something positive.
  std::vector<double> perturbed_values = nominal_values;
  perturbed_values[params::ic::AstroNorm] *= 1.5;
  ParameterWrapper perturbed(params::ic::number_of_parameters());
  perturbed.reset_parameter(perturbed_values.data());

  EXPECT_GT(likelihood.partial_llh(perturbed), 0.0);
}

// ---------------------------------------------------------------------------
// Randomized minimizer start values
// ---------------------------------------------------------------------------

// The multiplicative branch must reproduce NNMFit's draw, value * N(1, width),
// and the additive branch must actually move a parameter whose start value is
// 0 -- which is where NNMFit's purely multiplicative version gives up.
TEST(ParameterSeedingTest, DrawsMatchTheirBranches) {
  using ana::randomized_start_value;

  constexpr int    kDraws = 20000;
  constexpr double kValue = 1.5;
  constexpr double kStep  = 0.1;
  constexpr double kWidth = 0.08;

  std::mt19937_64 rng(12345);

  double sum = 0.0, sum_sq = 0.0;
  for (int i = 0; i < kDraws; ++i) {
    const double x = randomized_start_value(kValue, kStep, kWidth, rng);
    sum += x;
    sum_sq += x * x;
  }
  double mean = sum / kDraws;
  double sigma = std::sqrt(sum_sq / kDraws - mean * mean);

  // Multiplicative: mean == value, sigma == value * width. Tolerances are ~4
  // standard errors of the estimators at this sample size.
  EXPECT_NEAR(mean, kValue, 4.0 * kValue * kWidth / std::sqrt(kDraws));
  EXPECT_NEAR(sigma, kValue * kWidth, 0.05 * kValue * kWidth);

  // Additive fallback at zero: mean == 0, sigma == step. NNMFit returns
  // exactly 0 here, forever.
  sum = sum_sq = 0.0;
  for (int i = 0; i < kDraws; ++i) {
    const double x = randomized_start_value(0.0, kStep, kWidth, rng);
    sum += x;
    sum_sq += x * x;
  }
  mean = sum / kDraws;
  sigma = std::sqrt(sum_sq / kDraws - mean * mean);

  EXPECT_NEAR(mean, 0.0, 4.0 * kStep / std::sqrt(kDraws));
  EXPECT_NEAR(sigma, kStep, 0.05 * kStep);
}

// Same seed, same draws: a randomized start point must be reproducible from
// --seed alone (NNMFit's is not -- it uses numpy's global RNG).
TEST(ParameterSeedingTest, SameSeedSameDraw) {
  using ana::randomized_start_value;

  auto draw = [](const std::uint64_t seed) {
    std::mt19937_64     rng(seed);
    std::vector<double> out;
    for (int i = 0; i < 8; ++i) out.push_back(ana::randomized_start_value(1.5, 0.1, 0.08, rng));
    return out;
  };

  EXPECT_EQ(draw(4242), draw(4242));
  EXPECT_NE(draw(4242), draw(4243));
}

// The Asimov generation point must be separable from the minimiser start value
// (NNMFit's analysis.input_params vs its minimizer seeds), while a config that
// omits "AsimovValue" keeps generating Asimov at the start value.
TEST(InputParameterTest, AsimovValueDefaultsToStartValue) {
  static constexpr char kJson[] = R"JSON(
{
  "Parameter": [
    { "Name": "legacy",    "StartValue": 1.5, "StepWidth": 0.1, "Fixed": false, "Constrained": false },
    { "Name": "separated", "StartValue": 1.5, "StepWidth": 0.1, "AsimovValue": 2.25,
      "Fixed": false, "Constrained": false }
  ]
}
)JSON";

  boost::property_tree::ptree pt;
  std::istringstream          iss(kJson);
  boost::property_tree::read_json(iss, pt);

  const io::InputParameter parameters(pt.get_child("Parameter"));
  ASSERT_TRUE(parameters.size() == 2);

  EXPECT_EQ(parameters.parameters()[0].value(), 1.5);
  EXPECT_EQ(parameters.parameters()[0].asimov_value(), 1.5);

  EXPECT_EQ(parameters.parameters()[1].value(), 1.5);
  EXPECT_EQ(parameters.parameters()[1].asimov_value(), 2.25);
}

// Optional minimiser bounds ("LowerBound"/"UpperBound", NNMFit's per-parameter
// "range"). A config naming neither stays unbounded, which is what every config
// meant before bounds existed.
TEST(InputParameterTest, BoundsParseAndDefaults) {
  static constexpr char kJson[] = R"JSON(
{
  "Parameter": [
    { "Name": "free",  "StartValue": 1.0, "StepWidth": 0.1, "Fixed": false, "Constrained": false },
    { "Name": "both",  "StartValue": 0.5, "StepWidth": 0.1, "LowerBound": 0.0, "UpperBound": 5.0,
      "Fixed": false, "Constrained": false },
    { "Name": "lower", "StartValue": 1.0, "StepWidth": 0.1, "LowerBound": 0.0,
      "Fixed": false, "Constrained": false },
    { "Name": "upper", "StartValue": 1.0, "StepWidth": 0.1, "UpperBound": 3.0,
      "Fixed": false, "Constrained": false }
  ]
}
)JSON";

  boost::property_tree::ptree pt;
  std::istringstream          iss(kJson);
  boost::property_tree::read_json(iss, pt);

  const io::InputParameter parameters(pt.get_child("Parameter"));
  ASSERT_TRUE(parameters.size() == 4);
  const auto& p = parameters.parameters();

  EXPECT_FALSE(p[0].lower_bound().has_value());
  EXPECT_FALSE(p[0].upper_bound().has_value());

  ASSERT_TRUE(p[1].lower_bound().has_value());
  ASSERT_TRUE(p[1].upper_bound().has_value());
  EXPECT_EQ(*p[1].lower_bound(), 0.0);
  EXPECT_EQ(*p[1].upper_bound(), 5.0);

  EXPECT_TRUE(p[2].lower_bound().has_value());
  EXPECT_FALSE(p[2].upper_bound().has_value());

  EXPECT_FALSE(p[3].lower_bound().has_value());
  EXPECT_TRUE(p[3].upper_bound().has_value());
}

// A start value outside its own bounds, or an inverted range, is a config error.
// Clamping instead would start the fit somewhere the config never asked for --
// silently, which is the failure mode this whole comparison effort keeps hitting.
TEST(InputParameterTest, BoundsRejectBadConfigs) {
  auto parse = [](const char* json) {
    boost::property_tree::ptree pt;
    std::istringstream          iss(json);
    boost::property_tree::read_json(iss, pt);
    return io::InputParameter(pt.get_child("Parameter"));
  };

  EXPECT_THROW(parse(R"JSON({"Parameter": [
    { "Name": "below", "StartValue": -0.5, "StepWidth": 0.1, "LowerBound": 0.0,
      "Fixed": false, "Constrained": false }]})JSON"),
               std::invalid_argument);

  EXPECT_THROW(parse(R"JSON({"Parameter": [
    { "Name": "above", "StartValue": 9.0, "StepWidth": 0.1, "UpperBound": 3.0,
      "Fixed": false, "Constrained": false }]})JSON"),
               std::invalid_argument);

  EXPECT_THROW(parse(R"JSON({"Parameter": [
    { "Name": "inverted", "StartValue": 1.0, "StepWidth": 0.1, "LowerBound": 5.0,
      "UpperBound": 0.0, "Fixed": false, "Constrained": false }]})JSON"),
               std::invalid_argument);

  // On a bound exactly is allowed: NNMFit's own fitted PromptNorm sits at 0.0.
  EXPECT_NO_THROW(parse(R"JSON({"Parameter": [
    { "Name": "onedge", "StartValue": 0.0, "StepWidth": 0.1, "LowerBound": 0.0,
      "Fixed": false, "Constrained": false }]})JSON"));
}
