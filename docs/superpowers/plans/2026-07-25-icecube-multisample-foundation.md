# IceCube Multi-Sample Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the hardcoded single tracks-only IceCube likelihood into a config-driven, multi-sample composite with runtime N-dimensional binning, changing no physics.

**Architecture:** `ICLikelihood` becomes a composite that owns a `vector<SampleLikelihood>` (one per enabled sample) and sums their partial −2lnL, adding Gaussian pulls once. Each `SampleLikelihood` owns its `Sample`, its runtime `Binning`, its flux components, and its own `std::vector` histograms. A new `Binning` type replaces the `constexpr nBins`/`std::array` throughout. Samples, binnings and enable-flags come from the boost `property_tree` config.

**Tech Stack:** C++20/23, CMake, Eigen, Apache Arrow/Parquet, ROOT Minuit2, boost `property_tree`/`program_options`, OpenMP + Metal/CUDA GPU backends.

---

## Ground rules for this plan

- **This codebase has no unit-test framework.** Validation is integration-style: run `programs/LLHFit`, produce `Output.json`, compare with a recorded baseline via `tools/compare_output.py`. This plan adds one small **standalone unit-test executable** (plain `<cassert>` + exit code, registered with CTest) for the pure logic (`Binning`, CSR) — matching the repo's dependency-light style (it hand-rolls validation in bash/python rather than pulling gtest). Do **not** add gtest/catch2.
- **The golden gate.** Task 1 records an IceCube baseline `Output.json` from current `HEAD`. Every task after a refactor re-runs the single-sample config and must reproduce it. Use the **CPU backend** for the golden comparison (`"Backend": "cpu"`) so the check is bit-for-bit (`compare_output.py` with tolerance `0`). The Metal/CUDA FP32 path is checked separately with a tolerance.
- **Mechanical transforms.** Some tasks replace `BinArray`/`io::ic::Constants::nBins` across many files with a runtime buffer. Where a task is a mechanical pattern applied to code too large to paste verbatim, the step states the exact pattern, the exact signature change, and the exact file:symbol to change — follow it literally; do not invent new behavior. New self-contained units (the `Binning` type, config structs, `SampleLikelihood` interface, the composite loop) are given as full code.
- **Executable path:** the fit binary is `./build/programs/LLHFit/LLHFit` (a binary named `LLHFit` inside a directory named `LLHFit`). Use the full path in every run command.
- **Build command:** `cmake --build build -j8` (configure the `build/` dir first per `CLAUDE.md` if absent). **Never** commit `docs/superpowers/**` (spec/plan files stay local). Commit code only. No `Co-Authored-By: Claude` trailer in any commit message.

---

## File Structure

**New files:**
- `libraries/io/IceCube/Binning.h` / `Binning.cpp` — runtime N-dim analysis binning (axes, edges, `total_bins()`, `bin_index()`), parsed from config. Replaces the `constexpr` logic in `ICConstants.h`.
- `libraries/io/IceCube/SampleConfig.h` — POD describing one sample (name, enabled, binning ref, parquet/data paths, livetime, active components, per-sample `BranchNames`).
- `libraries/likelihood/IceCube/SampleLikelihood.h` / `SampleLikelihood.cpp` — one sample's prediction + partial −2lnL.
- `programs/ictests/ICTests.cpp` + `programs/ictests/CMakeLists.txt` — standalone unit-test executable, registered with CTest.

**Modified files:**
- `libraries/io/IceCube/ICConstants.h` — keep only genuinely global constants; bin logic moves to `Binning`.
- `libraries/io/IceCube/ICSample.h` — `sort_into_bins(int total_bins)` parameterized (no `Constants::nBins`).
- `libraries/io/IceCube/ICInputOptions.h` / `.cpp` — parse `Binnings` + `Samples`; expose `const std::vector<SampleConfig>&`.
- `libraries/io/IceCube/ICDataBase.h` / `.cpp` — load an enabled *set* of samples, each with its own `Binning`.
- `libraries/likelihood/IceCube/PowerlawFlux.h` / `.cpp`, `AtmosphericFlux.h` / `.cpp` — runtime-sized histograms from a `Binning`.
- `libraries/likelihood/IceCube/ICLikelihood.h` / `.cpp` — becomes the composite meta.
- `libraries/likelihood/IceCube/ICModule.cpp` — build enabled samples, hand them to the composite.
- `libraries/likelihood/IceCube/{MetalBackend.mm,CudaBackend.cpp,GpuBackend.h}` — `dispatch` group count from the sample's binning; verify no residual `Constants::nBins`.
- `libraries/io/IceCube/CMakeLists.txt`, `libraries/likelihood/IceCube/CMakeLists.txt`, `programs/CMakeLists.txt` — register new sources/targets.
- `configs/config_icecube.json` — migrate to `Binnings` + `Samples` (one sample == today).
- `tools/run_validation.sh` — add an IceCube golden-regression branch.

---

## Task 1: Record the IceCube golden baseline

**Files:**
- Create: `configs/config_icecube_tracks_cpu.json` (copy of current config, `"Backend": "cpu"`)
- Create (untracked): `Output.ic_baseline.json`

- [ ] **Step 1: Build current HEAD**

Run: `cmake --build build -j8`
Expected: builds `build/programs/LLHFit/LLHFit` with no errors.

- [ ] **Step 2: Make a CPU copy of the current IceCube config**

Copy `configs/config_icecube.json` to `configs/config_icecube_tracks_cpu.json` and set `"Backend": "cpu"` inside the `IceCube` object. Leave everything else identical. Point `TrackBaselineFilePath` at a parquet that exists locally (the test dataset).

- [ ] **Step 3: Generate and save the baseline**

Run:
```bash
./build/programs/LLHFit/LLHFit -c configs/config_icecube_tracks_cpu.json --silent
cp Output.json Output.ic_baseline.json
```
Expected: fit runs, prints `IC Asimov total events: ...`, writes `Output.json`. Record the printed Asimov total in the commit message for a human-visible anchor.

- [ ] **Step 4: Confirm the baseline compares equal to itself**

Run: `python3 tools/compare_output.py Output.ic_baseline.json Output.json`
Expected: exit 0, no diff lines.

- [ ] **Step 5: Commit**

```bash
git add configs/config_icecube_tracks_cpu.json
git commit -m "test: add CPU IceCube config for golden-regression baseline"
```
(Do **not** commit `Output.ic_baseline.json` — it is a local oracle, like `Output.baseline.json` for Double Chooz.)

---

## Task 2: `Binning` type + standalone unit-test executable

**Files:**
- Create: `libraries/io/IceCube/Binning.h`, `libraries/io/IceCube/Binning.cpp`
- Create: `programs/ictests/ICTests.cpp`, `programs/ictests/CMakeLists.txt`
- Modify: `libraries/io/IceCube/CMakeLists.txt`, `programs/CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

Create `programs/ictests/ICTests.cpp`:
```cpp
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
  // Same reference formula the old Constants::bin_index used.
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
```

- [ ] **Step 2: Create the `Binning` header**

Create `libraries/io/IceCube/Binning.h`:
```cpp
#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace io::ic {

  /** One analysis axis: a uniform grid in a derived reconstructed quantity. */
  struct Axis {
    enum class Kind { Log10Energy, CosZenith, Ra };
    Kind   kind;
    double lo;       // low edge in the axis' own units (log10 E, cos zenith, or radians)
    double hi;       // high edge
    int    n_bins;   // number of bins (edges = n_bins + 1)

    [[nodiscard]] double step() const noexcept { return (hi - lo) / n_bins; }
    /** Transform a raw event value (E in GeV, zenith in rad, RA in rad) to axis units. */
    [[nodiscard]] double project(double raw_value) const noexcept;
    /** Bin along this axis, or -1 if out of range. */
    [[nodiscard]] int index(double raw_value) const noexcept;
  };

  /** Parse "(lo, hi, n_bins)" for an axis of the named kind ("Log10Energy"|"CosZenith"|"Ra"). */
  [[nodiscard]] Axis parse_axis(std::string_view kind, std::string_view spec);

  /**
   * Runtime N-dimensional analysis binning. Row-major over the axis order:
   * flat = ((i0 * n1 + i1) * n2 + i2) ... . Replaces the old constexpr
   * Constants::nBins / bin_index for a single fixed grid.
   */
  class Binning {
   public:
    explicit Binning(std::vector<Axis> axes);

    [[nodiscard]] int total_bins() const noexcept { return m_TotalBins; }
    [[nodiscard]] std::size_t n_axes() const noexcept { return m_Axes.size(); }
    [[nodiscard]] std::span<const Axis> axes() const noexcept { return m_Axes; }

    /**
     * Flat bin index from one event's raw reco values, in axis order
     * (reco[0] for axis 0, ...). Returns -1 if the event is out of range on
     * any axis. `reco` must have at least n_axes() entries.
     */
    [[nodiscard]] int bin_index(std::span<const double> reco) const noexcept;

   private:
    std::vector<Axis> m_Axes;
    int               m_TotalBins;
  };

}  // namespace io::ic
```

- [ ] **Step 3: Implement `Binning`**

Create `libraries/io/IceCube/Binning.cpp`:
```cpp
#include "Binning.h"

#include <cmath>
#include <stdexcept>

namespace io::ic {

  double Axis::project(const double raw_value) const noexcept {
    switch (kind) {
      case Kind::Log10Energy: return std::log10(raw_value);
      case Kind::CosZenith:   return std::cos(raw_value);
      case Kind::Ra:          return raw_value;
    }
    return raw_value;
  }

  int Axis::index(const double raw_value) const noexcept {
    const double v = project(raw_value);
    if (v < lo || v >= hi) return -1;
    return static_cast<int>((v - lo) / step());
  }

  Axis parse_axis(const std::string_view kind, const std::string_view spec) {
    Axis::Kind k;
    if (kind == "Log10Energy")   k = Axis::Kind::Log10Energy;
    else if (kind == "CosZenith") k = Axis::Kind::CosZenith;
    else if (kind == "Ra")        k = Axis::Kind::Ra;
    else throw std::runtime_error("parse_axis: unknown axis kind '" + std::string(kind) + "'");

    // spec looks like "(lo, hi, n_bins)"
    std::string s(spec);
    for (char& c : s) if (c == '(' || c == ')' || c == ',') c = ' ';
    double lo = 0, hi = 0; int n = 0;
    if (std::sscanf(s.c_str(), "%lf %lf %d", &lo, &hi, &n) != 3 || n <= 0)
      throw std::runtime_error("parse_axis: bad spec '" + std::string(spec) + "' (want '(lo, hi, n_bins)')");
    return Axis{k, lo, hi, n};
  }

  Binning::Binning(std::vector<Axis> axes) : m_Axes(std::move(axes)) {
    if (m_Axes.empty()) throw std::runtime_error("Binning: needs at least one axis");
    m_TotalBins = 1;
    for (const Axis& a : m_Axes) m_TotalBins *= a.n_bins;
  }

  int Binning::bin_index(const std::span<const double> reco) const noexcept {
    int flat = 0;
    for (std::size_t d = 0; d < m_Axes.size(); ++d) {
      const int i = m_Axes[d].index(reco[d]);
      if (i < 0) return -1;
      flat = flat * m_Axes[d].n_bins + i;
    }
    return flat;
  }

}  // namespace io::ic
```

- [ ] **Step 4: Register the library source and the test target**

In `libraries/io/IceCube/CMakeLists.txt` add `Binning.cpp` to the IceCube io sources (follow the existing `target_sources`/source-list pattern in that file).

Create `programs/ictests/CMakeLists.txt`:
```cmake
add_executable(ICTests ICTests.cpp)
target_link_libraries(ICTests PRIVATE io)   # match the io target name used elsewhere
add_test(NAME ICTests COMMAND ICTests)
```
In `programs/CMakeLists.txt` add `add_subdirectory(ictests)` and ensure `enable_testing()` is active at the top-level `CMakeLists.txt` (add it if absent).

- [ ] **Step 5: Build and run the test — verify it passes**

Run:
```bash
cmake --build build -j8
ctest --test-dir build -R ICTests --output-on-failure
```
Expected: `ICTests: all passed`, CTest reports 1/1 passed.

- [ ] **Step 6: Commit**

```bash
git add libraries/io/IceCube/Binning.h libraries/io/IceCube/Binning.cpp \
        libraries/io/IceCube/CMakeLists.txt programs/ictests programs/CMakeLists.txt CMakeLists.txt
git commit -m "feat(icecube): add runtime N-dim Binning type with unit tests"
```

---

## Task 3: `SampleConfig` + parse `Binnings`/`Samples` from config

**Files:**
- Create: `libraries/io/IceCube/SampleConfig.h`
- Modify: `libraries/io/IceCube/ICInputOptions.h`, `libraries/io/IceCube/ICInputOptions.cpp`
- Modify: `programs/ictests/ICTests.cpp`

**This task is purely ADDITIVE and non-breaking.** It adds the `SampleConfig` type, a *tolerant* parser (does nothing if the config has no `Binnings`/`Samples`), a `samples()` getter, and a unit test. It does **not** migrate any live config and does **not** remove the old scalar keys/getters — the running fit path is untouched. Config migration + old-key removal happen atomically in Task 4 (loader switch), validated by the golden gate. Rationale: swapping the parser and the config in this task while the loader still reads `TrackBaselineFilePath` would leave the fit config inconsistent between tasks.

- [ ] **Step 1: Define `SampleConfig`**

Create `libraries/io/IceCube/SampleConfig.h`:
```cpp
#pragma once

#include "Binning.h"
#include "ICInputOptions.h"  // BranchNames

#include <string>
#include <vector>

namespace io::ic {

  /** One IceCube sample as described by the config's Samples list. */
  struct SampleConfig {
    std::string        name;
    bool               enabled = true;
    Binning            binning;                 // resolved from the named Binnings entry
    std::string        parquet;                 // MC baseline path
    std::string        data_path;               // real-data path ("" if Asimov)
    double             livetime = 1.0;
    std::vector<std::string> components;        // active flux components, e.g. {"astro","conventional","prompt"}
    BranchNames        branches;                // per-sample column names (defaults from BranchNames)

    [[nodiscard]] bool has_component(std::string_view c) const {
      for (const auto& x : components) if (x == c) return true;
      return false;
    }
  };

}  // namespace io::ic
```
Note: `Binning` has no default ctor, so give `SampleConfig` a constructor that takes the resolved `Binning`, or store `std::optional<Binning>` filled during parse. Simplest: add a `Binning` value and construct `SampleConfig` only after the binning is resolved (build a temporary via aggregate-with-move in the parser).

- [ ] **Step 2: Add the parser declaration**

In `libraries/io/IceCube/ICInputOptions.h`: add `#include "SampleConfig.h"` is **not** allowed (SampleConfig includes ICInputOptions for BranchNames — keep BranchNames in ICInputOptions.h and forward `struct SampleConfig;` here). Add:
```cpp
[[nodiscard]] const std::vector<SampleConfig>& samples() const noexcept { return m_Samples; }
```
and the member `std::vector<SampleConfig> m_Samples;`. Keep the existing scalar getters for now (they become unused after Task 6; remove them there).

- [ ] **Step 3: Parse `Binnings` + `Samples`**

Put the sample parsing in a free helper and call it **only when the `Samples` subtree exists**, so the old flat config keeps working until Task 4 migrates it. At the **end** of `read(...)` add:
```cpp
if (ic.get_child_optional("Samples"))
  m_Samples = parse_samples(ic);   // leaves m_Samples empty for the old flat config
```
where the free helper `static std::vector<SampleConfig> parse_samples(const boost::property_tree::ptree& ic)` contains:
```cpp
// --- Named binnings ---
std::map<std::string, Binning> binnings;
for (const auto& [bname, bnode] : ic.get_child("Binnings")) {
  std::vector<Axis> axes;
  std::stringstream axis_list(bnode.get<std::string>("axes"));
  std::string axis_kind;
  while (std::getline(axis_list, axis_kind, ',')) {
    // trim spaces
    axis_kind.erase(0, axis_kind.find_first_not_of(" \t"));
    axis_kind.erase(axis_kind.find_last_not_of(" \t") + 1);
    axes.push_back(parse_axis(axis_kind, bnode.get<std::string>(axis_kind)));
  }
  binnings.emplace(bname, Binning(std::move(axes)));
}

// --- Samples ---
std::vector<SampleConfig> out;
for (const auto& [sname, snode] : ic.get_child("Samples")) {
  const std::string binning_ref = snode.get<std::string>("binning");
  auto it = binnings.find(binning_ref);
  if (it == binnings.end())
    throw std::runtime_error("ICInputOptions: sample '" + sname + "' references unknown binning '" + binning_ref + "'");

  SampleConfig sc{.name = sname, .binning = it->second};
  sc.enabled   = snode.get<bool>("enabled", true);
  sc.parquet   = snode.get<std::string>("parquet");
  sc.data_path = snode.get<std::string>("data", "");
  sc.livetime  = snode.get<double>("livetime", 1.0);

  std::stringstream comp(snode.get<std::string>("components", ""));
  std::string c;
  while (std::getline(comp, c, ',')) {
    c.erase(0, c.find_first_not_of(" \t"));
    c.erase(c.find_last_not_of(" \t") + 1);
    if (!c.empty()) sc.components.push_back(c);
  }

  // per-sample Branches override the BranchNames defaults (reuse the existing block,
  // now reading from snode.get_child_optional("Branches") instead of the IceCube-level one)
  sc.branches = parse_branches(snode);   // factor the existing Branches block into a helper

  out.push_back(std::move(sc));
}
return out;
```
Factor the existing `IceCube.Branches` parsing (lines 53–74 of the current `ICInputOptions.cpp`) into a free helper `static BranchNames parse_branches(const boost::property_tree::ptree& node)` that reads `node.get_child_optional("Branches")`, and call it from `parse_samples` per sample. Add `#include <map>`, `<sstream>`, `"SampleConfig.h"`, `"Binning.h"` to `ICInputOptions.cpp`. **Do not touch the existing scalar/branch parsing or any config file in this task** — the old flat path stays intact.

- [ ] **Step 4: Add a parse test**

Append to `programs/ictests/ICTests.cpp` a test that constructs a `boost::property_tree::ptree` from a small in-memory JSON string with one binning + two samples (one `enabled:false`), runs the same parse logic, and asserts: 2 samples parsed, names/livetime correct, disabled flag read, `binning.total_bins() == 1485`, `components` split to 3 entries. (If the parse logic is private to `ICInputOptions::read`, expose a `static std::vector<SampleConfig> parse_samples(const ptree&)` free function used by both `read` and the test.)

- [ ] **Step 5: Build, run tests, verify**

Run:
```bash
cmake --build build -j8
ctest --test-dir build -R ICTests --output-on-failure
```
Expected: all asserts pass. (No fit run in this task — the parser is additive and not yet on the running path.)

- [ ] **Step 6: Commit**

```bash
git add libraries/io/IceCube/SampleConfig.h libraries/io/IceCube/ICInputOptions.h \
        libraries/io/IceCube/ICInputOptions.cpp programs/ictests/ICTests.cpp
git commit -m "feat(icecube): parse Binnings + Samples config into SampleConfig list"
```

---

## Task 4: Multi-sample loader (`ICDataBase` + parameterized `sort_into_bins`)

**Files:**
- Modify: `libraries/io/IceCube/ICSample.h`
- Modify: `libraries/io/IceCube/ICDataBase.h`, `libraries/io/IceCube/ICDataBase.cpp`
- Modify: `programs/ictests/ICTests.cpp`

- [ ] **Step 1: Parameterize `sort_into_bins`**

In `ICSample.h`, change `void sort_into_bins()` to `void sort_into_bins(int total_bins)` and replace the two uses of `Constants::nBins` inside it with the `total_bins` argument. Remove the `#include "ICConstants.h"` if it becomes unused (keep `ICParameter.h`).

- [ ] **Step 2: CSR invariant test for a non-tracks bin count**

In `ICTests.cpp`, add a test that fills a tiny `ICSample` (say 6 events) with hand-set `bin_idx` values against `total_bins = 4` (including one `-1`), calls `sort_into_bins(4)`, and asserts: out-of-range event dropped, `bin_offsets.size() == 5`, offsets monotonic non-decreasing, `bin_offsets.back() == size()`, and events grouped by bin.

- [ ] **Step 3: Make `ICDataBase` load a set of samples**

In `ICDataBase.h`, replace the single `ICSample m_Sample` with:
```cpp
[[nodiscard]] const ICSample& sample(std::size_t i) const noexcept { return m_Samples[i]; }
[[nodiscard]] std::size_t     n_samples() const noexcept { return m_Samples.size(); }
```
and members `std::vector<ICSample> m_Samples;`. Change the constructor to `explicit ICDataBase(const std::vector<SampleConfig>& samples);` and replace `read_track_baseline` with `arrow::Status read_sample(const SampleConfig& cfg, ICSample& out);`.

In `ICDataBase.cpp`: for each **enabled** `SampleConfig`, read its `cfg.parquet` using `cfg.branches` (identical column logic to the current `read_track_baseline`), scale by `cfg.livetime`, assign bins with `cfg.binning.bin_index({e_reco[i], reco_zenith[i]})` (build a 2-element array per event; extend to 3 when RA arrives in Phase 3), then `out.sort_into_bins(cfg.binning.total_bins())`. Push each populated `ICSample` into `m_Samples` **in the same order** as the enabled samples in the config (the composite relies on this order). Keep the existing per-event `get_double_column` helper and Arrow includes.

- [ ] **Step 4: Migrate the configs to the new `Binnings`/`Samples` layout**

This is the atomic switch: the loader now reads `samples()`, so the live configs must provide them. Rewrite the `IceCube` block of **both** `configs/config_icecube.json` and `configs/config_icecube_tracks_cpu.json` to:
```json
"IceCube": {
  "Backend": "cpu",
  "Likelihood": "Poisson",
  "UseData": false,
  "ERefGeV": 100000.0,
  "AstroReferenceIndex": 2.0,
  "AstroPerTypeNorm": false,
  "ConvDeltaGammaERef": 1000.0,
  "PromptDeltaGammaERef": 3800.0,
  "Binnings": {
    "tracks_2d": {
      "axes": "Log10Energy, CosZenith",
      "Log10Energy": "(2.5, 7.0, 45)",
      "CosZenith":   "(-1.0, 0.0872, 33)"
    }
  },
  "Samples": {
    "tracks": {
      "enabled": true,
      "binning": "tracks_2d",
      "parquet": "<the old TrackBaselineFilePath value>",
      "livetime": 3.0e8,
      "components": "astro, conventional, prompt"
    }
  }
}
```
Use `"Backend": "cpu"` in `config_icecube_tracks_cpu.json` (the golden config) and `"Backend": "metal"` in `config_icecube.json`. Fill `parquet` with the exact old `TrackBaselineFilePath` value from each file. Keep the top-level `Parameter` array unchanged. The old scalar keys (`TrackBaselineFilePath`, top-level `Livetime`) are removed — they are no longer read on the sample path (the old required parse in `read(...)` is now behind the "old flat config" branch; since these configs now HAVE `Samples`, remove those old required `.get` calls for `TrackBaselineFilePath`/`Livetime` from `read(...)` in this task so migrated configs don't trip them). **This is the task where the golden gate proves the migration is behavior-preserving** (Step 7).

- [ ] **Step 5: Update the caller signature (compile-only for now)**

In `ICModule.cpp`, change the `ICDataBase` construction to pass `m_InputOptions->samples()`. (The likelihood wiring is finished in Task 7; for now just make it compile — `ICLikelihood` still reads `sample(0)` via a temporary shim, see Step 6.)

- [ ] **Step 6: Temporary single-sample shim so the build stays green**

Until Task 7 replaces it, have `ICLikelihood` take `data_base.sample(0)` and `input_options.samples()[0].binning` — a one-line shim so the project builds and the golden config (one enabled sample) still runs. Mark it with `// TODO(Task 7): composite`.

- [ ] **Step 7: Build, run unit tests + golden regression**

Run:
```bash
cmake --build build -j8
ctest --test-dir build -R ICTests --output-on-failure
./build/programs/LLHFit/LLHFit -c configs/config_icecube_tracks_cpu.json --silent
python3 tools/compare_output.py Output.ic_baseline.json Output.json
```
Expected: unit tests pass; `compare_output.py` exits 0 (bit-for-bit). If it differs, the loader or the config migration changed event→bin assignment or a physics constant — fix before continuing. This step proves the new multi-sample loader + migrated config reproduce the original tracks-only output exactly.

- [ ] **Step 8: Commit**

```bash
git add libraries/io/IceCube/ICSample.h libraries/io/IceCube/ICDataBase.h \
        libraries/io/IceCube/ICDataBase.cpp libraries/io/IceCube/ICInputOptions.cpp \
        libraries/likelihood/IceCube/ICModule.cpp libraries/likelihood/IceCube/ICLikelihood.cpp \
        configs/config_icecube.json configs/config_icecube_tracks_cpu.json programs/ictests/ICTests.cpp
git commit -m "feat(icecube): load an enabled set of samples, each with its own binning"
```

---

## Task 5: Runtime-size the flux components

**Files:**
- Modify: `libraries/likelihood/IceCube/PowerlawFlux.h`, `PowerlawFlux.cpp`
- Modify: `libraries/likelihood/IceCube/AtmosphericFlux.h`, `AtmosphericFlux.cpp`
- Modify: `libraries/likelihood/IceCube/{GpuBackend.h,MetalBackend.mm,CudaBackend.cpp}`

- [ ] **Step 1: Replace the fixed histogram with a runtime vector — PowerlawFlux**

In `PowerlawFlux.h`: delete `using BinArray = std::array<double, io::ic::Constants::nBins>;` and change `BinArray m_Histogram{};` to `std::vector<double> m_Histogram;`. Add a `Binning` reference member and take it in the constructor:
```cpp
PowerlawFlux(const io::ic::ICSample&     sample,
             const io::ic::Binning&      binning,
             double e_ref_gev, double reference_index, bool per_type_norm,
             std::shared_ptr<GpuBackend> gpu = nullptr, bool need_per_event = false);
```
In `PowerlawFlux.cpp`: size `m_Histogram.assign(binning.total_bins(), 0.0);` in the constructor, and replace every `io::ic::Constants::nBins` in the recalc/dispatch with `static_cast<int>(m_Histogram.size())` (or a stored `m_TotalBins`). The GPU `dispatch` group count becomes `m_Histogram.size()`.

- [ ] **Step 2: Same transform — AtmosphericFlux**

Apply the identical change to `AtmosphericFlux.h/.cpp` (drop `BinArray`, `std::vector<double> m_Histogram;`, take `const Binning&`, size in ctor, replace `Constants::nBins`). The Barr-gradient GPU handles are unchanged.

- [ ] **Step 3: GPU dispatch count from the histogram size**

In `GpuBackend.h` the `dispatch` contract already says "dispatch nBins groups" — change its documented/used group count to come from the caller (the flux component passes its `m_Histogram.size()`). Verify `MetalBackend.mm` and `CudaBackend.cpp` take the group count as an argument and contain **no** `io::ic::Constants::nBins`. Grep to confirm:
```bash
grep -rn "Constants::nBins" libraries/likelihood/IceCube
```
Expected after this task: no matches in the flux/GPU files (only possibly in the soon-to-be-refactored `ICLikelihood`).

- [ ] **Step 4: Update the ICLikelihood shim to pass the binning**

Pass `input_options.samples()[0].binning` into `m_Astro` and `m_Atmo` construction in `ICLikelihood.cpp` (still the Task-4 shim; composite comes next).

- [ ] **Step 5: Build + golden regression (CPU, bit-for-bit)**

Run:
```bash
cmake --build build -j8
./build/programs/LLHFit/LLHFit -c configs/config_icecube_tracks_cpu.json --silent
python3 tools/compare_output.py Output.ic_baseline.json Output.json
```
Expected: exit 0. The runtime-sized path must reproduce the fixed-array result exactly on CPU.

- [ ] **Step 6: GPU parity check (if a device is present)**

Run the same fit with `"Backend": "metal"` (or `cuda`) and compare with a tolerance:
```bash
./build/programs/LLHFit/LLHFit -c configs/config_icecube.json --silent
python3 tools/compare_output.py Output.ic_baseline.json Output.json 1e-6
```
Expected: exit 0 within FP32 tolerance. (Skip if no GPU.)

- [ ] **Step 7: Commit**

```bash
git add libraries/likelihood/IceCube/PowerlawFlux.h libraries/likelihood/IceCube/PowerlawFlux.cpp \
        libraries/likelihood/IceCube/AtmosphericFlux.h libraries/likelihood/IceCube/AtmosphericFlux.cpp \
        libraries/likelihood/IceCube/GpuBackend.h libraries/likelihood/IceCube/MetalBackend.mm \
        libraries/likelihood/IceCube/CudaBackend.cpp libraries/likelihood/IceCube/ICLikelihood.cpp
git commit -m "refactor(icecube): runtime-size flux histograms from Binning (no Constants::nBins)"
```

---

## Task 6: Extract `SampleLikelihood`

**Files:**
- Create: `libraries/likelihood/IceCube/SampleLikelihood.h`, `SampleLikelihood.cpp`
- Modify: `libraries/likelihood/IceCube/CMakeLists.txt`
- Modify: `programs/ictests/ICTests.cpp`

- [ ] **Step 1: Define `SampleLikelihood`**

Create `libraries/likelihood/IceCube/SampleLikelihood.h`:
```cpp
#pragma once

#include "../../io/IceCube/Binning.h"
#include "../../io/IceCube/ICSample.h"
#include "../../io/IceCube/SampleConfig.h"
#include "../ParameterWrapper.h"
#include "AtmosphericFlux.h"
#include "GpuBackend.h"
#include "PowerlawFlux.h"

#include <memory>
#include <span>
#include <vector>

namespace ana::ic {

  /**
   * One IceCube sample's prediction and its partial −2lnL contribution. Owns its
   * flux components and its own runtime-sized histograms; knows nothing about
   * other samples. The meta ICLikelihood sums partial_llh() across samples and
   * adds the Gaussian pulls once.
   */
  class SampleLikelihood {
   public:
    SampleLikelihood(const io::ic::ICSample&      sample,
                     const io::ic::SampleConfig&  cfg,
                     const GlobalFluxSettings&    settings,   // e_ref, reference index, delta-gamma e_refs, per_type_norm
                     std::shared_ptr<GpuBackend>  gpu,
                     bool                         use_say);

    /** Recompute prediction for the current parameters and return this sample's −2lnL (no pulls). */
    [[nodiscard]] double partial_llh(const ParameterWrapper& parameter);

    /** Fill this sample's Asimov data from the nominal parameters. */
    void generate_asimov(const ParameterWrapper& nominal);

    [[nodiscard]] std::span<const double> predicted() const noexcept { return m_Predicted; }
    [[nodiscard]] std::span<const double> data() const noexcept { return m_Data; }

   private:
    const io::ic::ICSample& m_Sample;
    bool                    m_UseSAY;
    PowerlawFlux            m_Astro;
    AtmosphericFlux         m_Atmo;
    std::vector<double>     m_Predicted;
    std::vector<double>     m_Data;
    std::vector<double>     m_Ssq;

    bool assemble_prediction(const ParameterWrapper& parameter);
    void assemble_fluctuation();
  };

}  // namespace ana::ic
```
Note: `assemble_prediction` takes the `ParameterWrapper` because `SampleLikelihood` — unlike `ICLikelihood` — does not own one; the caller passes it into `partial_llh`/`generate_asimov`, and it is needed for the fluxes' `check_and_recalculate(parameter)`. `assemble_fluctuation()` stays parameterless (it only re-sums already-recalculated per-event weights).
Add a small `struct GlobalFluxSettings { double e_ref_gev, astro_reference_index, conv_delta_gamma_e_ref, prompt_delta_gamma_e_ref; bool astro_per_type_norm; };` (put it in `SampleLikelihood.h` or a shared header) built once from `ICInputOptions`.

- [ ] **Step 2: Implement `SampleLikelihood` (ADDITIVE — do not break `ICLikelihood`)**

Create `SampleLikelihood.cpp` by **copying and adapting** `assemble_prediction`, `assemble_fluctuation`, `generate_asimov_data`, and the SAY/Poisson term from the current `ICLikelihood.cpp`. **Do NOT remove them from `ICLikelihood.cpp` in this task** — `ICLikelihood` must keep building and running via its Task-4/5 shim. This creates temporary duplication (the assemble logic exists in both `ICLikelihood` and `SampleLikelihood`); Task 7 deletes the copies in `ICLikelihood` when it switches to the composite. Rationale: moving the logic out here would break `ICLikelihood`'s compile until Task 7.

Adaptations in the `SampleLikelihood` copy:
- size `m_Predicted/m_Data/m_Ssq` to `cfg.binning.total_bins()` in the ctor (never resize per-eval);
- loop bounds use `m_Predicted.size()` not `Constants::nBins`;
- `partial_llh` returns `calculate_say_likelihood(m_Data, m_Predicted, m_Ssq)` (or Poisson) **without** pulls;
- keep the "sum astro+atmo per event before squaring" ssq rule (current `ICLikelihood.cpp:106-124`);
- build `m_Astro`/`m_Atmo` from `settings` + `cfg.binning`.
Give `SampleLikelihood.cpp` its own copies of the free functions `calculate_poisson_likelihood`/`calculate_say_likelihood`/`square` (file-local `static`, or a small shared header). It is fine that `ICLikelihood.cpp` still has its own copies until Task 7 — Task 7 resolves the duplication.

- [ ] **Step 3: Register the source**

Add `SampleLikelihood.cpp` to `libraries/likelihood/IceCube/CMakeLists.txt` (follow the existing source-list pattern).

- [ ] **Step 4: Unit test — Asimov is the minimum**

In `ICTests.cpp`, add a test that builds a `SampleLikelihood` (CPU path, `gpu = nullptr`) over a tiny synthetic `ICSample` (a handful of events with all per-event columns populated — `e_true`, `astro_baseline`, `conv_baseline`, `conv_alt`, `prompt_baseline`, `prompt_alt`, `barr_conv[0..3]` — and a small `Binning`, then `sort_into_bins`). Call `generate_asimov(nominal)`, then assert **`partial_llh(nominal) < partial_llh(perturbed)`** where `perturbed` scales one norm (e.g. AstroNorm ×1.5) — i.e. the Asimov point is the minimum. Do NOT assert `≈ 0`: this codebase's `calculate_poisson_likelihood` does not subtract the saturated term, so the value at the minimum is a non-zero constant. Also assert `partial_llh(nominal)` is finite.

Note: `SampleLikelihood`, `PowerlawFlux`, `AtmosphericFlux` live in the heavy `icecube` library (ROOT/Arrow/GPU), while `ICTests` currently links only `io`. Add `icecube` to `ICTests`' `target_link_libraries` in `programs/ictests/CMakeLists.txt` so the test can construct a `SampleLikelihood`. Constructing a `ParameterWrapper` for `nominal`/`perturbed` may need the parameter count — use `params::ic::number_of_parameters()`.

- [ ] **Step 5: Build + tests**

Run:
```bash
cmake --build build -j8
ctest --test-dir build -R ICTests --output-on-failure
```
Expected: pass. (The full-fit golden check is exercised in Task 7 once the composite calls this.)

- [ ] **Step 6: Commit**

```bash
git add libraries/likelihood/IceCube/SampleLikelihood.h libraries/likelihood/IceCube/SampleLikelihood.cpp \
        libraries/likelihood/IceCube/CMakeLists.txt programs/ictests/CMakeLists.txt programs/ictests/ICTests.cpp
git commit -m "feat(icecube): add SampleLikelihood (per-sample prediction + partial LLH)"
```
(Do NOT stage `ICLikelihood.cpp` in this task — it is unchanged here.)

---

## Task 7: `ICLikelihood` becomes the composite; wire enable/disable; full validation

**Files:**
- Modify: `libraries/likelihood/IceCube/ICLikelihood.h`, `ICLikelihood.cpp`
- Modify: `libraries/likelihood/IceCube/ICModule.cpp`
- Modify: `libraries/io/IceCube/ICInputOptions.h`/`.cpp` (drop now-unused scalar getters)
- Modify: `tools/run_validation.sh`
- Create: `configs/config_icecube_tracksonly.json` (anchor)

- [ ] **Step 1: Rewrite `ICLikelihood` as the meta**

In `ICLikelihood.h`, replace the `m_Astro`/`m_Atmo`/`m_TotalPredicted`/`m_Data`/`m_Ssq` members with:
```cpp
std::shared_ptr<const io::ic::ICDataBase>          m_DataBase;
std::shared_ptr<GpuBackend>                        m_GpuBackend;
std::vector<std::unique_ptr<SampleLikelihood>>     m_Samples;
std::vector<std::tuple<int, double, double>>       m_Pulls;
```
Keep `predicted()`/`data()` but have them delegate to sample 0 (or drop them — check `ICWriteResults.h` usage first and adapt).

- [ ] **Step 2: Implement the composite**

In `ICLikelihood.cpp`:
```cpp
double ICLikelihood::calculate_likelihood(const double* parameter) {
  m_Parameter.reset_parameter(parameter);
  double llh = 0.0;
  for (auto& s : m_Samples)
    llh += s->partial_llh(m_Parameter);
  llh += calculate_pulls(m_Parameter);   // pulls added ONCE at the meta level
  return std::isfinite(llh) ? llh : 1.0e25;
}
```
Constructor: build `m_GpuBackend` once (existing `make_gpu_backend`), build `GlobalFluxSettings` from `input_options`, then for each enabled sample construct a `SampleLikelihood` (in the same order `ICDataBase` loaded them) sharing `m_GpuBackend`. Call `setup_pulls()` (unchanged) then each sample's `generate_asimov(nominal)`. Keep `initialize_data`/`use_data` throwing "not implemented" as today.

- [ ] **Step 3: Wire the module**

`ICModule.cpp`: `m_DataBase = std::make_shared<const io::ic::ICDataBase>(m_InputOptions->samples());` then `std::make_shared<ICLikelihood>(options, m_DataBase, *m_InputOptions)`. Remove the Task-4/5 shim.

- [ ] **Step 4: Golden regression through the composite (one enabled sample)**

Run:
```bash
cmake --build build -j8
./build/programs/LLHFit/LLHFit -c configs/config_icecube_tracks_cpu.json --silent
python3 tools/compare_output.py Output.ic_baseline.json Output.json
```
Expected: exit 0. The composite with a single enabled sample must reproduce the original tracks-only output bit-for-bit. **This is the Phase 1 hard gate.**

- [ ] **Step 5: Composite summation test (two copies of tracks)**

Create a temp config with the same sample listed twice (different names, same parquet/binning), all params fixed. Run it and confirm the printed total −2lnL is exactly twice the single-sample value at the same parameters, and that disabling one of the two returns the single value. (A quick way: run `LLHFit` on a two-sample vs one-sample config with all parameters `Fixed`, read the `####` printed value / `Output.json` LLH field, assert the 2× relation and that pulls did not double.) Document the observed numbers in the commit message.

- [ ] **Step 6: Enable/disable anchor config**

Create `configs/config_icecube_tracksonly.json` == the combined layout but with only `tracks` `enabled:true` (this is identical to the CPU golden here; in Phase 2 it becomes the tracks anchor of the 3-sample config). Run it and confirm it matches the baseline. Add an IceCube branch to `tools/run_validation.sh` mirroring the Double Chooz branch: if `Output.ic_baseline.json` exists, run `LLHFit -c configs/config_icecube_tracks_cpu.json --silent` and `compare_output.py`, else `skip`.

- [ ] **Step 7: Full validation suite**

Run: `tools/run_validation.sh --no-build`
Expected: Double Chooz + LinearRegression + IceCube branches all `ok` (or `skip` where inputs are absent), no `FAILED`.

- [ ] **Step 8: Commit**

```bash
git add libraries/likelihood/IceCube/ICLikelihood.h libraries/likelihood/IceCube/ICLikelihood.cpp \
        libraries/likelihood/IceCube/ICModule.cpp libraries/io/IceCube/ICInputOptions.h \
        libraries/io/IceCube/ICInputOptions.cpp tools/run_validation.sh \
        configs/config_icecube_tracksonly.json
git commit -m "feat(icecube): composite multi-sample likelihood with per-sample enable/disable"
```

---

## Self-review notes (traceability to the spec)

- **Composite meta + pulls once** → Task 6 (`SampleLikelihood.partial_llh` excludes pulls) + Task 7 Step 2.
- **Runtime N-dim binning, 3D-ready** → Task 2 (`Axis`/`Binning`, RA `Kind` present but unused until Phase 3).
- **Compile-time `params::ic` enum kept** → untouched; only wiring moves. Confirmed no task edits the enum.
- **Config-driven samples + named binnings + enable flags** → Task 3 (parse) + Task 7 (enable/disable, anchor config).
- **Per-sample branches/livetime/binning** → Task 3 (`SampleConfig`) + Task 4 (loader).
- **Golden gate: reproduce tracks-only exactly** → Tasks 1, 4, 5, 7 all re-run `compare_output.py` against `Output.ic_baseline.json` (CPU bit-for-bit; GPU with tolerance in Task 5 Step 6).
- **No allocation in hot loop** → histograms sized once in each component/`SampleLikelihood` ctor, overwritten per eval; verify no per-eval `assign`/`resize` in `partial_llh`.
- **Out of scope (Phases 2–3)**: veto components, MuonGun template, det-sys per-sample gradient, RA activation, galactic flux — none appear as tasks here. Correct per the spec.

---

## Phase 1 outcome + carry-over findings for Phase 2

**Status: Phase 1 COMPLETE.** All 7 tasks landed (`f864d03`, `7a42ae8`, `59d106e`, `374cb5b`, `d0372f5`, `84c86f2`, `293a594` + follow-up fixes). Golden gate reproduces the pre-refactor tracks-only fit bit-for-bit (Asimov 514973, LLH −6366527.142871824). Final code-quality review empirically verified, by config alone, that (a) a middle-disabled sample keeps the lockstep pairing correct and (b) a second sample with a *different* binning (1485 + 99 bins) loads and folds into the composite. Foundation confirmed ready.

**Fixed during Phase 1 close-out (beyond the original task list):**
- `SampleLikelihood::generate_asimov` now seeds `m_Ssq` when SAY is active. Previously the minimizer's first evaluation (start values == nominal → nothing recalculated) ran SAY with `ssq == 0`, silently degenerating to Poisson. Pre-existing bug inherited from the old `ICLikelihood`; the golden config is Poisson so the baseline is unaffected.
- `ICLikelihood` ctor now asserts the enabled-config count equals `ICDataBase::n_samples()` before indexing the unchecked `noexcept` `sample(i)`, so a future divergence between the two "enabled" filters fails loudly instead of being silent UB.

**Carry-over — do these FIRST in Phase 2 (from the final review):**
1. **Results writer is the last single-sample assumption, and it fails silently.** `libraries/results/IceCube/ICWriteResults.h:39-50` reads only `llh.data()`/`llh.predicted()` (which delegate to `m_Samples.front()`) and writes `io::ic::Constants::nEnergyBins`/`nZenithBins`. A two-sample fit today writes a plausible-looking JSON describing only sample 0 with a hardcoded 45×33 shape — **which also means the golden-regression harness is blind to a second sample.** Make the writer per-sample (emit an array of sample blocks with each sample's own binning) before trusting any multi-sample fit output. `ICLikelihood::predicted()/data()` are correspondingly mis-named for a composite.
2. **`SampleConfig::components` is parsed but inert.** `SampleLikelihood` builds `PowerlawFlux` + `AtmosphericFlux` unconditionally and `ICDataBase::read_sample` unconditionally reads conv/prompt/Barr columns, regardless of `components`. `has_component()` has zero callers. A Phase-2 cascade sample declaring `"components": "astro"` will still build the atmospheric flux and still require atmospheric/Barr columns in its parquet. Gate flux construction and column reads on `cfg.has_component(...)` — this is exactly the per-sample component masking Phase 2 needs (tracks excludes `muon, conventional_veto, prompt_veto`; cscd excludes `muontemplate, conventional, prompt`).
3. **No test covers the composite pairing** — the riskiest line in the refactor. Add an `ICTests` case for enabled/disabled sample pairing.
4. **Dead legacy surface to delete:** `ICInputOptions::{track_baseline_file_path,branch_names,livetime}()` + their members and the now-duplicated `IceCube.Branches` block (`ICInputOptions.cpp:55-79`, duplicated by `parse_branches` in `SampleConfig.cpp`); `Constants::bin_index()` (zero callers — delete so Phase 2 reaches for `Binning`); `configs/config_icecube_tracksonly.json` (byte-identical orphan of `config_icecube_tracks_cpu.json`, which is what `run_validation.sh` actually uses); duplicated `square` in `ICLikelihood.cpp` + `SampleLikelihood.cpp`.
5. **Stale comments that will mislead:** `SampleLikelihood.h:28,58-63`, `SAYLikelihood.cpp:26`, `PowerlawFlux.h:46`, `ICSample.h:14,44`, `ICInputOptions.h:25`, `ICModule.h:13`, `likelihood/IceCube/CMakeLists.txt:1`, `ICParameter.h:6` — several still describe a "tracks-only"/single-sample module or reference `ICLikelihood::assemble_prediction`, which no longer exists.

**Unrelated pre-existing issue (not caused by this refactor):** `tools/run_validation.sh` reports the Double Chooz branch as FAILED (small numeric drift in `FDI.accidental[...]` vs a Jul-23 `Output.baseline.json`). Verified that **zero** DoubleChooz files and zero shared-core files (`Fit`, `Likelihood`, `ParameterWrapper`, `write_results`, `Options`, `InputOptions`) changed across `eae04d3..HEAD`. Either the DC baseline is stale or it drifted in the pre-refactor WIP snapshot — needs the repo owner's call, not a Phase-2 task.
