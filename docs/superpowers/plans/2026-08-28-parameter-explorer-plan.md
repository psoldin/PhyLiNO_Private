# Interactive Parameter Explorer — Implementation Plan

Date: 2026-08-28
Spec: [2026-08-28-parameter-explorer-design.md](../specs/2026-08-28-parameter-explorer-design.md)

Verified against the tree at `8359c1c`. Every API named below was read, not
assumed. Two spec corrections landed from that pass: `ExplorerModel` uses
`Likelihood::parameter()` rather than owning a `ParameterWrapper`, and the
excluded atmospheric key is spelled `atmospheric` or `atmospheric_veto`
depending on `SampleConfig::wants_veto()`.

## Phase 0 — Measure one evaluation — DONE

Not a throwaway in the end. The measurement needs the same construction the
model needs, so it became `programs/ParamExplorer/Bench.cpp` against the
finished `ExplorerModel`, and it stays in the tree: the numbers below will move
when the likelihood does, and re-measuring should not mean rewriting the harness.

`io::Options` is built the way `LLHFit.C:1008` builds it — it takes `argc/argv`,
so a synthetic `{"PhyLiNOExplorer", "-c", <path>}` feeds the `("config,c", ...)`
option at `libraries/io/InputOptions.cpp:48`.

It sweeps every parameter rather than timing one: the spread across them turned
out to be four orders of magnitude, and one number would have hidden it.

Result: worst slider 237 ms, **Design B**, and the atmospheric split moved
behind a checkbox. See the spec's Latency section for the table and the
reasoning.

```bash
./build/bin/PhyLiNOExplorerBench configs/config_icecube_combined.json 10
```

## Phase 1 — `ExplorerModel`, no Qt

`programs/ParamExplorer/ExplorerModel.h` / `.cpp`.

Construction:

```cpp
ExplorerModel(const std::string& config_path);
```

- Build the synthetic argv, `io::Options`, `ana::ic::ICExperimentModule`.
- `create_likelihood(options, 0)`, `dynamic_pointer_cast<ana::ic::ICLikelihood>`;
  throw if null.
- Seed `m_Params` from `options->inputOptions().input_parameters().value(i)` for
  `i` in `[0, params::ic::number_of_parameters())` — 23, statically asserted in
  `libraries/io/IceCube/ICParameter.h`.

`parameters()` returns one `ParamInfo` per index, built from `InputParameter`:
`name(i)`, `value(i)`, `fixed(i)`, `lower_bound()`, `upper_bound()`. The bounds
are `std::optional`; apply the spec's fallbacks. `parameters()` returns all 23
with the `fixed` flag set — the GUI decides what to show, the model does not
filter.

`set(i, v)` writes `m_Params[i]`. `evaluate()` returns
`m_Llh->calculate_likelihood(m_Params.data())`.

`sample_names()` / `axes(sample)` walk `m_Llh->sample(i).config()`:
`.name` and `.binning.axes()`, with `io::ic::axis_kind_name(kind)` for the
label.

`marginalize(sample, axis)`:

- `const auto& s = m_Llh->sample(i);`
- components from `breakdown_for(s, m_Llh->parameter(), split)`: the split path
  is `result::ic::component_breakdown` itself, the default path reads the same
  accessors for the four entries it keeps and never triggers the per-event
  conv/prompt walk. Empty vectors are skipped; `systematicsDelta` is in neither
  stack.
- data from `s.data()`.
- Project each with the stride walk: `stride = product of n_bins after axis k`,
  `out[(flat / stride) % n_k] += in[flat]`.
- Edges from the `Axis`: `uniform()` → `lo + i * step()`, else `edges`.

Everything the breakdown returns is already in the analysis binning
(`in_analysis_bins` is applied inside `component_breakdown`), so the projection
sees one consistent layout and `Binning::total_bins()` is the loop bound.

`evaluate()` must run before `marginalize()`, since `component_breakdown`
requires the prediction to hold the current point. The model enforces this
itself: `marginalize()` calls `evaluate()` when a `set()` has happened since the
last one.

## Phase 2 — Tests in `programs/ictests`

Add `ExplorerModelTests.cpp` to the existing `ICTests` target
(`programs/ictests/CMakeLists.txt`). It already links `io icecube` and builds
parquet fixtures; the projection is pure arithmetic on `std::span`, so most of
it tests without a likelihood at all.

The five checks from the spec. Split them: 1, 3 and 5 exercise the projection
helper directly on synthetic bins and need no fixture; 2 and 4 need a built
likelihood and reuse the fixture pattern the `component_breakdown` test at
`programs/ictests/ICTests.cpp:1177` already sets up.

To keep 1/3/5 fixture-free, the stride walk lives as a free function taking
`(std::span<const double>, const Binning&, std::size_t axis)`, and
`ExplorerModel::marginalize` calls it.

## Phase 3 — Build integration

`programs/ParamExplorer/CMakeLists.txt`, added from `programs/CMakeLists.txt`
guarded by the Qt probe so the cluster build is untouched:

```cmake
find_package(Qt5 COMPONENTS Widgets Charts QUIET)
if(Qt5_FOUND)
  add_subdirectory(ParamExplorer)
else()
  message(STATUS "Qt5 not found -- skipping PhyLiNOExplorer")
endif()
```

The target links `io likelihood icecube results` plus `Qt5::Widgets
Qt5::Charts`, sets `CMAKE_AUTOMOC ON`, and needs
`target_include_directories(... ${CMAKE_SOURCE_DIR}/libraries/results/IceCube)`
— the same line `ictests` needs, since `ICComponentBreakdown.h` is header-only
and not exported by a target.

Homebrew's Qt5 is keg-only, so the configure step needs
`-DCMAKE_PREFIX_PATH=/opt/homebrew/opt/qt@5`. Document it in the target's
comment rather than hardcoding the path.

Verified present: `/opt/homebrew/opt/qt@5`, Qt 5.15.19, with
`lib/cmake/Qt5Charts`.

## Phase 4 — GUI

`MainWindow.h/.cpp` and `main.cpp`.

Widget tree per the spec: two combo boxes, `QChartView`, scrollable slider
panel, status bar. One row per parameter; `fixed` parameters render disabled
rather than hidden, so the layout does not reshuffle between configs.

Slider/spinbox wiring: 0–1000 integer ticks mapped onto `[lo, hi]`,
`QSignalBlocker` on the round trip, model `double` authoritative.

The refresh path is one method, `refresh()`: `evaluate()`, `marginalize()`,
`QLineSeries::replace()` per series, status bar update. The coalescing
single-shot `QTimer` calls it. Under Design C the same method is what the worker
posts back.

Series construction: cumulative-sum step polylines into `QAreaSeries` for the
stack, `QScatterSeries` for data, `QLogValueAxis` with the `1e-3` floor and a
linear/log toolbar toggle.

## Phase 5 — Verify

- `ctest -R ICTests` passes.
- `LLHFit` still configures and builds with Qt absent from `CMAKE_PREFIX_PATH`
  — the actual cluster condition, not an assumption about it.
- Launch on `configs/config_icecube_combined.json`; at the config start point
  with `UseData: false`, the data points sit on the stack. That is the spec's
  built-in correctness check and it is a visual one, so it belongs here rather
  than in the test suite.

## Order

0 → 1 → 2 → 3 → 4 → 5. Phase 0 gates 4's threading design and nothing else, so
1 and 2 can start before its number is in.
