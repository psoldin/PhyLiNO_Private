# Interactive Parameter Explorer — Design

Date: 2026-08-28
Status: Approved, ready for implementation planning

## Purpose

A desktop application for seeing what each fit parameter does to the predicted
histogram. Sliders set the 23 IceCube fit parameters; the plot shows the
marginalized prediction along a chosen axis, recomputed on every move, with the
current `-2lnL` displayed alongside.

This is a forward-model viewer, not a fitting tool. No minimizer runs. Moving a
slider evaluates the likelihood at that point and redraws — it does not profile
over the remaining parameters.

## Feasibility

Every piece the application needs is already public API:

| Need | Existing API |
| --- | --- |
| Parameters in, prediction out | `ICLikelihood::calculate_likelihood(const double*)` |
| Load data once, reuse across evaluations | `ICExperimentModule` caches `ICDataBase` and the GPU backend |
| Per-component histograms | `result::ic::component_breakdown(sample, parameter)` |
| Total prediction and data | `SampleLikelihood::predicted()`, `::data()` |
| Axis kinds and edges | `Binning::axes()`, `Axis` |
| Parameter names, bounds, start values | `params::ic` enum, the config's `Parameter` array |

Qt 5.15.19 with the Charts module is installed (`brew qt@5`).

## Architecture

New target `programs/ParamExplorer/` producing `bin/PhyLiNOExplorer`. Two
layers, with Qt confined to the upper one.

### `ExplorerModel` — no Qt, unit-testable

```
ExplorerModel(config_path)
  parameters()               -> vector<ParamInfo{name, index, value, lo, hi, fixed}>
  set(index, value)
  evaluate()                 -> double            // -2lnL
  sample_names()             -> vector<string>
  axes(sample)               -> vector<AxisInfo{kind_name, edges}>
  marginalize(sample, axis, split_atmospheric = false)
                             -> edges + vector<NamedHist{name, values}> + data hist
```

Owns `io::Options`, one `ICExperimentModule`, the `ICLikelihood` its
`create_likelihood` returns, and the `vector<double>` of current parameter
values. It does not own a `ParameterWrapper`: `Likelihood::parameter()` already
exposes the wrapper holding the last evaluated point with the config's
transform applied, which is exactly what `component_breakdown` requires.
Construction pays the parquet load once. `evaluate()` is a single
`calculate_likelihood(m_Params.data())`.

### `MainWindow` — Qt

Sample combo box and axis combo box on top, `QChartView` in the centre, a
scrollable slider panel below (one row per free parameter: label, `QSlider`,
`QDoubleSpinBox`), and a status bar showing `-2lnL` and its delta from the
config start point.

### Build integration

`find_package(Qt5 COMPONENTS Widgets Charts QUIET)`. When Qt is absent — the
cluster — the target is skipped and `LLHFit` builds unchanged. Qt is never a
hard dependency of the analysis code.

## Data flow

```
QSlider::valueChanged
  -> model.set(index, value)          // writes the array, nothing else
  -> restart QTimer (single-shot, 0 ms, coalescing)
timer fires
  -> llh = model.evaluate()
  -> hists = model.marginalize(sample, axis, split_checkbox)
  -> QLineSeries::replace() on each series
  -> status bar shows llh and delta
```

The coalescing timer is the primary responsiveness mechanism: a drag emits
dozens of `valueChanged` signals per second, and restarting a 0 ms single-shot
timer collapses them so only the newest value is evaluated, after the event
queue drains. `setTracking(false)` reduces that to one signal per release, so the
two together mean one evaluation per gesture.

## Latency

Measured on `config_icecube_combined.json` (3 samples, 1485 + 147 + 1 bins),
median of 10 moves per parameter, `bin/PhyLiNOExplorerBench`:

| Parameter group | `evaluate()` | `marginalize()` |
| --- | --- | --- |
| Norms (Astro, Conv, Prompt, Muon, MuonGun, Galactic) | ~0.01 ms | ~0.03 ms |
| Detector gradients (DOMEff, IceAbs, IceScat, HoleIceP0/P1) | ~0.01 ms | ~0.03 ms |
| SpectralIndex | 40 ms | ~0.03 ms |
| VetoThreshold | 56 ms | ~0.04 ms |
| Barr H/W/Y/Z, CRGrad, DeltaGamma | ~233 ms | ~0.03 ms |

Worst slider: 237 ms per move. The parquet load that precedes all of this is
12–16 s, paid once at startup.

The spread is the point. A norm rescales a cached histogram; a Barr parameter
re-walks every MC event. Seventeen of the 23 sliders are effectively free and
six are not.

**Design B: synchronous evaluation in the coalescing timer's slot, with
`setTracking(false)` so a slider evaluates on release rather than during the
drag.** No worker thread. At 237 ms worst case a release-triggered evaluation
reads as a brief pause, not as a hang, and the thread would cost cancellation,
stale-result handling and `ICLikelihood` shared across two threads to remove a
pause the user already associates with letting go of the slider.

### The atmospheric split is what costs

An earlier measurement put `marginalize()` at a flat ~163 ms for every
parameter, which would have forced the worker thread. All of it was
`AtmosphericFlux::breakdown()`, which re-walks every MC event to separate the
conventional and prompt halves. Every other component in the breakdown is a
cached span.

So the split became opt-in — a checkbox, off by default — and `marginalize()`
fell to 0.03 ms. The default stack draws the summed `atmospheric` entry; ticking
the box redraws with `atmospheric_conv` and `atmospheric_prompt` in its place
and reintroduces the 163 ms, which the same release-triggered path absorbs.

## Marginalization

The binning is row-major over its axes, so projecting onto axis `k` is a stride
walk: `stride_k` is the product of `n_bins` over the axes after `k`, and a flat
index maps to `(flat / stride_k) % n_k`. One pass over `total_bins`, no
allocation beyond the output.

Edges come from `Axis`: `lo + i * step` when `uniform()`, otherwise the explicit
`edges` vector, which is what the non-uniform cascade zenith binning requires.

### Components shown

The five additive entries of the seven that `component_breakdown` returns:
`astro`, `atmospheric_conv`, `atmospheric_prompt`, `template`, `galactic`.

Two returned keys are deliberately excluded. `atmospheric` -- spelled
`atmospheric_veto` when the sample declares the veto variant, so the exclusion
matches on both keys -- is the sum of the conv and prompt entries, so stacking it alongside them would double-count.
`systematicsDelta` is a signed correction to the total, not a component, and
stacking a signed quantity is meaningless. Empty spans — a sample declaring no
galactic component — are skipped rather than drawn as zero.

### Data overlay

`SampleLikelihood::data()`, marginalized identically. With `UseData: false` this
is the Asimov expectation, so at the config start point the data points sit
exactly on top of the stack — a built-in check that the marginalization is
correct.

## Plotting

QtCharts: `QAreaSeries` between cumulative-sum step polylines for the stack,
`QScatterSeries` for the data points, `QLogValueAxis` for counts.

QtCharts ships with the installed Qt5, provides axes, ticks, legend, zoom and
tooltips, and handles non-uniform bin edges because the stack is built from
explicit polyline points rather than bar categories. Hand-rolling a
`paintEvent` would mean ~250 lines of axis, tick and log-scale code that
QtCharts already provides. Embedding a ROOT `THStack` would reuse an existing
dependency, but ROOT 6 dropped `TQtWidget`, leaving a `TCanvas` attached to a
native window handle — fragile on macOS.

## Error handling

- **Missing parquet or malformed config.** The `ExplorerModel` constructor
  throws, as the loaders already do. `main` catches, shows a
  `QMessageBox::critical`, and exits non-zero.
- **Parameters without an upper bound.** `ConvNorm`, `PromptNorm` and
  `MuonNorm` declare `LowerBound` only. The fallback upper bound is
  `max(2 * StartValue, LowerBound + 1)`. The spinbox is not clamped to the
  slider range; typing past it rescales the slider.
- **Integer sliders.** Each slider spans 0–1000 ticks mapped linearly onto
  `[lo, hi]`. The model's `double` is authoritative and the spinbox displays it
  to four significant digits. `QSignalBlocker` guards the slider/spinbox
  round trip against feedback loops.
- **Log axis and empty bins.** `QLogValueAxis` cannot render values at or below
  zero, so those bins are drawn at a floor of `1e-3` counts. A linear/log
  toggle sits in the toolbar, since the single-bin cascade-muon sample makes a
  log axis pointless.
- **Non-finite `-2lnL`.** Displayed verbatim. It describes the parameter point
  and is not a bug to conceal.

## Testing

Unit tests go in the existing `programs/ictests` target, which already builds
topology parquet fixtures.

1. Marginalizing over each axis of a 2D binning yields the same total as the
   unmarginalized prediction.
2. The five stacked components sum bin-by-bin to `predicted()` minus
   `systematicsDelta` at the config start point.
3. Uniform axis edges match `lo + i * step`; the explicit-edge cascade zenith
   binning matches the config list exactly.
4. `set()` followed by two `evaluate()` calls at the same value returns an
   identical likelihood; a changed `AstroNorm` returns a different one.
5. Marginalizing a single-bin axis is the identity.

No GUI tests. Driving a `QSlider` through `QTest` would verify that Qt works,
not that the physics is right. The model layer holds all the logic and contains
no Qt precisely so that it can be tested directly.

## Out of scope

- Profiling: re-minimizing the remaining parameters per slider move.
- Loading fit results to centre sliders on a best-fit point with ±Nσ ranges.
- Multiple samples displayed simultaneously; one is selected at a time.
- Ratio or pull panels beneath the main plot.
- Error bars on the data points.
