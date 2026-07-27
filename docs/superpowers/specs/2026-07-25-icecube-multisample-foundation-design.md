# IceCube multi-sample foundation (Phase 1)

**Date:** 2026-07-25
**Status:** Approved design, pre-implementation
**Scope:** Phase 1 of 3 toward full NNMFit combined-config parity + galactic plane.

## Goal

Turn the hardcoded, single tracks-only IceCube likelihood into a **config-driven,
multi-sample composite** with **runtime N-dimensional binning**, without changing any
physics. Phase 1 is the foundation that Phases 2 (cascades) and 3 (galactic plane) build on.

End goal of the overall project (all phases): reproduce NNMFit `Fit_Configuration_Combined.yaml`
(tracks + `cscd_cascade` + `cscd_muon`, joint SAY fit, veto components) **plus** a galactic-plane
component measured with real RA sensitivity.

### Non-goals (Phase 1)

- No new flux physics (no veto components, no MuonGun template, no galactic flux).
- No RA / 3rd axis *activation* — the binning abstraction is built 3D-ready but dormant.
- No config-driven parameter registry — the compile-time `params::ic` enum stays the index source of truth.

## Context: current state

Single-sample, compile-time everything:

- `io::ic::Constants::nBins` is `constexpr` (45 energy x 33 cos-zenith); sizes `std::array<double,nBins>`
  (`BinArray`) across ~17 files.
- `ICSample` — one SoA (astro/conv/prompt baselines + `conv_alt`/`prompt_alt` + 4 Barr gradients).
- `ICDataBase` loads exactly one parquet (`read_track_baseline`).
- `ICLikelihood` — one `PowerlawFlux m_Astro` + one `AtmosphericFlux m_Atmo`, summed over the single grid,
  one `m_Data`; SAY/Poisson + Gaussian pulls.
- `params::ic` — flat enum, 14 params, single sample; `MuonNorm`/`DOMEff`/`IceAbs`/`IceScat` scaffolded (Fixed).
- Config: boost `property_tree` (JSON), parsed by `ICInputOptions::read(vm, config)` from the `IceCube` subtree;
  `configs/config_icecube.json` is flat single-sample. Top-level `Parameter` array drives free/fixed/constrained.
- GPU: `GpuBackend` (Metal/CUDA) shared across flux components; `dispatch` already launches a runtime count of
  per-bin groups, so it is largely bin-count-agnostic. Per-bin components (`MuonTemplate`, `DetectorSystematics`)
  stay CPU-side.

## Architecture

Composite pattern, reusing the existing framework base classes (`Fit` -> one `ExperimentModule` ->
one `Likelihood`). No new framework-level base classes.

### Components

**`ICExperimentModule`** (role unchanged) — parses config, builds the composite, hands one `Likelihood`
to `Fit`. Heavy parquet loading stays in `create_likelihood`.

**`ICLikelihood` becomes the meta / composite.** Owns `std::vector<std::unique_ptr<SampleLikelihood>>`
(only the config-enabled samples) and the single shared `GpuBackend`. Its `calculate_likelihood(p)`:

1. Load `p` into the shared `ParameterWrapper` once.
2. Sum each `SampleLikelihood::partial_llh()`.
3. Add global Gaussian pulls **once** (pulls are on shared params; per-sample would double-count).
4. Return total; non-finite -> `1e25`.

**`SampleLikelihood`** (new unit, one per sample) owns:

- its `Sample` (SoA + CSR bin layout) and its `Binning`;
- its active flux components (`PowerlawFlux`, `AtmosphericFlux`, and later `MuonTemplate`,
  `DetectorSystematics`, veto, galactic);
- its own runtime `std::vector` histograms `m_Predicted` / `m_Data` / `m_Ssq`, sized from `binning.total_bins()`;
- its own SAY/Poisson term.

It assembles prediction over *its* bins and returns *its* -2lnL for the shared parameter vector. It knows
nothing about other samples.

**`Binning`** descriptor — N axes (energy, cos-zenith, optional reco_ra), edges/counts, `total_bins()`,
`bin_index(event) -> int` (flat index or -1 out of range). Built from config. 2D in Phase 1; the RA axis is
supported by the type but only activated in Phase 3 by adding `reco_ra` to a binning's `axes`.

**`Sample` / `ICDataBase`** — config lists samples; each names its parquet, binning, livetime,
per-sample branch overrides, and active-component set. `ICDataBase` loads the enabled set (one SoA per sample).

### Enable / disable = anchor configs

Which `SampleLikelihood`s the meta constructs is driven entirely by per-sample `enabled` flags in config.
tracks-only, cascade-only, tracks+cascade differ **only** in config (`enabled` flags + which params are
`Fixed`) — no code change, no recompile. This is the anchor-config mechanism.

## Parameter layout

Minuit needs one fixed-length array and `ExperimentModule::number_of_parameters()` is a single number, so:

- **Full layout always allocated** (union of all samples' params). Disabling a sample does not shrink the
  array; its *unique* params go `Fixed` in config — the existing scaffolding pattern.
- **Shared physics params** (one index, read by multiple samples): `AstroNorm`, `SpectralIndex`,
  `ConvNorm`, `PromptNorm`, `BarrH/W/Y/Z`, `CRGrad`, `DeltaGamma` — the NNMFit YAML-anchor-shared set.
- **Per-sample params** (distinct index per sample): detector systematics (`DOMEff`/`IceAbs`/`IceScat`,
  different gradient pickle per sample — `_5up` for cscd) and template norms (`muontemplate` for tracks vs
  `muon` MuonGun for cscd). Phase 2 adds `VetoThreshold`; Phase 3 adds `GalacticNorm`.
- **Index source of truth stays the compile-time `params::ic` enum.** Config maps param *name* -> enum index
  and supplies free/fixed/prior. Adding a param is a recompile — accepted.
- Anchor configs remain valid at fixed layout: cascade-only fixes tracks-unique params (and vice versa);
  shared params still float (physically correct — same astro/atmo across samples).

Phase-1 concrete change: keep the enum, but move the assignment of parameters to samples/components out of
the single hardcoded block so each `SampleLikelihood` reads the shared indices it needs and its own per-sample
indices. Param count derived from the enabled layout, asserted against config.

## Config format

Reuse boost `property_tree` (JSON). Grow the `IceCube` section from flat single-sample to a named-`Binnings`
list + a `Samples` list. Illustrative (JSON):

```json
{
  "Experiment": "IceCube",
  "IceCube": {
    "Backend": "metal",
    "Likelihood": "SAY",
    "UseData": false,
    "Binnings": {
      "tracks_2d": {
        "axes": "energy, cos_zenith",
        "energy":     "(2.5, 7.0, 45)",
        "cos_zenith": "(-1.0, 0.0872, 33)"
      }
    },
    "Samples": {
      "tracks": {
        "enabled":    true,
        "binning":    "tracks_2d",
        "parquet":    ".../tracks_ftp_baseline.parquet",
        "data":       ".../data_tracks.parquet",
        "livetime":   410978234.97,
        "components": "astro, conventional, prompt, muontemplate",
        "Branches": {
          "reco_energy":     "energy_truncated",
          "reco_zenith":     "zenith_MPEFit",
          "conv_baseline":   "mceq_conv_H4a_SIBYLL23c"
        }
      }
    }
  },
  "Parameter": [ /* unchanged top-level array: name/start/step/fixed/constrained */ ]
}
```

Notes:

- `(min, max, N)` where **N = bin count** (fixes NNMFit's edges-vs-bins ambiguity; store bins internally,
  document clearly). Bin count in each axis, total = product.
- `enabled` per sample = the anchor-config toggle.
- `Binnings` are named and referenced so each sample points at its own; a sample gains an RA axis in Phase 3
  purely by adding `reco_ra` to its binning's `axes` string — no code change (the "dynamically adjustable
  third axis via config" requirement).
- `Branches` (already exists as `BranchNames`) moves under each sample (tracks `energy_truncated`, cscd
  `energy_monopod`), with struct defaults for anything omitted.
- `components` per sample selects which flux units the `SampleLikelihood` builds (drives the exclusion logic
  in Phase 2).
- The flat single-sample keys (`TrackBaselineFilePath`, top-level `Livetime`, ...) are replaced by the
  per-sample entries. A one-sample `Samples` block reproduces today's config.

## Data flow

Startup (`ICExperimentModule::create_likelihood`):

1. Parse `IceCube` ptree -> list of enabled `SampleConfig` (binning ref resolved, branches, livetime, components).
2. Per enabled sample: `ICDataBase` loads its parquet -> `Sample` SoA; assign bins via its `Binning::bin_index`;
   `sort_into_bins()` -> CSR `bin_offsets`.
3. Build meta `ICLikelihood`: one shared `GpuBackend`; construct each `SampleLikelihood` (owns its flux
   components; each uploads its columns/offsets to the backend once; sizes its `std::vector` histograms from
   `binning.total_bins()`).
4. Build Asimov (or load real data) per sample -> each sample's `m_Data`. Meta collects pull specs from shared
   params once.

Per `calculate_likelihood(p)` (hot path, no allocation):

1. Meta loads `p` into shared `ParameterWrapper`.
2. Per `SampleLikelihood`: components `check_and_recalculate` (skip if their params unchanged — existing
   caching), reassemble `m_Predicted` over its bins, compute its SAY/Poisson term.
3. Meta sums partials + global pulls once -> total.

Buffers are sized once at startup and overwritten, never reallocated per eval — preserves the current
no-alloc hot loop, now per-sample-sized instead of one `std::array`.

## Testing / validation

Hard gate: the new multi-sample machinery reproduces the current tracks-only output exactly.

1. **Golden regression (gate).** Snapshot the current `HEAD` tracks-only fit on the test parquet: Asimov
   total, per-bin prediction, and -2lnL at the default plus a few offset parameter points. After the refactor,
   a single-sample config (`Samples: { tracks: { enabled: true } }`, all else disabled) must reproduce those
   numbers — CPU path exact, FP32 GPU path within tolerance.
2. **Unit tests.** `Binning::bin_index` vs the old `Constants::bin_index` (identical for the tracks grid);
   `total_bins()`; CSR `sort_into_bins` invariants for a runtime-sized binning.
3. **Composite test.** Two-copies-of-tracks config -> meta -2lnL == 2x single (shared params). Confirms
   summation and that pulls are added once, not doubled.
4. **Enable/disable test.** Disabling a sample removes exactly its contribution; its unique params being
   `Fixed` does not perturb shared-param gradients.
5. **Backend parity.** CPU vs Metal/CUDA agree within FP32 tolerance on the multi-sample path (existing check,
   extended per sample).

## Affected code (indicative)

- `io/IceCube/ICConstants.h` — `constexpr nBins`/`bin_index` -> runtime `Binning` type (new file, e.g.
  `io/IceCube/Binning.{h,cpp}`); `ICConstants.h` retained only for genuinely global constants.
- `io/IceCube/ICSample.h` — unchanged column set for Phase 1, but `sort_into_bins` parameterized by
  `total_bins` instead of `Constants::nBins`.
- `io/IceCube/ICDataBase.{h,cpp}` — load an enabled *set* of samples; per-sample branches/binning.
- `io/IceCube/ICInputOptions.{h,cpp}` — parse `Binnings` + `Samples`; `BranchNames` per sample.
- `likelihood/IceCube/ICLikelihood.{h,cpp}` — split into meta `ICLikelihood` + new `SampleLikelihood`
  (`SampleLikelihood.{h,cpp}`).
- `likelihood/IceCube/{PowerlawFlux,AtmosphericFlux,MuonTemplate,DetectorSystematics}.*` — replace
  `BinArray`/`Constants::nBins` with runtime-sized buffers from the sample's `Binning`.
- `likelihood/IceCube/{Metal,Cuda}Backend.*`, `GpuBackend.h` — `dispatch` bin count from the sample's binning
  (already runtime-group); verify no residual `Constants::nBins`.
- `params/ic` (`ICParameter.h`) — enum kept; per-sample vs shared index wiring moved out of the single block.
- `configs/config_icecube.json` — migrate to `Binnings` + `Samples` (one-sample block == today).

## Roadmap (later phases, out of scope here)

- **Phase 2 — Cascades.** `conventional_veto`/`prompt_veto` = atmo x `VetoThreshold` passing-fraction reweight
  (new `log_PF_at100GeV_*` columns + `VetoThreshold` param); MuonGun `muon` template alongside Corsika
  `muontemplate`; per-sample component masking; `_5up` SnowStorm gradient variant; cscd binnings. Gate: match
  NNMFit combined config (3 samples, 2D, no galactic).
- **Phase 3 — Galactic plane.** Activate 3rd (RA) axis; add `reco_ra` column; `GalacticFlux` spline component
  (C++ spline-eval lib — decision point) + RA oversampling (~x10); `GalacticNorm` (+ optional morphology);
  3D GPU kernels. Gate: galactic injection/recovery.
