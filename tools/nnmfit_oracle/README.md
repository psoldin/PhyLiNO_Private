# NNMFit reference oracle

The IceCube cascade work (Phase 2) is validated against NNMFit run on the *same* local input
files. This directory holds the harness for producing that reference and the numbers it produced.

## Environment and inputs

| what | where |
|---|---|
| NNMFit venv | `/Users/soldin/Projects/IceCube/NNMFit/.venv` (python 3.11.15, aesara 2.9.4) |
| analysis config | `/Users/soldin/Downloads/Fit_Configuration_Combined_macOS.yaml` |
| recorded fit | `/Users/soldin/Projects/IceCube/NNMFit/NNMFit/Output.pickle` |
| datasets | `/Users/soldin/Downloads/nnmfit_files/datasets/{tracks,cscd_cascade,cscd_muon}_ftp_baseline`, data under `datasets/data/` |

The config uses NNMFit's newer schema: sample list in `analysis.datasets`, samples under a
top-level `datasets:` section, systematics under `systematics:` (no flat `config:` block).

## The recorded fit

Real-data SAY fit over all three samples, LBFGSB, converged
(`CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH`, 962 evaluations, 848 iterations),
`llh_value = 3997.6142420984984`.

```
tools/nnmfit_oracle/read_fit_result.py \
    /Users/soldin/Projects/IceCube/NNMFit/NNMFit/Output.pickle --json /tmp/nnmfit_fit_reference.json
```

| parameter | fitted | parameter | fitted |
|---|---|---|---|
| AstroNorm | 2.0102235093493097 | VetoThreshold | 0.14108708515445395 |
| SpectralIndex | 2.4973509081979963 | MuonGunNorm | 1.1362703633930007 |
| ConvNorm | 1.1862521739969658 | MuonNorm | 1.827017996655897 |
| PromptNorm | **0.0** (range boundary) | DOMEff | 1.01730784376685 |
| DeltaGamma | 0.07312317645432367 | IceAbs | 0.9951836691033408 |
| CRGrad | 0.5278438102260666 | IceScat | 1.0048480464614487 |
| BarrH | −0.059121890824429764 | HoleIceP0 | 0.19797631957351045 |
| BarrW | 0.2941434915474592 | HoleIceP1 | −0.06642331999754497 |
| BarrY | 0.048184445442831736 | | |
| BarrZ | −0.028540463688312537 | | |

Two things to keep in mind when comparing fits:

- NNMFit **randomizes its seeds** around the config defaults (`minimizer_seeds` in the pickle:
  `AstroNorm` 1.4515, `ConvNorm` 0.9677, `DOMEff` 0.8938, …), so only the minimum is comparable,
  not the trajectory.
- `PromptNorm` sits exactly at its NNMFit range boundary (0.0). PhyLiNO has no bounds plumbing, so
  it may walk negative there. That is a framework difference, not a defect.

## Likelihood convention (verified in the source)

`NNMFit/likelihoods/builder.py` builds `llh = Σ_bins log L + log_prior`, with
`log_prior = -((x - cv) / (σ√2))² = -½χ²` (`make_priors`), and
`NNMFit/core/nnm_fitter.py:681` minimises **`-llh_function`**. So NNMFit's reported `llh_value` is

```
-Σ log L + ½ χ²          (no factor 2; the saturated term is NOT subtracted for SAY)
```

PhyLiNO's `ICLikelihood` computes `-2 Σ log L + χ²` — i.e. **exactly twice** NNMFit's objective — and
then subtracts a first-call baseline (`m_LLHBaseLine`). Therefore:

> Absolute likelihood values are not comparable. Compare *differences* between parameter points,
> with `Δ(PhyLiNO) = 2 · Δ(NNMFit)`.

## Histogram dumps

```
tools/nnmfit_oracle/dump_histograms.sh [BASE_CONFIG] [OUT_DIR] [SAMPLE ...]
```

For each sample it writes `<sample>_total.pickle`, one `<sample>_<component>.pickle` per per-event
component, and `<sample>_no_template.pickle`. Each pickle holds `histograms` (µ per bin) and
`fluctuations` (σ² per bin). `nnmfit_set_excluded.py` does the config surgery: it sets that sample's
`excluded_components`, restricts `analysis.datasets` to the one sample, and switches
`analysis.analysis_type` to `asimov` — `make_histogram.py` dumps `get_data_hists()`, which is the
measured counts under `analysis_type: data` but the **model prediction** under `asimov`.

Runtime on an M1 Max: ~30 s for `cscd_muon`, ~30 s for `cscd_cascade`, ~80 s for `tracks`
(5 dumps each).

### Two NNMFit behaviours the harness works around

1. **A sample cannot consist of only a histogram component.** The muon templates are `TemplateFlux`
   (`IS_HIST_COMP`), and `histogram_builder.__make_total_fluctuation` leaves `weights` as the scalar
   `0.0` when no per-event component is present; aesara's `bincount` then dies with
   `'float' object has no attribute 'dtype'`. So the template is never dumped alone — take it as
   `total − no_template`. Cross-check: `no_template` must equal the sum of the per-event dumps.
2. **The SnowStorm gradient is a histogram-level additive term applied to every dump.** It does not
   scale with the components, so when the detector parameters are off their split values, each
   per-component dump contains the *full* delta and the per-component dumps stop being additive
   (they also get clipped at µ ≥ 0). At the split values the delta is identically zero and additivity
   is exact. Hence the three dump sets below.

### Dump sets

| directory | parameter point | per-component additivity |
|---|---|---|
| `/tmp/nnmfit_dumps` | config defaults (AstroNorm 1.5, SpectralIndex 2.4, ConvNorm 1.0, PromptNorm 0.5, VetoThreshold 0, det-sys at split) | exact |
| `/tmp/nnmfit_dumps_fitted_nosys` | fitted flux parameters, det-sys at split (`/tmp/nnmfit_fitted_nosys.yaml`) | exact |
| `/tmp/nnmfit_dumps_fitted` | fully fitted, det-sys included (`/tmp/nnmfit_fitted_point.yaml`) | broken by the gradient term — compare `total` only, or add PhyLiNO's `systematicsDelta` to each component |

The defaults point leaves `VetoThreshold = 0` and the det-sys deltas at zero, so it does **not**
exercise the veto exponent or the gradients. `fitted_nosys` is the set to use for per-component
comparisons: it exercises the veto (0.1411) and every atmospheric parameter while keeping the dumps
additive.

### Recorded µ sums (events)

`/tmp/nnmfit_dumps` — config defaults:

| sample | astro | conventional(_veto) | prompt(_veto) | template | total |
|---|---|---|---|---|---|
| tracks | 4182.2918 | 697399.4891 | 1179.4654 | 1138.1672 | 703899.4135 |
| cscd_cascade | 905.6846 | 9749.4762 | 153.2547 | 773.7884 | 11582.2039 |
| cscd_muon | 325.2242 | 8188.6717 | 63.2532 | 18066.2565 | 26643.4055 |

`/tmp/nnmfit_dumps_fitted_nosys` — fitted flux parameters, det-sys at split:

| sample | astro | conventional(_veto) | prompt(_veto) | template | total |
|---|---|---|---|---|---|
| tracks | 7183.4471 | 832819.1135 | 0.0000 | 2079.4519 | 842082.0124 |
| cscd_cascade | 1503.2852 | 10597.9610 | 0.0000 | 879.2328 | 12980.4789 |
| cscd_muon | 598.5481 | 9221.9159 | 0.0000 | 20528.1518 | 30348.6158 |

`/tmp/nnmfit_dumps_fitted` — fully fitted (per-component values include the gradient delta and are
clipped, hence non-additive; `total` is the meaningful column):

| sample | astro | conventional(_veto) | prompt(_veto) | template | total |
|---|---|---|---|---|---|
| tracks | 28517.3759 | 853758.6107 | 21758.0217 | 2079.4519 | 863021.2734 |
| cscd_cascade | 1692.0877 | 10787.0220 | 203.3010 | 879.2328 | 13169.2814 |
| cscd_muon | 956.7206 | 9580.0884 | 358.1725 | 20528.1518 | 30706.7883 |

The `prompt` columns are exactly 0 in the fitted sets because `PromptNorm` fitted to 0.0; the
non-zero entries in the last table are the gradient delta alone, clipped at zero.

Template anchors worth remembering, because they are pure arithmetic and catch a mis-scaled
template immediately: the MuonGun rates are 2.3425769796686227e-06 s⁻¹ (cascade) and
5.4694021304430105e-05 s⁻¹ (muon), and with livetime 330315015.11 s that is **773.7884** and
**18066.2565** events — exactly the dumped template columns at defaults.

σ² note: the reference fit uses `cscd_muongun_ALL_KDE_5up_manual_ssq_no_fluct.pickle`, whose
`template_fluctuation` is `None`, so the template contributes nothing to σ² (visible in the dumps:
`total` and `no_template` share the same σ² sum).

## Per-bin comparison against PhyLiNO

```
tools/nnmfit_oracle/make_probe_config.py configs/config_icecube_combined.json /tmp/probe_defaults.json
tools/nnmfit_oracle/make_probe_config.py configs/config_icecube_combined.json /tmp/probe_fitted_nosys.json \
    --set AstroNorm=2.0102235093493097 --set SpectralIndex=2.4973509081979963 \
    --set ConvNorm=1.1862521739969658 --set PromptNorm=0.0 --set DeltaGamma=0.07312317645432367 \
    --set CRGrad=0.5278438102260666 --set BarrH=-0.059121890824429764 --set BarrW=0.2941434915474592 \
    --set BarrY=0.048184445442831736 --set BarrZ=-0.028540463688312537 \
    --set VetoThreshold=0.14108708515445395 --set MuonGunNorm=1.1362703633930007 \
    --set MuonNorm=1.827017996655897

build/programs/LLHFit/LLHFit -c /tmp/probe_defaults.json --silent && cp Output.json /tmp/probe_defaults_output.json
build/programs/LLHFit/LLHFit -c /tmp/probe_fitted_nosys.json --silent && cp Output.json /tmp/probe_fitted_nosys_output.json

tools/nnmfit_oracle/compare_to_nnmfit.py /tmp/probe_defaults_output.json /tmp/nnmfit_dumps --tolerance 1e-6
tools/nnmfit_oracle/compare_to_nnmfit.py /tmp/probe_fitted_nosys_output.json /tmp/nnmfit_dumps_fitted_nosys --tolerance 1e-6
```

`make_probe_config.py` fixes every parameter (so `Fit::minimize()` evaluates the Asimov prediction
at the given point instead of moving it) and pins `Likelihood: SAY`, `Backend: cpu`, `UseData: false`
regardless of what the base config currently has.

Both points agree with NNMFit to floating-point precision (~1e-13 to 1e-16 relative) for every
sample and component that has a dump:

```
=== defaults ===
ok     tracks         astro              max rel dev 3.160e-16
ok     tracks         atmospheric        max rel dev 1.324e-14
ok     cscd_cascade   astro              max rel dev 2.485e-16
ok     cscd_cascade   atmospheric_veto   max rel dev 4.360e-15
ok     cscd_muon      astro              max rel dev 1.748e-16
ok     cscd_muon      atmospheric_veto   max rel dev 1.693e-13
all compared components agree

=== fitted_nosys ===
ok     tracks         astro              max rel dev 3.397e-16
ok     tracks         atmospheric        max rel dev 3.613e-16
ok     cscd_cascade   astro              max rel dev 1.830e-16
ok     cscd_cascade   atmospheric_veto   max rel dev 9.540e-17
ok     cscd_muon      astro              max rel dev 1.899e-16
ok     cscd_muon      atmospheric_veto   max rel dev 0.000e+00
all compared components agree
```

`tracks/atmospheric` only agrees at `fitted_nosys` (where CRGrad/Barr* are nonzero, so the CR-alt
and Barr-slope columns actually contribute) after fixing `ICDataBase.cpp`'s oscillation application
to scale only `conv_baseline`/`prompt_baseline`, not `conv_alt`/`prompt_alt`/`barr_conv[k]` — see
`libraries/io/IceCube/ICDataBase.cpp`'s `apply_survival` block and its comment for why NNMFit's
`OscillationsHook` only ever touches a flux's primary baseline-weight column
(`Flux.py::apply_hooks_for_flux`). At `defaults` both CRGrad and every Barr parameter are 0, so the
bug was invisible there — any future check of this kind should include a nonzero-systematics point,
not just defaults.

`template` is skipped everywhere: no individual `muontemplate`/`muon` pickle is dumped (see "Two
NNMFit behaviours the harness works around" above), and reconstructing `total − no_template` on the
NNMFit side isn't wired into `compare_to_nnmfit.py`.

A real-data comparison (`UseData: true` on both sides) is *not* run by this harness — do not run
LLHFit against real detector data without explicit authorization. If/when that check is wanted,
the same `compare_to_nnmfit.py` works against an `Output.json` produced with `UseData: true` and a
`dump_histograms.sh` run under `analysis_type: data`; script it, don't run it inline.

## Regenerating everything

```bash
V=/Users/soldin/Projects/IceCube/NNMFit/.venv/bin/python
tools/nnmfit_oracle/read_fit_result.py /Users/soldin/Projects/IceCube/NNMFit/NNMFit/Output.pickle \
    --json /tmp/nnmfit_fit_reference.json
tools/nnmfit_oracle/dump_histograms.sh                                    # defaults  -> /tmp/nnmfit_dumps
tools/nnmfit_oracle/dump_histograms.sh /tmp/nnmfit_fitted_nosys.yaml /tmp/nnmfit_dumps_fitted_nosys
tools/nnmfit_oracle/dump_histograms.sh /tmp/nnmfit_fitted_point.yaml  /tmp/nnmfit_dumps_fitted
```

The two fitted-point configs are generated from `Output.pickle` (see the plan, Task 1 Steps 4); the
dumps live under `/tmp` and are cheap to recreate.
