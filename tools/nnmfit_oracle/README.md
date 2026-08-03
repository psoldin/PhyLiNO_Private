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

PhyLiNO's `ICLikelihood` computes `-2 Σ log L + χ²` — i.e. **exactly twice** NNMFit's objective.
Since 2026-08-03 it applies **no baseline offset** (the first-call `m_LLHBaseLine` and the later
hardcoded constant are both gone), and its Poisson term subtracts the saturated likelihood exactly
as `LikelihoodBuilder.make_binwise_llh(substract_saturated=True)` does. Therefore:

> Absolute likelihood values *are* comparable: `PhyLiNO = 2 · NNMFit`, at any parameter point.

The factor 2 is the error-definition convention, not a normalisation difference: PhyLiNO reports
−2 log L with Minuit2's default `ErrorDef = 1`, NNMFit reports −log L with iminuit
`errordef = LIKELIHOOD = 0.5`. Same minimum, same parameter errors.

### Which term each code uses

| | NNMFit | PhyLiNO |
|---|---|---|
| SAY | `SAYLLH.compute_log_L`, saturated term **not** subtracted (`builder.py` raises `NotImplementedError` if asked) | `say_bin_log_likelihood` — same, unsubtracted |
| Poisson | `logP(k\|µ) − logP(k\|k)` (subtraction is the default and the only path `run_fit` takes) | `poisson_bin_log_likelihood_saturated` — same |
| µ ≤ 0 | `−690·k` for `k > 0`, else `0` | same |
| prior | `−((x − x₀)/(σ√2))²` | `χ² = ((x − x₀)/σ)²`, i.e. `−2 ×` theirs |

The saturated subtraction is why no baseline is needed: at the Asimov point every bin has `µ = k`,
so the Poisson total is exactly 0 instead of a bin-count-sized constant.

## Likelihood value parity

Two levels, one fast and offline, one requiring NNMFit runs.

**Per-bin golden values (offline, in the unit suite).**
`gen_llh_golden.py` dumps `log L` per bin straight out of NNMFit's aesara graphs
(`SAYLLH.compute_log_L`, `PoissonLLH.compute_log_L`, and the saturated-subtracted combination) for
a set of edge cases — non-integer `k`, `ssq = 0`, `ssq > µ²`, `ssq < 0`, `µ = 0`, `µ < 0`, tiny `µ`
with large `k`, and a 2.5e5-count bin:

```
/Users/soldin/Projects/IceCube/NNMFit/.venv/bin/python tools/nnmfit_oracle/gen_llh_golden.py \
    -o programs/ictests/LLHGolden.inc
```

The generated header is committed and asserted by `ICTests.LikelihoodParityTest.*` to 1e-12
relative. Regenerate it only when NNMFit's likelihood implementation changes.

**Whole-likelihood value at one fixed point (needs NNMFit).**

```
tools/nnmfit_oracle/compare_llh_value.py configs/config_icecube_combined.json \
    /Users/soldin/Downloads/Fit_Configuration_Combined_macOS.yaml --use-data --likelihood SAY
```

Both sides evaluate rather than fit: PhyLiNO through a probe config with every parameter `Fixed`,
NNMFit through `run_fit.py --fix` for every parameter (which makes `do_fit()` take its
"all parameters fixed → evaluate" branch; the script fails if `minimizer_info` comes back as
anything but `evaluated_only`). It then checks `PhyLiNO == 2 · NNMFit`.

Use `--use-data`: without it each code builds its *own* Asimov set from its own prediction at the
probe point, which makes the Poisson objective trivially 0 on both sides — a smoke test for the
absence of a baseline offset, not a parity gate. `--set NAME=VALUE` moves the point (PhyLiNO
names; translated via the script's `NAME_MAP`).

## The gradient livetime scaling (found 2026-08-03)

NNMFit multiplies every SnowStorm gradient by `livetime_scaling = t_analysis / t_gradients`
(`snowstorm_gradient.py:193`), reading `t_gradients` out of the config embedded in the gradient
pickle. The tracks gradients were produced on MC with `livetime = 387231573.49 s` against an
analysis livetime of `410978234.97 s` — a factor **1.061324187**. `export_nnmfit_inputs.py`
defaulted `--livetime-scale` to 1.0 and nobody passed it, so `gradients_tracks.txt` carried
`lt_scale 1.0` and every detector-systematic effect on tracks was 6% too small. The cascade
gradients were made at the analysis livetime, so they were always correct — which is why only
tracks disagreed.

**Why every earlier gate missed it.** All the per-bin gates ran at the split values
(`DOMEff = IceAbs = IceScat = 1`, hole-ice at its split), where `mu_delta` is identically zero and
the scale factor multiplies nothing. It only appears once a detector parameter moves, and then as
a smooth few-percent shift — indistinguishable from an interpolation artefact if you are only
looking at one number. It was caught by the *likelihood value* gate below, at
`DOMEff = 0.97`: tracks was 2.8% off (Δ = 338.14 in −2lnL) while cscd_cascade and cscd_muon agreed
to 1e-14 at the same point.

The exporter now derives the factor from the pickle and **refuses to run** without either
`--analysis-livetime` or an explicit `--livetime-scale`, because silently defaulting to 1.0 is
what made this invisible:

```
tools/export_nnmfit_inputs.py gradients \
    /Users/soldin/Downloads/nnmfit_files/snowstorm_ftp_all.pickle \
    /Users/soldin/Downloads/nnmfit_files/exported/gradients_tracks.txt \
    --det-conf IC86_pass2_SnowStorm_v2_tracks --bins 1485 --zenith-bins 33 \
    --analysis-livetime 410978234.97
```

Only the header line changes (`lt_scale`); `DetectorSystematics` applies it at runtime, including
`lt_scale²` on the covariance term. Re-exporting the two cascade files reproduces them byte for
byte.

**The general lesson**, which is the reason this section exists: a per-bin comparison at the
configuration's default point cannot see any term that is zero there. Detector systematics, Barr
slopes and `CRGrad` are all zero at defaults. Gate them at a moved point.

## Randomized start values

`LLHFit --randomizeSeeds` perturbs the minimizer's start point, the counterpart of NNMFit's
`randomize_param_seeds`, which `run_fit.py` applies **by default** (`--use_default_param_seeds`
turns it off). Only the start point moves: the data, Asimov included, is built from the configured
values either way.

| | NNMFit | PhyLiNO |
|---|---|---|
| draw | `np.random.normal(1, w) * value` | same, `w` = `--randomizeWidth` (default 0.08) |
| `w` | `0.25 * (hi - lo)` if bounded, else `0.08` | no bounds in this framework, so always `--randomizeWidth` |
| `value == 0` | stays `0` — never randomized | drawn from `N(0, StepWidth)` |
| fixed parameters | overwritten after the draw | not drawn |
| reproducible | no (numpy global RNG; the draw is recorded in the output pickle) | yes, from `--seed` |

The `value == 0` difference matters for the IceCube configs: `BarrH/W/Y/Z`, `CRGrad`,
`DeltaGamma` and `VetoThreshold` all start at 0, so NNMFit's multiplicative draw leaves every one
of them exactly at its configured value. `ana::randomized_start_value`
(`libraries/likelihood/ParameterSeeding.h`) falls back to an additive draw of width `StepWidth`
there; `ICTests.ParameterSeedingTest.*` covers both branches and the seed reproducibility.

The parity scripts never pass `--randomizeSeeds` — they pin every parameter, and a randomized
start would move the point being compared.

## Recorded likelihood-value results (2026-08-03)

All Asimov, all through `compare_llh_value.py`. "Moved point" means the Asimov set is generated at
the config values while the likelihood is evaluated somewhere else (`AsimovValue` / `--inject`),
which is the only way the Poisson term measures anything — at its own truth it is exactly 0.

| config | likelihood | point | backend | deviation from `2 · NNMFit` |
|---|---|---|---|---|
| combined 2D (`HEAD`) | SAY | defaults | cpu | 6.9e-14 |
| combined 3D | SAY | defaults | cpu | 1.2e-13 |
| combined 2D (`HEAD`) | Poisson | moved | cpu | 1.6e-14 |
| tracks 2D | Poisson | moved | cpu | 2.2e-14 |
| cscd_cascade 2D | Poisson | moved | cpu | 6.8e-14 |
| cscd_muon 2D | Poisson | moved | cpu | 3.0e-12 |
| combined 3D | SAY | moved | cpu | 6.1e-14 |
| combined 3D | SAY | moved | **metal** | 3.4e-08 |
| combined 2D (`HEAD`) | Poisson | moved | **metal** | 1.1e-06 |
| combined 3D | SAY | **all 21 parameters moved** | cpu | 3.2e-16 |

The moved points are `AstroNorm 1.8, ConvNorm 1.15, DeltaGamma 0.05, DOMEff 0.97, BarrH 0.1` (2D)
and additionally `AstroGamma1 1.5, AstroGamma2 2.9, AstroEBreak 4.6, GalacticNorm0/1 1.2` (3D).

The **all-parameters-moved** row is the strongest single check in this table: every one of the 21
shared parameters off its no-op value at once, so nothing is multiplied by zero. It is the only
run in which `CRGrad` (0.4) is non-zero — i.e. the only one where the `conv_alt`/`prompt_alt`
CR-gradient columns contribute at all — and the only one exercising all four Barr slopes and all
five detector-systematic gradient rows rather than `DOMEfficiency` alone. Reproduce with
`--set CRGrad=0.4 --set BarrW=-0.15 --set BarrY=0.12 --set BarrZ=-0.05 --set PromptNorm=0.8
--set MuonNorm=1.3 --set MuonGunNorm=1.2 --set VetoThreshold=0.12 --set IceAbs=1.03
--set IceScat=0.98 --set HoleIceP0=0.30 --set HoleIceP1=-0.08` on top of the 3D moved point.

**GPU backends.** Metal does not reach the CPU path's ~1e-13 because the flux kernels are FP32.
The useful way to state its accuracy is **absolute, not relative**: the Metal-vs-CPU difference
was `0.0127` on the 3D SAY point (value 3.7e5) and `0.0129` on the 2D Poisson point (value
1.2e4) — the same absolute error against likelihood values 30x apart, because it comes from FP32
rounding in the per-event flux sum, not from anything that scales with the likelihood. So the
relative tolerance a GPU run needs depends on how large the likelihood happens to be
(3e-8 for the first, 1e-6 for the second); `--tolerance 1e-4` covers both. A *relative* tolerance
tightened for the 3D case would spuriously fail the 2D one.

That absolute offset is ~0.013 in `-2lnL`, i.e. far below the 1-unit scale that matters for a fit
result, but it is not zero: do not use a GPU backend when comparing two likelihood values that
differ by less than ~0.1.

**CUDA cannot be tested on macOS** — `CudaBackend_stub.cpp` returns `available() == false`, so
`--backend cuda` silently falls back to the CPU path and proves nothing about the CUDA kernels
(the run will still print `ok`, which is exactly the kind of quiet false pass this directory's
notes keep warning about). Gate CUDA on a machine with a device, and check the
`ICLikelihood: CUDA backend using FP64/FP32 kernels` line actually appears.

## Converged-fit parity

```
tools/nnmfit_oracle/run_fit_parity.sh [OUT_DIR] [NNMFIT_CONFIG]
```

Stage 1 is the value check above, stage 2 fits with NNMFit from the **config default seeds**
(`--use_default_param_seeds`, so its usual seed randomisation is off and both codes start from the
same point), stage 3 fits with PhyLiNO on the same data and diffs every parameter plus the minimum
against `2 · llh_value`.

This fits real detector data, so it is a manual script — run it from a shell yourself; nothing
invokes it automatically. Two differences are expected and are not defects: the two minimisers
(Migrad vs LBFGSB) stop at different points within their own tolerances, and `PromptNorm` is
clipped at 0 by NNMFit's bounds while PhyLiNO, which has no bounds plumbing, may go slightly
negative.

> Real *detector* data has still never been compared. The minimiser is now gated by the
> pseudo-experiment run below, so what remains untested here is only the measured histograms
> themselves (`UseData: true` reading a data parquet). Owner: the user, deliberately — see the
> standing rule that data fits are run by hand, not by an agent.

## Converged-fit parity on a pseudo-experiment (2026-08-03)

```
tools/nnmfit_oracle/run_pseudo_fit_parity.sh [OUT_DIR] [SEED]
```

A Poisson draw on the Asimov prediction is integer-valued by construction, so it is a legitimate
dataset, and feeding the **same** draw to both codes makes their objectives identical bin for bin.
This gates what no fixed-point comparison can: the minimiser, the derivatives, and the fitted
parameter values with their uncertainties. It needs no detector data.

The draw is generated once by `make_pseudo_data.py` and handed to PhyLiNO as per-sample
`DataCounts` text files and to NNMFit as a `custom_data` / `custom_dataset` pickle — the NNMFit
branch that uses the array **verbatim without re-fluctuating it** (`nnm_fitter.py:487-501`). Each
framework drawing its own pseudo-experiment would compare two different datasets and prove
nothing.

Result at seed 20260803 (combined 2D, SAY, 18 free parameters):

| | value |
|---|---|
| PhyLiNO `-2lnL + chi2` | `7839.106994` (converged, EDM 1.9e-05) |
| `2 ×` NNMFit | `7839.157827` (`ABNORMAL_TERMINATION_IN_LNSRCH`, 805 evaluations) |
| difference | PhyLiNO **0.0508 lower** |

Every parameter agrees well inside its own fitted uncertainty: 15 of 18 below 0.05σ, and the three
largest are `PromptNorm` 0.221σ, `AstroNorm` 0.170σ, `SpectralIndex` 0.139σ. `PromptNorm` is the
known bounds difference — NNMFit clips it at exactly 0.0, PhyLiNO has no bounds plumbing and walks
to −0.123 — which also accounts for most of the likelihood gap and for the correlated shifts in
the other two. PhyLiNO reaching the lower minimum while NNMFit's line search gives up is a
minimiser difference, not an objective difference; the fixed-point gates above already prove the
objectives agree to 1e-16.

**Two traps when regenerating the draw.** The pickle must be written by *NNMFit's* interpreter —
one written by a newer numpy carries `numpy._core` references that NNMFit's numpy cannot unpickle
— and `read_fit_result.py` needs the fit run *without* `--skip_save_config`, or the result carries
no `settings` block for it to read.

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

build/programs/LLHFit/LLHFit -c /tmp/probe_defaults.json --fitOnly --silent && cp Output.json /tmp/probe_defaults_output.json
build/programs/LLHFit/LLHFit -c /tmp/probe_fitted_nosys.json --fitOnly --silent && cp Output.json /tmp/probe_fitted_nosys_output.json

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

## Comparing the fit itself against real data

`tools/nnmfit_oracle/run_ic_data_fit.sh [OUT_DIR]` fits `config_icecube_combined.json` against real
detector data (`UseData: true`) and prints the resulting parameters next to NNMFit's recorded fit
(`/tmp/nnmfit_fit_reference.json`). This is the only step in the Phase 2 acceptance gate that
touches real data, so it is a standalone script you run yourself from a shell — it is not run as
part of any automated check here. Expected, non-defect differences: `PromptNorm` may land slightly
negative for us where NNMFit clips it at its 0.0 range boundary (this framework has no bounds
plumbing), and the two sides' random/fixed seeding means only the minimum is comparable, not the
trajectory. Absolute likelihood values are not comparable either (see "Likelihood convention" above)
— only differences between points are.

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

## Phase 3a: the 3D (right-ascension + galactic plane) gate

The 3D configuration adds a right-ascension axis to `tracks` and `cscd_cascade`, a `CringeFITS`
galactic template on those two samples, and NNMFit's `AstroBPL` broken power law on all three.
`cscd_muon` deliberately stays 2D, so one fit mixes 2D and 3D samples.

```bash
V=/Users/soldin/Projects/IceCube/NNMFit/.venv/bin/python

# NNMFit side: generate the 3D analysis config from the Phase-2 one, then dump
"$V" tools/nnmfit_oracle/make_3d_config.py \
    /Users/soldin/Downloads/Fit_Configuration_Combined_macOS.yaml \
    /Users/soldin/Downloads/Fit_Configuration_Combined_3D_macOS.yaml
tools/nnmfit_oracle/dump_histograms.sh \
    /Users/soldin/Downloads/Fit_Configuration_Combined_3D_macOS.yaml /tmp/nnmfit_dumps_3d

# our side: evaluate at the same fixed point, then compare
tools/nnmfit_oracle/make_probe_config.py configs/config_icecube_combined_3d.json /tmp/probe_3d_defaults.json
build/programs/LLHFit/LLHFit -c /tmp/probe_3d_defaults.json --fitOnly --silent
cp Output.json /tmp/probe_3d_defaults_output.json
tools/nnmfit_oracle/compare_to_nnmfit.py /tmp/probe_3d_defaults_output.json /tmp/nnmfit_dumps_3d --tolerance 1e-8
```

Result at the defaults point (AstroNorm 1.77, gamma_1 1.31, gamma_2 2.74, e_break 4.4,
cringefits_norm 1.0, everything else as in the 2D defaults point):

```
ok     tracks         astro              max rel dev 1.931e-15
ok     tracks         atmospheric        max rel dev 1.321e-14
ok     tracks         galactic           max rel dev 1.701e-14
ok     cscd_cascade   astro              max rel dev 1.636e-15
ok     cscd_cascade   atmospheric_veto   max rel dev 4.432e-15
ok     cscd_cascade   galactic           max rel dev 6.834e-15
ok     cscd_muon      astro              max rel dev 1.270e-15
ok     cscd_muon      atmospheric_veto   max rel dev 1.693e-13
all compared components agree
```

### Recorded µ sums (events), 3D defaults point

| sample | astro | atmospheric(_veto) | galactic |
|---|---|---|---|
| tracks | 2887.1783 | 698578.9545 | 351.4272 |
| cscd_cascade | 697.6832 | 9902.7309 | 76.2660 |
| cscd_muon | 123.1199 | 8251.9248 | — (2D, no galactic component) |

The atmospheric sums are unchanged from the Phase-2 defaults point, as they must be: the RA
broadcast divides by `n_ra` and repeats `n_ra` times, which conserves the total exactly.

### Three harness traps this gate walked into, in order

Each produced a plausible-looking few-percent disagreement rather than an obvious failure, so
they are worth knowing about before trusting any future run of this comparison:

1. **Comparing a fitted prediction against a defaults-point dump.** `LLHFit` always minimises, so
   the comparison MUST go through `make_probe_config.py`. Skipping it showed up as a uniform
   sub-percent offset on every component, including samples the change under test never touched.
2. **A component defined but not activated.** `make_3d_config.py` added the galactic template
   under `components:` but not to `analysis.components`, which is the list NNMFit actually builds.
   Every galactic dump was then identically zero and `total - no_galactictemplate_cringefits`
   compared our real prediction against 0.0.
3. **A stale component name in `dump_histograms.sh`'s `ALL`.** After the astrophysical component
   was renamed `astro` -> `astro_brokenPL`, `excluded_except()` no longer excluded it, so every
   per-component dump silently carried the astrophysical flux on top. The tell was arithmetic:
   the `cscd_muon` `conventional_veto` dump came out at exactly its own value plus the astro sum.

The general lesson is that this harness fails quietly. Recording the µ sums above is what makes
the next regression detectable at a glance.
