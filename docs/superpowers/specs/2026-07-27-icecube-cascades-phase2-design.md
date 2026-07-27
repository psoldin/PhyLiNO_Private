# IceCube cascade samples (Phase 2)

**Date:** 2026-07-27
**Status:** Design, pre-implementation
**Scope:** Phase 2 of 3. Phase 1 (config-driven multi-sample composite + runtime binning) is
complete, including its carry-over cleanup (`8fb3460`).

## Goal

Reproduce the **structure of NNMFit `Fit_Configuration_Combined.yaml`**: three analysis samples
(`tracks`, `cscd_cascade`, `cscd_muon`) fitted jointly, with the veto-reweighted atmospheric
components the cascades use, the two muon templates, per-sample SnowStorm detector gradients, and
NNMFit's parameter priors. Phase 3 (galactic plane, RA axis) builds on this.

### Scope decisions

1. **Full structural parity, Asimov-first.** All components above are implemented. Real-data
   histogram loading is the last implementation task (the data parquets are local and the code path
   is small), so a later data fit needs no new design.
2. **Acceptance gate: NNMFit run locally, at two levels.** The environment and a reference fit
   already exist (see "Reference fit" below), so the gate is:
   - **Per bin, per component** — NNMFit's shipped `make_histogram.py` run once per sample and once
     per component (isolating a component by putting the others in that sample's
     `excluded_components`), diffed against our per-bin prediction at the same parameter values.
     Both sides are FP64 CPU, so agreement is expected at ~1e-10 relative, and a per-component
     mismatch localises the bug immediately.
   - **End to end** — our fitted parameters versus NNMFit's recorded fit, plus the seed-independent
     check of evaluating our −2lnL *at NNMFit's fitted point* and comparing against NNMFit's value
     there.
   Hand-computed `ICTests` assertions stay as the fast inner loop.
3. **Priors, steps and livetimes migrate to NNMFit's values everywhere**, and the recorded baselines
   are re-recorded once, deliberately (see §9).
4. **Out of scope:** RA / galactic (Phase 3), the `cscd_hybrid` sample, pseudo-experiments, NNMFit's
   `log10` bincount transformation (`bincount_transformation: plain` is what the combined config
   uses), and parameter bounds.
5. **In scope after inspection: the oscillation hook** (§10). It was going to be left as a no-op, but
   the combined config applies it to conv and prompt, so parity is impossible without it — and it
   turns out to be a static per-event factor, cheap to support.

## Reference fit (already available)

The NNMFit fit this phase must reproduce has been run locally, so the target numbers are known
before any code is written.

- **Environment:** `/Users/soldin/Projects/IceCube/NNMFit/.venv` (python 3.11.15, `aesara` 2.9.4)
  — no environment work needed.
- **Config:** `/Users/soldin/Downloads/Fit_Configuration_Combined_macOS.yaml`. Already local-pathed,
  and in NNMFit's **newer schema**: `analysis.datasets` (a list) instead of `detector_configs`, and
  top-level `systematics:` / `datasets:` sections instead of one flat `config:` block.
  `excluded_components` stays a comma-separated string inside `datasets.<name>`. Datasets live at
  `$NF/datasets/{tracks,cscd_cascade,cscd_muon}_ftp_baseline` with data under `$NF/datasets/data/`
  — the layout NNMFit itself expects, so our configs should point there too.
- **Result:** `/Users/soldin/Projects/IceCube/NNMFit/NNMFit/Output.pickle`, a dict with
  `llh_value`, `res_dict` (18 fitted values), `minimizer_seeds`, `minimizer_info` and the full
  `settings`. It is a **real-data** fit (`analysis_type: data`, `llh: SAYLLH`), converged with
  LBFGSB (`success: True`, 962 evaluations, 848 iterations), `llh_value = 3997.6142420984984`.

| parameter | fitted | parameter | fitted |
|---|---|---|---|
| `astro_norm` | 2.0102235093493097 | `effective_veto` | 0.14108708515445395 |
| `gamma_astro` | 2.4973509081979963 | `muongun_norm` | 1.1362703633930007 |
| `conv_norm` | 1.1862521739969658 | `muon_norm` | 1.827017996655897 |
| `prompt_norm` | **0.0** (at its range boundary) | `dom_eff` | 1.01730784376685 |
| `delta_gamma` | 0.07312317645432367 | `ice_abs` | 0.9951836691033408 |
| `CR_grad` | 0.5278438102260666 | `ice_scat` | 1.0048480464614487 |
| `barr_h` | −0.059121890824429764 | `ice_holep0` | 0.19797631957351045 |
| `barr_w` | 0.2941434915474592 | `ice_holep1` | −0.06642331999754497 |
| `barr_y` | 0.048184445442831736 | | |
| `barr_z` | −0.028540463688312537 | | |

Three consequences for this design:

1. **NNMFit seeds randomly around the defaults.** `minimizer_seeds` differs from every `default:`
   in the YAML (`astro_norm` 1.4515, `conv_norm` 0.9677, `dom_eff` 0.8938, …). Comparing fitted
   values is therefore a "same minimum" check, not a "same trajectory" one; the seed-independent
   comparison is our −2lnL evaluated at NNMFit's fitted point.
2. **`prompt_norm` sits exactly at 0.0**, its NNMFit range boundary. This framework has no bounds
   plumbing, so our fit can walk negative there. Expect a small difference in that parameter and do
   not treat it as a bug; the honest comparison is at fixed parameters.
3. **The reference fit used the no-fluctuation MuonGun template**
   (`cscd_muongun_ALL_KDE_5up_manual_ssq_no_fluct.pickle`, whose `template_fluctuation` is `None`),
   so its SAY σ² carries no template term. The parity config must use the same variant, and
   `TemplateFlux` must accept a missing fluctuation (exported as zeros). The fluctuation-carrying
   variant also exists (σ sums 1.152e-06 s⁻¹ cascade, 1.095e-06 s⁻¹ muon) for later comparison.
4. **Tracks really does use `muontemplate`** in the reference fit (`muon_norm` fitted to 1.827 under
   its 1.0 ± 0.5 prior), so exporting the Corsika `template_2d` pickle is required for parity, not
   optional.

## What the combined config actually asks for (measured, not assumed)

| | tracks | cscd_cascade | cscd_muon |
|---|---|---|---|
| bins | 45 × 33 = 1485 | 21 × 7 = **147** | 1 × 1 = **1** |
| energy axis | log10 E (2.5, 7.0) | log10 E (2.8, 7.0) | log10 E (2.6, 4.8) |
| zenith axis | cos θ (−1, 0.0872) uniform | cos θ **non-uniform** | cos θ (−1, 1) |
| reco | `energy_truncated`, `zenith_MPEFit` | `energy_monopod`, `zenith_monopod` | `energy_monopod`, `zenith_monopod` |
| livetime [s] | 410978234.97 | 330315015.11 | 330315015.11 |
| components | astro, conventional, prompt, muontemplate | astro, conventional_veto, prompt_veto, muon | astro, conventional_veto, prompt_veto, muon |
| gradients | `snowstorm_ftp_all.pickle` | `snowstorm_ftp_all_cscd_5up.pickle` | same `_5up` file | 

NNMFit binning strings are `(min, max, N_edges, spacing)`; our config stores **bin counts**
(N_edges − 1). The cascade `cscd-cos_5up` spacing is a hardcoded non-uniform edge list
(`NNMFit/binning/rectangular_binning.py`): cos θ ∈
`[-1, -0.76, -0.52, -0.28, -0.04, 0.2, 0.6, 1.0]`.

Facts confirmed from the local files:

- Both cascade MC baselines carry every column needed: `energy_monopod`/`zenith_monopod`
  (+`ra_monopod` for Phase 3), `powerlaw`, `mceq_{conv,pr}_{H4a,GST4}_SIBYLL23c`, the four
  `barr_*_mceq_H4a_SIBYLL23c` conventional slopes, and all six veto coefficients
  `log_PF_at100GeV_{conv,pr}_{a,b,c}`. Row counts: cascade 1,409,344; muon 587,538.
- The tracks baseline has **no** `log_PF` columns — consistent with tracks excluding the veto
  components, and a reason component-gated column reads (Phase 1 carry-over) had to land first.
- **NNMFit's standard mask is a no-op on both cascade MC baselines**: 0 of 1,409,344 and 0 of
  587,538 rows fail `energy_monopod_exists == 1 && energy_monopod_fit_status == 0 &&
  reco_dir_exists == 1 && reco_dir_fit_status == 0`. The MC is pre-cut, so the loader does not
  need a mask. The same count must be re-run on the *data* parquets in the data task.
- Muon templates are per-bin **rates** with a per-bin fluctuation:
  `cscd_muongun_ALL_KDE_5up_manual_ssq.pickle` is keyed by detector config and holds `template`
  (147 or 1 values), `template_fluctuation`, `energy_bins`, `zenith_bins` (ascending cos θ, equal
  to the 5up edges). The tracks Corsika pickle is single-dataset with `template_2d`.
- Gradient pickles are keyed by detector config, then by systematic name (`DOMEfficiency`,
  `IceAbsorption`, `IceScattering`, `HoleIceForward_p0`, `HoleIceForward_p1`), each with
  `gradient`, `gradient_error`, `factor`, `split_value` and `cross_correlations` per other
  systematic, plus the `binning`/`settings` used — enough to validate against our config at
  export time.

## Architecture

Phase 1's shape is unchanged: `ICLikelihood` (composite, pulls once) → one `SampleLikelihood` per
enabled sample → flux components. Phase 2 grows the component vocabulary and makes the last two
per-bin components runtime-sized and per-sample.

### 1. Non-uniform axes (`io::ic::Axis`)

`Axis` gains `std::vector<double> edges`. Empty means the existing uniform `(lo, hi, n_bins)`
fast path; non-empty means explicit edges, `lo`/`hi`/`n_bins` derived from them.
`Axis::index()` keeps the uniform arithmetic path and uses `std::ranges::upper_bound` for the
edge path. `parse_axis` accepts a second spelling: `"[-1.0, -0.76, ..., 1.0]"` (≥ 2 ascending
edges). Config spelling stays declarative — no named "cscd-cos_5up" magic constant, because the
edge list in the config is self-documenting and reusable.

### 2. Component vocabulary

`io::ic::component` grows to: `astro`, `conventional`, `prompt`, `conventional_veto`,
`prompt_veto`, `muontemplate`, `muon`. Mapping to units:

| declared components | unit built |
|---|---|
| `astro` | `PowerlawFlux` |
| `conventional` + `prompt` | `AtmosphericFlux` (veto off) |
| `conventional_veto` + `prompt_veto` | `AtmosphericFlux` (veto on, reads `log_PF_*`) |
| `muontemplate` | `TemplateFlux` with `MuonNorm` |
| `muon` | `TemplateFlux` with `MuonGunNorm` |

`parse_samples` validation extends: conv/prompt must be declared as a pair (already), veto conv/
prompt likewise, and a sample may declare **at most one** atmospheric variant (plain *or* veto) —
NNMFit's `excluded_components` guarantees this and mixing them would double-count. A sample may
declare at most one template component.

### 3. Veto: `AtmosphericFlux` gains an optional per-event reweight

The veto is a multiplicative per-event factor on exactly the weights `AtmosphericFlux` already
computes, so it belongs there rather than in a parallel class that would duplicate the whole
conv+prompt kernel. From `NNMFit/parameters/veto_threshold.py`:

```
e   = rescale_energy * 10^(VetoThreshold) - anchor_energy        # both 100 GeV in the config
PF_conv_i   = 10^(a_conv_i   + b_conv_i   * e + c_conv_i   * e^2)
PF_prompt_i = 10^(a_pr_i     + b_pr_i     * e + c_pr_i     * e^2)
conv_i   *= PF_conv_i
prompt_i *= PF_prompt_i
```

`e` is scalar per evaluation; only the six per-event coefficient columns are new. The recalculation
trigger gains `VetoThreshold`. The Metal and CUDA kernels take the six extra buffers plus a
`use_veto` flag and a `veto_e` scalar in the params struct; when veto is off the six slots are
bound to the already-uploaded `e_true` handle (the existing "always bind, never read" convention)
and the branch is skipped. `ICSample` gains `veto_conv[3]` / `veto_prompt[3]` columns, read only
when the sample declares a veto component.

### 4. `TemplateFlux` (generalises the scaffolded `MuonTemplate`)

A per-bin template, runtime-sized from the sample's `Binning`, parameterised by one norm index:

```
mu_b  = norm * template_b * livetime
ssq_b = (norm * fluctuation_b * livetime)^2      # only when SAY is active
```

matching `NNMFit/core/histogram_builder.py` (`ssq += (hist_fluctuation * livetime)**2`, with the
norm folded into the fluctuation graph). Templates are rates, so the livetime scaling mirrors what
`ICDataBase` already does to the per-event columns. A template pickle may carry
`template_fluctuation: None` (the `_no_fluct` MuonGun variant the reference fit used); the exporter
writes zeros for it and the σ² term then vanishes, which is exactly NNMFit's behaviour when
`make_fluctuations_graph` returns `None`. O(nBins) — CPU only, no GPU path. The loader
asserts the template's bin count equals the sample's `total_bins()` and (when the export carries
edges) that the edges match, since a silently mis-binned template is the obvious failure mode.

### 5. `DetectorSystematics`: runtime-sized, per-sample, µ **and** σ²

Histogram-level additive perturbation, applied to the sample's summed prediction (NNMFit's
`hist_parameter_overall: True` path applies it to total µ, after all components — `skip_syst`
only suppresses *per-component* systematics, which the combined config does not use):

```
D_k    = p_k - split_k                                   # k in {DOMEff, IceAbs, IceScat, HoleIceP0, HoleIceP1}
mu_add_b  = sum_k D_k * gradient_k_b * lt_scale
ssq_add_b = sum_k (D_k * gradient_error_k_b * lt_scale)^2
          + 2 * sum_{i<j} D_i * D_j * cov_ij_b
cov_ij_b  = factor_i * factor_j * lt_scale^2 * (err(up_i,up_j)^2 + err(lo_i,lo_j)^2
                                                - err(lo_i,up_j)^2 - err(up_i,lo_j)^2)
```

`lt_scale = livetime_analysis / livetime_gradients` (1.0 when they agree; the export script emits
it). The histogram↔gradient covariance term is correctly omitted: the FTP gradient configs set
`external_gradients: True`. µ is clipped at zero after the addition (already the behaviour).

### 6. `SampleLikelihood` becomes the assembly point

Members become: optional `PowerlawFlux`, optional `AtmosphericFlux`, optional `TemplateFlux`,
optional `DetectorSystematics`. Per evaluation:

1. per-event fluxes `check_and_recalculate`;
2. `m_Predicted[b] = astro_b + atmo_b` (absent components contribute nothing);
3. per-event ssq (astro + atmo summed **before** squaring — unchanged NNMFit rule);
4. add template µ and σ²;
5. add detector-systematics µ and σ²;
6. clip µ at zero, then SAY or Poisson.

No allocation in the hot path: every buffer is sized in the constructor. The template and
gradient contributions are recomputed only when their parameters changed, like the fluxes.

### 7. Parameter layout

The flat `params::ic` enum stays the index source of truth and grows by four, to 18:

- `VetoThreshold` — shared across the two cascade samples (one NNMFit `effective_veto` anchor).
- `MuonGunNorm` — cascade MuonGun `muon` norm, distinct from the tracks Corsika `MuonNorm`.
- `HoleIceP0`, `HoleIceP1` — the FTP gradient pickles carry them and their split values are
  **non-zero** (0.24901831812365854, −0.05678798504997925), so omitting them would bias the
  gradient sum.

Detector-systematics parameters stay **shared** across samples (NNMFit uses the same parameter
names for every detector config; only the gradient *file* differs per sample), so a per-sample
`DetectorSystematics` reads shared indices against its own gradients.

### 8. Config layout

Per sample, added keys (all optional, defaults preserve today's behaviour):

```json
"cscd_cascade": {
  "enabled": true,
  "binning": "cscd_cascade_2d",
  "parquet": ".../dataset_cscd_cascade_FTP_baseline_wCoords.parquet",
  "data":    ".../data_cscd_cascade_FTP_Monopod_wCoords.parquet",
  "livetime": 330315015.11,
  "components": "astro, conventional_veto, prompt_veto, muon",
  "Template": { "File": ".../cscd_muongun_cascade.txt", "Norm": "MuonGunNorm" },
  "Gradients": { "File": ".../gradients_cscd_cascade.txt" },
  "Branches": { "RecoEnergy": "energy_monopod", "RecoZenith": "zenith_monopod" }
}
```

Binnings gain the edge spelling:

```json
"cscd_cascade_2d": {
  "axes": "Log10Energy, CosZenith",
  "Log10Energy": "(2.8, 7.0, 21)",
  "CosZenith":   "[-1.0, -0.76, -0.52, -0.28, -0.04, 0.2, 0.6, 1.0]"
}
```

`VetoAnchorEnergy` / `VetoRescaleEnergy` live at the `IceCube` level (both 100.0, from the
NNMFit `additional` block).

Anchor configs differ only in `enabled` flags and which parameters are `Fixed`:
`config_icecube_tracks_cpu.json` (also the regression oracle), `config_icecube_cascades.json`,
`config_icecube_combined.json`. Every one of them carries NNMFit's livetimes, priors and starting
values (§9) — one source of truth, no config that is deliberately "wrong".

### 9. Priors, steps and the `StepWidth` overload

Today `InputParameter::Parameter` reads `StartValue` and `StepWidth`, and `StepWidth` is used for
**both** the Minuit step size and the Gaussian pull σ (`ICLikelihood::setup_pulls` reads
`parameters[i].uncertainty()`), while the pull's central value is the start value. That conflation
makes NNMFit's parameter set inexpressible: `muon_norm` has prior 1.0 ± 0.5 but wants a much smaller
step, and `delta_gamma` needs a small step with *no* prior at all.

`Parameter` therefore gains two optional keys, defaulting to today's behaviour so no other
experiment's config changes meaning:

```cpp
        , m_PriorValue(parameter.get<double>("PriorValue", m_Value))
        , m_PriorWidth(parameter.get<double>("PriorWidth", m_Uncertainty))
```

with `prior_value()` / `prior_width()` accessors. `ICLikelihood::setup_pulls` switches to those two;
`StepWidth` goes back to meaning only the minimiser step. Double Chooz keeps reading
`value()`/`uncertainty()` and is untouched — its configs specify neither new key, so the defaults
reproduce its current pulls exactly.

NNMFit's parity values, from the combined YAML:

| parameter | NNMFit start | NNMFit prior | ours before |
|---|---|---|---|
| `astro_norm`, `gamma_astro` | 1.5, 2.4 | none | same start, none |
| `conv_norm` | 1.0 | **none** | 0.4 |
| `prompt_norm` | **0.5** | **none** | 1.0, prior 0.5 |
| `delta_gamma` | 0.0 | **none** | 0.05 |
| `barr_h` / `barr_w` / `barr_y` / `barr_z` | 0.0 | 0.15 / 0.4 / 0.3 / 0.12 | 1.0 each |
| `CR_grad` | 0.0 | 1.0 | 1.0 |
| `muon_norm` (Corsika) | 1.0 | 1.0 ± 0.5 | fixed |
| `muongun_norm`, `effective_veto` | 1.0, 0.0 | none | — |
| det-sys (`dom_eff`, `ice_abs`, `ice_scat`) | 1.0 | none | fixed |
| `ice_holep0`, `ice_holep1` | 0.24901831812365854, −0.05678798504997925 | none | — |

Livetimes also become NNMFit's: tracks 410978234.97 s (was 3.0e8), both cascade samples
330315015.11 s.

**Consequence for the golden gate, handled deliberately.** The recorded baseline
(`chi2 = -6366527.142871824`, Asimov 514973) was produced with the old priors and livetime, so the
migration invalidates it. It stays the bit-for-bit oracle through every mechanical task, and is
re-recorded **once**, in the migration task, with two checks that the change is understood rather
than merely accepted: the Asimov total must scale by exactly 410978234.97 / 3.0e8 = 1.36993 (a
livetime is a linear factor on every weight), and running the migrated config with the *old*
livetime and priors must still reproduce the old baseline bit-for-bit. Both numbers are recorded in
that task's commit message, and the pre-migration values stay in this document as the historical
reference.

NNMFit's parameter *ranges* (`range:` entries, e.g. `effective_veto` ∈ [−1.301, 1.301]) are recorded
in the configs as comments only: this framework has no bounds plumbing, and adding it is not in
scope.

### 10. Oscillations: a required parity item, not a no-op

The combined config attaches `OscillationsHook` to **both** the conventional and prompt components,
with `NuCraft_OscillationProb.pickle`. Our `ICLikelihood` currently logs that it is unimplemented and
carries on — which cannot reproduce the reference fit. Reading
`NNMFit/fluxes/flux_hooks.py:96-155` shows the hook is much cheaper than "oscillations" suggests:

- it accounts for **νμ disappearance only**, applying the νμ/ν̄μ survival probability;
- the probability comes from a 2-D spline in (log10 E_true, zenith_true) per particle type
  (±14), evaluated **once at load time**;
- it multiplies the component's **baseline weights** in place — a static per-event factor with no
  parameter dependence and therefore no hot-path cost.

Design: a Python step evaluates the spline exactly (scipy, in the NNMFit venv) for every event of a
sample and writes a single-column parquet sidecar, row-aligned with that sample's baseline parquet.
`ICDataBase` reads the sidecar when the sample config names one and multiplies `conv_baseline`,
`conv_alt`, `prompt_baseline`, `prompt_alt` and the four Barr slopes by it, before the livetime
scaling. Exactness beats a C++ spline reimplementation, and row alignment is guaranteed because both
sides read the same file in the same order. The factor applies to atmospheric components only —
`astro_baseline` is untouched, matching the config, which attaches the hook to conv and prompt but
not to `astro`.

Alternative considered and rejected: exporting the spline on a dense grid and interpolating in C++.
It adds an interpolation error to a quantity we can have exactly, for no benefit — the sidecar is a
local development artefact like the parquets themselves.

### 11. Input-file pipeline

`tools/export_nnmfit_inputs.py` (stdlib `pickle` + `numpy` only — no pyarrow, which is not
installed locally) converts the pickles into plain text that C++ reads with no new dependency:

- **Template:** header `# template bins <N> livetime_hint <t>` then N lines `value fluctuation`,
  plus the bin edges as comments for the loader's assertion.
- **Gradients:** header `# gradients bins <N> params <K> lt_scale <s>`, then per systematic a
  `# param <name> split <v>` line and N `gradient gradient_error` lines, then per pair a
  `# cov <name_i> <name_j>` line and N values.

The script takes the detector-config key and the target bin count, and fails loudly when the
pickle's binning disagrees with what was asked for. It handles both template layouts: the
multi-dataset MuonGun pickles (keyed by detector config, `template` / `template_fluctuation`, the
latter possibly `None`) and the single-dataset Corsika tracks pickle (`template_2d`, no
detector-config level). Generated files are untracked (like the parquets); the script is committed
so the conversion is reproducible.

## Data flow (unchanged shape)

Startup: parse `Binnings`/`Samples` → `ICDataBase` loads each enabled sample's parquet (only the
declared components' columns) → assign bins via that sample's `Binning` → CSR sort → build one
`SampleLikelihood` per sample (fluxes upload their columns to the shared GPU backend; templates
and gradients load their text files) → Asimov (or, with the last task, real data) per sample.

Per evaluation: composite resets the shared `ParameterWrapper`, sums each sample's partial −2lnL,
adds the Gaussian pulls once.

## Testing

- **NNMFit diff (the gate).** A local venv (homebrew `python3.12`, `pip install -e ~/Projects/
  IceCube/NNMFit`) runs NNMFit's `make_histogram.py` on a copy of the combined YAML whose `/net/...`
  paths are rewritten to the local files. Dumps: total µ and σ² per sample, plus one run per
  component with every other component in `excluded_components`. Our per-bin `prediction` for the
  same sample, at the same parameter values, must agree per component. FP64 CPU path, so agreement
  is expected at ~1e-10 relative; a per-component mismatch localises the bug immediately (a
  transposed template, a wrong veto sign, a livetime applied twice).
- **Golden gate:** `configs/config_icecube_tracks_cpu.json` must stay bit-for-bit
  (`chi2 = -6366527.142871824`, Asimov 514973) after every mechanical task, and is re-recorded once
  in the prior/livetime migration task under the two checks described in §9. This is the same oracle
  Phase 1 established; it protects the tracks path while cascade code lands around it.
- **`ICTests` additions** (all pure, no parquet):
  - non-uniform `Axis::index` against a hand-written `upper_bound` reference, including the exact
    5up edges, boundary values and under/overflow;
  - `Binning` with mixed uniform + edge axes; `total_bins()` for the cascade grids (147, 1);
  - veto factor: hand-computed `10^(a + b·e + c·e²)` for two coefficient triples at
    `VetoThreshold ∈ {0, ±0.5}`, compared with `AtmosphericFlux` output on a synthetic sample, and
    the invariant that veto-off equals the Phase 1 result exactly;
  - `TemplateFlux`: µ = norm · template · livetime per bin, ssq = square of it with the
    fluctuation; bin-count mismatch throws;
  - `DetectorSystematics`: µ and σ² against the formulas above for two systematics with a known
    covariance term; zero perturbation ⇒ zero delta;
  - component-vocabulary validation: veto pair required, plain+veto rejected, two templates
    rejected;
  - single-bin sample (cscd_muon geometry) end to end through `SampleLikelihood`.
- **Integration:** cascade-only and combined Asimov runs recorded as new baselines; the combined
  total must equal the sum of the per-sample partials at fixed parameters (extends the Phase 1
  composite check to three samples with different binnings).
- **Backend parity:** CPU vs Metal on the combined config, tolerance as recorded for FP32
  (meaningful parameters < 1e-3).
- **Sanity anchors against NNMFit numbers we can compute:** the MuonGun template sums
  (2.3425769796686227e-06 s⁻¹ for cascade, 5.4694021304430105e-05 s⁻¹ for muon) times each
  livetime must equal the template component's Asimov contribution.
- **Prior plumbing:** an `ICTests` case asserting that a parameter with only `StepWidth` keeps
  today's pull (`prior_value == StartValue`, `prior_width == StepWidth`) and that explicit
  `PriorValue`/`PriorWidth` override it — the compatibility guarantee Double Chooz relies on.

## Risks

1. **Template zenith orientation.** NNMFit compares `template.zenith_bins` with
   `cos(configured zenith edges)[::-1]`. Our binning stores ascending cos θ, which matches the
   pickle's ascending edges — but a reversed template would still sum correctly per bin while
   being wrong per bin. The loader's edge assertion is what catches this; keep it strict.
2. **Row-major flattening of templates and gradients.** Both are flat 147-vectors; our flat index
   is energy-outer, zenith-inner. If NNMFit's flattening is the other way round the fit will look
   plausible and be wrong. The export script must assert the shape it flattens and record the
   order it assumed; the first cascade Asimov run is compared against a per-axis marginal to
   confirm.
3. **`cscd_muon` is a single bin.** Threadgroup-per-bin GPU kernels launch one group for ~590k
   events; correct but slow, and any "bins > 0" assumption gets exercised hard. The CPU path is
   the reference here.
4. **Cross-gradient covariance terms** are the least-tested part of the gradient formula and only
   matter for SAY. They are implemented but flagged: if a cascade SAY fit misbehaves, disabling
   just the covariance sum is the first bisection step.
5. **Data masks on real data.** The MC needs no mask; the data parquets are unverified. The data
   task re-runs the standard-mask count before trusting any data histogram.
6. ~~The NNMFit environment is the one unbounded task.~~ **Resolved:** the venv exists and works
   (`/Users/soldin/Projects/IceCube/NNMFit/.venv`, python 3.11.15, aesara 2.9.4) and a full fit has
   already run through it. What remains is producing the per-component histogram dumps, which is
   the same machinery the fit already exercised.
8. **NNMFit's `llh_value` convention is unverified.** 3997.61 is far too small to be a raw −2lnL sum
   over 1633 bins, so it is relative to something (a saturated or first-evaluation baseline).
   Task 1 pins the convention down empirically — by evaluating NNMFit's likelihood at two parameter
   points and comparing differences — before any comparison uses absolute values. Differences of
   −2lnL between points are convention-free and are what the gate should lean on.
7. **Prior migration touches shared core.** `InputParameter` is read by Double Chooz too. The new
   keys default to the old behaviour, and the Double Chooz branch of `run_validation.sh` is the
   check; note that branch currently reports a pre-existing drift (`Output.baseline.json` is stale
   or was recorded from a WIP snapshot), so record its state *before* the migration task to avoid
   attributing that drift to this work.

## Roadmap after this

**Phase 3 — galactic plane:** activate the RA axis (`Binning` is already N-dimensional; the
loader's 2-axis guard and the fixed 2-element reco array are the only blockers), add the
`GalacticFlux` spline component, RA oversampling (~×10), `GalacticNorm`, and 3D GPU kernels.
