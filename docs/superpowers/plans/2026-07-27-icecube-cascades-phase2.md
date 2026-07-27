# IceCube Cascade Samples (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit NNMFit's combined configuration — `tracks` + `cscd_cascade` + `cscd_muon` jointly, with veto-reweighted atmospheric components, muon templates, per-sample SnowStorm gradients and NNMFit's own priors and livetimes — and prove it per bin against NNMFit run locally.

**Architecture:** Phase 1's composite stays: `ICLikelihood` sums per-sample `SampleLikelihood` partials and adds pulls once. Phase 2 grows `Axis` with explicit edges, adds an optional per-event veto reweight inside `AtmosphericFlux`, turns the two scaffolded per-bin components (`MuonTemplate`, `DetectorSystematics`) into runtime-sized per-sample units contributing to both µ and σ², grows the flat `params::ic` enum by four indices, and separates the Gaussian-pull width from the minimiser step in `InputParameter`.

**Tech Stack:** C++23, CMake, Apache Arrow/Parquet, ROOT Minuit2, boost `property_tree`, OpenMP + Metal/CUDA kernels, Python 3.12 (`pickle`, `numpy`) for the input conversion, and NNMFit (aesara) in a local venv as the reference oracle.

Design doc: `docs/superpowers/specs/2026-07-27-icecube-cascades-phase2-design.md`.

---

## Ground rules for this plan

- **No unit-test framework.** Pure logic is tested in the standalone `programs/ictests/ICTests.cpp` executable (plain `<cassert>`, registered with CTest). Do **not** add gtest/catch2. Integration validation is `programs/LLHFit` + `tools/compare_output.py` against recorded baselines.
- **The golden gate.** `configs/config_icecube_tracks_cpu.json` (CPU backend, Poisson) must reproduce `Output.ic_baseline.json` **bit-for-bit** (`compare_output.py`, tolerance `0`) after Tasks 2–9: `chi2 = -6366527.142871824`, `IC Asimov total events: 514973`. Task 10 migrates priors and livetimes to NNMFit's values and re-records that baseline **once**, under the two equivalence checks spelled out there. If a baseline file is missing, regenerate it from `HEAD~` before starting.
- **The acceptance gate** is Task 13, at two levels: our per-bin prediction versus NNMFit's per sample and per component (from the dumps Task 1 produces, at NNMFit's default *and* fitted parameter points), plus our fitted parameters versus NNMFit's already-recorded fit. Absolute likelihood values are not comparable between the two frameworks — only differences are.
- **Executable path:** `./build/programs/LLHFit/LLHFit`. **Build:** `cmake --build build -j8`. **Tests:** `ctest --test-dir build -R ICTests --output-on-failure`.
- **Never commit `docs/superpowers/**`** (spec/plan stay local). Commit code only. **No `Co-Authored-By` trailer** in any commit message.
- **`NF` below means** `/Users/soldin/Downloads/nnmfit_files`; write absolute paths out in full inside config files. Use the `$NF/datasets/...` tree — the layout NNMFit itself reads, so both sides load byte-identical files. Local inputs (all present):
  - MC: `$NF/datasets/cscd_cascade_ftp_baseline/dataset_cscd_cascade_FTP_baseline_wCoords.parquet` (1,409,344 rows), `$NF/datasets/cscd_muon_ftp_baseline/dataset_cscd_muon_FTP_baseline_wCoords.parquet` (587,538 rows), `$NF/datasets/tracks_ftp_baseline/dataset_tracks_baseline.parquet` (13,906,852 rows)
  - data: `$NF/datasets/data/data_cscd_cascade_FTP_Monopod_wCoords.parquet`, `$NF/datasets/data/data_cscd_muon_wCoords.parquet`, `$NF/datasets/data/dataset_data_tracks_IC2010_to_IC2022_no_cscd_cascade_cscd_muon_wCoords.parquet`
  - templates: `$NF/cscd_muongun_ALL_KDE_5up_manual_ssq_no_fluct.pickle` (**the variant the reference fit used**; its `template_fluctuation` is `None`), `$NF/cscd_muongun_ALL_KDE_5up_manual_ssq.pickle` (with fluctuations), `$NF/Tracks_CorsikaMuon_Fullrange_drop_5lowEbins.pickle` (single-dataset, `template_2d`)
  - gradients: `$NF/snowstorm_ftp_all.pickle` (tracks), `$NF/snowstorm_ftp_all_cscd_5up.pickle` (both cascade samples)
  - **NNMFit reference:** venv `/Users/soldin/Projects/IceCube/NNMFit/.venv` (python 3.11.15, aesara 2.9.4); config `/Users/soldin/Downloads/Fit_Configuration_Combined_macOS.yaml`; recorded fit `/Users/soldin/Projects/IceCube/NNMFit/NNMFit/Output.pickle` (`llh_value 3997.6142420984984`, converged, real data + SAY)

---

## File Structure

**New files:**
- `libraries/likelihood/IceCube/TemplateFlux.h` / `.cpp` — per-bin template component (µ and σ²), runtime-sized, one norm parameter. Replaces `MuonTemplate.{h,cpp}` (deleted in Task 8).
- `tools/export_nnmfit_inputs.py` — pickle → plain-text converter for templates and gradients.
- `tools/export_oscillation_factors.py` — per-event νμ survival factors → parquet sidecar (Task 7b).
- `tools/nnmfit_oracle/` — the NNMFit reference harness: `README.md`, `read_fit_result.py`, `nnmfit_set_excluded.py`, `dump_histograms.sh`, `compare_to_nnmfit.py`.
- `configs/config_icecube_cascades.json`, `configs/config_icecube_combined.json` — cascade-only and 3-sample anchor configs.

**Modified files:**
- `libraries/io/IceCube/Binning.h` / `.cpp` — explicit-edge axes; `bin_event_counts` helper (Task 12).
- `libraries/io/IceCube/ICParameter.h` — enum grows by `MuonGunNorm`, `VetoThreshold`, `HoleIceP0`, `HoleIceP1` (14 → 18).
- `libraries/io/IceCube/SampleConfig.h` / `.cpp` — veto/template component names, per-sample `Template`/`Gradients` blocks, extended validation.
- `libraries/io/IceCube/BranchNames.h` — six `log_PF_at100GeV_*` column names.
- `libraries/io/IceCube/ICSample.h` — `veto_conv[3]`, `veto_prompt[3]` columns.
- `libraries/io/IceCube/ICDataBase.h` / `.cpp` — veto columns; per-sample data histograms (Task 12).
- `libraries/io/IceCube/ICInputOptions.h` / `.cpp` — `VetoAnchorEnergy`, `VetoRescaleEnergy`; drop the old global muon-template keys.
- `libraries/io/InputParameter.h` — optional `PriorValue` / `PriorWidth`, defaulting to `StartValue` / `StepWidth` (Task 10).
- `libraries/likelihood/IceCube/AtmosphericFlux.h` / `.cpp` — optional veto reweight, CPU + MSL + CUDA.
- `libraries/likelihood/IceCube/DetectorSystematics.h` / `.cpp` — runtime-sized, per-sample file, µ + σ², 5 systematics.
- `libraries/likelihood/IceCube/SampleLikelihood.h` / `.cpp` — owns the optional template + gradients, folds them into µ/σ², real-data setter.
- `libraries/likelihood/IceCube/ICLikelihood.cpp` — new global settings; pulls read the prior fields; real-data path.
- `libraries/likelihood/IceCube/CMakeLists.txt` — source list.
- `libraries/results/IceCube/ICWriteResults.h` — per-sample component totals (Task 11).
- `configs/config_icecube.json`, `configs/config_icecube_tracks_cpu.json`, `configs/config_icecube_tracksonly.json` — four new `Parameter` entries (Task 3), then NNMFit priors/livetimes (Task 10).
- `programs/ictests/ICTests.cpp` — all new unit tests.
- `tools/run_validation.sh` — cascade + combined regression branches (Task 11).

---

## Task 1: NNMFit reference oracle (per-component dumps + the recorded fit)

**Files:**
- Create: `tools/nnmfit_oracle/README.md`, `tools/nnmfit_oracle/nnmfit_set_excluded.py`, `tools/nnmfit_oracle/dump_histograms.sh`, `tools/nnmfit_oracle/read_fit_result.py`

No C++ changes. This task turns the already-available NNMFit run into the two reference artefacts Task 13 needs. **The environment and a converged fit already exist** — no installation, no path rewriting:

- venv: `/Users/soldin/Projects/IceCube/NNMFit/.venv` (python 3.11.15, aesara 2.9.4)
- config: `/Users/soldin/Downloads/Fit_Configuration_Combined_macOS.yaml` (already local-pathed; NNMFit's **newer schema** — `analysis.datasets` as a list, top-level `systematics:` and `datasets:` sections, no flat `config:` block)
- result: `/Users/soldin/Projects/IceCube/NNMFit/NNMFit/Output.pickle` (`llh_value = 3997.6142420984984`, converged LBFGSB, 18 fitted values in `res_dict`, randomized `minimizer_seeds`, full `settings`)

- [ ] **Step 1: Confirm the environment and read the recorded fit**

Create `tools/nnmfit_oracle/read_fit_result.py`:

```python
#!/usr/bin/env python3
"""Print an NNMFit Output.pickle fit result as JSON, for comparison with PhyLiNO.

Maps NNMFit's parameter names onto PhyLiNO's params::ic names so the two can be
diffed directly.

Usage: read_fit_result.py Output.pickle [--json OUT.json]
"""
import argparse
import json
import pickle

# NNMFit parameter name -> PhyLiNO config "Name"
NAME_MAP = {
    "astro_norm": "AstroNorm",
    "gamma_astro": "SpectralIndex",
    "conv_norm": "ConvNorm",
    "prompt_norm": "PromptNorm",
    "barr_h": "BarrH",
    "barr_w": "BarrW",
    "barr_y": "BarrY",
    "barr_z": "BarrZ",
    "CR_grad": "CRGrad",
    "delta_gamma": "DeltaGamma",
    "muon_norm": "MuonNorm",
    "muongun_norm": "MuonGunNorm",
    "effective_veto": "VetoThreshold",
    "dom_eff": "DOMEff",
    "ice_abs": "IceAbs",
    "ice_scat": "IceScat",
    "ice_holep0": "HoleIceP0",
    "ice_holep1": "HoleIceP1",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pickle_path")
    p.add_argument("--json", dest="json_path")
    args = p.parse_args()

    with open(args.pickle_path, "rb") as f:
        result = pickle.load(f)

    unmapped = sorted(set(result["res_dict"]) - set(NAME_MAP))
    if unmapped:
        raise SystemExit(f"unmapped NNMFit parameters: {unmapped} (extend NAME_MAP)")

    out = {
        "llh_value": result["llh_value"],
        "minimizer": result["minimizer_info"],
        "fitted": {NAME_MAP[k]: v for k, v in result["res_dict"].items()},
        "seeds": {NAME_MAP[k]: v for k, v in result["minimizer_seeds"].items()},
        "analysis_type": result["settings"]["analysis"]["analysis_type"],
        "llh": result["settings"]["analysis"]["llh"],
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.json_path:
        with open(args.json_path, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
```

Run:
```bash
V=/Users/soldin/Projects/IceCube/NNMFit/.venv/bin/python
$V -c "import aesara, NNMFit, sys; print(sys.version.split()[0], 'aesara', aesara.__version__)"
python3 tools/nnmfit_oracle/read_fit_result.py \
        /Users/soldin/Projects/IceCube/NNMFit/NNMFit/Output.pickle \
        --json /tmp/nnmfit_fit_reference.json
```
Expected: `3.11.15 aesara 2.9.4`, then a JSON block with `llh_value 3997.6142420984984`,
`minimizer.success true`, `analysis_type "data"`, `llh "SAYLLH"`, and 18 fitted values including
`AstroNorm 2.0102235093493097`, `SpectralIndex 2.4973509081979963`, `PromptNorm 0.0`,
`MuonNorm 1.827017996655897`, `MuonGunNorm 1.1362703633930007`,
`VetoThreshold 0.14108708515445395`. A `SystemExit` about unmapped parameters means NNMFit's config
gained a parameter this phase does not model — resolve that before continuing.

- [ ] **Step 2: Pin down NNMFit's `llh_value` convention**

3997.61 is far too small for a raw −2lnL sum over 1485 + 147 + 1 = 1633 bins, so it is relative to
some baseline. Absolute comparisons are worthless until the convention is known; differences between
parameter points are convention-free. Determine it from the source and confirm empirically:

```bash
V=/Users/soldin/Projects/IceCube/NNMFit/.venv/bin/python
grep -rn "def compute_log_L\|saturated\|llh_value" /Users/soldin/Projects/IceCube/NNMFit/NNMFit/likelihoods/ \
     /Users/soldin/Projects/IceCube/NNMFit/NNMFit/core/nnm_fitter.py | head -20
```
Then record in the README: whether NNMFit's reported value is `-2 log L`, `-log L`, or a saturated /
baseline-subtracted variant, and the sign convention. Cross-check the factor by re-evaluating
NNMFit's likelihood at the fitted point and at the seed point and comparing the difference with the
same difference computed from our fit later. Our own `ICLikelihood` already subtracts a first-call
baseline (`m_LLHBaseLine`), so **only differences are comparable on either side** — state that
explicitly in the README.

- [ ] **Step 3: Dump total and per-component histograms**

NNMFit's shipped `make_histogram.py` writes a pickle holding `mu` and `ssq` per detector config.
Isolating one component is done purely through config: put every *other* component into that
sample's `excluded_components`. Create `tools/nnmfit_oracle/nnmfit_set_excluded.py` (note the
**newer schema**: samples live under `datasets:`, and the sample list is `analysis.datasets`):

```python
#!/usr/bin/env python3
"""Write a copy of an NNMFit config with one sample's excluded_components set.

Targets the newer NNMFit schema: samples under `datasets:`, sample list in
`analysis.datasets`, `excluded_components` a comma-separated string.

Usage: nnmfit_set_excluded.py IN.yaml OUT.yaml SAMPLE "comp_a, comp_b"
"""
import sys

import yaml


def main():
    src, dst, sample, excluded = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    with open(src) as f:
        config = yaml.safe_load(f)

    datasets = config.get("datasets")
    if datasets is None or sample not in datasets:
        sys.exit(f"sample '{sample}' not under 'datasets:' in {src}")

    datasets[sample]["excluded_components"] = excluded
    # Build only this sample, so make_histogram does not touch the others.
    config["analysis"]["datasets"] = [sample]

    with open(dst, "w") as f:
        yaml.safe_dump(config, f)
    print(f"wrote {dst} (excluded: {excluded or 'none'})")


if __name__ == "__main__":
    main()
```

and `tools/nnmfit_oracle/dump_histograms.sh`:

```bash
#!/usr/bin/env bash
# Dump NNMFit's analysis histograms (mu, ssq) for the combined macOS config: once
# with every component the sample uses, then once per component with the others
# excluded. Output: /tmp/nnmfit_dumps/<sample>_<component|total>.pickle
#
# Usage: tools/nnmfit_oracle/dump_histograms.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/soldin/Projects/IceCube/NNMFit/.venv/bin/python
MAKE_HIST=/Users/soldin/Projects/IceCube/NNMFit/NNMFit/scripts/make_histogram.py
BASE=/Users/soldin/Downloads/Fit_Configuration_Combined_macOS.yaml
OUT=/tmp/nnmfit_dumps
mkdir -p "$OUT"

# sample -> the components it uses (the complement of its excluded_components)
SAMPLES=(
  "IC86_pass2_SnowStorm_v2_tracks:astro conventional prompt muontemplate"
  "IC86_pass2_SnowStorm_v2_cscd_cascade:astro conventional_veto prompt_veto muon"
  "IC86_pass2_SnowStorm_v2_cscd_muon:astro conventional_veto prompt_veto muon"
)
ALL="conventional conventional_veto prompt prompt_veto muon muontemplate astro"

for entry in "${SAMPLES[@]}"; do
  sample="${entry%%:*}"
  used="${entry#*:}"

  # total: exclude everything this sample does not use
  excluded=""
  for c in $ALL; do
    case " $used " in *" $c "*) ;; *) excluded="${excluded:+$excluded, }$c" ;; esac
  done
  "$PY" "$HERE/nnmfit_set_excluded.py" "$BASE" "$OUT/cfg_${sample}_total.yaml" "$sample" "$excluded"
  "$PY" "$MAKE_HIST" -c "$OUT/cfg_${sample}_total.yaml" -o "$OUT/${sample}_total.pickle"

  # one run per component: exclude all but that one
  for keep in $used; do
    excluded=""
    for c in $ALL; do
      [ "$c" = "$keep" ] || excluded="${excluded:+$excluded, }$c"
    done
    "$PY" "$HERE/nnmfit_set_excluded.py" "$BASE" "$OUT/cfg_${sample}_${keep}.yaml" "$sample" "$excluded"
    "$PY" "$MAKE_HIST" -c "$OUT/cfg_${sample}_${keep}.yaml" -o "$OUT/${sample}_${keep}.pickle"
  done
done

echo "dumps in $OUT"
```

Run:
```bash
chmod +x tools/nnmfit_oracle/dump_histograms.sh
tools/nnmfit_oracle/dump_histograms.sh
ls -la /tmp/nnmfit_dumps/*.pickle
```
Expected: 3 `*_total.pickle` plus 4 per-component pickles per sample (12 more), each holding `mu`
and `ssq` arrays of that sample's bin count (1485 / 147 / 1). If `make_histogram.py` needs extra
arguments in this NNMFit version, read its `make_parser()` and adjust the invocation — do not
work around it by editing NNMFit. Verify one dump:
```bash
/Users/soldin/Projects/IceCube/NNMFit/.venv/bin/python - <<'EOF'
import pickle, numpy as np
d = pickle.load(open('/tmp/nnmfit_dumps/IC86_pass2_SnowStorm_v2_cscd_cascade_total.pickle', 'rb'))
print(sorted(d.keys()))
print('mu shape', np.shape(d['mu']), 'sum', np.sum(d['mu']))
print('ssq shape', np.shape(d['ssq']), 'sum', np.sum(d['ssq']))
EOF
```
Expected: a `mu` of 147 values with a physically sensible sum (thousands of events, not 0 and not
1e30). Record every per-sample and per-component `mu` sum in the README.

Note which parameter values the dumps correspond to: `make_histogram` evaluates at the config's
`default:` values (`astro_norm` 1.5, `gamma_astro` 2.4, `conv_norm` 1.0, `prompt_norm` 0.5,
`effective_veto` 0.0, det-sys at their split values), **not** at the fitted point and **not** at the
randomized seeds. Task 13 must evaluate our prediction at exactly those values.

- [ ] **Step 4: Also dump at the fitted point**

The defaults are a weak test of the veto and gradient code (`effective_veto = 0` makes the veto
exponent's linear and quadratic terms vanish; det-sys deltas are zero at the split values). Produce a
second set at the fitted values by writing them into a config copy:

```bash
V=/Users/soldin/Projects/IceCube/NNMFit/.venv/bin/python
$V - <<'EOF'
import json, pickle, yaml

fit = pickle.load(open('/Users/soldin/Projects/IceCube/NNMFit/NNMFit/Output.pickle', 'rb'))
config = yaml.safe_load(open('/Users/soldin/Downloads/Fit_Configuration_Combined_macOS.yaml'))

# Overwrite each parameter's `default:` with its fitted value, wherever it appears.
def set_defaults(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, dict) and 'default' in value and key in fit['res_dict']:
                value['default'] = float(fit['res_dict'][key])
            set_defaults(value)
    elif isinstance(node, list):
        for item in node:
            set_defaults(item)

set_defaults(config)
# Systematics defaults are lists in the parameters order.
for name, block in config.get('systematics', {}).items():
    order = block.get('parameters', [])
    block['default'] = [float(fit['res_dict'][p]) for p in order]
yaml.safe_dump(config, open('/tmp/nnmfit_fitted_point.yaml', 'w'))
print('wrote /tmp/nnmfit_fitted_point.yaml')
EOF
```
then re-run the dump script against that config (temporarily point its `BASE` at
`/tmp/nnmfit_fitted_point.yaml` and its `OUT` at `/tmp/nnmfit_dumps_fitted`), and record those `mu`
sums too. Expected: the two dump sets differ, most visibly in the components carrying `effective_veto`
and the det-sys gradients — if they are identical, the parameter override did not take effect and the
comparison would be vacuous.

- [ ] **Step 5: Write the README**

Create `tools/nnmfit_oracle/README.md` documenting: the venv path and interpreter, the config and
`Output.pickle` locations, the exact commands above, the `llh_value` convention found in Step 2 (and
the warning that only *differences* are comparable, since our `ICLikelihood` subtracts a first-call
baseline), the fitted parameter table, the two dump sets with their parameter points, and the
per-component `mu` sums. State that the dumps live in `/tmp` and are regenerated by re-running the
script.

- [ ] **Step 6: Commit**

```bash
git add tools/nnmfit_oracle
git commit -m "tools: NNMFit oracle harness (per-component dumps + recorded fit reference)"
```

---

## Task 2: Non-uniform axes

**Files:**
- Modify: `libraries/io/IceCube/Binning.h`, `libraries/io/IceCube/Binning.cpp`
- Modify: `programs/ictests/ICTests.cpp`

- [ ] **Step 1: Write the failing tests**

Append to `programs/ictests/ICTests.cpp` (register both in `main()` right after `test_parse_axis_spec();`):

```cpp
// The cascade zenith axis is a hardcoded non-uniform cos-zenith edge list in
// NNMFit (rectangular_binning.py, "cscd-cos_5up"). Expressed here as explicit
// edges, Axis::index must bin by upper_bound over those edges.
static void test_non_uniform_axis_index() {
  const std::vector<double> edges{-1.0, -0.76, -0.52, -0.28, -0.04, 0.2, 0.6, 1.0};
  const Axis a = io::ic::parse_axis("CosZenith", "[-1.0, -0.76, -0.52, -0.28, -0.04, 0.2, 0.6, 1.0]");
  assert(a.n_bins == 7);
  assert(a.edges.size() == 8);
  assert(std::abs(a.lo + 1.0) < 1e-12);
  assert(std::abs(a.hi - 1.0) < 1e-12);

  // Reference: index of the bin containing cos(zenith), -1 outside [lo, hi).
  auto reference = [&edges](const double zenith_rad) -> int {
    const double cz = std::cos(zenith_rad);
    if (cz < edges.front() || cz >= edges.back()) return -1;
    for (std::size_t i = 0; i + 1 < edges.size(); ++i)
      if (cz >= edges[i] && cz < edges[i + 1]) return static_cast<int>(i);
    return -1;
  };

  for (double z : {0.0, 0.3, 0.8, 1.0, 1.2, 1.5708, 1.9, 2.4, 2.9, 3.14159, 3.2})
    assert(a.index(z) == reference(z));

  // Lower edge is inclusive, upper edge exclusive, in cos(zenith).
  assert(a.index(std::acos(-1.0)) == 0);     // cos = -1   -> first bin
  assert(a.index(std::acos(-0.76)) == 1);    // exactly an interior edge
  assert(a.index(std::acos(0.999999)) == 6); // last bin
  assert(a.index(std::acos(1.0)) == -1);     // cos = +1 == hi -> out of range
}

// A binning may mix a uniform energy axis with an explicit-edge zenith axis;
// this is exactly the cscd_cascade grid (21 x 7 = 147 bins).
static void test_mixed_binning_cascade_grid() {
  const Binning cascade({io::ic::parse_axis("Log10Energy", "(2.8, 7.0, 21)"),
                         io::ic::parse_axis("CosZenith",
                                            "[-1.0, -0.76, -0.52, -0.28, -0.04, 0.2, 0.6, 1.0]")});
  assert(cascade.total_bins() == 147);

  // 10^3 GeV is energy bin 1 ((3.0-2.8)/0.2 = 1); cos(zenith)=0.0 is zenith bin 4.
  const double reco[2] = {1000.0, 1.5707963267948966};
  assert(cascade.bin_index(reco) == 1 * 7 + 4);

  // The cscd_muon grid is a single bin.
  const Binning muon({io::ic::parse_axis("Log10Energy", "(2.6, 4.8, 1)"),
                      io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 1)")});
  assert(muon.total_bins() == 1);
  const double inside[2] = {1000.0, 1.5};
  assert(muon.bin_index(inside) == 0);
  const double too_soft[2] = {100.0, 1.5};
  assert(muon.bin_index(too_soft) == -1);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cmake --build build -j8`
Expected: compile error — `Axis` has no member `edges`, and `parse_axis` cannot take the bracket form (it currently `sscanf`s `"%lf %lf %d"`).

- [ ] **Step 3: Extend `Axis` in `libraries/io/IceCube/Binning.h`**

Replace the `Axis` struct with:

```cpp
  /**
   * One analysis axis in a derived reconstructed quantity. Either a uniform grid
   * ((lo, hi, n_bins), `edges` empty) or an explicit ascending edge list
   * (`edges.size() == n_bins + 1`), which is what NNMFit's non-uniform cascade
   * zenith binning ("cscd-cos_5up") needs.
   */
  struct Axis {
    enum class Kind { Log10Energy, CosZenith, Ra };
    Kind                kind;
    double              lo;
    double              hi;
    int                 n_bins;
    std::vector<double> edges;  // empty => uniform

    [[nodiscard]] bool   uniform() const noexcept { return edges.empty(); }
    [[nodiscard]] double step() const noexcept { return (hi - lo) / n_bins; }
    [[nodiscard]] double project(double raw_value) const noexcept;
    [[nodiscard]] int    index(double raw_value) const noexcept;
  };
```

Keep the `parse_axis` declaration but extend its comment:

```cpp
  /**
   * Parse an axis spec for the named kind ("Log10Energy"|"CosZenith"|"Ra"):
   *   "(lo, hi, n_bins)"        uniform grid
   *   "[e0, e1, ..., eN]"       explicit ascending edges, N bins
   */
  [[nodiscard]] Axis parse_axis(std::string_view kind, std::string_view spec);
```

- [ ] **Step 4: Implement in `libraries/io/IceCube/Binning.cpp`**

Replace `Axis::index` and `parse_axis` with:

```cpp
  int Axis::index(const double raw_value) const noexcept {
    const double v = project(raw_value);
    if (v < lo || v >= hi) return -1;
    if (uniform()) return static_cast<int>((v - lo) / step());
    // First edge strictly greater than v; v >= lo so the result is >= 1.
    const auto upper = std::ranges::upper_bound(edges, v);
    return static_cast<int>(std::distance(edges.begin(), upper)) - 1;
  }

  namespace {

    Axis::Kind parse_axis_kind(const std::string_view kind) {
      if (kind == "Log10Energy") return Axis::Kind::Log10Energy;
      if (kind == "CosZenith") return Axis::Kind::CosZenith;
      if (kind == "Ra") return Axis::Kind::Ra;
      throw std::runtime_error("parse_axis: unknown axis kind '" + std::string(kind) + "'");
    }

    // "(a, b, c)" / "[a, b, c]" -> the numbers, punctuation blanked out.
    std::vector<double> parse_numbers(const std::string_view spec) {
      std::string s(spec);
      for (char& c : s)
        if (c == '(' || c == ')' || c == '[' || c == ']' || c == ',') c = ' ';
      std::istringstream  in(s);
      std::vector<double> values;
      double              v = 0.0;
      while (in >> v) values.push_back(v);
      return values;
    }

  }  // namespace

  Axis parse_axis(const std::string_view kind, const std::string_view spec) {
    const Axis::Kind k       = parse_axis_kind(kind);
    const auto       trimmed = spec.substr(spec.find_first_not_of(" \t"));
    const auto       numbers = parse_numbers(spec);

    if (trimmed.starts_with('[')) {
      if (numbers.size() < 2)
        throw std::runtime_error("parse_axis: edge list '" + std::string(spec) +
                                 "' needs at least two edges");
      if (!std::ranges::is_sorted(numbers))
        throw std::runtime_error("parse_axis: edge list '" + std::string(spec) + "' is not ascending");
      return Axis{k, numbers.front(), numbers.back(), static_cast<int>(numbers.size()) - 1, numbers};
    }

    if (numbers.size() != 3 || numbers[2] <= 0.0)
      throw std::runtime_error("parse_axis: bad spec '" + std::string(spec) +
                               "' (want '(lo, hi, n_bins)' or '[e0, e1, ...]')");
    return Axis{k, numbers[0], numbers[1], static_cast<int>(numbers[2]), {}};
  }
```

Add `#include <algorithm>`, `#include <sstream>` and `#include <vector>` at the top of `Binning.cpp`; `Binning.h` already includes `<vector>`.

- [ ] **Step 5: Run the tests to verify they pass**

Run:
```bash
cmake --build build -j8
ctest --test-dir build -R ICTests --output-on-failure
```
Expected: `ICTests: all passed`.

- [ ] **Step 6: Golden regression**

Run:
```bash
./build/programs/LLHFit/LLHFit -c configs/config_icecube_tracks_cpu.json --silent
python3 tools/compare_output.py Output.ic_baseline.json Output.json
```
Expected: exit 0, `IDENTICAL`. The uniform path must be untouched.

- [ ] **Step 7: Commit**

```bash
git add libraries/io/IceCube/Binning.h libraries/io/IceCube/Binning.cpp programs/ictests/ICTests.cpp
git commit -m "feat(icecube): support explicit non-uniform axis edges in Binning"
```

---

## Task 3: Grow the parameter layout to 18

**Files:**
- Modify: `libraries/io/IceCube/ICParameter.h`
- Modify: `configs/config_icecube.json`, `configs/config_icecube_tracks_cpu.json`, `configs/config_icecube_tracksonly.json`
- Modify: `programs/ictests/ICTests.cpp`

The four new indices are needed by Tasks 6, 8 and 9. Adding them first, with every config updated in the same commit, keeps the config's `Parameter` array and `params::ic::number_of_parameters()` in lockstep.

- [ ] **Step 1: Extend the enum**

In `libraries/io/IceCube/ICParameter.h`, replace the scaffolded block and the derived counts:

```cpp
    // --- muon templates (one norm per template kind; a sample declares at most one) ---
    MuonNorm = _last_of_Flux_,  // Corsika muon template normalization (tracks)
    MuonGunNorm,                // MuonGun template normalization (cascade samples)

    // --- veto (NNMFit effective_veto): shared by every veto-reweighted sample ---
    VetoThreshold,

    // --- SnowStorm detector gradients (shared parameter names, per-sample gradient file) ---
    DOMEff,      // DOM efficiency
    IceAbs,      // bulk ice absorption
    IceScat,     // bulk ice scattering
    HoleIceP0,   // hole-ice forward p0
    HoleIceP1,   // hole-ice forward p1
    _last_of_General_
  };

  inline constexpr int nBarrParams =
    static_cast<int>(_last_of_Barr_) - static_cast<int>(BarrH);  // = 4 (H, W, Y, Z)

  inline constexpr int nDetSysParams =
    static_cast<int>(_last_of_General_) - static_cast<int>(DOMEff);  // = 5
```

and update the trailing assertion:

```cpp
  static_assert(nBarrParams == 4, "Expected 4 Barr parameters: H, W, Y, Z");
  static_assert(nDetSysParams == 5,
    "DOMEff, IceAbs, IceScat, HoleIceP0, HoleIceP1 -- the order the exported gradient file uses");
  static_assert(number_of_parameters() == 18,
    "10 flux/atmo + 2 template norms + VetoThreshold + 5 detector params. "
    "Update every config's Parameter array and this if the layout changes.");
```

Also refresh the doc comment's trailing block so it lists the new names.

- [ ] **Step 2: Update the three existing configs**

In **each** of `configs/config_icecube.json`, `configs/config_icecube_tracks_cpu.json`, `configs/config_icecube_tracksonly.json`, replace the tail of the `"Parameter"` array (from `"MuonNorm"` onwards) with exactly:

```json
    { "Name": "MuonNorm",      "StartValue": 1.0, "StepWidth": 0.1,  "Fixed": true,  "Constrained": false },
    { "Name": "MuonGunNorm",   "StartValue": 1.0, "StepWidth": 0.1,  "Fixed": true,  "Constrained": false },
    { "Name": "VetoThreshold", "StartValue": 0.0, "StepWidth": 0.1,  "Fixed": true,  "Constrained": false },
    { "Name": "DOMEff",        "StartValue": 1.0, "StepWidth": 0.1,  "Fixed": true,  "Constrained": false },
    { "Name": "IceAbs",        "StartValue": 1.0, "StepWidth": 0.1,  "Fixed": true,  "Constrained": false },
    { "Name": "IceScat",       "StartValue": 1.0, "StepWidth": 0.1,  "Fixed": true,  "Constrained": false },
    { "Name": "HoleIceP0",     "StartValue": 0.24901831812365854,  "StepWidth": 0.05, "Fixed": true, "Constrained": false },
    { "Name": "HoleIceP1",     "StartValue": -0.05678798504997925, "StepWidth": 0.02, "Fixed": true, "Constrained": false }
```

The `HoleIceP*` start values are NNMFit's FTP split values (`Snowstorm_Gradients_FTP.default`); every new parameter is `Fixed` so no fit result changes.

- [ ] **Step 3: Add a layout test**

Append to `ICTests.cpp` (register in `main()` before `test_parse_samples();`):

```cpp
// The parameter layout is the contract between the config's Parameter array, the
// Minuit index array and every component that reads a fixed index.
static void test_parameter_layout() {
  using namespace params::ic;
  assert(number_of_parameters() == 18);
  assert(nBarrParams == 4);
  assert(nDetSysParams == 5);
  // Barr block is contiguous in {H, W, Y, Z} order.
  assert(BarrW == BarrH + 1 && BarrY == BarrH + 2 && BarrZ == BarrH + 3);
  // Detector block is contiguous in the order the exported gradient file uses.
  assert(IceAbs == DOMEff + 1 && IceScat == DOMEff + 2);
  assert(HoleIceP0 == DOMEff + 3 && HoleIceP1 == DOMEff + 4);
  // The two template norms are distinct parameters.
  assert(MuonNorm != MuonGunNorm);
}
```

- [ ] **Step 4: Build and run the tests**

Run:
```bash
cmake --build build -j8
ctest --test-dir build -R ICTests --output-on-failure
```
Expected: pass. A `static_assert` failure here means the enum edit and the assertion disagree.

- [ ] **Step 5: Golden regression**

Run:
```bash
./build/programs/LLHFit/LLHFit -c configs/config_icecube_tracks_cpu.json --silent
python3 tools/compare_output.py Output.ic_baseline.json Output.json
```
Expected: exit 0 except for the four new `parameters.*` keys, which `compare_output.py` reports as
"only in right file". That is the one allowed diff in this task. Before re-recording, confirm by eye
that `chi2` is exactly `-6366527.142871824` and every pre-existing parameter value is unchanged:
```bash
cp Output.json Output.ic_baseline.json
```
If `chi2` moved, stop: Minuit's handling of the extra fixed variables changed something and needs
understanding, not a new baseline.

- [ ] **Step 6: Commit**

```bash
git add libraries/io/IceCube/ICParameter.h configs/config_icecube.json \
        configs/config_icecube_tracks_cpu.json configs/config_icecube_tracksonly.json \
        programs/ictests/ICTests.cpp
git commit -m "feat(icecube): add MuonGunNorm, VetoThreshold and hole-ice parameters"
```

---

## Task 4: Component vocabulary for veto and templates

**Files:**
- Modify: `libraries/io/IceCube/SampleConfig.h`, `libraries/io/IceCube/SampleConfig.cpp`
- Modify: `programs/ictests/ICTests.cpp`

Purely additive config surface: the names parse and validate, nothing consumes them yet.

- [ ] **Step 1: Write the failing tests**

In `ICTests.cpp`, extend `test_parse_samples_rejects_bad_components()`'s assertion block with:

```cpp
  assert(throws(R"(, "components": "astro, conventional_veto")"));            // veto pair incomplete
  assert(throws(R"(, "components": "astro, conventional, conventional_veto, prompt, prompt_veto")"));  // plain + veto
  assert(throws(R"(, "components": "astro, muon, muontemplate")"));           // two templates
  assert(!throws(R"(, "components": "astro, conventional_veto, prompt_veto")"));
  assert(!throws(R"(, "components": "astro, conventional, prompt")"));
```

and add a new test (register it in `main()` after `test_parse_samples();`):

```cpp
// The cascade samples declare the veto variants plus the MuonGun template, and
// carry their template/gradient file paths per sample.
static void test_parse_samples_cascade_entry() {
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
  assert(samples.size() == 1);
  const io::ic::SampleConfig& c = samples[0];

  assert(c.binning.total_bins() == 147);
  assert(c.wants_astro());
  assert(c.wants_atmospheric());
  assert(c.wants_veto());
  assert(c.template_file == "muongun_cascade.txt");
  assert(c.template_norm_index == params::ic::MuonGunNorm);
  assert(c.gradient_file == "gradients_cscd_cascade.txt");
  assert(c.branches.reco_energy == "energy_monopod");
  assert(c.branches.reco_zenith == "zenith_monopod");
  assert(c.data_path == "data_cscd_cascade.parquet");
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cmake --build build -j8`
Expected: compile error — `wants_veto`, `template_file`, `template_norm_index`, `gradient_file` do not exist.

- [ ] **Step 3: Extend `SampleConfig`**

In `libraries/io/IceCube/SampleConfig.h`, extend the component namespace:

```cpp
  namespace component {
    inline constexpr std::string_view Astro            = "astro";
    inline constexpr std::string_view Conventional     = "conventional";
    inline constexpr std::string_view Prompt           = "prompt";
    inline constexpr std::string_view ConventionalVeto = "conventional_veto";
    inline constexpr std::string_view PromptVeto       = "prompt_veto";
    inline constexpr std::string_view MuonTemplate     = "muontemplate";  // Corsika, tracks
    inline constexpr std::string_view MuonGun          = "muon";          // MuonGun, cascades
  }  // namespace component
```

add the members (after `branches`):

```cpp
    // Muon template, when the sample declares "muontemplate" or "muon": the
    // exported per-bin template file and which norm parameter scales it.
    std::string template_file;
    int         template_norm_index = -1;

    // Exported SnowStorm gradient file for this sample ("" = no detector systematics).
    std::string gradient_file;
```

and replace `wants_atmospheric()` with the veto-aware trio:

```cpp
    /**
     * Conventional + prompt atmospheric flux requested, plain or veto-reweighted.
     * AtmosphericFlux computes conv and prompt in one per-event pass, so each
     * variant's pair is enabled together; parse_samples() rejects a half pair and
     * rejects declaring both variants at once.
     */
    [[nodiscard]] bool wants_atmospheric() const noexcept {
      return has_component(component::Conventional) || has_component(component::Prompt) ||
             wants_veto();
    }

    /** Atmospheric components carry the NNMFit passing-fraction reweight. */
    [[nodiscard]] bool wants_veto() const noexcept {
      return has_component(component::ConventionalVeto) || has_component(component::PromptVeto);
    }

    /** A muon template component was declared (see template_file/template_norm_index). */
    [[nodiscard]] bool wants_template() const noexcept {
      return has_component(component::MuonTemplate) || has_component(component::MuonGun);
    }
```

- [ ] **Step 4: Parse and validate in `SampleConfig.cpp`**

Replace `validate_components` with:

```cpp
    bool is_known_component(const std::string& c) noexcept {
      return c == component::Astro || c == component::Conventional || c == component::Prompt ||
             c == component::ConventionalVeto || c == component::PromptVeto ||
             c == component::MuonTemplate || c == component::MuonGun;
    }

    void validate_components(const SampleConfig& sample) {
      if (sample.components.empty())
        throw std::runtime_error("parse_samples: sample '" + sample.name +
                                 "' declares no components (expected a \"components\" list, e.g. "
                                 "\"astro, conventional, prompt\")");

      for (const std::string& c : sample.components) {
        if (!is_known_component(c))
          throw std::runtime_error("parse_samples: sample '" + sample.name + "' declares unknown component '" + c +
                                   "' (supported: astro, conventional, prompt, conventional_veto, "
                                   "prompt_veto, muontemplate, muon)");
      }

      const bool plain = sample.has_component(component::Conventional);
      const bool veto  = sample.has_component(component::ConventionalVeto);
      if (plain != sample.has_component(component::Prompt))
        throw std::runtime_error("parse_samples: sample '" + sample.name +
                                 "' declares only one of 'conventional'/'prompt'; AtmosphericFlux computes both in "
                                 "one pass, so they must be enabled together");
      if (veto != sample.has_component(component::PromptVeto))
        throw std::runtime_error("parse_samples: sample '" + sample.name +
                                 "' declares only one of 'conventional_veto'/'prompt_veto'; they must be enabled "
                                 "together");
      if (plain && veto)
        throw std::runtime_error("parse_samples: sample '" + sample.name +
                                 "' declares both the plain and the veto atmospheric components; NNMFit excludes one "
                                 "variant per sample and enabling both would double-count");

      if (sample.has_component(component::MuonTemplate) && sample.has_component(component::MuonGun))
        throw std::runtime_error("parse_samples: sample '" + sample.name +
                                 "' declares two muon templates ('muontemplate' and 'muon'); pick one");

      if (sample.wants_template() && sample.template_file.empty())
        throw std::runtime_error("parse_samples: sample '" + sample.name +
                                 "' declares a muon template but has no \"Template\": { \"File\": ... } entry");
      if (!sample.wants_template() && !sample.template_file.empty())
        throw std::runtime_error("parse_samples: sample '" + sample.name +
                                 "' has a \"Template\" entry but declares neither 'muontemplate' nor 'muon'");
    }
```

In `parse_samples`, fill the new fields right after the `samples.push_back(...)` call and before `validate_components`:

```cpp
      SampleConfig& sample = samples.back();
      if (const auto tmpl = sample_node.get_child_optional("Template")) {
        sample.template_file = tmpl->get<std::string>("File");
        const std::string norm = tmpl->get<std::string>("Norm", "MuonNorm");
        if (norm == "MuonNorm")
          sample.template_norm_index = params::ic::MuonNorm;
        else if (norm == "MuonGunNorm")
          sample.template_norm_index = params::ic::MuonGunNorm;
        else
          throw std::runtime_error("parse_samples: sample '" + sample_name + "' Template.Norm '" + norm +
                                   "' is not a known template norm (expected MuonNorm or MuonGunNorm)");
      }
      if (const auto grad = sample_node.get_child_optional("Gradients"))
        sample.gradient_file = grad->get<std::string>("File");

      validate_components(sample);
```

(remove the old `validate_components(samples.back());` line).

- [ ] **Step 5: Build, test, golden regression**

Run:
```bash
cmake --build build -j8
ctest --test-dir build -R ICTests --output-on-failure
./build/programs/LLHFit/LLHFit -c configs/config_icecube_tracks_cpu.json --silent
python3 tools/compare_output.py Output.ic_baseline.json Output.json
```
Expected: tests pass, `compare_output.py` exits 0.

- [ ] **Step 6: Commit**

```bash
git add libraries/io/IceCube/SampleConfig.h libraries/io/IceCube/SampleConfig.cpp programs/ictests/ICTests.cpp
git commit -m "feat(icecube): parse veto and muon-template components per sample"
```

---

## Task 5: Load the veto coefficient columns

**Files:**
- Modify: `libraries/io/IceCube/BranchNames.h`, `libraries/io/IceCube/ICSample.h`, `libraries/io/IceCube/ICDataBase.cpp`
- Modify: `programs/ictests/ICTests.cpp`

- [ ] **Step 1: Write the failing test**

In `ICTests.cpp`, extend `test_sort_into_bins_csr_invariant()`. Insert right after `for (auto& g : s.barr_conv) g.resize(N);`:

```cpp
  for (auto& v : s.veto_conv) v.resize(N);
  for (auto& v : s.veto_prompt) v.resize(N);
```

inside the per-event seeding loop, after the Barr line:

```cpp
    for (int k = 0; k < 3; ++k) {
      s.veto_conv[k][i]   = 5000.0 * (k + 1) + idx;
      s.veto_prompt[k][i] = 9000.0 * (k + 1) + idx;
    }
```

and in the lockstep assertion loop:

```cpp
    for (int k = 0; k < 3; ++k) {
      assert(std::abs(s.veto_conv[k][i] - s.e_true[i] - 5000.0 * (k + 1)) < 1e-9);
      assert(std::abs(s.veto_prompt[k][i] - s.e_true[i] - 9000.0 * (k + 1)) < 1e-9);
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cmake --build build -j8`
Expected: compile error — `ICSample` has no `veto_conv` / `veto_prompt`.

- [ ] **Step 3: Add the columns**

In `libraries/io/IceCube/ICSample.h`, after the `barr_conv` member:

```cpp
    // NNMFit VetoThreshold coefficients: log10(passing fraction) is expanded to
    // second order around 100 GeV, per event and per component:
    //   log10(PF) = a + b*e + c*e^2,  index order {a, b, c}.
    // Populated only for samples declaring the veto components.
    std::array<std::vector<double>, 3> veto_conv;
    std::array<std::vector<double>, 3> veto_prompt;
```

and in `sort_into_bins`, after `for (auto& grad : barr_conv) reorder(grad);`:

```cpp
      for (auto& col : veto_conv) reorder(col);
      for (auto& col : veto_prompt) reorder(col);
```

In `libraries/io/IceCube/BranchNames.h`, after `barr_conv`:

```cpp
    // Veto passing-fraction coefficients {a, b, c}; only read for samples that
    // declare the veto components (the tracks baseline does not have them).
    std::array<std::string, 3> veto_conv = {
        "log_PF_at100GeV_conv_a", "log_PF_at100GeV_conv_b", "log_PF_at100GeV_conv_c"};
    std::array<std::string, 3> veto_prompt = {
        "log_PF_at100GeV_pr_a", "log_PF_at100GeV_pr_b", "log_PF_at100GeV_pr_c"};
```

In `libraries/io/IceCube/ICDataBase.cpp`, inside the `if (cfg.wants_atmospheric())` block, after the Barr loop:

```cpp
      if (cfg.wants_veto()) {
        for (int k = 0; k < 3; ++k) {
          ARROW_ASSIGN_OR_RAISE(out.veto_conv[k], get_double_column(*table, b.veto_conv[k]));
          ARROW_ASSIGN_OR_RAISE(out.veto_prompt[k], get_double_column(*table, b.veto_prompt[k]));
        }
      }
```

The veto coefficients are dimensionless — do **not** add them to the livetime scaling block.

- [ ] **Step 4: Build, test, golden regression**

Run:
```bash
cmake --build build -j8
ctest --test-dir build -R ICTests --output-on-failure
./build/programs/LLHFit/LLHFit -c configs/config_icecube_tracks_cpu.json --silent
python3 tools/compare_output.py Output.ic_baseline.json Output.json
```
Expected: tests pass, exit 0 (tracks declares no veto component, so nothing is read).

- [ ] **Step 5: Commit**

```bash
git add libraries/io/IceCube/BranchNames.h libraries/io/IceCube/ICSample.h \
        libraries/io/IceCube/ICDataBase.cpp programs/ictests/ICTests.cpp
git commit -m "feat(icecube): read veto passing-fraction columns for veto samples"
```

---

## Task 6: Veto reweight in `AtmosphericFlux` (CPU + GPU)

**Files:**
- Modify: `libraries/likelihood/IceCube/AtmosphericFlux.h`, `.cpp`
- Modify: `libraries/io/IceCube/ICInputOptions.h`, `.cpp`
- Modify: `libraries/likelihood/IceCube/SampleLikelihood.h`, `.cpp`
- Modify: `libraries/likelihood/IceCube/ICLikelihood.cpp`
- Modify: `programs/ictests/ICTests.cpp`

- [ ] **Step 1: Write the failing test**

Append to `ICTests.cpp` (register in `main()` after `test_sample_likelihood_component_masking();`):

```cpp
// The veto reweight is NNMFit's VetoThreshold parameter (parameters/veto_threshold.py):
//   e  = rescale * 10^p - anchor        (both 100 GeV in the combined config)
//   PF = 10^(a + b*e + c*e^2)           per event, per component
// applied multiplicatively to the conventional and prompt weights. Checked here
// against the same formula evaluated by hand on a one-bin sample, and against the
// invariant that veto-off reproduces the un-vetoed prediction exactly.
static void test_veto_reweight() {
  using ana::ic::AtmosphericFlux;
  using ana::ParameterWrapper;

  const Binning binning({io::ic::parse_axis("Log10Energy", "(2.0, 5.0, 1)"),
                         io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 1)")});
  assert(binning.total_bins() == 1);

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
  assert(std::abs(histogram_at(false, 0.0) - unvetoed) < 1e-12 * unvetoed);
  for (double p : {0.0, 0.5, -0.5, 1.301, -1.301})
    assert(std::abs(histogram_at(true, p) - reference(true, p)) < 1e-12 * unvetoed);
  // The veto suppresses the flux, and its parameter actually triggers a recalculation.
  assert(histogram_at(true, 0.0) < histogram_at(false, 0.0));
  assert(histogram_at(true, 1.0) != histogram_at(true, 0.0));
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cmake --build build -j8`
Expected: compile error — `AtmosphericFlux`'s constructor takes no veto arguments.

- [ ] **Step 3: Extend the `AtmosphericFlux` interface**

In `AtmosphericFlux.h`, extend the constructor and add members:

```cpp
    AtmosphericFlux(const io::ic::ICSample&       sample,
                    const io::ic::Binning&        binning,
                    double                        conv_delta_gamma_e_ref,
                    double                        prompt_delta_gamma_e_ref,
                    std::shared_ptr<GpuBackend>   gpu                 = nullptr,
                    bool                          need_per_event      = false,
                    bool                          use_veto            = false,
                    double                        veto_anchor_energy  = 100.0,
                    double                        veto_rescale_energy = 100.0);
```

private members, after `m_NeedPerEvent`:

```cpp
    bool   m_UseVeto;
    double m_VetoAnchorEnergy;
    double m_VetoRescaleEnergy;
```

and GPU handles, after `m_hBarr`:

```cpp
    std::array<int, 3> m_hVetoConv{};
    std::array<int, 3> m_hVetoPrompt{};
```

Extend the class doc comment with the veto formula (copy it from the spec's section 3).

- [ ] **Step 4: Implement the CPU path**

In `AtmosphericFlux.cpp`, add to the constructor's initialiser list (after `m_NeedPerEvent(need_per_event)`):

```cpp
    , m_UseVeto(use_veto)
    , m_VetoAnchorEnergy(veto_anchor_energy)
    , m_VetoRescaleEnergy(veto_rescale_energy)
```

In `recalculate`, after the `barr` array is filled:

```cpp
    // NNMFit VetoThreshold: the energy offset is scalar per evaluation; only the
    // second-order coefficients are per event.
    const double veto_e =
        m_UseVeto ? m_VetoRescaleEnergy * std::pow(10.0, parameter[VetoThreshold]) - m_VetoAnchorEnergy : 0.0;
```

inside the per-event loop, immediately before `event_total += conv_w;`:

```cpp
        if (m_UseVeto) {
          const double log_pf = m_Sample.veto_conv[0][i] + m_Sample.veto_conv[1][i] * veto_e +
                                m_Sample.veto_conv[2][i] * veto_e * veto_e;
          conv_w *= std::pow(10.0, log_pf);
        }
```

and before `event_total += prompt_w;`:

```cpp
        if (m_UseVeto) {
          const double log_pf = m_Sample.veto_prompt[0][i] + m_Sample.veto_prompt[1][i] * veto_e +
                                m_Sample.veto_prompt[2][i] * veto_e * veto_e;
          prompt_w *= std::pow(10.0, log_pf);
        }
```

In `check_and_recalculate`, add the veto trigger to the `changed` expression:

```cpp
        | (m_UseVeto && parameter.check_parameter_changed(VetoThreshold))
```

- [ ] **Step 5: Implement the GPU path**

In the `AtmoParams` struct (host copy **and** both kernel source strings, which must stay field-for-field identical), append two fields after `write_pe`:

```cpp
      int   use_veto;
      float veto_e;
```

In both kernels, insert the six veto buffers **before** `bin_offsets` (indices shift, per the `GpuBackend` convention that `bin_offsets` is the last input). Metal signature becomes:

```cpp
          device const float*  barr3        [[buffer(8)]],
          device const float*  veto_conv_a  [[buffer(9)]],
          device const float*  veto_conv_b  [[buffer(10)]],
          device const float*  veto_conv_c  [[buffer(11)]],
          device const float*  veto_pr_a    [[buffer(12)]],
          device const float*  veto_pr_b    [[buffer(13)]],
          device const float*  veto_pr_c    [[buffer(14)]],
          device const uint*   bin_offsets  [[buffer(15)]],
          constant AtmoParams& p            [[buffer(16)]],
          device float*        hist         [[buffer(17)]],
          device float*        per_event    [[buffer(18)]],
```

with the CUDA twin taking the same six `const float*` parameters in the same order before `bin_offsets`. In both kernel bodies, inside the event loop, before `event_total += cw;`:

```cpp
            if (p.use_veto) {
              const float log_pf = veto_conv_a[i] + veto_conv_b[i] * p.veto_e +
                                   veto_conv_c[i] * p.veto_e * p.veto_e;
              cw *= pow(10.0f, log_pf);
            }
```

and the `veto_pr_*` equivalent before `event_total += pw;` (CUDA uses `powf`).

In the constructor's GPU block, upload the veto columns; when the sample has none, bind the already-uploaded `e_true` handle so every argument is bound but never read:

```cpp
      for (int k = 0; k < 3; ++k) {
        m_hVetoConv[k]   = m_UseVeto ? m_Gpu->upload_column(sample.veto_conv[k].data(), M) : m_hETrue;
        m_hVetoPrompt[k] = m_UseVeto ? m_Gpu->upload_column(sample.veto_prompt[k].data(), M) : m_hETrue;
      }
```

In `recalculate_gpu`, set the two new params and extend the input list:

```cpp
    p.use_veto = m_UseVeto ? 1 : 0;
    p.veto_e   = static_cast<float>(
        m_UseVeto ? m_VetoRescaleEnergy * std::pow(10.0, parameter[VetoThreshold]) - m_VetoAnchorEnergy : 0.0);

    const int inputs[] = {m_hETrue,        m_hConvBase,     m_hConvAlt,      m_hPromptBase,
                          m_hPromptAlt,    m_hBarr[0],      m_hBarr[1],      m_hBarr[2],
                          m_hBarr[3],      m_hVetoConv[0],  m_hVetoConv[1],  m_hVetoConv[2],
                          m_hVetoPrompt[0], m_hVetoPrompt[1], m_hVetoPrompt[2], m_hOffsets};
    m_Gpu->dispatch("atmo_hist", inputs, 16, &p, sizeof(p), m_hHist, m_hPerEvent, m_Histogram.size());
```

- [ ] **Step 6: Plumb the veto settings through**

In `libraries/io/IceCube/ICInputOptions.h` add getters + members:

```cpp
    // NNMFit effective_veto "additional" block: the passing-fraction expansion
    // point and the scale the fit parameter is exponentiated against, in GeV.
    [[nodiscard]] double veto_anchor_energy() const noexcept { return m_VetoAnchorEnergy; }
    [[nodiscard]] double veto_rescale_energy() const noexcept { return m_VetoRescaleEnergy; }
```
```cpp
    double m_VetoAnchorEnergy  = 100.0;
    double m_VetoRescaleEnergy = 100.0;
```

and in `ICInputOptions.cpp`, next to the other scalar reads:

```cpp
    m_VetoAnchorEnergy  = ic.get<double>("VetoAnchorEnergy", m_VetoAnchorEnergy);
    m_VetoRescaleEnergy = ic.get<double>("VetoRescaleEnergy", m_VetoRescaleEnergy);
```

In `SampleLikelihood.h`, extend `GlobalFluxSettings`:

```cpp
  struct GlobalFluxSettings {
    double e_ref_gev;
    double astro_reference_index;
    double conv_delta_gamma_e_ref;
    double prompt_delta_gamma_e_ref;
    bool   astro_per_type_norm;
    double veto_anchor_energy;
    double veto_rescale_energy;
  };
```

In `SampleLikelihood.cpp`, pass the veto arguments when building the atmospheric flux:

```cpp
    if (cfg.wants_atmospheric())
      m_Atmo.emplace(sample,
                     cfg.binning,
                     settings.conv_delta_gamma_e_ref,
                     settings.prompt_delta_gamma_e_ref,
                     gpu,
                     use_say,
                     cfg.wants_veto(),
                     settings.veto_anchor_energy,
                     settings.veto_rescale_energy);
```

In `ICLikelihood.cpp`, fill the two new settings fields:

```cpp
        .veto_anchor_energy       = input_options.veto_anchor_energy(),
        .veto_rescale_energy      = input_options.veto_rescale_energy(),
```

- [ ] **Step 7: Build, test, golden regression**

Run:
```bash
cmake --build build -j8
ctest --test-dir build -R ICTests --output-on-failure
./build/programs/LLHFit/LLHFit -c configs/config_icecube_tracks_cpu.json --silent
python3 tools/compare_output.py Output.ic_baseline.json Output.json
```
Expected: tests pass; exit 0. Veto-off must be bit-for-bit: the added `if (m_UseVeto)` branches must not reorder any floating-point operation on the tracks path.

- [ ] **Step 8: Metal parity**

Run:
```bash
./build/programs/LLHFit/LLHFit -c configs/config_icecube.json --silent
python3 - <<'EOF'
import json
cpu = json.load(open('Output.ic_baseline.json'))
gpu = json.load(open('Output.json'))
print('chi2 rel', abs(cpu['chi2'] - gpu['chi2']) / abs(cpu['chi2']))
for k, v in cpu['parameters'].items():
    print(k, abs(v['value'] - gpu['parameters'][k]['value']))
EOF
```
Expected: `chi2 rel` ~1e-9, meaningful parameters within 1e-3 (the recorded FP32 tolerance). A larger jump means the buffer-index shift in the kernel is wrong.

- [ ] **Step 9: Commit**

```bash
git add libraries/likelihood/IceCube/AtmosphericFlux.h libraries/likelihood/IceCube/AtmosphericFlux.cpp \
        libraries/likelihood/IceCube/SampleLikelihood.h libraries/likelihood/IceCube/SampleLikelihood.cpp \
        libraries/likelihood/IceCube/ICLikelihood.cpp libraries/io/IceCube/ICInputOptions.h \
        libraries/io/IceCube/ICInputOptions.cpp programs/ictests/ICTests.cpp
git commit -m "feat(icecube): veto passing-fraction reweight in AtmosphericFlux"
```

---

## Task 7: Export NNMFit templates and gradients to text

**Files:**
- Create: `tools/export_nnmfit_inputs.py`

No C++ changes; this produces the files Tasks 8 and 9 read. Uses only `pickle` + `numpy`, both available in the system Python 3 (`pyarrow` is not installed and is not needed).

- [ ] **Step 1: Write the exporter**

Create `tools/export_nnmfit_inputs.py`:

```python
#!/usr/bin/env python3
"""Convert NNMFit muon-template and SnowStorm-gradient pickles to plain text.

The C++ side (TemplateFlux, DetectorSystematics) reads whitespace-separated
numbers with a small header, so no Python/pickle dependency leaks into the fit.

Usage:
  tools/export_nnmfit_inputs.py template  IN.pickle OUT.txt --det-conf NAME --bins N
  tools/export_nnmfit_inputs.py gradients IN.pickle OUT.txt --det-conf NAME --bins N \\
                                          [--livetime-scale S]

The detector-config names in the pickles are NNMFit's, e.g.
IC86_pass2_SnowStorm_v2_cscd_cascade.
"""
import argparse
import pickle
import sys

import numpy as np

# The order DetectorSystematics assumes, matching params::ic {DOMEff, IceAbs,
# IceScat, HoleIceP0, HoleIceP1}.
SYSTEMATICS = [
    "DOMEfficiency",
    "IceAbsorption",
    "IceScattering",
    "HoleIceForward_p0",
    "HoleIceForward_p1",
]

# Which cross_correlations error keys enter the covariance of two gradients
# (NNMFit snowstorm_gradient.__covariance_g1_g2): +up/up, +lo/lo, -lo/up, -up/lo.
COV_TERMS = [
    ("sys up-alt sys up", +1.0),
    ("sys low-alt sys low", +1.0),
    ("sys low-alt sys up", -1.0),
    ("sys up-alt sys low", -1.0),
]


def load(path, det_conf):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if det_conf is not None and det_conf in obj:
        return obj[det_conf]
    if det_conf is not None:
        sys.exit(
            f"detector config '{det_conf}' not in {path}; available: {sorted(obj.keys())}"
        )
    return obj


def flat(array, bins, what):
    a = np.asarray(array, dtype=float)
    if a.size != bins:
        sys.exit(f"{what}: expected {bins} values, pickle has {a.size} (shape {a.shape})")
    # Flattening is row-major: first axis (energy) outer, second (zenith) inner --
    # the same order io::ic::Binning uses for its flat index.
    return a.reshape(-1)


def export_template(entry, out, bins):
    # Two layouts: the multi-dataset MuonGun pickles use "template", the
    # single-dataset Corsika tracks pickle uses "template_2d". The `_no_fluct`
    # MuonGun variant (the one the reference fit used) stores
    # template_fluctuation: None -- exported as zeros, which makes the sigma^2
    # term vanish exactly as NNMFit's None fluctuation graph does.
    key = "template" if "template" in entry else "template_2d"
    if key not in entry:
        sys.exit(f"pickle entry has neither 'template' nor 'template_2d'; keys: {sorted(entry)}")
    template = flat(entry[key], bins, key)
    fluctuation = entry.get("template_fluctuation")
    fluctuation = (
        flat(fluctuation, bins, "template_fluctuation")
        if fluctuation is not None
        else np.zeros(bins)
    )
    with open(out, "w") as f:
        f.write(f"# template bins {bins}\n")
        f.write("# columns: template_rate fluctuation_rate (both per second)\n")
        if "energy_bins" in entry:
            f.write(f"# energy_bins {' '.join(repr(float(x)) for x in entry['energy_bins'])}\n")
        if "zenith_bins" in entry:
            f.write(f"# cos_zenith_bins {' '.join(repr(float(x)) for x in entry['zenith_bins'])}\n")
        for t, s in zip(template, fluctuation):
            f.write(f"{t!r} {s!r}\n")
    print(f"wrote {out}: {bins} bins, template sum {template.sum()!r} s^-1")


def export_gradients(entry, out, bins, livetime_scale):
    missing = [s for s in SYSTEMATICS if s not in entry]
    if missing:
        sys.exit(f"gradient pickle lacks systematics {missing}; has {sorted(entry.keys())}")

    with open(out, "w") as f:
        f.write(
            f"# gradients bins {bins} params {len(SYSTEMATICS)} lt_scale {livetime_scale!r}\n"
        )
        f.write("# per param: name split, then <bins> lines of 'gradient gradient_error'\n")
        for name in SYSTEMATICS:
            g = entry[name]
            gradient = flat(g["gradient"], bins, f"{name}.gradient")
            error = flat(g["gradient_error"], bins, f"{name}.gradient_error")
            f.write(f"# param {name} split {float(g['split_value'])!r}\n")
            for v, e in zip(gradient, error):
                f.write(f"{v!r} {e!r}\n")

        f.write("# per pair: names, then <bins> lines of 'covariance'\n")
        for i, name_i in enumerate(SYSTEMATICS):
            for name_j in SYSTEMATICS[i + 1 :]:
                corr = entry[name_i]["cross_correlations"][name_j]
                split_terms = np.zeros(bins)
                for key, sign in COV_TERMS:
                    split_terms += sign * flat(
                        np.asarray(corr[key]["error"]) ** 2, bins, f"{name_i}/{name_j} {key}"
                    )
                cov = (
                    float(entry[name_i]["factor"])
                    * float(entry[name_j]["factor"])
                    * split_terms
                )
                f.write(f"# cov {name_i} {name_j}\n")
                for v in cov:
                    f.write(f"{v!r}\n")
    print(f"wrote {out}: {bins} bins x {len(SYSTEMATICS)} systematics + 10 covariance pairs")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["template", "gradients"])
    p.add_argument("pickle_path")
    p.add_argument("out_path")
    p.add_argument("--det-conf", default=None)
    p.add_argument("--bins", type=int, required=True)
    p.add_argument(
        "--livetime-scale",
        type=float,
        default=1.0,
        help="analysis livetime / gradient livetime (NNMFit livetime_scaling)",
    )
    args = p.parse_args()

    entry = load(args.pickle_path, args.det_conf)
    if args.kind == "template":
        export_template(entry, args.out_path, args.bins)
    else:
        export_gradients(entry, args.out_path, args.bins, args.livetime_scale)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate every input file**

Run (output directory untracked, alongside the parquets):

```bash
NF=/Users/soldin/Downloads/nnmfit_files
mkdir -p "$NF/exported"
chmod +x tools/export_nnmfit_inputs.py

# The reference fit used the no-fluctuation MuonGun variant; export that one for
# parity, and the fluctuation-carrying one alongside it for later comparison.
tools/export_nnmfit_inputs.py template "$NF/cscd_muongun_ALL_KDE_5up_manual_ssq_no_fluct.pickle" \
    "$NF/exported/template_cscd_cascade.txt" \
    --det-conf IC86_pass2_SnowStorm_v2_cscd_cascade --bins 147
tools/export_nnmfit_inputs.py template "$NF/cscd_muongun_ALL_KDE_5up_manual_ssq_no_fluct.pickle" \
    "$NF/exported/template_cscd_muon.txt" \
    --det-conf IC86_pass2_SnowStorm_v2_cscd_muon --bins 1
tools/export_nnmfit_inputs.py template "$NF/cscd_muongun_ALL_KDE_5up_manual_ssq.pickle" \
    "$NF/exported/template_cscd_cascade_with_fluct.txt" \
    --det-conf IC86_pass2_SnowStorm_v2_cscd_cascade --bins 147
tools/export_nnmfit_inputs.py template "$NF/cscd_muongun_ALL_KDE_5up_manual_ssq.pickle" \
    "$NF/exported/template_cscd_muon_with_fluct.txt" \
    --det-conf IC86_pass2_SnowStorm_v2_cscd_muon --bins 1

# Tracks Corsika template: single-dataset pickle, array key "template_2d", so no
# --det-conf. Required for parity -- the reference fit floats MuonNorm (1.827).
tools/export_nnmfit_inputs.py template "$NF/Tracks_CorsikaMuon_Fullrange_drop_5lowEbins.pickle" \
    "$NF/exported/template_tracks.txt" --bins 1485

tools/export_nnmfit_inputs.py gradients "$NF/snowstorm_ftp_all_cscd_5up.pickle" \
    "$NF/exported/gradients_cscd_cascade.txt" \
    --det-conf IC86_pass2_SnowStorm_v2_cscd_cascade --bins 147
tools/export_nnmfit_inputs.py gradients "$NF/snowstorm_ftp_all_cscd_5up.pickle" \
    "$NF/exported/gradients_cscd_muon.txt" \
    --det-conf IC86_pass2_SnowStorm_v2_cscd_muon --bins 1
tools/export_nnmfit_inputs.py gradients "$NF/snowstorm_ftp_all.pickle" \
    "$NF/exported/gradients_tracks.txt" \
    --det-conf IC86_pass2_SnowStorm_v2_tracks --bins 1485
```

Expected output lines: cascade template sum `2.3425769796686227e-06` s⁻¹, muon template sum `5.4694021304430105e-05` s⁻¹ (identical in both MuonGun variants — only the fluctuation differs, `None` vs sums 1.152e-06 / 1.095e-06 s⁻¹). If a `--bins` count is rejected, the pickle's grid disagrees with the config and must be resolved before continuing (do not "fix" it by changing `--bins`).

If the tracks export fails on the shape (`template_2d` may be stored 2-D as 45 × 33 rather than flat), that is the flattening question from spec risk 2: reshape row-major with energy outer, and record in the commit message which orientation the pickle used. Do **not** transpose to make a number look nicer — Task 13's per-bin diff is what settles it.

- [ ] **Step 3: Sanity-check one exported file**

Run: `head -5 "$NF/exported/gradients_cscd_muon.txt"`
Expected: a `# gradients bins 1 params 5 lt_scale 1.0` header, then a `# param DOMEfficiency split 1.0` line and one `gradient gradient_error` pair.

- [ ] **Step 4: Commit**

```bash
git add tools/export_nnmfit_inputs.py
git commit -m "tools: export NNMFit template and gradient pickles to plain text"
```

---

## Task 7b: Oscillation survival factors (νμ disappearance)

**Files:**
- Create: `tools/export_oscillation_factors.py`
- Modify: `libraries/io/IceCube/SampleConfig.h`, `.cpp`, `libraries/io/IceCube/ICDataBase.cpp`
- Modify: `programs/ictests/ICTests.cpp`

The combined config attaches `OscillationsHook` to conventional **and** prompt, so parity is impossible while it is a no-op. Per `NNMFit/fluxes/flux_hooks.py:96-155` it is νμ-disappearance only: a 2-D spline in (log10 E_true, zenith_true) per particle type (±14), evaluated **once at load time**, multiplying the component's baseline weights. So it costs nothing in the hot loop — it is a per-event column applied at load.

- [ ] **Step 1: Export the exact per-event factors**

Create `tools/export_oscillation_factors.py` (run with the NNMFit venv, which has scipy and the same spline objects NNMFit uses, so the factors are exact rather than interpolated):

```python
#!/usr/bin/env python3
"""Evaluate NNMFit's OscillationsHook survival probability per MC event.

Writes a one-column parquet sidecar, row-aligned with the input baseline parquet,
holding the nu_mu / anti-nu_mu survival probability (1.0 for every other particle
type). NNMFit applies this factor to the conventional and prompt baseline weights
at load time; see NNMFit/fluxes/flux_hooks.py:96-155.

Run with the NNMFit venv:
  /Users/soldin/Projects/IceCube/NNMFit/.venv/bin/python \\
      tools/export_oscillation_factors.py MC.parquet SPLINE.pickle OUT.parquet
"""
import argparse
import pickle

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mc_parquet")
    p.add_argument("spline_pickle")
    p.add_argument("out_parquet")
    p.add_argument("--true-energy", default="MCPrimaryEnergy")
    p.add_argument("--true-zenith", default="MCPrimaryZenith")
    p.add_argument("--true-ptype", default="MCPrimaryType")
    args = p.parse_args()

    table = pq.read_table(
        args.mc_parquet, columns=[args.true_energy, args.true_zenith, args.true_ptype]
    )
    energy = np.asarray(table[args.true_energy], dtype=float)
    zenith = np.asarray(table[args.true_zenith], dtype=float)
    ptype = np.asarray(table[args.true_ptype], dtype=float).astype(int)

    with open(args.spline_pickle, "rb") as f:
        splines = pickle.load(f)["OscProb_Splines"]

    # Same loop as the hook: only nu_mu / anti-nu_mu are affected.
    factor = np.ones(len(ptype))
    for particle in (-14, 14):
        mask = ptype == particle
        if not mask.any():
            continue
        spline = splines[particle][abs(particle)]
        factor[mask] = spline(np.log10(energy[mask]), zenith[mask], grid=False)

    print(
        f"{args.mc_parquet}: {len(ptype)} events, "
        f"{int((ptype == 14).sum() + (ptype == -14).sum())} nu_mu, "
        f"factor min {factor.min():.6f} max {factor.max():.6f} mean {factor.mean():.6f}"
    )
    pq.write_table(pa.table({"osc_survival": pa.array(factor)}), args.out_parquet)
    print(f"wrote {args.out_parquet}")


if __name__ == "__main__":
    main()
```

Run:
```bash
V=/Users/soldin/Projects/IceCube/NNMFit/.venv/bin/python
NF=/Users/soldin/Downloads/nnmfit_files
SPLINE=$NF/NuCraft_OscillationProb.pickle
mkdir -p "$NF/exported"
for s in tracks:tracks_ftp_baseline/dataset_tracks_baseline.parquet \
         cscd_cascade:cscd_cascade_ftp_baseline/dataset_cscd_cascade_FTP_baseline_wCoords.parquet \
         cscd_muon:cscd_muon_ftp_baseline/dataset_cscd_muon_FTP_baseline_wCoords.parquet; do
  name="${s%%:*}"; rel="${s#*:}"
  $V tools/export_oscillation_factors.py "$NF/datasets/$rel" "$SPLINE" \
     "$NF/exported/osc_${name}.parquet"
done
```
Expected: one line per sample with the event count matching the known row counts (13,906,852 /
1,409,344 / 587,538) and a factor in (0, 1] — a `max` above 1.0 or a `min` of 0.0 means the wrong
spline key or a units mismatch (the hook feeds `log10(E)` and zenith in **radians**). Nothing else in
this task can be trusted until those ranges look physical.

- [ ] **Step 2: Write the failing test**

Append to `ICTests.cpp` (register in `main()` after `test_parse_samples_cascade_entry();`):

```cpp
// The oscillation sidecar is a per-event multiplicative factor on the atmospheric
// baselines only (NNMFit's OscillationsHook: nu_mu disappearance, applied to the
// conventional and prompt baseline weights at load time).
static void test_parse_samples_oscillation_entry() {
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
  assert(samples.size() == 2);
  assert(samples[0].oscillation_file == "osc_tracks.parquet");
  assert(samples[0].oscillation_branch == "osc_survival");
  assert(samples[1].oscillation_file.empty());
}
```

- [ ] **Step 3: Parse the sidecar path**

In `SampleConfig.h`, after `gradient_file`:

```cpp
    // Per-event nu_mu survival factor (NNMFit OscillationsHook), exported by
    // tools/export_oscillation_factors.py. Row-aligned with `parquet`; applied to
    // the atmospheric baselines at load time. "" = no oscillation reweight.
    std::string oscillation_file;
    std::string oscillation_branch = "osc_survival";
```

and in `SampleConfig.cpp`, next to the `Gradients` block:

```cpp
      if (const auto osc = sample_node.get_child_optional("Oscillations")) {
        sample.oscillation_file   = osc->get<std::string>("File");
        sample.oscillation_branch = osc->get<std::string>("Branch", sample.oscillation_branch);
      }
```

- [ ] **Step 4: Apply the factor in the loader**

In `ICDataBase.cpp`, inside `read_sample`'s `if (cfg.wants_atmospheric())` block, after the veto
columns and **before** the livetime scaling:

```cpp
      // NNMFit's OscillationsHook multiplies the atmospheric baseline weights by a
      // per-event nu_mu survival probability, once at load time. The factor lives
      // in a sidecar parquet, row-aligned with this sample's baseline file, so the
      // hot loop is untouched. Barr slopes are scaled too: they are derivatives of
      // the same conventional weight, so leaving them unscaled would change the
      // (slope / baseline) reweight ratios.
      if (!cfg.oscillation_file.empty()) {
        ARROW_ASSIGN_OR_RAISE(auto osc_table, read_parquet_file(cfg.oscillation_file));
        ARROW_ASSIGN_OR_RAISE(auto survival, get_double_column(*osc_table, cfg.oscillation_branch));
        if (survival.size() != out.conv_baseline.size())
          return arrow::Status::Invalid(
              "ICDataBase: oscillation sidecar '" + cfg.oscillation_file + "' has " +
              std::to_string(survival.size()) + " rows, the baseline parquet has " +
              std::to_string(out.conv_baseline.size()) + " (they must be row-aligned)");

        auto apply = [&survival](std::vector<double>& column) {
          for (std::size_t i = 0; i < column.size(); ++i) column[i] *= survival[i];
        };
        apply(out.conv_baseline);
        apply(out.conv_alt);
        apply(out.prompt_baseline);
        apply(out.prompt_alt);
        for (auto& slope : out.barr_conv) apply(slope);

        double mean = 0.0;
        for (const double v : survival) mean += v;
        std::cout << "IceCube sample '" << cfg.name << "': applied oscillation survival factors (mean "
                  << mean / static_cast<double>(survival.size()) << ")\n";
      }
```

Note the ordering: this runs **before** the livetime block, so both scalings compose, and before
`sort_into_bins`, so the sidecar's row order still matches the parquet's.

- [ ] **Step 5: Add the sidecar to the configs**

In `configs/config_icecube_cascades.json` and `configs/config_icecube_combined.json`, add to each
sample:

```json
        "Oscillations": { "File": "/Users/soldin/Downloads/nnmfit_files/exported/osc_<sample>.parquet" },
```

Leave the golden `config_icecube_tracks_cpu.json` **without** it: that config is the bit-for-bit
regression oracle, and adding a physics factor would change its numbers for reasons unrelated to the
refactor being guarded.

- [ ] **Step 6: Build, test, golden regression**

Run:
```bash
cmake --build build -j8
ctest --test-dir build -R ICTests --output-on-failure
./build/programs/LLHFit/LLHFit -c configs/config_icecube_tracks_cpu.json --silent
python3 tools/compare_output.py Output.ic_baseline.json Output.json
```
Expected: tests pass; exit 0 (the golden config declares no `Oscillations` block).

Then check the effect is real and physical on a sample that does use it:
```bash
./build/programs/LLHFit/LLHFit -c configs/config_icecube_cascades.json --silent
```
Expected: the printed mean survival factor is < 1.0 and the sample's Asimov total drops relative to a
run with the `Oscillations` block removed. Record both totals in the commit message.

- [ ] **Step 7: Commit**

```bash
git add tools/export_oscillation_factors.py libraries/io/IceCube/SampleConfig.h \
        libraries/io/IceCube/SampleConfig.cpp libraries/io/IceCube/ICDataBase.cpp \
        configs/config_icecube_cascades.json configs/config_icecube_combined.json \
        programs/ictests/ICTests.cpp
git commit -m "feat(icecube): apply NNMFit oscillation survival factors to atmospheric baselines"
```

Note: this task references the cascade/combined configs that Task 11 creates. If it is executed
before Task 11, do Steps 1–4, 6 (golden only) and 7 here, and fold Step 5 into Task 11's config
authoring.

---

## Task 8: `TemplateFlux` — per-bin muon templates

**Files:**
- Create: `libraries/likelihood/IceCube/TemplateFlux.h`, `.cpp`
- Delete: `libraries/likelihood/IceCube/MuonTemplate.h`, `.cpp`
- Modify: `libraries/likelihood/IceCube/SampleLikelihood.h`, `.cpp`, `libraries/likelihood/IceCube/CMakeLists.txt`
- Modify: `libraries/io/IceCube/ICInputOptions.h`, `.cpp`
- Modify: `programs/ictests/ICTests.cpp`

- [ ] **Step 1: Write the failing test**

Append to `ICTests.cpp` (register in `main()` after `test_veto_reweight();`); add `#include "TemplateFlux.h"`, `#include <fstream>` and `#include <cstdio>` to the includes if absent:

```cpp
// A muon template is a per-bin rate plus a per-bin fluctuation; the component
// scales both by its norm parameter and the sample livetime, matching NNMFit
// (histogram_builder: ssq += (hist_fluctuation * livetime)**2).
static void test_template_flux() {
  using ana::ic::TemplateFlux;
  using ana::ParameterWrapper;

  const Binning binning({io::ic::parse_axis("Log10Energy", "(2.0, 5.0, 3)"),
                         io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 1)")});
  assert(binning.total_bins() == 3);

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

  assert(flux.check_and_recalculate(parameter));

  const double rates[3]  = {1.0e-6, 2.0e-6, 4.0e-6};
  const double sigmas[3] = {2.0e-7, 3.0e-7, 5.0e-7};
  for (int b = 0; b < 3; ++b) {
    const double mu  = 2.0 * rates[b] * livetime;
    const double ssq = (2.0 * sigmas[b] * livetime) * (2.0 * sigmas[b] * livetime);
    assert(std::abs(flux.histogram()[b] - mu) < 1e-9 * mu);
    assert(std::abs(flux.fluctuation()[b] - ssq) < 1e-9 * ssq);
  }

  // Unchanged parameters must not trigger a recalculation.
  assert(!flux.check_and_recalculate(parameter));

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
  assert(threw);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cmake --build build -j8`
Expected: `TemplateFlux.h` not found.

- [ ] **Step 3: Write the header**

Create `libraries/likelihood/IceCube/TemplateFlux.h`:

```cpp
#pragma once

#include "../../io/IceCube/Binning.h"
#include "../ParameterWrapper.h"

#include <span>
#include <string>
#include <vector>

namespace ana::ic {

  /**
   * Binned template flux (NNMFit TemplateFlux): a fixed per-bin rate scaled by
   * one norm parameter, used for the atmospheric-muon templates -- Corsika
   * ("muontemplate", MuonNorm) for tracks and MuonGun ("muon", MuonGunNorm) for
   * the cascade samples.
   *
   *   mu_b  = norm * template_b * livetime
   *   ssq_b = (norm * fluctuation_b * livetime)^2
   *
   * matching NNMFit's histogram_builder, which multiplies the fluctuation graph
   * by the same parameters and adds its square to sigma^2. Template files are
   * produced by tools/export_nnmfit_inputs.py; both columns are rates (per
   * second), so the livetime scaling mirrors what ICDataBase does to the
   * per-event weights.
   *
   * O(nBins) work: CPU only, no GPU path.
   */
  class TemplateFlux {
   public:
    TemplateFlux(const io::ic::Binning& binning,
                 const std::string&     template_file,
                 int                    norm_index,
                 double                 livetime);
    ~TemplateFlux() = default;

    /** Rescale for the current parameters; false when the norm did not change. */
    bool check_and_recalculate(const ParameterWrapper& parameter);

    /** Predicted counts per analysis bin from this template. */
    [[nodiscard]] std::span<const double> histogram() const noexcept { return m_Histogram; }

    /** sigma^2 contribution per analysis bin (zero if the file carried no fluctuations). */
    [[nodiscard]] std::span<const double> fluctuation() const noexcept { return m_Fluctuation; }

   private:
    int                 m_NormIndex;
    std::vector<double> m_Template;     // rate * livetime, per bin
    std::vector<double> m_Sigma;        // fluctuation rate * livetime, per bin
    std::vector<double> m_Histogram;    // norm * m_Template
    std::vector<double> m_Fluctuation;  // (norm * m_Sigma)^2

    void load(const std::string& path, int total_bins, double livetime);
  };

}  // namespace ana::ic
```

- [ ] **Step 4: Write the implementation**

Create `libraries/likelihood/IceCube/TemplateFlux.cpp`:

```cpp
#include "TemplateFlux.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace ana::ic {

  TemplateFlux::TemplateFlux(const io::ic::Binning& binning,
                             const std::string&     template_file,
                             const int              norm_index,
                             const double           livetime)
    : m_NormIndex(norm_index) {
    const int total_bins = binning.total_bins();
    m_Histogram.assign(total_bins, 0.0);
    m_Fluctuation.assign(total_bins, 0.0);
    load(template_file, total_bins, livetime);
  }

  void TemplateFlux::load(const std::string& path, const int total_bins, const double livetime) {
    std::ifstream in(path);
    if (!in)
      throw std::runtime_error("TemplateFlux: cannot open template file '" + path + "'");

    // Header: "# template bins <N>"; the remaining comments (bin edges) are
    // informational. The bin count is a hard check -- a template binned
    // differently from the sample would mis-assign every bin while still
    // summing to a plausible total.
    int         declared_bins = -1;
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      if (line.front() != '#') break;
      std::istringstream header(line);
      std::string        hash, word, bins_key;
      if (header >> hash >> word && word == "template") {
        if (!(header >> bins_key >> declared_bins) || bins_key != "bins") declared_bins = -1;
      }
    }
    if (declared_bins != total_bins)
      throw std::runtime_error("TemplateFlux: template '" + path + "' declares " +
                               std::to_string(declared_bins) + " bins, the sample's binning has " +
                               std::to_string(total_bins));

    m_Template.assign(total_bins, 0.0);
    m_Sigma.assign(total_bins, 0.0);

    // `line` holds the first non-comment line read above; parse it, then the rest.
    std::istringstream first(line);
    double             rate = 0.0, sigma = 0.0;
    if (!(first >> rate >> sigma))
      throw std::runtime_error("TemplateFlux: template '" + path + "' has no data rows");
    m_Template[0] = rate * livetime;
    m_Sigma[0]    = sigma * livetime;

    for (int b = 1; b < total_bins; ++b) {
      if (!(in >> rate >> sigma))
        throw std::runtime_error("TemplateFlux: template '" + path + "' ended after " +
                                 std::to_string(b) + " of " + std::to_string(total_bins) + " bins");
      m_Template[b] = rate * livetime;
      m_Sigma[b]    = sigma * livetime;
    }

    double total = 0.0;
    for (const double v : m_Template) total += v;
    std::cout << "TemplateFlux: loaded " << total_bins << "-bin template from " << path << " (" << total
              << " events at norm 1)\n";
  }

  bool TemplateFlux::check_and_recalculate(const ParameterWrapper& parameter) {
    if (!parameter.check_parameter_changed(m_NormIndex)) return false;

    const double norm = parameter[m_NormIndex];
    for (std::size_t b = 0, n = m_Histogram.size(); b < n; ++b) {
      m_Histogram[b]   = norm * m_Template[b];
      const double sig = norm * m_Sigma[b];
      m_Fluctuation[b] = sig * sig;
    }
    return true;
  }

}  // namespace ana::ic
```

- [ ] **Step 5: Wire it into `SampleLikelihood`**

In `SampleLikelihood.h`: add `#include "TemplateFlux.h"`, and after `m_Atmo`:

```cpp
    std::optional<TemplateFlux> m_Template;
```

In `SampleLikelihood.cpp`'s constructor, after the atmospheric block:

```cpp
    if (cfg.wants_template())
      m_Template.emplace(cfg.binning, cfg.template_file, cfg.template_norm_index, cfg.livetime);
```

In `assemble_prediction`:

```cpp
    if (m_Template) changed |= m_Template->check_and_recalculate(parameter);
```
```cpp
    const std::span<const double> tmpl = m_Template ? m_Template->histogram() : std::span<const double>{};
```
```cpp
      if (!tmpl.empty()) total += tmpl[b];
```

In `assemble_fluctuation`, the per-event loop **overwrites** `m_Ssq[b]`, so histogram-level terms are added afterwards — and when no per-event flux exists nothing writes `m_Ssq` at all. Replace the trailing dispatch chain with:

```cpp
    if (!astro.empty() && !atmo.empty())
      accumulate([&](const std::size_t i) { return astro[i] + atmo[i]; });
    else if (!astro.empty())
      accumulate([&](const std::size_t i) { return astro[i]; });
    else if (!atmo.empty())
      accumulate([&](const std::size_t i) { return atmo[i]; });
    else
      std::ranges::fill(m_Ssq, 0.0);

    // Histogram-level fluctuation from the template component, added after the
    // per-event sum (NNMFit: ssq += (hist_fluctuation * livetime)**2).
    if (m_Template) {
      const std::span<const double> tmpl_ssq = m_Template->fluctuation();
      for (int b = 0; b < n_bins; ++b) m_Ssq[b] += tmpl_ssq[b];
    }
```

- [ ] **Step 6: Delete `MuonTemplate` and update CMake**

```bash
git rm libraries/likelihood/IceCube/MuonTemplate.h libraries/likelihood/IceCube/MuonTemplate.cpp
```

In `libraries/likelihood/IceCube/CMakeLists.txt`, replace the two `MuonTemplate.*` lines with:

```cmake
    TemplateFlux.h
    TemplateFlux.cpp
```

Also drop the now-stale `use_muon_template()` / `muon_template_file()` getters and their members from `libraries/io/IceCube/ICInputOptions.h` plus the two `ic.get<...>("UseMuonTemplate"...)` / `("MuonTemplateFile"...)` reads in `ICInputOptions.cpp`: the template is per-sample now.

- [ ] **Step 7: Build, test, golden regression**

Run:
```bash
cmake --build build -j8
ctest --test-dir build -R ICTests --output-on-failure
./build/programs/LLHFit/LLHFit -c configs/config_icecube_tracks_cpu.json --silent
python3 tools/compare_output.py Output.ic_baseline.json Output.json
```
Expected: tests pass; exit 0 (tracks declares no template).

- [ ] **Step 8: Commit**

```bash
git add libraries/likelihood/IceCube/TemplateFlux.h libraries/likelihood/IceCube/TemplateFlux.cpp \
        libraries/likelihood/IceCube/SampleLikelihood.h libraries/likelihood/IceCube/SampleLikelihood.cpp \
        libraries/likelihood/IceCube/CMakeLists.txt libraries/io/IceCube/ICInputOptions.h \
        libraries/io/IceCube/ICInputOptions.cpp programs/ictests/ICTests.cpp
git commit -m "feat(icecube): per-sample TemplateFlux with mu and sigma^2, replacing MuonTemplate"
```

---

## Task 9: Per-sample detector systematics with µ and σ²

**Files:**
- Modify: `libraries/likelihood/IceCube/DetectorSystematics.h`, `.cpp`
- Modify: `libraries/likelihood/IceCube/SampleLikelihood.h`, `.cpp`
- Modify: `programs/ictests/ICTests.cpp`

- [ ] **Step 1: Write the failing test**

Append to `ICTests.cpp` (register in `main()` after `test_template_flux();`):

```cpp
// SnowStorm gradients are a histogram-level additive perturbation of mu and
// sigma^2 (NNMFit snowstorm_gradient.make_graph, external_gradients: True so the
// histogram-gradient covariance is excluded):
//   D_k       = p_k - split_k
//   mu_add_b  = sum_k D_k * gradient_k_b * lt_scale
//   ssq_add_b = sum_k (D_k * error_k_b * lt_scale)^2 + 2 * sum_{i<j} D_i D_j cov_ij_b
static void test_detector_systematics() {
  using ana::ic::DetectorSystematics;
  using ana::ParameterWrapper;

  const Binning binning({io::ic::parse_axis("Log10Energy", "(2.0, 5.0, 2)"),
                         io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 1)")});
  assert(binning.total_bins() == 2);

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
    assert(systematics.mu_delta()[b] == 0.0);
    assert(systematics.ssq_delta()[b] == 0.0);
  }

  // Perturb two systematics.
  values[params::ic::DOMEff] = split[0] + 0.05;
  values[params::ic::IceAbs] = split[1] - 0.02;
  ParameterWrapper shifted(params::ic::number_of_parameters());
  shifted.reset_parameter(values.data());
  assert(systematics.check_and_recalculate(shifted));

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

    assert(std::abs(systematics.mu_delta()[b] - mu_add) < 1e-12 * std::abs(mu_add));
    assert(std::abs(systematics.ssq_delta()[b] - ssq_add) < 1e-12 * std::abs(ssq_add));
  }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cmake --build build -j8`
Expected: compile error — `DetectorSystematics` takes `(bool, const std::string&)` and has no `mu_delta()` / `ssq_delta()`.

- [ ] **Step 3: Rewrite the header**

Replace `libraries/likelihood/IceCube/DetectorSystematics.h` with:

```cpp
#pragma once

#include "../../io/IceCube/Binning.h"
#include "../../io/IceCube/ICParameter.h"
#include "../ParameterWrapper.h"

#include <array>
#include <span>
#include <string>
#include <vector>

namespace ana::ic {

  /**
   * SnowStorm detector-gradient systematics (NNMFit SnowStormGradient with
   * hist_parameter_overall: True), applied to one sample's summed prediction as
   * an additive perturbation of mu and sigma^2:
   *
   *   D_k       = (parameter_k - split_k) * lt_scale   k in {DOMEff .. HoleIceP1}
   *   mu_add_b  = sum_k D_k * gradient_k_b
   *   ssq_add_b = sum_k (D_k * gradient_error_k_b)^2 + 2 * sum_{i<j} D_i * D_j * cov_ij_b
   *
   * lt_scale is the analysis / gradient livetime ratio. The histogram-gradient
   * covariance term of NNMFit's formula is correctly absent: the FTP gradient
   * configs set external_gradients: True (the gradients come from independent MC).
   *
   * Gradients are per sample (each detector config has its own pickle; the
   * cascade samples share the _5up one) and are read from the text file produced
   * by tools/export_nnmfit_inputs.py, whose systematics order matches
   * params::ic {DOMEff .. HoleIceP1}. O(nBins) work: CPU only.
   */
  class DetectorSystematics {
   public:
    DetectorSystematics(const io::ic::Binning& binning, const std::string& gradient_file);
    ~DetectorSystematics() = default;

    bool check_and_recalculate(const ParameterWrapper& parameter);

    /** Additive contribution to the predicted counts per bin. */
    [[nodiscard]] std::span<const double> mu_delta() const noexcept { return m_MuDelta; }

    /** Additive contribution to sigma^2 per bin (SAY only). */
    [[nodiscard]] std::span<const double> ssq_delta() const noexcept { return m_SsqDelta; }

   private:
    static constexpr int nPairs = params::ic::nDetSysParams * (params::ic::nDetSysParams - 1) / 2;

    double                                        m_LivetimeScale = 1.0;
    std::array<double, params::ic::nDetSysParams> m_Split{};

    std::array<std::vector<double>, params::ic::nDetSysParams> m_Gradient;
    std::array<std::vector<double>, params::ic::nDetSysParams> m_GradientError;
    std::array<std::vector<double>, nPairs>                    m_Covariance;

    std::vector<double> m_MuDelta;
    std::vector<double> m_SsqDelta;

    void load(const std::string& path, int total_bins);
  };

}  // namespace ana::ic
```

- [ ] **Step 4: Rewrite the implementation**

Replace `libraries/likelihood/IceCube/DetectorSystematics.cpp` with:

```cpp
#include "DetectorSystematics.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace ana::ic {

  namespace {

    // The order tools/export_nnmfit_inputs.py writes, matching params::ic.
    constexpr const char* kSystematicNames[params::ic::nDetSysParams] = {
        "DOMEfficiency", "IceAbsorption", "IceScattering", "HoleIceForward_p0", "HoleIceForward_p1"};

    // Next non-empty line, comments included (the parser needs the section markers).
    bool next_line(std::istream& in, std::string& line) {
      while (std::getline(in, line))
        if (!line.empty()) return true;
      return false;
    }

  }  // namespace

  DetectorSystematics::DetectorSystematics(const io::ic::Binning& binning,
                                           const std::string&     gradient_file) {
    const int total_bins = binning.total_bins();
    m_MuDelta.assign(total_bins, 0.0);
    m_SsqDelta.assign(total_bins, 0.0);
    load(gradient_file, total_bins);
  }

  void DetectorSystematics::load(const std::string& path, const int total_bins) {
    std::ifstream in(path);
    if (!in)
      throw std::runtime_error("DetectorSystematics: cannot open gradient file '" + path + "'");

    std::string line;
    if (!next_line(in, line))
      throw std::runtime_error("DetectorSystematics: '" + path + "' is empty");

    // "# gradients bins <N> params <K> lt_scale <s>"
    {
      std::istringstream header(line);
      std::string        hash, word, bins_key, params_key, scale_key;
      int                bins = -1, params = -1;
      if (!(header >> hash >> word >> bins_key >> bins >> params_key >> params >> scale_key >>
            m_LivetimeScale))
        throw std::runtime_error("DetectorSystematics: '" + path +
                                 "' has an incomplete '# gradients bins <N> params <K> lt_scale <s>' header");
      if (word != "gradients" || bins_key != "bins" || params_key != "params" || scale_key != "lt_scale")
        throw std::runtime_error("DetectorSystematics: '" + path +
                                 "' has no '# gradients bins <N> params <K> lt_scale <s>' header");
      if (bins != total_bins)
        throw std::runtime_error("DetectorSystematics: '" + path + "' declares " + std::to_string(bins) +
                                 " bins, the sample's binning has " + std::to_string(total_bins));
      if (params != params::ic::nDetSysParams)
        throw std::runtime_error("DetectorSystematics: '" + path + "' declares " + std::to_string(params) +
                                 " systematics, expected " + std::to_string(params::ic::nDetSysParams));
    }

    auto read_marker = [&](const std::string& expected_kind) {
      while (next_line(in, line)) {
        if (line.front() != '#') continue;
        std::istringstream marker(line);
        std::string        hash, kind;
        marker >> hash >> kind;
        if (kind == expected_kind) return marker;
      }
      throw std::runtime_error("DetectorSystematics: '" + path + "' has no further '# " + expected_kind +
                               "' section");
    };

    for (int k = 0; k < params::ic::nDetSysParams; ++k) {
      std::istringstream marker = read_marker("param");
      std::string        name, split_key;
      marker >> name >> split_key >> m_Split[k];
      if (name != kSystematicNames[k])
        throw std::runtime_error("DetectorSystematics: '" + path + "' systematic " + std::to_string(k) +
                                 " is '" + name + "', expected '" + kSystematicNames[k] +
                                 "' (the export script's order must match params::ic)");
      m_Gradient[k].assign(total_bins, 0.0);
      m_GradientError[k].assign(total_bins, 0.0);
      for (int b = 0; b < total_bins; ++b)
        if (!(in >> m_Gradient[k][b] >> m_GradientError[k][b]))
          throw std::runtime_error("DetectorSystematics: '" + path + "' ran out of values in '" + name + "'");
      std::getline(in, line);  // consume the rest of the last data line
    }

    for (int p = 0; p < nPairs; ++p) {
      read_marker("cov");
      m_Covariance[p].assign(total_bins, 0.0);
      for (int b = 0; b < total_bins; ++b)
        if (!(in >> m_Covariance[p][b]))
          throw std::runtime_error("DetectorSystematics: '" + path + "' ran out of covariance values");
      std::getline(in, line);
    }

    std::cout << "DetectorSystematics: loaded " << params::ic::nDetSysParams << " gradients x " << total_bins
              << " bins (+ " << nPairs << " covariance pairs, livetime scale " << m_LivetimeScale << ") from "
              << path << '\n';
  }

  bool DetectorSystematics::check_and_recalculate(const ParameterWrapper& parameter) {
    using namespace params::ic;
    if (!parameter.check_parameter_changed(DOMEff, HoleIceP1)) return false;

    // The livetime scale is folded into the deviations, so the covariance term
    // picks up lt_scale^2 as the product of two of them (NNMFit's convention).
    double deviation[nDetSysParams];
    for (int k = 0; k < nDetSysParams; ++k)
      deviation[k] = (parameter[DOMEff + k] - m_Split[k]) * m_LivetimeScale;

    for (std::size_t b = 0, n = m_MuDelta.size(); b < n; ++b) {
      double mu  = 0.0;
      double ssq = 0.0;
      for (int k = 0; k < nDetSysParams; ++k) {
        mu += deviation[k] * m_Gradient[k][b];
        const double term = deviation[k] * m_GradientError[k][b];
        ssq += term * term;
      }
      int pair = 0;
      for (int i = 0; i < nDetSysParams; ++i)
        for (int j = i + 1; j < nDetSysParams; ++j, ++pair)
          ssq += 2.0 * deviation[i] * deviation[j] * m_Covariance[pair][b];
      m_MuDelta[b]  = mu;
      m_SsqDelta[b] = ssq;
    }
    return true;
  }

}  // namespace ana::ic
```

- [ ] **Step 5: Wire it into `SampleLikelihood`**

In `SampleLikelihood.h`: add `#include "DetectorSystematics.h"` and, after `m_Template`:

```cpp
    std::optional<DetectorSystematics> m_Systematics;
```

In `SampleLikelihood.cpp`'s constructor, after the template block:

```cpp
    if (!cfg.gradient_file.empty())
      m_Systematics.emplace(cfg.binning, cfg.gradient_file);
```

In `assemble_prediction`, add the gradient to the changed flag and to the sum **before** the clip (NNMFit clips µ after the gradient):

```cpp
    if (m_Systematics) changed |= m_Systematics->check_and_recalculate(parameter);
```
```cpp
    const std::span<const double> mu_delta =
        m_Systematics ? m_Systematics->mu_delta() : std::span<const double>{};
```
```cpp
      if (!mu_delta.empty()) total += mu_delta[b];
```

In `assemble_fluctuation`, after the template term:

```cpp
    if (m_Systematics) {
      const std::span<const double> sys_ssq = m_Systematics->ssq_delta();
      for (int b = 0; b < n_bins; ++b) m_Ssq[b] += sys_ssq[b];
    }
```

- [ ] **Step 6: Build, test, golden regression**

Run:
```bash
cmake --build build -j8
ctest --test-dir build -R ICTests --output-on-failure
./build/programs/LLHFit/LLHFit -c configs/config_icecube_tracks_cpu.json --silent
python3 tools/compare_output.py Output.ic_baseline.json Output.json
```
Expected: tests pass; exit 0 (the golden config has no `Gradients` block).

- [ ] **Step 7: Commit**

```bash
git add libraries/likelihood/IceCube/DetectorSystematics.h libraries/likelihood/IceCube/DetectorSystematics.cpp \
        libraries/likelihood/IceCube/SampleLikelihood.h libraries/likelihood/IceCube/SampleLikelihood.cpp \
        programs/ictests/ICTests.cpp
git commit -m "feat(icecube): per-sample SnowStorm gradients on mu and sigma^2"
```

---

## Task 10: Split the pull width from the step, migrate to NNMFit's priors and livetimes

**Files:**
- Modify: `libraries/io/InputParameter.h`
- Modify: `libraries/likelihood/IceCube/ICLikelihood.cpp`
- Modify: `configs/config_icecube.json`, `configs/config_icecube_tracks_cpu.json`, `configs/config_icecube_tracksonly.json`
- Modify: `programs/ictests/ICTests.cpp`

`StepWidth` currently serves as both the Minuit step and the Gaussian pull σ, and the pull's central value is the start value. NNMFit's parameter set is inexpressible that way (`muon_norm` has prior 1.0 ± 0.5 but wants a small step; `delta_gamma` wants a small step and *no* prior). This task separates them, then migrates every IceCube config to NNMFit's values and re-records the baseline once.

- [ ] **Step 1: Record the Double Chooz branch state first**

Run: `tools/run_validation.sh --no-build`
Expected: note verbatim what the Double Chooz branch reports (it is known to be `ok` or `skip` depending on whether `Output.baseline.json` exists locally; a pre-existing drift was observed on 2026-07-25). Recording it now prevents attributing a pre-existing failure to this task.

- [ ] **Step 2: Write the failing test**

Append to `ICTests.cpp` (register in `main()` after `test_parameter_layout();`):

```cpp
// The Gaussian pull width must be separable from the minimiser step, while a
// config that specifies neither prior key keeps today's meaning -- the
// compatibility guarantee the Double Chooz configs rely on.
static void test_prior_defaults_and_overrides() {
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
  assert(parameters.size() == 2);

  // Legacy entry: prior falls back to StartValue / StepWidth exactly as before.
  assert(parameters.parameters()[0].value() == 1.0);
  assert(parameters.parameters()[0].uncertainty() == 0.4);
  assert(parameters.parameters()[0].prior_value() == 1.0);
  assert(parameters.parameters()[0].prior_width() == 0.4);

  // Explicit entry: step and prior are independent.
  assert(parameters.parameters()[1].uncertainty() == 0.1);
  assert(parameters.parameters()[1].prior_value() == 1.2);
  assert(parameters.parameters()[1].prior_width() == 0.5);
}
```

Add `#include "InputParameter.h"` to `ICTests.cpp` (the `io` target's include directory is already on the test target).

- [ ] **Step 3: Run to verify failure**

Run: `cmake --build build -j8`
Expected: compile error — `Parameter` has no `prior_value()` / `prior_width()`.

- [ ] **Step 4: Extend `InputParameter`**

In `libraries/io/InputParameter.h`, extend the nested `Parameter` class:

```cpp
      explicit Parameter(const boost::property_tree::ptree& parameter)
        : m_Value(parameter.get<double>("StartValue"))
        , m_Uncertainty(parameter.get<double>("StepWidth"))
        // The Gaussian pull's central value and width. Optional: they default to
        // the start value and the step width, which is what every config meant
        // before the two were separable, so existing configs keep their pulls.
        , m_PriorValue(parameter.get<double>("PriorValue", m_Value))
        , m_PriorWidth(parameter.get<double>("PriorWidth", m_Uncertainty)) {
      }
```
```cpp
      /** Minimiser step width (Minuit's initial step). */
      [[nodiscard]] double uncertainty() const noexcept { return m_Uncertainty; }

      /** Central value of the Gaussian pull on a constrained parameter. */
      [[nodiscard]] double prior_value() const noexcept { return m_PriorValue; }

      /** Width (sigma) of the Gaussian pull on a constrained parameter. */
      [[nodiscard]] double prior_width() const noexcept { return m_PriorWidth; }
```
```cpp
      double m_Value;        ///< The start value of the parameter.
      double m_Uncertainty;  ///< The minimiser step width.
      double m_PriorValue;   ///< Central value of the Gaussian pull.
      double m_PriorWidth;   ///< Width of the Gaussian pull.
```

In `libraries/likelihood/IceCube/ICLikelihood.cpp`'s `setup_pulls`, switch to the prior fields (Double Chooz keeps reading `value()`/`uncertainty()`, so its behaviour is untouched):

```cpp
    for (std::size_t i = 0; i < parameters.size(); ++i) {
      if (constrained[i]) {
        std::cout << "IC pull: " << names[i]
                  << " CV=" << parameters[i].prior_value()
                  << " sigma=" << parameters[i].prior_width() << '\n';
        m_Pulls.emplace_back(static_cast<int>(i),
                             parameters[i].prior_value(),
                             parameters[i].prior_width());
      }
    }
```

- [ ] **Step 5: Prove the equivalence before migrating**

Run:
```bash
cmake --build build -j8
ctest --test-dir build -R ICTests --output-on-failure
./build/programs/LLHFit/LLHFit -c configs/config_icecube_tracks_cpu.json --silent
python3 tools/compare_output.py Output.ic_baseline.json Output.json
```
Expected: tests pass and the golden config — which specifies no prior keys — is still **bit-for-bit**. This is the check that the plumbing change alone is behaviour-preserving; only after it passes may the configs change.

- [ ] **Step 6: Migrate every IceCube config to NNMFit's values**

In `configs/config_icecube.json`, `configs/config_icecube_tracks_cpu.json` and
`configs/config_icecube_tracksonly.json`: set the tracks sample's `"livetime"` to `410978234.97`
and replace the whole `"Parameter"` array with:

```json
  "Parameter": [
    { "Name": "AstroNorm",     "StartValue": 1.5, "StepWidth": 0.1,  "Fixed": false, "Constrained": false },
    { "Name": "SpectralIndex", "StartValue": 2.4, "StepWidth": 0.1,  "Fixed": false, "Constrained": false },
    { "Name": "ConvNorm",      "StartValue": 1.0, "StepWidth": 0.1,  "Fixed": false, "Constrained": false },
    { "Name": "PromptNorm",    "StartValue": 0.5, "StepWidth": 0.1,  "Fixed": false, "Constrained": false },
    { "Name": "BarrH",         "StartValue": 0.0, "StepWidth": 0.05, "PriorValue": 0.0, "PriorWidth": 0.15, "Fixed": false, "Constrained": true  },
    { "Name": "BarrW",         "StartValue": 0.0, "StepWidth": 0.05, "PriorValue": 0.0, "PriorWidth": 0.4,  "Fixed": false, "Constrained": true  },
    { "Name": "BarrY",         "StartValue": 0.0, "StepWidth": 0.05, "PriorValue": 0.0, "PriorWidth": 0.3,  "Fixed": false, "Constrained": true  },
    { "Name": "BarrZ",         "StartValue": 0.0, "StepWidth": 0.05, "PriorValue": 0.0, "PriorWidth": 0.12, "Fixed": false, "Constrained": true  },
    { "Name": "CRGrad",        "StartValue": 0.0, "StepWidth": 0.1,  "PriorValue": 0.0, "PriorWidth": 1.0,  "Fixed": false, "Constrained": true  },
    { "Name": "DeltaGamma",    "StartValue": 0.0, "StepWidth": 0.05, "Fixed": false, "Constrained": false },
    { "Name": "MuonNorm",      "StartValue": 1.0, "StepWidth": 0.1,  "PriorValue": 1.0, "PriorWidth": 0.5,  "Fixed": true,  "Constrained": false },
    { "Name": "MuonGunNorm",   "StartValue": 1.0, "StepWidth": 0.1,  "Fixed": true,  "Constrained": false },
    { "Name": "VetoThreshold", "StartValue": 0.0, "StepWidth": 0.1,  "Fixed": true,  "Constrained": false },
    { "Name": "DOMEff",        "StartValue": 1.0, "StepWidth": 0.02, "Fixed": true,  "Constrained": false },
    { "Name": "IceAbs",        "StartValue": 1.0, "StepWidth": 0.02, "Fixed": true,  "Constrained": false },
    { "Name": "IceScat",       "StartValue": 1.0, "StepWidth": 0.02, "Fixed": true,  "Constrained": false },
    { "Name": "HoleIceP0",     "StartValue": 0.24901831812365854,  "StepWidth": 0.05, "Fixed": true, "Constrained": false },
    { "Name": "HoleIceP1",     "StartValue": -0.05678798504997925, "StepWidth": 0.02, "Fixed": true, "Constrained": false }
  ]
```

Values are NNMFit's: `prompt_norm` default 0.5; no prior on `conv_norm`, `prompt_norm` or
`delta_gamma`; Barr widths 0.15 / 0.4 / 0.3 / 0.12; `CR_grad` 1.0. `MuonNorm` keeps its NNMFit prior
recorded even while `Fixed` (its template is not exported yet), so enabling it later needs no config
archaeology. NNMFit's parameter ranges are not expressible here — do not add them.

- [ ] **Step 7: Two equivalence checks, then re-record the baseline**

Run:
```bash
./build/programs/LLHFit/LLHFit -c configs/config_icecube_tracks_cpu.json --silent
python3 - <<'EOF'
import json
old = json.load(open('Output.ic_baseline.json'))
new = json.load(open('Output.json'))
ratio = 410978234.97 / 3.0e8
print('Asimov old', old['dataTotal'], 'new', new['dataTotal'])
print('expected new/old', ratio, 'observed', new['dataTotal'] / old['dataTotal'])
print('chi2 old', old['chi2'], 'new', new['chi2'])
EOF
```
Expected: `observed` equals `expected` to ~1e-12 — a livetime is a linear factor on every per-event
weight, so the Asimov total must scale by exactly `410978234.97 / 3.0e8 = 1.3699274499`. `chi2`
changes (different livetime and different priors); that is the point of this task.

Then check the reverse direction: copy the migrated golden config, set the tracks livetime back to
`3.0e8` and drop the `PriorValue`/`PriorWidth` keys plus restore `Constrained: true` on ConvNorm
(0.4), PromptNorm (0.5 with StartValue 1.0), Barr* (1.0 each) and DeltaGamma (0.05), then run it and
`compare_output.py` against `Output.ic_baseline.json`.
Expected: bit-for-bit `IDENTICAL` — proving the migration changed only inputs, not code paths.

Now re-record:
```bash
./build/programs/LLHFit/LLHFit -c configs/config_icecube_tracks_cpu.json --silent
cp Output.json Output.ic_baseline.json
```
Record in the commit message: the pre-migration anchors (`chi2 = -6366527.142871824`, Asimov
514973), the observed livetime ratio, and the new `chi2` / Asimov total.

- [ ] **Step 8: Fix the parameter-0 seeding quirk**

`libraries/likelihood/Fit.cpp` (~line 68) seeds fit parameter **index 0 with a hardcoded 1.0**,
ignoring its config `StartValue`:

```cpp
      if (i != 0)
        m_Minimizer->SetVariable(i, names[i], parameters[i].value(), parameters[i].uncertainty());
      else
        m_Minimizer->SetVariable(i, names[i], 1.0, parameters[i].uncertainty());
```

For IceCube index 0 is `AstroNorm` (start 1.5), so an all-fixed evaluation silently runs at 1.0 — an
exactly 2/3 prediction deficit in the astro component, which would make Task 13's fixed-parameter
comparison meaningless. Replace both branches with the single correct call:

```cpp
      m_Minimizer->SetVariable(i, names[i], parameters[i].value(), parameters[i].uncertainty());
```

This is shared-core code, so verify Double Chooz explicitly in the next step: its parameter 0 must
either be nominally 1.0 (in which case nothing changes) or the change is a genuine fix that moves its
baseline — if the Double Chooz branch moves, stop and report rather than re-recording its baseline.

- [ ] **Step 9: Double Chooz must be unaffected**

Run:
```bash
cmake --build build -j8
tools/run_validation.sh --no-build
```
Expected: the Double Chooz branch reports exactly what Step 1 recorded. The new prior keys are absent
from its configs, so its pulls are unchanged by construction; the Step 8 change is only visible if
its parameter 0 has a `StartValue` other than 1.0 — check that config value and state it in the
commit message either way.

- [ ] **Step 10: Commit**

```bash
git add libraries/io/InputParameter.h libraries/likelihood/Fit.cpp \
        libraries/likelihood/IceCube/ICLikelihood.cpp \
        configs/config_icecube.json configs/config_icecube_tracks_cpu.json \
        configs/config_icecube_tracksonly.json programs/ictests/ICTests.cpp
git commit -m "feat(io): separate Gaussian prior from step width; adopt NNMFit priors and livetimes"
```

---

## Task 11: Cascade and combined configs, end-to-end validation

**Files:**
- Create: `configs/config_icecube_cascades.json`, `configs/config_icecube_combined.json`
- Modify: `libraries/results/IceCube/ICWriteResults.h`, `libraries/likelihood/IceCube/SampleLikelihood.h`
- Modify: `tools/run_validation.sh`

- [ ] **Step 1: Write the cascade-only config**

Create `configs/config_icecube_cascades.json`:

```json
{
  "Experiment": "IceCube",
  "IceCube": {
    "Backend": "cpu",
    "Likelihood": "SAY",
    "UseData": false,
    "ERefGeV": 100000.0,
    "AstroReferenceIndex": 2.0,
    "AstroPerTypeNorm": false,
    "ConvDeltaGammaERef": 1000.0,
    "PromptDeltaGammaERef": 3800.0,
    "VetoAnchorEnergy": 100.0,
    "VetoRescaleEnergy": 100.0,
    "Binnings": {
      "cscd_cascade_2d": {
        "axes": "Log10Energy, CosZenith",
        "Log10Energy": "(2.8, 7.0, 21)",
        "CosZenith":   "[-1.0, -0.76, -0.52, -0.28, -0.04, 0.2, 0.6, 1.0]"
      },
      "cscd_muon_2d": {
        "axes": "Log10Energy, CosZenith",
        "Log10Energy": "(2.6, 4.8, 1)",
        "CosZenith":   "(-1.0, 1.0, 1)"
      }
    },
    "Samples": {
      "cscd_cascade": {
        "enabled": true,
        "binning": "cscd_cascade_2d",
        "parquet": "/Users/soldin/Downloads/nnmfit_files/datasets/cscd_cascade_ftp_baseline/dataset_cscd_cascade_FTP_baseline_wCoords.parquet",
        "data":    "/Users/soldin/Downloads/nnmfit_files/datasets/data/data_cscd_cascade_FTP_Monopod_wCoords.parquet",
        "livetime": 330315015.11,
        "components": "astro, conventional_veto, prompt_veto, muon",
        "Template":  { "File": "/Users/soldin/Downloads/nnmfit_files/exported/template_cscd_cascade.txt", "Norm": "MuonGunNorm" },
        "Gradients": { "File": "/Users/soldin/Downloads/nnmfit_files/exported/gradients_cscd_cascade.txt" },
        "Branches":  { "RecoEnergy": "energy_monopod", "RecoZenith": "zenith_monopod" }
      },
      "cscd_muon": {
        "enabled": true,
        "binning": "cscd_muon_2d",
        "parquet": "/Users/soldin/Downloads/nnmfit_files/datasets/cscd_muon_ftp_baseline/dataset_cscd_muon_FTP_baseline_wCoords.parquet",
        "data":    "/Users/soldin/Downloads/nnmfit_files/datasets/data/data_cscd_muon_wCoords.parquet",
        "livetime": 330315015.11,
        "components": "astro, conventional_veto, prompt_veto, muon",
        "Template":  { "File": "/Users/soldin/Downloads/nnmfit_files/exported/template_cscd_muon.txt", "Norm": "MuonGunNorm" },
        "Gradients": { "File": "/Users/soldin/Downloads/nnmfit_files/exported/gradients_cscd_muon.txt" },
        "Branches":  { "RecoEnergy": "energy_monopod", "RecoZenith": "zenith_monopod" }
      }
    }
  },
  "Parameter": [ /* copy the migrated array from Task 10 Step 6, then apply the changes below */ ]
}
```

Copy the `Parameter` array verbatim from the migrated `config_icecube_tracks_cpu.json`, then change
only these entries (the cascade samples float what they constrain): `MuonGunNorm` `"Fixed": false`,
`VetoThreshold` `"Fixed": false`, and `DOMEff` / `IceAbs` / `IceScat` / `HoleIceP0` / `HoleIceP1`
`"Fixed": false`. `MuonNorm` stays `Fixed` (no cascade sample uses the Corsika template). Replace the
placeholder comment with the real array — JSON has no comments.

- [ ] **Step 2: Write the combined config**

Create `configs/config_icecube_combined.json` as the cascade config plus the tracks sample and its
binning, all three enabled. Add to `Binnings`:

```json
      "tracks_2d": {
        "axes": "Log10Energy, CosZenith",
        "Log10Energy": "(2.5, 7.0, 45)",
        "CosZenith":   "(-1.0, 0.0872, 33)"
      },
```

and to `Samples`:

```json
      "tracks": {
        "enabled": true,
        "binning": "tracks_2d",
        "parquet": "/Users/soldin/Downloads/nnmfit_files/datasets/tracks_ftp_baseline/dataset_tracks_baseline.parquet",
        "data":    "/Users/soldin/Downloads/nnmfit_files/datasets/data/dataset_data_tracks_IC2010_to_IC2022_no_cscd_cascade_cscd_muon_wCoords.parquet",
        "livetime": 410978234.97,
        "components": "astro, conventional, prompt, muontemplate",
        "Template":  { "File": "/Users/soldin/Downloads/nnmfit_files/exported/template_tracks.txt", "Norm": "MuonNorm" },
        "Gradients": { "File": "/Users/soldin/Downloads/nnmfit_files/exported/gradients_tracks.txt" },
        "Branches":  { "RecoEnergy": "energy_truncated", "RecoZenith": "zenith_MPEFit" }
      },
```

In the combined config's `Parameter` array, free `MuonNorm` (`"Fixed": false`) — the reference fit
floats it under its 1.0 ± 0.5 prior and fitted 1.827. Also update the cascade config's sample paths
to the same `$NF/datasets/...` tree.

- [ ] **Step 3: Run the cascade-only fit**

Run:
```bash
./build/programs/LLHFit/LLHFit -c configs/config_icecube_cascades.json --silent
```
Expected: both samples load (1,409,344 and 587,538 rows; 147 and 1 bins), `TemplateFlux` and
`DetectorSystematics` log their loads, and the fit converges. Sanity checks on `Output.json`:
- the cascade template at norm 1 is `2.3425769796686227e-06 × 330315015.11 ≈ 773.8` events and the
  muon-sample template `5.4694021304430105e-05 × 330315015.11 ≈ 18066` events — the numbers
  `TemplateFlux` prints at load;
- `samples[1].totalBins == 1` with one-entry `data`/`prediction` arrays;
- `dataTotal` finite and positive.

Save the baseline:
```bash
cp Output.json Output.ic_cascades_baseline.json
```

- [ ] **Step 4: Add the per-sample component breakdown to the results**

So a mis-scaled template or gradient is visible instead of hidden in a total, add accessors in
`SampleLikelihood.h`:

```cpp
    /** Per-bin prediction of one named part of this sample, for the results writer. */
    [[nodiscard]] std::span<const double> astro_histogram() const noexcept {
      return m_Astro ? m_Astro->histogram() : std::span<const double>{};
    }
    [[nodiscard]] std::span<const double> atmospheric_histogram() const noexcept {
      return m_Atmo ? m_Atmo->histogram() : std::span<const double>{};
    }
    [[nodiscard]] std::span<const double> template_histogram() const noexcept {
      return m_Template ? m_Template->histogram() : std::span<const double>{};
    }
    [[nodiscard]] std::span<const double> systematics_mu_delta() const noexcept {
      return m_Systematics ? m_Systematics->mu_delta() : std::span<const double>{};
    }
```

and in `ICWriteResults.h`'s `get_json_file`, inside the per-sample loop before the
`j["samples"].push_back(...)`:

```cpp
      auto sum_of = [](const std::span<const double> values) {
        double total = 0.0;
        for (const double v : values) total += v;
        return total;
      };
      nlohmann::json components = {
          {"astro", sum_of(sample.astro_histogram())},
          {"atmospheric", sum_of(sample.atmospheric_histogram())},
          {"template", sum_of(sample.template_histogram())},
          {"systematicsDelta", sum_of(sample.systematics_mu_delta())},
      };
```

then add `{"componentTotals", std::move(components)},` to the pushed object. Also emit the per-bin
component arrays under `{"componentBins", {...}}` using the same spans — Task 13 diffs those against
NNMFit per component, and totals alone would hide a shape error.

- [ ] **Step 5: Composite summation test across three binnings**

Run:
```bash
python3 - <<'EOF'
import json, subprocess, copy
base = json.load(open('configs/config_icecube_combined.json'))
for p in base['Parameter']:
    p['Fixed'] = True

def run(cfg, path):
    json.dump(cfg, open(path, 'w'), indent=2)
    out = subprocess.run(['./build/programs/LLHFit/LLHFit', '-c', path, '--silent'],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    return json.load(open('Output.json'))

total = run(base, '/tmp/ic_combined_fixed.json')
parts = {}
for name in base['IceCube']['Samples']:
    cfg = copy.deepcopy(base)
    for other in cfg['IceCube']['Samples']:
        cfg['IceCube']['Samples'][other]['enabled'] = (other == name)
    parts[name] = run(cfg, f'/tmp/ic_{name}_fixed.json')

print('combined chi2', total['chi2'], 'predTotal', total['predTotal'])
for name, out in parts.items():
    print(' ', name, 'chi2', out['chi2'], 'predTotal', out['predTotal'])
print('sum predTotal', sum(o['predTotal'] for o in parts.values()))
print('sum chi2', sum(o['chi2'] for o in parts.values()))
EOF
```
Expected: `combined predTotal` equals the sum of the parts to ~1e-9 relative. With every constrained
parameter starting at its prior central value the pull term is 0, so the `chi2` values must also add
up; if they do not, the composite is double-counting pulls. Record the observed numbers in the commit
message.

Note: parameter index 0 (`AstroNorm`) is forced to 1.0 by `Fit.cpp` regardless of `StartValue`, so
this all-fixed run evaluates at `AstroNorm = 1.0`, not 1.5. That is a known pre-existing quirk of the
shared core; it does not affect the additivity check (both sides see the same value).

- [ ] **Step 6: Extend `run_validation.sh`**

Add a cascade branch mirroring the existing IceCube one: if `Output.ic_cascades_baseline.json`
exists, run `LLHFit -c configs/config_icecube_cascades.json --silent` and `compare_output.py`, else
report `skip`. Follow the exact shape of the existing IceCube block (`IC_CONFIG`/`IC_BASELINE`
pattern, with `IC_CASCADE_CONFIG`/`IC_CASCADE_BASELINE` defaults and a mention in the script's header
comment).

- [ ] **Step 7: Full validation**

Run:
```bash
cmake --build build -j8
ctest --test-dir build -R ICTests --output-on-failure
tools/run_validation.sh --no-build
```
Expected: `ICTests` passes; the tracks branch is `ok` (bit-for-bit against the Task-10 baseline), the
cascade branch is `ok`, nothing `FAILED`.

- [ ] **Step 8: Metal parity on the combined config**

Run the combined config with `"Backend": "metal"` and compare parameter-by-parameter with the CPU run
using the snippet from Task 6 Step 8.
Expected: meaningful parameters within 1e-3. The single-bin `cscd_muon` sample exercises a
one-threadgroup dispatch — check its `predTotal` against the CPU value specifically.

- [ ] **Step 9: Commit**

```bash
git add configs/config_icecube_cascades.json configs/config_icecube_combined.json \
        libraries/results/IceCube/ICWriteResults.h libraries/likelihood/IceCube/SampleLikelihood.h \
        tools/run_validation.sh
git commit -m "feat(icecube): cascade and combined anchor configs with per-component output"
```

---

## Task 12: Real-data histograms

**Files:**
- Modify: `libraries/io/IceCube/Binning.h`, `.cpp` (the pure counting helper)
- Modify: `libraries/io/IceCube/ICDataBase.h`, `.cpp`
- Modify: `libraries/likelihood/IceCube/SampleLikelihood.h`, `.cpp`
- Modify: `libraries/likelihood/IceCube/ICLikelihood.cpp`
- Modify: `programs/ictests/ICTests.cpp`

- [ ] **Step 1: Check the data parquets against NNMFit's standard mask**

The MC baselines need no mask (measured: 0 rows dropped from both cascade files). The data files are
unverified, so measure before trusting them:

```bash
cat > /tmp/maskcount.cpp <<'EOF'
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>
#include <cstdio>
#include <memory>
#include <string>
int main(int argc, char** argv) {
  const std::string prefix = argv[2];
  auto infile = arrow::io::ReadableFile::Open(argv[1]).ValueOrDie();
  auto reader = parquet::arrow::OpenFile(infile, arrow::default_memory_pool()).ValueOrDie();
  std::shared_ptr<arrow::Table> table;
  if (!reader->ReadTable(&table).ok()) return 1;
  table = table->CombineChunks().ValueOrDie();
  auto get = [&](const std::string& n) { auto c = table->GetColumnByName(n); return c ? c->chunk(0) : nullptr; };
  auto ee = std::static_pointer_cast<arrow::UInt8Array>(get(prefix + "_exists"));
  auto es = std::static_pointer_cast<arrow::Int32Array>(get(prefix + "_fit_status"));
  auto de = std::static_pointer_cast<arrow::UInt8Array>(get("reco_dir_exists"));
  auto ds = std::static_pointer_cast<arrow::Int32Array>(get("reco_dir_fit_status"));
  if (!ee || !es || !de || !ds) { std::printf("missing mask columns\n"); return 2; }
  const int64_t n = table->num_rows();
  int64_t pass = 0;
  for (int64_t i = 0; i < n; ++i)
    pass += (ee->Value(i) == 1 && es->Value(i) == 0 && de->Value(i) == 1 && ds->Value(i) == 0);
  std::printf("%s: %lld rows, %lld pass (%.4f%% dropped)\n", argv[1], (long long)n, (long long)pass,
              100.0 * (n - pass) / n);
  return 0;
}
EOF
clang++ -std=c++20 -O2 -o /tmp/maskcount /tmp/maskcount.cpp \
  -I/opt/homebrew/opt/apache-arrow/include -L/opt/homebrew/opt/apache-arrow/lib -larrow -lparquet
NF=/Users/soldin/Downloads/nnmfit_files
/tmp/maskcount "$NF/datasets/data/data_cscd_cascade_FTP_Monopod_wCoords.parquet" energy_monopod
/tmp/maskcount "$NF/datasets/data/data_cscd_muon_wCoords.parquet" energy_monopod
/tmp/maskcount "$NF/datasets/data/dataset_data_tracks_IC2010_to_IC2022_no_cscd_cascade_cscd_muon_wCoords.parquet" energy_truncated
```
If any file drops a non-zero fraction, apply the same four conditions in `read_data_histogram` below
and record the fractions in the commit message. If all three drop 0%, note that and skip the mask.

- [ ] **Step 2: Write the failing test**

Append to `ICTests.cpp` (register in `main()` last):

```cpp
// Real data is a plain per-bin count in the sample's own binning: no weights, no
// livetime scaling. This tests the counting helper the loader uses.
static void test_data_histogram_counts() {
  const Binning binning({io::ic::parse_axis("Log10Energy", "(2.0, 5.0, 3)"),
                         io::ic::parse_axis("CosZenith", "(-1.0, 1.0, 2)")});
  assert(binning.total_bins() == 6);

  // Three events in bin (energy 0, zenith 0), one out of range.
  const std::vector<double> energies{316.0, 316.0, 316.0, 10.0};
  const std::vector<double> zeniths{2.0, 2.0, 2.0, 2.0};

  const std::vector<double> counts = io::ic::bin_event_counts(binning, energies, zeniths);
  assert(counts.size() == 6);
  assert(counts[0] == 3.0);
  for (std::size_t b = 1; b < counts.size(); ++b) assert(counts[b] == 0.0);
}
```

- [ ] **Step 3: Add the counting helper**

Declare it in `libraries/io/IceCube/Binning.h`, not in `ICDataBase.h`: it is pure binning logic, and
`ICTests` links `icecube` with Arrow attached PRIVATE, so a test including `ICDataBase.h` would fail
to find `arrow/type_fwd.h`.

```cpp
  /**
   * Per-bin event counts for a data sample: bins each (reco energy, reco zenith)
   * pair with `binning` and counts, dropping out-of-range events. No weights and
   * no livetime scaling -- real data is a count.
   */
  [[nodiscard]] std::vector<double> bin_event_counts(const Binning&             binning,
                                                    const std::vector<double>& reco_energy,
                                                    const std::vector<double>& reco_zenith);
```

and in `libraries/io/IceCube/Binning.cpp` (add `#include <array>`):

```cpp
  std::vector<double> bin_event_counts(const Binning&             binning,
                                       const std::vector<double>& reco_energy,
                                       const std::vector<double>& reco_zenith) {
    std::vector<double> counts(binning.total_bins(), 0.0);
    for (std::size_t i = 0, n = reco_energy.size(); i < n; ++i) {
      const std::array<double, 2> reco{reco_energy[i], reco_zenith[i]};
      const int                   bin = binning.bin_index(reco);
      if (bin >= 0) counts[bin] += 1.0;
    }
    return counts;
  }
```

- [ ] **Step 4: Load the data histograms**

In `ICDataBase.h` add the accessor and members:

```cpp
    /** Per-bin measured counts for sample i; empty when the config gave no data path. */
    [[nodiscard]] const std::vector<double>& data_histogram(std::size_t i) const noexcept {
      return m_DataHistograms[i];
    }
```
```cpp
    std::vector<std::vector<double>> m_DataHistograms;
    arrow::Status read_data_histogram(const SampleConfig& cfg, std::vector<double>& out);
```

In `ICDataBase.cpp`, implement it and call it from the constructor loop (once per enabled sample, so
`m_DataHistograms` stays index-aligned with `m_Samples`):

```cpp
  arrow::Status ICDataBase::read_data_histogram(const SampleConfig& cfg, std::vector<double>& out) {
    if (cfg.data_path.empty()) {
      out.clear();
      return arrow::Status::OK();
    }
    std::cout << "Reading IceCube data for sample '" << cfg.name << "': " << cfg.data_path << '\n';
    ARROW_ASSIGN_OR_RAISE(auto table, read_parquet_file(cfg.data_path));
    ARROW_ASSIGN_OR_RAISE(auto energy, get_double_column(*table, cfg.branches.reco_energy));
    ARROW_ASSIGN_OR_RAISE(auto zenith, get_double_column(*table, cfg.branches.reco_zenith));
    out = bin_event_counts(cfg.binning, energy, zenith);
    double total = 0.0;
    for (const double v : out) total += v;
    std::cout << "IceCube data '" << cfg.name << "': " << energy.size() << " rows, " << total
              << " in analysis range\n";
    return arrow::Status::OK();
  }
```
```cpp
      std::vector<double> data_histogram;
      const auto          data_status = read_data_histogram(cfg, data_histogram);
      if (!data_status.ok())
        throw std::runtime_error("Failed to read IceCube data for sample '" + cfg.name + "': " +
                                 data_status.ToString());
      m_DataHistograms.push_back(std::move(data_histogram));
```

- [ ] **Step 5: Use the data in the likelihood**

In `SampleLikelihood.h`:

```cpp
    /** Replace the Asimov expectation with measured counts (UseData). */
    void set_data(std::span<const double> counts);
```

in `SampleLikelihood.cpp`:

```cpp
  void SampleLikelihood::set_data(const std::span<const double> counts) {
    if (counts.size() != m_Data.size())
      throw std::runtime_error("SampleLikelihood: data histogram for sample '" + m_Config.name +
                               "' has " + std::to_string(counts.size()) + " bins, the binning has " +
                               std::to_string(m_Data.size()));
    std::ranges::copy(counts, m_Data.begin());

    // The SAY ssq describes MC statistics, so it still comes from the model; seed
    // it exactly as generate_asimov does.
    if (m_UseSAY) assemble_fluctuation();
  }
```

In `ICLikelihood.cpp`, replace the throwing branch of `initialize_data`:

```cpp
  void ICLikelihood::initialize_data(const bool use_data) {
    const auto&         parameters = m_Options->inputOptions().input_parameters().parameters();
    std::vector<double> nominal(params::ic::number_of_parameters());
    for (std::size_t i = 0; i < nominal.size(); ++i)
      nominal[i] = parameters[i].value();

    m_Parameter.reset_parameter(nominal.data());

    double total = 0.0;
    for (std::size_t k = 0; k < m_Samples.size(); ++k) {
      // The prediction at the nominal point is needed either way: as the Asimov
      // expectation, or to seed the SAY ssq before the measured counts replace it.
      m_Samples[k]->generate_asimov(m_Parameter);
      if (use_data) {
        const auto& counts = m_DataBase->data_histogram(k);
        if (counts.empty())
          throw std::runtime_error("ICLikelihood: UseData is true but sample " + std::to_string(k) +
                                   " has no \"data\" path in its config");
        m_Samples[k]->set_data(counts);
      }
      for (const double v : m_Samples[k]->data()) total += v;
    }
    std::cout << "IC " << (use_data ? "data" : "Asimov") << " total events: " << total << '\n';
  }
```

- [ ] **Step 6: Build, test, run a data fit**

Run:
```bash
cmake --build build -j8
ctest --test-dir build -R ICTests --output-on-failure
./build/programs/LLHFit/LLHFit -c configs/config_icecube_tracks_cpu.json --silent
python3 tools/compare_output.py Output.ic_baseline.json Output.json
python3 -c "
import json;c=json.load(open('configs/config_icecube_combined.json'));c['IceCube']['UseData']=True;json.dump(c,open('/tmp/ic_combined_data.json','w'),indent=2)"
./build/programs/LLHFit/LLHFit -c /tmp/ic_combined_data.json --silent
```
Expected: unit tests pass; the Asimov golden regression is still bit-for-bit; the data run prints
`IC data total events: <N>` where N matches the in-range row counts the loader reported, and
converges. Record the fitted `AstroNorm`/`SpectralIndex` in the commit message.

- [ ] **Step 7: Commit**

```bash
git add libraries/io/IceCube/Binning.h libraries/io/IceCube/Binning.cpp \
        libraries/io/IceCube/ICDataBase.h libraries/io/IceCube/ICDataBase.cpp \
        libraries/likelihood/IceCube/SampleLikelihood.h libraries/likelihood/IceCube/SampleLikelihood.cpp \
        libraries/likelihood/IceCube/ICLikelihood.cpp programs/ictests/ICTests.cpp
git commit -m "feat(icecube): per-sample real-data histograms"
```

---

## Task 13: Diff against the NNMFit oracle (the acceptance gate)

**Files:**
- Create: `tools/nnmfit_oracle/compare_to_nnmfit.py`

- [ ] **Step 1: Write the comparison script**

Create `tools/nnmfit_oracle/compare_to_nnmfit.py`:

```python
#!/usr/bin/env python3
"""Diff PhyLiNO's per-bin IceCube prediction against NNMFit's dumped histograms.

Reads an LLHFit Output.json (which carries per-sample componentBins) and the
pickles produced by tools/nnmfit_oracle/dump_histograms.sh, and reports the
largest relative deviation per sample and per component.

Usage: compare_to_nnmfit.py Output.json /tmp/nnmfit_dumps [--tolerance 1e-8]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# PhyLiNO sample name -> NNMFit detector config
SAMPLES = {
    "tracks": "IC86_pass2_SnowStorm_v2_tracks",
    "cscd_cascade": "IC86_pass2_SnowStorm_v2_cscd_cascade",
    "cscd_muon": "IC86_pass2_SnowStorm_v2_cscd_muon",
}

# PhyLiNO componentBins key -> the NNMFit components it sums over. PhyLiNO's
# AtmosphericFlux computes conventional and prompt in one pass, so it is compared
# against the sum of NNMFit's two components (veto variants for the cascades).
COMPONENTS = {
    "astro": ["astro"],
    "atmospheric": ["conventional", "prompt"],
    "atmospheric_veto": ["conventional_veto", "prompt_veto"],
    "template": ["muontemplate", "muon"],
}


def nnmfit_mu(dump_dir, det_conf, components):
    total = None
    for component in components:
        path = Path(dump_dir) / f"{det_conf}_{component}.pickle"
        if not path.exists():
            continue
        with open(path, "rb") as f:
            mu = np.asarray(pickle.load(f)["mu"], dtype=float).reshape(-1)
        total = mu if total is None else total + mu
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_json")
    p.add_argument("dump_dir")
    p.add_argument("--tolerance", type=float, default=1e-8)
    args = p.parse_args()

    with open(args.output_json) as f:
        output = json.load(f)

    failures = []
    for sample in output["samples"]:
        det_conf = SAMPLES.get(sample["name"])
        if det_conf is None:
            print(f"skip {sample['name']}: no NNMFit detector config mapped")
            continue

        for key, components in COMPONENTS.items():
            ours = sample.get("componentBins", {}).get(key)
            if not ours:
                continue
            theirs = nnmfit_mu(args.dump_dir, det_conf, components)
            if theirs is None:
                print(f"skip {sample['name']}/{key}: no NNMFit dump")
                continue

            ours = np.asarray(ours, dtype=float)
            if ours.shape != theirs.shape:
                failures.append(f"{sample['name']}/{key}: shape {ours.shape} vs {theirs.shape}")
                continue

            scale = max(np.abs(theirs).max(), 1e-300)
            deviation = np.abs(ours - theirs) / scale
            worst = int(np.argmax(deviation))
            status = "ok" if deviation[worst] <= args.tolerance else "FAILED"
            print(
                f"{status:6s} {sample['name']:14s} {key:18s} "
                f"max rel dev {deviation[worst]:.3e} in bin {worst} "
                f"(ours {ours[worst]!r} vs {theirs[worst]!r}); "
                f"sums {ours.sum()!r} vs {theirs.sum()!r}"
            )
            if status == "FAILED":
                failures.append(f"{sample['name']}/{key}: {deviation[worst]:.3e}")

    if failures:
        print("\nfailures:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\nall compared components agree")


if __name__ == "__main__":
    main()
```

Note the key names: Task 11 Step 4 must emit `componentBins.atmospheric` for a plain-atmospheric
sample and `componentBins.atmospheric_veto` for a veto sample (name the key from
`cfg.wants_veto()`), so this script compares like with like.

- [ ] **Step 2: Produce our prediction at NNMFit's parameter values**

Task 1 dumped two sets: at NNMFit's config **defaults** (`/tmp/nnmfit_dumps`) and at its **fitted**
values (`/tmp/nnmfit_dumps_fitted`). Build the matching all-`Fixed` configs so the written
`Output.json` holds the prediction at exactly those points rather than at a minimum. `Fit.cpp`'s
parameter-0 seeding was fixed in Task 10, so `AstroNorm` now takes its configured `StartValue`.

```bash
python3 - <<'EOF'
import json

fit = json.load(open('/tmp/nnmfit_fit_reference.json'))   # written in Task 1 Step 1
base = json.load(open('configs/config_icecube_combined.json'))
base['IceCube']['UseData'] = False

# NNMFit's config defaults, i.e. what /tmp/nnmfit_dumps corresponds to.
defaults = {'AstroNorm': 1.5, 'SpectralIndex': 2.4, 'ConvNorm': 1.0, 'PromptNorm': 0.5,
            'BarrH': 0.0, 'BarrW': 0.0, 'BarrY': 0.0, 'BarrZ': 0.0, 'CRGrad': 0.0,
            'DeltaGamma': 0.0, 'MuonNorm': 1.0, 'MuonGunNorm': 1.0, 'VetoThreshold': 0.0,
            'DOMEff': 1.0, 'IceAbs': 1.0, 'IceScat': 1.0,
            'HoleIceP0': 0.24901831812365854, 'HoleIceP1': -0.05678798504997925}

for label, values in (('defaults', defaults), ('fitted', fit['fitted'])):
    cfg = json.loads(json.dumps(base))
    missing = [p['Name'] for p in cfg['Parameter'] if p['Name'] not in values]
    assert not missing, f'no {label} value for {missing}'
    for p in cfg['Parameter']:
        p['StartValue'] = values[p['Name']]
        p['Fixed'] = True
    json.dump(cfg, open(f'/tmp/ic_oracle_{label}.json', 'w'), indent=2)
    print('wrote', f'/tmp/ic_oracle_{label}.json')
EOF

for label in defaults fitted; do
  ./build/programs/LLHFit/LLHFit -c /tmp/ic_oracle_$label.json --silent
  cp Output.json /tmp/ic_oracle_${label}_output.json
done
```
Expected: both runs exit immediately (nothing to minimise) and write per-sample `componentBins`. The
two outputs must differ — if they do not, the `StartValue` override did not reach the components and
the comparison would be vacuous.

- [ ] **Step 3: Run the per-bin diff at both points**

Run:
```bash
python3 tools/nnmfit_oracle/compare_to_nnmfit.py /tmp/ic_oracle_defaults_output.json \
        /tmp/nnmfit_dumps --tolerance 1e-8
python3 tools/nnmfit_oracle/compare_to_nnmfit.py /tmp/ic_oracle_fitted_output.json \
        /tmp/nnmfit_dumps_fitted --tolerance 1e-8
```
Expected: `ok` for every compared component of every sample in **both** runs; both sides are FP64
CPU, so agreement should be at the 1e-10 level. The defaults point leaves `VetoThreshold = 0` and the
det-sys parameters at their split values, so only the fitted point actually exercises the veto
exponent and the gradient sums — a pass at defaults alone proves little. Interpretation of failures:
- a constant ratio across all bins → a livetime, a norm convention (`per_type_norm` halving) or a
  units factor;
- agreement in sums but not per bin → a flattening/orientation mismatch (spec risks 1 and 2);
- one component off, others fine → that component's formula (veto sign, delta-gamma reference
  energy, gradient split value);
- agreement at defaults but not at the fitted point → the veto or gradient parameter dependence, not
  the baseline;
- cascade σ² off while µ agrees → the template fluctuation: the reference fit used the `_no_fluct`
  variant, so its template σ² term is zero.

- [ ] **Step 4: Compare the fit itself**

Run our fit on real data with the same samples and priors, then compare against NNMFit's recorded
result:

```bash
python3 -c "
import json;c=json.load(open('configs/config_icecube_combined.json'));c['IceCube']['UseData']=True;json.dump(c,open('/tmp/ic_combined_data.json','w'),indent=2)"
./build/programs/LLHFit/LLHFit -c /tmp/ic_combined_data.json --silent
python3 - <<'EOF'
import json
ours = json.load(open('Output.json'))
ref  = json.load(open('/tmp/nnmfit_fit_reference.json'))
print(f"{'parameter':16s} {'ours':>14s} {'NNMFit':>14s} {'abs diff':>12s}")
for name, theirs in sorted(ref['fitted'].items()):
    mine = ours['parameters'][name]['value']
    print(f'{name:16s} {mine:14.6f} {theirs:14.6f} {abs(mine - theirs):12.2e}')
print('our chi2 (baseline-subtracted)', ours['chi2'], 'NNMFit llh_value', ref['llh_value'])
EOF
```
Expected: the meaningful physics parameters (`AstroNorm`, `SpectralIndex`, `ConvNorm`, `CRGrad`,
`MuonNorm`, `MuonGunNorm`, `VetoThreshold`, det-sys) land at the same minimum. Known, expected
differences to state rather than chase:
- `PromptNorm`: NNMFit's range clips it at 0.0; this framework has no bounds, so ours may go
  slightly negative. Report the value; it is not a defect of this work.
- NNMFit seeds randomly (`ref['seeds']`), so trajectories differ — only the minimum is comparable.
- The absolute likelihood values are **not** comparable: NNMFit's `llh_value` follows its own
  convention (Task 1 Step 2) and our `chi2` is baseline-subtracted inside `ICLikelihood`. Compare
  *differences*: our −2lnL at NNMFit's fitted point (Step 2's `fitted` run) minus our −2lnL at our own
  minimum must be ≥ 0 and small; a large gap means the two sides sit at different minima.

- [ ] **Step 5: Record the result and commit**

Append both diff tables and the fit comparison to `tools/nnmfit_oracle/README.md` — the
per-component max deviations plus the parameter table are the Phase 2 acceptance record. Then:

```bash
git add tools/nnmfit_oracle
git commit -m "test(icecube): diff per-component predictions and the fit against NNMFit"
```

---

## Self-review notes (traceability to the spec)

- **NNMFit oracle (spec scope decision 2, "Reference fit", "Testing")** → Task 1 reads the recorded fit and produces the per-component dumps at two parameter points; Task 13 diffs both plus the fit itself; Task 11 Step 4 emits the per-bin component arrays the diff needs.
- **`Fit.cpp` parameter-0 seeding** (index 0 forced to 1.0, which would silently evaluate `AstroNorm` at 1.0 instead of 1.5 and invalidate the fixed-parameter comparison) → fixed in Task 10 Step 8, Double Chooz re-checked in Step 9.
- **Tracks Corsika template** → exported in Task 7 (`template_2d`, single-dataset layout) and enabled on the tracks sample in Task 11, because the reference fit floats `MuonNorm` (fitted 1.827); it is no longer a follow-up.
- **`_no_fluct` MuonGun template** → Task 7 exports both variants; the parity configs use the no-fluctuation one the reference fit used, and `TemplateFlux` treats a missing fluctuation as zeros.
- **Non-uniform axes (spec §1)** → Task 2, with the exact 5up edges and the mixed-axis cascade grid tested.
- **Component vocabulary (spec §2)** → Task 4; consumed in Tasks 6, 8, 9.
- **Veto reweight (spec §3)** → Tasks 5 (columns) + 6 (CPU, MSL, CUDA, settings), tested against the hand-evaluated NNMFit formula and the veto-off invariant.
- **`TemplateFlux` (spec §4)** → Task 8, µ and σ² with livetime folding, bin-count mismatch fatal.
- **`DetectorSystematics` (spec §5)** → Task 9, µ + diagonal σ² + cross-gradient covariance, per sample.
- **`SampleLikelihood` assembly order (spec §6)** → Tasks 6, 8, 9 each extend it; the clip stays last and the histogram-level σ² additions come after the per-event sum.
- **Parameter layout (spec §7)** → Task 3, with a layout test and every config updated in the same commit.
- **Config layout (spec §8)** → Tasks 4 (parsing) + 11 (the anchor configs).
- **Priors and the `StepWidth` overload (spec §9)** → Task 10: `PriorValue`/`PriorWidth` with backwards-compatible defaults, all configs migrated to NNMFit's values, baseline re-recorded under two equivalence checks, Double Chooz verified unaffected.
- **Export pipeline (spec §10)** → Task 7, including the exact commands and the two template-sum anchors.
- **Golden gate** → re-run in Tasks 2–9 against the frozen baseline, deliberately re-recorded in Task 10, and re-checked in Tasks 11–12.
- **Risk 1 (template orientation)** and **Risk 2 (flattening order)** → Task 7's exporter records the edges and asserts the size; Task 8's loader makes a bin-count mismatch fatal; Task 13's per-bin diff is what actually catches an orientation error.
- **Risk 3 (single-bin sample)** → Task 2's `cscd_muon` grid test, Task 11's Metal parity step.
- **Risk 4 (covariance terms)** → isolated in `DetectorSystematics::check_and_recalculate`, testable by zeroing `m_Covariance`.
- **Risk 5 (data masks)** → Task 12 Step 1 measures before trusting.
- **Risk 6 (NNMFit environment)** → resolved before the plan started: the venv exists and a full fit has run through it.
- **Risk 8 (`llh_value` convention)** → Task 1 Step 2 pins it down; Task 13 Step 4 compares differences only.
- **Risk 7 (shared-core prior change)** → Task 10 Steps 1, 5 and 8.
- **Oscillations (spec §10)** → Task 7b. Added after reading `flux_hooks.py`: the combined config applies `OscillationsHook` to conv and prompt, so parity is impossible without it, and it turns out to be a static per-event factor (νμ disappearance) applied at load — cheap to support exactly via a parquet sidecar.
- **Deliberately not tasks:** RA/galactic (Phase 3), `cscd_hybrid`, pseudo-experiments, and parameter bounds (the reason NNMFit's `PromptNorm` clips at 0.0 and ours may not).
