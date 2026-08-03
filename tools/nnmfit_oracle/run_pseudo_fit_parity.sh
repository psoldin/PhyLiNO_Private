#!/usr/bin/env bash
# Fit BOTH frameworks to the same pseudo-experiment and diff the results.
#
# This is the converged-fit parity check without touching detector data: a
# Poisson draw on the Asimov prediction is integer-valued by construction, so
# it is a valid dataset, and feeding the SAME draw to both codes makes their
# objectives identical bin for bin. What it gates that no fixed-point
# comparison can: the minimiser, the derivatives, and the fitted parameter
# values with their uncertainties.
#
# The draw is generated once (make_pseudo_data.py) and handed to
#   PhyLiNO : per-sample "DataCounts" text files
#   NNMFit  : analysis_type "custom_data" + "custom_dataset" pickle, the branch
#             that uses the array verbatim without re-fluctuating it
#
# Usage: tools/nnmfit_oracle/run_pseudo_fit_parity.sh [OUT_DIR] [SEED]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/tools/nnmfit_oracle"
OUT_DIR="${1:-/tmp/ic_pseudo_parity}"
SEED="${2:-20260803}"
PY=/Users/soldin/Projects/IceCube/NNMFit/.venv/bin/python
RUN_FIT=/Users/soldin/Projects/IceCube/NNMFit/NNMFit/scripts/run_fit.py

PHYLINO_CONFIG="${PHYLINO_CONFIG:-$ROOT/configs/config_icecube_combined.json}"
NNMFIT_CONFIG="${NNMFIT_CONFIG:-/Users/soldin/Downloads/Fit_Configuration_Combined_macOS.yaml}"
LIKELIHOOD="${LIKELIHOOD:-SAY}"

mkdir -p "$OUT_DIR"

echo "=== Stage 1: Asimov prediction at the truth point (every parameter fixed) ==="
"$HERE/make_probe_config.py" "$PHYLINO_CONFIG" "$OUT_DIR/truth.json"
(cd "$OUT_DIR" && "$ROOT/build/programs/LLHFit/LLHFit" -c "$OUT_DIR/truth.json" --fitOnly --silent)
mv "$OUT_DIR/Output.json" "$OUT_DIR/truth_output.json"

echo
echo "=== Stage 2: one Poisson draw, written for both frameworks ==="
# Generated with NNMFit's interpreter on purpose: a pickle written by a newer
# numpy carries numpy._core references that NNMFit's numpy cannot unpickle.
"$PY" "$HERE/make_pseudo_data.py" "$OUT_DIR/truth_output.json" "$OUT_DIR/pseudo" --seed "$SEED"

echo
echo "=== Stage 3: PhyLiNO fit ==="
"$PY" - "$PHYLINO_CONFIG" "$OUT_DIR" "$LIKELIHOOD" <<'EOF'
import json, sys
config_path, out_dir, likelihood = sys.argv[1:4]
cfg = json.load(open(config_path))
cfg["IceCube"]["UseData"] = True
cfg["IceCube"]["Likelihood"] = likelihood
cfg["IceCube"]["Backend"] = "cpu"
for name, sample in cfg["IceCube"]["Samples"].items():
    sample["DataCounts"] = f"{out_dir}/pseudo/pseudo_{name}.txt"
json.dump(cfg, open(f"{out_dir}/pseudo_fit_phylino.json", "w"), indent=2)
EOF
(cd "$OUT_DIR" && "$ROOT/build/programs/LLHFit/LLHFit" -c "$OUT_DIR/pseudo_fit_phylino.json" --fitOnly --silent)
mv "$OUT_DIR/Output.json" "$OUT_DIR/phylino_fit.json"

echo
echo "=== Stage 4: NNMFit fit on the same counts ==="
"$PY" - "$NNMFIT_CONFIG" "$OUT_DIR" <<'EOF'
import sys, yaml
config_path, out_dir = sys.argv[1:3]
cfg = yaml.safe_load(open(config_path))
cfg["analysis"]["analysis_type"] = "custom_data"
cfg["analysis"]["custom_dataset"] = f"{out_dir}/pseudo/pseudo_data_nnmfit.pickle"
yaml.safe_dump(cfg, open(f"{out_dir}/pseudo_fit_nnmfit.yaml", "w"))
EOF
# --use_default_param_seeds: NNMFit randomizes its seeds by default, and only a
# shared start point makes the two trajectories comparable.
"$PY" "$RUN_FIT" --configs "$OUT_DIR/pseudo_fit_nnmfit.yaml" \
    -o "$OUT_DIR/nnmfit_fit.pickle" --use_default_param_seeds --skip_save_config
"$HERE/read_fit_result.py" "$OUT_DIR/nnmfit_fit.pickle" --json "$OUT_DIR/nnmfit_fit.json" >/dev/null

echo
echo "=== Stage 5: diff ==="
python3 - "$OUT_DIR/phylino_fit.json" "$OUT_DIR/nnmfit_fit.json" <<'EOF'
import json, sys
ours = json.load(open(sys.argv[1]))
ref = json.load(open(sys.argv[2]))

print(f"{'parameter':16s} {'PhyLiNO':>15s} {'NNMFit':>15s} {'diff':>11s} {'diff/sigma':>11s}")
worst = worst_pull = 0.0
for name, theirs in sorted(ref["fitted"].items()):
    entry = ours["parameters"].get(name)
    if entry is None:
        continue
    mine, sigma = entry["value"], entry["error"]
    diff = abs(mine - theirs)
    pull = diff / sigma if sigma > 0 else float("nan")
    worst = max(worst, diff)
    if pull == pull:
        worst_pull = max(worst_pull, pull)
    print(f"{name:16s} {mine:15.8f} {theirs:15.8f} {diff:11.2e} {pull:11.2e}")

expected = 2.0 * ref["llh_value"]
print()
print(f"PhyLiNO minimum (-2lnL + chi2) : {ours['LLH']!r}   EDM {ours['EDM']!r}")
print(f"NNMFit  minimum (-lnL + chi2/2): {ref['llh_value']!r}")
print(f"2 * NNMFit                     : {expected!r}")
print(f"difference in -2lnL            : {ours['LLH'] - expected:+.6e}")
print(f"largest parameter difference   : {worst:.2e}  ({worst_pull:.2e} sigma)")
print()
print("How to read it: the two minimisers (Migrad vs LBFGSB) stop at different")
print("points within their own tolerances, so judge the parameter differences in")
print("units of the fitted uncertainty, not against 0. A minimum difference that")
print("is negative means PhyLiNO found the better point, positive means NNMFit did.")
EOF

echo
echo "wrote results to $OUT_DIR"
