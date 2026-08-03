#!/usr/bin/env bash
# Full input->output parity between PhyLiNO and NNMFit: same config, same data,
# same likelihood, same seeds -- compare the converged minimum and every fitted
# parameter.
#
# Three stages, each usable on its own:
#   1. likelihood VALUE at a fixed point (no minimisation on either side)
#   2. NNMFit fit from the config default seeds (no randomisation)
#   3. PhyLiNO fit from the same seeds, then a parameter-by-parameter diff
#
# Stages 1 and 3 fit REAL DETECTOR DATA. This script is therefore deliberately
# manual -- run it yourself from a shell; nothing here is invoked automatically.
#
# Usage: tools/nnmfit_oracle/run_fit_parity.sh [OUT_DIR] [NNMFIT_CONFIG]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/tools/nnmfit_oracle"
OUT_DIR="${1:-/tmp/ic_fit_parity}"
NNMFIT_CONFIG="${2:-/Users/soldin/Downloads/Fit_Configuration_Combined_macOS.yaml}"
PY=/Users/soldin/Projects/IceCube/NNMFit/.venv/bin/python
RUN_FIT=/Users/soldin/Projects/IceCube/NNMFit/NNMFit/scripts/run_fit.py

# Must match the NNMFit config's analysis.llh (SAYLLH -> SAY).
LIKELIHOOD="${LIKELIHOOD:-SAY}"
PHYLINO_CONFIG="${PHYLINO_CONFIG:-$ROOT/configs/config_icecube_combined.json}"

mkdir -p "$OUT_DIR"

echo "=== Stage 1: likelihood value at the config default point (real data) ==="
"$HERE/compare_llh_value.py" "$PHYLINO_CONFIG" "$NNMFIT_CONFIG" \
    --workdir "$OUT_DIR/value_defaults" --likelihood "$LIKELIHOOD" --use-data

echo
echo "=== Stage 2: NNMFit fit from the config default seeds ==="
# --use_default_param_seeds turns off NNMFit's seed randomisation, so both codes
# start from the same point and the comparison covers the trajectory's endpoint,
# not just "some minimum".
"$PY" "$RUN_FIT" --configs "$NNMFIT_CONFIG" \
    -o "$OUT_DIR/nnmfit_fit.pickle" --use_default_param_seeds
"$HERE/read_fit_result.py" "$OUT_DIR/nnmfit_fit.pickle" --json "$OUT_DIR/nnmfit_fit.json"

echo
echo "=== Stage 3: PhyLiNO fit on the same data, then diff ==="
DATA_CONFIG="$OUT_DIR/config_data.json"
python3 -c "
import json
c = json.load(open('$PHYLINO_CONFIG'))
c['IceCube']['UseData'] = True
c['IceCube']['Likelihood'] = '$LIKELIHOOD'
c['IceCube']['Backend'] = 'cpu'
json.dump(c, open('$DATA_CONFIG', 'w'), indent=2)
"
# --fitOnly: LLHFit's default entry point is the 2D scan, which writes a surface
# of Output_i_j files and no single result to compare against.
(cd "$ROOT" && ./build/programs/LLHFit/LLHFit -c "$DATA_CONFIG" --fitOnly --silent)
cp "$ROOT/Output.json" "$OUT_DIR/phylino_fit.json"

python3 - "$OUT_DIR/phylino_fit.json" "$OUT_DIR/nnmfit_fit.json" <<'EOF'
import json
import sys

ours = json.load(open(sys.argv[1]))
ref = json.load(open(sys.argv[2]))

print()
print(f"{'parameter':16s} {'PhyLiNO':>16s} {'NNMFit':>16s} {'abs diff':>12s}")
worst = 0.0
for name, theirs in sorted(ref["fitted"].items()):
    mine = ours["parameters"][name]["value"]
    worst = max(worst, abs(mine - theirs))
    print(f"{name:16s} {mine:16.8f} {theirs:16.8f} {abs(mine - theirs):12.2e}")

expected = 2.0 * ref["llh_value"]
print()
print(f"PhyLiNO minimum (-2lnL + chi2) : {ours['LLH']!r}")
print(f"NNMFit  minimum (-lnL + chi2/2): {ref['llh_value']!r}")
print(f"2 * NNMFit                     : {expected!r}")
print(f"difference                     : {abs(ours['LLH'] - expected):.6e}")
print(f"largest parameter difference   : {worst:.2e}")
print()
print("Reading the numbers:")
print("  The minima agree exactly only if both minimisers landed in the same")
print("  place; different minimisers (Migrad vs LBFGSB) stop at different")
print("  points within their own tolerance, so judge this against the fit's")
print("  own EDM, not against 0. A minimum difference far larger than the")
print("  parameter differences imply means the likelihoods themselves differ.")
print("  PromptNorm: NNMFit clips it at its range boundary (0.0); PhyLiNO has")
print("  no bounds plumbing, so it may land slightly negative instead.")
EOF

echo
echo "wrote results to $OUT_DIR"
