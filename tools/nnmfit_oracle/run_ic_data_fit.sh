#!/usr/bin/env bash
# Fit the combined IceCube config against REAL detector data and compare the
# minimum against NNMFit's already-recorded fit (Task 13 Step 4 of
# docs/superpowers/plans/2026-07-27-icecube-cascades-phase2.md).
#
# This is the only step in the Phase 2 acceptance gate that touches real
# data. It is deliberately a standalone script, not something run
# automatically by an agent: run it yourself, from a shell, when you want
# that comparison.
#
# Usage: tools/nnmfit_oracle/run_ic_data_fit.sh [OUT_DIR]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${1:-/tmp/ic_data_fit}"
mkdir -p "$OUT_DIR"

REF_JSON=/tmp/nnmfit_fit_reference.json
if [ ! -f "$REF_JSON" ]; then
  echo "missing $REF_JSON -- run first:" >&2
  echo "  tools/nnmfit_oracle/read_fit_result.py /Users/soldin/Projects/IceCube/NNMFit/NNMFit/Output.pickle --json $REF_JSON" >&2
  exit 1
fi

DATA_CONFIG="$OUT_DIR/config_icecube_combined_data.json"
python3 -c "
import json
c = json.load(open('$ROOT/configs/config_icecube_combined.json'))
c['IceCube']['UseData'] = True
c['IceCube']['Likelihood'] = 'SAY'
c['IceCube']['Backend'] = 'cpu'
json.dump(c, open('$DATA_CONFIG', 'w'), indent=2)
"

echo "Fitting $DATA_CONFIG against real data..."
(cd "$ROOT" && ./build/programs/LLHFit/LLHFit -c "$DATA_CONFIG" --silent)
cp "$ROOT/Output.json" "$OUT_DIR/Output.json"

python3 - "$OUT_DIR/Output.json" "$REF_JSON" <<'EOF'
import json
import sys

ours = json.load(open(sys.argv[1]))
ref = json.load(open(sys.argv[2]))

print(f"{'parameter':16s} {'ours':>14s} {'NNMFit':>14s} {'abs diff':>12s}")
for name, theirs in sorted(ref["fitted"].items()):
    mine = ours["parameters"][name]["value"]
    print(f"{name:16s} {mine:14.6f} {theirs:14.6f} {abs(mine - theirs):12.2e}")
print("our chi2 (baseline-subtracted)", ours["chi2"], "NNMFit llh_value", ref["llh_value"])
print()
print("Known, expected differences -- not defects:")
print("  PromptNorm: NNMFit clips it at its range boundary (0.0); this framework has")
print("  no bounds plumbing, so ours may land slightly negative instead.")
print("  Absolute likelihood values are NOT comparable (different conventions, see")
print("  tools/nnmfit_oracle/README.md); only compare differences between points.")
EOF

echo
echo "wrote $OUT_DIR/Output.json"
