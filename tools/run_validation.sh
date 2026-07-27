#!/usr/bin/env bash
#
# Validates a build against everything the experiment-module refactor promised:
#
#   1. the Double Chooz Asimov fit still reproduces the recorded baseline bit for bit,
#   2. the LinearRegression example still recovers its truth values,
#   3. a config without a usable "Experiment" key still fails with a helpful message.
#
# The Double Chooz part needs Output.baseline.json, which is deliberately untracked. Generate it
# once from a known-good build with:
#
#   ./build/programs/LLHFit/LLHFit -c "$DC_CONFIG" --silent && cp Output.json Output.baseline.json
#
# Without it (or without a Double Chooz config) that part is skipped, so a clean clone can still
# validate the parts that do not depend on the experiment inputs.
#
# The IceCube part needs Output.ic_baseline.json, which is deliberately untracked, generated the
# same way from configs/config_icecube_tracks_cpu.json. Without it that part is skipped.
#
# The IceCube cascade part is the same idea for configs/config_icecube_cascades.json and
# Output.ic_cascades_baseline.json (also untracked; needs the local NNMFit input files under
# ~/Downloads/nnmfit_files, so it is expected to skip on a machine without them).
#
# Usage: tools/run_validation.sh [--no-build]
#   DC_CONFIG       path to the Double Chooz config       (default ../PhyLiNO/config.json)
#   BASELINE        path to the recorded DC output        (default Output.baseline.json)
#   IC_CONFIG       path to the IceCube config             (default configs/config_icecube_tracks_cpu.json)
#   IC_BASELINE     path to the recorded IC output         (default Output.ic_baseline.json)
#   IC_CASCADE_CONFIG   path to the IceCube cascade config (default configs/config_icecube_cascades.json)
#   IC_CASCADE_BASELINE path to the recorded cascade output (default Output.ic_cascades_baseline.json)

set -u -o pipefail

cd "$(dirname "$0")/.."

DC_CONFIG=${DC_CONFIG:-../PhyLiNO/config.json}
BASELINE=${BASELINE:-Output.baseline.json}
LINREG_CONFIG=configs/config_linreg.json
IC_CONFIG=${IC_CONFIG:-configs/config_icecube_tracks_cpu.json}
IC_BASELINE=${IC_BASELINE:-Output.ic_baseline.json}
IC_CASCADE_CONFIG=${IC_CASCADE_CONFIG:-configs/config_icecube_cascades.json}
IC_CASCADE_BASELINE=${IC_CASCADE_BASELINE:-Output.ic_cascades_baseline.json}
LLHFIT=build/programs/LLHFit/LLHFit

failures=0
skips=0

pass() { printf '  ok      %s\n' "$1"; }
skip() { printf '  skip    %s\n' "$1"; skips=$((skips + 1)); }
fail() { printf '  FAILED  %s\n' "$1"; failures=$((failures + 1)); }

if [ "${1:-}" != "--no-build" ]; then
  echo "Building"
  if [ ! -d build ]; then
    echo "  no build directory - configure it first, see CLAUDE.md" >&2
    exit 1
  fi
  if ! cmake --build build -j8 > /tmp/phylino_build.log 2>&1; then
    tail -30 /tmp/phylino_build.log >&2
    echo "  build failed, see /tmp/phylino_build.log" >&2
    exit 1
  fi
fi

echo "Double Chooz reproduces the baseline"
if [ ! -f "$DC_CONFIG" ]; then
  skip "no Double Chooz config at $DC_CONFIG (set DC_CONFIG)"
elif [ ! -f "$BASELINE" ]; then
  skip "no baseline at $BASELINE (see the header of this script)"
elif ! "$LLHFIT" -c "$DC_CONFIG" --silent > /tmp/phylino_dc.log 2>&1; then
  tail -20 /tmp/phylino_dc.log >&2
  fail "the fit did not run"
elif python3 tools/compare_output.py "$BASELINE" Output.json > /tmp/phylino_cmp.log 2>&1; then
  pass "output identical to $BASELINE"
else
  head -20 /tmp/phylino_cmp.log >&2
  fail "output differs from $BASELINE"
fi

echo "LinearRegression recovers its truth values"
if ! "$LLHFIT" -c "$LINREG_CONFIG" --silent > /tmp/phylino_linreg.log 2>&1; then
  tail -20 /tmp/phylino_linreg.log >&2
  fail "the fit did not run"
elif python3 - "$LINREG_CONFIG" <<'PY'
import json, sys

with open(sys.argv[1]) as f:
    truth = json.load(f)["LinearRegression"]
with open("Output.json") as f:
    result = json.load(f)

problems = []
if not result["converged"]:
    problems.append("the fit did not converge")
for key, expected in (("a", truth["TruthA"]), ("b", truth["TruthB"])):
    if abs(result[key] - expected) > 1e-3:
        problems.append(f"{key} = {result[key]!r}, expected {expected!r}")
if result["chi2"] > 1e-6:
    problems.append(f"chi2 = {result['chi2']!r}, expected ~0 on Asimov data")
if len(result["x"]) != truth["NPoints"] or len(result["y"]) != truth["NPoints"]:
    problems.append(f"wrote {len(result['x'])}/{len(result['y'])} points, expected {truth['NPoints']}")

for problem in problems:
    print(problem, file=sys.stderr)
sys.exit(1 if problems else 0)
PY
then
  pass "a, b and chi2 match the configured truth"
else
  fail "the fit did not recover the truth values"
fi

echo "IceCube reproduces the baseline"
if [ ! -f "$IC_CONFIG" ]; then
  skip "no IceCube config at $IC_CONFIG (set IC_CONFIG)"
elif [ ! -f "$IC_BASELINE" ]; then
  skip "no baseline at $IC_BASELINE (see the header of this script)"
elif ! "$LLHFIT" -c "$IC_CONFIG" --silent > /tmp/phylino_ic.log 2>&1; then
  tail -20 /tmp/phylino_ic.log >&2
  fail "the fit did not run"
elif python3 tools/compare_output.py "$IC_BASELINE" Output.json > /tmp/phylino_ic_cmp.log 2>&1; then
  pass "output identical to $IC_BASELINE"
else
  head -20 /tmp/phylino_ic_cmp.log >&2
  fail "output differs from $IC_BASELINE"
fi

echo "IceCube cascades reproduce the baseline"
if [ ! -f "$IC_CASCADE_CONFIG" ]; then
  skip "no IceCube cascade config at $IC_CASCADE_CONFIG (set IC_CASCADE_CONFIG)"
elif [ ! -f "$IC_CASCADE_BASELINE" ]; then
  skip "no baseline at $IC_CASCADE_BASELINE (see the header of this script)"
elif ! "$LLHFIT" -c "$IC_CASCADE_CONFIG" --silent > /tmp/phylino_ic_cascade.log 2>&1; then
  tail -20 /tmp/phylino_ic_cascade.log >&2
  fail "the fit did not run"
elif python3 tools/compare_output.py "$IC_CASCADE_BASELINE" Output.json > /tmp/phylino_ic_cascade_cmp.log 2>&1; then
  pass "output identical to $IC_CASCADE_BASELINE"
else
  head -20 /tmp/phylino_ic_cascade_cmp.log >&2
  fail "output differs from $IC_CASCADE_BASELINE"
fi

echo "An unusable Experiment key is reported clearly"
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

python3 - "$LINREG_CONFIG" "$scratch" <<'PY'
import json, sys

config_path, scratch = sys.argv[1], sys.argv[2]
with open(config_path) as f:
    config = json.load(f)

missing = {k: v for k, v in config.items() if k != "Experiment"}
with open(scratch + "/missing.json", "w") as f:
    json.dump(missing, f)

with open(scratch + "/unknown.json", "w") as f:
    json.dump({**config, "Experiment": "NoSuchExperiment"}, f)
PY

for case in missing unknown; do
  output=$("$LLHFIT" -c "$scratch/$case.json" --silent 2>&1)
  status=$?
  if [ "$status" -eq 0 ]; then
    fail "a config with a $case Experiment key exited 0"
  elif ! printf '%s' "$output" | grep -q "Registered experiments:"; then
    printf '%s\n' "$output" >&2
    fail "the $case case did not list the registered experiments"
  else
    pass "the $case case fails with the registered experiments listed"
  fi
done

echo
if [ "$failures" -gt 0 ]; then
  echo "$failures check(s) failed"
  exit 1
fi
if [ "$skips" -gt 0 ]; then
  echo "all checks passed, $skips skipped"
  exit 0
fi
echo "all checks passed"
