#!/usr/bin/env bash
# Dump NNMFit's model histograms (mu, ssq) for the combined macOS config: once with
# every component a sample uses, then once per component with the others excluded.
# The per-component dumps are the oracle PhyLiNO's per-bin prediction is diffed
# against (tools/nnmfit_oracle/compare_to_nnmfit.py).
#
# Usage: tools/nnmfit_oracle/dump_histograms.sh [BASE_CONFIG] [OUT_DIR] [SAMPLE ...]
#   BASE_CONFIG  NNMFit analysis config (default: the combined macOS config)
#   OUT_DIR      output directory       (default: /tmp/nnmfit_dumps)
#   SAMPLE ...   restrict to these detector configs (default: all three)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/soldin/Projects/IceCube/NNMFit/.venv/bin/python
MAKE_HIST=/Users/soldin/Projects/IceCube/NNMFit/NNMFit/scripts/make_histogram.py

BASE="${1:-/Users/soldin/Downloads/Fit_Configuration_Combined_macOS.yaml}"
OUT="${2:-/tmp/nnmfit_dumps}"
shift || true
shift || true

# sample -> the components it uses (the complement of its excluded_components)
TRACKS="IC86_pass2_SnowStorm_v2_tracks"
CASCADE="IC86_pass2_SnowStorm_v2_cscd_cascade"
MUON="IC86_pass2_SnowStorm_v2_cscd_muon"
components_of() {
  case "$1" in
    "$TRACKS")  echo "astro conventional prompt muontemplate" ;;
    "$CASCADE") echo "astro conventional_veto prompt_veto muon" ;;
    "$MUON")    echo "astro conventional_veto prompt_veto muon" ;;
    *) echo "unknown sample '$1'" >&2; return 1 ;;
  esac
}

# The muon templates are histogram components (TemplateFlux, IS_HIST_COMP). NNMFit
# cannot build a sample whose ONLY component is a histogram component: its ssq path
# (histogram_builder.__make_total_fluctuation) leaves `weights` as the scalar 0.0
# and aesara's bincount then fails with "'float' object has no attribute 'dtype'".
# So instead of dumping the template alone, we dump everything EXCEPT the template
# ("<sample>_no_template.pickle"); the template contribution is
#   total - no_template
# and, as a cross-check, no_template must equal the sum of the per-event dumps.
template_of() {
  case "$1" in
    "$TRACKS") echo "muontemplate" ;;
    *)         echo "muon" ;;
  esac
}

ALL="conventional conventional_veto prompt prompt_veto muon muontemplate astro"
SAMPLES=("$@")
if [ ${#SAMPLES[@]} -eq 0 ]; then
  SAMPLES=("$TRACKS" "$CASCADE" "$MUON")
fi

mkdir -p "$OUT"

# Comma-joined list of every component in $ALL that is not in $2 (the keep list).
excluded_except() {
  local keep="$1" excluded="" component
  for component in $ALL; do
    case " $keep " in
      *" $component "*) ;;
      *) excluded="${excluded:+$excluded, }$component" ;;
    esac
  done
  echo "$excluded"
}

for sample in "${SAMPLES[@]}"; do
  used="$(components_of "$sample")"

  # "total": every component this sample actually uses
  "$PY" "$HERE/nnmfit_set_excluded.py" "$BASE" "$OUT/cfg_${sample}_total.yaml" \
        "$sample" "$(excluded_except "$used")"
  "$PY" "$MAKE_HIST" --configs "$OUT/cfg_${sample}_total.yaml" \
        -o "$OUT/${sample}_total.pickle"

  template="$(template_of "$sample")"

  # one dump per per-event component, everything else excluded
  for keep in $used; do
    [ "$keep" = "$template" ] && continue  # see template_of(): NNMFit cannot do this alone
    "$PY" "$HERE/nnmfit_set_excluded.py" "$BASE" "$OUT/cfg_${sample}_${keep}.yaml" \
          "$sample" "$(excluded_except "$keep")"
    "$PY" "$MAKE_HIST" --configs "$OUT/cfg_${sample}_${keep}.yaml" \
          -o "$OUT/${sample}_${keep}.pickle"
  done

  # everything except the template, so the template = total - no_template
  keep_no_template=""
  for component in $used; do
    [ "$component" = "$template" ] || keep_no_template="${keep_no_template:+$keep_no_template }$component"
  done
  "$PY" "$HERE/nnmfit_set_excluded.py" "$BASE" "$OUT/cfg_${sample}_no_template.yaml" \
        "$sample" "$(excluded_except "$keep_no_template")"
  "$PY" "$MAKE_HIST" --configs "$OUT/cfg_${sample}_no_template.yaml" \
        -o "$OUT/${sample}_no_template.pickle"
done

echo "dumps in $OUT"
