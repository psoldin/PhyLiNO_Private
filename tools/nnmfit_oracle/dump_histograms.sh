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
    "$TRACKS")  echo "astro_brokenPL conventional prompt muontemplate galactictemplate_cringefits" ;;
    "$CASCADE") echo "astro_brokenPL conventional_veto prompt_veto muon galactictemplate_cringefits" ;;
    "$MUON")    echo "astro_brokenPL conventional_veto prompt_veto muon" ;;
    *) echo "unknown sample '$1'" >&2; return 1 ;;
  esac
}

# Histogram components (TemplateFlux / GalacticTemplate, IS_HIST_COMP) cannot be
# dumped alone: the ssq path (histogram_builder.__make_total_fluctuation) leaves
# `weights` as the scalar 0.0 and aesara's bincount then fails with "'float' object
# has no attribute 'dtype'". So we dump everything EXCEPT the histogram components
# ("<sample>_no_hist.pickle"); their combined contribution is total - no_hist, and,
# as a cross-check, no_hist must equal the sum of the per-event dumps.
hist_components_of() {
  case "$1" in
    "$TRACKS") echo "muontemplate galactictemplate_cringefits" ;;
    "$MUON")   echo "muon" ;;
    *)         echo "muon galactictemplate_cringefits" ;;
  esac
}

# NOTE: the astrophysical component is named astro_brokenPL in the generated 3D
# config (NNMFit AstroBPL). It MUST appear here: excluded_except() only excludes
# names in this list, so a stale "astro" would leave the real component active in
# every per-component dump, silently adding the astrophysical flux to each one.
ALL="conventional conventional_veto prompt prompt_veto muon muontemplate astro_brokenPL galactictemplate_cringefits"
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

  hist="$(hist_components_of "$sample")"

  # one dump per per-event component, everything else excluded
  for keep in $used; do
    case " $hist " in *" $keep "*) continue ;; esac
    "$PY" "$HERE/nnmfit_set_excluded.py" "$BASE" "$OUT/cfg_${sample}_${keep}.yaml" \
          "$sample" "$(excluded_except "$keep")"
    "$PY" "$MAKE_HIST" --configs "$OUT/cfg_${sample}_${keep}.yaml" \
          -o "$OUT/${sample}_${keep}.pickle"
  done

  # everything except the histogram components, so their sum = total - no_hist
  keep_no_hist=""
  for component in $used; do
    case " $hist " in
      *" $component "*) ;;
      *) keep_no_hist="${keep_no_hist:+$keep_no_hist }$component" ;;
    esac
  done
  "$PY" "$HERE/nnmfit_set_excluded.py" "$BASE" "$OUT/cfg_${sample}_no_hist.yaml" \
        "$sample" "$(excluded_except "$keep_no_hist")"
  "$PY" "$MAKE_HIST" --configs "$OUT/cfg_${sample}_no_hist.yaml" \
        -o "$OUT/${sample}_no_hist.pickle"

  # ...and, for the galactic template, everything except that one component, so that
  # it can be isolated on its own as total - no_galactictemplate_cringefits. Keeping
  # the other per-event components alongside it is what makes the ssq path work.
  # (no_hist alone would only give muon template + galactic lumped together, which is
  # not enough to gate the galactic prediction on its own.)
  for drop in $hist; do
    [ "$drop" = "galactictemplate_cringefits" ] || continue
    keep_but_one=""
    for component in $used; do
      [ "$component" = "$drop" ] || keep_but_one="${keep_but_one:+$keep_but_one }$component"
    done
    "$PY" "$HERE/nnmfit_set_excluded.py" "$BASE" "$OUT/cfg_${sample}_no_${drop}.yaml" \
          "$sample" "$(excluded_except "$keep_but_one")"
    "$PY" "$MAKE_HIST" --configs "$OUT/cfg_${sample}_no_${drop}.yaml" \
          -o "$OUT/${sample}_no_${drop}.pickle"
  done
done

echo "dumps in $OUT"
