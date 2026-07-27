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

# PhyLiNO sample name -> number of zenith bins (the fast-varying axis in the
# row-major flat index both sides use). NNMFit's "cos"/"cscd-cos_5up" bin edges
# are built as np.sort(arccos(ascending_cos_values)): since arccos is
# monotonically decreasing, this mirrors the zenith axis relative to
# io::ic::Binning's own convention (verified numerically for both the uniform
# tracks axis and the explicit cscd_cascade edges: nnmfit_idx == zenith_bins-1
# - our_idx for every event). Unlike the exported templates/gradients (fixed at
# export time by tools/export_nnmfit_inputs.py's reverse_zenith_axis), the
# astro/atmospheric arrays compared here are each computed independently by its
# own framework, so the correction is applied here instead, to NNMFit's side.
ZENITH_BINS = {"tracks": 33, "cscd_cascade": 7, "cscd_muon": 1}


def reverse_zenith_axis(values, zenith_bins):
    """Relabel a flat (energy-outer, NNMFit-zenith-inner) array into our order."""
    if zenith_bins <= 1:
        return values
    return np.flip(values.reshape(-1, zenith_bins), axis=1).reshape(-1)

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
            d = pickle.load(f)
        h = d["histograms"]
        mu = h[det_conf] if isinstance(h, dict) else h
        mu = np.asarray(mu, dtype=float).reshape(-1)
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
            theirs = reverse_zenith_axis(theirs, ZENITH_BINS[sample["name"]])

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
