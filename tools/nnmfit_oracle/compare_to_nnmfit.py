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

# PhyLiNO sample name -> number of RA bins in the analysis binning (1 = no RA axis).
# With an RA axis the zenith axis is no longer the fast one, so the mirror applies to
# the MIDDLE axis of the (energy, zenith, RA) row-major array; flipping the last axis
# would mirror RA instead and leave zenith wrong. cscd_muon is 1: the analysis keeps
# that sample two-dimensional. This map must be edited alongside the config -- if a
# sample's RA binning changes, this map changes with it.
RA_BINS = {"tracks": 180, "cscd_cascade": 18, "cscd_muon": 1}


def reverse_zenith_axis(values, zenith_bins, ra_bins=1):
    """Relabel a flat (energy, zenith, RA) row-major array into our bin order."""
    if zenith_bins <= 1:
        return values
    return np.flip(values.reshape(-1, zenith_bins, ra_bins), axis=1).reshape(-1)

# PhyLiNO componentBins key -> the signed NNMFit dumps it is built from. PhyLiNO's
# AtmosphericFlux computes conventional and prompt in one pass, so it is compared
# against the sum of NNMFit's two components (veto variants for the cascades).
# Histogram components cannot be dumped on their own (see dump_histograms.sh), so the
# galactic template is reconstructed as total - no_galactictemplate_cringefits.
COMPONENTS = {
    "astro": [(1, "astro_brokenPL")],
    "atmospheric": [(1, "conventional"), (1, "prompt")],
    "atmospheric_veto": [(1, "conventional_veto"), (1, "prompt_veto")],
    "template": [(1, "muontemplate"), (1, "muon")],
    "galactic": [(1, "total"), (-1, "no_galactictemplate_cringefits")],
}


def read_mu(dump_dir, det_conf, dump):
    path = Path(dump_dir) / f"{det_conf}_{dump}.pickle"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        d = pickle.load(f)
    h = d["histograms"]
    mu = h[det_conf] if isinstance(h, dict) else h
    return np.asarray(mu, dtype=float).reshape(-1)


def nnmfit_mu(dump_dir, det_conf, terms):
    """Sum the signed dumps of `terms`, or None if the combination is unavailable.

    A missing additive dump just drops out of the sum (a sample that does not use
    that component), but a missing subtractive one would silently turn a difference
    into its minuend, so it makes the whole combination unavailable.
    """
    total = None
    for sign, dump in terms:
        mu = read_mu(dump_dir, det_conf, dump)
        if mu is None:
            if sign < 0:
                return None
            continue
        term = mu if sign > 0 else -mu
        total = term if total is None else total + term
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

        for key, terms in COMPONENTS.items():
            ours = sample.get("componentBins", {}).get(key)
            if not ours:
                continue
            theirs = nnmfit_mu(args.dump_dir, det_conf, terms)
            if theirs is None:
                print(f"skip {sample['name']}/{key}: no NNMFit dump")
                continue
            theirs = reverse_zenith_axis(
                theirs, ZENITH_BINS[sample["name"]], RA_BINS.get(sample["name"], 1)
            )

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
