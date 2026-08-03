#!/usr/bin/env python3
"""Draw one pseudo-experiment from PhyLiNO's Asimov prediction and write it in
both frameworks' input formats, so a fit can be compared on identical counts.

A Poisson draw on the expectation is already integer-valued, so this is the
"generate data, round to whole numbers, fit both" experiment -- with the one
requirement that both codes see the SAME draw. Each framework generating its
own pseudo-experiment would compare two different datasets.

  PhyLiNO : one text file per sample, one count per analysis bin, wired in with
            the sample's "DataCounts" config key (io::ic::SampleConfig).
  NNMFit  : one pickle {"histograms": {det_conf: array}}, wired in with
            analysis_type "custom_data" + "custom_dataset". That branch uses the
            array verbatim as data_hists -- it does NOT fluctuate it again
            (nnm_fitter.py:487-501), which is what makes the two identical.

The zenith axis is mirrored on the way out: NNMFit's bin edges are
np.sort(arccos(ascending_cos)), the opposite order from io::ic::Binning (see
compare_to_nnmfit.py's ZENITH_BINS note).

Usage:
  make_pseudo_data.py OUTPUT_JSON OUT_DIR --seed 12345
    OUTPUT_JSON  an LLHFit Output.json whose prediction is the truth to fluctuate
                 (produce it with every parameter Fixed at the truth point)
"""
import argparse
import json
import os
import pickle

import numpy as np

# PhyLiNO sample name -> (NNMFit detector config, zenith bins, RA bins)
SAMPLES = {
    "tracks": ("IC86_pass2_SnowStorm_v2_tracks", 33, 1),
    "cscd_cascade": ("IC86_pass2_SnowStorm_v2_cscd_cascade", 7, 1),
    "cscd_muon": ("IC86_pass2_SnowStorm_v2_cscd_muon", 1, 1),
}

# The 3D configs carry an RA axis on two of the samples; sizes are checked
# against the Output.json axes below, so a config change is caught here.
RA_3D = {"tracks": 180, "cscd_cascade": 18, "cscd_muon": 1}


def mirror_zenith(flat, energy_bins, zenith_bins, ra_bins):
    """PhyLiNO bin order -> NNMFit bin order (self-inverse)."""
    if zenith_bins == 1:
        return flat
    shaped = flat.reshape(energy_bins, zenith_bins, ra_bins)
    return shaped[:, ::-1, :].reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_json")
    parser.add_argument("out_dir")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.output_json) as f:
        result = json.load(f)

    rng = np.random.default_rng(args.seed)
    nnmfit_hists = {}

    for sample in result["samples"]:
        name = sample["name"]
        if name not in SAMPLES:
            raise SystemExit(f"unknown sample '{name}' (extend SAMPLES)")
        det_conf, zenith_bins, _ = SAMPLES[name]

        axes = sample["axes"]
        sizes = [int(a["nBins"]) for a in axes]
        expectation = np.asarray(sample["prediction"], dtype=float)
        if expectation.size != int(np.prod(sizes)):
            raise SystemExit(
                f"{name}: prediction has {expectation.size} bins, axes give {sizes}"
            )
        if sizes[1] != zenith_bins:
            raise SystemExit(
                f"{name}: config has {sizes[1]} zenith bins, this script expects "
                f"{zenith_bins} (extend SAMPLES)"
            )
        energy_bins = sizes[0]
        ra_bins = sizes[2] if len(sizes) > 2 else 1

        # The draw itself. Poisson of a non-negative expectation is integer by
        # construction, so no rounding step is needed (or wanted -- rounding a
        # draw would bias it).
        counts = rng.poisson(np.maximum(expectation, 0.0)).astype(float)

        ours_path = os.path.join(args.out_dir, f"pseudo_{name}.txt")
        with open(ours_path, "w") as f:
            f.write(f"# pseudo-data counts for '{name}', seed {args.seed}\n")
            f.write(f"# {counts.size} bins, {counts.sum():.0f} events, "
                    f"expectation {expectation.sum():.4f}\n")
            for value in counts:
                f.write(f"{value:.0f}\n")

        nnmfit_hists[det_conf] = mirror_zenith(counts, energy_bins, zenith_bins, ra_bins)
        print(f"{name}: {counts.sum():.0f} events drawn from {expectation.sum():.4f} "
              f"expected ({counts.size} bins) -> {ours_path}")

    pickle_path = os.path.join(args.out_dir, "pseudo_data_nnmfit.pickle")
    with open(pickle_path, "wb") as f:
        pickle.dump({"histograms": nnmfit_hists}, f)
    print(f"wrote {pickle_path} ({len(nnmfit_hists)} detector configs)")


if __name__ == "__main__":
    main()
