#!/usr/bin/env python3
"""Write a copy of an NNMFit config that isolates one sample and one component set.

Targets NNMFit's newer schema (as in Fit_Configuration_Combined_macOS.yaml):
samples live under `datasets:`, the sample list is `analysis.datasets`, and
`excluded_components` is a comma-separated string per sample.

Also switches `analysis.analysis_type` to `asimov` (unless --keep-analysis-type is
given), because make_histogram.py dumps `get_data_hists()`: with `analysis_type:
data` that is the measured counts, while with `asimov` it is the model prediction
mu at the config's parameter values, plus the expected fluctuations (ssq). The
model is what we diff PhyLiNO's prediction against.

Usage:
  nnmfit_set_excluded.py IN.yaml OUT.yaml SAMPLE "comp_a, comp_b" [--keep-analysis-type]
"""
import argparse
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_yaml")
    parser.add_argument("out_yaml")
    parser.add_argument("sample")
    parser.add_argument("excluded", help='comma-separated component names, "" for none')
    parser.add_argument("--keep-analysis-type", action="store_true")
    args = parser.parse_args()

    with open(args.in_yaml) as handle:
        config = yaml.safe_load(handle)

    datasets = config.get("datasets")
    if datasets is None or args.sample not in datasets:
        sys.exit(f"sample '{args.sample}' not under 'datasets:' in {args.in_yaml}")

    datasets[args.sample]["excluded_components"] = args.excluded
    # Build only this sample, so make_histogram does not touch the others.
    config["analysis"]["datasets"] = [args.sample]
    if not args.keep_analysis_type:
        config["analysis"]["analysis_type"] = "asimov"

    with open(args.out_yaml, "w") as handle:
        yaml.safe_dump(config, handle)
    print(f"wrote {args.out_yaml} (sample {args.sample}, excluded: {args.excluded or 'none'})")


if __name__ == "__main__":
    main()
