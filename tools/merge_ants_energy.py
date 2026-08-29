#!/usr/bin/env python3
"""Merge the ELEFANTS energy reconstructions into the NNMFit tracks dataset.

The two ANTS_energy files hold the same ~20.7M events reconstructed under two
topology hypotheses (throughgoing / starting).  Each carries, per event:

    truths   log10(E/GeV) of the quantity the network was trained on
             (Edep_primary_muon_closest_MMC)
    pred_0   the predicted log10(E/GeV)
    pred_1   the predicted LOG of the Gaussian's width, not the width itself

pred_1 being a log-sigma is not a guess: it is negative for 99.99% of rows, so
it cannot be a width, and the pull (pred_0 - truths) / exp(pred_1) has a spread
of order one while (pred_0 - truths) / 10**pred_1 is off by a factor ~7.  An
earlier version of this script took pred_1 for the width directly and produced a
negative sigma column, which is why the widths are now derived explicitly here
rather than left for the consumer to interpret.

Columns added per topology (prefix ELEFANTS_tg / ELEFANTS_st):

    _truth_log10   truths                     the kernel centre for a forward-
                                              folded (truth-centred) likelihood
    _mu_log10      pred_0                     reconstructed log10 energy
    _log_sigma     pred_1                     raw network output
    _sigma_log10   exp(pred_1)                width in dex, ready to use
    _mu            10**pred_0                 linear GeV        (--linear)
    _sigma         ln10 * mu * exp(pred_1)    linear GeV        (--linear)

Join key is (run_id, event_id, sub_event_id).  sub_run_id is NOT used: in the
ANTS files it is the uint32 sentinel 4294967295 for every row, and the sub-run
is already folded into run_id (run * 100000 + sub_run) in both datasets.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path("/Users/soldin/Downloads/nnmfit_files_shuyang/"
            "dataset_NT_FL_SnowStorm_tracks_NuMu.parquet")
ANTS_DIR = Path("/Users/soldin/Projects/DoubleChooz/software/FitPaper/"
                "PhyLiNO_Finish/ANTS_energy")

# file -> column prefix
TOPOLOGIES = {
    "throughgoing_energy.parquet": "ELEFANTS_tg",
    "starting_tracks_energy.parquet": "ELEFANTS_st",
}

KEY_MAIN = ["run_id", "event_id", "subevent_id"]
KEY_ANTS = ["run_id", "event_id", "sub_event_id"]


def load_topology(path, prefix, linear):
    df = pd.read_parquet(path, columns=KEY_ANTS + ["truths", "pred_0", "pred_1"])
    df = df.rename(columns={"truths": f"{prefix}_truth_log10",
                            "pred_0": f"{prefix}_mu_log10",
                            "pred_1": f"{prefix}_log_sigma"})
    # The ANTS columns are float32 on disk; ICDataBase only accepts double.
    for column in (f"{prefix}_truth_log10", f"{prefix}_mu_log10", f"{prefix}_log_sigma"):
        df[column] = df[column].astype("float64")

    # The width the network actually predicts.  Kept beside the raw output so a
    # consumer never has to know the convention.
    df[f"{prefix}_sigma_log10"] = np.exp(df[f"{prefix}_log_sigma"])

    if linear:
        df[f"{prefix}_mu"] = 10.0 ** df[f"{prefix}_mu_log10"]
        # delta method, from a width that is now genuinely a width
        df[f"{prefix}_sigma"] = np.log(10.0) * df[f"{prefix}_mu"] * df[f"{prefix}_sigma_log10"]

    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", type=Path, default=MAIN)
    ap.add_argument("--ants-dir", type=Path, default=ANTS_DIR)
    ap.add_argument("--out", type=Path, required=True,
                    help="output parquet path")
    ap.add_argument("--linear", action="store_true",
                    help="also add linear-space mu and delta-method sigma")
    args = ap.parse_args()

    df = pd.read_parquet(args.main)
    n_rows = len(df)
    print(f"main: {n_rows} rows, {df.shape[1]} columns")

    # The main frame carries an unnamed 4-level MultiIndex
    # (run, sub_run, event, subevent); merge() would drop it, so park it in a
    # column and restore afterwards.
    index_names = list(df.index.names)
    df = df.reset_index(names=[f"__idx_{i}" for i in range(df.index.nlevels)])
    index_cols = [f"__idx_{i}" for i in range(len(index_names))]

    for filename, prefix in TOPOLOGIES.items():
        right = load_topology(args.ants_dir / filename, prefix, args.linear)
        added = [c for c in right.columns if c.startswith(prefix)]
        print(f"{filename}: {len(right)} rows -> adding {added}")

        collisions = set(added) & set(df.columns)
        if collisions:
            raise SystemExit(f"column name collision: {sorted(collisions)}")
        if right.duplicated(KEY_ANTS).any():
            raise SystemExit(f"{filename} has duplicate join keys")

        df = df.merge(right, how="left",
                      left_on=KEY_MAIN, right_on=KEY_ANTS,
                      validate="one_to_one")
        df = df.drop(columns=[c for c in KEY_ANTS if c not in KEY_MAIN])

        unmatched = df[added[0]].isna().sum()
        print(f"  unmatched rows: {unmatched} "
              f"({100.0 * unmatched / n_rows:.4f} %)")

    assert len(df) == n_rows, f"row count changed: {n_rows} -> {len(df)}"

    df = df.set_index(index_cols)
    df.index.names = index_names

    df.to_parquet(args.out)
    print(f"wrote {args.out}: {len(df)} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    main()
