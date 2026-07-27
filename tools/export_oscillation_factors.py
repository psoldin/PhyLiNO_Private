#!/usr/bin/env python3
"""Evaluate NNMFit's OscillationsHook survival probability per MC event.

Writes a one-column parquet sidecar, row-aligned with the input baseline parquet,
holding the nu_mu / anti-nu_mu survival probability (1.0 for every other particle
type). NNMFit applies this factor to the conventional and prompt baseline weights
at load time; see NNMFit/fluxes/flux_hooks.py:96-155.

Run with the NNMFit venv (needs scipy + pyarrow, and the same spline objects
NNMFit uses, so the factors are exact rather than re-interpolated):
  /Users/soldin/Projects/IceCube/NNMFit/.venv/bin/python \
      tools/export_oscillation_factors.py MC.parquet SPLINE.pickle OUT.parquet
"""
import argparse
import pickle

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mc_parquet")
    p.add_argument("spline_pickle")
    p.add_argument("out_parquet")
    p.add_argument("--true-energy", default="MCPrimaryEnergy")
    p.add_argument("--true-zenith", default="MCPrimaryZenith")
    p.add_argument("--true-ptype", default="MCPrimaryType")
    args = p.parse_args()

    table = pq.read_table(
        args.mc_parquet, columns=[args.true_energy, args.true_zenith, args.true_ptype]
    )
    energy = np.asarray(table[args.true_energy], dtype=float)
    zenith = np.asarray(table[args.true_zenith], dtype=float)
    ptype = np.asarray(table[args.true_ptype], dtype=float).astype(int)

    with open(args.spline_pickle, "rb") as f:
        splines = pickle.load(f)["OscProb_Splines"]

    # Same loop as the hook: only nu_mu / anti-nu_mu are affected.
    factor = np.ones(len(ptype))
    for particle in (-14, 14):
        mask = ptype == particle
        if not mask.any():
            continue
        spline = splines[particle][abs(particle)]
        factor[mask] = spline(np.log10(energy[mask]), zenith[mask], grid=False)

    print(
        f"{args.mc_parquet}: {len(ptype)} events, "
        f"{int((ptype == 14).sum() + (ptype == -14).sum())} nu_mu, "
        f"factor min {factor.min():.6f} max {factor.max():.6f} mean {factor.mean():.6f}"
    )
    pq.write_table(pa.table({"osc_survival": pa.array(factor)}), args.out_parquet)
    print(f"wrote {args.out_parquet}")


if __name__ == "__main__":
    main()
