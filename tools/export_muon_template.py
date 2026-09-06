#!/usr/bin/env python3
"""Export NNMFit's muon-template pickle to the text file PhyLiNO's TemplateFlux reads.

NNMFit stores the template flattened in its own analysis order -- energy outer,
zenith *angle* inner, so ascending zenith and therefore DESCENDING cos(zenith).
io::ic::Binning is ascending cos(zenith), so the zenith axis is reversed on the
way out. (The pickle's own `zenith_bins` are ascending in cos, which is the
metadata for the *un*flipped array; NNMFit's TemplateFlux asserts against
`np.cos(bin_edges)[::-1]` and then uses the array as stored. See
NNMFit/fluxes/TemplateFlux.py.)

Both columns stay in Hz: PhyLiNO multiplies by the sample livetime at load time.
The fluctuation column is written verbatim, because that is the array NNMFit
itself squares into ssq -- copying it unchanged is what makes the two fits
comparable, whatever the template creator meant by it.

  python3 tools/export_muon_template.py muon_template_ANTS.pkl out.txt
"""
import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np


def fnum(v):
    """repr() of a plain Python float: round-trips exactly, parses in C++ istream."""
    return repr(float(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pickle_file", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n-energy", type=int, default=50)
    ap.add_argument("--n-zenith", type=int, default=33)
    args = ap.parse_args()

    with open(args.pickle_file, "rb") as f:
        pkl = pickle.load(f)
    digest = hashlib.sha256(args.pickle_file.read_bytes()).hexdigest()

    ne, nz = args.n_energy, args.n_zenith
    rate = np.asarray(pkl["template"], float).reshape(ne, nz)
    sigma = np.asarray(pkl["template_fluctuation"], float).reshape(ne, nz)

    # NNMFit order -> io::ic::Binning order.
    rate, sigma = rate[:, ::-1], sigma[:, ::-1]

    # The pickle carries the analysis grid it was built on; write it into the
    # header so a mismatched file stays diagnosable after the fact. PhyLiNO only
    # hard-checks the bin count, but these are what it should be checked against.
    e_edges = np.asarray(pkl["energy_bins"], float)          # GeV
    z_edges = np.asarray(pkl["zenith_bins"], float)          # cos(zenith), ascending
    if len(e_edges) != ne + 1 or len(z_edges) != nz + 1:
        raise SystemExit(f"grid mismatch: pickle has {len(e_edges)}/{len(z_edges)} edges, "
                         f"expected {ne + 1}/{nz + 1}")

    with open(args.out, "w") as f:
        f.write(f"# template bins {ne * nz}\n")
        f.write("# columns: template_rate fluctuation_rate (both per second)\n")
        f.write("# zenith axis relabelled into io::ic::Binning order (ascending cos(zenith))\n")
        f.write(f"# exported by tools/export_muon_template.py from {args.pickle_file.name}\n")
        f.write(f"# source sha256 {digest}\n")
        f.write(f"# source description: {pkl.get('description', '-')}\n")
        f.write(f"# energy_bins {' '.join(fnum(v) for v in e_edges)}\n")
        f.write(f"# cos_zenith_bins {' '.join(fnum(v) for v in z_edges)}\n")
        for t, s in zip(rate.ravel(), sigma.ravel()):
            f.write(f"{fnum(t)} {fnum(s)}\n")

    print(f"wrote {args.out}: {ne * nz} bins, "
          f"total {rate.sum():.6e} Hz, fluctuation {sigma.sum():.6e} Hz")
    print(f"source {args.pickle_file}  sha256 {digest}")


if __name__ == "__main__":
    main()
