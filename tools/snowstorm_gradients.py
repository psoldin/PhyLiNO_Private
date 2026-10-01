#!/usr/bin/env python3
"""Regenerate SnowStorm detector gradients in an arbitrary analysis binning.

Reimplements NNMFit's recipe (NNMFit/parameters/snowstorm/snowstorm_gradient_direct.py,
lines 146-169): for each systematic, split the ensemble at the midpoint of its
SnowStorm sampling range, histogram each half, and take

    gradient = 4 / (hi - lo) * (hist_upper - hist_lower)

which is d(mu)/d(parameter) for a uniformly sampled ensemble -- the two half-means
are separated by (hi - lo) / 2 and each half carries half the events.

What is written out is the RELATIVE gradient, gradient / hist_all, because the
ensemble and the analysis sample are different MC productions (22852-22861 versus
22042-22044) with different sizes, livetimes and selections. A fractional response
per bin transfers between them; an absolute one does not. That is the same
assumption NNMFit's own lt_scale makes, in scale-free form.

Bins with too few ensemble events to measure a difference are written as zero and
counted, rather than propagating noise into the Fisher matrix as if it were signal.
"""
import argparse
import sys

import numpy as np
import pyarrow.parquet as pq

# From the pickle's own settings (Snowstorm_Gradients_FTP_Tracksonly) and
# NNMFit example_config.yaml: sampling range, then split at its midpoint.
SYSTEMATICS = {
    "DOMEfficiency":     (0.9, 1.1),
    "IceAbsorption":     (0.9, 1.1),
    "IceScattering":     (0.9, 1.1),
    "HoleIceForward_p0": (-0.1, 0.5980366362473171),
    "HoleIceForward_p1": (-0.1135759700999585, 0.0),
}


def read_dump_header(path):
    """Axis definitions and category edges written by McVariance --dump-file."""
    header = {"categories": []}
    with open(path) as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            parts = line[1:].split()
            if parts[0] == "energy":
                header["energy"] = (float(parts[1]), float(parts[2]), int(parts[3]))
            elif parts[0] == "zenith":
                header["zenith"] = (float(parts[1]), float(parts[2]), int(parts[3]))
            elif parts[0] == "category":
                header["categories"].append(
                    {"name": parts[1], "n": int(parts[2]), "edges": [float(v) for v in parts[3:]]}
                )
            elif parts[0] == "bins":
                header["bins"] = int(parts[1])
    return header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--dump", required=True, help="binning definition from McVariance --dump-file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-events", type=int, default=20,
                    help="bins with fewer ensemble events on either side of a split get a zero gradient")
    ap.add_argument("--astro-norm", type=float, default=1.8)
    ap.add_argument("--gamma", type=float, default=2.52)
    args = ap.parse_args()

    head = read_dump_header(args.dump)
    e_lo, e_hi, n_e = head["energy"]
    z_lo, z_hi, n_z = head["zenith"]
    cats = head["categories"]

    cols = ["energy_truncated", "zenith_MPEFit", "MCPrimaryEnergy",
            "powerlaw", "mceq_conv_H4a_SIBYLL23c", "mceq_pr_H4a_SIBYLL23c"]
    cols += [c["name"] for c in cats] + list(SYSTEMATICS)
    table = pq.read_table(args.ensemble, columns=cols)
    d = {c: np.asarray(table[c], dtype=np.float64) for c in cols}

    # Nominal weighting. Only the RELATIVE gradient is exported, so this needs to
    # be a reasonable spectrum rather than the fit's exact one.
    e_true = d["MCPrimaryEnergy"]
    weight = (args.astro_norm * d["powerlaw"] * (e_true / 1.0e5) ** (2.0 - args.gamma)
              + d["mceq_conv_H4a_SIBYLL23c"] + d["mceq_pr_H4a_SIBYLL23c"])

    log_e = np.log10(np.where(d["energy_truncated"] > 0, d["energy_truncated"], np.nan))
    cos_z = np.cos(d["zenith_MPEFit"])

    ie = np.floor((log_e - e_lo) / ((e_hi - e_lo) / n_e)).astype(np.int64)
    iz = np.floor((cos_z - z_lo) / ((z_hi - z_lo) / n_z)).astype(np.int64)

    ok = np.isfinite(log_e) & np.isfinite(cos_z) & np.isfinite(weight)
    ok &= (ie >= 0) & (ie < n_e) & (iz >= 0) & (iz < n_z)

    flat = ie * n_z + iz
    n_bins = n_e * n_z
    for cat in cats:
        v = d[cat["name"]]
        k = np.searchsorted(np.asarray(cat["edges"]), v, side="right")
        flat = flat * cat["n"] + k
        n_bins *= cat["n"]
        ok &= np.isfinite(v)

    if n_bins != head["bins"]:
        sys.exit(f"bin count mismatch: dump says {head['bins']}, reconstructed {n_bins}")

    flat = flat[ok]
    w = weight[ok]
    print(f"ensemble: {ok.sum()} of {len(ok)} events inside the binning, {n_bins} bins "
          f"({ok.sum() / n_bins:.1f} per bin)")

    hist_all = np.bincount(flat, weights=w, minlength=n_bins)
    count_all = np.bincount(flat, minlength=n_bins)

    out = {}
    for name, (lo, hi) in SYSTEMATICS.items():
        split = 0.5 * (lo + hi)
        factor = 4.0 / (hi - lo)
        p = d[name][ok]

        upper, lower = p > split, p < split
        h_up = np.bincount(flat[upper], weights=w[upper], minlength=n_bins)
        h_dn = np.bincount(flat[lower], weights=w[lower], minlength=n_bins)
        n_up = np.bincount(flat[upper], minlength=n_bins)
        n_dn = np.bincount(flat[lower], minlength=n_bins)

        # Squared weights give the gradient's own statistical error, exactly as
        # NNMFit's covariance_gradient does.
        s_up = np.bincount(flat[upper], weights=w[upper] ** 2, minlength=n_bins)
        s_dn = np.bincount(flat[lower], weights=w[lower] ** 2, minlength=n_bins)

        grad = factor * (h_up - h_dn)
        err = factor * np.sqrt(s_up + s_dn)

        usable = (count_all > 0) & (n_up >= args.min_events) & (n_dn >= args.min_events) & (hist_all > 0)
        rel = np.zeros(n_bins)
        rel_err = np.zeros(n_bins)
        rel[usable] = grad[usable] / hist_all[usable]
        rel_err[usable] = err[usable] / hist_all[usable]
        out[name] = (rel, rel_err, usable)

        sig = np.abs(rel[usable]) / np.maximum(rel_err[usable], 1e-300)
        print(f"  {name:20s} split {split:+.4f} factor {factor:7.3f}  "
              f"{usable.sum():6d}/{n_bins} bins usable  median |grad|/err {np.median(sig):.2f}")

    with open(args.out, "w") as handle:
        handle.write(f"# snowstorm relative gradients, bins {n_bins} params {len(SYSTEMATICS)}\n")
        handle.write("# per param: name, then <bins> lines of 'rel_gradient rel_error usable'\n")
        for name, (rel, rel_err, usable) in out.items():
            handle.write(f"{name}\n")
            for b in range(n_bins):
                handle.write(f"{rel[b]:.8g} {rel_err[b]:.8g} {int(usable[b])}\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
