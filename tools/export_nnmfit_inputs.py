#!/usr/bin/env python3
"""Convert NNMFit muon-template and SnowStorm-gradient pickles to plain text.

The C++ side (TemplateFlux, DetectorSystematics) reads whitespace-separated
numbers with a small header, so no Python/pickle dependency leaks into the fit.

Usage:
  export_nnmfit_inputs.py template  IN.pickle OUT.txt [--det-conf NAME] --bins N
  export_nnmfit_inputs.py gradients IN.pickle OUT.txt --det-conf NAME --bins N \
                                     [--livetime-scale S]

The detector-config names in the pickles are NNMFit's, e.g.
IC86_pass2_SnowStorm_v2_cscd_cascade. --det-conf is omitted for single-dataset
template pickles (e.g. the tracks Corsika muon template).
"""
import argparse
import pickle
import sys

import numpy as np


def fnum(x):
    """repr() of a plain Python float, round-trip exact.

    numpy>=2.0 changed repr(np.float64(x)) to "np.float64(x)", which the C++
    reader (plain istream >> double) cannot parse. Casting to a native float
    first keeps the output numpy-version-independent.
    """
    return repr(float(x))


def reverse_zenith_axis(values, zenith_bins):
    """Undo NNMFit's zenith-axis mirror relative to io::ic::Binning's convention.

    NNMFit's cos/cscd-cos_5up bin edges are built as
    np.sort(arccos(ascending_cos_values)): arccos is monotonically DECREASING,
    so an ascending-cos input produces a descending-zenith-radians array, which
    np.sort then flips back to ascending zenith radians. The net effect,
    verified numerically against both the uniform "cos" and the explicit
    "cscd-cos_5up" edges: NNMFit's zenith bin index and ours satisfy
        nnmfit_zenith_idx == (zenith_bins - 1) - our_zenith_idx
    for every event, i.e. the zenith axis is exactly mirrored (energy is
    unaffected: NNMFit's "log" spacing is ascending already, matching ours).

    Any flat array NNMFit hands us (template rate, template fluctuation,
    gradient, gradient_error, or a covariance column) is row-major
    (energy outer, NNMFit's zenith inner); reshape to (n_energy, zenith_bins)
    and flip the zenith axis to relabel it into our own bin_index() order,
    which is what the C++ reader (TemplateFlux, DetectorSystematics) assumes.

    zenith_bins == 1 makes this a no-op (the single-bin cscd_muon sample).
    """
    if zenith_bins <= 1:
        return values
    if values.size % zenith_bins != 0:
        sys.exit(f"reverse_zenith_axis: {values.size} values not divisible by zenith_bins={zenith_bins}")
    reshaped = values.reshape(-1, zenith_bins)
    return np.flip(reshaped, axis=1).reshape(-1)


def reverse_zenith_axis_3d(values, zenith_bins, ra_bins):
    """As reverse_zenith_axis, for a flat (energy, zenith, RA) row-major array.

    With an RA axis present the zenith axis is no longer the fast one, so the flat
    array reshapes to (n_energy, zenith_bins, ra_bins) and the middle axis is the one
    that gets mirrored into io::ic::Binning's order.
    """
    if zenith_bins <= 1:
        return values
    block = zenith_bins * ra_bins
    if values.size % block != 0:
        sys.exit(
            f"reverse_zenith_axis_3d: {values.size} values not divisible by "
            f"zenith_bins*ra_bins={block}"
        )
    reshaped = values.reshape(-1, zenith_bins, ra_bins)
    return np.flip(reshaped, axis=1).reshape(-1)

# The order DetectorSystematics assumes, matching params::ic {DOMEff, IceAbs,
# IceScat, HoleIceP0, HoleIceP1}.
SYSTEMATICS = [
    "DOMEfficiency",
    "IceAbsorption",
    "IceScattering",
    "HoleIceForward_p0",
    "HoleIceForward_p1",
]

# Which cross_correlations error keys enter the covariance of two gradients
# (NNMFit snowstorm_gradient.__covariance_g1_g2): +up/up, +lo/lo, -lo/up, -up/lo.
COV_TERMS = [
    ("sys up-alt sys up", +1.0),
    ("sys low-alt sys low", +1.0),
    ("sys low-alt sys up", -1.0),
    ("sys up-alt sys low", -1.0),
]


def load(path, det_conf):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if det_conf is not None and det_conf in obj:
        return obj[det_conf]
    if det_conf is not None:
        sys.exit(
            f"detector config '{det_conf}' not in {path}; available: {sorted(obj.keys())}"
        )
    return obj


def flat(array, bins, what, zenith_bins):
    a = np.asarray(array, dtype=float)
    if a.size != bins:
        sys.exit(f"{what}: expected {bins} values, pickle has {a.size} (shape {a.shape})")
    # Flattening is row-major: first axis (energy) outer, second (zenith) inner --
    # the same order io::ic::Binning uses for its flat index. Verified against the
    # tracks Corsika pickle: its flat "template" equals template_2d.flatten() with
    # template_2d shaped (n_energy, n_zenith). The zenith sub-axis itself is then
    # relabelled into our own bin_index() order (see reverse_zenith_axis).
    return reverse_zenith_axis(a.reshape(-1), zenith_bins)


def export_template(entry, out, bins, zenith_bins):
    # Two layouts seen in practice: the multi-dataset MuonGun pickles carry a flat
    # "template", and so does the single-dataset Corsika tracks pickle (alongside
    # "template_2d", which is the same data pre-flatten -- "template" is used
    # directly whenever present). template_fluctuation may be None (the
    # "_no_fluct" MuonGun variant the reference fit used, and the Corsika tracks
    # pickle, which carries no fluctuation key at all) -- exported as zeros, which
    # makes the sigma^2 term vanish exactly as NNMFit's None fluctuation graph does.
    key = "template" if "template" in entry else "template_2d"
    if key not in entry:
        sys.exit(f"pickle entry has neither 'template' nor 'template_2d'; keys: {sorted(entry)}")
    template = flat(entry[key], bins, key, zenith_bins)
    fluctuation = entry.get("template_fluctuation")
    fluctuation = (
        flat(fluctuation, bins, "template_fluctuation", zenith_bins)
        if fluctuation is not None
        else np.zeros(bins)
    )
    with open(out, "w") as f:
        f.write(f"# template bins {bins}\n")
        f.write("# columns: template_rate fluctuation_rate (both per second)\n")
        f.write("# zenith axis relabelled into io::ic::Binning order (see reverse_zenith_axis)\n")
        if "energy_bins" in entry:
            f.write(f"# energy_bins {' '.join(repr(float(x)) for x in entry['energy_bins'])}\n")
        if "zenith_bins" in entry:
            f.write(f"# cos_zenith_bins {' '.join(repr(float(x)) for x in entry['zenith_bins'])}\n")
        for t, s in zip(template, fluctuation):
            f.write(f"{fnum(t)} {fnum(s)}\n")
    print(f"wrote {out}: {bins} bins, template sum {fnum(template.sum())} s^-1")


def export_gradients(entry, out, bins, livetime_scale, zenith_bins):
    missing = [s for s in SYSTEMATICS if s not in entry]
    if missing:
        sys.exit(f"gradient pickle lacks systematics {missing}; has {sorted(entry.keys())}")

    with open(out, "w") as f:
        f.write(
            f"# gradients bins {bins} params {len(SYSTEMATICS)} lt_scale {fnum(livetime_scale)}\n"
        )
        f.write("# per param: name split, then <bins> lines of 'gradient gradient_error'\n")
        f.write("# zenith axis relabelled into io::ic::Binning order (see reverse_zenith_axis)\n")
        for name in SYSTEMATICS:
            g = entry[name]
            gradient = flat(g["gradient"], bins, f"{name}.gradient", zenith_bins)
            error = flat(g["gradient_error"], bins, f"{name}.gradient_error", zenith_bins)
            f.write(f"# param {name} split {fnum(g['split_value'])}\n")
            for v, e in zip(gradient, error):
                f.write(f"{fnum(v)} {fnum(e)}\n")

        f.write("# per pair: names, then <bins> lines of 'covariance'\n")
        for i, name_i in enumerate(SYSTEMATICS):
            for name_j in SYSTEMATICS[i + 1 :]:
                corr = entry[name_i]["cross_correlations"][name_j]
                split_terms = np.zeros(bins)
                for key, sign in COV_TERMS:
                    split_terms += sign * flat(
                        np.asarray(corr[key]["error"]) ** 2, bins, f"{name_i}/{name_j} {key}", zenith_bins
                    )
                cov = (
                    float(entry[name_i]["factor"])
                    * float(entry[name_j]["factor"])
                    * split_terms
                )
                f.write(f"# cov {name_i} {name_j}\n")
                for v in cov:
                    f.write(f"{fnum(v)}\n")
    print(f"wrote {out}: {bins} bins x {len(SYSTEMATICS)} systematics + 10 covariance pairs")


def export_galactic(entry, out, bins, zenith_bins, ra_bins):
    """Write an NNMFit GalacticTemplate entry in the C++ TemplateFlux text format.

    The fluctuation column is deliberately all zeros: NNMFit's GalacticTemplate
    defines no make_fluctuations_graph, so histogram_builder excludes it from the ssq
    sum entirely (it logs "is a histogram component and excluded in the ssq
    calculation"). The pickle's rate_error array is therefore never read by NNMFit,
    and writing zeros makes our (norm * 0 * livetime)**2 term vanish the same way.
    """
    if "rate" not in entry:
        sys.exit(f"galactic pickle entry has no 'rate'; keys: {sorted(entry)}")
    rate = np.asarray(entry["rate"], dtype=float).reshape(-1)
    if rate.size != bins:
        sys.exit(f"galactic rate: expected {bins} values, pickle has {rate.size}")
    rate = reverse_zenith_axis_3d(rate, zenith_bins, ra_bins)

    settings = entry.get("binning_settings", {})
    with open(out, "w") as f:
        f.write(f"# template bins {bins}\n")
        f.write("# columns: template_rate fluctuation_rate (both per second)\n")
        f.write("# galactic template; fluctuation is zero (NNMFit excludes it from ssq)\n")
        f.write("# zenith axis relabelled into io::ic::Binning order (see reverse_zenith_axis_3d)\n")
        for key in ("reco_energy_binning", "reco_zenith_binning", "reco_ra_binning"):
            if key in settings:
                f.write(f"# {key} {settings[key]}\n")
        for value in rate:
            f.write(f"{fnum(value)} 0.0\n")
    print(f"wrote {out}: {bins} bins, rate sum {fnum(rate.sum())} s^-1")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["template", "gradients", "galactic"])
    p.add_argument("pickle_path")
    p.add_argument("out_path")
    p.add_argument("--det-conf", default=None)
    p.add_argument("--bins", type=int, required=True)
    p.add_argument(
        "--zenith-bins",
        type=int,
        required=True,
        help="number of zenith bins (the fast-varying axis); used to undo NNMFit's "
        "zenith-axis mirror (see reverse_zenith_axis). Pass 1 for a binning with no "
        "zenith axis or a single zenith bin.",
    )
    p.add_argument(
        "--livetime-scale",
        type=float,
        default=1.0,
        help="analysis livetime / gradient livetime (NNMFit livetime_scaling)",
    )
    p.add_argument(
        "--ra-bins",
        type=int,
        default=1,
        help="number of RA bins (the fast-varying axis of a galactic template); "
        "required for kind=galactic",
    )
    args = p.parse_args()

    entry = load(args.pickle_path, args.det_conf)
    if args.kind == "template":
        export_template(entry, args.out_path, args.bins, args.zenith_bins)
    elif args.kind == "galactic":
        if args.ra_bins < 2:
            sys.exit("kind=galactic needs --ra-bins >= 2")
        export_galactic(entry, args.out_path, args.bins, args.zenith_bins, args.ra_bins)
    else:
        export_gradients(entry, args.out_path, args.bins, args.livetime_scale, args.zenith_bins)


if __name__ == "__main__":
    main()
