#!/usr/bin/env python3
"""Convert NNMFit muon-template and SnowStorm-gradient pickles to plain text.

The C++ side (TemplateFlux, DetectorSystematics) reads whitespace-separated
numbers with a small header, so no Python/pickle dependency leaks into the fit.

Usage:
  export_nnmfit_inputs.py template  IN.pickle OUT.txt [--det-conf NAME] --bins N \
                                     [--rebin-energy LO,HI,NBINS]
  export_nnmfit_inputs.py gradients IN.pickle OUT.txt --det-conf NAME --bins N \
                                     [--rebin-energy LO,HI,NBINS] \
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


def rebin_energy_matrix(source_edges, lo, hi, n_bins):
    """Fractional overlap of each source log10(E) bin with each target bin.

    W[t, s] is the share of source bin s that lands in target bin t, taking the
    rate as uniform in log10(E) across a source bin. Whatever falls outside
    [lo, hi] is dropped, which is what the analysis binning does to it anyway.

    # ponytail: piecewise-constant density. Exact wherever the source bins are
    # finer than the target ones -- below log10(E) ~ 4 for the ANTS muon
    # template, which is >99.9% of its rate. Above that the source bins are the
    # wider ones and the split is a flat approximation, on a rate worth well
    # under one event per livetime. Rebuild the pickle on the analysis binning
    # if that ever stops being true.
    """
    target = np.linspace(lo, hi, n_bins + 1)
    src_lo, src_hi = source_edges[:-1], source_edges[1:]
    overlap = np.minimum(target[1:, None], src_hi[None, :]) - np.maximum(target[:-1, None], src_lo[None, :])
    return np.clip(overlap, 0.0, None) / (src_hi - src_lo)[None, :]


def rebin_energy(rate, fluctuation, source_edges, spec):
    """Move a (n_energy, n_zenith) rate/fluctuation pair onto a uniform log10(E) grid.

    Rates are additive, so they carry across with the overlap fractions. The
    fluctuations are per-bin sigmas over independent MC bin sums, so they add in
    quadrature with those same fractions.
    """
    lo, hi, n_bins = spec
    w = rebin_energy_matrix(source_edges, lo, hi, n_bins)
    out_rate = w @ rate
    kept = out_rate.sum() / rate.sum() if rate.sum() else 0.0
    print(
        f"energy rebin: {rate.shape[0]} -> {n_bins} bins over log10(E) [{lo}, {hi}]; "
        f"{kept:.4%} of the template rate is inside the analysis range"
    )
    return out_rate, np.sqrt((w**2) @ fluctuation**2)


def export_template(entry, out, bins, zenith_bins, rebin=None):
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
    rate = np.asarray(entry[key], dtype=float).reshape(-1)
    raw_fluctuation = entry.get("template_fluctuation")
    fluctuation = (
        np.asarray(raw_fluctuation, dtype=float).reshape(-1)
        if raw_fluctuation is not None
        else np.zeros_like(rate)
    )
    energy_edges = entry.get("energy_bins")
    if rebin is not None:
        if energy_edges is None:
            sys.exit("--rebin-energy needs the pickle's own 'energy_bins' edges; this one has none")
        if rate.size % zenith_bins:
            sys.exit(f"--rebin-energy: {rate.size} values not divisible by zenith_bins={zenith_bins}")
        rate, fluctuation = rebin_energy(
            rate.reshape(-1, zenith_bins),
            fluctuation.reshape(-1, zenith_bins),
            np.log10(np.asarray(energy_edges, dtype=float)),
            rebin,
        )
        energy_edges = np.logspace(rebin[0], rebin[1], rebin[2] + 1)
    template = flat(rate, bins, key, zenith_bins)
    fluctuation = flat(fluctuation, bins, "template_fluctuation", zenith_bins)
    with open(out, "w") as f:
        f.write(f"# template bins {bins}\n")
        f.write("# columns: template_rate fluctuation_rate (both per second)\n")
        f.write("# zenith axis relabelled into io::ic::Binning order (see reverse_zenith_axis)\n")
        if energy_edges is not None:
            f.write(f"# energy_bins {' '.join(repr(float(x)) for x in energy_edges)}\n")
        if "zenith_bins" in entry:
            f.write(f"# cos_zenith_bins {' '.join(repr(float(x)) for x in entry['zenith_bins'])}\n")
        for t, s in zip(template, fluctuation):
            f.write(f"{fnum(t)} {fnum(s)}\n")
    print(f"wrote {out}: {bins} bins, template sum {fnum(template.sum())} s^-1")


def gradient_livetime(entry, det_conf):
    """The livetime the gradients were computed at, from the pickle's own settings.

    NNMFit scales every gradient by `t_analysis / t_gradients`
    (snowstorm_gradient.py:193) and reads `t_gradients` out of the config it
    embedded in the gradient pickle. The tracks gradients were produced on MC
    with a livetime 6% shorter than the analysis livetime, so exporting them
    unscaled silently shrinks every detector-systematic effect -- invisible at
    the split values, where the gradient term is identically zero, and a ~3%
    likelihood error as soon as a detector parameter moves.

    Both config layouts NNMFit supports are handled: the newer one nests the
    per-dataset entries under "datasets", the older one has them at top level.
    """
    settings = entry.get("settings")
    if not settings or "config" not in settings:
        return None
    config = settings["config"]
    datasets = config.get("datasets", config)
    if det_conf not in datasets or "livetime" not in datasets[det_conf]:
        return None
    return float(datasets[det_conf]["livetime"])


def export_gradients(entry, out, bins, livetime_scale, zenith_bins, rebin=None):
    missing = [s for s in SYSTEMATICS if s not in entry]
    if missing:
        sys.exit(f"gradient pickle lacks systematics {missing}; has {sorted(entry.keys())}")

    # The gradient pickle carries its own binning as [energy_edges, zenith_edges].
    # When that energy axis is not the one the sample declares, every array in the
    # file moves onto the analysis grid the same way: gradients are per-bin counts
    # and add, errors and covariance terms are variances and add in quadrature.
    weights = None
    if rebin is not None:
        edges = entry.get("binning")
        if not edges:
            sys.exit("--rebin-energy needs the pickle's own 'binning' edges; this one has none")
        weights = rebin_energy_matrix(np.log10(np.asarray(edges[0], dtype=float)), *rebin)
        print(
            f"energy rebin: {weights.shape[1]} -> {weights.shape[0]} bins over "
            f"log10(E) [{rebin[0]}, {rebin[1]}]"
        )

    def prep(values, what, square=False):
        """Rebin (if asked) then relabel the zenith axis, as flat() does on its own."""
        a = np.asarray(values, dtype=float).reshape(-1)
        if weights is not None:
            if a.size % zenith_bins:
                sys.exit(f"{what}: {a.size} values not divisible by zenith_bins={zenith_bins}")
            m = weights**2 if square else weights
            a = (m @ a.reshape(-1, zenith_bins)).reshape(-1)
        return flat(a, bins, what, zenith_bins)

    with open(out, "w") as f:
        f.write(
            f"# gradients bins {bins} params {len(SYSTEMATICS)} lt_scale {fnum(livetime_scale)}\n"
        )
        f.write("# per param: name split, then <bins> lines of 'gradient gradient_error'\n")
        f.write("# zenith axis relabelled into io::ic::Binning order (see reverse_zenith_axis)\n")
        for name in SYSTEMATICS:
            g = entry[name]
            gradient = prep(g["gradient"], f"{name}.gradient")
            # gradient_error is a standard error (>= 0 in every pickle seen), so the
            # squared round trip is an identity when nothing is rebinned.
            error = np.sqrt(
                prep(np.asarray(g["gradient_error"], dtype=float) ** 2,
                     f"{name}.gradient_error", square=True)
            )
            f.write(f"# param {name} split {fnum(g['split_value'])}\n")
            for v, e in zip(gradient, error):
                f.write(f"{fnum(v)} {fnum(e)}\n")

        f.write("# per pair: names, then <bins> lines of 'covariance'\n")
        for i, name_i in enumerate(SYSTEMATICS):
            for name_j in SYSTEMATICS[i + 1 :]:
                corr = entry[name_i]["cross_correlations"][name_j]
                split_terms = np.zeros(bins)
                for key, sign in COV_TERMS:
                    split_terms += sign * prep(
                        np.asarray(corr[key]["error"]) ** 2, f"{name_i}/{name_j} {key}", square=True
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
        "--rebin-energy",
        default=None,
        metavar="LO,HI,NBINS",
        help="rebin the pickle's own log10(E) axis (read from its 'energy_bins' for "
        "kind=template, 'binning'[0] for kind=gradients) onto the uniform analysis "
        "grid, e.g. 2.5,7.0,45. Needed when the pickle was built on a different "
        "energy binning than the sample declares.",
    )
    p.add_argument(
        "--analysis-livetime",
        type=float,
        default=None,
        help="the analysis livetime (seconds) for kind=gradients. The gradient "
        "livetime is read from the pickle's own embedded config, and the ratio "
        "becomes NNMFit's livetime_scaling. Required unless --livetime-scale is "
        "given explicitly.",
    )
    p.add_argument(
        "--livetime-scale",
        type=float,
        default=None,
        help="analysis livetime / gradient livetime (NNMFit livetime_scaling), "
        "overriding the value derived from --analysis-livetime",
    )
    p.add_argument(
        "--ra-bins",
        type=int,
        default=1,
        help="number of RA bins (the fast-varying axis of a galactic template); "
        "required for kind=galactic",
    )
    args = p.parse_args()

    rebin = None
    if args.rebin_energy is not None:
        if args.kind == "galactic":
            sys.exit("--rebin-energy does not apply to kind=galactic")
        try:
            lo, hi, n_bins = args.rebin_energy.split(",")
            rebin = (float(lo), float(hi), int(n_bins))
        except ValueError:
            sys.exit(f"--rebin-energy expects LO,HI,NBINS, got '{args.rebin_energy}'")

    entry = load(args.pickle_path, args.det_conf)
    if args.kind == "template":
        export_template(entry, args.out_path, args.bins, args.zenith_bins, rebin)
    elif args.kind == "galactic":
        if args.ra_bins < 2:
            sys.exit("kind=galactic needs --ra-bins >= 2")
        export_galactic(entry, args.out_path, args.bins, args.zenith_bins, args.ra_bins)
    else:
        # Never fall back to 1.0 silently: that is exactly what produced a
        # 6%-too-small gradient term for tracks, undetectable at the split
        # values where the term vanishes.
        scale = args.livetime_scale
        if scale is None:
            if args.analysis_livetime is None:
                sys.exit(
                    "kind=gradients needs --analysis-livetime (the sample's livetime "
                    "in seconds, as in the PhyLiNO config) so the NNMFit livetime_scaling "
                    "can be derived, or an explicit --livetime-scale"
                )
            t_gradients = gradient_livetime(entry, args.det_conf)
            if t_gradients is None:
                sys.exit(
                    f"cannot read the gradient livetime for '{args.det_conf}' out of the "
                    f"pickle's embedded settings; pass --livetime-scale explicitly"
                )
            scale = args.analysis_livetime / t_gradients
            print(
                f"livetime scaling {scale!r} "
                f"(analysis {args.analysis_livetime!r} / gradients {t_gradients!r})"
            )
        export_gradients(entry, args.out_path, args.bins, scale, args.zenith_bins, rebin)


if __name__ == "__main__":
    main()
