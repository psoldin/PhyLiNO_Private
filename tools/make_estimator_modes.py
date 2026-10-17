#!/usr/bin/env python3
"""Write a muon template whose shape modes are the differences to other template estimators.

The bootstrap mode (MuonTmplMode1 from corsika-template/mode_prototype.py) covers how one
estimator's output scatters when its CORSIKA events are resampled. It does not cover the
choice of estimator itself -- other CORSIKA sets, weights, cuts or KDE settings -- and
templates built differently can disagree far beyond that scatter. Each alternative template
given here becomes one mode column

  d_k = s_k * T_alt,k - T_nominal,     s_k = sum(T_nominal) / sum(T_alt,k)   (default)

so that TemplateFlux's max(T + sum_k theta_k d_k, 0) is the nominal template at theta_k = 0
and reproduces alternative k exactly at theta_k = 1 (up to the rescaling s_k). With the prior
theta_k ~ N(0, 1), the spread between the estimators at hand is taken as the 1-sigma
uncertainty of the method. The rescaling makes every d_k a pure shape change (it sums to 0)
and leaves the normalization to MuonNorm; --keep-norm differences the templates as they are.

The nominal template's rate and fluctuation columns are copied unchanged. Mode columns the
nominal file already carries (a bootstrap mode) are kept in front with --keep-modes, so the
new modes follow them: with one kept mode, alternative 1 is MuonTmplMode2. At most
params::ic::nTemplateModes (3) modes fit. Use the result with the Poisson likelihood (README,
"Muon-template shape modes").

Every template file must be on the same grid (header bin count and bin-edge lines).

  python3 tools/make_estimator_modes.py nominal.txt out.txt alt1.txt [alt2.txt] \\
      [--keep-modes] [--keep-norm] [--livetime SECONDS]

Reported per mode: the closure T + d_k against the alternative, how much of the template the
mode moves, how much the clip at zero removes at theta = -1 and -2, and the overlap between
modes (cosine in the Poisson metric sum_b d_i d_j / T_b, 1 = same direction).
"""
import argparse
from pathlib import Path

import numpy as np

MAX_MODES = 3  # params::ic::nTemplateModes


def read_template(path):
    """(header lines, columns array [bins, 2 + modes], declared bin count, declared mode count)."""
    header, rows = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        (header if line.startswith("#") else rows).append(line)
    meta = {}
    for line in header:
        words = line[1:].split()
        if len(words) == 3 and words[0] == "template" and words[1] in ("bins", "modes"):
            meta[words[1]] = int(words[2])
    bins, modes = meta.get("bins"), meta.get("modes", 0)
    data = np.array([[float(x) for x in r.split()] for r in rows])
    if bins is None or data.shape != (bins, 2 + modes):
        raise SystemExit(f"{path}: header says {bins} bins and {modes} modes, rows have shape {data.shape}")
    return header, data, bins, modes


def grid_lines(header):
    return [line for line in header if line.split()[1:2] in (["energy_bins"], ["cos_zenith_bins"])]


def fnum(v):
    """repr() of a plain Python float: round-trips exactly, parses in C++ istream."""
    return repr(float(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nominal", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("alternatives", type=Path, nargs="+")
    ap.add_argument("--keep-modes", action="store_true",
                    help="keep the nominal file's own mode columns in front of the new ones")
    ap.add_argument("--keep-norm", action="store_true",
                    help="difference the templates as they are instead of rescaling to the nominal total")
    ap.add_argument("--livetime", type=float, help="seconds; reports events instead of fractions")
    args = ap.parse_args()

    header, nom, bins, nom_modes = read_template(args.nominal)
    T, sigma = nom[:, 0], nom[:, 1]
    kept = [nom[:, 2 + j] for j in range(nom_modes)] if args.keep_modes else []
    if nom_modes and not args.keep_modes:
        print(f"note: dropping the {nom_modes} mode column(s) of {args.nominal.name} (--keep-modes keeps them)")
    if len(kept) + len(args.alternatives) > MAX_MODES:
        raise SystemExit(f"{len(kept)} kept + {len(args.alternatives)} new modes; the layout holds {MAX_MODES}")
    if np.any(T < 0):
        raise SystemExit(f"{args.nominal}: negative template rates")

    scale = args.livetime or 1.0 / T.sum()  # events, or fractions of the nominal template
    unit = "events" if args.livetime else "of the nominal total"
    fmt = (lambda v: f"{v * scale:9.1f}") if args.livetime else (lambda v: f"{v * scale:8.1%}")

    new, notes = [], []
    for k, path in enumerate(args.alternatives):
        alt_header, alt, alt_bins, _ = read_template(path)
        if alt_bins != bins or grid_lines(alt_header) != grid_lines(header):
            raise SystemExit(f"{path}: not on the grid of {args.nominal} (bin count or bin-edge lines differ)")
        A = alt[:, 0]
        s = 1.0 if args.keep_norm else T.sum() / A.sum()
        d = s * A - T
        new.append(d)
        number = len(kept) + k + 1
        notes.append(f"# mode{number}: estimator variation, {'' if args.keep_norm else f'{s:.6g} x '}{path.name} "
                     f"- {args.nominal.name}{'' if args.keep_norm else ' (shape only, sums to 0)'}; "
                     f"+1 reproduces it. tools/make_estimator_modes.py")

        closure = np.max(np.abs(T + d - s * A)) / T.max()
        print(f"MuonTmplMode{number} = {path.name}" + ("" if args.keep_norm else f" x {s:.4f}"))
        print(f"  closure: max |T + d - alt| = {closure:.1e} of the largest bin")
        print(f"  sum d = {fmt(d.sum())}   moved (sum |d| / 2) = {fmt(np.abs(d).sum() / 2)}   {unit}")
        for theta in (-1.0, -2.0):
            shaped = T + theta * d
            neg = shaped < 0
            print(f"  theta = {theta:+.0f}: {neg.sum():4d} of {bins} bins clipped at 0, "
                  f"removing {fmt(-shaped[neg].sum())} {unit}")

    modes = kept + new
    if len(modes) > 1:
        w = np.where(T > 0, 1.0 / np.where(T > 0, T, 1.0), 0.0)
        names = [f"MuonTmplMode{j + 1}" for j in range(len(modes))]
        print("overlap (cosine in the Poisson metric):")
        for i in range(len(modes)):
            for j in range(i + 1, len(modes)):
                a, b = modes[i], modes[j]
                cos = (a * b * w).sum() / np.sqrt((a * a * w).sum() * (b * b * w).sum())
                print(f"  {names[i]} . {names[j]} = {cos:+.3f}")

    keep_header = [line for line in header
                   if not line.split()[1:2] in (["template"], ["columns:"]) and not line.startswith("# mode")]
    kept_notes = [line for line in header if line.startswith("# mode")] if args.keep_modes else []
    cols = " ".join(f"mode{j + 1}_rate" for j in range(len(modes)))
    with args.out.open("w") as out:
        out.write(f"# template bins {bins}\n# template modes {len(modes)}\n")
        out.write(f"# columns: template_rate fluctuation_rate {cols} (all per second)\n")
        for line in kept_notes + notes + keep_header:
            out.write(line + "\n")
        for b in range(bins):
            out.write(" ".join(fnum(v) for v in (T[b], sigma[b], *(m[b] for m in modes))) + "\n")
    print(f"wrote {args.out} ({bins} bins, {len(modes)} modes)")


if __name__ == "__main__":
    main()
