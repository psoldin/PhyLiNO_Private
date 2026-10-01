#!/usr/bin/env python3
"""Profiled uncertainty WITH detector systematics, for a candidate category binning.

Combines two things that live in different places:

  * the analysis-side histogram and its derivatives with respect to the free flux
    parameters, dumped by McVariance --dump-file;
  * the five SnowStorm detector gradients, regenerated from the event-level
    ensemble in exactly those bins by the recipe in snowstorm_gradients.py.

and reports sqrt([I^-1]_target,target) for the fine (category) binning and for the
2D binning it collapses to, so the gain can be read with the systematics in.

The detector gradients are relative (d ln mu / d p), so they enter as
d mu_b / d p = rel_b * mu_b. Bins where the ensemble cannot measure a gradient are
treated as having none, which is the conservative choice: it neither invents a
constraint on the systematic nor lets it absorb anything there.
"""
import argparse

import numpy as np
import pyarrow.parquet as pq

SYSTEMATICS = {
    "DOMEfficiency":     (0.9, 1.1),
    "IceAbsorption":     (0.9, 1.1),
    "IceScattering":     (0.9, 1.1),
    "HoleIceForward_p0": (-0.1, 0.5980366362473171),
    "HoleIceForward_p1": (-0.1135759700999585, 0.0),
}


def read_dump(path):
    head = {"categories": []}
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("#"):
                parts = line[1:].split()
                if parts[0] in ("energy", "zenith"):
                    head[parts[0]] = (float(parts[1]), float(parts[2]), int(parts[3]))
                elif parts[0] == "category":
                    head["categories"].append(
                        {"name": parts[1], "n": int(parts[2]), "edges": [float(v) for v in parts[3:]]})
                elif parts[0] == "params":
                    head["params"] = parts[2:]
            else:
                rows.append([float(v) for v in line.split()])
    a = np.asarray(rows)
    return head, a[:, 0], a[:, 1:].T          # mu, dmu[param][bin]


def ensemble_gradients(ensemble, head, n_bins, min_events, astro_norm, gamma):
    cats = head["categories"]
    e_lo, e_hi, n_e = head["energy"]
    z_lo, z_hi, n_z = head["zenith"]

    cols = ["energy_truncated", "zenith_MPEFit", "MCPrimaryEnergy",
            "powerlaw", "mceq_conv_H4a_SIBYLL23c", "mceq_pr_H4a_SIBYLL23c"]
    cols += [c["name"] for c in cats] + list(SYSTEMATICS)
    t = pq.read_table(ensemble, columns=cols)
    d = {c: np.asarray(t[c], dtype=np.float64) for c in cols}

    w = (astro_norm * d["powerlaw"] * (d["MCPrimaryEnergy"] / 1.0e5) ** (2.0 - gamma)
         + d["mceq_conv_H4a_SIBYLL23c"] + d["mceq_pr_H4a_SIBYLL23c"])
    with np.errstate(invalid="ignore", divide="ignore"):
        log_e = np.log10(np.where(d["energy_truncated"] > 0, d["energy_truncated"], np.nan))
    cos_z = np.cos(d["zenith_MPEFit"])

    ok = np.isfinite(log_e) & np.isfinite(cos_z) & np.isfinite(w)
    ie = np.zeros(len(ok), dtype=np.int64)
    iz = np.zeros(len(ok), dtype=np.int64)
    ie[ok] = np.floor((log_e[ok] - e_lo) / ((e_hi - e_lo) / n_e)).astype(np.int64)
    iz[ok] = np.floor((cos_z[ok] - z_lo) / ((z_hi - z_lo) / n_z)).astype(np.int64)
    ok &= (ie >= 0) & (ie < n_e) & (iz >= 0) & (iz < n_z)

    flat = ie * n_z + iz
    n_cat = 1
    for c in cats:
        v = d[c["name"]]
        flat = flat * c["n"] + np.searchsorted(np.asarray(c["edges"]), v, side="right")
        n_cat *= c["n"]
        ok &= np.isfinite(v)

    flat, w = flat[ok], w[ok]
    print(f"  ensemble: {ok.sum()} events in range, {ok.sum()/n_bins:.1f} per fine bin")

    def build(index, nb, label, parent=None, n_cat_local=1):
        """parent: relative gradients of the coarser binning this one refines.

        A fine bin the ensemble cannot measure inherits its parent's fractional
        response rather than being set to zero. Zeroing would quietly exempt that
        bin from the systematic altogether, which flatters a fine binning: 63% of
        the bins at eight categories are unmeasurable, and pretending the detector
        cannot move them is not conservative, it is wrong in the direction of the
        answer we are testing."""
        hist = np.bincount(index, weights=w, minlength=nb)
        grads, usable_frac = {}, {}
        for name, (lo, hi) in SYSTEMATICS.items():
            split, factor = 0.5 * (lo + hi), 4.0 / (hi - lo)
            p = d[name][ok]
            up, dn = p > split, p < split
            h_up = np.bincount(index[up], weights=w[up], minlength=nb)
            h_dn = np.bincount(index[dn], weights=w[dn], minlength=nb)
            n_up = np.bincount(index[up], minlength=nb)
            n_dn = np.bincount(index[dn], minlength=nb)
            good = (n_up >= min_events) & (n_dn >= min_events) & (hist > 0)
            rel = np.zeros(nb)
            if parent is not None:
                rel = np.repeat(parent[name], n_cat_local).copy()
            rel[good] = factor * (h_up[good] - h_dn[good]) / hist[good]
            grads[name] = rel
            usable_frac[name] = good.mean()
        print(f"  {label}: usable bins per systematic "
              + ", ".join(f"{k}={v:.0%}" for k, v in usable_frac.items()))
        return grads

    coarse = build(flat // n_cat, n_bins // n_cat, "2D ")
    fine   = build(flat, n_bins, "fine", parent=coarse, n_cat_local=n_cat)
    # The other extreme: assume the detector response has no category dependence
    # at all, so every fine bin simply carries its parent's. Together the two
    # bracket what the ensemble can and cannot tell us.
    inherited = {k: np.repeat(v, n_cat) for k, v in coarse.items()}
    return fine, coarse, inherited, n_cat


def profiled(mu, dmu, prior, target):
    keep = mu > 0
    n = len(dmu)
    fisher = np.zeros((n, n))
    for a in range(n):
        for b in range(a, n):
            fisher[a, b] = fisher[b, a] = np.sum(dmu[a][keep] * dmu[b][keep] / mu[keep])
    for a in range(n):
        if prior[a] > 0:
            fisher[a, a] += 1.0 / prior[a] ** 2
    cond = np.linalg.cond(fisher)
    return np.sqrt(np.linalg.inv(fisher)[target, target]), cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--target", default="SpectralIndex")
    ap.add_argument("--min-events", type=int, default=20)
    ap.add_argument("--astro-norm", type=float, default=1.8)
    ap.add_argument("--gamma", type=float, default=2.52)
    args = ap.parse_args()

    head, mu, dmu = read_dump(args.dump)
    names = list(head["params"])
    target = names.index(args.target)
    n_bins = len(mu)

    # Prior widths matching the config: only the Barr parameters and CRGrad are
    # constrained; the five detector parameters are not.
    prior_flux = {"BarrH": 0.15, "BarrW": 0.4, "BarrY": 0.3, "BarrZ": 0.12, "CRGrad": 1.0}
    prior = [prior_flux.get(n, 0.0) for n in names]

    fine_g, coarse_g, inherited_g, n_cat = ensemble_gradients(
        args.ensemble, head, n_bins, args.min_events, args.astro_norm, args.gamma)

    mu_c = mu.reshape(-1, n_cat).sum(axis=1)
    dmu_c = [d.reshape(-1, n_cat).sum(axis=1) for d in dmu]

    rows = []
    for label, m, dm, grads in (("2D (baseline)", mu_c, dmu_c, coarse_g),
                                ("2D x cat, measured", mu, list(dmu), fine_g),
                                ("2D x cat, inherited", mu, list(dmu), inherited_g)):
        flux_only, cond_a = profiled(m, dm, prior, target)
        dm_sys = list(dm) + [grads[s] * m for s in SYSTEMATICS]
        with_sys, cond_b = profiled(m, dm_sys, prior + [0.0] * len(SYSTEMATICS), target)
        rows.append((label, len(m), flux_only, with_sys, cond_a, cond_b))

    print(f"\n  profiled sigma on {args.target}\n")
    print(f"  {'binning':<20} {'bins':>8} {'flux only':>11} {'+ detector':>11} {'inflation':>10} {'cond(I)':>11}")
    for label, nb, a, b, ca, cb in rows:
        print(f"  {label:<20} {nb:>8} {a:>11.5f} {b:>11.5f} {b/a:>9.2f}x {cb:>11.3g}")
    print("\n  cond(I) is the Fisher matrix condition number: a tracks-only fit with ten free\n"
          "  flux parameters is close to degenerate, which is itself why the category axes\n"
          "  buy so much -- they break that degeneracy.")
    for i, tag in ((1, "measured gradients "), (2, "inherited gradients")):
        print(f"\n  category axes, {tag}:  flux only {100*(rows[i][2]/rows[0][2]-1):+.2f}%,"
              f"  with detector {100*(rows[i][3]/rows[0][3]-1):+.2f}%")


if __name__ == "__main__":
    main()
