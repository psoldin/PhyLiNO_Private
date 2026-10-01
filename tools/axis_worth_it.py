"""Is a candidate binning axis worth scanning? One Asimov fit, no scan.

    python3 axis_worth_it.py BestFit.pb.gz [emin_log10=5.0]

Within each energy bin above emin, compares how astro and atmospheric spread
over the category cells. Comparing within the bin is what isolates the axis:
aggregating over energy first would just re-measure the energy spectrum.
Also reports the Fisher gain sum(s^2/mu) of the split over the merged binning.
"""
import sys
import numpy as np
sys.path.insert(0, "tools")
from ic_result import load

res = load(sys.argv[1])
emin = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

for s in res["samples"]:
    axes = s["axes"]
    cat = [i for i, a in enumerate(axes) if a["kind"] == "Category"]
    if not cat:
        continue
    shape = [int(a["n_bins"]) for a in axes]
    comp = {c["name"]: np.array(c["bins"]).reshape(shape)
            for c in s["components_breakdown"] if c["bins"]}
    if "astro" not in comp:
        continue

    ie = next(i for i, a in enumerate(axes) if a["kind"] == "Log10Energy")
    e = axes[ie]
    edges = np.linspace(e["low"], e["high"], int(e["n_bins"]) + 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])

    mu = np.array(s["prediction"]).reshape(shape)
    astro = comp["astro"]
    atm = mu - astro
    # collapse everything except energy and the category axes
    keep = (ie, *cat)
    drop = tuple(i for i in range(len(shape)) if i not in keep)
    A = astro.sum(axis=drop).reshape(shape[ie], -1)
    B = atm.sum(axis=drop).reshape(shape[ie], -1)

    def info(sig, tot):
        m = tot > 0
        return np.sum(sig[m] ** 2 / tot[m])

    gain = info(astro, mu) / info(astro.sum(axis=tuple(cat)), mu.sum(axis=tuple(cat)))

    print(f"{s['name']}  cells={A.shape[1]}  (per-energy-bin, log10E > {emin})")
    print(f"  {'log10E':>7}  {'astro cell fracs':<28} {'atm cell fracs':<28} max|diff|")
    worst = 0.0
    for i in range(shape[ie]):
        if ctr[i] <= emin or A[i].sum() <= 0 or B[i].sum() <= 0:
            continue
        fa, fb = A[i] / A[i].sum(), B[i] / B[i].sum()
        d = np.abs(fa - fb).max()
        worst = max(worst, d)
        print(f"  {ctr[i]:7.2f}  {str(np.round(fa,3)):<28} {str(np.round(fb,3)):<28} {d:.3f}")
    print(f"\n  worst per-bin separation = {worst:.3f}")
    print(f"  Fisher gain sum(s^2/mu) split/merged = {gain:.3f}"
          f"  ->  sigma shrinks by {100*(1-1/np.sqrt(gain)):.1f}%")
