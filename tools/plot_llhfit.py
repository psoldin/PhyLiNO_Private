"""Marginalized energy / cos(zenith) spectra from an LLHFit result.

Reads either output format of ``write_ice_cube_results`` -- the pretty JSON or
the gzipped protobuf (decoded via ``ic_result.py``) -- and plots data vs. the
fitted prediction, marginalized onto one binning axis at a time.

Library:

    from plot_llhfit import load_result, marginalize, plot_sample

    res = load_result("BestFit.json")
    fig = plot_sample(res["samples"][0])            # every axis of the sample
    m   = marginalize(res["samples"][0], "CosZenith")  # edges + summed arrays

Command line:

    python3 plot_llhfit.py BestFit.pb.gz                 # window per sample
    python3 plot_llhfit.py BestFit.json -o spectra       # spectra_<sample>.pdf
    python3 plot_llhfit.py BestFit.json --sample cscd_cascade --axis CosZenith
    python3 plot_llhfit.py BestFit.json --e2 --linear      # E^2-weighted, linear y
    python3 plot_llhfit.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

__all__ = ["load_result", "marginalize", "bin_weights", "default_components",
           "plot_spectrum", "plot_sample"]

# atmospheric_conv + atmospheric_prompt re-sum the "atmospheric"/"atmospheric_veto"
# entry, so drawing all of them double-counts the atmospheric flux.
_SPLIT_ATM = ("atmospheric_conv", "atmospheric_prompt")
_SUM_ATM = ("atmospheric", "atmospheric_veto")

_AXIS_LABEL = {
    "Log10Energy": r"$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV})$",
    "CosZenith": r"$\cos\theta_{\mathrm{reco}}$",
    "Ra": "right ascension",
    "Category": "category bin",
}


def _load_pb(path):
    try:
        from ic_result import load
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ic_result import load
    return load(path)


def load_result(path):
    """Load a fit result from ``*.json``, ``*.pb`` or ``*.pb.gz``.

    Returns the JSON-ish layout with both writers' key spellings normalised:
    ``n_bins``, ``components`` (name -> per-bin array) and ``component_totals``.
    """
    raw = Path(path).read_bytes()
    # The writer only ever gzips the protobuf, so the magic settles the format.
    res = _load_pb(path) if raw[:2] == b"\x1f\x8b" or raw.lstrip()[:1] != b"{" else json.loads(raw)

    for sample in res["samples"]:
        for axis in sample["axes"]:
            axis["n_bins"] = int(axis.get("n_bins", axis.get("nBins")))
        if "componentBins" in sample:  # json writer
            sample["components"] = sample.pop("componentBins")
            sample["component_totals"] = sample.pop("componentTotals")
        else:  # protobuf writer
            breakdown = sample.pop("components_breakdown")
            sample["components"] = {c["name"]: c["bins"] for c in breakdown}
            sample["component_totals"] = {c["name"]: c["total"] for c in breakdown}
        sample.setdefault("dataTotal", sample.get("data_total"))
        sample.setdefault("predTotal", sample.get("pred_total"))
    return res


def _axis_index(sample, axis):
    if isinstance(axis, int):
        return axis
    for i, a in enumerate(sample["axes"]):
        if a["kind"] == axis:
            return i
    raise KeyError("sample %r has no %r axis" % (sample["name"], axis))


def bin_weights(sample, weight=None):
    """Per-bin flux weight, broadcastable over the sample's binning.

    ``None`` weights every bin by one; ``"E2"`` by the squared energy of its
    log-centre in GeV, so a marginalized cos(zenith) spectrum is weighted bin
    by bin before the energy axis is summed away.
    """
    if weight is None:
        return np.ones(1)
    if weight != "E2":
        raise ValueError("unknown weight %r (expected None or 'E2')" % (weight,))
    i = _axis_index(sample, "Log10Energy")
    a = sample["axes"][i]
    edges = np.linspace(a["low"], a["high"], a["n_bins"] + 1)
    energy = 10.0 ** (0.5 * (edges[1:] + edges[:-1]))
    return (energy ** 2).reshape([-1 if j == i else 1 for j in range(len(sample["axes"]))])


def marginalize(sample, axis=0, components=None, weight=None):
    """Sum data, prediction and components over every axis but ``axis``.

    ``axis`` is an axis kind ("Log10Energy", ...) or an index into the sample's
    axes, ``weight`` is passed to :func:`bin_weights`. Returns
    ``{"edges", "centers", "data", "data_err", "prediction", "components"}``.
    """
    i = _axis_index(sample, axis)
    a = sample["axes"][i]
    shape = [ax["n_bins"] for ax in sample["axes"]]
    drop = tuple(j for j in range(len(shape)) if j != i)
    w = bin_weights(sample, weight)

    def collapse(flat):
        return (np.asarray(flat, float).reshape(shape) * w).sum(axis=drop)

    # Poisson per bin, so the weighted sum carries sum(w^2 * N), not sqrt(N).
    variance = (np.asarray(sample["data"], float).reshape(shape) * w ** 2).sum(axis=drop)

    # A Category axis carries its real cut values elsewhere; its (low, high)
    # is only the dummy range the binner needed, so index the cells instead.
    edges = (np.arange(a["n_bins"] + 1) - 0.5 if a["kind"] == "Category"
             else np.linspace(a["low"], a["high"], a["n_bins"] + 1))
    keys = components if components is not None else default_components(sample)
    return {
        "axis": a,
        "edges": edges,
        "centers": 0.5 * (edges[1:] + edges[:-1]),
        "data": collapse(sample["data"]),
        "data_err": np.sqrt(variance),
        "prediction": collapse(sample["prediction"]),
        "components": {k: collapse(sample["components"][k]) for k in keys},
    }


def default_components(sample, split_atm=False):
    """Component keys that add up to the prediction exactly once."""
    drop = _SUM_ATM if split_atm else _SPLIT_ATM
    return [k for k, v in sample["components"].items() if k not in drop and len(v)]


def plot_spectrum(sample, axis=0, ax=None, rax=None, components=True, legend=True,
                  weight=None, logy=True):
    """Draw one marginalized spectrum; ``rax`` gets the data/prediction ratio."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    keys = components if isinstance(components, (list, tuple)) else (None if components else [])
    m = marginalize(sample, axis, keys, weight)
    e, c = m["edges"], m["centers"]

    ax.stairs(m["prediction"], e, color="k", lw=1.6, label="prediction")
    for name, bins in m["components"].items():
        ax.stairs(bins, e, lw=1.1, alpha=0.85, label=name)
    ax.errorbar(c, m["data"], yerr=m["data_err"], fmt="o", ms=3.5,
                color="k", capsize=0, label="data")

    if logy:
        ax.set_yscale("log")
        # Components that fall off a cliff (or go negative) would otherwise
        # stretch the log axis over a dozen decades, so ignore anything more
        # than six decades below the peak when framing it.
        shown = np.concatenate([m["prediction"], *m["components"].values()])
        shown = shown[shown > shown.max() * 1e-6]
        if shown.size:
            ax.set_ylim(bottom=0.5 * shown.min())
    ax.set_ylabel(r"$E^2 \times$ events [GeV$^2$]" if weight == "E2" else "events")
    ax.set_xlim(e[0], e[-1])
    if legend:
        ax.legend(fontsize=7, ncol=2)

    if rax is not None:
        pred = np.where(m["prediction"] > 0, m["prediction"], np.nan)
        rax.errorbar(c, m["data"] / pred, yerr=m["data_err"] / pred,
                     fmt="o", ms=3.5, color="k", capsize=0)
        rax.axhline(1.0, color="k", lw=0.8, ls="--")
        rax.set_ylabel("data/pred", fontsize=8)
        rax.set_xlim(e[0], e[-1])
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    label = _AXIS_LABEL.get(m["axis"]["kind"], m["axis"]["kind"])
    (rax or ax).set_xlabel(label)
    return ax


def plot_sample(sample, axes=None, ratio=True, figsize=None, components=True,
                weight=None, logy=True):
    """One figure for a sample: a column per binning axis, ratio panel below."""
    import matplotlib.pyplot as plt

    idx = [_axis_index(sample, a) for a in axes] if axes else range(len(sample["axes"]))
    idx = [i for i in idx if sample["axes"][i]["n_bins"] > 1] or list(idx)

    fig, grid = plt.subplots(
        2 if ratio else 1, len(idx), squeeze=False, sharex="col",
        figsize=figsize or (4.5 * len(idx), 5.0 if ratio else 3.8),
        gridspec_kw={"height_ratios": [3, 1]} if ratio else None,
    )
    for col, i in enumerate(idx):
        plot_spectrum(sample, i, ax=grid[0][col], components=components,
                      weight=weight, logy=logy,
                      rax=grid[-1][col] if ratio else None, legend=col == 0)
    fig.suptitle(sample["name"])
    fig.tight_layout()
    return fig


def _selftest():
    shape, edges = (3, 4), np.linspace(0, 1, 5)
    astro = np.arange(12, dtype=float).reshape(shape)
    atm = np.ones(shape) * 2.0
    sample = {
        "name": "t",
        "axes": [{"kind": "Log10Energy", "low": 2.0, "high": 5.0, "n_bins": 3},
                 {"kind": "CosZenith", "low": 0.0, "high": 1.0, "n_bins": 4}],
        "data": list((astro + atm).ravel()),
        "prediction": list((astro + atm).ravel()),
        "components": {"astro": list(astro.ravel()), "atmospheric": list(atm.ravel()),
                       "atmospheric_conv": list(atm.ravel()),
                       "atmospheric_prompt": list(np.zeros(12))},
    }
    assert default_components(sample) == ["astro", "atmospheric"]
    assert default_components(sample, split_atm=True) == ["astro", "atmospheric_conv", "atmospheric_prompt"]

    m = marginalize(sample, "CosZenith")
    assert np.allclose(m["edges"], edges)
    assert np.allclose(m["components"]["astro"], astro.sum(axis=0))
    assert np.allclose(m["prediction"], (astro + atm).sum(axis=0))
    assert np.allclose(m["data_err"], np.sqrt((astro + atm).sum(axis=0)))
    e = marginalize(sample, 0)
    assert np.allclose(e["prediction"], (astro + atm).sum(axis=1))
    assert e["prediction"].sum() == m["prediction"].sum() == (astro + atm).sum()

    # E^2 weighting: energy bins [2,3), [3,4), [4,5) in log10 -> log-centres.
    w = (10.0 ** np.array([2.5, 3.5, 4.5])) ** 2
    ew = marginalize(sample, "Log10Energy", weight="E2")
    assert np.allclose(ew["prediction"], (astro + atm).sum(axis=1) * w)
    assert np.allclose(ew["data_err"], np.sqrt((astro + atm).sum(axis=1)) * w)
    zw = marginalize(sample, "CosZenith", weight="E2")
    assert np.allclose(zw["components"]["astro"], (astro * w[:, None]).sum(axis=0))
    assert np.allclose(zw["data_err"], np.sqrt(((astro + atm) * w[:, None] ** 2).sum(axis=0)))
    print("selftest ok")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("result", nargs="?", help="*.json, *.pb or *.pb.gz fit result")
    p.add_argument("-o", "--output", help="save to <output>_<sample>.pdf instead of showing")
    p.add_argument("-s", "--sample", action="append", help="sample name (repeatable, default all)")
    p.add_argument("-a", "--axis", action="append", help="axis kind (repeatable, default all)")
    p.add_argument("--no-ratio", action="store_true")
    p.add_argument("--e2", action="store_true", help="weight every bin by its energy squared")
    p.add_argument("--linear", action="store_true", help="linear instead of log y-axis")
    p.add_argument("--split-atm", action="store_true",
                   help="draw the conv/prompt halves instead of the summed atmospheric flux")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.result:
        p.error("need a result file (or --selftest)")

    import matplotlib
    if args.output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = load_result(args.result)
    for sample in res["samples"]:
        if args.sample and sample["name"] not in args.sample:
            continue
        fig = plot_sample(sample, axes=args.axis, ratio=not args.no_ratio,
                          weight="E2" if args.e2 else None, logy=not args.linear,
                          components=default_components(sample, args.split_atm))
        if args.output:
            name = "%s_%s.pdf" % (args.output, sample["name"])
            fig.savefig(name)
            print(name)
    if not args.output:
        plt.show()


if __name__ == "__main__":
    sys.exit(main())
