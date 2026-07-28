#!/usr/bin/env python3
"""Repeat a 2D NNMFit muon-template pickle over an RA axis, for Binning_2D_to_3D configs.

NNMFit's TemplateFlux is a histogram component: it returns its template as mu without
ever calling the binning, so a 3D config needs a template that is already 3D. The
correct 3D template is the 2D one spread uniformly over RA, which is exactly what
Binning_2D_to_3D does to every non-histogram component: repeat(t, n_ra) / n_ra.

NNMFit's Template pydantic model (NNMFit/core/templates.py) also validates the
template's flat length against energy_bins/zenith_bins/ra_bins: if ra_bins is absent
it checks the 2D product only, and rejects the (now longer) 3D array. So this script
also writes an ra_bins field -- without it, the model validates the 3D array against
the 2D expected shape and fails.

Usage:
  make_3d_muon_templates.py IN.pickle OUT.pickle --ra-bins N [--ra-upper X] [--det-conf NAME ...]

With no --det-conf, every detector-config key in a multi-dataset pickle is converted
(and a single-dataset pickle is converted as-is).
"""
import argparse
import pickle

import numpy as np

TEMPLATE_KEYS = ("template", "template_2d", "template_fluctuation")


def repeat_entry(entry, ra_bins, ra_upper):
    """Return a copy of one detector-config entry with its arrays spread over RA."""
    out = dict(entry)
    for key in TEMPLATE_KEYS:
        value = entry.get(key)
        if value is None:
            continue
        flat = np.asarray(value, dtype=float).reshape(-1)
        out[key] = np.repeat(flat, ra_bins) / ra_bins
    # template_2d loses its 2D shape once RA is folded in; drop it so no consumer
    # reads a stale shape. "template" is the key NNMFit prefers when both exist.
    if "template" in out and "template_2d" in out:
        del out["template_2d"]
    elif "template_2d" in out:
        out["template"] = out.pop("template_2d")
    # Without this the Template model validates the 3D array against the 2D expected
    # shape and rejects it. Edges match the config's reco_ra_binning literally
    # (e.g. "(0,6.28319,19,lin)" -> 19 edges, 18 bins).
    out["ra_bins"] = np.linspace(0.0, ra_upper, ra_bins + 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("in_path")
    p.add_argument("out_path")
    p.add_argument("--ra-bins", type=int, required=True)
    p.add_argument(
        "--ra-upper",
        type=float,
        default=6.28319,
        help="upper RA edge, matching the config's reco_ra_binning (default 6.28319)",
    )
    p.add_argument("--det-conf", action="append", default=None)
    args = p.parse_args()

    with open(args.in_path, "rb") as f:
        obj = pickle.load(f)

    is_multi = isinstance(obj, dict) and any(
        isinstance(v, dict) and any(k in v for k in TEMPLATE_KEYS) for v in obj.values()
    )

    if is_multi:
        keys = args.det_conf or list(obj)
        out = dict(obj)
        for key in keys:
            if key not in obj:
                raise SystemExit(f"detector config '{key}' not in {args.in_path}; have {sorted(obj)}")
            out[key] = repeat_entry(obj[key], args.ra_bins, args.ra_upper)
            n = np.asarray(out[key]["template"]).size
            print(f"{key}: {n} bins after repeating over {args.ra_bins} RA bins")
    else:
        out = repeat_entry(obj, args.ra_bins, args.ra_upper)
        print(f"single dataset: {np.asarray(out['template']).size} bins")

    with open(args.out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
