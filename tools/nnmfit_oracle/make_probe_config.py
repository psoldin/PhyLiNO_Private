#!/usr/bin/env python3
"""Build a PhyLiNO IceCube config that evaluates the model at a fixed parameter
point instead of fitting.

LLHFit always calls Fit::minimize(); the only way to get its Asimov prediction
at an exact point (matching one of NNMFit's dumps, see
tools/nnmfit_oracle/README.md) is to mark every parameter Fixed so Migrad has
nothing to move. This also pins Likelihood/Backend/UseData so a stray edit to
the base config (e.g. mid-session experiments) can't silently change what gets
evaluated.

Usage:
  make_probe_config.py BASE_CONFIG OUT_CONFIG [--set NAME=VALUE ...]

Omitted --set NAME=VALUE pairs keep the base config's StartValue (used for the
"defaults" point). Values from /tmp/nnmfit_fitted_nosys.yaml go on the command
line for the "fitted_nosys" point.
"""
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base_config")
    p.add_argument("out_config")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    args = p.parse_args()

    overrides = {}
    for item in args.set:
        name, value = item.split("=", 1)
        overrides[name] = float(value)

    with open(args.base_config) as f:
        cfg = json.load(f)

    cfg["IceCube"]["Likelihood"] = "SAY"
    cfg["IceCube"]["Backend"] = "cpu"
    cfg["IceCube"]["UseData"] = False

    seen = set()
    for param in cfg["Parameter"]:
        if param["Name"] in overrides:
            param["StartValue"] = overrides[param["Name"]]
            seen.add(param["Name"])
        param["Fixed"] = True

    missing = set(overrides) - seen
    if missing:
        raise SystemExit(f"--set named parameters not in {args.base_config}: {sorted(missing)}")

    with open(args.out_config, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote {args.out_config}: {len(cfg['Parameter'])} parameters, all fixed, "
          f"{len(overrides)} overridden")


if __name__ == "__main__":
    main()
