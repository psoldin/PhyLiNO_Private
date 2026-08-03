#!/usr/bin/env python
"""Evaluate NNMFit's likelihood at a fully fixed parameter point and print it as
JSON. Run with NNMFit's interpreter.

Why not `run_fit.py --fix ... -o out.pickle`: with every parameter fixed,
`NNMFitter.do_fit` takes its evaluate-only branch and sets
`minimizer_info = "evaluated_only"` (a str), but `FitResult`'s pydantic model
declares that field a dict -- so run_fit.py computes the value and then dies in
`FitResult(**todump)` before writing anything. This driver calls the same
`do_fit` and prints the result instead of pickling it.

Usage:
  nnmfit_eval_llh.py CONFIG.yaml --fix NAME VALUE [--fix NAME VALUE ...]
                     [--json OUT.json]
"""
import argparse
import json
import sys

from NNMFit import AnalysisConfig
from NNMFit import NNMFitter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--fix", action="append", nargs=2, default=[],
                        metavar=("NAME", "VALUE"))
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    config_hdl = AnalysisConfig.from_yaml_files(args.config)
    fitter = NNMFitter(config_hdl)

    fixed = {name: float(value) for name, value in args.fix}

    # randomize_param_seeds is irrelevant with everything fixed, but leaving it
    # on would still perturb the printed seeds; turn it off so the reported
    # point is exactly what was asked for.
    llh, res_dict, minimizer_info = fitter.do_fit(
        fixed_pars=fixed, randomize_param_seeds=False
    )

    if minimizer_info != "evaluated_only":
        raise SystemExit(
            f"expected an evaluation, got minimizer_info={minimizer_info!r} -- "
            f"some parameter was left free"
        )

    out = {
        "llh_value": float(llh),
        "point": {k: float(v) for k, v in res_dict.items()},
        "fixed": fixed,
        "analysis_type": config_hdl.analysis_config["analysis_type"],
        "llh": config_hdl.analysis_config["llh"],
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.json_path:
        with open(args.json_path, "w") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
