#!/usr/bin/env python
"""Copy NNMFit's per-parameter ranges into a PhyLiNO config as LowerBound /
UpperBound. Run with NNMFit's interpreter.

NNMFit bounds every fit variable (`range:` in the component and systematics
config, read back through AnalysisConfig.get_params_and_bounds). PhyLiNO's
bounds are optional and off by default, so a config only gets them if they are
written in -- which is what this does, so a fit comparison is not confounded by
one side being bounded and the other not.

A `None` on either side means unbounded there and is simply not written; PhyLiNO
supports one-sided bounds (SetLowerLimitedVariable / SetUpperLimitedVariable).

Usage:
  apply_nnmfit_bounds.py PHYLINO_CONFIG NNMFIT_CONFIG OUT_CONFIG
"""
import argparse
import json
import sys

from NNMFit import AnalysisConfig

# PhyLiNO config "Name" -> NNMFit fit-variable name. Kept in step with
# compare_llh_value.py's NAME_MAP.
NAME_MAP = {
    "AstroNorm": "astro_norm",
    "SpectralIndex": "gamma_astro",
    "ConvNorm": "conv_norm",
    "PromptNorm": "prompt_norm",
    "BarrH": "barr_h",
    "BarrW": "barr_w",
    "BarrY": "barr_y",
    "BarrZ": "barr_z",
    "CRGrad": "CR_grad",
    "DeltaGamma": "delta_gamma",
    "MuonNorm": "muon_norm",
    "MuonGunNorm": "muongun_norm",
    "VetoThreshold": "effective_veto",
    "DOMEff": "dom_eff",
    "IceAbs": "ice_abs",
    "IceScat": "ice_scat",
    "HoleIceP0": "ice_holep0",
    "HoleIceP1": "ice_holep1",
    "AstroGamma1": "gamma_1",
    "AstroGamma2": "gamma_2",
    "AstroEBreak": "e_break",
    "GalacticNorm0": "cringefits_norm",
    "GalacticNorm1": "cringefits_norm",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phylino_config")
    parser.add_argument("nnmfit_config")
    parser.add_argument("out_config")
    parser.add_argument(
        "--also-from",
        action="append",
        default=[],
        metavar="NNMFIT_CONFIG",
        help="additional NNMFit config to take ranges from, for parameters the "
        "primary one does not define (a single-power-law config has no "
        "gamma_1/gamma_2/e_break, a 2D one no cringefits_norm). The primary "
        "config always wins where both define a parameter -- astro_norm is "
        "[0, None] in the 2D config but [0, 5] in the 3D one.",
    )
    args = parser.parse_args()

    _, bounds = AnalysisConfig.from_yaml_files(args.nnmfit_config).get_params_and_bounds()
    for extra in args.also_from:
        _, extra_bounds = AnalysisConfig.from_yaml_files(extra).get_params_and_bounds()
        for name, value in extra_bounds.items():
            bounds.setdefault(name, value)

    cfg = json.load(open(args.phylino_config))

    applied, skipped = 0, []
    for param in cfg["Parameter"]:
        nnmfit_name = NAME_MAP.get(param["Name"])
        if nnmfit_name is None or nnmfit_name not in bounds:
            skipped.append(param["Name"])
            continue
        lower, upper = bounds[nnmfit_name]
        if lower is not None:
            param["LowerBound"] = float(lower)
        if upper is not None:
            param["UpperBound"] = float(upper)

        # PhyLiNO rejects a StartValue outside its own bounds rather than
        # clamping it silently; report that here instead of at fit time.
        value = float(param["StartValue"])
        if (lower is not None and value < float(lower)) or (
            upper is not None and value > float(upper)
        ):
            raise SystemExit(
                f"{param['Name']}: StartValue {value} lies outside NNMFit's range "
                f"[{lower}, {upper}] -- fix the config before comparing"
            )
        applied += 1

    json.dump(cfg, open(args.out_config, "w"), indent=2)
    print(f"wrote {args.out_config}: bounded {applied} parameters")
    if skipped:
        print(f"no NNMFit range (left unbounded): {sorted(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
