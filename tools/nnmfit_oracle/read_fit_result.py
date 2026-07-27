#!/usr/bin/env python3
"""Print an NNMFit Output.pickle fit result as JSON, for comparison with PhyLiNO.

Maps NNMFit's parameter names onto PhyLiNO's config "Name" entries (params::ic) so
the two fits can be diffed directly.

Usage: read_fit_result.py Output.pickle [--json OUT.json]
"""
import argparse
import json
import pickle

# NNMFit parameter name -> PhyLiNO config "Name"
NAME_MAP = {
    "astro_norm": "AstroNorm",
    "gamma_astro": "SpectralIndex",
    "conv_norm": "ConvNorm",
    "prompt_norm": "PromptNorm",
    "barr_h": "BarrH",
    "barr_w": "BarrW",
    "barr_y": "BarrY",
    "barr_z": "BarrZ",
    "CR_grad": "CRGrad",
    "delta_gamma": "DeltaGamma",
    "muon_norm": "MuonNorm",
    "muongun_norm": "MuonGunNorm",
    "effective_veto": "VetoThreshold",
    "dom_eff": "DOMEff",
    "ice_abs": "IceAbs",
    "ice_scat": "IceScat",
    "ice_holep0": "HoleIceP0",
    "ice_holep1": "HoleIceP1",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle_path")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    with open(args.pickle_path, "rb") as handle:
        result = pickle.load(handle)

    unmapped = sorted(set(result["res_dict"]) - set(NAME_MAP))
    if unmapped:
        raise SystemExit(f"unmapped NNMFit parameters: {unmapped} (extend NAME_MAP)")

    out = {
        "llh_value": result["llh_value"],
        "minimizer": result["minimizer_info"],
        "fitted": {NAME_MAP[k]: v for k, v in result["res_dict"].items()},
        "seeds": {NAME_MAP[k]: v for k, v in result["minimizer_seeds"].items()},
        "analysis_type": result["settings"]["analysis"]["analysis_type"],
        "llh": result["settings"]["analysis"]["llh"],
        "samples": sorted(result["settings"]["datasets"]),
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.json_path:
        with open(args.json_path, "w") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
