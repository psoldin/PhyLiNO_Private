#!/usr/bin/env python3
"""Compare the *value* of the likelihood between PhyLiNO and NNMFit at one
fixed parameter point.

Both codes are asked to evaluate, not to fit: PhyLiNO via a probe config with
every parameter Fixed (make_probe_config.py), NNMFit via nnmfit_eval_llh.py,
which fixes every parameter so do_fit() takes its "all parameters fixed ->
evaluate" branch. The two objectives differ by a known factor only:

    PhyLiNO  = -2 * sum(log L) + chi2            (Minuit2 ErrorDef = 1)
    NNMFit   =     -sum(log L) + chi2 / 2        (iminuit errordef = LIKELIHOOD)

so the check is  PhyLiNO_LLH == 2 * NNMFit_llh_value  to floating-point
tolerance. Since PhyLiNO stopped subtracting a baseline offset, this is an
absolute comparison, not a difference-of-two-points one.

Usage:
  compare_llh_value.py PHYLINO_CONFIG NNMFIT_CONFIG [--set NAME=VALUE ...]
                       [--workdir DIR] [--likelihood SAY|Poisson]
                       [--use-data] [--tolerance 1e-9]

NAME is the PhyLiNO parameter name; it is translated to NNMFit's name via
NAME_MAP below, so both codes are pinned to the same point.

The comparison is only meaningful on the same data: pass --use-data to compare
on the real measured histograms (both sides then use their configured data),
which is the only mode in which the two codes' data histograms are guaranteed
to be identical. Without it, each code builds its own Asimov set from its own
prediction at the probe point, which makes the Poisson objective trivially 0 on
both sides -- useful as a smoke test, not as a gate.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

NNMFIT_PYTHON = "/Users/soldin/Projects/IceCube/NNMFit/.venv/bin/python"
NNMFIT_EVAL_LLH = os.path.join(HERE, "nnmfit_eval_llh.py")
LLHFIT = os.path.join(ROOT, "build", "programs", "LLHFit", "LLHFit")

# PhyLiNO config "Name" -> NNMFit parameter name (the inverse of
# read_fit_result.py's NAME_MAP, plus the parameters only the 3D/BPL configs
# have). Both PhyLiNO galactic norms map onto NNMFit's single shared
# cringefits_norm, so the script requires them to hold the same value.
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


def run(cmd, **kwargs):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def build_probe_config(base_config, out_config, overrides, likelihood, use_data, backend="cpu"):
    """Every parameter Fixed, values from the base config unless overridden.

    The base config's values stay in "AsimovValue", so the Asimov set is
    generated at the unmodified point while the evaluation happens at the
    overridden one. That is the only way to give the Poisson term something to
    measure: evaluated at its own truth it is exactly 0 by construction.
    """
    with open(base_config) as f:
        cfg = json.load(f)

    cfg["IceCube"]["Likelihood"] = likelihood
    cfg["IceCube"]["Backend"] = backend
    cfg["IceCube"]["UseData"] = bool(use_data)

    names = {p["Name"] for p in cfg["Parameter"]}
    missing = set(overrides) - names
    if missing:
        raise SystemExit(f"--set named parameters not in {base_config}: {sorted(missing)}")

    point = {}
    asimov_point = {}
    for param in cfg["Parameter"]:
        asimov_point[param["Name"]] = float(param.get("AsimovValue", param["StartValue"]))
        param["AsimovValue"] = asimov_point[param["Name"]]
        if param["Name"] in overrides:
            param["StartValue"] = overrides[param["Name"]]
        param["Fixed"] = True
        point[param["Name"]] = float(param["StartValue"])

    with open(out_config, "w") as f:
        json.dump(cfg, f, indent=2)
    return point, asimov_point


def phylino_llh(probe_config, workdir):
    """LLHFit at a fully fixed point: Migrad has nothing to move, so its
    reported minimum is the likelihood evaluated at that point.

    --fitOnly is required: without it LLHFit runs the 2D scan instead and never
    writes Output.json."""
    run([LLHFIT, "-c", probe_config, "--fitOnly", "--silent"], cwd=ROOT)
    output = os.path.join(workdir, "phylino_output.json")
    os.replace(os.path.join(ROOT, "Output.json"), output)
    with open(output) as f:
        return float(json.load(f)["LLH"])


def nnmfit_parameter_names(nnmfit_config):
    """The fit parameters NNMFit's config actually defines.

    PhyLiNO's parameter enum is fixed-size (params::ic), so a config carries
    entries its model does not use -- AstroGamma1/2, AstroEBreak and the
    galactic norms are present even in a single-power-law, no-galactic setup.
    Passing those to run_fit.py --fix aborts with "not fit parameters and
    cannot be set", so ask NNMFit which names exist and skip the rest.
    """
    script = (
        "import json;"
        "from NNMFit import AnalysisConfig;"
        f"cfg = AnalysisConfig.from_yaml_files({nnmfit_config!r});"
        "seeds, _ = cfg.get_params_and_bounds();"
        "print(json.dumps(sorted(seeds)))"
    )
    out = subprocess.run([NNMFIT_PYTHON, "-c", script], check=True,
                         capture_output=True, text=True)
    return set(json.loads(out.stdout.strip().splitlines()[-1]))


def to_nnmfit_names(point, known, what):
    """Translate a PhyLiNO parameter point into NNMFit's names.

    Entries whose NNMFit counterpart is not a fit parameter of this config are
    dropped: PhyLiNO's parameter enum is fixed-size (params::ic), so a config
    carries names its model does not use -- AstroGamma1/2, AstroEBreak and the
    galactic norms in a single-power-law, no-galactic setup.
    """
    unmapped = sorted(set(point) - set(NAME_MAP))
    if unmapped:
        raise SystemExit(f"no NNMFit name known for: {unmapped} (extend NAME_MAP)")

    translated = {}
    skipped = []
    for phylino_name, value in point.items():
        nnmfit_name = NAME_MAP[phylino_name]
        if nnmfit_name not in known:
            skipped.append(f"{phylino_name} ({nnmfit_name})")
            continue
        if nnmfit_name in translated and translated[nnmfit_name] != value:
            raise SystemExit(
                f"{phylino_name} and another parameter both map to NNMFit's "
                f"'{nnmfit_name}' but hold different values "
                f"({value} vs {translated[nnmfit_name]}); NNMFit has only one "
                f"such parameter, so the point is not representable there"
            )
        translated[nnmfit_name] = value

    if skipped:
        print(f"{what}: not fit parameters of this NNMFit config, skipped: {sorted(skipped)}")

    return translated


def nnmfit_llh(nnmfit_config, point, asimov_point, workdir):
    """Evaluate NNMFit's likelihood with every parameter fixed, via
    nnmfit_eval_llh.py (which asserts do_fit took its evaluate-only branch).

    asimov_point is passed as --inject, i.e. the point the Asimov data is built
    at, mirroring PhyLiNO's "AsimovValue". When it equals the evaluation point
    nothing is injected, so the command stays what it was.
    """
    known = nnmfit_parameter_names(nnmfit_config)

    fixed = to_nnmfit_names(point, known, "evaluation point")
    injected = to_nnmfit_names(asimov_point, known, "Asimov point")

    out_json = os.path.join(workdir, "nnmfit_evaluated.json")
    cmd = [NNMFIT_PYTHON, NNMFIT_EVAL_LLH, nnmfit_config, "--json", out_json]
    for name, value in sorted(fixed.items()):
        cmd += ["--fix", name, repr(value)]
    if injected != fixed:
        for name, value in sorted(injected.items()):
            cmd += ["--inject", name, repr(value)]
    run(cmd)

    with open(out_json) as handle:
        result = json.load(handle)
    return float(result["llh_value"])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phylino_config")
    parser.add_argument("nnmfit_config")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--likelihood", default="SAY", choices=["SAY", "Poisson"])
    parser.add_argument("--backend", default="cpu", choices=["cpu", "metal", "cuda"],
                        help="PhyLiNO compute backend. The GPU kernels are FP32 for some "
                             "components, so a GPU run needs a looser --tolerance than the "
                             "1e-12-ish a CPU run reaches.")
    parser.add_argument("--use-data", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-9,
                        help="relative tolerance on PhyLiNO == 2 * NNMFit")
    args = parser.parse_args()

    overrides = {}
    for item in args.set:
        name, value = item.split("=", 1)
        overrides[name] = float(value)

    workdir = args.workdir or tempfile.mkdtemp(prefix="llh_parity_")
    os.makedirs(workdir, exist_ok=True)
    print(f"workdir: {workdir}")

    probe_config = os.path.join(workdir, "probe.json")
    point, asimov_point = build_probe_config(args.phylino_config, probe_config,
                                             overrides, args.likelihood, args.use_data,
                                             args.backend)

    moved = {k: (asimov_point[k], v) for k, v in point.items() if asimov_point[k] != v}
    if moved:
        print("evaluating away from the Asimov point:")
        for name, (truth, value) in sorted(moved.items()):
            print(f"  {name}: Asimov {truth} -> evaluated at {value}")

    ours = phylino_llh(probe_config, workdir)
    theirs = nnmfit_llh(args.nnmfit_config, point, asimov_point, workdir)

    expected = 2.0 * theirs
    deviation = abs(ours - expected) / max(1.0, abs(expected))

    print()
    print(f"PhyLiNO  -2lnL + chi2 : {ours!r}")
    print(f"NNMFit   -lnL + chi2/2: {theirs!r}")
    print(f"2 * NNMFit            : {expected!r}")
    print(f"relative deviation    : {deviation:.3e} (tolerance {args.tolerance:g})")

    if deviation > args.tolerance:
        print("MISMATCH")
        return 1
    print("ok: likelihood values agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
