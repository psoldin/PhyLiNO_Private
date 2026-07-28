#!/usr/bin/env python3
"""Turn the Phase-2 combined NNMFit config into its Binning_2D_to_3D counterpart.

The result matches /Users/soldin/Downloads/Fit_Configuration_Final.yaml except for the
MuonGun template variant (kept fluctuation-free) -- see the Phase 3a design doc
section 8.

Changes, and nothing else:
  * tracks and cscd_cascade gain reco_ra + class_name Binning_2D_to_3D; cscd_muon
    stays Binning_2D, exactly as the final config has it (design doc section 7)
  * those two samples' muon template files are swapped for their _3D variants
  * the CringeFITS galactic component is added with a free norm, and excluded from
    cscd_muon
  * the single-power-law "astro" component is replaced by "astro_brokenPL", copied
    verbatim from Fit_Configuration_Final.yaml (AstroBPL), and the analysis
    component list is updated to match

Usage: make_3d_config.py IN.yaml OUT.yaml
"""
import sys

import yaml

NF = "/Users/soldin/Downloads/nnmfit_files"
TRACKS = "IC86_pass2_SnowStorm_v2_tracks"
CASCADE = "IC86_pass2_SnowStorm_v2_cscd_cascade"
MUON = "IC86_pass2_SnowStorm_v2_cscd_muon"

# cscd_muon is deliberately absent: it stays 2D.
#
# Invariant (not enforced by either script): these bin counts must match the
# --ra-bins value passed to make_3d_muon_templates.py for the same detector
# config (e.g. TRACKS here is 181 edges / 180 bins, matching --ra-bins 180 for
# the tracks template).
RA_BINNING = {
    TRACKS: "(0,6.28319,181,lin)",
    CASCADE: "(0,6.28319,19,lin)",
}

TEMPLATE_3D = {
    TRACKS: f"{NF}/Tracks_CorsikaMuon_Fullrange_drop_5lowEbins_3D.pickle",
    CASCADE: f"{NF}/cscd_muongun_ALL_KDE_5up_manual_ssq_no_fluct_3D.pickle",
}

# component name -> 3D template file. Keyed off the component name rather than a
# substring match on the template path, so an unrecognized component raises
# immediately instead of being silently misassigned.
TEMPLATE_BY_COMPONENT = {
    "muontemplate": TEMPLATE_3D[TRACKS],
    "muon": TEMPLATE_3D[CASCADE],
}

# component name -> (template file, norm parameter name)
GALACTIC = {
    "galactictemplate_cringefits": (
        f"{NF}/galactic_templates/combined/5up/CringeFITS_5up_3D.pickle",
        "cringefits_norm",
    ),
}

# Broken power law astrophysical component, copied verbatim from
# /Users/soldin/Downloads/Fit_Configuration_Final.yaml (components.astro_brokenPL),
# replacing the single-power-law "astro" component.
ASTRO_COMPONENT = "astro"
ASTRO_BPL_COMPONENT = "astro_brokenPL"
ASTRO_BPL = {
    "class": "FluxlessBase",
    "parameters": {
        "astro_BPL": {
            "additional": {
                "per_type_norm": False,
                "variable_mapping": {
                    "e_break": "e_break",
                    "index_1": "gamma_1",
                    "index_2": "gamma_2",
                    "norm": "astro_norm",
                },
            },
            "class": "AstroBPL",
            "default": [1.77, 1.31, 2.74, 4.4],
            "parameters": ["astro_norm", "gamma_1", "gamma_2", "e_break"],
            "prior": [None, None, None, None],
            "prior_width": [None, None, None, None],
            "range": [[0.0, 5.0], [-10.0, 3.0], [2.0, 4.0], [3.6, 5.0]],
        }
    },
}


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path) as f:
        cfg = yaml.safe_load(f)

    expected = {TRACKS, CASCADE, MUON}
    if set(cfg["datasets"]) != expected:
        raise SystemExit(
            f"unexpected datasets {sorted(cfg['datasets'])}, expected {sorted(expected)}"
        )

    for name, dataset in cfg["datasets"].items():
        excluded = [c.strip() for c in dataset["excluded_components"].split(",") if c.strip()]
        excluded = [ASTRO_BPL_COMPONENT if c == ASTRO_COMPONENT else c for c in excluded]

        if name in RA_BINNING:
            binning = dataset["analysis_binning"]
            binning["class_name"] = "Binning_2D_to_3D"
            binning["analysis_variables"] = ["reco_energy", "reco_zenith", "reco_ra"]
            binning["reco_ra_binning"] = RA_BINNING[name]
        else:
            # 2D sample: it can carry no galactic template (the component is stored in
            # the analysis binning and this one has no RA axis).
            excluded += list(GALACTIC)

        dataset["excluded_components"] = ", ".join(excluded)

    for comp_name, comp in cfg["components"].items():
        template = comp.get("additional", {}).get("template_file")
        if template is None:
            continue
        comp["additional"]["template_file"] = TEMPLATE_BY_COMPONENT[comp_name]

    # Swap the single-power-law astro component for the broken power law.
    del cfg["components"][ASTRO_COMPONENT]
    cfg["components"][ASTRO_BPL_COMPONENT] = ASTRO_BPL

    analysis_components = [
        c.strip() for c in cfg["analysis"]["components"].split(",") if c.strip()
    ]
    analysis_components = [
        ASTRO_BPL_COMPONENT if c == ASTRO_COMPONENT else c for c in analysis_components
    ]
    for comp_name, (template_file, norm_name) in GALACTIC.items():
        # Defining the component is not enough: NNMFit only builds what
        # analysis.components lists, so a galactic template left out of that list
        # contributes exactly zero and every dump silently agrees with itself.
        if comp_name not in analysis_components:
            analysis_components.append(comp_name)

        cfg["components"][comp_name] = {
            "class": "GalacticTemplate",
            "baseline_weights": "powerlaw",
            "additional": {"template_file": template_file},
            "parameters": {
                norm_name: {
                    "class": "Norm",
                    "default": 1.0,
                    "interpolate": False,
                    "range": [0.0, None],
                }
            },
        }

    cfg["analysis"]["components"] = ", ".join(analysis_components)

    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True, default_flow_style=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
