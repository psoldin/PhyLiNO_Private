# daemonflux explorer

Sliders for every daemonflux parameter (hadronic yields and GSF primary flux, in σ of
the calibration; 0 is the calibrated value) and the resulting neutrino flux per
(E, cos θ) bin: projections with a ratio to the nominal, E³·Φ(E), and the 2D map.
Conventional and prompt are shown separately and summed. See the docstring of
`daemonflux_explorer.py` for the panel layout and the conventions.

## Setup

Python 3.13 (daemonflux needs h5py, which has no wheel for 3.15 yet), with daemonflux
installed from a local checkout:

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ../../../daemonflux
```

## Run

```bash
.venv/bin/python daemonflux_explorer.py                 # release tables, downloaded once (~20 MB)
.venv/bin/python daemonflux_explorer.py --test-tables   # reduced test tables: 5 parameters, zenith 0°-18° only
.venv/bin/python daemonflux_explorer.py --snapshot out.png --set GSF_1=1 --species "νμ + ν̄μ"
```

Self-check of the bin integration: `.venv/bin/python test_daemonflux_explorer.py`.
