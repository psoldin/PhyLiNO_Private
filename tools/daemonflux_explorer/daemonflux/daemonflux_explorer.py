#!/usr/bin/env python3
"""Interactive explorer for the daemonflux parameters.

One slider per daemonflux parameter (hadronic and GSF, in units of the calibrated
sigma; 0 is the calibrated value) and six panels for the selected neutrino species:

    Φ per bin vs E        | E³·Φ(E), averaged over the cosθ range
    ratio to nominal vs E |
    2D Φ per bin          | Φ per bin vs cosθ
    (E × cosθ)            | ratio to nominal vs cosθ

"Φ per bin" is the flux integrated over the bin's ΔE and 2π·Δcosθ, so the 1D
projections are sums of the 2D map. Upgoing directions (cosθ < 0) use the flux at
|cosθ|. Conventional and prompt are drawn separately and summed; daemonflux leaves
the prompt flux uncalibrated, so it moves with a slider only if the tables carry a
prompt Jacobian.

Usage:
    daemonflux_explorer.py                     # daemonflux's default tables (downloads them once)
    daemonflux_explorer.py --test-tables       # the reduced tables from daemonflux/tests
    daemonflux_explorer.py --spl-file S.pkl --cal-file C.pkl
    daemonflux_explorer.py --snapshot out.png --set GSF_1=1   # render once, no window
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Display name -> daemonflux quantities summed for it.
SPECIES = {
    "νμ": ["numu"],
    "ν̄μ": ["antinumu"],
    "νe": ["nue"],
    "ν̄e": ["antinue"],
    "νμ + ν̄μ": ["numuflux"],
    "νe + ν̄e": ["nueflux"],
    "all ν": ["numuflux", "nueflux"],
}


def bin_integrate(values, x, subdiv, axis):
    """Trapezoid integral of `values` over each bin, along `axis`.

    `x` holds the nodes along that axis: nbins * subdiv + 1 of them, the bin edges
    being every subdiv-th node. Returns the array with that axis reduced to nbins.
    """
    values = np.moveaxis(values, axis, -1)
    segments = 0.5 * (values[..., 1:] + values[..., :-1]) * np.diff(x)
    per_bin = segments.reshape(*segments.shape[:-1], -1, subdiv).sum(axis=-1)
    return np.moveaxis(per_bin, -1, axis)


def fine_nodes(edges, subdiv):
    """Every bin of `edges` split into `subdiv` equal pieces (in the edges' own spacing)."""
    steps = np.linspace(0.0, 1.0, subdiv + 1)[:-1]
    inner = edges[:-1, None] + np.diff(edges)[:, None] * steps
    return np.append(inner.ravel(), edges[-1])


class FluxModel:
    """daemonflux evaluated on a (log E, cosθ) binning, without any GUI."""

    def __init__(self, flux, log_e_edges, cos_edges, subdiv):
        self.flux = flux
        self.subdiv = subdiv
        self.e_edges = 10.0**log_e_edges
        self.cos_edges = cos_edges

        self.log_e_nodes = fine_nodes(log_e_edges, subdiv)
        self.e_nodes = 10.0**self.log_e_nodes
        self.cos_nodes = fine_nodes(cos_edges, subdiv)

        params = flux.params
        self.names = list(params.known_parameters)
        self.groups = {name: params.groups[name] for name in self.names}

        # Mirror upgoing directions, then clip to the tabulated zenith range.
        tabulated = np.asarray(flux._get_flux_instance("")._zenith_deg_arr, dtype=float)
        zenith = np.degrees(np.arccos(np.clip(np.abs(self.cos_nodes), 0.0, 1.0)))
        clipped = np.clip(zenith, tabulated.min(), tabulated.max())
        self.clipped_fraction = np.mean(np.abs(clipped - zenith) > 1e-9)
        self.zenith_range = (tabulated.min(), tabulated.max())
        # daemonflux wants ascending zenith angles; map back to the nodes afterwards.
        self._zenith_unique, self._zenith_index = np.unique(clipped, return_inverse=True)

        quantities = set(flux.quantities)
        self.prompt_mode = (
            "pr_" if all(f"pr_{q}" in quantities for qs in SPECIES.values() for q in qs)
            else "total_" if all(f"total_{q}" in quantities for qs in SPECIES.values() for q in qs)
            else None
        )
        self._nominal = {}

    def _phi_nodes(self, quantity, shifts):
        """Φ (not E³Φ) on the fine (E, cosθ) nodes, shape (nE_nodes, ncos_nodes)."""
        e3phi = self.flux.flux(self.e_nodes, self._zenith_unique, quantity, params=shifts)
        e3phi = np.asarray(e3phi).reshape(len(self.e_nodes), len(self._zenith_unique))
        return e3phi[:, self._zenith_index] / self.e_nodes[:, None] ** 3

    def _components_nodes(self, species, shifts):
        conv = sum(self._phi_nodes(q, shifts) for q in SPECIES[species])
        if self.prompt_mode == "pr_":
            prompt = sum(self._phi_nodes(f"pr_{q}", shifts) for q in SPECIES[species])
        elif self.prompt_mode == "total_":
            # total_ = calibrated conventional + uncalibrated prompt; the difference at the
            # calibrated point is the prompt flux, which has no parameters of its own.
            prompt = sum(self._phi_nodes(f"total_{q}", {}) - self._phi_nodes(q, {})
                         for q in SPECIES[species])
        else:
            prompt = np.zeros_like(conv)
        return {"conventional": conv, "prompt": prompt}

    def evaluate(self, species, shifts):
        """Per-bin integrals and the E³Φ curve of every component, plus their sum.

        Returns {component: {"bins": (nE, ncos) in 1/(cm² s), "e3phi": (nE_nodes,)
        in GeV²/(cm² s sr), averaged over the cosθ range}}.
        """
        nodes = self._components_nodes(species, {k: v for k, v in shifts.items() if v != 0.0})
        nodes["sum"] = nodes["conventional"] + nodes["prompt"]

        result = {}
        cos_span = self.cos_edges[-1] - self.cos_edges[0]
        for name, phi in nodes.items():
            # ∫Φ dE = ∫Φ·E d(ln E): trapezoids in log E follow the steep spectrum closely.
            over_e = bin_integrate(phi * self.e_nodes[:, None], np.log(self.e_nodes), self.subdiv, axis=0)
            bins = 2.0 * np.pi * bin_integrate(over_e, self.cos_nodes, self.subdiv, axis=1)
            mean_phi = np.trapezoid(phi, self.cos_nodes, axis=1) / cos_span
            result[name] = {"bins": bins, "e3phi": mean_phi * self.e_nodes**3}
        return result

    def nominal(self, species):
        if species not in self._nominal:
            self._nominal[species] = self.evaluate(species, {})
        return self._nominal[species]

    def chi2(self, shifts):
        """Δχ² of the shifts under the calibration covariance (0 at the calibrated point)."""
        active = {k: v for k, v in shifts.items() if v != 0.0}
        return float(self.flux.chi2(active)) if active else 0.0


def ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


COLORS = {"conventional": "tab:blue", "prompt": "tab:orange", "sum": "black"}


class Figure:
    """The six panels, redrawn in place on every update."""

    def __init__(self, figure, model):
        self.fig = figure
        self.model = model
        # Row 2 and column 2 are spacers: tick labels of the ratio panel above the 2D
        # map, and the colorbar's label next to the cosθ panels.
        grid = figure.add_gridspec(5, 4, height_ratios=[3, 1.2, 0.45, 3, 1.2],
                                   width_ratios=[1, 0.035, 0.42, 1], hspace=0.08, wspace=0.05)
        self.ax_e = figure.add_subplot(grid[0, 0])
        self.ax_e_ratio = figure.add_subplot(grid[1, 0], sharex=self.ax_e)
        self.ax_2d = figure.add_subplot(grid[3:, 0])
        self.ax_cbar = figure.add_subplot(grid[3:, 1])
        self.ax_e3 = figure.add_subplot(grid[0:2, 3])
        self.ax_cos = figure.add_subplot(grid[3, 3])
        self.ax_cos_ratio = figure.add_subplot(grid[4, 3], sharex=self.ax_cos)
        figure.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.1)

    def draw(self, species, current, nominal):
        from matplotlib.colors import LogNorm

        m = self.model
        for ax in (self.ax_e, self.ax_e_ratio, self.ax_2d, self.ax_e3, self.ax_cos, self.ax_cos_ratio):
            ax.cla()

        for name, color in COLORS.items():
            if name == "prompt" and m.prompt_mode is None:
                continue
            cur, nom = current[name]["bins"], nominal[name]["bins"]
            label = name if name != "sum" else "conv. + prompt"
            self.ax_e.stairs(cur.sum(axis=1), m.e_edges, baseline=None, color=color, label=label)
            self.ax_e.stairs(nom.sum(axis=1), m.e_edges, baseline=None, color=color, ls="--", lw=0.8)
            self.ax_e_ratio.stairs(ratio(cur.sum(axis=1), nom.sum(axis=1)), m.e_edges, baseline=None, color=color)
            self.ax_cos.stairs(cur.sum(axis=0), m.cos_edges, baseline=None, color=color)
            self.ax_cos.stairs(nom.sum(axis=0), m.cos_edges, baseline=None, color=color, ls="--", lw=0.8)
            self.ax_cos_ratio.stairs(ratio(cur.sum(axis=0), nom.sum(axis=0)), m.cos_edges, baseline=None, color=color)
            self.ax_e3.plot(m.e_nodes, current[name]["e3phi"], color=color, label=label)
            self.ax_e3.plot(m.e_nodes, nominal[name]["e3phi"], color=color, ls="--", lw=0.8)

        self.ax_e.set(xscale="log", yscale="log", ylabel="Φ per bin [cm⁻² s⁻¹]",
                      title=f"{species}   (dashed: calibrated nominal)")
        self.ax_e.legend(fontsize="small")
        self.ax_e.tick_params(labelbottom=False)
        self.ax_e_ratio.set(xlabel="E [GeV]", ylabel="ratio")
        self.ax_cos.set(yscale="log", ylabel="Φ per bin [cm⁻² s⁻¹]")
        self.ax_cos.tick_params(labelbottom=False)
        self.ax_cos_ratio.set(xlabel="cos θ", ylabel="ratio")
        for ax in (self.ax_e_ratio, self.ax_cos_ratio):
            ax.axhline(1.0, color="grey", lw=0.6)
            ax.grid(alpha=0.3)
        self.ax_e3.set(xscale="log", yscale="log", xlabel="E [GeV]",
                       ylabel="E³·Φ [GeV² cm⁻² s⁻¹ sr⁻¹]",
                       title=f"mean over cos θ ∈ [{m.cos_edges[0]:g}, {m.cos_edges[-1]:g}]")
        self.ax_e3.legend(fontsize="small")

        total = current["sum"]["bins"]
        positive = total[total > 0]
        norm = LogNorm(positive.min(), positive.max()) if positive.size else None
        mesh = self.ax_2d.pcolormesh(m.e_edges, m.cos_edges, total.T, norm=norm, cmap="viridis")
        self.ax_2d.set(xscale="log", xlabel="E [GeV]", ylabel="cos θ")
        self.ax_cbar.cla()
        self.fig.colorbar(mesh, cax=self.ax_cbar, label="conv. + prompt per bin [cm⁻² s⁻¹]")


def load_flux(args):
    from daemonflux import Flux

    if args.test_tables:
        import daemonflux

        tests = Path(daemonflux.__file__).resolve().parents[2] / "tests"
        args.spl_file = str(tests / "test_daemonsplines_generic_202303_1.pkl")
        args.cal_file = str(tests / "test_calibration_default_202303_1.pkl")
    if args.spl_file:
        return Flux(spl_file=args.spl_file, cal_file=args.cal_file, debug=0)
    return Flux(location=args.location, debug=0)


def parse_sets(items, names):
    shifts = {}
    for item in items:
        name, _, value = item.partition("=")
        if name not in names:
            raise SystemExit(f"--set: unknown parameter {name!r}; known: {', '.join(names)}")
        shifts[name] = float(value)
    return shifts


def run_window(model, args, shifts):
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure as MplFigure
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox,
                                   QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea,
                                   QSlider, QVBoxLayout, QWidget)

    ticks_per_sigma = 100
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("daemonflux explorer")

    figure = Figure(MplFigure(figsize=(12, 9)), model)
    canvas = FigureCanvasQTAgg(figure.fig)

    species_box = QComboBox()
    species_box.addItems(list(SPECIES))
    reset = QPushButton("Reset")
    status = QLabel()
    status.setWordWrap(True)

    rows = {}  # name -> (slider, spin)
    groups_layout = QVBoxLayout()
    for group, title in (("hadronic", "Hadronic yields"), ("primary", "Primary flux (GSF)")):
        names = [n for n in model.names if model.groups[n] == group]
        if not names:
            continue
        box = QGroupBox(title)
        grid = QGridLayout(box)
        for row, name in enumerate(names):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-int(args.range_sigma * ticks_per_sigma), int(args.range_sigma * ticks_per_sigma))
            spin = QDoubleSpinBox()
            spin.setRange(-args.range_sigma, args.range_sigma)
            spin.setSingleStep(0.1)
            spin.setSuffix(" σ")
            grid.addWidget(QLabel(name), row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(spin, row, 2)
            rows[name] = (slider, spin)
        groups_layout.addWidget(box)
    groups_layout.addStretch()

    controls = QWidget()
    controls_layout = QVBoxLayout(controls)
    top = QHBoxLayout()
    top.addWidget(QLabel("Species"))
    top.addWidget(species_box, 1)
    top.addWidget(reset)
    controls_layout.addLayout(top)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    inner = QWidget()
    inner.setLayout(groups_layout)
    scroll.setWidget(inner)
    controls_layout.addWidget(scroll, 1)
    controls_layout.addWidget(status)
    controls.setMinimumWidth(380)

    central = QWidget()
    layout = QHBoxLayout(central)
    layout.addWidget(controls)
    layout.addWidget(canvas, 1)
    window.setCentralWidget(central)

    # Slider and spin box are two views of one value; the timer coalesces a burst of
    # changes into one redraw once the event queue drains.
    refresh_timer = QTimer()
    refresh_timer.setSingleShot(True)
    refresh_timer.setInterval(20)

    def refresh():
        species = species_box.currentText()
        current = {name: spin.value() for name, (_, spin) in rows.items()}
        figure.draw(species, model.evaluate(species, current), model.nominal(species))
        canvas.draw_idle()
        status.setText(status_text(model, current))

    def set_value(name, value):
        slider, spin = rows[name]
        for widget, v in ((slider, round(value * ticks_per_sigma)), (spin, value)):
            widget.blockSignals(True)
            widget.setValue(v)
            widget.blockSignals(False)
        refresh_timer.start()

    for name, (slider, spin) in rows.items():
        slider.valueChanged.connect(lambda tick, n=name: set_value(n, tick / ticks_per_sigma))
        spin.valueChanged.connect(lambda value, n=name: set_value(n, value))
    species_box.currentIndexChanged.connect(lambda _: refresh_timer.start())
    reset.clicked.connect(lambda: [set_value(name, 0.0) for name in rows])
    refresh_timer.timeout.connect(refresh)

    for name, value in shifts.items():
        set_value(name, value)
    refresh()
    window.resize(1600, 950)
    window.show()
    return app.exec()


def status_text(model, shifts):
    lines = [f"Δχ² (calibration prior): {model.chi2(shifts):.2f}"]
    if model.prompt_mode is None:
        lines.append("No prompt tables in this file.")
    elif model.prompt_mode == "total_":
        lines.append("Prompt = total − conventional at the calibrated point (no prompt parameters).")
    if model.clipped_fraction > 0:
        lo, hi = model.zenith_range
        lines.append(f"Tables cover zenith {lo:g}°–{hi:g}°: {model.clipped_fraction:.0%} of the cos θ "
                     f"nodes use the flux at the nearest tabulated angle.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tables = parser.add_mutually_exclusive_group()
    tables.add_argument("--test-tables", action="store_true", help="use the reduced tables from daemonflux/tests")
    tables.add_argument("--spl-file", help="spline/response file (default: daemonflux's download for --location)")
    parser.add_argument("--cal-file", help="calibration file belonging to --spl-file")
    parser.add_argument("--location", default="generic")
    parser.add_argument("--emin", type=float, default=1e2, help="GeV")
    parser.add_argument("--emax", type=float, default=1e6, help="GeV")
    parser.add_argument("--ebins", type=int, default=20)
    parser.add_argument("--cos-min", type=float, default=-1.0)
    parser.add_argument("--cos-max", type=float, default=1.0)
    parser.add_argument("--cos-bins", type=int, default=20)
    parser.add_argument("--subdiv", type=int, default=10, help="integration nodes per bin and axis")
    parser.add_argument("--range-sigma", type=float, default=3.0, help="slider range, ± this many σ")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=SIGMA",
                        help="start with a parameter shifted (repeatable)")
    parser.add_argument("--species", choices=list(SPECIES), default="νμ", help="species for --snapshot")
    parser.add_argument("--snapshot", metavar="PNG", help="render once to PNG instead of opening a window")
    args = parser.parse_args()

    flux = load_flux(args)
    log_e_edges = np.linspace(np.log10(args.emin), np.log10(args.emax), args.ebins + 1)
    cos_edges = np.linspace(args.cos_min, args.cos_max, args.cos_bins + 1)
    model = FluxModel(flux, log_e_edges, cos_edges, args.subdiv)
    shifts = parse_sets(args.set, model.names)

    try:
        model.nominal(args.species)
    except AssertionError as error:  # daemonflux's energy-range check
        raise SystemExit(f"{error} -- adjust --emin/--emax")

    if args.snapshot:
        from matplotlib.figure import Figure as MplFigure

        figure = Figure(MplFigure(figsize=(12, 9)), model)
        figure.draw(args.species, model.evaluate(args.species, shifts), model.nominal(args.species))
        figure.fig.text(0.01, 0.005, status_text(model, shifts), fontsize="small", va="bottom")
        figure.fig.savefig(args.snapshot, dpi=110)
        print(f"wrote {args.snapshot}")
        return 0
    return run_window(model, args, shifts)


if __name__ == "__main__":
    sys.exit(main())
