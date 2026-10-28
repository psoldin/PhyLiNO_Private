#!/usr/bin/env python3
"""Self-check for the per-bin integration in daemonflux_explorer.py.

Run: python tools/daemonflux_explorer/test_daemonflux_explorer.py
"""
import numpy as np

from daemonflux_explorer import FluxModel, bin_integrate, fine_nodes


class PowerLawFlux:
    """Stand-in for daemonflux.Flux: E³Φ = E^(3-γ) · (1 + cos²θ), one parameter."""

    gamma = 3.7

    class params:
        known_parameters = ["GSF_1"]
        groups = {"GSF_1": "primary"}

    quantities = ["numu", "pr_numu"]

    def _get_flux_instance(self, _):
        class Entry:
            _zenith_deg_arr = np.array([0.0, 90.0])

        return Entry()

    def flux(self, energy, zenith_deg, quantity, params):
        scale = 1.0 + params.get("GSF_1", 0.0) if quantity == "numu" else 1e-3
        c2 = np.cos(np.radians(zenith_deg)) ** 2
        return scale * np.outer(energy ** (3.0 - self.gamma), 1.0 + c2)

    def chi2(self, params):
        return sum(v**2 for v in params.values())


def test_fine_nodes_keep_the_edges():
    edges = np.array([0.0, 1.0, 3.0])
    nodes = fine_nodes(edges, 4)
    assert len(nodes) == 2 * 4 + 1
    assert np.allclose(nodes[::4], edges)


def test_bin_integrate_is_exact_for_a_linear_function():
    edges = np.linspace(0.0, 2.0, 3)
    x = fine_nodes(edges, 5)
    per_bin = bin_integrate(np.vstack([x, 2 * x]), x, 5, axis=1)
    assert np.allclose(per_bin, [[0.5, 1.5], [1.0, 3.0]])


def test_power_law_matches_the_analytic_integral():
    # The explorer's default granularity: 0.2-decade bins, 10 nodes per bin. The
    # trapezoid error there is ~0.1% for a γ = 3.7 spectrum (and cancels in ratios).
    model = FluxModel(PowerLawFlux(), np.linspace(2, 4, 11), np.linspace(-1, 1, 3), subdiv=10)
    bins = model.evaluate("νμ", {})["conventional"]["bins"]
    g = PowerLawFlux.gamma
    e = model.e_edges
    over_e = (e[1:] ** (1 - g) - e[:-1] ** (1 - g)) / (1 - g)
    # cosθ in [-1, 0] and [0, 1] after mirroring: ∫(1 + c²) dc = 4/3 for either half.
    expected = 2 * np.pi * np.outer(over_e, [4 / 3, 4 / 3])
    assert np.allclose(bins, expected, rtol=2e-3), bins / expected


def test_a_shift_moves_conventional_but_not_prompt():
    model = FluxModel(PowerLawFlux(), np.linspace(2, 4, 3), np.linspace(-1, 1, 3), subdiv=4)
    nominal, shifted = model.nominal("νμ"), model.evaluate("νμ", {"GSF_1": 0.5})
    assert np.allclose(shifted["conventional"]["bins"], 1.5 * nominal["conventional"]["bins"])
    assert np.allclose(shifted["prompt"]["bins"], nominal["prompt"]["bins"])
    assert model.chi2({"GSF_1": 0.5}) == 0.25


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
