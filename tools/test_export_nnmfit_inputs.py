#!/usr/bin/env python3
"""Self-check for the log10(E) rebinning in export_nnmfit_inputs.py.

Run: python tools/test_export_nnmfit_inputs.py
"""
import numpy as np

from export_nnmfit_inputs import rebin_energy, rebin_energy_matrix


def test_rate_is_conserved():
    # Source bins that tile the target range exactly: every column must sum to 1.
    source = np.linspace(2.5, 7.0, 91)  # 90 bins of 0.05 inside a 45-bin, 0.1 target
    w = rebin_energy_matrix(source, 2.5, 7.0, 45)
    assert np.allclose(w.sum(axis=0), 1.0), w.sum(axis=0)
    # Two source bins per target bin, so pairs merge and nothing else mixes.
    assert np.allclose(w[0, :2], 1.0) and w[0, 2:].max() == 0.0


def test_rate_outside_the_range_is_dropped():
    source = np.array([2.0, 2.5, 3.0])  # first bin entirely below the target range
    w = rebin_energy_matrix(source, 2.5, 3.0, 1)
    assert w[0, 0] == 0.0 and w[0, 1] == 1.0


def test_partial_overlap_splits_proportionally():
    source = np.array([2.5, 2.7])  # one 0.2-wide bin across two 0.1-wide targets
    w = rebin_energy_matrix(source, 2.5, 2.7, 2)
    assert np.allclose(w[:, 0], [0.5, 0.5])


def test_fluctuations_add_in_quadrature():
    source = np.linspace(2.5, 2.7, 3)  # two 0.1 bins merging into one 0.2 bin
    rate = np.array([[3.0], [4.0]])
    fluctuation = np.array([[3.0], [4.0]])
    out_rate, out_fluct = rebin_energy(rate, fluctuation, source, (2.5, 2.7, 1))
    assert np.allclose(out_rate, [[7.0]])           # rates add
    assert np.allclose(out_fluct, [[5.0]])          # sigmas add in quadrature


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
