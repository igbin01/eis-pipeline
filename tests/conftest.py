"""Shared fixtures / helpers for eispipe tests."""

from __future__ import annotations

import numpy as np
import pytest


def synth_coating(
    n: int = 40,
    R0: float = 50.0,
    Rpo: float = 8_000.0,
    C: float = 2e-7,
    f_hi: float = 1e5,
    f_lo: float = 1e-2,
):
    """A clean one-time-constant coated-metal spectrum: R0 in series with R||C.

    Returned impedance is in the internal capacitive-is-negative convention.
    """
    freq = np.logspace(np.log10(f_hi), np.log10(f_lo), n)
    w = 2 * np.pi * freq
    Z = R0 + 1.0 / (1.0 / Rpo + 1j * w * C)
    return freq, Z


def synth_degraded(
    n: int = 45,
    params=(40.0, 2e-7, 0.93, 8_000.0, 4e-6, 0.82, 2e6),
    f_hi: float = 3e4,
    f_lo: float = 3e-2,
    noise: float = 0.0,
    seed: int = 0,
):
    """A two-time-constant coated-metal spectrum via impedance.py.

    Circuit ``R0-p(CPE1,R1-p(CPE2,R2))`` == eispipe's ``degraded_coating``.
    Returns ``(freq, Z)`` in the internal capacitive-is-negative convention.
    """
    import warnings

    from impedance.models.circuits import CustomCircuit

    freq = np.logspace(np.log10(f_hi), np.log10(f_lo), n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cc = CustomCircuit("R0-p(CPE1,R1-p(CPE2,R2))", initial_guess=list(params))
        Z = cc.predict(freq, use_initial=True)
    if noise:
        rng = np.random.default_rng(seed)
        s = noise * np.abs(Z)
        Z = Z + rng.normal(scale=s) + 1j * rng.normal(scale=s)
    return freq, Z


@pytest.fixture
def coating_spectrum_arrays():
    return synth_coating()
