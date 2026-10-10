"""Synthetic-spectrum helpers shared across eispipe tests."""

from __future__ import annotations

import numpy as np


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
