"""Tests for eispipe.validate."""

from __future__ import annotations

import numpy as np
import pytest

from eispipe.spectrum import Spectrum, SpectrumSeries
from eispipe.validate import KKResult, kramers_kronig, validate_series
from conftest import synth_coating, synth_degraded


def test_kramers_kronig_passes_on_clean_synthetic():
    freq, Z = synth_coating()
    s = Spectrum(freq, Z, sample_id="A", immersion_time_h=12.0)

    r = kramers_kronig(s)

    assert isinstance(r, KKResult)
    assert r.passed
    assert r.max_resid < 0.01
    assert r.M >= 1
    assert r.freq.shape == r.resid_real.shape == r.resid_imag.shape == freq.shape
    assert r.sample_id == "A"
    assert r.immersion_time_h == 12.0


def test_kramers_kronig_flags_non_kk_data():
    # A mid-spectrum discontinuity is not KK-transformable.
    freq, Z = synth_degraded(noise=0.0)
    Z = Z.copy()
    Z[freq < np.median(freq)] *= 1.6
    s = Spectrum(freq, Z)

    r = kramers_kronig(s)
    assert not r.passed
    assert r.max_resid > r.threshold


def test_kramers_kronig_fit_type_complex_runs():
    freq, Z = synth_coating()
    r = kramers_kronig(Spectrum(freq, Z), fit_type="complex")
    assert np.isfinite(r.rms_resid)


def test_validate_series_frame():
    freq, Z = synth_coating()
    series = SpectrumSeries(
        [
            Spectrum(freq, Z, sample_id="A", immersion_time_h=t)
            for t in (168.0, 0.0, 24.0)
        ]
    )
    df = validate_series(series)

    assert list(df.columns) == [
        "sample_id",
        "immersion_time_h",
        "M",
        "mu",
        "max_resid",
        "rms_resid",
        "passed",
    ]
    assert list(df["immersion_time_h"]) == [0.0, 24.0, 168.0]
    assert df["passed"].all()
