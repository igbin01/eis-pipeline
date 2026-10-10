"""Tests for eispipe.validate."""

from __future__ import annotations

import numpy as np
import pytest

from eispipe.spectrum import Spectrum, SpectrumSeries
from eispipe.validate import KKResult, kramers_kronig, validate_series
from conftest import synth_degraded
from helpers import synth_coating


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


def test_kramers_kronig_soft_failure_on_zero_impedance_point_returns_failed_result():
    freq, Z = synth_degraded(noise=0.0)
    Zbad = Z.copy()
    Zbad[5] = 0.0 + 0.0j
    spec = Spectrum(freq, Zbad, sample_id="P1", immersion_time_h=24.0)

    result = kramers_kronig(spec)

    assert isinstance(result, KKResult)
    assert result.passed is False
    assert result.M == 0
    assert np.isnan(result.rms_resid)
    assert np.isnan(result.max_resid)
    assert np.isnan(result.mu)


def test_json_safe_event_payload_normalizes_nonfinite_numbers_for_browser_json_parse():
    import importlib.util
    from pathlib import Path

    server_path = Path(__file__).resolve().parents[1] / "app" / "server.py"
    spec = importlib.util.spec_from_file_location("eis_pipeline_server_under_test", server_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    payload = {
        "stage": "validate",
        "status": "done",
        "kk_pass_fail": [
            {"sample_id": "TEST", "immersion_time_h": 1.0, "M": 0, "mu": float("nan")}
        ],
    }

    safe = module._json_safe(payload)

    assert safe["kk_pass_fail"][0]["mu"] is None


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
