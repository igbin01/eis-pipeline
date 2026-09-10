"""Tests for eispipe.spectrum."""

from __future__ import annotations

import numpy as np
import pytest

from eispipe.spectrum import META_FIELDS, Spectrum, SpectrumSeries
from conftest import synth_coating


# --------------------------------------------------------------------------
# Spectrum construction & validation
# --------------------------------------------------------------------------


def test_basic_construction_and_derived_quantities():
    freq, Z = synth_coating()
    s = Spectrum(freq, Z, sample_id="A", immersion_time_h=0.0)

    assert len(s) == s.n_points == len(freq)
    np.testing.assert_allclose(s.Z_mag, np.abs(Z))
    np.testing.assert_allclose(s.Z_phase_deg, np.degrees(np.angle(Z)))
    # capacitive response -> negative imaginary part -> negative phase
    assert np.all(s.Z_phase_deg <= 1e-9)
    assert set(s.metadata) == set(META_FIELDS)


def test_length_mismatch_raises():
    freq, Z = synth_coating(n=20)
    with pytest.raises(ValueError, match="length mismatch"):
        Spectrum(freq[:-1], Z)


def test_non_finite_and_empty_raise():
    freq, Z = synth_coating(n=10)
    Zbad = Z.copy()
    Zbad[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        Spectrum(freq, Zbad)
    with pytest.raises(ValueError, match="no data points"):
        Spectrum(np.array([]), np.array([], dtype=complex))
    with pytest.raises(ValueError, match="strictly positive"):
        Spectrum(np.array([1.0, 0.0, -1.0]), np.array([1j, 1j, 1j]))


def test_ascending_frequency_is_reordered_descending():
    freq, Z = synth_coating(n=15)
    asc = np.argsort(freq)  # low -> high
    s = Spectrum(freq[asc], Z[asc])

    assert s.freq[0] > s.freq[-1]
    # the pairing between f and Z must survive the reordering
    order = np.argsort(freq)[::-1]
    np.testing.assert_allclose(s.freq, freq[order])
    np.testing.assert_allclose(s.Z, Z[order])


def test_metadata_range_validation():
    freq, Z = synth_coating(n=10)
    with pytest.raises(ValueError, match="immersion_time_h"):
        Spectrum(freq, Z, immersion_time_h=-1.0)
    with pytest.raises(ValueError, match="area_cm2"):
        Spectrum(freq, Z, area_cm2=0.0)


def test_with_metadata_returns_copy_and_rejects_unknown():
    freq, Z = synth_coating(n=10)
    s = Spectrum(freq, Z, sample_id="A", immersion_time_h=0.0)
    s2 = s.with_metadata(immersion_time_h=24.0, electrolyte="3.5% NaCl")

    assert s.immersion_time_h == 0.0  # original untouched
    assert s2.immersion_time_h == 24.0
    assert s2.electrolyte == "3.5% NaCl"
    assert s2.sample_id == "A"
    with pytest.raises(TypeError, match="unknown metadata field"):
        s.with_metadata(bogus=1)


def test_to_frame_shape_and_broadcast():
    freq, Z = synth_coating(n=12)
    s = Spectrum(freq, Z, sample_id="A", immersion_time_h=48.0)
    df = s.to_frame()

    assert list(df.columns) == [
        "sample_id",
        "immersion_time_h",
        "freq",
        "Z_real",
        "Z_imag",
        "Z_mag",
        "Z_phase_deg",
    ]
    assert len(df) == 12
    assert (df["sample_id"] == "A").all()
    assert (df["immersion_time_h"] == 48.0).all()
    np.testing.assert_allclose(df["freq"].to_numpy(), s.freq)
    np.testing.assert_allclose(df["Z_imag"].to_numpy(), s.Z.imag)


# --------------------------------------------------------------------------
# SpectrumSeries
# --------------------------------------------------------------------------


def _spec(t, sample_id="A", n=8):
    freq, Z = synth_coating(n=n)
    return Spectrum(freq, Z, sample_id=sample_id, immersion_time_h=t)


def test_series_sorts_by_immersion_time():
    series = SpectrumSeries([_spec(168), _spec(0), _spec(24)])
    np.testing.assert_allclose(series.times, [0, 24, 168])
    assert series.baseline.immersion_time_h == 0
    assert len(series) == 3
    assert series.sample_id == "A"


def test_series_requires_immersion_time():
    freq, Z = synth_coating(n=8)
    with pytest.raises(ValueError, match="immersion_time_h"):
        SpectrumSeries([_spec(0), Spectrum(freq, Z, sample_id="A")])


def test_series_rejects_mixed_sample_id():
    with pytest.raises(ValueError, match="multiple sample_id"):
        SpectrumSeries([_spec(0, "A"), _spec(24, "B")])


def test_series_warns_on_duplicate_times():
    with pytest.warns(UserWarning, match="duplicate immersion_time_h"):
        SpectrumSeries([_spec(24), _spec(24)])


def test_series_indexing_and_frame():
    series = SpectrumSeries([_spec(0), _spec(24), _spec(168)])

    assert isinstance(series[0], Spectrum)
    sub = series[:2]
    assert isinstance(sub, SpectrumSeries)
    np.testing.assert_allclose(sub.times, [0, 24])

    df = series.to_frame()
    assert len(df) == 3 * 8
    assert sorted(df["immersion_time_h"].unique()) == [0, 24, 168]
