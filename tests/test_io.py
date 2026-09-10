"""Tests for eispipe.io."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eispipe.io import (
    crop_frequencies,
    drop_inductive,
    load_csv,
    load_instrument,
    load_manifest,
    load_series,
    load_series_by_pattern,
)
from eispipe.spectrum import Spectrum, SpectrumSeries
from conftest import synth_coating


def write_csv(path, freq, Z, *, cols=("freq", "Zreal", "Zimag"), negate_imag=False):
    """Write a 3-column EIS CSV. ``negate_imag`` stores -Im(Z) (positive)."""
    imag = -Z.imag if negate_imag else Z.imag
    pd.DataFrame({cols[0]: freq, cols[1]: Z.real, cols[2]: imag}).to_csv(
        path, index=False
    )


# --------------------------------------------------------------------------
# load_csv
# --------------------------------------------------------------------------


def test_load_csv_standard_headers(tmp_path):
    freq, Z = synth_coating()
    p = tmp_path / "s.csv"
    write_csv(p, freq, Z)

    s = load_csv(p, sample_id="A", immersion_time_h=0.0, area_cm2=10.0)

    assert isinstance(s, Spectrum)
    np.testing.assert_allclose(s.freq, freq)
    np.testing.assert_allclose(s.Z, Z)
    assert s.sample_id == "A"
    assert s.area_cm2 == 10.0
    assert s.source_path == str(p)


@pytest.mark.parametrize(
    "cols",
    [
        ("Frequency (Hz)", "Re(Z)/ohm", "-Im(Z)/ohm"),
        ("f/Hz", "Z'", "Z''"),
        ("freq", "zreal", "zimag"),
    ],
)
def test_load_csv_header_variants(tmp_path, cols):
    freq, Z = synth_coating()
    p = tmp_path / "s.csv"
    negate = cols[2].strip().lower().startswith("-")
    write_csv(p, freq, Z, cols=cols, negate_imag=negate)

    s = load_csv(p)
    # regardless of how the file stored the sign, we recover the physics
    np.testing.assert_allclose(s.Z, Z, rtol=1e-9)


def test_load_csv_imag_sign_override(tmp_path):
    freq, Z = synth_coating()
    p = tmp_path / "s.csv"
    write_csv(p, freq, Z, cols=("freq", "Zreal", "Zimag"), negate_imag=True)

    as_measured = load_csv(p, imag_sign="as_measured")
    np.testing.assert_allclose(as_measured.Z.imag, -Z.imag)  # positive, as stored

    negated = load_csv(p, imag_sign="negate")
    np.testing.assert_allclose(negated.Z.imag, Z.imag)

    with pytest.raises(ValueError, match="imag_sign"):
        load_csv(p, imag_sign="weird")


def test_load_csv_unknown_metadata_kwarg(tmp_path):
    freq, Z = synth_coating()
    p = tmp_path / "s.csv"
    write_csv(p, freq, Z)
    with pytest.raises(TypeError, match="unexpected metadata keyword"):
        load_csv(p, panel="A")


def test_load_csv_column_detection_failures(tmp_path):
    freq, Z = synth_coating(n=10)
    p = tmp_path / "s.csv"
    pd.DataFrame({"a": freq, "b": Z.real, "c": Z.imag}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="could not find a 'freq'"):
        load_csv(p)

    p2 = tmp_path / "s2.csv"
    pd.DataFrame(
        {"freq": freq, "Zreal": Z.real, "Re(Z)": Z.real, "Zimag": Z.imag}
    ).to_csv(p2, index=False)
    with pytest.raises(ValueError, match="ambiguous 'real'"):
        load_csv(p2)


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------


def test_crop_frequencies():
    freq, Z = synth_coating(n=50)
    s = Spectrum(freq, Z)

    c = crop_frequencies(s, fmin=1.0, fmax=1e3)
    assert c.freq.min() >= 1.0
    assert c.freq.max() <= 1e3
    assert len(c) < len(s)
    # metadata / ordering preserved
    assert c.freq[0] > c.freq[-1]

    with pytest.raises(ValueError, match="removed every point"):
        crop_frequencies(s, fmin=1e9)


def test_drop_inductive_matches_impedance_helper():
    freq, Z = synth_coating(n=30)
    Z = Z.copy()
    Z[:3] = Z[:3].real + 5j  # fake inductive tail at high frequency
    s = Spectrum(freq, Z)

    d = drop_inductive(s)
    assert np.all(d.Z.imag < 0)
    assert len(d) == len(s) - 3

    from impedance.preprocessing import ignoreBelowX

    f_ref, Z_ref = ignoreBelowX(s.freq, s.Z)
    np.testing.assert_allclose(d.freq, f_ref)
    np.testing.assert_allclose(d.Z, Z_ref)


# --------------------------------------------------------------------------
# load_manifest
# --------------------------------------------------------------------------


def _manifest_df():
    return pd.DataFrame(
        {
            "filename": ["t168.csv", "t0.csv", "t24.csv"],
            "sample_id": ["A", "A", "A"],
            "immersion_time_h": [168, 0, 24],
            "area_cm2": [7.07, 7.07, 7.07],
            "electrolyte": ["3.5% NaCl"] * 3,
            "temperature_C": [25, 25, 25],
        }
    )


def test_load_manifest_valid_is_sorted(tmp_path):
    p = tmp_path / "manifest.csv"
    _manifest_df().to_csv(p, index=False)

    df = load_manifest(p)
    assert list(df["immersion_time_h"]) == [0, 24, 168]
    assert list(df["filename"]) == ["t0.csv", "t24.csv", "t168.csv"]


def test_load_manifest_missing_required_column(tmp_path):
    p = tmp_path / "m.csv"
    _manifest_df().drop(columns=["immersion_time_h"]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        load_manifest(p)


def test_load_manifest_missing_optional_column_warns(tmp_path):
    p = tmp_path / "m.csv"
    _manifest_df().drop(columns=["area_cm2", "electrolyte"]).to_csv(p, index=False)
    with pytest.warns(UserWarning, match="area_cm2"):
        load_manifest(p)


def test_load_manifest_range_checks(tmp_path):
    p = tmp_path / "m.csv"
    bad = _manifest_df()
    bad.loc[0, "immersion_time_h"] = -5
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="negative immersion_time_h"):
        load_manifest(p)

    p2 = tmp_path / "m2.csv"
    bad2 = _manifest_df()
    bad2.loc[0, "area_cm2"] = 0.0
    bad2.to_csv(p2, index=False)
    with pytest.raises(ValueError, match="non-positive area_cm2"):
        load_manifest(p2)


# --------------------------------------------------------------------------
# load_series
# --------------------------------------------------------------------------


def test_load_series_from_manifest_file(tmp_path):
    freq, Z = synth_coating()
    for name in ("t0.csv", "t24.csv", "t168.csv"):
        write_csv(tmp_path / name, freq, Z)
    mpath = tmp_path / "manifest.csv"
    _manifest_df().to_csv(mpath, index=False)

    series = load_series(mpath)

    assert isinstance(series, SpectrumSeries)
    np.testing.assert_allclose(series.times, [0, 24, 168])
    assert series.sample_id == "A"
    for s in series:
        assert s.area_cm2 == pytest.approx(7.07)
        assert s.electrolyte == "3.5% NaCl"
        assert s.temperature_C == 25
        assert s.source_path is not None


def test_load_series_from_dataframe_with_data_dir(tmp_path):
    freq, Z = synth_coating()
    for name in ("t0.csv", "t24.csv", "t168.csv"):
        write_csv(tmp_path / name, freq, Z)

    series = load_series(_manifest_df(), data_dir=tmp_path)
    np.testing.assert_allclose(series.times, [0, 24, 168])


# --------------------------------------------------------------------------
# load_series_by_pattern
# --------------------------------------------------------------------------


def test_load_series_by_pattern_default(tmp_path):
    freq, Z = synth_coating()
    write_csv(tmp_path / "panelA_0h.csv", freq, Z)
    write_csv(tmp_path / "panelA_168h.csv", freq, Z)
    write_csv(tmp_path / "notes.csv", freq, Z)  # no time -> skipped

    with pytest.warns(UserWarning, match="does not match time pattern"):
        series = load_series_by_pattern(
            sorted(tmp_path.glob("*.csv")), area_cm2=5.0
        )

    np.testing.assert_allclose(series.times, [0, 168])
    assert all(s.area_cm2 == 5.0 for s in series)


def test_load_series_by_pattern_sample_group(tmp_path):
    freq, Z = synth_coating()
    write_csv(tmp_path / "A-12h.csv", freq, Z)
    write_csv(tmp_path / "A-96h.csv", freq, Z)

    series = load_series_by_pattern(
        sorted(tmp_path.glob("*.csv")),
        pattern=r"(?P<sample>[A-Z])-(?P<time>\d+)h",
    )
    assert series.sample_id == "A"
    np.testing.assert_allclose(series.times, [12, 96])


def test_load_series_by_pattern_no_match_raises(tmp_path):
    freq, Z = synth_coating()
    write_csv(tmp_path / "nope.csv", freq, Z)
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="no files matched"):
            load_series_by_pattern(sorted(tmp_path.glob("*.csv")))


# --------------------------------------------------------------------------
# load_instrument
# --------------------------------------------------------------------------


def test_load_instrument_plain_csv(tmp_path):
    freq, Z = synth_coating()
    p = tmp_path / "raw.csv"  # headerless 3-column == impedance.readCSV format
    np.savetxt(p, np.c_[freq, Z.real, Z.imag], delimiter=",")

    s = load_instrument(p, sample_id="A", immersion_time_h=0.0)
    np.testing.assert_allclose(s.freq, freq)
    np.testing.assert_allclose(s.Z, Z)

    flipped = load_instrument(p, negate_imaginary=True)
    np.testing.assert_allclose(flipped.Z.imag, -Z.imag)


def test_load_instrument_unknown_extension_raises(tmp_path):
    p = tmp_path / "raw.xyz"
    p.write_text("nonsense")
    with pytest.raises(ValueError, match="cannot infer instrument"):
        load_instrument(p)
