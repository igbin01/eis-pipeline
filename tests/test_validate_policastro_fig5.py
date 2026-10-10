"""Tests for scripts/validate_policastro_fig5.py on synthetic digitized data."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import validate_policastro_fig5 as v  # noqa: E402

AREA = v.AREA_CM2


def _intact_Z(freq, R_e, R_po, Q, n):
    from impedance.models.circuits import CustomCircuit

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cc = CustomCircuit("R0-p(R1,CPE1)", initial_guess=[R_e, R_po, Q, n])
        return cc.predict(freq, use_initial=True)


def _write_digitized(d: Path, t: float, R_po, Q, n, *, flip_phase=False):
    """Mag and phase on different grids, as WebPlotDigitizer would give.

    Both stop at 0.1 Hz, the lowest frequency in the paper's Figure 5.
    """
    f_mag = np.logspace(5, -1, 61)
    f_ph = np.logspace(5, -1, 300)
    Z_mag = _intact_Z(f_mag, v.R_OMEGA_OHM, R_po, Q, n)
    Z_ph = _intact_Z(f_ph, v.R_OMEGA_OHM, R_po, Q, n)
    phase = np.degrees(np.angle(Z_ph))
    if flip_phase:
        phase = -phase
    mag_path, phase_path = v.digitized_paths(d, t)
    pd.DataFrame({"freq_hz": f_mag, "Z_mod_ohm": np.abs(Z_mag)}).to_csv(mag_path, index=False)
    pd.DataFrame({"freq_hz": f_ph, "phase_deg": phase}).to_csv(phase_path, index=False)
    return f_mag, Z_mag


def _raw_params(row):
    """Table 2 row (area-normalised) -> raw-ohm circuit parameters."""
    return row.Rpo_ohm_cm2 / AREA, row.Y0 * AREA, row.m


def test_reconstruct_matches_true_impedance(tmp_path):
    f_true, Z_true = _write_digitized(tmp_path, 0, 1.5e9, 1.2e-9, 0.95)
    freq, Z = v.reconstruct_spectrum(*v.digitized_paths(tmp_path, 0))
    np.testing.assert_allclose(freq, f_true)
    # phase interpolation error only (300-point phase grid)
    np.testing.assert_allclose(np.abs(Z), np.abs(Z_true), rtol=1e-12)
    np.testing.assert_allclose(np.angle(Z), np.angle(Z_true), atol=np.radians(0.05))


def test_sign_check_triggers_on_flipped_phase(tmp_path):
    _write_digitized(tmp_path, 0, 1.5e9, 1.2e-9, 0.95, flip_phase=True)
    with pytest.raises(v.SignConventionError, match="Z.imag >= 0"):
        v.reconstruct_spectrum(*v.digitized_paths(tmp_path, 0))


def test_predicted_phase_matches_R_CPE_circuit():
    freq = np.logspace(5, -2, 20)
    R, Y0, m = 2.26e10, 8e-11, 0.946
    expected = np.degrees(np.angle(_intact_Z(freq, 0.0, R, Y0, m)))
    np.testing.assert_allclose(v.predicted_phase_deg(R, Y0, m, freq), expected, atol=1e-9)


def test_run_recovers_table2_parameters(tmp_path, capsys):
    digit = tmp_path / "digitized"
    digit.mkdir()
    table2 = v.load_table2()
    times = (0, 840, 3192)
    for t in times:
        _write_digitized(digit, t, *_raw_params(table2[float(t)]))

    out = tmp_path / "out"
    series, fits, rows = v.run(digit, out, times_h=times, plots=True)

    assert (out / "manifest.csv").exists()
    manifest = pd.read_csv(out / "manifest.csv")
    assert list(manifest["immersion_time_h"]) == list(times)
    assert set(manifest["area_cm2"]) == {AREA}
    assert all(s.area_cm2 == AREA and s.temperature_C == 5.0 for s in series)

    assert all(r.success for r in fits)
    assert all(r.model_name == "intact_coating" for r in fits)
    assert all(r.kk_passed is not None for r in fits)
    assert [r["time_h"] for r in rows] == list(map(float, times))
    for r in rows:
        assert abs(r["Y0_pct"]) < 1.0
        assert abs(r["n_diff"]) < 0.005
        assert abs(r["Rpo_pct"]) < 2.0
        # the same formula as Table 2's Eq. 2, so only Table 2 rounding remains
        assert abs(r["Cc_pct"]) < 10.0
    # phase at 0.1 Hz: t=0 -46.6 deg, 840 h -19.7 deg, 3192 h -13.3 deg
    assert [r["Rpo_lower_bound_only"] for r in rows] == [True, False, False]
    for r in rows:
        assert r["f_lo_hz"] == pytest.approx(0.1)
        assert 0 < r["Zlo_over_Rpo"] < 1
    for t in times:
        assert (out / f"bode_t{t}h.png").exists()

    text = capsys.readouterr().out
    assert text.index("Step 3") < text.index("Step 4")
    assert "1-2 significant figures" in text
    assert "negligible" in text
    assert text.count("|Z(f_min)|/R_po_fit") == len(times)
    assert text.count("R_po is only a lower bound") == 1
    for line_start in ("  Y0 ", "  m  ", "  R_po", "  C_c"):
        assert line_start in text
    # Y0 and m are printed before R_po and C_c
    assert text.index("  Y0 ") < text.index("  m  ") < text.index("  R_po") < text.index("  C_c")


def test_rpo_flag_follows_phase_at_lowest_frequency(tmp_path):
    table2 = v.load_table2()
    for factor, flagged in ((1.0, True), (1e-2, False)):
        d = tmp_path / f"x{factor:g}"
        d.mkdir()
        R_po, Q, n = _raw_params(table2[0.0])
        # smaller R_po moves the corner above 0.1 Hz -> resistive at f_min
        _write_digitized(d, 0, R_po * factor, Q, n)
        _, _, rows = v.run(d, tmp_path / f"out{factor:g}", times_h=(0,), plots=False)
        assert rows[0]["Rpo_lower_bound_only"] is flagged
        assert (rows[0]["phase_lo_deg"] < v.RPO_LOWER_BOUND_PHASE_DEG) is flagged
