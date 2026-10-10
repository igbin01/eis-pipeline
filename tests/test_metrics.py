"""Tests for eispipe.metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import synth_degraded
from eispipe.fit import FitResult, FitSeries, fit_series
from eispipe.metrics import (
    breakpoint_frequency,
    breakpoint_frequency_measured,
    coating_capacitance,
    cpe_to_capacitance,
    double_layer_capacitance,
    metrics_table,
    ml_feature_table,
    water_uptake,
    water_uptake_series,
)
from eispipe.spectrum import Spectrum, SpectrumSeries


# --------------------------------------------------------------------------
# cpe_to_capacitance (Hsu-Mansfeld)
# --------------------------------------------------------------------------


def test_cpe_ideal_capacitor_returns_Q():
    assert cpe_to_capacitance(1e-6, 1.0, R=1000.0) == pytest.approx(1e-6)


def test_cpe_closed_form_matches_omega_max_form():
    Q, n, R = 3e-7, 0.88, 5_000.0
    omega_max = (Q * R) ** (-1.0 / n)
    a = cpe_to_capacitance(Q, n, R=R)
    b = cpe_to_capacitance(Q, n, omega_max=omega_max)
    assert a == pytest.approx(b, rel=1e-9)


def test_cpe_argument_validation():
    with pytest.raises(ValueError, match="exactly one of R or omega_max"):
        cpe_to_capacitance(1e-6, 0.9, R=1000.0, omega_max=10.0)
    with pytest.raises(ValueError, match="exactly one of R or omega_max"):
        cpe_to_capacitance(1e-6, 0.9)
    with pytest.raises(ValueError, match="exponent n"):
        cpe_to_capacitance(1e-6, 0.0, R=1000.0)
    with pytest.raises(ValueError, match="Q must be"):
        cpe_to_capacitance(-1e-6, 0.9, R=1000.0)


def test_cpe_degenerate_params_return_inf_not_raise():
    # tiny exponent + huge resistance overflows the closed form; a bad fit must
    # not crash the metrics table.
    C = cpe_to_capacitance(1.0, 0.01, R=1e10)
    assert np.isinf(C)


def test_cpe_n_above_one_warns_and_returns_nan():
    # n slightly above 1 is common in real fits; one bad time point should not
    # abort a whole series, so this warns instead of raising.
    with pytest.warns(UserWarning, match="n=1.5"):
        C = cpe_to_capacitance(1e-6, 1.5, R=1000.0)
    assert np.isnan(C)


# --------------------------------------------------------------------------
# fixtures: hand-built FitResults (no optimiser involved)
# --------------------------------------------------------------------------


def _fit(t, *, model="degraded_coating", Qc=2e-7, nc=0.9, Rpo=8000.0,
         area=7.07, ok=True):
    freq = np.logspace(4, -1, 20)
    params = {"R_e": 40.0, "Q_c": Qc, "n_c": nc, "R_po": Rpo,
              "Q_dl": 5e-6, "n_dl": 0.8, "R_ct": 2e6}
    if model == "intact_coating":
        params = {"R_e": 40.0, "R_po": Rpo, "Q_c": Qc, "n_c": nc}
    return FitResult(
        model_name=model,
        circuit="x",
        params=params,
        param_errors={k: 0.0 for k in params},
        freq=freq,
        Z_data=np.zeros_like(freq, dtype=complex),
        Z_fit=np.zeros_like(freq, dtype=complex),
        rmse=1.0,
        chi2_norm=0.0,
        weighting="modulus",
        success=ok,
        sample_id="P1",
        immersion_time_h=float(t),
        area_cm2=area,
    )


def test_coating_capacitance_uses_Rpo_and_area():
    f = _fit(0, Qc=2e-7, nc=0.9, Rpo=8000.0, area=10.0)
    expected = cpe_to_capacitance(2e-7, 0.9, R=8000.0)
    assert coating_capacitance(f) == pytest.approx(expected)
    assert coating_capacitance(f, per_area=True) == pytest.approx(expected / 10.0)


def test_double_layer_capacitance_none_for_intact():
    assert double_layer_capacitance(_fit(0, model="intact_coating")) is None
    val = double_layer_capacitance(_fit(0, model="degraded_coating"))
    assert val == pytest.approx(cpe_to_capacitance(5e-6, 0.8, R=2e6))


def test_breakpoint_frequency_matches_definition():
    f = _fit(0, Qc=2e-7, nc=0.9, Rpo=8000.0)
    Cc = coating_capacitance(f)
    assert breakpoint_frequency(f) == pytest.approx(1.0 / (2 * np.pi * 8000.0 * Cc))


# --------------------------------------------------------------------------
# water uptake
# --------------------------------------------------------------------------


def test_water_uptake_reference_and_scale():
    assert water_uptake(1e-7, 1e-7) == 0.0
    # C_t / C_0 == epsilon_water  ->  Xv == 1
    assert water_uptake(80e-7, 1e-7, epsilon_water=80.0) == pytest.approx(1.0)


def test_water_uptake_accepts_fitresults():
    c0 = _fit(0, Qc=1e-7, nc=1.0, Rpo=8000.0)
    ct = _fit(24, Qc=2e-7, nc=1.0, Rpo=8000.0)
    # n = 1 -> C == Q, ratio 2
    assert water_uptake(ct, c0) == pytest.approx(np.log10(2) / np.log10(80.0))


def test_water_uptake_series_and_reference_selection():
    fits = FitSeries([
        _fit(0, Qc=1e-7, nc=1.0),
        _fit(24, Qc=2e-7, nc=1.0),
        _fit(72, Qc=4e-7, nc=1.0),
    ])
    df = water_uptake_series(fits)
    assert list(df["immersion_time_h"]) == [0, 24, 72]
    assert df["water_uptake_vol_frac"].iloc[0] == 0.0
    assert df["water_uptake_vol_frac"].is_monotonic_increasing

    # reference by (nearest) immersion time
    df2 = water_uptake_series(fits, reference=24)
    assert df2["water_uptake_vol_frac"].iloc[1] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# metrics_table
# --------------------------------------------------------------------------


def test_metrics_table_columns_and_content():
    fits = FitSeries([
        _fit(0, Qc=1e-7, nc=0.95, Rpo=1e6),
        _fit(168, Qc=8e-7, nc=0.90, Rpo=5e4),
    ])
    df = metrics_table(fits)

    for col in ("sample_id", "immersion_time_h", "model", "success", "kk_passed",
                "rmse", "R_e", "R_po", "C_c", "f_break", "water_uptake_vol_frac",
                "R_ct", "C_dl", "R_po_ohm_cm2", "C_c_F_per_cm2"):
        assert col in df.columns
    assert df["water_uptake_vol_frac"].iloc[0] == 0.0
    assert df["water_uptake_vol_frac"].iloc[1] > 0
    assert df["R_po_ohm_cm2"].iloc[0] == pytest.approx(1e6 * 7.07)


def test_metrics_table_marks_failed_fit():
    fits = FitSeries([_fit(0, ok=True), _fit(24, ok=False)])
    df = metrics_table(fits)
    assert bool(df["success"].iloc[1]) is False
    assert np.isnan(df["C_c"].iloc[1])


def test_ml_feature_table_has_flat_columns_and_na_then_diff_rates():
    # Build a tiny degraded-series-style FitSeries from the shared synthetic helper.
    specs = []
    for idx, t in enumerate([0.0, 24.0, 72.0]):
        freq, Z = synth_degraded(noise=0.0, seed=idx + 1)
        specs.append(
            Spectrum(freq, Z, sample_id="P1", immersion_time_h=t, area_cm2=7.07)
        )
    fits = fit_series(SpectrumSeries(specs), "degraded_coating", validate=False)

    table = ml_feature_table(fits)

    expected_cols = [
        "immersion_time_h",
        "R_po",
        "C_c",
        "n_c",
        "chi2_norm",
        "water_uptake_vol_frac",
        "R_po_rate",
        "C_c_rate",
        "n_c_rate",
        "chi2_norm_rate",
        "water_uptake_vol_frac_rate",
    ]
    assert list(table.columns) == expected_cols
    assert len(table) == len(fits)

    # First row can't be diffed against a prior point, so its derivative columns
    # must stay missing even if the series retains its normal time ordering.
    for rate_col in [
        "R_po_rate",
        "C_c_rate",
        "n_c_rate",
        "chi2_norm_rate",
        "water_uptake_vol_frac_rate",
    ]:
        assert pd.isna(table.loc[0, rate_col])

    # On the second and later rows, the derivative columns are finite differences
    # in the feature divided by the elapsed immersion-time gap in hours.
    for row_idx in range(1, len(table)):
        dt = table.loc[row_idx, "immersion_time_h"] - table.loc[row_idx - 1, "immersion_time_h"]
        for base_col, rate_col in [
            ("R_po", "R_po_rate"),
            ("C_c", "C_c_rate"),
            ("n_c", "n_c_rate"),
            ("chi2_norm", "chi2_norm_rate"),
            ("water_uptake_vol_frac", "water_uptake_vol_frac_rate"),
        ]:
            expected = (table.loc[row_idx, base_col] - table.loc[row_idx - 1, base_col]) / dt
            assert table.loc[row_idx, rate_col] == pytest.approx(expected)


def test_metrics_table_without_area_omits_normalised_columns():
    fits = FitSeries([_fit(0, area=None), _fit(24, area=None)])
    df = metrics_table(fits)
    assert "R_po_ohm_cm2" not in df.columns
    assert "C_c" in df.columns
