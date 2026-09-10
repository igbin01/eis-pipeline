"""Tests for eispipe.fit."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from eispipe.fit import FitResult, FitSeries, fit_series, fit_spectrum
from eispipe.spectrum import Spectrum, SpectrumSeries
from conftest import synth_degraded


def _degraded_series(times=(0.0, 24.0, 72.0, 168.0), noise=0.01):
    specs = []
    for i, t in enumerate(times):
        # gentle degradation: coating Q up, pore R down
        p = (40.0, 2e-7 * (1 + 0.02 * t), 0.93, 8_000.0 / (1 + 0.03 * t),
             4e-6, 0.82, 2e6 / (1 + 0.05 * t))
        freq, Z = synth_degraded(params=p, noise=noise, seed=i + 1)
        specs.append(
            Spectrum(freq, Z, sample_id="P1", immersion_time_h=t, area_cm2=7.07)
        )
    return SpectrumSeries(specs)


def test_fit_spectrum_recovers_parameters():
    freq, Z = synth_degraded(noise=0.0)
    spec = Spectrum(freq, Z, sample_id="P1", immersion_time_h=0.0, area_cm2=10.0)

    res = fit_spectrum(spec, "degraded_coating")

    assert isinstance(res, FitResult)
    assert res.success
    assert set(res.params) == set(res.param_errors) == set(
        ("R_e", "Q_c", "n_c", "R_po", "Q_dl", "n_dl", "R_ct")
    )
    # true R_po = 8000, R_ct = 2e6
    assert res.params["R_po"] == pytest.approx(8_000.0, rel=0.05)
    assert res.params["R_ct"] == pytest.approx(2e6, rel=0.1)
    assert res.rmse < 1.0
    assert res.Z_fit.shape == freq.shape
    assert res.area_cm2 == 10.0


def test_fit_spectrum_rejects_bad_weighting():
    freq, Z = synth_degraded(noise=0.0)
    with pytest.raises(ValueError, match="weighting"):
        fit_spectrum(Spectrum(freq, Z), "degraded_coating", weighting="l2")


def test_fit_spectrum_raises_runtimeerror_on_failure():
    freq, Z = synth_degraded(noise=0.0)
    with pytest.raises(RuntimeError, match="fit failed"):
        # wrong number of initial-guess values for the circuit
        fit_spectrum(Spectrum(freq, Z), "intact_coating", initial_guess=[1.0, 2.0])


def test_seed_from_previous_defaults_to_false():
    sig = inspect.signature(fit_series)
    assert sig.parameters["seed_from_previous"].default is False


def test_fit_series_shape_and_frame():
    series = _degraded_series()
    fits = fit_series(series, "degraded_coating", validate=False)

    assert isinstance(fits, FitSeries)
    assert len(fits) == len(series)
    np.testing.assert_allclose(fits.times, [0, 24, 72, 168])

    df = fits.to_frame()
    assert len(df) == len(series)
    for col in ("R_po", "R_po_err", "Q_c", "kk_passed", "immersion_time_h"):
        assert col in df.columns
    assert df["kk_passed"].isna().all()  # validate=False


def test_fit_series_validates_without_skipping():
    series = _degraded_series(noise=0.0)
    # make the last spectrum non-stationary / non-KK
    bad = series[-1]
    Zb = bad.Z.copy()
    Zb[bad.freq < np.median(bad.freq)] *= 1.6
    specs = list(series[:-1]) + [
        Spectrum(bad.freq, Zb, sample_id="P1", immersion_time_h=bad.immersion_time_h)
    ]
    series = SpectrumSeries(specs)

    fits = fit_series(series, "degraded_coating", validate=True)

    assert len(fits) == len(series)  # nothing skipped
    assert fits[-1].kk_passed is False
    assert all(isinstance(r.kk_passed, bool) for r in fits)
    # the flagged point was still fit
    assert fits[-1].success


@pytest.mark.filterwarnings("ignore:divide by zero", "ignore:invalid value")
def test_fit_series_soft_failure_continues():
    freq, Z = synth_degraded(noise=0.0)
    good = Spectrum(freq, Z, sample_id="P1", immersion_time_h=0.0)
    Zbad = Z.copy()
    Zbad[5] = 0.0 + 0.0j  # non-finite modulus weight -> optimiser bails
    bad = Spectrum(freq, Zbad, sample_id="P1", immersion_time_h=24.0)
    fits = fit_series(SpectrumSeries([good, bad]), "intact_coating", validate=False)

    assert len(fits) == 2
    assert fits[0].success
    assert not fits[1].success
    assert "fail" in fits[1].message.lower()
    assert np.isnan(fits[1].rmse)
    assert list(fits.successful.times) == [0.0]
