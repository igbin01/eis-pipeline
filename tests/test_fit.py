"""Tests for eispipe.fit."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from eispipe.fit import FitResult, FitSeries, fit_series, fit_spectrum
from eispipe.models import get_model
from eispipe.spectrum import Spectrum, SpectrumSeries
from eispipe.validate import kramers_kronig
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
    # true params: R_e=40, Q_c=2e-7, n_c=0.93, R_po=8000, Q_dl=4e-6, n_dl=0.82,
    # R_ct=2e6 (see synth_degraded's default). On noise-free data the fit
    # should recover every one of them tightly -- Q_c and n_c matter because
    # every metric in eispipe.metrics derives from them.
    assert res.params["R_e"] == pytest.approx(40.0, rel=0.01)
    assert res.params["Q_c"] == pytest.approx(2e-7, rel=0.01)
    assert res.params["n_c"] == pytest.approx(0.93, rel=0.01)
    assert res.params["R_po"] == pytest.approx(8_000.0, rel=0.01)
    assert res.params["Q_dl"] == pytest.approx(4e-6, rel=0.01)
    assert res.params["n_dl"] == pytest.approx(0.82, rel=0.01)
    assert res.params["R_ct"] == pytest.approx(2e6, rel=0.01)
    assert res.rmse < 1.0
    assert res.Z_fit.shape == freq.shape
    assert res.area_cm2 == 10.0


def test_fit_spectrum_param_errors_finite_and_scale_with_noise():
    true_params = (40.0, 2e-7, 0.93, 8_000.0, 4e-6, 0.82, 2e6)

    def _errors(noise):
        freq, Z = synth_degraded(params=true_params, noise=noise, seed=3)
        spec = Spectrum(freq, Z, sample_id="P1", immersion_time_h=0.0)
        res = fit_spectrum(spec, "degraded_coating")
        assert res.success
        return res.param_errors

    low = _errors(0.005)
    high = _errors(0.05)

    for name in ("R_e", "Q_c", "n_c", "R_po", "Q_dl", "n_dl", "R_ct"):
        assert np.isfinite(low[name]) and low[name] > 0
        assert np.isfinite(high[name]) and high[name] > 0
        assert high[name] > low[name]


def test_fit_spectrum_converges_when_Rct_dominates_Rpo():
    # Regression for the models.py guess-heuristic bug: with R_ct/R_po = 250
    # and a sweep reaching 1e-2 Hz, the old whole-span R_po guess (~1e6, off
    # by ~100x) sent the optimiser to a spurious local minimum even with
    # noise=0. This uses the default data-driven guess end to end (no
    # initial_guess override) -- the case that broke before the fix.
    freq, Z = synth_degraded(
        n=50, params=(40.0, 2e-7, 0.93, 8_000.0, 4e-6, 0.82, 2e6),
        f_hi=10 ** 4.5, f_lo=1e-2,
    )
    spec = Spectrum(freq, Z, sample_id="P1", immersion_time_h=0.0)

    res = fit_spectrum(spec, "degraded_coating")

    # The 1e-2 Hz sweep doesn't fully resolve the R_ct plateau, so this
    # checks convergence to the right ballpark, not tight precision -- the
    # point is that it no longer lands on the ~1e6-off spurious minimum.
    assert res.success
    assert res.params["R_po"] == pytest.approx(8_000.0, rel=0.1)
    assert res.params["R_ct"] == pytest.approx(2e6, rel=0.2)


def test_predict_uses_named_params_not_dict_order():
    from impedance.models.circuits import CustomCircuit

    cm = get_model("intact_coating")
    true_values = {"R_e": 40.0, "R_po": 8_000.0, "Q_c": 2e-7, "n_c": 0.9}
    ordered_values = [true_values[n] for n in cm.param_names]
    cc = CustomCircuit(cm.circuit, initial_guess=ordered_values)
    freq = np.logspace(4, -1, 20)
    Z_expected = cc.predict(freq, use_initial=True)

    # deliberately not in cm.param_names order
    shuffled_params = {k: true_values[k] for k in ("n_c", "R_po", "R_e", "Q_c")}
    assert tuple(shuffled_params) != cm.param_names
    shuffled_errors = {k: 0.1 for k in shuffled_params}

    res = FitResult(
        model_name="intact_coating",
        circuit=cm.circuit,
        params=shuffled_params,
        param_errors=shuffled_errors,
        freq=freq,
        Z_data=Z_expected,
        Z_fit=Z_expected,
        rmse=0.0,
        chi2_norm=0.0,
        weighting="none",
    )

    Z_pred = res.predict(freq)
    np.testing.assert_allclose(Z_pred, Z_expected, rtol=1e-6)


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


def test_fit_series_on_progress_callback_receives_index_total_and_result():
    series = _degraded_series()
    seen = []

    def on_progress(index, total, result):
        seen.append((index, total, result, result.success))

    fits = fit_series(series, "degraded_coating", validate=False, on_progress=on_progress)

    assert isinstance(fits, FitSeries)
    assert len(seen) == len(series)
    assert [idx for idx, _, _, _ in seen] == list(range(len(series)))
    assert all(total == len(series) for _, total, _, _ in seen)
    assert [result.immersion_time_h for _, _, result, _ in seen] == [0.0, 24.0, 72.0, 168.0]


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


# ---------------------------------------------------------------------------
# x_scale regression: Policastro et al. 2023 Table 2 coatings at GOhm scale
# ---------------------------------------------------------------------------

POLICASTRO_AREA_CM2 = 14.6
# t (h): (R_po ohm*cm^2, Y0 S*s^n*cm^-2, m) from Table 2, 0.01 M NaCl, 5 degC
POLICASTRO_TABLE2 = {
    0: (22.6e9, 0.08e-9, 0.946),
    840: (3.07e9, 0.19e-9, 0.898),
    3192: (1.31e9, 0.29e-9, 0.872),
}
POLICASTRO_R_E = 4.0  # Table 1, R_Omega


def _policastro_spectrum(t_h, noise=0.0, seed=0):
    """intact_coating spectrum in raw ohms on the paper's grid (1e5..0.1 Hz, 10/decade)."""
    from impedance.models.circuits import CustomCircuit

    R_po, Y0, m = POLICASTRO_TABLE2[t_h]
    true = {
        "R_e": POLICASTRO_R_E,
        "R_po": R_po / POLICASTRO_AREA_CM2,
        "Q_c": Y0 * POLICASTRO_AREA_CM2,
        "n_c": m,
    }
    freq = np.logspace(5, -1, 61)
    cc = CustomCircuit(
        "R0-p(R1,CPE1)",
        initial_guess=[true["R_e"], true["R_po"], true["Q_c"], true["n_c"]],
    )
    Z = cc.predict(freq, use_initial=True)
    if noise:
        rng = np.random.default_rng(seed)
        Z = Z * (1 + noise * (rng.normal(size=Z.size) + 1j * rng.normal(size=Z.size)))
    spec = Spectrum(freq, Z, immersion_time_h=t_h, area_cm2=POLICASTRO_AREA_CM2)
    return spec, true


@pytest.mark.filterwarnings("ignore:Simulating circuit")
@pytest.mark.parametrize("t_h", sorted(POLICASTRO_TABLE2))
def test_fit_recovers_GOhm_coating_noise_free(t_h):
    spec, true = _policastro_spectrum(t_h)
    res = fit_spectrum(spec, "intact_coating")  # model's own data-driven guess
    assert res.success
    for name in ("R_po", "Q_c", "n_c"):
        assert res.params[name] == pytest.approx(true[name], rel=1e-4), name


@pytest.mark.filterwarnings("ignore:Simulating circuit")
@pytest.mark.parametrize("t_h", sorted(POLICASTRO_TABLE2))
def test_fit_recovers_GOhm_coating_with_noise(t_h):
    spec, true = _policastro_spectrum(t_h, noise=0.01, seed=42)
    res = fit_spectrum(spec, "intact_coating")
    assert res.success
    for name in ("Q_c", "n_c"):
        assert res.params[name] == pytest.approx(true[name], rel=0.05), name


@pytest.mark.filterwarnings("ignore:Simulating circuit")
def test_x_scale_is_overridable_and_default_matters():
    spec, true = _policastro_spectrum(0)
    unscaled = fit_spectrum(spec, "intact_coating", x_scale=1.0)
    # scipy's unscaled default stalls at GOhm scale -- the reason for "jac"
    assert unscaled.params["Q_c"] != pytest.approx(true["Q_c"], rel=0.01)
    assert inspect.signature(fit_spectrum).parameters["x_scale"].default == "jac"
    assert inspect.signature(fit_series).parameters["x_scale"].default == "jac"


def test_fit_series_passes_x_scale_through(monkeypatch):
    import eispipe.fit as fit_mod

    seen = []
    real = fit_mod.fit_spectrum

    def spy(*args, **kwargs):
        seen.append(kwargs.get("x_scale"))
        return real(*args, **kwargs)

    monkeypatch.setattr(fit_mod, "fit_spectrum", spy)
    freq, Z = synth_degraded(noise=0.0)
    series = SpectrumSeries([Spectrum(freq, Z, immersion_time_h=0.0)])
    fit_series(series, "intact_coating", validate=False, x_scale=1.0)
    fit_series(series, "intact_coating", validate=False)
    assert seen == [1.0, "jac"]


@pytest.mark.filterwarnings("ignore:Simulating circuit")
@pytest.mark.parametrize("noise", [0.0, 0.01])
@pytest.mark.parametrize("t_h", sorted(POLICASTRO_TABLE2))
def test_kramers_kronig_passes_on_GOhm_coating(t_h, noise):
    spec, _ = _policastro_spectrum(t_h, noise=noise, seed=42)
    kk = kramers_kronig(spec)
    print(
        f"\nKK t={t_h} h noise={noise:.0%}: passed={kk.passed} M={kk.M} "
        f"mu={kk.mu:.3f} rms_resid={kk.rms_resid:.3e} max_resid={kk.max_resid:.3e} "
        f"threshold={kk.threshold:.3e}"
    )
    assert kk.passed is True
    assert np.isfinite(kk.rms_resid) and kk.rms_resid <= kk.threshold
