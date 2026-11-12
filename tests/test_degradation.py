"""Tests for eispipe.degradation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from eispipe.degradation import (
    MIN_POINTS,
    RateEstimate,
    ThresholdForecast,
    describe_trend,
    detect_rate_change,
    estimate_rate,
    per_year,
    simulate_series,
    time_to_threshold,
)

# ---------------------------------------------------------------------------
# estimate_rate
# ---------------------------------------------------------------------------


def test_estimate_rate_exact_on_noise_free_line():
    t = np.linspace(0.0, 10.0, 21)
    est = estimate_rate(t, 0.3 + 0.12 * t)

    assert est.rate == pytest.approx(0.12, abs=1e-12)
    assert est.low == pytest.approx(0.12, abs=1e-12)
    assert est.high == pytest.approx(0.12, abs=1e-12)
    assert est.intercept == pytest.approx(0.3, abs=1e-12)
    assert (est.n, est.level, est.significant) == (21, 0.95, True)


@pytest.mark.slow
def test_estimate_rate_95_interval_covers_true_rate_in_90_to_99_percent():
    covered = 0
    for seed in range(200):
        s = simulate_series("constant", seed=seed)  # Gaussian noise plus outliers
        est = estimate_rate(s.times, s.values)
        covered += est.low <= s.truth["rate"] <= est.high

    assert 0.90 <= covered / 200 <= 0.99


def test_theil_sen_resists_outliers_that_move_least_squares():
    t = np.arange(30.0)
    y = 1.0 + 0.1 * t
    y[-3:] += 5.0  # three wild readings at the end

    ols_slope = np.polyfit(t, y, 1)[0]
    assert abs(ols_slope - 0.1) > 0.05  # least squares is dragged by the outliers

    assert estimate_rate(t, y).rate == pytest.approx(0.1, abs=1e-12)


def test_estimate_rate_window_uses_only_recent_points():
    s = simulate_series("step", noise=0.0, outlier_fraction=0.0)
    window = s.times[-1] - s.truth["change_time"] - 1.0  # strictly after the change

    est = estimate_rate(s.times, s.values, window=window)

    assert est.rate == pytest.approx(s.truth["rate_after"], abs=1e-12)
    assert est.n == int(np.sum(s.times >= s.times[-1] - window))


def test_estimate_rate_too_few_points_gives_nan():
    est = estimate_rate([0.0, 1.0], [1.0, 2.0])
    assert math.isnan(est.rate) and math.isnan(est.low) and math.isnan(est.high)
    assert est.n == 2
    assert not est.significant

    windowed = estimate_rate(np.arange(10.0), np.arange(10.0), window=1.0)  # 2 points left
    assert math.isnan(windowed.rate)


def test_per_year_conversions():
    assert per_year(1.0, "h") == pytest.approx(8766.0)
    assert per_year(1.0, "d") == pytest.approx(365.25)
    assert per_year(2.0, "year") == 2.0
    yearly = RateEstimate(rate=0.001, low=0.0008, high=0.0012, intercept=0.0, n=10).per_year("d")
    assert (yearly.rate, yearly.low, yearly.high) == pytest.approx((0.36525, 0.2922, 0.4383))
    with pytest.raises(ValueError, match="time unit"):
        per_year(1.0, "fortnight")


def test_vectorised_theil_sen_matches_scipy():
    from eispipe.degradation import _theil_sen

    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0, 10, 25))
    Y = 0.2 * t + rng.normal(0, 0.3, size=(5, t.size))

    slopes, intercepts = _theil_sen(t, Y)

    for row, slope, intercept in zip(Y, slopes, intercepts, strict=True):
        ref = stats.theilslopes(row, t, method="joint")
        assert slope == pytest.approx(ref.slope, abs=1e-12)
        assert intercept == pytest.approx(ref.intercept, abs=1e-12)


def test_estimate_rate_intercept_uses_all_points():
    # the "joint" intercept, median(y - rate * t), rather than one built from
    # median(y) alone, which hinges on the one or two middle readings
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 10.0, 30)
    y = 0.4 + 0.1 * t + rng.normal(0, 0.05, t.size)

    est = estimate_rate(t, y)

    joint = stats.theilslopes(y, t, method="joint").intercept
    separate = stats.theilslopes(y, t, method="separate").intercept
    assert joint != pytest.approx(separate, abs=1e-9)
    assert est.intercept == pytest.approx(joint, abs=1e-12)


# ---------------------------------------------------------------------------
# detect_rate_change
# ---------------------------------------------------------------------------


def test_detect_rate_change_exact_on_noise_free_step():
    s = simulate_series("step", noise=0.0, outlier_fraction=0.0, n=61, duration=10.0,
                        change_time=6.0)

    change = detect_rate_change(s.times, s.values, min_segment=5)

    assert change is not None
    assert change.time == pytest.approx(6.0, abs=1e-9)
    assert change.rate_before == pytest.approx(s.truth["rate"], abs=1e-9)
    assert change.rate_after == pytest.approx(s.truth["rate_after"], abs=1e-9)


def test_detect_rate_change_finds_change_within_tolerance():
    for seed in range(20):
        s = simulate_series("step", seed=seed, outlier_fraction=0.0)

        change = detect_rate_change(s.times, s.values, min_segment=5)

        assert change is not None, seed
        assert abs(change.time - s.truth["change_time"]) <= 0.5, seed
        assert change.rate_before == pytest.approx(s.truth["rate"], abs=0.015)
        assert change.rate_after == pytest.approx(s.truth["rate_after"], abs=0.015)
        assert change.evidence >= 10.0


def test_detect_rate_change_does_not_invent_changes_on_straight_data():
    invented = []
    for seed in range(50):
        s = simulate_series("constant", seed=seed, outlier_fraction=0.0)
        if detect_rate_change(s.times, s.values, min_segment=5) is not None:
            invented.append(seed)

    assert invented == []


def test_detect_rate_change_noise_free_straight_line_has_no_change():
    t = np.linspace(0.0, 10.0, 40)
    assert detect_rate_change(t, 2.0 + 0.05 * t, min_segment=5) is None


# ---------------------------------------------------------------------------
# time_to_threshold
# ---------------------------------------------------------------------------


def test_time_to_threshold_exact_on_noise_free_line():
    t = np.linspace(0.0, 10.0, 41)

    fc = time_to_threshold(t, 0.1 * t, threshold=1.5, horizon=20.0)

    assert fc.ok and fc.reason is None
    assert fc.time_to_cross == pytest.approx(5.0, abs=1e-9)
    assert fc.low == pytest.approx(5.0, abs=1e-9)
    assert fc.high == pytest.approx(5.0, abs=1e-9)
    assert fc.crossing_time == pytest.approx(15.0, abs=1e-9)


def test_time_to_threshold_falling_series_towards_a_lower_limit():
    t = np.linspace(0.0, 10.0, 21)

    fc = time_to_threshold(t, 10.0 - 0.5 * t, threshold=2.0, horizon=20.0)

    assert fc.time_to_cross == pytest.approx(6.0, abs=1e-9)


@pytest.mark.slow
def test_forecast_95_interval_coverage_is_close_to_nominal():
    # two-sided on purpose: a conservative interval (~99%) fails as well as a
    # too-narrow one
    hits, n_sims = 0, 400
    for seed in range(n_sims):
        s = simulate_series("constant", n=40, rate=0.1, seed=seed)
        true_time_to_cross = (1.5 - s.truth["start_value"]) / s.truth["rate"] - s.times[-1]

        fc = time_to_threshold(s.times, s.values, threshold=1.5, horizon=50.0,
                               n_boot=500, seed=seed)

        assert fc.ok, (seed, fc.reason)
        hits += fc.low <= true_time_to_cross <= fc.high

    assert 0.90 <= hits / n_sims <= 0.98


def test_non_significant_trend_returns_none_with_reason():
    t = np.arange(20.0)
    y = 1.0 + 0.05 * (-1.0) ** np.arange(20)  # zig-zag with no trend

    fc = time_to_threshold(t, y, threshold=2.0, horizon=100.0)

    assert not fc.ok
    assert (fc.time_to_cross, fc.low, fc.high, fc.crossing_time) == (None, None, None, None)
    assert "not statistically significant" in fc.reason


def test_trend_heading_away_returns_none_with_reason():
    t = np.linspace(0.0, 10.0, 30)

    fc = time_to_threshold(t, 1.0 - 0.05 * t, threshold=2.0, horizon=100.0)

    assert fc.time_to_cross is None
    assert "heads away" in fc.reason


def test_threshold_already_crossed_returns_none_with_reason():
    t = np.linspace(0.0, 10.0, 30)

    fc = time_to_threshold(t, 0.1 * t, threshold=0.5, horizon=100.0)

    assert fc.time_to_cross is None
    assert "already crossed" in fc.reason


def test_crossing_beyond_horizon_returns_none_with_reason():
    t = np.linspace(0.0, 10.0, 30)

    fc = time_to_threshold(t, 0.1 * t, threshold=1.5, horizon=2.0)

    assert fc.time_to_cross is None
    assert "horizon" in fc.reason


def test_time_to_threshold_too_few_points_returns_none_with_reason():
    fc = time_to_threshold([0.0, 1.0], [0.0, 0.1], threshold=1.0, horizon=100.0)

    assert fc.time_to_cross is None
    assert "too few points" in fc.reason


def test_time_to_threshold_is_reproducible_with_a_seed():
    s = simulate_series("constant", seed=4)
    a = time_to_threshold(s.times, s.values, threshold=1.5, horizon=50.0, n_boot=300, seed=1)
    b = time_to_threshold(s.times, s.values, threshold=1.5, horizon=50.0, n_boot=300, seed=1)
    assert (a.time_to_cross, a.low, a.high) == (b.time_to_cross, b.low, b.high)


# ---------------------------------------------------------------------------
# simulate_series
# ---------------------------------------------------------------------------


def test_simulate_series_kinds_are_exact_without_noise():
    const = simulate_series("constant", noise=0.0, start_value=0.2, rate=0.1)
    np.testing.assert_allclose(const.values, 0.2 + 0.1 * const.times)

    step = simulate_series("step", noise=0.0, rate=0.1, rate_after=0.02, change_time=6.0)
    slopes = np.diff(step.values) / np.diff(step.times)
    np.testing.assert_allclose(slopes[step.times[1:] <= 6.0], 0.1)
    np.testing.assert_allclose(slopes[step.times[:-1] >= 6.0], 0.02)

    accel = simulate_series("accelerating", noise=0.0, rate=0.1, acceleration=0.02)
    np.testing.assert_allclose(accel.values, 0.1 * accel.times + 0.01 * accel.times**2)


def test_simulate_series_seed_noise_and_outliers():
    a = simulate_series("constant", seed=1)
    b = simulate_series("constant", seed=1)
    c = simulate_series("constant", seed=2)
    np.testing.assert_array_equal(a.values, b.values)
    assert not np.array_equal(a.values, c.values)

    big = simulate_series("constant", n=2000, noise=0.01, outlier_fraction=0.05,
                          outlier_size=10.0, seed=0)
    assert 0.03 <= big.outliers.mean() <= 0.07
    residual = big.values - big.clean
    assert np.all(np.abs(residual[big.outliers]) > 5 * 0.01)
    assert np.std(residual[~big.outliers]) == pytest.approx(0.01, rel=0.1)

    with pytest.raises(ValueError, match="kind"):
        simulate_series("wobbly")


# ---------------------------------------------------------------------------
# describe_trend
# ---------------------------------------------------------------------------

RISING = RateEstimate(rate=0.12, low=0.09, high=0.15, intercept=0.0, n=30)


def _forecast(rate=RISING, **fields) -> ThresholdForecast:
    base = dict(threshold=0.5, horizon=10.0, level=0.95, rate=rate, time_to_cross=3.2,
                low=2.4, high=4.6, crossing_time=13.2, reason=None, n_boot=1000)
    base.update(fields)
    return ThresholdForecast(**base)


def test_describe_trend_example_wording():
    assert describe_trend(_forecast(), quantity="Metal loss", unit="mm", time_unit="year") == (
        "Metal loss is rising about 0.12 mm/year (0.09 to 0.15). "
        "At this pace it reaches the 0.5 mm limit in about 3.2 years (2.4 to 4.6)."
    )


@pytest.mark.parametrize(
    "result, quantity, unit, time_unit, expected",
    [
        (
            RateEstimate(rate=-0.034, low=-0.041, high=-0.027, intercept=5.0, n=40),
            "Film resistance", "kOhm", "day",
            "Film resistance is falling about 0.034 kOhm/day (0.027 to 0.041).",
        ),
        (
            RateEstimate(rate=0.004, low=-0.01, high=0.018, intercept=0.0, n=40),
            "Metal loss", "mm", "year",
            "Metal loss shows no clear trend: about 0.004 mm/year (-0.010 to 0.018).",
        ),
        (
            RateEstimate(rate=math.nan, low=math.nan, high=math.nan, intercept=math.nan, n=2),
            "Metal loss", "mm", "year",
            "Metal loss: too few points to estimate a trend.",
        ),
        (
            _forecast(time_to_cross=None, low=None, high=None, crossing_time=None,
                      reason="the trend heads away from the threshold"),
            "Metal loss", "mm", "year",
            "Metal loss is rising about 0.12 mm/year (0.09 to 0.15). "
            "It is not forecast to reach the 0.5 mm limit: the trend heads away from the threshold.",
        ),
        (
            _forecast(high=math.inf),
            "Metal loss", "mm", "year",
            "Metal loss is rising about 0.12 mm/year (0.09 to 0.15). "
            "At this pace it reaches the 0.5 mm limit in about 3.2 years (2.4 to more than 10).",
        ),
    ],
)
def test_describe_trend_wordings(result, quantity, unit, time_unit, expected):
    assert describe_trend(result, quantity=quantity, unit=unit, time_unit=time_unit) == expected


def test_describe_trend_rejects_other_objects():
    with pytest.raises(TypeError):
        describe_trend(0.12, quantity="Metal loss", unit="mm")


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------

CALLS = {
    "estimate_rate": lambda t, y: estimate_rate(t, y),
    "detect_rate_change": lambda t, y: detect_rate_change(t, y, min_segment=3),
    "time_to_threshold": lambda t, y: time_to_threshold(t, y, threshold=10.0, horizon=100.0),
}
GOOD_T = np.arange(10.0)
GOOD_Y = 0.1 * GOOD_T


@pytest.mark.parametrize("name", CALLS)
@pytest.mark.parametrize(
    "times, values, match",
    [
        (GOOD_T, np.where(GOOD_T == 4, np.nan, GOOD_Y), "finite"),
        (np.where(GOOD_T == 4, np.nan, GOOD_T), GOOD_Y, "finite"),
        (GOOD_T[::-1], GOOD_Y, "increasing"),
        (np.r_[GOOD_T[:5], GOOD_T[4:9]], GOOD_Y, "increasing"),  # repeated time
        (GOOD_T, GOOD_Y[:-1], "same length"),
    ],
    ids=["nan-value", "nan-time", "unsorted", "repeated-time", "length-mismatch"],
)
def test_invalid_series_raise(name, times, values, match):
    with pytest.raises(ValueError, match=match):
        CALLS[name](times, values)


def test_detect_rate_change_needs_enough_points():
    with pytest.raises(ValueError, match="at least"):
        detect_rate_change(np.arange(9.0), np.arange(9.0), min_segment=5)
    with pytest.raises(ValueError, match="min_segment"):
        detect_rate_change(np.arange(20.0), np.arange(20.0), min_segment=2)


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda: estimate_rate(GOOD_T, GOOD_Y, level=1.5), "level"),
        (lambda: estimate_rate(GOOD_T, GOOD_Y, window=0.0), "window"),
        (lambda: time_to_threshold(GOOD_T, GOOD_Y, 10.0, horizon=0.0), "horizon"),
        (lambda: time_to_threshold(GOOD_T, GOOD_Y, 10.0, horizon=5.0, level=0.0), "level"),
        (lambda: time_to_threshold(GOOD_T, GOOD_Y, 10.0, horizon=5.0, n_boot=0), "n_boot"),
    ],
)
def test_invalid_arguments_raise(call, match):
    with pytest.raises(ValueError, match=match):
        call()


def test_min_points_is_three():
    assert MIN_POINTS == 3


# ---------------------------------------------------------------------------
# change detection: power, and optional outlier screening
# ---------------------------------------------------------------------------


def _power(noise, ratio, outlier_fraction, screen=False, seeds=range(200)):
    """Share of runs where a halving-style rate change at mid-series is found
    (reported within 1 time unit of the truth)."""
    found = 0
    for seed in seeds:
        s = simulate_series("step", n=60, duration=10.0, rate=0.1, rate_after=0.1 * ratio,
                            change_time=5.0, noise=noise, outlier_fraction=outlier_fraction,
                            seed=seed)
        change = detect_rate_change(s.times, s.values, min_segment=5, screen_outliers=screen)
        found += change is not None and abs(change.time - 5.0) <= 1.0
    return found / len(seeds)


@pytest.mark.slow
def test_power_floor_clearest_case_without_outliers():
    # noise 0.01, rate drops to a quarter: must essentially always be found
    assert _power(noise=0.01, ratio=0.25, outlier_fraction=0.0) >= 0.98


@pytest.mark.slow
def test_power_floor_clearest_case_with_outliers():
    assert _power(noise=0.01, ratio=0.25, outlier_fraction=0.05) >= 0.95
    assert _power(noise=0.01, ratio=0.25, outlier_fraction=0.05, screen=True) >= 0.98


@pytest.mark.slow
def test_screening_restores_power_under_outliers():
    # rate halves, moderate noise, 5% outliers: plain least squares loses the
    # change in many runs; screening finds it again
    assert _power(noise=0.02, ratio=0.5, outlier_fraction=0.05, screen=True) >= 0.95


@pytest.mark.slow
def test_screening_does_not_invent_changes_on_straight_data_with_outliers():
    invented = sum(
        detect_rate_change(s.times, s.values, min_segment=5, screen_outliers=True) is not None
        for s in (simulate_series("constant", seed=seed) for seed in range(200))
    )
    assert invented <= 2


def test_screening_is_off_by_default_and_reports_what_it_dropped():
    import inspect

    assert inspect.signature(detect_rate_change).parameters["screen_outliers"].default is False

    clean = simulate_series("step", noise=0.0, outlier_fraction=0.0, n=61, change_time=6.0)
    screened = detect_rate_change(clean.times, clean.values, min_segment=5, screen_outliers=True)
    assert screened.n_screened == 0
    assert screened.time == pytest.approx(6.0, abs=1e-9)

    noisy = simulate_series("step", noise=0.02, outlier_fraction=0.0, seed=1)
    values = noisy.values.copy()
    values[[10, 40]] += 1.0  # two wild readings
    plain = detect_rate_change(noisy.times, values, min_segment=5)
    with_screen = detect_rate_change(noisy.times, values, min_segment=5, screen_outliers=True)
    assert plain is None or plain.n_screened == 0
    assert with_screen.n_screened >= 2


# ---------------------------------------------------------------------------
# package exports
# ---------------------------------------------------------------------------


def test_degradation_api_is_exported_from_the_package():
    import eispipe
    from eispipe import degradation

    names = [
        "estimate_rate", "detect_rate_change", "time_to_threshold", "simulate_series",
        "describe_trend", "RateEstimate", "RateChange", "ThresholdForecast", "SimulatedSeries",
    ]
    for name in names:
        assert getattr(eispipe, name) is getattr(degradation, name), name
        assert name in eispipe.__all__, name


# ---------------------------------------------------------------------------
# correlated noise: dependence="auto" (default) vs "independent"
# ---------------------------------------------------------------------------


def _ar1(rho, seed, n=60, outlier_fraction=0.0):
    return simulate_series("constant", n=n, rate=0.1, noise=0.02, ar1_rho=rho,
                           outlier_fraction=outlier_fraction, seed=seed)


def test_simulate_series_ar1_noise_has_requested_correlation_and_size():
    s = simulate_series("constant", n=20_000, noise=0.02, ar1_rho=0.8, outlier_fraction=0.0, seed=0)
    e = s.values - s.clean
    assert np.std(e) == pytest.approx(0.02, rel=0.05)
    assert np.corrcoef(e[:-1], e[1:])[0, 1] == pytest.approx(0.8, abs=0.02)
    # rho 0 is the old independent noise, draw for draw
    np.testing.assert_array_equal(simulate_series("constant", seed=3, ar1_rho=0.0).values,
                                  simulate_series("constant", seed=3).values)


def test_rho_is_estimated_within_tolerance():
    # average over 200 runs of 60 readings within 0.05 of the truth
    for rho in (0.5, 0.8):
        est = [estimate_rate(s.times, s.values).rho for s in (_ar1(rho, seed) for seed in range(200))]
        assert np.mean(est) == pytest.approx(rho, abs=0.05), rho
    independent = [estimate_rate(s.times, s.values).rho
                   for s in (_ar1(0.0, seed) for seed in range(200))]
    assert np.mean(independent) < 0.1
    assert min(independent) >= 0.0  # negative correlation is never used to narrow


def test_rate_estimate_reports_rho_and_effective_sample_size():
    s = _ar1(0.8, seed=1)
    est = estimate_rate(s.times, s.values)
    assert est.dependence == "auto"
    assert 0.0 < est.rho <= 0.95
    assert est.n_eff == pytest.approx(est.n * (1 - est.rho) / (1 + est.rho))

    old = estimate_rate(s.times, s.values, dependence="independent")
    assert (old.dependence, old.rho, old.n_eff) == ("independent", 0.0, float(old.n))
    assert est.rate == old.rate  # same slope, wider interval
    assert est.high - est.low > old.high - old.low

    fc = time_to_threshold(s.times, s.values, threshold=1.5, horizon=50.0, n_boot=200)
    assert (fc.rho, fc.n_eff) == (fc.rate.rho, fc.rate.n_eff)


def test_independent_mode_reproduces_the_previous_numbers_exactly():
    # recorded from the implementation before dependence= existed
    recorded = {
        5: ((0.10002050872023205, 0.09805924693455706, 0.10194695944737199, -0.005257291640160255),
            (5.054318232446739, 4.84728692361341, 5.2491882879579785)),
        6: ((0.09857411200667665, 0.09702101088855959, 0.10043413502668838, 0.010960317584450305),
            (5.097783014007966, 4.923685787467197, 5.32336132508181)),
    }
    for seed, (rate, forecast) in recorded.items():
        s = simulate_series("constant", seed=seed)
        r = estimate_rate(s.times, s.values, dependence="independent")
        f = time_to_threshold(s.times, s.values, threshold=1.5, horizon=50.0, n_boot=400,
                              seed=seed, dependence="independent")
        assert (r.rate, r.low, r.high, r.intercept) == rate
        assert (f.time_to_cross, f.low, f.high) == forecast


def test_noise_free_data_needs_no_widening():
    t = np.linspace(0.0, 10.0, 41)
    est = estimate_rate(t, 0.1 * t)
    assert est.rho == 0.0 and est.low == pytest.approx(0.1) and est.high == pytest.approx(0.1)


def test_too_few_points_to_estimate_rho():
    t = np.arange(8.0)
    est = estimate_rate(t, 0.1 * t + 0.01 * (-1.0) ** t)
    assert math.isnan(est.rho)
    assert est.n_eff == float(est.n)  # falls back to the independent interval


def test_dependence_must_be_known():
    with pytest.raises(ValueError, match="dependence"):
        estimate_rate(GOOD_T, GOOD_Y, dependence="ar2")
    with pytest.raises(ValueError, match="dependence"):
        time_to_threshold(GOOD_T, GOOD_Y, 10.0, horizon=5.0, dependence="sometimes")


def _coverage(rho, outlier_fraction=0.0, runs=400, n_boot=400):
    rate_hits, fc_hits, rate_w, fc_w = 0, 0, [], []
    for seed in range(runs):
        s = _ar1(rho, seed, outlier_fraction=outlier_fraction)
        est = estimate_rate(s.times, s.values)
        rate_hits += est.low <= 0.1 <= est.high
        rate_w.append(est.high - est.low)
        fc = time_to_threshold(s.times, s.values, threshold=1.5, horizon=50.0, n_boot=n_boot,
                               seed=seed)
        fc_hits += fc.ok and fc.low <= 15.0 - s.times[-1] <= fc.high
        fc_w.append(fc.high - fc.low if fc.ok else np.nan)
    return rate_hits / runs, fc_hits / runs, np.mean(rate_w), np.nanmean(fc_w)


@pytest.mark.slow
def test_auto_coverage_with_independent_noise_stays_93_to_97():
    rate_cov, fc_cov, _, _ = _coverage(0.0, outlier_fraction=0.05)
    assert 0.93 <= rate_cov <= 0.97
    assert 0.93 <= fc_cov <= 0.97


@pytest.mark.slow
def test_auto_coverage_with_correlated_noise():
    _, _, rate_w0, fc_w0 = _coverage(0.0)

    rate_cov, fc_cov, _, _ = _coverage(0.5)
    assert 0.90 <= rate_cov <= 0.98
    assert 0.90 <= fc_cov <= 0.98

    rate_cov, fc_cov, rate_w, fc_w = _coverage(0.8)
    assert rate_cov >= 0.88
    assert fc_cov >= 0.88
    assert rate_w >= 2.0 * rate_w0  # clearly wider than for independent noise
    assert fc_w >= 2.0 * fc_w0


def test_describe_trend_mentions_strong_correlation():
    correlated = RateEstimate(rate=0.12, low=0.05, high=0.19, intercept=0.0, n=60,
                              rho=0.8, n_eff=6.7, dependence="auto")
    assert describe_trend(correlated, quantity="Metal loss", unit="mm") == (
        "Metal loss is rising about 0.12 mm/year (0.05 to 0.19). "
        "Readings are strongly correlated in time, so the range is wide."
    )
    fc = _forecast(rate=correlated, time_to_cross=3.2, low=1.8, high=7.5)
    assert describe_trend(fc, quantity="Metal loss", unit="mm") == (
        "Metal loss is rising about 0.12 mm/year (0.05 to 0.19). "
        "At this pace it reaches the 0.5 mm limit in about 3.2 years (1.8 to 7.5). "
        "Readings are strongly correlated in time, so the range is wide."
    )
    mild = RateEstimate(rate=0.12, low=0.09, high=0.15, intercept=0.0, n=60, rho=0.3,
                        n_eff=32.3, dependence="auto")
    assert describe_trend(mild, quantity="Metal loss", unit="mm") == (
        "Metal loss is rising about 0.12 mm/year (0.09 to 0.15)."
    )


# ---------------------------------------------------------------------------
# detect_rate_change under correlated noise (dependence="auto", the default)
# ---------------------------------------------------------------------------


def test_detect_independent_mode_reproduces_the_previous_results_exactly():
    # recorded from the implementation before dependence= existed here
    recorded = [
        ("step", 7, {}, (6.271186440677965, 0.10340317436079931, -0.0007951833607399428,
                         46.10452841715329, 38, 23, 0)),
        ("step", 8, {"screen_outliers": True},
         (5.932203389830509, 0.10058541513706871, 0.023624918449382845, 104.37258614078024,
          36, 24, 1)),
        ("constant", 9, {}, None),
    ]
    for kind, seed, kw, expected in recorded:
        s = simulate_series(kind, seed=seed)
        c = detect_rate_change(s.times, s.values, 5, dependence="independent", **kw)
        got = None if c is None else (c.time, c.rate_before, c.rate_after, c.evidence,
                                      c.n_before, c.n_after, c.n_screened)
        assert got == expected, (kind, seed)


def test_detect_reports_rho_and_mode():
    import inspect

    assert inspect.signature(detect_rate_change).parameters["dependence"].default == "auto"
    s = simulate_series("step", rate=0.1, rate_after=0.0, change_time=5.0, noise=0.02,
                        ar1_rho=0.5, outlier_fraction=0.0, seed=2)
    auto = detect_rate_change(s.times, s.values, 5)
    old = detect_rate_change(s.times, s.values, 5, dependence="independent")
    assert auto is not None and old is not None
    assert auto.dependence == "auto" and auto.rho > 0
    assert (old.dependence, old.rho) == ("independent", 0.0)
    assert auto.time == old.time  # same best split; only the evidence is discounted
    assert auto.evidence < old.evidence
    with pytest.raises(ValueError, match="dependence"):
        detect_rate_change(s.times, s.values, 5, dependence="maybe")


@pytest.mark.slow
def test_detect_false_changes_on_straight_correlated_data():
    # targets: at most 5% at rho 0.5 and 10% at rho 0.8 (60 readings, no outliers)
    for rho, limit in ((0.5, 0.05), (0.8, 0.10)):
        false = sum(
            detect_rate_change(s.times, s.values, 5) is not None
            for s in (simulate_series("constant", ar1_rho=rho, outlier_fraction=0.0, seed=k)
                      for k in range(200))
        )
        assert false / 200 <= limit, (rho, false)


@pytest.mark.slow
def test_detect_auto_costs_little_power_with_independent_noise():
    for noise, ratio in ((0.02, 0.75), (0.05, 0.5)):
        found = {"auto": 0, "independent": 0}
        for k in range(200):
            s = simulate_series("step", rate=0.1, rate_after=0.1 * ratio, change_time=5.0,
                                noise=noise, outlier_fraction=0.0, seed=k)
            for mode in found:
                c = detect_rate_change(s.times, s.values, 5, dependence=mode)
                found[mode] += c is not None and abs(c.time - 5.0) <= 1.0
        assert found["auto"] >= found["independent"] - 10, (noise, ratio, found)  # <= 5 points


# ---------------------------------------------------------------------------
# short series: the rho estimate must not swing wildly
# ---------------------------------------------------------------------------


def test_short_series_rho_estimate_is_stabilised():
    widen = []
    for k in range(300):
        s = simulate_series("constant", n=10, outlier_fraction=0.0, seed=k)  # independent
        auto = estimate_rate(s.times, s.values)
        indep = estimate_rate(s.times, s.values, dependence="independent")
        widen.append((auto.high - auto.low) / (indep.high - indep.low))
    # unstabilised (n r1 + 2) / (n - 6) widened independent 10-point series ~1.9x
    assert np.mean(widen) < 1.4


def test_rho_estimator_unchanged_from_24_readings():
    from eispipe.degradation import _lag1_correlation

    rng = np.random.default_rng(11)
    for n in (24, 30, 60):
        e = rng.normal(size=n) + 0.6 * np.r_[0.0, rng.normal(size=n - 1)]
        d = e - e.mean()
        r1 = float(d[:-1] @ d[1:] / (d @ d))
        expected = min(max((n * r1 + 2) / (n - 6), 0.0), 0.95)
        assert _lag1_correlation(e, scale=1.0) == pytest.approx(expected, abs=1e-15)
