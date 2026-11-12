"""Rate, change-point and threshold forecasting for drifting measurement series.

Sensor-agnostic: the functions take any series of readings that drift over
time -- metal loss from a corrosion probe, inhibitor film resistance, coating
pore resistance -- as two arrays, ``times`` and ``values``. Rates come out in
value units per time unit of ``times``; :func:`per_year` converts them.

* :func:`estimate_rate` -- robust straight-line rate with a confidence interval.
* :func:`detect_rate_change` -- one point where the rate changed, or ``None``.
* :func:`time_to_threshold` -- when the trend reaches a limit, with an interval,
  or a plain reason why no forecast is given.
* :func:`simulate_series` -- synthetic series with known answers, for testing.
* :func:`describe_trend` -- one plain-language sentence for a report.

Assumptions shared by all functions (in plain language):

* Readings are independent: the scatter of one reading around the trend does
  not depend on the scatter of the previous one. Slowly wandering errors
  (autocorrelation) make every interval here too narrow.
* Over the stretch of data used, the underlying change is a straight line
  (or, for :func:`detect_rate_change`, two straight lines joined at one point).
* Forecasts assume the current trend simply continues; they cannot anticipate
  a process change that has not happened yet.

TODO verify against ASTM G16 and Sen (1968): the Theil-Sen rate and its
confidence interval, and the use of a linear rate for corrosion data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy import stats

__all__ = [
    "MIN_POINTS",
    "RateChange",
    "RateEstimate",
    "SimulatedSeries",
    "ThresholdForecast",
    "describe_trend",
    "detect_rate_change",
    "estimate_rate",
    "per_year",
    "simulate_series",
    "time_to_threshold",
]

#: Fewest points for which a rate and its interval are reported.
MIN_POINTS = 3

#: How many of each time unit make one year (a year is 365.25 days).
UNITS_PER_YEAR = {
    "s": 365.25 * 86_400.0,
    "min": 365.25 * 1_440.0,
    "h": 365.25 * 24.0,
    "d": 365.25,
    "week": 365.25 / 7.0,
    "year": 1.0,
}


def per_year(rate, time_unit: str):
    """Convert a rate per ``time_unit`` into a rate per year (365.25 days).

    ``time_unit`` is one of ``"s"``, ``"min"``, ``"h"``, ``"d"``, ``"week"``,
    ``"year"``. Works on numbers and NumPy arrays.
    """
    try:
        factor = UNITS_PER_YEAR[time_unit]
    except KeyError:
        raise ValueError(
            f"unknown time unit {time_unit!r}; use one of {sorted(UNITS_PER_YEAR)}"
        ) from None
    return rate * factor


# ---------------------------------------------------------------------------
# result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateEstimate:
    """A straight-line rate with a confidence interval.

    ``rate`` is the change in value per unit time (negative when falling);
    ``low``/``high`` bound it at confidence ``level``. ``intercept`` is the
    fitted value at time 0: the median of ``value - rate * time`` over all
    readings. All of them are NaN when there were fewer than
    :data:`MIN_POINTS` points.
    """

    rate: float
    low: float
    high: float
    intercept: float
    n: int
    level: float = 0.95

    @property
    def significant(self) -> bool:
        """True if the interval excludes zero (a real rise or fall)."""
        return bool(self.low > 0 or self.high < 0)

    def per_year(self, time_unit: str) -> RateEstimate:
        """The same estimate with ``rate``, ``low`` and ``high`` per year."""
        factor = per_year(1.0, time_unit)
        return replace(self, rate=self.rate * factor, low=self.low * factor, high=self.high * factor)


@dataclass(frozen=True)
class RateChange:
    """A detected change of rate.

    ``time`` is when the rate changed, ``rate_before``/``rate_after`` the
    slopes of the two joined straight segments, ``evidence`` the drop in BIC
    against a single straight line (larger is stronger), and
    ``n_before``/``n_after`` the points in each segment (the point at the
    change belongs to both). ``n_screened`` is how many readings the optional
    outlier screening left out (0 when screening is off).
    """

    time: float
    rate_before: float
    rate_after: float
    evidence: float
    n_before: int
    n_after: int
    n_screened: int = 0


@dataclass(frozen=True)
class ThresholdForecast:
    """When a trend is expected to reach ``threshold``.

    ``time_to_cross`` (median) and ``low``/``high`` (interval at ``level``)
    are measured from the last reading; ``crossing_time`` is the median on the
    series' own time axis. ``high`` is ``math.inf`` when the upper end lies
    beyond ``horizon``. When no forecast is given, those four are ``None`` and
    ``reason`` says why in plain language.
    """

    threshold: float
    horizon: float
    level: float
    rate: RateEstimate
    time_to_cross: float | None
    low: float | None
    high: float | None
    crossing_time: float | None
    reason: str | None
    n_boot: int

    @property
    def ok(self) -> bool:
        """True if a forecast was made."""
        return self.time_to_cross is not None and math.isfinite(self.time_to_cross)


@dataclass(frozen=True)
class SimulatedSeries:
    """A synthetic series from :func:`simulate_series`.

    ``values`` are the noisy readings, ``clean`` the noise-free truth,
    ``outliers`` a boolean mask of the readings given an outlier, and
    ``truth`` the parameters used (``rate``, ``rate_after``, ``change_time``,
    ``acceleration``, ``start_value``, ``noise``).
    """

    kind: str
    times: np.ndarray
    values: np.ndarray
    clean: np.ndarray
    outliers: np.ndarray
    truth: dict


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _series(times, values) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or t.shape != y.shape:
        raise ValueError("times and values must be 1-D arrays of the same length")
    bad = ~(np.isfinite(t) & np.isfinite(y))
    if bad.any():
        raise ValueError(
            f"times and values must be finite; {int(bad.sum())} point(s) are NaN or "
            f"infinite -- drop missing readings first"
        )
    if t.size > 1 and np.any(np.diff(t) <= 0):
        raise ValueError(
            "times must be strictly increasing; sort the series and merge readings "
            "taken at the same time"
        )
    return t, y


def _check_level(level: float) -> None:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be between 0 and 1, got {level!r}")


def _theil_sen(t: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Theil-Sen slope and intercept for every row of ``Y`` (same ``t`` for all).

    The same estimator as ``scipy.stats.theilslopes(..., method="joint")``:
    slope = median of all pairwise slopes, intercept = median(y - slope * t).
    Vectorised for the bootstrap; memory is bounded by chunking.
    """
    Y = np.atleast_2d(Y)
    i, j = np.triu_indices(t.size, k=1)
    dt = t[j] - t[i]
    slopes = np.empty(Y.shape[0])
    chunk = max(1, int(4_000_000 // max(1, i.size)))
    for start in range(0, Y.shape[0], chunk):
        block = Y[start : start + chunk]
        slopes[start : start + chunk] = np.median((block[:, j] - block[:, i]) / dt, axis=1)
    intercepts = np.median(Y - slopes[:, None] * t, axis=1)
    return slopes, intercepts


# ---------------------------------------------------------------------------
# rate
# ---------------------------------------------------------------------------


def estimate_rate(times, values, window=None, *, level=0.95) -> RateEstimate:
    """Robust straight-line rate of change with a confidence interval.

    The rate is the Theil-Sen slope (``scipy.stats.theilslopes``): the median
    of the slopes between every pair of readings. A few wild readings barely
    move it, unlike an ordinary least-squares slope. The interval at ``level``
    (default 95%) is the distribution-free one SciPy computes from the ranks of
    those pairwise slopes. The intercept is the median of
    ``value - rate * time`` over all readings.

    ``window`` (same units as ``times``) restricts the fit to readings taken
    within that span before the last one, e.g. the last 90 days. With fewer
    than :data:`MIN_POINTS` readings the rate and interval are NaN.

    Assumptions: the change is a straight line over the readings used; the
    scatter around it is independent from reading to reading and does not
    change over time. If the rate itself changed (see
    :func:`detect_rate_change`), use ``window`` or pass only the readings after
    the change.

    TODO verify against ASTM G16 and Sen (1968).
    """
    _check_level(level)
    t, y = _series(times, values)
    if window is not None:
        if not window > 0:
            raise ValueError(f"window must be > 0, got {window!r}")
        if t.size:
            keep = t >= t[-1] - window
            t, y = t[keep], y[keep]
    if t.size < MIN_POINTS:
        nan = math.nan
        return RateEstimate(rate=nan, low=nan, high=nan, intercept=nan, n=int(t.size), level=level)
    # "joint" intercept: median(y - rate * t) uses every reading; the
    # "separate" one, median(y) - rate * median(t), hinges on the one or two
    # middle readings and made forecast intervals conservative.
    fit = stats.theilslopes(y, t, alpha=level, method="joint")
    return RateEstimate(
        rate=float(fit.slope),
        low=float(fit.low_slope),
        high=float(fit.high_slope),
        intercept=float(fit.intercept),
        n=int(t.size),
        level=level,
    )


# ---------------------------------------------------------------------------
# change point
# ---------------------------------------------------------------------------


def _sse(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid), beta


#: Outlier screening: neighbours on each side for the local median, and the
#: cutoff in robust standard deviations (1.4826 * median absolute deviation).
_SCREEN_HALF_WINDOW = 3
_SCREEN_CUTOFF = 3.5


def _screen_outliers(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean mask of readings to keep.

    Removes the overall Theil-Sen line, then compares each reading with the
    median of itself and its 3 neighbours on each side; a reading is dropped
    if that difference is more than 3.5 robust standard deviations from the
    typical difference. The local median follows a genuine bend in the trend,
    so a real change of rate is not mistaken for outliers.
    """
    fit = stats.theilslopes(y, t, method="joint")
    detrended = y - (fit.intercept + fit.slope * t)
    h = _SCREEN_HALF_WINDOW
    local = np.array([np.median(detrended[max(0, i - h) : i + h + 1]) for i in range(t.size)])
    deviation = detrended - local
    centre = np.median(deviation)
    sigma = 1.4826 * np.median(np.abs(deviation - centre))
    if sigma <= 1e-9 * max(float(np.ptp(y)), 1e-300):
        return np.ones(t.size, dtype=bool)  # (near) noise-free: nothing to screen
    return np.abs(deviation - centre) <= _SCREEN_CUTOFF * sigma


def detect_rate_change(
    times, values, min_segment, *, min_bic_drop=10.0, screen_outliers=False
) -> RateChange | None:
    """Find one point where the rate changed, or return ``None``.

    Every reading that leaves at least ``min_segment`` readings on each side
    is tried as the change time. For each, two straight segments joined at
    that time are fitted by least squares (the value is continuous; only the
    slope changes, as when an inhibitor dose slows corrosion). The best
    candidate is kept only if it beats a single straight line clearly.

    Why BIC and not an F-test: the change time is chosen by searching all
    candidates for the best fit. A textbook F-test p-value assumes the time
    was fixed in advance, so after the search it calls changes "significant"
    on straight, noisy data far more often than its nominal level. BIC
    instead charges the search for the change time as an extra fitted
    parameter:

        BIC = n * ln(SSE / n) + k * ln(n)

    with ``n`` the number of readings, ``SSE`` the sum of squared residuals,
    and ``k`` the parameter count: 2 for one straight line (intercept, slope)
    and 4 for two joined segments (intercept, slope before, change of slope,
    change time). The noise variance is a parameter of both models, so it
    cancels and is not counted. ``evidence`` is BIC(line) - BIC(segments).

    A change is reported only if that drop is at least ``min_bic_drop``
    (default 10). The value was chosen from simulation, not taken from a
    published scale (TODO verify against Kass and Raftery (1995)). In
    ``scripts/validate_degradation.py`` (500 straight series each, no
    screening): threshold 2 gave 6 false changes without outliers and 25 with
    5% outliers; 6 gave 0 and 13; 10 gave 0 and 3; 15 gave 0 and 2. Raising
    it from 6 to 10 cost no power for a moderate change (rate to 0.75x at
    noise 0.02: 92.8% found at both); 15 cut power for a weak one (rate
    halving at noise 0.05: 79% at 10, 65% at 15).

    ``screen_outliers=True`` (off by default) first drops readings that sit
    far from their neighbours: the overall Theil-Sen line is removed, each
    reading is compared with the median of itself and its 3 neighbours on
    each side, and readings more than 3.5 robust standard deviations away are
    left out. ``n_screened`` reports how many. Least squares is sensitive to
    outliers, and screening undoes most of the damage (below).

    How often a real change is found (``scripts/validate_degradation.py``:
    60 readings over 10 time units, rate 0.1 per unit changing at
    mid-series, "found" = reported within 1 time unit of the truth, 200 runs
    per case):

    * Without outliers, a rate that drops to a quarter or to half is found in
      every run at noise 0.01 and 0.02; at noise 0.05 in 97% (quarter) and
      78% (half). A drop to three quarters is found in 100%, 94% and 5% of
      runs at noise 0.01, 0.02 and 0.05 -- too small to see at high noise.
    * With 5% outliers and no screening, power falls sharply: a halving at
      noise 0.02 is found in 77% of runs, a drop to a quarter at noise 0.05
      in 19%.
    * With 5% outliers and ``screen_outliers=True``, power returns to within
      a few points of the outlier-free figures (100% and 96% for those two
      cases).
    * False changes on straight data with 5% outliers: 6 in 500 without
      screening, 0 in 500 with it; without outliers, 1 in 500 either way.

    Assumptions: at most one change; straight segments; independent scatter
    with the same spread throughout. Without screening, a few wild readings
    can create or hide a change.

    Raises ``ValueError`` if ``min_segment < 3`` or there are fewer than
    ``2 * min_segment`` readings.
    """
    t, y = _series(times, values)
    if int(min_segment) != min_segment or min_segment < MIN_POINTS:
        raise ValueError(f"min_segment must be an integer >= {MIN_POINTS}, got {min_segment!r}")
    if not min_bic_drop >= 0:
        raise ValueError(f"min_bic_drop must be >= 0, got {min_bic_drop!r}")
    if t.size < 2 * min_segment:
        raise ValueError(
            f"need at least 2 * min_segment = {2 * min_segment} readings, got {t.size}"
        )
    n_screened = 0
    if screen_outliers:
        keep = _screen_outliers(t, y)
        if keep.sum() >= 2 * min_segment:  # never screen below what the search needs
            n_screened = int((~keep).sum())
            t, y = t[keep], y[keep]
    n = t.size
    scale = max(float(np.ptp(y)), float(np.max(np.abs(y))))
    if scale == 0.0:
        return None  # all zero: nothing changes
    # Treat residuals at floating-point noise level as an exact fit, so a
    # perfect straight line is not "improved" by rounding error.
    sse_floor = n * (1e-9 * scale) ** 2

    tc = t - t.mean()
    ones = np.ones(n)
    sse_line, _ = _sse(np.column_stack([ones, tc]), y)

    best = None
    for k in range(min_segment - 1, n - min_segment + 1):
        hinge = np.maximum(0.0, t - t[k])
        sse, beta = _sse(np.column_stack([ones, tc, hinge]), y)
        if best is None or sse < best[0]:
            best = (sse, k, beta)
    sse_hinge, k, beta = best

    def bic(sse: float, n_params: int) -> float:
        return n * math.log(max(sse, sse_floor) / n) + n_params * math.log(n)

    drop = bic(sse_line, 2) - bic(sse_hinge, 4)
    if drop < min_bic_drop:
        return None
    return RateChange(
        time=float(t[k]),
        rate_before=float(beta[1]),
        rate_after=float(beta[1] + beta[2]),
        evidence=float(drop),
        n_before=k + 1,
        n_after=n - k,
        n_screened=n_screened,
    )


# ---------------------------------------------------------------------------
# threshold forecast
# ---------------------------------------------------------------------------

#: Jitter added to bootstrap residuals, in robust standard deviations.
_JITTER = 0.5


def time_to_threshold(times, values, threshold, horizon, level=0.95, n_boot=1000, seed=0):
    """Forecast when the trend reaches ``threshold``, with an interval.

    Fits the Theil-Sen line of :func:`estimate_rate` to all readings given,
    then repeats the fit ``n_boot`` times on resampled data (a residual
    bootstrap: the fitted line plus the observed scatter, drawn at random with
    replacement). Each repeat gives a crossing time; the result is their
    median and the central ``level`` share (default 95%), all measured from
    the last reading. ``horizon`` is how far beyond the last reading the
    forecast may reach; repeats that do not cross within it count as "later
    than the horizon", and an upper bound there is reported as ``math.inf``.

    Two small corrections keep the interval honest: the residuals are scaled
    up by sqrt(n / (n - 2)) because fitting the line made them slightly
    smaller than the real scatter, and each resampled residual gets a little
    jitter (half a robust standard deviation, rescaled so the overall spread
    is unchanged) so that repeated residuals do not pile the Theil-Sen slopes
    up at the original value. In ``scripts/validate_degradation.py`` (5%
    outliers, 500 runs each) the 95% interval contained the true crossing
    time in 94.4%, 94.6% and 95.4% of runs with 20, 40 and 60 readings; an
    earlier check on other seeds gave 92.6% with 20 readings, so treat short
    series as slightly optimistic.

    No number is given -- ``time_to_cross`` is ``None`` and ``reason`` says
    why -- when there are too few readings, the rate is not significantly
    different from zero at ``level``, the fitted line already crossed the
    threshold within the observed period, the trend heads away from it, or
    the median crossing lies beyond ``horizon``.

    Assumptions: the current straight-line trend continues unchanged; the
    scatter is independent from reading to reading and looks the same in the
    future as in the past. If the rate changed recently, pass only the
    readings since the change. The interval reflects scatter around the
    trend, not the chance that the process itself changes.

    Same ``seed``, same result.
    """
    _check_level(level)
    if not horizon > 0:
        raise ValueError(f"horizon must be > 0, got {horizon!r}")
    if int(n_boot) != n_boot or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot!r}")
    t, y = _series(times, values)
    threshold = float(threshold)
    rate = estimate_rate(t, y, level=level)

    def no_forecast(reason: str) -> ThresholdForecast:
        return ThresholdForecast(
            threshold=threshold, horizon=horizon, level=level, rate=rate,
            time_to_cross=None, low=None, high=None, crossing_time=None,
            reason=reason, n_boot=int(n_boot),
        )

    if rate.n < MIN_POINTS:
        return no_forecast(f"too few points ({rate.n}; need at least {MIN_POINTS})")
    if not rate.significant:
        return no_forecast(
            f"the trend is not statistically significant "
            f"(the {level:.0%} interval of the rate includes zero)"
        )
    first = rate.intercept + rate.rate * t[0]
    last = rate.intercept + rate.rate * t[-1]
    if (threshold - first) * (threshold - last) <= 0:
        return no_forecast("the fitted trend already crossed the threshold during the observed period")
    if rate.rate * (threshold - last) < 0:
        return no_forecast("the trend heads away from the threshold")

    n = t.size
    fitted = rate.intercept + rate.rate * t
    # Fitted residuals are slightly smaller than the real errors (two line
    # parameters were fitted to the same data): scale them back up.
    resid = (y - fitted) * math.sqrt(n / (n - 2))
    rng = np.random.default_rng(seed)
    draws = resid[rng.integers(0, n, size=(int(n_boot), n))]
    # Smoothed bootstrap: resampling with replacement repeats residuals, and
    # every pair of readings that draws the same residual gets exactly the
    # original slope, piling the Theil-Sen slopes up at the original value
    # (an interval too narrow). A little jitter (0.5 robust SD), rescaled so
    # the spread of typical residuals is unchanged, breaks those ties.
    sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    if sigma > 0:
        draws = (draws + _JITTER * sigma * rng.standard_normal(draws.shape)) / math.sqrt(
            1.0 + _JITTER**2
        )
    slopes, intercepts = _theil_sen(t, fitted + draws)

    side = math.copysign(1.0, threshold - last)  # which side of the threshold we are on
    gap = side * (threshold - (intercepts + slopes * t[-1]))  # distance still to go
    speed = side * slopes  # > 0 when moving towards the threshold
    times_to_cross = np.full(int(n_boot), np.inf)
    towards = speed > 0
    times_to_cross[towards] = gap[towards] / speed[towards]
    times_to_cross[gap <= 0] = 0.0  # this repeat is already at the threshold
    times_to_cross[times_to_cross > horizon] = np.inf

    tail = (1.0 - level) / 2.0
    low, median, high = np.quantile(times_to_cross, [tail, 0.5, 1.0 - tail], method="inverted_cdf")
    if not math.isfinite(median):
        return no_forecast(f"the crossing is not expected within the forecast horizon ({horizon:g})")
    return ThresholdForecast(
        threshold=threshold, horizon=horizon, level=level, rate=rate,
        time_to_cross=float(median), low=float(low), high=float(high),
        crossing_time=float(t[-1] + median), reason=None, n_boot=int(n_boot),
    )


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------


def simulate_series(
    kind,
    *,
    n=60,
    duration=10.0,
    start_value=0.0,
    rate=0.1,
    noise=0.02,
    outlier_fraction=0.05,
    outlier_size=10.0,
    change_time=None,
    rate_after=None,
    acceleration=None,
    seed=0,
) -> SimulatedSeries:
    """Synthetic series with known answers, for testing the estimators.

    ``kind``:

    * ``"constant"`` -- a straight line, ``start_value + rate * t``.
    * ``"step"`` -- the rate changes from ``rate`` to ``rate_after`` at
      ``change_time`` (default 60% of ``duration``, ``rate_after`` default
      ``rate / 5``), with no jump in value -- e.g. an inhibitor dose.
    * ``"accelerating"`` -- the rate grows steadily:
      ``start_value + rate * t + acceleration * t**2 / 2`` (default
      ``acceleration = rate / duration``, doubling the rate by the end).

    ``n`` evenly spaced readings from 0 to ``duration`` get independent
    Gaussian noise of standard deviation ``noise``; each reading has
    probability ``outlier_fraction`` of an extra error of
    ``± outlier_size * noise``. Same ``seed``, same series.
    """
    if kind not in ("constant", "step", "accelerating"):
        raise ValueError(f"kind must be 'constant', 'step' or 'accelerating', got {kind!r}")
    if n < 2 or not duration > 0:
        raise ValueError("need n >= 2 and duration > 0")
    if not 0.0 <= outlier_fraction <= 1.0 or noise < 0:
        raise ValueError("need noise >= 0 and 0 <= outlier_fraction <= 1")
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, float(duration), int(n))
    truth = {"start_value": start_value, "rate": rate, "noise": noise,
             "rate_after": None, "change_time": None, "acceleration": None}

    if kind == "constant":
        clean = start_value + rate * t
    elif kind == "step":
        change_time = 0.6 * duration if change_time is None else change_time
        rate_after = rate / 5.0 if rate_after is None else rate_after
        clean = start_value + rate * t + (rate_after - rate) * np.maximum(0.0, t - change_time)
        truth.update(change_time=change_time, rate_after=rate_after)
    else:
        acceleration = rate / duration if acceleration is None else acceleration
        clean = start_value + rate * t + 0.5 * acceleration * t**2
        truth.update(acceleration=acceleration)

    values = clean + rng.normal(0.0, noise, size=t.size) if noise > 0 else clean.copy()
    outliers = rng.random(t.size) < outlier_fraction
    signs = rng.choice([-1.0, 1.0], size=t.size)
    values = values + np.where(outliers, signs * outlier_size * noise, 0.0)
    return SimulatedSeries(kind=kind, times=t, values=values, clean=clean,
                           outliers=outliers, truth=truth)


# ---------------------------------------------------------------------------
# plain-language summary
# ---------------------------------------------------------------------------


def _decimals(*values: float) -> int:
    """Decimals that show the largest finite magnitude with 2 significant figures."""
    finite = [abs(v) for v in values if v is not None and math.isfinite(v)]
    biggest = max(finite, default=0.0)
    if biggest == 0.0:
        return 0
    return max(0, 1 - math.floor(math.log10(biggest)))


def _fmt(value: float, decimals: int) -> str:
    value = round(value, decimals)
    if value == 0:
        value = 0.0  # never print "-0.00"
    return f"{value:.{decimals}f}"


def _rate_sentence(rate: RateEstimate, quantity: str, unit: str, time_unit: str) -> str:
    if math.isnan(rate.rate):
        return f"{quantity}: too few points to estimate a trend."
    per = f"{unit}/{time_unit}" if unit else f"per {time_unit}"
    if not rate.significant:
        d = _decimals(rate.rate, rate.low, rate.high)
        return (
            f"{quantity} shows no clear trend: about {_fmt(rate.rate, d)} {per} "
            f"({_fmt(rate.low, d)} to {_fmt(rate.high, d)})."
        )
    direction = "rising" if rate.rate > 0 else "falling"
    lo, hi = sorted((abs(rate.low), abs(rate.high)))
    d = _decimals(rate.rate, lo, hi)
    return (
        f"{quantity} is {direction} about {_fmt(abs(rate.rate), d)} {per} "
        f"({_fmt(lo, d)} to {_fmt(hi, d)})."
    )


def describe_trend(result, *, quantity, unit, time_unit="year") -> str:
    """One plain-language summary of a rate or a threshold forecast.

    ``result`` is a :class:`RateEstimate` or a :class:`ThresholdForecast`.
    ``quantity`` names what is measured ("Metal loss"), ``unit`` its unit
    ("mm") and ``time_unit`` the unit of the series' time axis ("year").
    Numbers are shown with two significant figures (the interval with the same
    decimals) and are not converted: convert rates first with
    :meth:`RateEstimate.per_year` if needed. For example::

        Metal loss is rising about 0.12 mm/year (0.09 to 0.15). At this pace
        it reaches the 0.5 mm limit in about 3.2 years (2.4 to 4.6).
    """
    if isinstance(result, RateEstimate):
        return _rate_sentence(result, quantity, unit, time_unit)
    if not isinstance(result, ThresholdForecast):
        raise TypeError(
            f"describe_trend needs a RateEstimate or ThresholdForecast, got {type(result).__name__}"
        )
    first = _rate_sentence(result.rate, quantity, unit, time_unit)
    limit = f"the {result.threshold:g} {unit} limit" if unit else f"the {result.threshold:g} limit"
    if not result.ok:
        return f"{first} It is not forecast to reach {limit}: {result.reason}."
    plural = time_unit if time_unit.endswith("s") else f"{time_unit}s"
    d = _decimals(result.time_to_cross, result.low, result.high)
    upper = (
        _fmt(result.high, d) if math.isfinite(result.high) else f"more than {result.horizon:g}"
    )
    return (
        f"{first} At this pace it reaches {limit} in about {_fmt(result.time_to_cross, d)} "
        f"{plural} ({_fmt(result.low, d)} to {upper})."
    )
