"""Reproduce the evidence table for eispipe.degradation on fresh seeds.

Every number comes from seeded simulations (``simulate_series``) whose seeds
the unit tests do not use: the tests use seeds below 1000, this script uses
10_000 and up. The unit tests do not import or run this script.

    python scripts/validate_degradation.py           # full run, about a minute
    python scripts/validate_degradation.py --quick   # 5x fewer runs, rougher numbers

Sections:
  1. 95% rate-interval coverage (estimate_rate)
  2. false alarms on straight data, with and without outliers (detect_rate_change)
  3. change-time error on a known step
  4. 95% forecast-interval coverage (time_to_threshold) for 20, 40, 60 readings
  5. power: how often a real change is found, by noise, effect size and outliers
  6. how false alarms and power vary with the BIC threshold (2, 6, 10, 15)
  7. rate- and forecast-interval coverage when the noise is correlated over
     time (AR(1), rho 0.5 and 0.8), next to independent noise
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from eispipe.degradation import (
    detect_rate_change,
    estimate_rate,
    simulate_series,
    time_to_threshold,
)

# Shared simulation settings: 60 readings over 10 time units, rate 0.1 per
# unit, Gaussian noise 0.02, outliers of 10 x noise (5% when switched on).
MIN_SEGMENT = 5
NOISES = (0.01, 0.02, 0.05)
RATIOS = (0.25, 0.5, 0.75)  # rate after the change / rate before
THRESHOLDS = (2.0, 6.0, 10.0, 15.0)


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def rate_coverage(runs: int) -> float:
    hits = 0
    for k in range(runs):
        s = simulate_series("constant", seed=10_000 + k)
        est = estimate_rate(s.times, s.values)
        hits += est.low <= s.truth["rate"] <= est.high
    return hits / runs


def false_alarms(runs: int, outliers: float, screen: bool) -> int:
    count = 0
    for k in range(runs):
        s = simulate_series("constant", outlier_fraction=outliers, seed=11_000 + k)
        count += (
            detect_rate_change(s.times, s.values, MIN_SEGMENT, screen_outliers=screen) is not None
        )
    return count


def change_time_error(runs: int) -> tuple[float, np.ndarray]:
    errors, found = [], 0
    for k in range(runs):
        s = simulate_series("step", outlier_fraction=0.0, seed=12_000 + k)
        change = detect_rate_change(s.times, s.values, MIN_SEGMENT)
        if change is not None:
            found += 1
            errors.append(abs(change.time - s.truth["change_time"]))
    return found / runs, np.asarray(errors)


def forecast_coverage(runs: int, n: int, n_boot: int = 500) -> float:
    hits = 0
    for k in range(runs):
        s = simulate_series("constant", n=n, rate=0.1, seed=13_000 + k)
        true_time = (1.5 - s.truth["start_value"]) / s.truth["rate"] - s.times[-1]
        fc = time_to_threshold(s.times, s.values, threshold=1.5, horizon=50.0,
                               n_boot=n_boot, seed=13_000 + k)
        hits += fc.ok and fc.low <= true_time <= fc.high
    return hits / runs


def step_series(noise: float, ratio: float, outliers: float, seed: int):
    return simulate_series("step", rate=0.1, rate_after=0.1 * ratio, change_time=5.0,
                           noise=noise, outlier_fraction=outliers, seed=seed)


def found(change, true_time: float = 5.0) -> bool:
    return change is not None and abs(change.time - true_time) <= 1.0


def power(runs: int, noise: float, ratio: float, outliers: float, screen: bool) -> float:
    hits = 0
    for k in range(runs):
        s = step_series(noise, ratio, outliers, seed=14_000 + k)
        hits += found(detect_rate_change(s.times, s.values, MIN_SEGMENT, screen_outliers=screen))
    return hits / runs


def evidence(series) -> tuple[float, float | None]:
    """Best-split BIC drop (0 if below 0) and the change time, for any threshold."""
    change = detect_rate_change(series.times, series.values, MIN_SEGMENT, min_bic_drop=0.0)
    return (change.evidence, change.time) if change is not None else (0.0, None)


def threshold_sweep(runs: int) -> list[tuple[float, int, int, float, float]]:
    straight = [evidence(simulate_series("constant", outlier_fraction=0.0, seed=15_000 + k))
                for k in range(runs)]
    straight_out = [evidence(simulate_series("constant", outlier_fraction=0.05, seed=15_000 + k))
                    for k in range(runs)]
    moderate = [evidence(step_series(0.02, 0.75, 0.0, seed=16_000 + k)) for k in range(runs)]
    hard = [evidence(step_series(0.05, 0.5, 0.0, seed=16_000 + k)) for k in range(runs)]

    def hit_rate(results, thr):
        return np.mean([e >= thr and tau is not None and abs(tau - 5.0) <= 1.0 for e, tau in results])

    rows = []
    for thr in THRESHOLDS:
        rows.append((
            thr,
            sum(e >= thr for e, _ in straight),
            sum(e >= thr for e, _ in straight_out),
            hit_rate(moderate, thr),
            hit_rate(hard, thr),
        ))
    return rows


def correlated_series(rho: float, seed: int, n: int = 60, noise: float = 0.02):
    """Straight line (rate 0.1) plus AR(1) noise with total standard deviation ``noise``.

    Each reading's error is ``rho`` times the previous one plus fresh noise;
    the fresh part is scaled by sqrt(1 - rho**2) so the overall spread stays
    ``noise`` whatever ``rho`` is. ``rho = 0`` is independent noise. No outliers.
    """
    line = simulate_series("constant", n=n, rate=0.1, noise=0.0, outlier_fraction=0.0)
    rng = np.random.default_rng(seed)
    errors = np.empty(n)
    errors[0] = rng.normal(0.0, noise)
    fresh = noise * np.sqrt(1.0 - rho**2)
    for i in range(1, n):
        errors[i] = rho * errors[i - 1] + rng.normal(0.0, fresh)
    return line.times, line.values + errors


def correlated_coverage(rho: float, runs_rate: int, runs_forecast: int) -> tuple[float, float]:
    rate_hits = 0
    for k in range(runs_rate):
        t, y = correlated_series(rho, seed=17_000 + k)
        est = estimate_rate(t, y)
        rate_hits += est.low <= 0.1 <= est.high
    fc_hits = 0
    for k in range(runs_forecast):
        t, y = correlated_series(rho, seed=18_000 + k)
        true_time = 1.5 / 0.1 - t[-1]
        fc = time_to_threshold(t, y, threshold=1.5, horizon=50.0, n_boot=500, seed=18_000 + k)
        fc_hits += fc.ok and fc.low <= true_time <= fc.high
    return rate_hits / runs_rate, fc_hits / runs_forecast


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="5x fewer runs")
    args = parser.parse_args(argv)
    scale = 5 if args.quick else 1
    n_cov, n_fa, n_ct, n_fc, n_pw, n_sw = (r // scale for r in (1000, 500, 200, 500, 200, 500))
    start = time.monotonic()

    print("eispipe.degradation -- evidence table (seeds >= 10000, not used by the unit tests)")
    print("Simulations: 60 readings over 10 time units, rate 0.1, noise 0.02, outliers 10 x noise.\n")

    print(f"1. Rate interval (estimate_rate, 95%), 5% outliers, {n_cov} runs")
    print(f"   covers the true rate in {pct(rate_coverage(n_cov))}   (target about 95%)\n")

    print(f"2. False alarms on straight data (detect_rate_change, BIC drop >= 10), {n_fa} runs each")
    print(f"   {'outliers':>10} {'screening off':>15} {'screening on':>14}")
    for outl in (0.0, 0.05):
        off, on = false_alarms(n_fa, outl, False), false_alarms(n_fa, outl, True)
        print(f"   {outl:>10.0%} {f'{off}/{n_fa}':>15} {f'{on}/{n_fa}':>14}")
    print()

    rate_found, errs = change_time_error(n_ct)
    print(f"3. Change time, rate 0.1 -> 0.02 at t = 6, no outliers, {n_ct} runs")
    print(f"   found in {pct(rate_found)}; |time error| median {np.median(errs):.3f}, "
          f"95th percentile {np.percentile(errs, 95):.3f}, max {errs.max():.3f} (time units)\n")

    print(f"4. Forecast interval (time_to_threshold, 95%), 5% outliers, {n_fc} runs each")
    for n in (20, 40, 60):
        print(f"   {n:>3} readings: covers the true crossing time in {pct(forecast_coverage(n_fc, n))}"
              f"   (target 93-97%)")
    print()

    print(f"5. Power: rate changes at mid-series (t = 5), found within +-1 time unit, {n_pw} runs each")
    print(f"   {'noise':>6} {'ratio':>6} | {'0% outliers':>12} | {'5% outliers':>12} {'+ screening':>12}")
    for noise in NOISES:
        for ratio in RATIOS:
            p0 = power(n_pw, noise, ratio, 0.0, False)
            p5 = power(n_pw, noise, ratio, 0.05, False)
            p5s = power(n_pw, noise, ratio, 0.05, True)
            print(f"   {noise:>6.2f} {ratio:>6.2f} | {pct(p0):>12} | {pct(p5):>12} {pct(p5s):>12}")
    print("   (ratio = rate after / rate before; 0.25 is the clearest change, 0.75 the subtlest)\n")

    print(f"6. BIC threshold (min_bic_drop), screening off, {n_sw} runs each")
    print(f"   {'threshold':>9} | {'false alarms':>13} {'false alarms':>13} | "
          f"{'power, 0% outliers':>19}")
    print(f"   {'':>9} | {'0% outliers':>13} {'5% outliers':>13} | {'n.02 r.75':>9} {'n.05 r.5':>9}")
    for thr, fa0, fa5, p_mod, p_hard in threshold_sweep(n_sw):
        mark = "  <- default" if thr == 10.0 else ""
        print(f"   {thr:>9g} | {f'{fa0}/{n_sw}':>13} {f'{fa5}/{n_sw}':>13} | "
              f"{pct(p_mod):>9} {pct(p_hard):>9}{mark}")
    print("   (power columns: change at t = 5 with no outliers; n = noise, r = rate after / before)\n")

    n_cr, n_cf = 1000 // scale, 500 // scale
    print("7. Correlated noise: AR(1) from reading to reading, total noise SD 0.02, no outliers,")
    print(f"   60 readings; rate interval {n_cr} runs, forecast interval {n_cf} runs per row")
    print(f"   {'noise':>24} | {'rate 95% interval':>18} {'forecast 95% interval':>22}")
    for rho, label in ((0.0, "independent (rho 0)"), (0.5, "correlated, rho 0.5"),
                       (0.8, "correlated, rho 0.8")):
        rate_cov, fc_cov = correlated_coverage(rho, n_cr, n_cf)
        print(f"   {label:>24} | {pct(rate_cov):>18} {pct(fc_cov):>22}")
    print("   (both intervals assume independent noise; correlated noise makes them too narrow)")

    print(f"\n(done in {time.monotonic() - start:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
