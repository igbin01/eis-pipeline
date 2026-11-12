# eispipe.degradation: design decisions and rejected alternatives

This records the alternatives that were measured and not adopted for
`src/eispipe/degradation.py`, so the choices can be revisited with the
numbers in hand. Current figures for the adopted methods are produced by
`scripts/validate_degradation.py` and quoted in the module's docstrings.

All numbers below come from seeded simulations: a straight line with rate 0.1
per time unit over 10 time units, Gaussian noise of SD 0.02 (AR(1) where a
`rho` is given, with the same overall SD), and a 95% target unless stated.
They are simulation results, not guarantees for real data. Sources named
for rejected methods are given for orientation only and have not been
checked against the originals; the sources the adopted code relies on, and
their verification status, are in `docs/literature_ledger.md`.

## Forecast interval (`time_to_threshold`), independent noise

| Variant | 20 readings | 40 readings | 60 readings |
|---|---|---|---|
| "separate" Theil-Sen intercept, plain residual bootstrap | -- | 99.4% (too wide) | -- |
| "joint" intercept, plain residual bootstrap | 88.8% | 92.3% | 91.9% |
| + residuals scaled by sqrt(n/(n-2)) | 90.2% | 93.3% | 92.4% |
| + smoothed bootstrap (jitter 0.5 robust SD) only | 91.9% | 94.5% | 93.8% |
| **"joint" + scaling + smoothing (adopted)** | 92.6% | 94.6% | 94.0% |

The "separate" intercept (median(y) - slope * median(t)) depends on the one
or two middle readings; the bootstrap partly undid that error, so the
interval was centred better than its width assumed. Slope uncertainty was not
counted twice: the bootstrap SDs were within 6-13% of the true spreads.

## Change detection (`detect_rate_change`), independent noise

* **F-test instead of BIC:** rejected without simulation. The change time
  is chosen by searching, which a fixed-time F-test p-value does not allow
  for.
* **Outlier screening on the two-segment fit's residuals** (instead of the
  adopted local-median rule): restored power about as well (e.g. 99.5% vs
  72% without screening, noise 0.02, rate halving, 5% outliers), but false
  changes on straight data with 5% outliers stayed at 5-6 in 200. The fit
  can bend towards a cluster of outliers near an end. The adopted rule
  (detrend with Theil-Sen, compare with the median of +-3 neighbours, cutoff
  3.5 robust SD) gave 0 in 200.

## Correlated noise: estimating rho

Lag-1 autocorrelation of Theil-Sen residuals, 60 readings, mean over 400 runs:

| True rho | raw r1 | (n r1 + 1)/(n - 3) | (n r1 + 2)/(n - 6) (adopted) |
|---|---|---|---|
| 0.0 | 0.035 | 0.044 | 0.056 |
| 0.5 | 0.426 | 0.466 | 0.511 |
| 0.8 | 0.694 | 0.748 | 0.806 |

Shrinking the estimate towards 0 for small n was considered and rejected:
the raw estimate is already biased low, most of all for short series. For
10-20 readings the denominator is floored at 0.75 n. Before the floor, 10
independent readings gave a mean estimate of 0.28 and widened intervals
1.9-fold; after it, 0.17 and 1.3-fold. Other options measured at 10-30
readings:

* **Mean-only correction (n r1 + 1)/(n - 3):** under-corrected; coverage at
  rho 0.5 was 81-92%.
* **No estimate below 20 readings:** stable, but left 10-19 readings
  unprotected (72-85% coverage at rho 0.5).

## Correlated noise: rate interval (`estimate_rate`)

60 readings, 400 runs per cell:

| Variant | rho 0 | rho 0.5 | rho 0.8 | width at rho 0.8 vs rho 0 |
|---|---|---|---|---|
| **Sen interval stretched by sqrt(n / n_eff) (adopted)** | 94.8% | 93.5% | 92.0% | 3.0x |
| Kendall-statistic variance multiplied by n / n_eff | 94.8% | 94.2% | 94.8% | 10.6x |
| Hamed-Rao-style factor with AR(1) rank correlations | 96.0% | 93.9% | 90.2% | 2.6x |

* **Inflating the variance of Kendall's statistic** (the idea behind the
  Hamed and Rao 1998 correction) covered well, but at rho 0.8 it reached into
  the noisy extreme pairwise slopes: 10.6 times wider, against a theoretical
  sqrt(9) = 3.
* **The Hamed-Rao factor in AR(1) form** (rank autocorrelations
  (6/pi) asin(rho^k / 2)) under-covered at rho 0.8. That run used the earlier
  rho estimator, (n r1 + 1)/(n - 3).
* **Trend-free pre-whitening** (Yue et al. 2002) tests whether a trend
  exists; it does not give an interval, so it does not apply.

## Correlated noise: forecast interval (`time_to_threshold`)

60 readings, 400 runs per cell, bias-corrected rho:

| Variant | rho 0 (5% outliers) | rho 0.5 | rho 0.8 |
|---|---|---|---|
| **Plain bootstrap interval stretched by sqrt(n / n_eff) (adopted)** | 94.2% | 93.2% | 90.8% |
| AR(1) sieve bootstrap (resample innovations, rebuild residuals) | 93.5% | 92.2% | 90.2% |
| Moving-block bootstrap, block length (2 rho/(1-rho^2))^(2/3) n^(1/3) | 93.5% | 85.0% | 66.2% |

* **The moving-block bootstrap** (Kuensch 1989; block-length rule as
  attributed to Carlstein 1986) failed because 60 readings make only a few
  blocks of about 9, too few to reproduce the spread. Longer blocks made it
  worse.
* **The sieve bootstrap** (Kreiss 1992; Buehlmann 1997) met the targets but
  was slightly less accurate than the stretch at every correlation.
* **The adopted stretch** reduces exactly to the independent bootstrap
  interval when rho is 0.

## Correlated noise: change detection (`detect_rate_change`)

60 readings, no outliers. The first two columns are false changes on straight
data in 500 runs; the power columns are for a rate halving at noise 0.02
(200 runs).

| Variant | false, rho 0.5 | false, rho 0.8 | power rho 0 | power rho 0.5 | power rho 0.8 |
|---|---|---|---|---|---|
| no allowance (old) | 23 | 180 | 100% | 98.5% | 92.0% |
| **discount by n_eff/n, rho from segment residuals (adopted, before gate)** | 3 | 4 | 100% | 94.0% | 51.0% |
| same, rho corrected for the extra parameter | 2 | 3 | 100% | 89.0% | 37.5% |
| same, rho from the line's residuals | 0 | 0 | 0% | 0% | 0% |
| AR(1) errors in both models (Prais-Winsten) | 1 | 2 | 100% | 84.5% | 44.5% |

* **Taking rho from the line's residuals** reads a real bend as correlation
  and hides every change.
* **Fitting AR(1) errors (Prais-Winsten)** controlled false changes as well,
  but lost more power.
* **A significance gate was added** afterwards: discount only if the raw
  lag-1 autocorrelation exceeds 1.645 / sqrt(n). Without it, small chance
  correlations on independent data cost up to 6.5 points of power (noise
  0.05, rate halving: 75.0% to 68.5%). With it, the largest loss on
  independent data was 0.5 points, and false changes were 4 / 500 at rho 0.5
  and 9 / 500 at rho 0.8. The gate is used only for change detection. The
  intervals use rho without it, because there a slightly too-wide interval
  is the safe error.
