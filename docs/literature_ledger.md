# eispipe.degradation: literature ledger

Every published source that `src/eispipe/degradation.py` leans on, what the
code actually takes from it, and how far that has been checked. The code
points to rows here as `ledger L1` etc. Simulation evidence for the parts
that are not covered by a source is in `scripts/validate_degradation.py` and
`docs/degradation_design.md`.

Status words: **confirmed** = the claim was checked against the source or a
reliable description of it; **partly confirmed** = checked for a narrower
case than the code uses; **not checked** = relied on from memory.

| # | Claim | Source | What the code relies on | Verification status | Who verified it, how |
|---|---|---|---|---|---|
| L1 | The Theil-Sen slope (median of pairwise slopes) and its distribution-free confidence interval from the ranks of those slopes | Sen, P. K. (1968), *J. Amer. Statist. Assoc.* 63, 1379-1389 | `estimate_rate`: slope, `low`/`high` via `scipy.stats.theilslopes`; the same slope in the bootstrap of `time_to_threshold` | **Confirmed** for the slope and interval. The intercept is **not** defined in Sen (1968); the code's intercept (median of `value - slope * time`) follows SciPy's documented `method="joint"` option. | Claude (AI assistant, project chat), 2026-11-12, web search of secondary sources (SciPy's `theilslopes` documentation, which cites Sen (1968) and states that the intercept is not part of it); primary source not opened. |
| L2 | A BIC drop of 10 or more counts as "very strong evidence" for the better model (the 2 ln BF scale: 0-2 not worth more than a bare mention, 2-6 positive, 6-10 strong, > 10 very strong) | Kass, R. E. and Raftery, A. E. (1995), *J. Amer. Statist. Assoc.* 90, 773-795 | `detect_rate_change`: default `min_bic_drop=10` | **Confirmed** through secondary sources quoting the scale; the paper itself was not opened. The value was also chosen independently by simulation (false alarms vs power, script section 6), so the code does not depend on the scale being exact. | Claude (AI assistant, project chat), 2026-11-12, web search of secondary sources (quoting the 2 ln BF scale); primary source not opened. |
| L3 | With AR(1) noise of lag-1 correlation rho, n readings carry the information of n_eff = n (1 - rho) / (1 + rho) independent ones | von Storch, H. and Zwiers, F. W. (1999), *Statistical Analysis in Climate Research*, Cambridge | `estimate_rate`, `time_to_threshold`: intervals stretched by sqrt(n / n_eff); `detect_rate_change`: likelihood ratio discounted by n_eff / n | **Partly confirmed.** Confirmed for the variance of a *mean* through other sources. Applying the same factor to a trend *slope* (and to a likelihood ratio) is supported only by simulation: coverage 94.1% / 92.9% at rho 0.5 / 0.8 for the rate interval, 93.6% / 90.6% for the forecast (60 readings; script section 7). | Claude (AI assistant, project chat), 2026-11-12, web search of secondary sources (for the mean case); primary source not opened. Slope extension: simulation during development (script sections 7-9, design doc). |
| L4 | The sample lag-1 autocorrelation of a series with an estimated mean is biased by about -(1 + 3 rho) / n | Kendall, M. G. (1954), *Biometrika* 41, 403-404; Marriott, F. H. C. and Pope, J. A. (1954), *Biometrika* 41, 390-402 | `_lag1_correlation`: the correction `(n r1 + 2) / max(n - 6, 0.75 n)` | **Partly confirmed.** Confirmed for a series with an estimated mean (one parameter). The constant 2 for residuals of a fitted *line* (two parameters) is a heuristic, checked by simulation only: mean estimates 0.31 / 0.51 / 0.81 at true 0.3 / 0.5 / 0.8 on 60 readings. The denominator floor of 0.75 n for short series is an engineering choice, not from any source. | Claude (AI assistant, project chat), 2026-11-12, web search of secondary sources (describing the Kendall / Marriott-Pope result for the mean); primary source not opened. Line constant and floor: simulation during development (design doc, "estimating rho"). |
| L5 | General guidance on the statistical treatment of corrosion data (only the title was confirmed; the content has not been read) | ASTM G16, *Standard Guide for Applying Statistics to Analysis of Corrosion Data* | Nothing specific: no formula or number in the code comes from it. It is named in the module docstring only as the standard a corrosion-rate analysis would eventually be checked against. | **Not checked.** To read when library access is available. | Nobody yet. |
| L6 | Residuals of a least-squares fit should be scaled up by sqrt(n / (n - p)) before a residual bootstrap, because fitting p parameters makes them smaller than the true errors | Davison, A. C. and Hinkley, D. V. (1997), *Bootstrap Methods and Their Application*, Cambridge, ch. 6 | `time_to_threshold`: residuals multiplied by sqrt(n / (n - 2)) before resampling | **Not checked** against the book. The effect was measured: the rescaling moved forecast coverage from 92.3% to 93.3% at 40 readings (design doc, forecast table); the Theil-Sen line is not least squares, so the textbook factor is applied by analogy. | Nobody has opened the source. Effect: simulation during development. |

Rows L1 to L4 should be re-confirmed against the primary sources by the
project owner when library access is available; until then they count as
confirmed through secondary sources only.

## Notes

* No source was consulted for the smoothed-bootstrap jitter (0.5 robust SD),
  the one-sided 5% gate on rho in `detect_rate_change`, the outlier-screening
  rule (local median of +-3 neighbours, cutoff 3.5 robust SD), or the
  `MAX_RHO = 0.95` cap. All four are engineering choices checked by
  simulation and recorded in `docs/degradation_design.md`.
* Sources named only for rejected alternatives (Hamed and Rao 1998, Yue et
  al. 2002, Kuensch 1989, Carlstein 1986, Kreiss 1992, Buehlmann 1997) have no
  code behind them and are listed in the design document, not here.
