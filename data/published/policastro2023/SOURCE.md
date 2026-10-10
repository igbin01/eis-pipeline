# Policastro et al. 2023 — validation source

Policastro, S.A.; et al. "Experimental and Numerical Investigation into the
Effect of Water Uptake on the Capacitance of an Organic Coating." *Materials*
2023, 16, 3623. doi:[10.3390/ma16103623](https://doi.org/10.3390/ma16103623)

## Condition used

Epoxy/polyurethane panel aged in 0.01 M NaCl at 5 degC, exposure area
14.6 cm^2 (Section 2.2). Rpo, Y0, m in Table 2 are already area-normalised at
that exposure area.

## Sources used

- **Table 2** — Rpo, Y0, m, and the paper's own computed coating capacitance
  Cc (via their Equation 2) at each immersion time. Hardcoded into
  `table2_values.csv` and used by `scripts/validate_policastro2023.py` for
  the Step 1 numeric check against `eispipe.metrics.cpe_to_capacitance`.
- **Figure 5c,d** — Bode magnitude/phase for the same 0.01 M NaCl, 5 degC
  condition, one curve per immersion time. Not yet digitized; see
  `digitized/README.md`.

## Immersion times (h)

0, 168, 336, 504, 840, 1008, 1176, 1344, 1512, 1680, 3192

## Circuit models

- Figure 6b (aged panel): `R0-p(CPE1,R1-p(CPE2,R2))` == eispipe's
  `degraded_coating`, with R_e=R_Omega, R_po/Q_c/n_c for the coating CPE,
  R_ct/Q_dl/n_dl for the substrate CPE.
- Figure 6a / Table 1 (pretreated panel): == eispipe's `intact_coating`.
