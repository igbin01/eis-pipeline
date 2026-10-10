"""Numeric validation of eispipe.metrics.cpe_to_capacitance against Table 2 of

Policastro, S.A.; et al. "Experimental and Numerical Investigation into the
Effect of Water Uptake on the Capacitance of an Organic Coating." Materials
2023, 16, 3623. doi:10.3390/ma16103623

Table 2 gives Rpo, Y0, m and the paper's own computed coating capacitance Cc
for an epoxy/polyurethane panel aged in 0.01 M NaCl at 5 degC over 3192 h,
via their Equation 2: Cc = (Y0 * Rpo)^(1/m) / Rpo -- algebraically the
Hsu-Mansfeld closed form implemented by cpe_to_capacitance(Y0, m, R=Rpo).

Rpo and Y0 in Table 2 are already area-normalised at the paper's 14.6 cm^2
exposure area (Section 2.2), so no re-normalisation is applied here -- units
are compared like-for-like (Ohm*cm^2, S*s^n*cm^-2, F/cm^2).
"""

from __future__ import annotations

from eispipe.metrics import cpe_to_capacitance

# Time (h), Rpo (Ohm*cm^2), Y0 (S*s^n*cm^-2), m, published Cc (F/cm^2)
ROWS = [
    (0, 22.6e9, 0.08e-9, 0.946, 0.09e-9),
    (168, 6.13e9, 0.14e-9, 0.914, 0.14e-9),
    (336, 5.55e9, 0.15e-9, 0.912, 0.14e-9),
    (504, 5.26e9, 0.15e-9, 0.909, 0.15e-9),
    (840, 3.07e9, 0.19e-9, 0.898, 0.18e-9),
    (1008, 2.63e9, 0.21e-9, 0.893, 0.20e-9),
    (1176, 2.77e9, 0.21e-9, 0.896, 0.19e-9),
    (1344, 2.63e9, 0.21e-9, 0.894, 0.20e-9),
    (1512, 2.34e9, 0.22e-9, 0.891, 0.20e-9),
    (1680, 2.04e9, 0.23e-9, 0.886, 0.21e-9),
    (3192, 1.31e9, 0.29e-9, 0.872, 0.25e-9),
]


def main() -> None:
    header = f"{'t (h)':>7}  {'Rpo (ohm.cm2)':>14}  {'Y0':>10}  {'m':>6}  " \
             f"{'Cc_published (nF/cm2)':>22}  {'Cc_eispipe (nF/cm2)':>20}  {'diff':>7}"
    print(header)
    print("-" * len(header))
    for t, Rpo, Y0, m, Cc_pub in ROWS:
        Cc_calc = cpe_to_capacitance(Y0, m, R=Rpo)
        diff_pct = 100.0 * (Cc_calc - Cc_pub) / Cc_pub
        print(
            f"{t:7.0f}  {Rpo:14.3e}  {Y0:10.3e}  {m:6.3f}  "
            f"{Cc_pub * 1e9:22.3f}  {Cc_calc * 1e9:20.3f}  {diff_pct:+6.1f}%"
        )


if __name__ == "__main__":
    main()
