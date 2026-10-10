"""End-to-end demo of the eispipe package: spectrum -> fit -> metrics -> plots.

Simulates a degraded_coating spectrum at six immersion times, with pore
resistance falling and the coating CPE magnitude rising over time (a coating
losing barrier performance). ``area_cm2=14.6`` matches the exposure area used
in the Policastro et al. 2023 validation (see
``data/published/policastro2023/SOURCE.md``).

Writes ``demo_metrics.csv``, ``demo_bode.png``, ``demo_rpo_vs_time.png``
alongside this script, and prints the metrics table.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eispipe.fit import fit_series
from eispipe.metrics import metrics_table
from eispipe.plots import bode, plot_metric
from eispipe.spectrum import Spectrum, SpectrumSeries

OUT_DIR = Path(__file__).parent
IMMERSION_TIMES_H = (0, 168, 336, 672, 1344, 2016)
AREA_CM2 = 14.6

# True degraded_coating params (R_e, Q_c, n_c, R_po, Q_dl, n_dl, R_ct) in
# circuit order -- coating CPE rises and pore resistance falls with time.
TRUE_R_E = 40.0
TRUE_Q_C0, Q_C_RATE = 2e-7, 0.0015
TRUE_N_C = 0.93
TRUE_R_PO0, R_PO_RATE = 8_000.0, 0.0025
TRUE_Q_DL, TRUE_N_DL = 4e-6, 0.82
TRUE_R_CT0, R_CT_RATE = 2e6, 0.001
NOISE = 0.01


def simulate_spectrum(t_h: float, *, seed: int) -> Spectrum:
    from impedance.models.circuits import CustomCircuit

    params = [
        TRUE_R_E,
        TRUE_Q_C0 * (1 + Q_C_RATE * t_h),
        TRUE_N_C,
        TRUE_R_PO0 / (1 + R_PO_RATE * t_h),
        TRUE_Q_DL,
        TRUE_N_DL,
        TRUE_R_CT0 / (1 + R_CT_RATE * t_h),
    ]
    freq = np.logspace(4.5, -2, 50)
    cc = CustomCircuit("R0-p(CPE1,R1-p(CPE2,R2))", initial_guess=params)
    Z = cc.predict(freq, use_initial=True)

    rng = np.random.default_rng(seed)
    scale = NOISE * np.abs(Z)
    Z = Z + rng.normal(scale=scale) + 1j * rng.normal(scale=scale)

    return Spectrum(
        freq=freq,
        Z=Z,
        sample_id="DEMO1",
        immersion_time_h=t_h,
        area_cm2=AREA_CM2,
    )


def main() -> None:
    series = SpectrumSeries(
        [simulate_spectrum(t, seed=i + 1) for i, t in enumerate(IMMERSION_TIMES_H)]
    )

    fits = fit_series(series, "degraded_coating", validate=True)
    df = metrics_table(fits)

    csv_path = OUT_DIR / "demo_metrics.csv"
    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5, 6))
    for r in fits:
        bode(
            Spectrum(r.freq, r.Z_data, immersion_time_h=r.immersion_time_h),
            fit=r,
            axes=axes,
        )
    axes[0].set_title("Bode: all immersion times")
    bode_path = OUT_DIR / "demo_bode.png"
    fig.tight_layout()
    fig.savefig(bode_path, dpi=150)
    plt.close(fig)

    ax = plot_metric(df, "R_po", logy=True)
    ax.set_title("R_po vs. immersion time")
    rpo_path = OUT_DIR / "demo_rpo_vs_time.png"
    ax.figure.tight_layout()
    ax.figure.savefig(rpo_path, dpi=150)
    plt.close(ax.figure)

    print(f"wrote {csv_path}")
    print(f"wrote {bode_path}")
    print(f"wrote {rpo_path}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
