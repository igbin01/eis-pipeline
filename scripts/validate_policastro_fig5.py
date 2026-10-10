"""Validate the eispipe fit against Figure 5c,d of

Policastro, S.A.; et al. "Experimental and Numerical Investigation into the
Effect of Water Uptake on the Capacitance of an Organic Coating." Materials
2023, 16, 3623. doi:10.3390/ma16103623

Figure 5c,d is the Bode magnitude/phase of the 0.01 M NaCl, 5 degC panel --
the same condition as Table 2 (``data/published/policastro2023/
table2_values.csv``). Exposure area 14.6 cm^2; Table 1 gives R_Omega = 4 Ohm.

Inputs, digitized with WebPlotDigitizer into ``data/published/policastro2023/
digitized/``, for each t in ``TIMES_H``:

    t{t}h_mag.csv     freq_hz, Z_mod_ohm   (raw ohms, NOT ohm*cm^2)
    t{t}h_phase.csv   freq_hz, phase_deg   (negative for capacitive)

Steps: rebuild complex Z from |Z| and phase, write a CSV + manifest and load
it with ``load_series``, cross-check the digitized phase against Table 2,
fit ``intact_coating`` with ``fit_series(validate=True)``, compare fitted
parameters with Table 2, and save one Bode plot per time point.

``intact_coating`` rather than ``degraded_coating``: the paper held the
substrate branch fixed, because it is unobservable next to a ~GOhm coating,
so a free 7-parameter fit would be underdetermined.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eispipe.fit import FitResult, fit_series  # noqa: E402
from eispipe.io import load_series  # noqa: E402
from eispipe.metrics import cpe_to_capacitance  # noqa: E402

PAPER_DIR = REPO_ROOT / "data" / "published" / "policastro2023"
DIGITIZED_DIR = PAPER_DIR / "digitized"
TABLE2_PATH = PAPER_DIR / "table2_values.csv"
OUT_DIR = REPO_ROOT / "reports" / "policastro2023_fig5"

TIMES_H = (0, 840, 3192)
AREA_CM2 = 14.6
R_OMEGA_OHM = 4.0  # Table 1
ELECTROLYTE = "0.01 M NaCl"
TEMPERATURE_C = 5.0
SAMPLE_ID = "policastro2023_fig5"
MODEL = "intact_coating"
#: R_po is reported as only a lower bound when the digitized phase at the
#: lowest digitized frequency is more negative than this: the coating is still
#: capacitive there, so the low-frequency resistive plateau was not reached.
RPO_LOWER_BOUND_PHASE_DEG = -20.0


class SignConventionError(ValueError):
    """Reconstructed impedance is mostly non-capacitive (phase sign flipped?)."""


@dataclass
class Table2Row:
    time_h: float
    Rpo_ohm_cm2: float
    Y0: float  # S s^m cm^-2
    m: float
    Cc_F_cm2: float


def load_table2(path: Path = TABLE2_PATH) -> dict[float, Table2Row]:
    df = pd.read_csv(path)
    return {
        float(r.time_h): Table2Row(
            float(r.time_h), float(r.Rpo_ohm_cm2), float(r.Y0), float(r.m),
            float(r.Cc_published_F_cm2),
        )
        for r in df.itertuples()
    }


def digitized_paths(digitized_dir: Path, t: float) -> tuple[Path, Path]:
    stem = f"t{t:g}h"
    return digitized_dir / f"{stem}_mag.csv", digitized_dir / f"{stem}_phase.csv"


# ---------------------------------------------------------------------------
# step 1: reconstruct Z
# ---------------------------------------------------------------------------


def reconstruct_spectrum(mag_path: Path, phase_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(freq, Z)`` on the magnitude file's frequencies.

    Phase is interpolated linearly in log10(f). Magnitude points outside the
    digitized phase range are dropped rather than extrapolated. Raises
    :class:`SignConventionError` if more than half of ``Z.imag`` is >= 0.
    """
    mag = pd.read_csv(mag_path)
    ph = pd.read_csv(phase_path)
    for df, path, cols in ((mag, mag_path, ("freq_hz", "Z_mod_ohm")),
                           (ph, phase_path, ("freq_hz", "phase_deg"))):
        df.columns = [str(c).strip() for c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name}: missing column(s) {missing}; found {list(df.columns)}")

    f_mag = mag["freq_hz"].to_numpy(float)
    z_mod = mag["Z_mod_ohm"].to_numpy(float)
    order = np.argsort(ph["freq_hz"].to_numpy(float))
    logf_ph = np.log10(ph["freq_hz"].to_numpy(float)[order])
    phase = ph["phase_deg"].to_numpy(float)[order]

    logf_mag = np.log10(f_mag)
    inside = (logf_mag >= logf_ph[0]) & (logf_mag <= logf_ph[-1])
    n_out = int(np.sum(~inside))
    if n_out:
        print(f"  note: {mag_path.name}: dropped {n_out} magnitude point(s) outside "
              f"the digitized phase range {10**logf_ph[0]:.3g}-{10**logf_ph[-1]:.3g} Hz")
    f_mag, z_mod, logf_mag = f_mag[inside], z_mod[inside], logf_mag[inside]
    if f_mag.size == 0:
        raise ValueError(f"{mag_path.name} and {phase_path.name} share no frequency range")

    phi = np.radians(np.interp(logf_mag, logf_ph, phase))
    Z = z_mod * (np.cos(phi) + 1j * np.sin(phi))

    n_bad = int(np.sum(Z.imag >= 0))
    if n_bad > Z.size / 2:
        raise SignConventionError(
            f"{phase_path.name}: {n_bad} of {Z.size} reconstructed points have "
            f"Z.imag >= 0. phase_deg must be negative for a capacitive coating; "
            f"check the sign used when digitizing Figure 5d."
        )
    return f_mag, Z


# ---------------------------------------------------------------------------
# step 2: write CSV + manifest, load via load_series
# ---------------------------------------------------------------------------


def write_series(spectra: dict[float, tuple[np.ndarray, np.ndarray]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for t, (freq, Z) in spectra.items():
        fname = f"t{t:g}h_Z.csv"
        pd.DataFrame(
            {"freq_hz": freq, "Zreal_ohm": Z.real, "Zimag_ohm": Z.imag}
        ).to_csv(out_dir / fname, index=False)
        rows.append({
            "filename": fname,
            "sample_id": SAMPLE_ID,
            "immersion_time_h": t,
            "area_cm2": AREA_CM2,
            "electrolyte": ELECTROLYTE,
            "temperature_C": TEMPERATURE_C,
        })
    manifest = out_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


# ---------------------------------------------------------------------------
# step 3: phase cross-check
# ---------------------------------------------------------------------------


def predicted_phase_deg(Rpo: float, Y0: float, m: float, freq) -> np.ndarray:
    """Phase of ``R/(1 + R*Y0*(j*w)^m)``; area-normalised R and Y0 (areas cancel)."""
    w = 2 * np.pi * np.asarray(freq, dtype=float)
    Z = Rpo / (1 + Rpo * Y0 * (1j * w) ** m)
    return np.degrees(np.angle(Z))


def print_phase_check(series, table2: dict[float, Table2Row]) -> None:
    print("\n=== Step 3: phase cross-check (Table 2 R_po, Y0, m vs digitized phase) ===")
    print("Predicted from Z = R_po/(1 + R_po*Y0*(jw)^m), area-normalised. "
          "Mismatches are reported, not fatal.")
    hdr = f"{'t (h)':>6}  {'end':>4}  {'f (Hz)':>10}  {'pred (deg)':>10}  {'digit (deg)':>11}  {'diff':>7}"
    print(hdr)
    print("-" * len(hdr))
    re_ratio = []
    for spec in series:
        row = table2.get(spec.immersion_time_h)
        if row is None:
            print(f"{spec.immersion_time_h:6g}  no Table 2 row")
            continue
        for label, idx in (("low", int(np.argmin(spec.freq))), ("high", int(np.argmax(spec.freq)))):
            f = spec.freq[idx]
            pred = float(predicted_phase_deg(row.Rpo_ohm_cm2, row.Y0, row.m, f))
            dig = float(spec.Z_phase_deg[idx])
            print(f"{spec.immersion_time_h:6g}  {label:>4}  {f:10.3g}  {pred:10.1f}  {dig:11.1f}  {dig - pred:+7.1f}")
        re_ratio.append(R_OMEGA_OHM / float(np.min(spec.Z_mag)))
    if re_ratio:
        print(f"Note: R_e = {R_OMEGA_OHM:g} Ohm (Table 1) is left out of the prediction; it is "
              f"negligible here (at most {100 * max(re_ratio):.2g}% of the smallest "
              f"digitized |Z|).")


# ---------------------------------------------------------------------------
# steps 4-5: fit and compare
# ---------------------------------------------------------------------------


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def compare_fit(res: FitResult, row: Table2Row, area_cm2: float = AREA_CM2) -> dict:
    """Area-normalised fitted values next to Table 2's."""
    p = res.params
    Y0_fit = p["Q_c"] / area_cm2
    Rpo_fit = p["R_po"] * area_cm2
    Cc_fit = cpe_to_capacitance(Y0_fit, p["n_c"], R=Rpo_fit)
    i_lo = int(np.argmin(res.freq))
    z_lo = float(np.abs(res.Z_data[i_lo]))
    phase_lo = float(np.degrees(np.angle(res.Z_data[i_lo])))
    return {
        "time_h": res.immersion_time_h,
        "Y0_fit": Y0_fit, "Y0_table": row.Y0, "Y0_pct": pct(Y0_fit, row.Y0),
        "n_fit": p["n_c"], "m_table": row.m, "n_diff": p["n_c"] - row.m,
        "Rpo_fit": Rpo_fit, "Rpo_table": row.Rpo_ohm_cm2, "Rpo_pct": pct(Rpo_fit, row.Rpo_ohm_cm2),
        "Rpo_lower_bound_only": phase_lo < RPO_LOWER_BOUND_PHASE_DEG,
        "f_lo_hz": float(res.freq[i_lo]),
        "phase_lo_deg": phase_lo,
        "Z_lo_ohm": z_lo,
        "Zlo_over_Rpo": z_lo / p["R_po"],
        "Cc_fit": Cc_fit, "Cc_table": row.Cc_F_cm2, "Cc_pct": pct(Cc_fit, row.Cc_F_cm2),
        "chi2_norm": res.chi2_norm, "kk_passed": res.kk_passed,
    }


def print_comparison(fits, table2: dict[float, Table2Row]) -> list[dict]:
    print(f"\n=== Step 5: fitted {MODEL} vs Table 2 (area {AREA_CM2} cm^2) ===")
    print("Table 2 values are given to only 1-2 significant figures, so differences of "
          "several percent are within its rounding.")
    rows = []
    for res in fits:
        t = res.immersion_time_h
        print(f"\n-- t = {t:g} h --")
        if not res.success:
            print(f"  fit FAILED: {res.message}")
            continue
        row = table2.get(t)
        if row is None:
            print("  no Table 2 row for this time")
            continue
        c = compare_fit(res, row)
        rows.append(c)
        print(f"  Y0   fit Q_c/A = {c['Y0_fit']:.3e}  table {c['Y0_table']:.3e} S s^n cm^-2  ({c['Y0_pct']:+.1f}%)")
        print(f"  m    fit n_c   = {c['n_fit']:.3f}      table {c['m_table']:.3f}            ({c['n_diff']:+.3f})")
        print(f"  R_po fit R*A   = {c['Rpo_fit']:.3e}  table {c['Rpo_table']:.3e} Ohm cm^2   ({c['Rpo_pct']:+.1f}%)")
        print(f"       |Z(f_min)|/R_po_fit = {c['Zlo_over_Rpo']:.3f}, phase at f_min "
              f"({c['f_lo_hz']:.3g} Hz) = {c['phase_lo_deg']:.1f} deg")
        if c["Rpo_lower_bound_only"]:
            print(f"       FLAG: R_po is only a lower bound -- phase at f_min is more negative "
                  f"than {RPO_LOWER_BOUND_PHASE_DEG:g} deg, so the dc plateau is not reached.")
        print(f"  C_c  eispipe   = {c['Cc_fit']:.3e}  table {c['Cc_table']:.3e} F cm^-2     ({c['Cc_pct']:+.1f}%)")
        print(f"  chi2_norm = {c['chi2_norm']:.2e}, KK passed = {c['kk_passed']}")
    return rows


# ---------------------------------------------------------------------------
# step 6: plots
# ---------------------------------------------------------------------------


def save_bode_plots(series, fits, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from eispipe.plots import bode

    paths = []
    for spec, res in zip(series, fits):
        ax_mag, _ = bode(spec, fit=res)
        ax_mag.set_title(f"Policastro 2023 Fig 5c,d, t = {spec.immersion_time_h:g} h")
        path = out_dir / f"bode_t{spec.immersion_time_h:g}h.png"
        ax_mag.figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(ax_mag.figure)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------


def run(
    digitized_dir: Path = DIGITIZED_DIR,
    out_dir: Path = OUT_DIR,
    *,
    times_h=TIMES_H,
    table2_path: Path = TABLE2_PATH,
    plots: bool = True,
):
    table2 = load_table2(table2_path)

    print("=== Step 1: reconstruct Z from digitized |Z| and phase ===")
    spectra = {}
    for t in times_h:
        mag_path, phase_path = digitized_paths(digitized_dir, t)
        freq, Z = reconstruct_spectrum(mag_path, phase_path)
        print(f"  t = {t:g} h: {freq.size} points, {freq.min():.3g}-{freq.max():.3g} Hz")
        spectra[float(t)] = (freq, Z)

    print("\n=== Step 2: write CSV + manifest, load with load_series ===")
    manifest = write_series(spectra, out_dir)
    series = load_series(manifest, imag_sign="as_measured")
    print(f"  wrote {manifest}; loaded {len(series)} spectra")

    print_phase_check(series, table2)

    print(f"\n=== Step 4: fit_series(model={MODEL!r}, validate=True) ===")
    fits = fit_series(series, MODEL, validate=True)
    print(f"  {fits}")

    rows = print_comparison(fits, table2)

    if plots:
        print("\n=== Step 6: Bode plots ===")
        for p in save_bode_plots(series, fits, out_dir):
            print(f"  {p}")
    return series, fits, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--digitized-dir", type=Path, default=DIGITIZED_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    try:
        run(args.digitized_dir, args.out_dir)
    except (SignConventionError, FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
