"""Coating-condition metrics derived from fitted circuit parameters.

The pipeline's reason to exist: turn a :class:`~eispipe.fit.FitSeries` into
numbers a corrosion engineer tracks over immersion time -- effective coating
capacitance, Brasher-Kingsbury water uptake, pore resistance, and breakpoint
frequency.

All functions read the named parameters produced by :mod:`eispipe.models`
(``Q_c``, ``n_c``, ``R_po`` for the coating loop in both the intact and the
degraded model; ``Q_dl``, ``n_dl``, ``R_ct`` for the substrate loop in the
degraded model).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .fit import FitResult, FitSeries
from .spectrum import Spectrum

__all__ = [
    "cpe_to_capacitance",
    "coating_capacitance",
    "double_layer_capacitance",
    "breakpoint_frequency",
    "breakpoint_frequency_measured",
    "water_uptake",
    "water_uptake_series",
    "metrics_table",
    "ml_feature_table",
]

#: Relative permittivity of water near room temperature, used by
#: :func:`water_uptake`.
EPSILON_WATER = 80.0


def cpe_to_capacitance(
    Q: float,
    n: float,
    R: float | None = None,
    *,
    omega_max: float | None = None,
) -> float:
    r"""Convert a constant-phase element ``(Q, n)`` to an effective capacitance.

    Hsu-Mansfeld method:

    .. math::

        C = Q \, \omega_{max}^{\,n-1}

    where :math:`\omega_{max} = 2\pi f_{max}` is the angular frequency at which
    the imaginary impedance of the ``R \| CPE`` loop is largest. For an ideal
    ``R \| CPE`` time constant :math:`\omega_{max} = (Q R)^{-1/n}`, which gives
    the closed form used when ``R`` is supplied:

    .. math::

        C = Q^{1/n} \, R^{(1-n)/n}

    Pass exactly one of ``R`` (closed form) or ``omega_max`` (direct form).
    ``n = 1`` returns ``Q`` unchanged (an ideal capacitor).

    Reference
    ---------
    Hsu, C. H.; Mansfeld, F. "Technical Note: Concerning the Conversion of the
    Constant Phase Element Parameter Y0 into a Capacitance." *Corrosion* 2001,
    57 (9), 747-748. doi:10.5006/1.3280607
    """
    if (R is None) == (omega_max is None):
        raise ValueError("pass exactly one of R or omega_max")
    if n <= 0.0:
        raise ValueError(f"CPE exponent n must be > 0, got {n}")
    if n > 1.0:
        warnings.warn(
            f"CPE exponent n={n} > 1 is non-physical for an ideal CPE; "
            "returning NaN instead of aborting the series",
            stacklevel=2,
        )
        return float("nan")
    if Q <= 0:
        raise ValueError(f"CPE magnitude Q must be > 0, got {Q}")

    if omega_max is None:
        if R <= 0:
            raise ValueError(f"loop resistance R must be > 0, got {R}")
        # log space: a degenerate fit (tiny n, huge R) otherwise overflows.
        with np.errstate(over="ignore"):
            log_C = np.log(Q) / n + np.log(R) * (1.0 - n) / n
            return float(np.exp(log_C))
    if omega_max <= 0:
        raise ValueError(f"omega_max must be > 0, got {omega_max}")
    with np.errstate(over="ignore"):
        return float(Q * omega_max ** (n - 1.0))


def _require_fit(fit: FitResult) -> None:
    if not fit.success:
        raise ValueError(f"fit at t={fit.immersion_time_h} h did not succeed")


def coating_capacitance(fit: FitResult, *, per_area: bool = False) -> float:
    """Effective coating capacitance from the ``Q_c, n_c, R_po`` loop.

    Uses :func:`cpe_to_capacitance` with ``R = R_po`` (the resistance in
    parallel with the coating CPE at high frequency, in both models). Returns
    farads, or F/cm^2 if ``per_area`` and ``fit.area_cm2`` is set.

    In the ``degraded_coating`` model, ``R_po`` is taken as the effective
    parallel resistance of the coating loop even though the substrate
    double-layer branch (``Q_dl, n_dl, R_ct``) hangs off the far side of it.
    This holds because, at the coating loop's own characteristic frequency,
    the double-layer CPE's impedance is small enough that the substrate
    branch approaches a short relative to ``R_ct`` -- so ``R_po`` alone
    dominates the parallel combination seen by the coating CPE.
    """
    _require_fit(fit)
    p = fit.params
    C = cpe_to_capacitance(p["Q_c"], p["n_c"], R=p["R_po"])
    if per_area:
        if not fit.area_cm2:
            raise ValueError("per_area requested but fit.area_cm2 is not set")
        C /= fit.area_cm2
    return C


def double_layer_capacitance(fit: FitResult, *, per_area: bool = False) -> float | None:
    """Substrate double-layer capacitance from ``Q_dl, n_dl, R_ct``.

    Returns ``None`` when the model has no substrate loop (e.g.
    ``intact_coating``). Farads, or F/cm^2 if ``per_area``.
    """
    _require_fit(fit)
    p = fit.params
    if not {"Q_dl", "n_dl", "R_ct"} <= set(p):
        return None
    C = cpe_to_capacitance(p["Q_dl"], p["n_dl"], R=p["R_ct"])
    if per_area:
        if not fit.area_cm2:
            raise ValueError("per_area requested but fit.area_cm2 is not set")
        C /= fit.area_cm2
    return C


def breakpoint_frequency(fit: FitResult) -> float:
    r"""Coating breakpoint frequency :math:`f_b = 1 / (2\pi R_{po} C_c)`.

    The frequency where the coating loop's phase passes -45 deg. It scales with
    the delaminated / ionically-conducting area fraction, so a rising
    :math:`f_b` over immersion time tracks disbondment.

    This is computed from the fitted ``R_po`` and ``C_c = f(Q_c, n_c, R_po)``,
    so it equals the phase-peak frequency of an ideal ``R_po \| CPE`` loop *by
    construction* -- it is not an independent measurement, only a
    re-expression of ``Q_c``, ``n_c``, and ``R_po``. For a value derived
    directly from the measured phase, independent of any fit, see
    :func:`breakpoint_frequency_measured`.
    """
    _require_fit(fit)
    Cc = coating_capacitance(fit)
    with np.errstate(divide="ignore", over="ignore"):
        return float(1.0 / (2.0 * np.pi * fit.params["R_po"] * Cc))


def breakpoint_frequency_measured(spec: Spectrum) -> float:
    """Measured coating breakpoint frequency from the raw phase, no fit.

    Finds the frequency at which the measured phase angle crosses -45 deg,
    by linear interpolation in log-frequency between the two bracketing data
    points. Unlike :func:`breakpoint_frequency`, this depends only on the
    measured spectrum, not on any fitted ``Q_c, n_c, R_po``.
    """
    freq = spec.freq
    phase = spec.Z_phase_deg
    target = -45.0
    diff = phase - target
    for i in range(diff.size - 1):
        if diff[i] == 0:
            return float(freq[i])
        if (diff[i] > 0) != (diff[i + 1] > 0):
            log_f0, log_f1 = np.log10(freq[i]), np.log10(freq[i + 1])
            t = diff[i] / (diff[i] - diff[i + 1])
            return float(10.0 ** (log_f0 + t * (log_f1 - log_f0)))
    if diff[-1] == 0:
        return float(freq[-1])
    raise ValueError("measured phase never crosses -45 deg")


def water_uptake(
    C_t: float | FitResult,
    C_0: float | FitResult,
    *,
    epsilon_water: float = EPSILON_WATER,
) -> float:
    r"""Brasher-Kingsbury volume fraction of absorbed water.

    .. math::

        X_v = \frac{\log_{10}(C_t / C_0)}{\log_{10}(\varepsilon_{water})}

    with :math:`\varepsilon_{water} \approx 80`. ``C_0`` is the *reference*
    capacitance -- conventionally the dry-film capacitance before immersion,
    against which later capacitances ``C_t`` are compared. Brasher-Kingsbury
    assumes a dry reference; when no dry measurement exists and ``C_0`` is
    taken from the earliest *immersed* spectrum instead (as
    :func:`water_uptake_series` and :func:`metrics_table` do), the result
    underestimates uptake by whatever water was absorbed before that first
    immersed measurement. Either argument may be a capacitance in farads or a
    :class:`~eispipe.fit.FitResult` (coating capacitance is then computed via
    :func:`coating_capacitance`).

    Reference
    ---------
    Brasher, D. M.; Kingsbury, A. H. "Electrical Measurements in the Study of
    Immersed Paint Coatings on Metal. I. Comparison between Capacitance and
    Gravimetric Methods of Estimating Water-Uptake." *J. Appl. Chem.* 1954,
    4 (2), 62-72. doi:10.1002/jctb.5010040202
    """
    ct = coating_capacitance(C_t) if isinstance(C_t, FitResult) else float(C_t)
    c0 = coating_capacitance(C_0) if isinstance(C_0, FitResult) else float(C_0)
    if ct <= 0 or c0 <= 0:
        raise ValueError("capacitances must be positive")
    if epsilon_water <= 1:
        raise ValueError("epsilon_water must be > 1")
    return float(np.log10(ct / c0) / np.log10(epsilon_water))


def _reference_capacitance(fits: FitSeries, reference) -> tuple[float, float | None]:
    """Return ``(C_0, reference_time_h)`` for the resolved reference fit."""
    if isinstance(reference, (int, float)) and not isinstance(reference, bool):
        # nearest immersion time, among successful fits only
        ok = fits.successful
        if len(ok) == 0:
            raise ValueError(
                f"no successful fit available to use as reference for t={reference} h"
            )
        times = ok.times
        idx = int(np.argmin(np.abs(times - float(reference))))
        ref_fit = ok[idx]
        return coating_capacitance(ref_fit), ref_fit.immersion_time_h
    if reference == "first":
        for r in fits:
            if r.success:
                return coating_capacitance(r), r.immersion_time_h
        raise ValueError("no successful fit to use as reference")
    raise ValueError("reference must be 'first' or an immersion time in hours")


def water_uptake_series(
    fits: FitSeries,
    *,
    reference="first",
    epsilon_water: float = EPSILON_WATER,
) -> pd.DataFrame:
    """Water uptake at every immersion time relative to a reference spectrum.

    ``reference`` is ``"first"`` (earliest successful fit) or an immersion time
    in hours (nearest fit is used). Columns: ``immersion_time_h``, ``C_c``,
    ``water_uptake_vol_frac``.
    """
    c0, _ = _reference_capacitance(fits, reference)
    rows = []
    for r in fits:
        if r.success:
            cc = coating_capacitance(r)
            xv = float(np.log10(cc / c0) / np.log10(epsilon_water))
        else:
            cc, xv = np.nan, np.nan
        rows.append(
            {
                "immersion_time_h": r.immersion_time_h,
                "C_c": cc,
                "water_uptake_vol_frac": xv,
            }
        )
    return pd.DataFrame(rows)


def ml_feature_table(fits: FitSeries) -> pd.DataFrame:
    """Return one row per immersion time as a flat sklearn-friendly feature table.

    Columns are the requested model inputs:
    ``immersion_time_h, R_po, C_c, n_c, chi2_norm,
    water_uptake_vol_frac`` followed by per-feature finite-difference rate
    columns with names ``*_rate``. The first row is the only row that cannot
    have a prior point to diff against, so its derivative columns are ``NaN``.
    """
    rows = []
    for result in fits:
        if not result.success:
            continue
        p = result.params
        cc = coating_capacitance(result)
        water_up = np.nan
        try:
            # Match the existing metrics convention by taking the first successful
            # fit as the reference for the water-uptake curve. Keep this pure.
            c0 = coating_capacitance(fits.successful[0])
            water_up = float(np.log10(cc / c0) / np.log10(EPSILON_WATER))
        except Exception:
            water_up = np.nan

        rows.append(
            {
                "immersion_time_h": result.immersion_time_h,
                "R_po": p["R_po"],
                "C_c": cc,
                "n_c": p["n_c"],
                "chi2_norm": result.chi2_norm,
                "water_uptake_vol_frac": water_up,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    # Add the per-feature first-difference rate columns.
    for col in ("R_po", "C_c", "n_c", "chi2_norm", "water_uptake_vol_frac"):
        frame[f"{col}_rate"] = np.nan

    for idx in range(1, len(frame)):
        dt = frame.loc[idx, "immersion_time_h"] - frame.loc[idx - 1, "immersion_time_h"]
        if dt == 0:
            dt = np.nan
        for col in ("R_po", "C_c", "n_c", "chi2_norm", "water_uptake_vol_frac"):
            if np.isfinite(dt):
                frame.loc[idx, f"{col}_rate"] = (
                    frame.loc[idx, col] - frame.loc[idx - 1, col]
                ) / dt

    return frame[[
        "immersion_time_h",
        "R_po",
        "C_c",
        "n_c",
        "chi2_norm",
        "water_uptake_vol_frac",
        "R_po_rate",
        "C_c_rate",
        "n_c_rate",
        "chi2_norm_rate",
        "water_uptake_vol_frac_rate",
    ]]


def metrics_table(
    fits: FitSeries,
    *,
    reference="first",
    epsilon_water: float = EPSILON_WATER,
) -> pd.DataFrame:
    """Canonical per-immersion-time metrics table.

    One row per fit. Always: ``sample_id, immersion_time_h, model, success,
    kk_passed, rmse, R_e, R_po, C_c, f_break, f_break_measured,
    water_uptake_vol_frac, reference_time_h``. ``f_break_measured`` is the
    fit-independent phase-crossing breakpoint frequency (see
    :func:`breakpoint_frequency_measured`); ``reference_time_h`` records which
    immersion time's capacitance was used as the water-uptake reference
    ``C_0`` (NaN if no reference was resolvable). When the fit exposes a
    substrate loop: ``R_ct, C_dl``. When ``area_cm2`` is set:
    ``R_po_ohm_cm2, C_c_F_per_cm2``.
    """
    try:
        c0, ref_time_h = _reference_capacitance(fits, reference)
    except ValueError:
        c0, ref_time_h = np.nan, np.nan

    rows = []
    for r in fits:
        row: dict = {
            "sample_id": r.sample_id,
            "immersion_time_h": r.immersion_time_h,
            "model": r.model_name,
            "success": r.success,
            "kk_passed": r.kk_passed,
            "rmse": r.rmse,
            "reference_time_h": ref_time_h,
        }
        measured_spec = Spectrum(freq=r.freq, Z=r.Z_data)
        try:
            row["f_break_measured"] = breakpoint_frequency_measured(measured_spec)
        except ValueError:
            row["f_break_measured"] = np.nan
        if r.success:
            p = r.params
            cc = coating_capacitance(r)
            row.update(
                {
                    "R_e": p.get("R_e"),
                    "R_po": p["R_po"],
                    "C_c": cc,
                    "f_break": breakpoint_frequency(r),
                    "water_uptake_vol_frac": (
                        float(np.log10(cc / c0) / np.log10(epsilon_water))
                        if np.isfinite(c0)
                        else np.nan
                    ),
                }
            )
            cdl = double_layer_capacitance(r)
            if cdl is not None:
                row["R_ct"] = p["R_ct"]
                row["C_dl"] = cdl
            if r.area_cm2:
                row["R_po_ohm_cm2"] = p["R_po"] * r.area_cm2
                row["C_c_F_per_cm2"] = cc / r.area_cm2
        else:
            row.update(
                {"R_e": np.nan, "R_po": np.nan, "C_c": np.nan, "f_break": np.nan,
                 "water_uptake_vol_frac": np.nan}
            )
        rows.append(row)

    return pd.DataFrame(rows)
