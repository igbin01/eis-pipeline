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

import numpy as np
import pandas as pd

from .fit import FitResult, FitSeries

__all__ = [
    "cpe_to_capacitance",
    "coating_capacitance",
    "double_layer_capacitance",
    "breakpoint_frequency",
    "water_uptake",
    "water_uptake_series",
    "metrics_table",
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
    if not (0.0 < n <= 1.0):
        raise ValueError(f"CPE exponent n must be in (0, 1], got {n}")
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
    """
    _require_fit(fit)
    Cc = coating_capacitance(fit)
    with np.errstate(divide="ignore", over="ignore"):
        return float(1.0 / (2.0 * np.pi * fit.params["R_po"] * Cc))


def water_uptake(
    C_t: float | FitResult,
    C_0: float | FitResult,
    *,
    epsilon_water: float = EPSILON_WATER,
) -> float:
    r"""Brasher-Kingsbury volume fraction of absorbed water.

    .. math::

        X_v = \frac{\log_{10}(C_t / C_0)}{\log_{10}(\varepsilon_{water})}

    with :math:`\varepsilon_{water} \approx 80`. ``C_0`` is the dry / freshly
    immersed coating capacitance and ``C_t`` the value after immersion. Either
    argument may be a capacitance in farads or a :class:`~eispipe.fit.FitResult`
    (coating capacitance is then computed via :func:`coating_capacitance`).

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


def _reference_capacitance(fits: FitSeries, reference) -> float:
    if isinstance(reference, (int, float)) and not isinstance(reference, bool):
        # nearest immersion time
        times = fits.times
        idx = int(np.argmin(np.abs(times - float(reference))))
        return coating_capacitance(fits[idx])
    if reference == "first":
        for r in fits:
            if r.success:
                return coating_capacitance(r)
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
    c0 = _reference_capacitance(fits, reference)
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


def metrics_table(
    fits: FitSeries,
    *,
    reference="first",
    epsilon_water: float = EPSILON_WATER,
) -> pd.DataFrame:
    """Canonical per-immersion-time metrics table.

    One row per fit. Always: ``sample_id, immersion_time_h, model, success,
    kk_passed, rmse, R_e, R_po, C_c, f_break, water_uptake_vol_frac``. When the
    fit exposes a substrate loop: ``R_ct, C_dl``. When ``area_cm2`` is set:
    ``R_po_ohm_cm2, C_c_F_per_cm2``.
    """
    try:
        c0 = _reference_capacitance(fits, reference)
    except ValueError:
        c0 = np.nan

    rows = []
    for r in fits:
        row: dict = {
            "sample_id": r.sample_id,
            "immersion_time_h": r.immersion_time_h,
            "model": r.model_name,
            "success": r.success,
            "kk_passed": r.kk_passed,
            "rmse": r.rmse,
        }
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
