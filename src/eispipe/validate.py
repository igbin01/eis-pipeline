"""Kramers-Kronig validation via the linear KK (Lin-KK) test.

Wraps :func:`impedance.validation.linKK` (Schonleber et al. 2014) and reduces
its output to a single :class:`KKResult` per spectrum, with a pass/fail verdict
against a residual threshold.

A **failing** KK test means the spectrum is not KK-transformable -- usually
non-stationarity (the coating changed during the sweep), drift, or a bad
contact. For a slowly-degrading coating measured with a slow sweep this is a
real physical signal, not just noise, so downstream code flags failures rather
than discarding them.

Reference
---------
Schonleber, M.; Klotz, D.; Ivers-Tiffee, E. "A Method for Improving the
Robustness of linear Kramers-Kronig Validity Tests." Electrochimica Acta 2014,
131, 20-27. doi:10.1016/j.electacta.2014.01.034
"""

from __future__ import annotations

import contextlib
import io as _io
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .spectrum import Spectrum, SpectrumSeries

__all__ = ["KKResult", "kramers_kronig", "validate_series"]

#: Default RMS relative-residual ceiling for :attr:`KKResult.passed`. A clean
#: spectrum fit with ideal RC elements still leaves ~1% residual wherever the
#: real response is a CPE (non-ideal), so the pass ceiling sits above that
#: discretisation floor.
DEFAULT_THRESHOLD = 0.02

#: Default number of RC elements. We fit a fixed dense set (``c=None``) rather
#: than the Schonleber ``mu`` auto-stop: the auto-stop routinely underfits
#: smooth spectra and then reports huge residuals for data that is in fact
#: KK-compliant. A sum of RC elements is itself KK-compliant, so a large fixed
#: M cannot "fake" a pass for a non-KK spectrum -- it can only track noise.
DEFAULT_MAX_M = 35


@dataclass
class KKResult:
    """Outcome of a Lin-KK test on one spectrum.

    Attributes
    ----------
    freq, Z_fit:
        Frequencies and the impedance of the fitted KK-compliant model.
    resid_real, resid_imag:
        Relative residuals ``(Z - Z_fit) / |Z|``, per point.
    M:
        Number of RC elements the test used.
    mu:
        Schonleber over/under-fitting measure (the test adds elements until
        ``mu < c``).
    max_resid, rms_resid:
        Max and RMS of ``|resid|`` pooled over real and imaginary parts.
    threshold, passed:
        ``passed`` is ``rms_resid <= threshold`` (RMS, not max: one noisy point
        should not fail an otherwise clean spectrum).
    sample_id, immersion_time_h:
        Carried from the spectrum for series bookkeeping.
    """

    freq: np.ndarray
    Z_fit: np.ndarray
    resid_real: np.ndarray
    resid_imag: np.ndarray
    M: int
    mu: float
    max_resid: float
    rms_resid: float
    threshold: float
    passed: bool
    sample_id: str | None = None
    immersion_time_h: float | None = None

    def __repr__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"KKResult({verdict}, immersion_time_h={self.immersion_time_h!r}, "
            f"M={self.M}, rms_resid={self.rms_resid:.2%})"
        )


def kramers_kronig(
    spec: Spectrum,
    *,
    c: float | None = None,
    max_M: int = DEFAULT_MAX_M,
    fit_type: str = "complex",
    add_cap: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
) -> KKResult:
    """Run the Lin-KK test on a single spectrum.

    Parameters
    ----------
    c:
        Cutoff for the Schonleber ``mu`` over/under-fitting measure. ``None``
        (default) disables the auto-stop and fits exactly ``max_M`` RC
        elements -- see :data:`DEFAULT_MAX_M` for why. Pass a float (e.g. the
        ``impedance.py`` default ``0.85``) to re-enable it.
    max_M:
        Number of RC elements when ``c is None``; the ceiling otherwise.
    fit_type:
        ``"real"``, ``"imag"`` or ``"complex"`` -- which component(s) the KK
        model is fit to. Residuals for both are always returned; ``"complex"``
        is the most sensitive.
    add_cap:
        Add a series capacitance (helps data with no low-frequency intercept).
    threshold:
        RMS relative-residual ceiling for :attr:`KKResult.passed`.

    Notes
    -----
    A **fail** on data you trust usually means non-stationarity (the coating
    changed during the sweep) or drift -- which is real information, not a
    reason to discard the point. Downstream (:func:`~eispipe.fit.fit_series`)
    flags failures and fits them anyway.
    """
    from impedance.validation import linKK

    # Two library-interaction workarounds, both scoped to this call:
    #  - linKK prints progress to stdout every 10 RC elements -> silence it.
    #  - impedance 1.7.1's eval_linKK builds a circuit string with repr() of
    #    NumPy scalars; under NumPy >= 2 that repr is "np.float64(...)", which
    #    then fails to eval. legacy='1.25' printing restores the bare-number
    #    repr it expects.
    with contextlib.redirect_stdout(_io.StringIO()), np.printoptions(legacy="1.25"):
        M, mu, Z_fit, resid_real, resid_imag = linKK(
            spec.freq, spec.Z, c=c, max_M=max_M, fit_type=fit_type, add_cap=add_cap
        )

    pooled = np.concatenate([np.abs(resid_real), np.abs(resid_imag)])
    max_resid = float(np.max(pooled))
    rms_resid = float(np.sqrt(np.mean(pooled**2)))

    return KKResult(
        freq=np.asarray(spec.freq, dtype=float),
        Z_fit=np.asarray(Z_fit, dtype=complex),
        resid_real=np.asarray(resid_real, dtype=float),
        resid_imag=np.asarray(resid_imag, dtype=float),
        M=int(M),
        mu=float(mu),
        max_resid=max_resid,
        rms_resid=rms_resid,
        threshold=float(threshold),
        passed=bool(rms_resid <= threshold),
        sample_id=spec.sample_id,
        immersion_time_h=spec.immersion_time_h,
    )


def validate_series(series: SpectrumSeries, **kwargs) -> pd.DataFrame:
    """Run :func:`kramers_kronig` on every spectrum in *series*.

    Returns one row per immersion time: ``sample_id, immersion_time_h, M, mu,
    max_resid, rms_resid, passed``. Keyword arguments pass through to
    :func:`kramers_kronig`.
    """
    rows = []
    for spec in series:
        r = kramers_kronig(spec, **kwargs)
        rows.append(
            {
                "sample_id": r.sample_id,
                "immersion_time_h": r.immersion_time_h,
                "M": r.M,
                "mu": r.mu,
                "max_resid": r.max_resid,
                "rms_resid": r.rms_resid,
                "passed": r.passed,
            }
        )
    return pd.DataFrame(rows)
