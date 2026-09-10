"""Equivalent-circuit fitting and parameter extraction.

:func:`fit_spectrum` fits one :class:`~eispipe.spectrum.Spectrum` with a
:mod:`eispipe.models` circuit and returns a :class:`FitResult` carrying named
parameters, their confidence intervals, the fitted impedance and goodness-of-fit.

:func:`fit_series` fits a whole :class:`~eispipe.spectrum.SpectrumSeries` and
returns a :class:`FitSeries`.

Two deliberate defaults:

* ``seed_from_previous=False`` -- each spectrum is fit from its own data-driven
  initial guess. Seeding fit *n+1* from fit *n* produces smooth parameter
  trajectories that can be an artefact of a bad earlier fit.
* Kramers-Kronig failures are **not** skipped. When ``validate=True`` each
  result gets a ``kk_passed`` flag, but every spectrum is fit regardless -- a
  coating that changes during a slow sweep is exactly what you want to see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from .models import CircuitModel, get_model
from .spectrum import Spectrum, SpectrumSeries
from .validate import kramers_kronig

__all__ = ["FitResult", "FitSeries", "fit_spectrum", "fit_series"]


@dataclass
class FitResult:
    """Result of fitting one spectrum.

    Attributes
    ----------
    model_name, circuit:
        The model used.
    params, param_errors:
        ``{name: value}`` and ``{name: 1-sigma confidence}`` (NaN if
        ``impedance.py`` could not estimate the covariance).
    freq, Z_data, Z_fit:
        Input frequencies, measured impedance, fitted impedance.
    rmse:
        Root-mean-square of ``|Z_data - Z_fit|`` (absolute, ohms).
    weighting:
        ``"modulus"`` (fit residuals divided by ``|Z|``) or ``"none"``.
    success, message:
        ``success`` is False if the optimiser raised; ``message`` holds the
        error text.
    kk_passed:
        Lin-KK verdict, filled in by :func:`fit_series` when ``validate=True``;
        ``None`` if KK was not run.
    sample_id, immersion_time_h, area_cm2:
        Carried from the spectrum.
    """

    model_name: str
    circuit: str
    params: dict[str, float]
    param_errors: dict[str, float]
    freq: np.ndarray
    Z_data: np.ndarray
    Z_fit: np.ndarray
    rmse: float
    weighting: str
    success: bool = True
    message: str = ""
    kk_passed: bool | None = None
    sample_id: str | None = None
    immersion_time_h: float | None = None
    area_cm2: float | None = None

    def predict(self, freq: np.ndarray | None = None) -> np.ndarray:
        """Impedance of the fitted circuit at *freq* (default: the fit grid)."""
        if not self.success:
            raise RuntimeError("fit did not succeed; no model to predict from")
        from impedance.models.circuits import CustomCircuit

        cc = CustomCircuit(self.circuit, initial_guess=list(self.params.values()))
        cc.parameters_ = np.array(list(self.params.values()), dtype=float)
        cc.conf_ = np.array(list(self.param_errors.values()), dtype=float)
        grid = self.freq if freq is None else np.asarray(freq, dtype=float)
        return cc.predict(grid)

    def to_row(self) -> dict:
        """Flatten to a single dict row (params + ``<name>_err`` + metadata)."""
        row: dict = {
            "sample_id": self.sample_id,
            "immersion_time_h": self.immersion_time_h,
            "area_cm2": self.area_cm2,
            "model": self.model_name,
            "rmse": self.rmse,
            "success": self.success,
            "kk_passed": self.kk_passed,
        }
        row.update(self.params)
        row.update({f"{k}_err": v for k, v in self.param_errors.items()})
        return row

    def __repr__(self) -> str:
        state = "ok" if self.success else f"FAILED: {self.message}"
        return (
            f"FitResult({self.model_name}, immersion_time_h={self.immersion_time_h!r}, "
            f"rmse={self.rmse:.3g}, {state})"
        )


class FitSeries:
    """Ordered :class:`FitResult` collection for one sample over immersion time."""

    def __init__(self, results: list[FitResult]) -> None:
        if not results:
            raise ValueError("FitSeries requires at least one result")
        self._results = sorted(
            results,
            key=lambda r: (r.immersion_time_h is None, r.immersion_time_h),
        )

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self) -> Iterator[FitResult]:
        return iter(self._results)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return FitSeries(self._results[key])
        return self._results[key]

    @property
    def times(self) -> np.ndarray:
        return np.array([r.immersion_time_h for r in self._results], dtype=float)

    @property
    def successful(self) -> "FitSeries":
        return FitSeries([r for r in self._results if r.success])

    def to_frame(self) -> pd.DataFrame:
        """One row per immersion time: metadata, params, ``<name>_err``, GoF."""
        return pd.DataFrame([r.to_row() for r in self._results])

    def __repr__(self) -> str:
        n_ok = sum(r.success for r in self._results)
        return f"FitSeries(n={len(self)}, ok={n_ok}, model={self._results[0].model_name!r})"


# ---------------------------------------------------------------------------


def fit_spectrum(
    spec: Spectrum,
    model: str | CircuitModel,
    *,
    initial_guess=None,
    weighting: str = "modulus",
    bounds=None,
) -> FitResult:
    """Fit one spectrum.

    Parameters
    ----------
    model:
        A :class:`~eispipe.models.CircuitModel` or a registry name
        (``"intact_coating"``, ``"degraded_coating"``).
    initial_guess:
        Explicit starting parameters in circuit order; default is the model's
        data-driven guess for this spectrum.
    weighting:
        ``"modulus"`` (weight residuals by ``1/|Z|``; the standard choice when
        measurement variances are unknown) or ``"none"``.
    bounds:
        ``(lower, upper)`` sequences passed straight to ``impedance.py``;
        default is its per-element bounds (``0..inf``, CPE exponent ``0..1``).

    Raises
    ------
    RuntimeError
        If the optimiser fails. :func:`fit_series` catches this per-spectrum;
        call it directly if you want a soft failure.
    """
    from impedance.models.circuits import CustomCircuit
    from impedance.models.circuits.fitting import rmse as _rmse

    cm = get_model(model)
    if weighting not in ("modulus", "none"):
        raise ValueError("weighting must be 'modulus' or 'none'")
    guess = list(initial_guess) if initial_guess is not None else cm.initial_guess(spec)

    try:
        cc = CustomCircuit(cm.circuit, initial_guess=guess, name=cm.name)
        cc.fit(
            spec.freq,
            spec.Z,
            bounds=bounds,
            weight_by_modulus=(weighting == "modulus"),
        )
    except Exception as exc:  # bad guess length / optimiser / linear-algebra failure
        raise RuntimeError(f"{cm.name} fit failed: {exc}") from exc

    values = np.asarray(cc.parameters_, dtype=float)
    confs = (
        np.asarray(cc.conf_, dtype=float)
        if cc.conf_ is not None
        else np.full(values.shape, np.nan)
    )
    Z_fit = cc.predict(spec.freq)

    return FitResult(
        model_name=cm.name,
        circuit=cm.circuit,
        params=cm.named_params(values),
        param_errors=dict(zip(cm.param_names, (float(x) for x in confs))),
        freq=np.asarray(spec.freq, dtype=float),
        Z_data=np.asarray(spec.Z, dtype=complex),
        Z_fit=np.asarray(Z_fit, dtype=complex),
        rmse=float(_rmse(Z_fit, spec.Z)),
        weighting=weighting,
        success=True,
        sample_id=spec.sample_id,
        immersion_time_h=spec.immersion_time_h,
        area_cm2=spec.area_cm2,
    )


def fit_series(
    series: SpectrumSeries,
    model: str | CircuitModel,
    *,
    weighting: str = "modulus",
    seed_from_previous: bool = False,
    validate: bool = True,
    kk_kwargs: dict | None = None,
    bounds=None,
) -> FitSeries:
    """Fit every spectrum in *series*.

    Parameters
    ----------
    seed_from_previous:
        If True, seed each fit with the previous *successful* fit's parameters
        instead of a fresh data-driven guess. Off by default -- see the module
        docstring.
    validate:
        If True, run :func:`~eispipe.validate.kramers_kronig` on each spectrum
        and record the verdict in ``FitResult.kk_passed``. KK failures are
        never skipped.
    kk_kwargs:
        Extra keyword arguments for :func:`~eispipe.validate.kramers_kronig`.

    A spectrum whose optimiser fails yields a ``FitResult`` with
    ``success=False`` and the error message; the series continues.
    """
    cm = get_model(model)
    kk_kwargs = kk_kwargs or {}
    results: list[FitResult] = []
    prev_params: list[float] | None = None

    for spec in series:
        guess = prev_params if (seed_from_previous and prev_params is not None) else None
        try:
            res = fit_spectrum(
                spec, cm, initial_guess=guess, weighting=weighting, bounds=bounds
            )
        except RuntimeError as exc:
            res = FitResult(
                model_name=cm.name,
                circuit=cm.circuit,
                params={n: np.nan for n in cm.param_names},
                param_errors={n: np.nan for n in cm.param_names},
                freq=np.asarray(spec.freq, dtype=float),
                Z_data=np.asarray(spec.Z, dtype=complex),
                Z_fit=np.full(spec.freq.shape, np.nan, dtype=complex),
                rmse=float("nan"),
                weighting=weighting,
                success=False,
                message=str(exc),
                sample_id=spec.sample_id,
                immersion_time_h=spec.immersion_time_h,
                area_cm2=spec.area_cm2,
            )

        if validate:
            res.kk_passed = kramers_kronig(spec, **kk_kwargs).passed

        if res.success:
            prev_params = list(res.params.values())
        results.append(res)

    return FitSeries(results)
