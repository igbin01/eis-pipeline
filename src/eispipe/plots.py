"""Matplotlib plotting helpers.

Every function takes an optional ``ax`` (or ``axes``) and returns it; none of
them create a figure the caller can't reach, set a title, or call
``savefig`` -- output and layout are the caller's job.

* :func:`nyquist` / :func:`bode` -- one spectrum, optionally with a fit overlay.
* :func:`nyquist_series` -- every spectrum in a series, coloured by immersion time.
* :func:`kk_residuals` -- Lin-KK residuals vs. frequency.
* :func:`plot_metric` -- any column of a metrics table vs. immersion time.
"""

from __future__ import annotations

import numpy as np

from .fit import FitResult
from .spectrum import SpectrumSeries
from .validate import KKResult

__all__ = ["nyquist", "bode", "nyquist_series", "kk_residuals", "plot_metric"]


def _get_ax(ax):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    return ax


def nyquist(spec, *, fit: FitResult | None = None, ax=None, label="data", **kwargs):
    """Nyquist plot (``Z'`` vs. ``-Z''``) of one spectrum, optional fit overlay.

    Extra keyword arguments go to the data ``plot`` call.
    """
    ax = _get_ax(ax)
    ax.plot(spec.Z.real, -spec.Z.imag, "o", label=label, **kwargs)
    if fit is not None and fit.success:
        Zf = fit.Z_fit
        ax.plot(Zf.real, -Zf.imag, "-", label="fit")
    ax.set_xlabel(r"$Z'$ / $\Omega$")
    ax.set_ylabel(r"$-Z''$ / $\Omega$")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    return ax


def bode(spec, *, fit: FitResult | None = None, axes=None):
    """Bode plot: ``|Z|`` and phase vs. frequency on two stacked axes.

    Returns the ``(ax_mag, ax_phase)`` pair.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(2, 1, sharex=True, figsize=(5, 6))
    ax_mag, ax_phase = axes

    ax_mag.loglog(spec.freq, spec.Z_mag, "o", label="data")
    ax_phase.semilogx(spec.freq, spec.Z_phase_deg, "o", label="data")
    if fit is not None and fit.success:
        Zf = fit.Z_fit
        ax_mag.loglog(fit.freq, np.abs(Zf), "-", label="fit")
        ax_phase.semilogx(fit.freq, np.degrees(np.angle(Zf)), "-", label="fit")

    ax_mag.set_ylabel(r"$|Z|$ / $\Omega$")
    ax_phase.set_ylabel(r"phase / deg")
    ax_phase.set_xlabel("frequency / Hz")
    ax_mag.legend()
    return ax_mag, ax_phase


def nyquist_series(series: SpectrumSeries, *, ax=None, cmap="viridis", colorbar=True):
    """Overlaid Nyquist plots for a whole series, coloured by immersion time."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    ax = _get_ax(ax)
    times = series.times
    norm = Normalize(vmin=float(times.min()), vmax=float(times.max()))
    cmap_obj = plt.get_cmap(cmap)

    for spec, t in zip(series, times):
        ax.plot(
            spec.Z.real,
            -spec.Z.imag,
            "o-",
            ms=3,
            color=cmap_obj(norm(t)),
        )

    ax.set_xlabel(r"$Z'$ / $\Omega$")
    ax.set_ylabel(r"$-Z''$ / $\Omega$")
    ax.set_aspect("equal", adjustable="datalim")
    if colorbar:
        sm = ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        ax.figure.colorbar(sm, ax=ax, label="immersion time / h")
    return ax


def kk_residuals(kk: KKResult, *, axes=None):
    """Plot Lin-KK relative residuals (real and imaginary) vs. frequency.

    A horizontal band at +/- ``kk.threshold`` marks the pass criterion.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(figsize=(6, 4))
    ax = axes

    ax.semilogx(kk.freq, 100 * kk.resid_real, "o-", label=r"$\Delta_{real}$", ms=3)
    ax.semilogx(kk.freq, 100 * kk.resid_imag, "s-", label=r"$\Delta_{imag}$", ms=3)
    ax.axhspan(-100 * kk.threshold, 100 * kk.threshold, color="0.85", zorder=0)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("frequency / Hz")
    ax.set_ylabel("relative residual / %")
    ax.legend()
    return ax


def plot_metric(df, y, *, x="immersion_time_h", ax=None, logy=False, **kwargs):
    """Plot column ``y`` of a metrics DataFrame against ``x``.

    ``kwargs`` pass to ``plot``. Set ``logy=True`` for quantities that span
    decades (``R_po``, ``f_break``).
    """
    ax = _get_ax(ax)
    kwargs.setdefault("marker", "o")
    ax.plot(df[x], df[y], **kwargs)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    return ax
