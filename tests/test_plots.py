"""Smoke tests for eispipe.plots (Agg backend, no files written)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eispipe import plots  # noqa: E402
from eispipe.fit import fit_spectrum  # noqa: E402
from eispipe.spectrum import Spectrum, SpectrumSeries  # noqa: E402
from eispipe.validate import kramers_kronig  # noqa: E402
from conftest import synth_coating, synth_degraded  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.fixture
def spec():
    freq, Z = synth_coating()
    return Spectrum(freq, Z, sample_id="A", immersion_time_h=0.0, area_cm2=7.07)


def test_nyquist_with_and_without_fit(spec):
    ax = plots.nyquist(spec)
    assert ax.get_xlabel()

    fit = fit_spectrum(spec, "intact_coating")
    ax2 = plots.nyquist(spec, fit=fit)
    # data + fit -> two lines
    assert len(ax2.get_lines()) == 2


def test_bode_returns_two_axes(spec):
    fit = fit_spectrum(spec, "intact_coating")
    ax_mag, ax_phase = plots.bode(spec, fit=fit)
    assert ax_mag.get_yscale() == "log"
    assert ax_phase.get_xscale() == "log"


def test_nyquist_series_adds_colorbar():
    freq, Z = synth_degraded()
    series = SpectrumSeries(
        [Spectrum(freq, Z * (1 + 0.05 * k), sample_id="A", immersion_time_h=float(t))
         for k, t in enumerate((0, 24, 72, 168))]
    )
    fig, ax = plt.subplots()
    plots.nyquist_series(series, ax=ax)
    # a colorbar adds a second Axes to the figure
    assert len(fig.axes) == 2


def test_kk_residuals(spec):
    r = kramers_kronig(spec)
    ax = plots.kk_residuals(r)
    assert ax.get_ylabel() == "relative residual / %"


def test_plot_metric_logy():
    df = pd.DataFrame(
        {"immersion_time_h": [0, 24, 72], "R_po": [1e6, 1e5, 1e4]}
    )
    ax = plots.plot_metric(df, "R_po", logy=True)
    assert ax.get_yscale() == "log"
    assert len(ax.get_lines()) == 1
