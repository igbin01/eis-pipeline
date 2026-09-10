"""eispipe -- EIS analysis pipeline for organic coating degradation.

Public API (stable):

    Spectrum, SpectrumSeries      -- in-memory data types (see ``spectrum``)
    load_csv, load_instrument     -- read a single sweep (see ``io``)
    load_manifest, load_series,
    load_series_by_pattern        -- read an immersion-time series (see ``io``)
    crop_frequencies, drop_inductive  -- preprocessing (see ``io``)

    kramers_kronig, validate_series, KKResult   -- KK validation (see ``validate``)
    CircuitModel, get_model, MODELS             -- circuit definitions (see ``models``)
    fit_spectrum, fit_series, FitResult, FitSeries  -- fitting (see ``fit``)
    coating_capacitance, double_layer_capacitance,
    cpe_to_capacitance, breakpoint_frequency,
    water_uptake, water_uptake_series, metrics_table  -- metrics (see ``metrics``)

The ``plots`` module is imported explicitly (``from eispipe import plots``) so
that importing the package does not pull in matplotlib.
"""

from .spectrum import Spectrum, SpectrumSeries
from .io import (
    load_csv,
    load_instrument,
    load_manifest,
    load_series,
    load_series_by_pattern,
    crop_frequencies,
    drop_inductive,
)
from .validate import KKResult, kramers_kronig, validate_series
from .models import CircuitModel, MODELS, get_model
from .fit import FitResult, FitSeries, fit_spectrum, fit_series
from .metrics import (
    cpe_to_capacitance,
    coating_capacitance,
    double_layer_capacitance,
    breakpoint_frequency,
    water_uptake,
    water_uptake_series,
    metrics_table,
)

__version__ = "0.1.0"

__all__ = [
    "Spectrum",
    "SpectrumSeries",
    "load_csv",
    "load_instrument",
    "load_manifest",
    "load_series",
    "load_series_by_pattern",
    "crop_frequencies",
    "drop_inductive",
    "KKResult",
    "kramers_kronig",
    "validate_series",
    "CircuitModel",
    "MODELS",
    "get_model",
    "FitResult",
    "FitSeries",
    "fit_spectrum",
    "fit_series",
    "cpe_to_capacitance",
    "coating_capacitance",
    "double_layer_capacitance",
    "breakpoint_frequency",
    "water_uptake",
    "water_uptake_series",
    "metrics_table",
    "__version__",
]
