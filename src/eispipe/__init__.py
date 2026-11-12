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
    breakpoint_frequency_measured,
    water_uptake, water_uptake_series, metrics_table,
    ml_feature_table                                  -- metrics (see ``metrics``)
    estimate_rate, detect_rate_change, time_to_threshold,
    simulate_series, describe_trend, RateEstimate, RateChange,
    ThresholdForecast, SimulatedSeries   -- drift rates and forecasts (see ``degradation``)

The ``plots`` module is imported explicitly (``from eispipe import plots``) so
that importing the package does not pull in matplotlib.
"""

from .degradation import (
    RateChange,
    RateEstimate,
    SimulatedSeries,
    ThresholdForecast,
    describe_trend,
    detect_rate_change,
    estimate_rate,
    simulate_series,
    time_to_threshold,
)
from .fit import FitResult, FitSeries, fit_series, fit_spectrum
from .io import (
    crop_frequencies,
    drop_inductive,
    load_csv,
    load_instrument,
    load_manifest,
    load_series,
    load_series_by_pattern,
)
from .metrics import (
    breakpoint_frequency,
    breakpoint_frequency_measured,
    coating_capacitance,
    cpe_to_capacitance,
    double_layer_capacitance,
    metrics_table,
    ml_feature_table,
    water_uptake,
    water_uptake_series,
)
from .models import MODELS, CircuitModel, get_model
from .spectrum import Spectrum, SpectrumSeries
from .validate import KKResult, kramers_kronig, validate_series

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
    "breakpoint_frequency_measured",
    "water_uptake",
    "water_uptake_series",
    "metrics_table",
    "ml_feature_table",
    "estimate_rate",
    "detect_rate_change",
    "time_to_threshold",
    "simulate_series",
    "describe_trend",
    "RateEstimate",
    "RateChange",
    "ThresholdForecast",
    "SimulatedSeries",
    "__version__",
]
