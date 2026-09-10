"""Core in-memory representations for EIS data.

A :class:`Spectrum` is one impedance sweep (frequency vs. complex impedance)
together with the metadata every downstream step needs. A
:class:`SpectrumSeries` is an ordered set of spectra for a single sample across
immersion time -- the unit the coating metrics operate on.

Sign convention
---------------
Complex impedance is stored in the physics / ``impedance.py`` convention::

    Z = Z' + i Z''

so a capacitive response has a **negative** imaginary part (``Z.imag < 0``) and
Nyquist plots show ``-Z.imag`` on the y-axis. Readers in :mod:`eispipe.io` are
responsible for converting files that store ``-Z''`` (positive-for-capacitive)
into this convention.

Frequencies are stored in descending order (high -> low), the usual EIS sweep
direction; the constructor reorders the data if given ascending input.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

#: Metadata fields carried on a :class:`Spectrum` (everything except the arrays).
META_FIELDS: tuple[str, ...] = (
    "sample_id",
    "immersion_time_h",
    "area_cm2",
    "electrolyte",
    "temperature_C",
    "source_path",
)


@dataclass
class Spectrum:
    """One EIS sweep plus sample/measurement metadata.

    Parameters
    ----------
    freq:
        Frequencies in Hz, 1-D, strictly positive.
    Z:
        Complex impedance in ohms, same length as ``freq``, capacitive response
        negative-imaginary (see module docstring).
    sample_id:
        Identifier for the coated panel / coupon.
    immersion_time_h:
        Hours of electrolyte immersion at the time of measurement. ``0`` is the
        dry / freshly-immersed baseline used as the water-uptake reference.
    area_cm2:
        Exposed electrode area in cm^2, used to area-normalise capacitance and
        resistance. Must be positive if given.
    electrolyte:
        Free-text electrolyte description (e.g. ``"3.5 wt% NaCl"``).
    temperature_C:
        Measurement temperature in degrees Celsius.
    source_path:
        Path the data was loaded from, for provenance.
    """

    freq: np.ndarray
    Z: np.ndarray
    sample_id: str | None = None
    immersion_time_h: float | None = None
    area_cm2: float | None = None
    electrolyte: str | None = None
    temperature_C: float | None = None
    source_path: str | None = None

    def __post_init__(self) -> None:
        self.freq = np.asarray(self.freq, dtype=float)
        self.Z = np.asarray(self.Z, dtype=complex)

        if self.freq.ndim != 1 or self.Z.ndim != 1:
            raise ValueError("freq and Z must be 1-D arrays")
        if self.freq.shape != self.Z.shape:
            raise ValueError(
                f"freq and Z length mismatch: {self.freq.shape[0]} vs {self.Z.shape[0]}"
            )
        if self.freq.size == 0:
            raise ValueError("spectrum has no data points")
        if not np.all(np.isfinite(self.freq)):
            raise ValueError("freq contains non-finite values")
        if np.any(self.freq <= 0):
            raise ValueError("freq must be strictly positive")
        if not (np.all(np.isfinite(self.Z.real)) and np.all(np.isfinite(self.Z.imag))):
            raise ValueError("Z contains non-finite values")

        # Store high -> low frequency; reorder if the input was not non-increasing.
        if not np.all(np.diff(self.freq) <= 0):
            order = np.argsort(self.freq)[::-1]
            self.freq = self.freq[order]
            self.Z = self.Z[order]

        if self.immersion_time_h is not None:
            self.immersion_time_h = float(self.immersion_time_h)
            if self.immersion_time_h < 0:
                raise ValueError("immersion_time_h must be >= 0")
        if self.area_cm2 is not None:
            self.area_cm2 = float(self.area_cm2)
            if self.area_cm2 <= 0:
                raise ValueError("area_cm2 must be > 0")
        if self.temperature_C is not None:
            self.temperature_C = float(self.temperature_C)

    # -- size / derived quantities ------------------------------------------

    def __len__(self) -> int:
        return int(self.freq.size)

    @property
    def n_points(self) -> int:
        return int(self.freq.size)

    @property
    def Z_mag(self) -> np.ndarray:
        """``|Z|`` in ohms."""
        return np.abs(self.Z)

    @property
    def Z_phase_deg(self) -> np.ndarray:
        """Phase angle of ``Z`` in degrees (negative for capacitive)."""
        return np.degrees(np.angle(self.Z))

    @property
    def metadata(self) -> dict:
        """The non-array fields as a plain dict."""
        return {name: getattr(self, name) for name in META_FIELDS}

    # -- transforms --------------------------------------------------------

    def with_metadata(self, **changes) -> "Spectrum":
        """Return a copy with some metadata fields replaced.

        Only names in :data:`META_FIELDS` are accepted.
        """
        bad = set(changes) - set(META_FIELDS)
        if bad:
            raise TypeError(f"unknown metadata field(s): {sorted(bad)}")
        return replace(self, **changes)

    def _replace_data(self, freq: np.ndarray, Z: np.ndarray) -> "Spectrum":
        """Return a copy with new arrays and identical metadata."""
        return replace(self, freq=freq, Z=Z)

    # -- export ----------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """Long-format DataFrame: one row per frequency point.

        Columns: ``sample_id, immersion_time_h, freq, Z_real, Z_imag,
        Z_mag, Z_phase_deg``. Metadata values are broadcast down the column so
        that frames from several spectra concatenate cleanly.
        """
        n = self.n_points
        return pd.DataFrame(
            {
                "sample_id": np.repeat(self.sample_id, n),
                "immersion_time_h": np.repeat(self.immersion_time_h, n),
                "freq": self.freq,
                "Z_real": self.Z.real,
                "Z_imag": self.Z.imag,
                "Z_mag": self.Z_mag,
                "Z_phase_deg": self.Z_phase_deg,
            }
        )

    def __repr__(self) -> str:
        f = self.freq
        return (
            f"Spectrum(sample_id={self.sample_id!r}, "
            f"immersion_time_h={self.immersion_time_h!r}, "
            f"n_points={self.n_points}, "
            f"freq={f[0]:.3g}..{f[-1]:.3g} Hz)"
        )


class SpectrumSeries:
    """Ordered collection of :class:`Spectrum` for one sample over immersion time.

    Every member must have ``immersion_time_h`` set, and all members must share
    the same ``sample_id`` (``None`` is allowed only if *every* member is
    ``None``). Members are held sorted by ascending immersion time.
    """

    def __init__(self, spectra: Iterable[Spectrum]) -> None:
        spectra = list(spectra)
        if not spectra:
            raise ValueError("SpectrumSeries requires at least one spectrum")

        missing = [i for i, s in enumerate(spectra) if s.immersion_time_h is None]
        if missing:
            raise ValueError(
                f"every spectrum needs immersion_time_h; missing at index {missing}"
            )

        ids = {s.sample_id for s in spectra}
        if len(ids) > 1:
            raise ValueError(f"spectra span multiple sample_id values: {sorted(map(str, ids))}")

        ordered = sorted(spectra, key=lambda s: s.immersion_time_h)
        times = [s.immersion_time_h for s in ordered]
        dups = {t for t in times if times.count(t) > 1}
        if dups:
            warnings.warn(
                f"duplicate immersion_time_h values in series: {sorted(dups)}",
                stacklevel=2,
            )

        self._spectra: list[Spectrum] = ordered
        self._sample_id = next(iter(ids))

    # -- container protocol ---------------------------------------------

    def __len__(self) -> int:
        return len(self._spectra)

    def __iter__(self) -> Iterator[Spectrum]:
        return iter(self._spectra)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return SpectrumSeries(self._spectra[key])
        return self._spectra[key]

    # -- accessors ------------------------------------------------------

    @property
    def sample_id(self) -> str | None:
        return self._sample_id

    @property
    def times(self) -> np.ndarray:
        """Immersion times in hours, ascending."""
        return np.array([s.immersion_time_h for s in self._spectra], dtype=float)

    @property
    def baseline(self) -> Spectrum:
        """Earliest-immersion spectrum -- the water-uptake reference (t0)."""
        return self._spectra[0]

    # -- export -------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """Concatenate every member's :meth:`Spectrum.to_frame`, sorted by time."""
        return pd.concat(
            [s.to_frame() for s in self._spectra], ignore_index=True
        )

    def __repr__(self) -> str:
        t = self.times
        return (
            f"SpectrumSeries(sample_id={self._sample_id!r}, n={len(self)}, "
            f"immersion_time_h={t[0]:g}..{t[-1]:g})"
        )
