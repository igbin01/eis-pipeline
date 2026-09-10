"""Loading EIS data from files, and light preprocessing.

Three entry points, in order of preference:

* :func:`load_csv` -- a plain CSV / delimited text export with named columns.
* :func:`load_instrument` -- a vendor binary/text export, via ``impedance.py``'s
  reader collection (Gamry, BioLogic, Autolab, ZPlot, VersaStudio, PARSTAT,
  PowerSuite, CH Instruments).
* :func:`load_series` -- a whole immersion-time series described by a **sidecar
  manifest CSV** (the canonical way to attach immersion time and area).
  :func:`load_series_by_pattern` is the fallback when there is no manifest and
  the immersion time is encoded in the filename.

Preprocessing helpers (:func:`crop_frequencies`, :func:`drop_inductive`) live
here rather than in a separate module; they take a :class:`~eispipe.spectrum.Spectrum`
and return a new one.

File mtime is never used to infer immersion time -- copying files between
machines rewrites it.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .spectrum import META_FIELDS, Spectrum, SpectrumSeries

__all__ = [
    "load_csv",
    "load_instrument",
    "load_manifest",
    "load_series",
    "load_series_by_pattern",
    "crop_frequencies",
    "drop_inductive",
]

# Metadata a caller may pass through to Spectrum via **metadata (source_path is
# set by the loaders themselves).
_PASSTHROUGH_META = tuple(f for f in META_FIELDS if f != "source_path")

# ---------------------------------------------------------------------------
# column detection for load_csv
# ---------------------------------------------------------------------------

_FREQ_KEYS = {"freq", "frequency", "f", "fhz", "freqhz", "frequencyhz", "hz", "w", "omega"}
_REAL_EXACT = {"zreal", "rez", "zprime", "zr", "zre", "real", "z"}
_IMAG_EXACT = {"zimag", "imz", "zi", "zii", "zim", "imag", "zdoubleprime", "zprimeprime"}


def _norm_header(name: str) -> str:
    """Normalise a column header for matching.

    Lower-cases, drops parenthetical content and anything after ``/`` (units),
    and keeps only ``a-z0-9`` plus the prime/quote marks that distinguish
    ``Z'`` from ``Z''`` (curly and double-prime unicode folded to ASCII).
    """
    s = str(name).strip().lower()
    s = re.sub(r"\(.*?\)", "", s)          # "re(z)" -> "re"
    s = s.split("/")[0]                    # "re(z)/ohm" -> "re(z)"
    s = s.replace("’", "'").replace("″", "''").replace("”", "''")
    s = s.replace('"', "''")
    s = re.sub(r"[^a-z0-9']", "", s)
    return s


def _header_is_negated_imag(name: str) -> bool:
    """True if the raw header marks a negated imaginary column, e.g. ``-Im(Z)``."""
    s = str(name).strip().lower().replace(" ", "")
    return s.startswith("-") or s.startswith("neg") or "-im" in s or "minusim" in s


def _classify(name: str) -> str | None:
    """Classify a header as ``"freq"``, ``"imag"``, ``"real"`` or ``None``.

    Frequency is tested first, then imaginary, then real -- ``Z''`` also ends in
    ``'`` so the imaginary test must win over the real one.
    """
    n = _norm_header(name)
    if n in _FREQ_KEYS:
        return "freq"
    if (
        n in _IMAG_EXACT
        or n.endswith("''")
        or n.startswith("im")
        or "imag" in n
    ):
        return "imag"
    if (
        n in _REAL_EXACT
        or n.endswith("'")
        or n.startswith("re")
        or "real" in n
    ):
        return "real"
    return None


def _detect_column(columns: Sequence[str], kind: str) -> str:
    """Pick the frequency / real / imaginary column from *columns*.

    ``kind`` is ``"freq"``, ``"real"`` or ``"imag"``. Raises ``ValueError`` if
    zero or more than one column matches.
    """
    matches = [col for col in columns if _classify(col) == kind]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"could not find a {kind!r} column in {list(columns)}; "
            f"pass the column name explicitly"
        )
    raise ValueError(
        f"ambiguous {kind!r} column: {matches} in {list(columns)}; "
        f"pass the column name explicitly"
    )


def _split_meta(kwargs: dict) -> dict:
    bad = set(kwargs) - set(_PASSTHROUGH_META)
    if bad:
        raise TypeError(f"unexpected metadata keyword(s): {sorted(bad)}")
    return {k: v for k, v in kwargs.items() if v is not None}


# ---------------------------------------------------------------------------
# single-spectrum loaders
# ---------------------------------------------------------------------------


def load_csv(
    path: str | Path,
    *,
    freq_col: str | None = None,
    real_col: str | None = None,
    imag_col: str | None = None,
    imag_sign: str = "auto",
    delimiter: str = ",",
    **metadata,
) -> Spectrum:
    """Read one sweep from a delimited text file with a header row.

    Column names are auto-detected (case/spacing/unit tolerant): frequency from
    names like ``freq``, ``Frequency (Hz)``, ``f/Hz``; real part from ``Zreal``,
    ``Re(Z)``, ``Z'``; imaginary part from ``Zimag``, ``Im(Z)``, ``Z''``,
    ``-Im(Z)``. Override any of them with ``freq_col`` / ``real_col`` /
    ``imag_col``.

    ``imag_sign`` controls conversion to the internal
    capacitive-is-negative convention:

    * ``"auto"`` (default) -- negate iff the imaginary column header is marked
      negative (``-Im(Z)``, ``neg...``); otherwise take values as written.
      Detection uses the header only, never the data; pass an explicit value if
      the header is unclear.
    * ``"as_measured"`` -- use values exactly as in the file.
    * ``"negate"`` -- flip the sign of the imaginary part.

    Extra keyword arguments (``sample_id``, ``immersion_time_h``, ``area_cm2``,
    ``electrolyte``, ``temperature_C``) are attached to the returned
    :class:`~eispipe.spectrum.Spectrum`.
    """
    path = Path(path)
    meta = _split_meta(metadata)

    df = pd.read_csv(path, delimiter=delimiter)
    df.columns = [str(c).strip() for c in df.columns]

    fcol = freq_col or _detect_column(df.columns, "freq")
    rcol = real_col or _detect_column(df.columns, "real")
    icol = imag_col or _detect_column(df.columns, "imag")

    freq = df[fcol].to_numpy(dtype=float)
    zr = df[rcol].to_numpy(dtype=float)
    zi = df[icol].to_numpy(dtype=float)

    if imag_sign == "auto":
        negate = _header_is_negated_imag(icol)
    elif imag_sign == "negate":
        negate = True
    elif imag_sign == "as_measured":
        negate = False
    else:
        raise ValueError(
            f"imag_sign must be 'auto', 'as_measured' or 'negate', got {imag_sign!r}"
        )
    if negate:
        zi = -zi

    Z = zr + 1j * zi
    return Spectrum(freq=freq, Z=Z, source_path=str(path), **meta)


# extension -> impedance.py instrument key
_INSTRUMENT_BY_EXT = {
    ".dta": "gamry",
    ".mpt": "biologic",
    ".mpr": "biologic",
    ".z": "zplot",
    ".par": "versastudio",
}


def load_instrument(
    path: str | Path,
    instrument: str | None = None,
    *,
    negate_imaginary: bool = False,
    **metadata,
) -> Spectrum:
    """Read one sweep from a vendor export via ``impedance.preprocessing.readFile``.

    ``instrument`` is one of ``gamry, autolab, biologic, parstat, zplot,
    versastudio, powersuite, chinstruments``. If omitted it is guessed from the
    file extension (``.dta`` -> gamry, ``.mpt`` -> biologic, ``.z`` -> zplot,
    ``.par`` -> versastudio); pass it explicitly for anything else.

    ``impedance.py``'s readers are trusted to return impedance in the internal
    capacitive-is-negative convention. Set ``negate_imaginary=True`` if a
    particular export is reversed.
    """
    from impedance.preprocessing import readFile

    path = Path(path)
    meta = _split_meta(metadata)

    if instrument is None:
        instrument = _INSTRUMENT_BY_EXT.get(path.suffix.lower())
    if instrument is None and path.suffix.lower() not in {".csv", ".txt"}:
        raise ValueError(
            f"cannot infer instrument from extension {path.suffix!r}; "
            f"pass instrument= explicitly"
        )

    freq, Z = readFile(str(path), instrument=instrument)
    freq = np.asarray(freq, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    if negate_imaginary:
        Z = Z.real - 1j * Z.imag

    return Spectrum(freq=freq, Z=Z, source_path=str(path), **meta)


# ---------------------------------------------------------------------------
# manifest + series loaders
# ---------------------------------------------------------------------------

MANIFEST_REQUIRED: tuple[str, ...] = ("filename", "immersion_time_h")
MANIFEST_OPTIONAL: tuple[str, ...] = (
    "sample_id",
    "area_cm2",
    "electrolyte",
    "temperature_C",
)


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Read and validate a sidecar manifest CSV.

    Required columns: ``filename``, ``immersion_time_h``. Expected-but-optional:
    ``sample_id``, ``area_cm2``, ``electrolyte``, ``temperature_C`` (a warning
    is emitted for any that are absent). Numeric columns are coerced and range
    checked (``immersion_time_h >= 0``, ``area_cm2 > 0``). Rows are returned
    sorted by ``immersion_time_h``.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    missing_required = [c for c in MANIFEST_REQUIRED if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"manifest {path.name} missing required column(s) {missing_required}; "
            f"present: {list(df.columns)}"
        )

    missing_optional = [c for c in MANIFEST_OPTIONAL if c not in df.columns]
    if missing_optional:
        warnings.warn(
            f"manifest {path.name} has no {missing_optional} column(s); "
            f"those metadata fields will be unset",
            stacklevel=2,
        )

    df["filename"] = df["filename"].astype(str).str.strip()
    df["immersion_time_h"] = pd.to_numeric(df["immersion_time_h"], errors="raise")
    if (df["immersion_time_h"] < 0).any():
        raise ValueError("manifest has negative immersion_time_h")
    if "area_cm2" in df.columns:
        df["area_cm2"] = pd.to_numeric(df["area_cm2"], errors="raise")
        if (df["area_cm2"] <= 0).any():
            raise ValueError("manifest has non-positive area_cm2")
    if "temperature_C" in df.columns:
        df["temperature_C"] = pd.to_numeric(df["temperature_C"], errors="raise")

    return df.sort_values("immersion_time_h", kind="stable").reset_index(drop=True)


def _loader_for(path: Path, imag_sign: str) -> Callable[..., Spectrum]:
    if path.suffix.lower() in {".csv", ".txt"}:
        return lambda p, **kw: load_csv(p, imag_sign=imag_sign, **kw)
    return load_instrument


def load_series(
    manifest: str | Path | pd.DataFrame,
    data_dir: str | Path | None = None,
    *,
    loader: Callable[..., Spectrum] | None = None,
    imag_sign: str = "auto",
) -> SpectrumSeries:
    """Build a :class:`~eispipe.spectrum.SpectrumSeries` from a manifest.

    ``manifest`` is a path to a manifest CSV (see :func:`load_manifest`) or an
    already-loaded DataFrame. Each row's ``filename`` is resolved relative to
    ``data_dir`` (default: the manifest's own directory, or the current
    directory when a DataFrame is passed).

    A per-row loader is chosen by extension -- ``.csv`` / ``.txt`` go to
    :func:`load_csv` (honouring ``imag_sign``), everything else to
    :func:`load_instrument`. Pass ``loader`` to override for every row.
    """
    if isinstance(manifest, pd.DataFrame):
        df = manifest.copy()
        base = Path(data_dir) if data_dir is not None else Path.cwd()
    else:
        mpath = Path(manifest)
        df = load_manifest(mpath)
        base = Path(data_dir) if data_dir is not None else mpath.parent

    spectra: list[Spectrum] = []
    for row in df.to_dict("records"):
        fpath = base / str(row["filename"])
        meta = {
            k: row[k]
            for k in _PASSTHROUGH_META
            if k in row and pd.notna(row[k])
        }
        use_loader = loader or _loader_for(fpath, imag_sign)
        spectra.append(use_loader(fpath, **meta))

    return SpectrumSeries(spectra)


#: Default filename time pattern for :func:`load_series_by_pattern` -- a number
#: immediately followed by ``h``/``hr``/``hrs``, e.g. ``panelA_168h.csv``.
DEFAULT_TIME_PATTERN = r"(?P<time>\d+(?:\.\d+)?)\s*h(?:r|rs)?\b"


def load_series_by_pattern(
    paths: Iterable[str | Path] | str | Path,
    *,
    pattern: str = DEFAULT_TIME_PATTERN,
    loader: Callable[..., Spectrum] | None = None,
    imag_sign: str = "auto",
    **metadata,
) -> SpectrumSeries:
    """Fallback series loader: parse immersion time from each filename.

    Use this only when there is no manifest. ``pattern`` is a regex applied to
    each file's name; it must contain a named group ``time`` (hours) and may
    contain a named group ``sample`` (used as ``sample_id`` when present).
    Files whose name does not match are skipped with a warning.

    ``paths`` may be an iterable of paths or a single glob string. Extra
    keyword metadata (e.g. ``area_cm2``, ``electrolyte``) is applied to every
    spectrum, and does not override a ``sample`` captured from the name.
    """
    if isinstance(paths, (str, Path)) and any(ch in str(paths) for ch in "*?["):
        file_list = sorted(Path().glob(str(paths)))
    elif isinstance(paths, (str, Path)):
        file_list = [Path(paths)]
    else:
        file_list = [Path(p) for p in paths]

    common_meta = _split_meta(metadata)
    rx = re.compile(pattern, re.IGNORECASE)

    spectra: list[Spectrum] = []
    for fpath in file_list:
        m = rx.search(fpath.name)
        if not m:
            warnings.warn(f"{fpath.name!r} does not match time pattern; skipped", stacklevel=2)
            continue
        meta = dict(common_meta)
        meta["immersion_time_h"] = float(m.group("time"))
        if "sample" in rx.groupindex and m.group("sample"):
            meta["sample_id"] = m.group("sample")
        use_loader = loader or _loader_for(fpath, imag_sign)
        spectra.append(use_loader(fpath, **meta))

    if not spectra:
        raise ValueError("no files matched the time pattern")
    return SpectrumSeries(spectra)


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------


def crop_frequencies(
    spec: Spectrum,
    fmin: float | None = None,
    fmax: float | None = None,
) -> Spectrum:
    """Return a copy keeping only points with ``fmin <= freq <= fmax``.

    ``None`` means unbounded on that side. Metadata is preserved.
    """
    freq = spec.freq
    keep = np.ones(freq.shape, dtype=bool)
    if fmin is not None:
        keep &= freq >= fmin
    if fmax is not None:
        keep &= freq <= fmax
    if not keep.any():
        raise ValueError(f"crop_frequencies removed every point (fmin={fmin}, fmax={fmax})")
    return spec._replace_data(freq[keep], spec.Z[keep])


def drop_inductive(spec: Spectrum) -> Spectrum:
    """Return a copy with inductive / above-axis points removed.

    Drops every point whose imaginary part is non-negative (``Z.imag >= 0``),
    i.e. points on or above the real axis of the Nyquist plot -- high-frequency
    lead inductance and mutual-inductance artefacts. Equivalent to
    ``impedance.preprocessing.ignoreBelowX``.
    """
    keep = spec.Z.imag < 0
    if not keep.any():
        raise ValueError("drop_inductive removed every point")
    return spec._replace_data(spec.freq[keep], spec.Z[keep])
