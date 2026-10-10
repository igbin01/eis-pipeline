"""Equivalent-circuit definitions for organic-coating EIS.

Each model is a :class:`CircuitModel`: an ``impedance.py`` circuit string, the
human-readable parameter names aligned to that string's parameter order, and a
data-driven initial-guess heuristic. Build an ``impedance.py`` model with
:meth:`CircuitModel.build`, or look one up by name with :func:`get_model`.

Models
------
``intact_coating``  -- ``R0-p(R1,CPE1)``
    One time constant: electrolyte resistance ``R_e`` in series with the coating
    pore resistance ``R_po`` in parallel with the coating capacitance
    ``CPE_c`` (``Q_c``, ``n_c``). Appropriate while the film is a good barrier.

``degraded_coating`` -- ``R0-p(CPE1,R1-p(CPE2,R2))``
    Two nested time constants. The substrate charge-transfer branch
    (``R_ct`` in parallel with the double-layer ``CPE_dl``) sits *inside* the
    pore-resistance branch, because the electrolyte can only reach the metal
    through the coating pores -- ``R_po`` is in series with the interfacial
    process, and the whole thing is in parallel with the coating capacitance.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .spectrum import Spectrum

__all__ = ["CircuitModel", "MODELS", "get_model", "INTACT_COATING", "DEGRADED_COATING"]


# ---------------------------------------------------------------------------
# data-driven initial-guess helpers
# ---------------------------------------------------------------------------


def _estimate_Re(spec: Spectrum) -> float:
    """Electrolyte resistance ~ real part of Z at the highest frequency."""
    re_hi = float(spec.Z.real[np.argmax(spec.freq)])
    return re_hi if re_hi > 1e-6 else 1.0


def _estimate_Rp(spec: Spectrum, Re: float) -> float:
    """Low-frequency resistive span above ``Re`` (semicircle diameter)."""
    re_lo = float(spec.Z.real[np.argmin(spec.freq)])
    span = re_lo - Re
    if not np.isfinite(span) or span <= 0:
        span = float(np.ptp(spec.Z.real))
    return max(span, 1.0e2)


#: Minimum peak-to-flanking-valley height (in degrees of phase) required
#: before :func:`_find_arc_boundary` accepts a candidate as a genuine
#: separation between the two arcs, rather than a noise ripple.
_MIN_ARC_SEPARATION_PROMINENCE_DEG = 5.0


def _find_arc_boundary(phase_deg: np.ndarray) -> int | None:
    """Index of the phase peak separating two nested RC arcs, or ``None``.

    Both CPE branches of a two-time-constant spectrum drive the phase toward
    0 deg at the high- and low-frequency ends and toward a more negative
    extreme near each arc's own characteristic frequency; between the two
    arcs the phase rises back toward 0 at a local maximum. That maximum,
    unlike a local extremum in ``|Z''|``, stays visible even when the two
    arcs' resistances differ by orders of magnitude, because phase is a
    ratio (``atan2(Z'', Z')``) and so is insensitive to the huge difference
    in the arcs' absolute scale that swamps a magnitude-based search.

    Every interior local maximum is a candidate; the one is picked whose
    prominence (drop to the lower of its two flanking minima) is largest,
    and only accepted if that prominence clears
    ``_MIN_ARC_SEPARATION_PROMINENCE_DEG`` -- small ripples from measurement
    noise produce shallow local maxima near the sweep's ends (phase trends
    toward 0 at both extremes anyway) that would otherwise be mistaken for a
    genuine arc boundary.
    """
    if phase_deg.size < 5:
        return None
    is_local_max = (phase_deg[1:-1] > phase_deg[:-2]) & (phase_deg[1:-1] > phase_deg[2:])
    candidates = np.flatnonzero(is_local_max) + 1
    if candidates.size == 0:
        return None

    best_idx, best_prominence = None, 0.0
    for idx in candidates:
        left_min = float(np.min(phase_deg[:idx]))
        right_min = float(np.min(phase_deg[idx:]))
        prominence = float(phase_deg[idx]) - max(left_min, right_min)
        if prominence > best_prominence:
            best_idx, best_prominence = int(idx), prominence

    if best_idx is None or best_prominence < _MIN_ARC_SEPARATION_PROMINENCE_DEG:
        return None
    return best_idx


def _estimate_Rpo_Rct(spec: Spectrum, Re: float) -> tuple[float, float]:
    """Split the low-frequency resistive span into a coating and a substrate share.

    Used only for ``degraded_coating``'s two nested loops, where
    :func:`_estimate_Rp`'s whole-span estimate is the wrong guess for
    ``R_po`` alone: both CPEs approach open circuits at low frequency, so
    that span is ``R_po + R_ct``, not ``R_po`` by itself. Coatings routinely
    have ``R_ct`` orders of magnitude larger than ``R_po``, and conflating
    them starts the optimiser orders of magnitude off -- it then often
    converges to a spurious local minimum instead of the true one,
    reproducible even at zero noise.

    Instead, this looks for the frequency where the two arcs separate (see
    :func:`_find_arc_boundary`) and uses ``Z'`` there, minus ``R_e``, as the
    ``R_po`` estimate; ``R_ct`` takes the remainder of the whole span. If no
    clear separation is found -- a single, undifferentiated arc, e.g. because
    ``R_po`` and ``R_ct`` are actually close in magnitude -- this falls back
    to splitting the whole span evenly between the two and emits a warning,
    since that fallback is a much rougher starting point.
    """
    order = np.argsort(spec.freq)[::-1]
    zr = spec.Z.real[order]
    total_span = max(float(zr[-1]) - Re, 1.0e2)

    phase_deg = np.degrees(np.angle(spec.Z[order]))
    boundary = _find_arc_boundary(phase_deg)

    if boundary is None:
        warnings.warn(
            "degraded_coating initial guess: no clear separation between the "
            "coating and substrate arcs (single undifferentiated arc); "
            "falling back to an even split of the low-frequency span between "
            "R_po and R_ct",
            stacklevel=3,
        )
        half_span = total_span / 2.0
        return half_span, half_span

    Rpo = float(zr[boundary]) - Re
    Rpo = min(max(Rpo, 1.0), total_span * 0.95)
    Rct = max(total_span - Rpo, 1.0)
    return Rpo, Rct


def _estimate_C(spec: Spectrum) -> float:
    """Capacitance from the high-frequency capacitive branch, ``C ~ -1/(w Z'')``.

    Uses the median over the highest-frequency decade of points that are
    actually capacitive (``Z'' < 0``), which is robust to a noisy first point.
    """
    order = np.argsort(spec.freq)[::-1]
    f = spec.freq[order]
    zim = spec.Z.imag[order]
    hf = f >= (f[0] / 10.0)
    cap = hf & (zim < 0)
    if not cap.any():
        cap = zim < 0
    if not cap.any():
        return 1e-8
    C = -1.0 / (2 * np.pi * f[cap] * zim[cap])
    return float(np.median(C))


# ---------------------------------------------------------------------------
# model definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CircuitModel:
    """A named equivalent-circuit model.

    Attributes
    ----------
    name:
        Registry key.
    circuit:
        ``impedance.py`` circuit string.
    param_names:
        Domain names for each free parameter, in the same order
        ``impedance.py`` returns them (verify with
        ``CustomCircuit(circuit, ...).get_param_names()``).
    units:
        Unit string per parameter, same order.
    description:
        One-line summary.
    _guess:
        Callable ``spec -> list[float]`` giving a data-driven initial guess.
    """

    name: str
    circuit: str
    param_names: tuple[str, ...]
    units: tuple[str, ...]
    description: str
    _guess: Callable[[Spectrum], list[float]] = field(repr=False)

    def __post_init__(self) -> None:
        if len(self.param_names) != len(self.units):
            raise ValueError("param_names and units length mismatch")

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def initial_guess(self, spec: Spectrum) -> list[float]:
        """Data-driven starting parameters for *spec*, in circuit order."""
        guess = [float(x) for x in self._guess(spec)]
        if len(guess) != self.n_params:
            raise ValueError(
                f"{self.name}: guess has {len(guess)} values, expected {self.n_params}"
            )
        return guess

    def build(self, initial_guess=None, *, spec: Spectrum | None = None, constants=None):
        """Return an unfitted ``impedance.py`` ``CustomCircuit``.

        Provide ``initial_guess`` explicitly, or pass ``spec`` to derive one
        from data via :meth:`initial_guess`.
        """
        from impedance.models.circuits import CustomCircuit

        if initial_guess is None:
            if spec is None:
                raise ValueError("pass initial_guess or spec")
            initial_guess = self.initial_guess(spec)
        return CustomCircuit(
            self.circuit, initial_guess=list(initial_guess), constants=constants or {},
            name=self.name,
        )

    def named_params(self, values) -> dict[str, float]:
        """Zip a raw parameter vector into a ``{name: value}`` dict."""
        values = list(values)
        if len(values) != self.n_params:
            raise ValueError(f"expected {self.n_params} values, got {len(values)}")
        return dict(zip(self.param_names, (float(v) for v in values)))


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------


def _guess_intact(spec: Spectrum) -> list[float]:
    Re = _estimate_Re(spec)
    Rpo = _estimate_Rp(spec, Re)
    C = _estimate_C(spec)
    # R0, R1(=R_po), CPE1_0(=Q_c), CPE1_1(=n_c)
    return [Re, Rpo, C, 0.90]


def _guess_degraded(spec: Spectrum) -> list[float]:
    Re = _estimate_Re(spec)
    Rpo, Rct = _estimate_Rpo_Rct(spec, Re)
    Cc = _estimate_C(spec)
    # R0, CPE1_0(=Q_c), CPE1_1(=n_c), R1(=R_po), CPE2_0(=Q_dl), CPE2_1(=n_dl), R2(=R_ct)
    return [Re, Cc, 0.90, Rpo, max(100.0 * Cc, 1e-6), 0.80, Rct]


INTACT_COATING = CircuitModel(
    name="intact_coating",
    circuit="R0-p(R1,CPE1)",
    param_names=("R_e", "R_po", "Q_c", "n_c"),
    units=("Ohm", "Ohm", "Ohm^-1 s^n", ""),
    description="One time constant: electrolyte R, coating pore R || coating CPE.",
    _guess=_guess_intact,
)

DEGRADED_COATING = CircuitModel(
    name="degraded_coating",
    circuit="R0-p(CPE1,R1-p(CPE2,R2))",
    param_names=("R_e", "Q_c", "n_c", "R_po", "Q_dl", "n_dl", "R_ct"),
    units=("Ohm", "Ohm^-1 s^n", "", "Ohm", "Ohm^-1 s^n", "", "Ohm"),
    description=(
        "Two nested time constants: coating CPE in parallel with "
        "(pore R in series with substrate charge-transfer R || double-layer CPE)."
    ),
    _guess=_guess_degraded,
)

MODELS: dict[str, CircuitModel] = {
    INTACT_COATING.name: INTACT_COATING,
    DEGRADED_COATING.name: DEGRADED_COATING,
}


def get_model(model: str | CircuitModel) -> CircuitModel:
    """Resolve a model name (or pass-through a :class:`CircuitModel`)."""
    if isinstance(model, CircuitModel):
        return model
    try:
        return MODELS[model]
    except KeyError:
        raise KeyError(
            f"unknown model {model!r}; available: {sorted(MODELS)}"
        ) from None
