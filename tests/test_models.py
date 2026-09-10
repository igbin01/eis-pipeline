"""Tests for eispipe.models."""

from __future__ import annotations

import numpy as np
import pytest

from eispipe.models import (
    DEGRADED_COATING,
    INTACT_COATING,
    MODELS,
    CircuitModel,
    get_model,
)
from eispipe.spectrum import Spectrum
from conftest import synth_coating, synth_degraded


def test_registry_and_get_model():
    assert set(MODELS) == {"intact_coating", "degraded_coating"}
    assert get_model("intact_coating") is INTACT_COATING
    assert get_model(DEGRADED_COATING) is DEGRADED_COATING
    with pytest.raises(KeyError, match="unknown model"):
        get_model("nope")


def test_circuit_strings_and_param_order_are_pinned():
    # Regression guard: the degraded model must NOT reuse an element label and
    # must nest the charge-transfer branch inside the pore-resistance branch.
    assert INTACT_COATING.circuit == "R0-p(R1,CPE1)"
    assert INTACT_COATING.param_names == ("R_e", "R_po", "Q_c", "n_c")

    assert DEGRADED_COATING.circuit == "R0-p(CPE1,R1-p(CPE2,R2))"
    assert DEGRADED_COATING.param_names == (
        "R_e",
        "Q_c",
        "n_c",
        "R_po",
        "Q_dl",
        "n_dl",
        "R_ct",
    )


@pytest.mark.parametrize("model", [INTACT_COATING, DEGRADED_COATING])
def test_param_names_match_impedance(model: CircuitModel):
    from impedance.models.circuits import CustomCircuit

    assert len(model.param_names) == len(model.units) == model.n_params
    cc = CustomCircuit(model.circuit, initial_guess=[1.0] * model.n_params)
    assert len(cc.get_param_names()[0]) == model.n_params


@pytest.mark.parametrize(
    "model, arrays",
    [
        (INTACT_COATING, synth_coating()),
        (DEGRADED_COATING, synth_degraded()),
    ],
)
def test_initial_guess_is_sane(model, arrays):
    freq, Z = arrays
    spec = Spectrum(freq, Z)
    guess = model.initial_guess(spec)

    assert len(guess) == model.n_params
    assert np.all(np.isfinite(guess))
    d = model.named_params(guess)
    assert d["R_e"] > 0
    assert d["R_po"] > 0
    assert d["Q_c"] > 0
    assert 0 < d["n_c"] <= 1


def test_build_returns_unfitted_customcircuit():
    freq, Z = synth_coating()
    spec = Spectrum(freq, Z)
    cc = INTACT_COATING.build(spec=spec)

    assert cc.circuit == "R0-p(R1,CPE1)"
    assert cc._is_fit() is False
    assert len(cc.initial_guess) == INTACT_COATING.n_params

    with pytest.raises(ValueError, match="pass initial_guess or spec"):
        INTACT_COATING.build()


def test_named_params_length_check():
    with pytest.raises(ValueError, match="expected 4 values"):
        INTACT_COATING.named_params([1, 2, 3])
