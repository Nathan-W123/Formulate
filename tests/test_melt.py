"""Melt properties of a polymer: whether it melts, how thick, how slack.

The interesting cases here are the refusals. An amorphous polymer has no
melting point, and saying so is an answer rather than a coverage gap; a melt
viscosity without a chain length is not a number at all; and WLF referenced to
Tg has a range that a hot-melt nozzle sits outside of.
"""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MonomerUnit,
    PolymerSpec,
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import Prediction, PredictionStatus
from formulate.core.quantity import Quantity
from formulate.experts.base import PredictionRequest
from formulate.experts.melt import (
    AMORPHOUS,
    MELTING_POINTS,
    SURFACE_TENSION_DGDT,
    PolymerMeltExpert,
    melt_viscosity,
)

PE, PS, PMMA = "[*]CC[*]", "[*]CC(c1ccccc1)[*]", "[*]CC(C)(C(=O)OC)[*]"
NYLON66 = "[*]NCCCCCCNC(=O)CCCCC(=O)[*]"


def _polymer(*repeat_units: str, mn_kg_mol: float | None = None):
    monomers = tuple(
        MonomerUnit(smiles=s, mole_fraction=1.0 / len(repeat_units)) for s in repeat_units
    )
    kwargs = {}
    if mn_kg_mol is not None:
        kwargs["number_average_molar_mass"] = Quantity(value=mn_kg_mol, unit="kg/mol")
    return Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(monomers=monomers, **kwargs),
        conditions=Conditions.standard(),
    )


def _predict(candidate, prop, *, temperature_k=298.15, context=None):
    request = PredictionRequest(
        candidate=candidate,
        properties=frozenset({prop}),
        conditions=Conditions(temperature=Quantity(value=temperature_k, unit="K")),
        context=context or {},
    )
    return next(p for p in PolymerMeltExpert().predict(request) if p.property == prop)


def _context(tg_k: float, me_kg_mol: float) -> dict[str, Prediction]:
    """Stand in for the upstream experts this one depends on."""
    return {
        "glass_transition_temperature": Prediction(
            property="glass_transition_temperature",
            quantity=Quantity(value=tg_k, unit="K"),
            expert_id="stub",
        ),
        "entanglement_molar_mass": Prediction(
            property="entanglement_molar_mass",
            quantity=Quantity(value=me_kg_mol, unit="kg/mol"),
            expert_id="stub",
        ),
    }


# -- melting point ---------------------------------------------------------


@requires_rdkit
def test_a_semicrystalline_polymer_gets_its_measured_melting_point():
    prediction = _predict(_polymer(NYLON66), "melting_point")
    assert prediction.quantity is not None
    assert prediction.quantity.to_canonical().value == pytest.approx(538.0)


@requires_rdkit
def test_an_amorphous_polymer_is_refused_because_it_has_no_melting_point():
    """Not a coverage gap. Polystyrene softens; it does not melt."""
    prediction = _predict(_polymer(PS), "melting_point")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    assert "no melting point" in " ".join(prediction.notes)


@requires_rdkit
def test_an_untabulated_polymer_is_refused_differently_from_an_amorphous_one():
    """The two refusals must not read alike: one is a gap, one is a fact."""
    gap = _predict(_polymer("[*]CC(CC)[*]"), "melting_point")
    fact = _predict(_polymer(PMMA), "melting_point")
    assert gap.quantity is None and fact.quantity is None
    assert "cannot be estimated" in " ".join(gap.notes)
    assert "softens through" in " ".join(fact.notes)


def test_no_polymer_is_both_amorphous_and_given_a_melting_point():
    assert not set(MELTING_POINTS) & set(AMORPHOUS)


# -- surface tension -------------------------------------------------------


@requires_rdkit
def test_surface_tension_is_corrected_from_room_temperature_to_the_melt():
    cold = _predict(_polymer(PE), "surface_tension", temperature_k=298.15)
    hot = _predict(_polymer(PE), "surface_tension", temperature_k=473.15)
    drop = cold.quantity.value - hot.quantity.value
    assert drop == pytest.approx(-SURFACE_TENSION_DGDT * 175.0, rel=1e-6)
    assert hot.quantity.value < cold.quantity.value


@requires_rdkit
def test_a_copolymer_is_refused_rather_than_averaged():
    prediction = _predict(_polymer(PS, PE), "surface_tension")
    assert prediction.status is PredictionStatus.UNSUPPORTED


# -- melt viscosity --------------------------------------------------------


@requires_rdkit
def test_viscosity_without_a_chain_length_is_refused_not_guessed():
    """The same repeat unit spans six orders between an oligomer and a polymer."""
    prediction = _predict(
        _polymer(PS), "shear_viscosity", temperature_k=423.15,
        context=_context(373.0, 18.1),
    )
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "molar mass" in " ".join(prediction.notes)


@requires_rdkit
def test_an_unentangled_chain_is_refused_because_the_power_law_does_not_hold():
    prediction = _predict(
        _polymer(PS, mn_kg_mol=12.0), "shear_viscosity", temperature_k=423.15,
        context=_context(373.0, 18.1),
    )
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "entanglement threshold" in " ".join(prediction.notes)


@requires_rdkit
def test_wlf_refuses_far_above_the_glass_transition():
    """A hot-melt nozzle sits outside the range WLF is referenced over."""
    prediction = _predict(
        _polymer(PE, mn_kg_mol=50.0), "shear_viscosity", temperature_k=473.15,
        context=_context(198.0, 1.15),
    )
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "WLF" in " ".join(prediction.notes)


@requires_rdkit
def test_below_the_glass_transition_it_is_a_solid_not_a_melt():
    prediction = _predict(
        _polymer(PS, mn_kg_mol=100.0), "shear_viscosity", temperature_k=300.0,
        context=_context(373.0, 18.1),
    )
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "solid, not a melt" in " ".join(prediction.notes)


@requires_rdkit
def test_viscosity_in_range_carries_an_order_of_magnitude_of_uncertainty():
    """Universal WLF constants are worth about a decade, and must say so."""
    prediction = _predict(
        _polymer(PS, mn_kg_mol=100.0), "shear_viscosity", temperature_k=443.15,
        context=_context(373.0, 18.1),
    )
    assert prediction.quantity is not None
    ratio = prediction.uncertainty.std / prediction.quantity.value
    assert 4.0 < ratio < 5.0     # (10^1 - 1)/2


def test_the_viscosity_form_is_anchored_at_the_definition_of_tg():
    """At Tg and the critical mass it must return the value that defines Tg."""
    assert melt_viscosity(2.0, 1.0, 373.0, 373.0) == pytest.approx(1e12)


def test_viscosity_falls_with_temperature_and_rises_with_chain_length():
    hot = melt_viscosity(50.0, 10.0, 373.0, 443.0)
    cold = melt_viscosity(50.0, 10.0, 373.0, 403.0)
    longer = melt_viscosity(100.0, 10.0, 373.0, 443.0)
    assert hot < cold
    assert longer > hot
    assert longer / hot == pytest.approx(2.0**3.4, rel=1e-6)
