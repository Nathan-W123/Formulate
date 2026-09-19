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
def test_an_unentangled_chain_gets_rouse_not_the_reptation_power():
    """A wax lives below the entanglement threshold, so a blend cannot be
    scored without this branch - but it is the other law, not an
    extrapolation of the 3.4 power into a regime it does not hold in."""
    prediction = _predict(
        _polymer(PS, mn_kg_mol=12.0), "shear_viscosity", temperature_k=423.15,
        context=_context(373.0, 18.1),
    )
    assert prediction.quantity is not None
    joined = " ".join(prediction.notes)
    assert "unentangled" in joined
    assert "carries no load" in joined


def test_the_two_chain_length_branches_join_at_the_threshold():
    """Rouse and reptation agree at the critical mass, which is where both hold."""
    from formulate.experts.melt import CRITICAL_OVER_ENTANGLEMENT, melt_viscosity

    me = 1.15
    critical = CRITICAL_OVER_ENTANGLEMENT * me
    below = melt_viscosity(critical * 0.999, me, 198.0, 250.0)
    above = melt_viscosity(critical * 1.001, me, 198.0, 250.0)
    assert below == pytest.approx(above, rel=5e-3)


def test_a_wax_is_orders_of_magnitude_thinner_than_the_polymer():
    """Which is the whole reason a blend can carry a longer backbone."""
    from formulate.experts.melt import melt_viscosity

    wax = melt_viscosity(0.8, 1.15, 198.0, 473.15, 27.0e3)
    polymer = melt_viscosity(20.0, 1.15, 198.0, 473.15, 27.0e3)
    assert polymer / wax > 1000.0


@requires_rdkit
def test_above_the_wlf_range_arrhenius_carries_it_and_says_so():
    """A hot-melt nozzle sits past where WLF is referenced, so a tabulated
    flow activation energy carries the curve the rest of the way."""
    prediction = _predict(
        _polymer(PE, mn_kg_mol=50.0), "shear_viscosity", temperature_k=473.15,
        context=_context(198.0, 1.15),
    )
    assert prediction.quantity is not None
    # Measured HDPE at this chain length and 200 C is a few thousand Pa.s.
    assert 500.0 < prediction.quantity.value < 10000.0
    assert "Arrhenius" in " ".join(prediction.notes)


@requires_rdkit
def test_without_an_activation_energy_the_melt_regime_is_refused():
    """The Arrhenius branch is tabulated, so a polymer outside the table gets
    a refusal rather than a WLF extrapolation two hundred degrees past its range."""
    prediction = _predict(
        _polymer("[*]CC(CC)[*]", mn_kg_mol=50.0), "shear_viscosity",
        temperature_k=473.15, context=_context(200.0, 2.0),
    )
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "activation energy" in " ".join(prediction.notes)


@requires_rdkit
def test_the_two_viscosity_branches_join_without_a_step():
    """WLF hands over to Arrhenius at Tg + WLF_RANGE_K.

    The value is continuous by construction - Arrhenius starts from whatever
    WLF says at the crossover - which is what is asserted here. The *slope* is
    not continuous, and is not claimed to be: the two forms have different
    temperature dependences and joining them at a point is the whole idea.
    """
    from formulate.experts.melt import WLF_RANGE_K, melt_viscosity

    tg, crossover = 198.0, 198.0 + WLF_RANGE_K
    at = melt_viscosity(50.0, 1.15, tg, crossover, 27.0e3)
    just_above = melt_viscosity(50.0, 1.15, tg, crossover + 1e-9, 27.0e3)
    assert just_above == pytest.approx(at, rel=1e-9)


@requires_rdkit
def test_atactic_polypropylene_is_not_given_the_isotactic_melting_point():
    """Keyed on the repeat unit the two are identical, and one does not melt."""
    from formulate.core.candidate import Tacticity

    atactic = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(
            monomers=(MonomerUnit(smiles="[*]CC(C)[*]"),), tacticity=Tacticity.ATACTIC
        ),
        conditions=Conditions.standard(),
    )
    isotactic = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(
            monomers=(MonomerUnit(smiles="[*]CC(C)[*]"),), tacticity=Tacticity.ISOTACTIC
        ),
        conditions=Conditions.standard(),
    )
    assert _predict(atactic, "melting_point").quantity is None
    assert _predict(isotactic, "melting_point").quantity is not None


@requires_rdkit
def test_unstated_tacticity_is_refused_rather_than_assumed():
    prediction = _predict(_polymer("[*]CC(C)[*]"), "melting_point")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "does not state its tacticity" in " ".join(prediction.notes)


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


@requires_rdkit
def test_polycaprolactone_melts_low_enough_to_handle():
    """The safety-relevant entry: 60 C against polyethylene's 135 and the
    200 C a hot-melt nozzle runs at. Molten polymer sticks to skin, so the
    melting point of the material is a burn risk, not just a process setting."""
    prediction = _predict(_polymer("[*]CCCCCC(=O)O[*]"), "melting_point")
    assert prediction.quantity.to_canonical().value == pytest.approx(333.0)


@requires_rdkit
def test_a_measured_modulus_does_not_need_a_chain_dimension():
    """Gating it there refused polycaprolactone a stiffness sitting in a table,
    and with it the whole low-melting branch of the search."""
    from formulate.core.candidate import Candidate, MaterialClass
    from formulate.experts import polymer_registry
    from formulate.experts.base import PredictionRequest as Req
    from formulate.experts.mechanical import CHAIN_DIMENSIONS

    assert "[*]CCCCCC(=O)O[*]" not in CHAIN_DIMENSIONS, "the point of the test"

    candidate = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(monomers=(MonomerUnit(smiles="[*]CCCCCC(=O)O[*]"),)),
        conditions=Conditions.standard(),
    )
    registry = polymer_registry()
    wanted = frozenset({"youngs_modulus", "glass_transition_temperature", "amorphous_density"})
    context = {}
    for expert in registry.resolution_order(
        registry.experts_for(wanted, MaterialClass.POLYMER)
    ):
        for prediction in expert.predict(
            Req(candidate=candidate, properties=wanted,
                conditions=candidate.conditions, context=dict(context))
        ):
            if prediction.is_usable:
                context.setdefault(prediction.property, prediction)
    assert context["youngs_modulus"].quantity.to_canonical().value == pytest.approx(0.4e9)
