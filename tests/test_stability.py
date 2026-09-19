"""The top of the processing window, and whether there is a window at all."""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MonomerUnit,
    PolymerSpec,
    Tacticity,
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import PredictionStatus
from formulate.core.quantity import Quantity
from formulate.experts.base import PredictionRequest
from formulate.experts.stability import (
    DECOMPOSITION_REFERENCE,
    DECOMPOSITION_SPREAD,
    ThermalStabilityExpert,
    group_decomposition,
    named_decomposition,
)

PAA = "[*]CC(C(=O)O)[*]"
PET = "[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]"
PE = "[*]CC[*]"
PP = "[*]CC(C)[*]"


def _predict(repeat, prop, tacticity=Tacticity.ATACTIC):
    candidate = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(
            monomers=(MonomerUnit(smiles=repeat),),
            tacticity=tacticity,
            number_average_molar_mass=Quantity(value=50.0, unit="kg/mol"),
        ),
        conditions=Conditions.standard(),
    )
    request = PredictionRequest(
        candidate=candidate,
        properties=frozenset({prop}),
        conditions=Conditions(temperature=Quantity(value=298.15, unit="K")),
    )
    return next(p for p in ThermalStabilityExpert().predict(request) if p.property == prop)


# -- the case this expert exists for ---------------------------------------


@requires_rdkit
def test_poly_acrylic_acid_does_not_survive_a_two_hundred_degree_melt():
    """The candidate a blend search ranked first for a 200 C process. Nothing
    it reported was wrong; nothing asked the question that disqualifies it."""
    prediction = _predict(PAA, "decomposition_temperature")
    assert prediction.quantity is not None
    assert prediction.quantity.to("degC").value < 200.0
    assert "anhydride" in " ".join(prediction.notes)


@requires_rdkit
def test_the_named_reactions_fire_and_the_clean_melts_do_not():
    named = ["[*]CC(C(=O)O)[*]", "[*]CC(C)(C(=O)O)[*]", "[*]CC(Cl)[*]",
             "[*]CC(C#N)[*]", "[*]CC(O)[*]"]
    clean = [PET, "[*]CC(C)(C(=O)OC)[*]", "[*]CCCCCC(=O)O[*]", "[*]CCO[*]",
             "[*]NCCCCCC(=O)[*]"]
    for repeat in named:
        assert named_decomposition(repeat) is not None, repeat
    for repeat in clean:
        assert named_decomposition(repeat) is None, repeat


@requires_rdkit
def test_capping_the_attachment_points_does_not_invent_a_functional_group():
    """The bug this caught, and it is the class of bug this repository keeps
    finding: PET written [*]OCCO... becomes [H]OCCO... when capped, whose
    terminal OX2H1 on a CX4 is a hydroxyl. PET matched the poly(vinyl alcohol)
    dehydration rule and was told it decomposes at 200 C - a polymer that is
    melt-spun at 280 every day."""
    assert named_decomposition(PET) is None
    assert named_decomposition("[*]CC(O)[*]") is not None


# -- the group sum ---------------------------------------------------------


@requires_rdkit
def test_the_group_sum_reproduces_its_own_reference_set():
    worst = 0.0
    for repeat, measured in DECOMPOSITION_REFERENCE.items():
        predicted = group_decomposition(repeat)
        assert predicted is not None, repeat
        worst = max(worst, max(predicted / measured, measured / predicted))
    assert worst < DECOMPOSITION_SPREAD


@requires_rdkit
def test_the_bar_carried_is_the_held_out_one_not_the_in_sample_one():
    prediction = _predict(PE, "decomposition_temperature")
    relative = prediction.uncertainty.std / prediction.quantity.value
    assert relative == pytest.approx(DECOMPOSITION_SPREAD - 1.0, rel=1e-6)


@requires_rdkit
def test_an_aromatic_backbone_outlasts_an_aliphatic_one():
    aromatic = group_decomposition(PET)
    aliphatic = group_decomposition("[*]OCCOC(=O)CCCCC(=O)[*]")
    assert aromatic > aliphatic


# -- crystallisability -----------------------------------------------------


@requires_rdkit
def test_an_atactic_chain_with_a_stereocentre_cannot_crystallise():
    prediction = _predict(PAA, "crystallisability")
    assert prediction.quantity is not None
    assert prediction.quantity.value == 0.0
    assert "orientation lock" in " ".join(prediction.notes)


@requires_rdkit
def test_a_regular_chain_can():
    prediction = _predict(PE, "crystallisability")
    assert prediction.quantity.value == 1.0


@requires_rdkit
def test_an_unstated_tacticity_is_refused_rather_than_picked():
    """The two answers are a fibre and a piece of string."""
    prediction = _predict(PP, "crystallisability", tacticity=Tacticity.UNSPECIFIED)
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "picking" in " ".join(prediction.notes)


@requires_rdkit
def test_tacticity_decides_it_where_it_decides_crystallinity():
    assert _predict(PP, "crystallisability", Tacticity.ISOTACTIC).quantity.value == 1.0
    assert _predict(PP, "crystallisability", Tacticity.ATACTIC).quantity.value == 0.0


@requires_rdkit
def test_a_copolymer_is_refused_for_both_properties():
    candidate = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(
            monomers=(
                MonomerUnit(smiles=PE, mole_fraction=0.5),
                MonomerUnit(smiles=PP, mole_fraction=0.5),
            ),
            number_average_molar_mass=Quantity(value=50.0, unit="kg/mol"),
        ),
        conditions=Conditions.standard(),
    )
    request = PredictionRequest(
        candidate=candidate,
        properties=frozenset({"decomposition_temperature", "crystallisability"}),
        conditions=Conditions(temperature=Quantity(value=298.15, unit="K")),
    )
    for prediction in ThermalStabilityExpert().predict(request):
        assert prediction.status is PredictionStatus.UNSUPPORTED
