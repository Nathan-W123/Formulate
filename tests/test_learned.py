"""The learned boiling point, and the two gates that make it safe to ship.

This expert is the only fitted model in the panel and it is deliberately narrow.
It exists because a discovery run generated thirty-five novel structures and
Joback could type almost none of them, so none could be ranked: in this panel
the boiling point is the root of the dependency chain, and a generator whose
inventions cannot be scored discovers nothing.
"""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import molecule_candidate
from formulate.core.conditions import Conditions
from formulate.experts.base import PredictionRequest
from formulate.experts.joback import JobackThermalExpert
from formulate.experts.learned import (
    LearnedBoilingPointExpert,
    boiling_point_model,
    sklearn_available,
)

pytestmark = [
    requires_rdkit,
    pytest.mark.skipif(not sklearn_available(), reason="scikit-learn is not installed"),
]

#: Structures the evolutionary explorer actually produced when asked for a
#: high-boiling, low-freezing, hydrophobic liquid. Every one is a cyclic
#: carbonate or lactone, and Joback has a group for none of them.
GENERATED = [
    "CCCCC1COC(=O)O1",
    "CC1(C)COC(=O)O1",
    "O=C1OCC(CC(F)(F)F)O1",
    "CCC1(C)COC(=O)O1",
]


def _ask(expert, smiles):
    request = PredictionRequest(
        candidate=molecule_candidate(smiles),
        properties=frozenset({"normal_boiling_point"}),
        conditions=Conditions.standard(),
    )
    return next(iter(expert.predict(request)))


# -- the gap it closes -----------------------------------------------------


@pytest.mark.parametrize("smiles", GENERATED)
def test_it_answers_where_group_contribution_has_no_groups(smiles):
    assert _ask(JobackThermalExpert(), smiles).quantity is None
    prediction = _ask(LearnedBoilingPointExpert(), smiles)
    assert prediction.quantity is not None
    # A cyclic carbonate boils well above water and well below decomposition.
    assert 350.0 < prediction.quantity.to("K").value < 600.0


def test_the_generated_structures_are_inside_the_training_distribution():
    """They are not exotic - they are simply absent from a 1987 table.

    Worth asserting, because the interesting claim is not that a model returns
    a number for anything. It is that these particular molecules are ordinary
    enough to have thousands of measured relatives, and only group contribution
    finds them unfamiliar.
    """
    expert = LearnedBoilingPointExpert()
    for smiles in GENERATED:
        assert expert.assess_domain(molecule_candidate(smiles)).in_domain


# -- the gates -------------------------------------------------------------


@pytest.mark.parametrize("smiles", ["O", "C"])
def test_a_molecule_the_trees_disagree_about_is_refused(smiles):
    """Water came back at 700 K against a true 373, and passed the distance test.

    In a standardised descriptor space a three-atom molecule is not obviously
    far from anything, so nearest-neighbour distance let it through. The forest
    itself knew - it quoted plus or minus 486 K - and the refusal now listens to
    that instead.
    """
    prediction = _ask(LearnedBoilingPointExpert(), smiles)
    assert prediction.quantity is None
    assert "disagree" in " ".join(prediction.notes)


def test_the_spread_gate_is_calibrated_rather_than_chosen():
    from formulate.experts.learned import _MAX_RELATIVE_SPREAD

    # Measured over the held-out set: no gate gives 51.8 K, this gives 26.6 K
    # while still answering 91 per cent of compounds.
    assert 0.10 <= _MAX_RELATIVE_SPREAD <= 0.20


def test_the_stated_uncertainty_is_earned_not_asserted():
    """prefer() selects on the tightest bound, so an unearned one hijacks it."""
    model = boiling_point_model()
    assert 0.60 <= model["held_out_coverage"] <= 0.76
    assert model["n_train"] > 5000
    assert model["n_held_out"] > 1000


def test_joback_still_wins_where_it_has_groups():
    """This expert is a fallback, and the notes say so rather than implying more."""
    toluene = "Cc1ccccc1"
    joback = _ask(JobackThermalExpert(), toluene)
    learned = _ask(LearnedBoilingPointExpert(), toluene)
    assert joback.quantity is not None and learned.quantity is not None
    measured = 383.75
    assert abs(joback.quantity.to("K").value - measured) < abs(
        learned.quantity.to("K").value - measured
    )
    assert "Joback is the better estimate" in " ".join(learned.notes)


# -- reproducibility -------------------------------------------------------


def test_the_repository_carries_data_rather_than_a_pickled_model():
    """A fitted forest is 34 MB, opaque and version-fragile; the data is 307 KB."""
    import json

    from formulate.experts.learned import _DATA

    assert _DATA.exists()
    assert _DATA.stat().st_size < 1_000_000
    document = json.loads(_DATA.read_text())
    assert len(document["rows"]) > 10_000
    assert "JOBACK is excluded deliberately" in document["provenance"]


def test_the_training_data_excludes_the_calibration_compounds():
    """Otherwise the benchmark would be scoring the model on its own training set."""
    import json

    from rdkit import Chem

    from formulate.experts.learned import _DATA
    from formulate.exploration.database import load_reference_compounds

    held_out = set()
    for record in load_reference_compounds():
        mol = Chem.MolFromSmiles(record["smiles"])
        if mol is not None:
            held_out.add(Chem.MolToInchiKey(mol))

    for smiles, _ in json.loads(_DATA.read_text())["rows"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        assert Chem.MolToInchiKey(mol) not in held_out, smiles


# --------------------------------------------------------------------------
# The abstraction, exercised by a second property with different units
# --------------------------------------------------------------------------


def test_every_learnable_property_declares_what_the_base_class_must_not_assume():
    """The gaps this table closed were all silent on the boiling point.

    A noise floor of one kelvin, a relative-spread denominator floored at one
    kelvin, and the letter K in three messages were hard-coded rather than
    declared. On a refractive index of 1.36 the noise floor alone forced a
    stated spread of 1.0, the gate read that as the trees disagreeing by 73 per
    cent, and the expert declined every molecule it was asked about.
    """
    from formulate.core.properties import get_property
    from formulate.experts.learned import LEARNABLE

    for prop, spec in LEARNABLE.items():
        definition = get_property(prop)
        assert spec.filename.endswith(".json")
        assert spec.noise_floor > 0
        assert spec.scale_floor > 0
        assert spec.comparison and spec.provenance_note
        if spec.unit:
            # The declared unit must be the one the property is registered in.
            from formulate.core.units import convert

            convert(1.0, spec.unit, definition.canonical_unit)


def test_a_value_is_printed_to_where_its_noise_floor_sits():
    """Printing further claims a precision the property does not have."""
    from formulate.experts.learned import LEARNABLE

    assert LEARNABLE["normal_boiling_point"].format(351.42) == "351 K"
    assert LEARNABLE["refractive_index"].format(1.36123) == "1.3612"


def test_the_noise_floor_is_per_property_rather_than_one_kelvin_everywhere():
    from formulate.experts.learned import LEARNABLE

    assert LEARNABLE["refractive_index"].noise_floor < 0.01
    assert LEARNABLE["normal_boiling_point"].noise_floor == 1.0


@requires_rdkit
def test_the_refractive_index_expert_answers_rather_than_declining_everything():
    from formulate.core.candidate import molecule_candidate
    from formulate.core.conditions import Conditions
    from formulate.experts.base import PredictionRequest
    from formulate.experts.learned import LearnedRefractiveIndexExpert

    expert = LearnedRefractiveIndexExpert()
    if not expert.is_available():
        pytest.skip(expert.unavailable_reason())

    answered = 0
    for smiles in ("CCO", "Cc1ccccc1", "c1ccccc1", "CCCCCC", "CC(C)=O"):
        prediction = expert.predict(
            PredictionRequest(
                candidate=molecule_candidate(smiles),
                properties=frozenset({"refractive_index"}),
                conditions=Conditions.standard(),
            )
        )[0]
        if prediction.quantity is not None:
            answered += 1
            assert 1.2 < prediction.quantity.value < 2.0
            assert 0.0 < prediction.uncertainty.std < 0.1
    assert answered == 5


@requires_rdkit
def test_the_refractive_index_expert_states_its_units_and_provenance():
    from formulate.core.candidate import molecule_candidate
    from formulate.core.conditions import Conditions
    from formulate.experts.base import PredictionRequest
    from formulate.experts.learned import LearnedRefractiveIndexExpert

    expert = LearnedRefractiveIndexExpert()
    if not expert.is_available():
        pytest.skip(expert.unavailable_reason())
    prediction = expert.predict(
        PredictionRequest(
            candidate=molecule_candidate("Cc1ccccc1"),
            properties=frozenset({"refractive_index"}),
            conditions=Conditions.standard(),
        )
    )[0]
    assert prediction.quantity.unit in ("", "dimensionless")
    assert prediction.expert_version
    assert prediction.provenance is not None
    assert " K" not in prediction.uncertainty.basis, "a kelvin leaked into a dimensionless property"
    assert any("refractive index" in note for note in prediction.notes)
    assert any("Lorentz-Lorenz" in note for note in prediction.notes)


@requires_rdkit
def test_inference_is_reproducible():
    """Same structure, same answer, twice, from a freshly built expert.

    To well inside the property's noise floor rather than to the last bit:
    scikit-learn averages its trees across threads, so the summation order
    varies and two identical predictions can differ by one unit in the last
    place. Demanding bit-identity would make this test fail on a difference of
    2e-16 in a quantity whose fourth decimal is already noise, which is a
    statement about floating-point reduction rather than about the model.
    """
    from formulate.core.candidate import molecule_candidate
    from formulate.core.conditions import Conditions
    from formulate.experts.base import PredictionRequest
    from formulate.experts.learned import LearnedRefractiveIndexExpert

    def once():
        expert = LearnedRefractiveIndexExpert()
        if not expert.is_available():
            pytest.skip(expert.unavailable_reason())
        return expert.predict(
            PredictionRequest(
                candidate=molecule_candidate("CCCCO"),
                properties=frozenset({"refractive_index"}),
                conditions=Conditions.standard(),
            )
        )[0]

    from formulate.experts.learned import LEARNABLE

    floor = LEARNABLE["refractive_index"].noise_floor
    first, second = once(), once()
    assert first.quantity.value == pytest.approx(second.quantity.value, abs=floor / 1000)
    assert first.uncertainty.std == pytest.approx(second.uncertainty.std, abs=floor / 1000)
    assert first.expert_version == second.expert_version


@requires_rdkit
def test_a_structure_far_outside_the_training_data_is_refused():
    from formulate.core.candidate import molecule_candidate
    from formulate.experts.learned import LearnedRefractiveIndexExpert

    expert = LearnedRefractiveIndexExpert()
    if not expert.is_available():
        pytest.skip(expert.unavailable_reason())
    exotic = expert.assess_domain(molecule_candidate("C[Pt](C)(C)C"))
    ordinary = expert.assess_domain(molecule_candidate("CCCCO"))
    assert ordinary.in_domain
    assert not exotic.in_domain or exotic.score <= ordinary.score


@requires_rdkit
def test_an_unparseable_structure_fails_rather_than_raising():
    from formulate.core.candidate import molecule_candidate
    from formulate.core.prediction import PredictionStatus
    from formulate.core.conditions import Conditions
    from formulate.experts.base import PredictionRequest
    from formulate.experts.learned import LearnedRefractiveIndexExpert

    expert = LearnedRefractiveIndexExpert()
    if not expert.is_available():
        pytest.skip(expert.unavailable_reason())
    prediction = expert.predict(
        PredictionRequest(
            candidate=molecule_candidate("C[Xx]C"),
            properties=frozenset({"refractive_index"}),
            conditions=Conditions.standard(),
        )
    )[0]
    assert prediction.status in (PredictionStatus.FAILED, PredictionStatus.OUT_OF_DOMAIN)


def test_the_rejected_permittivity_model_is_recorded_with_its_reason():
    """So the experiment is not repeated, and so the reason is not "it was bad".

    It was not simply bad: gated it looks excellent, and that number is an
    artefact of the gate admitting only nonpolar molecules.
    """
    from formulate.experts.learned import _RELATIVE_PERMITTIVITY_REJECTED

    assert "nonpolar" in _RELATIVE_PERMITTIVITY_REJECTED
    assert "not shipped" in _RELATIVE_PERMITTIVITY_REJECTED


def test_the_rejected_models_are_not_registered():
    from formulate.core.candidate import MaterialClass
    from formulate.experts import default_registry

    registry = default_registry()
    for prop in ("relative_permittivity",):
        assert registry.coverage([prop], MaterialClass.MOLECULE) == {} or not registry.coverage(
            [prop], MaterialClass.MOLECULE
        ).get(prop)
