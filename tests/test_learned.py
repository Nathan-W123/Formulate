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
