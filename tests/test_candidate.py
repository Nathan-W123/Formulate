"""The canonical candidate object."""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import (
    Candidate,
    ComponentRole,
    ConstraintViolation,
    MaterialClass,
    MixtureComponent,
    MixtureSpec,
    MoleculeSpec,
    MonomerRole,
    MonomerUnit,
    PolymerSpec,
    dedupe,
    molecule_candidate,
)
from formulate.core.conditions import Conditions
from formulate.core.errors import CandidateError
from formulate.core.quantity import Quantity


def test_payload_must_match_material_class():
    with pytest.raises(CandidateError):
        Candidate(material_class=MaterialClass.MOLECULE)
    with pytest.raises(CandidateError):
        Candidate(
            material_class=MaterialClass.MOLECULE,
            molecule=MoleculeSpec(smiles="CCO"),
            mixture=MixtureSpec(
                components=(
                    MixtureComponent(fraction=1.0, molecule=MoleculeSpec(smiles="CCO")),
                )
            ),
        )


@requires_rdkit
def test_identity_is_independent_of_smiles_spelling():
    assert molecule_candidate("OCC").candidate_id == molecule_candidate("CCO").candidate_id


@requires_rdkit
def test_dedupe_collapses_equivalent_structures():
    assert len(dedupe([molecule_candidate("OCC"), molecule_candidate("CCO")])) == 1


def test_conditions_are_part_of_identity_but_not_of_structure():
    base = molecule_candidate("CCO")
    hot = base.with_conditions(Conditions(temperature=Quantity(value=400.0, unit="K")))
    assert hot.candidate_id != base.candidate_id
    assert hot.structure_id == base.structure_id


def test_attaching_results_does_not_change_identity():
    from formulate.core.candidate import CandidateResults

    base = molecule_candidate("CCO")
    scored = base.with_results(CandidateResults(scalar_score=0.9))
    assert scored.candidate_id == base.candidate_id


def test_polymer_fractions_must_sum_to_one():
    with pytest.raises(Exception):
        PolymerSpec(
            monomers=(
                MonomerUnit(smiles="[*]CC[*]", mole_fraction=0.7),
                MonomerUnit(smiles="[*]CC(C)[*]", mole_fraction=0.2),
            )
        )


def test_polymer_end_groups_are_excluded_from_the_fraction_sum():
    polymer = PolymerSpec(
        monomers=(
            MonomerUnit(smiles="[*]CC[*]", mole_fraction=1.0),
            MonomerUnit(smiles="[*]C", mole_fraction=0.02, role=MonomerRole.END_GROUP),
        )
    )
    assert len(polymer.monomers) == 2


def test_mixture_fractions_must_sum_to_one():
    with pytest.raises(Exception):
        MixtureSpec(
            components=(
                MixtureComponent(fraction=0.5, molecule=MoleculeSpec(smiles="CCO")),
            )
        )


@requires_rdkit
def test_mixture_rejects_duplicate_components():
    with pytest.raises(Exception):
        MixtureSpec(
            components=(
                MixtureComponent(fraction=0.5, molecule=MoleculeSpec(smiles="CCO")),
                MixtureComponent(fraction=0.5, molecule=MoleculeSpec(smiles="OCC")),
            )
        )


def test_mixture_component_needs_exactly_one_payload():
    with pytest.raises(Exception):
        MixtureComponent(fraction=1.0)


def test_primary_smiles_of_a_mixture_is_the_largest_component():
    mixture = MixtureSpec(
        components=(
            MixtureComponent(
                role=ComponentRole.SOLVENT, fraction=0.8, molecule=MoleculeSpec(smiles="CCO")
            ),
            MixtureComponent(
                role=ComponentRole.SOLUTE, fraction=0.2, molecule=MoleculeSpec(smiles="c1ccccc1")
            ),
        )
    )
    candidate = Candidate(material_class=MaterialClass.MIXTURE, mixture=mixture)
    assert candidate.primary_smiles == "CCO"
    assert set(candidate.all_smiles()) == {"CCO", "c1ccccc1"}


def test_constraint_violation_renders_readably():
    violation = ConstraintViolation(
        property="normal_boiling_point", requirement="in range", margin=-5.0, hard=True
    )
    assert "hard" in str(violation)
    assert "normal_boiling_point" in str(violation)
