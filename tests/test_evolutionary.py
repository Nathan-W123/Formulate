"""Evolutionary search: operators, selection, diversity and reproducibility."""

from __future__ import annotations

import random

import pytest

from conftest import requires_rdkit

from formulate import chem
from formulate.coordination import DeterministicCoordinator, RunConfig
from formulate.core.candidate import (
    Candidate,
    ComponentRole,
    MaterialClass,
    MixtureComponent,
    MixtureSpec,
    MoleculeSpec,
    MonomerUnit,
    PolymerSpec,
    molecule_candidate,
)
from formulate.exploration.evolutionary import EvolutionaryExplorer, EvolutionConfig
from formulate.exploration.operators import (
    DEFAULT_OPERATORS,
    AtomDeletion,
    AtomSubstitution,
    BondOrderChange,
    FragmentAppend,
    RingFusion,
    brics_crossover,
)
from formulate.targets.spec import TargetSpec

pytestmark = requires_rdkit

_SPEC = TargetSpec.from_dict(
    {
        "name": "evolution test",
        "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
        "requirements": [
            {
                "property": "normal_boiling_point",
                "direction": "in_range",
                "lower": "60 degC",
                "upper": "160 degC",
                "hard": True,
            },
            {"property": "logp", "direction": "target", "target": 2.0, "lower": 0.0, "upper": 4.0},
            {
                "property": "synthetic_accessibility",
                "direction": "minimize",
                "lower": 1,
                "upper": 5,
            },
        ],
    }
)


@pytest.fixture(scope="module")
def scored_pool():
    run = DeterministicCoordinator(config=RunConfig(pool_size=30)).run(_SPEC)
    return [entry.candidate for entry in run.ranking.ranked]


# -- operators -------------------------------------------------------------


@pytest.mark.parametrize("operator", list(DEFAULT_OPERATORS), ids=lambda o: o.id)
def test_every_operator_emits_only_parseable_structures(operator):
    """An operator may propose nonsense, but never something that will not parse."""
    for smiles in ("CC(=O)Oc1ccccc1C(=O)O", "CCCCCC", "c1ccccc1", "CCOC(C)=O", "CC(C)O"):
        for proposal in operator.propose(smiles, random.Random(3), 4):
            assert chem.mol_from_smiles(proposal) is not None
            assert proposal != smiles


@pytest.mark.parametrize("operator", list(DEFAULT_OPERATORS), ids=lambda o: o.id)
def test_operators_are_deterministic_given_a_seed(operator):
    a = operator.propose("CC(=O)Oc1ccccc1C(=O)O", random.Random(11), 4)
    b = operator.propose("CC(=O)Oc1ccccc1C(=O)O", random.Random(11), 4)
    assert a == b


def test_atom_substitution_changes_an_element():
    proposals = AtomSubstitution().propose("CCCO", random.Random(1), 5)
    assert proposals
    assert any(chem.elements(p) != chem.elements("CCCO") for p in proposals)


def test_fragment_append_grows_the_molecule():
    for proposal in FragmentAppend().propose("CCO", random.Random(1), 4):
        assert (
            chem.descriptors(proposal)["heavy_atom_count"]
            > chem.descriptors("CCO")["heavy_atom_count"]
        )


def test_atom_deletion_shrinks_the_molecule():
    for proposal in AtomDeletion().propose("CCCCO", random.Random(1), 4):
        assert (
            chem.descriptors(proposal)["heavy_atom_count"]
            < chem.descriptors("CCCCO")["heavy_atom_count"]
        )


def test_atom_deletion_refuses_to_annihilate_a_tiny_molecule():
    assert AtomDeletion().propose("CO", random.Random(1), 4) == []


def test_ring_closure_makes_rings_of_a_sane_size():
    """Unrestricted closure emitted strained bicyclics that sanitise but cannot be made."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    for proposal in RingFusion().propose("CCCCCCC", random.Random(2), 5):
        mol = chem.mol_from_smiles(proposal)
        assert rdMolDescriptors.CalcNumRings(mol) >= 1
        sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
        assert all(5 <= size <= 7 for size in sizes), f"{proposal} has ring sizes {sizes}"


def test_ring_closure_does_not_bridge_two_ring_atoms():
    for proposal in RingFusion().propose("c1ccccc1CCC", random.Random(4), 6):
        mol = chem.mol_from_smiles(proposal)
        sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
        assert all(size >= 5 for size in sizes), f"{proposal} -> {sizes}"


def test_bond_order_change_alters_unsaturation():
    proposals = BondOrderChange().propose("CCCC", random.Random(1), 3)
    assert any("=" in p for p in proposals)


def test_crossover_recombines_both_parents():
    products = brics_crossover("CC(=O)Oc1ccccc1C(=O)O", "CCOc1ccccc1", random.Random(3), 4)
    assert products
    for product in products:
        assert chem.mol_from_smiles(product) is not None
        assert product not in {"CC(=O)Oc1ccccc1C(=O)O", "CCOc1ccccc1"}


def test_crossover_is_seed_dependent_and_reproducible():
    same = brics_crossover("CC(=O)Oc1ccccc1C(=O)O", "CCOc1ccccc1", random.Random(3), 4)
    again = brics_crossover("CC(=O)Oc1ccccc1C(=O)O", "CCOc1ccccc1", random.Random(3), 4)
    other = brics_crossover("CC(=O)Oc1ccccc1C(=O)O", "CCOc1ccccc1", random.Random(77), 4)
    assert same == again
    assert same != other


def test_crossover_of_an_unparseable_parent_is_empty_not_an_error():
    assert brics_crossover("not-a-smiles", "CCO", random.Random(1), 2) == []


# -- explorer --------------------------------------------------------------


def test_an_unevaluated_pool_yields_no_offspring():
    """Evolution needs parents; inventing an origin would be a fabricated start."""
    assert EvolutionaryExplorer().propose(_SPEC, 5, scored=[], seed=1) == []
    unscored = [molecule_candidate("CCO"), molecule_candidate("CCCO")]
    assert EvolutionaryExplorer().propose(_SPEC, 5, scored=unscored, seed=1) == []


def test_offspring_are_produced_with_lineage(scored_pool):
    children = EvolutionaryExplorer().propose(_SPEC, 10, scored=scored_pool, seed=1)
    assert children
    parent_ids = {c.candidate_id for c in scored_pool}
    for child in children:
        assert child.parent_ids
        assert set(child.parent_ids) <= parent_ids
        assert child.generation_strategy.startswith("evolutionary")
        assert child.provenance is not None
        assert child.results is None


def test_offspring_never_repeat_a_parent(scored_pool):
    children = EvolutionaryExplorer().propose(_SPEC, 15, scored=scored_pool, seed=2)
    assert not ({c.structure_id for c in children} & {c.structure_id for c in scored_pool})


def test_offspring_are_unique_within_a_batch(scored_pool):
    children = EvolutionaryExplorer().propose(_SPEC, 20, scored=scored_pool, seed=4)
    ids = [c.structure_id for c in children]
    assert len(set(ids)) == len(ids)


def test_proposals_are_reproducible_and_seed_sensitive(scored_pool):
    a = EvolutionaryExplorer().propose(_SPEC, 12, scored=scored_pool, seed=5)
    b = EvolutionaryExplorer().propose(_SPEC, 12, scored=scored_pool, seed=5)
    c = EvolutionaryExplorer().propose(_SPEC, 12, scored=scored_pool, seed=6)
    assert [x.structure_id for x in a] == [x.structure_id for x in b]
    assert [x.structure_id for x in a] != [x.structure_id for x in c]


def test_proposals_do_not_depend_on_the_order_of_the_input_pool(scored_pool):
    """Parents are sorted internally, so a caller's ordering cannot change a run."""
    forward = EvolutionaryExplorer().propose(_SPEC, 10, scored=scored_pool, seed=8)
    reversed_pool = list(reversed(scored_pool))
    backward = EvolutionaryExplorer().propose(_SPEC, 10, scored=reversed_pool, seed=8)
    assert [c.structure_id for c in forward] == [c.structure_id for c in backward]


def test_the_scaffold_cap_stops_one_family_monopolising_a_batch(scored_pool):
    explorer = EvolutionaryExplorer(EvolutionConfig(max_per_scaffold=2))
    children = explorer.propose(_SPEC, 20, scored=scored_pool, seed=9)
    counts: dict[str, int] = {}
    for child in children:
        key = explorer._scaffold_key(child)
        counts[key] = counts.get(key, 0) + 1
    assert counts
    assert max(counts.values()) <= 2


def test_growth_is_bounded_relative_to_the_parents(scored_pool):
    explorer = EvolutionaryExplorer(EvolutionConfig(max_growth_per_generation=2))
    children = explorer.propose(_SPEC, 15, scored=scored_pool, seed=10)
    largest_parent = max(
        chem.descriptors(c.primary_smiles).get("heavy_atom_count", 0.0)
        for c in scored_pool
        if c.primary_smiles
    )
    for child in children:
        assert chem.descriptors(child.primary_smiles)["heavy_atom_count"] <= largest_parent + 2


def test_the_spec_heavy_atom_ceiling_is_respected(scored_pool):
    spec = _SPEC.model_copy(
        update={"structural": _SPEC.structural.model_copy(update={"max_heavy_atoms": 8})}
    )
    children = EvolutionaryExplorer().propose(spec, 15, scored=scored_pool, seed=11)
    for child in children:
        assert chem.descriptors(child.primary_smiles)["heavy_atom_count"] <= 8


def test_infeasible_candidates_are_not_bred_from_when_feasible_ones_exist(scored_pool):
    feasible = {c.candidate_id for c in scored_pool if c.results.feasible}
    assert feasible, "fixture should contain feasible candidates"
    children = EvolutionaryExplorer().propose(_SPEC, 12, scored=scored_pool, seed=12)
    for child in children:
        assert set(child.parent_ids) <= feasible


def test_search_improves_the_frontier_over_generations():
    """The point of the whole subsystem: hypervolume must not go backwards."""
    coordinator = DeterministicCoordinator(config=RunConfig(pool_size=40))
    run = coordinator.run(_SPEC)
    pool = [entry.candidate for entry in run.ranking.ranked]
    explorer = EvolutionaryExplorer()

    history = [run.ranking.hypervolume]
    for generation in range(1, 4):
        children = explorer.propose(_SPEC, 20, scored=pool, seed=generation)
        if not children:
            break
        run = coordinator.run(_SPEC, candidates=pool + children)
        pool = [entry.candidate for entry in run.ranking.ranked]
        history.append(run.ranking.hypervolume)

    assert len(history) >= 3
    assert history[-1] >= history[0]
    assert all(b >= a - 1e-9 for a, b in zip(history, history[1:])), history


# -- non-molecular classes -------------------------------------------------


def _polymer_candidate():
    return Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(
            monomers=(
                MonomerUnit(smiles="[*]CC[*]", mole_fraction=0.6),
                MonomerUnit(smiles="[*]CC(C)[*]", mole_fraction=0.4),
            )
        ),
    )


def _mixture_candidate():
    return Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=(
                MixtureComponent(
                    role=ComponentRole.SOLVENT, fraction=0.7, molecule=MoleculeSpec(smiles="CCO")
                ),
                MixtureComponent(
                    role=ComponentRole.SOLUTE,
                    fraction=0.3,
                    molecule=MoleculeSpec(smiles="c1ccccc1"),
                ),
            )
        ),
    )


def test_polymer_mutation_keeps_the_schema_valid():
    explorer = EvolutionaryExplorer()
    children = explorer._mutate(_polymer_candidate(), random.Random(1))
    assert children
    for child in children:
        chain = [m for m in child.polymer.monomers if m.role.value != "end_group"]
        assert abs(sum(m.mole_fraction for m in chain) - 1.0) < 1e-6


def test_mixture_mutation_keeps_fractions_summing_to_one():
    explorer = EvolutionaryExplorer()
    children = explorer._mutate(_mixture_candidate(), random.Random(2))
    assert children
    for child in children:
        assert abs(sum(c.fraction for c in child.mixture.components) - 1.0) < 1e-6


def test_recipe_crossover_produces_a_valid_mixture():
    explorer = EvolutionaryExplorer()
    other = Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=(
                MixtureComponent(
                    role=ComponentRole.SOLVENT, fraction=0.5, molecule=MoleculeSpec(smiles="CCCO")
                ),
                MixtureComponent(
                    role=ComponentRole.ADDITIVE, fraction=0.5, molecule=MoleculeSpec(smiles="CCOC(C)=O")
                ),
            )
        ),
    )
    children = explorer._crossover(_mixture_candidate(), other, random.Random(3))
    for child in children:
        assert abs(sum(c.fraction for c in child.mixture.components) - 1.0) < 1e-6
