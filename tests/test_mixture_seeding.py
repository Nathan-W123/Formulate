"""Seeding the first formulations.

Three explorers could each work on a mixture once one existed and none would
make the first one. The evolutionary explorer said seeding belongs to the
database or optimisation explorers, the Bayesian explorer said proposing a
formulation from nothing belongs to the evolutionary and retrieval explorers,
and the database explorer returns molecules only. So a mixture target produced
an empty pool and nothing said why.
"""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import (
    CandidateResults,
    MaterialClass,
    molecule_candidate,
)
from formulate.exploration.bayesopt import BayesOptExplorer
from formulate.exploration.database import ReferenceDatabaseExplorer
from formulate.exploration.evolutionary import EvolutionaryExplorer
from formulate.exploration.mixtures import MixtureSeedConfig, MixtureSeedExplorer
from formulate.targets.spec import TargetSpec

_MIXTURE_SPEC = TargetSpec.from_dict(
    {
        "name": "a coating solvent blend",
        "material_classes": ["mixture"],
        "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
        "requirements": [
            {"property": "liquid_density", "direction": "in_range",
             "lower": "0.80 g/cm^3", "upper": "0.95 g/cm^3"},
            {"property": "mixing_stability", "direction": "maximize"},
        ],
    }
)

_MOLECULE_SPEC = TargetSpec.from_dict(
    {
        "name": "a molecule only",
        "material_classes": ["molecule"],
        "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
        "requirements": [{"property": "normal_boiling_point", "direction": "maximize"}],
    }
)


def _components(candidate):
    return [c.molecule.smiles for c in candidate.mixture.components]


# -- the gap this closes ---------------------------------------------------


@requires_rdkit
def test_no_other_explorer_proposes_a_formulation_from_a_cold_start():
    """The circular deferral, asserted so it cannot come back quietly."""
    for explorer in (
        ReferenceDatabaseExplorer(),
        EvolutionaryExplorer(),
        BayesOptExplorer(),
    ):
        assert explorer.propose(_MIXTURE_SPEC, 10, scored=(), seed=1) == []


@requires_rdkit
def test_the_seeder_proposes_formulations_from_a_cold_start():
    proposals = MixtureSeedExplorer().propose(_MIXTURE_SPEC, 10, scored=(), seed=1)
    assert len(proposals) == 10
    for candidate in proposals:
        assert candidate.material_class is MaterialClass.MIXTURE
        assert candidate.mixture is not None
        assert len(candidate.mixture.components) == 2
        assert candidate.generation_strategy == "mixture:seed"


@requires_rdkit
def test_the_default_explorer_set_reaches_mixtures():
    """The wiring, not just the class: a factory that forgets it is the bug."""
    from formulate.coordination.iterative import default_iterative_coordinator

    explorers = default_iterative_coordinator().explorers
    proposals = [
        candidate
        for explorer in explorers
        for candidate in explorer.propose(_MIXTURE_SPEC, 4, scored=(), seed=1)
    ]
    assert proposals
    assert all(c.material_class is MaterialClass.MIXTURE for c in proposals)


# -- what it proposes, and in what order -----------------------------------


@requires_rdkit
def test_proposals_are_dealt_in_rounds_so_early_ones_span_the_pool():
    """Ten binary blends should touch twenty compounds, not five.

    Enumerating pairs in index order spends the budget on one compound.
    Ordering by the largest index is barely better: it exhausts every pair
    inside a growing prefix, which put ten proposals inside five compounds.
    """
    proposals = MixtureSeedExplorer().propose(_MIXTURE_SPEC, 10, scored=(), seed=1)
    used = {smiles for c in proposals for smiles in _components(c)}
    assert len(used) == 20


@requires_rdkit
def test_every_component_set_is_offered_exactly_once():
    """A round defers what it cannot place; it must not drop or repeat it."""
    pool = 6
    explorer = MixtureSeedExplorer(MixtureSeedConfig(max_components=pool))
    proposals = explorer.propose(_MIXTURE_SPEC, 500, scored=(), seed=3)

    sets = [frozenset(_components(c)) for c in proposals]
    assert len(sets) == pool * (pool - 1) // 2
    assert len(set(sets)) == len(sets)


@requires_rdkit
def test_the_same_seed_gives_the_same_proposals():
    first = MixtureSeedExplorer().propose(_MIXTURE_SPEC, 8, scored=(), seed=7)
    again = MixtureSeedExplorer().propose(_MIXTURE_SPEC, 8, scored=(), seed=7)
    other = MixtureSeedExplorer().propose(_MIXTURE_SPEC, 8, scored=(), seed=8)
    assert [c.structure_id for c in first] == [c.structure_id for c in again]
    assert [c.structure_id for c in first] != [c.structure_id for c in other]


@requires_rdkit
def test_fractions_sum_to_one_and_components_are_distinct():
    """MixtureSpec validates both, so a violation raises rather than ranks."""
    for candidate in MixtureSeedExplorer().propose(_MIXTURE_SPEC, 20, scored=(), seed=2):
        components = candidate.mixture.components
        assert sum(c.fraction for c in components) == pytest.approx(1.0)
        assert len({c.canonical_key() for c in components}) == len(components)


# -- what it refuses to decide ---------------------------------------------


@requires_rdkit
def test_it_does_not_pre_judge_whether_the_components_will_mix():
    """Section 3 keeps exploration and evaluation apart.

    Dropping an immiscible pair here would move a decision out of the expert
    that can justify it - UNIFAC scores hexane and water at about -15 against
    +1.6 for ethanol and water - and into one that would be guessing. Over a
    large budget the seeder must therefore offer pairs the panel will reject.
    """
    from formulate.experts.activity import UNIFACActivityExpert

    if not UNIFACActivityExpert().is_available():
        pytest.skip("thermo's UNIFAC tables are not installed")

    proposals = MixtureSeedExplorer().propose(_MIXTURE_SPEC, 400, scored=(), seed=1)
    pairs = [frozenset(_components(c)) for c in proposals]
    assert frozenset({"CCCCCC", "O"}) in pairs, "an immiscible pair was filtered out upstream"


@requires_rdkit
def test_a_molecule_only_target_gets_nothing_and_costs_nothing():
    assert MixtureSeedExplorer().propose(_MOLECULE_SPEC, 10, scored=(), seed=1) == []


# -- where the components come from ----------------------------------------


@requires_rdkit
def test_components_come_from_this_run_once_enough_have_been_scored():
    """Blending compounds the panel has already scored beats a fixed list."""
    scored = []
    for smiles in ("CCO", "CC(C)=O", "CCCCCC", "Cc1ccccc1"):
        candidate = molecule_candidate(smiles)
        scored.append(candidate.with_results(CandidateResults(predictions=())))

    proposals = MixtureSeedExplorer().propose(_MIXTURE_SPEC, 6, scored=scored, seed=1)
    assert proposals
    known = {"CCO", "CC(C)=O", "CCCCCC", "Cc1ccccc1"}
    for candidate in proposals:
        assert set(_components(candidate)) <= known


@requires_rdkit
def test_an_unscored_molecule_does_not_count_towards_the_pool():
    """A candidate with no results has not been evaluated, only proposed."""
    unscored = [molecule_candidate(s) for s in ("CCO", "CC(C)=O", "CCCCCC")]
    proposals = MixtureSeedExplorer().propose(_MIXTURE_SPEC, 6, scored=unscored, seed=1)

    used = {smiles for c in proposals for smiles in _components(c)}
    assert not used <= {"CCO", "CC(C)=O", "CCCCCC"}


@requires_rdkit
def test_forbidden_elements_are_honoured():
    spec = TargetSpec.from_dict(
        {
            "name": "no halogens",
            "material_classes": ["mixture"],
            "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
            "structural": {"forbidden_elements": ["Cl", "N", "S"]},
            "requirements": [{"property": "liquid_density", "direction": "maximize"}],
        }
    )
    proposals = MixtureSeedExplorer().propose(spec, 60, scored=(), seed=1)
    assert proposals

    from rdkit import Chem

    for candidate in proposals:
        for smiles in _components(candidate):
            elements = {a.GetSymbol() for a in Chem.MolFromSmiles(smiles).GetAtoms()}
            assert not elements & {"Cl", "N", "S"}


# -- configuration ---------------------------------------------------------


def test_a_composition_that_does_not_describe_the_blend_is_refused():
    with pytest.raises(ValueError, match="cannot describe a blend"):
        MixtureSeedExplorer(
            MixtureSeedConfig(components_per_blend=3, compositions=((0.5, 0.5),))
        )


def test_a_composition_that_does_not_sum_to_one_is_refused():
    with pytest.raises(ValueError, match="does not sum to one"):
        MixtureSeedExplorer(MixtureSeedConfig(compositions=((0.5, 0.4),)))


@requires_rdkit
def test_ternary_blends_when_asked_for():
    explorer = MixtureSeedExplorer(
        MixtureSeedConfig(
            components_per_blend=3,
            compositions=((1 / 3, 1 / 3, 1 / 3),),
            max_components=9,
        )
    )
    proposals = explorer.propose(_MIXTURE_SPEC, 5, scored=(), seed=1)
    assert len(proposals) == 5
    for candidate in proposals:
        assert len(candidate.mixture.components) == 3
        assert sum(c.fraction for c in candidate.mixture.components) == pytest.approx(1.0)
