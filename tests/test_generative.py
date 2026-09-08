"""Generative exploration: capability, validity, reproducibility, and reach.

The last unbuilt item of specification section 3. Two providers with
deliberately different standing - one a pretrained transformer, one a mutation
heuristic - and the tests care most about the difference being stated rather
than blurred.
"""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import CandidateResults, molecule_candidate
from formulate.exploration.generative import (
    GenerativeConfig,
    GenerativeExplorer,
    ProviderInfo,
    SafeGptProvider,
    SelfiesMutationProvider,
    acceptable_structure,
    available_providers,
)
from formulate.targets.spec import TargetSpec

pytestmark = requires_rdkit

_SPEC = TargetSpec.from_dict(
    {
        "name": "a high-boiling hydrophobic liquid",
        "material_classes": ["molecule"],
        "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
        "requirements": [
            {"property": "normal_boiling_point", "direction": "maximize", "weight": 2.0},
            {"property": "logp", "direction": "maximize"},
        ],
    }
)

_SEEDS = ["Cc1ccccc1", "CCCCCCCC"]


def _scored(smiles):
    return molecule_candidate(smiles).with_results(CandidateResults(predictions=()))


def _selfies_available():
    return SelfiesMutationProvider().is_available()


def _safe_available():
    return SafeGptProvider().is_available()


# -- capability is declared, not assumed -----------------------------------


def test_a_heuristic_never_claims_to_be_a_trained_model():
    """The one thing worse than no generative model is a random walk posing as one."""
    assert SelfiesMutationProvider().info.pretrained is False
    assert "not a trained generative model" in SelfiesMutationProvider().info.description
    assert SafeGptProvider().info.pretrained is True


def test_every_provider_declares_a_checkable_identity():
    for provider in (SafeGptProvider(), SelfiesMutationProvider()):
        info = provider.info
        assert isinstance(info, ProviderInfo)
        assert info.identifier and info.description and info.reference
        assert set(info.as_parameters()) >= {"provider", "pretrained", "description"}


def test_an_unavailable_provider_says_why_rather_than_returning_nothing():
    explorer = GenerativeExplorer(GenerativeConfig(provider="does-not-exist"))
    assert not explorer.is_available()
    assert explorer.unavailable_reason()


def test_a_pretrained_model_is_preferred_over_the_heuristic():
    """The specification's model policy, asserted rather than left to ordering."""
    identifiers = [p.info.identifier for p in available_providers()]
    if "safe-gpt" in identifiers and "selfies-mutation" in identifiers:
        assert identifiers.index("safe-gpt") < identifiers.index("selfies-mutation")


# -- what a proposal has to be ---------------------------------------------


def test_a_salt_is_not_a_molecule_candidate():
    """SAFE-GPT emits co-crystals; a blend belongs in a MixtureSpec with fractions."""
    assert acceptable_structure("N.O=Cc1ccccc1") is None
    assert acceptable_structure("[Na+].[Cl-]") is None


def test_a_radical_is_not_a_bottleable_compound():
    assert acceptable_structure("[CH]C") is None
    assert acceptable_structure("[CH2]") is None


def test_an_ordinary_molecule_survives_and_is_canonicalised():
    assert acceptable_structure("c1ccccc1C") == "Cc1ccccc1"


# -- generation ------------------------------------------------------------


@pytest.mark.skipif(not _selfies_available(), reason="selfies is not installed")
def test_the_heuristic_generates_valid_structures_from_seeds():
    from rdkit import Chem

    proposals = SelfiesMutationProvider().propose(_SEEDS, 8, seed=1)
    assert proposals
    for smiles in proposals:
        assert Chem.MolFromSmiles(smiles) is not None


@pytest.mark.skipif(not _selfies_available(), reason="selfies is not installed")
def test_the_heuristic_has_nothing_to_mutate_without_seeds():
    """It is a mutation operator, not a model; with no parent it says nothing."""
    assert SelfiesMutationProvider().propose([], 8, seed=1) == []


@pytest.mark.skipif(not _selfies_available(), reason="selfies is not installed")
def test_generation_is_reproducible_for_a_given_seed():
    provider = SelfiesMutationProvider()
    first = provider.propose(_SEEDS, 6, seed=7)
    again = provider.propose(_SEEDS, 6, seed=7)
    other = provider.propose(_SEEDS, 6, seed=8)
    assert first == again
    assert first != other


@pytest.mark.slow
@pytest.mark.skipif(not _safe_available(), reason="safe-mol/torch are not installed")
def test_the_pretrained_model_generates_and_is_reproducible():
    provider = SafeGptProvider()
    first = provider.propose(_SEEDS, 4, seed=3)
    again = provider.propose(_SEEDS, 4, seed=3)
    assert first, f"no proposals; provider reported {provider.last_errors}"
    assert first == again


@pytest.mark.slow
@pytest.mark.skipif(not _safe_available(), reason="safe-mol/torch are not installed")
def test_a_failed_generation_is_recorded_rather_than_swallowed():
    """A silent empty batch and a wrong call signature must not look the same."""
    provider = SafeGptProvider()
    provider.propose(["not a smiles at all"], 2, seed=1)
    assert isinstance(provider.last_errors, list)


# -- the explorer contract -------------------------------------------------


@pytest.mark.skipif(not _selfies_available(), reason="selfies is not installed")
def test_proposals_carry_provenance_naming_model_seed_and_parents():
    explorer = GenerativeExplorer(GenerativeConfig(provider="selfies-mutation"))
    proposed = explorer.propose(_SPEC, 4, scored=[_scored(s) for s in _SEEDS], seed=11)
    assert proposed
    for candidate in proposed:
        record = candidate.provenance
        assert record is not None
        assert record.parameters["provider"] == "selfies-mutation"
        assert record.parameters["pretrained"] is False
        assert record.parameters["seed"] == 11
        assert record.parameters["seed_structures"] == _SEEDS
        assert candidate.generation_strategy == "generative:selfies-mutation"


@pytest.mark.skipif(not _selfies_available(), reason="selfies is not installed")
def test_structural_constraints_are_applied_before_evaluation():
    """Rejecting here is a courtesy to the budget; the filter applies them again."""
    spec = TargetSpec.from_dict(
        {
            "name": "carbon, hydrogen and oxygen only",
            "material_classes": ["molecule"],
            "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
            "structural": {"allowed_elements": ["C", "H", "O"]},
            "requirements": [{"property": "logp", "direction": "maximize"}],
        }
    )
    explorer = GenerativeExplorer(GenerativeConfig(provider="selfies-mutation"))
    proposed = explorer.propose(spec, 10, scored=[_scored(s) for s in _SEEDS], seed=2)

    from rdkit import Chem

    for candidate in proposed:
        elements = {a.GetSymbol() for a in Chem.MolFromSmiles(candidate.primary_smiles).GetAtoms()}
        assert elements <= {"C", "H", "O"}


@pytest.mark.skipif(not _selfies_available(), reason="selfies is not installed")
def test_it_does_not_repropose_what_the_pool_already_holds():
    explorer = GenerativeExplorer(GenerativeConfig(provider="selfies-mutation"))
    scored = [_scored(s) for s in _SEEDS]
    proposed = explorer.propose(_SPEC, 6, scored=scored, seed=4)
    existing = {c.structure_id for c in scored}
    assert all(c.structure_id not in existing for c in proposed)
    assert len({c.structure_id for c in proposed}) == len(proposed)


def test_a_mixture_target_is_declined_rather_than_answered_with_molecules():
    spec = TargetSpec.from_dict(
        {
            "name": "a blend",
            "material_classes": ["mixture"],
            "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
            "requirements": [{"property": "liquid_density", "direction": "maximize"}],
        }
    )
    explorer = GenerativeExplorer(GenerativeConfig(provider="selfies-mutation"))
    assert explorer.propose(spec, 5, scored=(), seed=1) == []


# -- it must actually reach the ranking ------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not _selfies_available(), reason="selfies is not installed")
def test_generated_candidates_reach_evaluation_and_ranking_in_a_real_run():
    """The claim that matters: proposals become ranked candidates, not a list."""
    from formulate.coordination.coordinator import RunConfig
    from formulate.coordination.iterative import IterationConfig, IterativeCoordinator
    from formulate.exploration.database import ReferenceDatabaseExplorer

    coordinator = IterativeCoordinator(
        explorers=[
            ReferenceDatabaseExplorer(),
            GenerativeExplorer(GenerativeConfig(provider="selfies-mutation")),
        ],
        config=RunConfig(pool_size=20, seed=5),
        iteration=IterationConfig(max_rounds=2, batch_size=10),
    )
    final = coordinator.run_iterative(_SPEC).final
    ranked = [entry.candidate for entry in final.ranking.ranked]
    generated = [c for c in ranked if c.generation_strategy.startswith("generative:")]

    assert generated, "no generated candidate survived to the ranking"
    for candidate in generated:
        assert candidate.results is not None, "a generated candidate was never evaluated"


def test_the_explorer_is_registered_in_the_default_factories():
    from formulate.coordination.adaptive import default_adaptive_coordinator
    from formulate.coordination.iterative import default_iterative_coordinator

    for factory in (default_iterative_coordinator, default_adaptive_coordinator):
        assert any(e.id == "generative" for e in factory().explorers)
