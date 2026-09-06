"""End-to-end runs, feasibility diagnosis, filters and the evidence store."""

from __future__ import annotations

import json

import pytest

from conftest import requires_rdkit

from formulate.coordination import DeterministicCoordinator, RunConfig, analyze
from formulate.core.candidate import molecule_candidate
from formulate.evaluation.engine import EvaluationConfig
from formulate.exploration.database import ReferenceDatabaseExplorer, load_reference_compounds
from formulate.exploration.filters import CandidateFilter
from formulate.store.evidence import EvidenceStore
from formulate.targets.spec import StructuralConstraints, TargetSpec

_IMPOSSIBLE = TargetSpec.from_dict(
    {
        "name": "impossible",
        "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
        "requirements": [
            {
                "property": "normal_boiling_point",
                "direction": "in_range",
                "lower": "400 degC",
                "upper": "500 degC",
                "hard": True,
            },
            {
                "property": "synthetic_accessibility",
                "direction": "minimize",
                "lower": 1,
                "upper": 1.2,
                "hard": True,
            },
        ],
    }
)


@requires_rdkit
def test_a_full_run_produces_a_ranked_feasible_set(solvent_spec):
    run = DeterministicCoordinator(config=RunConfig(pool_size=30)).run(solvent_spec)
    assert run.evaluated > 0
    assert run.ranking.feasible_count > 0
    assert run.ranking.frontier
    assert run.ranking.axes


@requires_rdkit
def test_the_report_carries_evidence_and_guardrails(solvent_spec):
    report = DeterministicCoordinator(config=RunConfig(pool_size=20)).run(solvent_spec).report(
        top_k=3
    )
    assert "WHAT THIS RUN DOES NOT ESTABLISH" in report
    assert "not experimental proof" in report
    assert "utility" in report
    assert "via joback" in report


@requires_rdkit
def test_assumptions_are_surfaced_in_the_report():
    spec = TargetSpec.from_dict(
        {
            "requirements": [
                {"property": "logp", "direction": "in_range", "lower": 0, "upper": 4}
            ],
            "assumptions": [
                {"statement": "Odour was read as volatility.", "source": "inferred"}
            ],
        }
    )
    report = DeterministicCoordinator(config=RunConfig(pool_size=10)).run(spec).report()
    assert "Odour was read as volatility." in report


@requires_rdkit
def test_an_impossible_request_explains_which_constraint_eliminated_everything():
    """Section 1: identify incompatible constraints and useful relaxations."""
    run = DeterministicCoordinator(config=RunConfig(pool_size=50)).run(_IMPOSSIBLE)
    assert run.ranking.feasible_count == 0
    assert run.feasibility.infeasible

    description = run.feasibility.describe()
    assert "No candidate satisfied every hard constraint" in description
    assert "normal_boiling_point" in description

    diagnosis = next(
        d for d in run.feasibility.diagnoses if d.property == "normal_boiling_point"
    )
    assert diagnosis.eliminated > 0
    assert diagnosis.observed is not None


@requires_rdkit
def test_the_diagnosis_states_that_it_only_covers_evaluated_candidates():
    run = DeterministicCoordinator(config=RunConfig(pool_size=20)).run(_IMPOSSIBLE)
    assert "only the candidates that were evaluated" in run.feasibility.describe()


@requires_rdkit
def test_joint_incompatibility_is_reported():
    run = DeterministicCoordinator(config=RunConfig(pool_size=50)).run(_IMPOSSIBLE)
    assert run.feasibility.jointly_unsatisfied


def test_analysis_of_an_empty_pool_is_safe():
    analysis = analyze([], _IMPOSSIBLE)
    assert not analysis.infeasible
    assert analysis.evaluated == 0


@requires_rdkit
def test_a_run_can_evaluate_a_supplied_pool_instead_of_exploring(solvent_spec, small_pool):
    run = DeterministicCoordinator().run(solvent_spec, candidates=small_pool)
    assert run.evaluated == len(small_pool)


@requires_rdkit
def test_filters_run_before_experts_and_record_reasons(solvent_spec):
    spec = solvent_spec.model_copy(
        update={"structural": StructuralConstraints(allowed_elements=("C", "H"))}
    )
    run = DeterministicCoordinator(config=RunConfig(pool_size=50)).run(spec)
    assert run.filters.rejected
    assert any("outside the allowed set" in r for r in run.filters.rejection_counts)


@requires_rdkit
def test_a_fully_filtered_pool_returns_cleanly(solvent_spec):
    spec = solvent_spec.model_copy(
        update={"structural": StructuralConstraints(allowed_elements=("Xe",))}
    )
    run = DeterministicCoordinator(config=RunConfig(pool_size=20)).run(spec)
    assert run.evaluated == 0
    assert "No candidates were evaluated" in run.report()


@requires_rdkit
def test_filter_rejects_charged_species_by_default():
    result = CandidateFilter(StructuralConstraints()).check(
        molecule_candidate("CC(=O)[O-]")
    )
    assert not result.passed
    assert "charged" in result.reasons[0]


@requires_rdkit
def test_filter_honours_required_substructures():
    constraints = StructuralConstraints(required_smarts=("[OX2H]",))
    assert CandidateFilter(constraints).check(molecule_candidate("CCO")).passed
    assert not CandidateFilter(constraints).check(molecule_candidate("CCCC")).passed


@requires_rdkit
def test_filter_honours_forbidden_substructures():
    constraints = StructuralConstraints(forbidden_smarts=("[CX3]=[OX1]",))
    assert not CandidateFilter(constraints).check(molecule_candidate("CC(C)=O")).passed
    assert CandidateFilter(constraints).check(molecule_candidate("CCO")).passed


@requires_rdkit
def test_runs_are_reproducible(solvent_spec):
    config = RunConfig(pool_size=25, evaluation=EvaluationConfig(max_workers=4))
    first = DeterministicCoordinator(config=config).run(solvent_spec)
    second = DeterministicCoordinator(config=config).run(solvent_spec)
    assert [r.candidate.candidate_id for r in first.ranking.ranked] == [
        r.candidate.candidate_id for r in second.ranking.ranked
    ]
    assert first.ranking.hypervolume == pytest.approx(second.ranking.hypervolume)


def test_the_reference_set_is_valid_and_unique():
    compounds = load_reference_compounds()
    assert len(compounds) >= 40
    names = [c["name"] for c in compounds]
    assert len(set(names)) == len(names)
    for compound in compounds:
        assert "smiles" in compound and "boiling_point_c" in compound


@requires_rdkit
def test_the_explorer_does_not_repropose_known_structures(solvent_spec):
    explorer = ReferenceDatabaseExplorer()
    first = explorer.propose(solvent_spec, 5)
    again = explorer.propose(solvent_spec, 5, scored=first)
    assert not ({c.structure_id for c in first} & {c.structure_id for c in again})


@requires_rdkit
def test_the_evidence_store_round_trips_a_run(tmp_path, solvent_spec):
    run = DeterministicCoordinator(config=RunConfig(pool_size=15)).run(solvent_spec)
    store = EvidenceStore(tmp_path)
    path = store.write(run)

    document = store.read(path)
    assert document["schema"] == "formulate/design-run/1"
    assert document["record_id"].startswith("run-")
    assert document["target"]["name"] == solvent_spec.name
    assert len(document["candidates"]) == len(run.ranking.ranked)
    assert store.list_runs() == [path]
    json.dumps(document)  # must stay JSON-serialisable


@requires_rdkit
def test_the_record_id_is_content_addressed(tmp_path, solvent_spec):
    config = RunConfig(pool_size=15)
    store = EvidenceStore(tmp_path)
    first = store.record(DeterministicCoordinator(config=config).run(solvent_spec))
    second = store.record(DeterministicCoordinator(config=config).run(solvent_spec))
    assert first["record_id"] == second["record_id"]
