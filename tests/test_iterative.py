"""The iterate loop and the Phase 2 search metrics."""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.coordination import RunConfig
from formulate.coordination.iterative import (
    IterationConfig,
    IterativeCoordinator,
    default_iterative_coordinator,
)
from formulate.coordination.metrics import RoundRecord, SearchMetrics, top_k_recall
from formulate.exploration.database import ReferenceDatabaseExplorer
from formulate.targets.spec import TargetSpec

_SPEC = TargetSpec.from_dict(
    {
        "name": "iterative test",
        "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
        "requirements": [
            {
                "property": "normal_boiling_point",
                "direction": "target",
                "target": "150 degC",
                "lower": "110 degC",
                "upper": "190 degC",
                "hard": True,
            },
            {"property": "logp", "direction": "target", "target": 2.5, "lower": 0.5, "upper": 4.5},
            {
                "property": "synthetic_accessibility",
                "direction": "minimize",
                "lower": 1,
                "upper": 5,
            },
        ],
    }
)


def _record(index, hypervolume, evaluated, feasible=0, proposed=10, frontier=3):
    return RoundRecord(
        index=index,
        proposed=proposed,
        rejected_by_filters=0,
        newly_evaluated=proposed,
        total_evaluated=evaluated,
        feasible=feasible,
        hypervolume=hypervolume,
        frontier_size=frontier,
        mean_structural_distance=0.8,
        best_scalar=0.5,
    )


# -- metrics ---------------------------------------------------------------


def test_success_rate_is_over_generated_not_surviving_candidates():
    """Measured over survivors it would approach 1.0 and say nothing useful."""
    metrics = SearchMetrics(rounds=[_record(0, 0.1, 40, feasible=10, proposed=100)])
    assert metrics.success_rate == pytest.approx(0.1)


def test_success_rate_is_undefined_with_nothing_generated():
    assert SearchMetrics().success_rate is None


def test_hypervolume_gain_and_rate():
    metrics = SearchMetrics(rounds=[_record(0, 0.10, 50), _record(1, 0.30, 100)])
    assert metrics.hypervolume_gain == pytest.approx(0.20)
    assert metrics.hypervolume_per_evaluation == pytest.approx(0.002)


def test_sample_efficiency_finds_the_first_round_reaching_a_fraction():
    metrics = SearchMetrics(
        rounds=[_record(0, 0.50, 40), _record(1, 0.90, 80), _record(2, 1.00, 160)]
    )
    assert metrics.evaluations_to_reach(0.9) == 80
    assert metrics.evaluations_to_reach(1.0) == 160
    assert metrics.evaluations_to_reach(0.4) == 40


def test_failure_modes_and_contributions_aggregate_across_rounds():
    first = _record(0, 0.1, 10)
    second = _record(1, 0.2, 20)
    object.__setattr__(first, "contributions", {"database": 5, "evolutionary": 2})
    object.__setattr__(second, "contributions", {"evolutionary": 4})
    object.__setattr__(first, "rejection_reasons", {"charged": 3})
    object.__setattr__(second, "rejection_reasons", {"charged": 1, "too big": 2})
    metrics = SearchMetrics(rounds=[first, second])
    assert metrics.contributions == {"evolutionary": 6, "database": 5}
    assert metrics.failure_modes == {"charged": 4, "too big": 2}


@requires_rdkit
def test_top_k_recall_matches_on_canonical_structure():
    """A benchmark written in a different but equivalent spelling must still match."""
    assert top_k_recall(["CCO", "c1ccccc1"], ["OCC"], 2) == pytest.approx(1.0)
    assert top_k_recall(["CCO", "c1ccccc1"], ["OCC", "CCCCCC"], 2) == pytest.approx(0.5)
    assert top_k_recall(["CCO"], ["CCCCCC"], 1) == pytest.approx(0.0)


def test_top_k_recall_is_undefined_without_a_benchmark():
    assert top_k_recall(["CCO"], [], 5) is None


def test_metrics_describe_an_empty_run_without_crashing():
    assert "No search rounds" in SearchMetrics().describe()


# -- the loop --------------------------------------------------------------


@requires_rdkit
def test_the_loop_improves_the_frontier_and_records_every_round():
    coordinator = default_iterative_coordinator(
        RunConfig(pool_size=30), IterationConfig(max_rounds=3, batch_size=15)
    )
    run = coordinator.run_iterative(_SPEC)

    assert len(run.metrics.rounds) >= 2
    assert len(run.history) == len(run.metrics.rounds)
    history = run.metrics.hypervolume_history
    assert history[-1] >= history[0]
    assert all(b >= a - 1e-9 for a, b in zip(history, history[1:])), history
    assert run.metrics.stop_reason


@requires_rdkit
def test_evolution_actually_contributes_once_scores_are_fed_back():
    """The single-pass coordinator cannot do this: it has no scored parents."""
    coordinator = default_iterative_coordinator(
        RunConfig(pool_size=30), IterationConfig(max_rounds=3, batch_size=15)
    )
    run = coordinator.run_iterative(_SPEC)
    contributions = run.metrics.contributions
    assert any(k.startswith("evolutionary") for k in contributions), contributions


@requires_rdkit
def test_the_round_budget_stops_the_search():
    coordinator = default_iterative_coordinator(
        RunConfig(pool_size=20), IterationConfig(max_rounds=2, batch_size=10)
    )
    run = coordinator.run_iterative(_SPEC)
    assert len(run.metrics.rounds) <= 3
    assert "rounds" in run.metrics.stop_reason or "plateau" in run.metrics.stop_reason


@requires_rdkit
def test_the_evaluation_budget_stops_the_search():
    coordinator = default_iterative_coordinator(
        RunConfig(pool_size=20),
        IterationConfig(max_rounds=10, batch_size=10, max_evaluations=45),
    )
    run = coordinator.run_iterative(_SPEC)
    assert run.metrics.total_evaluated >= 45
    assert "budget" in run.metrics.stop_reason


@requires_rdkit
def test_a_reached_frontier_target_stops_the_search():
    coordinator = default_iterative_coordinator(
        RunConfig(pool_size=30),
        IterationConfig(max_rounds=10, batch_size=15, target_frontier_size=2),
    )
    run = coordinator.run_iterative(_SPEC)
    assert "frontier" in run.metrics.stop_reason


def test_a_plateau_stops_the_search():
    """The predicate is checked directly: a flat frontier must end the run."""
    coordinator = IterativeCoordinator(
        explorers=[ReferenceDatabaseExplorer()],
        iteration=IterationConfig(max_rounds=99, plateau_rounds=2, hypervolume_tolerance=1e-4),
    )
    flat = SearchMetrics(rounds=[_record(i, 0.5, 10 * (i + 1)) for i in range(4)])
    assert "plateau" in coordinator._stop_reason(flat, coordinator.iteration) or "no more than" in (
        coordinator._stop_reason(flat, coordinator.iteration)
    )

    rising = SearchMetrics(rounds=[_record(i, 0.1 * (i + 1), 10 * (i + 1)) for i in range(4)])
    assert coordinator._stop_reason(rising, coordinator.iteration) == ""


@requires_rdkit
def test_an_exhausted_explorer_ends_the_run_rather_than_spinning():
    """Database retrieval runs out; with nothing else there is no new candidate."""
    coordinator = IterativeCoordinator(
        explorers=[ReferenceDatabaseExplorer()],
        config=RunConfig(pool_size=100),
        iteration=IterationConfig(max_rounds=8, batch_size=25),
    )
    run = coordinator.run_iterative(_SPEC)
    assert run.metrics.stop_reason


@requires_rdkit
def test_the_run_is_reproducible():
    def once():
        coordinator = default_iterative_coordinator(
            RunConfig(pool_size=25), IterationConfig(max_rounds=2, batch_size=12)
        )
        run = coordinator.run_iterative(_SPEC)
        return (
            [e.candidate.candidate_id for e in run.final.ranking.ranked],
            run.metrics.hypervolume_history,
        )

    assert once() == once()


@requires_rdkit
def test_the_report_carries_both_the_results_and_the_search():
    coordinator = default_iterative_coordinator(
        RunConfig(pool_size=25), IterationConfig(max_rounds=2, batch_size=12)
    )
    report = coordinator.run_iterative(_SPEC).report(top_k=2)
    assert "FORMULATE DESIGN RUN" in report
    assert "SEARCH" in report
    assert "Stopped because" in report
    assert "WHAT THIS RUN DOES NOT ESTABLISH" in report


@requires_rdkit
def test_seed_candidates_override_exploration():
    from formulate.core.candidate import molecule_candidate

    seeds = [molecule_candidate(s) for s in ("CCO", "c1ccccc1", "CCCCCC")]
    coordinator = default_iterative_coordinator(
        RunConfig(pool_size=30), IterationConfig(max_rounds=1, batch_size=5)
    )
    run = coordinator.run_iterative(_SPEC, seed_candidates=seeds)
    assert run.metrics.rounds[0].proposed == 3
