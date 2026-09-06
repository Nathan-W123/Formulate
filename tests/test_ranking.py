"""Pareto machinery and the precedence rules of the ranker."""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import CandidateResults, ConstraintViolation, molecule_candidate
from formulate.ranking.diversity import max_min_selection, mean_pairwise_distance
from formulate.ranking.pareto import (
    crowding_distance,
    dominates,
    hypervolume,
    non_dominated_sort,
)
from formulate.ranking.ranker import Ranker, RankingConfig
from formulate.targets.spec import TargetSpec

_SPEC = TargetSpec.from_dict(
    {
        "requirements": [
            {"property": "logp", "direction": "maximize", "lower": 0, "upper": 1},
            {
                "property": "synthetic_accessibility",
                "direction": "maximize",
                "lower": 1,
                "upper": 10,
            },
        ]
    }
)


def _candidate(smiles, label, objectives, feasible=True):
    violations = (
        ()
        if feasible
        else (ConstraintViolation(property="logp", requirement="x", hard=True),)
    )
    return molecule_candidate(smiles, label=label).with_results(
        CandidateResults(
            objective_vector=objectives,
            feasible=feasible,
            constraint_violations=violations,
            scalar_score=sum(objectives.values()) / len(objectives),
        )
    )


def test_dominance_requires_better_somewhere_and_worse_nowhere():
    assert dominates([1.0, 1.0], [0.5, 0.5])
    assert not dominates([1.0, 0.0], [0.0, 1.0])
    assert not dominates([0.5, 0.5], [0.5, 0.5])


def test_dominance_ignores_floating_point_noise():
    assert not dominates([0.5 + 1e-15, 0.5], [0.5, 0.5])


def test_dominance_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        dominates([1.0], [1.0, 2.0])


def test_non_dominated_sort_layers_the_pool():
    assert non_dominated_sort([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.4, 0.4]]) == [0, 0, 0, 1]


def test_non_dominated_sort_handles_an_empty_pool():
    assert non_dominated_sort([]) == []


def test_crowding_distance_puts_the_extremes_first():
    distances = crowding_distance([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    assert distances[0] == float("inf")
    assert distances[1] == float("inf")
    assert distances[2] < float("inf")


@pytest.mark.parametrize(
    "points,expected",
    [
        ([[1.0, 1.0]], 1.0),
        ([[1.0, 0.0], [0.0, 1.0]], 0.0),
        ([[1.0, 0.5], [0.5, 1.0]], 0.75),
        ([[0.5, 0.5]], 0.25),
        ([[1.0, 1.0, 1.0]], 1.0),
        ([[1.0, 1.0, 0.5], [0.5, 1.0, 1.0]], 0.75),
    ],
)
def test_hypervolume_matches_analytic_values(points, expected):
    assert hypervolume(points) == pytest.approx(expected)


def test_hypervolume_ignores_dominated_points():
    assert hypervolume([[1.0, 1.0], [0.5, 0.5]]) == pytest.approx(1.0)


def test_hypervolume_of_an_empty_front_is_zero():
    assert hypervolume([]) == 0.0


def test_hypervolume_respects_a_reference_point():
    assert hypervolume([[1.0, 1.0]], [0.5, 0.5]) == pytest.approx(0.25)


@requires_rdkit
def test_feasible_candidates_always_outrank_infeasible_ones():
    """Even when the infeasible one scores better on every objective."""
    pool = [
        _candidate("CCO", "great but infeasible", {"logp": 1.0, "synthetic_accessibility": 1.0},
                   feasible=False),
        _candidate("c1ccccc1", "modest but feasible",
                   {"logp": 0.1, "synthetic_accessibility": 0.1}),
    ]
    result = Ranker().rank(pool, _SPEC)
    assert result.ranked[0].candidate.label == "modest but feasible"
    assert result.feasible_count == 1


@requires_rdkit
def test_eliminate_policy_drops_infeasible_candidates():
    pool = [
        _candidate("CCO", "bad", {"logp": 1.0, "synthetic_accessibility": 1.0}, feasible=False),
        _candidate("c1ccccc1", "good", {"logp": 0.5, "synthetic_accessibility": 0.5}),
    ]
    result = Ranker(RankingConfig(infeasible_policy="eliminate")).rank(pool, _SPEC)
    assert len(result.ranked) == 1


@requires_rdkit
def test_an_objective_missing_for_some_candidates_is_excluded_from_dominance():
    """A candidate must not dominate by virtue of not having been measured."""
    pool = [
        _candidate("CCO", "complete", {"logp": 0.5, "synthetic_accessibility": 0.5}),
        _candidate("c1ccccc1", "partial", {"logp": 0.9}),
    ]
    result = Ranker().rank(pool, _SPEC)
    assert result.axes == ("logp",)
    assert result.dropped_axes == ("synthetic_accessibility",)


@requires_rdkit
def test_strict_mode_raises_on_partial_objectives():
    pool = [
        _candidate("CCO", "complete", {"logp": 0.5, "synthetic_accessibility": 0.5}),
        _candidate("c1ccccc1", "partial", {"logp": 0.9}),
    ]
    with pytest.raises(ValueError):
        Ranker(RankingConfig(allow_partial_objectives=False)).rank(pool, _SPEC)


@requires_rdkit
def test_ranking_writes_front_and_rank_back_onto_results():
    pool = [
        _candidate("CCO", "a", {"logp": 0.9, "synthetic_accessibility": 0.9}),
        _candidate("c1ccccc1", "b", {"logp": 0.1, "synthetic_accessibility": 0.1}),
    ]
    result = Ranker().rank(pool, _SPEC)
    top = result.ranked[0]
    assert top.candidate.results.aggregate_rank == 1
    assert top.candidate.results.pareto_front == 0


def test_ranking_an_empty_pool_is_safe():
    assert Ranker().rank([], _SPEC).ranked == []


@requires_rdkit
def test_diverse_selection_keeps_the_best_candidate_first():
    pool = [
        _candidate("c1ccccc1", "benzene", {"logp": 0.9, "synthetic_accessibility": 0.9}),
        _candidate("Cc1ccccc1", "toluene", {"logp": 0.8, "synthetic_accessibility": 0.8}),
        _candidate("CCCCCCO", "hexanol", {"logp": 0.7, "synthetic_accessibility": 0.7}),
    ]
    result = Ranker().rank(pool, _SPEC)
    picked = result.diverse_top(2)
    assert picked[0].candidate.label == "benzene"
    # Hexanol is structurally further from benzene than toluene is.
    assert picked[1].candidate.label == "hexanol"


@requires_rdkit
def test_max_min_selection_returns_everything_when_asked_for_too_much():
    candidates = [molecule_candidate(s) for s in ("CCO", "c1ccccc1")]
    assert sorted(max_min_selection(candidates, 5)) == [0, 1]


@requires_rdkit
def test_mean_pairwise_distance_is_bounded():
    candidates = [molecule_candidate(s) for s in ("CCO", "c1ccccc1", "CCCCCC")]
    distance = mean_pairwise_distance(candidates)
    assert 0.0 <= distance <= 1.0


def test_mean_pairwise_distance_of_a_single_candidate_is_undefined():
    assert mean_pairwise_distance([molecule_candidate("CCO")]) is None
