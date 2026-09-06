"""Phase 5: the adaptive policy and the benchmark that judges it."""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.coordination import RunConfig, ValidationPolicy
from formulate.coordination.adaptive import (
    Action,
    ActionEstimate,
    AdaptiveConfig,
    default_adaptive_coordinator,
    objective_coverage,
    progress,
)
from formulate.coordination.benchmark import (
    ArmResult,
    BenchmarkResult,
    IncomparableTarget,
    check_comparable,
)
from formulate.targets.spec import TargetSpec

_SPEC = TargetSpec.from_dict(
    {
        "name": "adaptive test",
        "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
        "structural": {"allowed_elements": ["C", "H", "O"], "max_heavy_atoms": 12},
        "requirements": [
            {
                "property": "normal_boiling_point",
                "direction": "in_range",
                "lower": "60 degC",
                "upper": "170 degC",
                "hard": True,
            },
            {"property": "homo_lumo_gap", "direction": "maximize", "lower": 4, "upper": 9},
            {
                "property": "synthetic_accessibility",
                "direction": "minimize",
                "lower": 1,
                "upper": 5,
            },
        ],
    }
)


# -- the value model -------------------------------------------------------


def test_value_density_is_gain_per_second():
    estimate = ActionEstimate(Action.GENERATE, 0.5, 2.0, "")
    assert estimate.value_density == pytest.approx(0.25)


def test_a_free_action_has_no_value_density_rather_than_infinite():
    assert ActionEstimate(Action.STOP, 1.0, 0.0, "").value_density == 0.0


def test_shrinkage_keeps_a_stalled_action_reachable():
    """One zero observation must not disqualify an action for the whole run.

    The unshrunk version spent its entire budget on validations that returned
    nothing, because a single unlucky first round set generation's observed
    yield to zero and locked it out.
    """
    from formulate.coordination.adaptive import _shrunk

    assert _shrunk([], 1e-3, 2.0) == pytest.approx(1e-3)
    assert _shrunk([0.0], 1e-3, 2.0) > 0.0
    # Sustained evidence still dominates the prior.
    assert _shrunk([0.0] * 20, 1e-3, 2.0) < _shrunk([0.0], 1e-3, 2.0)
    assert _shrunk([1.0] * 10, 1e-3, 2.0) > 0.5


# -- the progress signal ---------------------------------------------------


@requires_rdkit
def test_coverage_notices_an_objective_no_expert_supplies():
    """A frontier with one identically zero axis has zero volume however good
    the rest is, so hypervolume alone cannot steer toward acquiring that axis."""
    from formulate.coordination import DeterministicCoordinator

    run = DeterministicCoordinator(config=RunConfig(pool_size=10)).run(_SPEC)
    coverage, uncovered = objective_coverage(run, _SPEC)
    assert "homo_lumo_gap" in uncovered
    assert coverage < 1.0
    # Coverage dominates while anything is unmeasured.
    assert progress(run, _SPEC) == pytest.approx(coverage)


@requires_rdkit
def test_progress_switches_to_hypervolume_once_everything_is_covered():
    from formulate.coordination import DeterministicCoordinator

    spec = TargetSpec.from_dict(
        {
            "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
            "requirements": [
                {"property": "logp", "direction": "maximize", "lower": 0, "upper": 5}
            ],
        }
    )
    run = DeterministicCoordinator(config=RunConfig(pool_size=10)).run(spec)
    coverage, uncovered = objective_coverage(run, spec)
    assert coverage == 1.0 and not uncovered
    assert progress(run, spec) == pytest.approx(1.0 + run.ranking.hypervolume)


# -- the benchmark ---------------------------------------------------------


def test_a_pool_relative_target_is_refused_rather_than_mismeasured():
    """Two arms with different pools would be measured with different rulers."""
    floating = TargetSpec.from_dict(
        {"requirements": [{"property": "logp", "direction": "maximize"}]}
    )
    with pytest.raises(IncomparableTarget) as excinfo:
        check_comparable(floating)
    assert "scale from the candidate pool" in str(excinfo.value)


def test_an_anchored_target_is_accepted():
    check_comparable(_SPEC)


def _arm(name, seed, hv):
    return ArmResult(
        arm=name, seed=seed, seconds=10.0, hypervolume=hv, feasible=5,
        evaluated=20, frontier=3, coverage=1.0,
    )


def test_a_benchmark_can_return_a_negative_result():
    """A comparison that can only confirm is not a comparison."""
    result = BenchmarkResult(target="t")
    for seed in range(4):
        result.pairs.append((_arm("fixed", seed, 0.5), _arm("adaptive", seed, 0.3)))
    assert result.mean_difference < 0
    assert result.losses == 4
    assert "KEEP THE FIXED PIPELINE" in result.verdict


def test_a_marginal_win_does_not_license_adoption():
    """Winning on the mean is not enough when the sign test cannot rule out chance."""
    result = BenchmarkResult(target="t")
    result.pairs.append((_arm("fixed", 0, 0.50), _arm("adaptive", 0, 0.60)))
    result.pairs.append((_arm("fixed", 1, 0.50), _arm("adaptive", 1, 0.45)))
    assert result.mean_difference > 0
    assert "KEEP THE FIXED PIPELINE" in result.verdict


def test_a_consistent_win_licenses_adoption():
    result = BenchmarkResult(target="t")
    for seed in range(6):
        result.pairs.append((_arm("fixed", seed, 0.40), _arm("adaptive", seed, 0.60)))
    assert result.wins == 6
    assert result.sign_test_p() < 0.10
    assert "ADOPT THE ADAPTIVE COORDINATOR" in result.verdict


def test_all_ties_decide_nothing():
    result = BenchmarkResult(target="t")
    for seed in range(3):
        result.pairs.append((_arm("fixed", seed, 0.5), _arm("adaptive", seed, 0.5)))
    assert result.ties == 3
    assert result.sign_test_p() is None
    assert "KEEP THE FIXED PIPELINE" in result.verdict


def test_an_empty_benchmark_decides_nothing():
    assert "nothing is decided" in BenchmarkResult(target="t").verdict


def test_the_sign_test_matches_the_exact_binomial():
    result = BenchmarkResult(target="t")
    for seed in range(5):
        result.pairs.append((_arm("fixed", seed, 0.4), _arm("adaptive", seed, 0.6)))
    # Five wins from five decisive pairs: two-sided exact p = 2 * (1/32).
    assert result.sign_test_p() == pytest.approx(2 / 32)


# -- the loop --------------------------------------------------------------


@requires_rdkit
@pytest.mark.slow
def test_the_policy_acquires_an_uncovered_objective_then_stops_paying_for_it():
    coordinator = default_adaptive_coordinator(
        RunConfig(pool_size=16),
        AdaptiveConfig(
            budget_seconds=60,
            batch_size=10,
            max_actions=10,
            validation=ValidationPolicy(max_candidates=1, max_seconds=30),
        ),
    )
    run = coordinator.run_adaptive(_SPEC)

    assert run.actions
    assert Action.VALIDATE.value in run.action_counts
    # Coverage reached, so the frontier has non-zero volume.
    coverage, uncovered = objective_coverage(run.final, _SPEC)
    assert coverage == 1.0, uncovered
    assert run.final.ranking.hypervolume > 0.0


@requires_rdkit
@pytest.mark.slow
def test_physics_evidence_survives_a_later_generation_round():
    """Re-evaluation rebuilds results from expert predictions and would drop it,
    which showed up as a validated objective losing its coverage."""
    coordinator = default_adaptive_coordinator(
        RunConfig(pool_size=16),
        AdaptiveConfig(
            budget_seconds=60,
            batch_size=10,
            max_actions=10,
            validation=ValidationPolicy(max_candidates=1, max_seconds=30),
        ),
    )
    run = coordinator.run_adaptive(_SPEC)

    validated_actions = [a for a in run.actions if a.chosen is Action.VALIDATE]
    if not validated_actions:
        pytest.skip("the policy never chose to validate in this budget")
    first_validation = validated_actions[0].index
    later = [a for a in run.actions if a.index > first_validation and a.chosen is Action.GENERATE]
    for action in later:
        assert action.actual_gain >= -1e-9, (
            "a generation round after validation lost progress, which means the "
            "physics evidence was discarded by re-evaluation"
        )


@requires_rdkit
@pytest.mark.slow
def test_the_report_shows_every_decision_and_its_outcome():
    coordinator = default_adaptive_coordinator(
        RunConfig(pool_size=14),
        AdaptiveConfig(budget_seconds=30, batch_size=8, max_actions=5),
    )
    report = coordinator.run_adaptive(_SPEC).report(top_k=2)
    assert "ADAPTIVE POLICY" in report
    assert "Decisions:" in report
    assert "WHAT THIS RUN DOES NOT ESTABLISH" in report
