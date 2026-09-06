"""Selective physics validation: what gets validated, by what, and what changes."""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.coordination import DeterministicCoordinator, RunConfig
from formulate.coordination.validation import (
    NOT_VALIDATABLE_REASONS,
    VALIDATABLE,
    Disagreement,
    ValidationMethod,
    ValidationPolicy,
    merge_prediction,
    physics_prediction,
    select_targets,
    validatable_properties,
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import Prediction
from formulate.core.quantity import Quantity, Uncertainty, UncertaintyKind
from formulate.targets.spec import TargetSpec

_SPEC = TargetSpec.from_dict(
    {
        "name": "validation test",
        "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
        "requirements": [
            {
                "property": "normal_boiling_point",
                "direction": "in_range",
                "lower": "60 degC",
                "upper": "170 degC",
                "hard": True,
            },
            {"property": "homo_lumo_gap", "direction": "maximize", "lower": 4, "upper": 9},
            {"property": "logp", "direction": "target", "target": 2.0, "lower": 0, "upper": 4},
        ],
    }
)


# -- what physics may be asked ---------------------------------------------


def test_the_permitted_mapping_splits_requested_properties():
    permitted, refused = validatable_properties(_SPEC)
    assert permitted["homo_lumo_gap"] is ValidationMethod.QUANTUM
    assert "normal_boiling_point" in refused
    assert "logp" in refused


@pytest.mark.parametrize("prop", ["normal_boiling_point", "logp", "synthetic_accessibility"])
def test_properties_outside_reach_are_refused_with_a_physical_reason(prop):
    """A refusal must say why, not merely that it is unsupported."""
    assert prop not in VALIDATABLE
    reason = NOT_VALIDATABLE_REASONS[prop]
    assert len(reason) > 40


def test_quantum_is_never_offered_a_bulk_property():
    for prop in ("liquid_density", "self_diffusion_coefficient", "shear_viscosity"):
        assert ValidationMethod.QUANTUM not in VALIDATABLE[prop]


def test_dynamics_is_never_offered_an_electronic_property():
    for prop in ("homo_lumo_gap", "dipole_moment", "electronic_energy"):
        assert ValidationMethod.DYNAMICS not in VALIDATABLE[prop]


# -- selection -------------------------------------------------------------


@requires_rdkit
def test_selection_prefers_a_property_no_expert_could_supply():
    run = DeterministicCoordinator(config=RunConfig(pool_size=12)).run(_SPEC)
    ranked = [entry.candidate for entry in run.ranking.ranked]
    targets, _ = select_targets(ranked, _SPEC, ValidationPolicy(max_candidates=3))
    assert targets
    assert all(t.property == "homo_lumo_gap" for t in targets)
    assert all(t.method is ValidationMethod.QUANTUM for t in targets)


@requires_rdkit
def test_selection_respects_its_budget():
    run = DeterministicCoordinator(config=RunConfig(pool_size=20)).run(_SPEC)
    ranked = [entry.candidate for entry in run.ranking.ranked]
    targets, _ = select_targets(ranked, _SPEC, ValidationPolicy(max_candidates=2))
    assert len(targets) <= 2


@requires_rdkit
def test_infeasible_candidates_are_not_validated():
    """Confirming a property of a candidate that breaks a hard constraint changes nothing."""
    run = DeterministicCoordinator(config=RunConfig(pool_size=20)).run(_SPEC)
    ranked = [entry.candidate for entry in run.ranking.ranked]
    targets, _ = select_targets(ranked, _SPEC, ValidationPolicy(max_candidates=10))
    for target in targets:
        assert target.candidate.results.feasible


def test_a_spec_with_nothing_validatable_selects_nothing():
    spec = TargetSpec.from_dict(
        {"requirements": [{"property": "logp", "direction": "minimize", "lower": 0, "upper": 5}]}
    )
    targets, refused = select_targets([], spec, ValidationPolicy())
    assert targets == []
    assert "logp" in refused


# -- merging and disagreement ----------------------------------------------


def _prediction(value, std, expert_id="expert", prop="homo_lumo_gap"):
    return Prediction(
        property=prop,
        quantity=Quantity(value=value, unit="eV"),
        uncertainty=Uncertainty(std=std, kind=UncertaintyKind.EPISTEMIC, basis="test"),
        expert_id=expert_id,
        conditions=Conditions.standard(),
    )


def test_inverse_variance_combination_favours_the_tighter_estimate():
    expert = _prediction(5.0, 1.0, "expert")
    physics = _prediction(7.0, 0.25, "qm")
    merged, disagreement = merge_prediction(expert, physics)
    combined = merged.quantity.to("eV").value
    # The physics estimate is sixteen times more precise, so the combination
    # must sit close to it rather than halfway.
    assert 6.5 < combined < 7.0
    assert merged.uncertainty.std < 0.25
    assert disagreement is not None


def test_physics_does_not_automatically_replace_the_expert():
    """A converged calculation with a known systematic error is not obviously better."""
    expert = _prediction(5.0, 0.1, "expert")
    physics = _prediction(9.0, 3.0, "qm")
    merged, _ = merge_prediction(expert, physics)
    combined = merged.quantity.to("eV").value
    assert combined == pytest.approx(5.0, abs=0.1)


def test_a_missing_uncertainty_forces_replacement_and_says_so():
    expert = _prediction(5.0, None, "expert")
    physics = _prediction(7.0, 0.5, "qm")
    merged, _ = merge_prediction(expert, physics)
    assert merged.quantity.to("eV").value == pytest.approx(7.0)
    assert any("could not be combined" in note for note in merged.notes)


def test_significant_disagreement_is_flagged_in_the_merged_notes():
    expert = _prediction(5.0, 0.1, "expert")
    physics = _prediction(9.0, 0.1, "qm")
    merged, disagreement = merge_prediction(expert, physics)
    assert disagreement.is_significant()
    assert abs(disagreement.z_score) > 20
    assert any("overconfident" in note for note in merged.notes)


def test_a_z_score_is_undefined_without_any_stated_uncertainty():
    disagreement = Disagreement(
        property="homo_lumo_gap",
        expert_id="e",
        expert_value=5.0,
        expert_std=None,
        physics_value=9.0,
        physics_std=None,
        unit="eV",
    )
    assert disagreement.z_score is None
    assert not disagreement.is_significant()
    assert "undefined" in disagreement.describe()


def test_consistent_values_are_not_flagged():
    expert = _prediction(5.0, 1.0)
    physics = _prediction(5.3, 1.0, "qm")
    _, disagreement = merge_prediction(expert, physics)
    assert not disagreement.is_significant()


def test_a_physics_prediction_uses_the_same_record_as_an_expert_one():
    """One record type is what lets the ranker treat both identically."""
    prediction = physics_prediction(
        "homo_lumo_gap",
        Quantity(value=7.0, unit="eV"),
        Uncertainty(std=2.0, kind=UncertaintyKind.EPISTEMIC, basis="method error"),
        backend="qm:pyscf",
        method="b3lyp/6-31g",
        conditions=Conditions.standard(),
    )
    assert isinstance(prediction, Prediction)
    assert prediction.is_usable
    assert prediction.expert_id == "qm:pyscf"


# -- end to end ------------------------------------------------------------


@requires_rdkit
@pytest.mark.slow
def test_validated_values_reach_the_objective_vector_and_the_ranking():
    """Merging a prediction is not enough; the objective vector must be rebuilt."""
    from formulate.coordination import (
        IterationConfig,
        default_validating_coordinator,
    )

    coordinator = default_validating_coordinator(
        RunConfig(pool_size=12),
        IterationConfig(max_rounds=1, batch_size=6),
        ValidationPolicy(max_candidates=2, max_seconds=120),
    )
    run = coordinator.run_validated(_SPEC)

    validated = [
        entry.candidate
        for entry in run.final.ranking.ranked
        if entry.candidate.results.prediction_for("homo_lumo_gap") is not None
    ]
    assert validated, "no candidate received a validated gap"
    for candidate in validated:
        assert "homo_lumo_gap" in candidate.results.objective_vector
        assert candidate.results.objective_vector["homo_lumo_gap"] > 0.0
        assert candidate.results.simulation_ids

    assert "PHYSICS VALIDATION" in run.report(top_k=2)
