"""Selective physics validation: what gets validated, by what, and what changes."""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.coordination import DeterministicCoordinator, RunConfig
from formulate.coordination.validation import (
    DYNAMICS_PROTOCOLS,
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


@pytest.mark.parametrize(
    "prop", ["normal_boiling_point", "logp", "synthetic_accessibility", "electronic_energy"]
)
def test_properties_outside_reach_are_refused_with_a_physical_reason(prop):
    """A refusal must say why, not merely that it is unsupported."""
    assert prop not in VALIDATABLE
    reason = NOT_VALIDATABLE_REASONS[prop]
    assert len(reason) > 40


def test_quantum_is_never_offered_a_bulk_property():
    for prop in ("liquid_density", "self_diffusion_coefficient", "work_of_separation"):
        assert ValidationMethod.QUANTUM not in VALIDATABLE[prop]
    # And a bulk property with no workflow at all is offered to neither method.
    assert "shear_viscosity" not in VALIDATABLE
    assert "shear_viscosity" in NOT_VALIDATABLE_REASONS


def test_dynamics_is_never_offered_an_electronic_property():
    for prop in ("homo_lumo_gap", "dipole_moment"):
        assert ValidationMethod.DYNAMICS not in VALIDATABLE[prop]


def test_absolute_electronic_energy_is_not_a_rankable_objective():
    """It is monotone in electron count, so ranking on it sorts by size.

    HF/STO-3G gives -75 Hartree for water, -152 for ethanol and -228 for
    benzene. A cross-candidate objective over those numbers measures how big
    each molecule is and nothing else.
    """
    assert "electronic_energy" not in VALIDATABLE
    assert "monotone" in NOT_VALIDATABLE_REASONS["electronic_energy"]


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


def test_quantum_results_carry_the_conditions_they_were_computed_at():
    """A gas-phase result at zero Kelvin is not a result at 25 degrees.

    Stamping it with the requested conditions would make the section 11
    condition check pass on exactly the claim it was written to catch.
    """
    from formulate.coordination.validation import VACUUM_ZERO_KELVIN

    assert VACUUM_ZERO_KELVIN.temperature_k == 0.0
    assert VACUUM_ZERO_KELVIN.environment == "vacuum"


def test_a_condition_independent_property_is_not_rejected_on_conditions():
    """A frontier gap has no temperature to disagree about."""
    from formulate.coordination.validation import VACUUM_ZERO_KELVIN
    from formulate.evaluation.engine import EvaluationConfig
    from formulate.evaluation.scoring import OutcomeStatus, score_requirement

    requirement = next(r for r in _SPEC.requirements if r.property == "homo_lumo_gap")
    prediction = Prediction(
        property="homo_lumo_gap",
        quantity=Quantity(value=7.0, unit="eV"),
        uncertainty=Uncertainty(std=2.0, kind=UncertaintyKind.EPISTEMIC, basis="method"),
        expert_id="qm:pyscf",
        conditions=VACUUM_ZERO_KELVIN,
    )
    outcome = score_requirement(
        requirement,
        _SPEC,
        [prediction],
        requirement.desirability(),
        EvaluationConfig(),
    )
    assert outcome.status is not OutcomeStatus.CONDITION_MISMATCH
    assert outcome.effective_utility is not None


def test_a_condition_dependent_property_is_still_rejected_on_conditions():
    """The exemption must not leak to properties that genuinely vary."""
    from formulate.core.conditions import Conditions
    from formulate.evaluation.engine import EvaluationConfig
    from formulate.evaluation.scoring import OutcomeStatus, score_requirement
    from formulate.targets.spec import TargetSpec

    spec = TargetSpec.from_dict(
        {
            "conditions": {"temperature": "200 degC"},
            "requirements": [
                {
                    "property": "surface_tension",
                    "direction": "minimize",
                    "lower": 0.01,
                    "upper": 0.05,
                }
            ],
        }
    )
    prediction = Prediction(
        property="surface_tension",
        quantity=Quantity(value=0.02, unit="N/m"),
        expert_id="test",
        conditions=Conditions.standard(),
    )
    outcome = score_requirement(
        spec.requirements[0],
        spec,
        [prediction],
        spec.requirements[0].desirability(),
        EvaluationConfig(),
    )
    assert outcome.status is OutcomeStatus.CONDITION_MISMATCH


# -- every dynamics property must name its own protocol --------------------


def test_every_dynamics_property_has_a_protocol():
    """The invariant whose absence let a density be answered with an energy.

    Protocol selection used to be an if-else ending in a default, so every
    property but the radius of gyration ran the cohesive-energy workflow and
    came back in joules per mole. The dimensionality check caught it, but as an
    uncaught exception out of the middle of a design run rather than as a
    refusal.
    """
    dynamics = {
        prop for prop, methods in VALIDATABLE.items() if ValidationMethod.DYNAMICS in methods
    }
    assert dynamics
    assert dynamics <= set(DYNAMICS_PROTOCOLS), sorted(dynamics - set(DYNAMICS_PROTOCOLS))
    # And nothing is mapped that is not offered.
    assert set(DYNAMICS_PROTOCOLS) <= dynamics


def test_every_protocol_name_resolves_to_a_real_workflow():
    from formulate.physics.md import MDProtocol

    for prop, name in DYNAMICS_PROTOCOLS.items():
        assert MDProtocol(name), prop


def test_a_property_with_no_workflow_is_refused_with_a_physical_reason():
    spec = TargetSpec.from_dict(
        {
            "name": "not obtainable here",
            "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
            "requirements": [
                {
                    "property": "shear_viscosity",
                    "direction": "minimize",
                    "lower": "0 Pa*s",
                    "upper": "0.01 Pa*s",
                },
                {
                    "property": "cohesive_energy_density",
                    "direction": "maximize",
                    "lower": "0 Pa",
                    "upper": "1e9 Pa",
                },
            ],
        }
    )
    permitted, refused = validatable_properties(spec)
    assert permitted == {}
    assert "Green-Kubo" in refused["shear_viscosity"]
    # Not a vague "unsupported": the cluster workflow produces the wrong
    # dimension, and saying so is what stops someone wiring it up again.
    assert "energy per mole" in refused["cohesive_energy_density"]


@requires_rdkit
@pytest.mark.parametrize(
    "prop", ["liquid_density", "self_diffusion_coefficient", "work_of_separation"]
)
def test_a_bulk_property_declines_with_its_cost_rather_than_raising(prop):
    """The dynamics module was written to refuse these. It must get the chance."""
    from formulate.coordination.validation import (
        PhysicsValidator,
        ValidationTarget,
    )
    from formulate.core.candidate import molecule_candidate
    from formulate.physics.md.base import REQUIREMENTS, MDProtocol

    target = ValidationTarget(
        candidate=molecule_candidate("CCO"),
        property=prop,
        method=ValidationMethod.DYNAMICS,
        value_score=1.0,
        rationale="test",
    )
    prediction, reason = PhysicsValidator(policy=ValidationPolicy())._run_target(target, _SPEC)

    assert prediction is None
    requirement = REQUIREMENTS[MDProtocol(DYNAMICS_PROTOCOLS[prop])]
    assert str(requirement.min_molecules) in reason
    assert "periodic" in reason
