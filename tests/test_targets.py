"""Target specification and desirability functions."""

from __future__ import annotations

import pytest

from formulate.targets.desirability import Desirability, Direction, pool_anchors
from formulate.targets.spec import Requirement, StructuralConstraints, TargetSpec


def test_maximize_and_minimize_are_mirror_images():
    up = Desirability(direction=Direction.MAXIMIZE, lower=0.0, upper=10.0)
    down = Desirability(direction=Direction.MINIMIZE, lower=0.0, upper=10.0)
    assert up(0.0) == pytest.approx(0.0)
    assert up(10.0) == pytest.approx(1.0)
    assert down(0.0) == pytest.approx(1.0)
    assert down(10.0) == pytest.approx(0.0)
    assert up(5.0) == pytest.approx(down(5.0))


def test_values_outside_the_anchors_are_clamped():
    up = Desirability(direction=Direction.MAXIMIZE, lower=0.0, upper=10.0)
    assert up(-5.0) == 0.0
    assert up(15.0) == 1.0


def test_target_desirability_peaks_at_the_target():
    d = Desirability(direction=Direction.TARGET, lower=340.0, target=350.0, upper=360.0)
    assert d(350.0) == pytest.approx(1.0)
    assert d(345.0) == pytest.approx(0.5)
    assert d(355.0) == pytest.approx(0.5)
    assert d(339.0) == 0.0
    assert d(361.0) == 0.0


def test_in_range_is_flat_inside_and_zero_outside():
    d = Desirability(direction=Direction.IN_RANGE, lower=1.0, upper=3.0)
    assert d(2.0) == 1.0
    assert d(0.9) == 0.0
    assert d(3.1) == 0.0


def test_shape_exponent_demands_closeness():
    lenient = Desirability(direction=Direction.MAXIMIZE, lower=0.0, upper=1.0, shape=1.0)
    strict = Desirability(direction=Direction.MAXIMIZE, lower=0.0, upper=1.0, shape=3.0)
    assert strict(0.5) < lenient(0.5)


def test_risk_adjustment_always_moves_away_from_the_ideal():
    up = Desirability(direction=Direction.MAXIMIZE, lower=0.0, upper=10.0)
    nominal, risk = up.evaluate(5.0, 2.0)
    assert risk < nominal

    target = Desirability(direction=Direction.TARGET, lower=340.0, target=350.0, upper=360.0)
    # Above the target the pessimistic shift must go further above, not back toward it.
    _, risk_above = target.evaluate(355.0, 3.0)
    assert risk_above < target(355.0)
    _, risk_below = target.evaluate(345.0, 3.0)
    assert risk_below < target(345.0)


def test_risk_adjustment_is_a_no_op_without_uncertainty():
    up = Desirability(direction=Direction.MAXIMIZE, lower=0.0, upper=10.0)
    assert up.evaluate(5.0, None) == up.evaluate(5.0, 0.0)


def test_desirability_validates_its_own_shape():
    with pytest.raises(ValueError):
        Desirability(direction=Direction.MAXIMIZE, lower=1.0, upper=1.0)
    with pytest.raises(ValueError):
        Desirability(direction=Direction.TARGET, lower=1.0, target=5.0, upper=3.0)
    with pytest.raises(ValueError):
        Desirability(direction=Direction.MAXIMIZE, lower=0.0, upper=1.0, shape=0.0)


def test_pool_anchors_handle_a_degenerate_pool():
    low, high = pool_anchors([5.0, 5.0])
    assert low < high


def test_shorthand_quantities_are_parsed_with_units():
    spec = TargetSpec.from_dict(
        {
            "requirements": [
                {
                    "property": "normal_boiling_point",
                    "direction": "target",
                    "target": "150 degC",
                    "lower": "120 degC",
                    "upper": "180 degC",
                }
            ]
        }
    )
    requirement = spec.requirements[0]
    assert requirement.target_canonical == pytest.approx(423.15)
    assert requirement.lower_canonical == pytest.approx(393.15)


def test_bare_numbers_are_read_in_canonical_units():
    spec = TargetSpec.from_dict(
        {
            "requirements": [
                {"property": "logp", "direction": "in_range", "lower": 0, "upper": 3}
            ]
        }
    )
    assert spec.requirements[0].upper_canonical == pytest.approx(3.0)


def test_target_direction_requires_bounds():
    with pytest.raises(Exception):
        Requirement(property="logp", direction=Direction.TARGET)


def test_hard_bounded_direction_requires_the_bound_it_must_not_cross():
    with pytest.raises(Exception):
        Requirement(property="logp", direction=Direction.MINIMIZE, hard=True)


def test_lower_above_upper_is_rejected():
    with pytest.raises(Exception):
        TargetSpec.from_dict(
            {
                "requirements": [
                    {"property": "logp", "direction": "in_range", "lower": 5, "upper": 1}
                ]
            }
        )


def test_duplicate_requirements_at_the_same_conditions_are_rejected():
    with pytest.raises(Exception):
        TargetSpec.from_dict(
            {
                "requirements": [
                    {"property": "logp", "direction": "minimize", "lower": 0, "upper": 5},
                    {"property": "logp", "direction": "maximize", "lower": 0, "upper": 5},
                ]
            }
        )


def test_unanchored_optimisation_needs_the_pool():
    requirement = Requirement(property="logp", direction=Direction.MAXIMIZE)
    assert requirement.needs_pool_anchors
    with pytest.raises(ValueError):
        requirement.desirability()
    derived = requirement.desirability([0.0, 1.0, 2.0])
    assert derived.pool_relative


def test_weights_are_normalised_over_soft_requirements_only():
    spec = TargetSpec.from_dict(
        {
            "requirements": [
                {
                    "property": "normal_boiling_point",
                    "direction": "in_range",
                    "lower": 300,
                    "upper": 400,
                    "hard": True,
                },
                {"property": "logp", "direction": "minimize", "lower": 0, "upper": 5, "weight": 3.0},
                {
                    "property": "synthetic_accessibility",
                    "direction": "minimize",
                    "lower": 1,
                    "upper": 5,
                    "weight": 1.0,
                },
            ]
        }
    )
    weights = spec.normalized_weights()
    assert "normal_boiling_point" not in weights
    assert weights["logp"] == pytest.approx(0.75)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_structural_constraints_reject_contradictions():
    with pytest.raises(Exception):
        StructuralConstraints(allowed_elements=("C",), forbidden_elements=("C",))
    with pytest.raises(Exception):
        StructuralConstraints(min_heavy_atoms=10, max_heavy_atoms=5)


def test_spec_roundtrips_through_json():
    spec = TargetSpec.from_dict(
        {
            "requirements": [
                {"property": "logp", "direction": "in_range", "lower": 0, "upper": 3}
            ]
        }
    )
    assert TargetSpec.model_validate_json(spec.to_json()).properties == spec.properties


def test_pool_anchors_do_not_depend_on_a_direction():
    """The pool's range is the pool's range; Desirability orients it."""
    import inspect

    assert "direction" not in inspect.signature(pool_anchors).parameters
    assert pool_anchors([1.0, 2.0, 3.0]) == pool_anchors([3.0, 1.0, 2.0])


def test_two_cure_schedules_are_not_one_duplicate_requirement():
    """describe() renders conditions for a person and drops the processing history."""
    spec = TargetSpec.from_dict(
        {
            "name": "cured coating",
            "conditions": {"temperature": "25 degC"},
            "requirements": [
                {
                    "property": "youngs_modulus",
                    "direction": "maximize",
                    "lower": "1 GPa",
                    "upper": "5 GPa",
                    "conditions": {"temperature": "25 degC", "processing": ["cure 120C 30min"]},
                },
                {
                    "property": "youngs_modulus",
                    "direction": "maximize",
                    "lower": "1 GPa",
                    "upper": "5 GPa",
                    "conditions": {"temperature": "25 degC", "processing": ["cure 150C 10min"]},
                },
            ],
        }
    )
    assert len(spec.requirements) == 2


def test_the_same_condition_spelled_two_ways_is_still_a_duplicate():
    with pytest.raises(ValueError, match="Duplicate requirement"):
        TargetSpec.from_dict(
            {
                "name": "two spellings",
                "conditions": {"temperature": "25 degC"},
                "requirements": [
                    {
                        "property": "normal_boiling_point",
                        "direction": "in_range",
                        "lower": "60 degC",
                        "upper": "160 degC",
                        "conditions": {"temperature": "25 degC"},
                    },
                    {
                        "property": "normal_boiling_point",
                        "direction": "in_range",
                        "lower": "60 degC",
                        "upper": "160 degC",
                        "conditions": {"temperature": "298.15 K"},
                    },
                ],
            }
        )
