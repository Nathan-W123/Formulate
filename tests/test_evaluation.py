"""Dispatch, caching, and the rules that stop absence becoming a neutral score."""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import molecule_candidate
from formulate.core.conditions import Conditions
from formulate.core.prediction import Prediction, PredictionStatus
from formulate.core.quantity import Quantity, Uncertainty, UncertaintyKind
from formulate.evaluation.engine import EvaluationConfig, EvaluationEngine
from formulate.evaluation.scoring import (
    OutcomeStatus,
    build_desirabilities,
    score_candidate,
    score_pool,
    score_requirement,
)
from formulate.store.cache import PredictionCache
from formulate.targets.spec import TargetSpec

_BP_SPEC = TargetSpec.from_dict(
    {
        "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
        "requirements": [
            {
                "property": "normal_boiling_point",
                "direction": "in_range",
                "lower": "60 degC",
                "upper": "160 degC",
                "hard": True,
            }
        ],
    }
)


def _prediction(value_k, std=5.0, conditions=None, in_domain=True):
    from formulate.core.quantity import ApplicabilityDomain

    return Prediction(
        property="normal_boiling_point",
        quantity=Quantity(value=value_k, unit="K"),
        uncertainty=Uncertainty(std=std, kind=UncertaintyKind.EPISTEMIC),
        applicability=ApplicabilityDomain(in_domain=in_domain, score=1.0 if in_domain else 0.2),
        status=PredictionStatus.OK if in_domain else PredictionStatus.OUT_OF_DOMAIN,
        expert_id="test",
        conditions=conditions or Conditions.standard(),
    )


def test_missing_prediction_is_never_scored_neutral():
    """The core rule of section 4: absence is not a mid-range result."""
    config = EvaluationConfig()
    outcome = score_requirement(
        _BP_SPEC.requirements[0], _BP_SPEC, [], _BP_SPEC.requirements[0].desirability(), config
    )
    assert outcome.status is OutcomeStatus.MISSING
    assert outcome.effective_utility is None
    assert outcome.utility != 0.5


def test_an_unverifiable_hard_requirement_makes_a_candidate_infeasible():
    """A hard constraint that could not be checked must not pass by default."""
    results, outcomes = score_candidate(
        molecule_candidate("CCO"), [], _BP_SPEC, build_desirabilities(_BP_SPEC, [[]]),
        EvaluationConfig(),
    )
    assert not results.feasible
    assert results.hard_violations
    assert "could not be verified" in results.hard_violations[0].reason


def test_missing_property_is_recorded_and_explained():
    results, _ = score_candidate(
        molecule_candidate("CCO"), [], _BP_SPEC, build_desirabilities(_BP_SPEC, [[]]),
        EvaluationConfig(),
    )
    assert results.missing_properties == ("normal_boiling_point",)


def test_a_failed_attempt_is_used_to_explain_the_absence():
    failure = Prediction.failed("normal_boiling_point", "joback", "group assignment failed")
    outcome = score_requirement(
        _BP_SPEC.requirements[0], _BP_SPEC, [failure],
        _BP_SPEC.requirements[0].desirability(), EvaluationConfig(),
    )
    assert "joback" in outcome.reason
    assert "group assignment failed" in outcome.reason


def test_a_prediction_at_the_wrong_conditions_does_not_answer_the_requirement():
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
        conditions=Conditions.standard(),
        expert_id="test",
    )
    outcome = score_requirement(
        spec.requirements[0], spec, [prediction],
        spec.requirements[0].desirability(), EvaluationConfig(),
    )
    assert outcome.status is OutcomeStatus.CONDITION_MISMATCH
    assert outcome.effective_utility is None


def test_out_of_domain_predictions_are_penalised_not_discarded():
    config = EvaluationConfig(out_of_domain_penalty=0.5, use_risk_adjusted=False)
    requirement = _BP_SPEC.requirements[0]
    good = score_requirement(
        requirement, _BP_SPEC, [_prediction(370.0)], requirement.desirability(), config
    )
    flagged = score_requirement(
        requirement, _BP_SPEC, [_prediction(370.0, in_domain=False)],
        requirement.desirability(), config,
    )
    assert flagged.status is OutcomeStatus.OUT_OF_DOMAIN
    assert flagged.effective_utility == pytest.approx(good.effective_utility * 0.5)


def test_a_bound_met_only_within_uncertainty_is_flagged():
    requirement = _BP_SPEC.requirements[0]  # 333.15 K to 433.15 K
    outcome = score_requirement(
        requirement, _BP_SPEC, [_prediction(335.0, std=10.0)],
        requirement.desirability(), EvaluationConfig(),
    )
    assert outcome.violation is None
    assert outcome.constraint_at_risk


def test_a_comfortable_pass_is_not_flagged_at_risk():
    requirement = _BP_SPEC.requirements[0]
    outcome = score_requirement(
        requirement, _BP_SPEC, [_prediction(383.0, std=5.0)],
        requirement.desirability(), EvaluationConfig(),
    )
    assert not outcome.constraint_at_risk


def test_a_breached_bound_becomes_a_violation():
    requirement = _BP_SPEC.requirements[0]
    outcome = score_requirement(
        requirement, _BP_SPEC, [_prediction(300.0)], requirement.desirability(),
        EvaluationConfig(),
    )
    assert outcome.violation is not None
    assert outcome.violation.hard
    assert outcome.violation.margin < 0


def test_risk_adjusted_ranking_can_be_switched_off():
    requirement = _BP_SPEC.requirements[0]
    prediction = _prediction(340.0, std=20.0)
    risky = score_requirement(
        requirement, _BP_SPEC, [prediction], requirement.desirability(),
        EvaluationConfig(use_risk_adjusted=True),
    )
    nominal = score_requirement(
        requirement, _BP_SPEC, [prediction], requirement.desirability(),
        EvaluationConfig(use_risk_adjusted=False),
    )
    assert risky.effective_utility <= nominal.effective_utility


def test_pool_relative_anchors_are_marked():
    spec = TargetSpec.from_dict(
        {"requirements": [{"property": "logp", "direction": "maximize"}]}
    )
    predictions = [
        [Prediction(property="logp", quantity=Quantity(value=v, unit=""), expert_id="t")]
        for v in (1.0, 2.0, 3.0)
    ]
    desirabilities = build_desirabilities(spec, predictions)
    assert desirabilities["logp"].pool_relative


def test_an_unscorable_requirement_is_reported_not_guessed():
    spec = TargetSpec.from_dict(
        {"requirements": [{"property": "logp", "direction": "maximize", "hard": False}]}
    )
    desirabilities = build_desirabilities(spec, [[]])
    outcome = score_requirement(
        spec.requirements[0], spec, [], desirabilities["logp"], EvaluationConfig()
    )
    assert outcome.status is OutcomeStatus.UNSCORABLE


@requires_rdkit
def test_engine_pulls_in_dependencies_of_requested_properties(registry, solvent_spec):
    engine = EvaluationEngine(registry, EvaluationConfig(max_workers=1))
    needed = engine._properties_needed(solvent_spec)
    assert "critical_temperature" in needed  # required by surface tension
    assert "normal_boiling_point" in needed


@requires_rdkit
def test_engine_reports_uncovered_properties(registry):
    spec = TargetSpec.from_dict(
        {
            "material_classes": ["polymer"],
            "requirements": [
                {"property": "logp", "direction": "minimize", "lower": 0, "upper": 5}
            ],
        }
    )
    from formulate.core.candidate import (
        Candidate,
        MaterialClass,
        MonomerUnit,
        PolymerSpec,
    )

    polymer = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(monomers=(MonomerUnit(smiles="[*]CC[*]"),)),
    )
    _, report = EvaluationEngine(registry, EvaluationConfig(max_workers=1)).predict(
        [polymer], spec
    )
    assert "logp" in report.uncovered_properties


@requires_rdkit
def test_cache_prevents_repeated_calculation(registry, solvent_spec, small_pool):
    cache = PredictionCache()
    engine = EvaluationEngine(registry, EvaluationConfig(max_workers=1), cache)
    engine.predict(small_pool, solvent_spec)
    first_misses = cache.misses
    engine.predict(small_pool, solvent_spec)
    assert cache.hits >= first_misses
    assert cache.hit_rate > 0.0


@requires_rdkit
def test_cache_persists_across_instances(tmp_path, registry, solvent_spec, small_pool):
    path = tmp_path / "cache.jsonl"
    engine = EvaluationEngine(registry, EvaluationConfig(max_workers=1), PredictionCache(path))
    engine.predict(small_pool, solvent_spec)

    reloaded = PredictionCache(path)
    assert len(reloaded) > 0


def test_a_corrupt_cache_line_loses_one_entry_not_the_file(tmp_path):
    path = tmp_path / "cache.jsonl"
    path.write_text('{"broken": true}\nnot json at all\n', encoding="utf-8")
    assert len(PredictionCache(path)) == 0


@requires_rdkit
def test_concurrent_and_serial_dispatch_agree(registry, solvent_spec, small_pool):
    serial, _ = EvaluationEngine(registry, EvaluationConfig(max_workers=1)).predict(
        small_pool, solvent_spec
    )
    parallel, _ = EvaluationEngine(registry, EvaluationConfig(max_workers=4)).predict(
        small_pool, solvent_spec
    )
    assert [len(p) for p in serial] == [len(p) for p in parallel]


@requires_rdkit
def test_scoring_preserves_raw_predictions(registry, solvent_spec, small_pool):
    """Section 4: preserve raw physical predictions alongside the utilities."""
    engine = EvaluationEngine(registry, EvaluationConfig(max_workers=1))
    predictions, _ = engine.predict(small_pool, solvent_spec)
    scored, _, _ = score_pool(small_pool, predictions, solvent_spec, engine.config)
    for candidate, raw in zip(scored, predictions):
        assert len(candidate.results.predictions) == len(raw)
        bp = candidate.results.prediction_for("normal_boiling_point")
        if bp is not None:
            assert bp.quantity.unit == "kelvin"
