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


# -- one rule for choosing between two predictions -------------------------


def _density(value, unit, std, expert_id, in_domain=True, score=1.0):
    from formulate.core.quantity import ApplicabilityDomain

    return Prediction(
        property="amorphous_density",
        quantity=Quantity(value=value, unit=unit),
        uncertainty=Uncertainty(std=std, kind=UncertaintyKind.EPISTEMIC),
        applicability=ApplicabilityDomain(in_domain=in_domain, score=score),
        status=PredictionStatus.OK if in_domain else PredictionStatus.OUT_OF_DOMAIN,
        expert_id=expert_id,
        conditions=Conditions.standard(),
    )


def test_the_tighter_spread_is_decided_in_one_unit():
    """An uncertainty is in the expert's unit, so raw floats are not comparable."""
    from formulate.core.prediction import prefer

    # 40 kg/m^3 is tighter than 0.05 g/cm^3, which is 50 kg/m^3. Comparing the
    # numbers as written would say 0.05 < 40 and pick the looser one.
    tight = _density(1074.0, "kg/m^3", 40.0, "si")
    loose = _density(1.074, "g/cm^3", 0.05, "cgs")
    assert prefer(tight, loose)
    assert not prefer(loose, tight)


def test_a_stated_spread_beats_an_unstated_one_in_both_directions():
    from formulate.core.prediction import prefer

    stated = _density(1074.0, "kg/m^3", 40.0, "stated")
    silent = _density(1074.0, "kg/m^3", None, "silent")
    assert prefer(stated, silent)
    assert not prefer(silent, stated)


def test_dispatch_scoring_and_the_results_record_pick_the_same_prediction():
    """Three places used to restate this rule and two of them got it wrong.

    Both predictions here are in domain with equal applicability, which is the
    exact tie a sort on applicability alone cannot break - and the two that
    sorted took whichever happened to come first.
    """
    from formulate.core.candidate import CandidateResults
    from formulate.core.prediction import best_prediction, prefer
    from formulate.evaluation.scoring import _best_prediction

    loose = _density(1200.0, "kg/m^3", 90.0, "estimate")
    tight = _density(1195.0, "kg/m^3", 12.0, "measured")

    for order in ([loose, tight], [tight, loose]):
        assert best_prediction(order, "amorphous_density").expert_id == "measured"
        assert _best_prediction(order, "amorphous_density").expert_id == "measured"
        results = CandidateResults(predictions=tuple(order))
        assert results.prediction_for("amorphous_density").expert_id == "measured"

        folded = None
        for prediction in order:
            if folded is None or prefer(prediction, folded):
                folded = prediction
        assert folded.expert_id == "measured"


def test_an_out_of_domain_prediction_never_wins_on_a_tighter_spread():
    from formulate.core.prediction import best_prediction

    confident_but_outside = _density(1200.0, "kg/m^3", 1.0, "outside", in_domain=False, score=0.2)
    honest_and_inside = _density(1100.0, "kg/m^3", 80.0, "inside")
    for order in ([confident_but_outside, honest_and_inside], [honest_and_inside, confident_but_outside]):
        assert best_prediction(order, "amorphous_density").expert_id == "inside"


def test_a_cache_entry_is_not_reused_for_a_different_question():
    """The ask is part of the key, because the answer depends on it.

    An expert returns the intersection of its coverage with the request, and
    sees only the upstream predictions that request produced. A cache persisted
    from one target specification was serving a later one the earlier ask's
    filtered result.
    """
    from formulate.store.cache import PredictionCache

    conditions = Conditions.standard()
    one = PredictionCache.key("cand", "joback", "1", conditions, ["normal_boiling_point"])
    two = PredictionCache.key(
        "cand", "joback", "1", conditions, ["normal_boiling_point", "melting_point"]
    )
    assert one != two

    # The set is what matters, not the order it arrived in.
    reordered = PredictionCache.key(
        "cand", "joback", "1", conditions, ["melting_point", "normal_boiling_point"]
    )
    assert reordered == two


@requires_rdkit
def test_two_specifications_share_an_entry_for_an_expert_neither_ask_reaches():
    """Keying on the whole request would miss on a property the expert ignores."""
    from formulate.experts import default_registry
    from formulate.store.cache import PredictionCache

    cache = PredictionCache()
    engine = EvaluationEngine(default_registry(), EvaluationConfig(), cache)
    candidate = molecule_candidate("CCO")

    def spec_with(extra):
        return TargetSpec.from_dict(
            {
                "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
                "requirements": [
                    {
                        "property": "normal_boiling_point",
                        "direction": "in_range",
                        "lower": "60 degC",
                        "upper": "160 degC",
                    }
                ]
                + extra,
            }
        )

    engine.predict([candidate], spec_with([]))
    hits_before = cache.hits
    # logp is answered by crippen, which the thermal experts never see, so
    # their entries must still hit.
    engine.predict(
        [candidate],
        spec_with([{"property": "logp", "direction": "in_range", "lower": -1.0, "upper": 5.0}]),
    )
    assert cache.hits > hits_before
