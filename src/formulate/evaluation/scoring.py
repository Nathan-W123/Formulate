"""Turning raw predictions into constraints and dimensionless objectives.

Specification section 4: "Convert requested properties into dimensionless
utility/constraint functions.  Preserve raw physical predictions. ...  Hard
constraints eliminate or heavily penalize candidates.  Missing/uncertain
predictions must not silently become neutral scores."

Three rules follow from that last sentence and are enforced here:

* A property with no usable prediction is recorded as missing.  It is never
  scored 0.5.  Under the default policy it scores zero - conservative, and
  visible in the report - and a hard requirement that cannot be checked makes
  the candidate infeasible rather than passing by default.
* An out-of-domain prediction is used but penalised, and the penalty is
  recorded, because discarding it silently would hide a coverage gap.
* A prediction made at conditions incompatible with the requirement does not
  answer that requirement at all (section 11).

Raw predictions are never overwritten; utilities live alongside them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from formulate.core.candidate import Candidate, CandidateResults, ConstraintViolation
from formulate.core.conditions import ConditionMatch, match_conditions
from formulate.core.prediction import Prediction, best_prediction
from formulate.core.properties import get_property
from formulate.targets.desirability import Desirability
from formulate.targets.spec import Requirement, TargetSpec

from .engine import EvaluationConfig


class OutcomeStatus(str, Enum):
    """How a single requirement fared for one candidate."""

    OK = "ok"
    OUT_OF_DOMAIN = "out_of_domain"
    #: No expert produced a usable value.
    MISSING = "missing"
    #: A value exists but not at conditions that answer the requirement.
    CONDITION_MISMATCH = "condition_mismatch"
    #: The requirement needs pool anchors and the pool supplied none.
    UNSCORABLE = "unscorable"


@dataclass(frozen=True, slots=True)
class RequirementOutcome:
    """The full audit trail for one requirement against one candidate."""

    requirement: Requirement
    status: OutcomeStatus
    prediction: Prediction | None = None
    #: Predicted value in the property's canonical unit.
    value: float | None = None
    #: Desirability at the predicted value.
    utility: float | None = None
    #: Desirability at a pessimistic bound derived from the uncertainty.
    risk_adjusted_utility: float | None = None
    #: The utility actually used for ranking, after any domain penalty.
    effective_utility: float | None = None
    condition_match: ConditionMatch | None = None
    violation: ConstraintViolation | None = None
    #: True when the bound is met by the predicted value but not by the
    #: pessimistic bound. The candidate passes, but only within its own error
    #: bar, which section 12 treats as a calibration question rather than a
    #: clean pass.
    constraint_at_risk: bool = False
    reason: str = ""

    @property
    def contributes(self) -> bool:
        return self.effective_utility is not None

    def describe(self) -> str:
        prop = self.requirement.property
        if self.status in (OutcomeStatus.MISSING, OutcomeStatus.UNSCORABLE):
            return f"{prop}: {self.status.value} - {self.reason}"
        if self.status is OutcomeStatus.CONDITION_MISMATCH:
            return f"{prop}: condition mismatch - {self.reason}"
        unit = get_property(prop).canonical_unit
        flag = " [out of domain]" if self.status is OutcomeStatus.OUT_OF_DOMAIN else ""
        risk = " [meets the bound only within its uncertainty]" if self.constraint_at_risk else ""
        return (
            f"{prop}: {self.value:.6g} {unit} -> utility {self.effective_utility:.3f}"
            f"{flag}{risk}"
        )


def build_desirabilities(
    spec: TargetSpec, predictions_by_candidate: Sequence[Sequence[Prediction]]
) -> dict[str, Desirability | None]:
    """Construct one desirability per requirement.

    Requirements that state no anchors take them from the observed pool; the
    resulting functions are marked pool-relative so a report can say that the
    score is comparable within this run only.
    """
    out: dict[str, Desirability | None] = {}
    for req in spec.requirements:
        if not req.needs_pool_anchors:
            out[req.property] = req.desirability()
            continue
        values = _pool_values(req, predictions_by_candidate)
        if not values:
            out[req.property] = None
            continue
        out[req.property] = req.desirability(values)
    return out


def _pool_values(
    req: Requirement, predictions_by_candidate: Sequence[Sequence[Prediction]]
) -> list[float]:
    unit = get_property(req.property).canonical_unit
    values: list[float] = []
    for predictions in predictions_by_candidate:
        best = _best_prediction(predictions, req.property)
        if best is not None and best.quantity is not None:
            values.append(best.quantity.to(unit).value)
    return values


def _best_prediction(predictions: Sequence[Prediction], prop: str) -> Prediction | None:
    # Deliberately the shared rule and not a local sort. Sorting on
    # applicability alone leaves a measured value tied with a group-contribution
    # estimate, and the pool statistics that anchor pool-relative targets were
    # being built from whichever of the two happened to be dispatched first.
    return best_prediction(predictions, prop)


def score_requirement(
    req: Requirement,
    spec: TargetSpec,
    predictions: Sequence[Prediction],
    desirability: Desirability | None,
    config: EvaluationConfig,
) -> RequirementOutcome:
    """Evaluate one requirement against one candidate's predictions."""
    prop = req.property
    unit = get_property(prop).canonical_unit

    if desirability is None:
        return RequirementOutcome(
            requirement=req,
            status=OutcomeStatus.UNSCORABLE,
            reason=(
                f"{prop} states no absolute anchors and no candidate in the pool produced "
                "a value to derive them from"
            ),
            violation=_missing_violation(req, "no value anywhere in the pool"),
        )

    prediction = _best_prediction(predictions, prop)
    if prediction is None or prediction.quantity is None:
        reason = _explain_absence(predictions, prop)
        return RequirementOutcome(
            requirement=req,
            status=OutcomeStatus.MISSING,
            reason=reason,
            violation=_missing_violation(req, reason),
        )

    required_conditions = spec.conditions_for(req)
    match = match_conditions(
        required_conditions,
        prediction.conditions,
        temperature_tolerance_k=config.temperature_tolerance_k,
        pressure_rtol=config.pressure_rtol,
    )
    # A property the registry marks condition-independent has no temperature or
    # pressure to disagree about. A frontier orbital gap computed in vacuum at
    # zero Kelvin answers a request stated at 25 degrees, because the quantity
    # itself does not vary with either. Enforcing a match for such a property
    # would reject every quantum result on a technicality; skipping it for a
    # condition-dependent one would be the far worse error, which is why the
    # distinction is read from the registry rather than from the caller.
    condition_independent = not get_property(prop).condition_dependent
    if not match.compatible and not condition_independent:
        reason = "; ".join(match.issues)
        return RequirementOutcome(
            requirement=req,
            status=OutcomeStatus.CONDITION_MISMATCH,
            prediction=prediction,
            condition_match=match,
            reason=reason,
            violation=_missing_violation(req, f"only available at other conditions: {reason}"),
        )

    value = prediction.quantity.to(unit).value
    std = prediction.uncertainty.converted(prediction.quantity.unit, unit).std
    nominal, risk = desirability.evaluate(value, std, config.risk_k)
    base = risk if config.use_risk_adjusted else nominal

    out_of_domain = not prediction.applicability.in_domain
    effective = base * config.out_of_domain_penalty if out_of_domain else base

    violation = _bound_violation(req, value, unit)
    at_risk = violation is None and std is not None and std > 0.0 and (
        _bound_violation(
            req, desirability.pessimistic_value(value, std, config.risk_k), unit
        )
        is not None
    )
    return RequirementOutcome(
        requirement=req,
        status=OutcomeStatus.OUT_OF_DOMAIN if out_of_domain else OutcomeStatus.OK,
        prediction=prediction,
        value=value,
        utility=nominal,
        risk_adjusted_utility=risk,
        effective_utility=effective,
        condition_match=match,
        violation=violation,
        constraint_at_risk=at_risk,
        reason="; ".join(prediction.applicability.warnings) if out_of_domain else "",
    )


def _explain_absence(predictions: Sequence[Prediction], prop: str) -> str:
    """Say *why* a property is missing, using the failed attempts as evidence."""
    attempts = [p for p in predictions if p.property == prop]
    if not attempts:
        return "no registered expert covers this property for this material class"
    reasons = []
    for attempt in attempts:
        note = "; ".join(attempt.notes) or attempt.status.value
        reasons.append(f"{attempt.expert_id}: {note}")
    return " | ".join(reasons)


def _missing_violation(req: Requirement, reason: str) -> ConstraintViolation | None:
    """A hard requirement that cannot be checked is a violation, not a pass."""
    if not req.hard:
        return None
    return ConstraintViolation(
        property=req.property,
        requirement=req.describe_constraint(),
        hard=True,
        reason=f"hard requirement could not be verified ({reason})",
    )


def _bound_violation(req: Requirement, value: float, unit: str) -> ConstraintViolation | None:
    lower, upper = req.constraint_bound()
    margin: float | None = None
    if lower is not None and value < lower:
        margin = value - lower
    elif upper is not None and value > upper:
        margin = value - upper
    if margin is None:
        return None
    side = (
        f"below the lower bound of {lower:.6g}"
        if margin < 0
        else f"above the upper bound of {upper:.6g}"
    )
    return ConstraintViolation(
        property=req.property,
        requirement=req.describe_constraint(),
        margin=margin,
        hard=req.hard,
        reason=f"predicted {value:.6g} {unit} is {side} {unit}",
    )


def score_candidate(
    candidate: Candidate,
    predictions: Sequence[Prediction],
    spec: TargetSpec,
    desirabilities: dict[str, Desirability | None],
    config: EvaluationConfig,
) -> tuple[CandidateResults, list[RequirementOutcome]]:
    """Produce the results block for one candidate."""
    outcomes = [
        score_requirement(req, spec, predictions, desirabilities.get(req.property), config)
        for req in spec.requirements
    ]

    objective_vector: dict[str, float] = {}
    missing: list[str] = []
    violations: list[ConstraintViolation] = []

    for outcome in outcomes:
        prop = outcome.requirement.property
        if outcome.violation is not None:
            violations.append(outcome.violation)

        if outcome.effective_utility is not None:
            objective_vector[prop] = outcome.effective_utility
            continue

        missing.append(prop)
        # Absent a value, credit is not given. Recorded in missing_properties
        # so the report can explain the zero rather than leaving it unexplained.
        if config.missing_objective_policy == "penalize":
            objective_vector[prop] = 0.0

    feasible = not any(v.hard for v in violations)
    scalar = _scalar_baseline(spec, objective_vector)

    results = CandidateResults(
        predictions=tuple(predictions),
        objective_vector=objective_vector,
        missing_properties=tuple(missing),
        constraint_violations=tuple(violations),
        feasible=feasible,
        scalar_score=scalar,
    )
    return results, outcomes


def _scalar_baseline(spec: TargetSpec, objective_vector: dict[str, float]) -> float | None:
    """Weighted sum of soft objectives.

    Section 4 permits combining utilities "with explicit weights only for a
    scalar baseline"; the Pareto frontier is the real ranking and does not
    consult this.  When a spec states only hard requirements there are no soft
    weights, so the available objectives are weighted equally.
    """
    weights = spec.normalized_weights()
    if not weights:
        available = [v for k, v in objective_vector.items()]
        return sum(available) / len(available) if available else None
    total_weight = sum(w for prop, w in weights.items() if prop in objective_vector)
    if total_weight <= 0:
        return None
    return sum(
        objective_vector[prop] * weight
        for prop, weight in weights.items()
        if prop in objective_vector
    ) / total_weight


def score_pool(
    candidates: Sequence[Candidate],
    predictions_by_candidate: Sequence[Sequence[Prediction]],
    spec: TargetSpec,
    config: EvaluationConfig,
) -> tuple[list[Candidate], list[list[RequirementOutcome]], dict[str, Desirability | None]]:
    """Score an entire pool, deriving any pool-relative anchors first."""
    desirabilities = build_desirabilities(spec, predictions_by_candidate)
    scored: list[Candidate] = []
    all_outcomes: list[list[RequirementOutcome]] = []
    for candidate, predictions in zip(candidates, predictions_by_candidate):
        results, outcomes = score_candidate(
            candidate, predictions, spec, desirabilities, config
        )
        scored.append(candidate.with_results(results))
        all_outcomes.append(outcomes)
    return scored, all_outcomes, desirabilities
