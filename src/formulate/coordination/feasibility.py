"""Diagnosing infeasible requests.

Specification section 1: "Failure behavior: identify incompatible constraints
and estimate which constraint relaxations create feasible regions."

Returning "no candidates found" is not an answer.  When every candidate fails,
the useful output is which constraint did the eliminating, whether any pair of
constraints is jointly unsatisfiable, and how far a bound would have to move
to admit something.  This module computes that from the evaluated pool.

The estimates are only as good as the pool: a relaxation that would admit a
candidate the search never proposed cannot be seen here, and the report says
so rather than implying the analysis is exhaustive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from formulate.core.candidate import Candidate
from formulate.core.properties import get_property
from formulate.targets.spec import Requirement, TargetSpec


@dataclass(frozen=True, slots=True)
class ConstraintDiagnosis:
    """How one hard constraint behaved across the pool."""

    property: str
    requirement: str
    eliminated: int
    #: Candidates that violate this constraint and no other hard one, so
    #: relaxing this alone would admit them.
    would_admit: int
    #: The bound this constraint would need for the nearest miss to pass.
    suggested_bound: str | None = None
    #: Range of predicted values seen across the pool.
    observed: tuple[float, float] | None = None
    unverifiable: int = 0

    def describe(self) -> str:
        unit = get_property(self.property).canonical_unit
        lines = [
            f"{self.property}: eliminated {self.eliminated} candidate(s)"
        ]
        if self.unverifiable:
            lines.append(f", {self.unverifiable} of them because the value could not be predicted")
        if self.would_admit:
            lines.append(f"; relaxing it alone would admit {self.would_admit}")
        if self.suggested_bound:
            lines.append(f"; nearest miss passes at {self.suggested_bound}")
        if self.observed is not None:
            lines.append(
                f"; pool spans {self.observed[0]:.6g} to {self.observed[1]:.6g} {unit}"
            )
        return "".join(lines)


@dataclass
class FeasibilityAnalysis:
    """Why a request produced no feasible candidates, and what would help."""

    feasible_count: int = 0
    evaluated: int = 0
    diagnoses: list[ConstraintDiagnosis] = field(default_factory=list)
    #: Pairs of hard constraints no candidate satisfied simultaneously.
    jointly_unsatisfied: list[tuple[str, str]] = field(default_factory=list)

    @property
    def infeasible(self) -> bool:
        return self.feasible_count == 0 and self.evaluated > 0

    def describe(self) -> str:
        if not self.infeasible:
            return (
                f"{self.feasible_count} of {self.evaluated} candidates satisfy every "
                "hard constraint."
            )
        lines = [
            f"No candidate satisfied every hard constraint ({self.evaluated} evaluated).",
            "",
            "Constraint analysis:",
        ]
        lines.extend(f"  - {d.describe()}" for d in self.diagnoses)
        if self.jointly_unsatisfied:
            lines.append("")
            lines.append("Constraints never satisfied together by any candidate in this pool:")
            lines.extend(f"  - {a} and {b}" for a, b in self.jointly_unsatisfied)
        lines.append("")
        lines.append(
            "This analysis covers only the candidates that were evaluated. A relaxation "
            "that would admit a structure the search never proposed cannot be detected here."
        )
        return "\n".join(lines)


def analyze(candidates: Sequence[Candidate], spec: TargetSpec) -> FeasibilityAnalysis:
    """Diagnose which hard constraints are doing the eliminating."""
    scored = [c for c in candidates if c.results is not None]
    hard = spec.hard_requirements
    analysis = FeasibilityAnalysis(
        feasible_count=sum(1 for c in scored if c.results.feasible),  # type: ignore[union-attr]
        evaluated=len(scored),
    )
    if not scored or not hard:
        return analysis

    violated_by: dict[str, set[int]] = {req.property: set() for req in hard}
    for index, candidate in enumerate(scored):
        for violation in candidate.results.hard_violations:  # type: ignore[union-attr]
            violated_by.setdefault(violation.property, set()).add(index)

    for req in hard:
        offenders = violated_by.get(req.property, set())
        only_this = {
            i
            for i in offenders
            if not any(other != req.property and i in violated_by.get(other, set())
                       for other in violated_by)
        }
        analysis.diagnoses.append(
            _diagnose(req, scored, offenders, only_this)
        )

    analysis.diagnoses.sort(key=lambda d: (-d.would_admit, -d.eliminated))

    for i, first in enumerate(hard):
        for second in hard[i + 1:]:
            a, b = violated_by.get(first.property, set()), violated_by.get(second.property, set())
            if a and b and len(a | b) == len(scored):
                analysis.jointly_unsatisfied.append(
                    (first.describe(), second.describe())
                )
    return analysis


def _diagnose(
    req: Requirement,
    scored: Sequence[Candidate],
    offenders: set[int],
    only_this: set[int],
) -> ConstraintDiagnosis:
    unit = get_property(req.property).canonical_unit

    values: list[float] = []
    unverifiable = 0
    nearest: float | None = None
    for index in offenders:
        prediction = scored[index].results.prediction_for(req.property)  # type: ignore[union-attr]
        if prediction is None or prediction.quantity is None:
            unverifiable += 1
            continue
        values.append(prediction.quantity.to(unit).value)

    for candidate in scored:
        prediction = candidate.results.prediction_for(req.property)  # type: ignore[union-attr]
        if prediction is not None and prediction.quantity is not None:
            value = prediction.quantity.to(unit).value
            if nearest is None or _distance_to_bound(req, value) < _distance_to_bound(req, nearest):
                nearest = value

    observed: tuple[float, float] | None = None
    all_values = [
        c.results.prediction_for(req.property).quantity.to(unit).value  # type: ignore[union-attr]
        for c in scored
        if c.results.prediction_for(req.property) is not None  # type: ignore[union-attr]
        and c.results.prediction_for(req.property).quantity is not None  # type: ignore[union-attr]
    ]
    if all_values:
        observed = (min(all_values), max(all_values))

    suggestion: str | None = None
    if nearest is not None and _distance_to_bound(req, nearest) > 0:
        lower, upper = req.constraint_bound()
        if upper is not None and nearest > upper:
            suggestion = f"upper bound {nearest:.6g} {unit} (currently {upper:.6g})"
        elif lower is not None and nearest < lower:
            suggestion = f"lower bound {nearest:.6g} {unit} (currently {lower:.6g})"

    return ConstraintDiagnosis(
        property=req.property,
        requirement=req.describe(),
        eliminated=len(offenders),
        would_admit=len(only_this),
        suggested_bound=suggestion,
        observed=observed,
        unverifiable=unverifiable,
    )


def _distance_to_bound(req: Requirement, value: float) -> float:
    """How far a value sits outside the requirement's bounds; 0 when inside."""
    lower, upper = req.constraint_bound()
    if lower is not None and value < lower:
        return lower - value
    if upper is not None and value > upper:
        return value - upper
    return 0.0
