"""Rendering a design run for a human reader.

Specification section 9, stage 10: "Report top recipes, conditions, evidence,
uncertainties, trade-offs, and infeasible constraints."

The report is not decoration.  A ranked list without its uncertainties, its
out-of-domain flags and its guardrails is exactly the artefact section 13
warns against - one that invites a reader to treat a model score as
experimental proof.  Everything that qualifies a number is printed next to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from formulate.core.properties import get_property
from formulate.evaluation.scoring import OutcomeStatus

if TYPE_CHECKING:  # pragma: no cover
    from .coordinator import DesignRun

_RULE = "=" * 72
_THIN = "-" * 72

#: Standing scientific guardrails, from specification section 13. They are
#: printed on every report because they qualify every number in it.
_GUARDRAILS = (
    "Requested properties do not uniquely determine a structure. Inverse design "
    "is many-to-many, and a request may have no feasible answer at all.",
    "A high model score is not experimental proof and not evidence of "
    "synthesisability. Nothing here has been made or measured.",
    "These predictors are correlations and heuristics over single molecules. "
    "Bulk, formulation and processing behaviour is not established by them.",
    "No quantum or molecular-dynamics validation has run in this phase, so no "
    "result here carries physics evidence.",
)


def render_report(run: "DesignRun", *, top_k: int = 10) -> str:
    """Render a complete run report."""
    parts = [
        _header(run),
        _pipeline(run),
        _feasibility(run),
        _ranking(run, top_k),
        _caveats(run),
    ]
    return "\n".join(p for p in parts if p)


def _header(run: "DesignRun") -> str:
    spec = run.spec
    lines = [
        _RULE,
        f"FORMULATE DESIGN RUN - {spec.name}",
        _RULE,
        "",
        f"Conditions      {spec.conditions.describe()}",
        f"Material class  {', '.join(c.value for c in spec.material_classes)}",
        "",
        "Requirements",
    ]
    lines.extend(f"  - {req.describe()}" for req in spec.requirements)
    if spec.assumptions:
        lines += ["", "Assumptions made on your behalf"]
        lines.extend(f"  - {a}" for a in spec.assumptions)
    return "\n".join(lines)


def _pipeline(run: "DesignRun") -> str:
    lines = ["", _THIN, "PIPELINE", _THIN, ""]
    lines.append(f"Candidates proposed   {run.proposed}")
    lines.append(run.filters.describe())
    lines.append("")
    lines.append(run.dispatch.describe())

    pool_relative = run.pool_relative_properties
    if pool_relative:
        lines += [
            "",
            "Scored on a pool-relative scale (the request stated no absolute anchors, so "
            "0 and 1 were taken from the range observed in this pool): "
            + ", ".join(pool_relative),
            "Those utilities are comparable within this run only.",
        ]
    return "\n".join(lines)


def _feasibility(run: "DesignRun") -> str:
    return "\n".join(["", _THIN, "FEASIBILITY", _THIN, "", run.feasibility.describe()])


def _ranking(run: "DesignRun", top_k: int) -> str:
    ranking = run.ranking
    if not ranking.ranked:
        return "\n".join(["", _THIN, "RESULTS", _THIN, "", "No candidates were evaluated."])

    lines = ["", _THIN, "RESULTS", _THIN, "", ranking.describe(limit=0).rstrip(), ""]

    entries = ranking.top(top_k)
    for entry in entries:
        lines.extend(_candidate_block(run, entry))
    return "\n".join(lines)


def _candidate_block(run: "DesignRun", entry) -> list[str]:
    candidate = entry.candidate
    results = candidate.results
    name = candidate.label or candidate.primary_smiles or candidate.candidate_id
    smiles = candidate.primary_smiles

    header = f"#{entry.rank}  {name}"
    if smiles and smiles != name:
        header += f"   {smiles}"
    lines = ["", header]
    lines.append(
        f"    front {entry.front}   baseline score "
        + ("n/a" if entry.scalar is None else f"{entry.scalar:.3f}")
        + ("" if entry.feasible else "   INFEASIBLE")
    )
    lines.append(f"    id {candidate.candidate_id}   via {candidate.generation_strategy}")

    for outcome in run.outcomes_for(candidate):
        lines.extend(_outcome_lines(outcome))

    if results is not None and results.constraint_violations:
        lines.append("    Constraint violations")
        for violation in results.constraint_violations:
            lines.append(f"      ! {violation}")
    return lines


def _outcome_lines(outcome) -> list[str]:
    prop = outcome.requirement.property
    if outcome.status in (OutcomeStatus.MISSING, OutcomeStatus.UNSCORABLE):
        return [f"    {prop:<32} not available - {outcome.reason}"]
    if outcome.status is OutcomeStatus.CONDITION_MISMATCH:
        return [f"    {prop:<32} wrong conditions - {outcome.reason}"]

    unit = get_property(prop).canonical_unit
    unit_text = "" if unit == "dimensionless" else f" {unit}"
    prediction = outcome.prediction
    spread = ""
    if prediction is not None:
        uncertainty = prediction.canonical_uncertainty
        if uncertainty.std is not None and uncertainty.std > 0:
            spread = f" +/- {uncertainty.std:.4g}"

    line = (
        f"    {prop:<32} {outcome.value:>12.6g}{unit_text}{spread}"
        f"   utility {outcome.effective_utility:.3f}"
    )
    lines = [line]

    detail: list[str] = []
    if prediction is not None:
        detail.append(f"via {prediction.expert_id}")
    if outcome.status is OutcomeStatus.OUT_OF_DOMAIN:
        detail.append("OUT OF DOMAIN")
    if outcome.constraint_at_risk:
        detail.append("meets its bound only within uncertainty")
    if (
        outcome.utility is not None
        and outcome.risk_adjusted_utility is not None
        and abs(outcome.utility - outcome.risk_adjusted_utility) > 1e-9
    ):
        detail.append(
            f"utility {outcome.utility:.3f} at the predicted value, "
            f"{outcome.risk_adjusted_utility:.3f} at the pessimistic bound"
        )
    if detail:
        lines.append(f"        {'; '.join(detail)}")

    if prediction is not None:
        for warning in prediction.applicability.warnings:
            lines.append(f"        domain: {warning}")
    return lines


def _caveats(run: "DesignRun") -> str:
    lines = ["", _THIN, "WHAT THIS RUN DOES NOT ESTABLISH", _THIN, ""]
    lines.extend(f"  - {g}" for g in _GUARDRAILS)

    uncovered = run.dispatch.uncovered_properties
    if uncovered:
        lines += [
            "",
            "  - No registered expert covers these requested properties, so they were "
            "not evaluated at all: " + ", ".join(uncovered),
        ]
    return "\n".join(lines)
