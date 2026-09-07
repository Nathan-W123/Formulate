"""Expert dispatch.

Specification section 5, the deterministic baseline coordinator: "dispatch
candidates to applicable experts, validate units/conditions, normalize
objectives, enforce constraints, deduplicate results".

This module owns the dispatch half.  It runs every applicable expert over
every candidate - not a chosen subset - so that trade-offs between properties
are comparable across the pool, as section 4 requires.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Sequence

from formulate.core.candidate import Candidate
from formulate.core.conditions import Conditions
from formulate.core.prediction import Prediction, prefer
from formulate.experts.base import Expert, PredictionRequest
from formulate.experts.registry import ExpertRegistry
from formulate.store.cache import PredictionCache
from formulate.targets.spec import TargetSpec


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Knobs for dispatch and scoring."""

    #: Threads used to evaluate candidates concurrently.
    max_workers: int = 4
    #: Standard deviations used when computing the risk-adjusted utility.
    risk_k: float = 1.0
    #: Rank on the risk-adjusted utility rather than the nominal one.
    use_risk_adjusted: bool = True
    #: Multiplier applied to the utility of an out-of-domain prediction.
    out_of_domain_penalty: float = 0.5
    #: How a soft objective with no usable prediction is treated.
    #: "penalize" scores it zero and records why; "exclude" drops it from the
    #: objective vector and renormalises. Neither silently scores it neutral.
    missing_objective_policy: str = "penalize"
    temperature_tolerance_k: float = 5.0
    pressure_rtol: float = 0.05


@dataclass
class DispatchReport:
    """What happened during one dispatch pass."""

    experts_run: tuple[str, ...] = ()
    uncovered_properties: tuple[str, ...] = ()
    unavailable_experts: dict[str, str] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    predictions_made: int = 0

    def describe(self) -> str:
        lines = [f"Experts run: {', '.join(self.experts_run) or 'none'}"]
        if self.uncovered_properties:
            lines.append(
                "No expert covers: " + ", ".join(self.uncovered_properties)
            )
        for expert_id, reason in sorted(self.unavailable_experts.items()):
            lines.append(f"Expert {expert_id} unavailable: {reason}")
        total = self.cache_hits + self.cache_misses
        if total:
            lines.append(
                f"Cache: {self.cache_hits} hits / {total} lookups "
                f"({self.cache_hits / total:.0%})"
            )
        lines.append(f"Predictions produced: {self.predictions_made}")
        return "\n".join(lines)


class EvaluationEngine:
    """Runs the applicable expert panel over a candidate pool."""

    def __init__(
        self,
        registry: ExpertRegistry,
        config: EvaluationConfig | None = None,
        cache: PredictionCache | None = None,
    ) -> None:
        self.registry = registry
        self.config = config or EvaluationConfig()
        self.cache = cache if cache is not None else PredictionCache()

    # -- dispatch ----------------------------------------------------------

    def predict(
        self, candidates: Sequence[Candidate], spec: TargetSpec
    ) -> tuple[list[list[Prediction]], DispatchReport]:
        """Predict every requested property for every candidate.

        Returns predictions parallel to ``candidates``, plus a report naming
        the properties no expert could cover.  A coverage gap is surfaced, not
        silently left blank.
        """
        report = DispatchReport()
        if not candidates:
            return [], report

        properties = self._properties_needed(spec)
        classes = {c.material_class for c in candidates}

        uncovered: set[str] = set()
        for material_class in classes:
            uncovered |= set(self.registry.uncovered(properties, material_class))
        report.uncovered_properties = tuple(sorted(uncovered))

        for expert in self.registry:
            if not expert.is_available():
                report.unavailable_experts[expert.id] = expert.unavailable_reason()

        run_ids: set[str] = set()
        results: list[list[Prediction]] = [[] for _ in candidates]

        def work(index: int) -> None:
            preds, used = self._predict_one_candidate(candidates[index], spec, properties)
            results[index] = preds
            run_ids.update(used)

        if self.config.max_workers > 1 and len(candidates) > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
                list(pool.map(work, range(len(candidates))))
        else:
            for index in range(len(candidates)):
                work(index)

        report.experts_run = tuple(sorted(run_ids))
        report.cache_hits = self.cache.hits
        report.cache_misses = self.cache.misses
        report.predictions_made = sum(len(p) for p in results)
        return results, report

    def _properties_needed(self, spec: TargetSpec) -> frozenset[str]:
        """Requested properties plus the dependencies experts need to supply them."""
        wanted = set(spec.properties)
        # Pull in dependencies transitively so that, for example, asking for a
        # surface tension also computes the critical constants it is built on.
        for _ in range(len(self.registry) + 1):
            added = False
            for expert in self.registry:
                if expert.supported_properties & wanted and not expert.dependencies <= wanted:
                    wanted |= expert.dependencies
                    added = True
            if not added:
                break
        return frozenset(wanted)

    def _predict_one_candidate(
        self, candidate: Candidate, spec: TargetSpec, properties: frozenset[str]
    ) -> tuple[list[Prediction], set[str]]:
        experts = self.registry.experts_for(properties, candidate.material_class)
        ordered = self.registry.resolution_order(experts)

        context: dict[str, Prediction] = {}
        out: list[Prediction] = []
        used: set[str] = set()

        for expert in ordered:
            conditions = self._conditions_for(expert, spec)
            # Keyed on what this expert will actually answer, not the whole
            # request: two specifications that differ only in a property this
            # expert does not cover should still share its entry.
            answered = expert.applicable_properties(properties, candidate.material_class)
            key = PredictionCache.key(
                candidate.candidate_id, expert.id, expert.version, conditions, answered
            )
            cached = self.cache.get(key)
            if cached is None:
                request = PredictionRequest(
                    candidate=candidate,
                    properties=properties,
                    conditions=conditions,
                    context=dict(context),
                )
                cached = expert.predict(request)
                self.cache.put(key, cached)
            if cached:
                used.add(expert.id)
            out.extend(cached)
            for pred in cached:
                if pred.is_usable:
                    incumbent = context.get(pred.property)
                    if incumbent is None or prefer(pred, incumbent):
                        context[pred.property] = pred
        return out, used

    def _conditions_for(self, expert: Expert, spec: TargetSpec) -> Conditions:
        """Conditions to evaluate an expert at.

        Requirements may state their own conditions; where several requirements
        an expert serves disagree, the spec-level conditions are used and the
        per-requirement condition check later reports the mismatch.
        """
        stated = [
            spec.conditions_for(req)
            for req in spec.requirements
            if req.property in expert.supported_properties and req.conditions is not None
        ]
        if len(stated) == 1:
            return stated[0]
        return spec.conditions
