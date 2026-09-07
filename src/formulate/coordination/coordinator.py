"""The deterministic MVP coordinator.

Specification section 5, baseline: "The MVP coordinator is deterministic:
dispatch candidates to applicable experts, validate units/conditions, normalize
objectives, enforce constraints, deduplicate results, compute Pareto/scalar
rankings, and control iteration budgets."

Deterministic is the point.  The adaptive coordinator of section 5 - choosing
the next *action* rather than the next candidate, and allocating expensive
compute by expected information gain - is Phase 5, and section 10 is explicit
that it "earns complexity only if it improves quality per unit compute".  That
comparison needs this baseline to exist first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from formulate.core.candidate import Candidate, dedupe
from formulate.evaluation.engine import DispatchReport, EvaluationConfig, EvaluationEngine
from formulate.evaluation.scoring import RequirementOutcome, score_pool
from formulate.experts.registry import ExpertRegistry
from formulate.exploration.base import Explorer
from formulate.exploration.filters import CandidateFilter, FilterReport
from formulate.ranking.ranker import Ranker, RankingConfig, RankingResult
from formulate.store.cache import PredictionCache
from formulate.targets.desirability import Desirability
from formulate.targets.spec import TargetSpec

from .feasibility import FeasibilityAnalysis, analyze
from .report import render_report


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Budget and behaviour for one design run."""

    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    #: Upper bound on candidates drawn from the explorers.
    pool_size: int = 100
    #: How many candidates the report details.
    top_k: int = 10
    seed: int = 0


@dataclass
class DesignRun:
    """Everything one run produced, including why it produced it."""

    spec: TargetSpec
    ranking: RankingResult
    dispatch: DispatchReport
    filters: FilterReport
    feasibility: FeasibilityAnalysis
    #: Per-candidate requirement audit, keyed by candidate id.
    outcomes: dict[str, list[RequirementOutcome]] = field(default_factory=dict)
    desirabilities: dict[str, Desirability | None] = field(default_factory=dict)
    proposed: int = 0
    evaluated: int = 0

    def outcomes_for(self, candidate: Candidate) -> list[RequirementOutcome]:
        return self.outcomes.get(candidate.candidate_id, [])

    def report(self, top_k: int = 10) -> str:
        """The section 9 stage 10 report."""
        return render_report(self, top_k=top_k)

    @property
    def pool_relative_properties(self) -> tuple[str, ...]:
        """Objectives whose scale came from the pool rather than the request."""
        return tuple(
            sorted(
                prop
                for prop, desirability in self.desirabilities.items()
                if desirability is not None and desirability.pool_relative
            )
        )


class DeterministicCoordinator:
    """Runs specify -> explore -> filter -> predict -> rank, once, reproducibly."""

    def __init__(
        self,
        registry: ExpertRegistry | None = None,
        explorers: Sequence[Explorer] | None = None,
        config: RunConfig | None = None,
        cache: PredictionCache | None = None,
    ) -> None:
        from formulate.experts import default_registry
        from formulate.exploration.database import ReferenceDatabaseExplorer

        self.registry = registry if registry is not None else default_registry()
        self.explorers = list(explorers) if explorers is not None else [ReferenceDatabaseExplorer()]
        self.config = config or RunConfig()
        self.engine = EvaluationEngine(self.registry, self.config.evaluation, cache)
        self.ranker = Ranker(self.config.ranking)

    def run(self, spec: TargetSpec, candidates: Sequence[Candidate] | None = None) -> DesignRun:
        """Execute one full pass.

        ``candidates`` overrides exploration, which is how the calibration
        check evaluates a fixed known set rather than a proposed one.
        """
        proposed = list(candidates) if candidates is not None else self._explore(spec)

        proposed = dedupe(proposed)

        filter_report = CandidateFilter(spec.structural, spec.conditions).apply(proposed)
        pool = list(filter_report.kept)

        if not pool:
            return DesignRun(
                spec=spec,
                ranking=RankingResult(),
                dispatch=DispatchReport(),
                filters=filter_report,
                feasibility=FeasibilityAnalysis(),
                proposed=len(proposed),
                evaluated=0,
            )

        predictions, dispatch = self.engine.predict(pool, spec)
        scored, outcomes, desirabilities = score_pool(pool, predictions, spec, self.config.evaluation)
        ranking = self.ranker.rank(scored, spec)
        feasibility = analyze(scored, spec)

        return DesignRun(
            spec=spec,
            ranking=ranking,
            dispatch=dispatch,
            filters=filter_report,
            feasibility=feasibility,
            outcomes={
                candidate.candidate_id: outcome
                for candidate, outcome in zip(scored, outcomes)
            },
            desirabilities=desirabilities,
            proposed=len(proposed),
            evaluated=len(scored),
        )

    def _explore(
        self,
        spec: TargetSpec,
        scored: Sequence[Candidate] = (),
        seed: int | None = None,
        budget: int | None = None,
    ) -> list[Candidate]:
        """Draw candidates from every explorer and merge into one pool.

        Section 3: "Multiple search strategies operate in parallel and merge
        into one deduplicated candidate pool."

        ``scored`` must contain candidates that have actually been evaluated.
        Passing the pool being accumulated in this very call would be a lie: at
        that moment none of it carries results, so a population-based explorer
        would find no parents and silently contribute nothing. Feeding real
        scores back is the iterate loop's job (section 3, "Iteration"), which
        is why this takes them as an argument rather than inventing them.
        """
        budget = self.config.pool_size if budget is None else budget
        already = list(scored)
        usable = [e for e in self.explorers if e.is_available()]
        if not usable or budget <= 0:
            return []

        effective_seed = self.config.seed if seed is None else seed
        pool: list[Candidate] = []

        # Split the budget between strategies rather than serving them in order.
        # First-come-first-served let whichever explorer ran first consume the
        # whole allocation: with retrieval ahead of evolution, evolution
        # contributed nothing for as long as the database had unseen compounds
        # left. Section 3 wants the strategies running in parallel and budget
        # reserved for distinct regions, which a fair share is the minimum
        # expression of.
        # A budget smaller than the number of strategies cannot be split evenly.
        # Handing the remainder to whoever happens to be first in the list
        # starves the same explorer every round - and it is always the last one,
        # which is the newest and least established. Rotating the order by the
        # seed spreads the shortfall across rounds instead of concentrating it.
        base, extra = divmod(budget, len(usable))
        offset = effective_seed % len(usable)
        rotated = usable[offset:] + usable[:offset]
        shares = [base + (1 if i < extra else 0) for i in range(len(rotated))]

        for explorer, share in zip(rotated, shares):
            room = min(share, budget - len(pool))
            if room <= 0:
                continue
            pool.extend(
                explorer.propose(spec, room, scored=already + pool, seed=effective_seed)
            )

        # Redistribute whatever the first pass left unspent, since an explorer
        # that has exhausted its source should not strand the budget.
        #
        # On a different seed: a seeded explorer asked twice with one seed
        # replays its opening draws and spends the second pass rediscovering
        # candidates already in the pool. The offset is large and odd rather
        # than +1 so that a round's second pass cannot collide with the next
        # round's first, which the iterate loop numbers sequentially.
        second_pass = effective_seed + 1013904223
        for explorer in rotated:
            remaining = budget - len(pool)
            if remaining <= 0:
                break
            pool.extend(
                explorer.propose(spec, remaining, scored=already + pool, seed=second_pass)
            )
        return pool
