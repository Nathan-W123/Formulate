"""The iterate loop.

Specification section 3: "generate batch -> cheap validity filters -> expert
scoring -> selection/Pareto archive -> update search populations/models ->
repeat until budget, convergence, or target satisfaction", and section 9
stage 7.

The single-pass coordinator cannot do this, and not by oversight: a
population-based explorer needs *scored* parents, and during a single pass
nothing has been scored yet. This loop is what closes that circuit, and it is
the whole reason the evolutionary explorer can contribute at all.

Re-ranking the accumulated pool each round rather than only the new arrivals
is deliberate. Pareto membership is a property of the pool, not of a
candidate, so a candidate that was on the frontier in round 2 may not be in
round 5, and only a full re-rank sees that. It is affordable because the
content-addressed cache means re-evaluation costs a dictionary lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from formulate.core.candidate import Candidate, dedupe
from formulate.targets.spec import TargetSpec

from .coordinator import DesignRun, DeterministicCoordinator, RunConfig
from .metrics import RoundRecord, SearchMetrics, top_k_recall
from .report import render_report


@dataclass(frozen=True, slots=True)
class IterationConfig:
    """When to keep searching and when to stop."""

    #: Search rounds after the initial seeding round.
    max_rounds: int = 5
    #: New candidates requested from the explorers each round.
    batch_size: int = 25
    #: Hard ceiling on distinct candidates evaluated, or None for no ceiling.
    max_evaluations: int | None = None
    #: Consecutive rounds of negligible hypervolume gain that end the search.
    plateau_rounds: int = 2
    #: Gain below this counts as no progress.
    hypervolume_tolerance: float = 1e-4
    #: Stop once the feasible frontier reaches this size, if set.
    target_frontier_size: int | None = None
    #: Known-good structures for the section 12 top-k recall measure.
    benchmark_smiles: tuple[str, ...] = ()
    top_k: int = 10


@dataclass
class IterativeRun:
    """The outcome of a multi-round search."""

    spec: TargetSpec
    final: DesignRun
    metrics: SearchMetrics
    #: Every round's DesignRun, oldest first, for auditing how the search moved.
    history: list[DesignRun] = field(default_factory=list)

    def report(self, top_k: int = 5) -> str:
        """The section 9 stage 10 report, preceded by how the search behaved."""
        search = self.metrics.describe()
        rule = "-" * 72
        return "\n".join(
            [render_report(self.final, top_k=top_k), "", rule, "SEARCH", rule, "", search]
        )


class IterativeCoordinator(DeterministicCoordinator):
    """Runs explore -> filter -> predict -> rank repeatedly, feeding scores back."""

    def __init__(
        self,
        *args,
        iteration: IterationConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.iteration = iteration or IterationConfig()

    def run_iterative(
        self, spec: TargetSpec, seed_candidates: Sequence[Candidate] | None = None
    ) -> IterativeRun:
        """Search until budget, convergence, or target satisfaction."""
        config = self.iteration
        metrics = SearchMetrics()
        history: list[DesignRun] = []

        pool: list[Candidate] = []
        run: DesignRun | None = None
        stop_reason = ""

        for index in range(config.max_rounds + 1):
            if index == 0:
                proposed = (
                    list(seed_candidates)
                    if seed_candidates is not None
                    else self._explore(spec, scored=(), seed=self.config.seed)
                )
            else:
                proposed = self._explore(
                    spec,
                    scored=pool,
                    seed=self.config.seed + index,
                    budget=config.batch_size,
                )
                # Only genuinely new structures count as progress.
                known = {c.structure_id for c in pool}
                proposed = [c for c in dedupe(proposed) if c.structure_id not in known]

            if not proposed and index > 0:
                stop_reason = "no explorer produced a new candidate"
                break

            previous_total = metrics.total_evaluated
            run = self.run(spec, candidates=list(pool) + list(proposed))
            history.append(run)
            pool = [entry.candidate for entry in run.ranking.ranked]

            metrics.rounds.append(
                self._record(index, len(proposed), previous_total, run, proposed)
            )

            stop_reason = self._stop_reason(metrics, config)
            if stop_reason:
                break

        if run is None:
            # Nothing was ever proposed: report the empty run rather than raising.
            run = self.run(spec, candidates=[])
            history.append(run)
            stop_reason = stop_reason or "no candidates were proposed"

        metrics.stop_reason = stop_reason or "maximum rounds reached"
        self._attach_recall(metrics, run, config)
        return IterativeRun(spec=spec, final=run, metrics=metrics, history=history)

    # -- internals ---------------------------------------------------------

    def _record(
        self,
        index: int,
        proposed_count: int,
        previous_total: int,
        run: DesignRun,
        proposed: Sequence[Candidate],
    ) -> RoundRecord:
        contributions: dict[str, int] = {}
        for candidate in proposed:
            strategy = candidate.generation_strategy
            contributions[strategy] = contributions.get(strategy, 0) + 1

        best = max(
            (e.scalar for e in run.ranking.ranked if e.feasible and e.scalar is not None),
            default=None,
        )
        return RoundRecord(
            index=index,
            proposed=proposed_count,
            rejected_by_filters=len(run.filters.rejected),
            newly_evaluated=max(0, run.evaluated - previous_total),
            total_evaluated=run.evaluated,
            feasible=run.ranking.feasible_count,
            hypervolume=run.ranking.hypervolume,
            frontier_size=len(run.ranking.frontier),
            mean_structural_distance=run.ranking.mean_diversity,
            best_scalar=best,
            contributions=contributions,
            rejection_reasons=dict(run.filters.rejection_counts),
        )

    def _stop_reason(self, metrics: SearchMetrics, config: IterationConfig) -> str:
        """The explicit stopping predicate.

        Checked in order of decisiveness: a satisfied target or an exhausted
        budget ends the run regardless of how well it was going, and a plateau
        only counts once there have been enough rounds to see one.
        """
        rounds = metrics.rounds
        if not rounds:
            return ""
        latest = rounds[-1]

        if (
            config.target_frontier_size is not None
            and latest.frontier_size >= config.target_frontier_size
        ):
            return (
                f"the feasible frontier reached {latest.frontier_size} candidates, "
                f"meeting the target of {config.target_frontier_size}"
            )

        if config.max_evaluations is not None and latest.total_evaluated >= config.max_evaluations:
            return (
                f"the evaluation budget of {config.max_evaluations} was reached "
                f"({latest.total_evaluated} evaluated)"
            )

        if len(rounds) > config.plateau_rounds:
            recent = rounds[-(config.plateau_rounds + 1):]
            gains = [b.hypervolume - a.hypervolume for a, b in zip(recent, recent[1:])]
            if all(gain <= config.hypervolume_tolerance for gain in gains):
                return (
                    f"hypervolume gained no more than {config.hypervolume_tolerance:g} "
                    f"for {config.plateau_rounds} consecutive rounds"
                )

        if latest.index >= config.max_rounds:
            return f"the maximum of {config.max_rounds} rounds was reached"
        return ""

    def _attach_recall(
        self, metrics: SearchMetrics, run: DesignRun, config: IterationConfig
    ) -> None:
        """Measure top-k recall when the caller supplied a benchmark."""
        if not config.benchmark_smiles:
            return
        ranked = [
            entry.candidate.primary_smiles
            for entry in run.ranking.ranked
            if entry.feasible and entry.candidate.primary_smiles
        ]
        metrics.benchmark_size = len(set(config.benchmark_smiles))
        metrics.top_k = config.top_k
        metrics.top_k_recall = top_k_recall(ranked, list(config.benchmark_smiles), config.top_k)


def default_iterative_coordinator(
    config: RunConfig | None = None, iteration: IterationConfig | None = None
) -> IterativeCoordinator:
    """An iterative coordinator seeded by the database and driven by evolution.

    Section 3 expects multiple strategies running in parallel and merging into
    one deduplicated pool: retrieval supplies a starting population that
    evolution then has something to work from.
    """
    from formulate.exploration.database import ReferenceDatabaseExplorer
    from formulate.exploration.evolutionary import EvolutionaryExplorer

    return IterativeCoordinator(
        explorers=[ReferenceDatabaseExplorer(), EvolutionaryExplorer()],
        config=config or RunConfig(),
        iteration=iteration,
    )
