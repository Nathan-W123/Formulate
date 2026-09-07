"""The adaptive coordinator.

Specification section 5: "Choose the next action rather than merely the next
candidate: generate more, diversify, call another expert, increase model
fidelity, run QM/MD, or stop.  Allocate expensive compute where expected
information gain or ranking impact is high."

The fixed coordinator answers "what next?" with a schedule.  This one answers
it with an estimate: every action is scored by the ranking movement it is
expected to buy, divided by what it costs, and the best ratio wins.  Stopping
is an action like any other and is chosen when nothing else clears its price.

Section 10 is blunt about the standard this must meet: the coordinator "earns
complexity only if it improves quality per unit compute".  That is why the
expected-value model is built from quantities the run actually measures - the
observed hypervolume gain per candidate in recent rounds, the observed cost of
a physics call - rather than from constants chosen to make the policy look
good.  A model fitted to the run can be wrong; a model invented to justify the
policy cannot even be checked.

No scientific decision routes through a language model.  Section 5 permits a
reasoning layer only for translating intent, decomposing niche requests, and
explaining, and this module contains none: given a seed, it is deterministic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from formulate.core.candidate import Candidate, dedupe
from formulate.targets.spec import TargetSpec

from .coordinator import DesignRun, RunConfig
from .iterative import IterationConfig, IterativeCoordinator
from .metrics import RoundRecord, SearchMetrics
from .report import render_report
from .validation import PhysicsValidator, ValidationPolicy, ValidationReport, select_targets


class Action(str, Enum):
    """What the coordinator may do next (specification section 5)."""

    #: Propose more candidates from the existing population.
    GENERATE = "generate"
    #: Propose candidates deliberately far from what has been seen.
    DIVERSIFY = "diversify"
    #: Spend a quantum or dynamics calculation on the most informative gap.
    VALIDATE = "validate"
    #: Nothing left is worth its cost.
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class ActionEstimate:
    """What an action is expected to buy, and what it will cost."""

    action: Action
    #: Expected improvement in the ranking, in hypervolume units.
    expected_gain: float
    #: Expected cost in seconds.
    expected_cost: float
    rationale: str

    @property
    def value_density(self) -> float:
        """Expected gain per second. The quantity the policy maximises."""
        if self.expected_cost <= 0:
            return 0.0
        return self.expected_gain / self.expected_cost

    def describe(self) -> str:
        return (
            f"{self.action.value:9s} gain {self.expected_gain:.5f} / "
            f"{self.expected_cost:6.1f} s = {self.value_density:.3e}   {self.rationale}"
        )


@dataclass(frozen=True, slots=True)
class AdaptiveConfig:
    """Budget and policy for an adaptive run."""

    #: Total wall-clock the coordinator may spend, in seconds.
    budget_seconds: float = 300.0
    #: Candidates proposed by one GENERATE or DIVERSIFY action.
    batch_size: int = 15
    #: Hard ceiling on actions, as a runaway guard rather than a policy.
    max_actions: int = 40
    #: Stop when the best action's expected gain per second falls below this.
    #: Zero means "stop only when nothing has positive expected gain".
    minimum_value_density: float = 0.0
    validation: ValidationPolicy = field(
        default_factory=lambda: ValidationPolicy(max_candidates=1, max_seconds=120.0)
    )
    #: Cost assumed for a physics call before one has been observed. Replaced by
    #: the measured cost as soon as the run has one.
    prior_validation_seconds: float = 25.0
    #: Expected hypervolume gain per candidate before any round has been seen.
    #: Deliberately optimistic so the first action is always to generate.
    prior_gain_per_candidate: float = 1e-3
    #: Expected gain from one physics call before one has been observed.
    prior_validation_gain: float = 5e-3
    #: Weight of the prior against observations, in pseudo-observations.
    #:
    #: Without it a single unlucky measurement is fatal: the first generation
    #: round returned no hypervolume at all, which set the observed yield to
    #: zero and locked generation out for the rest of the run while validation
    #: was chosen thirteen times, the last eight for no gain. Shrinking toward
    #: the prior keeps an action that has stopped paying from becoming
    #: permanently unreachable.
    prior_weight: float = 2.0


@dataclass
class ActionRecord:
    """One decision and what it actually produced."""

    index: int
    chosen: Action
    estimates: list[ActionEstimate]
    actual_cost: float
    actual_gain: float
    hypervolume_after: float
    detail: str = ""

    def describe(self) -> str:
        return (
            f"#{self.index:<3} {self.chosen.value:9s} cost {self.actual_cost:6.1f} s  "
            f"gain {self.actual_gain:+.5f}  hypervolume {self.hypervolume_after:.4f}  "
            f"{self.detail}"
        )


@dataclass
class AdaptiveRun:
    """The outcome of an adaptive run, including every decision it made."""

    spec: TargetSpec
    final: DesignRun
    metrics: SearchMetrics
    actions: list[ActionRecord] = field(default_factory=list)
    validation: ValidationReport | None = None
    seconds_spent: float = 0.0

    @property
    def action_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.actions:
            counts[record.chosen.value] = counts.get(record.chosen.value, 0) + 1
        return counts

    def describe_policy(self) -> str:
        lines = [
            f"Adaptive policy spent {self.seconds_spent:.1f} s over {len(self.actions)} actions",
            "  " + ", ".join(f"{k}: {v}" for k, v in sorted(self.action_counts.items())),
            "",
            "Decisions:",
        ]
        lines.extend(f"  {record.describe()}" for record in self.actions)
        return "\n".join(lines)

    def report(self, top_k: int = 5) -> str:
        rule = "-" * 72
        parts = [render_report(self.final, top_k=top_k), "", rule, "SEARCH", rule, "",
                 self.metrics.describe(), "", rule, "ADAPTIVE POLICY", rule, "",
                 self.describe_policy()]
        if self.validation is not None:
            parts += ["", rule, "PHYSICS VALIDATION", rule, "", self.validation.describe()]
        return "\n".join(parts)


def objective_coverage(run: DesignRun, spec: TargetSpec) -> tuple[float, tuple[str, ...]]:
    """Fraction of requested objectives any candidate has a real value for.

    Hypervolume alone cannot steer this search. An objective that no expert
    covers scores zero for every candidate, and a frontier with one identically
    zero axis has zero volume however good the others are - so a policy driven
    by hypervolume sees no progress at all and cannot tell that acquiring the
    missing axis is the only thing worth doing. Coverage is the signal that
    makes that visible.
    """
    requested = list(spec.properties)
    if not requested:
        return 1.0, ()
    uncovered: list[str] = []
    for prop in requested:
        has_value = any(
            entry.candidate.results is not None
            and entry.candidate.results.prediction_for(prop) is not None
            for entry in run.ranking.ranked
        )
        if not has_value:
            uncovered.append(prop)
    return 1.0 - len(uncovered) / len(requested), tuple(uncovered)


def progress(run: DesignRun, spec: TargetSpec) -> float:
    """The quantity the policy tries to increase.

    Coverage dominates while any objective is unmeasured, because until then
    the frontier volume is structurally zero and improving the measured axes
    buys nothing. Once everything is covered, hypervolume takes over.
    """
    coverage, _ = objective_coverage(run, spec)
    if coverage < 1.0:
        return coverage
    return 1.0 + run.ranking.hypervolume


class AdaptiveCoordinator(IterativeCoordinator):
    """Chooses the next action by expected ranking gain per unit cost."""

    def __init__(self, *args, adaptive: AdaptiveConfig | None = None, **kwargs) -> None:
        kwargs.setdefault("iteration", IterationConfig(max_rounds=0))
        super().__init__(*args, **kwargs)
        self.adaptive = adaptive or AdaptiveConfig()
        self._validator = PhysicsValidator(self.adaptive.validation)
        # Physics results survive re-evaluation only if they are held outside
        # the candidate. Every generation round re-runs the expert panel and
        # rebuilds each candidate's results from expert predictions alone, so a
        # validated value merged into a candidate is silently discarded the
        # next time anything is generated - which showed up as a validated
        # objective losing its coverage and the frontier collapsing back to
        # zero volume.
        self._physics: dict[str, list] = {}

    # -- the loop ----------------------------------------------------------

    def run_adaptive(self, spec: TargetSpec) -> AdaptiveRun:
        config = self.adaptive
        started = time.perf_counter()

        pool: list[Candidate] = []
        run = self.run(spec, candidates=[])
        metrics = SearchMetrics()
        actions: list[ActionRecord] = []
        validation: ValidationReport | None = None

        # Observed costs and yields, blended with the priors rather than
        # replacing them. This is what lets the policy be judged rather than
        # merely asserted.
        observed_gain_per_candidate: list[float] = []
        observed_validation_cost: list[float] = []
        observed_validation_gain: list[float] = []

        for index in range(config.max_actions):
            remaining = config.budget_seconds - (time.perf_counter() - started)
            if remaining <= 0:
                break

            estimates = self._estimate_actions(
                spec,
                run,
                pool,
                observed_gain_per_candidate,
                observed_validation_cost,
                observed_validation_gain,
                remaining,
            )
            best = max(estimates, key=lambda e: e.value_density)
            if best.action is Action.STOP or best.value_density <= config.minimum_value_density:
                actions.append(
                    ActionRecord(
                        index=index,
                        chosen=Action.STOP,
                        estimates=estimates,
                        actual_cost=0.0,
                        actual_gain=0.0,
                        hypervolume_after=run.ranking.hypervolume,
                        detail="no action cleared its cost",
                    )
                )
                break

            before = progress(run, spec)
            action_started = time.perf_counter()

            if best.action in (Action.GENERATE, Action.DIVERSIFY):
                run, pool, detail = self._act_generate(
                    spec, pool, best.action, index, config
                )
            else:
                run, pool, validation, detail = self._act_validate(
                    spec, pool, run, validation
                )

            cost = time.perf_counter() - action_started
            gain = progress(run, spec) - before

            if best.action in (Action.GENERATE, Action.DIVERSIFY) and config.batch_size:
                observed_gain_per_candidate.append(max(0.0, gain) / config.batch_size)
            elif best.action is Action.VALIDATE:
                observed_validation_cost.append(cost)
                observed_validation_gain.append(max(0.0, gain))

            actions.append(
                ActionRecord(
                    index=index,
                    chosen=best.action,
                    estimates=estimates,
                    actual_cost=cost,
                    actual_gain=gain,
                    hypervolume_after=run.ranking.hypervolume,
                    detail=detail,
                )
            )
            metrics.rounds.append(self._round_record(index, run, len(pool)))

        metrics.stop_reason = self._stop_summary(actions, config, started)
        return AdaptiveRun(
            spec=spec,
            final=run,
            metrics=metrics,
            actions=actions,
            validation=validation,
            seconds_spent=time.perf_counter() - started,
        )

    # -- the value model ---------------------------------------------------

    def _estimate_actions(
        self,
        spec: TargetSpec,
        run: DesignRun,
        pool: Sequence[Candidate],
        gains: list[float],
        validation_costs: list[float],
        validation_gains: list[float],
        remaining: float,
    ) -> list[ActionEstimate]:
        """Score every available action by expected gain per second."""
        config = self.adaptive
        estimates: list[ActionEstimate] = [
            ActionEstimate(Action.STOP, 0.0, 1.0, "the null action")
        ]

        # GENERATE. Expected gain is the recent observed hypervolume gain per
        # candidate, which decays naturally as the frontier saturates: once
        # rounds stop paying, this estimate falls and other actions win.
        recent = gains[-3:]
        gain_per_candidate = _shrunk(
            recent, config.prior_gain_per_candidate, config.prior_weight
        )
        generate_cost = self._generation_cost(config.batch_size)
        if generate_cost <= remaining:
            estimates.append(
                ActionEstimate(
                    Action.GENERATE,
                    gain_per_candidate * config.batch_size,
                    generate_cost,
                    (
                        f"{gain_per_candidate:.2e} hypervolume per candidate, from "
                        f"{len(recent)} observation(s) shrunk toward the prior"
                    ),
                )
            )

            # DIVERSIFY. Worth more when the pool has collapsed onto one family,
            # because a frontier made of near-duplicates is fragile: a single
            # wrong prediction moves all of it together.
            diversity = run.ranking.mean_diversity
            deficit = 1.0 if diversity is None else max(0.0, 1.0 - diversity)
            estimates.append(
                ActionEstimate(
                    Action.DIVERSIFY,
                    gain_per_candidate * config.batch_size * (0.5 + deficit),
                    generate_cost,
                    f"structural spread is {diversity:.2f}"
                    if diversity is not None
                    else "spread not yet measurable",
                )
            )

        # VALIDATE. Reuses the selection policy's own estimate of how much a
        # candidate-property pair could move the ranking, rather than a second
        # opinion about the same question.
        targets, _ = select_targets(
            [c for c in pool if c.results is not None], spec, config.validation
        )
        if targets:
            cost = (
                sum(validation_costs) / len(validation_costs)
                if validation_costs
                else config.prior_validation_seconds
            )
            if cost <= remaining:
                # The selection policy's value score says how much this pair
                # *could* move the ranking; the observed history says how much
                # validation actually has. The estimate is the score scaled by
                # the measured yield, so a run where validation keeps returning
                # nothing stops buying it.
                best_target = targets[0]
                measured = _shrunk(
                    validation_gains[-3:],
                    config.prior_validation_gain,
                    config.prior_weight,
                )
                _, uncovered = objective_coverage(run, spec)
                if best_target.property in uncovered:
                    # Acquiring an objective nothing has measured is worth a
                    # whole coverage step, and no amount of generating can
                    # substitute for it: this is the expected-information-gain
                    # case section 5 is describing.
                    gain = 1.0 / max(1, len(spec.properties))
                    why = f"{best_target.property} is not covered by any expert"
                else:
                    gain = best_target.value_score * measured
                    why = (
                        f"observed yield {measured:.2e} over "
                        f"{len(validation_gains)} call(s)"
                    )
                estimates.append(
                    ActionEstimate(
                        Action.VALIDATE,
                        gain,
                        cost,
                        f"{best_target.property} on "
                        f"{best_target.candidate.label or 'a frontier candidate'}: {why}",
                    )
                )
        return estimates

    def _generation_cost(self, batch: int) -> float:
        """Seconds a generation round is expected to take.

        Measured from the evaluation cache's own hit rate rather than assumed:
        a round that mostly re-evaluates cached candidates is nearly free.
        """
        per_candidate = 0.12
        cache = self.engine.cache
        total = cache.hits + cache.misses
        if total:
            per_candidate *= max(0.1, 1.0 - cache.hit_rate)
        return max(0.5, batch * per_candidate)

    # -- the actions -------------------------------------------------------

    def _act_generate(
        self,
        spec: TargetSpec,
        pool: list[Candidate],
        action: Action,
        index: int,
        config: AdaptiveConfig,
    ) -> tuple[DesignRun, list[Candidate], str]:
        seed = self.config.seed + index + (1000 if action is Action.DIVERSIFY else 0)
        proposed = self._explore(spec, scored=pool, seed=seed, budget=config.batch_size)
        known = {c.structure_id for c in pool}
        proposed = [c for c in dedupe(proposed) if c.structure_id not in known]
        if not proposed:
            run = self._reapply_physics(self.run(spec, candidates=list(pool)), spec)
            return run, [e.candidate for e in run.ranking.ranked], "no new candidate available"

        run = self._reapply_physics(
            self.run(spec, candidates=list(pool) + proposed), spec
        )
        return run, [e.candidate for e in run.ranking.ranked], f"{len(proposed)} proposed"

    def _reapply_physics(self, run: DesignRun, spec: TargetSpec) -> DesignRun:
        """Restore validated predictions that re-evaluation dropped.

        Expert dispatch rebuilds every candidate's results, so anything physics
        contributed has to be put back and the pool re-scored, or a validated
        objective silently reverts to uncovered.
        """
        if not self._physics:
            return run
        return self._rescore(
            [entry.candidate for entry in run.ranking.ranked], spec, run, merge_physics=True
        )

    def _act_validate(
        self,
        spec: TargetSpec,
        pool: list[Candidate],
        run: DesignRun,
        previous: ValidationReport | None,
    ) -> tuple[DesignRun, list[Candidate], ValidationReport, str]:
        ranked = [entry.candidate for entry in run.ranking.ranked]
        validated, report = self._validator.validate(ranked, spec)

        # Remember what physics contributed, keyed by candidate id. The id is
        # content-addressed, so the same structure re-proposed in a later round
        # carries the same key and its evidence can be restored.
        for candidate in validated:
            if candidate.results is None:
                continue
            physics = [
                p
                for p in candidate.results.predictions
                if p.expert_id.startswith(("qm:", "md:"))
            ]
            if physics:
                self._physics[candidate.candidate_id] = physics

        updated = self._rescore(validated, spec, run, merge_physics=True)

        merged = previous or report
        if previous is not None:
            merged.targets.extend(report.targets)
            merged.validated.extend(report.validated)
            merged.disagreements.extend(report.disagreements)
            merged.calls_made += report.calls_made
            merged.calls_avoided += report.calls_avoided
            merged.seconds_spent += report.seconds_spent
            merged.skipped.update(report.skipped)

        detail = f"{report.calls_made} calculation(s)"
        return updated, [e.candidate for e in updated.ranking.ranked], merged, detail

    def _rescore(
        self,
        candidates: Sequence[Candidate],
        spec: TargetSpec,
        run: DesignRun,
        *,
        merge_physics: bool,
    ) -> DesignRun:
        """Re-score and re-rank a pool, optionally restoring physics evidence."""
        from formulate.evaluation.scoring import score_pool

        prepared: list[Candidate] = []
        for candidate in candidates:
            results = candidate.results
            if merge_physics and results is not None:
                physics = self._physics.get(candidate.candidate_id)
                if physics:
                    known = {(p.property, p.expert_id) for p in results.predictions}
                    extra = tuple(p for p in physics if (p.property, p.expert_id) not in known)
                    if extra:
                        results = results.model_copy(
                            update={
                                "predictions": results.predictions + extra,
                                "simulation_ids": results.simulation_ids
                                + tuple(
                                    p.provenance.record_id for p in extra if p.provenance
                                ),
                            }
                        )
                        candidate = candidate.with_results(results)
            prepared.append(candidate)

        predictions = [
            list(c.results.predictions) if c.results is not None else [] for c in prepared
        ]
        rescored, outcomes, desirabilities = score_pool(
            prepared, predictions, spec, self.config.evaluation
        )
        evidence = {
            c.candidate_id: c.results.simulation_ids
            for c in prepared
            if c.results is not None and c.results.simulation_ids
        }
        rescored = [
            c.with_results(
                c.results.model_copy(update={"simulation_ids": evidence[c.candidate_id]})
            )
            if c.candidate_id in evidence and c.results is not None
            else c
            for c in rescored
        ]
        reranked = self.ranker.rank(rescored, spec)
        return DesignRun(
            spec=spec,
            ranking=reranked,
            dispatch=run.dispatch,
            filters=run.filters,
            feasibility=run.feasibility,
            outcomes={c.candidate_id: o for c, o in zip(rescored, outcomes)},
            desirabilities=desirabilities,
            proposed=run.proposed,
            evaluated=run.evaluated,
        )

    # -- bookkeeping -------------------------------------------------------

    def _round_record(self, index: int, run: DesignRun, pool_size: int) -> RoundRecord:
        best = max(
            (e.scalar for e in run.ranking.ranked if e.feasible and e.scalar is not None),
            default=None,
        )
        return RoundRecord(
            index=index,
            proposed=pool_size,
            rejected_by_filters=len(run.filters.rejected),
            newly_evaluated=0,
            total_evaluated=run.evaluated,
            feasible=run.ranking.feasible_count,
            hypervolume=run.ranking.hypervolume,
            frontier_size=len(run.ranking.frontier),
            mean_structural_distance=run.ranking.mean_diversity,
            best_scalar=best,
        )

    def _stop_summary(
        self, actions: list[ActionRecord], config: AdaptiveConfig, started: float
    ) -> str:
        if actions and actions[-1].chosen is Action.STOP:
            return "no remaining action was expected to pay for itself"
        if time.perf_counter() - started >= config.budget_seconds:
            return f"the {config.budget_seconds:.0f} s budget was spent"
        return f"the ceiling of {config.max_actions} actions was reached"


def _shrunk(observations: Sequence[float], prior: float, weight: float) -> float:
    """Blend observations with a prior, weighted in pseudo-observations.

    A plain mean of the observations lets one zero permanently disqualify an
    action, which is how the first version spent its whole budget on
    validations that returned nothing. Shrinking keeps a stalled action
    reachable while still letting sustained evidence dominate.
    """
    if not observations:
        return prior
    total = sum(observations) + prior * weight
    return total / (len(observations) + weight)


def default_adaptive_coordinator(
    config: RunConfig | None = None, adaptive: AdaptiveConfig | None = None
) -> AdaptiveCoordinator:
    """Retrieval, evolution and composition search, driven by the adaptive policy."""
    from formulate.exploration.bayesopt import BayesOptExplorer
    from formulate.exploration.database import ReferenceDatabaseExplorer
    from formulate.exploration.evolutionary import EvolutionaryExplorer
    from formulate.exploration.mixtures import MixtureSeedExplorer

    return AdaptiveCoordinator(
        explorers=[
            ReferenceDatabaseExplorer(),
            MixtureSeedExplorer(),
            EvolutionaryExplorer(),
            BayesOptExplorer(),
        ],
        config=config or RunConfig(),
        adaptive=adaptive,
    )
