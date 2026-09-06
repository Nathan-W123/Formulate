"""Aggregate ranking.

Specification section 5: the deterministic coordinator must "enforce
constraints, deduplicate results, compute Pareto/scalar rankings".

Order of precedence:

1. Feasibility.  A candidate breaching a hard constraint - or one whose hard
   constraint could not be checked at all - never outranks a feasible one.
2. Pareto front.  Front 0 is the set of candidates not beaten on every
   objective simultaneously.
3. The scalar baseline, used only to give a stable display order inside a
   front.  It is a tiebreak, not a claim that the higher-scoring candidate is
   better; that is what makes it a baseline rather than the ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from formulate.core.candidate import Candidate, CandidateResults
from formulate.targets.spec import TargetSpec

from .diversity import max_min_selection, mean_pairwise_distance
from .pareto import crowding_distance, hypervolume, non_dominated_sort


@dataclass(frozen=True, slots=True)
class RankingConfig:
    """How to turn scored candidates into an ordered list."""

    #: "rank_last" keeps infeasible candidates in the report, below every
    #: feasible one, with their violations attached - usually more useful than
    #: dropping them, because a near-miss explains the design space.
    #: "eliminate" removes them entirely.
    infeasible_policy: str = "rank_last"
    #: Objectives are dropped from dominance comparison when some candidate
    #: lacks them; set False to raise instead.
    allow_partial_objectives: bool = True


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """One candidate with its position established."""

    candidate: Candidate
    rank: int
    front: int
    crowding: float
    scalar: float | None
    feasible: bool

    @property
    def results(self) -> CandidateResults:
        return self.candidate.results  # type: ignore[return-value]

    def describe(self) -> str:
        name = self.candidate.label or self.candidate.primary_smiles or self.candidate.candidate_id
        flag = "" if self.feasible else "  INFEASIBLE"
        scalar = "n/a" if self.scalar is None else f"{self.scalar:.3f}"
        return f"#{self.rank} {name}  front {self.front}  baseline {scalar}{flag}"


@dataclass
class RankingResult:
    """The outcome of ranking a pool."""

    ranked: list[RankedCandidate] = field(default_factory=list)
    #: Objective names, in the order used for dominance comparison.
    axes: tuple[str, ...] = ()
    #: Objectives excluded because not every candidate had them.
    dropped_axes: tuple[str, ...] = ()
    hypervolume: float = 0.0
    feasible_count: int = 0
    infeasible_count: int = 0
    mean_diversity: float | None = None

    @property
    def frontier(self) -> list[RankedCandidate]:
        """Feasible, non-dominated candidates."""
        return [r for r in self.ranked if r.feasible and r.front == 0]

    def top(self, count: int) -> list[RankedCandidate]:
        return self.ranked[:count]

    def diverse_top(self, count: int) -> list[RankedCandidate]:
        """``count`` candidates that are both well ranked and structurally spread.

        Selection runs over the feasible candidates in rank order, seeded with
        the best one, so the top recommendation is never displaced by the
        diversity criterion.
        """
        pool = [r for r in self.ranked if r.feasible] or self.ranked
        if count >= len(pool):
            return pool
        picked = max_min_selection(
            [r.candidate for r in pool], count, seed_index=0, order=range(len(pool))
        )
        return [pool[i] for i in sorted(picked, key=lambda i: pool[i].rank)]

    def describe(self, limit: int = 10) -> str:
        lines = [
            f"Ranked {len(self.ranked)} candidates "
            f"({self.feasible_count} feasible, {self.infeasible_count} infeasible)",
            f"Objectives: {', '.join(self.axes) or 'none'}",
        ]
        if self.dropped_axes:
            lines.append(
                "Excluded from dominance (not available for every candidate): "
                + ", ".join(self.dropped_axes)
            )
        lines.append(f"Frontier size: {len(self.frontier)}   hypervolume: {self.hypervolume:.4f}")
        if self.mean_diversity is not None:
            lines.append(f"Mean structural distance: {self.mean_diversity:.3f}")
        lines.append("")
        lines.extend(r.describe() for r in self.ranked[:limit])
        return "\n".join(lines)


class Ranker:
    """Turns scored candidates into an ordered, Pareto-aware ranking."""

    def __init__(self, config: RankingConfig | None = None) -> None:
        self.config = config or RankingConfig()

    def rank(self, candidates: Sequence[Candidate], spec: TargetSpec) -> RankingResult:
        scored = [c for c in candidates if c.results is not None]
        if not scored:
            return RankingResult()

        axes, dropped = self._axes(scored, spec)
        pool = scored
        if self.config.infeasible_policy == "eliminate":
            pool = [c for c in scored if c.results.feasible]  # type: ignore[union-attr]
            if not pool:
                return RankingResult(
                    axes=axes,
                    dropped_axes=dropped,
                    infeasible_count=len(scored),
                )

        feasible = [c for c in pool if c.results.feasible]  # type: ignore[union-attr]
        infeasible = [c for c in pool if not c.results.feasible]  # type: ignore[union-attr]

        entries: list[RankedCandidate] = []
        entries.extend(self._rank_group(feasible, axes, feasible=True))
        entries.extend(self._rank_group(infeasible, axes, feasible=False))

        for position, entry in enumerate(entries, start=1):
            object.__setattr__(entry, "rank", position)

        frontier_vectors = [
            self._vector(e.candidate, axes) for e in entries if e.feasible and e.front == 0
        ]
        result = RankingResult(
            ranked=entries,
            axes=axes,
            dropped_axes=dropped,
            hypervolume=hypervolume(frontier_vectors) if frontier_vectors and axes else 0.0,
            feasible_count=len(feasible),
            infeasible_count=len(infeasible),
            mean_diversity=mean_pairwise_distance([e.candidate for e in entries]),
        )
        self._write_back(result)
        return result

    # -- internals ---------------------------------------------------------

    def _axes(
        self, candidates: Sequence[Candidate], spec: TargetSpec
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Objectives usable for dominance: those every candidate supplies.

        Comparing candidates on an axis some of them lack would let a
        candidate dominate by virtue of not having been measured.
        """
        requested = list(spec.properties)
        common: list[str] = []
        dropped: list[str] = []
        for prop in requested:
            if all(prop in c.results.objective_vector for c in candidates):  # type: ignore[union-attr]
                common.append(prop)
            else:
                dropped.append(prop)
        if dropped and not self.config.allow_partial_objectives:
            raise ValueError(
                "Objectives missing for some candidates: " + ", ".join(dropped)
            )
        return tuple(common), tuple(dropped)

    def _vector(self, candidate: Candidate, axes: Sequence[str]) -> list[float]:
        vector = candidate.results.objective_vector  # type: ignore[union-attr]
        return [vector.get(axis, 0.0) for axis in axes]

    def _rank_group(
        self, candidates: Sequence[Candidate], axes: Sequence[str], *, feasible: bool
    ) -> list[RankedCandidate]:
        if not candidates:
            return []
        if not axes:
            fronts = [0] * len(candidates)
            crowding = [0.0] * len(candidates)
        else:
            vectors = [self._vector(c, axes) for c in candidates]
            fronts = non_dominated_sort(vectors)
            crowding = [0.0] * len(candidates)
            for front_index in set(fronts):
                members = [i for i, f in enumerate(fronts) if f == front_index]
                distances = crowding_distance([vectors[i] for i in members])
                for position, i in enumerate(members):
                    crowding[i] = distances[position]

        entries = [
            RankedCandidate(
                candidate=candidate,
                rank=0,
                front=fronts[i],
                crowding=crowding[i],
                scalar=candidate.results.scalar_score,  # type: ignore[union-attr]
                feasible=feasible,
            )
            for i, candidate in enumerate(candidates)
        ]
        entries.sort(key=lambda e: (e.front, -(e.scalar if e.scalar is not None else -1.0)))
        return entries

    def _write_back(self, result: RankingResult) -> None:
        """Record front and rank on each candidate's results block."""
        for position, entry in enumerate(result.ranked):
            results = entry.candidate.results
            if results is None:
                continue
            updated = results.model_copy(
                update={"pareto_front": entry.front, "aggregate_rank": entry.rank}
            )
            result.ranked[position] = RankedCandidate(
                candidate=entry.candidate.with_results(updated),
                rank=entry.rank,
                front=entry.front,
                crowding=entry.crowding,
                scalar=entry.scalar,
                feasible=entry.feasible,
            )
