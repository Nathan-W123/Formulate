"""Search metrics for the iterative loop.

Specification section 12 names what a search must be judged on, and section 10
makes measuring it the deliverable of Phase 2: "measure hit rate, diversity,
sample efficiency, and failure modes."

Two of these are easy to report dishonestly and are defined carefully here:

* **Success rate** is over candidates *generated*, not over candidates
  surviving. Measuring it over survivors would rise toward 1.0 simply because
  the filters work, and would say nothing about whether the generator is
  proposing anything useful.
* **Hypervolume per evaluation** divides by the number of expert evaluations
  actually spent, so a search that reaches the same frontier by brute force
  scores worse than one that reaches it in a tenth of the calls. That is the
  whole point of the metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RoundRecord:
    """What one iteration of the loop achieved."""

    index: int
    proposed: int
    rejected_by_filters: int
    newly_evaluated: int
    #: Cumulative count of distinct candidates evaluated up to and including this round.
    total_evaluated: int
    feasible: int
    hypervolume: float
    frontier_size: int
    mean_structural_distance: float | None
    best_scalar: float | None
    #: New candidates contributed per generation strategy.
    contributions: dict[str, int] = field(default_factory=dict)
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    #: True when that hypervolume is zero because an objective is pinned at
    #: zero desirability across the whole frontier, rather than because the
    #: frontier stopped improving. The two are indistinguishable in the number,
    #: and the stopping rule reads the number.
    hypervolume_collapsed: bool = False
    #: The pinned objectives, for the report.
    pinned_axes: tuple[str, ...] = ()

    def describe(self) -> str:
        diversity = (
            "n/a" if self.mean_structural_distance is None
            else f"{self.mean_structural_distance:.3f}"
        )
        best = "n/a" if self.best_scalar is None else f"{self.best_scalar:.3f}"
        return (
            f"round {self.index:>2}  proposed {self.proposed:>4}  "
            f"filtered {self.rejected_by_filters:>4}  evaluated {self.total_evaluated:>4}  "
            f"feasible {self.feasible:>4}  "
            f"hypervolume {self.hypervolume:.4f}{'*' if self.hypervolume_collapsed else ''}  "
            f"frontier {self.frontier_size:>3}  diversity {diversity}  best {best}"
        )


@dataclass
class SearchMetrics:
    """Aggregate measures over a whole iterative run."""

    rounds: list[RoundRecord] = field(default_factory=list)
    stop_reason: str = ""
    #: Structures a benchmark says are good answers, when one was supplied.
    benchmark_size: int = 0
    top_k_recall: float | None = None
    top_k: int = 0

    # -- section 12 measures ----------------------------------------------

    @property
    def total_proposed(self) -> int:
        return sum(r.proposed for r in self.rounds)

    @property
    def total_evaluated(self) -> int:
        return self.rounds[-1].total_evaluated if self.rounds else 0

    @property
    def total_feasible(self) -> int:
        return self.rounds[-1].feasible if self.rounds else 0

    @property
    def success_rate(self) -> float | None:
        """Fraction of *generated* candidates that satisfy every hard constraint.

        Section 12 calls this the inverse-design success rate. Computing it over
        generated rather than surviving candidates is what makes it a statement
        about the generator instead of about the filters.
        """
        if not self.total_proposed:
            return None
        return self.total_feasible / self.total_proposed

    @property
    def hypervolume_history(self) -> list[float]:
        return [r.hypervolume for r in self.rounds]

    @property
    def hypervolume_gain(self) -> float:
        if len(self.rounds) < 2:
            return 0.0
        return self.rounds[-1].hypervolume - self.rounds[0].hypervolume

    @property
    def hypervolume_per_evaluation(self) -> float | None:
        """Frontier improvement bought per expert evaluation spent.

        None rather than zero when the measure has collapsed: a search whose
        hypervolume is pinned at zero by one objective bought no *measurable*
        improvement, which is not the same claim as buying none, and reporting
        0.0 makes it look like the second.
        """
        if self.hypervolume_collapsed:
            return None
        spent = self.total_evaluated
        return self.hypervolume_gain / spent if spent else None

    @property
    def hypervolume_collapsed(self) -> bool:
        """Every round's hypervolume was zeroed by a pinned objective."""
        return bool(self.rounds) and all(r.hypervolume_collapsed for r in self.rounds)

    @property
    def pinned_axes(self) -> tuple[str, ...]:
        return self.rounds[-1].pinned_axes if self.rounds else ()

    def evaluations_to_reach(self, fraction: float) -> int | None:
        """Evaluations needed to first reach ``fraction`` of the final hypervolume.

        Sample efficiency in the sense section 10 asks for: two searches that
        end at the same frontier are not equally good if one got there in a
        quarter of the calls.
        """
        if not self.rounds:
            return None
        target = self.rounds[-1].hypervolume * fraction
        for record in self.rounds:
            if record.hypervolume >= target:
                return record.total_evaluated
        return None

    @property
    def diversity_history(self) -> list[float | None]:
        return [r.mean_structural_distance for r in self.rounds]

    @property
    def contributions(self) -> dict[str, int]:
        """New candidates contributed per generation strategy across the run."""
        totals: dict[str, int] = {}
        for record in self.rounds:
            for strategy, count in record.contributions.items():
                totals[strategy] = totals.get(strategy, 0) + count
        return dict(sorted(totals.items(), key=lambda kv: -kv[1]))

    @property
    def failure_modes(self) -> dict[str, int]:
        """Why proposals were discarded, aggregated over the run.

        Section 10 asks Phase 2 to measure failure modes; the filters already
        record a reason for every rejection, so this is that record summed.
        """
        totals: dict[str, int] = {}
        for record in self.rounds:
            for reason, count in record.rejection_reasons.items():
                totals[reason] = totals.get(reason, 0) + count
        return dict(sorted(totals.items(), key=lambda kv: -kv[1]))

    # -- reporting ---------------------------------------------------------

    def describe(self) -> str:
        if not self.rounds:
            return "No search rounds were run."

        lines = ["Search progress"]
        lines.extend(f"  {r.describe()}" for r in self.rounds)
        lines.append("")
        lines.append(f"Stopped because: {self.stop_reason}")

        success = self.success_rate
        if success is not None:
            lines.append(
                f"Inverse-design success rate: {success:.1%} "
                f"({self.total_feasible} feasible of {self.total_proposed} generated)"
            )
        lines.append(
            f"Hypervolume: {self.rounds[0].hypervolume:.4f} -> "
            f"{self.rounds[-1].hypervolume:.4f} (gain {self.hypervolume_gain:+.4f})"
        )
        if self.hypervolume_collapsed:
            lines.append(
                "  That zero is a collapsed measure, not an unimproved one: no "
                "candidate on the frontier scores above zero desirability on "
                + ", ".join(self.pinned_axes)
                + ", and hypervolume is a product of edge lengths, so one such axis "
                "zeroes it whatever the others do. Read the frontier size and the "
                "best baseline score instead, and widen or drop that requirement if "
                "progress on it is what you wanted to see."
            )
        per_evaluation = self.hypervolume_per_evaluation
        if per_evaluation is not None:
            lines.append(
                f"Hypervolume gained per evaluation: {per_evaluation:.2e} "
                f"over {self.total_evaluated} evaluations"
            )
        for fraction in (0.9, 0.99):
            reached = self.evaluations_to_reach(fraction)
            if reached is not None:
                lines.append(
                    f"Reached {fraction:.0%} of the final hypervolume after "
                    f"{reached} evaluations"
                )
        final_diversity = self.rounds[-1].mean_structural_distance
        if final_diversity is not None:
            lines.append(f"Final mean structural distance: {final_diversity:.3f}")

        if self.top_k_recall is not None:
            lines.append(
                f"Top-{self.top_k} recall against the {self.benchmark_size}-structure "
                f"benchmark: {self.top_k_recall:.0%}"
            )

        contributions = self.contributions
        if contributions:
            lines.append("")
            lines.append("Candidates contributed per strategy:")
            lines.extend(f"  {strategy}: {count}" for strategy, count in contributions.items())

        failures = self.failure_modes
        if failures:
            lines.append("")
            lines.append("Proposals discarded, by reason:")
            lines.extend(f"  {count} x {reason}" for reason, count in failures.items())
        return "\n".join(lines)


def top_k_recall(
    ranked_smiles: list[str], benchmark_smiles: list[str], k: int
) -> float | None:
    """Fraction of known-good structures that appear in the top ``k``.

    Section 12: "Top-k recall against benchmark candidate sets when a known
    solution exists." Comparison is on canonical structure, so a benchmark
    written in a different but equivalent SMILES spelling still matches.
    """
    if not benchmark_smiles:
        return None

    from formulate import chem

    def canonical(values: list[str]) -> set[str]:
        out: set[str] = set()
        for value in values:
            canon = chem.canonical_smiles(value) if chem.rdkit_available() else value
            if canon:
                out.add(canon)
        return out

    wanted = canonical(benchmark_smiles)
    if not wanted:
        return None
    found = canonical(ranked_smiles[:k])
    return len(wanted & found) / len(wanted)
