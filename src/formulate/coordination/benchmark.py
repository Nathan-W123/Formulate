"""Does the adaptive coordinator earn its complexity?

Specification section 10, Phase 5: "compare fixed pipeline against
uncertainty/expected-value-driven compute allocation.  The coordinator earns
complexity only if it improves quality per unit compute."

That sentence is the deliverable, not the coordinator.  An adaptive policy
that has not been measured against the baseline has not completed the phase,
and a comparison that can only confirm is not a comparison.  So this module is
built to be able to return a negative result, and says so plainly when it does.

Three things decide whether the answer means anything.

**The currency.** The arms spend compute on different things: the adaptive one
may buy a quantum calculation where the fixed one buys candidates.  Equalising
the number of expert evaluations would hand the adaptive arm free physics.  The
budget is therefore wall-clock seconds, which is the only currency both arms
spend from the same purse.

**The axes.** Hypervolume is comparable between runs only when the utility
axes are fixed in advance.  A requirement with no absolute anchors takes its
scale from the pool it happens to see, so two arms with different pools would
be measured with different rulers.  Such targets are refused rather than
silently mismeasured.

**The pairing.** A single run proves nothing; both arms run on the same seed
and the same target, and the comparison is over the paired differences.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Sequence

from formulate.targets.spec import TargetSpec

from .adaptive import AdaptiveConfig, default_adaptive_coordinator, objective_coverage
from .coordinator import RunConfig
from .iterative import IterationConfig, default_validating_coordinator
from .validation import ValidationPolicy


class IncomparableTarget(ValueError):
    """The target cannot be benchmarked without changing what is measured."""


@dataclass(frozen=True, slots=True)
class ArmResult:
    """What one coordinator achieved on one seed."""

    arm: str
    seed: int
    seconds: float
    hypervolume: float
    feasible: int
    evaluated: int
    frontier: int
    coverage: float

    def describe(self) -> str:
        return (
            f"{self.arm:8s} seed {self.seed}  {self.seconds:6.1f} s  "
            f"hypervolume {self.hypervolume:.4f}  feasible {self.feasible:3d}  "
            f"evaluated {self.evaluated:3d}  coverage {self.coverage:.0%}"
        )


@dataclass
class BenchmarkResult:
    """The paired comparison and what it licenses."""

    target: str
    pairs: list[tuple[ArmResult, ArmResult]] = field(default_factory=list)

    @property
    def differences(self) -> list[float]:
        """Adaptive minus fixed, in hypervolume, one per seed."""
        return [adaptive.hypervolume - fixed.hypervolume for fixed, adaptive in self.pairs]

    @property
    def mean_difference(self) -> float:
        diffs = self.differences
        return statistics.fmean(diffs) if diffs else 0.0

    @property
    def wins(self) -> int:
        return sum(1 for d in self.differences if d > 1e-9)

    @property
    def losses(self) -> int:
        return sum(1 for d in self.differences if d < -1e-9)

    @property
    def ties(self) -> int:
        return len(self.differences) - self.wins - self.losses

    def sign_test_p(self) -> float | None:
        """Two-sided exact sign test over the non-tied pairs.

        A sign test rather than a t-test: with a handful of paired runs there
        is no basis for assuming normal differences, and the sign test needs
        only that a win and a loss are equally likely under the null. It is
        weak, which is the honest position at this sample size.
        """
        decisive = self.wins + self.losses
        if decisive == 0:
            return None
        from math import comb

        better = max(self.wins, self.losses)
        tail = sum(comb(decisive, k) for k in range(better, decisive + 1))
        return min(1.0, 2.0 * tail / (2**decisive))

    @property
    def verdict(self) -> str:
        """The adoption decision, stated as a rule rather than an impression.

        Adaptive is adopted only if it wins on the mean paired difference AND
        the sign test does not leave that mean indistinguishable from chance.
        Anything else leaves the fixed pipeline in place, which section 10
        treats as the expected outcome rather than a failure.
        """
        if not self.pairs:
            return "no paired runs were completed, so nothing is decided"
        p = self.sign_test_p()
        if self.mean_difference <= 0:
            return (
                f"KEEP THE FIXED PIPELINE: adaptive was worse on average "
                f"({self.mean_difference:+.4f} hypervolume over {len(self.pairs)} paired runs)"
            )
        if p is None or p > 0.10:
            return (
                f"KEEP THE FIXED PIPELINE: adaptive led by {self.mean_difference:+.4f} "
                f"hypervolume but won only {self.wins} of {len(self.pairs)} pairs"
                + (f", sign test p = {p:.2f}" if p is not None else "")
                + ", which is not distinguishable from chance at this sample size"
            )
        return (
            f"ADOPT THE ADAPTIVE COORDINATOR: it led by {self.mean_difference:+.4f} "
            f"hypervolume, winning {self.wins} of {len(self.pairs)} pairs "
            f"(sign test p = {p:.2f})"
        )

    def describe(self) -> str:
        lines = [f"Benchmark on target: {self.target}", ""]
        for fixed, adaptive in self.pairs:
            lines.append(f"  {fixed.describe()}")
            lines.append(f"  {adaptive.describe()}")
            lines.append(
                f"           difference {adaptive.hypervolume - fixed.hypervolume:+.4f}"
            )
            lines.append("")
        lines.append(
            f"Paired runs: {len(self.pairs)}   adaptive wins {self.wins}, "
            f"loses {self.losses}, ties {self.ties}"
        )
        lines.append(f"Mean paired difference: {self.mean_difference:+.4f} hypervolume")
        p = self.sign_test_p()
        if p is not None:
            lines.append(f"Exact two-sided sign test: p = {p:.3f}")
        lines.append("")
        lines.append(self.verdict)
        return "\n".join(lines)


def check_comparable(spec: TargetSpec) -> None:
    """Refuse a target whose utility axes are not fixed in advance.

    A requirement without absolute anchors takes its 0 and 1 from whichever
    pool it happens to see, so the two arms would be measured with different
    rulers and the difference between their hypervolumes would be an artefact
    of their pools rather than of their policies.
    """
    floating = [r.property for r in spec.requirements if r.needs_pool_anchors]
    if floating:
        raise IncomparableTarget(
            "these requirements take their scale from the candidate pool, so hypervolume "
            "is not comparable between two runs that saw different pools: "
            + ", ".join(floating)
            + ". Give them explicit lower and upper anchors to benchmark on this target."
        )


def run_benchmark(
    spec: TargetSpec,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    budget_seconds: float = 60.0,
    pool_size: int = 20,
    batch_size: int = 12,
    validation_candidates: int = 1,
) -> BenchmarkResult:
    """Run both coordinators on each seed and compare them pairwise.

    Each arm gets its own cache. Sharing one would let whichever arm ran second
    free-ride on the first's evaluations, which would measure the order the
    arms happened to run in rather than the policies.
    """
    check_comparable(spec)
    result = BenchmarkResult(target=spec.name)

    for seed in seeds:
        fixed = _run_fixed(spec, seed, budget_seconds, pool_size, batch_size, validation_candidates)
        adaptive = _run_adaptive(
            spec, seed, budget_seconds, pool_size, batch_size, validation_candidates
        )
        result.pairs.append((fixed, adaptive))
    return result


def _run_fixed(
    spec: TargetSpec,
    seed: int,
    budget: float,
    pool_size: int,
    batch: int,
    validations: int,
) -> ArmResult:
    """The Phase 2/3 pipeline: iterate a fixed number of rounds, then validate."""
    coordinator = default_validating_coordinator(
        RunConfig(pool_size=pool_size, seed=seed),
        IterationConfig(max_rounds=3, batch_size=batch),
        ValidationPolicy(max_candidates=validations, max_seconds=budget * 0.6),
    )
    started = time.perf_counter()
    run = coordinator.run_validated(spec)
    elapsed = time.perf_counter() - started
    coverage, _ = objective_coverage(run.final, spec)
    return ArmResult(
        arm="fixed",
        seed=seed,
        seconds=elapsed,
        hypervolume=run.final.ranking.hypervolume,
        feasible=run.final.ranking.feasible_count,
        evaluated=run.final.evaluated,
        frontier=len(run.final.ranking.frontier),
        coverage=coverage,
    )


def _run_adaptive(
    spec: TargetSpec,
    seed: int,
    budget: float,
    pool_size: int,
    batch: int,
    validations: int,
) -> ArmResult:
    coordinator = default_adaptive_coordinator(
        RunConfig(pool_size=pool_size, seed=seed),
        AdaptiveConfig(
            budget_seconds=budget,
            batch_size=batch,
            max_actions=25,
            validation=ValidationPolicy(max_candidates=validations, max_seconds=budget * 0.6),
        ),
    )
    started = time.perf_counter()
    run = coordinator.run_adaptive(spec)
    elapsed = time.perf_counter() - started
    coverage, _ = objective_coverage(run.final, spec)
    return ArmResult(
        arm="adaptive",
        seed=seed,
        seconds=elapsed,
        hypervolume=run.final.ranking.hypervolume,
        feasible=run.final.ranking.feasible_count,
        evaluated=run.final.evaluated,
        frontier=len(run.final.ranking.frontier),
        coverage=coverage,
    )
