"""End-to-end inverse design validation: hide a known material and look for it.

Specification section 12 asks for top-k recall "against benchmark candidate
sets when a known solution exists", and section 10 for rankings verified
against held-out known materials. Every other benchmark in this repository
measures one part - an expert against a handbook value, a potential against a
solver. This one measures the system: describe a material only by what it does,
take it away, and see whether the pipeline comes back with it.

Two questions live here and conflating them flatters the result:

**Retrieval.** The answer is left in the catalogue. Where does the ranking put
it? This tests the experts and the ranking, and nothing else - the search never
had to invent anything.

**Discovery.** The answer is removed from the catalogue. Can the explorers
reach it, or anything that meets the same targets? This is the harder question
and the one the project is actually about, and a run may legitimately fail it
while succeeding at retrieval.

Both are reported. A harness that only ran the first would report recall of one
and mean it as evidence of inverse design.

A window has to be wider than the method that answers it
-------------------------------------------------------
Desirability is risk-adjusted: a prediction is scored at one standard
deviation in the unfavourable direction, so a value whose uncertainty carries
it outside the acceptance window earns zero credit even when the nominal value
sits comfortably inside. That is the right conservative behaviour and it has a
sharp consequence for this harness.

Crippen's logp for ethanol is -0.0014 against a measured -0.31, which is well
inside a +/- 0.80 window: nominal desirability 0.393. Crippen's own error is
about 0.8, so the pessimistic value is 0.80, outside the window, and the
risk-adjusted desirability is exactly 0.000. Every candidate scores zero on
that axis, the frontier's logp coordinate is pinned at zero, and hypervolume -
a product of edge lengths from the origin - is zero however good the frontier
is on the other four objectives. The iterative coordinator stops on
"hypervolume gained nothing", so the search also stopped early and reported
convergence that had not happened.

So a recovery window narrower than roughly two standard deviations of the
answering method produces a target nothing can score. The ranking now names
such axes (``RankingResult.pinned_axes``), the stopping rule abstains rather
than reading the collapse as a plateau, and :class:`RecoveryResult` carries the
pinned axes so a zero in the output explains itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Sequence

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.conditions import Conditions
from formulate.core.quantity import Quantity
from formulate.targets.spec import Direction, Requirement, TargetSpec

#: Reference fields that can be turned into a requirement, with the unit they
#: are tabulated in and the property they describe. This is deliberately a
#: subset of the calibration set's mapping: a requirement the expert panel
#: cannot answer produces a spec nothing can satisfy, which measures coverage
#: rather than inverse design.
DESCRIBABLE = {
    "boiling_point_c": ("normal_boiling_point", "degC"),
    "melting_point_c": ("melting_point", "degC"),
    "density_25c": ("liquid_density", "g/cm^3"),
    "surface_tension_25c": ("surface_tension", "mN/m"),
    "logp": ("logp", ""),
    "critical_temperature_k": ("critical_temperature", "K"),
}

#: Half-width of the acceptance window, as a fraction of the property's own
#: observed spread across the reference set rather than of the hidden value.
#:
#: Scaling to the value is the obvious rule and it is wrong for any property
#: whose values straddle zero. Ethanol's logp is -0.31, so a ten per cent
#: relative window is +/- 0.031 - an order of magnitude tighter than Crippen's
#: own error, so every candidate scored exactly zero desirability on it, the
#: frontier's logp axis was pinned at zero, and the hypervolume that the
#: iterative search stops on collapsed to zero in every round of every run.
#: A window scaled to the spread means the same thing on logp as on a boiling
#: point: within a tenth of the range this property actually takes.
DEFAULT_TOLERANCE = 0.10


@dataclass(frozen=True, slots=True)
class HiddenMaterial:
    """A material described only by what it does."""

    name: str
    smiles: str
    #: Property -> (value, unit) taken from its measurements.
    properties: dict[str, tuple[float, str]]

    def describe(self) -> str:
        wants = ", ".join(
            f"{prop} {value:.4g} {unit}".rstrip()
            for prop, (value, unit) in self.properties.items()
        )
        return f"{self.name}: {wants}"


@dataclass
class RecoveryResult:
    """What one hidden-material experiment found."""

    hidden: HiddenMaterial
    mode: str
    spec: TargetSpec
    #: 1-based position of the hidden material, or None if it never appeared.
    rank: int | None = None
    pool_size: int = 0
    top_k: int = 10
    #: Objectives on which no frontier candidate scored above zero, which
    #: zeroes the hypervolume whatever the others did. Reported so the number
    #: below explains itself rather than reading as a search that achieved
    #: nothing.
    pinned_axes: tuple[str, ...] = ()
    #: Section 12 measures, carried through from the search.
    top_k_recall: float | None = None
    success_rate: float | None = None
    hypervolume_gain: float = 0.0
    diversity: float | None = None
    evaluations: int = 0
    physics_calls: int = 0
    seconds: float = 0.0
    #: Candidates the run put above the hidden material, best first.
    ranked_above: tuple[str, ...] = ()
    #: The explorer that produced the hidden structure, when it was found. A
    #: discovery arm that recovered the answer from the catalogue it was meant
    #: to be hidden from would be a leak rather than a result, so where it came
    #: from is recorded rather than assumed.
    found_by: str = ""
    notes: tuple[str, ...] = ()

    @property
    def recovered(self) -> bool:
        return self.rank is not None and self.rank <= self.top_k

    def describe(self) -> str:
        lines = [f"{self.hidden.name} ({self.mode})"]
        if self.rank is None:
            lines.append(
                f"    NOT FOUND in a pool of {self.pool_size}"
                + (" - the search never proposed it" if self.mode == "discovery" else "")
            )
        else:
            verdict = "recovered" if self.recovered else "found but outside the top k"
            lines.append(
                f"    rank {self.rank} of {self.pool_size} -> {verdict} (k = {self.top_k})"
            )
        if self.found_by:
            lines.append(f"    produced by: {self.found_by}")
        if self.ranked_above:
            lines.append("    ranked above it: " + ", ".join(self.ranked_above[:5]))
        if self.success_rate is not None:
            lines.append(
                f"    inverse-design success rate {self.success_rate:.0%}   "
                f"hypervolume gain {self.hypervolume_gain:+.4g}"
            )
        if self.pinned_axes:
            lines.append(
                "    that hypervolume is collapsed, not flat: nothing scores above zero "
                "risk-adjusted desirability on "
                + ", ".join(self.pinned_axes)
                + " (the window is narrower than the method answering it)"
            )
        if self.diversity is not None:
            lines.append(f"    frontier diversity {self.diversity:.3f}")
        lines.append(
            f"    {self.evaluations} evaluation(s), {self.physics_calls} physics call(s), "
            f"{self.seconds:.1f} s"
        )
        lines.extend(f"    {note}" for note in self.notes)
        return "\n".join(lines)


def describe_material(
    record: dict[str, Any], properties: Sequence[str] = (), limit: int = 5
) -> HiddenMaterial | None:
    """Turn a reference record into a behavioural description of itself.

    ``properties`` names reference fields to use; empty takes whatever the
    record has, up to ``limit``. Returns None when fewer than three usable
    values are present, because a material described by two numbers is not
    identified by them - dozens of compounds share any two.
    """
    wanted = list(properties) if properties else list(DESCRIBABLE)
    found: dict[str, tuple[float, str]] = {}
    for field_name in wanted:
        entry = DESCRIBABLE.get(field_name)
        value = record.get(field_name)
        if entry is None or value is None:
            continue
        found[entry[0]] = (float(value), entry[1])
        if len(found) >= limit:
            break
    if len(found) < 3:
        return None
    return HiddenMaterial(
        name=record.get("name", record["smiles"]), smiles=record["smiles"], properties=found
    )


def spec_for(
    hidden: HiddenMaterial,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    conditions: Conditions | None = None,
    material_classes: Sequence[MaterialClass] = (MaterialClass.MOLECULE,),
) -> TargetSpec:
    """A target specification that describes the hidden material and nothing else.

    Every requirement is a soft ``target`` with a window around the measured
    value. Making them hard would turn the experiment into a filter test: one
    property the panel estimates badly would eliminate the right answer outright
    and the run would report a recall of zero for a reason that has nothing to
    do with search.
    """
    requirements = []
    for prop, (value, unit) in hidden.properties.items():
        # The window is a fraction of the magnitude in the unit the value is
        # tabulated in. For a Celsius temperature that is the wrong thing to do
        # - ten per cent of 0 degrees is zero - so temperatures widen around
        # the absolute scale and are converted back.
        width = _window(prop, value, unit, tolerance)
        requirements.append(
            Requirement(
                property=prop,
                direction=Direction.TARGET,
                target=Quantity(value=value, unit=unit),
                lower=Quantity(value=value - width, unit=unit),
                upper=Quantity(value=value + width, unit=unit),
                # Demand closeness before granting credit: with shape 1 a
                # candidate at the edge of the window scores the same as one
                # halfway to it, and the ranking stops discriminating.
                shape=2.0,
                rationale=f"the hidden material's measured {prop}",
            )
        )
    return TargetSpec(
        name=f"recover {hidden.name} from its behaviour",
        requirements=tuple(requirements),
        conditions=conditions or Conditions.standard(),
        material_classes=tuple(material_classes),
        notes=(
            "Generated by the recovery harness: every requirement is one measured "
            "property of a material withheld from the search."
        ),
    )


def _window(prop: str, value: float, unit: str, tolerance: float) -> float:
    """Half-width of the acceptance window, in the value's own unit.

    ``tolerance`` multiplies the property's observed spread across the
    reference set. Where that spread is unknown - a property nothing in the set
    tabulates - it falls back to the magnitude of the value, and to the bare
    tolerance when the value itself is zero.
    """
    spread = _reference_spread().get(prop)
    if spread:
        return spread * tolerance
    if not value:
        return tolerance
    return abs(value) * tolerance


@lru_cache(maxsize=1)
def _reference_spread() -> dict[str, float]:
    """Observed range of each describable property, in the unit it is asked in.

    Computed once from the bundled reference set. It is a property of the
    catalogue rather than a constant, so it lives here rather than in a table
    that would drift the first time a compound is added.
    """
    from formulate.exploration.database import load_reference_compounds

    records = load_reference_compounds()
    spread: dict[str, float] = {}
    for field_name, entry in DESCRIBABLE.items():
        values = [
            float(record[field_name])
            for record in records
            if record.get(field_name) is not None
        ]
        if len(values) > 1:
            spread[entry[0]] = max(values) - min(values)
    return spread


def run_recovery(
    hidden: HiddenMaterial,
    *,
    mode: str = "discovery",
    coordinator=None,
    spec: TargetSpec | None = None,
    top_k: int = 10,
    rounds: int = 3,
    batch_size: int = 25,
    seed: int = 0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> RecoveryResult:
    """Run the pipeline against a behavioural description and look for the answer.

    ``mode`` is ``"discovery"`` (the answer is withheld from the catalogue) or
    ``"retrieval"`` (it is left in). The coordinator is built here when none is
    given, because the withholding has to reach the explorer that would
    otherwise hand the answer over.
    """
    from formulate.coordination.coordinator import RunConfig
    from formulate.coordination.iterative import (
        IterationConfig,
        default_iterative_coordinator,
    )

    if mode not in ("discovery", "retrieval"):
        raise ValueError(f"mode must be 'discovery' or 'retrieval', got {mode!r}")

    spec = spec or spec_for(hidden, tolerance=tolerance)
    if coordinator is None:
        coordinator = default_iterative_coordinator(
            config=RunConfig(seed=seed),
            iteration=IterationConfig(
                max_rounds=rounds,
                batch_size=batch_size,
                benchmark_smiles=(hidden.smiles,),
                top_k=top_k,
            ),
            exclude=(hidden.smiles,) if mode == "discovery" else (),
        )

    started = time.perf_counter()
    outcome = coordinator.run_iterative(spec)
    seconds = time.perf_counter() - started

    # A validating coordinator wraps the search in a ValidatedRun; an iterative
    # one returns the search itself. Both are accepted so a caller can measure
    # the same experiment with and without the physics stage.
    search = getattr(outcome, "search", outcome)
    ranked = list(search.final.ranking.ranked)
    rank, above, found_by = _locate(hidden.smiles, ranked)
    metrics = search.metrics

    return RecoveryResult(
        hidden=hidden,
        mode=mode,
        spec=spec,
        rank=rank,
        pool_size=len(ranked),
        top_k=top_k,
        top_k_recall=metrics.top_k_recall,
        success_rate=metrics.success_rate,
        hypervolume_gain=metrics.hypervolume_gain,
        pinned_axes=search.final.ranking.pinned_axes,
        diversity=_diversity(search.final),
        evaluations=metrics.total_evaluated,
        physics_calls=_physics_calls(outcome),
        seconds=seconds,
        ranked_above=above,
        found_by=found_by,
    )


def _locate(
    smiles: str, ranked: Sequence[Any]
) -> tuple[int | None, tuple[str, ...], str]:
    """Where the hidden material landed, what beat it, and what produced it.

    Matched on canonical structure. Matching on the string would miss the
    answer whenever an explorer rebuilt it with a different but equivalent
    spelling, and report a discovery failure that was a string comparison.
    """
    from formulate import chem

    def canonical(value: str | None) -> str:
        if not value:
            return ""
        if not chem.rdkit_available():
            return value
        return chem.canonical_smiles(value) or value

    wanted = canonical(smiles)
    above: list[str] = []
    for index, entry in enumerate(ranked, start=1):
        candidate: Candidate = entry.candidate
        if canonical(candidate.primary_smiles) == wanted:
            return index, tuple(above), candidate.generation_strategy
        above.append(candidate.label or candidate.primary_smiles or candidate.candidate_id[:12])
    return None, tuple(above), ""


def _diversity(run) -> float | None:
    """Mean pairwise structural distance across the ranked pool."""
    return run.ranking.mean_diversity


def _physics_calls(run) -> int:
    report = getattr(run, "validation", None)
    return getattr(report, "calls_made", 0) if report is not None else 0


@dataclass
class RecoverySuite:
    """Several hidden-material experiments and what they say together."""

    results: list[RecoveryResult] = field(default_factory=list)

    @property
    def recovery_rate(self) -> float | None:
        if not self.results:
            return None
        return sum(1 for r in self.results if r.recovered) / len(self.results)

    def by_mode(self, mode: str) -> list[RecoveryResult]:
        return [r for r in self.results if r.mode == mode]

    def describe(self) -> str:
        lines = [
            "End-to-end inverse design: known materials described by behaviour alone.",
            "",
        ]
        for mode in ("retrieval", "discovery"):
            group = self.by_mode(mode)
            if not group:
                continue
            recovered = sum(1 for r in group if r.recovered)
            lines.append(
                f"{mode.upper()}: {recovered} of {len(group)} recovered into the top k"
            )
            lines.extend(result.describe() for result in group)
            lines.append("")
        return "\n".join(lines).rstrip()
