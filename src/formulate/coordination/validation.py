"""Selective physics validation and re-ranking.

Specification section 9, stage 8: "Run QM/MD only on a small high-value/
uncertain subset using property-specific protocols", and stage 9: "Replace/
augment predictions with validated results; quantify disagreement and
confidence."

Three decisions carry the weight.

**Who gets validated.** Section 5: "Escalate candidates with high utility but
high uncertainty; avoid expensive validation of clearly dominated candidates."
Validating the best candidate is usually wasted budget - if it is already
clearly ahead, confirming it does not change the ranking. Budget belongs where
uncertainty could still reorder the list.

**What each method may be asked.** This is a hard scientific constraint, not a
routing preference. Quantum chemistry cannot produce a bulk density; molecular
dynamics cannot produce an orbital gap; neither can produce an odour. A request
outside the mapping is refused rather than approximated.

**How a validated value merges.** Physics does not automatically win. A
converged calculation at a level of theory with a known systematic error is not
obviously better than a correlation fitted to measurements of the very quantity
in question, so the two are combined by their stated uncertainties and the
disagreement is reported either way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from formulate.core.candidate import Candidate, CandidateResults
from formulate.core.conditions import Conditions
from formulate.core.prediction import Prediction, PredictionStatus
from formulate.core.properties import get_property
from formulate.core.provenance import ProvenanceKind, ProvenanceRecord
from formulate.core.quantity import ApplicabilityDomain, Quantity, Uncertainty, UncertaintyKind
from formulate.targets.spec import TargetSpec


#: Where an isolated-molecule quantum calculation actually applies. Every
#: result from the quantum backends is stamped with this rather than with the
#: requested conditions.
VACUUM_ZERO_KELVIN = Conditions(
    temperature=Quantity(value=0.0, unit="K"), environment="vacuum"
)


class ValidationMethod(str, Enum):
    QUANTUM = "quantum"
    DYNAMICS = "dynamics"


#: Which method may legitimately validate which property.
#:
#: A property absent from this mapping cannot be validated by physics at all in
#: this system, and asking is an error rather than a request to approximate.
#: Section 13: "Do not claim MD + QM derive every material property. Sensory/
#: biological, synthesis, manufacturing, aging, and many macroscale behaviors
#: require data-driven or higher-scale models/experiments."
VALIDATABLE: dict[str, frozenset[ValidationMethod]] = {
    "homo_lumo_gap": frozenset({ValidationMethod.QUANTUM}),
    "dipole_moment": frozenset({ValidationMethod.QUANTUM}),
    "interaction_energy": frozenset({ValidationMethod.QUANTUM}),
    "atomization_energy": frozenset({ValidationMethod.QUANTUM}),
    "radius_of_gyration": frozenset({ValidationMethod.DYNAMICS}),
    "self_diffusion_coefficient": frozenset({ValidationMethod.DYNAMICS}),
    "liquid_density": frozenset({ValidationMethod.DYNAMICS}),
    "cohesive_energy_density": frozenset({ValidationMethod.DYNAMICS}),
    "work_of_separation": frozenset({ValidationMethod.DYNAMICS}),
}

#: Properties computed from a periodic condensed phase rather than from a
#: single-molecule or cluster trajectory.
#:
#: These became possible when MMFF94 was made available as an OpenMM system.
#: They are expensive - a density is tens of minutes of wall clock per
#: candidate against seconds for a quantum observable - so the validation
#: budget refuses them unless it has been raised to suit, and says how long
#: they would need rather than silently skipping.
CONDENSED_PROTOCOLS: dict[str, str] = {
    "liquid_density": "density",
    "cohesive_energy_density": "cohesive_energy_density",
    "self_diffusion_coefficient": "self_diffusion",
}

#: Rough wall-clock cost of each condensed protocol on this installation, in
#: seconds, measured rather than guessed: MMFF94 through OpenMM's custom forces
#: runs a two-thousand-atom box at roughly twelve nanoseconds per day.
CONDENSED_COST_SECONDS: dict[str, float] = {
    "density": 3600.0,
    "cohesive_energy_density": 4200.0,
    "self_diffusion": 5400.0,
}

#: Which molecular-dynamics workflow produces which property.
#:
#: Every entry of VALIDATABLE marked DYNAMICS must appear here. Selecting a
#: protocol with an if-else that ends in a default is what made this necessary:
#: every property except the radius of gyration fell through to the
#: cohesive-energy workflow, whose result is an energy per mole, and stamping
#: that as a density raised an uncaught dimensionality error that killed the
#: run. A property with no protocol is refused before any of that, the way the
#: quantum path already refuses a property with no observable.
#:
#: Only two properties are left on this route. Density and self-diffusion moved
#: to CONDENSED_PROTOCOLS once MMFF94 could be run periodically; a property must
#: appear in exactly one of the two tables, and a test asserts it. Work of
#: separation stays here and declines, because an interface needs two slabs and
#: four hundred molecules, which is beyond what this installation will pay for.
DYNAMICS_PROTOCOLS: dict[str, str] = {
    "radius_of_gyration": "conformational_ensemble",
    "work_of_separation": "work_of_separation",
}

#: Why a property that looks physical is nevertheless not validatable here.
NOT_VALIDATABLE_REASONS: dict[str, str] = {
    "electronic_energy": (
        "an absolute electronic energy is monotone in the number of electrons "
        "(-75 Hartree for water, -152 for ethanol, -228 for benzene at HF/STO-3G), so "
        "ranking candidates on it would order them by size rather than by merit. Only "
        "differences between consistent calculations of the same system carry meaning, "
        "and a cross-candidate objective is not such a difference"
    ),
    "normal_boiling_point": (
        "a boiling point is a phase-equilibrium property requiring free energies of "
        "two coexisting phases, which neither a single-molecule calculation nor a "
        "short classical trajectory provides"
    ),
    "melting_point": (
        "a melting point requires the free-energy difference between a crystal and a "
        "liquid, including the crystal structure, which is not available here"
    ),
    "logp": (
        "a partition coefficient is a solvation free-energy difference between two "
        "solvents; obtaining it from simulation needs free-energy perturbation in "
        "both phases, far beyond the sampling reachable here"
    ),
    "aqueous_solubility_logs": (
        "solubility couples solvation free energy to the solid-state lattice energy, "
        "so it is not obtainable from a gas-phase or single-phase calculation"
    ),
    "synthetic_accessibility": (
        "synthesisability is a statement about available routes and reagents, not a "
        "physical observable; section 13 places it outside what QM or MD can establish"
    ),
    "shear_viscosity": (
        "a viscosity comes from a Green-Kubo integral of the stress autocorrelation "
        "or from non-equilibrium shear, and neither exists in this system; there is "
        "no dynamics workflow that produces it, adequate sampling or not"
    ),
    "surface_tension": (
        "surface tension requires a converged liquid-vapour interface, which needs a "
        "periodic condensed phase larger than the available potentials support"
    ),
}


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    """How much physics to buy, and for whom."""

    #: Candidates to validate. Section 9 stage 8 says "a small subset".
    max_candidates: int = 3
    #: Wall-clock ceiling for the whole validation stage.
    max_seconds: float = 600.0
    #: Restrict to candidates on or near the frontier. A clearly dominated
    #: candidate cannot be promoted into contention by one confirmed property.
    max_front: int = 1
    #: Molecules in a periodic condensed-phase box. Two hundred and fifty is
    #: above the minimum-image requirement for every liquid in the reference
    #: set and below the point where the cost stops being worth the accuracy.
    condensed_molecules: int = 250
    #: Starting density for a constant-volume run, g/cm^3. Only self-diffusion
    #: needs it, because there the box is fixed rather than measured.
    assumed_density: float = 0.85
    #: Feasible candidates only: validating an infeasible one cannot change
    #: the recommendation.
    feasible_only: bool = True
    #: Standard deviations of disagreement that count as significant.
    disagreement_sigma: float = 2.0


@dataclass(frozen=True, slots=True)
class ValidationTarget:
    """One candidate-property pair worth spending physics on."""

    candidate: Candidate
    property: str
    method: ValidationMethod
    #: Expected influence on the ranking, used to order the queue.
    value_score: float
    rationale: str


@dataclass
class Disagreement:
    """How far a validated value sits from what the expert predicted."""

    property: str
    expert_id: str
    expert_value: float
    expert_std: float | None
    physics_value: float
    physics_std: float | None
    unit: str

    @property
    def difference(self) -> float:
        return self.physics_value - self.expert_value

    @property
    def z_score(self) -> float | None:
        """Difference in units of the combined stated uncertainty.

        Undefined when neither side quoted an uncertainty: a difference is only
        surprising relative to how well the two claimed to know the answer.
        """
        variance = 0.0
        for std in (self.expert_std, self.physics_std):
            if std is not None:
                variance += std**2
        if variance <= 0:
            return None
        return self.difference / math.sqrt(variance)

    def is_significant(self, sigma: float = 2.0) -> bool:
        z = self.z_score
        return z is not None and abs(z) >= sigma

    def describe(self, sigma: float = 2.0) -> str:
        z = self.z_score
        z_text = "z undefined (no uncertainty quoted)" if z is None else f"z = {z:+.1f}"
        verdict = (
            "significant: the two disagree by more than their stated uncertainties allow"
            if self.is_significant(sigma)
            else "consistent within the stated uncertainties"
        )
        return (
            f"{self.property}: expert ({self.expert_id}) {self.expert_value:.4g} vs "
            f"physics {self.physics_value:.4g} {self.unit}, {z_text} - {verdict}"
        )


@dataclass
class ValidationReport:
    """What the validation stage did and what it changed."""

    targets: list[ValidationTarget] = field(default_factory=list)
    validated: list[Prediction] = field(default_factory=list)
    disagreements: list[Disagreement] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    seconds_spent: float = 0.0
    calls_made: int = 0
    calls_avoided: int = 0
    #: Candidate id -> (rank before, rank after).
    rank_changes: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def disagreement_rate(self) -> float | None:
        """Section 12's "physics disagreement rate"."""
        comparable = [d for d in self.disagreements if d.z_score is not None]
        if not comparable:
            return None
        return sum(1 for d in comparable if d.is_significant()) / len(comparable)

    def describe(self) -> str:
        if not self.targets and not self.skipped:
            return "No candidate was selected for physics validation."

        lines = [
            f"Validated {self.calls_made} calculation(s) on {len(self.targets)} target(s) "
            f"in {self.seconds_spent:.1f} s; {self.calls_avoided} avoided by selection."
        ]
        for target in self.targets:
            name = target.candidate.label or target.candidate.primary_smiles
            lines.append(
                f"  {name} / {target.property} via {target.method.value}: {target.rationale}"
            )

        if self.disagreements:
            lines.append("")
            lines.append("Expert versus physics:")
            lines.extend(f"  {d.describe()}" for d in self.disagreements)
            rate = self.disagreement_rate
            if rate is not None:
                lines.append(f"  disagreement rate: {rate:.0%}")

        if self.rank_changes:
            lines.append("")
            lines.append("Ranking changes after validation:")
            for candidate_id, (before, after) in sorted(
                self.rank_changes.items(), key=lambda kv: kv[1][1]
            ):
                direction = "up" if after < before else "down"
                lines.append(f"  {candidate_id[:20]}: #{before} -> #{after} ({direction})")

        if self.skipped:
            lines.append("")
            lines.append("Not validated:")
            lines.extend(f"  {prop}: {reason}" for prop, reason in sorted(self.skipped.items()))
        return "\n".join(lines)


def validatable_properties(spec: TargetSpec) -> tuple[dict[str, ValidationMethod], dict[str, str]]:
    """Split the requested properties into what physics may check and what it may not."""
    permitted: dict[str, ValidationMethod] = {}
    refused: dict[str, str] = {}
    for prop in spec.properties:
        methods = VALIDATABLE.get(prop)
        if methods:
            # A property with one permitted method takes it; the mapping does
            # not currently define a preference where both would apply.
            permitted[prop] = sorted(methods, key=lambda m: m.value)[0]
        else:
            refused[prop] = NOT_VALIDATABLE_REASONS.get(
                prop,
                "no quantum or dynamics protocol in this system produces this property",
            )
    return permitted, refused


def select_targets(
    ranked: Sequence[Candidate],
    spec: TargetSpec,
    policy: ValidationPolicy | None = None,
) -> tuple[list[ValidationTarget], dict[str, str]]:
    """Choose the small subset worth spending physics on.

    The value score multiplies three things: how uncertain the prediction is,
    how much the candidate is in contention, and how close it sits to a
    decision boundary. A confidently-predicted property on a clearly-leading
    candidate scores low, because confirming it changes nothing.
    """
    policy = policy or ValidationPolicy()
    permitted, refused = validatable_properties(spec)
    if not permitted:
        return [], refused

    targets: list[ValidationTarget] = []
    for candidate in ranked:
        results = candidate.results
        if results is None:
            continue
        if policy.feasible_only and not results.feasible:
            continue
        front = results.pareto_front if results.pareto_front is not None else 0
        if front > policy.max_front:
            continue

        for prop, method in permitted.items():
            prediction = results.prediction_for(prop)
            if prediction is None or prediction.quantity is None:
                # Nothing to check against, but physics can still supply the
                # value the experts could not - that is the highest-value case.
                targets.append(
                    ValidationTarget(
                        candidate=candidate,
                        property=prop,
                        method=method,
                        value_score=1.0,
                        rationale="no expert could supply this property, so physics adds a value rather than checking one",
                    )
                )
                continue

            score, rationale = _value_score(prediction, results, prop, front)
            targets.append(
                ValidationTarget(
                    candidate=candidate,
                    property=prop,
                    method=method,
                    value_score=score,
                    rationale=rationale,
                )
            )

    targets.sort(key=lambda t: (-t.value_score, t.candidate.candidate_id, t.property))
    return targets[: policy.max_candidates], refused


def _value_score(
    prediction: Prediction, results: CandidateResults, prop: str, front: int
) -> tuple[float, str]:
    unit = get_property(prop).canonical_unit
    quantity = prediction.quantity
    uncertainty = prediction.uncertainty.converted(quantity.unit, unit)

    relative = 0.0
    magnitude = abs(quantity.to(unit).value)
    if uncertainty.std is not None and magnitude > 0:
        relative = min(1.0, uncertainty.std / magnitude)
    elif uncertainty.std is None:
        # An unquantified uncertainty is not a small one.
        relative = 1.0

    utility = results.objective_vector.get(prop, 0.5)
    # A utility near the middle of the range is where a shift changes the
    # ordering; one already at 0 or 1 is unlikely to be moved by a correction.
    contention = 1.0 - abs(utility - 0.5) * 2.0
    domain_penalty = 1.0 if prediction.applicability.in_domain else 1.5
    front_weight = 1.0 / (1.0 + front)

    score = relative * (0.3 + 0.7 * contention) * domain_penalty * front_weight
    reasons = [f"relative uncertainty {relative:.0%}"]
    if not prediction.applicability.in_domain:
        reasons.append("the expert flagged this as out of its domain")
    if contention > 0.6:
        reasons.append("the utility sits where a correction could reorder the ranking")
    return score, "; ".join(reasons)


def merge_prediction(
    expert: Prediction | None,
    physics: Prediction,
    policy: ValidationPolicy | None = None,
) -> tuple[Prediction, Disagreement | None]:
    """Combine an expert prediction with a validated one.

    Where both quote an uncertainty they are combined by inverse variance,
    which is the maximum-likelihood combination for two independent estimates
    and, importantly, does not assume the physics is right. Where only one
    does, that one is taken and the fact is recorded.

    Physics does not automatically replace the expert. A converged calculation
    at a level of theory with a known systematic error is not obviously better
    than a correlation fitted to measurements of the quantity itself.
    """
    policy = policy or ValidationPolicy()
    if expert is None or expert.quantity is None or physics.quantity is None:
        return physics, None

    unit = get_property(physics.property).canonical_unit
    expert_value = expert.quantity.to(unit).value
    physics_value = physics.quantity.to(unit).value
    expert_std = expert.uncertainty.converted(expert.quantity.unit, unit).std
    physics_std = physics.uncertainty.converted(physics.quantity.unit, unit).std

    disagreement = Disagreement(
        property=physics.property,
        expert_id=expert.expert_id,
        expert_value=expert_value,
        expert_std=expert_std,
        physics_value=physics_value,
        physics_std=physics_std,
        unit=unit,
    )

    notes = list(physics.notes)
    if expert_std and physics_std and expert_std > 0 and physics_std > 0:
        weight_expert = 1.0 / expert_std**2
        weight_physics = 1.0 / physics_std**2
        combined = (expert_value * weight_expert + physics_value * weight_physics) / (
            weight_expert + weight_physics
        )
        combined_std = math.sqrt(1.0 / (weight_expert + weight_physics))
        notes.append(
            f"combined with the {expert.expert_id} prediction by inverse variance: "
            f"{expert_value:.4g} +/- {expert_std:.3g} and {physics_value:.4g} +/- "
            f"{physics_std:.3g} give {combined:.4g} +/- {combined_std:.3g} {unit}"
        )
        if disagreement.is_significant(policy.disagreement_sigma):
            notes.append(
                "the two disagree by more than their stated uncertainties allow, so at "
                "least one of them is overconfident; the combined value should be "
                "treated with suspicion rather than as a consensus"
            )
        merged = physics.model_copy(
            update={
                "quantity": Quantity(value=combined, unit=unit),
                "uncertainty": Uncertainty(
                    std=combined_std,
                    kind=UncertaintyKind.COMBINED,
                    basis=(
                        f"inverse-variance combination of {expert.expert_id} and "
                        f"{physics.expert_id}"
                    ),
                ),
                "notes": tuple(notes),
            }
        )
        return merged, disagreement

    notes.append(
        f"replaces the {expert.expert_id} prediction of {expert_value:.4g} {unit}; "
        "the two could not be combined because at least one quoted no uncertainty"
    )
    return physics.model_copy(update={"notes": tuple(notes)}), disagreement


def physics_prediction(
    prop: str,
    value: Quantity,
    uncertainty: Uncertainty,
    *,
    backend: str,
    method: str,
    conditions,
    provenance: ProvenanceRecord | None = None,
    notes: Sequence[str] = (),
    in_domain: bool = True,
    domain_warnings: Sequence[str] = (),
) -> Prediction:
    """Wrap a physics result in the same Prediction record experts produce.

    Using one record type is what lets the ranker treat a simulated value and a
    correlated one identically, which is the point of section 11's insistence
    that no module silently substitutes for another: the difference is recorded
    in the provenance, not in the shape of the data.
    """
    return Prediction(
        property=prop,
        quantity=value,
        uncertainty=uncertainty,
        applicability=ApplicabilityDomain(
            score=1.0 if in_domain else 0.3,
            in_domain=in_domain,
            warnings=tuple(domain_warnings),
            basis="physics calculation",
        ),
        status=PredictionStatus.OK if in_domain else PredictionStatus.OUT_OF_DOMAIN,
        expert_id=backend,
        expert_version="1",
        method=method,
        conditions=conditions,
        provenance=provenance
        or ProvenanceRecord(kind=ProvenanceKind.SIMULATION, producer=backend),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


#: Quantum observables reachable from one QMResult field.
_QM_OBSERVABLES = {
    "homo_lumo_gap": "homo_lumo_gap",
    "dipole_moment": "dipole_moment",
    "electronic_energy": "total_energy",
}


class PhysicsValidator:
    """Runs the selected physics and folds the results back into the ranking.

    Section 9 stages 8 and 9. The budget is spent in value order and stops at
    the first limit reached, so a validation stage never runs away: the number
    of calls avoided is reported alongside the number made, because section 12
    counts avoided calls as a success rather than an omission.
    """

    def __init__(
        self,
        policy: ValidationPolicy | None = None,
        qm_backend=None,
        md_engine=None,
    ) -> None:
        self.policy = policy or ValidationPolicy()
        self._qm = qm_backend
        self._md = md_engine

    # -- backends ----------------------------------------------------------

    def quantum(self):
        if self._qm is None:
            from formulate.physics.qm import PySCFBackend

            self._qm = PySCFBackend()
        return self._qm

    def dynamics(self):
        if self._md is None:
            from formulate.physics.md import MDEngine

            self._md = MDEngine()
        return self._md

    # -- the stage ---------------------------------------------------------

    def validate(
        self, ranked: Sequence[Candidate], spec: TargetSpec
    ) -> tuple[list[Candidate], ValidationReport]:
        """Validate a subset and return the pool with results merged in."""
        import time

        targets, refused = select_targets(ranked, spec, self.policy)
        report = ValidationReport(targets=list(targets), skipped=dict(refused))

        candidates = {c.candidate_id: c for c in ranked}
        before_rank = {
            c.candidate_id: (c.results.aggregate_rank or index + 1)
            for index, c in enumerate(ranked)
            if c.results is not None
        }

        started = time.perf_counter()
        for target in targets:
            elapsed = time.perf_counter() - started
            if elapsed > self.policy.max_seconds:
                report.calls_avoided += 1
                report.skipped.setdefault(
                    target.property,
                    f"the {self.policy.max_seconds:.0f} s validation budget was spent "
                    "before this target was reached",
                )
                continue

            prediction, diagnostics = self._run_target(target, spec)
            report.calls_made += 1
            if prediction is None:
                report.skipped.setdefault(target.property, diagnostics)
                continue

            candidate = candidates[target.candidate.candidate_id]
            merged, disagreement = self._merge_into(candidate, prediction)
            candidates[candidate.candidate_id] = merged
            report.validated.append(prediction)
            if disagreement is not None:
                report.disagreements.append(disagreement)

        report.seconds_spent = time.perf_counter() - started
        # Every candidate-property pair the policy declined to buy.
        report.calls_avoided += max(
            0, len(ranked) * len(validatable_properties(spec)[0]) - report.calls_made
        )

        updated = [candidates[c.candidate_id] for c in ranked]
        report.rank_changes = self._rank_changes(before_rank, updated)
        return updated, report

    # -- one target --------------------------------------------------------

    def _run_target(
        self, target: ValidationTarget, spec: TargetSpec
    ) -> tuple[Prediction | None, str]:
        from formulate.physics.geometry import geometry_from_smiles

        smiles = target.candidate.primary_smiles
        if smiles is None:
            return None, "the candidate carries no structure to build a geometry from"

        try:
            geometry = geometry_from_smiles(smiles, n_conformers=5)
        except Exception as exc:
            return None, f"could not build a 3D geometry ({exc})"

        if target.method is ValidationMethod.QUANTUM:
            return self._run_quantum(target, geometry, spec)
        return self._run_dynamics(target, geometry, spec)

    def _run_quantum(self, target: ValidationTarget, geometry, spec):
        from formulate.physics.qm import QMMethod, QMRequest

        backend = self.quantum()
        if not backend.is_available():
            return None, backend.unavailable_reason()

        if target.property == "atomization_energy":
            return self._run_atomization(target, geometry, backend)

        field = _QM_OBSERVABLES.get(target.property)
        if field is None:
            return None, f"no quantum observable maps to {target.property}"

        result = backend.run(
            QMRequest(
                geometry=geometry,
                method=QMMethod.DFT,
                basis="6-31g",
                xc="b3lyp",
                optimize_geometry=False,
            )
        )
        if not result.usable:
            return None, f"the calculation did not produce a usable result: {'; '.join(result.diagnostics)}"

        value = getattr(result, field, None)
        if value is None:
            return None, f"the calculation produced no {target.property}"

        return (
            physics_prediction(
                target.property,
                value,
                # A converged calculation is not an exact one. The uncertainty
                # quoted is the level of theory's known systematic error, not a
                # numerical convergence figure, which would be far too small.
                _quantum_uncertainty(target.property, value),
                backend=f"qm:{result.backend}",
                method=result.method_signature,
                # The conditions the calculation was actually performed at, not
                # the ones that were requested. A gas-phase result at zero
                # Kelvin is not a result at 25 degrees and one atmosphere, and
                # stamping it with the request would make the condition check
                # of section 11 pass on a claim it was written to catch.
                conditions=VACUUM_ZERO_KELVIN,
                provenance=result.provenance,
                notes=tuple(result.limitations) + tuple(result.diagnostics),
            ),
            "",
        )

    def _run_atomization(self, target: ValidationTarget, geometry, backend):
        """A molecule against its free atoms, which is several calculations.

        The free atoms are open shell - carbon a triplet, nitrogen a quartet -
        and computing them as closed-shell singlets converges perfectly well
        and returns an atomization energy wrong by hundreds of kJ/mol per atom.
        The multiplicities live in a table rather than a default for that
        reason.
        """
        from formulate.physics.qm.base import QMMethod, QMRequest
        from formulate.physics.qm.thermochemistry import (
            atomization_energy,
            unsupported_elements,
        )

        missing = unsupported_elements(geometry.symbols)
        if missing:
            return None, (
                "no ground-state multiplicity is tabulated for "
                + ", ".join(missing)
                + ", so there is no atomic reference to subtract"
            )

        template = QMRequest(
            geometry=geometry, method=QMMethod.DFT, basis="6-31g", xc="b3lyp",
            optimize_geometry=False,
        )
        result = atomization_energy(backend, geometry, template)
        if result is None:
            return None, "one of the atomic or molecular calculations did not converge"
        if result.diagnostics:
            return None, "; ".join(result.diagnostics)

        return (
            physics_prediction(
                target.property,
                Quantity(value=result.value, unit="J/mol"),
                Uncertainty(
                    # The systematic error of the level of theory, not a
                    # convergence figure. Measured against four experimental
                    # atomization energies, this basis under-binds by tens of
                    # kJ/mol per bond, and consistently in one direction.
                    std=abs(result.value) * 0.05,
                    kind=UncertaintyKind.EPISTEMIC,
                    basis=(
                        "systematic error of "
                        f"{result.method_signature} on an atomization energy, which a "
                        "small basis gets wrong by tens of kJ/mol per bond"
                    ),
                ),
                backend=f"qm:{backend.id}",
                method=result.method_signature,
                conditions=VACUUM_ZERO_KELVIN,
                notes=result.limitations,
            ),
            "",
        )

    def _run_dynamics(self, target: ValidationTarget, geometry, spec):
        from formulate.physics.md import MDProtocol, MDRequest

        if target.property in CONDENSED_PROTOCOLS:
            return self._run_condensed(target, spec)

        engine = self.dynamics()
        name = DYNAMICS_PROTOCOLS.get(target.property)
        if name is None:
            return None, f"no dynamics protocol in this system produces {target.property}"
        protocol = MDProtocol(name)
        request = MDRequest(
            geometry=geometry,
            protocol=protocol,
            calculator="GFN-FF",
            temperature_k=spec.conditions.temperature_k or 298.15,
            equilibration_steps=500,
            production_steps=3000,
            sample_interval=10,
        )
        result = engine.run(request)
        if not result.usable:
            return None, result.feasibility_reason or "; ".join(result.diagnostics)

        sampling = result.sampling
        return (
            physics_prediction(
                target.property,
                result.value,
                Uncertainty(
                    std=sampling.standard_error if sampling else None,
                    kind=UncertaintyKind.SAMPLING,
                    basis=(
                        f"block-averaged sampling error over {result.production_ps:.1f} ps; "
                        "finite-size error is separate and systematic"
                    ),
                ),
                backend="md:ase",
                method=f"{result.calculator} {protocol.value}",
                conditions=spec.conditions,
                provenance=result.provenance,
                notes=tuple(result.limitations) + tuple(result.diagnostics),
                in_domain=bool(sampling and sampling.converged),
                domain_warnings=tuple(result.diagnostics),
            ),
            "",
        )

    def _run_condensed(self, target: ValidationTarget, spec: TargetSpec):
        """Run a periodic condensed-phase protocol, or say what it would cost.

        These are the properties section 13 insists cannot come from a finite
        cluster, and they could not be attempted at all until MMFF94 became
        available as an OpenMM system. They are also the expensive ones: tens
        of minutes each against seconds for a quantum observable. A budget too
        small to hold one is not a reason to run a shorter version and present
        the result - a half-equilibrated box returns a density thirty per cent
        low with no outward sign - so the answer is the cost.
        """
        from formulate import chem
        from formulate.physics.md import condensed
        from formulate.physics.md.mmff import extract_parameters, openmm_available

        protocol = CONDENSED_PROTOCOLS[target.property]
        cost = CONDENSED_COST_SECONDS[protocol]
        if not openmm_available():
            return None, "OpenMM is not installed, so no periodic condensed phase is possible"
        if not chem.rdkit_available():
            return None, "RDKit is required to assign MMFF94 parameters"
        if cost > self.policy.max_seconds:
            return None, (
                f"a {protocol} run needs roughly {cost / 60:.0f} minutes of wall clock and "
                f"the validation budget is {self.policy.max_seconds / 60:.0f}; raise it to "
                "buy this one, because a shortened run does not fail, it just answers wrong"
            )

        smiles = target.candidate.primary_smiles
        if smiles is None:
            return None, "the candidate carries no structure to build a box from"
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            if mol is None or AllChem.EmbedMolecule(mol, randomSeed=0xF00D) != 0:
                return None, "could not embed a 3D conformer for a condensed-phase box"
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception as exc:
            return None, f"could not build a 3D structure ({type(exc).__name__}: {exc})"
        if extract_parameters(mol) is None:
            return None, "MMFF94 has no parameters for this structure"

        temperature = spec.conditions.temperature_k or 298.15
        molecules = self.policy.condensed_molecules
        try:
            if protocol == "self_diffusion":
                density = self.policy.assumed_density
                result = condensed.run_self_diffusion(
                    mol, molecules, temperature, density_g_cm3=density
                )
                value = Quantity(value=result.coefficient_m2_s, unit="m^2/s")
                uncertainty = Uncertainty(
                    std=result.error_m2_s,
                    kind=UncertaintyKind.SAMPLING,
                    basis=(
                        f"least-squares error over the {result.fitted_window_ps[0]:.0f}-"
                        f"{result.fitted_window_ps[1]:.0f} ps window where the mean squared "
                        f"displacement grows as t^{result.log_log_slope:.2f}"
                    ),
                )
                diagnostics = result.diagnostics
                in_domain = abs(result.log_log_slope - 1.0) <= 0.15
            else:
                liquid = condensed.run_npt(mol, molecules, temperature)
                if protocol == "density":
                    value = Quantity(value=liquid.density_g_cm3, unit="g/cm^3")
                    uncertainty = Uncertainty(
                        std=liquid.density_error,
                        kind=UncertaintyKind.SAMPLING,
                        basis=f"block-averaged over {liquid.production_ps:.0f} ps at constant pressure",
                    )
                    diagnostics = liquid.diagnostics
                    in_domain = not liquid.diagnostics
                else:
                    gas, gas_error = condensed.sample_isolated_energy(mol, temperature)
                    cohesive = condensed.cohesive_energy_density(liquid, gas, gas_error)
                    value = Quantity(value=cohesive.cohesive_energy_density_pa, unit="Pa")
                    uncertainty = Uncertainty(
                        std=cohesive.error_pa,
                        kind=UncertaintyKind.SAMPLING,
                        basis=(
                            f"propagated from a vaporisation energy of "
                            f"{cohesive.vaporisation_energy:.1f} kJ/mol and a molar volume of "
                            f"{cohesive.molar_volume_cm3:.1f} cm^3/mol"
                        ),
                    )
                    diagnostics = cohesive.diagnostics
                    in_domain = not cohesive.diagnostics
        except Exception as exc:  # a failed simulation is a skip, not a crash
            return None, f"the condensed-phase run failed ({type(exc).__name__}: {exc})"

        return (
            physics_prediction(
                target.property,
                value,
                uncertainty,
                backend="md:openmm",
                method=f"MMFF94 periodic {protocol}, {molecules} molecules",
                conditions=spec.conditions,
                notes=(
                    "MMFF94 parameters taken from RDKit and translated into OpenMM forces; "
                    "the translation reproduces RDKit's own energy to 1e-11 kcal/mol",
                    "MMFF94 was fitted to gas-phase geometries, not to liquids, and "
                    "under-binds a condensed phase. Measured here: cohesive energy 15 per "
                    "cent low for ethanol and 35 per cent low for hexane, density 26 per "
                    "cent low for ethanol. The deficit is in dispersion, so it is worst "
                    "for the least polar. Use these to rank candidates, where the bias is "
                    "shared, not as quantitative values",
                    "electrostatics by particle-mesh Ewald, which drops MMFF's 0.05 A "
                    "buffering; van der Waals truncated at the cutoff with a long-range "
                    "correction",
                )
                + tuple(diagnostics),
                in_domain=in_domain,
                domain_warnings=tuple(diagnostics),
            ),
            "",
        )

    # -- merging -----------------------------------------------------------

    def _merge_into(
        self, candidate: Candidate, prediction: Prediction
    ) -> tuple[Candidate, Disagreement | None]:
        results = candidate.results
        if results is None:
            return candidate, None

        expert = results.prediction_for(prediction.property)
        merged, disagreement = merge_prediction(expert, prediction, self.policy)

        simulation_ids = results.simulation_ids
        if merged.provenance is not None:
            simulation_ids = simulation_ids + (merged.provenance.record_id,)

        updated = results.model_copy(
            update={
                "predictions": results.predictions + (merged,),
                "simulation_ids": simulation_ids,
            }
        )
        return candidate.with_results(updated), disagreement

    def _rank_changes(
        self, before: dict[str, int], updated: Sequence[Candidate]
    ) -> dict[str, tuple[int, int]]:
        changes: dict[str, tuple[int, int]] = {}
        for index, candidate in enumerate(updated, start=1):
            previous = before.get(candidate.candidate_id)
            if previous is not None and previous != index:
                changes[candidate.candidate_id] = (previous, index)
        return changes


def _quantum_uncertainty(prop: str, value: Quantity) -> Uncertainty:
    """The level of theory's known systematic error for this observable.

    Quoting an SCF convergence threshold here would be dishonest by orders of
    magnitude: the calculation is converged to microhartrees and wrong by
    tenths of an electronvolt.
    """
    if prop == "homo_lumo_gap":
        return Uncertainty(
            std=abs(value.value) * 0.30,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                "hybrid functionals systematically underestimate frontier gaps, commonly "
                "by 30 percent or more against experiment; this is a method error, not a "
                "convergence error"
            ),
        )
    if prop == "dipole_moment":
        return Uncertainty(
            std=0.3,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                "a Pople double-zeta basis without diffuse functions typically errs by a "
                "few tenths of a debye, and overestimates polar molecules"
            ),
        )
    return Uncertainty(
        kind=UncertaintyKind.UNKNOWN,
        basis=(
            "absolute electronic energies are not comparable across levels of theory, so "
            "no interval is quoted"
        ),
    )
