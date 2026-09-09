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
    # The three below cost nothing extra. Each is arithmetic on a run that was
    # already being made and whose other output was already being kept: a
    # molar volume is the density run's box divided by what is in it, an
    # enthalpy of vaporisation is the cohesive run's energy difference plus RT,
    # and a Hildebrand parameter is the square root of its energy density.
    # Leaving them out was not a judgement about physics, it was three
    # properties being computed and discarded.
    #
    # A fourth looked equally free and is not. Every quantum calculation
    # produces a total electronic energy, and _QM_OBSERVABLES has always known
    # how to read it, but an absolute electronic energy is monotone in electron
    # count - minus seventy-five hartree for water, minus two hundred and
    # twenty-eight for benzene - so ranking candidates on it sorts them by size.
    # It stays out, as NOT_VALIDATABLE_REASONS has said all along.
    "molar_volume_liquid": frozenset({ValidationMethod.DYNAMICS}),
    "enthalpy_vaporization": frozenset({ValidationMethod.DYNAMICS}),
    "hildebrand_solubility_parameter": frozenset({ValidationMethod.DYNAMICS}),
    # These two were refused for years for a reason that turned out not to be
    # about physics. Both are components of the pressure tensor - a surface
    # tension is a difference between diagonal ones, a shear viscosity the
    # autocorrelation of an off-diagonal one - and OpenMM publishes no pressure
    # tensor at all. formulate.physics.md.stress recovers it by finite
    # difference, and with it both properties became ordinary condensed-phase
    # protocols. They are the most expensive ones here by a wide margin.
    "surface_tension": frozenset({ValidationMethod.DYNAMICS}),
    "shear_viscosity": frozenset({ValidationMethod.DYNAMICS}),
}

#: Ideal-gas constant in kJ/(mol K), for the RT that separates a potential
#: energy of vaporisation from an enthalpy of vaporisation.
_GAS_CONSTANT_KJ = 0.00831446261815324

#: Properties a physics route could reach but this system cannot, and why.
#:
#: Kept as a table rather than as silence so that the gap is a stated one. Gas
#: heat capacity is the case worth naming: it is standard thermochemistry from
#: vibrational frequencies under the rigid-rotor harmonic-oscillator model, and
#: it is unreachable here for one missing capability rather than for any deep
#: reason. No backend in this installation computes a Hessian.
PHYSICS_OUT_OF_REACH: dict[str, str] = {
    "heat_capacity_gas": (
        "needs vibrational frequencies, so it needs a Hessian, and no quantum backend "
        "here computes one; the partition function it would be assembled from is "
        "otherwise standard"
    ),
    "glass_transition_temperature": (
        "a molecular-dynamics glass transition depends on the cooling rate chosen, which "
        "is a dozen orders of magnitude faster than any experiment, and comes out tens of "
        "kelvin high; reporting one without that caveat would be worse than refusing"
    ),
    "melting_point": (
        "needs two-phase coexistence or free-energy integration between the solid and the "
        "liquid, neither of which is implemented, and a naive heating run superheats"
    ),
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
    "molar_volume_liquid": "density",
    "cohesive_energy_density": "cohesive_energy_density",
    "enthalpy_vaporization": "cohesive_energy_density",
    "hildebrand_solubility_parameter": "cohesive_energy_density",
    "self_diffusion_coefficient": "self_diffusion",
    "surface_tension": "surface_tension",
    "shear_viscosity": "shear_viscosity",
}

#: Rough wall-clock cost of each condensed protocol on this installation, in
#: seconds, measured rather than guessed: MMFF94 through OpenMM's custom forces
#: runs a two-thousand-atom box at roughly twelve nanoseconds per day.
CONDENSED_COST_SECONDS: dict[str, float] = {
    "density": 3600.0,
    "cohesive_energy_density": 4200.0,
    "self_diffusion": 5400.0,
    # The two pressure-tensor protocols are an order of magnitude dearer than
    # the rest, for two different reasons. A slab has to be several times a
    # cutoff thick before it has an interior, so its box is large before any
    # sampling starts, and it needs a constant-pressure run first to find the
    # density it will then be held at. A Green-Kubo integral needs the stress
    # every ten femtoseconds rather than the volume every picosecond, and each
    # of those samples is six single-point energies, so the finite difference
    # rather than the dynamics sets the cost. The slab was measured: 350
    # molecules of 2-butanone, 150 ps equilibration and 400 ps production,
    # 6.12 hours on two of this installation's four cores while a second run
    # shared the machine - so this figure is what a contended run costs, which
    # is the honest one to budget. The viscosity is still an estimate from
    # the density run's rate, deliberately generous: a wrong guess here refuses
    # a run it could have afforded, which costs a validation; the other
    # direction runs something it cannot finish, which costs the whole stage.
    "surface_tension": 22000.0,
    "shear_viscosity": 9000.0,
}

#: Measured systematic error of each force field in the condensed phase, as a
#: fraction of the value, keyed by property.
#:
#: This exists because the sampling error these protocols report is not the
#: error. A block average over a hundred picoseconds puts a few parts in a
#: thousand on a density; the force field puts twenty-six per cent on the same
#: number. Quoting only the first tells the ranker a wrong value is a certain
#: one, and the calibration suite exists to catch exactly that.
#:
#: Two rows, both measured against experiment at 298 K rather than assumed.
#: OPLS-AA over five liquids (ethanol, acetone, toluene, cyclohexane, hexane):
#: mean absolute error 1.2 per cent on density and 3.7 on the energy of
#: vaporisation. MMFF94 over one density (ethanol, 26 per cent low) and three
#: vaporisation energies (toluene 11, ethanol 15, hexane 3 per cent low), so
#: its density figure rests on a single point and is kept deliberately large
#: rather than refined on no evidence.
#:
#: Keyed by property and not by protocol, because one protocol yields several
#: numbers whose errors differ. A molar volume is a density inverted, so it
#: carries the same relative error. A cohesive energy density is an energy over
#: a volume, so it carries both in quadrature. A Hildebrand parameter is the
#: square root of that, so it carries half of it - a square root halves a
#: relative error, and quoting the energy density's figure on it would overstate
#: the uncertainty by a factor of two.
#:
#: Self-diffusion is absent from both rows because neither was measured for it,
#: and a coefficient that spans orders of magnitude is not a place to
#: interpolate a systematic error from a density. Runs of it say so instead.
#: Surface tension and shear viscosity are absent for the same reason and not
#: for a weaker one. The tension has been run once, on 2-butanone: 23.69 mN/m
#: against a measured 23.96, a deviation of 1.1 per cent inside a sampling
#: error of 12.5. That is agreement, and it is not a measurement of the
#: systematic error - a bias of five per cent would have been invisible under
#: that noise. Borrowing the density's 1.2 per cent would be worse than
#: admitting the gap: a tension is a small difference between two large
#: pressures and has no reason to inherit a density's accuracy. See
#: UNMEASURED_SYSTEMATIC below.
CONDENSED_SYSTEMATIC: dict[str, dict[str, float]] = {
    "opls-aa": {
        "liquid_density": 0.012,
        "molar_volume_liquid": 0.012,
        "enthalpy_vaporization": 0.037,
        "cohesive_energy_density": 0.039,
        "hildebrand_solubility_parameter": 0.020,
    },
    "mmff94": {
        "liquid_density": 0.26,
        "molar_volume_liquid": 0.26,
        "enthalpy_vaporization": 0.15,
        "cohesive_energy_density": 0.30,
        "hildebrand_solubility_parameter": 0.15,
    },
}


#: Protocols whose force-field systematic error has not been measured here.
#:
#: A missing row in CONDENSED_SYSTEMATIC is otherwise indistinguishable from a
#: systematic error of zero, and the difference matters: the uncertainty a run
#: of one of these reports is sampling error only, which is a floor rather than
#: an estimate. Naming them makes the gap assertable by a test instead of
#: something a reader has to notice.
UNMEASURED_SYSTEMATIC: frozenset[str] = frozenset(
    {"self_diffusion_coefficient", "surface_tension", "shear_viscosity"}
)


#: How each force field is named in a prediction's method string.
_FORCE_FIELD_NAMES = {"opls-aa": "OPLS-AA", "mmff94": "MMFF94"}

#: What a reader of a condensed-phase prediction needs to know about the force
#: field that produced it. Each figure here was measured against experiment.
_CONDENSED_NOTES: dict[str, tuple[str, ...]] = {
    "opls-aa": (
        "OPLS-AA types assigned from foyer's SMARTS definitions; the charges come "
        "from the type table rather than from a quantum calculation, which is why "
        "this force field is reachable here at all",
        "OPLS-AA was fitted to liquids. Against experiment at 298 K over ethanol, "
        "acetone, toluene, cyclohexane and hexane: density within 1.2 per cent mean "
        "absolute error, energy of vaporisation within 3.7",
        "the assigned types were checked to carry zero net charge, which is what "
        "catches this force field's silent failure on substituted aromatics and "
        "polychloroalkanes",
        "electrostatics by particle-mesh Ewald; van der Waals mixed geometrically as "
        "OPLS-AA requires, truncated at the cutoff with a long-range correction",
    ),
    "mmff94": (
        "MMFF94 parameters taken from RDKit and translated into OpenMM forces; "
        "the translation reproduces RDKit's own energy to 1e-11 kcal/mol",
        "MMFF94 was fitted to gas-phase geometries, not to liquids, and "
        "under-binds a condensed phase. Measured here: potential energy of "
        "vaporisation 11 per cent low for toluene, 15 per cent for ethanol, "
        "3 per cent for hexane, and a density 26 per cent low for ethanol. "
        "Use these to rank candidates, where the bias is shared, not as "
        "quantitative values",
        "electrostatics by particle-mesh Ewald, which drops MMFF's 0.05 A "
        "buffering; van der Waals truncated at the cutoff with a long-range "
        "correction",
    ),
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
        "a boiling point needs the saturated vapour pressure, and the slab that measures "
        "a surface tension is a liquid-vapour coexistence that could in principle supply "
        "it - but at 298 K an ordinary solvent puts less than one molecule in the vacuum "
        "gap. Measured on 2-butanone: 0.73 molecules on average across 400 ps, implying "
        "49 kPa against a real 12.6, because a count under one is not a statistic. "
        "Reaching one per cent on the pressure would need of order ten thousand "
        "molecules in the gap, which is a box a hundred times this one or a run a "
        "hundred times longer"
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
    #: Try a fast interatomic potential before spending a quantum calculation
    #: (section 8). Off means every quantum target goes straight to the solver.
    multi_fidelity: bool = True
    #: Relative uncertainty above which the cheap answer is not accepted as it
    #: stands. Ten per cent of an atomization energy is tens of kJ/mol, which
    #: is the scale at which a ranking on that property changes.
    escalate_above_relative_uncertainty: float = 0.10


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
    #: Calls to a fast potential, which are not quantum calls and are counted
    #: separately so a cheap run cannot be mistaken for an expensive one.
    cheap_calls: int = 0
    #: (target description, decision) for every multi-fidelity choice made.
    escalations: list[tuple[str, str]] = field(default_factory=list)
    #: Candidate id -> (rank before, rank after).
    rank_changes: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def disagreement_rate(self) -> float | None:
        """Section 12's "physics disagreement rate"."""
        comparable = [d for d in self.disagreements if d.z_score is not None]
        if not comparable:
            return None
        return sum(1 for d in comparable if d.is_significant()) / len(comparable)

    @property
    def priced_out(self) -> dict[str, str]:
        """Properties declined for cost alone, not for any physical reason."""
        return {
            prop: reason
            for prop, reason in self.skipped.items()
            if "wall clock" in reason and "budget is" in reason
        }

    def describe(self) -> str:
        if not self.targets and not self.skipped:
            return "No candidate was selected for physics validation."

        lines = []
        # Say it first and say it plainly. A run that validated nothing used to
        # report that as a list of per-property skip messages, each of them
        # accurate and none of them saying the thing that matters: no physics
        # ran, and the ranking in front of you rests entirely on correlations.
        # The default budget is six hundred seconds and the cheapest periodic
        # protocol needs several thousand, so that is what a default run does.
        priced_out = self.priced_out
        if not self.calls_made:
            if priced_out:
                lines.append(
                    "NO PHYSICS RAN. Every protocol was priced out of the validation "
                    "budget, so nothing here has been checked against a simulation and "
                    "the ranking rests on the expert panel alone. Physics is opt-in: "
                    "raise ValidationPolicy.max_seconds past the cost quoted below to "
                    "buy it, and expect to wait that long."
                )
            else:
                lines.append(
                    "No physics ran. The reasons below are physical rather than "
                    "budgetary, so raising the budget would not change them."
                )
            lines.append("")

        lines.append(
            f"Validated {self.calls_made} calculation(s) on {len(self.targets)} target(s) "
            f"in {self.seconds_spent:.1f} s; {self.calls_avoided} avoided by selection."
        )
        for target in self.targets:
            name = target.candidate.label or target.candidate.primary_smiles
            lines.append(
                f"  {name} / {target.property} via {target.method.value}: {target.rationale}"
            )

        if self.escalations:
            lines.append("")
            lines.append(
                f"Multi-fidelity: {self.cheap_calls} fast-potential call(s), each one "
                "deciding whether the quantum calculation was worth buying."
            )
            lines.extend(f"  {what}: {why}" for what, why in self.escalations)

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


#: Joules per mole in one electronvolt.
_J_PER_MOL_PER_EV = 96485.33212331001

#: Symbols to atomic numbers, for handing a geometry to a potential that speaks
#: atomic numbers rather than element symbols.
_ATOMIC_NUMBER_BY_SYMBOL = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Br": 35, "I": 53,
}

#: What a potential that returns energies and forces can actually answer. A
#: HOMO-LUMO gap or a dipole moment is not derivable from an energy surface, so
#: those targets escalate unconditionally rather than being approximated here.
_POTENTIAL_OBSERVABLES = frozenset({"atomization_energy"})

#: Properties where escalating to the quantum rung makes the answer worse, with
#: the measurement that established it.
#:
#: Section 8 describes a ladder and assumes the expensive rung is the accurate
#: one. For atomization energy at the basis this pipeline runs, it is not, and
#: that was worth checking rather than assuming. Against seven experimental
#: electronic well depths (D_0 plus zero-point energy, both handbook values):
#:
#:     MACE-OFF23-small   mean absolute error 13.0 kJ/mol, bias -11.5
#:     B3LYP/6-31G        mean absolute error 78.7 kJ/mol, bias -78.7
#:
#: The small basis under-binds every one of the seven, which is the textbook
#: failure of 6-31G on bond energies; the network was fitted to
#: wB97M-D3(BJ)/def2-TZVPPD, a far better level than this pipeline can afford
#: to run. So for this property the potential is not the cheap approximation to
#: the quantum answer - it is the better answer, six times over, and buying the
#: quantum calculation would spend six times the wall clock to get further from
#: the truth.
#:
#: Escalation for these properties therefore fires on one trigger only: the
#: molecule falling outside the network's element domain, where an
#: extrapolating neural network is worse than an under-binding basis. See
#: ``docs/BENCHMARKS.md``.
_POTENTIAL_IS_THE_BETTER_RUNG = frozenset({"atomization_energy"})

#: One sigma on the potential's atomization energy, from the mean absolute
#: error of 13.0 kJ/mol above under the normal-distribution conversion
#: sigma = 1.253 * MAE this repository uses elsewhere.
_POTENTIAL_ATOMIZATION_STD_J_PER_MOL = 16_300.0


def _potential_uncertainty(identifier: str, prop: str) -> Uncertainty:
    """Measured against experiment, not against the rung above.

    An earlier version quoted the spread against B3LYP/6-31G, which was 73
    kJ/mol and almost entirely that basis's own error rather than the
    network's. Stating it would have made the better method look like the
    worse one.
    """
    return Uncertainty(
        std=_POTENTIAL_ATOMIZATION_STD_J_PER_MOL,
        kind=UncertaintyKind.EPISTEMIC,
        basis=(
            f"mean absolute error of {identifier} against seven experimental electronic "
            "atomization energies, 13.0 kJ/mol, converted to a one-sigma spread; both "
            "sides are electronic energies against ground-state free atoms, with no "
            "zero-point correction on either"
        ),
    )


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
        potential=None,
    ) -> None:
        self.policy = policy or ValidationPolicy()
        self._qm = qm_backend
        self._md = md_engine
        self._potential = potential

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

            prediction, diagnostics = self._run_target(target, spec, report)
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
        self, target: ValidationTarget, spec: TargetSpec, report: "ValidationReport"
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
            return self._run_quantum(target, geometry, spec, report)
        return self._run_dynamics(target, geometry, spec)

    # -- the cheap rung ----------------------------------------------------

    def potential(self):
        """The fast potential, loaded once and shared across targets."""
        if self._potential is None:
            from formulate.physics.potentials import MacePotential

            self._potential = MacePotential()
        return self._potential

    def _cheap_pass(
        self, target: ValidationTarget, geometry, report: "ValidationReport"
    ):
        """Answer with the potential where it can, and say why when it cannot.

        Returns ``(prediction, decision)``. A prediction means the cheap rung
        settled it and the quantum calculation is not bought; ``None`` with a
        decision means it is. The decision is recorded either way, because a
        run that escalated everything and a run that escalated nothing are very
        different runs and the report has to be able to tell them apart.
        """
        from formulate.physics.potentials import (
            EscalationDecision,
            EscalationPolicy,
            UnsupportedSystem,
        )

        label = (
            f"{target.candidate.label or target.candidate.primary_smiles}"
            f" / {target.property}"
        )
        policy = EscalationPolicy(
            uncertainty_fraction=self.policy.escalate_above_relative_uncertainty,
            disagreement_sigma=self.policy.disagreement_sigma,
        )

        potential = self.potential()
        if not potential.is_available():
            reason = potential.unavailable_reason()
            report.escalations.append(
                (label, f"escalate to QM: no fast potential is installed ({reason})")
            )
            return None, None

        unknown = sorted({s for s in geometry.symbols if s not in _ATOMIC_NUMBER_BY_SYMBOL})
        if unknown:
            # An element this table has never heard of is outside every
            # organic potential's training set by construction. Reaching the
            # dictionary for it raised a KeyError, which surfaced as a crash
            # from a stage whose entire job is to decline things gracefully.
            decision = policy.decide(
                out_of_domain_reason=(
                    f"no atomic number is tabulated for {', '.join(unknown)}, so the "
                    "potential cannot even be asked about this molecule"
                )
            )
            report.escalations.append((label, decision.describe()))
            return None, decision

        numbers = [_ATOMIC_NUMBER_BY_SYMBOL[s] for s in geometry.symbols]
        refusal = potential.refusal(numbers)
        if refusal is not None:
            decision = policy.decide(out_of_domain_reason=refusal)
            report.escalations.append((label, decision.describe()))
            return None, decision

        if target.property not in _POTENTIAL_OBSERVABLES:
            report.escalations.append(
                (
                    label,
                    "escalate to QM: "
                    f"{potential.info.identifier} predicts energies and forces, and "
                    f"{target.property} is not derivable from them",
                )
            )
            return None, None

        try:
            value_ev = potential.atomization_energy_ev(numbers, geometry.positions)
        except UnsupportedSystem as exc:
            decision = policy.decide(out_of_domain_reason=str(exc))
            report.escalations.append((label, decision.describe()))
            return None, decision
        except Exception as exc:
            report.escalations.append(
                (label, f"escalate to QM: the fast potential failed ({exc})")
            )
            return None, None
        report.cheap_calls += 1

        value = Quantity(value=value_ev * _J_PER_MOL_PER_EV, unit="J/mol")
        uncertainty = _potential_uncertainty(potential.info.identifier, target.property)

        incumbent = None
        if target.candidate.results is not None:
            incumbent = target.candidate.results.prediction_for(target.property)
        other_value = other_std = None
        if incumbent is not None and incumbent.quantity is not None:
            other_value = incumbent.quantity.to("J/mol").value
            other_std = incumbent.uncertainty.converted(
                incumbent.quantity.unit, "J/mol"
            ).std

        if target.property in _POTENTIAL_IS_THE_BETTER_RUNG:
            # In domain, and the rung above is measurably worse here. Nothing
            # the disagreement trigger could find would be an argument for
            # buying it, so it is not consulted.
            decision = EscalationDecision(
                False,
                (
                    f"{potential.info.identifier} is in domain and, on this property, "
                    "6 times more accurate than the B3LYP/6-31G calculation the "
                    "escalation would buy (13.0 against 78.7 kJ/mol over seven "
                    "measured atomization energies)",
                ),
            )
        else:
            decision = policy.decide(
                value=value.value,
                uncertainty=uncertainty.std,
                other_value=other_value,
                other_uncertainty=other_std,
                rank=target.candidate.results.aggregate_rank
                if target.candidate.results is not None
                else None,
            )
        report.escalations.append((label, decision.describe()))
        if decision.escalate:
            return None, decision

        return (
            physics_prediction(
                target.property,
                value,
                uncertainty,
                backend=f"potential:{potential.info.identifier}",
                method=(
                    f"{potential.info.identifier} single point, atomization against the "
                    "model's own isolated-atom references"
                ),
                conditions=VACUUM_ZERO_KELVIN,
                notes=(
                    "electronic atomization energy: no zero-point correction, because a "
                    "single-point energy contains no vibrational information",
                    decision.describe(),
                ),
            ),
            decision,
        )

    def _run_quantum(
        self, target: ValidationTarget, geometry, spec, report: "ValidationReport"
    ):
        from formulate.physics.qm import QMMethod, QMRequest

        if self.policy.multi_fidelity:
            settled, _ = self._cheap_pass(target, geometry, report)
            if settled is not None:
                report.calls_avoided += 1
                return settled, ""

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
                    # convergence figure. Measured against seven experimental
                    # electronic atomization energies: mean absolute error 78.7
                    # kJ/mol, mean relative error 5.1 per cent, and every one of
                    # the seven under-bound. Five per cent is therefore the right
                    # magnitude but the wrong shape - the error is a one-signed
                    # bias, not a symmetric spread - and a symmetric interval is
                    # kept only because correcting a bias measured on seven small
                    # molecules would be fitting to the test set.
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

    @staticmethod
    def _condensed_force_field(mol) -> tuple[str | None, str]:
        """Pick the force field for a box, and say why if it is the weaker one.

        OPLS-AA is preferred wherever it will type the molecule. Its refusals
        are worth passing on verbatim: one of them is that the types it found
        do not carry a neutral molecule's charge, which is a statement about
        this specific structure rather than a missing feature, and a note
        saying so is more use than "OPLS-AA was unavailable".
        """
        from formulate.physics.md import opls
        from formulate.physics.md.mmff import extract_parameters

        reason = ""
        if not opls.available():
            reason = "OPLS-AA is not installed in this environment"
        else:
            refused = opls.refusal(mol)
            if refused is None:
                return "opls-aa", ""
            reason = refused

        if extract_parameters(mol) is None:
            return None, (
                f"neither force field can parameterise this structure: {reason}, "
                "and MMFF94 has no parameters for it either"
            )
        return "mmff94", reason

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
        from formulate.physics.md.mmff import openmm_available

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
        # OPLS-AA first, MMFF94 second, and the difference is not marginal:
        # over five liquids OPLS-AA lands within 1.2 per cent on density and
        # 3.7 on the energy of vaporisation, where MMFF94 is 26 per cent low on
        # ethanol's density. OPLS-AA only types about three quarters of the
        # reference set, though, so the fallback is what keeps the protocol
        # answerable at all, and the reason for it is carried through to the
        # notes rather than dropped.
        force_field, fallback_reason = self._condensed_force_field(mol)
        if force_field is None:
            return None, fallback_reason

        temperature = spec.conditions.temperature_k or 298.15
        molecules = self.policy.condensed_molecules

        if protocol in ("surface_tension", "shear_viscosity") and force_field != "opls-aa":
            return None, (
                f"a {protocol.replace('_', ' ')} is a component of the pressure tensor, and "
                "MMFF94 is 26 per cent low on a density here; a tensor built from a force "
                f"field that wrong is not worth the wall clock. OPLS-AA declined: {fallback_reason}"
            )
        if protocol == "surface_tension":
            from formulate.physics.md.interface import minimum_slab_molecules
            from rdkit.Chem import Descriptors

            needed = minimum_slab_molecules(
                float(Descriptors.MolWt(mol)), self.policy.assumed_density
            )
            if molecules < needed:
                return None, (
                    f"a slab of {molecules} molecules of this compound is under four "
                    f"cutoffs thick, so it has two interfaces and no bulk liquid between "
                    f"them; {needed} are needed, and the policy allows {molecules}"
                )

        try:
            if protocol == "self_diffusion":
                density = self.policy.assumed_density
                result = condensed.run_self_diffusion(
                    mol, molecules, temperature, density_g_cm3=density,
                    force_field=force_field,
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
            elif protocol in ("surface_tension", "shear_viscosity"):
                # Both are run at the density the liquid chooses for itself,
                # not at an assumed one. A slab held at the wrong density has
                # the wrong tension, and a viscosity is exponential in density,
                # so the constant-pressure run in front of these is not
                # preparation - it is part of the measurement.
                liquid = condensed.run_npt(
                    mol, molecules, temperature, force_field=force_field
                )
                caveats: tuple[str, ...] = ()
                if protocol == "surface_tension":
                    from formulate.physics.md import interface

                    result = interface.surface_tension(
                        mol, molecules, temperature, liquid.density_g_cm3, seed=11
                    )
                    value = Quantity(
                        value=result.surface_tension_mn_m * 1e-3, unit="N/m"
                    )
                    uncertainty = Uncertainty(
                        std=result.surface_tension_error * 1e-3,
                        kind=UncertaintyKind.SAMPLING,
                        basis=(
                            f"block-averaged over {result.production_ps:.0f} ps of a "
                            f"{result.box_nm[0]:.1f} x {result.box_nm[1]:.1f} x "
                            f"{result.box_nm[2]:.1f} nm slab holding "
                            f"{result.liquid_nm:.1f} nm of liquid"
                        ),
                    )
                    # A slab's capillary-wave bias arrives as a note rather
                    # than a diagnostic, so it reaches the reader without
                    # marking every box this size out of domain; the
                    # systematic term is what covers it.
                    caveats = result.notes
                else:
                    result = condensed.run_shear_viscosity(
                        mol,
                        molecules,
                        temperature,
                        density_g_cm3=liquid.density_g_cm3,
                        force_field=force_field,
                        seed=13,
                    )
                    value = Quantity(value=result.viscosity_pa_s, unit="Pa*s")
                    uncertainty = Uncertainty(
                        std=result.error_pa_s,
                        kind=UncertaintyKind.SAMPLING,
                        basis=(
                            "spread of the three independent shear components over a "
                            f"plateau read from {result.plateau_window_ps[0]:.1f} to "
                            f"{result.plateau_window_ps[1]:.1f} ps"
                        ),
                    )
                diagnostics = liquid.diagnostics + result.diagnostics + caveats
                in_domain = not result.diagnostics
            else:
                liquid = condensed.run_npt(
                    mol, molecules, temperature, force_field=force_field
                )
                if protocol == "density":
                    sampling = (
                        f"block-averaged over {liquid.production_ps:.0f} ps at constant "
                        "pressure"
                    )
                    if target.property == "liquid_density":
                        value = Quantity(value=liquid.density_g_cm3, unit="g/cm^3")
                        std = liquid.density_error
                    else:
                        # The same box, read the other way round. A molar volume
                        # is what one mole of the stuff occupies, so it is the
                        # molar mass over the density, and its relative error is
                        # the density's - the molar mass is exact.
                        value = Quantity(
                            value=liquid.molar_volume_cm3 * 1e-6, unit="m^3/mol"
                        )
                        std = (
                            value.value * liquid.density_error / liquid.density_g_cm3
                            if liquid.density_g_cm3
                            else None
                        )
                    uncertainty = Uncertainty(
                        std=std, kind=UncertaintyKind.SAMPLING, basis=sampling
                    )
                    diagnostics = liquid.diagnostics
                    in_domain = not liquid.diagnostics
                else:
                    gas, gas_error = condensed.sample_isolated_energy(
                        mol, temperature, force_field=force_field
                    )
                    cohesive = condensed.cohesive_energy_density(liquid, gas, gas_error)
                    sampling = (
                        f"propagated from a vaporisation energy of "
                        f"{cohesive.vaporisation_energy:.1f} kJ/mol and a molar volume of "
                        f"{cohesive.molar_volume_cm3:.1f} cm^3/mol"
                    )
                    relative = (
                        abs(cohesive.error_pa / cohesive.cohesive_energy_density_pa)
                        if cohesive.cohesive_energy_density_pa
                        else 0.0
                    )
                    if target.property == "cohesive_energy_density":
                        value = Quantity(value=cohesive.cohesive_energy_density_pa, unit="Pa")
                        std = cohesive.error_pa
                    elif target.property == "enthalpy_vaporization":
                        # The run measures the potential energy of vaporisation.
                        # The enthalpy is that plus the work of expanding into
                        # the vapour, which for an ideal gas is RT and for the
                        # liquid it left behind is negligible.
                        joules = (
                            cohesive.vaporisation_energy + _GAS_CONSTANT_KJ * temperature
                        ) * 1000.0
                        value = Quantity(value=joules, unit="J/mol")
                        std = (
                            math.hypot(liquid.energy_error, gas_error) * 1000.0
                            if gas_error is not None
                            else None
                        )
                    else:
                        # Hildebrand's parameter is the square root of the
                        # cohesive energy density, and a square root halves a
                        # relative error rather than preserving it.
                        density_pa = max(cohesive.cohesive_energy_density_pa, 0.0)
                        value = Quantity(value=math.sqrt(density_pa), unit="Pa^0.5")
                        std = 0.5 * relative * value.value
                    uncertainty = Uncertainty(
                        std=std, kind=UncertaintyKind.SAMPLING, basis=sampling
                    )
                    diagnostics = cohesive.diagnostics
                    in_domain = not cohesive.diagnostics
        except Exception as exc:  # a failed simulation is a skip, not a crash
            return None, f"the condensed-phase run failed ({type(exc).__name__}: {exc})"

        uncertainty = _widen_for_force_field(uncertainty, value, force_field, target.property)
        notes = _CONDENSED_NOTES[force_field]
        if fallback_reason:
            notes = notes + (
                f"OPLS-AA would have been the more accurate choice and was not used: "
                f"{fallback_reason}",
            )
        if target.property in UNMEASURED_SYSTEMATIC:
            notes = notes + (
                f"the force field's systematic error on a {target.property.replace('_', ' ')} "
                "was not measured here, so the quoted uncertainty is sampling error only "
                "and is a floor rather than an estimate",
            )

        return (
            physics_prediction(
                target.property,
                value,
                uncertainty,
                backend="md:openmm",
                method=(
                    f"{_FORCE_FIELD_NAMES[force_field]} periodic {protocol}, "
                    f"{molecules} molecules"
                ),
                conditions=spec.conditions,
                notes=notes + tuple(diagnostics),
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


def _widen_for_force_field(
    uncertainty: Uncertainty, value: Quantity, force_field: str, prop: str
) -> Uncertainty:
    """Add the force field's measured systematic error to the sampling error.

    Reporting the sampling error alone is the overconfidence the calibration
    suite is built to catch. A hundred picoseconds of block averaging puts a
    few parts in a thousand on an MMFF94 density; the force field puts
    twenty-six per cent on it. The two are independent, so they add in
    quadrature, and the basis string names both so the number can be argued
    with rather than taken on trust.
    """
    systematic_fraction = CONDENSED_SYSTEMATIC.get(force_field, {}).get(prop)
    if systematic_fraction is None:
        return uncertainty

    systematic = abs(value.value) * systematic_fraction
    sampling = uncertainty.std or 0.0
    return Uncertainty(
        std=math.hypot(sampling, systematic),
        kind=UncertaintyKind.EPISTEMIC,
        basis=(
            f"{uncertainty.basis}; widened in quadrature by {_FORCE_FIELD_NAMES[force_field]}'s "
            f"measured {systematic_fraction * 100:.1f} per cent systematic error on this "
            "observable, which dominates the sampling error and is what the value is "
            "actually worth"
        ),
    )


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
