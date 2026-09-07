"""Mixture phase behaviour from UNIFAC activity coefficients.

Specification section 12 asks a formulation to be judged on "validity of
recipes, phase/compatibility failures". Until now the only answer available was
a Hansen distance, which measures how alike two liquids are. Alike is not the
question. Two liquids can sit close in Hansen space and still separate, and the
quantity that decides it is the activity coefficient: how much each component
would rather be beside its own kind than its partner.

Modified UNIFAC (Dortmund) supplies that, and it is already installed. Group
assignment does not come from SMARTS matching here - it comes from the DDBST
compilation, keyed by CAS, which is a curated assignment rather than a guessed
one. Structures reach it the same way the Hansen expert reaches its table:
through an InChIKey, because a SMILES lookup fails on the commonest solvents.

Two things this module refuses to do.

The first is the reason it exists in this shape. When a pair of UNIFAC main
groups has no tabulated interaction parameter, the underlying implementation
does not raise - it treats the interaction as zero, which is the value for two
groups that get along perfectly. Trichloroethylene and water share no
parameter, so asking for it returns an infinite-dilution activity coefficient
of 3.2 for a pair that is in truth about four orders of magnitude worse. A
missing parameter therefore produces not an error but a confident prediction of
miscibility for compounds that do not mix. Every pair is checked against the
interaction table before any number is returned.

The second is cloud point and upper critical solution temperature. UNIFAC's
temperature dependence is fitted to vapour-liquid data, and extracting a
demixing temperature from it is wrong by tens to hundreds of kelvin and
sometimes gets the topology backwards. What is reported instead is the
stability of the mixture at the stated temperature, which is what UNIFAC can
actually support.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

#: Gas constant, J/(mol K).
_R = 8.31446261815324

#: Compositions scanned when testing whether a binary mixture is stable.
_STABILITY_GRID = 41


def unifac_available() -> bool:
    try:
        from thermo.unifac import DOUFIP2016, DOUFSG  # noqa: F401
    except Exception:
        return False
    return True


@functools.lru_cache(maxsize=4096)
def unifac_groups(smiles: str) -> tuple[tuple[int, int], ...] | None:
    """Modified-UNIFAC subgroup counts for a structure, or None.

    None means the compilation has no assignment for this compound, which is a
    reason to decline rather than to fragment it here: a hand-rolled
    fragmentation that disagrees with the one the interaction parameters were
    regressed against produces numbers that are internally consistent and
    externally wrong.
    """
    from thermo.unifac import UNIFAC_group_assignment_DDBST

    from .hansen import resolve_cas

    cas = resolve_cas(smiles)
    if cas is None:
        return None
    try:
        assignment = UNIFAC_group_assignment_DDBST(cas, "MODIFIED_UNIFAC")
    except Exception:
        return None
    if not assignment:
        return None
    return tuple(sorted(assignment.items()))


def _main_groups(groups: Sequence[tuple[int, int]]) -> set[int]:
    from thermo.unifac import DOUFSG

    return {DOUFSG[subgroup].main_group_id for subgroup, _ in groups}


def missing_interactions(
    groups: Sequence[Sequence[tuple[int, int]]]
) -> tuple[tuple[int, int], ...]:
    """Main-group pairs the interaction table does not cover.

    A non-empty result means the mixture cannot be evaluated. It must not be
    evaluated anyway: an absent parameter is silently read as zero, which says
    the two groups are perfectly compatible, so the missing case and the
    ideal case are indistinguishable in the output.
    """
    from thermo.unifac import DOUFIP2016, DOUFSG

    present = [_main_groups(g) for g in groups]
    missing: set[tuple[int, int]] = set()
    for index, first in enumerate(present):
        for second in present[index + 1 :]:
            for a in first:
                for b in second:
                    if a == b:
                        continue
                    if b not in DOUFIP2016.get(a, {}) or a not in DOUFIP2016.get(b, {}):
                        missing.add((min(a, b), max(a, b)))
    del DOUFSG
    return tuple(sorted(missing))


def group_name(main_group_id: int) -> str:
    from thermo.unifac import DOUFSG

    for subgroup in DOUFSG.values():
        if subgroup.main_group_id == main_group_id:
            return str(subgroup.main_group)
    return f"main group {main_group_id}"


@dataclass(frozen=True, slots=True)
class ActivityResult:
    """Activity coefficients and what they say about the mixture."""

    gammas: tuple[float, ...]
    #: RT * sum(x_i ln gamma_i), J/mol.
    excess_gibbs: float
    #: Minimum curvature of the Gibbs energy of mixing over composition, in RT.
    #: Negative means a composition exists at which the mixture separates.
    stability: float
    #: Composition at which that minimum occurs, for a binary.
    least_stable_fraction: float | None
    temperature_k: float
    notes: tuple[str, ...] = ()


def _gammas(groups, fractions, temperature: float) -> list[float]:
    from thermo.unifac import DOUFIP2016, DOUFSG, UNIFAC

    model = UNIFAC.from_subgroups(
        T=temperature,
        xs=list(fractions),
        chemgroups=[dict(g) for g in groups],
        subgroups=DOUFSG,
        interaction_data=DOUFIP2016,
        version=1,
    )
    return list(model.gammas())


def evaluate(
    groups: Sequence[Sequence[tuple[int, int]]],
    fractions: Sequence[float],
    temperature: float,
) -> ActivityResult | None:
    """Activity coefficients at the stated composition, plus phase stability.

    Stability is the second derivative of the molar Gibbs energy of mixing with
    respect to composition. Where it is negative the mixture lowers its energy
    by separating, which is the spinodal condition and is the honest way to
    answer "do these mix" from an activity model. It is scanned across
    composition rather than evaluated only at the requested one, because a
    formulation that is stable as written can still sit next to a composition
    that is not.
    """
    if len(groups) < 2:
        return None
    if missing_interactions(groups):
        return None

    gammas = _gammas(groups, fractions, temperature)
    excess = _R * temperature * sum(
        x * math.log(g) for x, g in zip(fractions, gammas) if x > 0 and g > 0
    )

    stability = float("inf")
    least_stable = None
    if len(groups) == 2:
        # g(x) = x ln(x gamma1) + (1-x) ln((1-x) gamma2), in units of RT.
        xs = np.linspace(0.02, 0.98, _STABILITY_GRID)
        energies = []
        for x in xs:
            g1, g2 = _gammas(groups, [float(x), float(1.0 - x)], temperature)
            energies.append(x * math.log(x * g1) + (1 - x) * math.log((1 - x) * g2))
        energies = np.array(energies)
        step = xs[1] - xs[0]
        curvature = (energies[2:] - 2 * energies[1:-1] + energies[:-2]) / step**2
        index = int(np.argmin(curvature))
        stability = float(curvature[index])
        least_stable = float(xs[index + 1])

    notes: list[str] = []
    if stability < 0:
        notes.append(
            f"the mixture is unstable near x = {least_stable:.2f} and separates into two "
            "phases there; the composition asked about may still be single-phase"
        )
    return ActivityResult(
        gammas=tuple(gammas),
        excess_gibbs=excess,
        stability=stability,
        least_stable_fraction=least_stable,
        temperature_k=temperature,
        notes=tuple(notes),
    )

# --------------------------------------------------------------------------
# The expert
# --------------------------------------------------------------------------


from formulate.core.candidate import Candidate, FractionBasis, MaterialClass  # noqa: E402
from formulate.core.prediction import Prediction  # noqa: E402
from formulate.core.properties import PropertyFamily  # noqa: E402
from formulate.core.provenance import SoftwareEnvironment  # noqa: E402
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind  # noqa: E402

from .base import Expert  # noqa: E402


class UNIFACActivityExpert(Expert):
    """Excess Gibbs energy and phase stability of a liquid mixture.

    Kept separate from :class:`~formulate.experts.mixture.MixtureExpert`
    because it answers a different question. That one evaluates each component
    through the molecular panel and applies a mixing rule; this one asks what
    the components do *to each other*, which no average over pure-component
    properties can produce. A Hansen distance says two liquids are alike; an
    activity coefficient says whether they mix.
    """

    id = "unifac"
    version = "1"
    method = "modified UNIFAC (Dortmund) over DDBST group assignments"
    family = PropertyFamily.CHEMICAL
    supported_classes = frozenset({MaterialClass.MIXTURE})
    supported_properties = frozenset({"excess_gibbs_energy", "mixing_stability"})

    def __init__(self, registry=None) -> None:
        self._registry = registry

    def is_available(self) -> bool:
        from formulate import chem

        return unifac_available() and chem.rdkit_available()

    def unavailable_reason(self) -> str:
        if not unifac_available():
            return "the thermo package does not expose modified UNIFAC here"
        return "RDKit is required to resolve a structure to its group assignment"

    def _software(self) -> SoftwareEnvironment:
        import thermo

        return SoftwareEnvironment.capture(thermo=thermo.__version__)

    def registry(self):
        if self._registry is None:
            from formulate.experts import molecular_registry

            self._registry = molecular_registry()
        return self._registry

    # -- composition -------------------------------------------------------

    def _mole_fractions(self, mixture, request) -> tuple[list[float] | None, str]:
        """Mole fractions, converting from whichever basis the recipe states.

        A recipe written by volume is the common case for a solvent blend, and
        converting it needs component densities. Those come from the molecular
        panel rather than from a guess, so that a component's density here is
        the same number it would have had as a candidate in its own right.
        """
        from rdkit.Chem import Descriptors

        from formulate import chem

        fractions = [c.fraction for c in mixture.components]
        if mixture.basis is FractionBasis.MOLE:
            return fractions, ""

        masses: list[float] = []
        for component in mixture.components:
            if component.molecule is None:
                return None, "a component is not a small molecule, so it has no group assignment"
            mol = chem.mol_from_smiles(component.molecule.smiles)
            if mol is None:
                return None, "a component structure could not be parsed"
            masses.append(float(Descriptors.MolWt(mol)))

        if mixture.basis is FractionBasis.MASS:
            moles = [f / m for f, m in zip(fractions, masses)]
        else:
            densities = self._component_densities(mixture, request)
            if densities is None:
                return None, (
                    "this recipe is stated by volume and at least one component has no "
                    "density, so it cannot be converted to the mole basis UNIFAC needs"
                )
            moles = [f * d / m for f, d, m in zip(fractions, densities, masses)]

        total = sum(moles)
        if total <= 0:
            return None, "the composition came out non-positive"
        return [m / total for m in moles], ""

    def _component_densities(self, mixture, request) -> list[float] | None:
        """Component densities from the molecular panel, dependencies resolved.

        The panel's density comes from a correlation that needs critical
        constants from another expert, so the experts have to run in dependency
        order with each one's output fed to the next. Calling them individually
        returns UNSUPPORTED for a missing dependency, which is correct
        behaviour and useless here.
        """
        from formulate.experts.mixture import MixtureExpert

        blender = MixtureExpert(self.registry())
        out: list[float] = []
        for component in mixture.components:
            properties = blender.component_properties(component, request)
            if properties.density_kg_m3 is None:
                return None
            out.append(properties.density_kg_m3 / 1000.0)  # kg/m^3 -> g/cm^3
        return out

    # -- prediction --------------------------------------------------------

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        basis = "modified UNIFAC (Dortmund), fitted to vapour-liquid equilibrium data"
        mixture = candidate.mixture
        if mixture is None or len(mixture.components) < 2:
            return ApplicabilityDomain.outside("a mixture needs at least two components", basis=basis)
        if any(c.polymer is not None for c in mixture.components):
            return ApplicabilityDomain.outside(
                "UNIFAC's groups and interaction parameters are fitted to small molecules; "
                "a polymer needs the free-volume terms of UNIFAC-FV, which is not here",
                basis=basis,
            )
        return ApplicabilityDomain(basis=basis)

    def _predict_one(self, prop, request, domain):
        mixture = request.candidate.mixture
        if mixture is None:
            return Prediction.unsupported(prop, self.id, "candidate carries no mixture")

        assignments = []
        for component in mixture.components:
            if component.molecule is None:
                return Prediction.unsupported(
                    prop, self.id, "a component is not a small molecule"
                )
            groups = unifac_groups(component.molecule.smiles)
            if groups is None:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"the DDBST compilation has no UNIFAC assignment for "
                    f"{component.molecule.smiles!r}",
                )
            assignments.append(groups)

        missing = missing_interactions(assignments)
        if missing:
            pairs = ", ".join(f"{group_name(a)}/{group_name(b)}" for a, b in missing)
            return Prediction.unsupported(
                prop,
                self.id,
                f"no interaction parameter is tabulated for {pairs}. An absent parameter "
                "is read as zero, which is the value for two groups that mix perfectly, "
                "so evaluating this would return confident miscibility for a pair that "
                "may not mix at all",
            )

        fractions, reason = self._mole_fractions(mixture, request)
        if fractions is None:
            return Prediction.unsupported(prop, self.id, reason)

        temperature = request.conditions.temperature_k
        if temperature is None:
            return Prediction.unsupported(
                prop, self.id, "an activity coefficient is temperature dependent and none was given"
            )

        result = evaluate(assignments, fractions, temperature)
        if result is None:
            return Prediction.unsupported(prop, self.id, "the activity model did not evaluate")

        if prop == "excess_gibbs_energy":
            value, unit, spread = result.excess_gibbs, "J/mol", 300.0
            basis = (
                "modified UNIFAC reproduces excess Gibbs energies of well-covered binaries "
                "to a few hundred J/mol; the spread is larger for aqueous and associating pairs"
            )
        else:
            if not math.isfinite(result.stability):
                return Prediction.unsupported(
                    prop,
                    self.id,
                    "phase stability is scanned over composition, which this "
                    "implementation does only for a binary mixture",
                )
            value, unit, spread = result.stability, "", 0.5
            basis = (
                "curvature of the Gibbs energy of mixing on a 41-point composition scan; "
                "the sign is the spinodal condition and is more reliable than the magnitude"
            )

        return self._make(
            prop,
            value,
            unit,
            request,
            domain,
            std=spread,
            kind=UncertaintyKind.EPISTEMIC,
            basis=basis,
            notes=tuple(result.notes)
            + (
                "activity coefficients: "
                + ", ".join(f"{g:.3f}" for g in result.gammas),
                "no cloud point or upper critical solution temperature is derived from "
                "this: UNIFAC's temperature dependence is fitted to vapour-liquid data "
                "and a demixing temperature taken from it is wrong by tens to hundreds "
                "of kelvin",
            ),
            temperature_k=round(temperature, 3),
        )
