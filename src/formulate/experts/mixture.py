"""Formulation properties from component properties.

Specification section 3 requires mixtures to carry composition and component
roles; section 12 asks for "validity of recipes, phase/compatibility failures,
sensitivity to composition/process variables".

A formulation expert cannot predict a blend from its own descriptors: it needs
each component's properties first.  Rather than duplicating the molecular
panel, this expert evaluates each component *as a molecule* through the
existing expert registry and then applies mixing rules to what comes back.
The delegation is the design: it means a component's density here is the same
number, from the same method with the same uncertainty, that it would have had
as a candidate in its own right.

Every mixing rule below is an ideal-mixing approximation.  Section 13 forbids
presenting these as validated bulk behaviour, and each prediction says which
assumption it rests on.
"""

from __future__ import annotations

from dataclasses import dataclass

from formulate.core.candidate import (
    Candidate,
    FractionBasis,
    MaterialClass,
    MixtureComponent,
    MoleculeSpec,
    molecule_candidate,
)
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest
from .hansen import hansen_distance, hansen_triple

_HANSEN_PROPERTIES = ("hansen_dispersion", "hansen_polar", "hansen_hydrogen_bonding")


@dataclass(frozen=True, slots=True)
class ComponentProperties:
    """What is known about one component of a formulation."""

    smiles: str
    role: str
    fraction: float
    molar_mass_g_mol: float | None = None
    density_kg_m3: float | None = None
    hansen: tuple[float, float, float] | None = None

    @property
    def molar_volume_cm3(self) -> float | None:
        if self.molar_mass_g_mol is None or not self.density_kg_m3:
            return None
        # g/mol / (g/cm^3) = cm^3/mol
        return self.molar_mass_g_mol / (self.density_kg_m3 / 1000.0)


class MixtureExpert(Expert):
    """Applies mixing rules over component properties."""

    id = "mixture"
    version = "1"
    method = "ideal mixing rules over component properties from the molecular panel"
    family = PropertyFamily.INTERFACIAL
    supported_classes = frozenset({MaterialClass.MIXTURE})
    supported_properties = frozenset(
        set(_HANSEN_PROPERTIES) | {"liquid_density", "hansen_distance"}
    )

    def __init__(self, registry=None) -> None:
        self._registry = registry

    def registry(self):
        """The molecular panel used to evaluate components.

        Constructed lazily to avoid an import cycle: the default registry
        imports every expert, including this one.
        """
        if self._registry is None:
            from formulate.experts import molecular_registry

            self._registry = molecular_registry()
        return self._registry

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "RDKit is required to read component structures"

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        mixture = candidate.mixture
        if mixture is None:
            return ApplicabilityDomain.outside("candidate carries no mixture")

        warnings: list[str] = []
        score = 1.0
        if any(component.polymer is not None for component in mixture.components):
            warnings.append(
                "a polymer component is present; these rules treat every component as a "
                "small-molecule liquid, and a polymer's contribution to a blend is not "
                "volume-additive in the same way"
            )
            score = min(score, 0.4)
        if mixture.phase_assumption.value in ("emulsion", "dispersion", "suspension"):
            warnings.append(
                f"the formulation is declared a {mixture.phase_assumption.value}, which is "
                "by definition not a single homogeneous phase; mixing rules describe the "
                "hypothetical single phase and not the dispersed system"
            )
            score = min(score, 0.3)
        return ApplicabilityDomain(
            score=score,
            in_domain=score > 0.3,
            warnings=tuple(warnings),
            basis="homogeneous liquid blends of small molecules near ideal mixing",
        )

    # -- component evaluation ---------------------------------------------

    def component_properties(
        self, component: MixtureComponent, request: PredictionRequest
    ) -> ComponentProperties:
        """Evaluate one component through the molecular panel."""
        from formulate.experts.base import PredictionRequest as Req

        payload = component.molecule or component.polymer
        smiles_list = component.all_smiles()
        smiles = smiles_list[0] if smiles_list else ""

        molar_mass = density = None
        triple = hansen_triple(smiles) if smiles else None
        if isinstance(payload, MoleculeSpec) and smiles:
            sub = molecule_candidate(smiles, conditions=request.conditions)
            registry = self.registry()
            wanted = frozenset(
                {
                    "molar_mass",
                    "liquid_density",
                    "critical_temperature",
                    "critical_pressure",
                    "critical_volume",
                    "normal_boiling_point",
                    "hansen_dispersion",
                    "hansen_polar",
                    "hansen_hydrogen_bonding",
                }
            )
            context: dict[str, Prediction] = {}
            for expert in registry.resolution_order(
                registry.experts_for(wanted, MaterialClass.MOLECULE)
            ):
                sub_request = Req(
                    candidate=sub,
                    properties=wanted,
                    conditions=request.conditions,
                    context=dict(context),
                )
                for prediction in expert.predict(sub_request):
                    if prediction.is_usable:
                        context.setdefault(prediction.property, prediction)

            mass = context.get("molar_mass")
            if mass is not None and mass.quantity is not None:
                molar_mass = mass.quantity.to("g/mol").value
            rho = context.get("liquid_density")
            if rho is not None and rho.quantity is not None:
                density = rho.quantity.to("kg/m^3").value

            # Whatever the component panel resolved, which may be a group
            # estimate. Going straight to the compilation instead meant a
            # blend was scoreable only when every component was already in a
            # handbook, and a designed formulation is mostly things that are
            # not: a specialty monomer, an initiator, an accelerator. One gap
            # took out the whole mixture.
            axes = tuple(
                context.get(f"hansen_{axis}")
                for axis in ("dispersion", "polar", "hydrogen_bonding")
            )
            if all(a is not None and a.quantity is not None for a in axes):
                triple = tuple(a.quantity.to("Pa^0.5").value for a in axes)  # type: ignore[union-attr]

        return ComponentProperties(
            smiles=smiles,
            role=component.role.value,
            fraction=component.fraction,
            molar_mass_g_mol=molar_mass,
            density_kg_m3=density,
            hansen=triple,
        )

    def volume_fractions(
        self, components: list[ComponentProperties], basis: FractionBasis
    ) -> list[float] | None:
        """Convert the declared fractions to volume fractions.

        Every mixing rule below is volume-based, and a mass fraction used where
        a volume fraction is required is a silent error of tens of percent
        whenever the components differ in density.
        """
        if basis is FractionBasis.VOLUME:
            return [c.fraction for c in components]

        volumes: list[float] = []
        for component in components:
            molar_volume = component.molar_volume_cm3
            if molar_volume is None or molar_volume <= 0:
                return None
            if basis is FractionBasis.MASS:
                if not component.molar_mass_g_mol:
                    return None
                # mass fraction / density gives a volume, up to a common factor.
                volumes.append(component.fraction / (component.density_kg_m3 / 1000.0))
            else:  # mole basis
                volumes.append(component.fraction * molar_volume)

        total = sum(volumes)
        return [v / total for v in volumes] if total > 0 else None

    # -- prediction --------------------------------------------------------

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        mixture = request.candidate.mixture
        if mixture is None:
            return Prediction.failed(prop, self.id, "candidate carries no mixture")

        components = [self.component_properties(c, request) for c in mixture.components]

        if prop == "hansen_distance":
            return self._hansen_distance(prop, components, request, domain)
        if prop in _HANSEN_PROPERTIES:
            return self._blend_hansen(prop, components, mixture.basis, request, domain)
        return self._blend_density(prop, components, mixture.basis, request, domain)

    def _blend_hansen(
        self, prop, components, basis, request, domain
    ) -> Prediction | None:
        missing = [c.smiles for c in components if c.hansen is None]
        if missing:
            return Prediction.unsupported(
                prop,
                self.id,
                (
                    "not every component has tabulated Hansen parameters "
                    f"({len(missing)} of {len(components)} missing), and a blend average "
                    "over a partial set would silently describe a different formulation"
                ),
            )

        fractions = self.volume_fractions(components, basis)
        if fractions is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "component densities were unavailable, so volume fractions could not be formed",
            )

        index = _HANSEN_PROPERTIES.index(prop)
        value = sum(f * c.hansen[index] for f, c in zip(fractions, components))
        return self._make(
            prop,
            value,
            "Pa^0.5",
            request,
            domain,
            std=800.0,
            kind=UncertaintyKind.COMBINED,
            basis=(
                "volume-fraction average of tabulated component values; the averaging "
                "assumes ideal mixing and adds to each component's own tabulation spread"
            ),
            notes=(
                "volume-fraction weighted; a mass-fraction average would differ by tens of "
                "percent where component densities differ",
            ),
        )

    def _hansen_distance(self, prop, components, request, domain) -> Prediction | None:
        usable = [c for c in components if c.hansen is not None]
        if len(usable) < 2:
            return Prediction.unsupported(
                prop,
                self.id,
                "at least two components with tabulated Hansen parameters are needed "
                "to form a distance",
            )

        worst = 0.0
        pair = ("", "")
        for i, first in enumerate(usable):
            for second in usable[i + 1:]:
                distance = hansen_distance(first.hansen, second.hansen)
                if distance > worst:
                    worst, pair = distance, (first.smiles, second.smiles)

        return self._make(
            prop,
            worst,
            "Pa^0.5",
            request,
            domain,
            std=1000.0,
            kind=UncertaintyKind.COMBINED,
            basis=(
                "propagated from the tabulation spread of the two components involved"
            ),
            notes=(
                f"the least compatible pair is {pair[0]} and {pair[1]}",
                (
                    "a distance alone does not decide miscibility: that needs the "
                    "interaction radius R0 of the material in question, which is measured "
                    "per polymer and is not defined for a solvent pair, so no relative "
                    "energy difference is reported here"
                ),
            ),
        )

    def _blend_density(self, prop, components, basis, request, domain) -> Prediction | None:
        if any(c.density_kg_m3 is None for c in components):
            return Prediction.unsupported(
                prop, self.id, "not every component has a predicted liquid density"
            )
        fractions = self.volume_fractions(components, basis)
        if fractions is None:
            return Prediction.unsupported(
                prop, self.id, "volume fractions could not be formed"
            )

        value = sum(f * c.density_kg_m3 for f, c in zip(fractions, components))
        return self._make(
            prop,
            value,
            "kg/m^3",
            request,
            domain,
            std=abs(value) * 0.08,
            kind=UncertaintyKind.COMBINED,
            basis=(
                "volume additivity, which ignores the excess volume of mixing; that "
                "excess is typically under a few percent for similar liquids and larger "
                "for strongly associating pairs such as water with an alcohol. Combined "
                "with the error already carried by each component's own density"
            ),
            notes=(
                "ideal mixing: the blend is assumed to occupy the sum of its components' "
                "volumes",
            ),
        )
