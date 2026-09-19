"""A polymer dissolved in a solvent: the spinning dope.

This is the material a web shooter actually holds. A cartridge carries a
fluid, the fluid leaves a hole, and what lands is a solid - which is dry
spinning, the route cellulose acetate and acrylic fibres have been made by
since the 1920s. The engine could not score one. ``polymer_blend_melt``
refuses a formulation containing a molecule, correctly, because its mixing
rules are over polymer components; so a 20% polystyrene in acetone came back
with eight of nine properties unanswered and the whole spinline chain - the
relaxation time, the strain hardening, the thinning ratio, the extrusion
pressure, the filament stability - collapsed behind a viscosity nobody
supplied.

Everything needed is already here and none of it is a lookup:

* the chain dimension from :func:`mechanical.predicted_chain_dimension`,
  which is the characteristic ratio from side-group bulk
* the solvent's own viscosity from ``liquid_transport``
* both densities, from ``polymer_density`` and ``interfacial``

**Intrinsic viscosity** comes from Flory-Fox, ``[eta] = Phi <R^2>^1.5 / M``,
which is the relation the chain dimension exists to feed. It is the one place
a solution viscosity is usually a table - Mark-Houwink ``K`` and ``a`` are
quoted per polymer, per solvent and per temperature - and going through the
chain dimension instead means any repeat unit can be asked.

Checked against intrinsic viscosities computed from published Mark-Houwink
constants at theta conditions, the Flory-Fox route lands between 0.88x and
1.60x: polyethylene 1.03, polystyrene 0.88 at both 100 and 200 kg/mol, PMMA
1.23, polypropylene 1.60. That is a screening number, and the prediction says
so.

**Concentration regime** decides which law applies, and the switch is
``c[eta]``, the coil overlap parameter. Below one the coils are separate and
Huggins applies; above one they interpenetrate and the viscosity climbs as a
high power of concentration. The two branches are joined at ``c[eta] = 1``,
where both hold - the same construction ``melt.py`` uses to join Rouse to
reptation at the critical mass, and for the same reason: an extrapolation of
one law into the other's regime is not a model of anything.

**What this expert does NOT do** is decide whether the polymer dissolves at
all. It assumes it does, states that assumption on every prediction, and
points at ``polymer_dissolution``, which answers the solubility question and
is honest about the part of it that is not derivable. A viscosity for a
suspension of undissolved polymer would be a number about a different
material.
"""

from __future__ import annotations

import math

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MixtureSpec,
    MonomerRole,
)
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest

#: Flory's constant in ``[eta] = Phi <R^2>^1.5 / M``, mol^-1.
#:
#: The universal value for a linear flexible chain under theta conditions,
#: with the intrinsic viscosity in cm^3/g and the mean-square end-to-end
#: distance in cm^2. It is universal only approximately - it drifts with chain
#: stiffness and with solvent quality, which is part of why the spread below
#: is what it is.
FLORY_CONSTANT = 2.5e23

#: Worst ratio of the Flory-Fox route against intrinsic viscosities computed
#: from published Mark-Houwink constants at theta conditions. Measured here
#: over polyethylene (1.03), polystyrene at two chain lengths (0.88), PMMA
#: (1.23) and polypropylene (1.60, the worst).
INTRINSIC_VISCOSITY_SPREAD = 1.61

#: How much a good solvent expands a coil over its theta dimension, as a
#: factor on the intrinsic viscosity.
#:
#: This is NOT applied. The theta value is what Flory-Fox gives, a good
#: solvent swells the coil, and ``[eta]`` goes as the cube of the expansion
#: factor - which for an ordinary polymer/good-solvent pair is between one and
#: about two. Rather than invent a solvent-quality model, the range is carried
#: as uncertainty and the notes say which end a given answer sits at. Deciding
#: it properly needs the Hansen distance AND an interaction radius, and this
#: repository has measured that the radius is not derivable from structure -
#: see ``polymer_dissolution``.
GOOD_SOLVENT_EXPANSION = 2.0

#: Huggins coefficient. Runs about 0.3 in a good solvent to 0.8 at theta; the
#: middle is used and the range is inside the spread already carried.
HUGGINS_COEFFICIENT = 0.4

#: Exponent on the overlap parameter above ``c[eta] = 1``.
#:
#: NOT the melt expert's 3.4, and the difference matters. That 3.4 is the
#: exponent in CHAIN LENGTH at fixed concentration - a melt is already at 100%
#: polymer and the only variable left is how long the chains are. Here the
#: variable is concentration, and raising it does two things at once: it adds
#: entanglements and it shrinks the screening length, so the coils contract
#: as well. Scaling theory gives 3.9 in a good solvent and about 4.7 at theta
#: (Rubinstein and Colby, Polymer Physics, 2003, ch. 9), and 4.3 is the middle
#: of that with the range carried as the spread.
#:
#: Using 3.4 here is what the first version of this module did, and it put a
#: 20% solution of 200 kg/mol polystyrene at 0.22 Pa s - thinner than a
#: spinning dope can be, since the whole point of a dope is that it is
#: viscous enough to draw.
ENTANGLED_EXPONENT = 4.3
ENTANGLED_EXPONENT_SPREAD = 0.4

#: One-sigma on the answer, in decades, before the inputs' own errors.
#: The solvent viscosity arrives with a bar of its own, and the intrinsic
#: viscosity with the Flory-Fox spread above; this is what the two-branch
#: concentration law adds on top of them.
REGIME_LOG_SIGMA = 0.3


#: Exponent in the entanglement dilution law, ``Me(phi) = Me / phi^alpha``.
#:
#: Solvent does not join the network, it dilutes it: the surviving
#: entanglements are pushed further apart, so the molar mass between them
#: rises as the polymer fraction falls. This is the same law - and the same
#: constant - ``blend.py`` applies to a short chain diluting a long one, for
#: exactly the same reason, and a solvent is the limiting case of a diluent.
#:
#: It is why a dope can carry a chain long enough to strain-harden and still
#: be pumpable: dilution buys the processability that chain length costs.
ENTANGLEMENT_DILUTION_EXPONENT = 1.0

#: A polymer solution's surface tension is its SOLVENT's, to within a few
#: mN/m, and the reason is not that the polymer is a small term - it is that
#: the lower-surface-tension component enriches at the surface, and at any
#: ordinary dope concentration that is the solvent. The gap between the two
#: components is carried as uncertainty rather than mixed, because a mixing
#: rule would have the polymer pulling the surface tension toward its own
#: value, which is the wrong direction.
SURFACE_ENRICHMENT_NOTE = (
    "the solvent's, because the lower-surface-tension component enriches at the surface "
    "and at a dope concentration that is the solvent"
)

def intrinsic_viscosity(r2_per_mass: float, molar_mass: float) -> float:
    """Theta-state intrinsic viscosity in dL/g, from Flory-Fox.

    ``r2_per_mass`` is ``<R^2>/M`` in A^2 per (g/mol), which is what the
    mechanical expert carries; ``molar_mass`` is in g/mol.
    """
    r2_cm2 = r2_per_mass * molar_mass * 1e-16
    return FLORY_CONSTANT * r2_cm2**1.5 / molar_mass / 100.0


def overlap_concentration(intrinsic: float) -> float:
    """``c* = 1/[eta]`` in g/dL: where the coils start to touch.

    A definition rather than a measurement, and the usual one. Other
    conventions put a numerical factor in front; none of them move the
    crossover by enough to matter against the spread carried here.
    """
    return 1.0 / intrinsic


def solution_viscosity(
    solvent_viscosity: float,
    intrinsic: float,
    concentration: float,
    *,
    exponent: float = ENTANGLED_EXPONENT,
) -> tuple[float, str]:
    """Viscosity of the solution in Pa s, and which branch produced it.

    Two laws joined where both hold. ``concentration`` is in g/dL and
    ``intrinsic`` in dL/g, so their product is the dimensionless overlap
    parameter that decides the regime.
    """
    overlap = concentration * intrinsic
    dilute = 1.0 + overlap + HUGGINS_COEFFICIENT * overlap**2
    if overlap <= 1.0:
        return solvent_viscosity * dilute, "dilute"
    # Joined at overlap = 1, where the Huggins form is still valid and the
    # power law starts. Continuous by construction; the slope is not, and is
    # not claimed to be.
    at_overlap = 1.0 + 1.0 + HUGGINS_COEFFICIENT
    return solvent_viscosity * at_overlap * overlap**exponent, "entangled"


class PolymerSolutionExpert(Expert):
    """Shear viscosity of a polymer dissolved in a solvent."""

    id = "polymer_solution"
    version = "1"
    method = (
        "Flory-Fox intrinsic viscosity from the predicted chain dimension, with a "
        "Huggins dilute branch joined at coil overlap to an entangled power law "
        "(Flory, Principles of Polymer Chemistry, 1953; Graessley, Polymeric Liquids "
        "and Networks, 2004)"
    )
    family = PropertyFamily.INTERFACIAL
    supported_classes = frozenset({MaterialClass.MIXTURE})
    #: The density comes free: the concentration the viscosity depends on is
    #: formed from it, so refusing to report it would mean computing a number
    #: and then throwing it away while the spinline expert downstream refuses
    #: for want of exactly that number.
    supported_properties = frozenset(
        {"shear_viscosity", "liquid_density", "entanglement_molar_mass", "surface_tension"}
    )

    def __init__(self, polymer_registry=None, molecular_registry=None) -> None:
        self._polymer = polymer_registry
        self._molecular = molecular_registry

    # -- panels, built lazily so the import stays one-way -------------------

    def _polymer_panel(self):
        if self._polymer is None:
            from formulate.experts import polymer_registry

            self._polymer = polymer_registry()
        return self._polymer

    def _molecular_panel(self):
        if self._molecular is None:
            from formulate.experts import default_registry

            # The molecular panel alone has no liquid viscosity; the transport
            # expert is Phase 6. Take the full registry and let the material
            # class of the sub-candidate restrict it, which it does.
            self._molecular = default_registry()
        return self._molecular

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "RDKit is required to read component structures"

    # -- shape of the formulation ------------------------------------------

    @staticmethod
    def _split(mixture: MixtureSpec):
        """(polymer components, molecular components), or None if neither shape."""
        polymers = [c for c in mixture.components if c.polymer is not None]
        molecules = [c for c in mixture.components if c.molecule is not None]
        if not polymers or not molecules:
            return None
        if len(polymers) + len(molecules) != len(mixture.components):
            return None
        return polymers, molecules

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        basis = (
            "linear flexible chains in a solvent that dissolves them, from the theta-state "
            "Flory-Fox intrinsic viscosity; checked against Mark-Houwink intrinsic "
            f"viscosities between 0.88x and {INTRINSIC_VISCOSITY_SPREAD:.2f}x"
        )
        mixture = candidate.mixture
        if mixture is None:
            return ApplicabilityDomain.outside("candidate carries no mixture", basis=basis)
        split = self._split(mixture)
        if split is None:
            return ApplicabilityDomain.outside(
                "this is not a polymer solution: it needs at least one polymer component "
                "and at least one molecular one. An all-polymer formulation is a blend, "
                "which polymer_blend_melt answers",
                basis=basis,
            )
        polymers, _ = split
        warnings: list[str] = []
        score = 1.0
        if len(polymers) > 1:
            warnings.append(
                "more than one dissolved polymer: the intrinsic viscosities are combined by "
                "mass fraction, which ignores that two dissolved chains interact with each "
                "other as well as with the solvent"
            )
            score = min(score, 0.5)
        for component in polymers:
            spec = component.polymer
            if spec is None:  # pragma: no cover - guarded by _split
                continue
            backbone = [m for m in spec.monomers if m.role is not MonomerRole.END_GROUP]
            if len(backbone) != 1:
                return ApplicabilityDomain.outside(
                    "a copolymer's chain dimension is not the average of its components', so "
                    "the Flory-Fox route has no dimension to use",
                    basis=basis,
                )
        return ApplicabilityDomain(
            score=score, in_domain=score > 0.3, warnings=tuple(warnings), basis=basis
        )

    # -- delegation ---------------------------------------------------------

    def _component_value(self, registry, candidate, prop, unit, conditions):
        """Run a panel over one component and read one property back."""
        from .base import PredictionRequest as Req

        wanted = {prop}
        for _ in range(len(registry) + 1):
            grown = set(wanted)
            for expert in registry:
                if expert.supported_properties & wanted:
                    grown |= expert.dependencies
            if grown == wanted:
                break
            wanted = grown

        context: dict[str, Prediction] = {}
        for expert in registry.resolution_order(
            registry.experts_for(wanted, candidate.material_class)
        ):
            request = Req(
                candidate=candidate,
                properties=frozenset(wanted),
                conditions=conditions,
                context=dict(context),
            )
            for prediction in expert.predict(request):
                existing = context.get(prediction.property)
                if existing is None or (existing.quantity is None and prediction.is_usable):
                    context[prediction.property] = prediction
        found = context.get(prop)
        if found is None or found.quantity is None:
            return None, None
        value = found.quantity.to(unit).value
        std = found.uncertainty.std if found.uncertainty is not None else None
        if std is not None:
            std = found.quantity.__class__(
                value=std, unit=found.quantity.unit
            ).to(unit).value
        return value, std

    # -- prediction ---------------------------------------------------------

    def _predict_one(self, prop, request: PredictionRequest, domain):
        from .mechanical import predicted_chain_dimension

        candidate = request.candidate
        mixture = candidate.mixture
        if mixture is None:
            return Prediction.unsupported(prop, self.id, "candidate carries no mixture")
        split = self._split(mixture)
        if split is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "this is not a polymer solution: it needs at least one polymer component and "
                "at least one molecular one. An all-polymer formulation is a blend, which "
                "polymer_blend_melt answers, and an all-molecular one is a solvent mixture",
            )
        polymers, molecules = split

        conditions = request.conditions
        temperature = conditions.temperature
        if temperature is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "a solution viscosity is meaningless without a temperature; state one in "
                "the requirement's conditions",
            )

        # -- the solvent, which sets the floor -----------------------------
        molecular = self._molecular_panel()
        solvent_terms, solvent_spreads, solvent_densities = [], [], []
        solvent_mass = sum(c.fraction for c in molecules)
        for component in molecules:
            sub = Candidate(
                material_class=MaterialClass.MOLECULE,
                molecule=component.molecule,
                conditions=candidate.conditions,
            )
            eta, eta_std = self._component_value(
                molecular, sub, "shear_viscosity", "Pa*s", conditions
            )
            if eta is None or eta <= 0:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"the solvent {component.molecule.smiles!r} has no predicted viscosity, "
                    "and a solution viscosity is its solvent's multiplied by what the "
                    "polymer adds - with no floor there is nothing to multiply",
                )
            rho, _ = self._component_value(
                molecular, sub, "liquid_density", "g/cm^3", conditions
            )
            if rho is None or rho <= 0:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"the solvent {component.molecule.smiles!r} has no predicted liquid "
                    "density, so the polymer's concentration in grams per decilitre cannot "
                    "be formed from a mass fraction",
                )
            weight = component.fraction / solvent_mass
            solvent_terms.append((weight, eta))
            solvent_spreads.append((weight, eta_std, eta))
            solvent_densities.append((component.fraction, rho))

        # Log-additive over the solvents, the same rule the blend expert uses.
        solvent_viscosity = math.exp(sum(w * math.log(e) for w, e in solvent_terms))
        solvent_log_sigma = 0.0
        for weight, std, value in solvent_spreads:
            own = 1.0 if std is None else math.log10(1.0 + 2.0 * std / value)
            solvent_log_sigma += weight * own

        # -- the polymer, which sets everything else ------------------------
        polymer_panel = self._polymer_panel()
        intrinsic_terms, polymer_densities = [], []
        polymer_mass = sum(c.fraction for c in polymers)
        for component in polymers:
            spec = component.polymer
            molar_mass = spec.number_average_molar_mass
            if molar_mass is None:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    "a dissolved polymer with no stated number-average molar mass has no "
                    "intrinsic viscosity: [eta] goes as the half power of chain length at "
                    "theta and higher in a good solvent, so the same repeat unit spans "
                    "orders of magnitude",
                )
            backbone = [m for m in spec.monomers if m.role is not MonomerRole.END_GROUP]
            repeat = backbone[0].smiles
            r2_per_mass = predicted_chain_dimension(repeat)
            if r2_per_mass is None or r2_per_mass <= 0:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"no chain dimension could be predicted for {repeat!r}, and Flory-Fox is "
                    "an expression in that dimension - there is nothing to put in it",
                )
            mass_g_mol = molar_mass.to("g/mol").value
            intrinsic_terms.append(
                (component.fraction / polymer_mass, intrinsic_viscosity(r2_per_mass, mass_g_mol))
            )
            sub = Candidate(
                material_class=MaterialClass.POLYMER,
                polymer=spec,
                conditions=candidate.conditions,
            )
            rho, _ = self._component_value(
                polymer_panel, sub, "amorphous_density", "g/cm^3", conditions
            )
            if rho is None or rho <= 0:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    "the dissolved polymer has no predicted density, so the solution's own "
                    "density - and with it the concentration the viscosity depends on - "
                    "cannot be formed",
                )
            polymer_densities.append((component.fraction, rho))

        intrinsic = sum(w * v for w, v in intrinsic_terms)

        # -- concentration --------------------------------------------------
        # Volume-additive: 1/rho = sum(w_i / rho_i). Ideal mixing, which for a
        # polymer in its own good solvent is good to a per cent or two and is
        # nowhere near the dominant error here.
        inverse = sum(f / r for f, r in polymer_densities + solvent_densities)
        solution_density = 1.0 / inverse          # g/cm^3
        concentration = polymer_mass * solution_density * 100.0   # g/dL

        # Polymer VOLUME fraction, which is what dilutes a network; the mass
        # fraction would be wrong by the density ratio.
        polymer_volume = sum(f / r for f, r in polymer_densities)
        solvent_volume = sum(f / r for f, r in solvent_densities)
        volume_fraction = polymer_volume / (polymer_volume + solvent_volume)

        if prop == "entanglement_molar_mass":
            melt_values, melt_spreads = [], []
            for component in polymers:
                sub = Candidate(
                    material_class=MaterialClass.POLYMER,
                    polymer=component.polymer,
                    conditions=candidate.conditions,
                )
                me, me_std = self._component_value(
                    polymer_panel, sub, "entanglement_molar_mass", "kg/mol", conditions
                )
                if me is None or me <= 0:
                    return Prediction.unsupported(
                        prop,
                        self.id,
                        "the dissolved polymer has no melt entanglement molar mass, and a "
                        "solution's is that value diluted - with nothing to dilute there is "
                        "no answer",
                    )
                melt_values.append((component.fraction / polymer_mass, me))
                melt_spreads.append(me_std)
            melt = sum(w * v for w, v in melt_values)
            diluted = melt / volume_fraction**ENTANGLEMENT_DILUTION_EXPONENT
            worst = max((s for s in melt_spreads if s is not None), default=None)
            spread = (
                diluted * (worst / melt) if worst is not None and melt > 0 else diluted * 0.5
            )
            return self._make(
                prop,
                diluted,
                "kg/mol",
                request,
                domain,
                std=spread,
                kind=UncertaintyKind.COMBINED,
                basis=(
                    "the melt value's own bar, carried through the dilution law, which "
                    "scales it rather than widening it"
                ),
                notes=(
                    f"Me = {melt:.2f} kg/mol in the melt, diluted to {diluted:.2f} at a "
                    f"polymer volume fraction of {volume_fraction:.2f}",
                    "solvent does not join the network, it dilutes it: the surviving "
                    "entanglements are pushed further apart. This is why a dope can carry a "
                    "chain long enough to strain-harden and still be pumpable",
                    f"the exponent is {ENTANGLEMENT_DILUTION_EXPONENT}, the same law and the "
                    "same constant the blend expert applies to a short chain diluting a long "
                    "one; a solvent is the limiting case of a diluent",
                ),
                conditions=conditions,
            )

        if prop == "surface_tension":
            gammas = []
            for component in molecules:
                sub = Candidate(
                    material_class=MaterialClass.MOLECULE,
                    molecule=component.molecule,
                    conditions=candidate.conditions,
                )
                gamma, gamma_std = self._component_value(
                    molecular, sub, "surface_tension", "N/m", conditions
                )
                if gamma is None or gamma <= 0:
                    return Prediction.unsupported(
                        prop,
                        self.id,
                        f"the solvent {component.molecule.smiles!r} has no predicted surface "
                        "tension, and a dope's surface is its solvent's",
                    )
                gammas.append((component.fraction / solvent_mass, gamma, gamma_std))
            value = sum(w * g for w, g, _ in gammas)
            stated = max((s for _, _, s in gammas if s is not None), default=None)
            # The polymer's own surface tension bounds how far enrichment can be
            # from complete, so the gap between the two is the honest width.
            polymer_gamma = None
            for component in polymers:
                sub = Candidate(
                    material_class=MaterialClass.POLYMER,
                    polymer=component.polymer,
                    conditions=candidate.conditions,
                )
                found, _ = self._component_value(
                    polymer_panel, sub, "surface_tension", "N/m", conditions
                )
                if found is not None:
                    polymer_gamma = found if polymer_gamma is None else max(polymer_gamma, found)
            gap = abs(polymer_gamma - value) if polymer_gamma is not None else value * 0.3
            spread = math.hypot(stated if stated is not None else value * 0.1, gap * 0.5)
            return self._make(
                prop,
                value,
                "N/m",
                request,
                domain,
                std=spread,
                kind=UncertaintyKind.COMBINED,
                basis=(
                    "the solvent's own bar, plus half the gap to the dissolved polymer's "
                    "surface tension: that gap is how far surface enrichment could fail to "
                    "be complete, and it bounds the answer from the only side it can move"
                ),
                notes=(
                    SURFACE_ENRICHMENT_NOTE,
                    (
                        f"the dissolved polymer's own surface tension is "
                        f"{polymer_gamma*1000:.1f} mN/m against the solvent's {value*1000:.1f}"
                        if polymer_gamma is not None
                        else "the dissolved polymer's own surface tension is not available, so "
                        "the width is a flat 30% rather than the gap to it"
                    ),
                    "NOT mixed by fraction: a mixing rule would pull the surface tension "
                    "toward the polymer's value, and enrichment pulls it the other way",
                ),
                conditions=conditions,
            )

        if prop == "liquid_density":
            return self._make(
                prop,
                solution_density * 1000.0,
                "kg/m^3",
                request,
                domain,
                std=solution_density * 1000.0 * 0.03,
                kind=UncertaintyKind.COMBINED,
                basis=(
                    "volume-additive over the components, 1/rho = sum(w/rho), on their own "
                    "predicted densities. Ideal mixing is good to a per cent or two for a "
                    "polymer in its own good solvent; three per cent covers that and the "
                    "component densities' own error"
                ),
                notes=(
                    f"{polymer_mass:.0%} polymer by mass in a solvent, so the solution sits "
                    "between the two component densities by construction",
                    "this is the density of the SOLUTION, not of the polymer: the spinline "
                    "expert needs it to turn a viscosity into a pressure and a relaxation "
                    "time, and a dissolved polymer has no amorphous density to give it",
                ),
                conditions=conditions,
            )

        value, branch = solution_viscosity(solvent_viscosity, intrinsic, concentration)
        overlap = concentration * intrinsic

        # -- uncertainty ----------------------------------------------------
        # In log space, because every term here is multiplicative. The
        # intrinsic viscosity enters the entangled branch raised to the
        # exponent, so its error is amplified there and that is carried.
        intrinsic_log = math.log10(INTRINSIC_VISCOSITY_SPREAD)
        expansion_log = math.log10(GOOD_SOLVENT_EXPANSION) / 2.0
        amplification = ENTANGLED_EXPONENT if branch == "entangled" else 1.0
        log_sigma = math.sqrt(
            solvent_log_sigma**2
            + (amplification * intrinsic_log) ** 2
            + (amplification * expansion_log) ** 2
            + REGIME_LOG_SIGMA**2
            + (
                (math.log10(max(overlap, 1.0 + 1e-9)) * ENTANGLED_EXPONENT_SPREAD) ** 2
                if branch == "entangled"
                else 0.0
            )
        )
        spread = value * (10.0**log_sigma - 1.0) / 2.0

        notes = [
            f"{branch} branch at c[eta] = {overlap:.2f}: "
            + (
                "the coils are separate and Huggins applies"
                if branch == "dilute"
                else f"the coils overlap and entangle, so viscosity climbs as the "
                f"{ENTANGLED_EXPONENT} power of the overlap parameter"
            ),
            f"[eta] = {intrinsic:.3f} dL/g from Flory-Fox on the predicted chain dimension, "
            f"c = {concentration:.1f} g/dL, c* = {overlap_concentration(intrinsic):.2f} g/dL",
            f"solvent viscosity {solvent_viscosity*1000:.2f} mPa.s, so the polymer multiplies "
            f"it by {value/solvent_viscosity:.3g}",
            "the intrinsic viscosity is the THETA value. A good solvent swells the coil and "
            f"[eta] goes as the cube of the expansion, up to about {GOOD_SOLVENT_EXPANSION:.0f}x; "
            "that range is carried in the uncertainty rather than modelled, because deciding "
            "it needs an interaction radius this repository has measured to be underivable",
            "THIS ASSUMES THE POLYMER DISSOLVES. Nothing here checks that; ask "
            "polymer_dissolution, which answers the solubility question and is explicit "
            "about the part of it that is not derivable. A viscosity for a suspension of "
            "undissolved polymer is a number about a different material",
        ]
        return self._make(
            prop,
            value,
            "Pa*s",
            request,
            domain,
            std=spread,
            kind=UncertaintyKind.COMBINED,
            basis=(
                f"{log_sigma:.2f} decades: the solvent's own bar, the Flory-Fox spread of "
                f"{INTRINSIC_VISCOSITY_SPREAD:.2f}x, the unmodelled good-solvent expansion, and "
                "the concentration law itself - with the first three amplified by the "
                "entangled exponent where that branch applies, because it raises them to a "
                "power"
            ),
            notes=tuple(notes),
            conditions=conditions,
        )
