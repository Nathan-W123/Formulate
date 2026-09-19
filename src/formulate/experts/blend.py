"""Polymer blends: what you get when one polymer will not do everything.

A single polymer couples properties that a design wants to set separately.
Melt viscosity goes as the 3.4 power of chain length and strength rises with
it too, so on one molar-mass knob a melt thin enough to extrude is a solid too
weak to carry load, and a solid strong enough to carry load is a melt too thick
to push. The hot-melt search in ``examples/web_shooter`` hit exactly that: the
feasible window on a single polyethylene was 5-8 kg/mol wide and the answer sat
an order of magnitude below any commercial grade.

Blending is how that coupling is broken, and it is what every real hot-melt
adhesive is: a long chain to carry load, a short one to thin the melt, and
their ratio as a second knob. Bimodal polyethylene is the same idea inside one
reactor.

The engine could already *represent* a polymer blend - ``MixtureComponent``
has carried a ``polymer`` field since Phase 4 - but nothing proposed one and no
expert scored one. The existing ``mixture`` expert declares a polymer component
out of domain in as many words, and it is right to: its rules are volume
fractions of small-molecule liquids.

What the mixing rules here are, and what each is worth:

**Viscosity is log-additive** in mass fraction. This is the standard first
approximation for a miscible blend and it is not a small correction: 10% of a
wax in a high polymer moves the melt viscosity by a factor of several, which is
the entire point of adding it. It fails for an immiscible blend, where the
minor phase forms droplets and the blend tracks the continuous phase instead,
so miscibility is checked and an immiscible blend is refused rather than
averaged.

**Modulus is bounded, not predicted.** Voigt (uniform strain) and Reuss
(uniform stress) are rigorous bounds for any two-phase arrangement, and where
a real blend falls between them is a question about morphology - co-continuous,
dispersed, layered - which nothing here knows. So the prediction is the
geometric mean of the bounds and the uncertainty spans them honestly, rather
than a point estimate that implies a morphology.

**Glass transition is Fox**, which is a genuine mixing rule for a miscible
blend and the same relation used to check whether a concentrated solution is
still a liquid.

**Melting point belongs to the crystalline component**, depressed by dilution.
A blend does not have one melting point; it has its crystalline component's,
lowered by however much amorphous material is dissolved in the melt. The
depression is real and is carried as an uncertainty rather than computed,
because Flory's expression needs an interaction parameter this repository does
not have.

**Density and surface tension are volume- and mass-additive** respectively,
which for a miscible blend of similar polymers is accurate enough that the
component uncertainties dominate.
"""

from __future__ import annotations

import math

from formulate.core.candidate import Candidate, MaterialClass, MixtureSpec
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest

#: Properties of the blend *as a melt*, wanted at a processing temperature.
MELT_PROPERTIES = frozenset({"shear_viscosity", "surface_tension"})

#: Properties of the blend *as a solid*, wanted at the temperature of use.
SOLID_PROPERTIES = frozenset(
    {
        "youngs_modulus",
        "glass_transition_temperature",
        "melting_point",
        "amorphous_density",
        "entanglement_molar_mass",
    }
)

#: Both, for anything that needs to know what a blend can be asked.
BLEND_PROPERTIES = MELT_PROPERTIES | SOLID_PROPERTIES

#: Exponent in the entanglement dilution law, ``Me(phi) = Me / phi^alpha``.
#:
#: This is the rule that decides whether a blend is still a polymer or has
#: become a wax, and it is why a blend cannot simply average its components'
#: entanglement masses. A short chain does not join the network - it dilutes
#: it, pushing the surviving entanglements further apart, exactly as a solvent
#: does. So the blend's entanglement mass is the *entangling* fraction's,
#: divided by how much of the blend that fraction is.
#:
#: The exponent is 1 for a theta diluent and about 1.3 in a good one. A low
#: oligomer of the same chemistry as the matrix is the theta-like case, so 1 is
#: used and the spread to 1.3 is carried as uncertainty.
ENTANGLEMENT_DILUTION_EXPONENT = 1.0

#: Hansen distance, MPa^0.5, beyond which two polymers are taken as immiscible.
#: Polymer-polymer miscibility is far stricter than polymer-solvent: the
#: entropy of mixing goes as 1/N for a chain, so it contributes almost nothing
#: and only a near-zero enthalpy leaves a blend miscible. Most polymer pairs
#: are immiscible, which is the correct default and the reason this is small.
MISCIBILITY_RA_LIMIT = 3.0


def log_additive(values: list[float], weights: list[float]) -> float:
    """``ln eta = sum w_i ln eta_i``. The standard blend viscosity rule."""
    return math.exp(sum(w * math.log(v) for v, w in zip(values, weights)))


#: Relative one-sigma used for a component whose own expert reported none.
#: A decade, because that is what the melt expert carries and a component with
#: no stated bar is not a component with no error.
_UNSTATED_COMPONENT_LOG_SIGMA = 1.0


def blend_log_spread(
    value: float,
    values: list[float],
    spreads: list[float | None],
    fractions: list[float],
) -> float:
    """One-sigma on a log-additive blend property, in linear units.

    The rule this accompanies is ``ln eta = sum w_i ln eta_i``, so the honest
    place to combine the components' errors is log space, where the rule is
    linear.

    This used to be ``value * 1.5``, with a comment claiming it carried "the
    components' own decade". It did not. It was a hardcoded 150% that never
    looked at the components at all, and a decade in log space is a factor of
    about 4.5 in linear units, not 1.5. The measured consequence: polyethylene
    at 50 kg/mol and at 2 kg/mol each carry 490%, and a bimodal blend of those
    two reported 150% - a blend claiming to be three times better known than
    either thing it is made of. An under-claimed bar is the worst kind of
    error here, because the ranker ranks on it.

    The components are combined as FULLY CORRELATED rather than independent:
    ``sigma_ln = sum w_i sigma_ln,i``. That is deliberate and it is the
    conservative choice. Independent quadrature would be wrong in the direction
    that matters, because these errors are not independent - the bimodal case
    is the same repeat unit at two chain lengths, scored by the same expert
    through the same universal WLF constants, so its two errors move together
    almost exactly. Treating them as correlated makes a blend of identical
    components inherit precisely that component's own bar, which is the
    behaviour any rule here has to reproduce to be believable.
    """
    log_sigma = 0.0
    for component_value, component_spread, weight in zip(values, spreads, fractions):
        if component_value <= 0:
            continue
        if component_spread is None:
            own = _UNSTATED_COMPONENT_LOG_SIGMA
        else:
            # Linear sigma to log10 sigma, by the same relation the melt expert
            # uses in reverse: sigma = value * (10**log_sigma - 1) / 2.
            own = math.log10(1.0 + 2.0 * component_spread / component_value)
        log_sigma += weight * own
    if log_sigma <= 0:
        log_sigma = _UNSTATED_COMPONENT_LOG_SIGMA
    return value * (10.0**log_sigma - 1.0) / 2.0


def voigt_reuss(values: list[float], fractions: list[float]) -> tuple[float, float]:
    """Upper and lower bounds on a two-phase modulus.

    Voigt is uniform strain and is the stiffest any arrangement can be; Reuss
    is uniform stress and is the softest. Every real morphology lies between.
    """
    voigt = sum(f * v for v, f in zip(values, fractions))
    reuss = 1.0 / sum(f / v for v, f in zip(values, fractions))
    return voigt, reuss


def fox_glass_transition(values: list[float], weights: list[float]) -> float:
    """``1/Tg = sum w_i / Tg_i``."""
    return 1.0 / sum(w / v for v, w in zip(values, weights))


class _BlendExpert(Expert):
    """Mixing rules over polymer components, from the polymer panel.

    Split into a melt-state and a solid-state expert below, and the split is
    not cosmetic. The evaluation engine evaluates an expert once, at one set of
    conditions, and resolves those from the requirements that expert serves; an
    expert serving both a melt viscosity at 200 C and a modulus at 25 C is
    asking it for something it cannot give, and what came back was a density
    refused for being asked at 200 C - correctly, since its packing factor was
    fitted at room temperature. The properties genuinely belong to two
    different states of the same material, so they are two experts.
    """

    supported_classes = frozenset({MaterialClass.MIXTURE})

    def __init__(self, registry=None) -> None:
        self._registry = registry

    def registry(self):
        """The polymer panel, built lazily so the import stays one-way."""
        if self._registry is None:
            from formulate.experts import polymer_registry

            self._registry = polymer_registry()
        return self._registry

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "RDKit is required to read component structures"

    # -- domain ------------------------------------------------------------

    @staticmethod
    def _all_polymer(mixture: MixtureSpec) -> bool:
        return all(c.polymer is not None for c in mixture.components)

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        basis = "mixing rules over polymer components"
        mixture = candidate.mixture
        if mixture is None:
            return ApplicabilityDomain.outside("candidate carries no mixture", basis=basis)
        if not self._all_polymer(mixture):
            return ApplicabilityDomain.outside(
                "this blend contains a small-molecule component; these rules mix polymer "
                "properties and a solvent or plasticiser does not enter them the same way",
                basis=basis,
            )
        if len(mixture.components) > 2:
            return ApplicabilityDomain(
                in_domain=True,
                score=0.5,
                basis=basis,
                notes=(
                    "the modulus bounds are two-phase and are applied pairwise here; a "
                    "ternary blend has a morphology they do not describe",
                ),
            )
        return ApplicabilityDomain(basis=basis)

    # -- components --------------------------------------------------------

    def _component_properties(
        self, mixture: MixtureSpec, request: PredictionRequest, wanted: frozenset[str]
    ) -> list[dict[str, Prediction]]:
        """Run the polymer panel over each component in turn."""
        from .base import PredictionRequest as Req

        requested = frozenset(wanted)
        registry = self.registry()
        # Pull in what the panel needs to answer, not just what was asked for.
        # Melt viscosity depends on a glass transition and an entanglement mass
        # from two other experts; requesting the property alone selects only the
        # expert that serves it, whose dependencies then arrive empty and which
        # correctly refuses. The engine does this closure for a top-level run
        # and a delegating expert has to do it for its own sub-runs.
        wanted = set(wanted)
        for _ in range(len(registry) + 1):
            grown = set(wanted)
            for expert in registry:
                if expert.supported_properties & wanted:
                    grown |= expert.dependencies
            if grown == wanted:
                break
            wanted = grown
        wanted = frozenset(wanted)

        out: list[dict[str, Prediction]] = []
        for component in mixture.components:
            sub = Candidate(
                material_class=MaterialClass.POLYMER,
                polymer=component.polymer,
                conditions=request.candidate.conditions,
            )
            context: dict[str, Prediction] = {}
            for expert in registry.resolution_order(
                registry.experts_for(wanted, MaterialClass.POLYMER)
            ):
                # A dependency is evaluated at the candidate's own conditions,
                # and only the property actually asked for carries the
                # requirement's. Passing the melt temperature to everything was
                # wrong and failed loudly: the density expert refuses at 200 C,
                # because its packing factor was fitted at room temperature and
                # carries no temperature dependence, and that refusal then took
                # out the entanglement mass and the viscosity behind it. A
                # blend's melt viscosity is wanted at the nozzle; the density it
                # rests on is a room-temperature quantity.
                serves_requested = bool(expert.supported_properties & requested)
                sub_request = Req(
                    candidate=sub,
                    properties=wanted,
                    conditions=request.conditions if serves_requested else sub.conditions,
                    context=dict(context),
                )
                for prediction in expert.predict(sub_request):
                    if prediction.is_usable:
                        context.setdefault(prediction.property, prediction)
            out.append(context)
        return out

    def _miscible(self, mixture: MixtureSpec) -> tuple[bool, str]:
        """Hansen distance between components, on the polymer expert's triples."""
        from .hansen import hansen_triple

        from formulate import chem

        # A bimodal blend is one chemistry at two chain lengths, so it is
        # miscible by construction and needs no solubility argument at all.
        # This is also the case Hansen cannot serve: the group set does not
        # decompose a polyester or a polyamide repeat unit, so requiring a
        # Hansen distance here would refuse exactly the blends that are most
        # obviously fine.
        units = {
            chem.canonical_smiles(c.polymer.monomers[0].smiles) or c.polymer.monomers[0].smiles
            for c in mixture.components
        }
        if len(units) == 1:
            return True, "one chemistry at two chain lengths, so miscible by construction"

        triples = []
        for component in mixture.components:
            chain = [m for m in component.polymer.monomers if m.mole_fraction > 0]
            # Cap the repeat unit by dropping the attachment points rather than
            # methylating them: the latter builds fragments the group set
            # cannot decompose even for polymers it otherwise covers.
            capped = chain[0].smiles.replace("[*]", "") if chain else None
            triple = hansen_triple(capped) if capped else None
            if triple is None:
                return False, (
                    "miscibility could not be judged: a component's Hansen parameters "
                    "are not available, and two different polymers are immiscible far "
                    "more often than not, so assuming otherwise would be the wrong default"
                )
            # hansen_triple answers in the registry's canonical Pa^0.5; the
            # miscibility limit below is quoted the way the literature quotes
            # it, in MPa^0.5, so the conversion happens once, here.
            triples.append(tuple(component / 1000.0 for component in triple))

        worst = 0.0
        for i, a in enumerate(triples):
            for b in triples[i + 1 :]:
                ra = math.sqrt(
                    4 * (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
                )
                worst = max(worst, ra)
        if worst > MISCIBILITY_RA_LIMIT:
            return False, (
                f"the components are {worst:.1f} MPa^0.5 apart in Hansen space, past the "
                f"{MISCIBILITY_RA_LIMIT:.0f} that polymer-polymer miscibility allows. A "
                "chain gains almost no entropy on mixing, so an immiscible blend forms "
                "two phases and tracks the continuous one rather than any average"
            )
        return True, f"components within {worst:.1f} MPa^0.5 in Hansen space"

    # -- the mixing rules --------------------------------------------------

    def _predict_one(self, prop, request: PredictionRequest, domain):
        mixture = request.candidate.mixture
        if mixture is None:
            return Prediction.failed(prop, self.id, "candidate carries no mixture")
        if not self._all_polymer(mixture):
            return Prediction.unsupported(
                prop, self.id, "every component must be a polymer for these rules"
            )

        miscible, why = self._miscible(mixture)
        if not miscible:
            return Prediction.unsupported(prop, self.id, why)

        fractions = [c.fraction for c in mixture.components]
        components = self._component_properties(mixture, request, frozenset({prop}))

        values, spreads, missing = [], [], []
        for index, context in enumerate(components):
            found = context.get(prop)
            if found is None or found.quantity is None:
                missing.append(index + 1)
            else:
                values.append(found.quantity.to_canonical().value)
                std = found.uncertainty.std if found.uncertainty is not None else None
                if std is not None:
                    # Into the same canonical units as the value.
                    std = found.quantity.__class__(
                        value=std, unit=found.quantity.unit
                    ).to_canonical().value
                spreads.append(std)
        if missing:
            return Prediction.unsupported(
                prop,
                self.id,
                f"component {', '.join(str(m) for m in missing)} of "
                f"{len(mixture.components)} has no {prop}, and a mixing rule over a "
                "missing value would be an average of one thing",
            )

        unit = _canonical_unit(prop)
        if prop == "shear_viscosity":
            value = log_additive(values, fractions)
            spread = blend_log_spread(value, values, spreads, fractions)
            note = (
                "log-additive in mass fraction: ln(eta) = sum w ln(eta). A tenth of a "
                "wax moves a high polymer's melt viscosity by a factor of several, "
                "which is the whole reason it is there"
            )
        elif prop == "youngs_modulus":
            voigt, reuss = voigt_reuss(values, fractions)
            value = math.sqrt(voigt * reuss)
            # The bounds coincide when the components are identical, and float
            # error then makes the difference very slightly negative.
            spread = max(0.0, (voigt - reuss) / 2.0)
            note = (
                f"between the Voigt bound ({voigt/1e9:.2f} GPa, uniform strain) and the "
                f"Reuss bound ({reuss/1e9:.2f} GPa, uniform stress). Where a real blend "
                "falls between them is a question about morphology, which is not "
                "predicted here, so the uncertainty spans the bounds"
            )
        elif prop == "entanglement_molar_mass":
            masses = [
                c.polymer.number_average_molar_mass.to_canonical().value
                if c.polymer.number_average_molar_mass is not None
                else None
                for c in mixture.components
            ]
            if any(m is None for m in masses):
                return Prediction.unsupported(
                    prop, self.id,
                    "a component states no molar mass, so whether it entangles at all "
                    "cannot be decided, and it is that which sets the blend's network",
                )
            # Only components long enough to entangle are part of the network.
            # The rest are diluent, however chemically identical they are.
            entangling = [
                (value, fraction)
                for value, fraction, mass in zip(values, fractions, masses)
                if mass >= 2.0 * value
            ]
            if not entangling:
                return Prediction.unsupported(
                    prop, self.id,
                    "no component in this blend is long enough to entangle, so it has no "
                    "load-bearing network at all: it is a wax, not a polymer, whatever "
                    "its stiffness says",
                )
            network_fraction = sum(f for _, f in entangling)
            base = min(v for v, _ in entangling)
            value = base / network_fraction**ENTANGLEMENT_DILUTION_EXPONENT
            high = base / network_fraction**1.3
            spread = abs(high - value)
            note = (
                f"{network_fraction:.0%} of this blend is long enough to entangle; the "
                f"rest dilutes the network rather than joining it, pushing the "
                f"entanglement mass from {base*1e3:.0f} to {value*1e3:.0f} g/mol. A "
                "short chain of the same chemistry is still a diluent"
            )
        elif prop == "glass_transition_temperature":
            value = fox_glass_transition(values, fractions)
            spread = max(abs(value - v) for v in values) * 0.2
            note = "Fox: 1/Tg = sum(w/Tg), a miscible blend having one transition"
        elif prop == "melting_point":
            value = max(values)
            crystalline = fractions[values.index(max(values))]
            spread = 10.0 + 30.0 * (1.0 - crystalline)
            note = (
                f"a blend has no melting point of its own: this is the highest-melting "
                f"component's, at {crystalline:.0%} of the blend. Dilution depresses it, "
                "and the depression is carried as uncertainty rather than computed "
                "because Flory's expression needs an interaction parameter this "
                "repository does not have"
            )
        else:  # amorphous_density, surface_tension
            value = sum(f * v for v, f in zip(values, fractions))
            spread = abs(max(values) - min(values)) * 0.1
            note = "additive in mass fraction over the components"

        return self._make(
            prop, value, unit, request, domain,
            std=spread, kind=UncertaintyKind.COMBINED,
            basis="propagated from the component predictions and the rule's own spread",
            notes=(note, why, f"over {len(values)} polymer components"),
        )


class PolymerBlendMeltExpert(_BlendExpert):
    """The blend as a melt: how thick it is and how hard its surface pulls."""

    id = "polymer_blend_melt"
    version = "1"
    method = (
        "log-additive melt viscosity and mass-additive surface tension over polymer "
        "components scored by the polymer panel"
    )
    family = PropertyFamily.INTERFACIAL
    supported_properties = MELT_PROPERTIES


class PolymerBlendSolidExpert(_BlendExpert):
    """The blend as a solid: how stiff, how dense, and where it softens."""

    id = "polymer_blend_solid"
    version = "1"
    method = (
        "Voigt-Reuss modulus bounds, Fox glass transition, the crystalline component's "
        "melting point and additive density over polymer components"
    )
    family = PropertyFamily.MECHANICAL
    supported_properties = SOLID_PROPERTIES


def _canonical_unit(prop: str) -> str:
    from formulate.core.properties import get_property

    return get_property(prop).canonical_unit
