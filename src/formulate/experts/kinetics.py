"""How fast a monomer polymerises, and how long a reacting liquid stays liquid.

The property registry had no reaction rate of any kind in it, and the gap was
not cosmetic. Every property here is a state property - a boiling point, a
density, a solubility parameter - and a material chosen for what it *becomes*
rather than for what it *is* cannot be searched on state properties alone. A
liquid that must solidify in flight is selected on a rate, and until this
module existed there was nothing for such a target to rank on.

Two coefficients decide a free-radical cure and they are known to very
different precision, which is the main thing a reader should take from this
file.

**Propagation is a benchmark quantity.** The IUPAC Working Party on Modeling
of Polymerization Kinetics and Processes spent two decades establishing k_p by
pulsed-laser polymerisation with size-exclusion chromatography, a method that
measures the coefficient directly rather than inferring it from an overall
rate. The Arrhenius parameters below are those recommendations. They are good
to something like ten per cent, and they are the reason a calculation like this
is worth doing at all.

**Termination is not.** k_t is diffusion controlled: it depends on chain
length, on conversion, and on the viscosity of a medium that the reaction
itself is thickening, and a single number for it is a fiction maintained for
convenience. The values here are chain-length-averaged, at low conversion, at
298 K, and they are order-of-magnitude. Since the rate goes as the inverse
square root of k_t, a factor of three in k_t is a factor of 1.7 in the answer -
which is tolerable for deciding whether a cure takes a second or an hour, and
not tolerable for anything finer. Predictions say so.

The third input is the initiation rate, and it is not a property of the
monomer at all. It is a property of the recipe: which initiator, how much, and
whether anything is helping it along. It arrives through ``Conditions.processing``
in the same way a solute arrives through ``Conditions.solutes`` - the target
states the regime it intends and the engine supplies what that implies -
because a cure time quoted without one is meaningless and defaulting to a
value would hide exactly the choice that dominates the answer.

One refusal in here does more work than any of the arithmetic. A cure time is
only interesting if what it produces is a solid, and half the fast monomers do
not make one: poly(butyl acrylate) has a glass transition at 219 K, so at
ambient temperature it is a rubber at every conversion and no amount of
reaction makes it stop flowing in the sense a load-bearing strand needs. The
homopolymer glass transition is tabulated alongside the kinetics and the
expert declines rather than reporting the time at which a liquid becomes a
gum.
"""

from __future__ import annotations

import math

from ..core.candidate import Candidate, MaterialClass
from ..core.properties import PropertyFamily
from ..core.quantity import ApplicabilityDomain, UncertaintyKind
from .base import Expert, Prediction

#: Molar gas constant, J/(mol K).
GAS_CONSTANT = 8.314462618

#: IUPAC-recommended propagation kinetics: monomer -> (A in L/mol/s, Ea in
#: kJ/mol, bulk monomer concentration in mol/L, homopolymer Tg in K).
#:
#: The Arrhenius pairs are the pulsed-laser benchmark values. Bulk monomer
#: concentrations are the neat liquid's density over its molar mass at 298 K.
#: Homopolymer glass transitions are the standard handbook values, and are
#: here because they decide whether a cure produces a solid or a gum.
MONOMERS: dict[str, tuple[float, float, float, float]] = {
    "styrene": (4.27e7, 32.5, 8.7, 373.0),
    "methyl methacrylate": (2.67e6, 22.36, 9.4, 378.0),
    "butyl methacrylate": (3.78e6, 22.9, 6.3, 293.0),
    "methyl acrylate": (1.66e7, 17.7, 11.1, 283.0),
    "ethyl acrylate": (1.80e7, 17.5, 9.2, 249.0),
    "butyl acrylate": (2.21e7, 17.9, 7.0, 219.0),
    "vinyl acetate": (1.47e7, 20.4, 10.8, 305.0),
}

#: Chain-length-averaged termination coefficients at 298 K and low conversion,
#: L/mol/s. Order-of-magnitude; see this module's opening note.
TERMINATION: dict[str, float] = {
    "styrene": 7.0e7,
    "methyl methacrylate": 2.5e7,
    "butyl methacrylate": 2.0e7,
    "methyl acrylate": 1.0e8,
    "ethyl acrylate": 1.2e8,
    "butyl acrylate": 1.5e8,
    "vinyl acetate": 3.0e7,
}

#: SMILES of each tabulated monomer, canonicalised on first use.
_SMILES: dict[str, str] = {
    "styrene": "C=Cc1ccccc1",
    "methyl methacrylate": "C=C(C)C(=O)OC",
    "butyl methacrylate": "C=C(C)C(=O)OCCCC",
    "methyl acrylate": "C=CC(=O)OC",
    "ethyl acrylate": "C=CC(=O)OCC",
    "butyl acrylate": "C=CC(=O)OCCCC",
    "vinyl acetate": "C=COC(C)=O",
}

#: Radical generation rate of each initiation regime at 298 K, mol/(L s), with
#: the recipe it assumes. These are the weakest numbers in the module and the
#: most influential: the cure time goes as the inverse square root of this, so
#: an order of magnitude here is a factor of three in the answer.
#:
#: "thermal-aibn" is azobisisobutyronitrile at 0.05 M with f = 0.6, using its
#: own Arrhenius decomposition (A = 1.58e15 /s, Ea = 128.4 kJ/mol), which at
#: 298 K gives a half life of about five months - that regime is in the table
#: precisely so that a target asking for a fast ambient cure can be shown why
#: the obvious initiator is not the answer. "redox-peroxide-amine" is benzoyl
#: peroxide with a tertiary aromatic amine, both at 0.05 M, the chemistry that
#: makes bone cement and two-part acrylic adhesives set at room temperature.
#: "photo-uv" is a cleavage photoinitiator under a few tens of mW/cm^2, which
#: needs a lamp the application may not have.
INITIATION: dict[str, tuple[float, str]] = {
    "thermal-aibn": (3.1e-9, "AIBN at 0.05 M, f = 0.6, thermal decomposition at 298 K"),
    "thermal-bpo": (6.0e-11, "benzoyl peroxide at 0.05 M, thermal decomposition at 298 K"),
    "redox-peroxide-amine": (
        3.0e-4,
        "benzoyl peroxide with a tertiary aromatic amine, both at 0.05 M",
    ),
    "photo-uv": (1.0e-3, "cleavage photoinitiator under a few tens of mW/cm^2 of UV"),
}

#: Conversion at which a linear chain-growth polymerisation stops flowing.
#:
#: Not a gel point. A monovinyl monomer makes linear chains and never gels; it
#: solidifies because the unreacted monomer that was plasticising it has been
#: consumed, so the mixture's own glass transition climbs through the ambient
#: temperature. Half conversion is where that happens for a monomer whose
#: homopolymer is glassy, to within the accuracy of everything else here.
SOLIDIFICATION_CONVERSION = 0.5


def _canonical(smiles: str) -> str:
    from .. import chem

    return chem.canonical_smiles(smiles) or smiles


def _identify(candidate: Candidate) -> str | None:
    """Which tabulated monomer this candidate is, by structure not by name."""
    smiles = candidate.primary_smiles
    if smiles is None:
        return None
    target = _canonical(smiles)
    for name, reference in _SMILES.items():
        if _canonical(reference) == target:
            return name
    return None


def propagation_constant(monomer: str, temperature_k: float) -> float:
    """k_p in L/(mol s) from the benchmark Arrhenius pair."""
    prefactor, activation_kj, _, _ = MONOMERS[monomer]
    return prefactor * math.exp(-activation_kj * 1e3 / (GAS_CONSTANT * temperature_k))


def cure_time(
    monomer: str,
    temperature_k: float,
    initiation_rate: float,
    *,
    conversion: float = SOLIDIFICATION_CONVERSION,
) -> float:
    """Seconds to reach ``conversion`` under a steady radical supply.

    Steady state puts the radical concentration at ``sqrt(R_i / 2 k_t)``, and
    a first-order consumption of monomer at that concentration integrates to
    ``-ln(1 - x) / (k_p [R])``. The approximation being made is that k_t holds
    at its low-conversion value throughout, which it does not: the
    Trommsdorff effect drops it by orders of magnitude as the medium thickens,
    which makes real cures *accelerate* towards the end. So this is an upper
    bound on the time, and a generous one - which is the safe direction for
    ruling a chemistry out and the unsafe direction for ruling one in.
    """
    kp = propagation_constant(monomer, temperature_k)
    radicals = math.sqrt(initiation_rate / (2.0 * TERMINATION[monomer]))
    return -math.log(1.0 - conversion) / (kp * radicals)


class _KineticsExpert(Expert):
    family = PropertyFamily.CHEMICAL
    supported_classes = frozenset({MaterialClass.MOLECULE})

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        monomer = _identify(candidate)
        if monomer is None:
            return ApplicabilityDomain(
                in_domain=False,
                score=0.0,
                basis=(
                    "the pulsed-laser benchmark set covers seven monomers and this is "
                    "not one of them; there is no group contribution for a propagation "
                    "rate coefficient, so a structure off the list has no route here"
                ),
            )
        return ApplicabilityDomain(
            in_domain=True,
            score=1.0,
            basis=f"{monomer} is in the IUPAC pulsed-laser benchmark set",
        )


class PropagationExpert(_KineticsExpert):
    """Propagation rate coefficient from the IUPAC benchmark Arrhenius pair."""

    id = "kinetics_propagation"
    version = "1"
    method = (
        "IUPAC-recommended propagation rate coefficient from pulsed-laser "
        "polymerisation with size-exclusion chromatography"
    )
    supported_properties = frozenset({"propagation_rate_constant"})

    def _predict_one(self, prop, request, domain) -> Prediction | None:
        monomer = _identify(request.candidate)
        if monomer is None:
            return Prediction.unsupported(prop, self.id, domain.basis)
        temperature = request.conditions.temperature_k
        if temperature is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "a propagation rate coefficient is an Arrhenius quantity and the request "
                "states no temperature",
            )
        value = propagation_constant(monomer, temperature)
        return self._make(
            prop,
            value,
            "L/mol/s",
            request,
            domain,
            # The benchmark values are quoted to about ten per cent, and the
            # Arrhenius extrapolation away from the fitted window costs more;
            # 298 K is inside it for every monomer here.
            std=value * 0.10,
            kind=UncertaintyKind.EPISTEMIC,
            basis="stated precision of the IUPAC pulsed-laser benchmark, about ten per cent",
            monomer=monomer,
        )


class FreeRadicalCureExpert(_KineticsExpert):
    """How long a monomer stays liquid under a stated initiation regime."""

    id = "cure_free_radical"
    version = "1"
    method = (
        "steady-state free-radical kinetics from benchmark k_p, order-of-magnitude "
        "k_t, and a stated initiation regime"
    )
    supported_properties = frozenset({"cure_time"})

    def _predict_one(self, prop, request, domain) -> Prediction | None:
        monomer = _identify(request.candidate)
        if monomer is None:
            return Prediction.unsupported(prop, self.id, domain.basis)

        temperature = request.conditions.temperature_k
        if temperature is None:
            return Prediction.unsupported(
                prop, self.id, "a cure time needs a temperature and the request states none"
            )

        regimes = [
            step for step in request.conditions.processing if step in INITIATION
        ]
        if not regimes:
            return Prediction.unsupported(
                prop,
                self.id,
                "a cure time is a property of the recipe and not of the monomer: it needs "
                "an initiation regime, named in the conditions' processing steps as one of "
                + ", ".join(sorted(INITIATION))
                + ". Defaulting to one would hide the choice that dominates the answer",
            )
        if len(regimes) > 1:
            return Prediction.unsupported(
                prop,
                self.id,
                f"the conditions name {len(regimes)} initiation regimes ({', '.join(regimes)}) "
                "and this expert models one at a time; two initiators together is a redox "
                "pair or an inhibitor, not a sum of rates",
            )
        regime = regimes[0]
        rate, recipe = INITIATION[regime]

        glass_transition = MONOMERS[monomer][3]
        if glass_transition <= temperature:
            return Prediction.unsupported(
                prop,
                self.id,
                f"poly({monomer}) has a glass transition at {glass_transition:.0f} K, at or "
                f"below the {temperature:.0f} K it would cure at, so it is a rubber at every "
                "conversion. There is a time at which this liquid stops being a liquid and it "
                "is not a time at which it becomes a solid",
            )

        value = cure_time(monomer, temperature, rate)
        # The rate goes as the inverse square root of the initiation rate and
        # of k_t, so an order of magnitude in either is a factor of about three
        # in the time. That, not the ten per cent on k_p, is the uncertainty.
        return self._make(
            prop,
            value,
            "s",
            request,
            domain,
            std=value * 2.0,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                "dominated by the inverse square root of an order-of-magnitude termination "
                "coefficient and an order-of-magnitude initiation rate, so a factor of about "
                "three either way"
            ),
            notes=(
                f"initiation regime {regime}: {recipe}",
                f"taken as the time to {SOLIDIFICATION_CONVERSION:.0%} conversion, where the "
                f"mixture's own glass transition climbs through {temperature:.0f} K; "
                f"poly({monomer}) is glassy at {glass_transition:.0f} K",
                "an upper bound: the termination coefficient is held at its low-conversion "
                "value, and the Trommsdorff effect drops it by orders of magnitude as the "
                "medium thickens, so real cures accelerate towards the end",
            ),
            monomer=monomer,
            regime=regime,
        )


def kinetics_experts() -> list[Expert]:
    return [PropagationExpert(), FreeRadicalCureExpert()]
