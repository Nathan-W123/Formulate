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
from dataclasses import dataclass

from ..core.candidate import Candidate, MaterialClass
from ..core.properties import PropertyFamily
from ..core.quantity import ApplicabilityDomain, UncertaintyKind
from .base import Expert, Prediction

#: Molar gas constant, J/(mol K).
GAS_CONSTANT = 8.314462618

@dataclass(frozen=True, slots=True)
class MonomerKinetics:
    """Everything needed to turn one monomer into a rate."""

    #: Arrhenius prefactor for propagation, L/(mol s).
    prefactor: float
    #: Propagation activation energy, kJ/mol.
    activation_kj: float
    #: Chain-length-averaged termination coefficient at 298 K, L/(mol s).
    termination: float
    #: Concentration of polymerisable double bonds in the neat liquid, mol/L.
    #: For a monofunctional monomer this is just its molar concentration; for a
    #: crosslinker it is that times the functionality.
    concentration: float
    #: Polymerisable double bonds per molecule.
    functionality: int
    #: Glass transition of the *linear* homopolymer, K. ``None`` for a
    #: crosslinker, where there is no linear homopolymer and the network's
    #: rigidity comes from the crosslinks rather than from chain stiffness.
    linear_tg_k: float | None
    source: str


#: Propagation kinetics per monomer.
#:
#: The seven monofunctional entries carry their own IUPAC pulsed-laser
#: benchmark Arrhenius pairs. The crosslinkers do not: nobody has run a
#: pulsed-laser experiment on a diacrylate, because the thing gels during the
#: measurement. They borrow the coefficient of their family, which is a real
#: approximation and worth stating plainly - k_p belongs to the propagating
#: radical and the double bond it attacks, and those are the same chemistry in
#: butyl acrylate and in hexanediol diacrylate. What differs is the medium, and
#: it differs in the direction that makes this conservative: a crosslinking
#: system stiffens as it reacts, which suppresses termination far more than
#: propagation and makes real cures faster than this predicts, not slower.
MONOMERS: dict[str, MonomerKinetics] = {
    "styrene": MonomerKinetics(4.27e7, 32.5, 7.0e7, 8.7, 1, 373.0, "IUPAC benchmark"),
    "methyl methacrylate": MonomerKinetics(
        2.67e6, 22.36, 2.5e7, 9.4, 1, 378.0, "IUPAC benchmark"
    ),
    "butyl methacrylate": MonomerKinetics(
        3.78e6, 22.9, 2.0e7, 6.3, 1, 293.0, "IUPAC benchmark"
    ),
    "methyl acrylate": MonomerKinetics(
        1.66e7, 17.7, 1.0e8, 11.1, 1, 283.0, "IUPAC benchmark"
    ),
    "ethyl acrylate": MonomerKinetics(
        1.80e7, 17.5, 1.2e8, 9.2, 1, 249.0, "IUPAC benchmark"
    ),
    "butyl acrylate": MonomerKinetics(
        2.21e7, 17.9, 1.5e8, 7.0, 1, 219.0, "IUPAC benchmark"
    ),
    "vinyl acetate": MonomerKinetics(
        1.47e7, 20.4, 3.0e7, 10.8, 1, 305.0, "IUPAC benchmark"
    ),
    # Crosslinkers. Concentrations are the functionality times the neat
    # liquid's molar concentration, from supplier densities (1.010, 1.051 and
    # 1.103 g/cm^3) rather than from the compilation, which carries none of
    # them. The density enters the rate linearly and the answers here are two
    # orders inside their target, so a few per cent on it changes nothing.
    "1,6-hexanediol diacrylate": MonomerKinetics(
        2.21e7, 17.9, 1.5e8, 8.93, 2, None, "acrylate family k_p (butyl acrylate)"
    ),
    "ethylene glycol dimethacrylate": MonomerKinetics(
        2.67e6, 22.36, 2.5e7, 10.60, 2, None, "methacrylate family k_p (MMA)"
    ),
    "trimethylolpropane triacrylate": MonomerKinetics(
        2.21e7, 17.9, 1.5e8, 11.17, 3, None, "acrylate family k_p (butyl acrylate)"
    ),
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
    "1,6-hexanediol diacrylate": "C=CC(=O)OCCCCCCOC(=O)C=C",
    "ethylene glycol dimethacrylate": "C=C(C)C(=O)OCCOC(=O)C(=C)C",
    "trimethylolpropane triacrylate": "C=CC(=O)OCC(CC)(COC(=O)C=C)COC(=O)C=C",
}

#: A polymerisable double bond: a terminal methylene on a non-aromatic carbon.
#: Every monomer in this table carries one or more, and the count is read from
#: the structure rather than tabulated, so a candidate's functionality cannot
#: disagree with what it is made of.
POLYMERISABLE = "[CH2]=[CX3]"

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


def functionality(smiles: str) -> int:
    """Polymerisable double bonds in one molecule, read from its structure."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts(POLYMERISABLE)))


def propagation_constant(monomer: str, temperature_k: float) -> float:
    """k_p in L/(mol s) from the Arrhenius pair."""
    row = MONOMERS[monomer]
    return row.prefactor * math.exp(
        -row.activation_kj * 1e3 / (GAS_CONSTANT * temperature_k)
    )


def radical_concentration(monomer: str, initiation_rate: float) -> float:
    """Steady-state radical concentration, ``sqrt(R_i / 2 k_t)``, in mol/L."""
    return math.sqrt(initiation_rate / (2.0 * MONOMERS[monomer].termination))


def kinetic_chain_length(monomer: str, temperature_k: float, initiation_rate: float) -> float:
    """Monomer units added per radical before it terminates.

    ``k_p [M] / (2 k_t [R])``: the rate one chain grows over the rate it dies.
    This is the quantity that decides when a crosslinking system gels, and it
    moves the opposite way to the overall rate - pushing the initiation rate up
    makes the polymerisation faster and the chains shorter, and short chains
    gel late. That trade-off is the reason a faster initiator is not a way to
    reach a two second cure.
    """
    row = MONOMERS[monomer]
    kp = propagation_constant(monomer, temperature_k)
    return kp * row.concentration / (2.0 * row.termination * radical_concentration(monomer, initiation_rate))


def gel_conversion(monomer: str, temperature_k: float, initiation_rate: float) -> float | None:
    """Conversion at which the system gels, or None for a linear polymer.

    Flory and Stockmayer: a chain-growth network gels once one crosslink has
    formed per weight-average primary chain, which puts the gel point at
    ``1 / (rho (DPw - 1))`` for ``rho`` pendant double bonds per repeat unit.
    A neat monomer of functionality ``f`` leaves ``f - 1`` of them, so a
    monofunctional monomer never gels at all - it makes linear chains, and
    stops flowing only when the unreacted monomer plasticising it has been
    consumed.

    That distinction is the whole reason this function exists. Solidifying by
    vitrification needs half the monomer converted; gelling needs a fraction of
    a per cent, because one crosslink per chain is enough and the chains are
    hundreds of units long. It is the difference between a minute and a tenth
    of a second, and it is not a refinement of the earlier calculation - the
    earlier calculation was answering a different question.

    Two things make this a lower bound on the gel conversion, and therefore on
    the time. Cyclisation wastes pendant double bonds on loops within the same
    chain, which do not count towards the network and push real gel points
    higher, by a factor of a few in a neat multifunctional monomer. And the
    treatment assumes termination by combination, which doubles the primary
    chain length; disproportionation would halve it and gel later.
    """
    row = MONOMERS[monomer]
    if row.functionality < 2:
        return None
    pendant = row.functionality - 1
    weight_average = 2.0 * kinetic_chain_length(monomer, temperature_k, initiation_rate)
    if weight_average <= 1.0:
        return None
    return min(1.0 / (pendant * (weight_average - 1.0)), 1.0)


def cure_time(
    monomer: str,
    temperature_k: float,
    initiation_rate: float,
    *,
    conversion: float | None = None,
) -> float:
    """Seconds for a reacting liquid to stop flowing.

    The endpoint depends on what kind of polymer it makes. A crosslinker stops
    flowing at its gel point; a monofunctional monomer has no gel point and
    stops flowing when it vitrifies, which is taken at half conversion.

    Steady state puts the radical concentration at ``sqrt(R_i / 2 k_t)``, and a
    first-order consumption of monomer at that concentration integrates to
    ``-ln(1 - x) / (k_p [R])``. The approximation is that k_t holds at its
    low-conversion value throughout, which it does not: the Trommsdorff effect
    drops it by orders of magnitude as the medium thickens, so real cures
    accelerate. This is therefore an upper bound on the time, which is the safe
    direction for ruling a chemistry out and the unsafe one for ruling it in.
    """
    if conversion is None:
        conversion = gel_conversion(monomer, temperature_k, initiation_rate)
        if conversion is None:
            conversion = SOLIDIFICATION_CONVERSION
    kp = propagation_constant(monomer, temperature_k)
    radicals = radical_concentration(monomer, initiation_rate)
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

        row = MONOMERS[monomer]
        gel = gel_conversion(monomer, temperature, rate)

        # The glass transition only decides anything for a linear polymer. A
        # crosslinked network is rigid because it is a single molecule, not
        # because its chains are stiff, so the homopolymer Tg of the linear
        # analogue says nothing about it - and applying the test anyway would
        # refuse every diacrylate, which is exactly the chemistry that gets a
        # cure into the seconds.
        if gel is None:
            if row.linear_tg_k is None:  # pragma: no cover - table invariant
                return Prediction.unsupported(
                    prop, self.id, f"{monomer} has neither a gel point nor a linear Tg"
                )
            if row.linear_tg_k <= temperature:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"poly({monomer}) has a glass transition at {row.linear_tg_k:.0f} K, at or "
                    f"below the {temperature:.0f} K it would cure at, so it is a rubber at "
                    "every conversion. There is a time at which this liquid stops being a "
                    "liquid and it is not a time at which it becomes a solid. It is "
                    "monofunctional, so it has no gel point to reach instead",
                )

        value = cure_time(monomer, temperature, rate)
        if gel is None:
            endpoint = (
                f"taken as the time to {SOLIDIFICATION_CONVERSION:.0%} conversion, where the "
                f"mixture's own glass transition climbs through {temperature:.0f} K; "
                f"poly({monomer}) is glassy at {row.linear_tg_k:.0f} K. It is monofunctional "
                "and never gels"
            )
            # Inverse square root of two order-of-magnitude quantities.
            spread = 2.0
            extra: tuple[str, ...] = ()
        else:
            endpoint = (
                f"taken as the time to gel at {gel * 100:.3g} per cent conversion, from "
                f"Flory-Stockmayer with {row.functionality - 1} pendant double bond(s) per "
                f"repeat unit and a weight-average primary chain of "
                f"{2 * kinetic_chain_length(monomer, temperature, rate):.0f}"
            )
            spread = 4.0
            extra = (
                "the gel conversion is a lower bound: cyclisation spends pendant double "
                "bonds on loops within the same chain, which do not join the network, and "
                "pushes real gel points higher by a factor of a few in a neat "
                "multifunctional monomer. The quoted uncertainty carries that",
                f"k_p is borrowed from the {row.source}; no pulsed-laser measurement exists "
                "for a crosslinker, because it gels during the experiment",
            )

        return self._make(
            prop,
            value,
            "s",
            request,
            domain,
            std=value * spread,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                "dominated by the inverse square root of an order-of-magnitude termination "
                "coefficient and an order-of-magnitude initiation rate, widened for a "
                "crosslinker by the cyclisation the gel point ignores"
            ),
            notes=(
                f"initiation regime {regime}: {recipe}",
                endpoint,
                "an upper bound in time: the termination coefficient is held at its "
                "low-conversion value, and the Trommsdorff effect drops it by orders of "
                "magnitude as the medium thickens, so real cures accelerate towards the end",
            )
            + extra,
            monomer=monomer,
            regime=regime,
            functionality=row.functionality,
        )


def kinetics_experts() -> list[Expert]:
    return [PropagationExpert(), FreeRadicalCureExpert()]
