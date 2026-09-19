"""What a polymer does in the melt: whether it melts, how thick it is, and
how hard its own surface pulls on it.

These three decide a melt-processing route and none of them was served for a
polymer. A run that asks for them gets ``No expert covers: melting_point,
shear_viscosity, surface_tension`` and eliminates every candidate for a value
that could not be predicted - which is correct behaviour and a useless answer.

Each is built from something the repository already has, and each refuses
where that something runs out.

**Melting point** comes in two halves, and only one of them is tabulated.

*Whether* a polymer melts is computed from the structure. A chain crystallises
when consecutive repeat units can be placed in a regular register, and that
fails when a backbone atom carries two different substituents - polypropylene's
hydrogen and methyl, polystyrene's hydrogen and phenyl, PMMA's methyl and
ester. Polyethylene's two hydrogens, PVDF's two fluorines and polyisobutylene's
two methyls are not stereocentres, so those chains are regular by construction.
That test replaced two lists of polymers and reproduces nineteen of their
twenty calls; the one it misses is listed as an exception with its reason,
which is kinetic rather than structural.

*At what temperature* it melts is still tabulated, and honestly so. Predicting
Tm from structure was tried here two ways - cohesive energy over backbone
flexibility, and a group-contribution melt transition function fitted to the
thirteen measurements this repository holds - and both came out at 150 to 250 K
of error, which is worse than useless for a process window. Thirteen points
cannot support a group fit, and the method that can (van Krevelen's, fitted to
hundreds of polymers) is a table this repository does not have. So a polymer
with no measured melting point is refused, and the refusal now reads
differently from "this polymer has no melting point at all", because the two
are different answers.

**Surface tension** is measured where a measurement exists and predicted from
the Sugden parachor otherwise. The measured route reads the same dispersive
and polar surface energies the adhesion expert does, because the total of
those two components *is* the surface tension; the parachor route is additive
over atoms, unsaturation and rings, so it covers any repeat unit RDKit can
read and lands within a third of the eight measurements that can check it.
Either way the same temperature correction is added: those values are for a
solid at room temperature, and a melt at 200 C is perceptibly slacker. Polymer
melts follow a near-linear ``d(gamma)/dT`` of about -0.06 mN/m/K over this
range, which is a large enough correction to matter - roughly 10 mN/m from room
temperature to a hot-melt nozzle - and well enough established to apply.

**Melt viscosity** is the one that is genuinely modelled, and the one to be
most careful about. The form is standard and the repository already holds both
inputs: viscosity scales as the 3.4 power of molar mass above the entanglement
threshold, which the mechanical expert computes from the packing length, and
the temperature dependence is WLF referenced to the glass transition, which the
Tg expert predicts. Anchoring at the conventional ``10^12 Pa s`` that *defines*
Tg closes the system with no fitted constant.

The honesty problem is that the universal WLF constants are universal only
approximately. Checked against polystyrene at 200 C the form lands about an
order of magnitude low, and that is not a bug to be tuned away - it is what
universal constants cost. So the prediction carries a one-sigma of a factor of
ten, stated as such.

A factor of ten sounds fatal and is not, because melt viscosity spans about
six orders of magnitude across ordinary polymers and chain lengths. A model
good to one order still separates a polymer that will pass a 4.4 mm orifice
from one that will not. What it cannot do is resolve the difference between 8
and 15 Pa s, and the uncertainty says so rather than implying otherwise - which
is the whole point of section 12's insistence that a stated uncertainty be
honest rather than flattering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MonomerRole,
    PolymerSpec,
    Tacticity,
)
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest

#: Viscosity at the glass transition, Pa s. This is a definition rather than a
#: measurement: Tg is conventionally the temperature at which a melt reaches
#: it. Using it as the anchor is what removes the need for a fitted constant.
VISCOSITY_AT_TG = 1.0e12

#: Williams-Landel-Ferry constants in their universal form, referenced to Tg.
WLF_C1, WLF_C2 = 17.44, 51.6

#: Entanglement threshold for viscosity is about twice the entanglement molar
#: mass; below it the melt is Rouse-like and the 3.4 power does not apply.
CRITICAL_OVER_ENTANGLEMENT = 2.0

#: Melt viscosity scales as this power of molar mass above the threshold.
REPTATION_EXPONENT = 3.4

#: And as this power below it. An unentangled chain drags through its
#: neighbours rather than reptating along a tube, and the Rouse result is
#: linear in molar mass. This is not an extrapolation of the 3.4 law into a
#: regime it does not hold in - it is the other law, joined at the threshold
#: where both are valid. A wax lives entirely on this branch, which is why a
#: blend cannot be scored without it.
ROUSE_EXPONENT = 1.0

#: One-sigma on the viscosity, in decades. See the module docstring: universal
#: WLF constants are worth about an order of magnitude, and claiming better
#: would be worse than claiming this.
VISCOSITY_LOG_SIGMA = 1.0

#: Sugden parachor increments, per atom, in the usual mixed units.
#:
#: This is what replaces the tabulated polymer surface energies as a *gate*.
#: The table answered for nine polymers; the parachor answers for any organic
#: structure, because it is additive over atoms and bonds rather than looked up
#: per substance - the same kind of scheme as the Joback groups this repository
#: already leans on.
#:
#: ``gamma = (P rho / M)^4``. Checked against the eight tabulated polymer
#: surface energies it reproduces them between 0.90x and 1.34x, worst case
#: poly(ethylene terephthalate). That is far looser than a measurement and far
#: better than nothing, and the prediction says so.
PARACHOR_ATOMS: dict[int, float] = {
    1: 17.1, 6: 4.8, 7: 12.5, 8: 20.0, 9: 27.5,
    14: 25.0, 16: 48.2, 17: 54.3, 35: 68.0, 53: 91.0,
}
PARACHOR_DOUBLE_BOND = 23.2
PARACHOR_TRIPLE_BOND = 46.4
PARACHOR_RING = 11.6
PARACHOR_SMALL_RING = 16.7

#: Worst ratio of the parachor route against the tabulated surface energies.
PARACHOR_SPREAD = 1.34


def parachor_surface_tension(repeat_unit: str, density_g_cm3: float) -> float | None:
    """Surface tension of a polymer from structure, mJ/m^2.

    Sugden's parachor is additive over atoms, unsaturation and rings, so this
    covers any repeat unit RDKit can parse instead of the nine that had a
    measured surface energy.
    """
    from rdkit import Chem

    # Capped rather than deleted: deleting an attachment point that sits inside
    # a branch leaves an empty "()" that RDKit refuses, which silently cost
    # every aromatic backbone - PEEK among them - its surface tension. The two
    # capping hydrogens are subtracted again below.
    mol = Chem.MolFromSmiles(repeat_unit.replace("[*]", "[H]"))
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    total = 0.0
    for atom in mol.GetAtoms():
        increment = PARACHOR_ATOMS.get(atom.GetAtomicNum())
        if increment is None:
            return None
        total += increment
    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            continue
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            total += PARACHOR_DOUBLE_BOND
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            total += PARACHOR_TRIPLE_BOND
    for ring in mol.GetRingInfo().AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            total += PARACHOR_RING + 3 * PARACHOR_DOUBLE_BOND
        elif len(ring) == 3:
            total += PARACHOR_SMALL_RING
        else:
            total += PARACHOR_RING

    mass = sum(a.GetMass() for a in mol.GetAtoms())
    # Dropping the attachment points added two capping hydrogens the repeat
    # unit does not have; both the parachor and the mass must lose them.
    total -= 2 * PARACHOR_ATOMS[1]
    mass -= 2 * 1.008
    if mass <= 0 or total <= 0:
        return None
    return (total * density_g_cm3 / mass) ** 4


#: Temperature coefficient of a polymer melt's surface tension, N/m/K.
SURFACE_TENSION_DGDT = -6.0e-5

#: Gas constant, J/(mol K).
GAS_CONSTANT = 8.31446261815324

#: How far above Tg the WLF form is carried before handing over to Arrhenius.
#: WLF is usually quoted as good from Tg to about Tg+100; this is the generous
#: end of that, and beyond it the two forms diverge fast.
WLF_RANGE_K = 100.0

#: Flow activation energies for the zero-shear melt viscosity, J/mol.
#:
#: Typical literature values for the melt regime, which is where they are
#: measured and where they apply. They vary with branching more than with
#: anything else - long-chain-branched polyethylene runs near 50 kJ/mol against
#: 27 for the linear polymer - so the entry here is for the linear or
#: conventional grade and a branched one is a different material for this
#: purpose. Quoted to no better than 15%, which is carried into the prediction.
FLOW_ACTIVATION_ENERGY: dict[str, float] = {
    "[*]CC[*]": 27.0e3,                              # polyethylene, linear
    "[*]CC(C)[*]": 42.0e3,                           # polypropylene
    "[*]CC(c1ccccc1)[*]": 105.0e3,                   # polystyrene
    "[*]CC(C)(C(=O)OC)[*]": 150.0e3,                 # PMMA
    "[*]CCO[*]": 28.0e3,                             # poly(ethylene oxide)
    "[*]CC(F)(F)[*]": 55.0e3,                        # PVDF
    "[*]NCCCCCC(=O)[*]": 65.0e3,                     # nylon-6
    "[*]NCCCCCCNC(=O)CCCCC(=O)[*]": 65.0e3,          # nylon-6,6
    "[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]": 75.0e3,        # PET
    "[*]CC(C)=CC[*]": 35.0e3,                        # cis-1,4-polyisoprene
    "[*]CC=CC[*]": 30.0e3,                           # 1,4-polybutadiene
    "[*]CC(C)(C)[*]": 50.0e3,                        # polyisobutylene
    "[*]CCCCCC(=O)O[*]": 40.0e3,                     # polycaprolactone
    "[*]OC(C)C(=O)[*]": 80.0e3,                      # polylactide
}

#: Relative one-sigma on a tabulated activation energy.
ACTIVATION_ENERGY_RTOL = 0.15

#: Flow activation energy from the glass transition, for a polymer the table
#: above does not carry.
#:
#: Both quantities measure the same thing from different ends: how much energy
#: it costs to move a chain segment past its neighbours. A stiff, polar or
#: bulky backbone raises the glass transition and raises the barrier to flow
#: together. Over the fourteen polymers above the correlation is r = +0.87 and
#: the fit is ``Ea(kJ/mol) = -47.3 + 0.403 Tg(K)``.
#:
#: Left out one polymer at a time and refitted, the worst held-out miss is
#: 1.66x, on PMMA - the ester side group hinders rotation more than its glass
#: transition alone reports. That ratio is carried as the one-sigma, which is
#: deliberately pessimistic: a worst case is not a standard deviation, and a
#: number that enters an exponential should be over-doubted rather than under.
#:
#: This is the difference between answering for fourteen polymers and
#: answering for any polymer whose glass transition can be predicted - which
#: is any polymer at all, since the group-contribution Tg expert has no table.
ACTIVATION_FROM_TG_SLOPE = 402.8
ACTIVATION_FROM_TG_INTERCEPT = -47.26e3

#: Held-out worst ratio of the correlation above, used as its one-sigma.
PREDICTED_ACTIVATION_RTOL = 0.66

#: Below roughly this the correlation would return a negative barrier, which is
#: not physics. A chain segment cannot flow more easily than a small molecule,
#: and 12 kJ/mol is about where the simplest flexible chains sit.
ACTIVATION_FLOOR = 12.0e3


def predicted_activation_energy(tg_kelvin: float) -> float:
    """Flow activation energy from the glass transition, J/mol."""
    return max(
        ACTIVATION_FLOOR,
        ACTIVATION_FROM_TG_INTERCEPT + ACTIVATION_FROM_TG_SLOPE * tg_kelvin,
    )

#: Chain regularity, from the structure, is what decides whether a polymer has
#: a crystalline phase at all - and it is computable rather than tabulated.
#:
#: A chain crystallises when consecutive repeat units can be placed in a
#: regular register. That fails when a backbone atom carries two *different*
#: substituents, because then each unit has a handedness and a chain built
#: without controlling it cannot repeat. Polyethylene's backbone carbons carry
#: two hydrogens, poly(vinylidene fluoride)'s carry two fluorines and
#: polyisobutylene's carry two methyls: all three are regular by construction.
#: Polypropylene's carry a hydrogen and a methyl, polystyrene's a hydrogen and
#: a phenyl, PMMA's a methyl and an ester: all three are configurational
#: stereocentres, and all three are amorphous in their ordinary atactic form.
#:
#: Substituents are compared by RDKit's canonical symmetry classes rather than
#: by element, so two branches count as the same substituent when they are
#: genuinely equivalent and not merely similar.
#:
#: Checked against the two tables this replaces, it gets nineteen of twenty:
#: every polymer that was listed amorphous is flagged as stereoirregular
#: except polyisobutylene, and every polymer with a tabulated melting point
#: comes out regular except the two whose own table entries already said
#: "isotactic only" and "stereoregular". The exception is listed below,
#: because the reason it is an exception is not structural.


def backbone_stereocentres(repeat_unit: str) -> list[str] | None:
    """Backbone atoms bearing two different substituents, as descriptions.

    An empty list means the chain is configurationally regular and will
    crystallise if anything will. ``None`` means the repeat unit could not be
    read at all - not the same answer, and not treated as one.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(repeat_unit)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        return None
    path = set(Chem.GetShortestPath(mol, stars[0], stars[1]))
    if len(path) < 2:
        return None
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))

    found: list[str] = []
    for idx in sorted(path):
        atom = mol.GetAtomWithIdx(idx)
        # A ring atom's neighbours run round the ring rather than off it, and
        # a ring in the backbone is rigid rather than handed.
        if atom.GetAtomicNum() == 0 or atom.IsInRing():
            continue
        subs = [nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetIdx() not in path]
        # Two substituents is what a stereocentre needs. One (an amide NH) is
        # planar; none is a methylene.
        if len(subs) != 2:
            continue
        if ranks[subs[0]] != ranks[subs[1]]:
            a, b = (mol.GetAtomWithIdx(s).GetSymbol() for s in subs)
            found.append(f"a backbone {atom.GetSymbol()} carrying both {a} and {b}")
    return found


#: The one polymer whose regularity the test above reads correctly and whose
#: behaviour it still gets wrong, and why that is not a structural question.
#:
#: Polyisobutylene's backbone carbons carry two methyls, so the chain is
#: regular and the test says it can crystallise. It can - under strain, which
#: is why a rubber band stiffens as you stretch it - and at rest it does not,
#: on a timescale anyone cares about. That is kinetics, and no amount of
#: looking at the repeat unit will produce it.
CRYSTALLISES_ONLY_UNDER_STRAIN: dict[str, str] = {
    "[*]CC(C)(C)[*]": (
        "polyisobutylene has a regular backbone and still does not crystallise at "
        "rest: the chain is too mobile to nucleate, and it orders only under strain"
    ),
}

#: Polymers that reach their decomposition temperature before their melting
#: point. This is thermal stability rather than crystallinity, which is why it
#: is separate from the regularity test above: a stereoregular sample of these
#: does have a crystalline phase, and heating it still does not give a melt.
DECOMPOSES_BEFORE_MELTING: dict[str, str] = {
    "[*]CC(C#N)[*]": (
        "polyacrylonitrile cyclises and decomposes above about 300 C, below where "
        "its crystalline phase would melt - which is why it is spun from solution "
        "and never from a melt"
    ),
}


@dataclass(frozen=True, slots=True)
class SolidModulus:
    """A measured room-temperature Young's modulus, Pa, and its spread."""

    modulus: float
    #: One-sigma, Pa. Dominated by degree of crystallinity, which is set by how
    #: the part was cooled, not by what it is made of.
    spread: float
    source: str


#: Measured Young's modulus at room temperature for the semicrystalline
#: polymers, Pa.
#:
#: These exist because the two-branch glass/rubber model cannot produce them.
#: It asks what the amorphous phase is doing, and a crystallite does not have a
#: glass transition to be above. Polyethylene at 25 C is a hundred degrees past
#: its Tg and the model returns 8 MPa; the measured value is about a thousand
#: times that. So a measurement is used where there is one, rather than a
#: composite model over a degree of crystallinity that is a processing variable
#: rather than a property.
#:
#: The spread is wide and is not measurement precision: crystallinity moves
#: with cooling rate, and a quenched and an annealed bar of one polymer differ
#: by tens of percent in stiffness.
SEMICRYSTALLINE_MODULUS: dict[str, SolidModulus] = {
    "[*]CC[*]": SolidModulus(1.0e9, 0.3e9, "high-density polyethylene, 23 C"),
    "[*]CC(C)[*]": SolidModulus(1.5e9, 0.3e9, "isotactic polypropylene, 23 C"),
    "[*]CC(F)(F)[*]": SolidModulus(2.0e9, 0.4e9, "poly(vinylidene fluoride), 23 C"),
    "[*]CCO[*]": SolidModulus(0.5e9, 0.2e9, "poly(ethylene oxide), 23 C"),
    "[*]NCCCCCC(=O)[*]": SolidModulus(2.7e9, 0.5e9, "nylon-6, dry, 23 C"),
    "[*]NCCCCCCNC(=O)CCCCC(=O)[*]": SolidModulus(2.8e9, 0.5e9, "nylon-6,6, dry, 23 C"),
    "[*]NCCCCCCCCCCC(=O)[*]": SolidModulus(1.3e9, 0.3e9, "nylon-11, 23 C"),
    "[*]NCCCCCCCCCCCC(=O)[*]": SolidModulus(1.4e9, 0.3e9, "nylon-12, 23 C"),
    "[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]": SolidModulus(2.8e9, 0.5e9, "PET, 23 C"),
    "[*]OCCCCOC(=O)c1ccc(cc1)C(=O)[*]": SolidModulus(2.4e9, 0.5e9, "PBT, 23 C"),
    "[*]C(F)(F)C(F)(F)[*]": SolidModulus(0.5e9, 0.2e9, "PTFE, 23 C"),
    "[*]CCCCCC(=O)O[*]": SolidModulus(0.4e9, 0.1e9, "polycaprolactone, 23 C"),
    "[*]OC(C)C(=O)[*]": SolidModulus(3.5e9, 0.5e9, "polylactide, 23 C"),
}


def semicrystalline_modulus(candidate: Candidate) -> SolidModulus | None:
    """Measured modulus for a semicrystalline homopolymer, or None.

    None for a copolymer, for a polymer with no measurement, and - importantly
    - for one whose crystallinity is decided by a tacticity the candidate has
    not stated. Atactic polypropylene is a tacky amorphous solid and must not
    be handed the isotactic polymer's 1.5 GPa.
    """
    spec = candidate.polymer
    if spec is None:
        return None
    repeat = _repeat_unit(spec)
    if repeat is None:
        return None
    if crystallinity(repeat, spec.tacticity).crystalline is not True:
        return None
    found = _match(repeat, SEMICRYSTALLINE_MODULUS)
    return found[1] if found is not None else None


@dataclass(frozen=True, slots=True)
class MeltingPoint:
    """A measured crystalline melting point, K, and what it depends on."""

    temperature: float
    #: One-sigma, K. Wide, because Tm moves with crystallinity and thermal
    #: history far more than it moves between compilations.
    spread: float
    source: str


#: Crystalline melting points, keyed by repeat-unit SMILES.
#:
#: Handbook values for the common semicrystalline polymers. The spread quoted
#: is not measurement precision: a polymer's melting endotherm shifts with
#: crystallinity, lamellar thickness and how fast it was cooled, and ten
#: degrees between a quenched and an annealed sample of one material is
#: ordinary. Tacticity matters more still, which is why the entry for
#: polypropylene is for the isotactic polymer and the atactic one - which is
#: what the bundled reference set actually carries - is amorphous and has no
#: melting point at all.
MELTING_POINTS: dict[str, MeltingPoint] = {
    "[*]CC[*]": MeltingPoint(408.0, 8.0, "polyethylene, linear; 135 C"),
    "[*]CC(C)[*]": MeltingPoint(438.0, 8.0, "polypropylene, ISOTACTIC only; 165 C"),
    "[*]CC(F)(F)[*]": MeltingPoint(450.0, 8.0, "poly(vinylidene fluoride); 177 C"),
    "[*]CCO[*]": MeltingPoint(338.0, 6.0, "poly(ethylene oxide); 65 C"),
    "[*]NCCCCCC(=O)[*]": MeltingPoint(493.0, 8.0, "nylon-6; 220 C"),
    "[*]NCCCCCCNC(=O)CCCCC(=O)[*]": MeltingPoint(538.0, 8.0, "nylon-6,6; 265 C"),
    "[*]NCCCCCCCCCCC(=O)[*]": MeltingPoint(463.0, 8.0, "nylon-11; 190 C"),
    "[*]NCCCCCCCCCCCC(=O)[*]": MeltingPoint(451.0, 8.0, "nylon-12; 178 C"),
    "[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]": MeltingPoint(538.0, 8.0, "PET; 265 C"),
    "[*]OCCCCOC(=O)c1ccc(cc1)C(=O)[*]": MeltingPoint(498.0, 8.0, "PBT; 225 C"),
    "[*]C(F)(F)C(F)(F)[*]": MeltingPoint(600.0, 10.0, "PTFE; 327 C"),
    # The low-melting pair. Polycaprolactone is the one polymer here a person
    # could handle molten without a burn unit - it is sold to hobbyists as
    # pellets you soften in hot water - and that makes it the only entry whose
    # melting point is a safety property rather than a processing one.
    "[*]CCCCCC(=O)O[*]": MeltingPoint(333.0, 5.0, "polycaprolactone; 60 C"),
    "[*]OC(C)C(=O)[*]": MeltingPoint(448.0, 10.0, "polylactide, stereoregular; 175 C"),
}

#: A polymer's crystallinity now comes from ``crystallinity(repeat, tacticity)``
#: rather than from a list of polymers someone remembered to add. The six
#: entries that list held - polystyrene, PMMA, PVC, poly(vinyl acetate),
#: polyacrylonitrile and polyisobutylene - are reproduced by the regularity
#: test above except for the last two, which are kept because their reasons
#: are not structural: see CRYSTALLISES_ONLY_UNDER_STRAIN and
#: DECOMPOSES_BEFORE_MELTING.


@dataclass(frozen=True, slots=True)
class Crystallinity:
    """Whether a polymer has a crystalline phase, and how that was decided."""

    #: True if it crystallises, False if it does not, None if the question is
    #: open because the candidate has not said which stereoisomer it is.
    crystalline: bool | None
    reason: str


def crystallinity(repeat_unit: str, tacticity: Tacticity) -> Crystallinity:
    """Does this polymer have a crystalline phase?

    Decided from chain regularity, not from a list. Three answers, and the
    third is the one that matters: an unspecified tacticity on a chain with a
    stereocentre leaves the question genuinely open, and answering it either
    way would hand a run a melting point for a material that may not melt.
    """
    strain = _match(repeat_unit, CRYSTALLISES_ONLY_UNDER_STRAIN)
    if strain is not None:
        return Crystallinity(False, str(strain[1]))
    decomposes = _match(repeat_unit, DECOMPOSES_BEFORE_MELTING)
    if decomposes is not None:
        return Crystallinity(False, str(decomposes[1]))

    centres = backbone_stereocentres(repeat_unit)
    if centres is None:
        return Crystallinity(None, "the repeat unit could not be read")
    if not centres:
        return Crystallinity(
            True,
            "the backbone carries no atom with two different substituents, so the "
            "chain is configurationally regular and can pack",
        )
    where = centres[0]
    if tacticity is Tacticity.ATACTIC:
        return Crystallinity(
            False,
            f"the chain has a stereocentre - {where} - and this candidate is the "
            "atactic one, so consecutive units cannot be placed in register",
        )
    if tacticity is Tacticity.UNSPECIFIED:
        return Crystallinity(
            None,
            f"the chain has a stereocentre - {where} - so whether it crystallises "
            "is decided by tacticity, which this candidate does not state",
        )
    return Crystallinity(
        True,
        f"the chain has a stereocentre - {where} - but this candidate is "
        f"{tacticity.value}, so the units repeat in register",
    )


def _repeat_unit(spec: PolymerSpec) -> str | None:
    """The single repeat unit of a homopolymer, or None for a copolymer."""
    chain = [m for m in spec.monomers if m.role is not MonomerRole.END_GROUP]
    return chain[0].smiles if len(chain) == 1 else None


def _match(smiles: str, table: dict) -> tuple[str, object] | None:
    from formulate import chem

    if chem.rdkit_available():
        canonical = chem.canonical_smiles(smiles)
        if canonical is not None:
            for key, value in table.items():
                if chem.canonical_smiles(key) == canonical:
                    return key, value
    return (smiles, table[smiles]) if smiles in table else None


def melt_viscosity(
    molar_mass: float,
    entanglement: float,
    tg: float,
    temperature: float,
    activation_energy: float | None = None,
) -> float:
    """Zero-shear melt viscosity, Pa s.

    Two regimes, joined where WLF stops being trustworthy.

    Near the glass transition the temperature dependence is WLF:
    ``eta = eta(Tg) * (M/Mc)^3.4 * 10^(-C1 dT / (C2 + dT))``, with the first
    factor fixed by the convention that *defines* Tg, so no constant is fitted.

    WLF is referenced to Tg and is not trustworthy much beyond ``WLF_RANGE_K``
    above it - which is a problem, because a melt-processing nozzle sits two to
    three hundred degrees above Tg for an ordinary semicrystalline polymer, and
    that is exactly the regime a hot-melt question asks about. There the melt
    follows an Arrhenius law with a polymer-specific flow activation energy.

    The two are joined at ``Tg + WLF_RANGE_K``: WLF fixes the value there, and
    Arrhenius carries it upward. That keeps the curve continuous and still
    introduces no fitted constant beyond the tabulated activation energy.
    """
    critical = CRITICAL_OVER_ENTANGLEMENT * entanglement
    exponent = REPTATION_EXPONENT if molar_mass >= critical else ROUSE_EXPONENT
    chain_factor = (molar_mass / critical) ** exponent

    def wlf(t: float) -> float:
        delta = t - tg
        return VISCOSITY_AT_TG * chain_factor * (10.0 ** (-WLF_C1 * delta / (WLF_C2 + delta)))

    crossover = tg + WLF_RANGE_K
    if temperature <= crossover:
        return wlf(temperature)
    if activation_energy is None:
        raise ValueError(
            "above the WLF range the melt needs a flow activation energy, and none "
            "is tabulated for this polymer"
        )
    return wlf(crossover) * math.exp(
        activation_energy / GAS_CONSTANT * (1.0 / temperature - 1.0 / crossover)
    )


class PolymerMeltExpert(Expert):
    """Melting point, melt viscosity and melt surface tension for a polymer."""

    id = "polymer_melt"
    version = "1"
    method = (
        "tabulated crystalline melting points; WLF-shifted reptation viscosity anchored "
        "at the viscosity that defines Tg, handing over to Arrhenius on a flow "
        "activation energy that is tabulated where known and predicted from the glass "
        "transition otherwise; surface tension from measured Owens-Wendt components "
        "where known and from the Sugden parachor otherwise, with a linear melt "
        "temperature correction"
    )
    family = PropertyFamily.INTERFACIAL
    supported_classes = frozenset({MaterialClass.POLYMER})
    supported_properties = frozenset({"melting_point", "shear_viscosity", "surface_tension"})
    dependencies = frozenset(
        {"glass_transition_temperature", "entanglement_molar_mass", "amorphous_density"}
    )

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        from formulate import chem

        return "" if chem.rdkit_available() else "RDKit is required to read a repeat unit"

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        basis = "measured melting points and surface energies; reptation form for viscosity"
        if candidate.polymer is None:
            return ApplicabilityDomain.outside("candidate carries no polymer", basis=basis)
        if _repeat_unit(candidate.polymer) is None:
            return ApplicabilityDomain.outside(
                "a copolymer's melt properties are not the average of its components'",
                basis=basis,
            )
        return ApplicabilityDomain(basis=basis)

    # -- the three properties ---------------------------------------------

    def _melting_point(self, prop, request, domain, repeat):
        phase = crystallinity(repeat, request.candidate.polymer.tacticity)
        if phase.crystalline is False:
            return Prediction.unsupported(
                prop,
                self.id,
                f"this polymer has no melting point: {phase.reason}. It softens through "
                "its glass transition over tens of degrees instead, which is not a set "
                "point a melt process can work to",
            )
        if phase.crystalline is None:
            return Prediction.unsupported(
                prop,
                self.id,
                f"whether this polymer melts at all is unanswered: {phase.reason}",
            )

        found = _match(repeat, MELTING_POINTS)
        if found is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "no measured melting point is tabulated for this polymer, and a melting "
                "point cannot be estimated from the repeat unit without knowing whether "
                "the polymer crystallises at all",
            )
        _, entry = found
        return self._make(
            prop, entry.temperature, "K", request, domain,
            std=entry.spread, kind=UncertaintyKind.ALEATORIC,
            basis=(
                "not measurement precision: a melting endotherm moves with crystallinity, "
                "lamellar thickness and cooling rate"
            ),
            notes=(
                f"tabulated: {entry.source}",
                "tacticity decides whether this polymer crystallises at all, and the "
                "tabulated value assumes the stereoregular form",
            ),
        )

    def _surface_tension(self, prop, request, domain, repeat):
        from .adhesion import POLYMER_SURFACES, SUBSTRATES

        temperature = request.conditions.temperature
        t_kelvin = temperature.to_canonical().value if temperature is not None else 298.15

        found = _match(repeat, POLYMER_SURFACES)
        if found is not None:
            energy = SUBSTRATES[found[1]]
            room = energy.total
            spread = energy.spread
            source = f"measured at room temperature: {energy.basis}"
        else:
            density = request.dependency_value("amorphous_density", "g/cm^3")
            if density is None:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    "no measured surface energy is tabulated for this polymer, and the "
                    "parachor that would replace it needs an amorphous density, which "
                    "no upstream expert could supply",
                )
            room = parachor_surface_tension(repeat, density)
            if room is None:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    "no measured surface energy is tabulated for this polymer, and the "
                    "parachor carries no increment for at least one element in this "
                    "repeat unit",
                )
            spread = room * (PARACHOR_SPREAD - 1.0)
            source = (
                f"predicted: Sugden parachor over the repeat unit at "
                f"{density:.3f} g/cm^3, reproducing the tabulated polymer surface "
                f"energies between 0.90x and {PARACHOR_SPREAD:.2f}x"
            )

        # mJ/m^2 is numerically N/m * 1000
        gamma = room / 1000.0 + SURFACE_TENSION_DGDT * (t_kelvin - 298.15)
        if gamma <= 0:
            return Prediction.failed(
                prop, self.id,
                f"the linear melt correction drives surface tension negative at "
                f"{t_kelvin:.0f} K, which means the extrapolation has left its range",
            )
        drift = SURFACE_TENSION_DGDT * (t_kelvin - 298.15)
        return self._make(
            prop, gamma, "N/m", request, domain,
            std=(spread / 1000.0) + abs(drift) * 0.25,
            kind=UncertaintyKind.COMBINED,
            basis=(
                "spread on the room-temperature surface energy, plus a quarter of the "
                "temperature correction, because the coefficient is a typical value "
                "rather than this polymer's own"
            ),
            notes=(
                source,
                f"corrected by {drift*1000:+.1f} mN/m to {t_kelvin:.0f} K at "
                f"{SURFACE_TENSION_DGDT*1000:.2f} mN/m/K",
            ),
        )

    def _shear_viscosity(self, prop, request, domain, repeat):
        spec = request.candidate.polymer
        molar_mass = spec.number_average_molar_mass
        if molar_mass is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "melt viscosity goes as the 3.4 power of chain length, so a polymer with "
                "no stated number-average molar mass has no melt viscosity: the same "
                "repeat unit spans six orders of magnitude between an oligomer and a "
                "high polymer",
            )
        mass = molar_mass.to_canonical().value  # kg/mol

        tg = request.context.get("glass_transition_temperature")
        me = request.context.get("entanglement_molar_mass")
        for name, dep in (("glass transition", tg), ("entanglement molar mass", me)):
            if dep is None or dep.quantity is None:
                return Prediction.unsupported(
                    prop, self.id,
                    f"melt viscosity needs this polymer's {name}, which no upstream expert "
                    f"could supply",
                )
        tg_k = tg.quantity.to_canonical().value
        me_kg = me.quantity.to_canonical().value

        critical = CRITICAL_OVER_ENTANGLEMENT * me_kg
        entangled = mass >= critical

        temperature = request.conditions.temperature
        if temperature is None:
            return Prediction.unsupported(
                prop, self.id,
                "melt viscosity is meaningless without a temperature; state one in the "
                "requirement's conditions",
            )
        t_kelvin = temperature.to_canonical().value
        if t_kelvin <= tg_k:
            return Prediction.unsupported(
                prop, self.id,
                f"at {t_kelvin:.0f} K this polymer is at or below its glass transition "
                f"({tg_k:.0f} K); it is a solid, not a melt",
            )
        above_wlf = t_kelvin > tg_k + WLF_RANGE_K
        activation = None
        activation_rtol = ACTIVATION_ENERGY_RTOL
        activation_source = ""
        if above_wlf:
            found = _match(repeat, FLOW_ACTIVATION_ENERGY)
            if found is None:
                activation = predicted_activation_energy(tg_k)
                activation_rtol = PREDICTED_ACTIVATION_RTOL
                activation_source = (
                    f"no flow activation energy is tabulated for this polymer, so it is "
                    f"predicted from its glass transition at "
                    f"{ACTIVATION_FROM_TG_SLOPE/1e3:.3f} kJ/mol/K"
                )
            else:
                activation = found[1]
                activation_source = "flow activation energy is tabulated for this polymer"

        eta = melt_viscosity(mass, me_kg, tg_k, t_kelvin, activation)
        # One decade of one-sigma, expressed in linear units for the ranker. The
        # Arrhenius branch adds the activation energy's own error on top, and it
        # enters an exponential, so it is propagated rather than waved at.
        log_sigma = VISCOSITY_LOG_SIGMA
        if activation is not None:
            exponent_sigma = (
                activation * activation_rtol / GAS_CONSTANT
                * abs(1.0 / t_kelvin - 1.0 / (tg_k + WLF_RANGE_K))
            )
            log_sigma = math.sqrt(log_sigma**2 + (exponent_sigma / math.log(10.0)) ** 2)
        sigma = eta * (10.0**log_sigma - 1.0) / 2.0
        branch = (
            f"WLF from Tg = {tg_k:.0f} K to {tg_k + WLF_RANGE_K:.0f} K, then Arrhenius "
            f"at {activation/1e3:.0f} kJ/mol to {t_kelvin:.0f} K - {activation_source}"
            if activation is not None
            else f"WLF shift from Tg = {tg_k:.0f} K to {t_kelvin:.0f} K"
        )
        return self._make(
            prop, eta, "Pa*s", request, domain,
            std=sigma, kind=UncertaintyKind.EPISTEMIC,
            basis=(
                f"{log_sigma:.1f} decades, one sigma. The WLF constants used are the "
                "universal ones, which against polystyrene at 200 C land about an order "
                "of magnitude low; a tighter claim would be flattering rather than "
                f"honest. Above the WLF range the activation energy's own "
                f"{activation_rtol*100:.0f}% is propagated through the exponential and "
                "added in quadrature"
            ),
            notes=(
                f"chain {mass*1e3:.0f} g/mol against a critical mass of {critical*1e3:.0f}: "
                + (
                    f"entangled, so reptation at the {REPTATION_EXPONENT} power"
                    if entangled
                    else f"unentangled, so Rouse drag at the {ROUSE_EXPONENT:.0f} power - "
                    "this is a wax, not a polymer, and carries no load"
                ),
                branch,
                "an order of magnitude still separates a melt that will pass a "
                "millimetre orifice from one that will not, because melt viscosity "
                "spans six orders across ordinary polymers",
            ),
            conditions=request.conditions,
        )

    def _predict_one(self, prop, request: PredictionRequest, domain):
        spec = request.candidate.polymer
        if spec is None:
            return Prediction.unsupported(prop, self.id, "candidate carries no polymer")
        repeat = _repeat_unit(spec)
        if repeat is None:
            return Prediction.unsupported(
                prop, self.id,
                "a copolymer's melt properties are not the mole-weighted average of its "
                "components'",
            )
        if prop == "melting_point":
            return self._melting_point(prop, request, domain, repeat)
        if prop == "surface_tension":
            return self._surface_tension(prop, request, domain, repeat)
        return self._shear_viscosity(prop, request, domain, repeat)
