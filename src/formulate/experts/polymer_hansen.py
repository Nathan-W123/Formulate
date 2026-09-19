"""Solubility parameters of a polymer repeat unit.

Nothing in this engine gave a polymer a solubility parameter before this
module.  That is why :mod:`formulate.experts.dissolution` can only answer for
the seven polymers whose spheres are tabulated, and why the whole "will this
dissolve, will these two blend" path was blind for anything else.

Method
------
Hoftyzer-Van Krevelen group contributions, evaluated over the repeat unit::

    delta_d = sum(F_di) / V
    delta_p = sqrt(sum(F_pi^2)) / V
    delta_h = sqrt(sum(E_hi) / V)

with ``V`` the repeat unit's molar volume.  The increment table is that of
van Krevelen & Te Nijenhuis, *Properties of Polymers*, 4th ed. (Elsevier,
2009), Table 7.10, reproduced as Table 3.2 in Hansen, *Hansen Solubility
Parameters: A User's Handbook*, 2nd ed. (CRC Press, 2007).  ``V`` is not
modelled a second time here: it is ``M_repeat / rho`` with ``rho`` a declared
dependency on ``amorphous_density``, which
:class:`~formulate.experts.polymer.PolymerDensityExpert` supplies.  The repeat
unit is isolated the way :mod:`formulate.experts.polymer` already does it -
fragment a trimer and a dimer built by ``link_repeat_units`` and difference
them - so the methyl caps cancel and every group is scored in a real chain
environment.

What it is worth, measured rather than claimed
----------------------------------------------
Two independent checks, neither of which fitted anything.  Every increment
below is transcribed, not tuned; increments that failed these checks were
dropped from the table and the repeat units needing them are refused.

*Liquids, where the reference is the same quantity the method predicts.*
Against the 50 bundled reference compounds, 35 are inside the shipped group
set and give RMSE 0.67 / 1.21 / 1.52 MPa^0.5 on delta_d / delta_p / delta_h
and 0.93 on the total, with bias -0.31 / +0.17 / -0.46.  A wider check over
120 solvents - every CAS where two of the four Hansen compilations in
``chemicals`` agree within 1.5 MPa^0.5 and ``thermo`` has a measured liquid
molar volume - gives 0.68 / 1.85 / 1.52 and 1.19 on the total.

*Polymers, where the reference is a different quantity.*  Against the five
usable handbook triples in ``dissolution.SOLUBILITY_SPHERES`` (polystyrene,
PMMA, PVC, poly(vinyl acetate), polycarbonate; polyamide 66 is refused and
cellulose acetate has no repeat unit at a fixed degree of substitution), run
end to end through this expert behind ``PolymerDensityExpert``'s own
amorphous density rather than through measured densities, the RMS
disagreement is 2.77 / 4.39 / 3.13 per component and 3.48 on the total, with
a systematic bias of -2.18 / -2.46 / -1.62.  Most of that is *not* group
error: a Hansen sphere centre is fitted to which solvents dissolve a polymer,
and is not the polymer's cohesion parameter.  Polystyrene's handbook centre
totals 22.5 MPa^0.5 where its measured Hildebrand parameter is about 18.6,
which is what this method returns.  The two numbers must not be substituted
for one another, and the shipped error bar is 3.9-6.1 rather than the 0.7-1.9
the liquid check alone would justify precisely because of that offset.  Sized
that way, 13 of the 15 component predictions fall inside their own one-sigma
bound; sized at the raw RMS, only 7 did, which is the failure mode
``formulate calibrate`` exists to catch.

Coverage: 46 of the 57 repeat units in ``data/reference_polymers.json``.  The
eleven refusals are the six amides, the four fluorine- or geminal-dihalide
repeat units, and poly(ethylene naphthalate).

Limitations
-----------
The number is the amorphous phase's, and a solubility parameter is necessary
but not sufficient: polyethylene comes out at 17.5 MPa^0.5 through this
engine's own density (16.5 at its measured one) and dissolves in nothing at
room temperature, because nothing here sees crystallinity.
Tacticity is invisible to an additive repeat-unit sum and is not in the error
bar.  Chain length is not: a sum over *interior* units is the high-polymer
limit, and below the same Mn threshold ``polymer.py`` uses the end groups
carry a real share of the cohesive energy - two hydroxyls on a 400 g/mol
poly(ethylene oxide) diol move delta_h by about 5 MPa^0.5, more than the bar
quoted here - so a short chain is marked out of domain rather than answered
as though it were infinite.

No molar volume is reported.  The repeat-unit volume ``M/rho`` is what every
parameter here is divided by, and it is recorded in the provenance of each
prediction, but it is *not* offered as ``molar_volume_liquid``: a glassy or
rubbery repeat unit is not a saturated liquid, which is the same distinction
``properties.py`` makes when it refuses to call an amorphous polymer density
``liquid_density``.  Ranking the two in one column would compare different
states of matter.  The name this wants is ``molar_volume_repeat_unit``, which
the registry does not have.

Van Krevelen's symmetry correction on ``F_p`` is *not* implemented - it could
not be reproduced against measured solvents - so geminal dihalides are
refused, and ring-type dipole cancellation is only warned about (1,4-dioxane
comes out at delta_p 6.6 against a measured 1.8).

Two entries are used only as far as they were published.  Ring closure is
tabulated for a five- or six-membered ring, which is also all the acceptance
set contains, so a strained ring, a macrocycle and a bicyclic skeleton are
refused rather than given the same 190.  The aromatic split is pinned by
phenyl and p-phenylene, so a benzene ring carrying three or four substituents
is answered but flagged: it is an extrapolation two steps past the last
tabulated ring, and poly(2,6-dimethyl-1,4-phenylene oxide) is the case in the
reference set that depends on it.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Mapping

from formulate.core.candidate import Candidate, MaterialClass, MonomerRole, PolymerSpec
from formulate.core.prediction import Prediction, PredictionStatus
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest
from .interfacial import propagate
from .polymer import (
    _MIN_NUMBER_AVERAGE_MOLAR_MASS,
    _architecture_reason,
    link_repeat_units,
    repeat_unit_mass,
)

# --------------------------------------------------------------------------
# The increment table
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GroupIncrement:
    """One row of the Hoftyzer-Van Krevelen table.

    ``f_d`` and ``f_p`` are molar attraction constants in
    ``(J cm^3)^0.5 / mol``, so that dividing by a molar volume in
    ``cm^3/mol`` gives ``(J/cm^3)^0.5``, which is ``MPa^0.5``.  ``e_h`` is a
    hydrogen-bonding cohesive energy in ``J/mol``.
    """

    name: str
    #: SMARTS matched over the *oligomer*, not over the bare repeat unit: a
    #: dangling valence is not a chemical environment any increment describes.
    smarts: str
    f_d: float
    f_p: float
    e_h: float


#: The transcribed table, in match priority order.
#:
#: Priority is chemical, not arbitrary.  A carboxylic acid must claim its own
#: carbonyl before the ester pattern sees it, and the ester before the bare
#: ketone, or -COOH would be scored as a ketone plus a hydroxyl and come out
#: at delta_h 25 instead of 13 for acetic acid.  Every pattern claims the
#: atoms it matches; an atom already claimed by a higher-priority group is
#: never double-counted, and any heavy atom still unclaimed at the end is a
#: refusal rather than a zero.
#:
#: Groups van Krevelen tabulates that are deliberately ABSENT here, and why:
#: amide (-CONH-), urethane, urea, the three amines, nitro, -S-, -Si-, -PO4,
#: -F, -Br, -I and -C#C-.  These could not be transcribed with confidence, and
#: where two neighbouring entries suggested a composite route it failed the
#: acceptance check - the -CO- plus -NH- route puts nylon-6,6 at a total
#: 20.1 MPa^0.5 against a measured Hildebrand parameter near 28.  A repeat
#: unit needing any of them is refused.  That is the whole of the amide, amine
#: and fluoropolymer gap, and it is a gap rather than a guess on purpose.
HVK_GROUPS: tuple[GroupIncrement, ...] = (
    # -- oxygenated functional groups, highest priority ---------------------
    GroupIncrement("-COOH", "[CX3](=[OX1])[OX2H1]", 530.0, 420.0, 10000.0),
    GroupIncrement("-COO-", "[CX3](=[OX1])[OX2;!$([OX2H1])]", 390.0, 490.0, 7000.0),
    GroupIncrement("-CHO", "[CX3H1]=[OX1]", 470.0, 800.0, 4500.0),
    GroupIncrement("-CO-", "[CX3](=[OX1])", 290.0, 770.0, 2000.0),
    GroupIncrement("-CN", "[CX2]#[NX1]", 430.0, 1100.0, 2500.0),
    GroupIncrement("-OH", "[OX2H1]", 210.0, 500.0, 20000.0),
    GroupIncrement("-O-", "[OX2]", 100.0, 400.0, 3000.0),
    GroupIncrement("-Cl", "[Cl]", 450.0, 550.0, 400.0),
    # -- aromatic carbon, per atom ------------------------------------------
    #
    # Van Krevelen tabulates whole rings: phenyl (C6H5) at F_d 1430 and
    # p-phenylene (C6H4) at 1270, both with F_p 110 and E_h 0.  Those two
    # published numbers determine a per-atom split exactly - 5a + b = 1430 and
    # 4a + 2b = 1270 give a = 265 for an aromatic CH and b = 105 for a
    # substituted aromatic carbon - and the split is used rather than the
    # whole-ring values so that a tri- or tetra-substituted ring, which the
    # published table does not list, is still scored from published numbers.
    # It is read off two tabulated rings, not an increment invented here, and
    # it is bounded twice: a ring with no substituents or with five or more is
    # refused outright, and a ring with three or four - past the last tabulated
    # entry rather than between the two - is answered but marked out of domain
    # by ``_COUNTED_FLAGS``, since nothing measured stands behind it.
    GroupIncrement("aromatic CH", "[cX3;H1]", 265.0, 0.0, 0.0),
    GroupIncrement("aromatic C(sub)", "[cX3;H0]", 105.0, 0.0, 0.0),
    # -- olefinic carbon ----------------------------------------------------
    #
    # Written as single-atom patterns with a recursive environment rather than
    # as "[CX3H1]=[CX3]".  A two-atom pattern on a symmetric double bond such
    # as -CH=CH- matches one atom set twice, RDKit uniquifies the duplicate
    # away, and the second carbon is left unassigned - which silently refused
    # cis-1,4-polybutadiene until it was caught.
    GroupIncrement("=CH2", "[CX3H2;$([CX3]=[CX3])]", 400.0, 0.0, 0.0),
    GroupIncrement("=CH-", "[CX3H1;$([CX3]=[CX3])]", 200.0, 0.0, 0.0),
    GroupIncrement("=C<", "[CX3H0;$([CX3]=[CX3])]", 70.0, 0.0, 0.0),
    # -- saturated carbon, lowest priority ----------------------------------
    GroupIncrement("-CH3", "[CX4H3]", 420.0, 0.0, 0.0),
    GroupIncrement("-CH2-", "[CX4H2]", 270.0, 0.0, 0.0),
    GroupIncrement(">CH-", "[CX4H1]", 80.0, 0.0, 0.0),
    GroupIncrement(">C<", "[CX4H0]", -70.0, 0.0, 0.0),
)

#: Ring corrections, counted per ring rather than per atom and therefore held
#: outside the SMARTS table.
#:
#: The aliphatic one is van Krevelen's ring closure at F_d 190: six -CH2-
#: alone put cyclohexane at delta_d 14.9 against a measured 16.8, and with the
#: correction at 16.6.  There is no aromatic equivalent, because the published
#: phenyl and p-phenylene values already contain it - adding 190 to benzene
#: takes it from 17.8 to 19.9 against a measured 18.4.  What the aromatic ring
#: does carry is a single F_p of 110 for the ring as a whole, which is why it
#: cannot see where the substituents sit.
RING_INCREMENTS: Mapping[str, GroupIncrement] = {
    "aliphatic ring": GroupIncrement("aliphatic ring", "", 190.0, 0.0, 0.0),
    "aromatic ring": GroupIncrement("aromatic ring", "", 0.0, 110.0, 0.0),
}

_INCREMENTS: Mapping[str, GroupIncrement] = {
    **{g.name: g for g in HVK_GROUPS},
    **RING_INCREMENTS,
}

#: The increments are 25 degC values, and the upstream packing factor carries
#: no temperature dependence either, so this is the window both are defensible
#: over.  It is the same window ``PolymerDensityExpert`` uses, deliberately:
#: widening it here would only produce a solubility parameter resting on a
#: density that was refused.  delta falls by roughly 0.02 MPa^0.5/K, so the
#: drift across the window is about +/-0.5 MPa^0.5, inside the error bar.
TEMPERATURE_WINDOW_K = (273.0, 323.0)

#: Held-out error of the method, per component, in MPa^0.5.
#:
#: Built from the larger of two measurements, never their average: what the
#: method misses on liquids, where the reference quantity is the quantity
#: predicted (RMSE 0.67 / 1.21 / 1.52 over 35 bundled compounds, 0.68 / 1.85 /
#: 1.52 over 120 cross-source-agreed solvents), and what it misses on
#: polymers, where the reference is a Hansen sphere centre and therefore a
#: different quantity (RMS 2.77 / 4.39 / 3.13 over the five usable handbook
#: triples).  The polymer term wins on every component and sets the bar.
#:
#: Both polymer figures are measured through this expert behind the density
#: expert it actually depends on, not through the measured densities, because
#: a bar quoted for a pipeline has to be measured on that pipeline; the two
#: differ by under 0.05 MPa^0.5 here, which is the evidence that the density
#: model is not the dominant term rather than a licence to quote either.
#:
#: The bar is NOT that RMS.  Most of the polymer disagreement is a systematic
#: offset (-2.18 / -2.46 / -1.62) rather than scatter, and a symmetric bar
#: centred on a biased prediction covers almost nothing until it is at least
#: the size of the offset: quoting the RMS directly left only 7 of the 15
#: component predictions inside their own one sigma, against the ~68 per cent
#: a correct estimate implies.  What is shipped is the offset plus the
#: residual scatter, ``|bias| + sqrt(RMS^2 - bias^2)``, which puts 13 of 15
#: inside - wide rather than narrow, which is the direction to err when the
#: alternative is a flattering number a ranking would believe.  The offset is
#: reported and not corrected for, because it is mostly the
#: sphere-centre-versus-cohesion-parameter difference and "correcting" it
#: would bake that confusion into the value.
#:
#: Five polymers cannot calibrate anything.  Used only as a floor over the
#: liquid RMSE it can widen the bar and never narrow it, which is the only
#: role a set that small can honestly play.
METHOD_STD_MPA_SQRT: Mapping[str, float] = {
    "hansen_dispersion": 3.89,
    "hansen_polar": 6.10,
    "hansen_hydrogen_bonding": 4.30,
    "hildebrand_solubility_parameter": 4.90,
}

#: Systematic offsets against the five handbook spheres, reported in the
#: prediction notes and deliberately NOT corrected for.  Correcting to a
#: validation set is how a validation set stops meaning anything, and most of
#: this offset is the sphere-centre-versus-cohesion-parameter difference
#: rather than group error, so "correcting" it would bake that confusion in.
POLYMER_BIAS_MPA_SQRT: Mapping[str, float] = {
    "hansen_dispersion": -2.18,
    "hansen_polar": -2.46,
    "hansen_hydrogen_bonding": -1.62,
    "hildebrand_solubility_parameter": -2.72,
}

#: Structural features the table handles badly, each with the bias measured on
#: the 120-solvent set, as (SMARTS, domain score, sentence).  These warn rather
#: than refuse: the value is still the best available and the direction of the
#: error is known, which is more use to a ranking than no value at all.
_DOMAIN_FLAGS: tuple[tuple[str, float, str], ...] = (
    (
        "[cX3][CX3]=[OX1]",
        0.5,
        "a carbonyl conjugated to an aromatic ring: the ring carries one F_p of 110 "
        "that cannot see the conjugation, and delta_p runs 3.9 MPa^0.5 low over the "
        "five such solvents in the acceptance set (dimethyl phthalate by 6.5)",
    ),
    (
        "[OX2;R]~[*;R]~[*;R]~[OX2;R]",
        0.5,
        "two ethers in one ring: the F_p terms are added in quadrature and cannot "
        "cancel, so a symmetric cyclic diether is over-predicted on the polar term - "
        "1,4-dioxane comes out at 6.6 MPa^0.5 against a measured 1.8",
    ),
    (
        "[CX3]=[CX3]",
        0.7,
        "an olefinic backbone: the =CH- increments carry E_h = 0, and delta_h runs "
        "2.1 MPa^0.5 low over the seventeen alkenes in the acceptance set",
    ),
    (
        "[CX2]#[NX1]",
        0.7,
        "a nitrile: delta_p runs 3.2 MPa^0.5 high over the six nitriles in the "
        "acceptance set",
    ),
    (
        "[OX2;!R][CX3;!R](=[OX1])[OX2;!R]",
        0.6,
        "an open-chain carbonate, which the table has no entry for: the ownership rule "
        "scores it as an ester plus an ether, which lands within 0.3 MPa^0.5 on delta_d "
        "for dimethyl and diethyl carbonate but up to 5.6 high on delta_h for diethyl "
        "carbonate, even though bisphenol-A polycarbonate comes out within 0.2/2.8/0.0 "
        "of its published sphere",
    ),
)

#: Flags that no SMARTS can express, counted the same trimer-minus-dimer way
#: as the ones above: (score, sentence).
#:
#: The aromatic split is pinned by exactly two published rings, phenyl and
#: p-phenylene.  One and two substituents are those two entries; three and four
#: are an extrapolation two steps past the last of them, and the bundled
#: acceptance set contains one trisubstituted benzene and no tetrasubstituted
#: one, so there is no measurement standing behind either.  It is scored rather
#: than refused because the alternative is refusing poly(2,6-dimethyl-1,4-
#: phenylene oxide), whose total lands at 19.3 MPa^0.5 against a literature
#: 19.0-19.6 - but the ranking is told the number rests on an extrapolation.
_COUNTED_FLAGS: tuple[tuple[float, str], ...] = (
    (
        0.7,
        "a benzene ring carrying three or four substituents: the per-atom split is "
        "fixed by the published phenyl (one substituent) and p-phenylene (two) entries "
        "and is an extrapolation beyond them, with one trisubstituted and no "
        "tetrasubstituted ring in the acceptance set to check it against",
    ),
)

#: g/cm^3 per kg/m^3.  The upstream density is canonical SI; the increment
#: table is in cm^3/mol, so the conversion happens once, here, on a plain
#: density where a factor of a thousand is obvious.  The one conversion that
#: matters - MPa^0.5 to Pa^0.5 - is left entirely to the shared unit registry,
#: because a hand-written factor on the square root of a pressure is the
#: mistake ``dissolution.py`` had to document.
_KG_M3_TO_G_CM3 = 1e-3


class RepeatUnitRefused(ValueError):
    """The repeat unit cannot be scored by this table, with the reason why."""


# --------------------------------------------------------------------------
# Fragmentation
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _compiled() -> tuple[tuple[GroupIncrement, object], ...]:
    from rdkit import Chem

    out = []
    for group in HVK_GROUPS:
        pattern = Chem.MolFromSmarts(group.smarts)
        if pattern is None:  # pragma: no cover - a typo in the table above
            raise ValueError(f"unparseable SMARTS for {group.name!r}")
        out.append((group, pattern))
    return tuple(out)


def _describe_unassigned(atom) -> str:
    """What is actually unscoreable about one atom, in terms a chemist can act on.

    Naming only the element misleads whenever the element *is* in the table.
    Poly(ethylene furanoate) refused for "no increment covering O" reads as a
    contradiction of the ``-O-`` row three lines above it; what is missing is
    an aromatic heterocycle, and saying so is the difference between a refusal
    a chemist can route around and one that looks like a bug.
    """
    symbol = atom.GetSymbol()
    if atom.GetIsAromatic():
        return (
            f"aromatic {symbol} (a heteroaromatic ring - furan, thiophene, pyridine - "
            "is not a benzene ring, and the phenyl and p-phenylene entries may not be "
            "borrowed for one)"
        )
    if atom.GetFormalCharge():
        return (
            f"formally charged {symbol} (these increments are for neutral organic "
            "groups; an ionomer's cohesion is Coulombic and no group sum reaches it)"
        )
    return symbol


def _count_groups(mol) -> dict[str, int]:
    """Assign every heavy atom to exactly one increment, or refuse.

    Ownership is what makes the sum additive: a pattern that would reuse an
    atom a higher-priority pattern already claimed is skipped, and an atom no
    pattern claims makes the whole repeat unit a refusal.  That is the
    difference between "this table has no amide increment" and "this amide
    contributed nothing", which are the same number and opposite claims.
    """
    owner: dict[int, str] = {}
    counts: dict[str, int] = {}
    for group, pattern in _compiled():
        for match in mol.GetSubstructMatches(pattern, uniquify=True):
            if any(index in owner for index in match):
                continue
            for index in match:
                owner[index] = group.name
            counts[group.name] = counts.get(group.name, 0) + 1

    unassigned = sorted(
        {
            _describe_unassigned(mol.GetAtomWithIdx(i))
            for i in range(mol.GetNumAtoms())
            if i not in owner
        }
    )
    if unassigned:
        raise RepeatUnitRefused(
            "the Hoftyzer-Van Krevelen table shipped here has no increment covering "
            + ", ".join(unassigned)
            + " in this environment, so this repeat unit is refused rather than scored "
            "with a hole in it; what was left out of the table rather than guessed is "
            "amide, amine, nitro, sulfur, silicon, phosphorus, fluorine, bromine, iodine "
            "and alkyne"
        )

    for name, count in _ring_counts(mol).items():
        counts[name] = counts.get(name, 0) + count
    return counts


def _ring_counts(mol) -> dict[str, int]:
    counts = {"aliphatic ring": 0, "aromatic ring": 0}
    for ring in mol.GetRingInfo().AtomRings():
        aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        counts["aromatic ring" if aromatic else "aliphatic ring"] += 1
    return {k: v for k, v in counts.items() if v}


@functools.lru_cache(maxsize=1)
def _cyclic_ester_pattern():
    from rdkit import Chem

    return Chem.MolFromSmarts("[CX3;R](=[OX1])[OX2;R]")


def _structural_refusal(mol) -> str | None:
    """Reasons a structure is outside the table even though every atom matched."""
    if not any(atom.GetSymbol() == "C" for atom in mol.GetAtoms()):
        return "these are organic-group increments and this fragment has no carbon"

    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        if any(ring_info.NumAtomRings(i) > 1 for i in ring):
            return (
                "a fused aromatic ring: the table lists phenyl and p-phenylene only, "
                "and a naphthalene is not two benzenes"
            )
        if len(ring) != 6 or any(mol.GetAtomWithIdx(i).GetSymbol() != "C" for i in ring):
            return (
                "a heteroaromatic or non-six-membered aromatic ring, for which no "
                "increment exists and no benzene value may be borrowed"
            )
        substituted = sum(1 for i in ring if mol.GetAtomWithIdx(i).GetTotalNumHs() == 0)
        if not 1 <= substituted <= 4:
            return (
                f"a benzene ring carrying {substituted} substituents, outside the phenyl "
                "(one) to p-phenylene (two) entries the per-atom split interpolates"
            )

    # Van Krevelen tabulates one ring-closure increment, for a five- or
    # six-membered ring.  A three-membered ring is strained and a macrocycle is
    # not a ring closure in the same sense; the 190 stands for what a normal
    # ring does to packing, and neither of those is that.  A bicyclic skeleton
    # is not two ring closures either, for the reason a naphthalene is not two
    # benzenes.
    for ring in ring_info.AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        if len(ring) not in (5, 6):
            return (
                f"a {len(ring)}-membered aliphatic ring: the ring-closure increment is "
                "tabulated for five- and six-membered rings only, and neither the strain "
                "of a smaller ring nor the freedom of a larger one is in that one number"
            )
        if any(ring_info.NumAtomRings(i) > 1 for i in ring):
            return (
                "a fused or bridged aliphatic ring: ring closure is one tabulated "
                "increment per isolated ring, and a bicyclic skeleton is not two of them"
            )

    for atom in mol.GetAtoms():
        halogens = sum(
            1 for n in atom.GetNeighbors() if n.GetSymbol() in ("F", "Cl", "Br", "I")
        )
        if halogens >= 2:
            return (
                "two or more halogens on one atom: van Krevelen's symmetry correction "
                "on F_p governs this case and is not implemented here, because the rule "
                "as stated gives a quarter-weight to both dichloromethane and carbon "
                "tetrachloride where the measured delta_p wants a half and a zero"
            )

    if mol.HasSubstructMatch(_cyclic_ester_pattern()):
        return (
            "an ester or carbonate carbonyl inside a ring: the open-chain increment "
            "assumes the dipoles partly cancel, and a ring locks them - "
            "gamma-butyrolactone and propylene carbonate come out 8 to 10 MPa^0.5 low "
            "on delta_p"
        )
    return None


@functools.lru_cache(maxsize=1)
def _compiled_flags() -> tuple[tuple[object, float, str], ...]:
    from rdkit import Chem

    return tuple(
        (Chem.MolFromSmarts(smarts), score, sentence)
        for smarts, score, sentence in _DOMAIN_FLAGS
    )


def _crowded_ring_count(mol) -> int:
    """Benzene rings carrying three or four substituents."""
    ring_info = mol.GetRingInfo()
    crowded = 0
    for ring in ring_info.AtomRings():
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        if sum(1 for i in ring if mol.GetAtomWithIdx(i).GetTotalNumHs() == 0) >= 3:
            crowded += 1
    return crowded


def _flag_table() -> tuple[tuple[float, str], ...]:
    """Every flag in the order :func:`_flag_counts` counts them."""
    return tuple(
        (score, sentence) for _, score, sentence in _compiled_flags()
    ) + _COUNTED_FLAGS


def _flag_counts(mol) -> tuple[int, ...]:
    return tuple(
        len(mol.GetSubstructMatches(pattern, uniquify=True))
        for pattern, _, _ in _compiled_flags()
    ) + (_crowded_ring_count(mol),)


def _domain_warnings(short: tuple[int, ...], long: tuple[int, ...]) -> list[tuple[float, str]]:
    """Flags the *interior* unit raises, as a trimer-minus-dimer difference.

    Counted as a difference for the same reason the groups are: the methyl cap
    turns the last linkage of every oligomer into something the repeat unit
    does not contain, and a warning raised by a cap would be a warning about a
    chain end the polymer does not have.
    """
    # ``strict`` because a silently truncated zip is how a flag added to one of
    # the two tables and not the other would stop being raised without anyone
    # noticing - the exact failure mode this module refuses elsewhere.
    return [
        (score, sentence)
        for (score, sentence), before, after in zip(
            _flag_table(), short, long, strict=True
        )
        if after > before
    ]


@dataclass(frozen=True, slots=True)
class UnitCohesion:
    """The three HvK sums for one interior repeat unit, before dividing by V."""

    #: sum(F_di), in (J cm^3)^0.5 / mol.
    f_d: float
    #: sum(F_pi^2), in J cm^3 / mol^2.  Kept squared because the polar term is
    #: a quadrature sum, and averaging two repeat units means averaging their
    #: square roots, not their squares.
    f_p_squared: float
    #: sum(E_hi), in J/mol.
    e_h: float
    #: Molar mass of the interior repeat unit, g/mol.
    mass: float
    groups: Mapping[str, int]
    warnings: tuple[tuple[float, str], ...] = ()

    def triple(self, molar_volume_cm3: float) -> tuple[float, float, float]:
        """(delta_d, delta_p, delta_h) in MPa^0.5 at this molar volume."""
        return (
            self.f_d / molar_volume_cm3,
            math.sqrt(self.f_p_squared) / molar_volume_cm3,
            math.sqrt(self.e_h / molar_volume_cm3),
        )


@functools.lru_cache(maxsize=1024)
def repeat_unit_cohesion(unit_smiles: str) -> UnitCohesion:
    """HvK sums for one *interior* repeat unit of ``unit_smiles``.

    Taken as a trimer-minus-dimer difference for the same reason
    ``polymer.repeat_unit_groups`` does it: the chain ends are identical in
    both, so the methyl caps cancel exactly, and the unit that survives the
    subtraction was scored in a real chain environment rather than with two
    dangling valences no increment describes.
    """
    counted: dict[int, dict[str, int]] = {}
    flags: dict[int, tuple[int, ...]] = {}
    for length in (2, 3):
        mol = link_repeat_units(unit_smiles, length)
        # The missing-increment test runs first so that a fluorinated repeat
        # unit is refused for the reason that actually disqualifies it - there
        # is no F increment here - rather than for the geminal-halogen rule it
        # also happens to trip.
        counted[length] = _count_groups(mol)
        refusal = _structural_refusal(mol)
        if refusal is not None:
            raise RepeatUnitRefused(refusal)
        flags[length] = _flag_counts(mol)
    warnings = _domain_warnings(flags[2], flags[3])

    difference = {
        name: counted[3].get(name, 0) - counted[2].get(name, 0)
        for name in set(counted[2]) | set(counted[3])
    }
    if any(count < 0 for count in difference.values()):
        # The two chain lengths decomposed inconsistently, so the difference is
        # not a repeat unit and must not be passed off as one.
        raise RepeatUnitRefused(
            "the trimer and the dimer decomposed inconsistently, so their difference "
            "is not one repeat unit"
        )
    difference = {name: count for name, count in difference.items() if count}

    f_d = f_p_squared = e_h = 0.0
    for name, count in difference.items():
        increment = _INCREMENTS[name]
        f_d += count * increment.f_d
        f_p_squared += count * increment.f_p**2
        e_h += count * increment.e_h
    return UnitCohesion(
        f_d=f_d,
        f_p_squared=f_p_squared,
        e_h=e_h,
        mass=repeat_unit_mass(unit_smiles),
        groups=difference,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------
# Chain-level assembly
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChainCohesion:
    """Every backbone repeat unit of a polymer, with its mole fraction."""

    units: tuple[tuple[UnitCohesion, float], ...]

    @property
    def mass(self) -> float:
        """Mole-fraction-weighted repeat-unit mass, g/mol."""
        return sum(unit.mass * fraction for unit, fraction in self.units)

    @property
    def warnings(self) -> tuple[tuple[float, str], ...]:
        seen: dict[str, float] = {}
        for unit, _ in self.units:
            for score, sentence in unit.warnings:
                seen[sentence] = min(seen.get(sentence, 1.0), score)
        return tuple((score, sentence) for sentence, score in seen.items())

    def triple(self, density_g_cm3: float) -> tuple[float, float, float]:
        """Volume-fraction-averaged (delta_d, delta_p, delta_h) in MPa^0.5.

        A random copolymer's cohesion parameter is the volume-fraction average
        of its comonomers', which is the rule :mod:`formulate.experts.mixture`
        already uses.  Each unit's triple is formed first and averaged after,
        never the other way round: mole-averaging group counts before taking
        ``sqrt(sum F_p^2)`` adds the comonomers' polar terms in quadrature and
        inflates delta_p for any pair that differs in polarity.

        It is still wrong for a *block* copolymer, which has two phases and
        two sets of parameters rather than one average of them; the prediction
        notes say so rather than the value hiding it.
        """
        volumes = [unit.mass * fraction / density_g_cm3 for unit, fraction in self.units]
        total = sum(volumes)
        out = [0.0, 0.0, 0.0]
        for (unit, _), volume in zip(self.units, volumes):
            triple = unit.triple(unit.mass / density_g_cm3)
            for i in range(3):
                out[i] += (volume / total) * triple[i]
        return (out[0], out[1], out[2])

    def molar_volume_cm3(self, density_g_cm3: float) -> float:
        return self.mass / density_g_cm3


def chain_cohesion(spec: PolymerSpec) -> ChainCohesion:
    """Score every backbone repeat unit of ``spec``, or raise."""
    backbone = [m for m in spec.monomers if m.role is not MonomerRole.END_GROUP]
    if not backbone:  # pragma: no cover - PolymerSpec already forbids this
        raise RepeatUnitRefused("the polymer carries no backbone repeat unit")
    units = []
    for monomer in backbone:
        try:
            units.append((repeat_unit_cohesion(monomer.smiles), monomer.mole_fraction))
        except RepeatUnitRefused:
            raise
        except ValueError as exc:
            # link_repeat_units refuses a SMILES it cannot read or one that
            # does not carry exactly two attachment points.
            raise RepeatUnitRefused(str(exc)) from exc
    return ChainCohesion(tuple(units))


#: How each predicted property is formed from a chain and a density, keyed by
#: property name: (callable(chain, rho) -> value, unit as produced).
#:
#: Everything is produced in the unit the published table is in - MPa^0.5 - and
#: converted by the shared unit registry.  Hand-converting a square root of a
#: pressure is exactly the mistake ``dissolution.py`` had to document after a
#: factor of a thousand made every real solvent for polystyrene score as a
#: non-solvent.
#:
#: ``molar_volume_liquid`` is deliberately NOT here even though ``M/rho`` is
#: computed for every prediction and recorded in its provenance.  That volume
#: is an amorphous repeat unit's at 25 degC; the registry defines
#: ``molar_volume_liquid`` as a *saturated liquid's*, and it separated
#: ``amorphous_density`` from ``liquid_density`` for exactly this reason.
#: Offering the two under one name would put a glassy repeat unit and a
#: solvent in the same ranked column, distinguished only by a note no ranker
#: reads.  The name this wants is ``molar_volume_repeat_unit``, which the
#: registry does not have; until it does, the number travels as provenance.
_FORMULAE = {
    "hansen_dispersion": (lambda c, rho: c.triple(rho)[0], "MPa^0.5"),
    "hansen_polar": (lambda c, rho: c.triple(rho)[1], "MPa^0.5"),
    "hansen_hydrogen_bonding": (lambda c, rho: c.triple(rho)[2], "MPa^0.5"),
    "hildebrand_solubility_parameter": (
        lambda c, rho: math.sqrt(sum(v * v for v in c.triple(rho))),
        "MPa^0.5",
    ),
}


# --------------------------------------------------------------------------
# The expert
# --------------------------------------------------------------------------


class PolymerHansenExpert(Expert):
    """Hansen and Hildebrand parameters of a polymer repeat unit.

    Depends on ``amorphous_density`` rather than modelling a repeat-unit
    volume a second time.  That is not only tidiness: delta_d and delta_p
    scale as ``1/V``, so a guessed density is a guessed solubility parameter.
    A density that never arrives is a refusal here, not a default.  A density
    that arrives *out of domain* - the packing factor was fitted over
    H/C/N/O/F/Cl only, so a siloxane gets a number with a warning rather than
    no number - is carried into this prediction's own domain, because a
    solubility parameter divided by a volume nobody trusts is not trustworthy
    either.
    """

    id = "polymer_hansen"
    version = "1"
    method = (
        "Hoftyzer-Van Krevelen group contributions over the repeat unit "
        "(van Krevelen & Te Nijenhuis, Properties of Polymers, 4th ed., Table 7.10), "
        "with the repeat-unit molar volume taken from an upstream amorphous density"
    )
    family = PropertyFamily.INTERFACIAL
    supported_classes = frozenset({MaterialClass.POLYMER})
    supported_properties = frozenset(_FORMULAE)
    dependencies = frozenset({"amorphous_density"})

    #: What a chemist would want said next to any of these five numbers.
    _STANDING_NOTE = (
        "crystallinity and tacticity are invisible to an additive repeat-unit sum, and "
        "a solubility parameter is necessary but not sufficient; polyethylene scores "
        "17.5 MPa^0.5 here and dissolves in nothing at room temperature"
    )

    #: The warning that belongs on a solubility parameter and on nothing else.
    #: Attaching it to the molar volume as well would make it wallpaper, and a
    #: caveat nobody reads is a caveat that is not there.
    _SPHERE_NOTE = (
        "a cohesion parameter of the amorphous phase, not a fitted Hansen solubility "
        "sphere centre: the two differ by about 2 to 4 MPa^0.5 on the five polymers "
        "checked here, and substituting this triple for a sphere centre moves every "
        "relative energy difference by roughly a fifth of a radius"
    )

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        from formulate import chem

        return "" if chem.rdkit_available() else "RDKit is required to read a repeat unit"

    def _software(self) -> SoftwareEnvironment:
        import rdkit

        return SoftwareEnvironment.capture(rdkit=rdkit.__version__)

    # -- domain ------------------------------------------------------------

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        basis = (
            "linear, uncrosslinked repeat units built from the transcribed "
            "Hoftyzer-Van Krevelen increments, checked against 35 bundled and 120 "
            "cross-source-agreed liquids (RMSE 0.67/1.21/1.52 MPa^0.5 on "
            "delta_d/delta_p/delta_h) and five published polymer spheres "
            "(RMS 2.77/4.39/3.13 measured through this expert)"
        )
        spec = candidate.polymer
        if spec is None:
            return ApplicabilityDomain.outside("candidate carries no polymer", basis=basis)

        architecture = _architecture_reason(spec)
        if architecture is not None:
            return ApplicabilityDomain.outside(architecture, basis=basis)

        try:
            chain = chain_cohesion(spec)
        except RepeatUnitRefused as exc:
            return ApplicabilityDomain.outside(str(exc), basis=basis)

        # A sum over *interior* repeat units is the high-polymer limit.  Below
        # the molar mass at which ``polymer.py`` stops trusting the same limit
        # for a glass transition, the end groups are a real share of the
        # cohesive energy and this sum cannot see them: two hydroxyls on a
        # 400 g/mol poly(ethylene oxide) diol raise delta_h from the 8.7 this
        # returns to about 13.3, which is larger than the bar quoted below.  The ends
        # are not modelled - ``PolymerSpec.end_groups`` is free text and the
        # table has no entry for most of what appears there - so the answer is
        # marked rather than corrected.
        mn = spec.number_average_molar_mass
        if mn is not None:
            mn_value = mn.to("g/mol").value
            if mn_value < _MIN_NUMBER_AVERAGE_MOLAR_MASS:
                return ApplicabilityDomain.outside(
                    f"Mn = {mn_value:.0f} g/mol: this is an additive sum over interior "
                    "repeat units and therefore the high-polymer limit, and at this "
                    "length the chain ends carry a share of the cohesive energy that "
                    f"the sum cannot see - below {_MIN_NUMBER_AVERAGE_MOLAR_MASS:.0f} "
                    "g/mol a hydroxyl-terminated oligomer can sit 5 MPa^0.5 above this "
                    "delta_h, more than the stated bar",
                    basis=basis,
                )

        warnings = [sentence for _, sentence in chain.warnings]
        score = min([1.0] + [value for value, _ in chain.warnings])

        # ``_architecture_reason`` refuses a stated network or a stated
        # crosslink density.  A monomer whose declared *role* is crosslinker,
        # with neither stated, is an under-specified network: the basis above
        # says uncrosslinked, so scoring it 1.0 would be this expert
        # contradicting its own domain statement.  ``polymer.analyse_chain``
        # puts the same sentence on the same case.
        if any(m.role is MonomerRole.CROSSLINKER for m in spec.monomers):
            warnings.append(
                "a crosslinker is present, and this is a sum over the repeat units of "
                "an uncrosslinked chain: a network holds cohesive energy in covalent "
                "junctions that no repeat-unit increment stands for, and the swelling "
                "a solubility parameter would predict for it is not dissolution"
            )
            score = min(score, 0.5)

        if len(chain.units) > 1:
            warnings.append(
                "more than one backbone repeat unit: averaged by volume fraction as a "
                "random copolymer, which is wrong for a block copolymer - a block "
                "copolymer has two phases and two sets of parameters"
            )
            score = min(score, 0.6)
        return ApplicabilityDomain(
            score=score, in_domain=score > 0.3, warnings=tuple(warnings), basis=basis
        )

    # -- prediction --------------------------------------------------------

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        spec = request.candidate.polymer
        if spec is None:
            return Prediction.unsupported(prop, self.id, "candidate carries no polymer")

        architecture = _architecture_reason(spec)
        if architecture is not None:
            return Prediction.unsupported(prop, self.id, architecture)

        temperature = request.conditions.temperature_k
        if temperature is None:
            return Prediction.unsupported(
                prop,
                self.id,
                f"{prop} here is a 25 degC group-table value divided by a molar volume "
                "that is itself temperature dependent, and no temperature was given",
            )
        low, high = TEMPERATURE_WINDOW_K
        if not low <= temperature <= high:
            return Prediction.unsupported(
                prop,
                self.id,
                f"the increments are 25 degC values and the upstream packing factor "
                f"carries no temperature dependence; {temperature:.0f} K is outside the "
                f"{low:.0f}-{high:.0f} K window over which both are defensible",
            )

        try:
            chain = chain_cohesion(spec)
        except RepeatUnitRefused as exc:
            return Prediction.unsupported(prop, self.id, str(exc))

        density = request.dependency("amorphous_density")
        if density is None or density.quantity is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "needs amorphous_density from an upstream expert to turn the repeat-unit "
                "mass into a molar volume, and none was available; no typical polymer "
                "density is substituted, because delta_d and delta_p scale as 1/V and a "
                "guessed density is a guessed solubility parameter",
            )
        # An out-of-domain volume makes an out-of-domain solubility parameter.
        # The upstream expert answers for a siloxane with a warning rather than
        # a refusal, and delta_d and delta_p are simply that density times a
        # group sum, so a status of OK here would launder the warning away.
        if density.status is PredictionStatus.OUT_OF_DOMAIN:
            domain = domain.merged_with(
                ApplicabilityDomain.outside(
                    "the upstream amorphous density is itself out of domain, and these "
                    "parameters are a group sum multiplied by it: "
                    + "; ".join(density.applicability.warnings),
                    basis=density.applicability.basis,
                )
            )

        rho = density.quantity.to("kg/m^3").value * _KG_M3_TO_G_CM3
        rho_std = density.uncertainty.converted(density.quantity.unit, "kg/m^3").std
        rho_std = None if rho_std is None else rho_std * _KG_M3_TO_G_CM3
        if rho <= 0.0:
            return Prediction.failed(
                prop, self.id, f"upstream amorphous density came back at {rho:.4g} g/cm^3"
            )

        formula, unit = _FORMULAE[prop]
        try:
            value, propagated = propagate(
                lambda v: formula(chain, v["density"]), {"density": (rho, rho_std)}
            )
        except (ValueError, ZeroDivisionError, OverflowError) as exc:  # pragma: no cover
            return Prediction.failed(prop, self.id, f"group sum failed: {exc}")

        std, basis = self._uncertainty(prop, propagated)
        volume = chain.molar_volume_cm3(rho)
        notes = [
            f"measured offset against the five published polymer spheres: "
            f"{POLYMER_BIAS_MPA_SQRT[prop]:+.2f} MPa^0.5, reported rather than "
            "corrected for",
            self._SPHERE_NOTE,
            self._STANDING_NOTE,
            f"divided by a repeat-unit molar volume of {volume:.1f} cm^3/mol, which is "
            "an amorphous repeat unit's and not a saturated liquid's, and is therefore "
            "recorded here rather than offered as molar_volume_liquid",
        ]
        for cohesion, fraction in chain.units:
            share = "" if len(chain.units) == 1 else f" (mole fraction {fraction:.3g})"
            notes.append(
                f"repeat-unit groups{share}: "
                + ", ".join(f"{n} x{c}" for n, c in sorted(cohesion.groups.items()))
            )
        if len(chain.units) > 1:
            notes.append(
                "volume-fraction average over "
                f"{len(chain.units)} backbone repeat units, as for a random copolymer; "
                "each unit's triple is formed before the average, never after"
            )

        return self._make(
            prop,
            value,
            unit,
            request,
            domain,
            std=std,
            kind=UncertaintyKind.COMBINED,
            basis=basis,
            notes=tuple(notes),
            repeat_unit_molar_volume_cm3=round(volume, 3),
            amorphous_density_g_cm3=round(rho, 4),
        )

    def _uncertainty(self, prop: str, propagated: float) -> tuple[float | None, str]:
        """The error bar, and the one sentence that defends it."""
        method = METHOD_STD_MPA_SQRT[prop]
        total = math.hypot(method, propagated)
        dominant = "the group table" if method > propagated else "the upstream density"
        return (
            total,
            f"what the method misses on liquids where the reference is the same quantity "
            f"(RMSE 0.67/1.21/1.52 MPa^0.5 on delta_d/delta_p/delta_h over 35 bundled "
            f"compounds), floored by what it misses on polymers where the reference is a "
            f"fitted sphere centre and therefore a different quantity - offset plus "
            f"residual scatter over five published spheres measured through this expert, "
            f"{method:.2f} MPa^0.5, which covers 13 of those 15 components at one sigma - "
            f"combined in quadrature with "
            f"{propagated:.2f} propagated from the upstream density's own bar; "
            f"{dominant} dominates",
        )
