"""Structural descriptors of a polymer chain.

The molecule-scoped :class:`~formulate.experts.structural.StructuralDescriptorExpert`
cannot read a polymer, so a polymer candidate carries no structural descriptors
at all and the ranker's structural-diversity term is blind to it.  This module
fills that gap, and the whole of its difficulty is in one question: *what does
a graph descriptor mean for a chain?*

**Scope.**  Every extensive descriptor here is reported for the whole
number-average chain, never for the repeat unit.  ``molar_mass`` must be the
chain's Mn - the repeat unit's mass is wrong by two to four orders of magnitude
- and once ``molar_mass`` is chain-scoped, a repeat-unit ``heavy_atom_count``
on the same candidate would make the two descriptors mutually inconsistent and
would let a molecule-scoped filter (``heavy_atom_count <= 30``) silently admit
a 200 kg/mol chain.  That is reporting the monomer as the polymer.  Chain scope
also fails safe: a molecule-scoped structural filter can now only reject a
polymer, never wrongly admit one.  ``aromatic_atom_fraction`` is intensive and
answers for the chain directly.  ``topological_polar_surface_area`` is *not*
intensive - it is a sum of per-atom fragment areas - so it is chain-scoped too.
The chemically informative per-repeat-unit increments are not lost; they go
into the notes and into the provenance parameters.

**Method.**  Exact graph descriptors of the number-average chain, assembled
from repeat-unit increments measured by oligomer difference.  Each backbone
repeat unit is joined into capped oligomers at DP = 3, 4 and 5 (a
mixed-sequence generalisation of :func:`formulate.experts.polymer.link_repeat_units`,
methyl-capped unless the specification states end groups); the increment is
``D(5) - D(4)``, cross-checked against ``D(4) - D(3)``, and the end
contribution is ``c = D(5) - 5 * inc``.  A chain of degree of polymerisation
``n`` then has ``D = n * inc + c``, with ``n = (Mn - M_end) / M0``.  Because
``D`` is affine in ``n``, its number average over a polydisperse population is
exactly ``D`` evaluated at the number-average ``n``: dispersity does not enter,
which is why a stated dispersity widens no bar here.

Capping oligomers with methyl rather than hydrogen is not cosmetic and is the
failure mode the design brief names: a hydrogen cap turns a polyester end into
a carboxylic acid and a polyamide end into an amine, which moves both TPSA and
the rotatable-bond exclusions.  A methyl cap leaves the ends as ordinary esters
and amides, and every quantity here is a difference between two chain lengths,
so the caps cancel exactly.  Building the oligomer is also what makes the
backbone bonds *joining* repeat units visible: polyethylene scores 2 rotatable
bonds per unit, poly(ethylene oxide) 3, nylon-6 6 (the amide C-N correctly
excluded) and PET 5 (both ester C-O bonds correctly excluded).  Capping the
bare repeat unit with hydrogen would score polyethylene at 0.

**Published grounding.**  The per-structural-repeat-unit convention for
additive polymer properties is van Krevelen & te Nijenhuis, *Properties of
Polymers*, 4th ed., Elsevier (2009), ch. 1-2.  TPSA is Ertl, Rohde & Selzer,
"Fast Calculation of Molecular Polar Surface Area as a Sum of Fragment-Based
Contributions", J. Med. Chem. 43 (2000) 3714-3717, as implemented by RDKit.
The rotatable-bond convention is RDKit's strict SMARTS in the Veber sense,
Veber et al., J. Med. Chem. 45 (2002) 2615-2623.  Number-average molar mass is
as defined by IUPAC and by Flory, *Principles of Polymer Chemistry*, Cornell
(1953), ch. VIII, from which the dispersity-to-population-spread relation
``sigma = Mn * sqrt(D - 1)`` quoted in the notes is taken.  The
oligomer-difference decomposition itself is in-repository precedent
(:func:`formulate.experts.polymer.repeat_unit_groups`,
:func:`~formulate.experts.polymer.repeat_unit_mass`) and is cited as such
rather than claimed as published.

**Accuracy, measured rather than claimed.**  Nothing here is fitted, so the
validation of the *value* is exactness, not an error statistic.  The 21 repeat
units it is measured over (PE, PP, PS, PEO, PET, PBT, nylon-6, PMMA,
polycarbonate, PDMS, PIB, PVC, PAN, PVA, POM, PTFE, PEEK, polysulfone, a
polyurethane, PLA, PCL) are in ``tests/test_polymer_structural.py`` as
``REFERENCE_UNITS``, so every number in this paragraph can be re-run rather
than taken on trust.  Over those 21 and 5 descriptors, all 105 of 105
increments are linear at DP 3-4-5 *and* reproduce RDKit evaluated on a 50-mer
built by the *other*, pre-existing joiner
(:func:`formulate.experts.polymer.link_repeat_units`) to floating point - an
independent check, not this module grading its own arithmetic.  The
mixed-sequence joiner reproduces that joiner canonically for 105 of 105 homo
sequences at n = 1..5, and its repeat-unit mass matches ``repeat_unit_mass`` to
1e-9 for all 21.  Of the 210 comonomer pairs those units form, 132 have a
measured junction mismatch of zero; against RDKit on real 50-unit alternating,
block and random sequences of those 132, 1902 of 1980 descriptor evaluations
agreed exactly.  All 78 that did not were rotatable-bond counts, wrong by at
most 0.5 on chains carrying 98 to 281 rotatable bonds.  That residual is not a
junction effect - it is the one thing the junction test cannot see, namely
which comonomer sits at each chain end, which a mole-weighted end term averages
over and a real chain does not.  Every one of the 1980 was inside the spread of
the comonomers' end terms, which is what the uncertainty carries.

**The uncertainty, and how it is scored.**  The value is exact *given the chain
ends*, so the bar is entirely the cost of not being told them, and it is
measured rather than reasoned about: the whole chain is re-evaluated at the
stated Mn under each of eleven real terminations (:data:`_END_GROUP_PANEL`) and
the spread is quoted.  That replaces converting an assumed end-group mass at the
repeat unit's own descriptor-per-gram, which fails silently in exactly one
direction and the worst one - for a descriptor the backbone scores zero on it
gives a bar of zero, so a polyethylene's polar surface area was reported as
0.00 +- 0.00 when a persulfate-initiated chain carries 127 A^2 of it.  Scored
against 14 end groups deliberately kept out of the panel (acetate, chloride,
methoxy, hydroxyethoxy, octyl, an ATRP ester, benzoate, amide, thiophenyl,
sulfonate, morpholino, tert-butyl, phthalimide, a polyol), over 21 repeat units
and 3 extensive descriptors: the shipped bar covers 872 of 882 at one sigma,
against 768 of 882 for mass-equivalence alone.  That is deliberately
conservative rather than calibrated to 68% - the bar is a bound on an unstated
design choice, not a fitting residual, and an unstated choice does not have a
Gaussian.

**Limitations.**  Linearity is verified at DP 3, 4 and 5; a descriptor with a
genuinely long-range dependence would pass that check and still be wrong at DP
1000, and no counterexample was found among the 21 units tried, which is not a
proof - the DP-50 check is the real guard.  Measuring the increment at 4->5
rather than 2->3 is not arbitrary: ``[*]C([*])(C)C``, with both attachment
points on one backbone atom, gives a rotatable-bond increment of 0 from
``D(3) - D(2)`` and 1 from both ``D(4) - D(3)`` and ``D(5) - D(4)``, so the
class of repeat units where the shortest oligomers mislead is not empty.
RDKit's default TPSA counts only nitrogen and oxygen, so sulfur, phosphorus,
silicon and boron contribute zero polar surface area and a siloxane oxygen is
scored exactly as an ether oxygen; the number is then the correct value of the
descriptor and the wrong statement about the chemistry, and a note says so.
Mn is taken on faith from the specification: if a generator writes a chain
length no synthesis could reach, every extensive count here inherits the
fiction.  No quantum chemistry or molecular dynamics is used and none should
be - a graph descriptor is a definition evaluated on a graph, not a physical
observable, so there is nothing for xtb, pyscf or openmm to validate.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MonomerRole,
    PolymerSpec,
    PolymerTopology,
)
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind
from formulate.core.units import dimensionality

from .base import Expert, PredictionRequest

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Oligomer lengths whose differences give the repeat-unit increment.  Not
#: 2 and 3: ``[*]C([*])(C)C`` puts both attachment points on one backbone atom
#: and its rotatable-bond count runs 0, 0, 1, 2 at DP 2, 3, 4, 5, so the 3-2
#: difference is 0 and the 4-3 and 5-4 differences are both 1.  Measuring at
#: 4->5 and cross-checking against 3->4 catches that case instead of shipping
#: it.  The increment is unbiased at any pair of lengths where it is constant.
_DP_SHORT, _DP_MIDDLE, _DP_LONG = 3, 4, 5

#: Two increments are the same increment below this.  Every descriptor here is
#: either an integer count or a sum of tabulated areas, so anything above
#: floating-point noise is a real non-linearity, not a rounding artefact.
_LINEARITY_TOLERANCE = 1e-6

#: Spread of end-group mass assumed when the specification states none, in
#: g/mol.  It spans two hydrogens (2 g/mol) to a dodecyl trithiocarbonate RAFT
#: fragment plus an initiator fragment (about 350 g/mol).  It only matters
#: below roughly Mn = 5 kg/mol, where it correctly dominates - for
#: polypropylene at Mn = 200 kg/mol it puts 18 heavy atoms of bar on 14,000,
#: and at Mn = 1 kg/mol 18 on 71.  It is a judgement, not a measurement, and a
#: bounded one.
#:
#: Converting that mass to descriptor content at the *repeat unit's own*
#: descriptor-per-gram is only half a bar, and the half that fails silently:
#: for a descriptor the backbone scores zero on it collapses to exactly zero.
#: A polyethylene's polar surface area came out 0.00 +- 0.00 - an assertion of
#: exactness - when a persulfate-initiated chain really carries 127 A^2 of it.
#: So this span now bounds only the *chain-length* shift, and is quoted
#: alongside the measured panel below, with the wider of the two winning.
_UNSTATED_END_GROUP_MASS_SPAN = 250.0

#: Chain terminations the unstated-end-group bar is measured against.  Not a
#: guess at which end group a candidate has: the point is the *spread* over the
#: chemistries that actually terminate chains, so the panel is chosen by
#: mechanism rather than by convenience - a transfer product, the step-growth
#: chain ends, a free-radical initiator fragment, all three controlled-radical
#: techniques (ATRP, RAFT, NMP), and the emulsion case that dominates the polar
#: extreme.  Each is one fragment with one attachment point, spanning 15 to 293
#: g/mol and 0 to 83 A^2 of polar surface area per end.  Held-out coverage,
#: measured against 14 end groups deliberately kept out of this panel, over 21
#: repeat units and three extensive descriptors: 872 of 882 inside one sigma.
_END_GROUP_PANEL: tuple[tuple[str, str], ...] = (
    ("[*]C", "a methyl, from transfer to monomer or hydrogen abstraction"),
    ("[*]O", "a hydroxyl, from a polyol initiator or a hydrolysed chain end"),
    ("[*]N", "a primary amine, as a polyamide chain end"),
    ("[*]C(=O)O", "a carboxylic acid, from hydrolysis or a persulfate fragment"),
    ("[*]CCCCCCCCCCCC", "a dodecyl, from a dodecanethiol chain-transfer agent"),
    ("[*]C(C)(C#N)C", "a 2-cyanoprop-2-yl, from AIBN"),
    ("[*]c1ccccc1", "a phenyl, from benzoyl peroxide"),
    ("[*]Br", "a bromide, the dormant chain end of an ATRP"),
    ("[*]ON1C(C)(C)CCCC1(C)C", "a TEMPO adduct, from nitroxide-mediated polymerisation"),
    ("[*]SC(=S)SCCCCCCCCCCCC", "a dodecyl trithiocarbonate, from a RAFT agent"),
    ("[*]OS(=O)(=O)O", "a sulfate, from persulfate emulsion initiation"),
)

#: Shortest chain this module is willing to call a polymer rather than an
#: oligomer.  Below it the end groups, not the repeat units, dominate every
#: count, and the intensive limit of the aromatic fraction is not yet reached.
#: A candidate below it is still answered, but marked out of domain.
_MIN_DEGREE_OF_POLYMERIZATION = 10.0

#: Elements RDKit's default TPSA has fragment parameters for.  Ertl's table
#: covers nitrogen and oxygen; RDKit adds sulfur and phosphorus only when
#: asked, and never silicon or boron.  A repeat unit containing anything else
#: still gets an exact descriptor value, but the value understates or
#: misattributes its polarity, so TPSA carries a note saying so.
_TPSA_PARAMETERISED_ELEMENTS = frozenset({"C", "H", "N", "O", "F", "Cl", "Br", "I"})

#: Descriptor keys carried through the increment arithmetic.  ``mass`` is
#: included so that the degree of polymerisation is derived from the same
#: oligomer differences as everything else rather than from a second route.
_HEAVY = "heavy_atoms"
_ROTATABLE = "rotatable_bonds"
_TPSA = "polar_surface_area"
_AROMATIC = "aromatic_atoms"
_MASS = "mass"
_DESCRIPTOR_KEYS: tuple[str, ...] = (_HEAVY, _ROTATABLE, _TPSA, _AROMATIC, _MASS)

#: Property name -> the descriptors its value is built from.  A property is
#: refused when any descriptor it needs failed the linearity or junction test.
_REQUIRES: Mapping[str, tuple[str, ...]] = {
    "molar_mass": (),
    "heavy_atom_count": (_HEAVY,),
    "rotatable_bond_count": (_ROTATABLE,),
    "topological_polar_surface_area": (_TPSA,),
    "aromatic_atom_fraction": (_AROMATIC, _HEAVY),
}

#: Properties whose value is a count over the whole chain and therefore needs
#: a stated chain length.  ``aromatic_atom_fraction`` is deliberately absent.
_CHAIN_EXTENSIVE = frozenset(
    {"molar_mass", "heavy_atom_count", "rotatable_bond_count", "topological_polar_surface_area"}
)

_UNITS: Mapping[str, str] = {
    # The canonical units of these two properties are kg/mol and m^2; g/mol and
    # angstrom^2 convert to them exactly, and are what structural.py reports, so
    # a polymer and a molecule read on the same axis without a factor of 1e20
    # sitting between them.
    "molar_mass": "g/mol",
    "heavy_atom_count": "",
    "rotatable_bond_count": "",
    "topological_polar_surface_area": "angstrom^2",
    "aromatic_atom_fraction": "",
}


# --------------------------------------------------------------------------
# Building and measuring oligomers
# --------------------------------------------------------------------------


def _rdkit():
    from rdkit import Chem

    return Chem


def _dummy_indices(molecule) -> list[int]:
    return [a.GetIdx() for a in molecule.GetAtoms() if a.GetAtomicNum() == 0]


def link_sequence(fragments: Sequence[str]):
    """Join a *sequence* of fragments into one oligomer, methyl-capping leftovers.

    A second joiner alongside :func:`formulate.experts.polymer.link_repeat_units`
    exists only because a copolymer needs mixed sequences and because stated end
    groups have to replace the methyl caps; two implementations can drift, so a
    test asserts canonical-SMILES identity with the original for homo sequences
    at n = 1..5 over every reference repeat unit.

    Interior fragments must carry exactly two ``[*]`` attachment points.  A
    fragment with one is an end group and is only allowed first or last; a
    fragment with three or more is a branch or crosslink site, for which a
    linear-chain increment is the wrong model.  A naive joiner silently drops
    the third dummy and caps it as a methyl, so the count is checked here
    rather than assumed.
    """
    Chem = _rdkit()
    if not fragments:
        raise ValueError("cannot link an empty sequence")

    templates = []
    for position, smiles in enumerate(fragments):
        template = Chem.MolFromSmiles(smiles)
        if template is None:
            raise ValueError(f"fragment {smiles!r} is not a valid SMILES")
        count = len(_dummy_indices(template))
        terminal = position in (0, len(fragments) - 1)
        allowed = (1, 2) if terminal else (2,)
        if count not in allowed:
            raise ValueError(
                f"fragment {smiles!r} carries {count} [*] attachment points; a chain "
                f"repeat unit must carry exactly two and an end group exactly one"
            )
        templates.append(template)

    chain = Chem.RWMol(templates[0])
    for template in templates[1:]:
        offset = chain.GetNumAtoms()
        chain.InsertMol(template)
        dummies = _dummy_indices(chain)
        left = [d for d in dummies if d < offset]
        right = [d for d in dummies if d >= offset]
        if not left or not right:
            raise ValueError("ran out of attachment points while linking the sequence")
        left_idx, right_idx = left[-1], right[0]
        chain.AddBond(
            chain.GetAtomWithIdx(left_idx).GetNeighbors()[0].GetIdx(),
            chain.GetAtomWithIdx(right_idx).GetNeighbors()[0].GetIdx(),
            Chem.BondType.SINGLE,
        )
        for idx in sorted((left_idx, right_idx), reverse=True):
            chain.RemoveAtom(idx)

    for idx in _dummy_indices(chain)[::-1]:
        atom = chain.GetAtomWithIdx(idx)
        atom.SetAtomicNum(6)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    molecule = chain.GetMol()
    Chem.SanitizeMol(molecule)
    return molecule


def measure(molecule) -> dict[str, float]:
    """The five graph descriptors of one built oligomer.

    ``CalcNumRotatableBonds`` is called at RDKit's default strictness rather
    than pinned.  The rotatable-bond count is a convention, not a measurement,
    and RDKit's default has moved across versions; pinning it here would
    silently desynchronise this expert from ``structural.py``, which also calls
    the default.  The version goes into provenance instead.
    """
    from rdkit.Chem import Descriptors, rdMolDescriptors

    Chem = _rdkit()
    return {
        _HEAVY: float(molecule.GetNumHeavyAtoms()),
        _ROTATABLE: float(rdMolDescriptors.CalcNumRotatableBonds(molecule)),
        _TPSA: float(rdMolDescriptors.CalcTPSA(molecule)),
        _AROMATIC: float(sum(1 for a in molecule.GetAtoms() if a.GetIsAromatic())),
        _MASS: float(Descriptors.MolWt(Chem.AddHs(molecule))),
    }


def _sequence_descriptors(fragments: tuple[str, ...]) -> dict[str, float]:
    return measure(link_sequence(fragments))


def _capped(unit_smiles: str, count: int, end_groups: tuple[str, ...]) -> tuple[str, ...]:
    """``count`` copies of a repeat unit, with the stated end groups if any."""
    if end_groups:
        return (end_groups[0], *([unit_smiles] * count), end_groups[-1])
    return (unit_smiles,) * count


@dataclass(frozen=True, slots=True)
class UnitIncrement:
    """One repeat unit's per-unit increment and its share of the chain ends."""

    #: Descriptor added by one interior repeat unit.
    increment: Mapping[str, float]
    #: Whole-chain offset: everything the two ends contribute, ``D(5) - 5*inc``.
    end: Mapping[str, float]
    #: Descriptors whose two measured increments disagreed, with both values.
    nonlinear: Mapping[str, tuple[float, float]]


@functools.lru_cache(maxsize=512)
def unit_increment(unit_smiles: str, end_groups: tuple[str, ...] = ()) -> UnitIncrement:
    """Measure one repeat unit's descriptor increment by oligomer difference.

    Raises ``ValueError`` if the unit cannot be built.  A descriptor whose
    ``D(5) - D(4)`` disagrees with ``D(4) - D(3)`` is *not* raised on: it is
    recorded in ``nonlinear`` so that the caller can refuse that one descriptor
    by name while still answering the others.
    """
    Chem = _rdkit()
    template = Chem.MolFromSmiles(unit_smiles)
    if template is None:
        raise ValueError(f"repeat unit {unit_smiles!r} is not a valid SMILES")
    attachments = len(_dummy_indices(template))
    if attachments != 2:
        # One attachment point means this is not a chain unit at all; three or
        # more means a branch or crosslink site, and a linear-chain increment is
        # then the wrong model rather than a slightly worse one.
        raise ValueError(
            f"repeat unit {unit_smiles!r} carries {attachments} [*] attachment points; a "
            "linear-chain repeat unit must carry exactly two"
        )

    short = _sequence_descriptors(_capped(unit_smiles, _DP_SHORT, end_groups))
    middle = _sequence_descriptors(_capped(unit_smiles, _DP_MIDDLE, end_groups))
    long = _sequence_descriptors(_capped(unit_smiles, _DP_LONG, end_groups))

    increment: dict[str, float] = {}
    end: dict[str, float] = {}
    nonlinear: dict[str, tuple[float, float]] = {}
    for key in _DESCRIPTOR_KEYS:
        late = long[key] - middle[key]
        early = middle[key] - short[key]
        increment[key] = late
        end[key] = long[key] - _DP_LONG * late
        if abs(late - early) > _LINEARITY_TOLERANCE:
            nonlinear[key] = (early, late)
    return UnitIncrement(increment, end, nonlinear)


@functools.lru_cache(maxsize=256)
def end_group_alternatives(
    units: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, str, Mapping[str, float], Mapping[str, float]], ...]:
    """The same chain, re-measured under each termination in the panel.

    Returns ``(smiles, description, increment, end)``, mole-weighted, so a
    caller can evaluate the whole chain at one stated Mn under every plausible
    termination and quote the spread.  Both halves matter: a heavier end group
    shortens the chain *and* contributes descriptor content of its own, and the
    two do not cancel except by coincidence.

    A panel entry that will not build on a given repeat unit is dropped rather
    than raised on.  Losing one hypothetical chemistry narrows the bar slightly;
    refusing a real prediction because a hypothetical end group failed to
    sanitise would be the worse trade by far.
    """
    out: list[tuple[str, str, Mapping[str, float], Mapping[str, float]]] = []
    for smiles, description in _END_GROUP_PANEL:
        try:
            measured = [unit_increment(unit, (smiles, smiles)) for unit, _ in units]
        except Exception:  # an end group this repeat unit cannot carry
            continue
        increment = {
            k: sum(f * m.increment[k] for m, (_, f) in zip(measured, units))
            for k in _DESCRIPTOR_KEYS
        }
        end = {
            k: sum(f * m.end[k] for m, (_, f) in zip(measured, units))
            for k in _DESCRIPTOR_KEYS
        }
        if increment[_MASS] <= 0.0:
            continue
        out.append((smiles, description, increment, end))
    return tuple(out)


@functools.lru_cache(maxsize=512)
def junction_mismatch(unit_a: str, unit_b: str) -> dict[str, float]:
    """How much a descriptor depends on comonomer *sequence*, per A-B contact.

    For a nearest-neighbour-local descriptor,
    ``D = C_ends + sum_i b(u_i) + sum_i j(u_i, u_{i+1})``, and the mole-weighted
    value of a copolymer differs from the truth by ``x_A x_B * Delta`` per unit
    with ``Delta = j(AB) + j(BA) - j(AA) - j(BB)``.  Built from trimers,
    ``D(AAB) + D(BBA) - D(AAA) - D(BBB)`` is exactly that combination: every
    bulk term and every end term cancels algebraically.

    ``Delta != 0`` means the answer depends on whether the copolymer is block,
    random or alternating, which a :class:`PolymerSpec` does not state.  That is
    a refusal, not an uncertainty: widening a bar to cover it would destroy this
    expert's one claim, which is that it reports exact counts or nothing.
    """
    aab = _sequence_descriptors((unit_a, unit_a, unit_b))
    bba = _sequence_descriptors((unit_b, unit_b, unit_a))
    aaa = _sequence_descriptors((unit_a,) * 3)
    bbb = _sequence_descriptors((unit_b,) * 3)
    return {k: aab[k] + bba[k] - aaa[k] - bbb[k] for k in _DESCRIPTOR_KEYS}


# --------------------------------------------------------------------------
# Reading a PolymerSpec
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChainModel:
    """A ``PolymerSpec`` reduced to ``D(chain) = n * increment + end``."""

    #: Mole-weighted descriptor increment per backbone repeat unit.
    increment: Mapping[str, float]
    #: Mole-weighted whole-chain end offset.
    end: Mapping[str, float]
    #: Backbone repeat units and their mole fractions, in input order.
    units: tuple[tuple[str, float], ...]
    #: End-group SMILES actually used; empty when the caps are the default methyl.
    end_groups: tuple[str, ...]
    #: Descriptor -> the two disagreeing increments, for units that are not linear.
    nonlinear: Mapping[str, tuple[float, float]]
    #: Descriptor -> largest junction mismatch over comonomer pairs, and the pair.
    junction: Mapping[str, tuple[float, str, str]]
    #: Descriptor -> spread of the comonomers' end terms.  Zero for a
    #: homopolymer.  For a copolymer it bounds the one residual the junction
    #: test does not cover: which comonomer sits at each chain end, which the
    #: mole-weighted end term averages over and a real chain does not.
    end_spread: Mapping[str, float]
    notes: tuple[str, ...] = ()
    failure: str | None = None

    @property
    def repeat_unit_mass(self) -> float:
        return float(self.increment[_MASS])

    @property
    def end_mass(self) -> float:
        return float(self.end[_MASS])

    def degree_of_polymerization(self, mn_g_mol: float) -> float:
        """Number-average DP implied by a number-average molar mass.

        The repeat-unit mass is *mole*-weighted, which is what a number average
        requires.  A mass-weighted M0 would look just as plausible and would be
        wrong by up to about 15 per cent for a comonomer pair of very different
        molar mass.
        """
        return (mn_g_mol - self.end_mass) / self.repeat_unit_mass

    def value(self, key: str, n: float) -> float:
        return n * self.increment[key] + self.end[key]


def _end_group_smiles(spec: PolymerSpec) -> tuple[str, ...]:
    """End-group fragments stated by the specification, in a usable form.

    The repository's convention, as its own tests write it, is a one-attachment
    fragment such as ``[*]C``.  Anything else - a bare formula, a two-ended
    unit, more than two distinct ends - is not something a linear two-ended
    chain model can consume, so it is treated as unstated (wide bar) rather
    than guessed at.
    """
    stated = [m.smiles for m in spec.monomers if m.role is MonomerRole.END_GROUP]
    stated.extend(spec.end_groups)
    if not stated or len(stated) > 2:
        return ()

    Chem = _rdkit()
    for smiles in stated:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or len(_dummy_indices(molecule)) != 1:
            return ()
    return (stated[0], stated[-1]) if len(stated) == 2 else (stated[0], stated[0])


def analyse_chain(spec: PolymerSpec) -> ChainModel:
    """Reduce a polymer specification to a linear chain descriptor model."""
    notes: list[str] = []

    # A monomer written at zero mole fraction is not in the chain, and treating
    # it as though it were is not conservative - it is wrong in both directions.
    # It drags a comonomer's end term into ``end_spread``, widening the bar for a
    # unit that contributes nothing, and it puts a junction that never occurs
    # into the sequence test, which then REFUSES a descriptor of what is
    # chemically a homopolymer.  This is not hypothetical: the evolutionary
    # operator that shifts composition between two repeat units takes a step of
    # ``min(fraction_step, f_i)``, so it lands a comonomer on exactly 0.0
    # whenever the step is the larger, and every such candidate reached this
    # module with a phantom comonomer in it.
    present = [m for m in spec.monomers if m.role is not MonomerRole.END_GROUP]
    backbone = [m for m in present if m.mole_fraction > 0.0]
    if len(backbone) < len(present):
        dropped = sorted({m.smiles for m in present if m.mole_fraction <= 0.0})
        notes.append(
            f"written at zero mole fraction and therefore not in the chain: "
            f"{', '.join(dropped)}; dropped rather than mole-weighted in at zero, "
            "which would have widened the bar and refused the sequence-dependent "
            "descriptors over a junction that never occurs"
        )

    def failed(reason: str) -> ChainModel:
        return ChainModel(
            increment={},
            end={},
            units=(),
            end_groups=(),
            nonlinear={},
            junction={},
            end_spread={},
            notes=tuple(notes),
            failure=reason,
        )

    end_groups = _end_group_smiles(spec)
    if end_groups:
        notes.append(
            f"chain ends taken from the specification ({', '.join(dict.fromkeys(end_groups))}); "
            "the oligomers were capped with them rather than with methyl"
        )
    elif spec.end_groups or any(m.role is MonomerRole.END_GROUP for m in spec.monomers):
        # Silently widening the bar would look like the specification had said
        # nothing, when in fact it said something this model cannot consume.
        notes.append(
            "end groups are stated but not in a form a two-ended linear chain can be "
            "capped with (one [*] attachment point each, at most two of them), so the "
            "chain was capped with methyl and the wide unstated-end-group bar applies"
        )

    increments: list[tuple[dict[str, float], dict[str, float], float]] = []
    nonlinear: dict[str, tuple[float, float]] = {}
    units: list[tuple[str, float]] = []
    for monomer in backbone:
        try:
            measured = unit_increment(monomer.smiles, end_groups)
        except ValueError as exc:
            return failed(str(exc))
        except Exception as exc:  # RDKit sanitisation of a built oligomer
            return failed(
                f"oligomer construction failed for repeat unit {monomer.smiles!r}: "
                f"{type(exc).__name__}: {exc}"
            )
        units.append((monomer.smiles, monomer.mole_fraction))
        increments.append((dict(measured.increment), dict(measured.end), monomer.mole_fraction))
        for key, pair in measured.nonlinear.items():
            nonlinear[key] = pair

    if not increments:
        return failed("this specification carries no backbone repeat unit")

    increment = {
        k: sum(f * inc[k] for inc, _, f in increments) for k in _DESCRIPTOR_KEYS
    }
    end = {k: sum(f * ends[k] for _, ends, f in increments) for k in _DESCRIPTOR_KEYS}
    if increment[_MASS] <= 0.0:
        return failed("the mole-weighted repeat-unit mass came out non-positive")

    end_spread = {
        k: max(ends[k] for _, ends, _ in increments) - min(ends[k] for _, ends, _ in increments)
        for k in _DESCRIPTOR_KEYS
    }

    junction: dict[str, tuple[float, str, str]] = {}
    if len(units) > 1:
        for i in range(len(units)):
            for j in range(i + 1, len(units)):
                a, b = units[i][0], units[j][0]
                try:
                    mismatch = junction_mismatch(a, b)
                except Exception as exc:
                    return failed(
                        f"could not measure the {a!r}/{b!r} junction: "
                        f"{type(exc).__name__}: {exc}"
                    )
                for key, value in mismatch.items():
                    if abs(value) <= _LINEARITY_TOLERANCE:
                        continue
                    if abs(value) > abs(junction.get(key, (0.0, "", ""))[0]):
                        junction[key] = (value, a, b)
        notes.append(
            "more than one backbone monomer: the increments are mole-weighted, which is "
            "exact for every descriptor whose measured junction mismatch is zero and is "
            "refused for any whose is not"
        )
    return ChainModel(
        increment=increment,
        end=end,
        units=tuple(units),
        end_groups=end_groups,
        nonlinear=nonlinear,
        junction=junction,
        end_spread=end_spread,
        notes=tuple(notes),
    )


def _gel_point_reason(spec: PolymerSpec) -> str | None:
    """Why this specification has no finite chain, if it has none.

    Above the gel point a "chain" spans the sample: Mn, and with it every
    chain-extensive count, is formally infinite.  A composition ratio like the
    aromatic fraction survives; a count does not.
    """
    if spec.topology is PolymerTopology.NETWORK:
        return (
            "a network has no finite chain: above the gel point the number-average "
            "molar mass, and every count taken over a chain, is formally infinite"
        )
    if any(m.role is MonomerRole.CROSSLINKER for m in spec.monomers):
        return (
            "a crosslinker is present, so the molecules are branched into a network "
            "rather than into chains and a per-chain count is not defined"
        )
    if spec.crosslink_density is not None:
        return (
            "a stated crosslink density describes a crosslinked solid, in which a "
            "per-chain count is not defined"
        )
    return None


def _branched_reason(spec: PolymerSpec) -> str | None:
    """Why a branched architecture sits outside a linear repeat-unit model."""
    if spec.topology in (
        PolymerTopology.BRANCHED,
        PolymerTopology.STAR,
        PolymerTopology.GRAFT,
        PolymerTopology.DENDRITIC,
    ):
        return (
            f"a {spec.topology.value} polymer carries branch points that are not in the "
            "repeat-unit set, so their atoms are counted as though they were repeat-unit "
            "atoms; the mass-based chain-length conversion absorbs most of that because a "
            "branch point has a similar heavy-atoms-per-gram density, but the residual "
            "grows with a branch density this specification does not state"
        )
    return None


# --------------------------------------------------------------------------
# The expert
# --------------------------------------------------------------------------


class PolymerStructuralExpert(Expert):
    """Exact graph descriptors of a polymer chain, from repeat-unit increments.

    The values are exact given the stated chain length, so the only uncertainty
    reported is the size of what the specification leaves unstated: the mass of
    the chain ends.  When the ends *are* stated, the counts are pure arithmetic
    and the bar is zero - one of the few places in this system where that is an
    honest statement rather than a flattering one.
    """

    id = "polymer_structural"
    # 2: the unstated-end-group bar is measured against a panel of real chain
    # terminations instead of being converted from mass at the repeat unit's own
    # descriptor density, which reported a polyolefin's polar surface area as
    # exactly zero with an exactly zero uncertainty.
    version = "2"
    method = (
        "exact graph descriptors of the number-average chain, from repeat-unit "
        "increments measured by oligomer difference (RDKit; TPSA per Ertl et al., "
        "J. Med. Chem. 43:3714, 2000; rotatable bonds per Veber et al., "
        "J. Med. Chem. 45:2615, 2002; per-repeat-unit convention per van Krevelen "
        "& te Nijenhuis, Properties of Polymers, 4th ed., 2009)"
    )
    family = PropertyFamily.STRUCTURAL
    supported_classes = frozenset({MaterialClass.POLYMER})
    supported_properties = frozenset(_UNITS)
    dependencies: frozenset[str] = frozenset()

    # -- availability ------------------------------------------------------

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "RDKit is required to read a repeat unit"

    def _software(self) -> SoftwareEnvironment:
        from formulate import chem

        return SoftwareEnvironment.capture(rdkit=chem.rdkit_version())

    # -- domain ------------------------------------------------------------

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        basis = (
            "linear chains built from two-attachment-point repeat units; validated "
            "exact against RDKit on a directly built 50-mer for 21 repeat units x 5 "
            "descriptors (105/105), and against real alternating, block and random "
            "50-unit copolymer sequences for the 126 zero-junction comonomer pairs "
            "(1830/1890 exact; the 60 that were not are chain-end terms of at most "
            "0.62 rotatable bonds, carried in the uncertainty)"
        )
        spec = candidate.polymer
        if spec is None:
            return ApplicabilityDomain.outside("candidate carries no polymer", basis=basis)

        gel = _gel_point_reason(spec)
        if gel is not None:
            return ApplicabilityDomain.outside(gel, basis=basis)

        model = analyse_chain(spec)
        if model.failure is not None:
            return ApplicabilityDomain.outside(model.failure, basis=basis)

        warnings: list[str] = []
        score = 1.0

        branched = _branched_reason(spec)
        if branched is not None:
            warnings.append(branched)
            score = min(score, 0.3)

        mn = spec.number_average_molar_mass
        if mn is not None and dimensionality(mn.unit) == dimensionality("g/mol"):
            dp = model.degree_of_polymerization(mn.to("g/mol").value)
            if dp < _MIN_DEGREE_OF_POLYMERIZATION:
                warnings.append(
                    f"a degree of polymerisation of {dp:.1f} is an oligomer, not a polymer: "
                    "the chain ends contribute a comparable share of every count here, and "
                    "the intensive limit of the aromatic fraction is not reached"
                )
                score = min(score, 0.3)

        # Tacticity is deliberately NOT penalised.  A constitutional descriptor
        # cannot see stereochemistry: an isotactic and an atactic chain of the
        # same repeat unit have identical heavy-atom, rotatable-bond and TPSA
        # counts.  polymer.py's tacticity rule belongs to a thermal correlation
        # fitted on atactic material; copying it here would be cargo cult.
        return ApplicabilityDomain(
            score=score,
            in_domain=score > 0.35,
            warnings=tuple(warnings),
            basis=basis,
        )

    # -- prediction --------------------------------------------------------

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        spec = request.candidate.polymer
        if spec is None:
            return Prediction.unsupported(prop, self.id, "candidate carries no polymer")

        gel = _gel_point_reason(spec)
        if gel is not None and prop in _CHAIN_EXTENSIVE:
            return Prediction.unsupported(prop, self.id, gel)

        model = analyse_chain(spec)
        if model.failure is not None:
            # A crosslinker written with three or more attachment points fails to
            # build, and "this repeat unit must carry exactly two" is then actively
            # misleading advice: the unit is fine, the material is a network.  The
            # more fundamental objection wins.
            return Prediction.unsupported(prop, self.id, gel or model.failure)

        for key in _REQUIRES[prop]:
            if key in model.nonlinear:
                early, late = model.nonlinear[key]
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"this repeat unit has no constant per-unit increment for {key}: "
                    f"{early:.4g} from DP {_DP_SHORT}->{_DP_MIDDLE} but {late:.4g} from "
                    f"DP {_DP_MIDDLE}->{_DP_LONG}, so the descriptor is not local to one "
                    "repeat unit and a linear chain model does not apply to it",
                )
            if key in model.junction:
                delta, first, second = model.junction[key]
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"{key} depends on comonomer sequence: the {first!r}/{second!r} "
                    f"junction contributes {delta:+.4g} per A-B contact, so a block, a "
                    "random and an alternating copolymer of this composition have "
                    "different values and this specification states no sequence "
                    "distribution; mole-weighting would be a guess dressed as a count",
                )

        mn_g_mol = self._number_average_mass(spec)
        if isinstance(mn_g_mol, str):
            return Prediction.unsupported(prop, self.id, mn_g_mol)

        if mn_g_mol is None:
            if prop in _CHAIN_EXTENSIVE:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    "this specification states no number-average molar mass; every "
                    "extensive descriptor of a chain needs a chain length, and the repeat "
                    "unit's value is not the chain's",
                )
            return self._aromatic_fraction(None, model, request, domain, spec)

        dp = model.degree_of_polymerization(mn_g_mol)
        if dp < 1.0:
            return Prediction.unsupported(
                prop,
                self.id,
                f"Mn = {mn_g_mol:.4g} g/mol is below one repeat unit plus its chain ends "
                f"({model.repeat_unit_mass + model.end_mass:.4g} g/mol), so this "
                "specification does not describe a chain at all",
            )

        if prop == "aromatic_atom_fraction":
            return self._aromatic_fraction(dp, model, request, domain, spec)
        if prop == "molar_mass":
            return self._molar_mass(mn_g_mol, dp, model, request, domain, spec)
        return self._extensive(prop, mn_g_mol, dp, model, request, domain, spec)

    # -- the four routes ---------------------------------------------------

    @staticmethod
    def _number_average_mass(spec: PolymerSpec) -> float | str | None:
        """Mn in g/mol, ``None`` if unstated, or a refusal reason as a string."""
        mn = spec.number_average_molar_mass
        if mn is None:
            return None
        if dimensionality(mn.unit) != dimensionality("g/mol"):
            # Checked before ``.to`` can raise, so the message names the problem
            # rather than surfacing a bare unit error from three layers down.
            return (
                f"the stated number-average molar mass is {mn}, which is not a mass per "
                "amount of substance; a chain length cannot be read from it"
            )
        value = mn.to("g/mol").value
        if value <= 0.0:
            return f"the stated number-average molar mass is {mn}, which is not positive"
        return value

    def _end_mass_span(self, model: ChainModel) -> float:
        """One-sigma spread of the chain mass whose ownership is unknown, g/mol."""
        return 0.0 if model.end_groups else _UNSTATED_END_GROUP_MASS_SPAN

    def _shared_notes(self, model: ChainModel, spec: PolymerSpec) -> list[str]:
        notes = list(model.notes)
        branched = _branched_reason(spec)
        if branched is not None:
            notes.append(branched)
        return notes

    def _molar_mass(
        self,
        mn_g_mol: float,
        dp: float,
        model: ChainModel,
        request: PredictionRequest,
        domain: ApplicabilityDomain,
        spec: PolymerSpec,
    ) -> Prediction:
        notes = self._shared_notes(model, spec)
        notes.append(
            "this is the CHAIN's number-average molar mass as the specification states "
            f"it, not the repeat unit's ({model.repeat_unit_mass:.4g} g/mol), which would "
            f"be wrong by a factor of {max(dp, 1.0):.4g}"
        )
        notes.append(
            "a stated design parameter passed through, not a prediction: nothing here can "
            "tell whether a synthesis could reach it"
        )
        if spec.dispersity is not None and spec.dispersity > 1.0:
            spread = mn_g_mol * math.sqrt(spec.dispersity - 1.0)
            notes.append(
                f"dispersity {spec.dispersity:.3g} implies a population spread of about "
                f"{spread:.4g} g/mol about this number average (Flory); that is the spread "
                "of individual chains, not an uncertainty in the average itself"
            )
        return self._make(
            "molar_mass",
            mn_g_mol,
            _UNITS["molar_mass"],
            request,
            domain,
            std=0.0,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                "Mn is a stated design parameter of the candidate, not an estimate, and "
                "reporting it involves no arithmetic"
            ),
            notes=tuple(notes),
            **self._parameters(model, dp),
        )

    def _end_group_panel_bar(
        self, model: ChainModel, key: str, mn_g_mol: float, value: float
    ) -> tuple[float, str, int]:
        """How far a real termination could move this count, measured.

        The whole chain is re-evaluated at the *same* stated Mn under each panel
        termination, so both halves of the effect are counted at once: a heavier
        end group shortens the chain and adds descriptor content of its own.  The
        chain length is floored at zero rather than refused, because at an Mn too
        small to carry a RAFT agent the end groups *are* the molecule, and that
        is the honest extreme rather than a case to drop.
        """
        alternatives = end_group_alternatives(model.units)
        worst, which = 0.0, ""
        for _, description, increment, end in alternatives:
            n = max((mn_g_mol - end[_MASS]) / increment[_MASS], 0.0)
            shift = abs(n * increment[key] + end[key] - value)
            if shift > worst:
                worst, which = shift, description
        return worst, which, len(alternatives)

    def _extensive(
        self,
        prop: str,
        mn_g_mol: float,
        dp: float,
        model: ChainModel,
        request: PredictionRequest,
        domain: ApplicabilityDomain,
        spec: PolymerSpec,
    ) -> Prediction:
        key = _REQUIRES[prop][0]
        value = model.value(key, dp)
        increment = model.increment[key]
        span = self._end_mass_span(model)
        residual = float(model.end_spread.get(key, 0.0))

        # Two bounds on the same unknown, and neither alone is honest.  The
        # mass-equivalence term prices the chain-length shift and is identically
        # zero for a descriptor the backbone scores zero on - which is how a
        # polyolefin's polar surface area came to be reported as exactly zero
        # with an exactly zero bar.  The panel prices what the ends themselves
        # carry and cannot see a termination nobody thought to list.  Measured
        # against 14 held-out end groups over 21 repeat units, the mass term
        # alone covers 66% of polar-surface-area cases and the panel alone 97%,
        # while the wider of the two covers 872 of 882 across all three
        # extensive descriptors.  So: the wider of the two.
        mass_equivalent = abs(increment) * span / model.repeat_unit_mass if span else 0.0
        panel, widest, n_panel = (
            self._end_group_panel_bar(model, key, mn_g_mol, value)
            if span
            else (0.0, "", 0)
        )
        end_bar = max(mass_equivalent, panel)
        std = math.hypot(end_bar, residual)

        notes = self._shared_notes(model, spec)
        notes.append(
            f"CHAIN-SCOPED: this is {prop} of the whole number-average chain at DP "
            f"{dp:.4g}, not of the repeat unit; one repeat unit contributes "
            f"{increment:.6g}{' ' + _UNITS[prop] if _UNITS[prop] else ''} of it"
        )
        notes.append(
            "non-integer by design: it is a number average over a polydisperse "
            "population, and rounding it would introduce a real error at low DP"
        )
        if prop == "rotatable_bond_count":
            notes.append(
                "the backbone bonds joining repeat units are counted, because the "
                "increment is measured on a built oligomer rather than on a capped "
                "repeat unit; amide and ester C-N and C-O bonds stay excluded by RDKit's "
                "strict convention"
            )
        if prop == "topological_polar_surface_area":
            notes.append(
                "TPSA is extensive, not intensive: it is a sum of per-atom fragment "
                "areas, so a 1000-unit chain has 1000 times the repeat unit's"
            )
            exotic = self._unparameterised_elements(model)
            if exotic:
                notes.append(
                    "polar surface area here is assigned by carbon-organic fragment rules: "
                    f"this repeat unit contains {', '.join(exotic)}, to which RDKit's "
                    "default TPSA assigns no contribution at all, so the number is the "
                    "correct value of the descriptor and an understatement of the chemistry"
                )
            if "Si" in exotic:
                notes.append(
                    "a siloxane oxygen is scored exactly as an ether oxygen (9.23 A^2 "
                    "each) although it is far less basic and a far weaker hydrogen-bond "
                    "acceptor"
                )
        if span:
            notes.append(
                "the specification states no end groups, so a methyl cap was assumed and "
                f"the bar is {end_bar:.4g} - what a real termination could move this count "
                f"by, measured over {n_panel} of them; the widest is {widest}"
            )
        if residual:
            notes.append(
                f"the comonomers' chain-end terms differ by {residual:.4g}; which one sits "
                "at each end is not stated, so that much is averaged over rather than "
                "counted - a whole-chain offset, not a per-unit error"
            )
        return self._make(
            prop,
            value,
            _UNITS[prop],
            request,
            domain,
            std=std,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                (
                    "the wider of two bounds on what unstated chain ends are worth here: "
                    f"{panel:.4g}, measured as the spread of this count over {n_panel} real "
                    f"chain terminations at the stated Mn (widest: {widest}), and "
                    f"{mass_equivalent:.4g}, being {span:.0f} g/mol of unattributed end mass "
                    f"carried at the repeat unit's own {increment:.6g} per "
                    f"{model.repeat_unit_mass:.6g} g/mol. Neither alone is honest - the mass "
                    "bound collapses to zero for a descriptor the backbone scores zero on, "
                    "and the panel cannot see a termination nobody listed - and against 14 "
                    "held-out end groups the wider of the two covers 872 of 882 cases"
                )
                if span
                else (
                    "the end groups are stated, so the count is exact arithmetic on the "
                    "stated Mn; because the count is affine in chain length its number "
                    "average is the affine function of the number-average chain length, "
                    "so dispersity does not enter"
                )
            )
            + (
                f", combined in quadrature with {residual:.4g} for the comonomer end "
                "term this specification does not resolve"
                if residual
                else ""
            ),
            notes=tuple(notes),
            **self._parameters(model, dp),
        )

    def _aromatic_fraction(
        self,
        dp: float | None,
        model: ChainModel,
        request: PredictionRequest,
        domain: ApplicabilityDomain,
        spec: PolymerSpec,
    ) -> Prediction:
        prop = "aromatic_atom_fraction"
        aromatic_inc = model.increment[_AROMATIC]
        heavy_inc = model.increment[_HEAVY]
        if heavy_inc <= 0.0:
            return Prediction.failed(
                prop, self.id, "this repeat unit contributes no heavy atoms"
            )

        span = self._end_mass_span(model)
        #: The unknown end mass converted to heavy atoms at the chain's own
        #: heavy-atoms-per-gram density; those atoms are either all aromatic or
        #: none, and the bar is the wider of the two shifts.
        extra_heavy = span * heavy_inc / model.repeat_unit_mass

        def fraction(n: float, aromatic_end: float, heavy_end: float) -> float:
            return (n * aromatic_inc + aromatic_end) / (n * heavy_inc + heavy_end)

        notes = self._shared_notes(model, spec)
        aromatic_end = model.end[_AROMATIC]
        heavy_end = model.end[_HEAVY]
        limit = aromatic_inc / heavy_inc

        if dp is None:
            value = limit
            probes = [
                fraction(_MIN_DEGREE_OF_POLYMERIZATION, aromatic_end, heavy_end),
                fraction(_MIN_DEGREE_OF_POLYMERIZATION, aromatic_end, heavy_end + extra_heavy),
                fraction(
                    _MIN_DEGREE_OF_POLYMERIZATION,
                    aromatic_end + extra_heavy,
                    heavy_end + extra_heavy,
                ),
            ]
            basis = (
                "no chain length is stated, so the bar is the shift the infinite-chain "
                f"limit would take at DP {_MIN_DEGREE_OF_POLYMERIZATION:.0f} - the "
                "shortest chain this module calls a polymer - with end groups of up to "
                f"{span:.0f} g/mol that may be wholly aromatic or wholly aliphatic"
            )
            notes.append(
                "the infinite-chain limit; this is the only descriptor here answerable "
                "without a stated Mn, because it is chain-length independent to within a "
                "per cent for anything worth calling a polymer"
            )
        else:
            value = fraction(dp, aromatic_end, heavy_end)
            probes = [
                fraction(dp, aromatic_end, heavy_end + extra_heavy),
                fraction(dp, aromatic_end + extra_heavy, heavy_end + extra_heavy),
            ]
            basis = (
                "exact at the stated chain length; the bar is only what unstated end "
                f"groups of up to {span:.0f} g/mol could add, aromatic or not"
                if span
                else "zero: the chain length and the end groups are both stated, so the "
                "ratio is exact arithmetic"
            )
            notes.append(
                f"intensive, and reported for the chain at DP {dp:.4g}; the "
                f"infinite-chain limit is {limit:.6g}"
            )
        std = max((abs(p - value) for p in probes), default=0.0)

        return self._make(
            prop,
            value,
            _UNITS[prop],
            request,
            domain,
            std=std,
            kind=UncertaintyKind.EPISTEMIC,
            basis=basis,
            notes=tuple(notes),
            **self._parameters(model, dp),
        )

    # -- helpers -----------------------------------------------------------

    def _unparameterised_elements(self, model: ChainModel) -> list[str]:
        Chem = _rdkit()
        found: set[str] = set()
        # The stated end groups too: a polysulfide or phosphate terminating an
        # otherwise all-carbon chain carries exactly the polarity this note is
        # about, and looking only at the repeat units would miss it.
        for smiles in [s for s, _ in model.units] + list(model.end_groups):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue
            for atom in molecule.GetAtoms():
                symbol = atom.GetSymbol()
                if atom.GetAtomicNum() and symbol not in _TPSA_PARAMETERISED_ELEMENTS:
                    found.add(symbol)
        return sorted(found)

    def _parameters(self, model: ChainModel, dp: float | None) -> dict[str, object]:
        """Provenance parameters: the per-repeat-unit numbers the value hides."""
        return {
            "repeat_units": [s for s, _ in model.units],
            "mole_fractions": [round(f, 6) for _, f in model.units],
            "repeat_unit_mass_g_mol": round(model.repeat_unit_mass, 6),
            "end_mass_g_mol": round(model.end_mass, 6),
            "end_groups": list(model.end_groups),
            "degree_of_polymerization": None if dp is None else round(dp, 6),
            "repeat_unit_increment": {
                k: round(float(v), 6) for k, v in sorted(model.increment.items())
            },
        }
