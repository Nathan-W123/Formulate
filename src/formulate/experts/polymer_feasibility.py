"""Can this polymer be *made*?  Monomer reconstruction plus polymerisation difficulty.

Nothing else in the engine asks whether a proposed repeat unit corresponds to a
chain anyone can build, so a search is otherwise free to return an
unsynthesisable backbone and rank it first.  The molecule-class ``feasibility``
expert cannot answer it: a repeat unit is a dummy-capped graph, not a molecule,
and its Ertl-Schuffenhauer score is not its monomer's.  Scoring the repeat unit
directly would be exactly the silent substitution this repository forbids.

The question is decomposed the way a polymer chemist decomposes it:

1. **Can you get the monomer?**  The likely monomer is *reconstructed* from the
   repeat unit by graph surgery, driven by substructure recognition of the named
   polymerisation chemistries - SMARTS where a pattern is the natural statement,
   explicit graph tests where the transform needs the atom indices it matched on
   - and then scored with Ertl, P. and Schuffenhauer,
   A., "Estimation of synthetic accessibility score of drug-like molecules based
   on molecular complexity and fragment contributions", J. Cheminform. 1:8
   (2009) - RDKit's contributed ``sascorer``, imported from the existing
   ``feasibility`` expert rather than re-implemented, so the two experts stay on
   one scale.
2. **Does that monomer polymerise?**  An ordinal difficulty term is added, taken
   from the standard limits on polymerisability in Odian, G., "Principles of
   Polymerization", 4th ed., Wiley (2004): ch. 2 for the Carothers conversion
   requirement in step growth, ch. 3 (§3-9) for the steric and
   ceiling-temperature limits on chain polymerisation, ch. 7 for ring size
   against ring-opening thermodynamics.

``score = clamp(1, 10, monomer_term + route_term)``, taking the **minimum** over
every route that matched, because a polymer only needs one route.  Taking the
minimum is deliberate: adding a recogniser can then only lower a score, never
raise one, so this expert cannot become more pessimistic as it learns more
chemistry.

Measured on this installation
-----------------------------
**Monomer identity**, which is the half that matters because a wrong monomer is
a confident wrong number, was checked by InChIKey against the known monomer for
30 named polymers (``tests/test_polymer_feasibility.py``): polyethylene to
ethylene, PVC to vinyl chloride, PTFE to tetrafluoroethylene, PEO to oxirane,
polyoxymethylene to formaldehyde, PLA to lactic acid *and* lactide,
polycaprolactone to caprolactone *and* 6-hydroxyhexanoic acid, PET to
terephthalic acid + ethylene glycol, nylon-6,6 to adipic acid +
hexamethylenediamine, PEEK to hydroquinone + 4,4'-difluorobenzophenone,
bisphenol-A polycarbonate to bisphenol A, poly(p-phenylene) to
1,4-dibromobenzene, PDMS to octamethylcyclotetrasiloxane.  **30 of 30 correct.**

**Route coverage** over the 57 reference polymers in
``data/reference_polymers.json``, every one of which was actually made: **57 of
57** get a route.

**Separation**, reported rather than tuned: 56 commodity reference polymers
against four hard cases (poly(p-phenylene) 6.42, head-to-head PVC 5.50,
polyacetylene 5.00, poly(alpha-methylstyrene) 3.56).  **Concordance 223 of 224
pairs.**  The one discordant pair is worth stating rather than hiding:
poly(2,6-dimethyl-1,4-phenylene oxide) comes out at 3.82, above
poly(alpha-methylstyrene).  It does so because of how the reference file *draws*
it - the methyls sit meta to the phenolic oxygen, so the reconstructed monomer is
3,5-dimethylphenol, whose free ortho positions make oxidative coupling give
branched carbon-carbon-linked product.  Drawn with the methyls ortho to the
oxygen, as the real polymer has them, the monomer comes back as
2,6-dimethylphenol and the score is 2.55, inside the commodity group.  The
penalty table was written from the polymerisation literature first and was not
fitted to this number.

**Typical error bars**, which are wide on purpose: polyethylene 2.00 +/- 1.22,
PET 2.50 +/- 1.22, poly(p-phenylene) 6.42 +/- 1.80, PDMS 5.08 +/- 2.12.

Accuracy and what it is not
---------------------------
Nothing here is fitted, so no held-out error can be quoted and none is claimed.
Only the *ordering* is claimed, and only the commodity-versus-laboratory
separation above is validated.  The absolute position of the number on the 1-10
scale is not established, so a specification that puts a hard threshold on
``synthetic_accessibility`` is cutting on a coordinate this expert does not
warrant.

The largest known defect is in the borrowed part.  The Ertl score is
*anti-correlated* with reality for the smallest and most important monomers.
Measured here: ethylene 4.09, propylene 3.01, vinyl chloride 3.71, formaldehyde
4.04, acrylonitrile 3.73, butadiene 3.64, tetrafluoroethylene 3.36, acetylene
4.78 - against styrene 1.55, terephthalic acid 1.30, benzene 1.00.  Raw Ertl
says polyethylene is harder to make than poly(ethylene terephthalate).  The
cause is visible in the scorer: its fragment term averages ChEMBL fragment
frequencies over very few fragments and nothing in ChEMBL is the size of
ethylene.  Measured on homologous series whose every member is a bulk commodity,
n-alkanes run 2.75 (C2) to 1.21 (C6) to 1.23 (C10) and 1-alkenes 4.09 (C2) to
2.08 (C6) to 1.83 (C10) - 2.9 units of variation with no change in synthetic
difficulty, plateauing near 1.2-2.0.  :data:`SMALL_MOLECULE_CEILING` corrects
that, as a ceiling only, and the correction is itself a judgement.

The same bias applies to rare elements, not just to size: octamethylcyclo-
tetrasiloxane, a bulk commodity, scores 4.08 and dichlorodimethylsilane 3.29, so
PDMS comes out near 5.1 when it is among the easiest polymers in the world to
make.  That is flagged out of domain and widened rather than corrected, because
correcting it would mean inventing a per-element offset.

Other limits, in the order they are likely to bite: a plausible-looking monomer
nobody has ever reported scores well, because Ertl is a prior over what has been
made before; head-to-tail regiochemistry is read off how the SMILES was written,
so a head-to-head charge may be punishing a generator's spelling; the backbone
is the shortest path between the attachment points, so a linkage lying off that
path is invisible; there is no reactivity-ratio model, so a copolymer of two
monomers that each homopolymerise happily is scored as though they copolymerise;
tacticity is charged at one flat rate, which cannot tell commercial isotactic
polypropylene from uncommercial isotactic PVC; and hazard, toxicity and
regulatory barriers - phosgene, isocyanates, fluorosurfactants - are not
accessibility, yet they decide real feasibility, and they leak into the route
terms unevenly rather than being modelled.

Several real chemistries are not recognised at all, and a polymer only reachable
through one of them is refused rather than scored.  Checked and confirmed
missing: ring-opening metathesis (ROMP) and acyclic diene metathesis (ADMET), so
polyoctenamer is refused; imide formation from a dianhydride, so the polyimides
are refused (a cut through the imide does not disconnect the ring, and the
anhydride is not reconstructed); the sulfide analogue of aromatic nucleophilic
substitution, so poly(phenylene sulfide) is refused; carbon-nitrogen bond
formation by amination, so a backbone joined through an aromatic nitrogen - an
N-arylene, and any polyimide RDKit perceives as aromatic - is refused; the
inorganic ring chemistries, so a phosphorus, boron, germanium or tin backbone is
refused (silicon alone has a recogniser); the redox chemistry of a heteroatom-
heteroatom bond, so a polydisulfide or a hydroxylamine-ether backbone is refused;
and any repeat unit whose two attachment points sit on the same atom, which is
how a polysilane is drawn.
A repeat unit written as a whole multiple of a smaller one - ``[*]CCCCCC[*]`` for
polyethylene, ``[*]CCOCCO[*]`` for poly(ethylene oxide) - is refused for the same
reason, because nothing here reduces a repeat unit to its smallest period.  A
false refusal costs a candidate its entire utility on this objective, since the
default missing-objective policy is to penalise; that is the correct direction to
fail in, but it is not free.

Three of those refusals - the aromatic nitrogen, the inorganic ring and the
heteroatom-heteroatom bond - were added by an adversarial review that caught the
ungated code returning a confident number on a monomer that does not exist:
``Brc1ccn(Br)c1``, a brominating agent, for an N-arylene; a four-membered
cyclodiphosphoxane at 2.5 *and in domain* for ``[*]P(C)(C)O[*]``, level with
poly(ethylene terephthalate); 1,2-dithietane at 2.5 for ``[*]CCSS[*]``.  Each is
now a refusal and each is pinned as a test.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MonomerRole,
    PolymerSpec,
    PolymerTopology,
    Tacticity,
)
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest
from .feasibility import _sascorer

#: Elements the Ertl fragment statistics actually cover.  Identical to the
#: molecule expert's list minus silicon and boron: those two are *in* that list
#: because a molecule containing them is still a molecule ChEMBL might describe,
#: but a siloxane or borazine *backbone* is a polymer chemistry the fragment
#: frequencies have never seen, and the measured consequence is a bulk commodity
#: (PDMS) scoring like a research compound.
_COMMON_ELEMENTS = frozenset({"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"})

#: Heavy-atom count at or below which the Ertl score is dominated by the
#: molecule being small rather than by it being hard.
#:
#: Measured on this installation over homologous series whose every member is a
#: bulk commodity: n-alkanes 2.75 (C2) -> 1.21 (C6) -> 1.23 (C10); 1-alkenes
#: 4.09 (C2) -> 2.08 (C6) -> 1.83 (C10).  Both plateau by about six heavy atoms
#: and stay near 1.2-2.0 thereafter.  Eight rather than six so that the common
#: one-ring monomers (vinyl chloride through methyl methacrylate) fall inside
#: the same correction as the gases.
SMALL_MOLECULE_HEAVY_ATOMS = 8

#: The value a small, otherwise unremarkable monomer's score is capped *at* -
#: never raised to.  The defence is one sentence: a trivially made molecule's
#: Ertl score falls to about two once the size term stops dominating, so a small
#: monomer scoring above that is being charged for being small, not for being
#: hard.  This is a judgement, not a calibration; it is charged 0.5 of extra
#: uncertainty whenever it is applied, because a correction is not a measurement.
SMALL_MOLECULE_CEILING = 2.0

#: Motifs that disqualify a small monomer from the ceiling above.  Each is a
#: reason a small molecule really *is* hard to make or does not survive being
#: made, so the ceiling must not rescue it: an alpha-lactone or cyclopropene
#: (strain plus a carbonyl or an alkene in a three-ring), a peroxide, an N-N
#: single bond, an azo or azide, a cumulated diene, a ketene.
_UNSTABLE_MOTIFS: tuple[tuple[str, str], ...] = (
    # ``[#6]`` rather than ``[CX3]``: RDKit perceives a three-membered ring
    # carrying two carbonyls (the oxiranedione an oxalate backbone closes to, and
    # its aza analogue) as AROMATIC, and an aromatic carbon does not match the
    # aliphatic ``C``.  Written the narrow way, the gate that keeps polyglycolide
    # on glycolide rather than on an alpha-lactone let those two rings straight
    # through.
    ("[#6;r3]=[OX1]", "a carbonyl inside a three-membered ring"),
    ("[#6;r3]=[#6]", "a double bond on a three-membered ring"),
    ("[OX2][OX2]", "a peroxide"),
    ("[NX3][NX3]", "an N-N single bond"),
    ("[NX2]=[NX2]", "an azo group"),
    ("[$(N=[N+]=[N-]),$([N-]=[N+]=N)]", "an azide"),
    ("[#6]=[#6]=[#6]", "a cumulated diene"),
    ("[#6]=[#6]=[OX1]", "a ketene"),
    ("[#8;r4][#6;r4][#8;r4]", "a four-membered acetal ring"),
)

#: Nominal spread of the Ertl score itself.  It carries no calibrated error;
#: the molecule-class ``feasibility`` expert records 1.0 as a nominal spread so
#: that ranking cannot treat it as exact, and the same number is kept here so
#: the two experts' error bars stay on one scale.
_ERTL_NOMINAL_STD = 1.0

#: Extra spread charged whenever :data:`SMALL_MOLECULE_CEILING` was applied.
_CEILING_STD = 0.5

#: Multiplier on the whole error bar for a candidate outside the domain.
_OUT_OF_DOMAIN_WIDENING = 1.5


class Mechanism(str, Enum):
    """Which kind of polymerisation a route is.

    Two repeat units can only share a chain if their mechanisms intersect: a
    vinyl and a lactone do not build one backbone, and averaging them would be
    the material-class substitution this repository forbids, one level down.
    """

    CHAIN_GROWTH = "chain growth"
    RING_OPENING = "ring-opening"
    STEP_GROWTH = "step growth"
    COUPLING = "coupling"


# --------------------------------------------------------------------------
# The polymerisation-difficulty table
# --------------------------------------------------------------------------
#
# Every number below is an *ordinal* judgement expressed in Ertl units.  The
# direction of each is from Odian; the size is not published by anyone, because
# no such scale exists.  Only the ordering they induce is claimed.  The spread
# beside each is the declared uncertainty of that judgement: 0.5 where thousands
# of commercial polymers stand behind the chemistry, 1.0 where it is real but
# much less common, 1.5 where the term is an extrapolation from a handful of
# cases.

#: Vinyl and diene chain growth: the general case, and the one the whole
#: commodity plastics industry rests on.  Nothing is charged for it.
_ROUTE_CHAIN_GROWTH = (0.0, 0.5)

#: Chain growth through a carbonyl (formaldehyde to polyoxymethylene).  Real and
#: commercial, but it needs low temperature, rigorous drying and end-capping to
#: stop the chain unzipping, which is why one polymer dominates the class.
_ROUTE_CARBONYL = (1.0, 1.0)

#: Chain growth of an alkyne.  Polyacetylene is a Nobel-prize material, not a
#: commodity: the product is insoluble and infusible and the polymerisation
#: needs a Ziegler-Natta or metathesis catalyst under inert conditions.
_ROUTE_ALKYNE = (3.0, 1.5)

#: 1,2-disubstituted alkene (a head-to-head backbone).  Odian §3-9b: such
#: monomers do not homopolymerise, for steric reasons plus a low enthalpy of
#: polymerisation; they only enter chains by copolymerisation.  Charged rather
#: than refused, because a copolymer or a non-radical route can reach the
#: structure.
_ROUTE_HEAD_TO_HEAD = (3.5, 1.5)

#: 1,1-disubstituted where one substituent is aryl and the other is not
#: hydrogen: the alpha-methylstyrene case.  The ceiling temperature is 61 C,
#: so high polymer simply does not form at ambient temperature.
_ROUTE_GEM_ARYL = (2.0, 1.5)

#: 1,1-disubstituted with two substituents both bulkier than a methyl or a
#: halogen (1,1-diphenylethylene and its relatives).  Propagation stops at the
#: dimer; the monomer is used as a chain-transfer probe, not a feedstock.
_ROUTE_GEM_BULKY = (3.0, 1.5)

#: Charged per fluorine on the two backbone termini, capped at four.
#:
#: This term exists because :data:`SMALL_MOLECULE_CEILING` erases a real
#: difference: it caps ethylene and tetrafluoroethylene at the same 2.0, when
#: ethylene comes off a cracker and TFE comes from chloroform via HCFC-22 by
#: pyrolysis and is then polymerised under pressure in specialised plant.  The
#: term restores, ordinally, what the ceiling removed.  Chlorine is not charged:
#: vinyl chloride is one step from ethylene and is made by the megatonne.
_ROUTE_PER_BACKBONE_FLUORINE = 0.4
_ROUTE_FLUORINE_CAP = 1.6

#: Ring-opening of a strained ring, three to seven members.  Epoxides, oxetanes,
#: tetrahydrofuran, the lactones and the lactams: mainstream, commercial, and
#: thermodynamically downhill because the ring strain pays for the entropy loss
#: (Odian ch. 7).
_ROUTE_ROP_STRAINED = (0.5, 0.5)

#: Ring-opening of an eight-membered or larger ring.  Macrolactone ROP is real
#: and is entropy-driven rather than strain-driven, which makes it far less
#: general and much more sensitive to catalyst and dilution.
_ROUTE_ROP_MACROCYCLE = (2.0, 1.5)

#: Step-growth polycondensation, AB or AA + BB.  Polyesters and polyamides:
#: mainstream.  What is charged instead is the Carothers requirement below.
_ROUTE_STEP_GROWTH = (0.5, 0.5)

#: Added to step growth when the linkage is a carbonate or a urethane.  Neither
#: monomer set is complete on its own: a carbonate needs phosgene or a
#: transesterification with diphenyl carbonate, and a urethane needs a
#: diisocyanate, itself made by phosgenating the amine.  The reconstruction
#: reports the diol and the diamine, which are real, and this term pays for the
#: C1 unit it deliberately does not invent a monomer for.
_ROUTE_PHOSGENE_DERIVED = (1.0, 1.0)

#: Aromatic nucleophilic substitution between a bisphenolate and an activated
#: dihaloarene - the PEEK and polysulfone route.  Established at scale but in
#: far fewer plants than polyester or polyamide chemistry, and it needs a dipolar
#: aprotic solvent at 300 C.
_ROUTE_SNAR = (1.0, 1.0)

#: Oxidative coupling of a phenol whose two ortho positions are blocked - the
#: poly(phenylene oxide) route.  Commercial, one polymer.
_ROUTE_OXIDATIVE_COUPLING = (1.0, 1.0)

#: The same coupling when the ortho positions are *free*.  Carbon-carbon coupling
#: then competes with carbon-oxygen coupling and the product is branched and
#: mixed rather than cleanly the polyether.  Charged, not refused: the structure
#: is reachable, just not cleanly.
_ROUTE_OXIDATIVE_COUPLING_OPEN = (2.0, 1.5)

#: Aryl-aryl coupling of a dihaloarene (Yamamoto, Suzuki).  Poly(p-phenylene) is
#: the standing example and it is intractable: the rigid rod precipitates from
#: solution after a handful of units, the polymer is infusible and insoluble, and
#: the coupling needs stoichiometric or catalytic nickel or palladium.  This is
#: the largest term in the table and it is an extrapolation from few cases.
_ROUTE_ARYL_ARYL = (5.0, 1.5)

#: Ring-opening of a cyclosiloxane to a polysiloxane.  Commercial and easy in
#: reality; the number is small because the chemistry is mainstream, and the
#: silicon *monomer* is where this expert is known to be wrong (see the module
#: docstring), which is handled by the domain flag rather than by this term.
_ROUTE_SILOXANE = (1.0, 1.0)

#: Added when the reconstructed monomer is an enol, which does not exist as an
#: isolable species: poly(vinyl alcohol) is made by polymerising vinyl acetate
#: and hydrolysing the polymer.  The acetate is scored instead and this pays for
#: the extra step.
_ROUTE_PROTECT_AND_HYDROLYSE = (1.0, 1.0)

#: Added when a stereoregular chain is asked for.  A stereospecific catalyst is
#: needed - metallocene or Ziegler-Natta for the polyolefins, anionic at low
#: temperature for the methacrylates.  One flat rate cannot tell commercial
#: isotactic polypropylene from uncommercial isotactic poly(vinyl chloride); that
#: limit is stated on every prediction that pays it.
_ROUTE_STEREOREGULAR = (1.0, 0.5)

#: Added when more than one backbone repeat unit is present.  It pays for the
#: composition control a copolymerisation needs; it does *not* know reactivity
#: ratios, so it cannot see that styrene and vinyl acetate refuse to copolymerise
#: at all.
_ROUTE_COPOLYMER = (0.5, 1.0)

#: Carothers (Odian ch. 2): in step growth the number-average degree of
#: polymerisation is 1/(1-p), so Xn = 200 needs 99.5% conversion and Xn = 500
#: needs 99.8%.  Those are achievable - PET is made at them - but only with
#: stoichiometry control, a high-vacuum finisher and a catalyst, so a target that
#: demands them is charged.  Chain growth pays nothing here: molar mass there is
#: set by the initiator-to-monomer ratio, not by conversion.
_CAROTHERS_STEPS: tuple[tuple[float, float, float], ...] = (
    (200.0, 0.5, 0.5),
    (500.0, 1.0, 1.0),
)

#: Ring sizes for which ring-opening is thermodynamically favourable in an
#: all-organic ring.  Three to five are strained; six is only favourable when the
#: ring carries a carbonyl (delta-valerolactone and lactide polymerise,
#: tetrahydropyran and cyclohexane do not - Odian ch. 7, the classic chair
#: argument); seven and up are strained again.
_UNSTRAINED_RING_SIZE = 6

#: The elements :data:`_UNSTRAINED_RING_SIZE` is a statement about.  Odian ch. 7
#: trades ring strain against the entropy of polymerisation for *organic* rings:
#: carbon with oxygen, nitrogen or sulfur in it.  Every other element keeps its
#: own size rules, and the measured consequence of ignoring that was a backbone
#: run through phosphorus, boron, germanium or tin being closed into a
#: four-membered inorganic ring that does not exist and then priced at the
#: cheapest route in the table - ``[*]P(C)(C)O[*]`` came back at 2.5, in domain,
#: level with poly(ethylene terephthalate).  Silicon is excluded here and handled
#: by :func:`_siloxane_route` instead, because the cyclosiloxanes that do
#: polymerise are the six- and eight-membered D3 and D4 rather than the
#: four-membered cyclodisiloxane this transform would otherwise close.
_ROP_RING_ELEMENTS = frozenset({"C", "N", "O", "S"})


# --------------------------------------------------------------------------
# Graph surgery on a repeat unit
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Route:
    """One way the repeat unit could have been made."""

    name: str
    mechanism: Mechanism
    #: Reconstructed monomer SMILES, in the order a chemist would name them.
    monomers: tuple[str, ...]
    penalty: float
    spread: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Veto:
    """A chemistry that was recognised and then found impossible."""

    reason: str


@dataclass(frozen=True, slots=True)
class UnitAnalysis:
    """Everything the recognisers found for one repeat unit."""

    smiles: str
    routes: tuple[Route, ...] = ()
    vetoes: tuple[Veto, ...] = ()
    failure: str | None = None
    notes: tuple[str, ...] = field(default=())

    @property
    def mechanisms(self) -> frozenset[Mechanism]:
        return frozenset(r.mechanism for r in self.routes)


def _rdkit() -> Any:
    from rdkit import Chem

    return Chem


def _parse(smiles: str) -> Any | None:
    Chem = _rdkit()
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:  # RDKit raises on a few malformed inputs rather than returning None
        return None


def _dummies(mol: Any) -> list[int]:
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]


def _smiles(mol: Any) -> str | None:
    Chem = _rdkit()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:  # pragma: no cover - sanitize almost always fails first
        return None


def _backbone(mol: Any) -> tuple[int, ...] | None:
    """The shortest path between the two attachment points.

    "Shortest" is a choice and it is recorded as a limitation: for a repeat unit
    whose chain runs through a ring by the longer route, the path taken here is
    not the chemical backbone and a linkage lying off it is invisible to every
    recogniser below.
    """
    Chem = _rdkit()
    dummies = _dummies(mol)
    if len(dummies) != 2:
        return None
    ends = []
    for idx in dummies:
        neighbours = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()]
        if len(neighbours) != 1:
            return None
        ends.append(neighbours[0])
    if ends[0] == ends[1]:
        return None
    path = Chem.GetShortestPath(mol, ends[0], ends[1])
    return tuple(path) if path else None


def _free_valence_atom(rw: Any, idx: int) -> None:
    """Let RDKit recompute hydrogens on an atom whose bonding just changed."""
    atom = rw.GetAtomWithIdx(idx)
    atom.SetNoImplicit(False)
    atom.SetNumExplicitHs(0)


def _strip_dummies(rw: Any) -> None:
    """Delete the attachment points and let their neighbours pick up hydrogens.

    The neighbours are marked before anything is removed, because deleting an
    atom renumbers every atom above it and an ``Atom`` handle taken beforehand
    does not survive the renumbering.
    """
    for idx in _dummies(rw):
        for neighbour in rw.GetAtomWithIdx(idx).GetNeighbors():
            neighbour.SetBoolProp("_formulate_freed", True)
    for idx in sorted(_dummies(rw), reverse=True):
        rw.RemoveAtom(idx)
    for atom in rw.GetAtoms():
        if atom.HasProp("_formulate_freed"):
            atom.SetNoImplicit(False)
            atom.SetNumExplicitHs(0)
            atom.ClearProp("_formulate_freed")


def _clear_stereo(rw: Any, atoms: Sequence[int]) -> None:
    """Drop stereo flags on bonds whose order this transform is about to change.

    A cis/trans flag on a bond that is about to become a different bond is not
    information, it is a leftover, and RDKit will refuse to sanitize a directional
    bond that no longer sits beside a double bond.
    """
    Chem = _rdkit()
    wanted = set(atoms)
    for bond in rw.GetBonds():
        if bond.GetBeginAtomIdx() in wanted or bond.GetEndAtomIdx() in wanted:
            bond.SetStereo(Chem.BondStereo.STEREONONE)
            bond.SetBondDir(Chem.BondDir.NONE)


def _has_exocyclic_double(mol: Any, idx: int, path: Sequence[int]) -> bool:
    """True when this backbone atom already carries a double bond off the chain."""
    Chem = _rdkit()
    on_path = set(path)
    atom = mol.GetAtomWithIdx(idx)
    for bond in atom.GetBonds():
        other = bond.GetOtherAtomIdx(idx)
        if other in on_path:
            continue
        if bond.GetBondType() is not Chem.BondType.SINGLE:
            return True
    return False


def _is_acyl(mol: Any, idx: int) -> bool:
    """True when this atom is a carbon doubly bonded to an oxygen.

    Tested for its own sake rather than through "carries any double bond",
    because an enol ether and an acrylate both carry one and neither is a
    linkage a polycondensation formed.
    """
    Chem = _rdkit()
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetAtomicNum() != 6:
        return False
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if bond.GetBondType() is Chem.BondType.DOUBLE and other.GetAtomicNum() == 8:
            return True
    return False


def _substituents(mol: Any, idx: int, path: Sequence[int]) -> list[Any]:
    """Heavy, non-dummy neighbours of a backbone terminus that are off the chain."""
    on_path = set(path)
    return [
        n
        for n in mol.GetAtomWithIdx(idx).GetNeighbors()
        if n.GetIdx() not in on_path and n.GetAtomicNum() != 0
    ]


def _branch_size(mol: Any, root: int, exclude: int) -> int:
    """Heavy-atom count of the branch hanging off ``root``, away from ``exclude``."""
    seen = {exclude}
    stack = [root]
    count = 0
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        count += 1
        stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors())
    return count


def _chain(unit_smiles: str, count: int) -> Any | None:
    """Link ``count`` copies of a repeat unit, keeping the terminal ``[*]``.

    The dummies are kept, unlike :func:`formulate.experts.polymer.link_repeat_units`,
    because they are what marks the interior of the oligomer.  Cutting a bare
    repeat unit instead of an oligomer is what turns polycaprolactone into
    hexanoic acid plus water: a linkage that straddles the periodic boundary is
    only a complete linkage once there is a neighbour on both sides.
    """
    Chem = _rdkit()
    template = _parse(unit_smiles)
    if template is None or len(_dummies(template)) != 2:
        return None
    chain = Chem.RWMol(template)
    for _ in range(count - 1):
        offset = chain.GetNumAtoms()
        addition = _parse(unit_smiles)
        if addition is None:  # pragma: no cover - parsed once already
            return None
        chain.InsertMol(addition)
        dummies = _dummies(chain)
        left = [d for d in dummies if d < offset][-1]
        right = [d for d in dummies if d >= offset][0]
        left_anchor = chain.GetAtomWithIdx(left).GetNeighbors()[0].GetIdx()
        right_anchor = chain.GetAtomWithIdx(right).GetNeighbors()[0].GetIdx()
        chain.AddBond(left_anchor, right_anchor, Chem.BondType.SINGLE)
        for idx in sorted((left, right), reverse=True):
            chain.RemoveAtom(idx)
    molecule = chain.GetMol()
    try:
        Chem.SanitizeMol(molecule)
    except Exception:
        return None
    return molecule


# --------------------------------------------------------------------------
# The Ertl term
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=4096)
def ertl_score(smiles: str) -> float | None:
    """Raw Ertl-Schuffenhauer accessibility of a molecule, or None."""
    scorer = _sascorer()
    mol = _parse(smiles)
    if scorer is None or mol is None:
        return None
    try:
        return float(scorer.calculateScore(mol))
    except Exception:  # pragma: no cover - the scorer is robust on parsed input
        return None


@functools.lru_cache(maxsize=4096)
def _unstable_motif(smiles: str) -> str | None:
    Chem = _rdkit()
    mol = _parse(smiles)
    if mol is None:  # pragma: no cover - callers parse first
        return None
    for pattern, description in _UNSTABLE_MOTIFS:
        query = Chem.MolFromSmarts(pattern)
        if query is not None and mol.HasSubstructMatch(query):
            return description
    return None


@functools.lru_cache(maxsize=4096)
def monomer_term(smiles: str) -> tuple[float, bool, str] | None:
    """Ertl score of a reconstructed monomer, with the small-molecule ceiling.

    Returns ``(value, ceiling_applied, note)``.
    """
    Chem = _rdkit()
    raw = ertl_score(smiles)
    if raw is None:
        return None
    mol = _parse(smiles)
    if mol is None:  # pragma: no cover
        return None

    if raw <= SMALL_MOLECULE_CEILING:
        return raw, False, ""

    heavy = mol.GetNumHeavyAtoms()
    if heavy > SMALL_MOLECULE_HEAVY_ATOMS:
        return raw, False, ""
    elements = {a.GetSymbol() for a in mol.GetAtoms()}
    if not elements <= _COMMON_ELEMENTS:
        return raw, False, ""
    if mol.GetRingInfo().NumRings() > 1:
        return raw, False, ""
    if Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False):
        return raw, False, ""
    motif = _unstable_motif(smiles)
    if motif is not None:
        return raw, False, ""

    return (
        SMALL_MOLECULE_CEILING,
        True,
        f"the Ertl score of {smiles} is {raw:.2f}, which is the fragment statistics charging "
        f"it for having only {heavy} heavy atoms rather than for being hard; capped at "
        f"{SMALL_MOLECULE_CEILING:.1f}, the value the n-alkane and 1-alkene series plateau at",
    )


# --------------------------------------------------------------------------
# Recognisers: repeat unit -> monomer(s)
# --------------------------------------------------------------------------


def _vinyl_route(mol: Any, path: Sequence[int]) -> Route | Veto | None:
    """Chain growth of an alkene: make the backbone bond a double bond."""
    Chem = _rdkit()
    if len(path) != 2:
        return None
    left, right = path
    atoms = [mol.GetAtomWithIdx(left), mol.GetAtomWithIdx(right)]
    # Both termini must be CARBON.  Ungated, this transform turns the PDMS
    # repeat unit into the silanone C[Si](C)=O, which is exactly the kind of
    # confident nonsense this gate exists to stop.
    if any(a.GetAtomicNum() != 6 for a in atoms):
        return None
    if any(a.GetIsAromatic() or a.IsInRing() for a in atoms):
        return None
    bond = mol.GetBondBetweenAtoms(left, right)
    if bond is None or bond.GetBondType() is not Chem.BondType.SINGLE:
        return None
    if any(_has_exocyclic_double(mol, idx, path) for idx in path):
        # A terminus that already carries a double bond would give a cumulene or
        # a ketene, not a monomer.
        return None

    subs = {idx: _substituents(mol, idx, path) for idx in path}
    heavy_subs = {
        idx: [a for a in group if a.GetSymbol() != "F"] for idx, group in subs.items()
    }
    counts = {idx: len(group) for idx, group in heavy_subs.items()}
    fluorines = sum(1 for group in subs.values() for a in group if a.GetSymbol() == "F")

    rw = Chem.RWMol(mol)
    _clear_stereo(rw, path)
    rw.GetBondBetweenAtoms(left, right).SetBondType(Chem.BondType.DOUBLE)
    _strip_dummies(rw)
    monomer = _smiles(rw)
    if monomer is None:
        return None

    notes: list[str] = []
    penalty, spread = _ROUTE_CHAIN_GROWTH

    total_heavy = sum(counts.values())
    if total_heavy >= 4:
        return Veto(
            f"the implied monomer is the tetrasubstituted alkene {monomer}; tetrasubstituted "
            "ethylenes are not known to undergo chain polymerisation (Odian §3-9b), and a "
            "1-to-10 accessibility scale has no value meaning 'impossible'"
        )
    if all(count >= 1 for count in counts.values()):
        penalty, spread = _ROUTE_HEAD_TO_HEAD
        notes.append(
            f"each backbone carbon carries a substituent, so the implied monomer is the "
            f"1,2-disubstituted alkene {monomer}, which does not homopolymerise (Odian "
            "§3-9b); this is read off how the repeat unit was written, so check the "
            "regiochemistry before believing the charge"
        )
    else:
        gem = max(counts, key=lambda idx: counts[idx])
        if counts[gem] == 2:
            group = heavy_subs[gem]
            aromatic = [a for a in group if a.GetIsAromatic()]
            bulky = [
                a
                for a in group
                if _branch_size(mol, a.GetIdx(), gem) > 1
                and a.GetSymbol() not in ("F", "Cl", "Br", "I")
            ]
            if len(bulky) == 2:
                penalty, spread = _ROUTE_GEM_BULKY
                notes.append(
                    "both substituents on one backbone carbon are larger than a methyl or a "
                    "halogen; propagation past the dimer is sterically blocked (Odian §3-9b)"
                )
            elif aromatic:
                penalty, spread = _ROUTE_GEM_ARYL
                notes.append(
                    "an aryl and a second substituent on the same backbone carbon puts the "
                    "ceiling temperature near ambient, as it does for alpha-methylstyrene "
                    "(61 C), so high polymer does not form on warming"
                )

    if fluorines:
        charge = min(fluorines * _ROUTE_PER_BACKBONE_FLUORINE, _ROUTE_FLUORINE_CAP)
        penalty += charge
        spread = max(spread, 1.0)
        notes.append(
            f"{fluorines} fluorine(s) on the double bond: a fluoroolefin is made by pyrolysis "
            "rather than by cracking and is polymerised under pressure in specialised plant, "
            f"charged {charge:.1f} because the small-monomer ceiling would otherwise price it "
            "the same as ethylene"
        )

    return Route(
        name="vinyl chain growth",
        mechanism=Mechanism.CHAIN_GROWTH,
        monomers=(monomer,),
        penalty=penalty,
        spread=spread,
        notes=tuple(notes),
    )


def _carbonyl_route(mol: Any, path: Sequence[int]) -> Route | None:
    """Chain growth through a carbon-oxygen double bond (formaldehyde to POM)."""
    Chem = _rdkit()
    if len(path) != 2:
        return None
    symbols = sorted(mol.GetAtomWithIdx(i).GetSymbol() for i in path)
    if symbols != ["C", "O"]:
        return None
    if any(mol.GetAtomWithIdx(i).IsInRing() for i in path):
        return None
    bond = mol.GetBondBetweenAtoms(*path)
    if bond is None or bond.GetBondType() is not Chem.BondType.SINGLE:
        return None
    carbon = next(i for i in path if mol.GetAtomWithIdx(i).GetSymbol() == "C")
    if _has_exocyclic_double(mol, carbon, path):
        # [*]OC(=O)[*] would become carbon dioxide, which is not a monomer.
        return None

    rw = Chem.RWMol(mol)
    _clear_stereo(rw, path)
    rw.GetBondBetweenAtoms(*path).SetBondType(Chem.BondType.DOUBLE)
    _strip_dummies(rw)
    monomer = _smiles(rw)
    if monomer is None:
        return None
    penalty, spread = _ROUTE_CARBONYL
    return Route(
        name="carbonyl chain growth",
        mechanism=Mechanism.CHAIN_GROWTH,
        monomers=(monomer,),
        penalty=penalty,
        spread=spread,
        notes=(
            "polymerisation of a carbonyl needs low temperature and rigorous drying, and the "
            "chain unzips unless the ends are capped",
        ),
    )


def _alkyne_route(mol: Any, path: Sequence[int]) -> Route | None:
    """Chain growth of an alkyne: the backbone double bond becomes a triple bond."""
    Chem = _rdkit()
    if len(path) != 2:
        return None
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in path):
        return None
    if any(mol.GetAtomWithIdx(i).GetIsAromatic() or mol.GetAtomWithIdx(i).IsInRing() for i in path):
        return None
    bond = mol.GetBondBetweenAtoms(*path)
    if bond is None or bond.GetBondType() is not Chem.BondType.DOUBLE:
        return None
    if any(len(_substituents(mol, i, path)) > 1 for i in path):
        return None

    rw = Chem.RWMol(mol)
    _clear_stereo(rw, path)
    rw.GetBondBetweenAtoms(*path).SetBondType(Chem.BondType.TRIPLE)
    _strip_dummies(rw)
    monomer = _smiles(rw)
    if monomer is None:
        return None
    penalty, spread = _ROUTE_ALKYNE
    return Route(
        name="alkyne chain growth",
        mechanism=Mechanism.CHAIN_GROWTH,
        monomers=(monomer,),
        penalty=penalty,
        spread=spread,
        notes=(
            "the product of an alkyne polymerisation is insoluble and infusible and the "
            "catalyst is air-sensitive; polyacetylene is a laboratory material",
        ),
    )


def _diene_route(mol: Any, path: Sequence[int]) -> Route | None:
    """1,4 addition of a conjugated diene: invert the bond orders along the backbone."""
    Chem = _rdkit()
    if len(path) != 4:
        return None
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in path):
        return None
    if any(mol.GetAtomWithIdx(i).GetIsAromatic() or mol.GetAtomWithIdx(i).IsInRing() for i in path):
        return None
    bonds = [mol.GetBondBetweenAtoms(path[i], path[i + 1]) for i in range(3)]
    if any(b is None for b in bonds):
        return None
    orders = [b.GetBondType() for b in bonds]
    if orders != [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.SINGLE]:
        return None
    if any(_has_exocyclic_double(mol, i, path) for i in (path[0], path[3])):
        return None

    rw = Chem.RWMol(mol)
    _clear_stereo(rw, path)
    rw.GetBondBetweenAtoms(path[0], path[1]).SetBondType(Chem.BondType.DOUBLE)
    rw.GetBondBetweenAtoms(path[1], path[2]).SetBondType(Chem.BondType.SINGLE)
    rw.GetBondBetweenAtoms(path[2], path[3]).SetBondType(Chem.BondType.DOUBLE)
    _strip_dummies(rw)
    monomer = _smiles(rw)
    if monomer is None:
        return None
    penalty, spread = _ROUTE_CHAIN_GROWTH
    return Route(
        name="1,4 diene chain growth",
        mechanism=Mechanism.CHAIN_GROWTH,
        monomers=(monomer,),
        penalty=penalty,
        spread=spread,
    )


def _ring_atoms_have_carbonyl(mol: Any, ring: Sequence[int]) -> bool:
    Chem = _rdkit()
    members = set(ring)
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() != 6:
            continue
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetIdx() in members:
                continue
            if bond.GetBondType() is Chem.BondType.DOUBLE and other.GetAtomicNum() == 8:
                return True
    return False


def _close_ring(unit_smiles: str, copies: int) -> tuple[str, int, bool] | None:
    """Bond the two backbone termini of ``copies`` linked units into a ring."""
    Chem = _rdkit()
    oligomer = _chain(unit_smiles, copies)
    if oligomer is None:
        return None
    path = _backbone(oligomer)
    if path is None or len(path) < 3:
        return None
    if any(oligomer.GetAtomWithIdx(i).GetIsAromatic() for i in path):
        # Ungated, closing an aromatic backbone produces `c1cc2ccc1-2` for
        # poly(p-phenylene) and a bridged bicycle for poly(phenylene oxide).
        return None
    if all(oligomer.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in path):
        # A saturated carbocycle does not ring-open: what drives ring-opening
        # polymerisation is a heteroatom the propagating centre can attack
        # (Odian ch. 7).  Cyclohexane is not a monomer, and the strained-olefin
        # route a cycloalkene would take - ROMP - is not recognised here at all.
        return None
    if any(
        oligomer.GetAtomWithIdx(i).GetSymbol() not in _ROP_RING_ELEMENTS for i in path
    ):
        # Odian's ring-size argument is about organic rings, so the transform is
        # confined to the elements it was written from (:data:`_ROP_RING_ELEMENTS`).
        # Silicon leaves by this gate and is picked up by :func:`_siloxane_route`;
        # phosphorus, boron, germanium and tin leave by it and are refused, which
        # is the honest answer for a ring nobody has put in a bottle.
        return None
    for i in range(len(path)):
        first = oligomer.GetAtomWithIdx(path[i])
        second = oligomer.GetAtomWithIdx(path[(i + 1) % len(path)])
        if first.GetAtomicNum() != 6 and second.GetAtomicNum() != 6:
            # A heteroatom-heteroatom bond in the ring - the closing bond
            # included, since that one is a ring bond too.  What Odian ch. 7
            # describes is a heteroatom flanked by carbon that a propagating
            # centre attacks; a ring closed across an O-O, S-S or N-O bond is a
            # peroxide, a disulfide or an oxaziridine, whose chemistry is redox
            # rather than polymerisation.  Ungated, ``[*]CCSS[*]`` came back as
            # 1,2-dithietane and ``[*]NOC[*]`` as oxaziridine, both scored 2.5 -
            # easier than poly(ethylene terephthalate) - on rings that are not
            # reagents.
            return None
    for i in range(len(path) - 1):
        bond = oligomer.GetBondBetweenAtoms(path[i], path[i + 1])
        if bond is None or bond.GetBondType() is not Chem.BondType.SINGLE:
            return None
    if oligomer.GetBondBetweenAtoms(path[0], path[-1]) is not None:
        return None

    rw = Chem.RWMol(oligomer)
    rw.AddBond(path[0], path[-1], Chem.BondType.SINGLE)
    _strip_dummies(rw)
    has_carbonyl = _ring_atoms_have_carbonyl(oligomer, path)
    monomer = _smiles(rw)
    if monomer is None:
        return None
    return monomer, len(path), has_carbonyl


def _ring_opening_routes(unit_smiles: str) -> list[Route | Veto]:
    """Ring-opening polymerisation: close the backbone into a ring and open it again."""
    out: list[Route | Veto] = []
    for copies in (1, 2):
        closed = _close_ring(unit_smiles, copies)
        if closed is None:
            continue
        monomer, size, has_carbonyl = closed
        if size < 3:
            continue
        if _unstable_motif(monomer) is not None:
            # An alpha-lactone is not a reagent.  Polylactide and polyglycolide
            # come from the cyclic DIMER, which is exactly why lactide and
            # glycolide are six-rings; try that instead of reporting a ring that
            # cannot be put in a bottle.  A dimer that is itself unisolable - the
            # 1,3-dioxetane polyoxymethylene would give - is dropped as well,
            # which leaves polyoxymethylene to be scored on formaldehyde.
            continue
        if size > _UNSTRAINED_RING_SIZE + 1 and not has_carbonyl:
            # Macrocyclic ring-opening is entropy-driven and needs the ester or
            # amide that makes the chain end reactive.  A large cyclic ether has
            # neither strain nor an acyl group, and crown ethers famously sit in
            # a bottle rather than polymerising.
            continue
        if size == _UNSTRAINED_RING_SIZE and not has_carbonyl:
            out.append(
                Veto(
                    f"the only ring-opening route would need the unstrained six-membered ring "
                    f"{monomer}; a six-ring with no carbonyl has no favourable free energy of "
                    "polymerisation (Odian ch. 7)"
                )
            )
            continue
        if size > _UNSTRAINED_RING_SIZE + 1:
            penalty, spread = _ROUTE_ROP_MACROCYCLE
            note = (
                f"a {size}-membered ring is not strained; its ring-opening is entropy-driven "
                "and far less general than epoxide or lactone chemistry"
            )
        else:
            penalty, spread = _ROUTE_ROP_STRAINED
            note = f"ring strain in a {size}-membered ring drives the polymerisation (Odian ch. 7)"
        out.append(
            Route(
                name=f"ring-opening of a {size}-membered ring"
                + (" (cyclic dimer)" if copies == 2 else ""),
                mechanism=Mechanism.RING_OPENING,
                monomers=(monomer,),
                penalty=penalty,
                spread=spread,
                notes=(note,),
            )
        )
        if out and isinstance(out[-1], Route):
            break
    return out


#: Fragments a step-growth cut can liberate that are not monomers: water,
#: ammonia, and the C1 units a carbonate or urethane linkage is built from.  Each
#: is what is left when an atom loses every acyl bond it had, not something a
#: chemist orders to build this backbone; the C1 units are paid for by
#: :data:`_ROUTE_PHOSGENE_DERIVED` instead of being passed off as a monomer.
_TRIVIAL_FRAGMENTS = ("O", "N", "OC(O)=O", "OC=O", "NC(N)=O", "NC(O)=O")


@functools.lru_cache(maxsize=1)
def _trivial_fragment_keys() -> frozenset[str]:
    Chem = _rdkit()
    out = set()
    for smiles in _TRIVIAL_FRAGMENTS:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            out.add(Chem.MolToSmiles(mol))
    return frozenset(out)


def _step_growth_route(unit_smiles: str) -> Route | None:
    """Polycondensation: cut every backbone acyl-heteroatom bond in a trimer."""
    Chem = _rdkit()
    oligomer = _chain(unit_smiles, 3)
    if oligomer is None:
        return None
    path = _backbone(oligomer)
    if path is None:
        return None
    on_path = set(path)

    cuts: list[tuple[int, int]] = []
    carbonate_like = False
    urethane_like = False
    urea_like = False
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        atom_a, atom_b = oligomer.GetAtomWithIdx(a), oligomer.GetAtomWithIdx(b)
        bond = oligomer.GetBondBetweenAtoms(a, b)
        if bond is None or bond.GetBondType() is not Chem.BondType.SINGLE:
            continue
        for acyl, hetero in ((atom_a, atom_b), (atom_b, atom_a)):
            if acyl.GetIsAromatic() or hetero.GetIsAromatic():
                continue
            if hetero.GetSymbol() not in ("O", "N"):
                continue
            if not _is_acyl(oligomer, acyl.GetIdx()):
                continue
            oxygens = sum(
                1
                for n in acyl.GetNeighbors()
                if n.GetSymbol() == "O" and n.GetIdx() in on_path
            )
            nitrogens = sum(
                1
                for n in acyl.GetNeighbors()
                if n.GetSymbol() == "N" and n.GetIdx() in on_path
            )
            if oxygens >= 2:
                carbonate_like = True
            if oxygens >= 1 and nitrogens >= 1:
                urethane_like = True
            if nitrogens >= 2:
                # A urea carbonyl, and the case the first two clauses missed: the
                # fragment cut out here is urea itself, which
                # :data:`_TRIVIAL_FRAGMENTS` discards on the stated grounds that
                # the C1 unit is paid for rather than passed off as a monomer.
                # Without this clause it was discarded and not paid for, so a
                # polyurea - which needs the same diisocyanate a polyurethane
                # does - was charged the bare polycondensation term.
                urea_like = True
            cuts.append((acyl.GetIdx(), hetero.GetIdx()))
            break

    if not cuts:
        return None

    rw = Chem.RWMol(oligomer)
    for acyl_idx, hetero_idx in cuts:
        rw.RemoveBond(acyl_idx, hetero_idx)
        _free_valence_atom(rw, hetero_idx)
        oxygen = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(acyl_idx, oxygen, Chem.BondType.SINGLE)
        _free_valence_atom(rw, oxygen)
    molecule = rw.GetMol()
    try:
        Chem.SanitizeMol(molecule)
    except Exception:
        return None

    trivial = _trivial_fragment_keys()
    monomers: list[str] = []
    for fragment in Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True):
        if _dummies(fragment):
            continue  # a terminal fragment: half a linkage, not a monomer
        smiles = Chem.MolToSmiles(fragment)
        if smiles in trivial or smiles in monomers:
            continue
        monomers.append(smiles)
    if not monomers:
        return None

    penalty, spread = _ROUTE_STEP_GROWTH
    notes = [
        "the linkage was cut in a trimer, not in the bare repeat unit, so a linkage that "
        "straddles the repeat boundary is a complete linkage"
    ]
    if carbonate_like or urethane_like or urea_like:
        extra, extra_spread = _ROUTE_PHOSGENE_DERIVED
        penalty += extra
        spread = math.hypot(spread, extra_spread)
        notes.append(
            "the backbone carries a carbonate, urethane or urea linkage; the C1 unit it "
            "needs is phosgene, a diaryl carbonate or a diisocyanate, which is charged here "
            "rather than invented as a monomer"
        )
    return Route(
        name="step-growth polycondensation",
        mechanism=Mechanism.STEP_GROWTH,
        monomers=tuple(monomers),
        penalty=penalty,
        spread=spread,
        notes=tuple(notes),
    )


_EWG_SMARTS = "[$([CX3]=[OX1]),$([SX4](=[OX1])=[OX1]),$([NX3](=[OX1])=[OX1]),$([CX2]#[NX1])]"


def _ring_is_activated(mol: Any, atom_idx: int) -> bool:
    """Does the aromatic ring containing ``atom_idx`` bear an electron-withdrawing group?"""
    Chem = _rdkit()
    query = Chem.MolFromSmarts(_EWG_SMARTS)
    if query is None:  # pragma: no cover
        return False
    withdrawing = {m[0] for m in mol.GetSubstructMatches(query)}
    if not withdrawing:
        return False
    for ring in mol.GetRingInfo().AtomRings():
        if atom_idx not in ring:
            continue
        for idx in ring:
            for neighbour in mol.GetAtomWithIdx(idx).GetNeighbors():
                if neighbour.GetIdx() in withdrawing:
                    return True
    return False


def _aryl_ether_route(unit_smiles: str) -> Route | Veto | None:
    """Aryl ether backbones: SNAr on an activated ring, or oxidative coupling."""
    Chem = _rdkit()
    oligomer = _chain(unit_smiles, 3)
    if oligomer is None:
        return None
    path = _backbone(oligomer)
    if path is None:
        return None
    on_path = set(path)

    ether_oxygens = []
    for idx in path:
        atom = oligomer.GetAtomWithIdx(idx)
        if atom.GetSymbol() != "O" or atom.GetIsAromatic() or atom.GetDegree() != 2:
            continue
        neighbours = [n for n in atom.GetNeighbors() if n.GetIdx() in on_path]
        if len(neighbours) == 2 and all(n.GetIsAromatic() for n in neighbours):
            ether_oxygens.append((idx, [n.GetIdx() for n in neighbours]))
    if not ether_oxygens:
        return None

    cuts: list[tuple[int, int]] = []
    for oxygen, neighbours in ether_oxygens:
        activated = [n for n in neighbours if _ring_is_activated(oligomer, n)]
        if activated:
            # SNAr forms the bond from the phenolate to the ring that carries the
            # electron-withdrawing group, so that is the bond to cut.  Each
            # oxygen may lose at most ONE bond; ungated, both PEEK and
            # poly(phenylene oxide) liberate free water instead of monomers.
            cuts.append((activated[0], oxygen))

    if cuts:
        rw = Chem.RWMol(oligomer)
        for aryl_idx, oxygen_idx in cuts:
            rw.RemoveBond(aryl_idx, oxygen_idx)
            _free_valence_atom(rw, oxygen_idx)
            fluorine = rw.AddAtom(Chem.Atom(9))
            rw.AddBond(aryl_idx, fluorine, Chem.BondType.SINGLE)
        molecule = rw.GetMol()
        try:
            Chem.SanitizeMol(molecule)
        except Exception:
            return None
        monomers: list[str] = []
        for fragment in Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True):
            if _dummies(fragment):
                continue
            smiles = Chem.MolToSmiles(fragment)
            if smiles not in monomers:
                monomers.append(smiles)
        if monomers:
            penalty, spread = _ROUTE_SNAR
            return Route(
                name="aromatic nucleophilic substitution",
                mechanism=Mechanism.STEP_GROWTH,
                monomers=tuple(monomers),
                penalty=penalty,
                spread=spread,
                notes=(
                    "the aryl halide is capped with fluorine because that is the leaving group "
                    "SNAr uses; the reaction needs a dipolar aprotic solvent near 300 C",
                ),
            )

    # No activation anywhere: the only remaining route is oxidative coupling of a
    # phenol, which needs the repeat unit to *be* a phenylene oxide.
    unit = _parse(unit_smiles)
    unit_path = _backbone(unit) if unit is not None else None
    if unit is None or unit_path is None:
        return None
    oxygens = [i for i in unit_path if unit.GetAtomWithIdx(i).GetSymbol() == "O"]
    aromatics = [i for i in unit_path if unit.GetAtomWithIdx(i).GetIsAromatic()]
    if len(oxygens) != 1 or len(aromatics) != len(unit_path) - 1:
        return Veto(
            "the backbone is an aryl ether with no electron-withdrawing group to activate "
            "nucleophilic substitution, and it is not a phenylene oxide, so oxidative coupling "
            "of a phenol does not apply either"
        )

    rw = Chem.RWMol(unit)
    _strip_dummies(rw)
    monomer = _smiles(rw)
    if monomer is None:  # pragma: no cover
        return None

    oxygen = oxygens[0]
    ipso = [n for n in unit.GetAtomWithIdx(oxygen).GetNeighbors() if n.GetIsAromatic()]
    blocked = False
    if ipso:
        ortho = [n for n in ipso[0].GetNeighbors() if n.GetIsAromatic()]
        blocked = len(ortho) == 2 and all(
            any(
                x.GetIdx() != ipso[0].GetIdx() and not x.GetIsAromatic()
                for x in a.GetNeighbors()
            )
            for a in ortho
        )
    if blocked:
        penalty, spread = _ROUTE_OXIDATIVE_COUPLING
        note = (
            "the phenol is blocked at both ortho positions, so the coupling goes cleanly "
            "through oxygen"
        )
    else:
        penalty, spread = _ROUTE_OXIDATIVE_COUPLING_OPEN
        note = (
            "the implied phenol has at least one free ortho position, so carbon-carbon coupling "
            "competes with the carbon-oxygen coupling and the product is branched rather than "
            "cleanly the polyether"
        )
    return Route(
        name="oxidative coupling of a phenol",
        mechanism=Mechanism.COUPLING,
        monomers=(monomer,),
        penalty=penalty,
        spread=spread,
        notes=(note,),
    )


def _aryl_aryl_route(mol: Any, path: Sequence[int]) -> Route | None:
    """All-aromatic backbone: the monomer is the dihaloarene a coupling uses."""
    Chem = _rdkit()
    if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in path):
        return None
    # Both attachment points must sit on aromatic CARBON, because the halide this
    # transform writes has to be the aryl halide a Yamamoto or Suzuki coupling
    # actually uses.  Ungated, a backbone joined through an aromatic *nitrogen* -
    # an N-arylene, and any polyimide RDKit perceives as aromatic - came back as
    # an N-bromo compound: `Brc1ccn(Br)c1`, which is a brominating agent rather
    # than a monomer, and a C-N bond is formed by amination, which is a chemistry
    # this module does not recognise at all.  A heteroatom *inside* the ring is
    # fine and stays: 2,5-dibromothiophene is the right monomer for
    # polythiophene, and its sulfur lies on the shortest path.
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in (path[0], path[-1])):
        return None
    rw = Chem.RWMol(mol)
    for idx in _dummies(rw):
        atom = rw.GetAtomWithIdx(idx)
        atom.SetAtomicNum(35)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    monomer = _smiles(rw)
    if monomer is None:
        return None
    penalty, spread = _ROUTE_ARYL_ARYL
    return Route(
        name="aryl-aryl coupling",
        mechanism=Mechanism.COUPLING,
        monomers=(monomer,),
        penalty=penalty,
        spread=spread,
        notes=(
            "a rigid all-aromatic chain precipitates from solution after a handful of units, "
            "so a Yamamoto or Suzuki coupling gives oligomers rather than high polymer, and the "
            "product is neither soluble nor fusible",
        ),
    )


def _siloxane_route(unit_smiles: str, mol: Any, path: Sequence[int]) -> Route | None:
    """Polysiloxane: the monomer is the cyclic tetramer, not the repeat unit."""
    symbols = sorted(mol.GetAtomWithIdx(i).GetSymbol() for i in path)
    if symbols != ["O", "Si"]:
        return None
    Chem = _rdkit()
    oligomer = _chain(unit_smiles, 4)
    if oligomer is None:
        return None
    ring_path = _backbone(oligomer)
    if ring_path is None:
        return None
    rw = Chem.RWMol(oligomer)
    if rw.GetBondBetweenAtoms(ring_path[0], ring_path[-1]) is not None:
        return None
    rw.AddBond(ring_path[0], ring_path[-1], Chem.BondType.SINGLE)
    _strip_dummies(rw)
    monomer = _smiles(rw)
    if monomer is None:
        return None
    penalty, spread = _ROUTE_SILOXANE
    return Route(
        name="ring-opening of a cyclosiloxane",
        mechanism=Mechanism.RING_OPENING,
        monomers=(monomer,),
        penalty=penalty,
        spread=spread,
        notes=(
            "the feedstock is the cyclic tetramer made by hydrolysing a dichlorosilane, not the "
            "repeat unit; the Ertl fragment statistics barely contain silicon, so this monomer "
            "score is the least trustworthy number this expert produces",
        ),
    )


def _enol_correction(route: Route) -> Route:
    """A reconstructed enol is not a monomer; the acetate is, plus a hydrolysis.

    Poly(vinyl alcohol) is the case that matters: the repeat unit reconstructs to
    vinyl alcohol, which tautomerises to acetaldehyde on formation, and the real
    route is to polymerise vinyl acetate and hydrolyse the polymer.
    """
    Chem = _rdkit()
    enol = Chem.MolFromSmarts("[CX3]=[CX3][OX2H1]")
    if enol is None:  # pragma: no cover
        return route
    replaced: list[str] = []
    changed = False
    for smiles in route.monomers:
        mol = _parse(smiles)
        if mol is None or not mol.HasSubstructMatch(enol):
            replaced.append(smiles)
            continue
        match = mol.GetSubstructMatch(enol)
        rw = Chem.RWMol(mol)
        oxygen = match[2]
        carbonyl = rw.AddAtom(Chem.Atom(6))
        ketone = rw.AddAtom(Chem.Atom(8))
        methyl = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(oxygen, carbonyl, Chem.BondType.SINGLE)
        rw.AddBond(carbonyl, ketone, Chem.BondType.DOUBLE)
        rw.AddBond(carbonyl, methyl, Chem.BondType.SINGLE)
        rw.GetAtomWithIdx(oxygen).SetNumExplicitHs(0)
        acetate = _smiles(rw)
        if acetate is None:  # pragma: no cover
            replaced.append(smiles)
            continue
        replaced.append(acetate)
        changed = True
    if not changed:
        return route
    penalty, spread = _ROUTE_PROTECT_AND_HYDROLYSE
    return Route(
        name=route.name + ", then hydrolysis of the polymer",
        mechanism=route.mechanism,
        monomers=tuple(replaced),
        penalty=route.penalty + penalty,
        spread=math.hypot(route.spread, spread),
        notes=route.notes
        + (
            "the reconstructed monomer is an enol, which tautomerises rather than sitting in a "
            "bottle; the acetate is polymerised and the polymer hydrolysed, and the extra step "
            "is charged",
        ),
    )


# --------------------------------------------------------------------------
# Analysing one repeat unit
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1024)
def analyse_repeat_unit(unit_smiles: str) -> UnitAnalysis:
    """Every polymerisation route this repeat unit could have come from."""
    mol = _parse(unit_smiles)
    if mol is None:
        return UnitAnalysis(unit_smiles, failure=f"the repeat unit {unit_smiles!r} does not parse")

    dummies = _dummies(mol)
    if len(dummies) != 2:
        return UnitAnalysis(
            unit_smiles,
            failure=(
                f"the repeat unit {unit_smiles!r} carries {len(dummies)} attachment point(s); a "
                "chain repeat unit needs exactly two [*], one at each end"
            ),
        )
    anchors = [
        n.GetIdx()
        for idx in dummies
        for n in mol.GetAtomWithIdx(idx).GetNeighbors()
    ]
    if len(anchors) == 2 and anchors[0] == anchors[1]:
        return UnitAnalysis(
            unit_smiles,
            failure=(
                f"both attachment points of {unit_smiles!r} sit on the same atom, so the repeat "
                "unit has a one-atom backbone; the recognisers here all need two backbone "
                "termini to work between, and a polysilane or a polymethylene written this way "
                "is outside every chemistry they cover"
            ),
        )
    path = _backbone(mol)
    if path is None:
        return UnitAnalysis(
            unit_smiles,
            failure=(
                f"the two attachment points of {unit_smiles!r} are not joined by a path through "
                "the structure, so there is no backbone to place in any chemistry"
            ),
        )

    routes: list[Route] = []
    vetoes: list[Veto] = []

    def offer(result: Route | Veto | None) -> None:
        if isinstance(result, Route):
            routes.append(_enol_correction(result))
        elif isinstance(result, Veto):
            vetoes.append(result)

    offer(_vinyl_route(mol, path))
    offer(_carbonyl_route(mol, path))
    offer(_alkyne_route(mol, path))
    offer(_diene_route(mol, path))
    offer(_aryl_aryl_route(mol, path))
    offer(_siloxane_route(unit_smiles, mol, path))
    for result in _ring_opening_routes(unit_smiles):
        offer(result)
    offer(_step_growth_route(unit_smiles))
    offer(_aryl_ether_route(unit_smiles))

    return UnitAnalysis(unit_smiles, tuple(routes), tuple(vetoes))


def score_route(route: Route) -> tuple[float, float, list[str]] | None:
    """Monomer term plus route term for one route, with its spread and notes."""
    terms: list[float] = []
    notes: list[str] = []
    ceiling_used = False
    for smiles in route.monomers:
        term = monomer_term(smiles)
        if term is None:
            return None
        value, capped, note = term
        terms.append(value)
        if capped:
            ceiling_used = True
            notes.append(note)
    if not terms:
        return None
    # The hardest monomer sets the accessibility: a route is not easier because
    # one of the two things it needs is cheap.
    monomer = max(terms)
    spread = math.hypot(_ERTL_NOMINAL_STD, route.spread)
    if ceiling_used:
        spread = math.hypot(spread, _CEILING_STD)
    notes.extend(route.notes)
    return monomer + route.penalty, spread, notes


# --------------------------------------------------------------------------
# The expert
# --------------------------------------------------------------------------


def _carothers_charge(spec: PolymerSpec, unit_smiles: str) -> tuple[float, float, str] | None:
    """What a demanding molar-mass target costs a step-growth route."""
    target = spec.number_average_molar_mass
    if target is None:
        return None
    from .polymer import repeat_unit_mass

    try:
        mass = repeat_unit_mass(unit_smiles)
    except Exception:
        return None
    if mass <= 0.0:
        return None
    degree = target.to("g/mol").value / mass
    charge: tuple[float, float, str] | None = None
    for threshold, penalty, spread in _CAROTHERS_STEPS:
        if degree >= threshold:
            charge = (
                penalty,
                spread,
                f"a number-average degree of polymerisation of {degree:.0f} needs a step-growth "
                f"conversion of {1.0 - 1.0 / degree:.4f} (Carothers); reaching it needs exact "
                "stoichiometry, a catalyst and a high-vacuum finisher",
            )
    return charge


def _polymer_components(candidate: Candidate) -> list[Candidate] | None:
    """Each component of an all-polymer formulation, as its own candidate.

    ``None`` when the candidate is not a formulation, or when any component is
    a molecule rather than a polymer. That second case is not a gap: a polymer
    dissolved in a solvent is a different question - what has to be MADE is the
    polymer, and the solvent is bought - and answering it with the same rule
    would quietly charge a formulation for a solvent nobody synthesises.
    """
    mixture = candidate.mixture
    if mixture is None or not mixture.components:
        return None
    out: list[Candidate] = []
    for component in mixture.components:
        if component.polymer is None:
            return None
        out.append(
            Candidate(
                material_class=MaterialClass.POLYMER,
                polymer=component.polymer,
                conditions=candidate.conditions,
            )
        )
    return out


class PolymerFeasibilityExpert(Expert):
    """Whether a repeat unit corresponds to a polymer anyone can make."""

    id = "polymer_feasibility"
    version = "1"
    method = (
        "Ertl-Schuffenhauer accessibility of a monomer reconstructed from the repeat unit "
        "(J. Cheminform. 1:8, 2009), adjusted for the difficulty of the matched "
        "polymerisation (Odian, Principles of Polymerization, 4th ed., 2004)"
    )
    family = PropertyFamily.FEASIBILITY
    supported_classes = frozenset({MaterialClass.POLYMER, MaterialClass.MIXTURE})
    supported_properties = frozenset({"synthetic_accessibility"})
    dependencies: frozenset[str] = frozenset()

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available() and _sascorer() is not None

    def unavailable_reason(self) -> str:
        from formulate import chem

        if not chem.rdkit_available():
            return "RDKit is not installed"
        if _sascorer() is None:
            return "RDKit's contributed SA_Score module could not be imported"
        return ""

    def _software(self) -> SoftwareEnvironment:
        from formulate import chem

        return SoftwareEnvironment.capture(rdkit=chem.rdkit_version())

    # -- domain ------------------------------------------------------------

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        from formulate import chem

        basis = (
            "linear homopolymers and random copolymers whose repeat unit maps onto one of the "
            "named polymerisation chemistries; the monomer half rests on the ChEMBL fragment "
            "frequencies behind the Ertl score, which barely cover silicon, boron, phosphorus "
            "or the metals"
        )
        spec = candidate.polymer
        if spec is None:
            components = _polymer_components(candidate)
            if components is None:
                return ApplicabilityDomain.outside(
                    "candidate carries no polymer, and is not a formulation whose "
                    "components are all polymers",
                    basis=basis,
                )
            # A blend is as far out of domain as its worst component.
            domains = [self.assess_domain(c) for c in components]

            worst = min(domains, key=lambda d: d.score)
            return ApplicabilityDomain(
                score=worst.score,
                in_domain=all(d.in_domain for d in domains),
                warnings=tuple(dict.fromkeys(w for d in domains for w in d.warnings)),
                basis=basis,
            )

        warnings: list[str] = []
        score = 1.0

        if spec.topology is not PolymerTopology.LINEAR:
            warnings.append(
                f"a {spec.topology.value} polymer is not a linear chain; the architecture is set "
                "by a branching or coupling step this reconstruction never sees"
            )
            score = min(score, 0.2)

        backbone = [m for m in spec.monomers if m.role is not MonomerRole.END_GROUP]
        if len(backbone) > 1:
            warnings.append(
                "more than one backbone repeat unit: the score assumes the monomers "
                "copolymerise, and there is no reactivity-ratio model here to say whether they "
                "do - styrene and vinyl acetate are the textbook pair that do not"
            )
            score = min(score, 0.4)

        # The attachment points come back from ``elements`` as "*"; they are a
        # notation, not a chemistry, so they are not exotic.
        exotic: set[str] = set()
        for monomer in spec.monomers:
            exotic |= chem.elements(monomer.smiles) - _COMMON_ELEMENTS - {"*"}
        if exotic:
            warnings.append(
                f"the repeat unit contains {', '.join(sorted(exotic))}; the fragment "
                "frequencies behind the monomer score barely cover these elements, and the "
                "measured consequence is that poly(dimethylsiloxane) - a bulk commodity - "
                "scores like a research compound"
            )
            score = min(score, 0.2)

        return ApplicabilityDomain(
            score=score,
            in_domain=score > 0.3,
            warnings=tuple(warnings),
            basis=basis,
        )

    # -- prediction --------------------------------------------------------

    def _blend(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        """A polymer blend is as hard to make as its hardest component.

        The same rule the mixture thermal expert applies to a molecular
        formulation, and for the same reason: a formulation is gated by the
        component nobody can supply, where an average would let an easy
        component hide a hard one.

        What this rule does NOT cover is stated in the notes rather than folded
        into the number. Blending is a processing step, not a synthesis - two
        polymers that each score 3 are not harder to SYNTHESISE for being mixed -
        and whether the pair is miscible at all is a different question, which
        the blend experts answer and this one must not appear to.
        """
        components = _polymer_components(request.candidate)
        if components is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "candidate carries no polymer. A formulation is answered here only when "
                "every component is a polymer: in a polymer dissolved in a solvent, what "
                "has to be made is the polymer and the solvent is bought, so charging the "
                "formulation for the solvent would be the wrong question",
            )

        per_component = []
        for component in components:
            sub = PredictionRequest(
                candidate=component,
                properties=frozenset({prop}),
                conditions=request.conditions,
                context=request.context,
            )
            per_component.append(self._predict_one(prop, sub, self.assess_domain(component)))

        refused = [p for p in per_component if p is None or p.quantity is None]
        if refused:
            reasons = [
                " ".join(p.notes) if p is not None and p.notes else "no value"
                for p in refused
            ]
            return Prediction.unsupported(
                prop,
                self.id,
                "a component of this blend cannot be made, or its route could not be "
                "identified, so the blend has no accessibility either: "
                + "; ".join(dict.fromkeys(reasons))[:400],
            )

        hardest = max(per_component, key=lambda p: p.quantity.value)
        # The spread is the widest of the components', not the hardest one's.
        # Which component is hardest is itself uncertain, so a blend cannot claim
        # a narrower bar than something it is made of.
        spread = max(
            (p.uncertainty.std for p in per_component if p.uncertainty.std is not None),
            default=None,
        )
        notes = [
            f"the hardest of {len(components)} components gates the blend, at "
            f"{hardest.quantity.value:.2f}",
        ]
        notes.extend(hardest.notes)
        notes.append(
            "blending itself is a processing step and is not charged here: two polymers that "
            "each score 3 are not harder to SYNTHESISE for being mixed"
        )
        notes.append(
            "this says nothing about whether the pair is miscible, which is a different "
            "question and one the blend experts answer"
        )
        return self._make(
            prop,
            hardest.quantity.value,
            "dimensionless",
            request,
            domain,
            std=spread,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                "the widest of the components' own bars, not the hardest component's: which "
                "component gates the blend is itself uncertain, so the blend cannot claim a "
                "narrower error than something it is made of"
            ),
            notes=tuple(notes),
        )

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        if _sascorer() is None:
            return Prediction.unsupported(prop, self.id, self.unavailable_reason())

        spec = request.candidate.polymer
        if spec is None:
            return self._blend(prop, request, domain)

        if spec.topology is PolymerTopology.NETWORK or spec.crosslink_density is not None:
            return Prediction.unsupported(
                prop,
                self.id,
                "a network's accessibility is set by the crosslinker and the cure chemistry, "
                "neither of which the repeat unit encodes",
            )
        if any(m.role is MonomerRole.CROSSLINKER for m in spec.monomers):
            return Prediction.unsupported(
                prop,
                self.id,
                "a crosslinker is declared, so what has to be made is a network; its "
                "accessibility is set by the cure chemistry rather than by the repeat unit",
            )

        backbone = [m for m in spec.monomers if m.role is not MonomerRole.END_GROUP]
        if not backbone:  # pragma: no cover - PolymerSpec forbids it
            return Prediction.unsupported(prop, self.id, "the polymer has no backbone repeat unit")

        analyses = [analyse_repeat_unit(m.smiles) for m in backbone]
        for analysis in analyses:
            if analysis.failure is not None:
                return Prediction.unsupported(prop, self.id, analysis.failure)

        empty = [a for a in analyses if not a.routes]
        if empty:
            reasons = []
            for analysis in empty:
                if analysis.vetoes:
                    reasons.extend(v.reason for v in analysis.vetoes)
                else:
                    # A refusal that only says "no" leaves a chemist with nothing
                    # to do next, and the commonest cause here is not an exotic
                    # backbone at all - it is a repeat unit written on a whole
                    # multiple of its own period, which nothing in this module
                    # reduces.  Say both.
                    reasons.append(
                        f"no polymerisation chemistry matched the backbone of "
                        f"{analysis.smiles!r}; the recognisers cover vinyl, diene, carbonyl "
                        "and alkyne chain growth, ring-opening of a strained organic ring, "
                        "polycondensation, aromatic nucleophilic substitution, oxidative "
                        "coupling of a phenol, aryl-aryl coupling and cyclosiloxane "
                        "ring-opening - check first that the repeat unit is written on its "
                        "smallest period, since '[*]CCCC[*]' is polyethylene written twice "
                        "and is refused for that reason alone"
                    )
            return Prediction.unsupported(
                prop,
                self.id,
                "; ".join(dict.fromkeys(reasons))
                + ". No number is reported: a 1-to-10 accessibility scale has no value "
                "meaning 'nobody can make this'",
            )

        shared = frozenset.intersection(*(a.mechanisms for a in analyses))
        if len(analyses) > 1 and not shared:
            listed = "; ".join(
                f"{a.smiles} -> {', '.join(sorted(m.value for m in a.mechanisms))}"
                for a in analyses
            )
            return Prediction.unsupported(
                prop,
                self.id,
                "the repeat units belong to incompatible polymerisation mechanisms and cannot "
                f"form one chain ({listed})",
            )

        # Each unit is scored on its own best route, restricted to the mechanism
        # the units share; the chain is as hard as its hardest unit.
        value = 0.0
        spread = 0.0
        notes: list[str] = []
        chosen: list[Route] = []
        for analysis in analyses:
            candidates = [r for r in analysis.routes if not shared or r.mechanism in shared]
            scored = [(score_route(r), r) for r in candidates]
            usable = [(s, r) for s, r in scored if s is not None]
            if not usable:
                return Prediction.failed(
                    prop, self.id, f"the Ertl scorer produced no value for {analysis.smiles!r}"
                )
            best, route = min(usable, key=lambda pair: pair[0][0])
            value = max(value, best[0])
            # The spread is taken over EVERY unit, not just the hardest one.
            # Pairing the hardest unit's value with only its own bar let a
            # copolymer claim a narrower error than one of the things it is made
            # of: poly(vinyl alcohol) carries +/-1.58 alone, but copolymerised
            # with PTFE - which is harder, and better known at +/-1.50 - the pair
            # reported +/-1.80 where the wider component implies +/-1.87.  Which
            # unit is hardest is itself uncertain, so the bar cannot shrink by
            # adding a comonomer.
            spread = max(spread, best[1])
            notes.append(
                f"{analysis.smiles}: {route.name} from {' + '.join(route.monomers)}"
            )
            notes.extend(best[2])
            chosen.append(route)

        if len(backbone) > 1:
            penalty, extra = _ROUTE_COPOLYMER
            value += penalty
            spread = math.hypot(spread, extra)
            notes.append(
                "a copolymer is charged for the composition control it needs; this expert has "
                "no reactivity-ratio model, so it cannot see whether the pair actually "
                "copolymerises at the stated composition"
            )

        if spec.tacticity in (Tacticity.ISOTACTIC, Tacticity.SYNDIOTACTIC):
            penalty, extra = _ROUTE_STEREOREGULAR
            value += penalty
            spread = math.hypot(spread, extra)
            article = "an" if spec.tacticity.value[0] in "aeiou" else "a"
            notes.append(
                f"{article} {spec.tacticity.value} chain needs a stereospecific catalyst; one flat "
                "charge cannot tell commercial isotactic polypropylene from isotactic "
                "poly(vinyl chloride), which nobody sells"
            )

        if any(r.mechanism is Mechanism.STEP_GROWTH for r in chosen):
            # The first backbone unit's mass stands in for a copolymer's average.
            # The charge is a coarse two-step threshold, so a few per cent of
            # error in the repeat-unit mass cannot move which step it lands on.
            charge = _carothers_charge(spec, backbone[0].smiles)
            if charge is not None:
                penalty, extra, note = charge
                value += penalty
                spread = math.hypot(spread, extra)
                notes.append(note)

        if not domain.in_domain:
            spread *= _OUT_OF_DOMAIN_WIDENING

        clamped = min(max(value, 1.0), 10.0)
        if clamped != value:
            notes.append(
                f"the sum of the monomer and polymerisation terms was {value:.2f} and has been "
                f"clamped to the registered 1-to-10 range; the ordering above 10 is not "
                "meaningful anyway"
            )

        notes.append(
            "this answers whether the POLYMER can be made, which is not the question the "
            "molecule-class feasibility expert answers, although both report a 1-to-10 "
            "synthetic_accessibility"
        )
        notes.append(
            "the monomer half is a fragment-frequency prior over what chemists have made "
            "before: a plausible monomer nobody has ever reported still scores well"
        )

        return self._make(
            prop,
            clamped,
            "",
            request,
            domain,
            std=spread,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                "nothing here is fitted, so no held-out error exists: the Ertl score carries no "
                f"calibrated error and keeps its published nominal spread of "
                f"{_ERTL_NOMINAL_STD:.1f}, and the polymerisation term is an ordinal judgement "
                "from the polymerisation literature quoted wider the fewer polymers stand "
                "behind it"
            ),
            notes=tuple(dict.fromkeys(notes)),
            routes=[r.name for r in chosen],
            monomers=sorted({m for r in chosen for m in r.monomers}),
        )
