"""Heteroatom backbones: the polyethers, carbonates, urethanes, siloxanes,
sulfones, sulfides and imides.

``vinyl.py`` writes chains whose backbone is carbon and ``condensation.py``
writes chains whose backbone is carbon joined by esters and amides.  Between
them they regenerate 50 of the 57 bundled reference polymers and cannot reach
the other eight, and the eight are not a scatter: every one of them has the
heteroatom *in* the backbone rather than hanging off it.  Poly(ethylene oxide),
poly(propylene oxide), poly(tetramethylene oxide), polyoxymethylene,
poly(oxytrimethylene), polycaprolactone, poly(2,6-dimethyl-1,4-phenylene oxide)
and poly(ether ether ketone) are this module's acceptance test, and it
regenerates all eight.

This is also where the property range ends.  Poly(ethylene oxide) is the most
water-soluble synthetic polymer in commerce and Kapton is the highest-Tg one;
neither is reachable from a carbon backbone, so without this family the engine's
answer to "give me something very polar" or "give me something that survives
300 C" was bounded by the wrong end of the wrong list.

What this module is careful about
---------------------------------
**It gates on chemistry, not on syntax.**  Nine gates, each naming the reaction
that happens *instead* of polymerisation: :func:`refusals` returns every
structure the grammar built and then refused, and the probe set at the bottom
exists so the gates are exercised rather than asserted.  Tetrahydropyran,
1,4-dioxane, thiane, glycidol, 3-ethyl-3-(hydroxymethyl)oxetane, glycidyl
methacrylate, ethylene peroxide, ethylene carbonate, propylene carbonate, the
bisphenol urethanes, the unactivated aryl ethers, poly(phenylene oxide) from
plain phenol, 2-methyltetrahydrofuran and tetrahydrothiophene are all built here
and all refused.

**It does not refuse silicon and sulfur, and that is deliberate.**
``vinyl.py`` refuses poly(vinyl bromide) because bromine is outside the element
set the density and Tg models were fitted over.  Doing the same here would
delete the silicones, the sulfones and the sulfides outright, and unlike vinyl
bromide those have no in-budget twin: there is no carbon-backbone polymer that
behaves like poly(dimethylsiloxane).  So they are generated and *marked*
``outside fitted elements``, and the density and Tg experts refuse them
downstream by their own domain check.  An expert saying "outside my domain" is
an answer; a library that never proposes the material is a silent gap.

**It says which polymers have a melt.**  Polyimides and poly(phenylene sulfide
sulfone)-type backbones reach their decomposition temperature before they flow,
which makes them unreachable for every melt-processing requirement and still the
right answer for a stiffness or service-temperature one.  Every unit carries a
:class:`Melt` and the ones that are not melt-processable say so in their name.

**It says that a polyurethane repeat unit is half a polymer.**  A real
polyurethane is a *segmented block copolymer*: a rigid diisocyanate/chain-
extender hard segment phase-separated from a polyether or polyester soft
segment, and the modulus, the resilience and the service window are all set by
that phase separation.  The units generated here are hard segments alone, which
is the honest thing to write down and is nothing like the material - so they
carry ``Melt.HARD_SEGMENT`` and say ``hard segment of a block copolymer`` in
their name.  Averaging a hard segment's properties and calling the answer a
polyurethane is exactly the substitution this repository forbids, so the label
is not decoration.

**Polycaprolactone is here, and it overlaps ``condensation.py`` on purpose.**
That module writes it as the AB polyester of 6-hydroxyhexanoic acid,
``[*]OCCCCCC(=O)[*]``; the bundled file writes it at the lactone cut,
``[*]CCCCCC(=O)O[*]``, and those two canonical SMILES differ, which is why the
measurement over the other two modules recorded polycaprolactone as unreachable.
It is the same periodic chain.  It is regenerated here at the bundled cut,
because caprolactone ring-opening is how the polymer is actually made, and this
is the one place in the three modules where two families write one chain.  No
other lactone is generated here: the lactones belong to ``condensation.py``.

Measured on this installation
-----------------------------
:func:`units` returns **171** repeat units - 42 polyurethane hard segments, 36
polyimides, 35 aryl polyethers, 23 polyethers, 19 polycarbonates, 7 silicones,
5 polysulfides, 3 polyacetals and polycaprolactone - and :func:`refusals`
returns **24** structures the grammar built and would not emit.  **39** of the
171 are polymers that are or have been sold, **40** have been made and
published, and **92** are combinations nobody appears to have run; every
monomer in all 171 is purchasable, 77 units entirely from commodities.

*Rediscovery.*  **8 of 8**.  Poly(ethylene oxide), poly(propylene oxide),
poly(tetramethylene oxide), polyoxymethylene, poly(oxytrimethylene),
polycaprolactone, poly(2,6-dimethyl-1,4-phenylene oxide) and poly(ether ether
ketone) all come back as canonical SMILES against canonical SMILES, and nothing
here collides with a reference polymer belonging to another module's family.

*Feasibility.*  ``PolymerFeasibilityExpert`` scores **144 of 171** and has no
route at all for **27**; :data:`NO_FEASIBILITY_ROUTE` names them and says which
three chemistries are missing from its table.  Scores run 2.05 to 6.56, median
3.34, against poly(ethylene terephthalate) at 2.50.

*Two places the expert and this module disagree, both left visible.*  The seven
silicones score 3.96 to 6.56, the top of the whole module, and the expert's own
source says why: its fragment statistics barely contain silicon, so a bulk
commodity prices like a research compound.  And poly(propylene carbonate), which
this module refuses to condense from 1,2-propanediol and emits from CO2 and
propylene oxide, is offered a ring-opening of propylene carbonate by the expert -
a monomer sold by the tonne as a solvent because that ring does not open.
Neither is corrected here; correcting one would mean fitting an expert to this
library.

*What the rest of the panel will then refuse.*  **35 of 171** contain silicon or
sulfur and are outside the element set the density and Tg models were fitted
over, so those two experts will decline them by their own domain check.  That is
the correct answer rather than a coverage gap, and it is in the label.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Elements the density and Tg models were fitted over, mirroring
#: ``formulate.experts.polymer._FITTED_ELEMENTS``.  Restated rather than
#: imported because a private name is not an interface, and a test asserts the
#: two are equal so drift fails loudly.
#:
#: Unlike ``vinyl.ELEMENT_BUDGET`` this is *not* a gate.  Silicon and sulfur are
#: outside it and the silicones, sulfones and sulfides are the whole reason this
#: family reaches the low-surface-energy and high-temperature ends of the
#: property range; refusing them would hide a real material behind a modelling
#: limit.  They are emitted and flagged, and the experts refuse them themselves.
ELEMENT_BUDGET = frozenset({"H", "C", "N", "O", "F", "Cl"})

#: A para-phenylene written as a connector: the first atom bonds to whatever
#: precedes it and the fourth to whatever follows, so ``"O" + PARA + "O"`` is
#: hydroquinone's contribution to a backbone.  This is the same convention
#: ``condensation.py`` uses for terephthalic acid, and it is what makes the
#: bisphenol and dianhydride libraries assemble by string substitution instead
#: of by hand.
PARA = "c1ccc(cc1)"

#: The same connector, 1,3 rather than 1,4.  A meta link is not a cosmetic
#: variant: it is the standard way of stopping a rigid aromatic chain from
#: crystallising, which is why Nomex is meta and Kevlar is para, and why
#: resorcinol and m-phenylenediamine are in the libraries below.
META = "c1cc(ccc1)"


class Availability(str, Enum):
    """How real a generated polymer is, which is not how makeable it is.

    The feasibility expert answers "could someone make this?".  This answers
    "has someone?", and a search needs both: a merely constructible candidate
    is a legitimate answer only if the run can see that nobody has made it.
    """

    #: Sold, or has been sold, as a material.  A trade or common name exists.
    COMMERCIAL = "commercial"
    #: Made and published, not a product.  The route is known to work.
    REPORTED = "reported"
    #: Every monomer is sold and the combination polymerises by the same
    #: chemistry as its commercial neighbours; nobody appears to have bothered.
    CONSTRUCTIBLE = "constructible"


class Scale(str, Enum):
    """Where the monomer is bought, which is a constraint on the answer."""

    #: Ten-kilotonne scale or more: a commodity with a market price.
    BULK = "bulk"
    #: A catalogue chemical, priced per kilogram rather than per tonne.
    CATALOGUE = "catalogue"


class Melt(str, Enum):
    """Whether the polymer has a melt at all, which decides half the panel.

    The melt expert makes this call from structure and thermal data; what is
    recorded here is the chemist's version of the same fact, so that a name in
    a ranking is legible without opening the prediction.  They are allowed to
    disagree and the disagreement is worth seeing.
    """

    #: Extruded, moulded or spun from the melt.  Everything ordinary.
    MELT = "melt processable"
    #: Decomposes before it flows.  The aromatic polyimides and the rigid
    #: all-para aromatics; they are made by casting a soluble precursor and
    #: cyclising it in place, and no melt-processing requirement can reach them.
    NONE = "not melt processable"
    #: The hard segment of a segmented block copolymer, written on its own.
    #: The real material is a two-phase copolymer and this unit is one phase.
    HARD_SEGMENT = "hard segment of a block copolymer"
    #: A fluid or a cured elastomer rather than a thermoplastic: the silicones
    #: and the liquid polysulfides have no melt-processing window because they
    #: are never solid before they are crosslinked.
    ELASTOMER = "fluid or cured elastomer, not melt processed"


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HeteroUnit:
    """One proposal, with everything a ranking needs in order to judge it."""

    name: str
    smiles: str
    subfamily: str
    monomers: tuple[str, ...]
    availability: Availability
    scale: Scale
    melt: Melt
    common_name: str = ""
    #: "direct" means the monomers named polymerise to this unit as written.
    #: Anything else is a route the structure does not reveal, and it also
    #: exempts the unit from the gate that would otherwise refuse it - which is
    #: the only way poly(ethylene carbonate) can be emitted without claiming
    #: that ethylene glycol and phosgene give a polymer.
    route: str = "direct"
    note: str = ""

    @property
    def outside_element_budget(self) -> bool:
        return bool(elements(self.smiles) - ELEMENT_BUDGET)

    @property
    def label(self) -> str:
        """The name :func:`units` emits: what it is, what it is called, how real.

        Every flag that changes what the number downstream *means* is in here,
        because ``(name, smiles)`` is the only channel an explorer taking pairs
        has, and a ranking that cannot tell a product from a possibility - or a
        melt from a thing that decomposes - has answered a different question.
        """
        parts: list[str] = []
        if self.common_name and self.common_name != self.name:
            parts.append(self.common_name)
        parts.append(self.availability.value)
        if self.scale is Scale.CATALOGUE:
            parts.append("catalogue monomer")
        if self.melt is not Melt.MELT:
            parts.append(self.melt.value)
        if self.outside_element_budget:
            parts.append("outside fitted elements")
        if self.name in NO_FEASIBILITY_ROUTE:
            parts.append("no feasibility route")
        return f"{self.name} ({', '.join(parts)})"


@dataclass(frozen=True, slots=True)
class Refusal:
    """A structure the grammar built and then refused, with the reason."""

    name: str
    smiles: str
    gate: str
    reason: str


@dataclass(frozen=True, slots=True)
class Monomer:
    """A difunctional monomer, stored as the fragment it contributes.

    ``core`` is the piece *between* the two functional groups, so the linkage
    is written once in a template rather than once per monomer.  ``short`` is
    what goes in a polymer name; ``name`` is what you would order.
    """

    name: str
    core: str
    short: str
    scale: Scale = Scale.BULK
    #: True when the two functional groups sit on aromatic carbon.  It changes
    #: what the monomer will react with - see the aryl-carbamate gate.
    aromatic: bool = False
    note: str = ""


#: The 27 units the engine's own ``PolymerFeasibilityExpert`` returns no score
#: for, measured by running all 171 through it.  Three chemistries are missing
#: from its route table and this is exactly which units they cost:
#:
#: * **imide formation from a dianhydride and a diamine** - 24 of the 36
#:   polyimides.  The other twelve are the ODPA and BPADA ones, and they are
#:   scored through nucleophilic aromatic substitution on a bis(halophthalimide),
#:   which is not a consolation prize: it is how commercial polyetherimide is
#:   actually made.  So the expert reaches every polyimide whose dianhydride
#:   carries an aryl ether and no other, which is a sharp and correct line.
#: * **sodium sulfide against a dihaloarene** - poly(p-phenylene sulfide).
#: * **a disulfide backbone** - the two Thiokols.  The expert refuses to close a
#:   ring across an S-S bond, on the stated grounds that a disulfide's chemistry
#:   is redox rather than polymerisation, which is right about the ring and
#:   wrong about these two polymers.
#:
#: This is a list of names rather than a rule because it is a *measurement*, and
#: a test asserts it still holds: if the expert grows an imide route, or loses
#: one, that is a change worth failing on rather than absorbing.  It is in the
#: label because a candidate with no synthetic-accessibility number is penalised
#: by the missing-objective policy, and that penalty should not be mistaken for
#: a judgement about the polymer.
NO_FEASIBILITY_ROUTE = frozenset(
    {
        "poly(p-phenylene sulfide)",
        "poly(ethylene disulfide)",
        "poly(oxydiethylene formal disulfide)",
    }
    | {
        f"poly({code}-{amine} imide)"
        for code in ("PMDA", "BPDA", "BTDA", "6FDA")
        for amine in ("PPD", "MPD", "ODA", "MDA", "DDS", "BAPP")
    }
)


def _rdkit() -> Any:
    from rdkit import Chem

    return Chem


@functools.lru_cache(maxsize=4096)
def canonical(smiles: str) -> str | None:
    """RDKit canonical SMILES, for comparing a generated unit with a bundled one."""
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


@functools.lru_cache(maxsize=4096)
def elements(smiles: str) -> frozenset[str]:
    """Every element in the repeat unit, hydrogens included, [*] excluded."""
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return frozenset()
    return frozenset(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms()) - {"*"}


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------
#
# A repeat unit carries exactly two [*], one at each end of the backbone, and
# must parse when they are CAPPED with [H] rather than deleted.  Each template
# below reproduces a bundled reference string character for character when its
# own monomer is substituted in, which is the cheapest possible check that the
# convention has been followed: see the rediscovery test.

#: Head-to-tail ring-opening of a 2-substituted oxirane.  ``{r}`` is the
#: substituent on the ring carbon, empty for ethylene oxide.  Substituting ""
#: gives poly(ethylene oxide) and "(C)" gives poly(propylene oxide), both
#: verbatim.
EPOXIDE = "[*]CC{r}O[*]"

#: Ring-opening of a 3,3-disubstituted oxetane.  "" gives poly(oxytrimethylene)
#: verbatim; "(CCl)(CCl)" gives Penton.
OXETANE = "[*]CC{r}CO[*]"

#: Ring-opening of tetrahydrofuran and its homologues: ``{core}`` is the
#: methylene run.  "CCCC" gives poly(tetramethylene oxide) verbatim.
CYCLIC_ETHER = "[*]{core}O[*]"

#: Ring-opening of a lactone, cut where the bundled file cuts polycaprolactone.
LACTONE = "[*]{core}C(=O)O[*]"

#: The aryl ether backbones: a bisphenol against an activated dihaloarene by
#: nucleophilic aromatic substitution.  Hydroquinone against
#: 4,4'-difluorobenzophenone gives poly(ether ether ketone) verbatim.
ARYL_ETHER = "[*]O{bisphenol}O{activated}[*]"

#: The AB form, from a monomer carrying both the phenol and the halide.
ARYL_ETHER_AB = "[*]O{activated}[*]"

#: Oxidative coupling of a 2,6-disubstituted phenol.  ``{r}`` is the ortho
#: substituent; "C" gives poly(2,6-dimethyl-1,4-phenylene oxide) verbatim.
PHENYLENE_OXIDE = "[*]Oc1cc({r})c([*])c({r})c1"

#: Carbonate.  ``{core}`` sits between the two carbonate oxygens.
CARBONATE = "[*]OC(=O)O{core}[*]"

#: Urethane, written diisocyanate-first so that the nitrogen an [H] cap lands
#: on is the urethane N-H it really is.  This is one hard segment.
URETHANE = "[*]N{iso}NC(=O)O{diol}OC(=O)[*]"

#: Siloxane.  Both substituents sit on the silicon; the backbone alternates
#: Si and O and cannot be written any other way.
SILOXANE = "[*][Si]{r}O[*]"

#: Thioether, the sulfur analogue of :data:`EPOXIDE` - thiirane opens the same
#: way oxirane does, and faster.
THIIRANE = "[*]CC{r}S[*]"


# --------------------------------------------------------------------------
# Cyclic ethers: the polyethers by ring-opening
# --------------------------------------------------------------------------

#: The epoxides, which are where this family's tonnage is: poly(ethylene oxide)
#: and poly(propylene oxide) between them are the polyol backbone of essentially
#: every flexible polyurethane foam made.
#:
#: Only monomers that are sold are here.  The three that look like obvious
#: members and are absent are absent for reasons the gates below state and the
#: probes exercise: glycidol carries a second hydroxyl and gives hyperbranched
#: polyglycerol rather than a chain, glycidyl methacrylate carries a
#: polymerisable ester and gives a network, and ethylene carbonate - which is an
#: oxirane plus CO2 - is a solvent, not a monomer.
#:
#: (name, substituent as written inside the template, short name, availability,
#: scale, note)
EPOXIDES: tuple[tuple[str, str, str, Availability, Scale, str], ...] = (
    ("ethylene oxide", "", "poly(ethylene oxide)", Availability.COMMERCIAL, Scale.BULK,
     "the most water-soluble synthetic polymer in commerce; sold as PEG below about "
     "20 kg/mol and as PEO above it"),
    ("propylene oxide", "(C)", "poly(propylene oxide)", Availability.COMMERCIAL, Scale.BULK,
     "the polyol half of flexible polyurethane foam"),
    ("1,2-butylene oxide", "(CC)", "poly(butylene oxide)", Availability.COMMERCIAL, Scale.BULK,
     "sold as a polyol; the extra methylene buys hydrophobicity over PPO"),
    ("1,2-epoxyhexane", "(CCCC)", "poly(1,2-hexylene oxide)", Availability.REPORTED,
     Scale.CATALOGUE, ""),
    ("1,2-epoxydodecane", "(CCCCCCCCCC)", "poly(1,2-dodecylene oxide)", Availability.REPORTED,
     Scale.CATALOGUE, "an alpha-olefin oxide; the side chain is long enough to crystallise"),
    ("epichlorohydrin", "(CCl)", "polyepichlorohydrin", Availability.COMMERCIAL, Scale.BULK,
     "the CO and ECO oil-resistant rubbers (Hydrin)"),
    ("styrene oxide", "(c1ccccc1)", "poly(styrene oxide)", Availability.REPORTED,
     Scale.CATALOGUE, ""),
    ("isobutylene oxide", "(C)(C)", "poly(isobutylene oxide)", Availability.REPORTED,
     Scale.CATALOGUE, "a 2,2-disubstituted oxirane still opens: the ring strain is 13 "
                      "kcal/mol and no substituent pattern removes it"),
    ("glycidyl methyl ether", "(COC)", "poly(glycidyl methyl ether)", Availability.REPORTED,
     Scale.CATALOGUE, ""),
    ("glycidyl ethyl ether", "(COCC)", "poly(glycidyl ethyl ether)", Availability.REPORTED,
     Scale.CATALOGUE, ""),
    ("n-butyl glycidyl ether", "(COCCCC)", "poly(n-butyl glycidyl ether)",
     Availability.REPORTED, Scale.BULK, "an epoxy reactive diluent, sold by the tonne"),
    ("tert-butyl glycidyl ether", "(COC(C)(C)C)", "poly(tert-butyl glycidyl ether)",
     Availability.REPORTED, Scale.CATALOGUE, ""),
    ("2-ethylhexyl glycidyl ether", "(COCC(CC)CCCC)", "poly(2-ethylhexyl glycidyl ether)",
     Availability.CONSTRUCTIBLE, Scale.BULK, ""),
    ("phenyl glycidyl ether", "(COc1ccccc1)", "poly(phenyl glycidyl ether)",
     Availability.REPORTED, Scale.BULK, ""),
    ("allyl glycidyl ether", "(COCC=C)", "poly(allyl glycidyl ether)", Availability.COMMERCIAL,
     Scale.BULK,
     "the cure site in ECO rubber. The pendant allyl survives the polymerisation rather "
     "than crosslinking it: an allyl group does not propagate, it transfers (Odian "
     "3-9c), which is exactly why it is used as a latent cure site"),
)

#: Cyclohexene oxide is written out rather than templated because its two
#: backbone carbons are ring atoms, so the substituent is a ring closure and not
#: a branch.  Cationic ring-opening of it is real and the polymer is rigid -
#: a cyclohexylene in the backbone is the aliphatic way to raise a Tg.
CYCLOHEXENE_OXIDE = HeteroUnit(
    name="poly(cyclohexene oxide)",
    smiles="[*]OC1CCCCC1[*]",
    subfamily="polyether",
    monomers=("cyclohexene oxide",),
    availability=Availability.REPORTED,
    scale=Scale.BULK,
    melt=Melt.MELT,
    note="cationic ring-opening; the trans-diaxial opening makes the chain irregular",
)

#: Oxetanes.  Four-membered, so still strained, and cationic rather than
#: anionic: an oxetane has no strain-relieved alkoxide to propagate.  Penton was
#: sold for years as a chlorinated engineering plastic before it was withdrawn,
#: which is exactly the kind of polymer a search should be able to re-find.
OXETANES: tuple[tuple[str, str, str, Availability, Scale, str], ...] = (
    ("oxetane", "", "poly(oxytrimethylene)", Availability.REPORTED, Scale.CATALOGUE,
     "the bundled POTM"),
    ("3,3-bis(chloromethyl)oxetane", "(CCl)(CCl)", "poly(3,3-bis(chloromethyl)oxetane)",
     Availability.COMMERCIAL, Scale.CATALOGUE,
     "sold as Penton, a chlorinated engineering thermoplastic, now withdrawn"),
    ("3,3-dimethyloxetane", "(C)(C)", "poly(3,3-dimethyloxetane)", Availability.REPORTED,
     Scale.CATALOGUE, ""),
    ("3-methyl-3-(methoxymethyl)oxetane", "(C)(COC)",
     "poly(3-methyl-3-(methoxymethyl)oxetane)", Availability.REPORTED, Scale.CATALOGUE,
     "a cationic-cure monomer sold for coatings"),
)

#: The larger cyclic ethers and the cyclic acetals, by methylene run.  The
#: missing member is the whole point: the five-membered ring (tetrahydrofuran)
#: polymerises and the six-membered one (tetrahydropyran) does not, because an
#: unstrained chair has no free energy of polymerisation to give (Odian ch. 7).
#: That is not a curiosity, it is the single most quoted fact about ring-opening
#: thermodynamics, and the gate for it is exercised by a probe rather than
#: enforced by leaving a row out of this tuple.
CYCLIC_ETHERS: tuple[tuple[str, str, str, Availability, Scale, str], ...] = (
    ("tetrahydrofuran", "CCCC", "poly(tetramethylene oxide)", Availability.COMMERCIAL,
     Scale.BULK, "PTMEG, the soft segment of every high-performance TPU and Spandex"),
    ("3-methyltetrahydrofuran", "CCC(C)C", "poly(3-methyltetramethylene oxide)",
     Availability.REPORTED, Scale.CATALOGUE,
     "sold as a copolymer with THF to give an amorphous polyether diol"),
    ("oxepane", "CCCCCC", "poly(hexamethylene oxide)", Availability.REPORTED, Scale.CATALOGUE,
     "a seven-ring is strained again, so it opens where the six-ring will not"),
    ("1,3-dioxolane", "COCC", "poly(1,3-dioxolane)", Availability.REPORTED, Scale.BULK,
     "a cyclic acetal: the chain is half polyoxymethylene and half poly(ethylene oxide)"),
    ("1,3-dioxepane", "COCCCC", "poly(1,3-dioxepane)", Availability.REPORTED, Scale.CATALOGUE,
     ""),
)

#: Polyoxymethylene is written out because its backbone is two atoms long and no
#: ring-opening template reaches it: the monomer is formaldehyde by chain growth
#: through the carbonyl, or the cyclic trimer trioxane by cationic ring-opening,
#: and both are run industrially.  The chain unzips from either end above about
#: 120 C, which is why every commercial grade is end-capped or copolymerised
#: with a few per cent of ethylene oxide - a fact about the ends, not the repeat
#: unit, and so invisible to everything downstream.
POLYOXYMETHYLENE = HeteroUnit(
    name="polyoxymethylene",
    smiles="[*]CO[*]",
    subfamily="polyacetal",
    monomers=("formaldehyde", "1,3,5-trioxane"),
    availability=Availability.COMMERCIAL,
    scale=Scale.BULK,
    melt=Melt.MELT,
    common_name="POM, acetal",
    note="unzips above ~120 C unless the chain ends are capped; commercial grades are "
         "copolymers with a few per cent ethylene oxide",
)

#: Polycaprolactone, at the cut the bundled file uses.  ``condensation.py``
#: writes the same chain from 6-hydroxyhexanoic acid and cuts it one bond
#: further along, so the two canonical SMILES differ and a caller deduplicating
#: on canonical SMILES will carry both.  That is stated in the module docstring
#: and is the only overlap between the three modules; no other lactone is
#: generated here.
POLYCAPROLACTONE = HeteroUnit(
    name="polycaprolactone",
    smiles=LACTONE.format(core="CCCCC"),
    subfamily="lactone polyester",
    monomers=("epsilon-caprolactone",),
    availability=Availability.COMMERCIAL,
    scale=Scale.BULK,
    melt=Melt.MELT,
    common_name="PCL",
    note="the same chain condensation.py writes from 6-hydroxyhexanoic acid, cut one bond "
         "further along; caprolactone ring-opening is the route actually used",
)


# --------------------------------------------------------------------------
# Aryl ethers
# --------------------------------------------------------------------------

#: The bisphenols.  These serve two templates - nucleophilic aromatic
#: substitution against an activated dihaloarene, and the carbonate - because
#: the phenolate is the nucleophile in both, and every one of them is a monomer
#: with a price per tonne or per kilogram.
#:
#: Bisphenol A carries the polycarbonate and polysulfone industries between
#: them.  Hydroquinone and 4,4'-biphenol are the rigid ones and are why this
#: family reaches a 220 C Tg without leaving melt processing.  Bisphenol S and
#: 4,4'-dihydroxybenzophenone are the two that arrive already carrying an
#: electron-withdrawing group, which matters for the activation gate below: it
#: cannot tell which half of a pair carried the leaving group, so no
#: unactivated dihaloarene is put in the library for them to meet.
BISPHENOLS: tuple[Monomer, ...] = (
    Monomer("hydroquinone", PARA, "hydroquinone", Scale.BULK, aromatic=True),
    Monomer("resorcinol", META, "resorcinol", Scale.BULK, aromatic=True,
            note="the meta link is the standard way to stop an aromatic chain crystallising"),
    Monomer("4,4'-biphenol", PARA + PARA, "biphenol", Scale.CATALOGUE, aromatic=True),
    Monomer("bisphenol A", PARA + "C(C)(C)" + PARA, "bisphenol A", Scale.BULK, aromatic=True),
    Monomer("bisphenol F", PARA + "C" + PARA, "bisphenol F", Scale.BULK, aromatic=True),
    Monomer("bisphenol S", PARA + "S(=O)(=O)" + PARA, "bisphenol S", Scale.BULK, aromatic=True),
    Monomer("bisphenol Z", PARA + "C5(CCCCC5)" + PARA, "bisphenol Z", Scale.CATALOGUE,
            aromatic=True, note="1,1-bis(4-hydroxyphenyl)cyclohexane; the cyclohexylidene "
                                "raises Tg over bisphenol A"),
    Monomer("bisphenol TMC", PARA + "C5(CC(C)CC(C)(C)C5)" + PARA, "bisphenol TMC",
            Scale.CATALOGUE, aromatic=True,
            note="the APEC high-heat polycarbonate bisphenol"),
    Monomer("4,4'-dihydroxybenzophenone", PARA + "C(=O)" + PARA, "dihydroxybenzophenone",
            Scale.CATALOGUE, aromatic=True),
    Monomer("4,4'-thiodiphenol", PARA + "S" + PARA, "thiodiphenol", Scale.CATALOGUE,
            aromatic=True),
)

#: The activated dihaloarenes.  Every entry carries a carbonyl or a sulfone
#: *para to both leaving groups*, because that is the whole mechanism: SNAr goes
#: through a Meisenheimer complex, and without an electron-withdrawing group to
#: hold the negative charge there is no reaction at 300 C or at any other
#: temperature.  An unactivated dihaloarene is therefore never put in this
#: library - it is built as a probe instead, and refused.
DIHALOARENES: tuple[Monomer, ...] = (
    Monomer("4,4'-difluorobenzophenone", PARA + "C(=O)" + PARA, "ether ketone", Scale.BULK,
            aromatic=True),
    Monomer("4,4'-dichlorodiphenyl sulfone", PARA + "S(=O)(=O)" + PARA, "ether sulfone",
            Scale.BULK, aromatic=True),
    Monomer("1,4-bis(4-fluorobenzoyl)benzene", PARA + "C(=O)" + PARA + "C(=O)" + PARA,
            "ether diketone", Scale.CATALOGUE, aromatic=True),
)

#: The AB aryl ethers: one monomer carrying both the phenol and the activated
#: halide, so no stoichiometry to control.  Both are how the polymer is made in
#: practice - Victrex PES is the chlorophenol route, not the bisphenol route.
ARYL_ETHER_AB_MONOMERS: tuple[tuple[str, str, str, str, Availability], ...] = (
    ("4-fluoro-4'-hydroxybenzophenone", PARA + "C(=O)" + PARA, "poly(ether ketone)", "PEK",
     Availability.COMMERCIAL),
    ("4-chloro-4'-hydroxydiphenyl sulfone", PARA + "S(=O)(=O)" + PARA, "poly(ether sulfone)",
     "PES", Availability.COMMERCIAL),
)

#: Aryl ether polymers that are or have been sold, keyed by (bisphenol,
#: dihaloarene).  Everything else is constructible: the pair reacts by the same
#: SNAr its commercial neighbours do, and nobody appears to have run it.
COMMERCIAL_ARYL_ETHERS: dict[tuple[str, str], str] = {
    ("hydroquinone", "4,4'-difluorobenzophenone"): "PEEK",
    ("bisphenol A", "4,4'-dichlorodiphenyl sulfone"): "PSU (Udel)",
    ("4,4'-biphenol", "4,4'-dichlorodiphenyl sulfone"): "PPSU (Radel R)",
}

REPORTED_ARYL_ETHERS: dict[tuple[str, str], str] = {
    ("hydroquinone", "1,4-bis(4-fluorobenzoyl)benzene"): "poly(ether ether diketone)",
    ("4,4'-biphenol", "4,4'-difluorobenzophenone"): "PEEK-biphenol",
    ("resorcinol", "4,4'-difluorobenzophenone"): "meta-PEEK",
    ("bisphenol S", "4,4'-dichlorodiphenyl sulfone"): "all-sulfone polyarylether",
    ("bisphenol F", "4,4'-dichlorodiphenyl sulfone"): "bisphenol-F polysulfone",
    ("4,4'-dihydroxybenzophenone", "4,4'-difluorobenzophenone"): "PEKEKK backbone",
}

#: The 2,6-disubstituted phenols, which reach this family by a different
#: reaction entirely: oxidative coupling with a copper-amine catalyst and air,
#: not substitution.  The ortho positions have to be blocked or carbon-carbon
#: coupling competes with carbon-oxygen coupling and the product is a branched
#: mixture - that is the gate, and plain phenol is the probe for it.
PHENOLS: tuple[tuple[str, str, str, str, Availability, Scale, str], ...] = (
    ("2,6-dimethylphenol", "C", "poly(2,6-dimethyl-1,4-phenylene oxide)", "PPE/PPO",
     Availability.COMMERCIAL, Scale.BULK,
     "blended with polystyrene as Noryl; the homopolymer's melt viscosity is the reason "
     "it is almost never used neat"),
    ("2,6-diphenylphenol", "-c5ccccc5", "poly(2,6-diphenyl-1,4-phenylene oxide)", "Tenax TA",
     Availability.COMMERCIAL, Scale.CATALOGUE,
     "the gas-chromatography adsorbent; thermally stable to 400 C"),
    ("2,6-dichlorophenol", "Cl", "poly(2,6-dichloro-1,4-phenylene oxide)", "",
     Availability.REPORTED, Scale.CATALOGUE, ""),
)


# --------------------------------------------------------------------------
# Carbonates
# --------------------------------------------------------------------------

#: The aliphatic diols for the carbonate template.  Two of them - ethylene
#: glycol and 1,2-propanediol - are in the list *in order to be refused*: a
#: 1,2-diol and a carbonate source close to the five-membered cyclic carbonate,
#: and ethylene carbonate and propylene carbonate are sold by the tonne as
#: battery-electrolyte solvents rather than polymerised.  The polymers are real
#: and are emitted separately, from carbon dioxide and the epoxide, which is the
#: route that actually reaches them.
CARBONATE_DIOLS: tuple[Monomer, ...] = (
    Monomer("ethylene glycol", "CC", "ethylene", Scale.BULK),
    Monomer("1,2-propanediol", "CC(C)", "propylene", Scale.BULK),
    Monomer("1,3-propanediol", "CCC", "trimethylene", Scale.BULK),
    Monomer("1,4-butanediol", "CCCC", "tetramethylene", Scale.BULK),
    Monomer("1,5-pentanediol", "CCCCC", "pentamethylene", Scale.CATALOGUE),
    Monomer("1,6-hexanediol", "CCCCCC", "hexamethylene", Scale.BULK),
    Monomer("neopentyl glycol", "CC(C)(C)C", "neopentylene", Scale.BULK),
    Monomer("1,4-cyclohexanedimethanol", "CC5CCC(CC5)C", "cyclohexylenedimethylene",
            Scale.BULK),
    Monomer("diethylene glycol", "CCOCC", "diethylene", Scale.BULK),
)

#: Polycarbonates that are or have been sold.
COMMERCIAL_CARBONATES: dict[str, str] = {
    "bisphenol A": "PC (Lexan, Makrolon)",
    "bisphenol TMC": "APEC high-heat PC",
    "1,3-propanediol": "poly(trimethylene carbonate), the bioresorbable",
    "1,6-hexanediol": "polycarbonate diol for PU",
}

REPORTED_CARBONATES: dict[str, str] = {
    "bisphenol F": "bisphenol-F PC",
    "bisphenol Z": "bisphenol-Z PC",
    "bisphenol S": "bisphenol-S PC",
    "4,4'-biphenol": "biphenol PC",
    "1,4-butanediol": "poly(tetramethylene carbonate)",
    "diethylene glycol": "CR-39 precursor diol",
}

#: The two polycarbonates that exist and cannot be condensed: carbon dioxide
#: copolymerised with the epoxide, over a zinc or cobalt catalyst.  Poly(propylene
#: carbonate) is sold at kilotonne scale as a sacrificial binder and as a CO2-
#: derived polyol.  They carry a non-direct route, which is what exempts them
#: from the cyclic-carbonate gate - the same device ``vinyl.py`` uses for
#: poly(vinyl alcohol).
CO2_CARBONATES: tuple[tuple[str, str, str, Availability], ...] = (
    ("ethylene oxide", "CC", "poly(ethylene carbonate)", Availability.REPORTED),
    ("propylene oxide", "CC(C)", "poly(propylene carbonate)", Availability.COMMERCIAL),
)


# --------------------------------------------------------------------------
# Urethanes
# --------------------------------------------------------------------------

#: The diisocyanates.  All six are made at scale, all six by phosgenating the
#: corresponding diamine, which is what the feasibility expert charges for when
#: it sees a urethane linkage rather than inventing a monomer for it.
#:
#: The aromatic three (MDI, TDI, PPDI) give the stiffer, higher-melting hard
#: segment; the aliphatic three (HDI, IPDI, H12MDI) give the light-stable one,
#: which is why every outdoor polyurethane coating is aliphatic.  That split is
#: not visible in any property this engine predicts - photo-oxidation is not on
#: the panel - so it is written down here instead of being silently lost.
DIISOCYANATES: tuple[Monomer, ...] = (
    Monomer("MDI", PARA + "C" + PARA, "MDI", Scale.BULK, aromatic=True,
            note="4,4'-methylenediphenyl diisocyanate, the largest-volume diisocyanate"),
    Monomer("TDI", "c1ccc(C)c(c1)", "TDI", Scale.BULK, aromatic=True,
            note="toluene-2,4-diisocyanate"),
    Monomer("PPDI", PARA, "PPDI", Scale.CATALOGUE, aromatic=True,
            note="p-phenylene diisocyanate; the most symmetric aromatic hard segment there is"),
    Monomer("HDI", "CCCCCC", "HDI", Scale.BULK,
            note="hexamethylene diisocyanate; light-stable, so it is the coatings one"),
    Monomer("IPDI", "CC5(C)CC(CC(C)(C)C5)", "IPDI", Scale.BULK,
            note="isophorone diisocyanate, from isophorone diamine"),
    Monomer("H12MDI", "C5CCC(CC5)CC5CCC(CC5)", "H12MDI", Scale.BULK,
            note="the hydrogenated MDI; aliphatic and still symmetric"),
)

#: The chain extenders.  These are short diols, and short is the point: the hard
#: segment is short and rigid so that it phase-separates from the soft segment
#: and acts as a physical crosslink.  Bisphenol A is in the list only to be
#: refused - an aryl isocyanate plus a phenol gives an aryl carbamate, which is
#: what a *blocked* isocyanate is, and it comes apart again around 120 C.
CHAIN_EXTENDERS: tuple[Monomer, ...] = (
    Monomer("ethylene glycol", "CC", "ethylene glycol", Scale.BULK),
    Monomer("1,4-butanediol", "CCCC", "1,4-butanediol", Scale.BULK,
            note="the standard TPU chain extender"),
    Monomer("1,6-hexanediol", "CCCCCC", "1,6-hexanediol", Scale.BULK),
    Monomer("diethylene glycol", "CCOCC", "diethylene glycol", Scale.BULK),
    Monomer("neopentyl glycol", "CC(C)(C)C", "neopentyl glycol", Scale.BULK),
    Monomer("1,4-cyclohexanedimethanol", "CC5CCC(CC5)C", "CHDM", Scale.BULK),
    Monomer("HQEE", "CCO" + PARA + "OCC", "HQEE", Scale.CATALOGUE,
            note="hydroquinone bis(2-hydroxyethyl) ether, the high-performance extender"),
    Monomer("bisphenol A", PARA + "C(C)(C)" + PARA, "bisphenol A", Scale.BULK, aromatic=True),
)

#: Hard segments that are the real thing rather than a combination.
COMMERCIAL_URETHANES: dict[tuple[str, str], str] = {
    ("MDI", "1,4-butanediol"): "the standard TPU hard segment",
    ("MDI", "ethylene glycol"): "MDI/EG hard segment",
    ("PPDI", "1,4-butanediol"): "PPDI/BDO, the high-resilience TPU",
    ("HDI", "1,4-butanediol"): "aliphatic TPU hard segment",
    ("MDI", "HQEE"): "MDI/HQEE, the high-temperature cast elastomer",
}


# --------------------------------------------------------------------------
# Siloxanes
# --------------------------------------------------------------------------

#: The silicones.  Silicon is outside :data:`ELEMENT_BUDGET` and these are
#: emitted anyway, flagged, because the alternative is a library with no
#: silicone in it at all.  Poly(dimethylsiloxane) has the lowest surface energy
#: and the lowest Tg (-123 C) of any commercial polymer and there is no carbon-
#: backbone substitute for it, so refusing it would not be conservative, it
#: would be a hole where the answer is.
#:
#: The engine's own two opinions about them are worth stating before anyone
#: reads a number: ``polymer.py`` records that its packing model puts PDMS at
#: 1.13 g/cm3 against a measured 0.97 and refuses the prediction on element
#: grounds, and ``polymer_feasibility.py`` records that the Ertl fragment
#: statistics barely contain silicon, which is why PDMS scores like a research
#: compound.  Both refusals are correct and both are visible in the label.
#:
#: None of these is a melt-processed thermoplastic: a silicone is a fluid until
#: it is crosslinked and a cured elastomer afterwards.
SILOXANES: tuple[tuple[str, str, str, str, Availability, Scale, str], ...] = (
    ("dimethyldichlorosilane", "(C)(C)", "poly(dimethylsiloxane)", "PDMS, silicone",
     Availability.COMMERCIAL, Scale.BULK,
     "the lowest surface energy and the lowest Tg in commercial use"),
    ("methylphenyldichlorosilane", "(C)(c1ccccc1)", "poly(methylphenylsiloxane)", "PMPS",
     Availability.COMMERCIAL, Scale.BULK,
     "the phenyl raises the refractive index and the thermal stability"),
    ("diphenyldichlorosilane", "(c1ccccc1)(c1ccccc1)", "poly(diphenylsiloxane)", "",
     Availability.REPORTED, Scale.CATALOGUE,
     "crystalline and infusible as the homopolymer; used only as a comonomer"),
    ("methylvinyldichlorosilane", "(C)(C=C)", "poly(methylvinylsiloxane)", "VMQ cure site",
     Availability.COMMERCIAL, Scale.BULK,
     "the vinyl is a hydrosilylation cure site, not a propagating group: it survives the "
     "siloxane equilibration that builds the chain"),
    ("methyldichlorosilane", "(C)([H])", "poly(methylhydrosiloxane)", "PMHS",
     Availability.COMMERCIAL, Scale.BULK,
     "sold as a crosslinker and as a mild reducing agent"),
    ("(3,3,3-trifluoropropyl)methyldichlorosilane", "(C)(CCC(F)(F)F)",
     "poly(3,3,3-trifluoropropylmethylsiloxane)", "FVMQ",
     Availability.COMMERCIAL, Scale.CATALOGUE,
     "the fuel-resistant silicone"),
    ("diethyldichlorosilane", "(CC)(CC)", "poly(diethylsiloxane)", "",
     Availability.REPORTED, Scale.CATALOGUE, ""),
)


# --------------------------------------------------------------------------
# Sulfides
# --------------------------------------------------------------------------

#: The thioethers.  Sulfur is outside :data:`ELEMENT_BUDGET` for the same
#: reason silicon is and is handled the same way.
#:
#: Two quite different chemistries are here.  Thiirane and its 2-substituted
#: relatives ring-open exactly as the epoxides do, only faster - sulfur is the
#: better nucleophile and the worse leaving group, so the propagation is cleaner
#: and the polymers are higher molar mass.  Poly(phenylene sulfide) is the other
#: one: sodium sulfide against p-dichlorobenzene in N-methylpyrrolidone, a
#: chemistry the feasibility expert's route table does not contain, which is why
#: it and it alone in this module carries ``no feasibility route``.
#:
#: The liquid polysulfides - the Thiokols - keep their sulfur-sulfur bond, and
#: it is load-bearing: the S-S is what lets the sealant be cured by an oxidant
#: and what makes it self-healing under stress.  The backbone gate below refuses
#: an O-O and an N-N and deliberately does not refuse an S-S, because a peroxide
#: is a radical initiator and a disulfide is a commercial aerospace sealant.
SULFIDES: tuple[HeteroUnit, ...] = (
    HeteroUnit(
        name="poly(ethylene sulfide)",
        smiles=THIIRANE.format(r=""),
        subfamily="polysulfide",
        monomers=("ethylene sulfide (thiirane)",),
        availability=Availability.REPORTED,
        scale=Scale.CATALOGUE,
        melt=Melt.MELT,
        note="crystalline, melting near 210 C; thiirane opens faster than oxirane does",
    ),
    HeteroUnit(
        name="poly(propylene sulfide)",
        smiles=THIIRANE.format(r="(C)"),
        subfamily="polysulfide",
        monomers=("propylene sulfide",),
        availability=Availability.REPORTED,
        scale=Scale.CATALOGUE,
        melt=Melt.MELT,
        note="the hydrophobic, oxidation-responsive block in PEG-PPS copolymers",
    ),
    HeteroUnit(
        name="poly(p-phenylene sulfide)",
        smiles="[*]S" + PARA + "[*]",
        subfamily="aryl polysulfide",
        monomers=("p-dichlorobenzene", "sodium sulfide"),
        availability=Availability.COMMERCIAL,
        scale=Scale.BULK,
        melt=Melt.MELT,
        common_name="PPS (Ryton)",
        note="melts at 285 C and is still melt processable; made in N-methylpyrrolidone by a "
             "chemistry the feasibility expert's route table does not contain",
    ),
    HeteroUnit(
        name="poly(ethylene disulfide)",
        smiles="[*]CCSS[*]",
        subfamily="polysulfide",
        monomers=("1,2-dichloroethane", "sodium disulfide"),
        availability=Availability.COMMERCIAL,
        scale=Scale.CATALOGUE,
        melt=Melt.ELASTOMER,
        common_name="Thiokol A",
        note="the first synthetic rubber sold in the United States; the S-S bond is the cure "
             "site, not a defect",
    ),
    HeteroUnit(
        name="poly(oxydiethylene formal disulfide)",
        smiles="[*]CCOCOCCSS[*]",
        subfamily="polysulfide",
        monomers=("bis(2-chloroethyl) formal", "sodium disulfide"),
        availability=Availability.COMMERCIAL,
        scale=Scale.CATALOGUE,
        melt=Melt.ELASTOMER,
        common_name="Thiokol LP",
        note="the liquid polysulfide in aircraft fuel-tank and insulating-glass sealants",
    ),
)


# --------------------------------------------------------------------------
# Imides
# --------------------------------------------------------------------------

#: The dianhydrides, each as the diimide it becomes, with ``{amine}`` marking
#: where the diamine hangs off the second imide nitrogen.  Ring-closure digits
#: run from 5 upward so that a diamine core is free to use 1 without colliding
#: with an open ring - which is a SMILES bookkeeping point and also the reason
#: these are written as whole templates rather than as connector cores: an imide
#: nitrogen is *inside* a ring, so there is nothing to thread a linear core
#: through.
#:
#: ``flexible`` records whether the monomer carries a kink or a swivel - an
#: ether, an isopropylidene, a meta link.  Two rigid halves give a chain that
#: reaches its decomposition temperature before it flows; one flexible half at
#: each end is what makes a polyimide a thermoplastic, and it is the whole
#: difference between Kapton, which is cast as the poly(amic acid) and imidised
#: in place, and Ultem, which is injection moulded.
DIANHYDRIDES: tuple[tuple[str, str, str, Scale, bool], ...] = (
    ("pyromellitic dianhydride", "PMDA",
     "[*]N5C(=O)c6cc7c(cc6C5=O)C(=O)N({amine})C7=O", Scale.BULK, False),
    ("3,3',4,4'-biphenyltetracarboxylic dianhydride", "BPDA",
     "[*]N5C(=O)c6ccc(cc6C5=O)-c6ccc7c(c6)C(=O)N({amine})C7=O", Scale.CATALOGUE, False),
    ("3,3',4,4'-benzophenonetetracarboxylic dianhydride", "BTDA",
     "[*]N5C(=O)c6ccc(cc6C5=O)C(=O)c6ccc7c(c6)C(=O)N({amine})C7=O", Scale.CATALOGUE, False),
    ("4,4'-oxydiphthalic anhydride", "ODPA",
     "[*]N5C(=O)c6ccc(cc6C5=O)Oc6ccc7c(c6)C(=O)N({amine})C7=O", Scale.CATALOGUE, True),
    ("4,4'-(hexafluoroisopropylidene)diphthalic anhydride", "6FDA",
     "[*]N5C(=O)c6ccc(cc6C5=O)C(C(F)(F)F)(C(F)(F)F)c6ccc7c(c6)C(=O)N({amine})C7=O",
     Scale.CATALOGUE, True),
    ("bisphenol A dianhydride", "BPADA",
     "[*]N5C(=O)c6ccc(Oc7ccc(cc7)C(C)(C)c7ccc(Oc8ccc9c(c8)C(=O)N({amine})C9=O)cc7)cc6C5=O",
     Scale.CATALOGUE, True),
)

#: The diamines.  All aromatic, which is a restriction rather than an oversight:
#: an aliphatic diamine and a dianhydride give a polyimide whose amic-acid
#: intermediate forms a salt instead of cyclising, and the aliphatic polyimides
#: that have been made are all made by a different route.  The whole point of
#: this sub-family is the top of the temperature range, and every material at
#: that end of it is wholly aromatic.
IMIDE_DIAMINES: tuple[Monomer, ...] = (
    Monomer("p-phenylenediamine", PARA, "PPD", Scale.BULK, aromatic=True),
    Monomer("m-phenylenediamine", META, "MPD", Scale.BULK, aromatic=True),
    Monomer("4,4'-oxydianiline", PARA + "O" + PARA, "ODA", Scale.BULK, aromatic=True),
    Monomer("4,4'-methylenedianiline", PARA + "C" + PARA, "MDA", Scale.BULK, aromatic=True),
    Monomer("4,4'-diaminodiphenyl sulfone", PARA + "S(=O)(=O)" + PARA, "DDS", Scale.BULK,
            aromatic=True),
    Monomer("BAPP", PARA + "O" + PARA + "C(C)(C)" + PARA + "O" + PARA, "BAPP",
            Scale.CATALOGUE, aromatic=True,
            note="2,2-bis[4-(4-aminophenoxy)phenyl]propane, the thermoplastic-polyimide diamine"),
)

#: Diamines that put a kink or a swivel in the chain.  p-Phenylenediamine is the
#: only one that does not, and a PMDA/PPD or BPDA/PPD polyimide is the most
#: intractable polymer in this module: it does not melt, does not dissolve, and
#: is handled only as the poly(amic acid).
_FLEXIBLE_DIAMINES = frozenset({"MPD", "ODA", "MDA", "DDS", "BAPP"})

#: Polyimides that are or have been sold, keyed by (dianhydride code, diamine
#: code).  These are the highest-Tg commercial polymers there are, and the
#: reason the family is worth the modules it costs.
COMMERCIAL_IMIDES: dict[tuple[str, str], str] = {
    ("PMDA", "ODA"): "Kapton",
    ("BPDA", "PPD"): "Upilex-S",
    ("BPDA", "ODA"): "Upilex-R",
    ("BPADA", "MPD"): "Ultem 1000 (polyetherimide)",
    ("BTDA", "MDA"): "Larc-TPI / thermoset polyimide matrix",
}

REPORTED_IMIDES: dict[tuple[str, str], str] = {
    ("PMDA", "PPD"): "the all-para polyimide; infusible and insoluble",
    ("6FDA", "PPD"): "a gas-separation membrane polyimide",
    ("6FDA", "ODA"): "a gas-separation membrane polyimide",
    ("ODPA", "PPD"): "Aurum-type thermoplastic polyimide",
    ("BPADA", "BAPP"): "a soluble polyetherimide",
}


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------
#
# A grammar is defined by what it refuses.  Every gate below names the reaction
# that happens *instead* of polymerisation, and every one has at least one probe
# at the bottom of this module that it is required to refuse, so the gates are
# exercised rather than asserted.

#: Backbone bonds between two heteroatoms that are not a polymer linkage.  An
#: O-O is a peroxide, which is a radical initiator rather than a chain; an N-N
#: is a hydrazine, an N-O a hydroxylamine, an O-S a sulfenate, and all of them
#: are redox chemistry rather than polymerisation.
#:
#: Two bonds are deliberately absent.  **S-S stays**, because the Thiokols are
#: commercial sealants whose disulfide is the cure site.  **Si-O stays**,
#: because it is the backbone of every silicone.  A gate written as "two
#: heteroatoms in a row" would have taken out both, which is how a plausible
#: rule deletes two real families.
FORBIDDEN_BACKBONE_BONDS = frozenset(
    {("N", "N"), ("N", "O"), ("N", "S"), ("O", "O"), ("O", "S")}
)

#: Monomers that are real compounds, look constructible, and have no
#: homopolymer, keyed by the repeat unit that would be written for them.  A
#: curated refusal is still a gate: what stops these is a free energy or a
#: cation stability, not a bond count, so no structural rule reaches them.
#:
#: Both entries are places where this module is *stricter than the engine*.
#: ``PolymerFeasibilityExpert`` offers a five-membered ring-opening route for
#: each and scores it, because its ring-size rule is about size and these two
#: fail on something else.
_NO_HOMOPOLYMER: dict[str, str] = {
    "[*]C(C)CCCO[*]": (
        "2-methyltetrahydrofuran does not homopolymerise: the oxocarbenium the cationic "
        "propagation goes through is stabilised by the methyl and stops handing the chain "
        "on. It is sold as a polymerisation solvent, which is the opposite of a monomer"
    ),
    "[*]CCCCS[*]": (
        "tetrahydrothiophene is an unstrained five-membered ring whose sulfur is a better "
        "nucleophile than the propagating sulfonium is an electrophile, so it caps chains "
        "instead of joining them; it is a stable solvent and a gas odorant"
    ),
}

#: SMARTS for a side group that polymerises in competition with the backbone,
#: with the reason.  An acrylic ester or a styrenic vinyl on a ring-opening
#: monomer gives a network, not a chain: the anionic or cationic initiator adds
#: to the double bond as readily as it opens the ring.
#:
#: An allyl ether is not here and must not be: allyl groups transfer rather than
#: propagate (Odian 3-9c), which is exactly why allyl glycidyl ether is sold as
#: a *latent* cure site and its homopolymer is linear.  A vinyl on silicon is
#: not here either, for the same reason: VMQ silicone is made with it in place.
_PENDANT_POLYMERISABLE: tuple[tuple[str, str], ...] = (
    ("[CX3]=[CX3][CX3](=[OX1])[OX2]", "an acrylic or methacrylic ester on the side group "
                                      "polymerises through the double bond under the same "
                                      "initiator that opens the ring, so the product is a "
                                      "network rather than a linear chain"),
    ("[c][CX3]=[CX3H2]", "a styrenic vinyl on the side group is a second polymerisable "
                         "group and makes the monomer a crosslinker"),
)


def _backbone_termini(mol: Any) -> tuple[int, ...] | None:
    """The chain of heavy atoms between the two attachment points."""
    Chem = _rdkit()
    dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
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


def _exocyclic_carbonyl(mol: Any, idx: int) -> bool:
    """Does this atom carry a double bond to oxygen?"""
    Chem = _rdkit()
    atom = mol.GetAtomWithIdx(idx)
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if bond.GetBondType() is Chem.BondType.DOUBLE and other.GetAtomicNum() == 8:
            return True
    return False


def _gate_structure(unit: HeteroUnit, mol: Any) -> tuple[str, str] | None:
    """Two attachment points, a backbone between them, and a cappable valence."""
    Chem = _rdkit()
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        return "structure", (
            f"a chain repeat unit needs exactly two [*]; this one has {len(dummies)}"
        )
    if _backbone_termini(mol) is None:
        return "structure", "the two attachment points are not joined by a path"
    if Chem.MolFromSmiles(unit.smiles.replace("[*]", "[H]")) is None:
        return "structure", "the unit does not parse once the attachment points are capped"
    return None


def _gate_free_hydroxyl(unit: HeteroUnit, mol: Any) -> tuple[str, str] | None:
    """A spare hydroxyl or primary amine makes the monomer a branch point."""
    for atom in mol.GetAtoms():
        spare = (
            (atom.GetAtomicNum() == 8 and atom.GetTotalNumHs() > 0)
            or (atom.GetAtomicNum() == 7 and atom.GetTotalNumHs() > 1)
        )
        if not spare:
            continue
        return "free-hydroxyl", (
            "the repeat unit carries a hydroxyl or a primary amine the backbone does not "
            "use, so the monomer is AB2 rather than AB: in a ring-opening polymerisation "
            "that group is an initiating site and the product is hyperbranched, and in a "
            "step growth it is a crosslink. Glycidol gives hyperbranched polyglycerol and "
            "3-ethyl-3-(hydroxymethyl)oxetane gives a hyperbranched polyether; neither "
            "gives the linear chain this repeat unit claims"
        )
    return None


def _gate_backbone_heteroatom_pair(unit: HeteroUnit, mol: Any) -> tuple[str, str] | None:
    """Two heteroatoms bonded to each other in the backbone, S-S and Si-O excepted."""
    path = _backbone_termini(mol)
    if path is None:
        return None
    for first, second in zip(path, path[1:]):
        pair = tuple(
            sorted((mol.GetAtomWithIdx(first).GetSymbol(), mol.GetAtomWithIdx(second).GetSymbol()))
        )
        if pair in FORBIDDEN_BACKBONE_BONDS:
            return "backbone-bond", (
                f"the backbone contains a {pair[0]}-{pair[1]} bond, which is a peroxide, a "
                "hydrazine, a hydroxylamine or a sulfenate depending on which it is; all of "
                "them are redox reagents rather than polymer linkages, and a polymeric "
                "peroxide in particular is an initiator that decomposes on warming. The "
                "disulfide of the Thiokols and the siloxane bond of the silicones are "
                "deliberately not in this list"
            )
    return None


def _gate_unstrained_ring(unit: HeteroUnit, mol: Any) -> tuple[str, str] | None:
    """A six-atom saturated backbone closes to a chair with nothing to gain."""
    path = _backbone_termini(mol)
    if path is None or len(path) != 6:
        return None
    Chem = _rdkit()
    symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in path]
    if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in path):
        return None
    if any(s not in ("C", "O", "S") for s in symbols):
        return None
    if all(s == "C" for s in symbols):
        return None  # cyclohexane is not a monomer, but neither is this a heteroatom chain
    if any(_exocyclic_carbonyl(mol, i) for i in path):
        return None  # a lactone or a carbonate: the carbonyl is what pays for the ring
    for first, second in zip(path, path[1:]):
        bond = mol.GetBondBetweenAtoms(first, second)
        if bond is None or bond.GetBondType() is not Chem.BondType.SINGLE:
            return None
    return "unstrained-ring", (
        "the monomer this repeat unit implies is an unstrained six-membered ring, and a "
        "six-ring with no carbonyl has no favourable free energy of polymerisation: the "
        "chair is as relaxed as the chain it would make, so the equilibrium sits on the "
        "monomer (Odian ch. 7). Tetrahydrofuran polymerises and tetrahydropyran does not; "
        "1,4-dioxane and thiane do not either, and all three sit in bottles as solvents"
    )


def _gate_cyclic_carbonate(unit: HeteroUnit, mol: Any) -> tuple[str, str] | None:
    """A 1,2-diol carbonate closes to the five-ring, which is a solvent."""
    if unit.route != "direct":
        return None
    path = _backbone_termini(mol)
    if path is None or len(path) != 5:
        return None
    on_path = set(path)
    for idx in path:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() != 6 or not _exocyclic_carbonyl(mol, idx):
            continue
        oxygens = [
            n for n in atom.GetNeighbors() if n.GetAtomicNum() == 8 and n.GetIdx() in on_path
        ]
        if len(oxygens) == 2:
            return "cyclic-carbonate", (
                "the two carbonate oxygens are two carbons apart, so what a diol and "
                "phosgene or a diaryl carbonate give here is the five-membered cyclic "
                "carbonate, not a chain: ethylene carbonate and propylene carbonate are "
                "made at hundreds of kilotonnes a year and sold as battery-electrolyte "
                "solvents precisely because that ring is stable and will not open. The "
                "polymers are real and are reached from carbon dioxide and the epoxide "
                "instead, which is the route this module emits them under"
            )
    return None


def _gate_aryl_carbamate(unit: HeteroUnit, mol: Any) -> tuple[str, str] | None:
    """A urethane built from a phenol is a blocked isocyanate, not a polymer."""
    path = _backbone_termini(mol)
    if path is None:
        return None
    on_path = set(path)
    for idx in path:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() != 6 or not _exocyclic_carbonyl(mol, idx):
            continue
        nitrogens = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 7]
        oxygens = [
            n
            for n in atom.GetNeighbors()
            if n.GetAtomicNum() == 8 and n.GetIdx() in on_path and n.GetTotalDegree() == 2
        ]
        if not nitrogens or not oxygens:
            continue
        for oxygen in oxygens:
            far = [n for n in oxygen.GetNeighbors() if n.GetIdx() != idx]
            if far and far[0].GetIsAromatic():
                return "aryl-carbamate", (
                    "the urethane oxygen sits on an aromatic ring, so the linkage is an "
                    "aryl carbamate made from an isocyanate and a phenol. That is the "
                    "definition of a *blocked* isocyanate: phenol-blocked isocyanates are "
                    "sold as one-component coatings precisely because the bond comes apart "
                    "again around 120-160 C and gives the isocyanate back. A polymer whose "
                    "backbone unzips below its processing temperature is not a polymer"
                )
    return None


def _gate_pendant_polymerisable(unit: HeteroUnit, mol: Any) -> tuple[str, str] | None:
    Chem = _rdkit()
    for smarts, reason in _PENDANT_POLYMERISABLE:
        query = Chem.MolFromSmarts(smarts)
        if query is not None and mol.HasSubstructMatch(query):
            return "pendant-polymerisable", reason
    return None


def _ring_is_activated(mol: Any, idx: int) -> bool:
    """Does the aromatic ring holding ``idx`` carry an electron-withdrawing group?

    This is the Meisenheimer question and nothing else: a carbonyl, a sulfone, a
    nitro or a nitrile somewhere on the ring is what holds the negative charge
    while the leaving group leaves.
    """
    Chem = _rdkit()
    # Written over atomic numbers rather than over C/S/N, because RDKit perceives
    # the pyromellitimide ring as aromatic and an aromatic imide carbonyl is
    # still a carbonyl: [CX3]=O does not match it and Kapton was refused.
    query = Chem.MolFromSmarts(
        "[$([#6X3]=[OX1]),$([#16X4](=[OX1])=[OX1]),$([#7X3](=[OX1])=[OX1]),$([#6X2]#[#7X1])]"
    )
    if query is None:  # pragma: no cover
        return False
    withdrawing = {m[0] for m in mol.GetSubstructMatches(query)}
    if not withdrawing:
        return False
    for ring in mol.GetRingInfo().AtomRings():
        if idx not in ring:
            continue
        for member in ring:
            for neighbour in mol.GetAtomWithIdx(member).GetNeighbors():
                if neighbour.GetIdx() in withdrawing:
                    return True
    return False


def _gate_aryl_ether(unit: HeteroUnit, mol: Any) -> tuple[str, str] | None:
    """An aryl ether backbone has exactly two routes, and needs to qualify for one.

    Known limit: this looks for an activating group *anywhere in the repeat
    unit* rather than on the ring that carried the leaving group, because the
    repeat unit no longer records which half was the halide.  A bisphenol that
    arrives already carrying a sulfone would therefore slip past it when paired
    with an unactivated dihaloarene - which is why :data:`DIHALOARENES` contains
    no unactivated member for it to be paired with.
    """
    path = _backbone_termini(mol)
    if path is None:
        return None
    on_path = set(path)
    ethers = [
        idx
        for idx in path
        if mol.GetAtomWithIdx(idx).GetSymbol() == "O"
        and not mol.GetAtomWithIdx(idx).GetIsAromatic()
        and any(n.GetIsAromatic() for n in mol.GetAtomWithIdx(idx).GetNeighbors())
        and all(
            n.GetAtomicNum() in (0, 6) and (n.GetIsAromatic() or n.GetIdx() not in on_path)
            for n in mol.GetAtomWithIdx(idx).GetNeighbors()
        )
    ]
    if not ethers:
        return None
    if any(_ring_is_activated(mol, idx) for idx in path):
        return None  # SNAr: PEEK, the polysulfones, the polyetherimides

    # Not activated anywhere, so the only route left is oxidative coupling of a
    # phenol, and that needs the unit to *be* a phenylene oxide with both ortho
    # positions blocked.
    aromatic = [i for i in path if mol.GetAtomWithIdx(i).GetIsAromatic()]
    oxygens = [i for i in path if mol.GetAtomWithIdx(i).GetSymbol() == "O"]
    blocked = False
    if len(oxygens) == 1 and len(aromatic) == len(path) - 1:
        ipso = [n for n in mol.GetAtomWithIdx(oxygens[0]).GetNeighbors() if n.GetIsAromatic()]
        if ipso:
            ring = next(
                (r for r in mol.GetRingInfo().AtomRings() if ipso[0].GetIdx() in r), ()
            )
            # Both ends of the ring are candidates, not just the one carrying the
            # oxygen written inside this repeat unit.  A periodic chain can be cut
            # at either C-O bond, and the bundled poly(2,6-dimethylphenylene
            # oxide) is cut at the other one: its methyls flank the [*] carbon
            # rather than the O carbon, and they are the same two methyls.
            candidates = [
                idx
                for idx in ring
                if any(
                    n.GetIdx() not in ring
                    and (n.GetAtomicNum() == 0 or n.GetIdx() == oxygens[0])
                    for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                )
            ]
            for candidate in candidates:
                ortho = [n for n in mol.GetAtomWithIdx(candidate).GetNeighbors()
                         if n.GetIdx() in ring]
                if len(ortho) == 2 and all(
                    any(x.GetIdx() not in ring and x.GetAtomicNum() != 1
                        for x in a.GetNeighbors())
                    for a in ortho
                ):
                    blocked = True
                    break
    if blocked:
        return None
    return "aryl-ether", (
        "an aryl ether backbone is reached by exactly two reactions and this one qualifies "
        "for neither. Nucleophilic aromatic substitution needs an electron-withdrawing group "
        "to stabilise the Meisenheimer complex, and there is none in the repeat unit: a "
        "bisphenolate and an unactivated dihaloarene do not react at 300 C or at any other "
        "temperature. Oxidative coupling of a phenol needs both ortho positions blocked, or "
        "carbon-carbon coupling competes with carbon-oxygen coupling and the product is "
        "branched and mixed rather than the linear polyether - which is why the commercial "
        "polymer is made from 2,6-dimethylphenol and not from phenol"
    )


#: Every gate, in the order a chemist would apply them.
_GATES = (
    _gate_structure,
    _gate_free_hydroxyl,
    _gate_backbone_heteroatom_pair,
    _gate_pendant_polymerisable,
    _gate_unstrained_ring,
    _gate_cyclic_carbonate,
    _gate_aryl_carbamate,
    _gate_aryl_ether,
)


@functools.lru_cache(maxsize=1)
def _no_homopolymer() -> dict[str, str]:
    out: dict[str, str] = {}
    for smiles, reason in _NO_HOMOPOLYMER.items():
        key = canonical(smiles)
        if key is None:  # pragma: no cover - a test parses the table
            raise ValueError(f"the refusal table entry {smiles!r} does not parse")
        out[key] = reason
    return out


def gate(unit: HeteroUnit) -> Refusal | None:
    """Refuse a proposal, or return None if the grammar will emit it."""
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(unit.smiles)
    if mol is None:
        return Refusal(unit.name, unit.smiles, "structure", "the repeat unit does not parse")
    reason = _no_homopolymer().get(canonical(unit.smiles) or unit.smiles)
    if reason is not None:
        return Refusal(unit.name, unit.smiles, "no-homopolymer", reason)
    for check in _GATES:
        verdict = check(unit, mol)
        if verdict is not None:
            return Refusal(unit.name, unit.smiles, verdict[0], verdict[1])
    return None


# --------------------------------------------------------------------------
# Probes: structures built here only so that the gates are exercised
# --------------------------------------------------------------------------


def _probe(name: str, smiles: str, monomer: str, note: str = "") -> HeteroUnit:
    return HeteroUnit(
        name=name,
        smiles=smiles,
        subfamily="probe",
        monomers=(monomer,),
        availability=Availability.CONSTRUCTIBLE,
        scale=Scale.CATALOGUE,
        melt=Melt.MELT,
        note=note,
    )


#: Tempting structures the grammar must refuse.  They live in the module rather
#: than in the test file because a gate with no counterexample beside it is a
#: claim, and because the list doubles as the argument for what is absent: every
#: one of these is a repeat unit a grammar for this family would otherwise emit,
#: and every one of the monomers named is a real chemical sold in a bottle.
PROBES: tuple[HeteroUnit, ...] = (
    _probe("poly(1,4-dioxane)", "[*]OCCOCC[*]", "1,4-dioxane",
           "the standard counterexample to ring-opening: THF polymerises, THP and "
           "dioxane do not"),
    _probe("poly(thiane)", "[*]CCCCCS[*]", "thiane (pentamethylene sulfide)"),
    _probe("poly(2-methyltetrahydrofuran)", "[*]C(C)CCCO[*]", "2-methyltetrahydrofuran"),
    _probe("poly(tetrahydrothiophene)", "[*]CCCCS[*]", "tetrahydrothiophene"),
    _probe("polyglycidol", "[*]CC(CO)O[*]", "glycidol",
           "polymerises, and to a hyperbranched polyglycerol rather than to this chain"),
    _probe("poly(3-ethyl-3-(hydroxymethyl)oxetane)", "[*]CC(CC)(CO)CO[*]",
           "3-ethyl-3-(hydroxymethyl)oxetane"),
    _probe("poly(glycidyl methacrylate) through the epoxide",
           "[*]CC(COC(=O)C(C)=C)O[*]", "glycidyl methacrylate"),
    _probe("poly(4-vinylphenyl glycidyl ether)", "[*]CC(COc1ccc(C=C)cc1)O[*]",
           "4-vinylphenyl glycidyl ether"),
    _probe("poly(ethylene peroxide)", "[*]CCOO[*]", "ethylene ozonide route"),
    _probe("poly(ethylenehydrazine)", "[*]CCNNCC[*]", "1,2-diaminoethane plus hydrazine"),
    _probe("poly(1,2-butylene carbonate) by condensation", "[*]OC(=O)OCC(CC)[*]",
           "1,2-butanediol plus diphenyl carbonate"),
    _probe("poly(1,4-phenylene oxide)", "[*]Oc1ccc(cc1)[*]", "phenol"),
    _probe("poly(2-methyl-1,4-phenylene oxide)", "[*]Oc1ccc([*])cc1C", "o-cresol"),
    _probe(
        "poly(bisphenol A-co-4,4'-biphenylene ether)",
        ARYL_ETHER.format(bisphenol=PARA + "C(C)(C)" + PARA, activated=PARA + PARA),
        "bisphenol A plus 4,4'-dichlorobiphenyl",
    ),
    _probe(
        "poly(hydroquinone-co-1,4-phenylene ether)",
        ARYL_ETHER.format(bisphenol=PARA, activated=PARA),
        "hydroquinone plus p-dichlorobenzene",
    ),
)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def _scale(*scales: Scale) -> Scale:
    return Scale.CATALOGUE if Scale.CATALOGUE in scales else Scale.BULK


def _polyethers() -> Iterator[HeteroUnit]:
    for monomer, substituent, polymer, availability, scale, note in EPOXIDES:
        yield HeteroUnit(
            name=polymer,
            smiles=EPOXIDE.format(r=substituent),
            subfamily="polyether",
            monomers=(monomer,),
            availability=availability,
            scale=scale,
            melt=Melt.MELT,
            note=note,
        )
    yield CYCLOHEXENE_OXIDE
    for monomer, substituent, polymer, availability, scale, note in OXETANES:
        yield HeteroUnit(
            name=polymer,
            smiles=OXETANE.format(r=substituent),
            subfamily="polyether",
            monomers=(monomer,),
            availability=availability,
            scale=scale,
            melt=Melt.MELT,
            note=note,
        )
    for monomer, core, polymer, availability, scale, note in CYCLIC_ETHERS:
        yield HeteroUnit(
            name=polymer,
            smiles=CYCLIC_ETHER.format(core=core),
            # A second oxygen inside the ring makes it a cyclic acetal rather
            # than a cyclic ether, and the chain an acetal rather than an ether.
            subfamily="polyacetal" if "O" in core else "polyether",
            monomers=(monomer,),
            availability=availability,
            scale=scale,
            melt=Melt.MELT,
            note=note,
        )
    # Tetrahydropyran is not in CYCLIC_ETHERS and its absence is the argument,
    # so it is built here and refused: a list that silently stops at five
    # members teaches nobody why the sixth is missing.
    yield _probe("poly(pentamethylene oxide)", CYCLIC_ETHER.format(core="CCCCC"),
                 "tetrahydropyran")
    yield POLYOXYMETHYLENE
    yield POLYCAPROLACTONE


def _aryl_ethers() -> Iterator[HeteroUnit]:
    for monomer, substituent, polymer, common, availability, scale, note in PHENOLS:
        yield HeteroUnit(
            name=polymer,
            smiles=PHENYLENE_OXIDE.format(r=substituent),
            subfamily="aryl polyether",
            monomers=(monomer, "oxygen"),
            availability=availability,
            scale=scale,
            melt=Melt.MELT,
            common_name=common,
            note=note or "oxidative coupling with a copper-amine catalyst and air",
        )
    for bisphenol in BISPHENOLS:
        for halide in DIHALOARENES:
            key = (bisphenol.name, halide.name)
            if key in COMMERCIAL_ARYL_ETHERS:
                availability = Availability.COMMERCIAL
                common = COMMERCIAL_ARYL_ETHERS[key]
            elif key in REPORTED_ARYL_ETHERS:
                availability = Availability.REPORTED
                common = REPORTED_ARYL_ETHERS[key]
            else:
                availability = Availability.CONSTRUCTIBLE
                common = ""
            yield HeteroUnit(
                name=f"poly({bisphenol.short} {halide.short})",
                smiles=ARYL_ETHER.format(bisphenol=bisphenol.core, activated=halide.core),
                subfamily="aryl polyether",
                monomers=(bisphenol.name, halide.name),
                availability=availability,
                scale=_scale(bisphenol.scale, halide.scale),
                melt=Melt.MELT,
                common_name=common,
                note="nucleophilic aromatic substitution in a dipolar aprotic solvent near "
                     "300 C; these are the melt-processable high-temperature thermoplastics",
            )
    for monomer, core, polymer, common, availability in ARYL_ETHER_AB_MONOMERS:
        yield HeteroUnit(
            name=polymer,
            smiles=ARYL_ETHER_AB.format(activated=core),
            subfamily="aryl polyether",
            monomers=(monomer,),
            availability=availability,
            scale=Scale.CATALOGUE,
            melt=Melt.MELT,
            common_name=common,
            note="one monomer carries both the phenol and the halide, so there is no "
                 "stoichiometry to control",
        )


def _carbonates() -> Iterator[HeteroUnit]:
    for diol in CARBONATE_DIOLS + BISPHENOLS:
        common = COMMERCIAL_CARBONATES.get(diol.name, "")
        if common:
            availability = Availability.COMMERCIAL
        elif diol.name in REPORTED_CARBONATES:
            availability = Availability.REPORTED
            common = REPORTED_CARBONATES[diol.name]
        else:
            availability = Availability.CONSTRUCTIBLE
            common = ""
        yield HeteroUnit(
            name=f"poly({diol.short} carbonate)",
            smiles=CARBONATE.format(core=diol.core),
            subfamily="polycarbonate",
            monomers=(diol.name, "diphenyl carbonate or phosgene"),
            availability=availability,
            scale=diol.scale,
            melt=Melt.MELT,
            common_name=common,
            note="the C1 unit is phosgene or a diaryl carbonate; the feasibility expert "
                 "charges for it rather than pretending it is a monomer",
        )
    for epoxide, core, polymer, availability in CO2_CARBONATES:
        yield HeteroUnit(
            name=polymer,
            smiles=CARBONATE.format(core=core),
            subfamily="polycarbonate",
            monomers=(epoxide, "carbon dioxide"),
            availability=availability,
            scale=Scale.BULK,
            melt=Melt.MELT,
            route="CO2 + epoxide copolymerisation",
            note="the only route to this chain: the diol condensation gives the cyclic "
                 "carbonate instead, which is a solvent. Zinc glutarate or a cobalt salen "
                 "catalyst, 40 wt% of the polymer is CO2",
        )


def _urethanes() -> Iterator[HeteroUnit]:
    for iso in DIISOCYANATES:
        for extender in CHAIN_EXTENDERS:
            key = (iso.short, extender.name)
            common = COMMERCIAL_URETHANES.get(key, "")
            availability = (
                Availability.COMMERCIAL if common else Availability.CONSTRUCTIBLE
            )
            yield HeteroUnit(
                name=f"poly({iso.short}-{extender.short} urethane)",
                smiles=URETHANE.format(iso=iso.core, diol=extender.core),
                subfamily="polyurethane",
                monomers=(iso.name, extender.name),
                availability=availability,
                scale=_scale(iso.scale, extender.scale),
                melt=Melt.HARD_SEGMENT,
                common_name=common,
                note="this is the HARD SEGMENT of a segmented block copolymer, not the "
                     "polymer: a real polyurethane is this alternating with a polyether or "
                     "polyester soft segment of 1-2 kg/mol, and the modulus, the resilience "
                     "and the service window are all set by the phase separation between "
                     "the two. Every number predicted for this unit describes one phase",
            )


def _siloxanes() -> Iterator[HeteroUnit]:
    for monomer, substituents, polymer, common, availability, scale, note in SILOXANES:
        yield HeteroUnit(
            name=polymer,
            smiles=SILOXANE.format(r=substituents),
            subfamily="polysiloxane",
            monomers=(monomer,),
            availability=availability,
            scale=scale,
            melt=Melt.ELASTOMER,
            common_name=common,
            note=note or "made by hydrolysing the dichlorosilane to the cyclic tetramer and "
                         "equilibrating it open",
        )


def _imides() -> Iterator[HeteroUnit]:
    for anhydride, code, template, scale, flexible in DIANHYDRIDES:
        for amine in IMIDE_DIAMINES:
            key = (code, amine.short)
            if key in COMMERCIAL_IMIDES:
                availability, common = Availability.COMMERCIAL, COMMERCIAL_IMIDES[key]
            elif key in REPORTED_IMIDES:
                availability, common = Availability.REPORTED, REPORTED_IMIDES[key]
            else:
                availability, common = Availability.CONSTRUCTIBLE, ""
            melts = flexible and amine.short in _FLEXIBLE_DIAMINES
            yield HeteroUnit(
                name=f"poly({code}-{amine.short} imide)",
                smiles=template.format(amine=amine.core + "[*]"),
                subfamily="polyimide",
                monomers=(anhydride, amine.name),
                availability=availability,
                scale=_scale(scale, amine.scale),
                melt=Melt.MELT if melts else Melt.NONE,
                common_name=common,
                note="made by casting the soluble poly(amic acid) and cyclising it in place "
                     "above 300 C; the imide itself does not dissolve"
                     + ("" if melts else ". No melt: it decomposes first"),
            )


def _proposals() -> Iterator[HeteroUnit]:
    yield from _polyethers()
    yield from _aryl_ethers()
    yield from _carbonates()
    yield from _urethanes()
    yield from _siloxanes()
    yield from SULFIDES
    yield from _imides()


@functools.lru_cache(maxsize=1)
def _built() -> tuple[tuple[HeteroUnit, ...], tuple[Refusal, ...]]:
    kept: list[HeteroUnit] = []
    refused: list[Refusal] = []
    seen: set[str] = set()
    for proposal in (*_proposals(), *PROBES):
        verdict = gate(proposal)
        if verdict is not None:
            refused.append(verdict)
            continue
        key = canonical(proposal.smiles)
        if key is None or key in seen:  # pragma: no cover - the structure gate parses first
            continue
        seen.add(key)
        kept.append(proposal)
    return tuple(kept), tuple(refused)


def records() -> list[HeteroUnit]:
    """Every generated repeat unit, with its monomers, availability and caveats."""
    return list(_built()[0])


def refusals() -> list[Refusal]:
    """Every structure the grammar built and then refused, with the reason."""
    return list(_built()[1])


def units() -> list[tuple[str, str]]:
    """(name, repeat-unit SMILES) for every heteroatom-backbone polymer proposed here."""
    return [(record.label, record.smiles) for record in _built()[0]]


# --------------------------------------------------------------------------
# What this module measured
# --------------------------------------------------------------------------

#: Every number in the module docstring, in one place, asserted by the test
#: file.  Pinning them is the point: a grammar that quietly grows or shrinks is
#: a grammar nobody is reading, and a rediscovery count that slips from 8 to 7
#: should be a failing test rather than a paragraph that went stale.
MEASURED: dict[str, Any] = {
    #: Repeat units emitted, and structures built and refused.
    "units": 171,
    "refusals": 24,
    "subfamilies": {
        "polyurethane": 42,
        "polyimide": 36,
        "aryl polyether": 35,
        "polyether": 23,
        "polycarbonate": 19,
        "polysiloxane": 7,
        "polysulfide": 4,
        "polyacetal": 3,
        "lactone polyester": 1,
        "aryl polysulfide": 1,
    },
    #: 39 polymers that are or have been sold, 40 made and published, 92 pairs
    #: nobody appears to have run.  Every monomer in all 171 is purchasable: 77
    #: units are built entirely from commodities and 94 need a catalogue chemical.
    "availability": {"commercial": 39, "reported": 40, "constructible": 92},
    "monomer_scale": {"bulk": 77, "catalogue": 94},
    #: 99 have a melt, 42 are polyurethane hard segments, 21 polyimides
    #: decompose first, 9 are silicones or liquid polysulfides that are never
    #: thermoplastic.
    "melt": {
        "melt processable": 99,
        "hard segment of a block copolymer": 42,
        "not melt processable": 21,
        "fluid or cured elastomer, not melt processed": 9,
    },
    #: Silicon or sulfur, so the density and Tg experts will refuse them.
    "outside_element_budget": 35,
    #: All eight of the bundled reference polymers this family owed, by
    #: canonical SMILES against canonical SMILES, and no polymer belonging to
    #: another module's family.
    "rediscovered": 8,
    #: ``PolymerFeasibilityExpert`` scores 144 of 171 and has no route for 27
    #: (see :data:`NO_FEASIBILITY_ROUTE`).  Scores run 2.05 to 6.56 with a
    #: median of 3.34, against poly(ethylene terephthalate) at 2.50 - higher
    #: than the condensation family's 2.42 and rightly so, because half of this
    #: module needs a C1 unit, an equilibration or a 300 C dipolar aprotic
    #: solvent that a polyester does not.
    "feasibility_scored": 144,
    "feasibility_refused": 27,
    "feasibility_median": 3.34,
    "feasibility_max": 6.56,
}
