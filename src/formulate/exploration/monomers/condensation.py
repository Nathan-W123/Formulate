"""Step-growth repeat units: the polyesters and the polyamides.

The engine's polymer search space was a list of 57 repeat units, so every
"best polymer" it has ever reported meant "best of 57".  This module replaces
the list with a *grammar* for one family - the one whose combinatorics are
richest, because each unit is a pair.  A polyester is a diacid crossed with a
diol, a polyamide is a diacid crossed with a diamine, and an AB polymer is a
single hydroxy or amino acid.  Fourteen diacids against twenty-five partners is
350 pairs before anything AB is counted, and every monomer in it is one that
can be bought.

Nothing here predicts anything.  Every property downstream is predicted from
structure - chain dimension from side-group bulk, crystallinity from backbone
regularity, Hansen parameters from group contributions - so a repeat unit that
nobody has tabulated is scored exactly as well as one that was.  The only thing
that was missing was something to propose it.

What this module is careful about
---------------------------------
**It gates.**  A grammar that emits syntactically valid nonsense is worse than
57 real entries, because the nonsense fills the ranking and buries the
candidates that can exist.  Seven things are refused, and every one of them
names the reaction that happens *instead* of polymerisation, with its ring size
where the competitor is a ring, rather than calling the combination unusual.
Five act on pairs and remove **100** of the 350 the templates would otherwise
write - :data:`GATE_PHENOL_ALIPHATIC_ACID` (30), :data:`GATE_CYCLIC_IMIDE` (22),
:data:`GATE_ARYLAMINE_ALIPHATIC_ACID` (20), :data:`GATE_ETHYLENEDIAMINE` (14),
:data:`GATE_PROPANEDIAMINE` (14) - and :func:`refusals` returns them with their
reason, so a test can assert on what the grammar will not say.  Two remove a
monomer from the library outright: :data:`GATE_MALONIC` and
:data:`GATE_ALPHA_AMINO`.

Three of the five pair gates name one competing ring, as does
:data:`GATE_ALPHA_AMINO`, and the ring sizes are the whole argument: five- and
six-membered rings close in preference to chain growth, which is Carothers'
rule, and seven-membered ones do not.  That is why
the commercial AA+BB nylon series starts at 1,4-butanediamine and at adipic acid
and not one carbon short of either - a shorter diamine closes an amidine on the
amide it just made, a shorter diacid closes an imide - and why the same two
diacids are perfectly good on the ester side, where the ring that closes is an
anhydride and an anhydride goes on reacting.

**It says what is real.**  Every unit carries an :class:`Availability`, written
into the name, separating polymers that are sold (PET, nylon-6,6, Kevlar) from
polymers that have been made and published but never sold, from pairs that are
merely constructible.  All three are legitimate candidates - the point of
searching a space rather than a list is to reach the third kind - but a run that
cannot tell them apart has not answered the question that was asked.

**It reaches the fibres.**  The aramids are in this family and they are the
reason it matters: p-phenylenediamine with terephthalic acid is Kevlar, the
highest-performance fibre in commercial production, and m-phenylenediamine with
isophthalic acid is Nomex.  Both are AA+BB polyamides of monomers that are
already in the lists, so the grammar reaches them without a special case, and a
test asserts it by name.  The thermotropic polyesters are here too -
poly(4-hydroxybenzoic acid) and the hydroquinone and biphenol terephthalates -
which is why hydroquinone and 4,4'-biphenol are in the diol list at all.

**It stays inside the fitted element set.**  Every monomer here is C, H, N and
O only, which is inside ``polymer._FITTED_ELEMENTS`` (H, C, N, O, F, Cl), so no
unit from this module can push the density or Tg models outside the elements
they were fitted over.

Measured on this installation
-----------------------------
:func:`units` returns **271** repeat units: 110 polyesters, 140 polyamides, 11
AB polyesters and 10 AB polyamides.  **34** are polymers that are or have been
sold, **29** have been made and published, and **208** are constructible pairs
nobody appears to have made; 168 are built from two commodity monomers and 103
need at least one catalogue chemical.

*Rediscovery.*  Of the 57 bundled reference polymers, **12** belong to this
family - PET, PBT, PEN, poly(ethylene adipate), nylon-6, -6,6, -11, -12,
polycaprolactone, polylactide, polyglycolide and poly(3-hydroxybutyrate) - and
the grammar regenerates **12 of 12**.  Eleven come back as byte-identical
canonical SMILES.  The twelfth, polycaprolactone, is regenerated one atom out
of phase: the reference file draws it ``[*]CCCCCC(=O)O[*]`` and this module
draws it ``[*]OCCCCCC(=O)[*]``, which is the same periodic chain cut at a
different bond.  That is a spelling difference and not a chemical one - both
spellings score 2.20 on ``PolymerFeasibilityExpert``, from the same
caprolactone - and :func:`periodic_key` is phase-invariant so that a caller
deduplicating against the bundled file does not count that polymer twice.

*Feasibility.*  Run through ``PolymerFeasibilityExpert``, **271 of 271** units
get a polymerisation route and a score; none is refused.  Scores run 1.80 to
4.42 with a median of 2.42, against poly(ethylene terephthalate) at 2.50 -
which is the right answer, since every unit here but the eighteen oxalates and
oxamides is made by the same polycondensation PET is, and those are made the
same way from the oxalate diester (:data:`OXALATE_ROUTE`).  Only 14 units score
above 3.0 and 12 of those are the isophorone-diamine polyamides, all at 4.42.
That number is the Ertl fragment prior charging a gem-dimethyl cycloaliphatic
diamine for being unusual in
ChEMBL, not a statement about isophorone diamine, which is made at roughly
100 kt/yr.  It is reported
rather than corrected, because correcting it would mean fitting the expert to
this library.

*What the panel then refuses.*  **63 of 271** get no glass transition, because
the Tg group-contribution table has no coefficient for the furan ring, the
1,4-cyclohexylene ring or the isophorone skeleton.  That is the engine refusing
rather than guessing, which is the correct direction to fail in, but it is not
free: the default missing-objective policy penalises, so those 63 candidates
enter the ranking handicapped for a reason that is about the table rather than
about the polymer.

What this module does not cover
-------------------------------
Polycarbonates and polyurethanes are step-growth and are *not* here: their
linkage needs a C1 unit - phosgene, a diaryl carbonate, a diisocyanate - which
is a different feedstock chemistry, and the feasibility expert already charges
it separately.  Unsaturated polyester resins are not here either, because the
material is the cured network and its repeat unit is not the thing that is
sold.  Polyimides are absent because the feasibility expert does not recognise
imide formation from a dianhydride and would refuse every one of them.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Availability(str, Enum):
    """How real a generated polymer is, which is not the same as how makeable.

    The feasibility expert answers "could someone make this?".  This answers
    "has someone?", which a search needs separately: a candidate that is
    merely constructible is still a legitimate answer, but only if the run can
    see that nobody has ever made it.
    """

    #: Sold, or has been sold, as a material.  A trade or common name exists.
    COMMERCIAL = "commercial"
    #: Made and published, but not a product.  The route is known to work.
    REPORTED = "reported"
    #: Both monomers are sold and the pair condenses by the same chemistry as
    #: its commercial neighbours; nobody appears to have bothered.
    CONSTRUCTIBLE = "constructible"


class Scale(str, Enum):
    """Where a monomer is bought from, which is a constraint on the answer."""

    #: Made at ten-kilotonne scale or more: a commodity with a market price.
    BULK = "bulk"
    #: A catalogue chemical - real, purchasable, priced per kilogram not per
    #: tonne.  A polymer built only from these is an answer to a different
    #: question than one built from commodities.
    CATALOGUE = "catalogue"


@dataclass(frozen=True, slots=True)
class Monomer:
    """One half of a step-growth pair, stored as the SMILES fragment it contributes.

    ``core`` is the fragment *between* the two functional groups, so that the
    repeat unit is pure string assembly and the functional groups are written
    once, in the templates below, rather than once per monomer.
    """

    name: str
    core: str
    scale: Scale
    #: Carbon count used for nylon-m,n naming, or a letter code for the ring
    #: diacids (T for terephthalic, I for isophthalic).  ``None`` where no
    #: standard code exists.
    code: str | None = None
    #: The adjective the polymer's systematic name uses: "hexamethylene" for
    #: 1,6-hexanediamine.  Kept separate from :attr:`code` because the industry
    #: numbers a nylon by carbon count and names a polyester by the divalent
    #: group, and one field cannot be both.
    adjective: str = ""
    #: True when the functional groups sit directly on an aromatic ring, which
    #: changes what the monomer will condense with (see the gates below).
    aromatic: bool = False


@dataclass(frozen=True, slots=True)
class CondensationUnit:
    """A generated repeat unit with everything a ranking needs to judge it."""

    name: str
    smiles: str
    availability: Availability
    #: "polyester", "polyamide", or the AB forms of each.
    subfamily: str
    #: The monomers, in the order a chemist would order them.
    monomers: tuple[str, ...]
    #: Trade or common name where one exists, else "".
    common_name: str = ""
    note: str = ""

    #: BULK when every monomer is a commodity, CATALOGUE when at least one is
    #: only a catalogue chemical.  A polymer whose monomer is priced per
    #: kilogram is an answer to a different question than one whose monomer is
    #: priced per tonne, so the label says which.
    monomer_scale: Scale = Scale.BULK

    @property
    def label(self) -> str:
        """The name :func:`units` emits: systematic, then what it is called and how real."""
        parts = [self.common_name] if self.common_name and self.common_name != self.name else []
        parts.append(self.availability.value)
        if self.monomer_scale is Scale.CATALOGUE:
            parts.append("catalogue monomer")
        return f"{self.name} ({', '.join(parts)})"


# --------------------------------------------------------------------------
# The SMILES convention
# --------------------------------------------------------------------------
#
# A repeat unit carries exactly two [*], one at each end of the backbone, and
# must parse when they are capped with [H] rather than deleted.  These four
# templates reproduce the bundled file exactly: substituting "CC" and
# "c1ccc(cc1)" into POLYESTER gives the reference string for PET character for
# character, and "CCCCCC" and "CCCC" into POLYAMIDE gives nylon-6,6.

#: Diol + diacid.  The ester oxygen nearest the left attachment point belongs
#: to the diol, which is why the diol goes between the two oxygens.
POLYESTER = "[*]O{partner}OC(=O){acid}C(=O)[*]"

#: Diamine + diacid, drawn the same way round as nylon-6,6 in the bundled file.
POLYAMIDE = "[*]N{partner}NC(=O){acid}C(=O)[*]"

#: A hydroxy acid: one ester per repeat unit rather than two.
AB_ESTER = "[*]O{core}C(=O)[*]"

#: An amino acid, a lactam, or the lactone/lactam a fermentation route reaches.
AB_AMIDE = "[*]N{core}C(=O)[*]"


# --------------------------------------------------------------------------
# The diacids
# --------------------------------------------------------------------------
#
# ``core`` is the fragment between the two carbonyl carbons, so oxalic acid's
# core is the empty string: its two carbonyls are bonded to each other.

#: The linear alpha,omega-dicarboxylic acids, plus the four ring diacids that
#: carry the engineering thermoplastics.
#:
#: Odd-numbered members (pimelic, azelaic) are kept even though almost nothing
#: is made from them, because they are the control: an odd methylene count
#: breaks the chain's ability to register its amide or ester groups with the
#: neighbouring chain, so a nylon-6,7 melts far below nylon-6,6 and a grammar
#: that silently dropped them would hide that from the search rather than let
#: it be measured.  Azelaic acid is in fact a commodity - it is ozonolysed from
#: oleic acid - and nylon-6,9 was sold.
#:
#: Oxalic acid is kept and is the one diacid here whose *free acid* is not the
#: reagent: it decomposes to CO2 and formic acid around 190 C, below where a
#: melt polycondensation runs.  The polymers are real anyway - poly(ethylene
#: oxalate) and the nylon-n,2 polyoxamides are a published family - because they
#: are made from dimethyl or dibutyl oxalate by transesterification or
#: aminolysis instead.  :data:`OXALATE_ROUTE` says so on every unit it touches,
#: because the reconstruction downstream will hand back oxalic acid and that is
#: not the bottle you would order.  Nothing cyclises here: an oxamide would have
#: to close a four-membered ring, which is why the C2 diacid escapes the imide
#: gate the C4 and C5 ones do not.
DIACIDS: tuple[Monomer, ...] = (
    Monomer("oxalic acid", "", Scale.BULK, code="2"),
    Monomer("succinic acid", "CC", Scale.BULK, code="4"),
    Monomer("glutaric acid", "CCC", Scale.BULK, code="5"),
    Monomer("adipic acid", "CCCC", Scale.BULK, code="6"),
    Monomer("pimelic acid", "CCCCC", Scale.CATALOGUE, code="7"),
    Monomer("suberic acid", "CCCCCC", Scale.CATALOGUE, code="8"),
    Monomer("azelaic acid", "CCCCCCC", Scale.BULK, code="9"),
    Monomer("sebacic acid", "CCCCCCCC", Scale.BULK, code="10"),
    Monomer("dodecanedioic acid", "CCCCCCCCCC", Scale.BULK, code="12"),
    Monomer("terephthalic acid", "c1ccc(cc1)", Scale.BULK, code="T", aromatic=True),
    Monomer("isophthalic acid", "c1cc(ccc1)", Scale.BULK, code="I", aromatic=True),
    Monomer(
        "2,6-naphthalenedicarboxylic acid",
        "c1ccc2cc(ccc2c1)",
        Scale.BULK,
        code="N",
        aromatic=True,
    ),
    Monomer(
        "1,4-cyclohexanedicarboxylic acid", "C1CCC(CC1)", Scale.BULK, code="CHDA"
    ),
    Monomer("furan-2,5-dicarboxylic acid", "c1ccc(o1)", Scale.CATALOGUE, code="F", aromatic=True),
)

#: Succinic and glutaric acid are kept as *diols'* partners and refused as
#: *diamines'*, and the asymmetry is the whole chemistry of the C4/C5 diacids.
#: Once one carboxyl of a succinic unit has become an amide, the second one is
#: three atoms from the amide nitrogen: it closes to the N-substituted
#: succinimide, a five-membered ring, and glutaric closes the same way to the
#: six-membered glutarimide.  Both rings are what you get when you heat the
#: diacid with a primary amine on purpose - that *is* the standard synthesis of
#: an N-alkylsuccinimide - and a nylon-salt melt runs at 210-280 C with the
#: water pulled off, which is the same condition.  The ring caps the chain end
#: it forms on, and a mid-chain one cleaves the backbone: it is the aspartyl
#: succinimide that cuts peptides at Asn-Gly, one ring size up from nothing.
#:
#: The ester side is untouched because the ester analogue is not a dead end.
#: An oxygen closing on the second carboxyl gives succinic *anhydride*, which
#: is still an acylating agent and still reacts on with the diol - which is why
#: poly(butylene succinate) is sold by the kilotonne while no AA+BB polyamide
#: anyone has sold uses a diacid shorter than adipic.  Adipic is where the imide
#: would have to be seven-membered, and that is the whole reason the commercial
#: nylon series starts there.
#:
#: An acid chloride at 0 C would reach these polyamides, since the ring needs
#: the heat.  That route is real and it is how the aramids below are made, but
#: it is not run for an aliphatic nylon, so it does not rescue the pair here.
GATE_CYCLIC_IMIDE = (
    "succinic or glutaric acid with any diamine: the second carboxyl closes onto "
    "the amide nitrogen to give the N-substituted succinimide or glutarimide, "
    "which caps the chain end it forms on"
)

#: Malonic acid is the one member of the linear series that is deliberately
#: absent, and it is absent for a reason a search should not have to rediscover:
#: a carboxyl beta to a second carbonyl decarboxylates, and malonic acid does
#: it at about 140 C.  Melt polycondensation runs at 200-280 C, so what comes
#: out of a malonate polyesterification is acetic acid and carbon dioxide, not
#: a chain.  This is a judgement about a temperature window rather than a law -
#: malonate esters have been condensed at low temperature with a titanium
#: catalyst - and it is recorded here rather than buried, because leaving
#: malonic in would have put fourteen candidates into the ranking that decompose
#: as they form.
GATE_MALONIC = (
    "malonic acid and its amide analogue: a carboxyl beta to a second carbonyl "
    "decarboxylates near 140 C, below any melt polycondensation"
)

#: Carried on every oxalate and oxamide unit.  It is not a caveat about whether
#: the polymer exists - poly(ethylene oxalate) does - but about which bottle the
#: route starts from, which is the one thing a structure-only reconstruction
#: downstream cannot tell you.
OXALATE_ROUTE = (
    "made from dimethyl or dibutyl oxalate, not from oxalic acid, which "
    "decomposes below polycondensation temperature"
)


# --------------------------------------------------------------------------
# The diols
# --------------------------------------------------------------------------

#: ``core`` sits between the two ester oxygens.
#:
#: The seven aliphatic diols are the ones the polyester industry actually
#: buys.  Neopentyl glycol and cyclohexanedimethanol are here because they are
#: the two standard ways of keeping an ester backbone amorphous or
#: hydrolytically stable without leaving the family, and diethylene glycol
#: because an ether oxygen in the backbone is the cheapest flexibility there is
#: - it is also the one diol here whose polymer the engine can be expected to
#: get wrong, since an ether oxygen changes the cohesive energy more than it
#: changes the chain dimension.
DIOLS: tuple[Monomer, ...] = (
    Monomer("ethylene glycol", "CC", Scale.BULK, adjective="ethylene"),
    Monomer("1,3-propanediol", "CCC", Scale.BULK, adjective="trimethylene"),
    Monomer("1,4-butanediol", "CCCC", Scale.BULK, adjective="tetramethylene"),
    Monomer("1,6-hexanediol", "CCCCCC", Scale.BULK, adjective="hexamethylene"),
    Monomer("neopentyl glycol", "CC(C)(C)C", Scale.BULK, adjective="neopentylene"),
    Monomer(
        "1,4-cyclohexanedimethanol",
        "CC1CCC(CC1)C",
        Scale.BULK,
        adjective="cyclohexylenedimethylene",
    ),
    Monomer("diethylene glycol", "CCOCC", Scale.BULK, adjective="diethylene"),
)

#: The diphenols, kept apart from the diols above because a phenol is not an
#: alcohol for this purpose: it is some five orders of magnitude less
#: nucleophilic, so it does not melt-esterify and the polymer is made either
#: from the diacid chloride at an interface or from the diphenol *diacetate* by
#: transesterification.  Both routes are practised, and only for the aromatic
#: diacids - which is exactly the gate below.
#:
#: Bisphenol A gives the polyarylates.  Hydroquinone and 4,4'-biphenol give the
#: rigid all-aromatic polyesters that the thermotropic liquid-crystal polymers
#: are built from, which is why they are worth the two extra entries: they are
#: the highest-modulus melt-processable chains in this family.
DIPHENOLS: tuple[Monomer, ...] = (
    Monomer(
        "bisphenol A",
        "c1ccc(cc1)C(C)(C)c1ccc(cc1)",
        Scale.BULK,
        adjective="bisphenol-A",
        aromatic=True,
    ),
    Monomer("hydroquinone", "c1ccc(cc1)", Scale.BULK, adjective="p-phenylene", aromatic=True),
    Monomer(
        "4,4'-biphenol",
        "c1ccc(cc1)c1ccc(cc1)",
        Scale.CATALOGUE,
        adjective="4,4'-biphenylene",
        aromatic=True,
    ),
)

GATE_PHENOL_ALIPHATIC_ACID = (
    "a diphenol with a non-aromatic diacid: a phenol will not melt-esterify, and "
    "the acid-chloride and diacetate routes that do work are practised only for "
    "the aromatic diacids"
)


# --------------------------------------------------------------------------
# The diamines
# --------------------------------------------------------------------------

#: ``core`` sits between the two nitrogens.
#:
#: The linear series runs to C12 because that is where the commercial nylons
#: stop (nylon-12,12 is the last one anyone has sold) and because the longer a
#: methylene run gets the more the chain behaves like polyethylene with
#: occasional amides, which the engine can already reach from the short members.
#: It *starts* at C2 even though the C2 and C3 members are refused by the gates
#: below, so that the refusal is a statement the grammar makes rather than a
#: monomer quietly missing from a list.
#: m-Xylylenediamine earns its place by being the monomer of nylon-MXD6, the
#: barrier polyamide.  Isophorone diamine earns its place as a commodity
#: cycloaliphatic diamine - roughly 100 kt/yr, for epoxy curing and for
#: isophorone diisocyanate - whose gem-dimethyl ring cannot pack into a crystal,
#: which is the structural trick the transparent polyamides use.  It is not the
#: diamine they are actually built from: Trogamid is trimethylhexamethylene-
#: diamine and Grilamid TR is bis(4-amino-3-methylcyclohexyl)methane, neither of
#: which is here, so the isophorone polyamides in this library are constructible
#: and are labelled that way rather than being sold as the transparent nylons.
DIAMINES: tuple[Monomer, ...] = (
    Monomer("1,2-ethanediamine", "CC", Scale.BULK, code="2", adjective="ethylene"),
    Monomer("1,3-propanediamine", "CCC", Scale.BULK, code="3", adjective="trimethylene"),
    Monomer("1,4-butanediamine", "CCCC", Scale.BULK, code="4", adjective="tetramethylene"),
    Monomer("1,5-pentanediamine", "CCCCC", Scale.BULK, code="5", adjective="pentamethylene"),
    Monomer("1,6-hexanediamine", "CCCCCC", Scale.BULK, code="6", adjective="hexamethylene"),
    Monomer("1,7-heptanediamine", "CCCCCCC", Scale.CATALOGUE, code="7", adjective="heptamethylene"),
    Monomer("1,8-octanediamine", "CCCCCCCC", Scale.CATALOGUE, code="8", adjective="octamethylene"),
    Monomer("1,9-nonanediamine", "CCCCCCCCC", Scale.BULK, code="9", adjective="nonamethylene"),
    Monomer("1,10-decanediamine", "CCCCCCCCCC", Scale.BULK, code="10", adjective="decamethylene"),
    Monomer(
        "1,11-undecanediamine",
        "CCCCCCCCCCC",
        Scale.CATALOGUE,
        code="11",
        adjective="undecamethylene",
    ),
    Monomer(
        "1,12-dodecanediamine",
        "CCCCCCCCCCCC",
        Scale.BULK,
        code="12",
        adjective="dodecamethylene",
    ),
    Monomer("m-xylylenediamine", "Cc1cc(ccc1)C", Scale.BULK, code="MXD", adjective="m-xylylene"),
    Monomer(
        "isophorone diamine",
        "CC1(C)CC(CC(C)(C)C1)",
        Scale.BULK,
        code="IPD",
        adjective="isophorone",
    ),
)

#: The aromatic diamines, which are the aramids and the reason this family is
#: the one a high-performance fibre comes from.  p-Phenylenediamine with
#: terephthalic acid is Kevlar; m-phenylenediamine with isophthalic acid is
#: Nomex.  Both are made in solution from the diacid chloride, not in a melt,
#: which is what the gate below records.
AROMATIC_DIAMINES: tuple[Monomer, ...] = (
    Monomer(
        "p-phenylenediamine",
        "c1ccc(cc1)",
        Scale.BULK,
        code="PPD",
        adjective="p-phenylene",
        aromatic=True,
    ),
    Monomer(
        "m-phenylenediamine",
        "c1cc(ccc1)",
        Scale.BULK,
        code="MPD",
        adjective="m-phenylene",
        aromatic=True,
    ),
)

GATE_ARYLAMINE_ALIPHATIC_ACID = (
    "an aromatic diamine with a non-aromatic diacid: an arylamine is too weak a "
    "base to form the nylon salt a melt polycondensation starts from, and the "
    "acid-chloride solution route that makes the aramids is run only against the "
    "aromatic diacids"
)

#: 1,2-Ethanediamine is excluded from the AA+BB polyamides and from nothing
#: else.  A 1,2-diamine and a carboxylic acid do not stop at the amide: the
#: second nitrogen closes onto the amide carbon and dehydrates to a
#: 2-substituted imidazoline, which is a five-membered aromatic ring and a chain
#: terminator.  That reaction is run deliberately, at scale, to make imidazoline
#: surfactants, and it is the reason there is no commercial nylon-2,n.
GATE_ETHYLENEDIAMINE = (
    "1,2-ethanediamine with any diacid: the second nitrogen closes onto the amide "
    "and dehydrates to a 2-substituted imidazoline, which terminates the chain"
)

#: 1,3-Propanediamine is refused for the same reason one ring size up, and the
#: ring size is the argument rather than an aggravation of it.  Its cyclisation
#: product is a 2-substituted 1,4,5,6-tetrahydropyrimidine, a *six*-membered
#: amidine, and six-membered rings are the ones Carothers found form in
#: preference to chain growth - the same rule that puts the alpha-amino acids
#: out through the diketopiperazine.  Refusing the five-ring and keeping the
#: six-ring would have had the rule backwards.
#:
#: The corroboration is negative and worth stating as such: the commercial
#: AA+BB nylon series starts at 1,4-butanediamine (PA46), and no nylon-3,n
#: appears in the tables below under any supplier or any paper, while
#: 2-substituted tetrahydropyrimidines are made deliberately from
#: 1,3-diaminopropane and a carboxylic acid at 150-200 C - melt-polycondensation
#: conditions.  A low-temperature acid-chloride route would give the polyamide,
#: since the amidine needs the dehydration, but that route is not run for an
#: aliphatic nylon and it is not counted here.
GATE_PROPANEDIAMINE = (
    "1,3-propanediamine with any diacid: the second nitrogen closes onto the amide "
    "and dehydrates to a 2-substituted tetrahydropyrimidine, the six-membered "
    "amidine, which terminates the chain"
)


# --------------------------------------------------------------------------
# The AB monomers
# --------------------------------------------------------------------------

#: Hydroxy acids.  ``core`` sits between the hydroxyl oxygen and the carbonyl
#: carbon, so glycolic acid's core is a single carbon.
#:
#: These are the AB polyesters that exist, and the list is short because the
#: chemistry is: a hydroxy acid either polymerises, or closes to a lactone that
#: polymerises, or closes to a lactone that sits in the bottle.  Every member
#: here does one of the first two.  Note that the engine's feasibility expert
#: will route 4-hydroxybutyric acid through ring-opening of gamma-butyrolactone
#: and be *wrong* about it - gamma-butyrolactone famously does not homopolymerise
#: - yet the polymer is right, and commercial, because it is made by
#: fermentation.  A structure-only engine cannot see that, which is a limit worth
#: knowing rather than a reason to drop a real material.
HYDROXY_ACIDS: tuple[tuple[str, str, Availability, str, str, Scale], ...] = (
    ("glycolic acid", "C", Availability.COMMERCIAL, "PGA", "polyglycolide, via the cyclic dimer", Scale.BULK),
    ("lactic acid", "C(C)", Availability.COMMERCIAL, "PLA", "polylactide, via lactide", Scale.BULK),
    ("3-hydroxypropionic acid", "CC", Availability.REPORTED, "P3HP", "also from beta-propiolactone", Scale.CATALOGUE),
    ("3-hydroxybutyric acid", "C(C)C", Availability.COMMERCIAL, "PHB", "bacterial polyester", Scale.CATALOGUE),
    (
        "3-hydroxyvaleric acid",
        "C(CC)C",
        Availability.REPORTED,
        "P3HV",
        "bacterial polyester; the homopolymer is published but only the PHBV "
        "copolymer is sold, so this is not a product",
        Scale.CATALOGUE,
    ),
    (
        "4-hydroxybutyric acid",
        "CCC",
        Availability.COMMERCIAL,
        "P4HB",
        "fermentation; gamma-butyrolactone itself does not homopolymerise",
        Scale.CATALOGUE,
    ),
    ("5-hydroxypentanoic acid", "CCCC", Availability.REPORTED, "PVL", "from delta-valerolactone", Scale.CATALOGUE),
    ("6-hydroxyhexanoic acid", "CCCCC", Availability.COMMERCIAL, "PCL", "from caprolactone", Scale.BULK),
    (
        "15-hydroxypentadecanoic acid",
        "CCCCCCCCCCCCCC",
        Availability.REPORTED,
        "PPDL",
        "entropy-driven ring-opening of pentadecalactone",
        Scale.CATALOGUE,
    ),
    (
        "4-hydroxybenzoic acid",
        "c1ccc(cc1)",
        Availability.COMMERCIAL,
        "Ekonol",
        "infusible homopolymer; the LCP backbone",
        Scale.BULK,
    ),
    (
        "6-hydroxy-2-naphthoic acid",
        "c1ccc2cc(ccc2c1)",
        Availability.REPORTED,
        "HNA homopolymer",
        "sold only copolymerised with 4-hydroxybenzoic acid, as Vectra",
        Scale.CATALOGUE,
    ),
)

#: Amino acids and lactams.  ``core`` sits between the nitrogen and the carbonyl
#: carbon, so 6-aminohexanoic acid - caprolactam - has five carbons in its core.
AMINO_ACIDS: tuple[tuple[str, str, Availability, str, str, Scale], ...] = (
    ("3-aminopropionic acid", "CC", Availability.REPORTED, "nylon-3", "from acrylamide", Scale.CATALOGUE),
    ("4-aminobutyric acid", "CCC", Availability.REPORTED, "nylon-4", "from 2-pyrrolidone", Scale.CATALOGUE),
    ("5-aminopentanoic acid", "CCCC", Availability.REPORTED, "nylon-5", "", Scale.CATALOGUE),
    ("6-aminohexanoic acid", "CCCCC", Availability.COMMERCIAL, "nylon-6", "from caprolactam", Scale.BULK),
    ("7-aminoheptanoic acid", "CCCCCC", Availability.REPORTED, "nylon-7", "", Scale.CATALOGUE),
    ("8-aminooctanoic acid", "CCCCCCC", Availability.REPORTED, "nylon-8", "", Scale.CATALOGUE),
    ("9-aminononanoic acid", "CCCCCCCC", Availability.REPORTED, "nylon-9", "", Scale.CATALOGUE),
    ("10-aminodecanoic acid", "CCCCCCCCC", Availability.REPORTED, "nylon-10", "", Scale.CATALOGUE),
    ("11-aminoundecanoic acid", "CCCCCCCCCC", Availability.COMMERCIAL, "nylon-11", "castor oil", Scale.BULK),
    (
        "12-aminododecanoic acid",
        "CCCCCCCCCCC",
        Availability.COMMERCIAL,
        "nylon-12",
        "from laurolactam",
        Scale.BULK,
    ),
)

#: The alpha-amino acids are excluded, and this is the gate most worth having.
#: Heating glycine or alanine does not give a polyamide: the two ends are close
#: enough that the dimer closes to a diketopiperazine, a stable six-membered
#: ring, and that is what comes out.  Polypeptides are made from
#: N-carboxyanhydrides - a different monomer, a ring-opening mechanism, and a
#: chemistry the feasibility expert does not recognise - so an alpha-amino acid
#: emitted here would be a repeat unit whose reconstructed monomer is real and
#: whose polymerisation is not.
GATE_ALPHA_AMINO = (
    "an alpha-amino acid: the dimer closes to a diketopiperazine rather than "
    "propagating, and polypeptides come from N-carboxyanhydrides instead"
)


# --------------------------------------------------------------------------
# Which pairs are real
# --------------------------------------------------------------------------
#
# Keyed by (partner name, diacid name).  Everything absent from this table is
# CONSTRUCTIBLE, which is a claim about the literature rather than about the
# chemistry: the pair condenses by the same reaction as its neighbours here.

#: Polyesters that are or have been sold, with what they are sold as.
COMMERCIAL_POLYESTERS: dict[tuple[str, str], str] = {
    ("ethylene glycol", "terephthalic acid"): "PET",
    ("1,3-propanediol", "terephthalic acid"): "PTT",
    ("1,4-butanediol", "terephthalic acid"): "PBT",
    ("ethylene glycol", "2,6-naphthalenedicarboxylic acid"): "PEN",
    ("1,4-cyclohexanedimethanol", "terephthalic acid"): "PCT",
    ("1,4-cyclohexanedimethanol", "1,4-cyclohexanedicarboxylic acid"): "PCCD",
    ("1,4-butanediol", "succinic acid"): "PBS",
    ("1,4-butanediol", "adipic acid"): "PBA polyol",
    ("ethylene glycol", "adipic acid"): "PEA polyol",
    ("1,6-hexanediol", "adipic acid"): "hexanediol adipate polyol",
    ("neopentyl glycol", "adipic acid"): "NPG adipate polyol",
    ("neopentyl glycol", "isophthalic acid"): "powder-coating polyester",
    ("bisphenol A", "terephthalic acid"): "polyarylate",
    ("bisphenol A", "isophthalic acid"): "polyarylate",
}

#: Polyesters published but not sold.
REPORTED_POLYESTERS: dict[tuple[str, str], str] = {
    ("ethylene glycol", "furan-2,5-dicarboxylic acid"): "PEF",
    ("1,4-butanediol", "furan-2,5-dicarboxylic acid"): "PBF",
    ("ethylene glycol", "oxalic acid"): "poly(ethylene oxalate)",
    ("1,4-butanediol", "sebacic acid"): "PBSe",
    ("ethylene glycol", "succinic acid"): "PES",
    ("hydroquinone", "terephthalic acid"): "LCP hard segment",
    ("4,4'-biphenol", "terephthalic acid"): "LCP hard segment",
    ("diethylene glycol", "isophthalic acid"): "saturated polyester resin",
}

#: Polyamides that are or have been sold.  The nylon-m,n codes are the industry's
#: own, m from the diamine and n from the diacid.
COMMERCIAL_POLYAMIDES: dict[tuple[str, str], str] = {
    ("1,4-butanediamine", "adipic acid"): "nylon-4,6",
    ("1,6-hexanediamine", "adipic acid"): "nylon-6,6",
    ("1,6-hexanediamine", "azelaic acid"): "nylon-6,9",
    ("1,6-hexanediamine", "sebacic acid"): "nylon-6,10",
    ("1,6-hexanediamine", "dodecanedioic acid"): "nylon-6,12",
    ("1,10-decanediamine", "sebacic acid"): "nylon-10,10",
    ("1,9-nonanediamine", "terephthalic acid"): "nylon-9,T",
    ("1,10-decanediamine", "terephthalic acid"): "nylon-10,T",
    ("m-xylylenediamine", "adipic acid"): "nylon-MXD6",
    ("p-phenylenediamine", "terephthalic acid"): "Kevlar",
    ("m-phenylenediamine", "isophthalic acid"): "Nomex",
}

#: Polyamides published but not sold as the homopolymer.
REPORTED_POLYAMIDES: dict[tuple[str, str], str] = {
    ("1,6-hexanediamine", "terephthalic acid"): "nylon-6,T",
    ("1,6-hexanediamine", "isophthalic acid"): "nylon-6,I",
    ("1,10-decanediamine", "dodecanedioic acid"): "nylon-10,12",
    ("1,12-dodecanediamine", "dodecanedioic acid"): "nylon-12,12",
    ("1,5-pentanediamine", "sebacic acid"): "nylon-5,10",
    ("p-phenylenediamine", "isophthalic acid"): "poly(p-phenylene isophthalamide)",
    ("m-phenylenediamine", "terephthalic acid"): "poly(m-phenylene terephthalamide)",
    ("p-phenylenediamine", "2,6-naphthalenedicarboxylic acid"): "rigid-rod aramid",
    ("m-xylylenediamine", "sebacic acid"): "nylon-MXD10",
}

#: The adipate polyols are sold, which is why they are COMMERCIAL, but they are
#: sold as hydroxyl-terminated oligomers for polyurethane rather than as
#: thermoplastics in their own right.  A ranking that reads "commercial" off the
#: label and expects a moulding resin would be misreading it.
_POLYOL_NOTE = (
    "sold as a hydroxyl-terminated oligomer for polyurethane, not as a "
    "high-molar-mass thermoplastic"
)

#: Notes attached to specific pairs where the availability flag alone would
#: mislead.  A homopolymer that melts above its own decomposition temperature is
#: not a material, however commercial its name is.
PAIR_NOTES: dict[tuple[str, str], str] = {
    ("ethylene glycol", "adipic acid"): _POLYOL_NOTE,
    ("1,4-butanediol", "adipic acid"): _POLYOL_NOTE,
    ("1,6-hexanediol", "adipic acid"): _POLYOL_NOTE,
    ("neopentyl glycol", "adipic acid"): _POLYOL_NOTE,
    ("neopentyl glycol", "isophthalic acid"): (
        "sold as a saturated coating resin: the linear diol-terminated chain, "
        "usually copolymerised with terephthalic acid and branched with a triol"
    ),
    ("1,6-hexanediamine", "terephthalic acid"): (
        "the homopolymer melts near 370 C, above where it decomposes; PA6T is sold "
        "only as a copolymer"
    ),
    ("p-phenylenediamine", "terephthalic acid"): (
        "spun from sulfuric acid solution as a lyotropic liquid crystal; it has no melt"
    ),
    ("m-phenylenediamine", "isophthalic acid"): (
        "solution-spun; the meta links stop the chain from being a rigid rod"
    ),
    ("hydroquinone", "terephthalic acid"): (
        "infusible as the homopolymer; it reaches a melt only copolymerised"
    ),
    ("4,4'-biphenol", "terephthalic acid"): (
        "infusible as the homopolymer; the Xydar backbone is a copolymer of it"
    ),
}


# --------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------

#: Diacid name -> the ester and amide forms used in a polymer name.
_ACID_FORMS: dict[str, tuple[str, str]] = {
    "oxalic acid": ("oxalate", "oxamide"),
    "succinic acid": ("succinate", "succinamide"),
    "glutaric acid": ("glutarate", "glutaramide"),
    "adipic acid": ("adipate", "adipamide"),
    "pimelic acid": ("pimelate", "pimelamide"),
    "suberic acid": ("suberate", "suberamide"),
    "azelaic acid": ("azelate", "azelamide"),
    "sebacic acid": ("sebacate", "sebacamide"),
    "dodecanedioic acid": ("dodecanedioate", "dodecanediamide"),
    "terephthalic acid": ("terephthalate", "terephthalamide"),
    "isophthalic acid": ("isophthalate", "isophthalamide"),
    "2,6-naphthalenedicarboxylic acid": ("2,6-naphthalate", "2,6-naphthalamide"),
    "1,4-cyclohexanedicarboxylic acid": (
        "1,4-cyclohexanedicarboxylate",
        "1,4-cyclohexanedicarboxamide",
    ),
    "furan-2,5-dicarboxylic acid": ("furan-2,5-dicarboxylate", "furan-2,5-dicarboxamide"),
}


def _nylon_code(diamine: Monomer, diacid: Monomer) -> str:
    """The industry's nylon-m,n code, where both halves have one."""
    if diamine.code is None or diacid.code is None:
        return ""
    if diamine.code.isdigit() and diacid.code.isdigit():
        return f"nylon-{diamine.code},{diacid.code}"
    if diamine.code.isdigit():
        return f"nylon-{diamine.code},{diacid.code}"
    return ""


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------


def _gate(partner: Monomer, diacid: Monomer, *, amide: bool) -> str | None:
    """Why this pair does not polymerise, or None if it does.

    Every gate here names the reaction that happens *instead*.  A gate whose
    justification is only "unusual" would be filtering the search rather than
    the chemistry, and the whole point of generating a space is to reach
    combinations nobody has tried.
    """
    if diacid.name == "malonic acid":  # pragma: no cover - not in DIACIDS
        return GATE_MALONIC
    if amide:
        if partner.name == "1,2-ethanediamine":
            return GATE_ETHYLENEDIAMINE
        if partner.name == "1,3-propanediamine":
            return GATE_PROPANEDIAMINE
        if partner.aromatic and not diacid.aromatic:
            return GATE_ARYLAMINE_ALIPHATIC_ACID
        if diacid.name in ("succinic acid", "glutaric acid"):
            return GATE_CYCLIC_IMIDE
    else:
        if partner.aromatic and not diacid.aromatic:
            return GATE_PHENOL_ALIPHATIC_ACID
    return None


def refusals() -> list[tuple[str, str, str]]:
    """Every (partner, diacid, reason) combination the grammar will not emit.

    A grammar is defined by what it refuses, so the refusals are a first-class
    return value rather than a silent ``continue`` inside a loop.
    """
    out: list[tuple[str, str, str]] = []
    for partner in DIOLS + DIPHENOLS:
        for diacid in DIACIDS:
            reason = _gate(partner, diacid, amide=False)
            if reason is not None:
                out.append((partner.name, diacid.name, reason))
    for partner in DIAMINES + AROMATIC_DIAMINES:
        for diacid in DIACIDS:
            reason = _gate(partner, diacid, amide=True)
            if reason is not None:
                out.append((partner.name, diacid.name, reason))
    return out


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def _scale(*monomers: Monomer) -> Scale:
    """A pair is only a commodity pair if both halves are."""
    return (
        Scale.CATALOGUE
        if any(m.scale is Scale.CATALOGUE for m in monomers)
        else Scale.BULK
    )


def _availability(
    key: tuple[str, str],
    commercial: dict[tuple[str, str], str],
    reported: dict[tuple[str, str], str],
    fallback_name: str,
) -> tuple[Availability, str]:
    if key in commercial:
        return Availability.COMMERCIAL, commercial[key]
    if key in reported:
        return Availability.REPORTED, reported[key]
    return Availability.CONSTRUCTIBLE, fallback_name


def _polyesters() -> Iterator[CondensationUnit]:
    for diol in DIOLS + DIPHENOLS:
        for diacid in DIACIDS:
            if _gate(diol, diacid, amide=False) is not None:
                continue
            ester, _ = _ACID_FORMS[diacid.name]
            name = f"poly({diol.adjective} {ester})"
            key = (diol.name, diacid.name)
            availability, common = _availability(
                key, COMMERCIAL_POLYESTERS, REPORTED_POLYESTERS, ""
            )
            note = PAIR_NOTES.get(key, "")
            if diacid.name == "oxalic acid":
                note = f"{note}; {OXALATE_ROUTE}" if note else OXALATE_ROUTE
            yield CondensationUnit(
                name=name,
                smiles=POLYESTER.format(partner=diol.core, acid=diacid.core),
                availability=availability,
                subfamily="polyester",
                monomers=(diol.name, diacid.name),
                common_name=common,
                note=note,
                monomer_scale=_scale(diol, diacid),
            )


def _polyamides() -> Iterator[CondensationUnit]:
    for diamine in DIAMINES + AROMATIC_DIAMINES:
        for diacid in DIACIDS:
            if _gate(diamine, diacid, amide=True) is not None:
                continue
            _, amide = _ACID_FORMS[diacid.name]
            name = f"poly({diamine.adjective} {amide})"
            key = (diamine.name, diacid.name)
            availability, common = _availability(
                key, COMMERCIAL_POLYAMIDES, REPORTED_POLYAMIDES, _nylon_code(diamine, diacid)
            )
            note = PAIR_NOTES.get(key, "")
            if diacid.name == "oxalic acid":
                note = f"{note}; {OXALATE_ROUTE}" if note else OXALATE_ROUTE
            yield CondensationUnit(
                name=name,
                smiles=POLYAMIDE.format(partner=diamine.core, acid=diacid.core),
                availability=availability,
                subfamily="polyamide",
                monomers=(diamine.name, diacid.name),
                common_name=common,
                note=note,
                monomer_scale=_scale(diamine, diacid),
            )


def _ab_units() -> Iterator[CondensationUnit]:
    for monomer, core, availability, common, note, scale in HYDROXY_ACIDS:
        yield CondensationUnit(
            name=f"poly({monomer})",
            smiles=AB_ESTER.format(core=core),
            availability=availability,
            subfamily="AB polyester",
            monomers=(monomer,),
            common_name=common,
            note=note,
            monomer_scale=scale,
        )
    for monomer, core, availability, common, note, scale in AMINO_ACIDS:
        yield CondensationUnit(
            name=f"poly({monomer})",
            smiles=AB_AMIDE.format(core=core),
            availability=availability,
            subfamily="AB polyamide",
            monomers=(monomer,),
            common_name=common,
            note=note,
            monomer_scale=scale,
        )


def catalogue() -> list[CondensationUnit]:
    """Every generated unit, with its availability, monomers and caveats.

    Deduplicated on the repeat unit itself: two different monomer pairs that
    write the same chain are the same polymer, and the first spelling wins.
    """
    seen: set[str] = set()
    out: list[CondensationUnit] = []
    for unit in (*_polyesters(), *_polyamides(), *_ab_units()):
        if unit.smiles in seen:
            continue
        seen.add(unit.smiles)
        out.append(unit)
    return out


def units() -> list[tuple[str, str]]:
    """(name, repeat-unit SMILES) for every polyester and polyamide generated here.

    The name carries the availability - "commercial", "reported" or
    "constructible" - because it is the only channel an explorer that takes
    ``(name, smiles)`` pairs has, and a ranking that cannot tell a product from
    a possibility has answered a different question than the one asked.
    """
    return [(unit.label, unit.smiles) for unit in catalogue()]


# --------------------------------------------------------------------------
# Phase-invariant identity
# --------------------------------------------------------------------------


def periodic_key(unit_smiles: str) -> str | None:
    """A canonical identity for a repeat unit that ignores where it was cut.

    ``[*]CCCCCC(=O)O[*]`` and ``[*]OCCCCCC(=O)[*]`` are the same chain - both
    are polycaprolactone - cut at a different bond, and they have different
    canonical SMILES.  Closing the unit into a ring removes the choice of cut
    entirely, which is what makes this comparable across two people's spellings.
    Returns None if the unit does not parse or does not have two attachment
    points on two different atoms.

    The ring this builds is a graph key, not a claim that the ring exists.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(unit_smiles)
    if mol is None:
        return None
    dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        return None
    anchors: list[int] = []
    for idx in dummies:
        neighbours = mol.GetAtomWithIdx(idx).GetNeighbors()
        if len(neighbours) != 1:
            return None
        anchors.append(neighbours[0].GetIdx())
    if anchors[0] == anchors[1]:
        return None
    rw = Chem.RWMol(mol)
    if rw.GetBondBetweenAtoms(anchors[0], anchors[1]) is not None:
        return None
    rw.AddBond(anchors[0], anchors[1], Chem.BondType.SINGLE)
    for idx in sorted(dummies, reverse=True):
        rw.RemoveAtom(idx)
    for atom in rw.GetAtoms():
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    ring: Any = rw.GetMol()
    try:
        Chem.SanitizeMol(ring)
    except Exception:
        return None
    return Chem.MolToSmiles(ring)
