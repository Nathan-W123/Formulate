"""Vinyl, vinylidene and diene chain growth: the repeat units this chemistry reaches.

The engine could already *score* a repeat unit nobody has tabulated - every
property on the gating path is predicted from structure - but it could only ever
*propose* the 57 units bundled in ``reference_polymers.json``.  This module
proposes the vinyl family, which is the largest of those 57 and the largest
share of world polymer tonnage.

What a grammar for this family has to get right is not the SMILES.  Writing
``[*]CC(R)[*]`` for arbitrary R is trivial and useless: the great majority of
alkenes you can draw do not give a polymer, and a ranking filled with them
buries the candidates that do.  So the generator is built the other way round -
from monomers that are sold, through carriers that are named polymer families,
and every proposal passes structural gates that encode the standard limits on
polymerisability (Odian, "Principles of Polymerization", 4th ed., ch. 3).

The gates are not decoration.  :func:`refusals` returns the proposals the
grammar built and then refused, each with the reason, and the probe set at the
bottom of this module exists so that the refusals are exercised rather than
asserted: allyl alcohol, divinylbenzene, 1,1-diphenylethylene, vinyl bromide,
2-butene, trichloroethylene, hexafluoropropylene, acrolein, 4-vinylphenol,
styrenesulfonic acid and vinyl isocyanate are all built here and all refused.

Three deliberate positions, because each looks like a bug otherwise:

*1,1-disubstitution is not gated on.*  Poly(alpha-methylstyrene) has a ceiling
temperature of 61 C and is still a real polymer in the reference set, and every
methacrylate is 1,1-disubstituted.  What is gated is the specific steric case -
two substituents both bulkier than a methyl, which stops propagation at the
dimer - and even that admits the itaconates, where one arm hangs off a CH2
spacer and the polymer is real.

*1,2-disubstitution is refused except for the fumarates.*  Almost no
1,2-disubstituted alkene homopolymerises.  The dialkyl fumarates do, radically,
to rigid rod-like polymers (Otsu's poly(dialkyl fumarate)s), and they are the
only family here where a name can be put to the polymer.

*Poly(vinyl alcohol), poly(vinylamine) and poly(4-hydroxystyrene) are emitted,
but vinyl alcohol, vinylamine and 4-vinylphenol are not monomers you can buy in
a bottle.*  The first two tautomerise and the third polymerises on standing, so
it is sold only as a dilute solution.  All three polymers are made by
hydrolysing a polymer instead - poly(vinyl acetate), poly(N-vinylformamide),
poly(4-acetoxystyrene) - so they carry ``route="post-polymerisation"``, and the
free-monomer gate refuses all three structures, the vinylphenols included, when
anything proposes them as direct polymerisations.

``availability`` is a fact about the *monomer*; ``product`` is a fact about the
*polymer*.  A catalogue monomer does not make a polymer anybody ships, and a
ranking that cannot tell "you could make this" from "you can buy this" answers
the wrong question.  :data:`COMMERCIAL_POLYMERS` carries the second claim, keyed
to the grade or trade name that makes it checkable.

Elements are confined to the set the density and Tg models were fitted over
(:data:`ELEMENT_BUDGET`), which is why poly(vinyl bromide), poly(methyl vinyl
sulfide) and the styrenesulfonates - all real - are absent and listed as
refusals rather than quietly dropped.

Measured on this installation.  137 repeat units, of which 40 have a commodity
monomer, 94 a catalogue monomer and 3 a monomer you would have to have made; 37
of the 137 are sold as polymers and the other 100 are constructible.  38 of the
57 bundled reference polymers are vinyl-family and all 38 are regenerated,
canonical SMILES against canonical SMILES.  All 26 probes are refused, and none
of them can reach :func:`units` even if one stops being refused.
``PolymerFeasibilityExpert`` scores all 137 and refuses none, median
2.00 on its 1-to-10 scale, 120 at or below 3.0; the ten above 3.5 are the two
exceptions this module argues for (the fumarates charged 3.5 for
1,2-substitution, the itaconates 3.0 for gem bulk), alpha-methylstyrene's
ceiling temperature, PTFE's monomer, and the isobornyl esters' terpene.  Those
charges are the expert disagreeing with the exception, not agreeing with it,
and that disagreement is worth leaving visible.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from typing import Any, Iterator

#: Elements the downstream models were fitted over.  This mirrors
#: ``formulate.experts.polymer._FITTED_ELEMENTS``; it is restated rather than
#: imported because a private name is not an interface, and a test asserts the
#: two are equal so that drift is a failure rather than a silent widening.
#: Generating outside it would hand the ranker a density and a Tg from an
#: extrapolation nobody measured, which is worse than not proposing the polymer.
ELEMENT_BUDGET = frozenset({"H", "C", "N", "O", "F", "Cl"})

#: How hard the monomer is to get hold of, worst to best read right to left.
#: "commodity" means megatonne scale, "commercial" means a catalogue item in
#: kilogram quantities, "literature" means it has been made and polymerised and
#: published but you would be ordering a custom synthesis.  A search that
#: returns a repeat unit whose monomer nobody sells has answered a different
#: question than the one asked, so this is carried on every record.
AVAILABILITY_ORDER: tuple[str, ...] = ("commodity", "commercial", "literature")

#: A substituent branch of this many heavy atoms or fewer counts as "small" for
#: the gem-disubstitution gate: one heavy atom is a methyl or a halogen.  The
#: number is the line Odian draws (§3-9b): propagation survives a methyl on the
#: same carbon as almost anything, and fails when both groups are larger.
_SMALL_BRANCH_HEAVY_ATOMS = 1


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RepeatUnit:
    """One proposal: the polymer, the monomer you would buy, and how to get it."""

    name: str
    smiles: str
    monomer: str
    family: str
    availability: str
    #: "direct" means the monomer polymerises to this unit.  "post-polymerisation"
    #: means the chain is made from a different monomer and converted afterwards,
    #: which is the only honest way to write poly(vinyl alcohol).
    route: str = "direct"
    #: The product that makes this a polymer somebody *sells*, or "" for a unit
    #: that is merely constructible.  :data:`availability` is a fact about the
    #: monomer and answers a different question: 2-methylstyrene and vinyl
    #: laurate are both catalogue monomers, and neither poly(2-methylstyrene)
    #: nor poly(vinyl laurate) is an article of commerce.  A ranking that cannot
    #: tell the two apart will happily return the second as "the best polymer".
    product: str = ""
    note: str = ""


@dataclass(frozen=True, slots=True)
class Refusal:
    """A structure the grammar built and then refused, with the reason."""

    name: str
    smiles: str
    gate: str
    reason: str


def _rdkit() -> Any:
    from rdkit import Chem

    return Chem


def _weaker(first: str, second: str) -> str:
    return max((first, second), key=AVAILABILITY_ORDER.index)


# --------------------------------------------------------------------------
# Substituents
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Group:
    """A substituent, written as it appears inside the branch of a repeat unit.

    ``alpha_methyl`` carries the (polymer, monomer) names of the 1,1-disubstituted
    form with a methyl on the same carbon, or ``None`` where that monomer has no
    polymer anyone can name - which is the commonest case.  Naming both forms by
    hand rather than deriving the second one is the point: it is what keeps
    "methacrylate" from turning into "isopropenyl methyl ether".
    """

    smiles: str
    kind: str
    polymer: str
    monomer: str
    availability: str
    alpha_methyl: tuple[str, str] | None = None
    alpha_methyl_availability: str = "commercial"
    note: str = ""


#: Alkyl substituents: the polyolefins.  Radical initiation gives only oligomer
#: here - the allylic hydrogen transfers - so the 1-alkenes need a Ziegler-Natta
#: or metallocene catalyst and the two 1,1-dialkyl members of the alpha-methyl
#: column, isobutylene and 2-methyl-1-butene, need a cationic one instead, since
#: coordination catalysis does not touch a 1,1-dialkylethylene.  Which of the
#: two it is is a fact about the plant rather than about the structure, so
#: neither is charged.  2-ethylhexyl is deliberately absent: it appears below as
#: an ester alcohol, because the alkene that would put it directly on the
#: backbone is 3-ethyl-1-heptene, which nobody sells.
_ALKYL: tuple[Group, ...] = (
    Group("C", "alkyl", "polypropylene", "propylene", "commodity",
          ("polyisobutylene", "isobutylene"), "commodity"),
    Group("CC", "alkyl", "poly(1-butene)", "1-butene", "commodity",
          ("poly(2-methyl-1-butene)", "2-methyl-1-butene"), "commercial"),
    Group("CCC", "alkyl", "poly(1-pentene)", "1-pentene", "commercial"),
    Group("CCCC", "alkyl", "poly(1-hexene)", "1-hexene", "commodity"),
    Group("CCCCCC", "alkyl", "poly(1-octene)", "1-octene", "commodity"),
    Group("CCCCCCCC", "alkyl", "poly(1-decene)", "1-decene", "commodity",
          note="oligomeric poly(1-decene) is the polyalphaolefin base oil"),
    Group("C(C)C", "alkyl", "poly(3-methyl-1-butene)", "3-methyl-1-butene", "commercial"),
    Group("CC(C)C", "alkyl", "poly(4-methyl-1-pentene)", "4-methyl-1-pentene", "commodity"),
    Group("C(C)(C)C", "alkyl", "poly(3,3-dimethyl-1-butene)", "3,3-dimethyl-1-butene",
          "commercial"),
    Group("C1CCCCC1", "alkyl", "poly(vinylcyclohexane)", "vinylcyclohexane", "commercial"),
)

#: Styrenics.  Ring substitution is where the family earns its breadth: it moves
#: Tg and surface energy without touching the polymerisation, which is exactly
#: the kind of axis an inverse design engine should be free to sweep.  Only the
#: alpha-methyl form of styrene itself is named, because 4-methyl-alpha-
#: methylstyrene and its relatives are monomers without polymers.
_STYRENIC: tuple[Group, ...] = (
    Group("c1ccccc1", "aryl", "polystyrene", "styrene", "commodity",
          ("poly(alpha-methylstyrene)", "alpha-methylstyrene"), "commercial"),
    Group("c1ccc(C)cc1", "aryl", "poly(4-methylstyrene)", "4-methylstyrene", "commodity"),
    Group("c1cccc(C)c1", "aryl", "poly(3-methylstyrene)", "3-methylstyrene", "commercial"),
    Group("c1ccccc1C", "aryl", "poly(2-methylstyrene)", "2-methylstyrene", "commercial"),
    Group("c1ccc(C(C)(C)C)cc1", "aryl", "poly(4-tert-butylstyrene)", "4-tert-butylstyrene",
          "commercial"),
    Group("c1ccc(Cl)cc1", "aryl", "poly(4-chlorostyrene)", "4-chlorostyrene", "commercial"),
    Group("c1ccc(F)cc1", "aryl", "poly(4-fluorostyrene)", "4-fluorostyrene", "commercial"),
    Group("c1c(F)c(F)c(F)c(F)c1F", "aryl", "poly(pentafluorostyrene)", "pentafluorostyrene",
          "commercial"),
    Group("c1ccc(OC)cc1", "aryl", "poly(4-methoxystyrene)", "4-methoxystyrene", "commercial"),
    Group("c1ccc(OC(C)=O)cc1", "aryl", "poly(4-acetoxystyrene)", "4-acetoxystyrene",
          "commercial",
          note="hydrolysed after polymerisation to poly(4-hydroxystyrene), the photoresist"),
    Group("c1ccc2ccccc2c1", "aryl", "poly(2-vinylnaphthalene)", "2-vinylnaphthalene",
          "commercial"),
    Group("c1ccccn1", "aryl", "poly(2-vinylpyridine)", "2-vinylpyridine", "commercial"),
    Group("c1ccncc1", "aryl", "poly(4-vinylpyridine)", "4-vinylpyridine", "commercial"),
)

#: N-vinyl monomers: the nitrogen is the attachment, so there is no allylic
#: hydrogen and no enamine.  Poly(N-vinylpyrrolidone) is the second-largest
#: water-soluble synthetic polymer after poly(acrylamide); the rest are real
#: and sold.  N-vinylformamide is here because hydrolysing its polymer is the
#: industrial route to poly(vinylamine).
_N_VINYL: tuple[Group, ...] = (
    Group("N1CCCC1=O", "n-vinyl", "poly(N-vinylpyrrolidone)", "N-vinylpyrrolidone", "commodity"),
    Group("N1CCCCCC1=O", "n-vinyl", "poly(N-vinylcaprolactam)", "N-vinylcaprolactam",
          "commercial"),
    Group("n1c2ccccc2c2ccccc21", "n-vinyl", "poly(N-vinylcarbazole)", "N-vinylcarbazole",
          "commercial"),
    Group("n1ccnc1", "n-vinyl", "poly(1-vinylimidazole)", "1-vinylimidazole", "commercial"),
    Group("NC=O", "n-vinyl", "poly(N-vinylformamide)", "N-vinylformamide", "commercial"),
    Group("NC(C)=O", "n-vinyl", "poly(N-vinylacetamide)", "N-vinylacetamide", "commercial"),
)

#: Halogen, nitrile, acid, ketone and the amides.  The alpha-methyl column here
#: is where the methacrylic family other than the esters comes from.  2-halo-
#: propenes are absent on purpose: there is no poly(2-chloropropene).
_FUNCTIONAL: tuple[Group, ...] = (
    Group("Cl", "halogen", "poly(vinyl chloride)", "vinyl chloride", "commodity"),
    Group("F", "halogen", "poly(vinyl fluoride)", "vinyl fluoride", "commodity"),
    Group("C#N", "nitrile", "polyacrylonitrile", "acrylonitrile", "commodity",
          ("poly(methacrylonitrile)", "methacrylonitrile"), "commercial"),
    Group("C(=O)O", "acid", "poly(acrylic acid)", "acrylic acid", "commodity",
          ("poly(methacrylic acid)", "methacrylic acid"), "commodity"),
    Group("C(N)=O", "amide", "polyacrylamide", "acrylamide", "commodity",
          ("polymethacrylamide", "methacrylamide"), "commercial"),
    Group("C(=O)NC", "amide", "poly(N-methylacrylamide)", "N-methylacrylamide", "commercial",
          ("poly(N-methylmethacrylamide)", "N-methylmethacrylamide"), "literature"),
    Group("C(=O)N(C)C", "amide", "poly(N,N-dimethylacrylamide)", "N,N-dimethylacrylamide",
          "commercial",
          ("poly(N,N-dimethylmethacrylamide)", "N,N-dimethylmethacrylamide"), "commercial"),
    Group("C(=O)NC(C)C", "amide", "poly(N-isopropylacrylamide)", "N-isopropylacrylamide",
          "commercial",
          ("poly(N-isopropylmethacrylamide)", "N-isopropylmethacrylamide"), "commercial"),
    Group("C(=O)NC(C)(C)C", "amide", "poly(N-tert-butylacrylamide)", "N-tert-butylacrylamide",
          "commercial",
          ("poly(N-tert-butylmethacrylamide)", "N-tert-butylmethacrylamide"), "commercial"),
    Group("C(=O)NCCO", "amide", "poly(N-(2-hydroxyethyl)acrylamide)",
          "N-(2-hydroxyethyl)acrylamide", "commercial"),
    Group("C(=O)Nc1ccccc1", "amide", "poly(N-phenylacrylamide)", "N-phenylacrylamide",
          "literature"),
    Group("C(=O)NC(C)(C)CC(C)=O", "amide", "poly(diacetone acrylamide)",
          "diacetone acrylamide", "commercial"),
    Group("C(C)=O", "ketone", "poly(methyl vinyl ketone)", "methyl vinyl ketone", "commercial",
          ("poly(methyl isopropenyl ketone)", "methyl isopropenyl ketone"), "commercial"),
    Group("C(=O)c1ccccc1", "ketone", "poly(phenyl vinyl ketone)", "phenyl vinyl ketone",
          "literature"),
)


# --------------------------------------------------------------------------
# Carriers: one functional group, many alcohols or acids
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Residue:
    """An alcohol or acid residue hung off a carrier group."""

    smiles: str
    name: str
    availability: str
    carriers: frozenset[str]


#: Alcohol residues.  Every one of these is an alcohol sold by the tonne or by
#: the kilogram, and every (residue, carrier) pair below is a monomer in a
#: catalogue - which is the whole reason the acrylate and methacrylate families
#: are generated combinatorially while the styrenics are not: esterification is
#: the one step in this family that really is a free variable.
_ESTER = "ester"
_ETHER = "ether"
_ALCOHOLS: tuple[Residue, ...] = (
    Residue("C", "methyl", "commodity", frozenset({_ESTER, _ETHER})),
    Residue("CC", "ethyl", "commodity", frozenset({_ESTER, _ETHER})),
    Residue("CCC", "n-propyl", "commercial", frozenset({_ESTER})),
    Residue("C(C)C", "isopropyl", "commercial", frozenset({_ESTER})),
    Residue("CCCC", "n-butyl", "commodity", frozenset({_ESTER, _ETHER})),
    Residue("CC(C)C", "isobutyl", "commercial", frozenset({_ESTER, _ETHER})),
    Residue("C(C)(C)C", "tert-butyl", "commercial", frozenset({_ESTER})),
    Residue("CCCCCC", "n-hexyl", "commercial", frozenset({_ESTER})),
    Residue("CC(CC)CCCC", "2-ethylhexyl", "commodity", frozenset({_ESTER, _ETHER})),
    Residue("CCCCCCCCCCCC", "lauryl", "commercial", frozenset({_ESTER, _ETHER})),
    Residue("C1CCCCC1", "cyclohexyl", "commercial", frozenset({_ESTER, _ETHER})),
    Residue("Cc1ccccc1", "benzyl", "commercial", frozenset({_ESTER})),
    Residue("c1ccccc1", "phenyl", "commercial", frozenset({_ESTER, _ETHER})),
    Residue("C1CC2CCC1(C)C2(C)C", "isobornyl", "commercial", frozenset({_ESTER})),
    Residue("CCO", "2-hydroxyethyl", "commodity", frozenset({_ESTER, _ETHER})),
    Residue("CC(O)C", "2-hydroxypropyl", "commercial", frozenset({_ESTER})),
    Residue("CC1CO1", "glycidyl", "commodity", frozenset({_ESTER})),
    Residue("CCOC", "2-methoxyethyl", "commercial", frozenset({_ESTER})),
    Residue("CC(F)(F)F", "2,2,2-trifluoroethyl", "commercial", frozenset({_ESTER})),
    Residue("CCN(C)C", "2-(dimethylamino)ethyl", "commercial", frozenset({_ESTER})),
    Residue("CCCCO", "4-hydroxybutyl", "commercial", frozenset({_ETHER})),
    Residue("CCCl", "2-chloroethyl", "commercial", frozenset({_ETHER})),
)

#: Vinyl esters: the acid is on the oxygen, not the carbon, which inverts the
#: polarity of the monomer and is why vinyl acetate copolymerises with almost
#: nothing that styrene likes.  Written as whole fragments rather than as a
#: template plus residue because vinyl formate has no residue at all.
_VINYL_ESTERS: tuple[tuple[str, str, str], ...] = (
    ("OC=O", "formate", "commercial"),
    ("OC(=O)C", "acetate", "commodity"),
    ("OC(=O)CC", "propionate", "commercial"),
    ("OC(=O)CCC", "butyrate", "commercial"),
    ("OC(=O)C(C)(C)C", "pivalate", "commercial"),
    ("OC(=O)C(CC)CCCC", "2-ethylhexanoate", "commercial"),
    ("OC(=O)c1ccccc1", "benzoate", "commercial"),
    ("OC(=O)CCCCCCCCCCC", "laurate", "commercial"),
    ("OC(=O)CCCCCCCCCCCCCCCCC", "stearate", "commercial"),
    ("OC(=O)C(F)(F)F", "trifluoroacetate", "commercial"),
)


def _carrier_groups() -> Iterator[Group]:
    """Acrylates, methacrylates, vinyl esters and vinyl ethers."""
    for residue in _ALCOHOLS:
        if _ESTER in residue.carriers:
            yield Group(
                f"C(=O)O{residue.smiles}",
                "ester",
                f"poly({residue.name} acrylate)",
                f"{residue.name} acrylate",
                residue.availability,
                (f"poly({residue.name} methacrylate)", f"{residue.name} methacrylate"),
                residue.availability,
            )
        if _ETHER in residue.carriers:
            # Vinyl ethers are cationic monomers: the oxygen lone pair that makes
            # them polymerise at all also makes the radical route useless, so
            # none of them carries an alpha-methyl form - 2-methoxypropene is a
            # real monomer with no polymer to its name.
            yield Group(
                f"O{residue.smiles}",
                "ether",
                f"poly({residue.name} vinyl ether)",
                f"{residue.name} vinyl ether",
                _weaker(residue.availability, "commercial"),
            )
    for fragment, ester, availability in _VINYL_ESTERS:
        yield Group(
            fragment,
            "vinyl-ester",
            f"poly(vinyl {ester})",
            f"vinyl {ester}",
            availability,
        )


# --------------------------------------------------------------------------
# Backbones that are not built from a substituent table
# --------------------------------------------------------------------------

#: Fluoroolefins.  These four are the whole family: the monomers are made by
#: pyrolysis rather than by cracking and polymerised under pressure, and there
#: is no fifth member because hexafluoropropylene's ceiling temperature puts its
#: homopolymer out of reach (it is a comonomer only - FEP, Viton).
_FLUOROPOLYMERS: tuple[RepeatUnit, ...] = (
    RepeatUnit("polytetrafluoroethylene", "[*]C(F)(F)C(F)(F)[*]", "tetrafluoroethylene",
               "fluoroolefin", "commodity"),
    RepeatUnit("poly(vinylidene fluoride)", "[*]CC(F)(F)[*]", "vinylidene fluoride",
               "fluoroolefin", "commodity"),
    RepeatUnit("poly(trifluoroethylene)", "[*]C(F)C(F)(F)[*]", "trifluoroethylene",
               "fluoroolefin", "commercial"),
    RepeatUnit("poly(chlorotrifluoroethylene)", "[*]C(F)(F)C(F)(Cl)[*]",
               "chlorotrifluoroethylene", "fluoroolefin", "commercial"),
)

#: Vinylidene chloride is its own entry rather than an alpha-chloro column on
#: the halogen groups, because the 1,1-dihalide is the only gem-dihalogen that
#: polymerises: 1-chloro-1-fluoroethylene has no polymer.
_VINYLIDENE_CHLORIDE = RepeatUnit(
    "poly(vinylidene chloride)", "[*]CC(Cl)(Cl)[*]", "vinylidene chloride",
    "vinylidene halide", "commodity",
)

#: The 1,2-disubstituted exception.  Dialkyl fumarates homopolymerise radically
#: to rigid, rod-like chains with very high Tg (Otsu); the maleates, their cis
#: isomers, do not, which is a stereochemical distinction this SMILES convention
#: does not record and so is a stated limitation rather than a gate.
_FUMARATES: tuple[tuple[str, str, str], ...] = (
    ("C", "dimethyl", "commodity"),
    ("CC", "diethyl", "commercial"),
    ("C(C)C", "diisopropyl", "commercial"),
    ("CCCC", "di-n-butyl", "commercial"),
)

#: The gem-bulky exception.  Both arms of an itaconate are esters, which the
#: steric gate would normally refuse, and the polymers are nonetheless real:
#: one arm hangs off a CH2 spacer, one carbon further from the backbone, which
#: is the whole reason propagation survives.  Itaconic acid is fermented from
#: sugar, so these are also the bio-derived members of the family.
_ITACONATES: tuple[tuple[str, str, str], ...] = (
    ("C", "dimethyl", "commercial"),
    ("CCCC", "di-n-butyl", "commercial"),
)

#: Dienes.  1,4 addition keeps one double bond in the backbone, which is what
#: makes these the rubbers - low Tg, and a site for sulfur cure.  The 1,2 and
#: 3,4 forms put that double bond in the side group instead and are real,
#: commercial polymers in their own right (syndiotactic 1,2-polybutadiene;
#: 3,4-polyisoprene in tyre tread).  Geometry is not written: the reference set
#: writes cis-1,4-polybutadiene as [*]CC=CC[*] with no stereo, and matching it
#: matters more than a stereo flag no downstream model reads.
_DIENES: tuple[RepeatUnit, ...] = (
    RepeatUnit("1,4-polybutadiene", "[*]CC=CC[*]", "1,3-butadiene", "diene", "commodity"),
    RepeatUnit("1,4-polyisoprene", "[*]CC(C)=CC[*]", "isoprene", "diene", "commodity"),
    RepeatUnit("polychloroprene", "[*]CC(Cl)=CC[*]", "chloroprene", "diene", "commodity"),
    RepeatUnit("1,4-poly(2,3-dimethylbutadiene)", "[*]CC(C)=C(C)C[*]", "2,3-dimethylbutadiene",
               "diene", "commercial", note="the original methyl rubber"),
    RepeatUnit("1,4-poly(1,3-pentadiene)", "[*]C(C)C=CC[*]", "1,3-pentadiene (piperylene)",
               "diene", "commercial"),
    RepeatUnit("1,2-polybutadiene", "[*]CC(C=C)[*]", "1,3-butadiene", "diene", "commodity",
               note="the pendant vinyl is what makes this one curable and crosslinkable"),
    RepeatUnit("3,4-polyisoprene", "[*]CC(C(C)=C)[*]", "isoprene", "diene", "commodity"),
)

#: Polymers made by converting a chain rather than by polymerising their own
#: monomer.  Each is a commodity; none of the three free monomers exists as an
#: isolable species (vinyl alcohol and vinylamine tautomerise, 4-vinylphenol
#: polymerises in the bottle).  Emitting them with the route recorded is the
#: difference between proposing a real material and proposing a monomer that
#: cannot be bought.
_POST_POLYMERISATION: tuple[RepeatUnit, ...] = (
    RepeatUnit("poly(vinyl alcohol)", "[*]CC(O)[*]", "vinyl acetate", "post-polymerisation",
               "commodity", route="post-polymerisation",
               note="made by hydrolysing poly(vinyl acetate); vinyl alcohol tautomerises "
                    "to acetaldehyde and is not a monomer"),
    RepeatUnit("poly(vinylamine)", "[*]CC(N)[*]", "N-vinylformamide", "post-polymerisation",
               "commercial", route="post-polymerisation",
               note="made by hydrolysing poly(N-vinylformamide); free vinylamine "
                    "tautomerises to acetaldimine"),
    RepeatUnit("poly(4-hydroxystyrene)", "[*]CC(c1ccc(O)cc1)[*]", "4-acetoxystyrene",
               "post-polymerisation", "commercial", route="post-polymerisation",
               note="made by hydrolysing poly(4-acetoxystyrene); 4-vinylphenol "
                    "polymerises spontaneously"),
)


#: The units that are sold *as polymers*, each keyed to the product that makes
#: the claim checkable rather than to a bare True.  The bar is deliberately one
#: a reader can argue with: the homopolymer is bought under a grade or a trade
#: name.  Everything not listed here is constructible from a monomer somebody
#: sells, which is a much weaker claim and is the one :data:`availability`
#: makes.  Absence is therefore not a statement that the polymer is unreal -
#: poly(N-isopropylacrylamide) and the poly(dialkyl fumarate)s are real and are
#: not listed - only that nobody ships it.
COMMERCIAL_POLYMERS: dict[str, str] = {
    "polyethylene": "every grade from LDPE to UHMWPE",
    "polypropylene": "isotactic PP",
    "polyisobutylene": "BASF Oppanol B; butyl rubber is its copolymer with isoprene",
    "poly(1-butene)": "isotactic PB-1, LyondellBasell Toppyl, hot-water pipe",
    "poly(4-methyl-1-pentene)": "Mitsui TPX",
    "poly(1-decene)": "the polyalphaolefin base oils, sold as oligomer rather than as "
                      "high polymer",
    "polystyrene": "GPPS and HIPS",
    "poly(alpha-methylstyrene)": "the AMS tackifier resins (Eastman Kristalex)",
    "poly(4-hydroxystyrene)": "the 248 nm photoresist resin",
    "poly(vinyl chloride)": "PVC",
    "poly(vinylidene chloride)": "Saran",
    "poly(vinyl fluoride)": "Tedlar film",
    "poly(vinylidene fluoride)": "Arkema Kynar, Solvay Solef",
    "polytetrafluoroethylene": "Teflon PTFE",
    "poly(chlorotrifluoroethylene)": "Daikin Neoflon PCTFE, 3M Kel-F",
    "polyacrylonitrile": "acrylic fibre and the carbon-fibre precursor",
    "poly(acrylic acid)": "Lubrizol Carbopol; the superabsorbents are its crosslinked salt",
    "polyacrylamide": "the SNF and Kemira flocculants",
    "poly(methyl methacrylate)": "Perspex, Plexiglas, Lucite",
    "poly(ethyl methacrylate)": "Lucite Elvacite 2042",
    "poly(n-butyl methacrylate)": "Lucite Elvacite 2044",
    "poly(isobutyl methacrylate)": "Lucite Elvacite 2045",
    "poly(2-hydroxyethyl methacrylate)": "the poly-HEMA soft contact lens hydrogel",
    "poly(vinyl acetate)": "the wood-glue and gum-base homopolymer",
    "poly(vinyl alcohol)": "Kuraray Poval, Sekisui Selvol",
    "poly(vinylamine)": "BASF Lupamin",
    "poly(N-vinylpyrrolidone)": "Ashland PVP K-grades, BASF Luvitec",
    "poly(N-vinylcaprolactam)": "BASF Luvicap",
    "poly(N-vinylcarbazole)": "Luvican M170, the photoconductor",
    "poly(methyl vinyl ether)": "BASF Lutonal M",
    "poly(ethyl vinyl ether)": "BASF Lutonal A",
    "poly(isobutyl vinyl ether)": "BASF Lutonal I",
    "1,4-polybutadiene": "BR, the tyre and HIPS-toughening rubber",
    "1,4-polyisoprene": "synthetic IR (Goodyear Natsyn); natural rubber is the same chain",
    "polychloroprene": "Neoprene, Arlanxeo Baypren",
    "1,2-polybutadiene": "JSR RB, the syndiotactic 1,2 grade",
    "3,4-polyisoprene": "the high-3,4 tread rubbers",
}


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------

#: Monomers that look constructible, are real compounds, and have no
#: homopolymer.  Each is keyed by the repeat unit that would be written for it.
#: A curated refusal is still a gate: these are the cases no structural rule
#: reaches, because what stops them is a ceiling temperature or a degenerative
#: transfer step rather than a bond count.
_NO_HOMOPOLYMER: dict[str, str] = {
    "[*]C(F)(F)C(F)(C(F)(F)F)[*]": (
        "hexafluoropropylene has a ceiling temperature near ambient and gives no "
        "homopolymer; it is a comonomer only (FEP, Viton)"
    ),
    "[*]CC(C)(OC)[*]": (
        "2-methoxypropene is a real cationic monomer with no polymer to its name: the "
        "oxocarbenium it forms is too stable to propagate"
    ),
    "[*]CC(C)(Cl)[*]": (
        "2-chloropropene does not homopolymerise; the allylic hydrogens transfer and "
        "there is no poly(2-chloropropene)"
    ),
}

#: SMARTS for groups that destroy the monomer before it can be polymerised, each
#: paired with the reason.  These are not "unstable molecules" in general - an
#: acid chloride sits happily in a bottle - they are groups whose reaction with
#: the initiator, the solvent or their own neighbour outruns propagation.
_SELF_REACTIVE: tuple[tuple[str, str], ...] = (
    ("[CX3H1](=O)[#6]", "an aldehyde polymerises through the carbonyl as well as the "
                        "alkene, so the product is a crosslinked mixture rather than a "
                        "vinyl chain (acrolein)"),
    ("[CX3](=O)[Cl,Br,I]", "an acyl halide hydrolyses on contact with the water any "
                           "emulsion or suspension polymerisation runs in"),
    ("[NX2]=[CX2]=[OX1]", "an isocyanate reacts with every protic species present and "
                          "polymerises through the N=C bond in competition with the alkene"),
    ("[CX3](=O)[OX2][CX3]=[OX1]", "an anhydride acylates the initiator and hydrolyses"),
    ("[OX2][OX2]", "a peroxide is an initiator, not a monomer"),
    ("[CX2]#[CX2H1]", "a terminal alkyne adds to the propagating radical and terminates it "
                      "(vinylacetylene is a chain-transfer agent, not a comonomer)"),
    ("[CX2]=[CX2]=[*]", "a cumulated diene is not a monomer"),
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


def _off_path(mol: Any, idx: int, path: tuple[int, ...]) -> list[Any]:
    on_path = set(path)
    return [
        n
        for n in mol.GetAtomWithIdx(idx).GetNeighbors()
        if n.GetIdx() not in on_path and n.GetAtomicNum() != 0
    ]


def _branch_heavy_atoms(mol: Any, root: int, exclude: int) -> int:
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


def _is_ester_carbonyl(mol: Any, atom: Any) -> bool:
    Chem = _rdkit()
    if atom.GetAtomicNum() != 6:
        return False
    double_o = single_o = False
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetAtomicNum() != 8:
            continue
        if bond.GetBondType() is Chem.BondType.DOUBLE:
            double_o = True
        elif bond.GetBondType() is Chem.BondType.SINGLE:
            single_o = True
    return double_o and single_o


def _is_itaconate_arm(mol: Any, atom: Any, root: int) -> bool:
    """A -CH2-ester arm: bulk held one carbon away from the backbone.

    This is the whole of the itaconate exception to the gem-steric gate, and it
    is deliberately narrow.  What the alpha carbon of a propagating chain end
    sees in dimethyl itaconate is a methylene, exactly as it does in methyl
    methacrylate; the ester that would block it sits a bond further out.  An
    ethyl group also presents a methylene and is *not* an itaconate arm, which
    is why the far end has to be the ester: 2-ethyl-1-butene has no polymer.
    """
    if atom.GetAtomicNum() != 6 or atom.GetTotalNumHs() != 2:
        return False
    beyond = [
        n
        for n in atom.GetNeighbors()
        if n.GetIdx() != root and n.GetAtomicNum() not in (0, 1)
    ]
    return len(beyond) == 1 and _is_ester_carbonyl(mol, beyond[0])


def _gate_structure(unit: RepeatUnit, mol: Any) -> tuple[str, str] | None:
    """Two attachment points, a backbone between them, and a cappable valence."""
    Chem = _rdkit()
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        return "structure", (
            f"a chain repeat unit needs exactly two [*]; this one has {len(dummies)}"
        )
    if _backbone_termini(mol) is None:
        return "structure", "the two attachment points are not joined by a path"
    capped = Chem.MolFromSmiles(unit.smiles.replace("[*]", "[H]"))
    if capped is None:
        return "structure", "the unit does not parse once the attachment points are capped"
    return None


def _gate_elements(unit: RepeatUnit, mol: Any) -> tuple[str, str] | None:
    Chem = _rdkit()
    present = {a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms()} - {"*"}
    foreign = present - ELEMENT_BUDGET
    if foreign:
        return "elements", (
            f"contains {', '.join(sorted(foreign))}, outside the element set the density and "
            "Tg models were fitted over; the polymer may well be real, but its predicted "
            "properties would be an extrapolation nobody measured"
        )
    return None


def _acylated(mol: Any, atom: Any) -> bool:
    """True when this heteroatom hangs off a carbonyl - an amide or an ester.

    This is the line between an enamine and an enamide, and it is the whole
    reason N-vinylformamide and N-vinylpyrrolidone are monomers while vinylamine
    is not: delocalising the lone pair into the carbonyl removes the tautomer.
    """
    Chem = _rdkit()
    for neighbour in atom.GetNeighbors():
        if neighbour.GetAtomicNum() != 6:
            continue
        for bond in neighbour.GetBonds():
            other = bond.GetOtherAtom(neighbour)
            if bond.GetBondType() is Chem.BondType.DOUBLE and other.GetAtomicNum() == 8:
                return True
    return False


def _aromatic_hydroxyl(mol: Any, atom: Any) -> bool:
    """True when this ring atom belongs to a ring system carrying a free phenol.

    The ring is walked rather than pattern-matched on one atom because the
    hydroxyl of a vinylphenol need not sit on the carbon the vinyl is attached
    to - ortho, meta and para are all the same monomer problem - and because a
    fused system (a vinylnaphthol) is the same problem again.
    """
    seen: set[int] = set()
    stack = [atom.GetIdx()]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        current = mol.GetAtomWithIdx(idx)
        for bond in current.GetBonds():
            neighbour = bond.GetOtherAtom(current)
            # Aromatic *bonds*, not aromatic atoms: the single bond of a
            # biphenyl joins two ring systems, and a hydroxyl on the far ring of
            # one is a different molecule's problem.
            if bond.GetIsAromatic():
                stack.append(neighbour.GetIdx())
            elif neighbour.GetAtomicNum() == 8 and neighbour.GetTotalNumHs() > 0:
                return True
    return False


def _gate_free_monomer(unit: RepeatUnit, mol: Any) -> tuple[str, str] | None:
    """An -OH or a free amine on the backbone means the monomer is an enol or enamine."""
    if unit.route == "post-polymerisation":
        return None
    path = _backbone_termini(mol)
    if path is None:
        return None
    for idx in path:
        for neighbour in _off_path(mol, idx, path):
            if neighbour.GetIsAromatic() and _aromatic_hydroxyl(mol, neighbour):
                return "free-monomer", (
                    "the monomer would be a vinylphenol. The ring hydroxyl initiates the "
                    "monomer's own double bond, so 4-vinylphenol polymerises in the bottle "
                    "and is sold only as a dilute solution, never neat; the real chain is "
                    "made by polymerising 4-acetoxystyrene and hydrolysing it afterwards, "
                    "which is what the post-polymerisation records in this module say"
                )
            element = neighbour.GetAtomicNum()
            if element == 8 and neighbour.GetTotalNumHs() > 0:
                kind = "an enol"
            elif (
                element == 7
                and not neighbour.GetIsAromatic()
                and not _acylated(mol, neighbour)
            ):
                kind = "an enamine"
            else:
                continue
            return "free-monomer", (
                f"the monomer would be {kind}, which tautomerises rather than sitting in a "
                "bottle: vinyl alcohol becomes acetaldehyde and vinylamine becomes "
                "acetaldimine. The real chains are made by hydrolysing a polymer, which is "
                "what the post-polymerisation records in this module say"
            )
    return None


def _gate_allylic(unit: RepeatUnit, mol: Any) -> tuple[str, str] | None:
    """An sp3 C-H carrying a heteroatom, directly on the backbone: an allyl monomer."""
    path = _backbone_termini(mol)
    if path is None:
        return None
    for idx in path:
        for neighbour in _off_path(mol, idx, path):
            if neighbour.GetAtomicNum() != 6 or neighbour.GetHybridization().name != "SP3":
                continue
            # No hydrogen, no transfer: a perfluoroalkyl on the double bond is not
            # an allyl monomer, whatever else is wrong with it.
            if neighbour.GetTotalNumHs() == 0:
                continue
            hetero = [
                n.GetSymbol()
                for n in neighbour.GetNeighbors()
                if n.GetAtomicNum() not in (1, 6, 0)
            ]
            if hetero:
                return "allylic", (
                    "the monomer is an allyl compound: the allylic C-H next to the "
                    f"{hetero[0]} transfers to the propagating radical faster than the "
                    "double bond adds, so degradative chain transfer caps the product at "
                    "oligomer (Odian §3-9c). Allyl alcohol, allyl acetate and allyl "
                    "chloride all fail this way"
                )
    return None


def _gate_pendant_alkene(unit: RepeatUnit, mol: Any) -> tuple[str, str] | None:
    """A second polymerisable double bond that is not conjugated to the first."""
    Chem = _rdkit()
    path = _backbone_termini(mol)
    if path is None:
        return None
    on_path = set(path)
    attached = on_path | {
        n.GetIdx() for idx in path for n in mol.GetAtomWithIdx(idx).GetNeighbors()
    }
    for bond in mol.GetBonds():
        if bond.GetBondType() is not Chem.BondType.DOUBLE or bond.GetIsAromatic():
            continue
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        if begin.GetAtomicNum() != 6 or end.GetAtomicNum() != 6:
            continue
        if begin.GetIdx() in attached or end.GetIdx() in attached:
            # Hanging directly off a backbone carbon, so in the monomer it was
            # conjugated with the double bond that became the backbone: this is a
            # 1,3-diene polymerised 1,2, which is what 1,2-polybutadiene and
            # 3,4-polyisoprene are, and the chain is still linear.
            continue
        return "pendant-alkene", (
            "a second vinyl group sits too far from the backbone to be the other half of a "
            "1,3-diene, so the monomer is difunctional and the product is a network rather "
            "than a linear chain (divinylbenzene, allyl methacrylate, the glycol "
            "dimethacrylates). This explorer proposes linear chains"
        )
    return None


def _gate_self_reactive(unit: RepeatUnit, mol: Any) -> tuple[str, str] | None:
    Chem = _rdkit()
    for smarts, reason in _SELF_REACTIVE:
        query = Chem.MolFromSmarts(smarts)
        if query is not None and mol.HasSubstructMatch(query):
            return "self-reactive", reason
    path = _backbone_termini(mol)
    if path is not None:
        ring_info = mol.GetRingInfo()
        for idx in path:
            for neighbour in _off_path(mol, idx, path):
                for ring in ring_info.AtomRings():
                    if neighbour.GetIdx() in ring and len(ring) == 3 and any(
                        mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring
                    ):
                        return "self-reactive", (
                            "an epoxide directly on the double bond opens under any cationic "
                            "or anionic initiator before the alkene adds; butadiene monoxide "
                            "polymerises through the ring, not the vinyl"
                        )
    return None


def _gate_substitution(unit: RepeatUnit, mol: Any) -> tuple[str, str] | None:
    """The substitution pattern on the two carbons the monomer's double bond became."""
    path = _backbone_termini(mol)
    if path is None or len(path) != 2:
        # A diene backbone is four atoms long and is checked by its own gate.
        return None
    halogens = {9, 17, 35, 53}
    groups: dict[int, list[Any]] = {idx: _off_path(mol, idx, path) for idx in path}
    heavy = {
        idx: [a for a in group if a.GetAtomicNum() not in halogens]
        for idx, group in groups.items()
    }
    halide = {
        idx: [a for a in group if a.GetAtomicNum() in halogens] for idx, group in groups.items()
    }

    left, right = path
    if groups[left] and groups[right]:
        every = groups[left] + groups[right]
        if all(a.GetAtomicNum() in halogens for a in every):
            chlorines = sum(1 for a in every if a.GetAtomicNum() == 17)
            if chlorines <= 1 and all(a.GetAtomicNum() in (9, 17) for a in every):
                return None  # PTFE, poly(trifluoroethylene), PCTFE
            return "substitution", (
                "a chlorine on each backbone carbon makes the monomer a 1,2-dihaloethylene; "
                "trichloroethylene and 1,2-dichloroethylene are chain-transfer agents, not "
                "monomers"
            )
        if (
            len(heavy[left]) == 1
            and len(heavy[right]) == 1
            and not halide[left]
            and not halide[right]
            and all(_is_ester_carbonyl(mol, a) for a in heavy[left] + heavy[right])
        ):
            return None  # the dialkyl fumarates
        return "substitution", (
            "each backbone carbon carries a substituent, so the monomer is a "
            "1,2-disubstituted alkene; those do not homopolymerise - the steric barrier to "
            "addition plus a small enthalpy of polymerisation (Odian §3-9b) - and the only "
            "family that does is the dialkyl fumarates"
        )

    crowded = left if groups[left] else right
    if len(heavy[crowded]) + len(halide[crowded]) > 2:
        return "substitution", (
            "three or more substituents on one backbone carbon is not a valence a "
            "monosubstituted alkene can reach"
        )
    if len(heavy[crowded]) == 2:
        sizes = [
            _branch_heavy_atoms(mol, a.GetIdx(), crowded) for a in heavy[crowded]
        ]
        spaced = [a for a in heavy[crowded] if _is_itaconate_arm(mol, a, crowded)]
        if min(sizes) > _SMALL_BRANCH_HEAVY_ATOMS and not spaced:
            return "substitution", (
                "both substituents on one backbone carbon are bulkier than a methyl, so "
                "propagation stops at the dimer (Odian §3-9b): 1,1-diphenylethylene is sold as "
                "a chain-transfer probe for exactly this reason. Note this is not a charge on "
                "1,1-disubstitution itself - alpha-methylstyrene and every methacrylate pass - "
                "and an arm held one carbon away by a CH2, as in the itaconates, passes too"
            )
    return None


def _gate_diene(unit: RepeatUnit, mol: Any) -> tuple[str, str] | None:
    """A four-atom backbone has to be the 1,4 residue of a conjugated diene."""
    Chem = _rdkit()
    path = _backbone_termini(mol)
    if path is None or len(path) != 4:
        return None
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in path):
        return "diene", "a four-atom backbone here has to be all carbon to be a diene residue"
    orders = [mol.GetBondBetweenAtoms(path[i], path[i + 1]).GetBondType() for i in range(3)]
    if orders != [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.SINGLE]:
        return "diene", (
            "the four-atom backbone does not carry the single-double-single pattern 1,4 "
            "addition of a 1,3-diene leaves behind"
        )
    for idx in (path[0], path[3]):
        if len(_off_path(mol, idx, path)) > 1:
            return "diene", (
                "a 1,4-diene residue whose terminal carbon carries two substituents implies a "
                "tetrasubstituted diene, which does not polymerise"
            )
    return None


#: Every gate, in the order a chemist would apply them: can it be written, is it
#: in the element budget, does the monomer exist, does it survive its own
#: functional groups, does the substitution pattern let a chain grow.
_GATES = (
    _gate_structure,
    _gate_elements,
    _gate_free_monomer,
    _gate_self_reactive,
    _gate_allylic,
    _gate_pendant_alkene,
    _gate_substitution,
    _gate_diene,
)


@functools.lru_cache(maxsize=1)
def _no_homopolymer() -> dict[str, str]:
    """:data:`_NO_HOMOPOLYMER`, keyed by canonical SMILES rather than by how it is written."""
    out: dict[str, str] = {}
    for smiles, reason in _NO_HOMOPOLYMER.items():
        key = canonical(smiles)
        if key is None:  # pragma: no cover - the table is checked by a test
            raise ValueError(f"the refusal table entry {smiles!r} does not parse")
        out[key] = reason
    return out


def gate(unit: RepeatUnit) -> Refusal | None:
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


def canonical(smiles: str) -> str | None:
    """RDKit canonical SMILES, for comparing a generated unit with a bundled one."""
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


# --------------------------------------------------------------------------
# Probes: structures built here only so that the gates are exercised
# --------------------------------------------------------------------------

#: Tempting structures the grammar must refuse.  They are carried in the module
#: rather than in the test file because a gate with no counterexample beside it
#: is a claim, and because the list doubles as the argument for what is absent:
#: every one of these is a repeat unit some other vinyl grammar would emit.
PROBES: tuple[RepeatUnit, ...] = (
    RepeatUnit("poly(vinyl alcohol) proposed as a direct polymerisation", "[*]CC(O)[*]",
               "vinyl alcohol", "probe", "literature"),
    RepeatUnit("poly(vinylamine) proposed as a direct polymerisation", "[*]CC(N)[*]",
               "vinylamine", "probe", "literature"),
    RepeatUnit("poly(4-hydroxystyrene) proposed as a direct polymerisation",
               "[*]CC(c1ccc(O)cc1)[*]", "4-vinylphenol", "probe", "literature"),
    RepeatUnit("poly(sodium styrenesulfonate) as the free acid",
               "[*]CC(c1ccc(S(=O)(=O)O)cc1)[*]", "4-styrenesulfonic acid", "probe",
               "commodity"),
    RepeatUnit("poly(allyl alcohol)", "[*]CC(CO)[*]", "allyl alcohol", "probe", "commodity"),
    RepeatUnit("poly(allyl acetate)", "[*]CC(COC(C)=O)[*]", "allyl acetate", "probe",
               "commercial"),
    RepeatUnit("poly(allyl chloride)", "[*]CC(CCl)[*]", "allyl chloride", "probe", "commodity"),
    RepeatUnit("poly(divinylbenzene)", "[*]CC(c1ccc(C=C)cc1)[*]", "divinylbenzene", "probe",
               "commodity"),
    RepeatUnit("poly(allyl methacrylate)", "[*]CC(C)(C(=O)OCC=C)[*]", "allyl methacrylate",
               "probe", "commercial"),
    RepeatUnit("poly(1,1-diphenylethylene)", "[*]CC(c1ccccc1)(c1ccccc1)[*]",
               "1,1-diphenylethylene", "probe", "commercial"),
    RepeatUnit("poly(2-ethyl-1-butene)", "[*]CC(CC)(CC)[*]", "2-ethyl-1-butene", "probe",
               "commercial"),
    RepeatUnit("poly(2-butene)", "[*]C(C)C(C)[*]", "2-butene", "probe", "commodity"),
    RepeatUnit("poly(beta-methylstyrene)", "[*]C(C)C(c1ccccc1)[*]", "beta-methylstyrene",
               "probe", "commercial"),
    RepeatUnit("poly(stilbene)", "[*]C(c1ccccc1)C(c1ccccc1)[*]", "stilbene", "probe",
               "commercial"),
    RepeatUnit("poly(vinyl bromide)", "[*]CC(Br)[*]", "vinyl bromide", "probe", "commercial"),
    RepeatUnit("poly(methyl vinyl sulfide)", "[*]CC(SC)[*]", "methyl vinyl sulfide", "probe",
               "commercial"),
    RepeatUnit("poly(acrolein)", "[*]CC(C=O)[*]", "acrolein", "probe", "commodity"),
    RepeatUnit("poly(acryloyl chloride)", "[*]CC(C(=O)Cl)[*]", "acryloyl chloride", "probe",
               "commercial"),
    RepeatUnit("poly(vinyl isocyanate)", "[*]CC(N=C=O)[*]", "vinyl isocyanate", "probe",
               "literature"),
    RepeatUnit("poly(butadiene monoxide)", "[*]CC(C1CO1)[*]", "1,2-epoxy-3-butene", "probe",
               "commercial"),
    RepeatUnit("poly(vinylacetylene)", "[*]CC(C#C)[*]", "vinylacetylene", "probe", "literature"),
    RepeatUnit("poly(hexafluoropropylene)", "[*]C(F)(F)C(F)(C(F)(F)F)[*]",
               "hexafluoropropylene", "probe", "commodity"),
    RepeatUnit("poly(2-methoxypropene)", "[*]CC(C)(OC)[*]", "2-methoxypropene", "probe",
               "commercial"),
    RepeatUnit("poly(2-chloropropene)", "[*]CC(C)(Cl)[*]", "2-chloropropene", "probe",
               "commercial"),
    RepeatUnit("poly(trichloroethylene)", "[*]C(Cl)C(Cl)(Cl)[*]", "trichloroethylene", "probe",
               "commodity"),
    RepeatUnit("poly(2,3-dimethyl-2-butene)", "[*]C(C)(C)C(C)(C)[*]", "2,3-dimethyl-2-butene",
               "probe", "commercial"),
)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def _proposals() -> Iterator[RepeatUnit]:
    """Everything the grammar builds, gated and ungated alike."""
    # Polyethylene is the degenerate member of the mono-substituted family and
    # has to be written out: "[*]CC([H])[*]" is not how this convention writes a
    # hydrogen, and "[*]C[*]" would be polymethylene, a one-atom backbone.
    yield RepeatUnit("polyethylene", "[*]CC[*]", "ethylene", "polyolefin", "commodity")

    for group in (*_ALKYL, *_STYRENIC, *_N_VINYL, *_FUNCTIONAL, *_carrier_groups()):
        yield RepeatUnit(
            group.polymer,
            f"[*]CC({group.smiles})[*]",
            group.monomer,
            group.kind,
            group.availability,
            note=group.note,
        )
        if group.alpha_methyl is not None:
            polymer, monomer = group.alpha_methyl
            yield RepeatUnit(
                polymer,
                f"[*]CC(C)({group.smiles})[*]",
                monomer,
                group.kind,
                group.alpha_methyl_availability,
            )

    yield _VINYLIDENE_CHLORIDE
    yield from _FLUOROPOLYMERS

    for residue, name, availability in _FUMARATES:
        ester = f"C(=O)O{residue}"
        yield RepeatUnit(
            f"poly({name} fumarate)",
            f"[*]C({ester})C({ester})[*]",
            f"{name} fumarate",
            "fumarate",
            availability,
            note="one of the few 1,2-disubstituted alkenes with a homopolymer; the chain is "
                 "rod-like and the Tg is very high",
        )

    for residue, name, availability in _ITACONATES:
        ester = f"C(=O)O{residue}"
        yield RepeatUnit(
            f"poly({name} itaconate)",
            f"[*]CC(C{ester})({ester})[*]",
            f"{name} itaconate",
            "itaconate",
            availability,
            note="itaconic acid is fermented from sugar, so this arm of the family is "
                 "bio-derived",
        )

    yield from _DIENES
    yield from _POST_POLYMERISATION


@functools.lru_cache(maxsize=1)
def _built() -> tuple[tuple[RepeatUnit, ...], tuple[Refusal, ...], tuple[RepeatUnit, ...]]:
    """The library, the refusals, and any probe that got past the gates.

    The probes are gated in a loop of their own rather than concatenated onto
    the proposals, because a probe is a structure this module asserts is *not* a
    polymer: if a gate regresses, the old arrangement shipped it to the ranker
    under a name like "poly(vinyl bromide)" and the only thing standing between
    that and a search result was a test remembering to look.  Now a probe that
    escapes goes in the third tuple, where :func:`escapes` and a test can find
    it, and never into :func:`units`.
    """
    kept: list[RepeatUnit] = []
    refused: list[Refusal] = []
    escaped: list[RepeatUnit] = []
    seen: dict[str, str] = {}
    for proposal in _proposals():
        verdict = gate(proposal)
        if verdict is not None:
            refused.append(verdict)
            continue
        key = canonical(proposal.smiles)
        if key is None:  # pragma: no cover - the structure gate parses first
            continue
        if key in seen:
            continue
        seen[key] = proposal.name
        kept.append(replace(proposal, product=COMMERCIAL_POLYMERS.get(proposal.name, "")))
    for candidate in PROBES:
        verdict = gate(candidate)
        if verdict is None:
            escaped.append(candidate)
        else:
            refused.append(verdict)
    return tuple(kept), tuple(refused), tuple(escaped)


def records() -> list[RepeatUnit]:
    """Every generated repeat unit, with its monomer, family and availability."""
    return list(_built()[0])


def refusals() -> list[Refusal]:
    """Every structure the grammar built and then refused, with the reason."""
    return list(_built()[1])


def escapes() -> list[RepeatUnit]:
    """Probes that got past every gate.  Non-empty means a gate has regressed."""
    return list(_built()[2])


def commercial() -> list[RepeatUnit]:
    """The units somebody sells as a polymer, as opposed to merely being able to make."""
    return [record for record in _built()[0] if record.product]


def units() -> list[tuple[str, str]]:
    """(name, repeat-unit SMILES) for every vinyl-family polymer this module proposes."""
    return [(record.name, record.smiles) for record in _built()[0]]
