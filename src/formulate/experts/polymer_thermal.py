"""How much heat a polymer crystal costs to melt, per mole of repeat unit.

``melt.py`` answers *whether* a polymer melts and *at what temperature*, and
it reports honestly that the temperature could not be modelled: two structure
routes to Tm were fitted here and rejected at 250 to 350 K of error, which on
a 333-600 K range is 94 to 131 per cent of the span the model had to resolve.
That result does not transfer to the enthalpy, and the first job of this
module was to find out which way it goes rather than to assume.

It goes the other way, and by a measurable margin. Held out, the enthalpy of
fusion lands at **5.9 kJ/mol rms against a 6.7-57.8 kJ/mol range: 11.5 per
cent of the span**, about eight times better than the melting point in the
units that matter. The reason is structural rather than luck. The enthalpy of fusion
is extensive, so most of its variation across polymers is simply "how large is
the repeat unit"; the melting point is ``Tm = dHm/dSm``, a ratio in which the
size cancels and only the hard part survives. A negative result would have
been the lazy answer here, not the honest one.

**The gate comes first, and it is reused rather than reinvented.**
``melt.crystallinity(repeat_unit, tacticity)`` decides whether there is a
crystal to melt at all, before any number is computed. An amorphous polymer is
refused: returning zero would assert that melting it costs nothing, which is a
different claim and a false one - there is no melting endotherm to integrate.
A polymer whose tacticity is unstated is refused differently, because the
question is undecided rather than decided against: isotactic polystyrene
absorbs about 86 J/g on melting and the atactic polymer absorbs nothing.

**Route 1, preferred: the measured perfect-crystal enthalpy.**  Twenty-five
measurements over twenty-four repeat units - polystyrene has two, one per
stereoregular form - tabulated from Wunderlich's ATHAS compilation
(B. Wunderlich, *Macromolecular Physics, Vol. 3: Crystal Melting*, Academic
Press, 1980; *Thermal Analysis of Polymeric Materials*, Springer, 2005),
cross-checked against Brandrup, Immergut & Grulke, *Polymer Handbook*, 4th
ed., Wiley, 1999.  Thirteen are marked ``fit``, seven ``validation`` - never
seen by any coefficient - and five ``reference``: real measurements kept out
of both because they lie outside the model's domain and exist to document
where that boundary is.  A ``reference`` entry still answers when it is
matched, because a measurement is a measurement; what it never does is inform
or score a coefficient.

**Route 2, fallback: one parameter.**  ``dHm = a x (N + 1)``, where ``N``
counts the backbone atoms of the repeat unit and ``a = 2381.3 J/mol``, fitted
here by relative least squares to the thirteen ``fit`` entries.  The form is
Wunderlich's bead rule - the entropy of fusion of a flexible macromolecule is
roughly constant per mobile backbone unit, 7-12 J/(K mol) per bead - and the
coefficient is ours, so it is quoted only by its held-out error.  In sample it
reproduces the thirteen to 21.3%, which is what a fit does by construction;
left out one at a time it scores **23.4%**, and on the seven polymers no
coefficient ever saw, **10.0%**.  The pessimistic figure, 23.4% rounded up to
24%, is what every model prediction carries, and 16 of the 20 held-out
residuals fall inside it against the 14 that 68 per cent coverage would imply.
Worst held-out misses: polylactide +50.4%, nylon-6,6 -39.4%, polyoxymethylene
-28.2%.

**Why ``N + 1`` and not ``N``.**  The strict proportionality is the obvious
form for an extensive quantity and it is *not* what the measurements support.
Fitted through the origin it scores 26.5% left out and 14.7% on the validation
seven, and its residual is ordered in ``N`` rather than scattered: all five of
the two-backbone-atom polymers sit below the line (polyethylene -28%,
polypropylene -32%, PTFE -28%, polyoxymethylene -40%, PVDF -11% held out) and
the long aliphatic esters sit above it.  That is a bias, not noise, and a
symmetric error bar cannot represent it - which matters here more than the rms
does, because almost every polymer that reaches the *model* rather than the
table is a vinyl with two backbone atoms.  Allowing an intercept removes the
ordering; the two-parameter fit then returns an intercept/slope ratio of
**0.994**, i.e. one backbone atom, so the correction is a counting offset of
one unit per repeat unit rather than a second fitted parameter.  Sweeping the
offset for the best held-out score returns 0.95, and it is held at exactly 1.
What the offset stands for physically is not established: the small repeat
units in the table are also the tightly packing all-carbon and oxymethylene
chains, so repeat-unit size and backbone chemistry are confounded in thirteen
points and the offset absorbs some of both.  It is kept because it is the
smaller claim - one integer, no extra fitted parameter - and because it is
better on *every* measure: in sample, left out, on the validation seven, on
one-sigma coverage, and on the small-``N`` bias (-1.6% against -13.9%).

**What was tried and rejected, with its number.**

*Joback group contributions* over the same thirteen, reusing
``polymer.repeat_unit_groups``: eleven groups over thirteen measurements.
Left out one at a time it scores 20.7%, better than the one-parameter model -
and on the seven polymers it never saw it scores **49.2%**, five times worse.
Eleven parameters on thirteen points reproduce their own chemistries and
do not transfer; leave-one-out flatters a group scheme because the removed
polymer's groups are still carried by its near-neighbours.  It also answers
for only 44 of the 57 repeat units in ``data/reference_polymers.json`` - the
other thirteen carry a group none of the thirteen fitted polymers has, so it
has no coefficient for them - against 57 of 57 for the backbone count.
Rejected on both numbers.

*Wunderlich bead counting* - a backbone aromatic ring as two beads rather than
four atoms - scores 24.8% left out and 16.1% on the validation set, worse than
atom counting on both, and it needs a ring convention that atom counting does
not.

*Two-parameter variants*, each of which must earn its parameter held out and
none of which does, against 23.4% / 10.0% for one parameter and the fixed
offset: a free intercept 24.8% / 10.0%, backbone atoms + amide count 23.2% /
10.8%, flexible + aromatic backbone atoms 23.8% / 10.4%, backbone + side-chain
atoms 27.5% / 9.6%.  The free intercept is the interesting one: it is the form
that motivated the ``+1``, and once the offset is fixed at one the second
parameter has nothing left to buy.

*Cruder extensive measures*, to check that the backbone count is doing real
work: proportional to repeat-unit molar mass scores 43.6% held out, and per
heavy atom of the repeat unit 54.3%.

*The published Wunderlich route* ``dHm = s x N(beads) x Tm`` is genuinely
better per point, and is not taken, because it buys no coverage: a melting
point exists in this repository only as ``melt.MELTING_POINTS``, thirteen
polymers that already have a tabulated enthalpy here.  It would trade accuracy
for polymers that need no model and answer nothing for the ones that do.
Investigated, measured, declined.

**What is reported, and what is not.**  Every value here is the enthalpy of
fusion of the **100 per cent crystalline** polymer, **per mole of repeat
unit**.  A real sample absorbs that times its degree of crystallinity, which
is set by how the part was cooled - a processing variable, not a property of
the material - and this expert cannot supply it and must not appear to.  An
HDPE bar at 65% crystallinity absorbs about 190 J/g, not the 293 J/g reported
here.  The registry has one name, ``enthalpy_fusion``, for both that ceiling
and the ordinary whole-molecule heat of fusion the Joback expert reports, so
every prediction says in its notes which one it is.  The honest fix is two
property names, ``enthalpy_fusion_crystalline`` and
``degree_of_crystallinity``, which this module is not allowed to add.

**Where it refuses, and the measurement behind each refusal.**  A siloxane
backbone: the fitted line is 160% high on poly(dimethylsiloxane), because an
Si-O crystal holds together with far less energy per backbone atom than
anything in the table.  An unsaturated backbone: cis-1,4-polybutadiene +29%
but cis-1,4-polyisoprene +171%, and the only two such polymers that can be
checked differ by 2.1x per backbone atom and 2.6x per gram - that is not an
error bar, it is an absence of knowledge.  A pendant group of more than six
heavy atoms, which crystallises on its own account while the model counts
side-chain atoms as contributing nothing.  A pendant element outside
{C, N, O, F}: chlorine, sulfur and bromine appear nowhere in the table.  And a
fused or corner-cut backbone ring, where the count stops describing the unit
it is meant to count: the count is of atoms on the shortest path between the
two attachment points, so a 2,3-linked naphthalene reports two backbone atoms
for a ten-atom rigid unit.  Every backbone ring in the thirteen is a
para-linked six-membered ring giving four of its six atoms to that path.

Between those, a pendant of two to six heavy atoms is answered and marked out
of domain with a wider bar, 34% rather than 24%.  The extra term is not
measured scatter but a known blind spot: isotactic and syndiotactic
polystyrene share one repeat unit and differ by 1.62x, 8.96 against 5.52
kJ/mol, and no structure-only model can resolve a crystal polymorph.  The
half-spread of that pair, 23.8%, is added in quadrature to the model's 24%.

**What it answers but marks.**  A polymer whose chain is short enough that its
ends are a material fraction of it, a network or dendritic architecture, and a
stated crosslink density: in each of those the number is still the best
available statement of the perfect crystal's enthalpy, and in each the
candidate in hand is not the material the number describes.  The short chain
also widens the bar, by the fraction of repeat units that sit at a chain end
and do not crystallise with the rest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
from .melt import DECOMPOSES_BEFORE_MELTING, crystallinity
from .polymer import repeat_unit_mass

# --------------------------------------------------------------------------
# The measurements
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FusionEnthalpy:
    """A measured heat of fusion of the perfect crystal, J/mol of repeat unit."""

    value: float
    #: One-sigma as a fraction of the value. See FUSION_ENTHALPIES for why it
    #: is never measurement precision.
    rtol: float
    #: ``fit`` informs the coefficient, ``validation`` scores it and was never
    #: fitted, ``reference`` does neither because the polymer lies outside the
    #: model's domain - it is carried to document that boundary, and it still
    #: answers when it is matched.
    role: str
    source: str
    #: Set only where the measured value genuinely depends on the stereoisomer
    #: and both numbers are known, which here is polystyrene alone.
    tacticity: Tacticity | None = None


#: Relative one-sigma on an ordinary tabulated entry.
#:
#: This is not measurement precision. The heat of fusion of a 100 per cent
#: crystalline polymer is never measured, because nobody has ever made one: it
#: is obtained by measuring samples of known crystallinity and extrapolating to
#: chi = 1, and different extrapolations of the same quantity disagree. Ten per
#: cent is the ordinary spread between compilations.
TABULATED_RTOL = 0.10

#: And twice that where the compilations disagree outright: polypropylene is
#: quoted anywhere from 165 to 209 J/g, nylon-6,6 at 255 or 301, polyoxy-
#: methylene at 250 or 326. A number two sources give differently by 30 per
#: cent does not get a 10 per cent error bar.
DISPUTED_RTOL = 0.20

#: Enthalpy of fusion of the 100 per cent crystalline polymer, J/mol of repeat
#: unit, keyed by repeat-unit SMILES as this repository writes them.
#:
#: Wunderlich's ATHAS compilation, cross-checked against the Polymer Handbook.
#: The five values the brief names are reproduced exactly on conversion:
#: polyethylene 293 J/g (4.11 kJ/mol per CH2, so 8.22 kJ/mol for the C2H4
#: repeat unit this repository uses), nylon-6 230 J/g, PET 140 J/g, isotactic
#: polypropylene 207 J/g, poly(ethylene oxide) 197 J/g.
#:
#: The fit/validation split is kept in the table itself, the same arrangement
#: as ``data/reference_polymers.json``, so that the seven validation entries
#: can be *reported* as measurements while no coefficient ever saw them.
FUSION_ENTHALPIES: dict[str, tuple[FusionEnthalpy, ...]] = {
    # -- fit: the thirteen the coefficient was fitted to --------------------
    "[*]CC[*]": (FusionEnthalpy(8220.0, TABULATED_RTOL, "fit", "polyethylene; 293 J/g"),),
    "[*]CC(C)[*]": (
        FusionEnthalpy(
            8700.0, DISPUTED_RTOL, "fit",
            "polypropylene, stereoregular; 207 J/g, quoted from 165 to 209",
        ),
    ),
    "[*]CC(F)(F)[*]": (
        FusionEnthalpy(6700.0, TABULATED_RTOL, "fit", "poly(vinylidene fluoride); 105 J/g"),
    ),
    "[*]C(F)(F)C(F)(F)[*]": (
        FusionEnthalpy(8200.0, TABULATED_RTOL, "fit", "PTFE; 82 J/g"),
    ),
    "[*]CO[*]": (
        FusionEnthalpy(
            9790.0, DISPUTED_RTOL, "fit",
            "polyoxymethylene; 326 J/g, and quoted elsewhere at 250",
        ),
    ),
    "[*]CCO[*]": (
        FusionEnthalpy(8660.0, TABULATED_RTOL, "fit", "poly(ethylene oxide); 197 J/g"),
    ),
    "[*]CCCCO[*]": (
        FusionEnthalpy(14000.0, TABULATED_RTOL, "fit", "poly(tetramethylene oxide); 194 J/g"),
    ),
    "[*]OC(C)C(=O)[*]": (
        FusionEnthalpy(6700.0, TABULATED_RTOL, "fit", "polylactide, stereoregular; 93 J/g"),
    ),
    "[*]CCCCCC(=O)O[*]": (
        FusionEnthalpy(15900.0, TABULATED_RTOL, "fit", "polycaprolactone; 139 J/g"),
    ),
    "[*]NCCCCCC(=O)[*]": (
        FusionEnthalpy(26000.0, TABULATED_RTOL, "fit", "nylon-6; 230 J/g"),
    ),
    "[*]NCCCCCCNC(=O)CCCCC(=O)[*]": (
        FusionEnthalpy(
            57800.0, DISPUTED_RTOL, "fit",
            "nylon-6,6; 255 J/g, and quoted elsewhere at 301",
        ),
    ),
    "[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]": (
        FusionEnthalpy(26900.0, TABULATED_RTOL, "fit", "PET; 140 J/g"),
    ),
    "[*]OCCCCOC(=O)c1ccc(cc1)C(=O)[*]": (
        FusionEnthalpy(31300.0, TABULATED_RTOL, "fit", "PBT; 142 J/g"),
    ),
    # -- validation: never seen by the coefficient --------------------------
    "[*]OC(C)CC(=O)[*]": (
        FusionEnthalpy(12570.0, TABULATED_RTOL, "validation", "poly(3-hydroxybutyrate); 146 J/g"),
    ),
    "[*]NCCCCCCCCCCC(=O)[*]": (
        FusionEnthalpy(34640.0, DISPUTED_RTOL, "validation", "nylon-11; 189 J/g, held with less confidence"),
    ),
    "[*]NCCCCCCCCCCCC(=O)[*]": (
        FusionEnthalpy(41240.0, DISPUTED_RTOL, "validation", "nylon-12; 209 J/g, held with less confidence"),
    ),
    "[*]Oc1ccc(Oc2ccc(C(=O)c3ccc([*])cc3)cc2)cc1": (
        FusionEnthalpy(37480.0, TABULATED_RTOL, "validation", "PEEK; 130 J/g"),
    ),
    "[*]OCC(=O)[*]": (
        FusionEnthalpy(10620.0, DISPUTED_RTOL, "validation", "polyglycolide; 183 J/g, quoted from 180 to 207"),
    ),
    "[*]OCCOC(=O)CCCCC(=O)[*]": (
        FusionEnthalpy(24100.0, DISPUTED_RTOL, "validation", "poly(ethylene adipate); 140 J/g"),
    ),
    "[*]CC(CC)[*]": (
        FusionEnthalpy(7010.0, TABULATED_RTOL, "validation", "poly(1-butene), isotactic form I; 125 J/g"),
    ),
    # -- reference: measurements outside the model's domain -----------------
    # These five are why the refusals below are worded as measurements rather
    # than as opinions. Each still answers for its own polymer; none of them
    # licenses the model on any other.
    "[*][Si](C)(C)O[*]": (
        FusionEnthalpy(
            2750.0, TABULATED_RTOL, "reference",
            "poly(dimethylsiloxane); 37 J/g - the fitted line is 160% high here",
        ),
    ),
    "[*]CC=CC[*]": (
        FusionEnthalpy(
            9200.0, TABULATED_RTOL, "reference",
            "cis-1,4-polybutadiene; 170 J/g - the fitted line is 29% high here",
        ),
    ),
    "[*]CC(C)=CC[*]": (
        FusionEnthalpy(
            4400.0, TABULATED_RTOL, "reference",
            "cis-1,4-polyisoprene; 65 J/g - the fitted line is 171% high here",
        ),
    ),
    "[*]CC(c1ccccc1)[*]": (
        FusionEnthalpy(
            8960.0, TABULATED_RTOL, "reference", "isotactic polystyrene; 86 J/g",
            tacticity=Tacticity.ISOTACTIC,
        ),
        FusionEnthalpy(
            5520.0, TABULATED_RTOL, "reference", "syndiotactic polystyrene; 53 J/g",
            tacticity=Tacticity.SYNDIOTACTIC,
        ),
    ),
}


# --------------------------------------------------------------------------
# The one-parameter model
# --------------------------------------------------------------------------

#: Enthalpy of fusion per counted backbone unit of the repeat unit, J/mol.
#:
#: Fitted here by relative least squares - minimising the sum of squared
#: *fractional* residuals, because the quantity spans a factor of nine and an
#: absolute loss would be a fit to nylon-6,6 alone - over the thirteen ``fit``
#: entries above. The form is Wunderlich's: the entropy of fusion of a flexible
#: macromolecule is roughly constant per mobile backbone unit, so the enthalpy
#: is roughly proportional to how many of them there are.
#:
#: Quoted only by its held-out error, never its in-sample one. See the module
#: docstring: 21.3% in sample, 23.4% left out one at a time, 10.0% on seven
#: polymers no coefficient ever saw.
BACKBONE_UNIT_ENTHALPY = 2381.3

#: Backbone units counted per repeat unit beyond the atoms on the shortest path
#: between the attachment points.
#:
#: Not a second fitted parameter, and not free. Fitted through the origin the
#: model's residual is ordered in chain length rather than scattered - all five
#: two-backbone-atom polymers below the line, the long aliphatic esters above
#: it - which is a bias an error bar cannot describe, and it lands on exactly
#: the repeat units the model is actually used for. A two-parameter fit that is
#: allowed an intercept returns intercept/slope = 0.994, one backbone atom, and
#: sweeping the offset for the best held-out score returns 0.95. So the offset
#: is held at the integer 1 and the model keeps one fitted coefficient.
BACKBONE_COUNT_OFFSET = 1

#: One-sigma on a model prediction inside the domain, as a fraction.
#:
#: max(leave-one-out over the thirteen = 23.4%, validation over the seven =
#: 10.0%), rounded up. The same "quote the pessimistic held-out figure" rule
#: ``polymer.GlassTransitionModel.sigma`` uses, and for the same reason:
#: leave-one-out reuses polymers that informed the model's shape and the
#: validation set is small, so neither alone is trustworthy. Sixteen of the
#: twenty held-out residuals fall inside it, against the fourteen that 68%
#: coverage implies - wide by two polymers, which is the safe direction.
MODEL_RTOL = 0.24

#: Half-spread between isotactic and syndiotactic polystyrene: one repeat unit,
#: two crystal polymorphs, 8.96 against 5.52 kJ/mol, a factor of 1.62. No
#: structure-only model can resolve which crystal forms, so a bulky pendant
#: earns this in quadrature on top of the model's own error.
POLYMORPH_RTOL = 0.238

#: One-sigma for a bulky-pendant polymer, as a fraction: 34%. Widened for a
#: known blind spot rather than for measured scatter, which is stated as such
#: on the prediction rather than blended into a single number.
OUT_OF_DOMAIN_RTOL = math.sqrt(MODEL_RTOL**2 + POLYMORPH_RTOL**2)

#: Backbone elements the thirteen fitted polymers cover. A siloxane backbone is
#: the one out-of-set case that can be checked and the model is 160% high on
#: it, so anything outside this set is refused rather than widened.
MODEL_BACKBONE_ELEMENTS = frozenset({"C", "N", "O"})

#: Pendant elements the table covers. Chlorine, sulfur and bromine appear
#: nowhere in it; poly(vinyl chloride) is quoted near 11 kJ/mol against the
#: model's 7.1, and that value could not be established well enough here to
#: fit, so the honest move is to decline rather than to widen.
MODEL_PENDANT_ELEMENTS = frozenset({"C", "N", "O", "F"})

#: A pendant of one heavy atom - a methyl, a fluorine, a hydroxyl, a carbonyl
#: oxygen - is what the fitted set contains, and is where the model is in
#: domain.
IN_DOMAIN_PENDANT_ATOMS = 1

#: Above this the side chain crystallises on its own account. The model counts
#: side-chain atoms as contributing nothing, so it would miss by the whole
#: side-chain contribution rather than by a percentage - which is a different
#: kind of error from the one the error bar describes.
MAX_PENDANT_ATOMS = 6

#: Largest repeat unit the model has been checked on: PEEK, fifteen backbone
#: atoms. Linear extrapolation in an extensive quantity is the safe direction,
#: but a repeat unit twice that size gets an answer nobody has checked, so it
#: is marked rather than refused.
MAX_CHECKED_BACKBONE_ATOMS = 15

#: Atoms per backbone ring that the count is allowed to miss, on average.
#:
#: The count is of atoms on the shortest path between the two attachment
#: points, so a ring is counted by the atoms the path crosses. Every backbone
#: ring in the thirteen is a para-linked six-membered ring, which gives four of
#: its six atoms to the path and hides two - PET hides two of twelve, PEEK six
#: of twenty-one. A ring linked any other way, or fused to the one the path
#: crosses, hides more: a 2,3-linked naphthalene hides eight of its ten and
#: reports two backbone atoms for a ten-atom rigid unit, which is not a
#: percentage error in the count but a different count.
MAX_RING_ATOMS_OFF_PATH = 2

#: Below this number-average molar mass the chain ends are a large enough
#: fraction of the chain that the infinite-chain perfect crystal is not what a
#: sample would show. Matches the threshold ``polymer.py`` uses.
_SHORT_CHAIN_MOLAR_MASS = 2000.0

#: Repeat units at each end of a chain taken not to crystallise with the rest.
#:
#: One per end is the crudest defensible statement of the chain-end deficit and
#: it is used only to *size* an uncertainty, never to correct a value: a chain
#: of m repeat units then carries a fractional deficit of 2/m, which is 6% at
#: the 2000 g/mol threshold and 30% for a ten-unit oligomer. The real deficit
#: depends on lamellar thickness and on how the part was cooled, neither of
#: which this expert knows, so the number is deliberately a floor on the
#: uncertainty rather than an estimate of the error.
_NON_CRYSTALLISING_END_UNITS = 2.0


@dataclass(frozen=True, slots=True)
class RepeatUnitShape:
    """What the one-parameter model and its gates need from a repeat unit."""

    #: Atoms on the shortest path between the two attachment points.
    backbone_atoms: int
    #: Backbone atoms the count does not see: the atoms of a ring the path
    #: crosses, or of a ring fused to one, that the path itself does not touch.
    #: Two per ring for a para-linked six-membered ring, which is what the
    #: thirteen contain; more than that and the count stops describing the unit.
    ring_atoms_off_path: int
    #: Rings that are part of the backbone, counted so the line above can be
    #: read per ring rather than per repeat unit.
    backbone_rings: int
    #: Element symbols of those atoms, plus any ring they sit in.
    backbone_elements: frozenset[str]
    #: True when two backbone atoms are joined by a non-aromatic double or
    #: triple bond.
    unsaturated_backbone: bool
    #: Heavy-atom count of each branch hanging off the backbone.
    pendant_sizes: tuple[int, ...]
    pendant_elements: frozenset[str]


def repeat_unit_shape(repeat_unit: str) -> RepeatUnitShape | None:
    """Read a repeat unit, or None if it cannot be read as a chain segment.

    A backbone ring is treated as backbone rather than as a pendant: the two
    ring carbons a terephthalate path does not pass through are part of the
    same rigid unit, not a substituent hanging off it.  The absorption runs to
    a fixed point rather than in one pass, because a ring fused to a backbone
    ring is backbone too: one pass over ``AtomRings()`` picks up a ring only if
    it meets the core *as the core stands when that ring is reached*, so the
    outer rings of an anthracene backbone survive as an eight-atom "pendant"
    if the ring perception happens to list them first.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(repeat_unit)
    if mol is None:
        return None
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        return None
    path = [i for i in Chem.GetShortestPath(mol, stars[0], stars[1]) if i not in stars]
    if not path:
        return None

    core = set(path)
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    growing = True
    while growing:
        growing = False
        for ring in rings:
            if core & ring and not ring <= core:
                core |= ring
                growing = True

    unsaturated = False
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (
            a in core
            and b in core
            and not bond.GetIsAromatic()
            and bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
        ):
            unsaturated = True

    excluded = core | set(stars)
    sizes: list[int] = []
    elements: set[str] = set()
    for idx in sorted(core):
        for neighbour in mol.GetAtomWithIdx(idx).GetNeighbors():
            start = neighbour.GetIdx()
            if start in excluded:
                continue
            seen, stack, branch = set(excluded), [start], []
            while stack:
                here = stack.pop()
                if here in seen:
                    continue
                seen.add(here)
                branch.append(here)
                stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(here).GetNeighbors())
            excluded |= set(branch)
            sizes.append(len(branch))
            elements.update(mol.GetAtomWithIdx(k).GetSymbol() for k in branch)

    return RepeatUnitShape(
        backbone_atoms=len(path),
        ring_atoms_off_path=len(core) - len(path),
        backbone_rings=sum(1 for ring in rings if ring <= core),
        backbone_elements=frozenset(mol.GetAtomWithIdx(i).GetSymbol() for i in core),
        unsaturated_backbone=unsaturated,
        pendant_sizes=tuple(sorted(sizes, reverse=True)),
        pendant_elements=frozenset(elements),
    )


def model_enthalpy_fusion(backbone_atoms: int) -> float:
    """Enthalpy of fusion of the perfect crystal, J/mol of repeat unit."""
    return BACKBONE_UNIT_ENTHALPY * (backbone_atoms + BACKBONE_COUNT_OFFSET)


def model_refusal(shape: RepeatUnitShape) -> str | None:
    """Why the one-parameter model must not answer for this repeat unit.

    Each reason carries the measurement that establishes it. A gate with no
    number behind it is an opinion, and an opinion is not a reason to refuse
    any more than it is a reason to answer.
    """
    if shape.ring_atoms_off_path > MAX_RING_ATOMS_OFF_PATH * shape.backbone_rings:
        return (
            f"the backbone is counted by the {shape.backbone_atoms} atoms on the "
            f"shortest path between the attachment points, and this repeat unit puts "
            f"{shape.ring_atoms_off_path} further backbone atoms off that path in "
            f"{shape.backbone_rings} ring(s) - a fused or corner-cut ring system rather "
            f"than the para-linked six-membered rings the thirteen contain, which hide "
            f"two atoms each. A 2,3-linked naphthalene reports two backbone atoms for a "
            "ten-atom rigid unit; that is not a percentage error in the count, it is a "
            "different count, and the fitted coefficient has no meaning against it"
        )
    outside = sorted(shape.backbone_elements - MODEL_BACKBONE_ELEMENTS)
    if outside:
        return (
            f"the backbone carries {', '.join(outside)}, and every polymer this "
            "coefficient was fitted to has a backbone of carbon, nitrogen and oxygen "
            "only. The one out-of-set backbone that can be checked is the siloxane of "
            "poly(dimethylsiloxane), where the fitted line is 160% high: an Si-O "
            "crystal holds together with far less energy per backbone atom than "
            "anything in the table, and no error bar drawn from the table covers that"
        )
    if shape.unsaturated_backbone:
        return (
            "the backbone carries a non-aromatic double or triple bond, and the two "
            "such polymers that can be checked disagree with each other, not merely "
            "with the model: the fitted line is 29% high on cis-1,4-polybutadiene and "
            "171% high on cis-1,4-polyisoprene, which differ by 2.1x per backbone atom "
            "and 2.6x per gram. That is an absence of knowledge rather than an error bar"
        )
    if shape.pendant_sizes and shape.pendant_sizes[0] > MAX_PENDANT_ATOMS:
        return (
            f"the largest pendant group carries {shape.pendant_sizes[0]} heavy atoms, "
            f"more than the {MAX_PENDANT_ATOMS} this model has been checked to. A side "
            "chain that long crystallises on its own account, and the model counts "
            "side-chain atoms as contributing nothing at all - so it would miss by the "
            "whole side-chain contribution rather than by a percentage, which is not "
            "the kind of error the stated uncertainty describes"
        )
    unusual = sorted(shape.pendant_elements - MODEL_PENDANT_ELEMENTS)
    if unusual:
        return (
            f"a pendant group carries {', '.join(unusual)}, which appears in no polymer "
            "in the table. Poly(vinyl chloride) is the nearest check and is quoted near "
            "11 kJ/mol against this model's 7.1, and that value could not be "
            "established here well enough to fit; widening the bar to cover a number "
            "this model has no information about would be a worse answer than declining"
        )
    return None


def _repeat_unit(spec: PolymerSpec) -> str | None:
    """The single repeat unit of a homopolymer, or None for a copolymer."""
    chain = [m for m in spec.monomers if m.role is not MonomerRole.END_GROUP]
    return chain[0].smiles if len(chain) == 1 else None


def decomposition_reason(repeat_unit: str) -> str | None:
    """Why this polymer never reaches its melting point, if it does not.

    ``melt.crystallinity`` folds two different facts into one ``False``: a
    chain that cannot pack, and a chain that packs perfectly well but cyclises
    or degrades below the temperature at which the crystal would melt.  Those
    are opposite statements about whether a crystal exists, and a refusal that
    tells a chemist polyacrylonitrile "has no crystal to melt" is one they
    would reject - ``melt.py``'s own comment on the table says a stereoregular
    sample of these *does* have a crystalline phase.  So the second case is
    separated out and answered in its own words.
    """
    from formulate import chem

    if repeat_unit in DECOMPOSES_BEFORE_MELTING:
        return DECOMPOSES_BEFORE_MELTING[repeat_unit]
    if not chem.rdkit_available():
        return None
    canonical = chem.canonical_smiles(repeat_unit)
    if canonical is None:
        return None
    for key, reason in DECOMPOSES_BEFORE_MELTING.items():
        if chem.canonical_smiles(key) == canonical:
            return reason
    return None


def _safe_repeat_unit_mass(repeat_unit: str) -> float | None:
    """Repeat-unit molar mass, or None where it cannot be built.

    Only the notes and the chain-end fraction need it, so a repeat unit RDKit
    can read as a chain segment but cannot be linked into a trimer must not
    turn a good measurement into a failed prediction.
    """
    try:
        mass = repeat_unit_mass(repeat_unit)
    except (ValueError, RuntimeError):
        return None
    return mass if mass > 0.0 else None


@dataclass(frozen=True, slots=True)
class MaterialCaveats:
    """What the candidate says about itself that neither route can answer.

    These attach to a *measured* perfect-crystal enthalpy exactly as they do to
    a modelled one, because they are not statements about the model: a network
    is not the linear chain whose crystal was measured, and a twenty-mer is not
    the infinite chain the extrapolation to chi = 1 assumes.  Route 1 used to
    build itself an unconditionally in-domain assessment and so reported a
    crosslinked, 900 g/mol polyethylene at the full tabulated value, in domain,
    with a 10% measurement bar.
    """

    warnings: tuple[str, ...]
    #: Fraction of repeat units sitting at a chain end, and so not in the
    #: lattice interior. Zero unless the chain is short enough to matter.
    end_fraction: float = 0.0


def material_caveats(spec: PolymerSpec, mass: float | None) -> MaterialCaveats:
    """Architecture and chain length, read against what the table describes."""
    warnings: list[str] = []
    if spec.topology in (PolymerTopology.NETWORK, PolymerTopology.DENDRITIC):
        warnings.append(
            f"this candidate is a {spec.topology.value} polymer and every value here "
            "describes the perfect crystal of the corresponding linear chain; "
            "crosslinks and dense branching bar that state rather than change its "
            "enthalpy, so the number is a ceiling the candidate cannot approach and "
            "not a quantity a scan of it would find"
        )
    elif spec.crosslink_density is not None:
        warnings.append(
            "a crosslink density is stated; crosslinks suppress how much of this "
            "ceiling a sample can reach without changing the ceiling itself, and the "
            "thirteen are uncrosslinked linear chains"
        )

    end_fraction = 0.0
    chain = spec.number_average_molar_mass
    if chain is not None and chain.to("g/mol").value < _SHORT_CHAIN_MOLAR_MASS:
        grams_per_mole = chain.to("g/mol").value
        warning = (
            f"the stated chain is {grams_per_mole:.0f} g/mol; this value is the "
            "infinite-chain limit and a chain this short melts lower and with less heat "
            "per repeat unit because its ends do not crystallise"
        )
        # Without a repeat-unit mass the deficit cannot be sized, and an
        # unsized caveat is still a caveat - it just does not widen anything.
        if mass:
            units = grams_per_mole / mass
            end_fraction = min(_NON_CRYSTALLISING_END_UNITS / units, 1.0) if units else 1.0
            warning += (
                f", about {units:.0f} of them here. Taking one repeat unit at each end "
                f"as lost puts the deficit at {end_fraction*100:.0f}%, which is added to "
                "the uncertainty rather than subtracted from the value, because how much "
                "of a short chain crystallises depends on how it was cooled"
            )
        warnings.append(warning)
    return MaterialCaveats(warnings=tuple(warnings), end_fraction=end_fraction)


def tabulated_enthalpy(repeat_unit: str, tacticity: Tacticity) -> FusionEnthalpy | None:
    """The measured perfect-crystal enthalpy for this polymer, or None.

    Keyed on the repeat unit, and on the tacticity where - as for polystyrene -
    the two stereoregular forms crystallise into different lattices and both
    numbers are known. A tacticity-specific entry is never reached with an
    unstated tacticity, because the crystallinity gate refuses first.
    """
    from formulate import chem

    entries: tuple[FusionEnthalpy, ...] | None = FUSION_ENTHALPIES.get(repeat_unit)
    if entries is None and chem.rdkit_available():
        canonical = chem.canonical_smiles(repeat_unit)
        if canonical is not None:
            for key, value in FUSION_ENTHALPIES.items():
                if chem.canonical_smiles(key) == canonical:
                    entries = value
                    break
    if entries is None:
        return None
    for entry in entries:
        if entry.tacticity is tacticity:
            return entry
    for entry in entries:
        if entry.tacticity is None:
            return entry
    return None


# --------------------------------------------------------------------------
# The expert
# --------------------------------------------------------------------------


class PolymerThermalExpert(Expert):
    """Enthalpy of fusion of a semicrystalline polymer, per mole of repeat unit.

    Two routes behind one gate. The gate is chain regularity, borrowed whole
    from ``melt.crystallinity``: a polymer with no crystalline phase has no
    enthalpy of fusion, and a polymer whose tacticity has not been stated has
    not been asked a well-posed question yet. Behind it, a measurement where
    one exists and a one-parameter backbone count otherwise.
    """

    id = "polymer_thermal"
    version = "1"
    method = (
        "enthalpy of fusion of the 100% crystalline polymer, per mole of repeat unit: "
        "crystallinity gated on chain regularity (melt.crystallinity), then a measured "
        "perfect-crystal value from Wunderlich's ATHAS compilation (Macromolecular "
        "Physics Vol. 3, 1980; Thermal Analysis of Polymeric Materials, 2005) "
        "cross-checked against the Polymer Handbook 4th ed., 1999, or otherwise one "
        "fitted parameter on Wunderlich's bead form, dHm = 2381.3 J/mol per (backbone "
        "atom + 1), which scores 23.4% left out one at a time over 13 polymers and "
        "10.0% over 7 never fitted. NOT the heat a real sample absorbs: multiply by a "
        "degree of crystallinity this expert does not supply"
    )
    family = PropertyFamily.THERMAL
    supported_classes = frozenset({MaterialClass.POLYMER})
    supported_properties = frozenset({"enthalpy_fusion"})
    dependencies: frozenset[str] = frozenset()

    # -- availability ------------------------------------------------------

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        from formulate import chem

        return "" if chem.rdkit_available() else "RDKit is required to read a repeat unit"

    def _software(self) -> SoftwareEnvironment:
        from formulate import chem

        return SoftwareEnvironment.capture(rdkit=chem.rdkit_version())

    # -- domain ------------------------------------------------------------

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        basis = (
            "20 semicrystalline homopolymers with a tabulated perfect-crystal enthalpy "
            "of fusion (13 fitted, 7 held out); the one-parameter backbone count is "
            "checked from 2 to 15 backbone atoms, on carbon/nitrogen/oxygen backbones "
            "with pendant groups of one heavy atom"
        )
        spec = candidate.polymer
        if spec is None:
            return ApplicabilityDomain.outside("candidate carries no polymer", basis=basis)
        repeat = _repeat_unit(spec)
        if repeat is None:
            return ApplicabilityDomain.outside(
                "a copolymer's crystal is not the mole-weighted average of its "
                "components' crystals",
                basis=basis,
            )
        shape = repeat_unit_shape(repeat)
        if shape is None:
            return ApplicabilityDomain.outside(
                "the repeat unit could not be read as a chain segment carrying exactly "
                "two attachment points",
                basis=basis,
            )

        caveats = material_caveats(spec, _safe_repeat_unit_mass(repeat))

        # A measurement is in domain by definition. The gates below describe
        # where the *model* may be used, and they have no authority over a
        # number that was measured rather than computed. What the measurement
        # does not escape is the candidate: a crosslinked or oligomeric
        # polyethylene is not the polyethylene the number was measured on.
        if tabulated_enthalpy(repeat, spec.tacticity) is not None:
            measured_basis = (
                basis + "; this polymer has a measured value, so the model's domain "
                "does not apply to it"
            )
            if caveats.warnings:
                return ApplicabilityDomain.outside(
                    *caveats.warnings, score=0.5, basis=measured_basis
                )
            return ApplicabilityDomain(basis=measured_basis)

        refusal = model_refusal(shape)
        if refusal is not None:
            return ApplicabilityDomain.outside(refusal, basis=basis)

        warnings: list[str] = list(caveats.warnings)
        score = 0.5 if warnings else 1.0
        largest = shape.pendant_sizes[0] if shape.pendant_sizes else 0
        if largest > IN_DOMAIN_PENDANT_ATOMS:
            warnings.append(
                f"the largest pendant group carries {largest} heavy atoms; every polymer "
                "the coefficient was fitted to carries at most one, and a bulky side "
                "group also opens the crystal-polymorph question that isotactic and "
                "syndiotactic polystyrene answer 1.62x apart"
            )
            score = min(score, 0.4)
        if shape.backbone_atoms > MAX_CHECKED_BACKBONE_ATOMS:
            warnings.append(
                f"{shape.backbone_atoms} backbone atoms is beyond the "
                f"{MAX_CHECKED_BACKBONE_ATOMS} this model has been checked to; linear "
                "extrapolation is the safe direction for an extensive quantity and it "
                "is still an extrapolation"
            )
            score = min(score, 0.5)

        return ApplicabilityDomain(
            score=score, in_domain=not warnings, warnings=tuple(warnings), basis=basis
        )

    # -- prediction --------------------------------------------------------

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        spec = request.candidate.polymer
        if spec is None:
            return Prediction.unsupported(prop, self.id, "candidate carries no polymer")
        repeat = _repeat_unit(spec)
        if repeat is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "a copolymer has no single perfect crystal: a comonomer that will not "
                "fit the lattice is excluded from it, which depresses both the melting "
                "point and the heat of fusion, and the result is not the mole-weighted "
                "average of the two homopolymers' crystals",
            )
        shape = repeat_unit_shape(repeat)
        if shape is None:
            return Prediction.unsupported(
                prop,
                self.id,
                f"the repeat unit {repeat!r} could not be read as a chain segment with "
                "exactly two [*] attachment points, so its backbone cannot be traced",
            )

        # The gate, before any number. Whether there is a crystal to melt is a
        # different question from how much melting it costs, and it is answered
        # first because a wrong answer to it cannot be rescued by an error bar.
        phase = crystallinity(repeat, spec.tacticity)
        decomposes = decomposition_reason(repeat)
        if phase.crystalline is False and decomposes is not None:
            # A crystal that is never reached is not an absent crystal, and
            # saying so would be a refusal a chemist would throw back.
            return Prediction.unsupported(
                prop,
                self.id,
                f"this polymer does have a crystalline phase and never reaches the "
                f"temperature at which it would melt: {decomposes}. So there is no "
                "melting endotherm to integrate and no enthalpy of fusion to measure - "
                "the quantity this expert reports is the heat absorbed at a melting "
                "transition, and this polymer has none. Its crystal cohesive energy is "
                "a different quantity and is not tabulated here",
            )
        if phase.crystalline is False:
            return Prediction.unsupported(
                prop,
                self.id,
                f"this polymer has no enthalpy of fusion because it has no crystal to "
                f"melt: {phase.reason}. Reporting zero would be a different claim and a "
                "false one - that melting this polymer costs nothing - when in fact "
                "there is no melting endotherm to integrate at all",
            )
        if phase.crystalline is None:
            return Prediction.unsupported(
                prop,
                self.id,
                f"whether this polymer has a crystalline phase is undecided rather than "
                f"decided against: {phase.reason}. State the tacticity and the question "
                "has an answer - isotactic polystyrene absorbs about 86 J/g on melting "
                "and the atactic polymer absorbs none at all",
            )

        mass = _safe_repeat_unit_mass(repeat)
        caveats = material_caveats(spec, mass)
        common = self._common_notes()

        entry = tabulated_enthalpy(repeat, spec.tacticity)
        if entry is not None:
            return self._from_measurement(prop, request, entry, mass, caveats, common)

        refusal = model_refusal(shape)
        if refusal is not None:
            return Prediction.unsupported(prop, self.id, refusal)

        return self._from_model(prop, request, domain, shape, mass, caveats, common)

    # -- the two routes ----------------------------------------------------

    def _from_measurement(
        self,
        prop: str,
        request: PredictionRequest,
        entry: FusionEnthalpy,
        mass: float | None,
        caveats: MaterialCaveats,
        common: tuple[str, ...],
    ) -> Prediction:
        """A tabulated perfect-crystal value, which no model error applies to."""
        measured_basis = (
            "a measured perfect-crystal enthalpy of fusion; the one-parameter "
            "model's domain does not apply to a value that was not modelled"
        )
        measured_domain = (
            ApplicabilityDomain.outside(*caveats.warnings, score=0.5, basis=measured_basis)
            if caveats.warnings
            else ApplicabilityDomain(basis=measured_basis)
        )
        rtol = math.hypot(entry.rtol, caveats.end_fraction)
        role = {
            "fit": "this measurement informed the fitted coefficient",
            "validation": "held out: no coefficient here ever saw this measurement",
            "reference": (
                "held out of both the fit and the validation score because this polymer "
                "lies outside the one-parameter model's domain; it is carried to "
                "document that boundary and it answers for itself"
            ),
        }[entry.role]
        return self._make(
            prop,
            entry.value,
            "J/mol",
            request,
            measured_domain,
            std=entry.value * rtol,
            kind=UncertaintyKind.COMBINED,
            basis=(
                f"{entry.rtol*100:.0f}% of the value, which is not measurement "
                "precision: a 100% crystalline polymer is never measured, only "
                "extrapolated to from samples of known crystallinity, and different "
                "extrapolations of the same quantity disagree by about this much"
                + (
                    f", widened in quadrature to {rtol*100:.0f}% for the chain-end "
                    "deficit of the short chain this candidate states"
                    if caveats.end_fraction
                    else ""
                )
            ),
            notes=(
                f"measured: {entry.source}",
                role,
                *self._per_gram_note(entry.value, mass),
                *common,
                *caveats.warnings,
            ),
            route="tabulated",
            role=entry.role,
        )

    def _from_model(
        self,
        prop: str,
        request: PredictionRequest,
        domain: ApplicabilityDomain,
        shape: RepeatUnitShape,
        mass: float | None,
        caveats: MaterialCaveats,
        common: tuple[str, ...],
    ) -> Prediction:
        """The one fitted parameter, quoted by its held-out error."""
        value = model_enthalpy_fusion(shape.backbone_atoms)
        bulky = bool(shape.pendant_sizes) and shape.pendant_sizes[0] > IN_DOMAIN_PENDANT_ATOMS
        rtol = math.hypot(OUT_OF_DOMAIN_RTOL if bulky else MODEL_RTOL, caveats.end_fraction)
        if bulky:
            basis = (
                f"{MODEL_RTOL*100:.0f}% held out, widened to "
                f"{OUT_OF_DOMAIN_RTOL*100:.0f}% in quadrature for a known blind spot "
                "rather than for measured scatter: isotactic and syndiotactic "
                "polystyrene share this repeat-unit description and absorb 8.96 and "
                "5.52 kJ/mol, a factor of 1.62 that no structure-only model can resolve"
            )
        else:
            basis = (
                f"{MODEL_RTOL*100:.0f}% of the value: the larger of leave-one-out over "
                "the 13 fitted polymers (23.4%) and the 7 never fitted (10.0%), rounded "
                "up. In sample it is 21.3%, which is quoted here only to show that the "
                "held-out figure is not the flattering one"
            )
        if caveats.end_fraction:
            basis += (
                f", and widened again to {rtol*100:.0f}% for the chain-end deficit of "
                "the short chain this candidate states"
            )

        notes = [
            f"predicted: ({shape.backbone_atoms} backbone atoms + "
            f"{BACKBONE_COUNT_OFFSET}) x {BACKBONE_UNIT_ENTHALPY:.0f} J/mol, one "
            "parameter fitted here to 13 polymers on Wunderlich's bead form over a "
            "counting offset of one that a free intercept puts at 0.994",
            *self._per_gram_note(value, mass),
            "a Joback group fit over the same 13 was tried and rejected: 20.7% left out "
            "one at a time but 49.2% on 7 polymers it never saw, and it answers for 44 "
            "of the 57 repeat units in the reference library against 57 of 57 for this",
        ]
        notes.extend(domain.warnings)
        notes.extend(common)
        notes.extend(w for w in caveats.warnings if w not in domain.warnings)
        return self._make(
            prop,
            value,
            "J/mol",
            request,
            domain,
            std=value * rtol,
            kind=UncertaintyKind.EPISTEMIC,
            basis=basis,
            notes=tuple(notes),
            route="model",
            backbone_atoms=shape.backbone_atoms,
        )

    # -- what every prediction has to say ----------------------------------

    @staticmethod
    def _per_gram_note(value: float, mass: float | None) -> tuple[str, ...]:
        """The per-gram reading, where the repeat-unit mass could be built."""
        if mass is None:
            return (
                "the repeat-unit molar mass could not be built, so this value is not "
                "restated per gram here",
            )
        return (f"= {value / mass:.0f} J/g of repeat unit (repeat unit {mass:.2f} g/mol)",)

    def _common_notes(self) -> tuple[str, ...]:
        """The reading of the number, and the two ways it is misread.

        Both are live: the registry carries one name for this quantity and for
        the whole heat of fusion of a small molecule, and nothing downstream
        multiplies by a degree of crystallinity because nothing downstream has
        one to multiply by.
        """
        notes = [
            "per mole of REPEAT UNIT, not per gram and not per mole of chain",
            "this is the 100% CRYSTALLINE polymer: a real sample absorbs this times its "
            "degree of crystallinity, which is set by how the part was cooled and is a "
            "processing variable rather than a property of the material - an HDPE bar at "
            "65% crystallinity absorbs about 190 J/g against the 293 J/g reported here "
            "for polyethylene",
            "not comparable with a molecular enthalpy of fusion under the same property "
            "name: the Joback expert reports a real, attainable heat of fusion and this "
            "reports an unattainable ceiling",
        ]
        return tuple(notes)
