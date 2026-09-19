"""The top of the processing window, and whether there is a window at all.

A ranking that put poly(acrylic acid) first for a 200 C melt process was not
wrong about any number it reported. It was wrong because nothing asked the
question that disqualifies it: poly(acrylic acid) does not melt at 200 C, it
DEHYDRATES. Adjacent carboxylic acids on the same backbone condense to a cyclic
anhydride from about 150 C and keep going, so what leaves the die is not the
polymer that entered the cartridge.

That failure has a shape worth naming, because it is the shape of every other
gap in this engine. The registry had a melting point and a glass transition -
the bottom of a processing window - and nothing at all for the top. A window
with only one end is not a window, and a search handed one will happily propose
a material whose two ends are in the wrong order.

Two things are served here.

**decomposition_temperature** is where chemistry starts. It comes from group
contributions over the repeat unit, in the same additive spirit as the Joback
groups and the Sugden parachor already in this repository, plus a set of
SMARTS gates for the specific chemistries that decompose far below what a
group sum would suggest. The gates matter more than the sum: they are what
separates a polymer that degrades gradually from one that has a named reaction
waiting at a particular temperature.

**crystallisability** is the prior question behind "rapid crystallisation or
other orientation lock". ``melt.crystallinity`` has computed it from chain
regularity since the melting-point tables were removed, and it was reachable
only as a *refusal* on the melting point - so a specification could be told
"this polymer has no melting point" but could not RANK on it. A drawn filament
of an amorphous polymer has no orientation lock: the chains are stretched, the
glass freezes them, and they relax back over hours to months. That is the
difference between a fibre and a piece of string that slowly becomes a puddle,
and it deserves to be a requirement rather than a footnote.

Accuracy. The group sum reproduces the eleven measured onset temperatures
below at 1.10x in sample and 1.37x left out one at a time, worst
poly(ethylene terephthalate). That is not good enough to choose between two
stable polymers - it is good enough to separate a polymer with a processing
window from one with none, which is the decision this property exists to make.
The gated chemistries are not predicted at all; they are stated, with their
reaction named, because a group sum cannot see a reaction.
"""

from __future__ import annotations

from dataclasses import dataclass

from formulate.core.candidate import Candidate, MaterialClass, MonomerRole, Tacticity
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest


@dataclass(frozen=True, slots=True)
class NamedDecomposition:
    """A chemistry that fires at a stated temperature, with its reaction."""

    temperature: float
    spread: float
    reaction: str


#: Repeat units whose decomposition is a NAMED REACTION rather than gradual
#: bond scission, keyed by a SMARTS over the repeat unit.
#:
#: These are stated rather than predicted, and the distinction is the point. A
#: group sum answers "how hot before bonds in general start breaking"; it has
#: no way to see that two carboxylic acids four atoms apart on one chain will
#: find each other. Where such a reaction exists it fires far below the group
#: sum and it decides the processing window on its own, so it overrides.
#:
#: Each entry names the reaction, because a refusal that only gives a number
#: leaves a chemist with nothing to check.
NAMED_DECOMPOSITION: dict[str, NamedDecomposition] = {
    # Poly(acrylic acid) and poly(methacrylic acid). Anhydride formation from
    # about 150 C, and it is not reversible on cooling.
    "[CX4][CX4]([CX3](=O)[OX2H1])": NamedDecomposition(
        423.0,
        25.0,
        "adjacent carboxylic acids on the backbone condense to a cyclic anhydride, "
        "losing water, from about 150 C - so a melt above that is not a melt of the "
        "same polymer, and the change is not undone by cooling",
    ),
    # Poly(vinyl chloride). Dehydrochlorination with autocatalysis.
    "[CX4][CX4]([Cl])": NamedDecomposition(
        473.0,
        20.0,
        "dehydrochlorination from about 200 C, releasing HCl which then catalyses more "
        "of the same - which is why PVC is never processed without a stabiliser and why "
        "its processing window is narrower than its softening point suggests",
    ),
    # Polyacrylonitrile. Nitrile cyclisation - the reaction carbon fibre is made by.
    "[CX4][CX4]([CX2]#[NX1])": NamedDecomposition(
        523.0,
        25.0,
        "the nitriles cyclise to a ladder polymer from about 250 C, which is the first "
        "step of carbon-fibre manufacture and the reason polyacrylonitrile is spun from "
        "solution and never from a melt",
    ),
    # Poly(vinyl alcohol). Dehydration to polyene.
    "[CX4][CX4]([OX2H1])": NamedDecomposition(
        473.0,
        25.0,
        "dehydration to a conjugated polyene from about 200 C, which colours the polymer "
        "before it flows",
    ),
}

#: Group contributions to the decomposition onset, K, over a fitted base.
#:
#: Additive over the repeat unit like every other group scheme here, and
#: FITTED to the eleven measured onsets below rather than asserted. The first
#: version of this table was written from chemical intuition - an aromatic ring
#: worth +120 K, a base of 400 - and it came out 0.59x to 0.82x against every
#: reference, which is to say uniformly and badly low. Intuition got the
#: ORDERING right and the magnitudes wrong, which is the usual way for a table
#: nobody checked.
#:
#: Only four groups survive. Eleven points cannot support eight, and the amide,
#: thioether, siloxane and quaternary-carbon terms all fitted to zero - not
#: because they do nothing, but because one or two references each cannot
#: separate them from the base. They are left out rather than kept at a value
#: the data does not support, so a polyamide and a polyethylene get the same
#: answer here and the uncertainty is what says that is a limitation.
#:
#: The ordering that survives is the well-established one: an aromatic ring in
#: the backbone raises the onset, fluorine raises it further, and an ester or
#: ether oxygen lowers it because the heteroatom bond is the weak one.
DECOMPOSITION_BASE = 636.5
DECOMPOSITION_GROUPS: dict[str, float] = {
    "c1ccccc1": 38.8,                 # aromatic ring anywhere on the unit
    "[F]": 32.9,                      # C-F, per fluorine
    "[CX3](=O)[OX2]": -27.4,          # ester: the weak link in a polyester
    "[OX2;!$(O[CX3]=O)]": -17.4,      # ether oxygen, excluding the ester's own
}

#: Worst ratio of the group sum against a held-out reference. In sample it is
#: 1.10; left out one at a time it is 1.37, on poly(ethylene terephthalate),
#: with a median of 1.08. The held-out figure is the one carried.
DECOMPOSITION_SPREAD = 1.37

#: Measured onset of decomposition in nitrogen, K, for the polymers used to
#: check the group sum. Compilations differ by tens of degrees because the
#: onset depends on heating rate and on what counts as onset, which is part of
#: why the spread carried is what it is.
DECOMPOSITION_REFERENCE: dict[str, float] = {
    "[*]CC[*]": 673.0,                              # polyethylene
    "[*]CC(C)[*]": 653.0,                           # polypropylene
    "[*]CC(c1ccccc1)[*]": 633.0,                    # polystyrene
    "[*]CC(C)(C(=O)OC)[*]": 553.0,                  # PMMA
    "[*]C(F)(F)C(F)(F)[*]": 773.0,                  # PTFE
    "[*]CCO[*]": 623.0,                             # poly(ethylene oxide)
    "[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]": 673.0,        # PET
    "[*]NCCCCCC(=O)[*]": 673.0,                     # nylon-6
    "[*]CCCCCC(=O)O[*]": 623.0,                     # polycaprolactone
    "[*]OC(C)C(=O)[*]": 573.0,                      # polylactide
    "[*]CC(F)(F)[*]": 693.0,                        # PVDF
}


def _unit_mol(repeat_unit: str):
    """The repeat unit with its attachment points INTACT.

    Capping ``[*]`` with ``[H]`` is what the parachor and the group schemes do,
    because they sum over atoms and the caps can be subtracted again. A SMARTS
    cannot subtract anything: capping invents functional groups that are not in
    the polymer, and this module was caught by exactly that. Poly(ethylene
    terephthalate) written ``[*]OCCOC(=O)...`` becomes ``[H]OCCOC(=O)...`` when
    capped, whose terminal ``[OX2H1]`` on a ``[CX4]`` is a hydroxyl - so PET
    matched the poly(vinyl alcohol) dehydration rule and was told it decomposes
    at 200 C, a polymer that is melt-spun at 280 every day.

    Leaving the dummies in place means the ester oxygen is an ``OX2`` with no
    hydrogen, which is what it actually is.
    """
    from rdkit import Chem

    return Chem.MolFromSmiles(repeat_unit)


def group_decomposition(repeat_unit: str) -> float | None:
    """Onset of decomposition from the repeat unit, K, or None."""
    from rdkit import Chem

    mol = _unit_mol(repeat_unit)
    if mol is None:
        return None
    total = DECOMPOSITION_BASE
    for smarts, increment in DECOMPOSITION_GROUPS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:  # pragma: no cover - the table is fixed
            continue
        total += increment * len(mol.GetSubstructMatches(pattern, uniquify=True))
    return total


def named_decomposition(repeat_unit: str) -> NamedDecomposition | None:
    """A specific decomposition reaction for this repeat unit, or None."""
    from rdkit import Chem

    mol = _unit_mol(repeat_unit)
    if mol is None:
        return None
    for smarts, entry in NAMED_DECOMPOSITION.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return entry
    return None


class ThermalStabilityExpert(Expert):
    """Where the processing window ends, and whether the chain can crystallise."""

    id = "thermal_stability"
    version = "1"
    method = (
        "group contributions to the decomposition onset over the repeat unit, overridden "
        "by named decomposition chemistries matched as SMARTS; crystallisability from "
        "backbone regularity (see experts.melt.crystallinity)"
    )
    family = PropertyFamily.THERMAL
    supported_classes = frozenset({MaterialClass.POLYMER})
    supported_properties = frozenset({"decomposition_temperature", "crystallisability"})

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "RDKit is required to read a repeat unit"

    @staticmethod
    def _repeat(candidate: Candidate) -> str | None:
        spec = candidate.polymer
        if spec is None:
            return None
        chain = [m for m in spec.monomers if m.role is not MonomerRole.END_GROUP]
        return chain[0].smiles if len(chain) == 1 else None

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        basis = (
            "group contributions checked against eleven measured decomposition onsets "
            f"between 0.88x and {DECOMPOSITION_SPREAD:.2f}x, with named chemistries stated "
            "rather than predicted"
        )
        if candidate.polymer is None:
            return ApplicabilityDomain.outside("candidate carries no polymer", basis=basis)
        if self._repeat(candidate) is None:
            return ApplicabilityDomain.outside(
                "a copolymer decomposes by whichever of its units goes first, and which one "
                "that is depends on sequence as well as composition",
                basis=basis,
            )
        return ApplicabilityDomain(basis=basis)

    def _predict_one(self, prop, request: PredictionRequest, domain):
        from .melt import crystallinity

        repeat = self._repeat(request.candidate)
        if repeat is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "candidate carries no single repeat unit: a copolymer decomposes by "
                "whichever of its units goes first, and which one that is depends on "
                "sequence as well as composition",
            )

        if prop == "crystallisability":
            spec = request.candidate.polymer
            tacticity = spec.tacticity if spec is not None else Tacticity.UNSPECIFIED
            phase = crystallinity(repeat, tacticity)
            if phase.crystalline is None:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"whether this chain can crystallise is genuinely open: {phase.reason}. "
                    "Answering 0 or 1 here would be picking, and the two answers are a "
                    "fibre and a piece of string",
                )
            return self._make(
                prop,
                1.0 if phase.crystalline else 0.0,
                "dimensionless",
                request,
                domain,
                std=0.0,
                kind=UncertaintyKind.EPISTEMIC,
                basis=(
                    "a structural test with no fitted constant: it is right or it is wrong, "
                    "and against the two tables it replaced it was right nineteen times in "
                    "twenty"
                ),
                notes=(
                    phase.reason,
                    "a drawn filament of a chain that cannot crystallise has no orientation "
                    "lock: the chains are stretched, the glass freezes them, and they relax "
                    "back over hours to months",
                    "this is NOT a degree of crystallinity, which is set by how a part was "
                    "cooled and is a processing variable rather than a property",
                ),
            )

        named = named_decomposition(repeat)
        if named is not None:
            return self._make(
                prop,
                named.temperature,
                "K",
                request,
                domain,
                std=named.spread,
                kind=UncertaintyKind.ALEATORIC,
                basis=(
                    "a stated reaction rather than a group sum, so the spread is the spread "
                    "between compilations of where that reaction becomes fast, not a model "
                    "error"
                ),
                notes=(
                    named.reaction,
                    "this overrides the group contribution, which would put the onset far "
                    "higher: a group sum answers how hot before bonds in general break and "
                    "cannot see a specific reaction waiting",
                ),
            )

        value = group_decomposition(repeat)
        if value is None:
            return Prediction.unsupported(
                prop, self.id, "the repeat unit could not be read"
            )
        return self._make(
            prop,
            value,
            "K",
            request,
            domain,
            std=value * (DECOMPOSITION_SPREAD - 1.0),
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                f"group contributions fitted to eleven measured onsets: 1.10x in sample and "
                f"{DECOMPOSITION_SPREAD:.2f}x left out one at a time, which is the figure "
                "carried. Too wide to choose between two stable polymers; wide enough to "
                "separate a polymer with a processing window from one with none"
            ),
            notes=(
                "no named decomposition chemistry matched this repeat unit, so this is the "
                "gradual-scission estimate rather than a reaction with a temperature",
                "onset depends on heating rate and on what counts as onset; compilations "
                "differ by tens of degrees, which is part of the spread carried here",
            ),
        )
