"""What a substance does to the person handling it, from a curated table only.

The engine chose a benzoyl peroxide / N,N-dimethyl-p-toluidine redox pair to
cure a hexanediol diacrylate base, and every one of those three is a hazard the
run could not see.  The diacrylate is a skin sensitiser, which is the classic
occupational injury of acrylate chemistry and is permanent once acquired.  The
peroxide is an organic peroxide and a sensitiser as well.  The amine is toxic by
all three routes.  None of that cost the recipe a point, because hazard was not
a property, and a property the registry does not carry cannot lose a candidate
anything.

**Every value here is looked up and none is inferred.**  That is the whole
design of this module and it is worth being explicit about why, because
inferring would be easy and there are structural alerts for sensitisation in
every toxicology textbook - a Michael acceptor, an epoxide, an isocyanate.  Two
reasons not to.  The first is that they are wrong often enough to matter in both
directions: methyl methacrylate is a Michael acceptor and a sensitiser, methyl
acrylate is a Michael acceptor and a sensitiser, and butyl acetate is neither
while looking like an ester that could be either.  The second is what a false
negative costs.  A wrong boiling point wastes a synthesis; a wrong "not a
sensitiser" is somebody's hands, for the rest of their working life.  So a
structure not in the table gets a refusal naming itself, and the refusal is the
correct answer rather than a gap in coverage.

**Three endpoints, and the third is here to stop the other two lying.**  A
screen carrying only skin sensitisation and acute toxicity would report benzene
as unobjectionable: benzene is not a sensitiser and is not acutely toxic at any
GHS category, and it is a category 1A carcinogen.  A hazard expert that returned
two clean numbers for benzene would be worse than no hazard expert at all,
because a clean bill of health is acted on and a missing one is not.  So
carcinogenicity is carried as well, and the endpoints stay separate properties
rather than being averaged into one "safety" score - averaging would let a
strong pass on one endpoint pay for a failure on another, and hazards do not
trade off that way.

**What the table still does not carry**, and therefore what a spec built on it
still cannot see: reproductive toxicity, specific target organ toxicity,
mutagenicity, aspiration hazard, flammability, environmental hazard, and every
exposure limit.  N,N-dimethylformamide is in this table as acute category 4 and
is a reproductive toxicant, which is the reason it is being restricted in
Europe, and that fact is invisible here.  Every prediction says so in its own
notes.  This is a hazard *screen*, and screening out is all it can do; nothing
in it is a regulatory determination or a substitute for a safety data sheet.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from formulate.core.candidate import Candidate, MaterialClass, MonomerRole
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest

#: Source classes, named per row so a reader can tell a harmonised legal
#: classification from a supplier's own.
CLP = "EU CLP Annex VI harmonised classification"
SELF = "GHS self-classification, consistent across major supplier safety data sheets"
POLYMER = "high polymers are not classified as such; see the row's own note"

#: The category reserved for a substance screened against every route of the
#: endpoint and classified on none.  Not the same as absent from the table,
#: which is a refusal.
NOT_CLASSIFIED_ACUTE = 5
NOT_CLASSIFIED_CARCINOGEN = 4


@dataclass(frozen=True, slots=True)
class HazardRow:
    """One substance's classification on the three endpoints carried here."""

    name: str
    #: Every hazard statement code the source lists, including endpoints this
    #: module does not turn into a property.  Reported in the notes so a reader
    #: sees what the screen is not looking at.
    codes: tuple[str, ...]
    #: True where the source lists H317 (Skin Sens. 1, 1A or 1B).
    skin_sensitiser: bool
    #: Most severe GHS acute toxicity category over oral, dermal and
    #: inhalation.  5 means screened and classified on none.
    acute_category: int
    #: 1 for GHS carcinogenicity 1A, 2 for 1B, 3 for category 2, 4 for
    #: screened and not classified.  ``None`` where this module's compiler was
    #: not confident enough to record one, which is a refusal for that endpoint
    #: and not a pass.
    carcinogen_category: int | None
    source: str
    note: str = ""


def _row(
    smiles: str,
    name: str,
    codes: str,
    *,
    sens: bool = False,
    acute: int = NOT_CLASSIFIED_ACUTE,
    carc: int | None = NOT_CLASSIFIED_CARCINOGEN,
    source: str = CLP,
    note: str = "",
) -> tuple[str, HazardRow]:
    return smiles, HazardRow(
        name=name,
        codes=tuple(c.strip() for c in codes.split(",") if c.strip()),
        skin_sensitiser=sens,
        acute_category=acute,
        carcinogen_category=carc,
        source=source,
        note=note,
    )


#: The curated table, keyed on SMILES as written and matched on canonical form.
#:
#: It covers every structure in the bundled molecular catalogue, every repeat
#: unit in the polymer catalogue, and the two initiator components the cure runs
#: put in the recipe but never in the pool.  A structure outside it is refused.
HAZARDS: dict[str, HazardRow] = dict(
    [
        # --- solvents and small molecules from the reference set ---------------
        _row("O", "water", ""),
        _row(
            "CO",
            "methanol",
            "H225, H301, H311, H331, H370",
            acute=3,
            note="toxic by all three routes and specifically to the optic nerve",
        ),
        _row("CCO", "ethanol", "H225"),
        _row("CCCO", "1-propanol", "H225, H318, H336"),
        _row("CC(C)O", "isopropanol", "H225, H319, H336"),
        _row("CCCCO", "1-butanol", "H226, H302, H315, H318, H335, H336", acute=4),
        _row("CC(C)=O", "acetone", "H225, H319, H336"),
        _row("CCC(C)=O", "2-butanone", "H225, H319, H336"),
        _row("O=C1CCCCC1", "cyclohexanone", "H226, H302, H332, H315, H318", acute=4),
        _row("CCOCC", "diethyl ether", "H224, H302, H336", acute=4),
        _row(
            "C1CCOC1",
            "tetrahydrofuran",
            "H225, H302, H319, H335, H351",
            acute=4,
            carc=3,
        ),
        _row(
            "C1COCCO1",
            "1,4-dioxane",
            "H225, H319, H335, H351",
            carc=3,
        ),
        _row("COc1ccccc1", "anisole", "H226", source=SELF),
        _row("CCOC(C)=O", "ethyl acetate", "H225, H319, H336"),
        _row("CC(=O)O", "acetic acid", "H226, H314"),
        _row("CCCCC", "pentane", "H225, H304, H336, H411"),
        _row("CCCCCC", "hexane", "H225, H304, H315, H336, H361f, H373, H411"),
        _row("CCCCCCC", "heptane", "H225, H304, H315, H336, H410"),
        _row("CCCCCCCC", "octane", "H225, H304, H315, H336, H410"),
        _row("CCCCCCCCCC", "decane", "H226, H304, H410", source=SELF),
        _row("CCCCCCCCCCCC", "dodecane", "H304", source=SELF),
        _row("C1CCCCC1", "cyclohexane", "H225, H304, H315, H336, H410"),
        _row(
            "c1ccccc1",
            "benzene",
            "H225, H304, H315, H319, H340, H350, H372",
            carc=1,
            note=(
                "the reason this module carries a carcinogenicity endpoint at all: "
                "benzene is not a sensitiser and is not acutely toxic at any category, "
                "so a screen without it would return two clean numbers for a known "
                "human carcinogen"
            ),
        ),
        _row("Cc1ccccc1", "toluene", "H225, H304, H315, H336, H361d, H373"),
        _row("Cc1ccc(C)cc1", "p-xylene", "H226, H304, H312, H315, H332", acute=4),
        _row(
            "C=Cc1ccccc1",
            "styrene",
            "H226, H304, H315, H319, H332, H335, H361d, H372",
            acute=4,
            carc=3,
        ),
        _row("c1ccc2ccccc2c1", "naphthalene", "H228, H302, H351, H410", acute=4, carc=3),
        _row(
            "Oc1ccccc1",
            "phenol",
            "H301, H311, H331, H314, H341, H373",
            acute=3,
        ),
        _row(
            "Nc1ccccc1",
            "aniline",
            "H301, H311, H331, H317, H318, H341, H351, H372, H410",
            sens=True,
            acute=3,
            carc=3,
        ),
        _row(
            "O=[N+]([O-])c1ccccc1",
            "nitrobenzene",
            "H301, H311, H331, H351, H360F, H372, H411",
            acute=3,
            carc=3,
        ),
        _row("c1ccncc1", "pyridine", "H225, H302, H312, H332, H315, H319", acute=4),
        _row("CC#N", "acetonitrile", "H225, H302, H312, H332, H319", acute=4),
        _row("ClCCl", "dichloromethane", "H315, H319, H335, H336, H351", carc=3),
        _row(
            "ClC(Cl)Cl",
            "chloroform",
            "H302, H315, H319, H331, H351, H361d, H372",
            acute=3,
            carc=3,
        ),
        _row(
            "ClC(Cl)(Cl)Cl",
            "carbon tetrachloride",
            "H301, H311, H331, H351, H372, H420",
            acute=3,
            carc=3,
        ),
        _row("Clc1ccccc1", "chlorobenzene", "H226, H332, H411", acute=4),
        _row("CS(C)=O", "dimethyl sulfoxide", "", source=SELF),
        _row(
            "CN(C)C=O",
            "N,N-dimethylformamide",
            "H226, H312, H332, H319, H360D",
            acute=4,
            note=(
                "acute category 4 and a reproductive toxicant, which is the reason it "
                "is being restricted in Europe and is an endpoint this table does not "
                "carry"
            ),
        ),
        _row("CC1COC(=O)O1", "propylene carbonate", "H319", source=SELF),
        _row("O=C1CCCO1", "gamma-butyrolactone", "H302, H318, H336", acute=4),
        _row("OCCO", "ethylene glycol", "H302, H373", acute=4),
        _row("OCC(O)CO", "glycerol", "", source=SELF),
        _row(
            "CC1=CCC(CC1)C(=C)C",
            "limonene",
            "H226, H304, H315, H317, H410",
            sens=True,
            note="the oxidation products, not the fresh terpene, carry the sensitisation",
        ),
        _row("CCOC(=O)C(C)O", "ethyl lactate", "H226, H318, H335", source=SELF),
        _row("CC(C)CC(C)=O", "methyl isobutyl ketone", "H225, H319, H332, H335", acute=4),
        _row("CCCCOC(C)=O", "butyl acetate", "H226, H336"),
        _row("CC(C)OC(C)C", "diisopropyl ether", "H224, H336, H351", carc=3),
        _row(
            "ClCCCl",
            "1,2-dichloroethane",
            "H225, H302, H315, H319, H335, H350",
            acute=4,
            carc=2,
        ),
        _row("Cc1cc(C)cc(C)c1", "mesitylene", "H226, H304, H315, H335, H411", source=SELF),
        _row(
            "CN1CCCC1=O",
            "n-methylpyrrolidone",
            "H315, H319, H335, H360D",
            note="a reproductive toxicant, which this table does not carry as a property",
        ),
        # --- monomers ---------------------------------------------------------
        # Every acrylate and methacrylate in the catalogue is a skin sensitiser.
        # That is not a structural inference drawn here; it is what each one's
        # own classification says, and it is the single most common occupational
        # injury in acrylate chemistry.
        _row(
            "C=CC(=O)OC",
            "methyl acrylate",
            "H225, H302, H312, H332, H315, H317, H319, H335",
            sens=True,
            acute=4,
        ),
        _row(
            "C=CC(=O)OCC",
            "ethyl acrylate",
            "H225, H302, H312, H332, H315, H317, H319, H335, H351",
            sens=True,
            acute=4,
            carc=3,
        ),
        _row(
            "C=CC(=O)OCCCC",
            "butyl acrylate",
            "H226, H315, H317, H319, H332, H335",
            sens=True,
            acute=4,
        ),
        _row(
            "C=C(C)C(=O)OC",
            "methyl methacrylate",
            "H225, H315, H317, H335",
            sens=True,
        ),
        _row(
            "C=C(C)C(=O)OCCCC",
            "butyl methacrylate",
            "H226, H315, H317, H335",
            sens=True,
        ),
        _row(
            "C=COC(C)=O",
            "vinyl acetate",
            "H225, H332, H335, H351",
            acute=4,
            carc=3,
        ),
        _row(
            "C=CC(=O)OCCCCCCOC(=O)C=C",
            "1,6-hexanediol diacrylate",
            "H315, H317, H319, H335, H400",
            sens=True,
            note=(
                "the base five earlier runs selected. A difunctional acrylate and a "
                "skin sensitiser, which is the classification that matters most for a "
                "two-part system mixed at a nozzle: the person holding it meets the "
                "uncured monomer, not the network"
            ),
        ),
        _row(
            "C=C(C)C(=O)OCCOC(=O)C(=C)C",
            "ethylene glycol dimethacrylate",
            "H315, H317, H319, H335",
            sens=True,
        ),
        _row(
            "C=CC(=O)OCC(CC)(COC(=O)C=C)COC(=O)C=C",
            "trimethylolpropane triacrylate",
            "H315, H317, H319, H335, H411",
            sens=True,
        ),
        # --- the initiator system, which was in the recipe and never in the pool
        _row(
            "O=C(OOC(=O)c1ccccc1)c1ccccc1",
            "dibenzoyl peroxide",
            "H242, H317, H319, H400",
            sens=True,
            note=(
                "H242 is an organic peroxide: it is self-reactive, it decomposes "
                "exothermically, and it is shipped wet with water or phlegmatiser for "
                "that reason. Neither the self-reactivity nor the oxidising hazard is "
                "a property this table turns into a number, so a spec screening on "
                "sensitisation and acute toxicity alone sees only half of what is "
                "wrong with putting it in a hand-held cartridge"
            ),
        ),
        _row(
            "CN(C)c1ccc(C)cc1",
            "N,N-dimethyl-p-toluidine",
            "H301, H311, H331, H373, H411",
            acute=3,
            carc=None,
            note=(
                "the accelerator half of the redox pair: toxic by mouth, by skin and "
                "by inhalation, all three at category 3. Carcinogenicity is left "
                "unrecorded rather than set to 'not classified': several aromatic "
                "amines are carcinogens, aniline among them and in the row above, and "
                "recording a four here would assert a screening nobody did"
            ),
        ),
        # --- polymer repeat units --------------------------------------------
        # A high polymer is not classified as a hazardous substance: it is not
        # absorbed, so there is nothing to be acutely toxic or sensitising. The
        # rows exist so the panel can say that with a source behind it rather
        # than by finding nothing and refusing. What they do NOT say is that the
        # material is safe to make: every one of these is polymerised from a
        # monomer with its own classification, and several of those monomers are
        # in the rows above.
        _row("[*]CC([*])c1ccccc1", "polystyrene", "", source=POLYMER),
        _row("[*]CC([*])(C)C(=O)OC", "poly(methyl methacrylate)", "", source=POLYMER),
        _row("[*]CC([*])OC(C)=O", "poly(vinyl acetate)", "", source=POLYMER),
        _row("[*]CC[*]", "polyethylene", "", source=POLYMER),
        _row("[*]CC([*])C", "polypropylene", "", source=POLYMER),
        _row("[*]NCCCCCC(=O)[*]", "polyamide 6", "", source=POLYMER),
        _row("[*]NCCCCCCNC(=O)CCCCC(=O)[*]", "polyamide 6,6", "", source=POLYMER),
        _row("[*]NCCCCCCCCCCCC(=O)[*]", "polyamide 12", "", source=POLYMER),
        _row(
            "[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]",
            "poly(ethylene terephthalate)",
            "",
            source=POLYMER,
        ),
        _row(
            "[*]OCCCCOC(=O)c1ccc(cc1)C(=O)[*]",
            "poly(butylene terephthalate)",
            "",
            source=POLYMER,
        ),
        _row("[*]CCCCCC(=O)O[*]", "polycaprolactone", "", source=POLYMER),
        _row("[*]OC(C)C(=O)[*]", "polylactide", "", source=POLYMER),
        _row("[*]CCCCO[*]", "poly(tetramethylene oxide)", "", source=POLYMER),
        _row(
            "[*]OCCCCOC(=O)Nc1ccc(Cc2ccc(NC(=O)[*])cc2)cc1",
            "thermoplastic polyurethane hard segment",
            "",
            source=POLYMER,
            note=(
                "the polymer is not classified; the diisocyanate it is made from is a "
                "respiratory and skin sensitiser, and a thermoplastic polyurethane "
                "processed above about 200 C can liberate some of it again"
            ),
        ),
        _row(
            "[*]CC([*])C(=O)OCCCCCCOC(=O)C([*])C[*]",
            "crosslinked poly(1,6-hexanediol diacrylate)",
            "H317",
            sens=True,
            carc=None,
            source=SELF,
            note=(
                "the classification is inherited from the monomer by a decision "
                "recorded here rather than derived from the network's structure. Two "
                "reasons it is the right inheritance. A two-part acrylate is mixed at "
                "the point of use, so the uncured diacrylate is what reaches skin. And "
                "a radical cure stops short of complete conversion, so residual "
                "monomer stays in the article and leaches - which is why cured acrylate "
                "articles are themselves a recognised cause of acrylate sensitisation. "
                "Sensitisation inherits and carcinogenicity is left unrecorded, because "
                "the inheritance argument is about what reaches skin and there is no "
                "equivalent argument, or datum, for the other endpoint"
            ),
        ),
    ]
)


@lru_cache(maxsize=1)
def _canonical_table() -> dict[str, HazardRow]:
    """The table re-keyed on canonical SMILES.

    Written with explicit ``[*]`` attachment points for the repeat units,
    because that is how a repeat unit is written everywhere else here; RDKit
    canonicalises those to ``*``, so a lookup against the table as written
    misses every polymer in it.
    """
    from formulate import chem

    if not chem.rdkit_available():
        return dict(HAZARDS)
    out: dict[str, HazardRow] = {}
    for smiles, row in HAZARDS.items():
        out[chem.canonical_smiles(smiles) or smiles] = row
    return out


def hazard_row(smiles: str) -> HazardRow | None:
    """The table row for a structure, or ``None`` if it is not in the table."""
    from formulate import chem

    if not chem.rdkit_available():
        return HAZARDS.get(smiles)
    canonical = chem.canonical_smiles(smiles)
    if canonical is None:
        return None
    return _canonical_table().get(canonical)


def _structures(candidate: Candidate) -> list[str]:
    """The structures a person handling this candidate is exposed to.

    For a polymer that is its repeat units, end groups excluded: an end group
    is present at parts per thousand and is not what anyone touches.  For a
    mixture it is every component, because a formulation's hazard is the worst
    of its parts and not an average of them - a hundredth of a per cent of a
    sensitiser sensitises.
    """
    if candidate.polymer is not None:
        return [
            m.smiles
            for m in candidate.polymer.monomers
            if m.role is not MonomerRole.END_GROUP
        ]
    return candidate.all_smiles()


class GHSHazardExpert(Expert):
    """Skin sensitisation, acute toxicity and carcinogenicity, by lookup only.

    Applies to molecules, polymers and formulations.  For anything with more
    than one structure the answer is the worst of them, never a
    composition-weighted mean: a formulation containing one per cent of a
    sensitiser is a sensitiser, and averaging would let ninety-nine per cent of
    an inert carrier buy it a pass.

    Every structure must be in the table.  One unknown component and the whole
    candidate is refused, with the unknown named, because a mixture whose worst
    component was not looked at has not been screened.
    """

    id = "hazard_ghs"
    version = "1"
    method = "curated GHS classification table; lookup only, nothing inferred from structure"
    family = PropertyFamily.SPECIALIZED
    supported_classes = frozenset(
        {MaterialClass.MOLECULE, MaterialClass.POLYMER, MaterialClass.MIXTURE}
    )
    supported_properties = frozenset(
        {"skin_sensitiser", "acute_toxicity_category", "carcinogen_category"}
    )

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "RDKit is required to match a structure"

    def _missing(self, candidate: Candidate) -> list[str]:
        return [s for s in _structures(candidate) if hazard_row(s) is None]

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        basis = f"{len(HAZARDS)} structures with a recorded GHS classification"
        structures = _structures(candidate)
        if not structures:
            return ApplicabilityDomain.outside("candidate carries no structure", basis=basis)
        missing = self._missing(candidate)
        if missing:
            return ApplicabilityDomain.outside(
                "not in the hazard table: "
                + ", ".join(sorted(missing))
                + ". A classification is a laboratory result and there is no route to "
                "one from a structure that is worth putting a person's skin behind",
                basis=basis,
            )
        return ApplicabilityDomain(basis="a recorded classification for every structure present")

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        candidate = request.candidate
        structures = _structures(candidate)
        if not structures:
            return Prediction.unsupported(prop, self.id, "candidate carries no structure")

        missing = self._missing(candidate)
        if missing:
            return Prediction.unsupported(
                prop,
                self.id,
                "no recorded GHS classification for "
                + ", ".join(sorted(missing))
                + ". Refusing rather than inferring: a structural alert for skin "
                "sensitisation is wrong often enough in both directions that a false "
                "negative here would be somebody's hands",
            )

        rows = [hazard_row(s) for s in structures]
        assert all(row is not None for row in rows)
        rows = [row for row in rows if row is not None]

        if prop == "skin_sensitiser":
            worst = [row for row in rows if row.skin_sensitiser]
            value = 1.0 if worst else 0.0
            driver = worst[0] if worst else rows[0]
            reason = (
                f"{', '.join(sorted(r.name for r in worst))} carries H317"
                if worst
                else "no component in this candidate carries H317"
            )
        elif prop == "acute_toxicity_category":
            driver = min(rows, key=lambda r: r.acute_category)
            value = float(driver.acute_category)
            reason = (
                f"{driver.name} at category {driver.acute_category}"
                if driver.acute_category < NOT_CLASSIFIED_ACUTE
                else "no component is classified for acute toxicity by any route"
            )
        else:
            scored = [row for row in rows if row.carcinogen_category is not None]
            if len(scored) != len(rows):
                unscored = sorted(r.name for r in rows if r.carcinogen_category is None)
                return Prediction.unsupported(
                    prop,
                    self.id,
                    "the table records no carcinogenicity classification for "
                    + ", ".join(unscored)
                    + "; an unrecorded endpoint is a refusal and not a pass",
                )
            driver = min(scored, key=lambda r: r.carcinogen_category or 0)
            value = float(driver.carcinogen_category or NOT_CLASSIFIED_CARCINOGEN)
            reason = (
                f"{driver.name} at {_CARCINOGEN_NAMES[int(value)]}"
                if value < NOT_CLASSIFIED_CARCINOGEN
                else "no component is classified as a carcinogen"
            )

        codes = sorted({code for row in rows for code in row.codes})
        notes = [
            reason,
            "the worst component, not a composition-weighted average: a formulation "
            "containing one per cent of a sensitiser is a sensitiser",
            "every hazard statement on the structures present: "
            + (", ".join(codes) if codes else "none"),
            "this screen carries skin sensitisation, acute toxicity and "
            "carcinogenicity only. Reproductive toxicity, target-organ toxicity, "
            "mutagenicity, aspiration, flammability, environmental hazard and every "
            "exposure limit are outside it, and a pass here is not a clean bill",
            "sources: " + ", ".join(sorted({row.source for row in rows if row.source})),
        ]
        notes.extend(row.note for row in rows if row.note)

        return self._make(
            prop,
            value,
            "",
            request,
            domain,
            # A classification is a recorded categorical fact. It carries real
            # uncertainty - classifications are revised, and two jurisdictions
            # disagree - but not an uncertainty this expert can put a number on,
            # and inventing one would make it comparable to a prediction it is
            # not comparable to.
            std=0.0,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                "a recorded classification rather than an estimate; the zero is the "
                "lookup being exact, not the classification being certain"
            ),
            notes=tuple(notes),
            structures=len(structures),
        )


#: What each carcinogenicity code means, for the note that reports it.
_CARCINOGEN_NAMES = {
    1: "GHS category 1A, known human carcinogen",
    2: "GHS category 1B, presumed human carcinogen",
    3: "GHS category 2, suspected human carcinogen",
    4: "screened and not classified",
}


def hazard_experts() -> list[Expert]:
    return [GHSHazardExpert()]
