"""Hansen solubility parameters.

Specification section 4 lists solubility and compatibility under the chemical
and interfacial expert families, and section 12 makes formulation compatibility
a success metric.

A single Hildebrand parameter cannot answer a compatibility question.  Ethanol
and nitromethane have almost the same total cohesive energy density and are
poor substitutes for one another, because ethanol holds its cohesion in
hydrogen bonds and nitromethane in dipolar interactions.  Splitting the
parameter into dispersion, polar and hydrogen-bonding components is what makes
the distance between two materials meaningful.

Values come from a curated published database rather than a correlation, per
the section 4 model policy: reuse an adequate open source rather than fit
something new.  A compound absent from it returns no value at all, because a
Hansen triple estimated from a total solubility parameter would be an
underdetermined guess dressed as a measurement.
"""

from __future__ import annotations

import functools
import math

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest

try:  # pragma: no cover
    from chemicals import CAS_from_any
    from chemicals.solubility import hansen_delta_d, hansen_delta_h, hansen_delta_p

    _HAVE_CHEMICALS = True
    _IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    _HAVE_CHEMICALS = False
    _IMPORT_ERROR = str(exc)


_COMPONENTS = {
    "hansen_dispersion": "d",
    "hansen_polar": "p",
    "hansen_hydrogen_bonding": "h",
}

#: Spread between the published compilations for the same compound. The
#: databases disagree by roughly this much on well-studied solvents, and rather
#: more on rare ones, so it is the honest floor on a looked-up value rather
#: than the zero a database lookup might suggest.
_COMPILATION_SPREAD_PA_SQRT = 500.0  # 0.5 MPa^0.5


@functools.lru_cache(maxsize=4096)
def resolve_cas(smiles: str) -> str | None:
    """Resolve a structure to a CAS number through its InChIKey.

    The InChIKey route is used rather than a SMILES lookup because the
    database matches SMILES as an exact string: benzene, toluene, acetone and
    dimethyl sulfoxide all fail to resolve from a perfectly valid SMILES, while
    all ten test solvents resolved from their InChIKeys.
    """
    from formulate import chem

    if not _HAVE_CHEMICALS or not chem.rdkit_available():
        return None
    key = chem.inchi_key(smiles)
    if key is None:
        return None
    try:
        return CAS_from_any("InChIKey=" + key)
    except Exception:
        return None


@functools.lru_cache(maxsize=4096)
def hansen_triple(smiles: str) -> tuple[float, float, float] | None:
    """Return (dispersion, polar, hydrogen bonding) in Pa^0.5, or None."""
    cas = resolve_cas(smiles)
    if cas is None:
        return None
    try:
        values = (hansen_delta_d(cas), hansen_delta_p(cas), hansen_delta_h(cas))
    except Exception:
        return None
    if any(v is None for v in values):
        return None
    return tuple(float(v) for v in values)  # type: ignore[return-value]


def hansen_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Hansen distance Ra between two materials, in the same unit as the inputs.

    Ra^2 = 4 (dD1 - dD2)^2 + (dP1 - dP2)^2 + (dH1 - dH2)^2

    The factor of four on the dispersion term is not a fitting constant. It
    makes the solubility region in Hansen space a sphere rather than an
    ellipsoid: doubling the dispersion axis is what empirically renders the
    three components commensurable, and dropping it distorts every distance.
    """
    return (
        4.0 * (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    ) ** 0.5


def total_solubility_parameter(triple: tuple[float, float, float]) -> float:
    """The Hildebrand parameter implied by a Hansen triple.

    delta^2 = dD^2 + dP^2 + dH^2, which is what makes the split a
    decomposition rather than a separate quantity.
    """
    return (triple[0] ** 2 + triple[1] ** 2 + triple[2] ** 2) ** 0.5


class HansenSolubilityExpert(Expert):
    """Looks up the three Hansen components for a molecule."""

    id = "hansen"
    version = "1"
    method = "curated Hansen solubility parameter compilation (via the chemicals package)"
    family = PropertyFamily.INTERFACIAL
    supported_classes = frozenset({MaterialClass.MOLECULE})
    supported_properties = frozenset(_COMPONENTS)

    def is_available(self) -> bool:
        from formulate import chem

        return _HAVE_CHEMICALS and chem.rdkit_available()

    def unavailable_reason(self) -> str:
        from formulate import chem

        if not _HAVE_CHEMICALS:
            return f"the chemicals package is not installed ({_IMPORT_ERROR})"
        if not chem.rdkit_available():
            return "RDKit is required to resolve a structure to an InChIKey"
        return ""

    def _software(self) -> SoftwareEnvironment:
        import chemicals

        from formulate import chem

        return SoftwareEnvironment.capture(
            chemicals=chemicals.__version__, rdkit=chem.rdkit_version()
        )

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        smiles = candidate.molecule.smiles if candidate.molecule else None
        if smiles is None:
            return ApplicabilityDomain.outside("candidate carries no molecule")
        if hansen_triple(smiles) is None:
            return ApplicabilityDomain.outside(
                "this compound is not in the Hansen compilation",
                basis="a curated table of measured solvents, not a correlation",
            )
        return ApplicabilityDomain(
            basis=(
                "a tabulated measurement for this compound; the compilations differ from "
                "one another by a few tenths of a MPa^0.5"
            )
        )

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        smiles = request.candidate.molecule.smiles  # type: ignore[union-attr]
        triple = hansen_triple(smiles)
        if triple is None:
            # An estimate is not produced. Three components cannot be recovered
            # from one total solubility parameter: the split is underdetermined,
            # and a guess presented alongside looked-up values for other
            # candidates would corrupt every distance computed against it.
            return Prediction.unsupported(
                prop,
                self.id,
                (
                    "this compound is not in the Hansen compilation, and the three "
                    "components cannot be recovered from a single total solubility "
                    "parameter, so no value is estimated"
                ),
            )

        index = "dph".index(_COMPONENTS[prop])
        return self._make(
            prop,
            triple[index],
            "Pa^0.5",
            request,
            domain,
            std=_COMPILATION_SPREAD_PA_SQRT,
            kind=UncertaintyKind.ALEATORIC,
            basis=(
                "spread between the published compilations for the same compound, "
                "about 0.5 MPa^0.5 for well-studied solvents and larger for rare ones; "
                "a tabulated value is a measurement, not an exact number"
            ),
            notes=(
                "measured and tabulated rather than estimated",
                f"CAS {resolve_cas(smiles)}",
            ),
        )


# --------------------------------------------------------------------------
# Estimating a Hansen triple where the compilation has none
# --------------------------------------------------------------------------

#: Hoftyzer-Van Krevelen group contributions: (SMARTS, Fd, Fp, Eh).
#:
#: ``Fd`` and ``Fp`` are in (MJ/m^3)^0.5 cm^3/mol and ``Eh`` in J/mol, from
#: Van Krevelen's table. The triple follows as
#:
#:     delta_D = sum(Fd) / V
#:     delta_P = sqrt(sum(Fp^2)) / V
#:     delta_H = sqrt(sum(Eh) / V)
#:
#: The dispersion term adds linearly, the polar term **in quadrature**, and
#: the hydrogen bonding term as an energy. Getting the middle one wrong is
#: easy and quiet: summing Fp linearly instead put the mean absolute error on
#: the polar component at 2.42 MPa^0.5 against the 0.87 it should be, and
#: nothing about the answers looked unusual.
#:
#: The table is deliberately short. Every group here is one whose value was
#: checked against the compilation over the reference set; a molecule
#: containing anything else is refused rather than estimated from a partial
#: sum, because a missing group does not add noise, it removes a contribution
#: and leaves an ordinary looking number. Dimethyl sulfoxide with no
#: sulfoxide group comes back at 11.8 against a measured 18.4.
_HVK_GROUPS: tuple[tuple[str, float, float, float], ...] = (
    ("[CX4H3]", 420.0, 0.0, 0.0),
    ("[CX4H2]", 270.0, 0.0, 0.0),
    ("[CX4H1]", 80.0, 0.0, 0.0),
    ("[CX4H0]", -70.0, 0.0, 0.0),
    ("[CX3H2]=[CX3]", 400.0, 0.0, 0.0),
    ("[CX3H1](=[CX3])", 200.0, 0.0, 0.0),
    ("[CX3H0;$([CX3]=[CX3])]", 70.0, 0.0, 0.0),
    # Per aromatic carbon, so a phenyl adds up to Van Krevelen's 1430 and 110.
    # The polar figure is 110/sqrt(6) rather than 110/6, because the polar
    # term adds in quadrature: splitting it evenly gave a phenyl 44 instead
    # of 110, and polystyrene a polar parameter of 0.4 against a real 4.5.
    ("[cX3]", 238.0, 44.9, 0.0),
    # The ester oxygen must not itself be bonded to oxygen, or a peroxyester
    # matches and a peroxide is typed as two esters. Every atom is covered
    # in that case, so the unmatched-atom refusal cannot see it: benzoyl
    # peroxide came back as an ordinary 18.2 / 3.5 / 8.4.
    ("[CX3](=[OX1])[OX2;!$([OX2][OX2])]", 390.0, 490.0, 7000.0),
    ("[CX3;$([CX3]([#6])[#6])]=[OX1]", 290.0, 770.0, 2000.0),
    ("[OX2;$([OX2]([#6])[#6]);!$([OX2][CX3]=O)]", 100.0, 400.0, 3000.0),
    ("[OX2H]", 210.0, 500.0, 20000.0),
    ("[NX3;H2]", 280.0, 0.0, 8400.0),
    ("[NX3;H0;!$(NC=O)]", 20.0, 800.0, 5000.0),
    ("[Cl]", 450.0, 550.0, 400.0),
    ("[CX2]#[NX1]", 430.0, 1100.0, 2500.0),
)

#: A carbon carrying two or more halogens. The ``-Cl`` contribution is
#: calibrated for an isolated substituent and does not survive being stacked:
#: with it, carbon tetrachloride comes out at a polar 11.3 against a measured
#: zero. Refused rather than corrected, since one compound is no basis for a
#: correction.
_POLYHALOGENATED = "[#6](-[F,Cl,Br,I])-[F,Cl,Br,I]"

#: One-sigma spreads, as 1.253 times the mean absolute error measured against
#: the compilation over the forty reference structures this method accepts.
HVK_SIGMA: dict[str, float] = {
    "hansen_dispersion": 1.12,
    "hansen_polar": 1.01,
    "hansen_hydrogen_bonding": 1.48,
}


@functools.lru_cache(maxsize=1)
def _hvk_compiled():
    from rdkit import Chem

    return (
        tuple((Chem.MolFromSmarts(s), fd, fp, eh) for s, fd, fp, eh in _HVK_GROUPS),
        Chem.MolFromSmarts(_POLYHALOGENATED),
    )


def hoftyzer_van_krevelen(smiles: str, molar_volume_cm3: float) -> tuple[float, float, float] | None:
    """A Hansen triple in MPa^0.5, or None if any atom is left uncovered."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or molar_volume_cm3 <= 0.0:
        return None
    groups, polyhalogen = _hvk_compiled()
    if mol.HasSubstructMatch(polyhalogen):
        return None

    used: set[int] = set()
    dispersion = polar_squared = bonding = 0.0
    for pattern, fd, fp, eh in groups:
        for match in mol.GetSubstructMatches(pattern):
            if any(atom in used for atom in match):
                continue
            used.update(match)
            dispersion += fd
            polar_squared += fp * fp
            bonding += eh
    # Attachment points are bonds to the next repeat unit, not atoms, and a
    # polymer's repeat unit is written with them. Leaving them in is what makes
    # the H counts right - the backbone carbon of polystyrene is a CH2 because
    # two of its four connections are dummies - so they are skipped here rather
    # than stripped, which would turn that CH2 into a CH3.
    real = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 0}
    if real - used:
        return None
    return (
        dispersion / molar_volume_cm3,
        math.sqrt(polar_squared) / molar_volume_cm3,
        math.sqrt(bonding / molar_volume_cm3),
    )


class GroupContributionHansenExpert(Expert):
    """A Hansen triple by Hoftyzer-Van Krevelen, where no compilation has one.

    The compilation covers ordinary solvents and stops there. Every component
    of a designed formulation tends to fall outside it - a specialty monomer,
    an initiator, an accelerator - and the mixture layer needs all three
    components of every one of them, so a single gap takes out the whole
    blend. Asked for the four components of a cured jet, the lookup expert
    reported four of four missing and the mixture expert then had nothing to
    average.

    The estimate is worth having only because it declines. It refuses on any
    atom no group covers, on a polyhalogenated carbon, and on a missing molar
    volume, because each of those failure modes produces a plausible number
    rather than an obvious one.
    """

    id = "hansen_group_contribution"
    version = "1"
    method = (
        "Hoftyzer-Van Krevelen group contribution (Van Krevelen, Properties of "
        "Polymers, 4th ed., table 7.9)"
    )
    family = PropertyFamily.INTERFACIAL
    supported_properties = frozenset(
        {"hansen_dispersion", "hansen_polar", "hansen_hydrogen_bonding"}
    )
    supported_classes = frozenset({MaterialClass.MOLECULE})
    #: The triple is an energy density, so every component divides by the
    #: molar volume; without one there is no estimate at all.
    dependencies = frozenset({"liquid_density"})

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        smiles = candidate.primary_smiles
        if smiles is None:
            return ApplicabilityDomain(in_domain=False, score=0.0, basis="no structure")
        return ApplicabilityDomain(
            in_domain=True,
            score=1.0,
            basis=(
                "measured against the compilation over the forty reference structures "
                "this method accepts: mean absolute error 0.89, 0.80 and 1.18 MPa^0.5 "
                "on the dispersion, polar and hydrogen bonding components"
            ),
        )

    def _predict_one(self, prop, request, domain) -> Prediction | None:
        smiles = request.candidate.primary_smiles
        if smiles is None:
            return Prediction.unsupported(prop, self.id, "the candidate carries no structure")

        density = request.dependency_value("liquid_density", "kg/m^3")
        if density is None or density <= 0.0:
            return Prediction.unsupported(
                prop,
                self.id,
                "a Hansen component is an energy per unit volume and no liquid density "
                "reached this expert, so there is no volume to divide by",
            )
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return Prediction.unsupported(prop, self.id, "the structure could not be parsed")
        molar_volume = float(Descriptors.MolWt(molecule)) / (density / 1000.0)

        triple = hoftyzer_van_krevelen(smiles, molar_volume)
        if triple is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "this structure contains a group the table does not cover, or a carbon "
                "carrying more than one halogen. A partial sum silently drops a "
                "contribution and returns an ordinary looking number, so it is refused",
            )
        index = {
            "hansen_dispersion": 0,
            "hansen_polar": 1,
            "hansen_hydrogen_bonding": 2,
        }[prop]
        value = triple[index]

        # The group method's own spread, plus whatever the molar volume brings:
        # every component goes as one over V, so the density's relative error
        # passes straight through.
        incumbent = request.dependency("liquid_density")
        relative_volume = 0.0
        if incumbent is not None and incumbent.uncertainty.std and incumbent.quantity:
            relative_volume = abs(incumbent.uncertainty.std / incumbent.quantity.value)
        std = math.hypot(HVK_SIGMA[prop], value * relative_volume)

        return self._make(
            prop,
            value,
            "MPa^0.5",
            request,
            domain,
            std=std,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                f"1.253 times the mean absolute error of this group method against the "
                f"compilation ({HVK_SIGMA[prop]:.2f} MPa^0.5), with the molar volume's "
                f"own {relative_volume * 100:.0f} per cent carried through"
            ),
            notes=(
                f"molar volume {molar_volume:.1f} cm^3/mol, from the liquid density this "
                "expert was given rather than from a group estimate of its own",
                "estimated, not tabulated: the compilation has no entry for this structure",
            ),
            molar_volume_cm3=round(molar_volume, 3),
        )
