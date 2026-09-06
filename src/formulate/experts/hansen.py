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
