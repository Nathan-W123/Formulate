"""Measured pure-component properties.

Specification section 4's model policy asks for open sources to be reused
where they are scientifically adequate rather than for new predictors to be
fitted.  A compiled measurement is the strongest case of that: where a value
has been measured, estimating it instead is a choice to be less accurate.

This expert therefore sits alongside the group-contribution and correlation
experts rather than replacing them.  Where a compound has been measured it
supplies the measurement with a small uncertainty; where it has not, it
supplies nothing at all, and the estimating experts answer.  The evaluation
engine already prefers the in-domain prediction with the tighter spread, so
no precedence logic is needed here - the measurement wins because it is
better, not because it is privileged.
"""

from __future__ import annotations

import functools

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest
from .hansen import resolve_cas

try:  # pragma: no cover
    from chemicals import MW, Tb, Tm

    _HAVE_CHEMICALS = True
    _IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    _HAVE_CHEMICALS = False
    _IMPORT_ERROR = str(exc)


#: Property -> (lookup, unit, one-sigma spread, what the spread represents).
#:
#: The spreads are not measurement precision, which would be far smaller. They
#: are the disagreement between compilations for the same substance, which
#: comes from sample purity, pressure correction and which original source was
#: preferred. Quoting instrument precision here would be the more precise lie.
_LOOKUPS = {
    "normal_boiling_point": (
        "Tb",
        "K",
        0.5,
        "spread between compilations for a measured boiling point, driven by purity and "
        "the pressure correction to one atmosphere rather than by instrument precision",
    ),
    "melting_point": (
        "Tm",
        "K",
        1.0,
        "spread between compilations for a measured melting point; polymorphism and "
        "purity move it more than the measurement does",
    ),
    "molar_mass": (
        "MW",
        "g/mol",
        0.0,
        "computed exactly from the standard atomic weights",
    ),
}


@functools.lru_cache(maxsize=4096)
def measured_value(prop: str, smiles: str) -> float | None:
    """Look up a measured value for a structure, or None if it is not tabulated."""
    if not _HAVE_CHEMICALS:
        return None
    cas = resolve_cas(smiles)
    if cas is None:
        return None
    name = _LOOKUPS[prop][0]
    try:
        value = {"Tb": Tb, "Tm": Tm, "MW": MW}[name](cas)
    except Exception:
        return None
    return None if value is None else float(value)


class MeasuredPropertyExpert(Expert):
    """Supplies compiled experimental values where they exist."""

    id = "measured"
    version = "1"
    method = "compiled experimental pure-component data (via the chemicals package)"
    family = PropertyFamily.THERMAL
    supported_classes = frozenset({MaterialClass.MOLECULE})
    supported_properties = frozenset(_LOOKUPS)

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

        return SoftwareEnvironment.capture(chemicals=chemicals.__version__)

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        smiles = candidate.molecule.smiles if candidate.molecule else None
        if smiles is None:
            return ApplicabilityDomain.outside("candidate carries no molecule")
        if resolve_cas(smiles) is None:
            return ApplicabilityDomain.outside(
                "this structure could not be resolved to a compiled substance",
                basis="substances present in the experimental compilation",
            )
        return ApplicabilityDomain(
            basis="a measured value for this exact substance, not an estimate"
        )

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        smiles = request.candidate.molecule.smiles  # type: ignore[union-attr]
        value = measured_value(prop, smiles)
        if value is None:
            # Silence, not a guess. The estimating experts cover this compound,
            # and a fabricated "measurement" would outrank them on uncertainty.
            return Prediction.unsupported(
                prop,
                self.id,
                "this substance has no compiled measurement for this property",
            )

        _, unit, spread, basis = _LOOKUPS[prop]
        return self._make(
            prop,
            value,
            unit,
            request,
            domain,
            std=spread,
            kind=UncertaintyKind.ALEATORIC,
            basis=basis,
            notes=(
                "measured, not estimated",
                f"CAS {resolve_cas(smiles)}",
            ),
        )
