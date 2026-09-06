"""Lipophilicity and refractivity: Wildman-Crippen atomic contributions.

Method: Wildman, S. A. and Crippen, G. M., "Prediction of Physicochemical
Parameters by Atomic Contributions", J. Chem. Inf. Comput. Sci. 39 (1999)
868-873.  Evaluated through RDKit's implementation.
"""

from __future__ import annotations

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest

_SPEC = {
    "logp": (
        "crippen_logp",
        "",
        1.0,
        "Wildman & Crippen (1999) report about 0.68 log units on their own 9920-compound "
        "set; roughly 1.0 log unit is the accepted spread on external sets, which is the "
        "value used here",
    ),
    "molar_refractivity": (
        "crippen_mr",
        "cm^3/mol",
        2.5,
        "atomic-contribution molar refractivity; approximately 2.5 cm^3/mol",
    ),
}


class CrippenLipophilicityExpert(Expert):
    """Octanol/water partition coefficient and molar refractivity."""

    id = "crippen"
    version = "1"
    method = "Wildman-Crippen atomic contributions (J. Chem. Inf. Comput. Sci. 39:868, 1999)"
    family = PropertyFamily.CHEMICAL
    supported_classes = frozenset({MaterialClass.MOLECULE})
    supported_properties = frozenset(_SPEC)

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "RDKit is not installed"

    def _software(self) -> SoftwareEnvironment:
        from formulate import chem

        return SoftwareEnvironment.capture(rdkit=chem.rdkit_version())

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        from formulate import chem

        smiles = candidate.molecule.smiles if candidate.molecule else None
        if smiles is None or chem.mol_from_smiles(smiles) is None:
            return ApplicabilityDomain.outside("candidate carries no parseable molecule")

        warnings: list[str] = []
        score = 1.0
        descriptors = chem.descriptors(smiles)

        if descriptors.get("formal_charge", 0.0):
            warnings.append(
                "the atomic contributions were fitted to neutral species; for an ion the "
                "octanol/water partition coefficient is pH-dependent and this value is a "
                "neutral-form estimate"
            )
            score = min(score, 0.25)

        if descriptors.get("molar_mass", 0.0) > 800:
            warnings.append("molar mass is above the range the parameters were fitted over")
            score = min(score, 0.5)

        return ApplicabilityDomain(
            score=score,
            in_domain=score > 0.3,
            warnings=tuple(warnings),
            basis="neutral drug-like and small organic molecules",
        )

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        from formulate import chem

        smiles = request.candidate.molecule.smiles  # type: ignore[union-attr]
        key, unit, std, basis = _SPEC[prop]
        descriptors = chem.descriptors(smiles)
        if key not in descriptors:
            return Prediction.failed(prop, self.id, f"RDKit produced no {prop}")
        return self._make(
            prop,
            descriptors[key],
            unit,
            request,
            domain,
            std=std,
            kind=UncertaintyKind.EPISTEMIC,
            basis=basis,
        )
