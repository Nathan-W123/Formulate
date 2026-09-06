"""Synthesis feasibility and structural alerts.

Specification section 4 lists a feasibility family covering "synthesis /
manufacturing heuristics, data-domain detection, and optional cost /
environmental constraints".

Section 13 is explicit that a high score here is not proof of synthesisability.
The synthetic accessibility score is a fragment-frequency heuristic: it says
whether a structure looks like the things chemists have made before, which is
a prior, not a route.  Every prediction from this expert carries that caveat.
"""

from __future__ import annotations

import functools
import os
from typing import Any

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest

#: Elements common enough in organic synthesis that their presence is unremarkable.
_COMMON_ELEMENTS = frozenset({"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si"})


@functools.lru_cache(maxsize=1)
def _sascorer() -> Any | None:
    """Load RDKit's contributed synthetic accessibility scorer, if present."""
    try:
        import sys

        from rdkit.Chem import RDConfig

        path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if path not in sys.path:
            sys.path.append(path)
        import sascorer  # type: ignore[import-not-found]

        return sascorer
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _pains_catalog() -> Any | None:
    """PAINS structural-alert catalogue, if RDKit provides one."""
    try:
        from rdkit.Chem import FilterCatalog

        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        return FilterCatalog.FilterCatalog(params)
    except Exception:
        return None


class SynthesisFeasibilityExpert(Expert):
    """Scores how much a structure resembles previously synthesised chemistry."""

    id = "feasibility"
    version = "1"
    method = (
        "Ertl-Schuffenhauer synthetic accessibility score "
        "(J. Cheminform. 1:8, 2009) with structural alerts"
    )
    family = PropertyFamily.FEASIBILITY
    supported_classes = frozenset({MaterialClass.MOLECULE})
    supported_properties = frozenset({"synthetic_accessibility"})

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available() and _sascorer() is not None

    def unavailable_reason(self) -> str:
        from formulate import chem

        if not chem.rdkit_available():
            return "RDKit is not installed"
        if _sascorer() is None:
            return "RDKit's contributed SA_Score module could not be imported"
        return ""

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

        exotic = chem.elements(smiles) - _COMMON_ELEMENTS
        if exotic:
            warnings.append(
                f"contains {', '.join(sorted(exotic))}, which the fragment frequencies "
                "underlying this score barely cover"
            )
            score = min(score, 0.4)

        return ApplicabilityDomain(
            score=score,
            in_domain=score > 0.3,
            warnings=tuple(warnings),
            basis="organic structures resembling the PubChem fragment statistics the score was built on",
        )

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        from formulate import chem

        scorer = _sascorer()
        if scorer is None:
            return Prediction.unsupported(prop, self.id, self.unavailable_reason())

        smiles = request.candidate.molecule.smiles  # type: ignore[union-attr]
        mol = chem.mol_from_smiles(smiles)
        if mol is None:
            return Prediction.failed(prop, self.id, f"SMILES {smiles!r} could not be parsed")

        value = float(scorer.calculateScore(mol))

        notes = [
            "a fragment-frequency heuristic on a 1 (easy) to 10 (hard) scale; it is a prior "
            "over what has been made before, not a synthetic route, and a good score is not "
            "evidence that this compound can be made",
        ]
        notes.extend(self._alerts(mol))

        return self._make(
            prop,
            value,
            "",
            request,
            domain,
            std=1.0,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                "the score is a heuristic with no calibrated error; 1.0 is recorded as a "
                "nominal spread so that ranking cannot treat it as exact"
            ),
            notes=tuple(notes),
        )

    def _alerts(self, mol: Any) -> list[str]:
        """Structural alerts worth surfacing alongside the score."""
        catalog = _pains_catalog()
        if catalog is None:
            return []
        try:
            entry = catalog.GetFirstMatch(mol)
        except Exception:
            return []
        if entry is None:
            return []
        return [
            f"matches the PAINS structural alert {entry.GetDescription()!r}; such substructures "
            "interfere with many assays and are often excluded from screening libraries"
        ]
