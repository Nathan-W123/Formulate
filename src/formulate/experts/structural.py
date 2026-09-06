"""Structural descriptors.

Not a property predictor in the modelling sense: these are computed exactly
from the graph.  They are exposed through the expert interface so that a
target can constrain molar mass or ring count in the same language as it
constrains a boiling point, and so the provenance chain stays uniform.

Because the values are exact, their uncertainty is zero - one of the few
places in this system where that is an honest statement.
"""

from __future__ import annotations

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest

_UNITS = {
    "molar_mass": "g/mol",
    "heavy_atom_count": "",
    "rotatable_bond_count": "",
    "topological_polar_surface_area": "angstrom^2",
    "aromatic_atom_fraction": "",
}


class StructuralDescriptorExpert(Expert):
    """Exact graph descriptors computed with RDKit."""

    id = "structural"
    version = "1"
    method = "RDKit graph descriptors (exact, not estimated)"
    family = PropertyFamily.STRUCTURAL
    supported_classes = frozenset({MaterialClass.MOLECULE, MaterialClass.MIXTURE})
    supported_properties = frozenset(_UNITS)

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

        for smiles in candidate.all_smiles():
            if chem.mol_from_smiles(smiles) is None:
                return ApplicabilityDomain.outside(f"SMILES {smiles!r} could not be parsed")
        return ApplicabilityDomain(basis="exact graph computation; valid for any parseable structure")

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        from formulate import chem

        candidate = request.candidate
        notes: tuple[str, ...] = ()

        if candidate.material_class is MaterialClass.MIXTURE:
            value = self._mixture_value(prop, candidate)
            if value is None:
                return Prediction.unsupported(
                    prop, self.id, f"{prop} is not defined as a mixture average"
                )
            notes = (
                "composition-weighted average over mixture components on the "
                f"{candidate.mixture.basis.value} basis",
            )
        else:
            smiles = candidate.molecule.smiles  # type: ignore[union-attr]
            descriptors = chem.descriptors(smiles)
            if prop not in descriptors:
                return Prediction.failed(prop, self.id, f"RDKit produced no {prop}")
            value = descriptors[prop]

        return self._make(
            prop,
            value,
            _UNITS[prop],
            request,
            domain,
            std=0.0,
            kind=UncertaintyKind.EPISTEMIC,
            basis="computed exactly from the molecular graph",
            notes=notes,
        )

    def _mixture_value(self, prop: str, candidate: Candidate) -> float | None:
        """Composition-weighted descriptor across a formulation.

        Only meaningful for extensive-per-mole descriptors; a weighted mean of
        an aromatic fraction is a summary, not a property of the blend, so the
        note attached to the prediction says so.
        """
        from formulate import chem

        mixture = candidate.mixture
        if mixture is None:
            return None
        total = 0.0
        for component in mixture.components:
            smis = component.all_smiles()
            if not smis:
                return None
            descriptors = chem.descriptors(smis[0])
            if prop not in descriptors:
                return None
            total += component.fraction * descriptors[prop]
        return total
