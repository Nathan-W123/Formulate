"""Aqueous solubility: the ESOL correlation.

Method: Delaney, J. S., "ESOL: Estimating Aqueous Solubility Directly from
Molecular Structure", J. Chem. Inf. Comput. Sci. 44 (2004) 1000-1005.

    logS = 0.16 - 0.63 clogP - 0.0062 MW + 0.066 RB - 0.74 AP

with S in mol/L at 298 K, clogP the Crippen partition coefficient, MW the
molar mass, RB the rotatable-bond count and AP the aromatic proportion.
"""

from __future__ import annotations

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.conditions import Conditions, Phase
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, Quantity, UncertaintyKind

from .base import Expert, PredictionRequest

#: Published coefficients, in the order (intercept, clogP, MW, RB, AP).
_COEFFICIENTS = (0.16, -0.63, -0.0062, 0.066, -0.74)

#: Delaney reports an average absolute error near 0.75 log units.
_STD = 0.75

_REFERENCE_TEMPERATURE_K = 298.0


class ESOLSolubilityExpert(Expert):
    """Aqueous solubility of a neutral organic solid or liquid at 298 K."""

    id = "esol"
    version = "1"
    method = "ESOL correlation (Delaney, J. Chem. Inf. Comput. Sci. 44:1000, 2004)"
    family = PropertyFamily.CHEMICAL
    supported_classes = frozenset({MaterialClass.MOLECULE})
    supported_properties = frozenset({"aqueous_solubility_logs"})

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

        descriptors = chem.descriptors(smiles)
        warnings: list[str] = []
        score = 1.0

        if descriptors.get("formal_charge", 0.0):
            warnings.append(
                "ESOL was fitted to neutral species; solubility of an ion is pH-dependent"
            )
            score = min(score, 0.2)

        mass = descriptors.get("molar_mass", 0.0)
        if mass > 700:
            warnings.append("molar mass is above the range ESOL was fitted over")
            score = min(score, 0.4)

        # Small, highly polar molecules are often water-miscible, for which a
        # finite solubility is not a meaningful quantity.
        if mass < 100 and descriptors.get("crippen_logp", 0.0) < 0.0:
            warnings.append(
                "small and hydrophilic: this compound may be fully water-miscible, in which "
                "case a finite log solubility is not physically meaningful and ESOL is known "
                "to underpredict"
            )
            score = min(score, 0.45)

        # ESOL was fitted largely on drug-like compounds. Two related groups sit
        # outside that set and have their insolubility systematically
        # underpredicted, by several log units in the worst cases.
        logp = descriptors.get("crippen_logp", 0.0)
        if logp > 4.0:
            warnings.append(
                f"clogP of {logp:.1f} is more hydrophobic than the drug-like set ESOL was "
                "fitted on; the correlation underpredicts how insoluble such compounds are"
            )
            score = min(score, 0.5 if logp < 6.0 else 0.15)

        # Saturated hydrocarbons are the clearest failure mode. ESOL's only
        # hydrophobicity term is clogP, and its aromatic-proportion term does
        # not apply to them, so nothing in the correlation captures the
        # hydrophobic effect of an aliphatic chain. Errors reach 2-4 log units
        # (octane: -2.3 predicted vs -5.2 experimental).
        if (
            chem.elements(smiles) <= {"C", "H"}
            and descriptors.get("aromatic_atom_fraction", 0.0) == 0.0
            and descriptors.get("heavy_atom_count", 0.0) >= 5
        ):
            warnings.append(
                "saturated hydrocarbon: ESOL carries no descriptor that captures the "
                "hydrophobic effect of an aliphatic chain and underpredicts insolubility "
                "by 2-4 log units for this class"
            )
            score = min(score, 0.4 if logp <= 5.0 else 0.15)

        return ApplicabilityDomain(
            score=score,
            in_domain=score > 0.3,
            warnings=tuple(warnings),
            basis="neutral small organic molecules with measurable, non-miscible solubility",
        )

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        from formulate import chem

        smiles = request.candidate.molecule.smiles  # type: ignore[union-attr]
        d = chem.descriptors(smiles)
        if not d:
            return Prediction.failed(prop, self.id, "RDKit produced no descriptors")

        intercept, c_logp, c_mw, c_rb, c_ap = _COEFFICIENTS
        value = (
            intercept
            + c_logp * d["crippen_logp"]
            + c_mw * d["molar_mass"]
            + c_rb * d["rotatable_bond_count"]
            + c_ap * d["aromatic_atom_fraction"]
        )

        # ESOL is defined in water at 298 K regardless of the request's conditions.
        conditions = Conditions(
            temperature=Quantity(value=_REFERENCE_TEMPERATURE_K, unit="K"),
            environment="water",
            phase=Phase.SOLUTION,
        )
        notes = ("log10 of solubility in mol/L, in water at 298 K",)
        requested_t = request.conditions.temperature_k
        if requested_t is not None and abs(requested_t - _REFERENCE_TEMPERATURE_K) > 5.0:
            notes += (
                f"the request asks for {requested_t:.1f} K but ESOL carries no temperature "
                "dependence; this value remains a 298 K estimate",
            )

        return self._make(
            prop,
            value,
            "",
            request,
            domain,
            std=_STD,
            kind=UncertaintyKind.EPISTEMIC,
            basis="average absolute error near 0.75 log units reported by Delaney (2004)",
            conditions=conditions,
            notes=notes,
        )
