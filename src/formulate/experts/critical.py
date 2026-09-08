"""Critical constants from atoms rather than from groups.

Joback is the panel's usual source for critical constants and it is the better
estimate for two of the three - but it types molecules by matching a table of
41 functional groups, and when nothing matches it returns nothing at all. A
discovery run asked for a high-boiling low-freezing liquid generated
thirty-five cyclic carbonates and lactones, and Joback could type almost none
of them, which left them without critical constants and therefore without a
density or a surface tension, since both are derived from those.

Wilson-Jasperson counts atoms and rings instead. Every organic structure has
atoms, so it answers where a group table cannot, and it needs only a boiling
point - which the learned expert now supplies for exactly the molecules Joback
refuses. Fedors does the same for critical volume.

Measured over the fifty reference compounds, using their measured boiling
points so that this tests these methods rather than whatever fed them:

    critical temperature   Wilson-Jasperson 19.5 K    Joback 27.8 K
    critical pressure      Wilson-Jasperson 5.5e5 Pa  Joback 3.1e5 Pa
    critical volume        Fedors 1.25e-5 m^3/mol     Joback 8.4e-6 m^3/mol

So this expert is better on one of the three and worse on two, and none of that
needs a precedence rule. The spreads below are the measured errors, and
:func:`~formulate.evaluation.engine.prefer` selects the tighter one per
property: it takes Wilson-Jasperson's critical temperature and Joback's
pressure and volume, on the numbers alone. Where Joback declines, these are the
only answers and they win by default.
"""

from __future__ import annotations

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest

#: One-sigma spreads, as 1.253 times the mean absolute error measured over the
#: reference set - the conversion for a normal distribution. Quoting the
#: original papers' figures would repeat the mistake Joback's critical
#: temperature made: a fitting-set number that does not survive this chemistry.
_SPREAD = {
    "critical_temperature": (
        24.4,
        "1.253 times the 19.5 K mean absolute error measured over the fifty reference "
        "compounds, which is better than Joback manages on the same set",
    ),
    "critical_pressure": (
        6.92e5,
        "1.253 times the 5.5 bar mean absolute error measured over the fifty reference "
        "compounds; Joback is the better estimate here and the panel prefers it on "
        "this number alone",
    ),
    "critical_volume": (
        1.56e-5,
        "1.253 times the mean absolute error of the Fedors group method over the fifty "
        "reference compounds; Joback is the better estimate where it can type the "
        "molecule",
    ),
}


class AtomicCriticalExpert(Expert):
    """Critical constants for structures a group table cannot match."""

    id = "critical_atomic"
    version = "1"
    method = "Wilson-Jasperson atomic contributions, and Fedors for critical volume"
    family = PropertyFamily.THERMAL
    supported_classes = frozenset({MaterialClass.MOLECULE})
    supported_properties = frozenset(_SPREAD)
    #: Wilson-Jasperson takes the boiling point as an input. Declaring it lets
    #: the registry run the boiling-point experts first, so a measurement is
    #: used where one exists and the learned estimate where none does.
    dependencies = frozenset({"normal_boiling_point"})

    def is_available(self) -> bool:
        from formulate import chem

        try:
            from thermo import Fedors, Wilson_Jasperson  # noqa: F401
        except Exception:
            return False
        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        from formulate import chem

        if not chem.rdkit_available():
            return "RDKit is required to count atoms and rings"
        try:
            from thermo import Fedors, Wilson_Jasperson  # noqa: F401
        except Exception as exc:
            return f"thermo is not installed ({exc})"
        return ""

    def _software(self) -> SoftwareEnvironment:
        import thermo

        return SoftwareEnvironment.capture(thermo=thermo.__version__)

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        if candidate.molecule is None:
            return ApplicabilityDomain.outside("candidate carries no molecule")
        from formulate import chem

        if chem.mol_from_smiles(candidate.molecule.smiles) is None:
            return ApplicabilityDomain.outside("this structure could not be parsed")
        return ApplicabilityDomain(
            basis="atomic contributions, which every organic structure has, rather than "
            "a table of functional groups that a novel structure may not match"
        )

    def _hydrogenated(self, smiles: str):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        return None if mol is None else Chem.AddHs(mol)

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        smiles = request.candidate.molecule.smiles  # type: ignore[union-attr]
        mol = self._hydrogenated(smiles)
        if mol is None:
            return Prediction.failed(prop, self.id, "this structure could not be parsed")

        std, basis = _SPREAD[prop]
        if prop == "critical_volume":
            from thermo import Fedors

            try:
                value, status, *_ = Fedors(mol)
            except Exception as exc:
                return Prediction.failed(prop, self.id, f"Fedors failed: {exc}")
            if status != "OK" or value is None:
                return Prediction.unsupported(
                    prop, self.id, f"Fedors has no increment for part of this structure: {status}"
                )
            return self._make(prop, float(value), "m^3/mol", request, domain,
                              std=std, kind=UncertaintyKind.EPISTEMIC, basis=basis)

        boiling = request.dependency("normal_boiling_point")
        if boiling is None or boiling.quantity is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "Wilson-Jasperson takes the normal boiling point as an input and no "
                "expert upstream supplied one",
            )
        from thermo import Wilson_Jasperson

        try:
            tc, pc, tc_missing, pc_missing = Wilson_Jasperson(
                mol, Tb=boiling.quantity.to("K").value
            )
        except Exception as exc:
            return Prediction.failed(prop, self.id, f"Wilson-Jasperson failed: {exc}")

        value = tc if prop == "critical_temperature" else pc
        missing = tc_missing if prop == "critical_temperature" else pc_missing
        if value is None or missing:
            return Prediction.unsupported(
                prop,
                self.id,
                "Wilson-Jasperson has no atomic increment for part of this structure",
            )
        unit = "K" if prop == "critical_temperature" else "Pa"
        return self._make(
            prop, float(value), unit, request, domain,
            std=std, kind=UncertaintyKind.EPISTEMIC, basis=basis,
            notes=(
                f"built on a boiling point from {boiling.expert_id}, so its error carries "
                "into this one",
            ),
        )
