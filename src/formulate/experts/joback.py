"""Thermal and phase-behaviour expert: Joback group contribution.

Method: Joback, K. G. and Reid, R. C., "Estimation of Pure-Component
Properties from Group-Contributions", Chem. Eng. Commun. 57 (1987) 233-243.

Reuses the validated implementation in the open ``thermo`` package rather than
re-entering the group table, per the model policy of specification section 4.
This module contributes the applicability domain, the uncertainty statement and
the canonical-unit handling that ``thermo`` does not provide.
"""

from __future__ import annotations

import functools
from typing import Any

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.conditions import Conditions
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, Quantity, UncertaintyKind

from .base import Expert, PredictionRequest

try:  # pragma: no cover
    from thermo.group_contribution import Joback as _Joback

    _HAVE_THERMO = True
    _THERMO_ERROR = ""
except Exception as exc:  # pragma: no cover
    _HAVE_THERMO = False
    _THERMO_ERROR = str(exc)

#: Elements the Joback group table covers.
_SUPPORTED_ELEMENTS = frozenset({"C", "H", "N", "O", "S", "F", "Cl", "Br", "I"})

#: Heavy-atom count beyond which group contribution is extrapolating well past
#: the compounds Joback was fitted to.
_LARGE_MOLECULE_THRESHOLD = 30

#: One-sigma epistemic uncertainties, in each property's canonical unit.
#:
#: The values are the average absolute errors Joback and Reid report for their
#: own fitting set.  They are therefore optimistic for novel chemistry, which
#: the ``basis`` string records; a held-out molecule should be expected to do
#: worse.  Melting point is the exception and is inflated deliberately - see
#: the note below.
_UNCERTAINTY: dict[str, tuple[float, str]] = {
    "normal_boiling_point": (
        12.9,
        "average absolute error reported by Joback & Reid (1987) on their fitting set; "
        "optimistic for chemistry outside it",
    ),
    "melting_point": (
        40.0,
        "Joback & Reid (1987) report 22.6 K on their fitting set, but melting point is "
        "governed by crystal packing that group contribution cannot represent and errors "
        "above 100 K occur for symmetric aromatics (benzene: 171 K predicted vs 278.7 K "
        "experimental). Inflated to 40 K as a practical estimate; treat any melting-point "
        "ranking from this expert as indicative only",
    ),
    "critical_temperature": (
        4.76,
        "average absolute error reported by Joback & Reid (1987); assumes an accurate "
        "boiling point, and here the boiling point is itself estimated",
    ),
    "critical_pressure": (
        2.06e5,
        "average absolute error reported by Joback & Reid (1987), 2.06 bar",
    ),
    "critical_volume": (
        7.54e-6,
        "average absolute error reported by Joback & Reid (1987), 7.54 cm^3/mol",
    ),
    "enthalpy_vaporization": (
        1.09e3,
        "average absolute error reported by Joback & Reid (1987), 1.09 kJ/mol, "
        "at the normal boiling point",
    ),
    "enthalpy_fusion": (
        2.2e3,
        "average absolute error reported by Joback & Reid (1987), 2.2 kJ/mol",
    ),
    "heat_capacity_gas": (
        4.0,
        "ideal-gas heat capacity polynomial; Joback reports a few percent over the "
        "fitted temperature range, taken here as 4 J/mol/K",
    ),
}

#: Conditions each property is defined at, beyond the request's own.
_AT_BOILING_POINT = frozenset({"enthalpy_vaporization"})

#: Measured one-sigma errors over the fifty reference compounds, replacing the
#: published figures where those proved optimistic, in each canonical unit.
#:
#: Joback and Reid quote average absolute errors on their own fitting set, and
#: for three properties those do not survive contact with this compound mix:
#: a stated 4.76 K on the critical temperature caught 17 per cent of the
#: reference compounds inside one sigma where a correct estimate catches about
#: 68. The values below are 1.253 times the measured mean absolute error, which
#: is the conversion for a normal distribution.
#:
#: The split is by hydrogen-bond donor count, following the interfacial
#: expert, because that is where the failures concentrate and the effect is
#: large: on the enthalpy of vaporisation the associating compounds err by 4.0
#: kJ/mol against 1.7 for the rest, and acetic acid - which dimerises in the
#: vapour, so no monomeric group method can be right about it - is out by 13.3.
#: Critical volume, enthalpy of fusion and gas heat capacity show no such
#: pattern and take a single figure; inventing a factor for them would be
#: decoration.
#:
#: These spreads are calibrated on the reference set, so coverage measured over
#: that same set is not independent evidence that they are right. They replace
#: figures fitted to somebody else's set, which had the same problem and was
#: additionally about the wrong chemistry.
_MEASURED_SPREAD: dict[str, tuple[float, float]] = {
    # property: (no hydrogen-bond donor, one or more)
    "critical_temperature": (29.0, 54.0),
    "critical_pressure": (2.28e5, 9.12e5),
    "enthalpy_vaporization": (2170.0, 4960.0),
    "critical_volume": (1.05e-5, 1.05e-5),
    "enthalpy_fusion": (2620.0, 2620.0),
    "heat_capacity_gas": (2.45, 2.45),
}

_MEASURED_BASIS = (
    "1.253 times the mean absolute error measured over the fifty reference compounds, "
    "split by hydrogen-bond donor count; replaces the figure Joback and Reid report for "
    "their own fitting set, which is optimistic for this chemistry"
)


@functools.lru_cache(maxsize=4096)
def _joback_for(smiles: str) -> Any:
    """Construct and cache a Joback estimator for a SMILES string."""
    return _Joback(smiles)


class JobackThermalExpert(Expert):
    """Estimates thermal and phase properties from molecular groups."""

    id = "joback"
    version = "1"
    method = "Joback group contribution (Joback & Reid, Chem. Eng. Commun. 57:233, 1987)"
    family = PropertyFamily.THERMAL
    supported_classes = frozenset({MaterialClass.MOLECULE})
    supported_properties = frozenset(_UNCERTAINTY)
    #: Joback's critical-property correlations take the boiling point as an
    #: input, and Joback & Reid state that a measured one should be used when
    #: available. Declaring the dependency makes the registry run a lookup
    #: expert first, so the critical constants are built on the measurement
    #: rather than on this method's own estimate of it.
    dependencies = frozenset({"normal_boiling_point"})

    def is_available(self) -> bool:
        from formulate import chem

        return _HAVE_THERMO and chem.rdkit_available()

    def unavailable_reason(self) -> str:
        from formulate import chem

        if not _HAVE_THERMO:
            return f"the 'thermo' package is not installed ({_THERMO_ERROR})"
        if not chem.rdkit_available():
            return "RDKit is not installed"
        return ""

    def _software(self) -> SoftwareEnvironment:
        import thermo

        from formulate import chem

        return SoftwareEnvironment.capture(thermo=thermo.__version__, rdkit=chem.rdkit_version())

    # -- domain ------------------------------------------------------------

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        from formulate import chem

        smiles = candidate.molecule.smiles if candidate.molecule else None
        if smiles is None:
            return ApplicabilityDomain.outside("candidate carries no molecule")

        mol = chem.mol_from_smiles(smiles)
        if mol is None:
            return ApplicabilityDomain.outside(f"SMILES {smiles!r} could not be parsed")

        warnings: list[str] = []
        score = 1.0

        present = chem.elements(smiles)
        unsupported = present - _SUPPORTED_ELEMENTS
        if unsupported:
            return ApplicabilityDomain.outside(
                f"elements outside the Joback group table: {', '.join(sorted(unsupported))}",
                basis="Joback covers C, H, N, O, S and the halogens only",
            )

        if candidate.molecule.charge != 0 or chem.descriptors(smiles).get("formal_charge", 0.0):
            warnings.append("Joback was parameterised on neutral species; this one is charged")
            score = min(score, 0.2)

        heavy = mol.GetNumHeavyAtoms()
        if heavy > _LARGE_MOLECULE_THRESHOLD:
            warnings.append(
                f"{heavy} heavy atoms is well beyond the small molecules Joback was fitted to; "
                "group-contribution error grows with size"
            )
            score = min(score, 0.4)
        elif heavy > 20:
            score = min(score, 0.75)

        if _HAVE_THERMO:
            try:
                est = _joback_for(smiles)
                if not est.success:
                    return ApplicabilityDomain.outside(
                        f"Joback group assignment failed: {est.status}",
                        basis="every atom must map to a Joback group",
                    )
            except Exception as exc:
                return ApplicabilityDomain.outside(f"Joback group assignment raised: {exc}")

        return ApplicabilityDomain(
            score=score,
            in_domain=not warnings or score > 0.3,
            warnings=tuple(warnings),
            basis="neutral small organic molecules composed of tabulated Joback groups",
        )

    # -- prediction --------------------------------------------------------

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        candidate = request.candidate
        smiles = candidate.molecule.smiles if candidate.molecule else None
        if smiles is None:
            return Prediction.failed(prop, self.id, "candidate carries no molecule")

        est = _joback_for(smiles)
        if not est.success:
            return Prediction.failed(prop, self.id, f"Joback group assignment failed: {est.status}")

        counts = est.counts
        value, unit, conditions, notes = self._compute(prop, est, counts, request)
        if value is None:
            return Prediction.failed(prop, self.id, f"Joback produced no value for {prop}")

        std, basis = _UNCERTAINTY[prop]
        measured = _MEASURED_SPREAD.get(prop)
        if measured is not None:
            from formulate import chem

            associating = int(chem.descriptors(smiles).get("hbd", 0.0)) > 0
            std = measured[1 if associating else 0]
            basis = _MEASURED_BASIS
        return self._make(
            prop,
            value,
            unit,
            request,
            domain,
            std=std,
            kind=UncertaintyKind.EPISTEMIC,
            basis=basis,
            conditions=conditions,
            notes=notes,
            groups=str(dict(sorted(counts.items()))),
        )

    def _external_boiling_point(self, request: PredictionRequest) -> float | None:
        """A boiling point from someone other than this expert, in Kelvin.

        Only an outside value is used. Passing this expert's own estimate back
        into its critical-property correlation would change nothing, and taking
        it from the context indiscriminately would risk feeding the correlation
        its own output.
        """
        prediction = request.dependency("normal_boiling_point")
        if prediction is None or prediction.quantity is None:
            return None
        if prediction.expert_id == self.id:
            return None
        return prediction.quantity.to("K").value

    def _compute(
        self, prop: str, est: Any, counts: dict[int, int], request: PredictionRequest
    ) -> tuple[float | None, str, Conditions | None, tuple[str, ...]]:
        """Return ``(value, unit, conditions, notes)`` for one property."""
        if prop == "normal_boiling_point":
            return est.Tb(counts), "K", None, ()
        if prop == "melting_point":
            return (
                est.Tm(counts),
                "K",
                None,
                ("Joback melting points are the least reliable output of the method",),
            )
        if prop == "critical_temperature":
            measured_tb = self._external_boiling_point(request)
            if measured_tb is not None:
                return (
                    est.Tc(counts, measured_tb),
                    "K",
                    None,
                    (
                        "computed from a measured boiling point, which is how Joback "
                        "and Reid intend the correlation to be used",
                    ),
                )
            return (
                est.Tc(counts),
                "K",
                None,
                ("computed from the Joback-estimated boiling point, so their errors compound",),
            )
        if prop == "critical_pressure":
            return est.Pc(counts, est.atom_count), "Pa", None, ()
        if prop == "critical_volume":
            return est.Vc(counts), "m^3/mol", None, ()
        if prop == "enthalpy_vaporization":
            tb = est.Tb(counts)
            at_tb = Conditions(temperature=Quantity(value=tb, unit="K"))
            return (
                est.Hvap(counts),
                "J/mol",
                at_tb,
                ("defined at the normal boiling point, not at the requested temperature",),
            )
        if prop == "enthalpy_fusion":
            return est.Hfus(counts), "J/mol", None, ()
        if prop == "heat_capacity_gas":
            temperature = request.conditions.temperature_k
            if temperature is None:
                return (
                    None,
                    "J/mol/K",
                    None,
                    ("ideal-gas heat capacity needs a temperature, and none was given",),
                )
            return (
                est.Cpig(temperature),
                "J/mol/K",
                request.conditions,
                ("ideal-gas value; not the liquid or solid heat capacity",),
            )
        return None, "", None, ()
