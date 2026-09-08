"""Refractive index from Lorentz-Lorenz, an actual physical relation.

Section 4 asks for a mixture of experts and section 13 warns against
reimplementing what physics already gives you. The refractive index of a liquid
is one of the few properties where a closed-form relation genuinely works:

    (n^2 - 1) / (n^2 + 2) = R_m / V_m

``R_m`` is the molar refraction, which is additive over bonds and is what
RDKit's Crippen ``MolMR`` computes; ``V_m`` is the molar volume, molar mass over
density. Rearranged, ``n = sqrt((1 + 2x) / (1 - x))`` with ``x = R_m / V_m``.

Why this expert and the learned one both exist
    Measured on 385 compounds that have a tabulated liquid density, with a
    random forest refitted on the other 4026 so it had never seen any of them:

        Lorentz-Lorenz, measured density     MAE 0.0123   RMSE 0.0378
        Lorentz-Lorenz, estimated density    MAE 0.1995   RMSE 2.5501
        learned forest                       MAE 0.0165   RMSE 0.0342

    The physics wins when it is given a real density and is catastrophic when
    it is not - the estimated-density row is the same equation fed a
    Joback-then-Rackett molar volume, and its RMSE is sixty-seven times worse
    than its own MAE because a molar volume slightly too small sends ``x``
    toward one and ``n`` toward infinity.

    So neither route is simply better, and no precedence rule is written here.
    The density's own uncertainty is propagated through the equation, so a
    prediction built on a measured density states a narrow spread and one built
    on an estimate states a wide one, and ``prefer()`` - which selects on the
    tightest stated in-domain uncertainty - picks between them on the evidence.
    That is the section 4 mechanism working as intended rather than a table of
    which expert wins.
"""

from __future__ import annotations

import math

from formulate.core.candidate import MaterialClass
from formulate.core.prediction import Prediction, PredictionStatus
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import ProvenanceKind, ProvenanceRecord
from formulate.core.quantity import ApplicabilityDomain, Quantity, Uncertainty, UncertaintyKind

from .base import Expert, PredictionRequest
from .interfacial import propagate

#: Units each dependency is converted to before entering the equation.
_DEPENDENCY_UNITS = {
    "liquid_density": "kg/m^3",
    "molar_mass": "g/mol",
    "molar_refractivity": "cm^3/mol",
}

#: The route's error given a tabulated density, measured against 385 tabulated
#: refractive indices: mean absolute error 0.0123, converted to a one-sigma
#: spread by the sigma = 1.253 * MAE relation used elsewhere here.
#:
#: This figure already contains Crippen's molar-refraction error, because the
#: 385 predictions it was measured from were made with Crippen molar
#: refractions. Propagating the molar refractivity's own stated uncertainty on
#: top of it would count that error twice, and did: Crippen states 2.5 cm^3/mol
#: on toluene's 31.2, eight per cent, which propagates to about 0.047 in n and
#: made this expert quote +/- 0.049 on an answer that was in fact within 0.0012
#: of the measured value. The ranking then preferred the learned route, which
#: was fourteen times further out and said so more confidently.
_EQUATION_STD = 0.0123 * 1.253

#: Only the density's uncertainty is propagated, for that reason. It is also
#: the input that actually varies: a tabulated density is good to a kilogram
#: per cubic metre and a Rackett estimate to tens, and that difference is what
#: this expert exists to report honestly.
_PROPAGATED_INPUTS = frozenset({"liquid_density"})

#: Crippen's molar refraction is itself a fitted quantity. Its contribution is
#: inside the 0.0123 above rather than separated out, because separating it
#: would need molar refractions measured independently of refractive indices,
#: and those are derived from refractive indices.
_CRIPPEN_NOTE = (
    "molar refraction from Crippen's atomic contributions, which are fitted to "
    "measured refractive indices; the equation's error and the fit's are not "
    "separable here and both sit inside the stated spread"
)


def lorentz_lorenz(molar_refraction_cm3: float, molar_volume_cm3: float) -> float:
    """``n`` from a molar refraction and a molar volume, both in cm^3/mol.

    Raises ``ValueError`` when the ratio reaches one, which is where the
    equation has a pole: a molar volume at or below the molar refraction
    describes a medium denser than the polarisability it is made of, which is
    not a liquid but an arithmetic accident of two estimates disagreeing.
    """
    if molar_volume_cm3 <= 0.0:
        raise ValueError("molar volume must be positive")
    ratio = molar_refraction_cm3 / molar_volume_cm3
    if not 0.0 < ratio < 1.0:
        raise ValueError(
            f"molar refraction is {ratio:.3f} of the molar volume; Lorentz-Lorenz has "
            "a pole at one and is meaningless at or above it"
        )
    return math.sqrt((1.0 + 2.0 * ratio) / (1.0 - ratio))


class LorentzLorenzExpert(Expert):
    """Refractive index from a density and a molar refraction."""

    id = "lorentz_lorenz"
    version = "1"
    family = PropertyFamily.ELECTRICAL
    supported_properties = frozenset({"refractive_index"})
    supported_classes = frozenset({MaterialClass.MOLECULE})
    dependencies = frozenset(_DEPENDENCY_UNITS)

    def assess_domain(self, candidate) -> ApplicabilityDomain:
        return ApplicabilityDomain(
            score=1.0,
            in_domain=True,
            basis=(
                "Lorentz-Lorenz applies to any non-absorbing isotropic liquid; the "
                "domain question here is whether the density it is given is trustworthy, "
                "and that is carried in the uncertainty rather than in this score"
            ),
        )

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        if request.candidate.molecule is None:
            return Prediction.unsupported(
                prop, self.id, "needs a single molecular structure"
            )
        temperature = request.conditions.temperature_k
        if temperature is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "refractive index is temperature dependent through the density, and no "
                "temperature was given",
            )

        inputs: dict[str, tuple[float, float | None]] = {}
        sources: list[str] = []
        for name, unit in _DEPENDENCY_UNITS.items():
            upstream = request.dependency(name)
            if upstream is None or upstream.quantity is None:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"needs {name} from an upstream expert, and none was available",
                )
            inputs[name] = (
                upstream.quantity.to(unit).value,
                upstream.uncertainty.converted(upstream.quantity.unit, unit).std,
            )
            sources.append(f"{name} from {upstream.expert_id}")

        # A refractive index of a liquid is a property of the liquid.
        from .measured import not_liquid_at

        wrong_phase = not_liquid_at(request.candidate.molecule.smiles, temperature)
        if wrong_phase:
            return Prediction.unsupported(prop, self.id, wrong_phase)

        def equation(values):
            molar_volume = values["molar_mass"] / (values["liquid_density"] / 1000.0)
            return lorentz_lorenz(values["molar_refractivity"], molar_volume)

        # The molar mass is exact and the molar refraction's error is already
        # inside _EQUATION_STD, so neither is given a spread to propagate.
        varying = {
            name: (value, std if name in _PROPAGATED_INPUTS else None)
            for name, (value, std) in inputs.items()
        }
        try:
            value, propagated = propagate(equation, varying)
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            return Prediction.failed(prop, self.id, f"Lorentz-Lorenz failed: {exc}")

        # The equation's own error and whatever the density carried, in
        # quadrature. When the density is measured the first term dominates and
        # this expert states about 0.015; when it is estimated the second does
        # and the spread widens by an order of magnitude, which is what lets
        # prefer() choose the learned route instead without a precedence rule.
        combined = math.sqrt(_EQUATION_STD**2 + (propagated or 0.0) ** 2)
        density_std = inputs["liquid_density"][1]
        dominant = (
            "the density's own uncertainty"
            if propagated and propagated > _EQUATION_STD
            else "the equation itself"
        )

        return Prediction(
            property=prop,
            quantity=Quantity(value=value, unit=""),
            uncertainty=Uncertainty(
                std=combined,
                kind=UncertaintyKind.EPISTEMIC,
                basis=(
                    f"Lorentz-Lorenz with a Crippen molar refraction reproduces 385 "
                    f"tabulated refractive indices to {_EQUATION_STD:.4f} one sigma when "
                    "given a tabulated density, which already includes the molar "
                    "refraction's own error; "
                    + (
                        f"combined in quadrature with {propagated:.4f} propagated from a "
                        f"density stated to {density_std:.4g} kg/m^3"
                        if propagated
                        else "the density stated no uncertainty, so nothing was propagated"
                    )
                    + f"; {dominant} dominates"
                ),
            ),
            applicability=domain,
            status=PredictionStatus.OK,
            expert_id=self.id,
            expert_version=self.version,
            method="Lorentz-Lorenz with Crippen molar refraction",
            conditions=request.conditions,
            provenance=ProvenanceRecord(
                kind=ProvenanceKind.PREDICTION,
                producer=self.id,
                producer_version=self.version,
                parameters={"temperature_k": temperature},
            ),
            notes=(_CRIPPEN_NOTE, "derived from " + ", ".join(sources)),
        )
