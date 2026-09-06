"""Interfacial and liquid-state expert built on empirical correlations.

Nothing here is a fitted machine-learning model; these are the standard
engineering correlations, evaluated on critical constants supplied by an
upstream expert:

* liquid molar volume and density - Rackett equation.
* surface tension - Brock-Bird corresponding-states correlation.
* Hildebrand solubility parameter - from the Watson-corrected enthalpy of
  vaporisation and the liquid molar volume.

Because the inputs are themselves estimates, this expert propagates their
uncertainty numerically rather than reporting only its own correlation error.
A surface tension derived from an estimated critical pressure cannot honestly
claim the accuracy the correlation shows on measured constants.

References: Reid, Prausnitz and Poling, "The Properties of Gases and Liquids",
4th ed., for the Rackett and Brock-Bird correlations and the Watson relation.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest

R_GAS = 8.31446261815324  # J/(mol K)

#: Correlation error, as a fraction of the value, on *measured* inputs, for a
#: normal (non-associating) fluid.
_METHOD_RELATIVE_ERROR = {
    "molar_volume_liquid": 0.05,
    "liquid_density": 0.05,
    "surface_tension": 0.10,
    "hildebrand_solubility_parameter": 0.07,
}

#: Multiplier on the correlation error for hydrogen-bonding liquids, indexed by
#: hydrogen-bond donor count (capped at 2).
#:
#: Corresponding-states correlations assume the only intermolecular forces are
#: dispersion and dipolar. Hydrogen bonding breaks that assumption, and the
#: error grows sharply with the number of donors. Measured against the bundled
#: reference compounds, surface tension carries a mean absolute error of
#: 2.0 mN/m at zero donors, 9.9 at one and 21-53 at two or three; liquid
#: density runs 4% / 10% / 18% over the same groups. Reporting one fixed error
#: bar across all three groups would be overconfident for alcohols and acids,
#: which is precisely the failure section 12 asks the system to detect.
#:
#: The multipliers are chosen so that the fraction of reference compounds
#: falling inside their own stated one-sigma bound lands near the ~68% a
#: correct estimate implies, erring slightly wide rather than slightly narrow.
_ASSOCIATION_FACTOR: dict[str, dict[int, float]] = {
    "surface_tension": {0: 1.0, 1: 2.5, 2: 4.5},
    "hildebrand_solubility_parameter": {0: 1.0, 1: 3.0, 2: 4.0},
    "liquid_density": {0: 1.0, 1: 2.5, 2: 3.5},
    "molar_volume_liquid": {0: 1.0, 1: 2.5, 2: 3.5},
}


def _hbd_count(smiles: str) -> int:
    from formulate import chem

    return int(chem.descriptors(smiles).get("hbd", 0.0))


def association_factor(prop: str, hbd: int) -> float:
    """Error multiplier for ``prop`` given a hydrogen-bond donor count."""
    return _ASSOCIATION_FACTOR[prop][min(hbd, 2)]

_DEPENDENCY_UNITS = {
    "critical_temperature": "K",
    "critical_pressure": "Pa",
    "critical_volume": "m^3/mol",
    "normal_boiling_point": "K",
    "enthalpy_vaporization": "J/mol",
    "molar_mass": "g/mol",
}

_NEEDED: dict[str, tuple[str, ...]] = {
    "molar_volume_liquid": ("critical_temperature", "critical_pressure", "critical_volume"),
    "liquid_density": (
        "critical_temperature", "critical_pressure", "critical_volume", "molar_mass",
    ),
    "surface_tension": (
        "critical_temperature", "critical_pressure", "normal_boiling_point",
    ),
    "hildebrand_solubility_parameter": (
        "critical_temperature", "critical_pressure", "critical_volume",
        "normal_boiling_point", "enthalpy_vaporization",
    ),
}


# -- correlations -----------------------------------------------------------


def rackett_molar_volume(tc: float, pc: float, vc: float, temperature: float) -> float:
    """Saturated liquid molar volume in m^3/mol via the Rackett equation.

    The Rackett compressibility is approximated by the critical
    compressibility computed from the supplied constants, which is the usual
    fallback when no measured Zra is available.
    """
    z_ra = pc * vc / (R_GAS * tc)
    reduced = temperature / tc
    return (R_GAS * tc / pc) * z_ra ** (1.0 + (1.0 - reduced) ** (2.0 / 7.0))


def brock_bird_surface_tension(tc: float, pc: float, tb: float, temperature: float) -> float:
    """Surface tension in N/m via the Brock-Bird correlation."""
    pc_bar = pc / 1e5
    reduced_boiling = tb / tc
    reduced = temperature / tc
    q = (
        0.1196
        * (1.0 + reduced_boiling * math.log(pc_bar / 1.01325) / (1.0 - reduced_boiling))
        - 0.279
    )
    sigma_dyn_cm = (
        pc_bar ** (2.0 / 3.0) * tc ** (1.0 / 3.0) * q * (1.0 - reduced) ** (11.0 / 9.0)
    )
    return sigma_dyn_cm * 1e-3  # dyn/cm -> N/m


def watson_enthalpy_vaporization(hvap_tb: float, tb: float, tc: float, temperature: float) -> float:
    """Correct an enthalpy of vaporisation from Tb to another temperature."""
    numerator = 1.0 - temperature / tc
    denominator = 1.0 - tb / tc
    return hvap_tb * (numerator / denominator) ** 0.38


def hildebrand_parameter(
    hvap_tb: float, tb: float, tc: float, vc: float, pc: float, temperature: float
) -> float:
    """Hildebrand solubility parameter in Pa^0.5."""
    hvap_t = watson_enthalpy_vaporization(hvap_tb, tb, tc, temperature)
    molar_volume = rackett_molar_volume(tc, pc, vc, temperature)
    cohesive_energy_density = (hvap_t - R_GAS * temperature) / molar_volume
    if cohesive_energy_density <= 0.0:
        raise ValueError("non-positive cohesive energy density")
    return math.sqrt(cohesive_energy_density)


# -- uncertainty propagation ------------------------------------------------


def propagate(
    fn: Callable[[Mapping[str, float]], float],
    inputs: Mapping[str, tuple[float, float | None]],
) -> tuple[float, float]:
    """Propagate input uncertainties through ``fn`` by central differences.

    Returns ``(value, std)``.  Inputs are assumed independent, which
    understates the true spread when several come from the same
    group-contribution fit; the prediction's ``basis`` string records that
    caveat rather than hiding it.
    """
    nominal = {k: v for k, (v, _) in inputs.items()}
    value = fn(nominal)

    variance = 0.0
    for key, (magnitude, std) in inputs.items():
        if not std:
            continue
        step = std
        up, down = dict(nominal), dict(nominal)
        up[key] = magnitude + step
        down[key] = magnitude - step
        try:
            derivative = (fn(up) - fn(down)) / (2.0 * step)
        except (ValueError, ZeroDivisionError, OverflowError):
            # A one-sided difference where the perturbation leaves the valid range.
            try:
                derivative = (fn(up) - value) / step
            except Exception:
                continue
        variance += (derivative * std) ** 2
    return value, math.sqrt(variance)


class InterfacialCorrelationExpert(Expert):
    """Liquid density, surface tension and cohesion from critical constants."""

    id = "interfacial"
    version = "1"
    method = "Rackett, Brock-Bird and Watson correlations on estimated critical constants"
    family = PropertyFamily.INTERFACIAL
    supported_classes = frozenset({MaterialClass.MOLECULE})
    supported_properties = frozenset(_METHOD_RELATIVE_ERROR)
    dependencies = frozenset(_DEPENDENCY_UNITS)

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        from formulate import chem

        smiles = candidate.molecule.smiles if candidate.molecule else None
        if smiles is None or chem.mol_from_smiles(smiles) is None:
            return ApplicabilityDomain.outside("candidate carries no parseable molecule")

        descriptors = chem.descriptors(smiles)
        warnings: list[str] = []
        score = 1.0

        # Brock-Bird and Rackett are corresponding-states correlations. They
        # were developed for normal fluids; strongly hydrogen-bonding liquids
        # (alcohols, acids, water, amides) deviate systematically.
        hbd = int(descriptors.get("hbd", 0.0))
        if hbd >= 1:
            warnings.append(
                f"{hbd} hydrogen-bond donor(s): corresponding-states correlations are "
                "developed for normal fluids and deviate systematically for strongly "
                "associating liquids such as alcohols, acids and amides. Against the "
                "reference set, surface tension errs by about 10 mN/m at one donor and "
                "20-50 mN/m at two or more"
            )
            score = min(score, 0.5 if hbd == 1 else 0.2)

        if descriptors.get("formal_charge", 0.0):
            return ApplicabilityDomain.outside(
                "these correlations do not apply to ionic species",
                basis="neutral normal fluids",
            )

        return ApplicabilityDomain(
            score=score,
            in_domain=score > 0.3,
            warnings=tuple(warnings),
            basis="neutral, weakly associating liquids below their critical temperature",
        )

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        temperature = request.conditions.temperature_k
        if temperature is None:
            return Prediction.unsupported(
                prop, self.id, f"{prop} is temperature dependent and no temperature was given"
            )

        inputs: dict[str, tuple[float, float | None]] = {}
        for name in _NEEDED[prop]:
            unit = _DEPENDENCY_UNITS[name]
            pred = request.dependency(name)
            if pred is None or pred.quantity is None:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"needs {name} from an upstream expert, and none was available",
                )
            value = pred.quantity.to(unit).value
            std = pred.uncertainty.converted(pred.quantity.unit, unit).std
            inputs[name] = (value, std)

        tc = inputs["critical_temperature"][0]
        if temperature >= tc:
            return Prediction.failed(
                prop,
                self.id,
                f"requested temperature {temperature:.1f} K is at or above the estimated "
                f"critical temperature {tc:.1f} K; there is no liquid phase to describe",
            )

        fn, unit = self._correlation(prop, temperature)
        try:
            value, propagated = propagate(fn, inputs)
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            return Prediction.failed(prop, self.id, f"correlation failed: {exc}")

        smiles = request.candidate.molecule.smiles  # type: ignore[union-attr]
        hbd = _hbd_count(smiles)
        factor = association_factor(prop, hbd)
        relative = _METHOD_RELATIVE_ERROR[prop] * factor
        method_std = abs(value) * relative
        total = math.hypot(method_std, propagated)

        dominant = "propagated input error" if propagated > method_std else "correlation error"
        return self._make(
            prop,
            value,
            unit,
            request,
            domain,
            std=total,
            kind=UncertaintyKind.COMBINED,
            basis=(
                f"correlation error {relative:.0%} on measured inputs ({method_std:.4g}), "
                + (
                    f"widened {factor:.1f}x for {hbd} hydrogen-bond donor(s), "
                    if factor > 1.0
                    else ""
                )
                + f"combined in quadrature with {propagated:.4g} propagated from the "
                f"estimated critical constants; {dominant} dominates. Inputs are treated "
                "as independent, which understates the spread when several come from the "
                "same group-contribution fit"
            ),
            notes=(
                f"derived from upstream estimates of {', '.join(sorted(inputs))}, "
                "not from measured constants",
            ),
            temperature_k=temperature,
        )

    def _correlation(
        self, prop: str, temperature: float
    ) -> tuple[Callable[[Mapping[str, float]], float], str]:
        if prop == "molar_volume_liquid":
            return (
                lambda v: rackett_molar_volume(
                    v["critical_temperature"], v["critical_pressure"],
                    v["critical_volume"], temperature,
                ),
                "m^3/mol",
            )
        if prop == "liquid_density":
            return (
                lambda v: (v["molar_mass"] * 1e-3)
                / rackett_molar_volume(
                    v["critical_temperature"], v["critical_pressure"],
                    v["critical_volume"], temperature,
                ),
                "kg/m^3",
            )
        if prop == "surface_tension":
            return (
                lambda v: brock_bird_surface_tension(
                    v["critical_temperature"], v["critical_pressure"],
                    v["normal_boiling_point"], temperature,
                ),
                "N/m",
            )
        return (
            lambda v: hildebrand_parameter(
                v["enthalpy_vaporization"], v["normal_boiling_point"],
                v["critical_temperature"], v["critical_volume"],
                v["critical_pressure"], temperature,
            ),
            "Pa^0.5",
        )
