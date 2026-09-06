"""Operating conditions and condition matching.

Specification section 11: "predictions are invalid outside stated
temperature/pressure/environment unless explicitly modeled."  Conditions are
therefore attached to targets and to predictions alike, and the evaluation
layer compares them before a prediction is allowed to satisfy a requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator

from .quantity import Quantity
from .units import dimensionality

STANDARD_TEMPERATURE_K = 298.15
STANDARD_PRESSURE_PA = 101325.0


class Phase(str, Enum):
    SOLID = "solid"
    LIQUID = "liquid"
    GAS = "gas"
    SOLUTION = "solution"
    MELT = "melt"
    UNSPECIFIED = "unspecified"


class Conditions(BaseModel):
    """The physical context a property is defined at.

    Every field is optional: an unstated condition means "not specified",
    which is different from "standard".  The distinction matters because an
    unstated condition cannot be silently matched against a stated one.
    """

    model_config = ConfigDict(frozen=True)

    temperature: Quantity | None = None
    pressure: Quantity | None = None
    #: Surrounding medium, e.g. "air", "nitrogen", "water", "vacuum".
    environment: str | None = None
    phase: Phase | None = None
    #: Free-form processing / cure history for polymers and formulations.
    processing: tuple[str, ...] = ()
    #: Surfaces or interfaces present, e.g. ("aluminium-oxide",).
    surfaces: tuple[str, ...] = ()

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: Quantity | None) -> Quantity | None:
        if v is not None and dimensionality(v.unit) != "[temperature]":
            raise ValueError(f"temperature must have temperature units, got {v.unit!r}")
        return v

    @field_validator("pressure")
    @classmethod
    def _check_pressure(cls, v: Quantity | None) -> Quantity | None:
        if v is not None and dimensionality(v.unit) != "[mass] / [length] / [time] ** 2":
            raise ValueError(f"pressure must have pressure units, got {v.unit!r}")
        return v

    @classmethod
    def standard(cls) -> "Conditions":
        """298.15 K, 1 atm, unspecified environment."""
        return cls(
            temperature=Quantity(value=STANDARD_TEMPERATURE_K, unit="kelvin"),
            pressure=Quantity(value=STANDARD_PRESSURE_PA, unit="pascal"),
        )

    @property
    def temperature_k(self) -> float | None:
        return None if self.temperature is None else self.temperature.to("kelvin").value

    @property
    def pressure_pa(self) -> float | None:
        return None if self.pressure is None else self.pressure.to("pascal").value

    def merged_with(self, other: "Conditions") -> "Conditions":
        """Overlay ``other`` onto this, with ``other`` taking precedence."""
        return Conditions(
            temperature=other.temperature or self.temperature,
            pressure=other.pressure or self.pressure,
            environment=other.environment or self.environment,
            phase=other.phase or self.phase,
            processing=other.processing or self.processing,
            surfaces=other.surfaces or self.surfaces,
        )

    def describe(self) -> str:
        bits = []
        if self.temperature is not None:
            bits.append(f"T={self.temperature.to('kelvin').value:.6g} K")
        if self.pressure is not None:
            bits.append(f"P={self.pressure.to('pascal').value:.6g} Pa")
        if self.environment:
            bits.append(f"env={self.environment}")
        if self.phase:
            bits.append(f"phase={self.phase.value}")
        if self.surfaces:
            bits.append(f"surfaces={'+'.join(self.surfaces)}")
        return ", ".join(bits) or "unspecified"


@dataclass(frozen=True, slots=True)
class ConditionMatch:
    """Result of comparing the conditions of a prediction against a request."""

    compatible: bool
    #: Reasons the match failed or was only partial.
    issues: tuple[str, ...] = ()
    #: True when the request specified a condition the prediction did not.
    unstated: tuple[str, ...] = ()

    @property
    def exact(self) -> bool:
        return self.compatible and not self.issues and not self.unstated


def match_conditions(
    required: Conditions,
    available: Conditions,
    *,
    temperature_tolerance_k: float = 5.0,
    pressure_rtol: float = 0.05,
) -> ConditionMatch:
    """Decide whether a prediction made at ``available`` may answer ``required``.

    An unstated condition on the prediction side is reported rather than
    assumed to be standard: silently defaulting is exactly the substitution
    section 11 forbids.
    """
    issues: list[str] = []
    unstated: list[str] = []

    req_t, avl_t = required.temperature_k, available.temperature_k
    if req_t is not None:
        if avl_t is None:
            unstated.append("prediction states no temperature")
        elif abs(req_t - avl_t) > temperature_tolerance_k:
            issues.append(
                f"temperature mismatch: required {req_t:.6g} K, prediction at {avl_t:.6g} K "
                f"(tolerance {temperature_tolerance_k:g} K)"
            )

    req_p, avl_p = required.pressure_pa, available.pressure_pa
    if req_p is not None:
        if avl_p is None:
            unstated.append("prediction states no pressure")
        elif req_p > 0 and abs(req_p - avl_p) / req_p > pressure_rtol:
            issues.append(
                f"pressure mismatch: required {req_p:.6g} Pa, prediction at {avl_p:.6g} Pa"
            )

    if required.environment and available.environment:
        if required.environment != available.environment:
            issues.append(
                f"environment mismatch: required {required.environment!r}, "
                f"prediction in {available.environment!r}"
            )
    elif required.environment and not available.environment:
        unstated.append("prediction states no environment")

    if required.phase and available.phase and required.phase != available.phase:
        issues.append(
            f"phase mismatch: required {required.phase.value}, "
            f"prediction for {available.phase.value}"
        )

    for surface in required.surfaces:
        if surface not in available.surfaces:
            unstated.append(f"prediction does not model surface {surface!r}")

    return ConditionMatch(
        compatible=not issues, issues=tuple(issues), unstated=tuple(unstated)
    )
