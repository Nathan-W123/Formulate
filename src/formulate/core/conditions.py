"""Operating conditions and condition matching.

Specification section 11: "predictions are invalid outside stated
temperature/pressure/environment unless explicitly modeled."  Conditions are
therefore attached to targets and to predictions alike, and the evaluation
layer compares them before a prediction is allowed to satisfy a requirement.
"""

from __future__ import annotations

import math
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


class Spinline(BaseModel):
    """The geometry a filament is made at.

    Every dead end in ``examples/web_shooter`` was geometric rather than
    chemical - a strand too thick to solidify in time, a jet too thin to
    survive its own surface tension, an orifice too narrow to push anything
    through. None of it was visible to the engine, because a candidate carried
    what a material *is* and nothing about the shape it is made into, and a
    solidification time is not a property of polyethylene. It is a property of
    polyethylene at a diameter.

    So geometry enters where the substrate does. ``surfaces`` set the
    precedent: adhesion is a property of an interface, the substrate is not
    part of the candidate, and it is read from the conditions. A filament
    diameter is the same kind of thing - a condition of use, not a property of
    the material - and putting it here keeps the candidate a material and lets
    a spec state the shape it intends.
    """

    model_config = ConfigDict(frozen=True)

    #: Diameter of the filament as it leaves the die.
    die_diameter: Quantity | None = None
    #: Draw-down ratio between the die and the solidified filament. A melt
    #: spinning line runs at 10-100; the filament that arrives is thinner than
    #: the hole it came from by the square root of this.
    draw_ratio: float | None = None
    #: Speed of the filament along the spinline.
    line_speed: Quantity | None = None
    #: Length of the straight section of the die.
    die_land: Quantity | None = None
    #: Number of filaments run in parallel into one bundle.
    filament_count: int | None = None

    @property
    def final_diameter(self) -> Quantity | None:
        """Diameter after draw-down, which is what sets solidification."""
        if self.die_diameter is None:
            return None
        if self.draw_ratio is None or self.draw_ratio <= 0:
            return self.die_diameter
        value = self.die_diameter.to("m").value / math.sqrt(self.draw_ratio)
        return Quantity(value=value, unit="m")

    def identity_payload(self) -> dict:
        def _value(quantity, unit):
            if quantity is None:
                return None
            return float(f"{quantity.to(unit).value:.12g}")

        return {
            "die_diameter_m": _value(self.die_diameter, "m"),
            "draw_ratio": self.draw_ratio,
            "line_speed_m_s": _value(self.line_speed, "m/s"),
            "die_land_m": _value(self.die_land, "m"),
            "filament_count": self.filament_count,
        }

    def describe(self) -> str:
        bits = []
        if self.die_diameter is not None:
            bits.append(f"die {self.die_diameter.to('m').value*1e6:.0f} um")
        if self.draw_ratio is not None:
            bits.append(f"draw {self.draw_ratio:g}x")
        if self.line_speed is not None:
            bits.append(f"{self.line_speed.to('m/s').value:g} m/s")
        if self.filament_count is not None:
            bits.append(f"x{self.filament_count}")
        return ", ".join(bits)


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
    #: Materials that have to be dissolved, e.g. ("polystyrene",). Named
    #: rather than given as parameters, so that a target states the behaviour
    #: it wants and the engine supplies what that implies. Mirrors
    #: ``surfaces``, which the adhesion expert already reads the same way.
    solutes: tuple[str, ...] = ()
    #: The geometry a filament is made at. Absent for anything that is not
    #: being spun, which is most candidates.
    spinline: Spinline | None = None

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: Quantity | None) -> Quantity | None:
        if v is not None and dimensionality(v.unit) != dimensionality("kelvin"):
            raise ValueError(f"temperature must have temperature units, got {v.unit!r}")
        return v

    @field_validator("pressure")
    @classmethod
    def _check_pressure(cls, v: Quantity | None) -> Quantity | None:
        if v is not None and dimensionality(v.unit) != dimensionality("pascal"):
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
            solutes=other.solutes or self.solutes,
            spinline=other.spinline or self.spinline,
        )

    def identity_payload(self) -> dict:
        """Canonical form for content addressing.

        Every other quantity in the identity chain is canonicalised before it
        is hashed: a molecule through its canonical SMILES, a polymer's molar
        mass through g/mol. Conditions were hashed as written, so a candidate
        at "25 degC, 1 atm" and the same candidate at "298.15 K, 101325 Pa"
        were two different candidates with two separate cache entries, and
        nothing in the system could notice they were the same request.

        ``processing`` keeps its order because a cure schedule is a sequence;
        ``surfaces`` is a set of what is present, so it is sorted.
        """
        def _value(quantity, unit):
            if quantity is None:
                return None
            # Twelve significant figures: far finer than any condition is
            # known to, and coarse enough that two spellings of the same
            # condition cannot differ in the last bit of a conversion.
            return float(f"{quantity.to(unit).value:.12g}")

        return {
            "temperature_k": _value(self.temperature, "kelvin"),
            "pressure_pa": _value(self.pressure, "pascal"),
            "environment": self.environment,
            "phase": None if self.phase is None else self.phase.value,
            "processing": list(self.processing),
            "surfaces": sorted(self.surfaces),
            "solutes": sorted(self.solutes),
            "spinline": None if self.spinline is None else self.spinline.identity_payload(),
        }

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
        if self.solutes:
            bits.append(f"solutes={'+'.join(self.solutes)}")
        if self.spinline is not None:
            bits.append(f"spinline[{self.spinline.describe()}]")
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

    for solute in required.solutes:
        if solute not in available.solutes:
            unstated.append(f"prediction does not model solute {solute!r}")

    return ConditionMatch(
        compatible=not issues, issues=tuple(issues), unstated=tuple(unstated)
    )
