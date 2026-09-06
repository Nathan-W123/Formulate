"""Dimensional scalars and first-class uncertainty.

Specification section 11 makes uncertainty a first-class field "throughout" and
requires that aleatoric, model (epistemic) and simulation sampling error be
distinguishable.  A bare float is therefore never an acceptable carrier for a
predicted property anywhere in this codebase.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import DimensionalityError
from .units import canonical_unit, convert, convert_delta, dimensionality, parse_unit


class Quantity(BaseModel):
    """A scalar magnitude with a unit.

    Immutable.  Arithmetic is deliberately not implemented: quantities in this
    system are results to be compared and converted, and ad-hoc arithmetic on
    them is how offset-unit bugs get introduced.
    """

    model_config = ConfigDict(frozen=True)

    value: float
    unit: str = Field(default="dimensionless")

    @field_validator("unit")
    @classmethod
    def _validate_unit(cls, v: str) -> str:
        return str(parse_unit(v))

    @property
    def dimensionality(self) -> str:
        return dimensionality(self.unit)

    def to(self, unit: str) -> "Quantity":
        """Return this quantity converted to ``unit``."""
        return Quantity(value=convert(self.value, self.unit, unit), unit=unit)

    def to_canonical(self) -> "Quantity":
        """Return this quantity in the canonical unit for its dimension."""
        target = canonical_unit(self.unit)
        return self if str(parse_unit(target)) == self.unit else self.to(target)

    def is_compatible_with(self, other: "Quantity") -> bool:
        return self.dimensionality == other.dimensionality

    def __str__(self) -> str:
        if self.unit == "dimensionless":
            return f"{self.value:.6g}"
        return f"{self.value:.6g} {self.unit}"


class UncertaintyKind(str, Enum):
    """Where an uncertainty estimate comes from.

    These are not interchangeable.  A published training-set error is a model
    error and does not shrink with more sampling; a molecular-dynamics standard
    error does.  Section 11 requires them to stay distinguishable.
    """

    #: Irreducible scatter in the underlying measurement/data.
    ALEATORIC = "aleatoric"
    #: Model error, e.g. published held-out RMSE of a correlation.
    EPISTEMIC = "epistemic"
    #: Finite-time / finite-size sampling error from a simulation.
    SAMPLING = "sampling"
    #: Several sources combined; the components are no longer separable.
    COMBINED = "combined"
    #: No usable estimate.  Never treated as zero.
    UNKNOWN = "unknown"


class Uncertainty(BaseModel):
    """Uncertainty attached to a predicted value.

    ``std`` is a one-sigma spread expressed in the *same unit* as the value it
    accompanies.  ``basis`` is free text recording where the number came from,
    so that a reviewer can tell a published RMSE from a guess.
    """

    model_config = ConfigDict(frozen=True)

    std: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    kind: UncertaintyKind = UncertaintyKind.UNKNOWN
    basis: str = ""

    @field_validator("std")
    @classmethod
    def _non_negative(cls, v: float | None) -> float | None:
        if v is not None and (v < 0 or math.isnan(v)):
            raise ValueError("Uncertainty std must be non-negative and finite.")
        return v

    @property
    def is_informative(self) -> bool:
        """True when this carries a usable spread estimate."""
        return self.std is not None or (self.ci_low is not None and self.ci_high is not None)

    def converted(self, from_unit: str, to_unit: str) -> "Uncertainty":
        """Convert the spread to another unit.

        Uses difference semantics, so a 12.9 K spread stays 12.9 when expressed
        in degrees Celsius rather than becoming -260.
        """
        if dimensionality(from_unit) != dimensionality(to_unit):
            raise DimensionalityError(
                f"Cannot convert uncertainty from {from_unit!r} to {to_unit!r}."
            )
        f = convert_delta

        def _c(x: float | None) -> float | None:
            return None if x is None else f(x, from_unit, to_unit)

        # Confidence-interval endpoints are absolute values, not spreads.
        return Uncertainty(
            std=_c(self.std),
            ci_low=None if self.ci_low is None else convert(self.ci_low, from_unit, to_unit),
            ci_high=None if self.ci_high is None else convert(self.ci_high, from_unit, to_unit),
            kind=self.kind,
            basis=self.basis,
        )

    @classmethod
    def unknown(cls, reason: str = "") -> "Uncertainty":
        return cls(kind=UncertaintyKind.UNKNOWN, basis=reason)

    def __str__(self) -> str:
        if self.std is not None:
            return f"+/-{self.std:.4g} (1 sigma, {self.kind.value})"
        if self.ci_low is not None and self.ci_high is not None:
            return f"[{self.ci_low:.4g}, {self.ci_high:.4g}] ({self.kind.value})"
        return "uncertainty unknown"


class ApplicabilityDomain(BaseModel):
    """How far a candidate sits outside the region a model was built for.

    Section 4 requires every prediction to report an applicability-domain
    score, and section 12 measures the system on its "ability to identify
    out-of-domain expert predictions".
    """

    model_config = ConfigDict(frozen=True)

    #: 1.0 = squarely inside the model's domain, 0.0 = definitely outside.
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    in_domain: bool = True
    warnings: tuple[str, ...] = ()
    basis: str = ""

    @classmethod
    def outside(cls, *warnings: str, score: float = 0.0, basis: str = "") -> "ApplicabilityDomain":
        return cls(score=score, in_domain=False, warnings=tuple(warnings), basis=basis)

    def merged_with(self, other: "ApplicabilityDomain") -> "ApplicabilityDomain":
        """Combine two domain assessments conservatively (worst case wins)."""
        return ApplicabilityDomain(
            score=min(self.score, other.score),
            in_domain=self.in_domain and other.in_domain,
            warnings=tuple(dict.fromkeys(self.warnings + other.warnings)),
            basis="; ".join(x for x in (self.basis, other.basis) if x),
        )
