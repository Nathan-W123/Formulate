"""The prediction record produced by every expert.

Specification section 4: "Every prediction returns value/distribution, units,
uncertainty/confidence, applicability-domain score, model/version, conditions,
and provenance."

Section 4 also states that "missing/uncertain predictions must not silently
become neutral scores".  That is enforced structurally here: a prediction that
did not produce a value carries an explicit :class:`PredictionStatus` and a
``quantity`` of ``None``, so downstream code cannot mistake absence for a
mid-range result.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conditions import Conditions
from .properties import get_property, validate_unit_for
from .provenance import ProvenanceRecord
from .quantity import ApplicabilityDomain, Quantity, Uncertainty


class PredictionStatus(str, Enum):
    """Outcome of an attempted prediction."""

    #: A value was produced and the candidate is inside the model's domain.
    OK = "ok"
    #: A value was produced but the candidate is outside the model's domain.
    OUT_OF_DOMAIN = "out_of_domain"
    #: This expert does not cover this property or material class at all.
    UNSUPPORTED = "unsupported"
    #: The expert was applicable but failed (bad structure, backend error).
    FAILED = "failed"

    @property
    def has_value(self) -> bool:
        return self in (PredictionStatus.OK, PredictionStatus.OUT_OF_DOMAIN)


class Prediction(BaseModel):
    """A single property prediction with everything needed to judge it."""

    model_config = ConfigDict(frozen=True)

    property: str
    quantity: Quantity | None = None
    uncertainty: Uncertainty = Field(default_factory=Uncertainty.unknown)
    applicability: ApplicabilityDomain = Field(default_factory=ApplicabilityDomain)
    status: PredictionStatus = PredictionStatus.OK

    expert_id: str = ""
    expert_version: str = "0"
    #: Human-readable description of the method, e.g. "Joback (1987)".
    method: str = ""
    #: Conditions this prediction is valid at.
    conditions: Conditions = Field(default_factory=Conditions)
    provenance: ProvenanceRecord | None = None
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> "Prediction":
        get_property(self.property)  # raises for unregistered names
        if self.status.has_value:
            if self.quantity is None:
                raise ValueError(
                    f"Prediction of {self.property!r} has status {self.status.value} "
                    "but no quantity."
                )
            validate_unit_for(self.property, self.quantity.unit)
        elif self.quantity is not None:
            raise ValueError(
                f"Prediction of {self.property!r} has status {self.status.value} "
                "and must not carry a quantity."
            )
        return self

    @property
    def canonical(self) -> Quantity | None:
        """The predicted value in the property's canonical unit."""
        return None if self.quantity is None else self.quantity.to_canonical()

    @property
    def canonical_uncertainty(self) -> Uncertainty:
        """Uncertainty converted to the property's canonical unit."""
        if self.quantity is None:
            return self.uncertainty
        return self.uncertainty.converted(self.quantity.unit, get_property(self.property).canonical_unit)

    @property
    def is_usable(self) -> bool:
        """True when this prediction may participate in ranking.

        Out-of-domain predictions are usable but must be penalised, not
        discarded silently; unsupported and failed ones carry no value at all.
        """
        return self.status.has_value

    @classmethod
    def unsupported(cls, prop: str, expert_id: str, reason: str) -> "Prediction":
        return cls(
            property=prop,
            status=PredictionStatus.UNSUPPORTED,
            expert_id=expert_id,
            notes=(reason,),
            uncertainty=Uncertainty.unknown(reason),
        )

    @classmethod
    def failed(cls, prop: str, expert_id: str, reason: str) -> "Prediction":
        return cls(
            property=prop,
            status=PredictionStatus.FAILED,
            expert_id=expert_id,
            notes=(reason,),
            uncertainty=Uncertainty.unknown(reason),
        )

    def __str__(self) -> str:
        if not self.status.has_value:
            return f"{self.property}: {self.status.value} ({'; '.join(self.notes)})"
        flag = "" if self.applicability.in_domain else " [out of domain]"
        return f"{self.property}: {self.quantity} {self.uncertainty}{flag}"
