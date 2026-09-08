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
from typing import Sequence

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


# --------------------------------------------------------------------------
# Choosing between two predictions of the same property
# --------------------------------------------------------------------------


def canonical_std(prediction: Prediction) -> float | None:
    """One-sigma spread in the property's canonical unit, or None.

    An ``Uncertainty`` is expressed in the same unit as the value it
    accompanies, and an expert may report in whatever unit reads best.  Two
    spreads are therefore only comparable once both are in the same unit, and
    comparing the raw floats would rank a density expert reporting 0.05 g/cm^3
    ahead of one reporting 40 kg/m^3 - which is the tighter of the two by a
    factor of twenty in the wrong direction.
    """
    if prediction.quantity is None or prediction.uncertainty.std is None:
        return None
    unit = get_property(prediction.property).canonical_unit
    return prediction.uncertainty.converted(prediction.quantity.unit, unit).std


def comparable_spread(prediction: Prediction) -> float | None:
    """The spread of a prediction, on the scale its property's error lives on.

    Absolute for most properties, relative for the ones the registry marks as
    carrying multiplicative error. Converting a unit is not enough to make two
    spreads comparable: a property that runs over orders of magnitude also
    needs the right *kind* of spread, or the comparison silently rewards
    whichever expert predicted the smallest number, since a small prediction
    carries a small absolute error whether or not it is any good.

    See ``PropertyDef.multiplicative_error`` for the case that found this - a
    viscosity that was a factor of twenty-four low and 85 per cent uncertain
    beating one that was 31 per cent uncertain, on absolute spread.
    """
    std = canonical_std(prediction)
    if std is None:
        return None
    if not get_property(prediction.property).multiplicative_error:
        return std
    unit = get_property(prediction.property).canonical_unit
    value = abs(prediction.quantity.to(unit).value)
    if value == 0.0:
        # A zero prediction has no scale to be relative to, and dividing by it
        # would make an unfalsifiable answer look infinitely certain.
        return float("inf")
    return std / value


def prefer(candidate: Prediction, incumbent: Prediction) -> bool:
    """True when ``candidate`` should displace ``incumbent``.

    The single authority on which of two predictions for the same property is
    better: in-domain first, then applicability, then the tighter spread - on
    the scale that property's error lives on, which is relative rather than
    absolute for the ones that run over orders of magnitude.  It
    lives beside :class:`Prediction` rather than in the evaluation layer
    because three places need it - dispatch, scoring and the results record -
    and the lowest of those is ``core``.  A second copy of the rule would let
    two parts of the system disagree about which prediction the user is
    actually being shown, and both the calibration report and the scoring pool
    have done exactly that: they sorted on applicability alone, which is a tie
    between a measured value and a group-contribution estimate, and then took
    whichever happened to come first.

    Ties keep the incumbent, so a fold over predictions in dispatch order is
    stable.
    """
    if candidate.applicability.in_domain != incumbent.applicability.in_domain:
        return candidate.applicability.in_domain
    if candidate.applicability.score != incumbent.applicability.score:
        return candidate.applicability.score > incumbent.applicability.score
    a = comparable_spread(candidate)
    b = comparable_spread(incumbent)
    if a is not None and b is not None:
        return a < b
    # A stated spread beats an unstated one: an expert that admits how wrong it
    # might be has said more than one that did not.
    return a is not None and b is None


def best_prediction(
    predictions: "Sequence[Prediction]", prop: str
) -> Prediction | None:
    """The usable prediction for ``prop`` that :func:`prefer` ranks highest."""
    best: Prediction | None = None
    for prediction in predictions:
        if prediction.property != prop or not prediction.is_usable:
            continue
        if best is None or prefer(prediction, best):
            best = prediction
    return best
