"""Dimensionless desirability functions.

Specification section 4: "Convert requested properties into dimensionless
utility/constraint functions.  Preserve raw physical predictions."

The shapes are Derringer-Suich desirability functions, the standard formulation
for multi-response optimisation.  A desirability is a pure function of a value
in canonical units; it never sees a candidate, an expert, or a rank, which is
what keeps normalisation auditable.

Two values are produced for every prediction: the desirability at the predicted
value, and a *risk-adjusted* desirability evaluated at a pessimistic bound
derived from the prediction's uncertainty.  Section 4 forbids letting an
uncertain prediction quietly score as if it were certain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Direction(str, Enum):
    """What "better" means for a property."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    #: Hit a value, with deviation either side penalised.
    TARGET = "target"
    #: Anywhere inside a band is equally acceptable.
    IN_RANGE = "in_range"


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@dataclass(frozen=True, slots=True)
class Desirability:
    """A dimensionless utility in [0, 1] over values in canonical units.

    ``lower`` and ``upper`` are the anchors at which desirability reaches its
    extremes.  ``target`` is required for :attr:`Direction.TARGET`.  ``shape``
    is the Derringer-Suich exponent: 1 is linear, >1 demands values close to
    the ideal before granting credit, <1 is lenient.
    """

    direction: Direction
    lower: float | None = None
    upper: float | None = None
    target: float | None = None
    shape: float = 1.0
    #: True when the anchors were derived from the candidate pool rather than
    #: stated by the user.  Pool-relative utilities are not comparable between
    #: runs and must be reported as such.
    pool_relative: bool = False

    def __post_init__(self) -> None:
        if self.shape <= 0:
            raise ValueError("Desirability shape exponent must be positive.")
        if self.direction is Direction.TARGET:
            if self.target is None or self.lower is None or self.upper is None:
                raise ValueError("TARGET desirability needs lower, target and upper.")
            if not (self.lower <= self.target <= self.upper):
                raise ValueError(
                    f"TARGET desirability requires lower <= target <= upper, got "
                    f"{self.lower} / {self.target} / {self.upper}."
                )
        elif self.direction is Direction.IN_RANGE:
            if self.lower is None and self.upper is None:
                raise ValueError("IN_RANGE desirability needs at least one bound.")
        else:
            if self.lower is None or self.upper is None:
                raise ValueError(
                    f"{self.direction.value} desirability needs both anchors "
                    "(the value scoring 0 and the value scoring 1)."
                )
            if math.isclose(self.lower, self.upper):
                raise ValueError("Desirability anchors must differ.")

    # -- evaluation --------------------------------------------------------

    def __call__(self, value: float) -> float:
        """Desirability of ``value``."""
        if math.isnan(value):
            raise ValueError("Desirability is undefined for NaN.")
        d = self.direction
        if d is Direction.IN_RANGE:
            if self.lower is not None and value < self.lower:
                return 0.0
            if self.upper is not None and value > self.upper:
                return 0.0
            return 1.0
        if d is Direction.MAXIMIZE:
            lo, hi = float(self.lower), float(self.upper)  # type: ignore[arg-type]
            return _clamp01((value - lo) / (hi - lo)) ** self.shape
        if d is Direction.MINIMIZE:
            lo, hi = float(self.lower), float(self.upper)  # type: ignore[arg-type]
            # `lower` is the ideal, `upper` the point of zero credit.
            return _clamp01((hi - value) / (hi - lo)) ** self.shape
        # TARGET
        lo, tgt, hi = float(self.lower), float(self.target), float(self.upper)  # type: ignore[arg-type]
        if value < lo or value > hi:
            return 0.0
        if math.isclose(value, tgt):
            return 1.0
        if value < tgt:
            return ((value - lo) / (tgt - lo)) ** self.shape if tgt > lo else 1.0
        return ((hi - value) / (hi - tgt)) ** self.shape if hi > tgt else 1.0

    def pessimistic_value(self, value: float, std: float, k: float = 1.0) -> float:
        """Shift ``value`` by ``k`` standard deviations in the unfavourable direction.

        For a two-sided target the unfavourable direction is *away* from the
        target, so the shift always reduces desirability rather than
        accidentally improving it.
        """
        if std <= 0.0:
            return value
        delta = k * std
        d = self.direction
        if d is Direction.MAXIMIZE:
            return value - delta
        if d is Direction.MINIMIZE:
            return value + delta
        if d is Direction.TARGET:
            tgt = float(self.target)  # type: ignore[arg-type]
            return value + delta if value >= tgt else value - delta
        # IN_RANGE: push toward whichever edge is nearer.
        lo, hi = self.lower, self.upper
        if lo is not None and hi is not None:
            mid = 0.5 * (lo + hi)
            return value + delta if value >= mid else value - delta
        if lo is not None:
            return value - delta
        return value + delta

    def evaluate(self, value: float, std: float | None = None, k: float = 1.0) -> tuple[float, float]:
        """Return ``(desirability, risk_adjusted_desirability)``.

        When no uncertainty is available the two are equal, and the caller is
        expected to record that the risk adjustment is uninformed rather than
        treat the prediction as certain.
        """
        nominal = self(value)
        if std is None or std <= 0.0:
            return nominal, nominal
        return nominal, self(self.pessimistic_value(value, std, k))

    def describe(self, unit: str = "") -> str:
        suffix = f" {unit}" if unit else ""
        d = self.direction
        if d is Direction.IN_RANGE:
            lo = "-inf" if self.lower is None else f"{self.lower:.6g}"
            hi = "+inf" if self.upper is None else f"{self.upper:.6g}"
            return f"in range [{lo}, {hi}]{suffix}"
        if d is Direction.TARGET:
            return (
                f"target {self.target:.6g}{suffix} "
                f"(zero credit outside [{self.lower:.6g}, {self.upper:.6g}])"
            )
        ideal, zero = (self.upper, self.lower) if d is Direction.MAXIMIZE else (self.lower, self.upper)
        note = " [pool-relative]" if self.pool_relative else ""
        return f"{d.value}: 1.0 at {ideal:.6g}{suffix}, 0.0 at {zero:.6g}{suffix}{note}"


def pool_anchors(values: list[float], direction: Direction, *, pad: float = 0.05) -> tuple[float, float]:
    """Derive MINIMIZE/MAXIMIZE anchors from the observed candidate pool.

    Used only when the request states no absolute anchors.  The resulting
    desirabilities are comparable *within* one run and not between runs, which
    is why :attr:`Desirability.pool_relative` records the fact.
    """
    finite = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    if not finite:
        raise ValueError("Cannot derive pool anchors from an empty set of values.")
    lo, hi = min(finite), max(finite)
    if math.isclose(lo, hi):
        # A degenerate pool: widen by a unit so the function stays well defined.
        span = abs(lo) * pad if lo else 1.0
        lo, hi = lo - span, hi + span
    else:
        span = (hi - lo) * pad
        lo, hi = lo - span, hi + span
    return (lo, hi) if direction is Direction.MINIMIZE else (lo, hi)
