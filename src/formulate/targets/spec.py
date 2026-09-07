"""Machine-readable target specifications.

Specification section 5: "Natural-language goals must become explicit
properties, units, conditions, objective direction, tolerance, priority, and
hard/soft status.  Ambiguous targets require assumptions to be surfaced rather
than hidden."

This module is the machine-readable end of that translation.  It contains no
language model: turning vague intent into a :class:`TargetSpec` is the job of
the reasoning layer, and this type is the contract it must produce.  Keeping
the two apart is what makes the scientific ranking reproducible.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from formulate.core.candidate import MaterialClass
from formulate.core.conditions import Conditions
from formulate.core.hashing import canonical_json
from formulate.core.properties import get_property, validate_unit_for
from formulate.core.quantity import Quantity
from formulate.core.units import convert

from .desirability import Desirability, Direction


class AssumptionSource(str, Enum):
    """How firmly an assumption is held."""

    #: The user said it.
    STATED = "stated"
    #: Inferred from context by the reasoning layer.
    INFERRED = "inferred"
    #: A system default applied because nothing was said.
    DEFAULT = "default"


class Assumption(BaseModel):
    """Something the system decided that the user did not say.

    Surfaced in every report.  An assumption that is not written down is a
    hidden modelling choice, which section 5 forbids.
    """

    model_config = ConfigDict(frozen=True)

    statement: str
    source: AssumptionSource = AssumptionSource.DEFAULT
    made_by: str = ""
    affects: tuple[str, ...] = ()

    def __str__(self) -> str:
        who = f" ({self.made_by})" if self.made_by else ""
        return f"[{self.source.value}]{who} {self.statement}"


class Requirement(BaseModel):
    """One requested property, fully specified.

    Anchors (``lower``/``upper``) define where desirability reaches 0 and 1.
    For MINIMIZE and MAXIMIZE they may be omitted, in which case the evaluation
    layer derives them from the candidate pool and marks the resulting utility
    pool-relative.
    """

    model_config = ConfigDict(frozen=True)

    property: str
    direction: Direction
    target: Quantity | None = None
    lower: Quantity | None = None
    upper: Quantity | None = None
    #: Relative importance among soft objectives. Ignored for hard constraints.
    weight: float = Field(default=1.0, gt=0.0)
    #: A hard requirement eliminates or heavily penalises a violating candidate;
    #: a soft one only reduces its utility (specification section 4).
    hard: bool = False
    #: Derringer-Suich exponent; >1 demands closeness before granting credit.
    shape: float = Field(default=1.0, gt=0.0)
    #: Conditions for this requirement; falls back to the spec-level conditions.
    conditions: Conditions | None = None
    rationale: str = ""

    @model_validator(mode="after")
    def _validate(self) -> "Requirement":
        get_property(self.property)  # raises for an unregistered name
        for field in ("target", "lower", "upper"):
            q: Quantity | None = getattr(self, field)
            if q is not None:
                validate_unit_for(self.property, q.unit)

        lo, hi = self.lower_canonical, self.upper_canonical
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(
                f"Requirement on {self.property!r} has lower ({lo}) above upper ({hi})."
            )

        if self.direction is Direction.TARGET:
            if self.target is None:
                raise ValueError(
                    f"Requirement on {self.property!r} with direction 'target' needs a target."
                )
            if lo is None or hi is None:
                raise ValueError(
                    f"Requirement on {self.property!r} with direction 'target' needs both "
                    "lower and upper bounds delimiting acceptable deviation."
                )
        elif self.direction is Direction.IN_RANGE:
            if lo is None and hi is None:
                raise ValueError(
                    f"Requirement on {self.property!r} with direction 'in_range' needs a bound."
                )
        if self.hard and self.direction in (Direction.MINIMIZE, Direction.MAXIMIZE):
            bound = hi if self.direction is Direction.MINIMIZE else lo
            if bound is None:
                raise ValueError(
                    f"Hard {self.direction.value} requirement on {self.property!r} needs the "
                    f"bound it must not cross "
                    f"({'upper' if self.direction is Direction.MINIMIZE else 'lower'})."
                )
        return self

    # -- canonical accessors ----------------------------------------------

    @property
    def canonical_unit(self) -> str:
        return get_property(self.property).canonical_unit

    def _canon(self, q: Quantity | None) -> float | None:
        return None if q is None else convert(q.value, q.unit, self.canonical_unit)

    @property
    def target_canonical(self) -> float | None:
        return self._canon(self.target)

    @property
    def lower_canonical(self) -> float | None:
        return self._canon(self.lower)

    @property
    def upper_canonical(self) -> float | None:
        return self._canon(self.upper)

    @property
    def needs_pool_anchors(self) -> bool:
        """True when desirability cannot be built without seeing the pool."""
        if self.direction in (Direction.TARGET, Direction.IN_RANGE):
            return False
        return self.lower_canonical is None or self.upper_canonical is None

    def desirability(
        self, pool_values: Iterable[float] | None = None
    ) -> Desirability:
        """Build the desirability function for this requirement.

        ``pool_values`` supplies observed values in canonical units, used only
        when the requirement omits anchors.
        """
        lo, hi, tgt = self.lower_canonical, self.upper_canonical, self.target_canonical

        if self.direction in (Direction.TARGET, Direction.IN_RANGE):
            return Desirability(
                direction=self.direction, lower=lo, upper=hi, target=tgt, shape=self.shape
            )

        if lo is None or hi is None:
            from .desirability import pool_anchors

            if pool_values is None:
                raise ValueError(
                    f"Requirement on {self.property!r} omits anchors and no candidate pool "
                    "was supplied to derive them from."
                )
            plo, phi = pool_anchors(list(pool_values))
            lo = lo if lo is not None else plo
            hi = hi if hi is not None else phi
            return Desirability(
                direction=self.direction, lower=lo, upper=hi, shape=self.shape,
                pool_relative=True,
            )
        return Desirability(direction=self.direction, lower=lo, upper=hi, shape=self.shape)

    def constraint_bound(self) -> tuple[float | None, float | None]:
        """The interval a candidate must lie in to satisfy this requirement.

        ``(None, None)`` when the requirement imposes no absolute bound.
        """
        lo, hi, tgt = self.lower_canonical, self.upper_canonical, self.target_canonical
        if self.direction is Direction.MINIMIZE:
            return (None, hi)
        if self.direction is Direction.MAXIMIZE:
            return (lo, None)
        if self.direction is Direction.IN_RANGE:
            return (lo, hi)
        return (lo, hi) if tgt is not None else (None, None)

    def describe_constraint(self) -> str:
        """The requirement clause alone, without the property name or hard/soft tag.

        Used inside a ConstraintViolation, whose own rendering already states
        both, so that the two do not read back doubled.
        """
        unit = self.canonical_unit
        try:
            shape = self.desirability().describe(unit)
        except ValueError:
            shape = f"{self.direction.value} (anchors from pool)"
        cond = f" at {self.conditions.describe()}" if self.conditions else ""
        return f"{shape}{cond}"

    def describe(self) -> str:
        kind = "hard" if self.hard else f"soft(w={self.weight:g})"
        return f"{self.property} [{kind}]: {self.describe_constraint()}"


class StructuralConstraints(BaseModel):
    """Constraints on structure rather than on a predicted property.

    Applied by the cheap validity filters of specification section 3, before
    any expert is called.
    """

    model_config = ConfigDict(frozen=True)

    allowed_elements: tuple[str, ...] = ()
    forbidden_elements: tuple[str, ...] = ()
    #: SMARTS patterns that disqualify a candidate.
    forbidden_smarts: tuple[str, ...] = ()
    #: SMARTS patterns at least one of which must be present.
    required_smarts: tuple[str, ...] = ()
    max_heavy_atoms: int | None = None
    min_heavy_atoms: int | None = None
    allow_charged: bool = False
    #: Formulation limits.
    max_components: int | None = None
    min_components: int | None = None

    @model_validator(mode="after")
    def _validate(self) -> "StructuralConstraints":
        overlap = set(self.allowed_elements) & set(self.forbidden_elements)
        if overlap:
            raise ValueError(f"Elements both allowed and forbidden: {sorted(overlap)}")
        if (
            self.max_heavy_atoms is not None
            and self.min_heavy_atoms is not None
            and self.max_heavy_atoms < self.min_heavy_atoms
        ):
            raise ValueError("max_heavy_atoms is below min_heavy_atoms.")
        return self


class TargetSpec(BaseModel):
    """A complete, machine-readable statement of desired behavior."""

    model_config = ConfigDict(frozen=True)

    name: str = "unnamed target"
    requirements: tuple[Requirement, ...]
    #: Default operating conditions for every requirement that states none.
    conditions: Conditions = Field(default_factory=Conditions.standard)
    #: Material classes the search may propose.
    material_classes: tuple[MaterialClass, ...] = (MaterialClass.MOLECULE,)
    structural: StructuralConstraints = Field(default_factory=StructuralConstraints)
    assumptions: tuple[Assumption, ...] = ()
    notes: str = ""

    @model_validator(mode="after")
    def _validate(self) -> "TargetSpec":
        if not self.requirements:
            raise ValueError("A TargetSpec needs at least one requirement.")
        seen: set[tuple[str, str]] = set()
        for req in self.requirements:
            # The canonical identity, not describe(): that renders conditions
            # for a human and omits the processing history, so two requirements
            # on one property at different cure schedules were rejected with a
            # message asserting their conditions were identical. It also
            # rendered units as written, which made the same condition spelled
            # two ways look like two conditions.
            conditions = req.conditions or self.conditions
            key = (req.property, canonical_json(conditions.identity_payload()))
            if key in seen:
                raise ValueError(
                    f"Duplicate requirement on {req.property!r} at identical conditions; "
                    "merge them into one."
                )
            seen.add(key)
        return self

    # -- accessors ---------------------------------------------------------

    @property
    def properties(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(r.property for r in self.requirements))

    @property
    def hard_requirements(self) -> tuple[Requirement, ...]:
        return tuple(r for r in self.requirements if r.hard)

    @property
    def soft_requirements(self) -> tuple[Requirement, ...]:
        return tuple(r for r in self.requirements if not r.hard)

    def conditions_for(self, req: Requirement) -> Conditions:
        """Effective conditions for ``req``: spec-level, overlaid by its own."""
        return self.conditions.merged_with(req.conditions) if req.conditions else self.conditions

    def requirement_for(self, prop: str) -> Requirement | None:
        for r in self.requirements:
            if r.property == prop:
                return r
        return None

    def normalized_weights(self) -> dict[str, float]:
        """Soft-objective weights normalised to sum to one.

        Used only for the scalar baseline of section 4; the Pareto frontier
        does not consult them.
        """
        soft = self.soft_requirements
        total = sum(r.weight for r in soft)
        if total <= 0:
            return {}
        return {r.property: r.weight / total for r in soft}

    def with_assumption(self, *assumptions: Assumption) -> "TargetSpec":
        return self.model_copy(update={"assumptions": self.assumptions + tuple(assumptions)})

    def describe(self) -> str:
        lines = [f"Target: {self.name}", f"Conditions: {self.conditions.describe()}"]
        lines.append(f"Material classes: {', '.join(c.value for c in self.material_classes)}")
        lines.append("Requirements:")
        lines.extend(f"  - {r.describe()}" for r in self.requirements)
        if self.assumptions:
            lines.append("Assumptions:")
            lines.extend(f"  - {a}" for a in self.assumptions)
        return "\n".join(lines)

    # -- serialisation -----------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TargetSpec":
        """Build a spec from a plain mapping (parsed JSON or YAML)."""
        return cls.model_validate(_expand_shorthand(data))

    @classmethod
    def from_file(cls, path: str | Path) -> "TargetSpec":
        """Load a spec from a ``.json``, ``.yaml`` or ``.yml`` file."""
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"{p} does not contain a target specification mapping.")
        return cls.from_dict(data)

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent, exclude_none=True)


def _q(value: Any, default_unit: str) -> Any:
    """Accept ``5``, ``"5 K"``, ``{"value": 5, "unit": "K"}`` for a quantity."""
    if value is None or isinstance(value, dict):
        return value
    if isinstance(value, (int, float)):
        return {"value": float(value), "unit": default_unit}
    if isinstance(value, str):
        parts = value.strip().split(maxsplit=1)
        try:
            magnitude = float(parts[0])
        except ValueError:
            raise ValueError(f"Cannot read {value!r} as a quantity.") from None
        unit = parts[1] if len(parts) > 1 else default_unit
        return {"value": magnitude, "unit": unit}
    return value


def _expand_shorthand(data: dict[str, Any]) -> dict[str, Any]:
    """Allow terse hand-written specs.

    Quantities may be written as bare numbers (interpreted in the property's
    canonical unit) or as "value unit" strings.  Conditions accept
    ``temperature: 25 degC``.
    """
    out = dict(data)

    reqs = []
    for raw in out.get("requirements", []):
        req = dict(raw)
        prop = req.get("property")
        unit = get_property(prop).canonical_unit if prop else "dimensionless"
        for field in ("target", "lower", "upper"):
            if field in req:
                req[field] = _q(req[field], unit)
        if "conditions" in req and isinstance(req["conditions"], dict):
            req["conditions"] = _expand_conditions(req["conditions"])
        reqs.append(req)
    if reqs:
        out["requirements"] = reqs

    if isinstance(out.get("conditions"), dict):
        out["conditions"] = _expand_conditions(out["conditions"])
    return out


def _expand_conditions(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    if "temperature" in out:
        out["temperature"] = _q(out["temperature"], "kelvin")
    if "pressure" in out:
        out["pressure"] = _q(out["pressure"], "pascal")
    for key in ("processing", "surfaces"):
        if key in out and isinstance(out[key], str):
            out[key] = [out[key]]
    return out
