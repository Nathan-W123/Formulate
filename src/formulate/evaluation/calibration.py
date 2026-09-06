"""Calibration of experts against measured values.

Specification section 10, Phase 1: "Verify rankings against held-out known
materials."  Section 12 lists "property prediction error/calibration on
held-out materials" as the first success metric.

Two distinct things are measured here, and conflating them is a common error:

* **Accuracy** - how far predictions land from measured values.
* **Calibration** - whether the stated uncertainty is honest.  A model with a
  3 K error that claims 1 K is worse for decision-making than one with a 15 K
  error that claims 15 K, because the first will silently pass candidates that
  do not belong.  For a correct one-sigma estimate roughly 68% of predictions
  should fall inside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from formulate.core.candidate import Candidate, MaterialClass, MoleculeSpec
from formulate.core.conditions import Conditions
from formulate.core.properties import get_property
from formulate.core.quantity import Quantity
from formulate.experts.base import PredictionRequest
from formulate.experts.registry import ExpertRegistry

#: Reference fields mapped onto registry properties and their units.
REFERENCE_PROPERTIES = {
    "boiling_point_c": ("normal_boiling_point", "degC"),
    "melting_point_c": ("melting_point", "degC"),
    "density_25c": ("liquid_density", "g/cm^3"),
    "surface_tension_25c": ("surface_tension", "mN/m"),
}


@dataclass
class PropertyCalibration:
    """Accuracy and calibration for one property across a reference set."""

    property: str
    unit: str
    count: int = 0
    errors: list[float] = field(default_factory=list)
    stated_std: list[float] = field(default_factory=list)
    worst: tuple[str, float] | None = None
    out_of_domain: int = 0

    @property
    def mean_absolute_error(self) -> float:
        return sum(abs(e) for e in self.errors) / len(self.errors) if self.errors else math.nan

    @property
    def bias(self) -> float:
        """Mean signed error; a large value means a systematic offset."""
        return sum(self.errors) / len(self.errors) if self.errors else math.nan

    @property
    def rmse(self) -> float:
        if not self.errors:
            return math.nan
        return math.sqrt(sum(e * e for e in self.errors) / len(self.errors))

    @property
    def within_one_sigma(self) -> float | None:
        """Fraction of predictions inside their own stated one-sigma bound."""
        pairs = [(e, s) for e, s in zip(self.errors, self.stated_std) if s and s > 0]
        if not pairs:
            return None
        return sum(1 for e, s in pairs if abs(e) <= s) / len(pairs)

    @property
    def within_two_sigma(self) -> float | None:
        pairs = [(e, s) for e, s in zip(self.errors, self.stated_std) if s and s > 0]
        if not pairs:
            return None
        return sum(1 for e, s in pairs if abs(e) <= 2 * s) / len(pairs)

    def verdict(self) -> str:
        """A short judgement on whether the stated uncertainty is trustworthy."""
        coverage = self.within_one_sigma
        if coverage is None:
            return "no stated uncertainty to assess"
        if coverage < 0.45:
            return "OVERCONFIDENT - the stated uncertainty is too small"
        if coverage > 0.90:
            return "conservative - the stated uncertainty is wider than observed error"
        return "reasonable - close to the ~68% a correct one-sigma estimate implies"

    def describe(self) -> str:
        lines = [
            f"{self.property} ({self.unit}), n = {self.count}",
            f"    mean absolute error {self.mean_absolute_error:.4g}"
            f"   RMSE {self.rmse:.4g}   bias {self.bias:+.4g}",
        ]
        coverage = self.within_one_sigma
        if coverage is not None:
            lines.append(
                f"    within 1 sigma {coverage:.0%}   within 2 sigma "
                f"{self.within_two_sigma:.0%}   -> {self.verdict()}"
            )
        if self.worst is not None:
            lines.append(f"    worst case {self.worst[0]} off by {abs(self.worst[1]):.4g}")
        if self.out_of_domain:
            lines.append(f"    {self.out_of_domain} prediction(s) flagged out of domain")
        return "\n".join(lines)


def calibrate(
    registry: ExpertRegistry,
    compounds: Sequence[dict],
    *,
    conditions: Conditions | None = None,
) -> dict[str, PropertyCalibration]:
    """Predict every reference property and compare against measured values."""
    conditions = conditions or Conditions.standard()
    wanted = {prop for prop, _ in REFERENCE_PROPERTIES.values()}
    # Pull in the dependencies the derived properties need.
    for _ in range(len(registry) + 1):
        for expert in registry:
            if expert.supported_properties & wanted:
                wanted |= expert.dependencies

    results = {
        prop: PropertyCalibration(property=prop, unit=unit)
        for prop, unit in REFERENCE_PROPERTIES.values()
    }

    for record in compounds:
        candidate = Candidate(
            material_class=MaterialClass.MOLECULE,
            molecule=MoleculeSpec(smiles=record["smiles"]),
            conditions=conditions,
            label=record.get("name", ""),
        )
        predictions = _predict(registry, candidate, frozenset(wanted), conditions)

        for field_name, (prop, unit) in REFERENCE_PROPERTIES.items():
            measured = record.get(field_name)
            if measured is None:
                continue
            prediction = predictions.get(prop)
            if prediction is None or prediction.quantity is None:
                continue

            predicted = prediction.quantity.to(unit).value
            reference = Quantity(value=float(measured), unit=unit).value
            error = predicted - reference

            entry = results[prop]
            entry.count += 1
            entry.errors.append(error)
            std = prediction.uncertainty.converted(prediction.quantity.unit, unit).std
            entry.stated_std.append(std if std is not None else 0.0)
            if not prediction.applicability.in_domain:
                entry.out_of_domain += 1
            if entry.worst is None or abs(error) > abs(entry.worst[1]):
                entry.worst = (record.get("name", record["smiles"]), error)

    return {prop: entry for prop, entry in results.items() if entry.count}


def _predict(
    registry: ExpertRegistry,
    candidate: Candidate,
    wanted: frozenset[str],
    conditions: Conditions,
):
    """Run the panel over one candidate, honouring expert dependencies."""
    experts = registry.resolution_order(
        registry.experts_for(wanted, candidate.material_class)
    )
    context: dict = {}
    for expert in experts:
        request = PredictionRequest(
            candidate=candidate, properties=wanted, conditions=conditions, context=dict(context)
        )
        for prediction in expert.predict(request):
            if prediction.is_usable:
                incumbent = context.get(prediction.property)
                if incumbent is None or (
                    prediction.applicability.in_domain and not incumbent.applicability.in_domain
                ):
                    context[prediction.property] = prediction
    return context


def describe(results: dict[str, PropertyCalibration]) -> str:
    lines = [
        "Phase 1 calibration against the bundled reference compounds.",
        "",
        "These are commonly tabulated handbook values, not a curated benchmark. Treat the",
        "numbers below as a regression guard on the expert panel, not as a scientific",
        "evaluation of the underlying methods.",
        "",
    ]
    for entry in results.values():
        lines.append(entry.describe())
        lines.append("")
    return "\n".join(lines).rstrip()
