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
from formulate.core.quantity import Quantity
from formulate.experts.base import PredictionRequest
from formulate.experts.registry import ExpertRegistry

from .engine import prefer

#: Reference fields mapped onto registry properties and their units.
REFERENCE_PROPERTIES = {
    "boiling_point_c": ("normal_boiling_point", "degC"),
    "melting_point_c": ("melting_point", "degC"),
    "density_25c": ("liquid_density", "g/cm^3"),
    "surface_tension_25c": ("surface_tension", "mN/m"),
    # Seven properties that produced numbers with nothing here to check them
    # against. The measurements were installed the whole time and simply never
    # pulled into the reference set, so the accuracy of the estimator behind
    # each was a citation to its original paper rather than anything this
    # repository had measured.
    "critical_temperature_k": ("critical_temperature", "K"),
    "critical_pressure_pa": ("critical_pressure", "Pa"),
    "critical_volume_m3_mol": ("critical_volume", "m^3/mol"),
    # Keyed to each compound's own boiling point rather than to a fixed
    # temperature, because that is the state the group method is defined at.
    "enthalpy_vaporization_tb_j_mol": (
        "enthalpy_vaporization", "J/mol", "boiling_point_c",
    ),
    "enthalpy_fusion_j_mol": ("enthalpy_fusion", "J/mol"),
    "heat_capacity_gas_298k_j_mol_k": ("heat_capacity_gas", "J/mol/K"),
    "logp": ("logp", ""),
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
    #: Compound -> why its prediction was not comparable with the reference.
    skipped_on_conditions: dict = field(default_factory=dict)
    #: Expert id -> how many of these comparisons it answered.
    answered_by: dict = field(default_factory=dict)

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

    #: Experts that answer from a compiled measurement rather than estimating.
    #: Comparing one of these against a reference table compiled from the same
    #: sources measures whether the database agrees with itself.
    LOOKUP_EXPERTS = frozenset({"measured"})

    @property
    def self_comparison(self) -> str:
        """Why this row is a consistency check rather than an accuracy figure.

        Empty when the property was answered by an estimating expert. The
        default `formulate calibrate` run reported a mean absolute error of
        exactly zero on the critical temperature over fifty compounds, and a
        boiling point good to 0.22 degrees, which read as extraordinary
        accuracy and were the compiled measurement being compared against the
        table it was compiled from.
        """
        lookups = {
            expert: count
            for expert, count in self.answered_by.items()
            if expert in self.LOOKUP_EXPERTS
        }
        if not lookups:
            return ""
        answered = sum(lookups.values())
        return (
            f"{answered} of {self.count} answered by a compiled measurement "
            f"({', '.join(sorted(lookups))}), so this is the database agreeing with "
            "itself rather than an estimator being tested; restrict to one expert to "
            "measure that expert"
        )

    def verdict(self) -> str:
        """A short judgement on whether the stated uncertainty is trustworthy."""
        if self.self_comparison:
            return "NOT AN ACCURACY MEASUREMENT - " + self.self_comparison
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
        elif self.self_comparison:
            lines.append(f"    -> {self.verdict()}")
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
    expert_id: str | None = None,
) -> dict[str, PropertyCalibration]:
    """Predict every reference property and compare against measured values.

    ``expert_id`` restricts the comparison to one expert. That matters because
    a lookup expert answering from a compiled measurement will reproduce the
    reference values almost exactly, which is correct behaviour but would
    quietly turn a regression guard on an estimating method into a test that
    the database agrees with itself. Guarding an estimator means asking that
    estimator.
    """
    conditions = conditions or Conditions.standard()
    wanted = {entry[0] for entry in REFERENCE_PROPERTIES.values()}
    # Pull in the dependencies the derived properties need.
    for _ in range(len(registry) + 1):
        for expert in registry:
            if expert.supported_properties & wanted:
                wanted |= expert.dependencies

    results = {
        entry[0]: PropertyCalibration(property=entry[0], unit=entry[1])
        for entry in REFERENCE_PROPERTIES.values()
    }

    for record in compounds:
        candidate = Candidate(
            material_class=MaterialClass.MOLECULE,
            molecule=MoleculeSpec(smiles=record["smiles"]),
            conditions=conditions,
            label=record.get("name", ""),
        )
        predictions = _predict(
            registry, candidate, frozenset(wanted), conditions, expert_id=expert_id
        )

        for field_name, entry in REFERENCE_PROPERTIES.items():
            prop, unit = entry[0], entry[1]
            #: Which record field carries the temperature this reference is
            #: measured at, when it is not the requested one.
            at_field = entry[2] if len(entry) > 2 else None
            measured = record.get(field_name)
            if measured is None:
                continue
            prediction = predictions.get(prop)
            if prediction is None or prediction.quantity is None:
                continue

            # A prediction stamped at conditions the reference is not measured
            # at is not comparable to it, and comparing anyway is silent. This
            # guard exists because that happened here: Joback's enthalpy of
            # vaporisation is defined at the normal boiling point, the
            # reference was tabulated at 298 K, and the comparison reported a
            # mean error of 7.3 kJ/mol and a one-sigma coverage of 7 per cent.
            # Referencing it at the boiling point instead gave 2.3 kJ/mol and
            # 49 per cent. Nothing was wrong with the expert.
            # A reference keyed to a compound's own boiling point and a
            # prediction made at that boiling point describe the same state,
            # even though the two numbers for where it lies differ - the
            # estimator's boiling point is itself an estimate. Comparing the
            # numbers would reject every such pair; comparing the states is
            # what was meant. The residual difference between the estimated and
            # measured boiling point is a real extra error and lands in the
            # property's own spread, where it belongs.
            mismatch = (
                "" if at_field is not None
                else _condition_mismatch(prediction, conditions.temperature_k)
            )
            if mismatch:
                results[prop].skipped_on_conditions[record.get("name", "")] = mismatch
                continue

            predicted = prediction.quantity.to(unit).value
            reference = Quantity(value=float(measured), unit=unit).value
            error = predicted - reference

            entry = results[prop]
            entry.count += 1
            entry.answered_by[prediction.expert_id] = (
                entry.answered_by.get(prediction.expert_id, 0) + 1
            )
            entry.errors.append(error)
            std = prediction.uncertainty.converted(prediction.quantity.unit, unit).std
            entry.stated_std.append(std if std is not None else 0.0)
            if not prediction.applicability.in_domain:
                entry.out_of_domain += 1
            if entry.worst is None or abs(error) > abs(entry.worst[1]):
                entry.worst = (record.get("name", record["smiles"]), error)

    return {prop: entry for prop, entry in results.items() if entry.count}


def _condition_mismatch(prediction, expected_k: float | None) -> str:
    """Why this prediction cannot be compared with a reference at ``requested``.

    Only temperature is checked, because that is what the reference values are
    keyed on and what silently differs. A prediction that states no temperature
    is taken at its word rather than assumed to disagree.
    """
    stated = prediction.conditions.temperature_k if prediction.conditions else None
    wanted = expected_k
    if stated is None or wanted is None:
        return ""
    if abs(stated - wanted) <= 1.0:
        return ""
    return (
        f"predicted at {stated:.1f} K against a reference measured at {wanted:.1f} K"
    )


def _predict(
    registry: ExpertRegistry,
    candidate: Candidate,
    wanted: frozenset[str],
    conditions: Conditions,
    expert_id: str | None = None,
):
    """Run the panel over one candidate, honouring expert dependencies."""
    experts = registry.resolution_order(
        registry.experts_for(wanted, candidate.material_class)
    )
    context: dict = {}
    for expert in experts:
        keep = expert_id is None or expert.id == expert_id
        request = PredictionRequest(
            candidate=candidate, properties=wanted, conditions=conditions, context=dict(context)
        )
        for prediction in expert.predict(request):
            if not prediction.is_usable:
                continue
            # Upstream experts still run, because a dependent expert needs
            # their values; they just do not get to answer for the property
            # under test when one expert was named.
            if not keep and prediction.property in _REFERENCE_TARGETS:
                context.setdefault("__" + prediction.property, prediction)
                continue
            incumbent = context.get(prediction.property)
            if incumbent is None or prefer(prediction, incumbent):
                context[prediction.property] = prediction
    return context


#: Properties the calibration compares; only these are filtered by expert.
_REFERENCE_TARGETS = frozenset(entry[0] for entry in REFERENCE_PROPERTIES.values())


def describe(results: dict[str, PropertyCalibration]) -> str:
    lines = [
        "Phase 1 calibration against the bundled reference compounds.",
        "",
        "These are commonly tabulated handbook values, not a curated benchmark. Treat the",
        "numbers below as a regression guard on the expert panel, not as a scientific",
        "evaluation of the underlying methods.",
        "",
    ]
    tautological = [name for name, entry in results.items() if entry.self_comparison]
    if tautological:
        lines.extend(
            [
                "SOME ROWS BELOW MEASURE NOTHING. " + ", ".join(sorted(tautological)) + " "
                "were answered by an expert that looks the value up rather than",
                "estimating it, against a reference table compiled from the same sources.",
                "Those rows are consistency checks; pass --expert to measure an estimator.",
                "",
            ]
        )
    for entry in results.values():
        lines.append(entry.describe())
        lines.append("")
    return "\n".join(lines).rstrip()
