"""The property registry and the prediction record."""

from __future__ import annotations

import pytest

from formulate.core.conditions import Conditions
from formulate.core.errors import DimensionalityError, UnknownPropertyError
from formulate.core.prediction import Prediction, PredictionStatus
from formulate.core.properties import (
    PROPERTY_REGISTRY,
    PropertyFamily,
    get_property,
    properties_in_family,
    validate_unit_for,
)
from formulate.core.quantity import Quantity, Uncertainty, UncertaintyKind


def test_unknown_property_is_an_error_not_a_warning():
    with pytest.raises(UnknownPropertyError):
        get_property("vibes")


def test_unit_validation_accepts_any_compatible_unit():
    validate_unit_for("normal_boiling_point", "degC")
    validate_unit_for("normal_boiling_point", "K")
    with pytest.raises(DimensionalityError):
        validate_unit_for("normal_boiling_point", "Pa")


def test_every_registered_property_has_a_canonical_unit():
    for name, definition in PROPERTY_REGISTRY.items():
        assert definition.canonical_unit
        assert definition.name == name


def test_family_lookup():
    assert "normal_boiling_point" in properties_in_family(PropertyFamily.THERMAL)


def test_prediction_rejects_wrong_dimension():
    with pytest.raises(Exception):
        Prediction(
            property="normal_boiling_point", quantity=Quantity(value=1.0, unit="Pa")
        )


def test_failed_prediction_may_not_carry_a_value():
    with pytest.raises(Exception):
        Prediction(
            property="normal_boiling_point",
            status=PredictionStatus.FAILED,
            quantity=Quantity(value=300.0, unit="K"),
        )


def test_valued_status_requires_a_quantity():
    with pytest.raises(Exception):
        Prediction(property="normal_boiling_point", status=PredictionStatus.OK)


def test_absence_is_explicit_rather_than_a_neutral_value():
    """A prediction that could not be made carries no number at all."""
    unsupported = Prediction.unsupported("logp", "joback", "not covered")
    assert unsupported.quantity is None
    assert not unsupported.is_usable
    assert unsupported.status is PredictionStatus.UNSUPPORTED

    failed = Prediction.failed("logp", "crippen", "bad structure")
    assert failed.quantity is None
    assert not failed.is_usable


def test_canonical_accessors_convert_value_and_uncertainty_together():
    p = Prediction(
        property="normal_boiling_point",
        quantity=Quantity(value=100.0, unit="degC"),
        uncertainty=Uncertainty(std=5.0, kind=UncertaintyKind.EPISTEMIC),
        conditions=Conditions.standard(),
    )
    assert p.canonical.value == pytest.approx(373.15)
    assert p.canonical_uncertainty.std == pytest.approx(5.0)
