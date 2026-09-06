"""Unit registry: the invariant that nothing is ranked before dimensions agree."""

from __future__ import annotations

import pytest

from formulate.core import units
from formulate.core.errors import DimensionalityError, UnitError


def test_canonical_units_are_si_for_registered_dimensions():
    assert units.canonical_unit("degC") == "kelvin"
    assert units.canonical_unit("bar") == "pascal"
    assert units.canonical_unit("kJ/mol") == "joule / mole"
    assert units.canonical_unit("g/cm^3") == "kilogram / meter ** 3"
    assert units.canonical_unit("mN/m") == "newton / meter"
    assert units.canonical_unit("MPa^0.5") == "pascal ** 0.5"
    assert units.canonical_unit("") == "dimensionless"


def test_offset_temperature_converts_by_offset_not_scale():
    assert units.convert(25.0, "degC", "K") == pytest.approx(298.15)
    assert units.convert(212.0, "degF", "degC") == pytest.approx(100.0)


def test_uncertainty_conversion_uses_difference_semantics():
    """A 12.9 K spread is 12.9 degrees Celsius, not -260."""
    assert units.convert_delta(12.9, "K", "degC") == pytest.approx(12.9)
    # Fahrenheit degrees are smaller, so the number changes but the offset does not apply.
    assert units.convert_delta(1.0, "K", "degF") == pytest.approx(1.8)


def test_incompatible_dimensions_are_rejected():
    with pytest.raises(DimensionalityError):
        units.convert(1.0, "K", "Pa")
    with pytest.raises(DimensionalityError):
        units.convert_delta(1.0, "K", "Pa")


def test_unparseable_unit_raises():
    with pytest.raises(UnitError):
        units.parse_unit("not-a-unit-@@")


def test_compatibility_check():
    assert units.are_compatible("degC", "K")
    assert not units.are_compatible("degC", "Pa")


def test_offset_unit_detection():
    assert units.is_offset_unit("degC")
    assert not units.is_offset_unit("K")
