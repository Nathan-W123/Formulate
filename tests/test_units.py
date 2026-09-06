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


@pytest.mark.parametrize(
    "spelling_a,spelling_b",
    [
        ("debye", "coulomb * meter"),
        ("debye", "elementary_charge * angstrom"),
        ("kJ/mol", "J/mol"),
        ("bar", "Pa"),
        ("degC", "K"),
        ("mN/m", "N/m"),
        ("cP", "Pa*s"),
        ("eV", "J"),
        ("g/cm^3", "kg/m^3"),
        ("MPa^0.5", "Pa^0.5"),
        ("cm^3/mol", "m^3/mol"),
    ],
)
def test_equivalent_spellings_share_one_dimension_key(spelling_a, spelling_b):
    """pint's dimensionality repr is not order-stable across spellings.

    Keying anything on the raw repr made are_compatible("debye",
    "coulomb * meter") return False and rejected a physically valid unit.
    The key must be canonically ordered so equivalent spellings agree, and
    must resolve to the same canonical unit.
    """
    assert units.dimensionality(spelling_a) == units.dimensionality(spelling_b)
    assert units.are_compatible(spelling_a, spelling_b)
    assert units.canonical_unit(spelling_a) == units.canonical_unit(spelling_b)


def test_genuinely_different_dimensions_stay_incompatible():
    for a, b in [("K", "Pa"), ("J", "J/mol"), ("N/m", "Pa"), ("m^2/s", "m^3/mol")]:
        assert not units.are_compatible(a, b), f"{a} and {b} must not be compatible"


def test_a_unit_stated_in_any_valid_spelling_is_accepted_for_its_property():
    from formulate.core.properties import validate_unit_for

    validate_unit_for("dipole_moment", "coulomb * meter")
    validate_unit_for("dipole_moment", "debye")
    validate_unit_for("homo_lumo_gap", "J")
    validate_unit_for("shear_viscosity", "cP")
