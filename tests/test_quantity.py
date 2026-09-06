"""Quantities, uncertainty and applicability domains."""

from __future__ import annotations

import pytest

from formulate.core.errors import DimensionalityError
from formulate.core.quantity import (
    ApplicabilityDomain,
    Quantity,
    Uncertainty,
    UncertaintyKind,
)


def test_quantity_converts_and_canonicalises():
    q = Quantity(value=25.0, unit="degC")
    assert q.to("K").value == pytest.approx(298.15)
    assert q.to_canonical().unit == "kelvin"
    assert Quantity(value=31.3, unit="kJ/mol").to_canonical().value == pytest.approx(31300.0)


def test_quantity_is_frozen():
    q = Quantity(value=1.0, unit="K")
    with pytest.raises(Exception):
        q.value = 2.0  # type: ignore[misc]


def test_uncertainty_rejects_negative_spread():
    with pytest.raises(Exception):
        Uncertainty(std=-1.0)


def test_uncertainty_conversion_preserves_spread_across_offset_units():
    u = Uncertainty(std=12.9, kind=UncertaintyKind.EPISTEMIC)
    assert u.converted("K", "degC").std == pytest.approx(12.9)


def test_uncertainty_conversion_rejects_wrong_dimension():
    with pytest.raises(DimensionalityError):
        Uncertainty(std=1.0).converted("K", "Pa")


def test_unknown_uncertainty_is_not_informative():
    assert not Uncertainty.unknown("no model").is_informative
    assert Uncertainty(std=1.0).is_informative


def test_applicability_merge_is_conservative():
    a = ApplicabilityDomain.outside("bad", score=0.2)
    b = ApplicabilityDomain(score=0.9)
    merged = a.merged_with(b)
    assert merged.score == pytest.approx(0.2)
    assert merged.in_domain is False
    assert "bad" in merged.warnings
