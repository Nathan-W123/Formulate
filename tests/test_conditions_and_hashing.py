"""Condition matching and content addressing."""

from __future__ import annotations

import pytest

from formulate.core.conditions import Conditions, Phase, match_conditions
from formulate.core.hashing import canonical_json, cache_key, content_hash
from formulate.core.quantity import Quantity


def test_standard_conditions():
    c = Conditions.standard()
    assert c.temperature_k == pytest.approx(298.15)
    assert c.pressure_pa == pytest.approx(101325.0)


def test_conditions_reject_wrong_dimension():
    with pytest.raises(Exception):
        Conditions(temperature=Quantity(value=1.0, unit="Pa"))
    with pytest.raises(Exception):
        Conditions(pressure=Quantity(value=1.0, unit="K"))


def test_temperature_mismatch_is_incompatible():
    required = Conditions(temperature=Quantity(value=150.0, unit="degC"))
    available = Conditions.standard()
    match = match_conditions(required, available)
    assert not match.compatible
    assert "temperature mismatch" in match.issues[0]


def test_temperature_within_tolerance_is_compatible():
    required = Conditions(temperature=Quantity(value=298.15, unit="K"))
    available = Conditions(temperature=Quantity(value=300.0, unit="K"))
    assert match_conditions(required, available).compatible


def test_unstated_condition_is_reported_not_assumed():
    """An unstated condition must never be silently treated as standard."""
    match = match_conditions(Conditions(environment="water"), Conditions.standard())
    assert match.compatible
    assert match.unstated
    assert not match.exact


def test_phase_mismatch_is_incompatible():
    match = match_conditions(
        Conditions(phase=Phase.LIQUID), Conditions(phase=Phase.GAS)
    )
    assert not match.compatible


def test_merge_prefers_the_overlay():
    base = Conditions.standard()
    overlay = Conditions(temperature=Quantity(value=400.0, unit="K"))
    merged = base.merged_with(overlay)
    assert merged.temperature_k == pytest.approx(400.0)
    assert merged.pressure_pa == pytest.approx(101325.0)


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


def test_content_hash_is_stable_and_namespaced():
    payload = {"x": 1.0, "y": [1, 2, 3]}
    assert content_hash(payload) == content_hash(dict(payload))
    assert content_hash(payload, prefix="a") != content_hash(payload, prefix="b")


def test_float_precision_absorbs_last_bit_drift():
    assert content_hash({"v": 0.1 + 0.2}) == content_hash({"v": 0.30000000000000004})


def test_cache_key_changes_with_method_version():
    conditions = Conditions.standard()
    first = cache_key(
        candidate_hash="c", method="joback", method_version="1", conditions=conditions
    )
    second = cache_key(
        candidate_hash="c", method="joback", method_version="2", conditions=conditions
    )
    assert first != second


def test_cache_key_changes_with_conditions():
    a = cache_key(
        candidate_hash="c", method="m", method_version="1", conditions=Conditions.standard()
    )
    b = cache_key(
        candidate_hash="c",
        method="m",
        method_version="1",
        conditions=Conditions(temperature=Quantity(value=400.0, unit="K")),
    )
    assert a != b
