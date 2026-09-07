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


def test_the_same_conditions_written_two_ways_are_one_candidate():
    """Content addressing has to canonicalise units, or it is address-by-spelling."""
    from formulate.core.candidate import molecule_candidate
    from formulate.core.conditions import Conditions
    from formulate.core.quantity import Quantity

    celsius = molecule_candidate(
        "CCO",
        conditions=Conditions(
            temperature=Quantity(value=25.0, unit="degC"),
            pressure=Quantity(value=1.0, unit="atm"),
        ),
    )
    kelvin = molecule_candidate(
        "CCO",
        conditions=Conditions(
            temperature=Quantity(value=298.15, unit="K"),
            pressure=Quantity(value=101325.0, unit="Pa"),
        ),
    )
    assert celsius.candidate_id == kelvin.candidate_id

    warmer = molecule_candidate(
        "CCO", conditions=Conditions(temperature=Quantity(value=50.0, unit="degC"))
    )
    assert celsius.candidate_id != warmer.candidate_id


def test_a_cache_entry_is_shared_between_two_spellings_of_one_condition():
    from formulate.core.conditions import Conditions
    from formulate.core.quantity import Quantity
    from formulate.store.cache import PredictionCache

    celsius = Conditions(temperature=Quantity(value=25.0, unit="degC"))
    kelvin = Conditions(temperature=Quantity(value=298.15, unit="K"))
    assert PredictionCache.key("c", "e", "1", celsius, ["logp"]) == PredictionCache.key(
        "c", "e", "1", kelvin, ["logp"]
    )


def test_a_cure_schedule_keeps_its_order_and_a_surface_list_does_not():
    from formulate.core.conditions import Conditions

    forward = Conditions(processing=("dry 40C", "cure 120C"))
    backward = Conditions(processing=("cure 120C", "dry 40C"))
    # Two different histories: one bakes before curing, the other after.
    assert forward.identity_payload() != backward.identity_payload()

    one = Conditions(surfaces=("aluminium-oxide", "glass"))
    other = Conditions(surfaces=("glass", "aluminium-oxide"))
    # The same two surfaces are present either way.
    assert one.identity_payload() == other.identity_payload()


def test_a_backend_upgrade_changes_a_provenance_id_but_a_new_machine_does_not():
    """Provenance identity tracks the method, not the host that ran it."""
    from formulate.core.provenance import (
        ProvenanceKind,
        ProvenanceRecord,
        SoftwareEnvironment,
    )

    def record(**software):
        return ProvenanceRecord(
            kind=ProvenanceKind.PREDICTION,
            producer="joback",
            producer_version="1",
            parameters={"property": "normal_boiling_point"},
            software=SoftwareEnvironment(formulate_version="0.1", **software),
        )

    # RDKit's conformer generation and PySCF's defaults both move between
    # releases, so a value computed against a different one is a different
    # value and must not share a provenance id.
    assert record(backends={"rdkit": "2026.03.6"}).record_id != record(
        backends={"rdkit": "2024.09.1"}
    ).record_id

    # The operating system and the interpreter patch level are properties of
    # the machine, and re-running tomorrow elsewhere must still match.
    assert (
        record(backends={"rdkit": "1"}, platform="Linux-x", python_version="3.11.9").record_id
        == record(backends={"rdkit": "1"}, platform="Darwin-y", python_version="3.12.1").record_id
    )
