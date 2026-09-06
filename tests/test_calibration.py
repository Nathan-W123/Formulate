"""Phase 1 calibration: accuracy, and whether stated uncertainty is honest.

These are regression guards on the expert panel, not a scientific benchmark.
The reference values are commonly tabulated handbook figures; the thresholds
are set loosely enough that ordinary variation does not fail the suite, and
tightly enough that a genuine regression does.
"""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.evaluation.calibration import calibrate, describe
from formulate.exploration.database import load_reference_compounds

pytestmark = requires_rdkit


@pytest.fixture(scope="module")
def results(request):
    from formulate.experts import default_registry

    return calibrate(default_registry(), load_reference_compounds())


def test_every_calibrated_property_has_a_useful_sample(results):
    for name in ("normal_boiling_point", "melting_point", "liquid_density", "surface_tension"):
        assert name in results, f"{name} produced no predictions at all"
        assert results[name].count >= 20


def test_boiling_point_accuracy_matches_the_published_method_error(results):
    """Joback reports about 12.9 K; anything far worse means something broke."""
    assert results["normal_boiling_point"].mean_absolute_error < 25.0


def test_melting_point_is_poor_but_not_unbounded(results):
    """Joback melting points are known to be weak; this guards the magnitude."""
    assert results["melting_point"].mean_absolute_error < 45.0


def test_liquid_density_accuracy(results):
    assert results["liquid_density"].mean_absolute_error < 0.10


def test_surface_tension_accuracy(results):
    assert results["surface_tension"].mean_absolute_error < 9.0


@pytest.mark.parametrize(
    "prop",
    ["normal_boiling_point", "melting_point", "liquid_density", "surface_tension"],
)
def test_stated_uncertainty_is_not_overconfident(results, prop):
    """A one-sigma bound that catches far fewer than 68% is lying to the ranker."""
    coverage = results[prop].within_one_sigma
    assert coverage is not None
    assert coverage >= 0.5, f"{prop} claims a tighter error bar than it earns"


@pytest.mark.parametrize(
    "prop",
    ["normal_boiling_point", "melting_point", "liquid_density", "surface_tension"],
)
def test_two_sigma_coverage_is_high(results, prop):
    assert results[prop].within_two_sigma >= 0.80


@pytest.mark.parametrize(
    "prop",
    ["normal_boiling_point", "melting_point", "liquid_density", "surface_tension"],
)
def test_no_property_is_reported_as_overconfident(results, prop):
    assert "OVERCONFIDENT" not in results[prop].verdict()


def test_the_worst_cases_are_the_associating_compounds(results):
    """Errors should concentrate where the domain warnings say they will."""
    worst = results["surface_tension"].worst
    assert worst is not None
    assert worst[0] in {"glycerol", "ethylene glycol", "acetic acid", "water"}


def test_description_states_that_this_is_not_a_benchmark(results):
    assert "not a curated benchmark" in describe(results)
