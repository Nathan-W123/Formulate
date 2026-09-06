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
    """Joback only.

    The panel now includes a lookup expert that answers from compiled
    measurements, and it correctly wins for every reference compound. That is
    the right behaviour and it would also make this suite meaningless: a
    regression guard on a group-contribution estimator has to ask that
    estimator, not the database the references came from.
    """
    from formulate.experts import default_registry

    return calibrate(default_registry(), load_reference_compounds(), expert_id="joback")


@pytest.fixture(scope="module")
def panel_results():
    """The whole panel, measurement included, as a user would see it."""
    from formulate.experts import default_registry

    return calibrate(default_registry(), load_reference_compounds())


def test_every_calibrated_property_has_a_useful_sample(panel_results):
    for name in ("normal_boiling_point", "melting_point", "liquid_density", "surface_tension"):
        assert name in panel_results, f"{name} produced no predictions at all"
        assert panel_results[name].count >= 20


def test_a_measured_value_beats_the_estimate_and_the_panel_uses_it(
    results, panel_results
):
    """The engine prefers the tighter in-domain prediction with no special case."""
    for prop in ("normal_boiling_point", "melting_point"):
        assert panel_results[prop].mean_absolute_error < results[prop].mean_absolute_error
        assert panel_results[prop].mean_absolute_error < 3.0


def test_boiling_point_accuracy_matches_the_published_method_error(results):
    """Joback reports about 12.9 K; anything far worse means something broke."""
    assert results["normal_boiling_point"].mean_absolute_error < 25.0


def test_melting_point_is_poor_but_not_unbounded(results):
    """Joback melting points are known to be weak; this guards the magnitude."""
    assert results["melting_point"].mean_absolute_error < 45.0


def test_liquid_density_accuracy(panel_results):
    assert panel_results["liquid_density"].mean_absolute_error < 0.10


def test_surface_tension_accuracy(panel_results):
    assert panel_results["surface_tension"].mean_absolute_error < 9.0


@pytest.mark.parametrize(
    "prop",
    ["normal_boiling_point", "melting_point", "liquid_density", "surface_tension"],
)
def test_stated_uncertainty_is_not_overconfident(panel_results, prop):
    """A one-sigma bound that catches far fewer than 68% is lying to the ranker."""
    coverage = panel_results[prop].within_one_sigma
    assert coverage is not None
    assert coverage >= 0.5, f"{prop} claims a tighter error bar than it earns"


@pytest.mark.parametrize(
    "prop",
    ["normal_boiling_point", "melting_point", "liquid_density", "surface_tension"],
)
def test_two_sigma_coverage_is_high(panel_results, prop):
    assert panel_results[prop].within_two_sigma >= 0.80


@pytest.mark.parametrize(
    "prop",
    ["normal_boiling_point", "melting_point", "liquid_density", "surface_tension"],
)
def test_no_property_is_reported_as_overconfident(panel_results, prop):
    assert "OVERCONFIDENT" not in panel_results[prop].verdict()


def test_the_worst_cases_are_the_associating_compounds(panel_results):
    """Errors should concentrate where the domain warnings say they will."""
    worst = panel_results["surface_tension"].worst
    assert worst is not None
    assert worst[0] in {"glycerol", "ethylene glycol", "acetic acid", "water"}


def test_description_states_that_this_is_not_a_benchmark(panel_results):
    assert "not a curated benchmark" in describe(panel_results)
