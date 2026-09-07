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
    """Was 0.064 g/cm^3 from a corresponding-states correlation alone.

    The correlation needs a critical temperature, gets it from Joback group
    contribution, and is wrong by a third of a gram per cubic centimetre on
    methanol. Compiled densities now cover forty-three of the forty-eight
    tabulated compounds to better than a tenth of a per cent, and the five they
    miss are the ones with no data method in the compilation - they still fall
    through to the correlation and are what is left of this number.
    """
    assert panel_results["liquid_density"].mean_absolute_error < 0.03


def test_water_has_a_density_at_all(panel_results):
    """It had none, and that took the formulation panel down with it.

    Joback cannot type a molecule with no carbon, so water had no critical
    temperature, so the correlation could not give it a density. A blend
    containing water then lost its own density and all three volume-weighted
    Hansen parameters, because volume fractions could not be formed without
    every component's density.
    """
    from formulate.experts.measured import measured_value

    density = measured_value("liquid_density", "O")
    assert density is not None
    assert density == pytest.approx(997.05, abs=1.0)  # kg/m^3 at 25 degrees C


def test_surface_tension_accuracy(panel_results):
    # Was 9 mN/m when a corresponding-states correlation answered for every
    # compound. With measurements covering the associating ones it sits near
    # 0.3, so the guard is tightened to where it can still catch a regression.
    assert panel_results["surface_tension"].mean_absolute_error < 2.0


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


def test_the_associating_compounds_are_no_longer_the_worst_cases(panel_results):
    """They were, and a measured lookup is what stopped them being.

    Brock-Bird is a corresponding-states correlation for non-associating
    fluids, and on the ones it was never meant for it did not fail quietly: it
    refused water outright and overestimated ethanol by 63 per cent and
    ethylene glycol by 65. Measured surface tensions now cover them, the panel
    prefers a measurement over an estimate on uncertainty alone, and the mean
    absolute error over the reference set fell to a few tenths of a millinewton
    per metre. What is left is no longer concentrated on the hydrogen bonders.
    """
    result = panel_results["surface_tension"]
    assert result.count >= 30
    assert result.mean_absolute_error < 1.0

    worst = result.worst
    assert worst is not None
    assert worst[0] not in {"glycerol", "ethylene glycol", "acetic acid", "water"}
    assert worst[1] < 5.0


def test_description_states_that_this_is_not_a_benchmark(panel_results):
    assert "not a curated benchmark" in describe(panel_results)
