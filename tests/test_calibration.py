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
    # Was 0.27 over forty compounds. Atomic critical constants then made two
    # more answerable - dimethyl sulfoxide and dimethylformamide, which no
    # compilation measures and which Joback could not type - and Brock-Bird
    # handles both badly, at around half. The average is worse because the
    # coverage is wider, which is the trade rather than a regression: the
    # bound is raised to admit it and stays tight enough to catch one.
    assert result.count >= 42
    assert result.mean_absolute_error < 1.5

    worst = result.worst
    assert worst is not None
    assert worst[0] not in {"glycerol", "ethylene glycol", "acetic acid", "water"}


def test_surface_tension_errors_stay_inside_the_bars_they_quote(panel_results):
    """Accuracy fell as coverage grew. Honesty is what must not.

    The worst case used to be under 5 mN/m, because every compound the panel
    answered had a measurement behind it. It is now 22, for dimethyl sulfoxide,
    which no compilation measures and which only became answerable at all once
    atomic critical constants replaced the group table Joback could not apply
    to a sulfoxide. Raising that bound and moving on would be the wrong lesson:
    a correlation reaching further is allowed to be less accurate, and is not
    allowed to be quietly confident about it.

    So this asserts the property that actually protects the ranking. Every
    prediction, measured or correlated, has to sit inside twice the uncertainty
    it quotes.
    """
    result = panel_results["surface_tension"]
    outside = [
        (error, stated)
        for error, stated in zip(result.errors, result.stated_std)
        if stated and abs(error) > 2.0 * stated
    ]
    assert not outside, f"{len(outside)} predictions fall outside their own two sigma"


def test_description_states_that_this_is_not_a_benchmark(panel_results):
    assert "not a curated benchmark" in describe(panel_results)


# --------------------------------------------------------------------------
# The seven properties that had nothing to check them against
# --------------------------------------------------------------------------

_NEWLY_REFERENCED = [
    ("critical_temperature", 40.0, "K"),
    ("critical_pressure", 5.0e5, "Pa"),
    ("critical_volume", 1.5e-5, "m^3/mol"),
    ("enthalpy_vaporization", 3500.0, "J/mol"),
    ("enthalpy_fusion", 3200.0, "J/mol"),
    ("heat_capacity_gas", 3.5, "J/mol/K"),
]


@pytest.mark.parametrize(("prop", "bound", "unit"), _NEWLY_REFERENCED)
def test_the_estimator_is_measured_rather_than_cited(results, prop, bound, unit):
    """Each of these used to rest on the accuracy its authors reported.

    A citation is not a measurement. Joback and Reid quote average absolute
    errors on their own fitting set; over this compound mix the critical
    temperature is out by 28 K against a quoted 4.8, because the set includes
    associating compounds their method was not built for.
    """
    assert prop in results
    assert results[prop].count >= 40
    assert results[prop].mean_absolute_error < bound


def test_logp_is_measured_too(panel_results):
    assert panel_results["logp"].count >= 40
    assert panel_results["logp"].mean_absolute_error < 1.0


@pytest.mark.parametrize("prop", [p for p, _, _ in _NEWLY_REFERENCED])
def test_the_newly_measured_properties_are_not_overconfident(results, prop):
    """The point of adding the data. Two of these failed when it arrived.

    Critical temperature caught 17 per cent of compounds inside its own one
    sigma and the enthalpy of vaporisation caught 7, where a correct estimate
    catches about 68. Both were invisible while the properties had no reference
    values at all.
    """
    coverage = results[prop].within_one_sigma
    assert coverage is not None
    assert coverage >= 0.5, f"{prop} claims a tighter error bar than it earns"


def test_the_vaporisation_reference_is_at_the_boiling_point_not_at_298(results):
    """Getting this wrong tripled the apparent error and hid a real one.

    Joback's enthalpy of vaporisation is defined at the normal boiling point.
    Compared against a reference tabulated at 298 K it showed a mean error of
    7.3 kJ/mol and a one-sigma coverage of 7 per cent, which looks exactly like
    an overconfident expert and was nothing of the kind.
    """
    from formulate.evaluation.calibration import REFERENCE_PROPERTIES

    entry = REFERENCE_PROPERTIES["enthalpy_vaporization_tb_j_mol"]
    assert entry[0] == "enthalpy_vaporization"
    assert entry[2] == "boiling_point_c"
    assert results["enthalpy_vaporization"].mean_absolute_error < 3500.0


def test_a_prediction_at_the_wrong_temperature_is_not_compared():
    """The guard that would have caught it, rather than a note not to repeat it."""
    from formulate.core.conditions import Conditions
    from formulate.core.quantity import Quantity
    from formulate.evaluation.calibration import _condition_mismatch
    from formulate.core.prediction import Prediction

    at_boiling = Prediction(
        property="enthalpy_vaporization",
        expert_id="test",
        expert_version="1",
        method="test",
        quantity=Quantity(value=1.0, unit="J/mol"),
        conditions=Conditions.standard().model_copy(
            update={"temperature": Quantity(value=351.6, unit="K")}
        ),
    )
    assert "351.6" in _condition_mismatch(at_boiling, 298.15)
    assert _condition_mismatch(at_boiling, 351.6) == ""


def test_associating_compounds_get_a_wider_bar_where_the_data_shows_they_need_one():
    """And do not get one where it does not.

    Over the reference set the enthalpy of vaporisation errs by 4.0 kJ/mol for
    hydrogen-bond donors against 1.7 for the rest. Critical volume, enthalpy of
    fusion and gas heat capacity show no such split, and inventing a factor for
    them would be decoration rather than calibration.
    """
    from formulate.experts.joback import _MEASURED_SPREAD

    for prop in ("critical_temperature", "critical_pressure", "enthalpy_vaporization"):
        plain, associating = _MEASURED_SPREAD[prop]
        assert associating > plain * 1.5, prop
    for prop in ("critical_volume", "enthalpy_fusion", "heat_capacity_gas"):
        plain, associating = _MEASURED_SPREAD[prop]
        assert plain == associating, prop


# --------------------------------------------------------------------------
# A row that measures nothing must say so
# --------------------------------------------------------------------------


def test_a_lookup_answer_is_not_reported_as_accuracy():
    """The default run reported a critical temperature good to exactly zero.

    Over fifty compounds, mean absolute error 0.0000, and a boiling point good
    to 0.22 degrees. Both read as extraordinary accuracy and were the compiled
    measurement being compared against the table it was compiled from.
    """
    from formulate.evaluation.calibration import PropertyCalibration

    lookup = PropertyCalibration(
        property="critical_temperature", unit="K", count=50,
        errors=[0.0] * 50, stated_std=[1.0] * 50, answered_by={"measured": 50},
    )
    assert lookup.self_comparison
    assert "NOT AN ACCURACY MEASUREMENT" in lookup.verdict()
    assert "NOT AN ACCURACY MEASUREMENT" in lookup.describe()


def test_an_estimator_is_judged_on_its_uncertainty_as_before():
    from formulate.evaluation.calibration import PropertyCalibration

    estimator = PropertyCalibration(
        property="normal_boiling_point", unit="K", count=10,
        # Seven of ten inside the stated one sigma, which is the ~68% a
        # correct estimate implies. An earlier fixture put all ten inside and
        # was then correctly judged conservative.
        errors=[5.0, -4.0, 4.0, -5.0, 3.0, -4.0, 5.0, -9.0, 8.0, -10.0],
        stated_std=[6.0] * 10, answered_by={"joback": 10},
    )
    assert estimator.self_comparison == ""
    assert "reasonable" in estimator.verdict()


def test_a_mixed_row_still_names_the_lookup_share():
    from formulate.evaluation.calibration import PropertyCalibration

    mixed = PropertyCalibration(
        property="liquid_density", unit="g/cm^3", count=10,
        errors=[0.0] * 10, stated_std=[0.01] * 10,
        answered_by={"measured": 7, "interfacial": 3},
    )
    assert "7 of 10" in mixed.self_comparison


@requires_rdkit
def test_the_report_warns_before_the_numbers_rather_than_after():
    """A reader who stops at the first table must have been told already."""
    from formulate.evaluation.calibration import calibrate, describe
    from formulate.experts import default_registry
    from formulate.exploration import load_reference_compounds

    text = describe(calibrate(default_registry(), load_reference_compounds()))
    assert "SOME ROWS BELOW MEASURE NOTHING" in text
    assert text.index("MEASURE NOTHING") < text.index("mean absolute error")


@requires_rdkit
def test_restricting_to_an_estimator_produces_a_real_measurement():
    from formulate.evaluation.calibration import calibrate
    from formulate.experts import default_registry
    from formulate.exploration import load_reference_compounds

    results = calibrate(
        default_registry(), load_reference_compounds(), expert_id="joback"
    )
    boiling = results["normal_boiling_point"]
    assert boiling.self_comparison == ""
    assert boiling.mean_absolute_error > 5.0, "a group method is not exact"
    assert "measured" not in boiling.answered_by
