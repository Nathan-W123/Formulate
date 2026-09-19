"""Filament properties, which are properties of a material at a geometry.

The point of this expert is that solidification, extrusion pressure and
capillary survival are not properties of a polymer at all. They are properties
of a polymer at a diameter, and every dead end in the web-shooter example was
one of them being invisible to a search over materials.
"""

from __future__ import annotations

import pytest

from formulate.core.candidate import Candidate, MaterialClass, MonomerUnit, PolymerSpec
from formulate.core.conditions import Conditions, Spinline
from formulate.core.prediction import Prediction, PredictionStatus
from formulate.core.quantity import Quantity
from formulate.experts.base import PredictionRequest
from formulate.experts.spinline import (
    SpinlineExpert,
    breakup_length,
    extrusion_pressure,
    solidification_time,
)

PE = "[*]CC[*]"


def _spinline(die_um=900.0, draw=225.0, speed=9.0, land_mm=1.8, count=80):
    return Spinline(
        die_diameter=Quantity(value=die_um * 1e-6, unit="m"),
        draw_ratio=draw,
        line_speed=Quantity(value=speed, unit="m/s"),
        die_land=Quantity(value=land_mm * 1e-3, unit="m"),
        filament_count=count,
    )


def _dep(prop, value, unit):
    return Prediction(
        property=prop, quantity=Quantity(value=value, unit=unit), expert_id="stub"
    )


def _predict(prop, spinline=None, context=None):
    candidate = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(monomers=(MonomerUnit(smiles=PE),)),
        conditions=Conditions.standard(),
    )
    request = PredictionRequest(
        candidate=candidate,
        properties=frozenset({prop}),
        conditions=Conditions(
            temperature=Quantity(value=473.15, unit="K"),
            spinline=spinline if spinline is not None else _spinline(),
        ),
        context=context
        or {
            "shear_viscosity": _dep("shear_viscosity", 1000.0, "Pa*s"),
            "amorphous_density": _dep("amorphous_density", 910.0, "kg/m^3"),
            "surface_tension": _dep("surface_tension", 0.0225, "N/m"),
        },
    )
    return next(p for p in SpinlineExpert().predict(request) if p.property == prop)


# -- geometry is a condition, and its absence is a refusal -----------------


def test_without_a_geometry_a_filament_property_is_refused():
    candidate = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(monomers=(MonomerUnit(smiles=PE),)),
        conditions=Conditions.standard(),
    )
    request = PredictionRequest(
        candidate=candidate,
        properties=frozenset({"solidification_time"}),
        conditions=Conditions(temperature=Quantity(value=473.15, unit="K")),
    )
    prediction = next(
        p for p in SpinlineExpert().predict(request) if p.property == "solidification_time"
    )
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "defined at a geometry" in " ".join(prediction.notes)


def test_draw_down_sets_the_diameter_that_matters():
    """A 900 um die drawn 225x makes a 60 um filament, not a 900 um one."""
    assert _spinline().final_diameter.to("m").value == pytest.approx(60e-6, rel=1e-6)


# -- the square law, which is the whole finding ---------------------------


def test_solidification_picks_the_regime_from_the_biot_number():
    """The first version took conduction unconditionally, which is the Bi >> 1
    limit and wrong for anything fine. A 60 um filament is air-film limited and
    five times slower than that model gave."""
    from formulate.experts.spinline import (
        POLYMER_CONDUCTIVITY,
        heat_transfer_coefficient,
    )

    fine = 60e-6
    biot = heat_transfer_coefficient(fine, 9.0) * (fine / 2) / POLYMER_CONDUCTIVITY
    assert biot < 1.0, "a fine filament is limited by the air film"
    conduction_only = (0.5 * fine / 2) ** 2 / 1e-7
    assert solidification_time(fine, 9.0) > 3 * conduction_only


def test_setting_still_rises_steeply_with_diameter():
    """Which regime applies changes the number, not the finding: setting is
    geometry, and a fat strand cannot be made to set quickly by chemistry."""
    assert solidification_time(4.4e-3, 9.0) > 5.0
    assert solidification_time(60e-6, 9.0) < 0.05
    ratio = solidification_time(240e-6, 9.0) / solidification_time(60e-6, 9.0)
    assert ratio > 3.0


def test_the_engine_reports_the_drawn_filament_in_tens_of_milliseconds():
    """Not the 2.25 ms the conduction model claimed. Still inside a 333 ms
    flight, but by a factor of thirty rather than a hundred and fifty."""
    value = _predict("solidification_time").quantity.to_canonical().value
    assert 0.005 < value < 0.05


# -- pressure is spent at the die, at the die's own speed -----------------


def test_the_melt_moves_at_the_line_speed_divided_by_the_draw_ratio():
    """Mass conservation. Using the line speed here overstated the pressure by
    the draw ratio and made every geometry look impossible."""
    fast = _predict("extrusion_pressure", _spinline(draw=25.0))
    slow = _predict("extrusion_pressure", _spinline(draw=225.0))
    assert slow.quantity.value < fast.quantity.value


def test_opening_the_die_and_drawing_harder_buys_pressure_at_equal_filament():
    """Both make a 60 um filament; the wider die costs far less pressure."""
    narrow = _predict("extrusion_pressure", _spinline(300.0, 25.0, land_mm=0.6))
    wide = _predict("extrusion_pressure", _spinline(1200.0, 400.0, land_mm=2.4))
    assert wide.quantity.value < narrow.quantity.value / 5.0


def test_pressure_rises_with_viscosity():
    thick = _predict(
        "extrusion_pressure",
        context={
            "shear_viscosity": _dep("shear_viscosity", 5000.0, "Pa*s"),
            "amorphous_density": _dep("amorphous_density", 910.0, "kg/m^3"),
            "surface_tension": _dep("surface_tension", 0.0225, "N/m"),
        },
    )
    thin = _predict("extrusion_pressure")
    assert thick.quantity.value > thin.quantity.value


# -- drawing is what makes the filament survive ---------------------------


def test_drawing_stabilises_against_capillary_breakup():
    """The impossibility result was about a free jet. Tension changes it."""
    free = breakup_length(60e-6, 9.0, 1000.0, 910.0, 0.0225, draw_ratio=1.0)
    drawn = breakup_length(60e-6, 9.0, 1000.0, 910.0, 0.0225, draw_ratio=225.0)
    assert drawn > free * 100.0


def test_a_drawn_filament_solidifies_long_before_it_beads():
    prediction = _predict("filament_stability")
    assert prediction.quantity.value > 1.0


def test_an_undrawn_jet_gets_no_stabilisation_and_says_so():
    prediction = _predict("filament_stability", _spinline(draw=1.0))
    assert "undrawn" in " ".join(prediction.notes)


# -- the pieces agree with each other --------------------------------------


def test_extrusion_pressure_matches_hagen_poiseuille_by_hand():
    value = extrusion_pressure(
        diameter=1e-3, land=2e-3, speed=0.02, viscosity=1000.0, density=910.0
    )
    expected = 8 * 1000.0 * 2e-3 * 0.02 / (0.5e-3) ** 2 + 910.0 * 0.02**2 / 2
    assert value == pytest.approx(expected)


def test_a_missing_dependency_is_refused_rather_than_assumed():
    prediction = _predict(
        "extrusion_pressure",
        context={"amorphous_density": _dep("amorphous_density", 910.0, "kg/m^3")},
    )
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "melt viscosity" in " ".join(prediction.notes)
