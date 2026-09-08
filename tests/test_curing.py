"""Setting by reaction, and the heat that comes with it."""

from __future__ import annotations

import pytest

from formulate.processing import (
    POLYMERISATION,
    CureConditions,
    CureVerdict,
    adiabatic_temperature_rise,
    assess_cure,
    impact_energy,
    jet_thrust,
    thermal_time,
)
from formulate.processing.curing import retained_fraction


def _cure(**overrides):
    base = dict(
        monomer="methyl methacrylate", reactive_fraction=0.15, cure_time=2.0,
        radius=2.1e-3, flight_time=0.5,
    )
    base.update(overrides)
    return CureConditions(**base)


def test_neat_acrylic_would_cook_itself():
    """340 K on a monomer that boils at 100 C: the exotherm is the constraint."""
    assert adiabatic_temperature_rise("methyl methacrylate", 1.0) == pytest.approx(
        340.0, rel=0.02
    )


def test_the_rise_is_linear_in_what_actually_reacts():
    full = adiabatic_temperature_rise("methyl methacrylate", 1.0)
    third = adiabatic_temperature_rise("methyl methacrylate", 1 / 3)
    assert third == pytest.approx(full / 3, rel=1e-9)


def test_every_tabulated_monomer_has_a_physical_enthalpy():
    """Vinyl chain polymerisation runs 50 to 90 kJ per mole of double bond."""
    for monomer, (enthalpy, molar_mass, double_bonds) in POLYMERISATION.items():
        assert 40.0 < enthalpy < 100.0, monomer
        assert double_bonds >= 1, monomer
        assert molar_mass > 0.0, monomer
        # Wide enough for a crosslinker. The bound was written when the table
        # held only monofunctional vinyls; a triacrylate legitimately weighs
        # three hundred, and the invariant that actually constrains the
        # chemistry is the enthalpy per double bond checked above.
        assert 50.0 < molar_mass < 400.0, monomer


def test_an_untabulated_monomer_is_refused():
    with pytest.raises(KeyError, match="polymerisation enthalpy"):
        adiabatic_temperature_rise("polydimethylsiloxane", 0.3)


def test_a_thick_strand_traps_its_own_heat():
    """The geometry that makes reaction the only workable mechanism is the
    geometry that keeps its heat in."""
    assert thermal_time(2.1e-3) == pytest.approx(7.6, rel=0.05)
    assert thermal_time(2.5e-6) < 1e-4


def test_the_retained_fraction_is_right_at_both_limits():
    assert retained_fraction(0.0, 7.6) == pytest.approx(1.0)
    assert retained_fraction(1e6, 7.6) < 0.001
    # And monotone between them.
    assert retained_fraction(1.0, 7.6) > retained_fraction(5.0, 7.6)


def test_dilution_is_what_keeps_it_below_a_burn():
    hot = assess_cure(_cure(reactive_fraction=0.30))
    cool = assess_cure(_cure(reactive_fraction=0.15))
    assert hot.burns_on_contact
    assert not cool.burns_on_contact
    assert cool.peak_temperature_c < 70.0


def test_setting_before_it_lands_is_a_failure_not_a_success():
    """A solid rod bounces off the target and the strand has no anchor."""
    result = assess_cure(_cure(cure_time=0.2))
    assert result.verdict is CureVerdict.TOO_FAST
    assert any("bounce rather than bond" in limit for limit in result.limitations)


def test_setting_far_too_late_is_also_a_failure():
    assert assess_cure(_cure(cure_time=30.0)).verdict is CureVerdict.TOO_SLOW


def test_the_usable_window_is_between_landing_and_a_few_seconds():
    for cure_time in (0.5, 1.0, 2.0, 5.0):
        assert assess_cure(_cure(cure_time=cure_time)).verdict is CureVerdict.USABLE


def test_no_cure_kinetics_are_claimed():
    """The cure time is an input. Nothing here derives it from a recipe."""
    joined = " ".join(assess_cure(_cure()).limitations)
    assert "no cure kinetics are predicted" in joined.lower()
    assert "lumped one-parameter model" in joined


def test_the_shot_carries_paintball_energy():
    import math

    volume = math.pi * 2.1e-3**2 * 20.0 * 0.5
    energy = impact_energy(volume, 1050.0, 20.0)
    assert 20.0 < energy < 40.0  # J: above a paintball, near an air rifle


def test_the_thrust_is_trivial_next_to_the_hanging_load():
    """5.8 N to fire it; 800 N to hang off it, through the same wrist."""
    thrust = jet_thrust(1050.0, 2.1e-3, 20.0)
    assert 4.0 < thrust < 8.0
    assert 800.0 / thrust > 100
