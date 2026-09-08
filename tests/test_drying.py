"""Does the filament dry before it lands?

Three times - flight, evaporation, diffusion - and a verdict of dry, skinned or
wet. These guard the arithmetic and the direction of each dependence; what the
calculation says about a web shooter is in docs/BENCHMARKS.md.
"""

from __future__ import annotations

import math

import pytest

from formulate.processing import (
    DryingConditions,
    DryingRegime,
    assess_drying,
    fuller_diffusion_volume,
    fuller_diffusivity,
)
from formulate.processing.drying import (
    critical_radius,
    diffusion_time,
    evaporation_time,
    mass_transfer_coefficient,
)


def _filament(**overrides):
    base = dict(
        radius=2e-5, velocity=20.0, distance=10.0, vapour_pressure=12091.0,
        solvent_molar_mass=72.11, solvent_diffusion_volume=88.19,
        density=880.0, solvent_fraction=0.80, diffusivity_in_filament=1e-11,
    )
    base.update(overrides)
    return DryingConditions(**base)


# -- the gas-phase side ----------------------------------------------------


def test_the_fuller_volume_matches_a_hand_sum():
    """2-butanone, C4H8O: 4(15.9) + 8(2.31) + 6.11."""
    assert fuller_diffusion_volume("CCC(C)=O") == pytest.approx(
        4 * 15.9 + 8 * 2.31 + 6.11, rel=1e-9
    )


def test_an_aromatic_ring_is_subtracted():
    """Fuller's table carries a -18.3 correction per aromatic ring."""
    benzene = fuller_diffusion_volume("c1ccccc1")
    assert benzene == pytest.approx(6 * 15.9 + 6 * 2.31 - 18.3, rel=1e-9)


def test_an_element_with_no_tabulated_volume_returns_nothing():
    assert fuller_diffusion_volume("[Si](C)(C)(C)C") is None


def test_fuller_reproduces_the_measured_air_diffusivity_of_a_common_solvent():
    """2-butanone in air at 298 K is about 9 mm^2/s."""
    value = fuller_diffusivity(298.15, 101325.0, 72.11, 88.19)
    assert value == pytest.approx(9.2e-6, rel=0.1)


def test_diffusivity_falls_with_pressure_and_rises_with_temperature():
    base = fuller_diffusivity(298.15, 101325.0, 72.11, 88.19)
    assert fuller_diffusivity(298.15, 202650.0, 72.11, 88.19) < base
    assert fuller_diffusivity(350.0, 101325.0, 72.11, 88.19) > base


def test_a_faster_thinner_filament_transfers_mass_faster():
    slow = mass_transfer_coefficient(2e-5, 5.0, 9.2e-6)
    fast = mass_transfer_coefficient(2e-5, 40.0, 9.2e-6)
    thin = mass_transfer_coefficient(5e-6, 5.0, 9.2e-6)
    assert fast > slow
    assert thin > slow


# -- the two times ---------------------------------------------------------


def test_evaporation_time_grows_with_radius():
    coefficient = mass_transfer_coefficient(2e-5, 20.0, 9.2e-6)
    thin = evaporation_time(_filament(radius=1e-5), coefficient)
    thick = evaporation_time(_filament(radius=1e-4), coefficient)
    assert thick > thin


def test_a_less_volatile_solvent_takes_longer_to_evaporate():
    """Cyclohexanone at 0.54 kPa against 2-butanone at 12.1."""
    coefficient = mass_transfer_coefficient(2e-5, 20.0, 9.2e-6)
    fast = evaporation_time(_filament(vapour_pressure=12091.0), coefficient)
    slow = evaporation_time(_filament(vapour_pressure=542.0), coefficient)
    assert slow > 10 * fast


def test_diffusion_time_uses_the_cylinder_mode_not_a_bare_square():
    """R^2/D would understate it by a factor of 5.78."""
    value = diffusion_time(1e-5, 1e-11)
    assert value == pytest.approx(1e-10 / (2.404826**2 * 1e-11), rel=1e-9)
    assert value < 1e-10 / 1e-11


def test_diffusion_time_is_quadratic_in_radius():
    assert diffusion_time(2e-5, 1e-11) == pytest.approx(4 * diffusion_time(1e-5, 1e-11))


def test_the_critical_radius_is_where_diffusion_just_keeps_up():
    radius = critical_radius(0.5, 1e-11)
    assert diffusion_time(radius, 1e-11) == pytest.approx(0.5, rel=1e-9)
    # About five microns for a half-second flight, which is spider-silk fine.
    assert 3e-6 < radius < 8e-6


# -- the verdict -----------------------------------------------------------


def test_a_fine_filament_dries_in_flight():
    result = assess_drying(_filament(radius=5e-6, diffusivity_in_filament=1e-10))
    assert result.regime is DryingRegime.DRY
    assert result.flight_time > result.diffusion_time


def test_a_thicker_one_skins_over_a_wet_core():
    """Evaporation keeps up and diffusion does not, which is the usual case."""
    result = assess_drying(_filament(radius=1e-4, diffusivity_in_filament=1e-10))
    assert result.regime is DryingRegime.SKINNED
    assert result.evaporation_time < result.flight_time < result.diffusion_time
    assert "lands tacky" in " ".join(result.limitations)


def test_a_thick_one_lands_wet():
    result = assess_drying(_filament(radius=1e-3))
    assert result.regime is DryingRegime.WET
    assert result.evaporation_time > result.flight_time


def test_diffusion_is_the_limiting_step_for_anything_but_the_finest_filament():
    result = assess_drying(_filament(radius=1e-4))
    assert result.limiting_step == "diffusion out of the core"


def test_a_longer_flight_dries_more():
    short = assess_drying(_filament(radius=2e-5, distance=2.0))
    long_ = assess_drying(_filament(radius=2e-5, distance=40.0))
    assert long_.flight_time > short.flight_time
    assert long_.critical_radius > short.critical_radius


def test_dropping_the_diffusion_coefficient_moves_the_verdict():
    """It is the dominant uncertainty and the assessment says so."""
    wet_ish = assess_drying(_filament(radius=5e-6, diffusivity_in_filament=1e-13))
    dry = assess_drying(_filament(radius=5e-6, diffusivity_in_filament=1e-10))
    assert dry.regime is DryingRegime.DRY
    assert wet_ish.regime is not DryingRegime.DRY
    assert any("input, not a prediction" in limit for limit in dry.limitations)


def test_every_assessment_states_that_evaporation_is_a_lower_bound():
    joined = " ".join(assess_drying(_filament()).limitations)
    assert "lower bound" in joined
    assert "held at saturation" in joined


def test_the_description_carries_the_three_times_and_the_radius_that_would_help():
    text = assess_drying(_filament()).describe()
    for token in ("flight", "evaporation", "diffusion"):
        assert token in text
    assert "would have to be under" in text
    assert "um in radius" in text


def test_the_load_bearing_and_drying_radii_are_incompatible():
    """The result the whole calculation exists to produce.

    A strand thin enough to dry in half a second is a few microns across; one
    thick enough to hold a person at a well-drawn 58 MPa is about 4 mm. The
    ratio of their areas is how many filaments a bundle would need.
    """
    drying_radius = critical_radius(0.5, 1e-11)
    load_radius = math.sqrt(800.0 / (58e6) / math.pi)
    assert load_radius > 100 * drying_radius
    assert (load_radius / drying_radius) ** 2 > 1e4
