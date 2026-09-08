"""Does a jet become a fibre or drops?

The standard dimensionless grouping, which decides a regime and not a fibre.
These tests guard the arithmetic and the refusals; what the criterion is worth
is recorded in docs/BENCHMARKS.md.
"""

from __future__ import annotations

import math

import pytest

from formulate.processing import (
    MARK_HOUWINK,
    JetRegime,
    SpinningConditions,
    assess_jet,
    deborah_number,
    ohnesorge_number,
    overlap_concentration,
    rayleigh_time,
    zimm_relaxation_time,
)
from formulate.processing.spinning import UnknownPair, intrinsic_viscosity


def _dope(**overrides):
    base = dict(
        nozzle_radius=1.5e-4, velocity=20.0, solvent_viscosity=3.95e-4,
        density=850.0, surface_tension=0.024, molar_mass=500e3,
        concentration=0.10, polymer="polystyrene", solvent="2-butanone",
    )
    base.update(overrides)
    return SpinningConditions(**base)


# -- the pieces ------------------------------------------------------------


def test_the_rayleigh_time_is_the_inertio_capillary_scale():
    """sqrt(rho R^3 / sigma): 0.35 ms for a 0.3 mm jet of a light organic."""
    t = rayleigh_time(850.0, 1.5e-4, 0.024)
    assert t == pytest.approx(math.sqrt(850.0 * 1.5e-4**3 / 0.024))
    assert 1e-4 < t < 1e-3


def test_a_bigger_jet_breaks_up_more_slowly():
    assert rayleigh_time(850.0, 3e-4, 0.024) > rayleigh_time(850.0, 1.5e-4, 0.024)


def test_higher_surface_tension_breaks_a_jet_up_faster():
    assert rayleigh_time(850.0, 1.5e-4, 0.072) < rayleigh_time(850.0, 1.5e-4, 0.024)


def test_the_ohnesorge_number_is_dimensionless_and_small_for_a_thin_solvent():
    oh = ohnesorge_number(3.95e-4, 850.0, 1.5e-4, 0.024)
    assert 0.001 < oh < 0.05


def test_the_overlap_concentration_is_the_reciprocal_intrinsic_viscosity():
    assert overlap_concentration(100.0) == pytest.approx(0.01)


def test_mark_houwink_gives_the_published_intrinsic_viscosity():
    """Polystyrene in 2-butanone, K = 3.9e-2 cm^3/g and a = 0.58."""
    value = intrinsic_viscosity("polystyrene", "2-butanone", 150e3)
    assert value == pytest.approx(3.9e-2 * 150e3**0.58, rel=1e-9)
    assert 20.0 < value < 60.0  # cm^3/g, the right order for this pair


def test_a_better_solvent_swells_the_coil_more():
    """The exponent carries solvent quality: toluene is better for PS than MEK."""
    mek = intrinsic_viscosity("polystyrene", "2-butanone", 1e6)
    toluene = intrinsic_viscosity("polystyrene", "toluene", 1e6)
    assert toluene > mek


def test_an_untabulated_pair_is_refused_rather_than_borrowed():
    """The exponent differs enough between solvents to change [eta] threefold."""
    with pytest.raises(UnknownPair, match="Mark-Houwink"):
        intrinsic_viscosity("polystyrene", "acetone", 150e3)
    with pytest.raises(UnknownPair):
        intrinsic_viscosity("polyamide 66", "2-butanone", 150e3)


def test_every_tabulated_exponent_is_physical():
    """0.5 is a theta solvent and 0.8 a good one; outside that is not a coil."""
    for (polymer, solvent), (k, a) in MARK_HOUWINK.items():
        assert 0.45 <= a <= 0.85, f"{polymer}/{solvent}"
        assert 0.0 < k < 1.0


def test_the_zimm_time_scales_with_mass_and_solvent_viscosity():
    a = zimm_relaxation_time(40.0, 150e3, 4e-4, 298.15)
    assert zimm_relaxation_time(40.0, 300e3, 4e-4, 298.15) > a
    assert zimm_relaxation_time(40.0, 150e3, 8e-4, 298.15) == pytest.approx(2 * a)
    # A dilute coil relaxes in microseconds, not milliseconds.
    assert 1e-9 < a < 1e-4


def test_the_deborah_number_is_a_ratio_of_the_two_times():
    assert deborah_number(2e-3, 1e-3) == pytest.approx(2.0)


# -- the verdict -----------------------------------------------------------


def test_a_dilute_short_chain_dope_sprays():
    """Not entangled and relaxing far faster than capillarity: drops."""
    result = assess_jet(_dope(molar_mass=150e3, concentration=0.05))
    assert result.regime is JetRegime.SPRAY
    assert result.deborah < 1.0
    assert not result.entangled


def test_raising_the_molar_mass_turns_a_spray_into_a_filament():
    """The single strongest lever, because the relaxation time follows it steeply."""
    low = assess_jet(_dope(molar_mass=150e3, concentration=0.10))
    high = assess_jet(_dope(molar_mass=500e3, concentration=0.10))
    assert low.regime is JetRegime.SPRAY
    assert high.regime is JetRegime.FILAMENT
    assert high.deborah > low.deborah


def test_a_narrower_nozzle_favours_a_filament():
    """The Rayleigh time falls as R^1.5, so the same dope is more elastic in a
    finer jet."""
    wide = assess_jet(_dope(molar_mass=150e3, concentration=0.20, nozzle_radius=5e-4))
    narrow = assess_jet(_dope(molar_mass=150e3, concentration=0.20, nozzle_radius=1.5e-4))
    assert narrow.deborah > wide.deborah


def test_the_band_around_the_threshold_is_reported_as_marginal():
    """The transition is not sharp and rounding it to a verdict would say more
    than the criterion supports."""
    seen = set()
    for concentration in [0.01 * i for i in range(5, 31)]:
        seen.add(assess_jet(_dope(molar_mass=150e3, concentration=concentration)).regime)
    assert JetRegime.MARGINAL in seen
    assert JetRegime.SPRAY in seen and JetRegime.FILAMENT in seen


def test_an_unentangled_dope_says_so_even_when_it_would_form_a_filament():
    result = assess_jet(_dope(molar_mass=500e3, concentration=0.02))
    assert not result.entangled
    assert any("entanglement threshold" in limit for limit in result.limitations)


def test_a_gel_is_flagged_rather_than_reported_as_a_clean_filament():
    """At 2 MDa and 30 per cent the relaxation time is over a minute.

    The Deborah number is then enormous and means nothing about break-up: the
    dope will fracture at the nozzle rather than flow through it.
    """
    result = assess_jet(_dope(molar_mass=2e6, concentration=0.30))
    assert result.relaxation_time > 1.0
    assert any("rubbery gel" in limit for limit in result.limitations)


def test_every_assessment_states_what_it_does_not_decide():
    joined = " ".join(assess_jet(_dope()).limitations)
    assert "not a fibre" in joined
    assert "solvent leaving the filament" in joined


def test_the_description_carries_the_numbers_behind_the_verdict():
    text = assess_jet(_dope()).describe()
    for token in ("Deborah", "Ohnesorge", "c/c*", "[eta]"):
        assert token in text


# -- pushing it through the nozzle -----------------------------------------


def test_extrusion_pressure_is_hagen_poiseuille():
    from formulate.processing import extrusion_pressure

    assert extrusion_pressure(1.0, 2.5e-4, 20.0, 0.01) == pytest.approx(
        8.0 * 1.0 * 0.01 * 20.0 / 2.5e-4**2
    )


def test_halving_the_nozzle_quadruples_the_pressure():
    from formulate.processing import extrusion_pressure

    wide = extrusion_pressure(1.0, 5e-4, 20.0, 0.01)
    narrow = extrusion_pressure(1.0, 2.5e-4, 20.0, 0.01)
    assert narrow == pytest.approx(4 * wide)


def test_a_polystyrene_melt_cannot_be_shot_through_a_fine_nozzle():
    """The result that rules out a hot-melt web shooter as a direct shot.

    At 210 C - already near where polystyrene starts to degrade - the
    zero-shear pressure to drive it through a 0.5 mm nozzle at 20 m/s is
    around 10^5 bar. A hydraulic hand tool reaches about 700.
    """
    from formulate.processing import extrusion_pressure

    melt_viscosity = 377.0  # Pa s at 210 C, from the panel
    fast = extrusion_pressure(melt_viscosity, 2.5e-4, 20.0, 0.01)
    slow = extrusion_pressure(melt_viscosity, 2.5e-4, 0.5, 0.01)
    assert fast / 1e5 > 10000  # bar
    assert slow / 1e5 > 1000
    # Slower extrusion with draw-down is the only route that gets close.
    assert fast / slow == pytest.approx(40.0)


# -- the thick ballistic jet -----------------------------------------------


def test_a_thin_fast_jet_of_a_thin_liquid_is_turbulent_and_atomises():
    """Which is why a fire hose makes spray rather than a rod of water."""
    from formulate.processing import breakup_length, reynolds_number

    assert reynolds_number(1050.0, 20.0, 4.2e-3, 2e-3) > 2000
    assert breakup_length(1050.0, 20.0, 4.2e-3, 0.030, 2e-3) is None


def test_enough_viscosity_makes_the_same_jet_laminar_and_coherent():
    from formulate.processing import breakup_length, reynolds_number

    assert reynolds_number(1050.0, 20.0, 4.2e-3, 50e-3) < 2000
    length = breakup_length(1050.0, 20.0, 4.2e-3, 0.030, 50e-3)
    assert length is not None
    assert length > 10.0, "it has to outlast a ten metre shot"


def test_a_thicker_faster_jet_carries_further_before_breaking_up():
    from formulate.processing import breakup_length

    slow = breakup_length(1050.0, 10.0, 4.2e-3, 0.030, 100e-3)
    fast = breakup_length(1050.0, 20.0, 4.2e-3, 0.030, 100e-3)
    thin = breakup_length(1050.0, 20.0, 2.0e-3, 0.030, 100e-3)
    assert fast > slow
    assert fast > thin


def test_the_weber_number_of_a_thick_fast_jet_is_enormous():
    """Which is the whole reason it carries: inertia swamps capillarity."""
    from formulate.processing import weber_number

    assert weber_number(1050.0, 20.0, 4.2e-3, 0.030) > 10000


def test_a_thick_nozzle_makes_the_shot_affordable():
    """The number that reopened the question.

    A polymer melt through half a millimetre needs about 10^5 bar. A resin of
    a hundredth the viscosity through a nozzle eight times wider needs under a
    bar, because pressure carries viscosity linearly and the radius squared.
    """
    from formulate.processing import extrusion_pressure

    melt = extrusion_pressure(377.0, 2.5e-4, 20.0, 0.01)
    resin = extrusion_pressure(100e-3, 2.1e-3, 20.0, 0.02)
    assert melt / 1e5 > 10000        # bar
    assert resin / 1e5 < 1.0         # bar
    assert melt / resin > 1e4


def test_solution_viscosity_inverts_its_own_target():
    """The formulation is specified by c[eta], so the round trip has to hold."""
    from formulate.processing.spinning import overlap_for_viscosity, solution_viscosity

    for base in (1.0e-3, 4.25e-3, 2.0e-2):
        for target in (2.0e-3, 1.0e-1, 1.0):
            if target <= base:
                continue
            overlap = overlap_for_viscosity(target, base)
            assert solution_viscosity(1.0, overlap, base) == pytest.approx(target, rel=1e-9)


def test_the_two_regimes_join_at_the_overlap_concentration():
    """Huggins below, entanglement above, and no step between them."""
    from formulate.processing.spinning import solution_viscosity

    base = 4.25e-3
    below = solution_viscosity(1.0, 1.0 - 1e-9, base)
    above = solution_viscosity(1.0, 1.0 + 1e-9, base)
    assert below == pytest.approx(above, rel=1e-6)


def test_only_the_product_of_concentration_and_intrinsic_viscosity_matters():
    """Why a ratio can be quoted without Mark-Houwink constants.

    Halving the intrinsic viscosity and doubling the concentration is the same
    solution as far as this model is concerned, which is what lets the recipe
    fix c[eta] and leave the split to whoever knows the polymer.
    """
    from formulate.processing.spinning import solution_viscosity

    base = 4.25e-3
    assert solution_viscosity(80.0, 0.0226, base) == pytest.approx(
        solution_viscosity(160.0, 0.0113, base), rel=1e-9
    )


def test_more_polymer_is_always_thicker():
    from formulate.processing.spinning import solution_viscosity

    base = 4.25e-3
    values = [solution_viscosity(1.0, c, base) for c in (0.1, 0.5, 0.9, 1.5, 3.0, 6.0)]
    assert values == sorted(values)
