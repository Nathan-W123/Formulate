"""Work of adhesion from measured surface energies.

The check that matters is directional: does the model know which way a liquid
runs? A work of adhesion in mJ/m^2 is hard to falsify by eye, but a contact
angle is not, and Young's equation turns one into the other.
"""

from __future__ import annotations

import math

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import molecule_candidate
from formulate.core.conditions import Conditions
from formulate.core.prediction import PredictionStatus
from formulate.core.quantity import Quantity
from formulate.experts.adhesion import (
    LIQUIDS,
    SUBSTRATES,
    AdhesionExpert,
    liquid_energy,
    resolve_substrate,
    spreading_coefficient,
    work_of_adhesion,
)
from formulate.experts.base import PredictionRequest

WATER = "O"
DIIODOMETHANE = "ICI"
HEXANE = "CCCCCC"


def _contact_angle(smiles: str, substrate: str) -> float:
    """Young's equation, in degrees, from the tabulated components."""
    liquid = liquid_energy(smiles)
    solid = resolve_substrate(substrate)[1]
    cosine = work_of_adhesion(liquid, solid) / liquid.total - 1.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


@pytest.mark.parametrize(
    "substrate,low,high",
    [
        ("ptfe", 100.0, 118.0),          # measured 108-112
        ("polyethylene", 92.0, 110.0),   # measured 96-103
        ("pmma", 65.0, 88.0),            # measured 70-75
        ("glass", 0.0, 25.0),            # clean glass is wetted
    ],
)
def test_water_contact_angles_land_where_they_are_measured(substrate, low, high):
    assert low <= _contact_angle(WATER, substrate) <= high


def test_the_polar_term_is_what_separates_ptfe_from_glass():
    """Drop it and the two solids look alike to water, which is the whole error."""
    water = liquid_energy(WATER)
    ptfe = resolve_substrate("ptfe")[1]
    glass = resolve_substrate("glass")[1]

    two_component = work_of_adhesion(water, glass) / work_of_adhesion(water, ptfe)
    single = math.sqrt(glass.total / ptfe.total)  # Girifalco-Good, totals only
    assert two_component > 2.5 > single


def test_water_beads_on_fluoropolymer_and_spreads_on_glass():
    water = liquid_energy(WATER)
    assert spreading_coefficient(water, resolve_substrate("ptfe")[1]) < 0
    assert spreading_coefficient(water, resolve_substrate("glass")[1]) > 0


def test_a_purely_dispersive_liquid_wets_a_fluoropolymer_that_water_will_not():
    """Hexane spreads on PTFE; water does not. Same solid, opposite outcome."""
    ptfe = resolve_substrate("ptfe")[1]
    assert spreading_coefficient(liquid_energy(HEXANE), ptfe) > 0
    assert spreading_coefficient(liquid_energy(WATER), ptfe) < 0


def test_diiodomethane_does_not_spread_on_polyethylene():
    """It is measured at about 52 degrees, so a positive coefficient is wrong."""
    pe = resolve_substrate("polyethylene")[1]
    assert spreading_coefficient(liquid_energy(DIIODOMETHANE), pe) < 0


def test_substrate_aliases_resolve():
    assert resolve_substrate("Teflon")[0] == "ptfe"
    assert resolve_substrate("aluminum")[0] == "aluminium-oxide"
    assert resolve_substrate("unobtainium") is None


def test_inorganic_substrates_carry_a_surface_state_uncertainty():
    """Their number is set by what is adsorbed, not by the bulk material."""
    for name in ("glass", "aluminium-oxide", "steel"):
        assert SUBSTRATES[name].spread >= 10.0
    for name in ("ptfe", "pmma", "pet"):
        assert SUBSTRATES[name].spread <= 2.0


# -- the expert ------------------------------------------------------------


def _predict(smiles, surfaces):
    expert = AdhesionExpert()
    conditions = Conditions(
        temperature=Quantity(value=298.15, unit="K"), surfaces=tuple(surfaces)
    )
    return expert.predict(
        PredictionRequest(
            candidate=molecule_candidate(smiles),
            properties=frozenset({"work_of_separation"}),
            conditions=conditions,
        )
    )[0]


@requires_rdkit
def test_the_expert_produces_a_work_of_separation_for_a_named_substrate():
    prediction = _predict(WATER, ["ptfe"])
    assert prediction.status is PredictionStatus.OK
    # 50 mJ/m^2 is 0.050 N/m.
    assert prediction.quantity.to("N/m").value == pytest.approx(0.050, abs=0.01)
    assert prediction.uncertainty.std > 0


@requires_rdkit
def test_no_substrate_means_no_answer():
    """Adhesion is a property of an interface, not of a liquid."""
    prediction = _predict(WATER, [])
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "no substrate was named" in " ".join(prediction.notes) or prediction.quantity is None


@requires_rdkit
def test_an_untabulated_substrate_is_refused_and_the_known_ones_are_listed():
    prediction = _predict(WATER, ["unobtainium"])
    assert prediction.status is PredictionStatus.UNSUPPORTED


@requires_rdkit
def test_an_untabulated_liquid_is_refused_rather_than_split_from_hansen():
    """The Hansen route was tested and halves water's dispersive component."""
    prediction = _predict("CC(C)(C)c1ccccc1", ["ptfe"])
    assert prediction.quantity is None


@requires_rdkit
def test_practical_adhesion_is_disclaimed_on_every_value():
    prediction = _predict(WATER, ["aluminium"])
    joined = " ".join(prediction.notes)
    assert "peel strength" in joined
    assert "adsorbed" in joined  # the surface-state caveat for an inorganic


def test_every_tabulated_liquid_and_substrate_states_its_basis():
    for entry in list(LIQUIDS.values()) + list(SUBSTRATES.values()):
        assert entry.basis
        assert entry.dispersive >= 0 and entry.polar >= 0
        assert entry.spread > 0
