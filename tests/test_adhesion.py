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


# ---------------------------------------------------------------------------
# Tack: the question the work of adhesion above does not answer
# ---------------------------------------------------------------------------


ROOM = Conditions(
    temperature=Quantity(value=298.15, unit="K"), pressure=Quantity(value=1.0, unit="atm")
)


def _tack(modulus_pa: float | None, *, relative: float = 0.18, conditions=ROOM):
    """Run the tack expert against a stated shear modulus."""
    from formulate.core.candidate import polymer_candidate
    from formulate.core.prediction import Prediction
    from formulate.core.quantity import ApplicabilityDomain, Uncertainty, UncertaintyKind
    from formulate.experts.adhesion import DahlquistTackExpert

    context = {}
    if modulus_pa is not None:
        context["shear_modulus"] = Prediction(
            property="shear_modulus",
            quantity=Quantity(value=modulus_pa, unit="Pa"),
            uncertainty=Uncertainty(
                std=modulus_pa * relative, kind=UncertaintyKind.EPISTEMIC, basis="test"
            ),
            applicability=ApplicabilityDomain(basis="test"),
            expert_id="test_modulus",
        )
    candidate = polymer_candidate("[*]CC([*])c1ccccc1", conditions=conditions)
    return DahlquistTackExpert().predict(
        PredictionRequest(
            candidate=candidate,
            properties=frozenset({"tack"}),
            conditions=conditions,
            context=context,
        )
    )[0]


def test_hold_and_stick_want_moduli_three_orders_of_magnitude_apart():
    """The reason the expert exists, stated as arithmetic rather than as a view.

    A filament that carries load is a glass and a glass is about 1e9 Pa; tack
    needs 1e5. No single material occupies both, so a design that has to hold
    AND stick has to put the two functions in two materials. That is not a
    preference between architectures - it is the only available one.
    """
    from formulate.experts.adhesion import DAHLQUIST_HIGH
    from formulate.experts.mechanical import _GLASSY_MODULUS, shear_from_young

    glassy_shear = shear_from_young(_GLASSY_MODULUS, "glassy")
    assert glassy_shear / DAHLQUIST_HIGH > 1000.0
    # And the panel says so of the material it actually selected.
    verdict = _tack(glassy_shear)
    assert verdict.is_usable and verdict.quantity.value == 0.0
    assert "no single material does both" in " ".join(verdict.notes)


def test_a_soft_enough_polymer_is_called_tacky():
    verdict = _tack(2.0e4)
    assert verdict.is_usable
    assert verdict.quantity.value == 1.0
    assert "wet a rough surface" in " ".join(verdict.notes)


def test_a_stiff_polymer_is_called_not_tacky_and_says_by_how_much():
    verdict = _tack(1.0e7)
    assert verdict.is_usable
    assert verdict.quantity.value == 0.0
    assert "too stiff to deform into the asperities" in " ".join(verdict.notes)


def test_a_modulus_straddling_the_band_is_refused_rather_than_decided():
    """The criterion has a soft edge - it is quoted as 1e5 as often as 3e5 -
    and the width is the uncertainty rather than sloppy citation.

    Amorphous polypropylene is the standing example: the panel puts its shear
    modulus at 3.5e5 Pa with a factor-of-two spread, which covers the whole
    band. Inside it the answer is decided by the difference between the storage
    modulus at 1 Hz the criterion states and the static plateau this panel
    produces, and that difference is not modelled.
    """
    from formulate.experts.adhesion import DAHLQUIST_HIGH, DAHLQUIST_LOW

    verdict = _tack(3.5e5, relative=1.0)
    assert verdict.status is PredictionStatus.UNSUPPORTED
    reason = " ".join(verdict.notes)
    assert "straddles the criterion" in reason
    assert "inventing the decision" in reason
    assert DAHLQUIST_LOW < DAHLQUIST_HIGH


def test_the_spread_is_read_as_a_factor_and_not_as_an_amount():
    """A rubbery modulus carries a relative spread of one, so a plus-or-minus
    would reach zero and call every rubber tacky."""
    # 4e6 Pa with a factor-of-two spread runs 2e6 to 8e6: clear of the band.
    verdict = _tack(4.0e6, relative=1.0)
    assert verdict.is_usable and verdict.quantity.value == 0.0
    assert "a factor of 2.00 either way" in " ".join(verdict.notes)


def test_without_a_modulus_it_refuses_and_says_surface_energy_will_not_do():
    verdict = _tack(None)
    assert verdict.status is PredictionStatus.UNSUPPORTED
    assert "surface energy does not substitute" in " ".join(verdict.notes)


def test_the_classification_is_exact_because_it_is_only_issued_clear_of_the_band():
    verdict = _tack(1.0e7)
    assert verdict.uncertainty.std == 0.0
    assert "refuses rather than reporting a zero or a one" in verdict.uncertainty.basis


def test_every_prediction_says_it_is_not_a_bond_strength():
    for modulus in (2.0e4, 1.0e7):
        notes = " ".join(_tack(modulus).notes)
        assert "not a measurement" in notes
        assert "viscoelastic dissipation" in notes
        assert "1 Hz" in notes


@requires_rdkit
def test_no_polymer_in_the_catalogue_is_tacky():
    """The finding this expert was built to produce.

    The pool holds sixteen commodity thermoplastics and a cured network, and
    not one of them meets the criterion: twelve are confidently too stiff, one
    straddles the band, and four have no modulus at all. A tackifying resin is
    a low-molar-mass oligomer rather than an entangled polymer, and every
    modulus model in this panel is for a high polymer - so the material the
    architecture needs is not merely absent from the catalogue, it is outside
    what the panel can currently score.
    """
    from formulate.core.candidate import Candidate, MaterialClass
    from formulate.core.prediction import prefer
    from formulate.experts import default_registry
    from formulate.exploration.database import load_polymers, polymer_spec

    wanted = frozenset(
        {"tack", "shear_modulus", "glass_transition_temperature", "amorphous_density"}
    )
    registry = default_registry()
    verdicts = {}
    for record in load_polymers():
        candidate = Candidate(
            material_class=MaterialClass.POLYMER,
            polymer=polymer_spec(record),
            conditions=ROOM,
        )
        context: dict = {}
        for expert in registry.resolution_order(
            registry.experts_for(wanted, MaterialClass.POLYMER)
        ):
            for prediction in expert.predict(
                PredictionRequest(
                    candidate=candidate,
                    properties=wanted,
                    conditions=ROOM,
                    context=dict(context),
                )
            ):
                if prediction.is_usable and (
                    prediction.property not in context
                    or prefer(prediction, context[prediction.property])
                ):
                    context[prediction.property] = prediction
        answer = context.get("tack")
        verdicts[record["abbreviation"]] = None if answer is None else answer.quantity.value

    assert not any(v == 1.0 for v in verdicts.values()), verdicts
    assert sum(v == 0.0 for v in verdicts.values()) == 12
    assert sum(v is None for v in verdicts.values()) == 5
    # The one that straddles is the one a hot-melt formulator would reach for,
    # which is the useful part of the refusal.
    assert verdicts["PP"] is None
    # And the classic tackifier base cannot be scored at all: no chain
    # dimension is tabulated for a copolymer repeat unit, so no modulus.
    assert verdicts["EVA-18"] is None and verdicts["EVA-40"] is None
