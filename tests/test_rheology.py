"""Shear and extensional viscosity.

``shear_viscosity`` sat in the property registry from Phase 1 with nothing
behind it and a recorded reason why physics could not supply it. These are the
correlation routes that could have all along, plus the one extensional claim
that is exact and the refusal that stands where it is not.
"""

from __future__ import annotations

import math

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import (
    MaterialClass,
    MonomerUnit,
    PolymerSpec,
    Candidate,
    molecule_candidate,
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import PredictionStatus
from formulate.core.quantity import Quantity
from formulate.evaluation.engine import prefer
from formulate.experts import default_registry
from formulate.experts.base import PredictionRequest
from formulate.experts.rheology import (
    TROUTON_RATIO,
    CorrespondingStatesViscosityExpert,
    JobackViscosityExpert,
    MeasuredViscosityExpert,
    TroutonExtensionalExpert,
    acentric_factor,
    joback_viscosity,
    letsou_stiel_viscosity,
)

_WANT = frozenset(
    {
        "shear_viscosity", "extensional_viscosity", "critical_temperature",
        "critical_pressure", "normal_boiling_point", "molar_mass",
    }
)

#: Measured liquid viscosities at 25 C, mPa s, from the usual compilations.
_MEASURED = {
    "CCC(C)=O": 0.394,      # 2-butanone
    "CCO": 1.074,           # ethanol
    "Cc1ccccc1": 0.560,     # toluene
    "CCOC(C)=O": 0.423,     # ethyl acetate
    "OCCO": 16.1,           # ethylene glycol
}


def _resolve(smiles, conditions=None):
    registry = default_registry()
    conditions = conditions or Conditions.standard()
    candidate = molecule_candidate(smiles)
    context, offers = {}, []
    for expert in registry.resolution_order(
        registry.experts_for(_WANT, MaterialClass.MOLECULE)
    ):
        request = PredictionRequest(
            candidate=candidate, properties=_WANT, conditions=conditions,
            context=dict(context),
        )
        for prediction in expert.predict(request):
            offers.append(prediction)
            if not prediction.is_usable:
                continue
            incumbent = context.get(prediction.property)
            if incumbent is None or prefer(prediction, incumbent):
                context[prediction.property] = prediction
    return context, offers


# -- the correlations ------------------------------------------------------


@requires_rdkit
@pytest.mark.parametrize("smiles,measured", list(_MEASURED.items()))
def test_joback_is_within_a_factor_of_two(smiles, measured):
    value = joback_viscosity(smiles, 298.15)
    if value is None:
        pytest.skip("Joback cannot type this structure")
    assert abs(math.log10(value * 1000 / measured)) < math.log10(2.0)


def test_letsou_stiel_reproduces_a_hand_calculation():
    """2-butanone: Tc 535.5 K, Pc 4.15 MPa, omega 0.329, M 72.11."""
    published = letsou_stiel_viscosity(
        298.15, 72.11, 535.5, 4.15e6, 0.329, corrected=False
    )
    assert published == pytest.approx(3.17e-4, rel=0.05)


def test_the_fitted_offset_raises_letsou_stiel_by_the_stated_factor():
    from formulate.experts.rheology import _LETSOU_STIEL_OFFSET

    published = letsou_stiel_viscosity(298.15, 72.11, 535.5, 4.15e6, 0.329, corrected=False)
    corrected = letsou_stiel_viscosity(298.15, 72.11, 535.5, 4.15e6, 0.329)
    assert corrected / published == pytest.approx(10.0**-_LETSOU_STIEL_OFFSET, rel=1e-9)
    assert corrected > published, "the correlation under-predicts, so the offset raises it"


def test_a_temperature_above_the_critical_point_has_no_liquid_viscosity():
    assert letsou_stiel_viscosity(600.0, 72.11, 535.5, 4.15e6, 0.329) is None


def test_the_acentric_factor_comes_from_constants_the_panel_produces():
    value = acentric_factor(352.8, 535.5, 4.15e6)  # 2-butanone
    assert value == pytest.approx(0.32, abs=0.06)


def test_a_result_outside_any_liquid_is_refused():
    """VISWANATH_NATARAJAN_2E returns 7470 Pa s for 2-butanone.

    Seven orders of magnitude high, and it looks exactly like a number. The
    plausibility bound exists because that method is in the same table as the
    ones that work.
    """
    from formulate.experts.rheology import _PLAUSIBLE_RANGE

    assert _PLAUSIBLE_RANGE[0] < 3.9e-4 < _PLAUSIBLE_RANGE[1]   # 2-butanone
    assert _PLAUSIBLE_RANGE[0] < 1.4 < _PLAUSIBLE_RANGE[1]       # glycerol
    assert _PLAUSIBLE_RANGE[0] < 2.2e-4 < _PLAUSIBLE_RANGE[1]    # pentane
    assert not _PLAUSIBLE_RANGE[0] < 7470.0 < _PLAUSIBLE_RANGE[1]


# -- the experts -----------------------------------------------------------


@requires_rdkit
@pytest.mark.parametrize("smiles,measured", list(_MEASURED.items()))
def test_the_panel_lands_close_to_the_measured_viscosity(smiles, measured):
    context, _ = _resolve(smiles)
    viscosity = context.get("shear_viscosity")
    assert viscosity is not None, f"no route answered for {smiles}"
    value = viscosity.quantity.to("Pa*s").value * 1000
    assert abs(math.log10(value / measured)) < math.log10(1.5)


@requires_rdkit
def test_the_measured_route_wins_where_it_has_data():
    context, _ = _resolve("Cc1ccccc1")
    assert context["shear_viscosity"].expert_id == "viscosity_measured"


@requires_rdkit
def test_corresponding_states_states_the_widest_spread_of_the_three():
    """Which is why prefer() never picks it when another route answered."""
    _, offers = _resolve("CCC(C)=O")
    spreads = {
        o.expert_id: o.uncertainty.std
        for o in offers
        if o.property == "shear_viscosity" and o.quantity is not None
    }
    assert spreads["viscosity_corresponding_states"] > spreads["viscosity_joback"]
    assert spreads["viscosity_joback"] > spreads["viscosity_measured"]


@requires_rdkit
def test_every_route_refuses_without_a_temperature():
    """A liquid viscosity roughly halves over twenty degrees."""
    for expert in (
        MeasuredViscosityExpert(), JobackViscosityExpert(),
        CorrespondingStatesViscosityExpert(),
    ):
        prediction = expert.predict(
            PredictionRequest(
                candidate=molecule_candidate("CCO"),
                properties=frozenset({"shear_viscosity"}),
                conditions=Conditions(),
            )
        )[0]
        assert prediction.status is PredictionStatus.UNSUPPORTED
        assert "temperature" in prediction.notes[0]


@requires_rdkit
def test_a_solid_has_no_liquid_viscosity():
    for expert in (MeasuredViscosityExpert(), JobackViscosityExpert()):
        prediction = expert.predict(
            PredictionRequest(
                candidate=molecule_candidate("c1ccc2ccccc2c1"),  # naphthalene, mp 80 C
                properties=frozenset({"shear_viscosity"}),
                conditions=Conditions.standard(),
            )
        )[0]
        assert prediction.quantity is None


@requires_rdkit
def test_the_measured_route_does_not_reach_the_estimating_methods():
    """thermo drops through to Letsou-Stiel and Joback without saying so.

    A measured expert that is quietly an estimator is worse than none, because
    the report would name a compilation for a number that came from a group
    table.
    """
    from formulate.experts.rheology import _MEASURED_VISCOSITY_METHODS

    assert "LETSOU_STIEL" not in _MEASURED_VISCOSITY_METHODS
    assert "JOBACK" not in _MEASURED_VISCOSITY_METHODS
    assert "VISWANATH_NATARAJAN_2E" not in _MEASURED_VISCOSITY_METHODS


@requires_rdkit
def test_the_uncertainty_is_a_factor_rather_than_an_amount():
    """A viscosity wrong by 2x is wrong by 0.5 mPa s at one and 50 Pa s at a
    hundred, so a single absolute spread describes neither."""
    _, offers = _resolve("CCC(C)=O")
    joback = next(
        o for o in offers
        if o.expert_id == "viscosity_joback" and o.quantity is not None
    )
    assert joback.uncertainty.ci_low is not None
    assert joback.uncertainty.ci_high is not None
    value = joback.quantity.to("Pa*s").value
    # The interval is multiplicative: the same factor either side.
    assert (value / joback.uncertainty.ci_low) == pytest.approx(
        joback.uncertainty.ci_high / value, rel=1e-9
    )
    assert "factor" in joback.uncertainty.basis


# -- extensional -----------------------------------------------------------


@requires_rdkit
def test_the_extensional_viscosity_is_exactly_three_times_the_shear_one():
    context, _ = _resolve("Cc1ccccc1")
    shear = context["shear_viscosity"].quantity.to("Pa*s").value
    extensional = context["extensional_viscosity"].quantity.to("Pa*s").value
    assert extensional == pytest.approx(TROUTON_RATIO * shear, rel=1e-12)


@requires_rdkit
def test_the_trouton_result_adds_no_error_of_its_own():
    """The ratio is exact, so the whole spread is the shear viscosity's."""
    context, _ = _resolve("Cc1ccccc1")
    shear, extensional = context["shear_viscosity"], context["extensional_viscosity"]
    assert extensional.uncertainty.std == pytest.approx(
        TROUTON_RATIO * shear.uncertainty.std, rel=1e-12
    )
    assert "exact for a Newtonian liquid" in extensional.uncertainty.basis
    assert shear.expert_id in extensional.uncertainty.basis


@requires_rdkit
def test_a_polymer_is_refused_an_extensional_viscosity():
    """The whole point of the expert.

    A spinning dope strain-hardens: its extensional viscosity rises by orders
    of magnitude as chains stretch, and depends on strain rate and on strain
    history. Three times the shear viscosity is not an approximation to that,
    so returning it would be a fabricated capability rather than a rough one.
    """
    candidate = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(
            monomers=(MonomerUnit(smiles="[*]CC([*])c1ccccc1", fraction=1.0),),
            number_average_molar_mass=Quantity(value=150000.0, unit="g/mol"),
        ),
        conditions=Conditions.standard(),
    )
    prediction = TroutonExtensionalExpert().predict(
        PredictionRequest(
            candidate=candidate,
            properties=frozenset({"extensional_viscosity"}),
            conditions=Conditions.standard(),
        )
    )
    # Either it declines the class outright, or it says why in the note.
    assert not prediction or all(p.quantity is None for p in prediction)


def test_the_extensional_expert_covers_liquids_only():
    expert = TroutonExtensionalExpert()
    assert expert.supported_classes == frozenset({MaterialClass.MOLECULE})


@requires_rdkit
def test_extensional_refuses_without_a_shear_viscosity_upstream():
    prediction = TroutonExtensionalExpert().predict(
        PredictionRequest(
            candidate=molecule_candidate("CCO"),
            properties=frozenset({"extensional_viscosity"}),
            conditions=Conditions.standard(),
        )
    )[0]
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "shear_viscosity" in prediction.notes[0]


@requires_rdkit
def test_the_extensional_result_says_it_is_newtonian_only():
    context, _ = _resolve("Cc1ccccc1")
    notes = " ".join(context["extensional_viscosity"].notes)
    assert "Newtonian" in notes
    assert "spinning dope" in notes


# -- melt viscosity --------------------------------------------------------


def _polymer(unit="[*]CC([*])c1ccccc1", temperature=453.15):
    from formulate.core.candidate import Candidate, MonomerUnit, PolymerSpec

    return Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(
            monomers=(MonomerUnit(smiles=unit, fraction=1.0),),
            number_average_molar_mass=Quantity(value=500000.0, unit="g/mol"),
        ),
        conditions=Conditions(
            temperature=Quantity(value=temperature, unit="K"),
            pressure=Quantity(value=101325.0, unit="Pa"),
        ),
    )


def _melt(unit="[*]CC([*])c1ccccc1", temperature=453.15):
    registry = default_registry()
    wanted = frozenset({"glass_transition_temperature", "shear_viscosity"})
    candidate = _polymer(unit, temperature)
    context: dict = {}
    for expert in registry.resolution_order(
        registry.experts_for(wanted, MaterialClass.POLYMER)
    ):
        request = PredictionRequest(
            candidate=candidate, properties=wanted,
            conditions=candidate.conditions, context=dict(context),
        )
        for prediction in expert.predict(request):
            if not prediction.is_usable:
                continue
            incumbent = context.get(prediction.property)
            if incumbent is None or prefer(prediction, incumbent):
                context[prediction.property] = prediction
    return context


def test_wlf_reaches_the_glass_transition_viscosity_at_the_glass_transition():
    """10^12 Pa s at Tg is the rheological definition of Tg, not a fit."""
    from formulate.experts.rheology import wlf_melt_viscosity

    assert wlf_melt_viscosity(373.15, 373.15, 13.7, 50.0) == pytest.approx(1e12, rel=1e-9)


def test_the_melt_thins_steeply_with_temperature():
    from formulate.experts.rheology import wlf_melt_viscosity

    hot = wlf_melt_viscosity(473.15, 373.15, 13.7, 50.0)
    warm = wlf_melt_viscosity(423.15, 373.15, 13.7, 50.0)
    assert hot < warm
    # Fifty degrees is worth more than two orders of magnitude near Tg.
    assert warm / hot > 100


def test_polystyrene_at_two_hundred_degrees_is_in_the_right_decade():
    """A melt viscosity anyone can check: of order 10^3 Pa s.

    The universal WLF pair gives 3 Pa s here, three orders low, which is why
    the table carries measured constants and refuses without them.
    """
    from formulate.experts.rheology import wlf_melt_viscosity

    value = wlf_melt_viscosity(473.15, 373.15, 13.7, 50.0)
    assert 100.0 < value < 10000.0


def test_below_the_glass_transition_there_is_no_melt():
    from formulate.experts.rheology import wlf_melt_viscosity

    assert wlf_melt_viscosity(350.0, 373.15, 13.7, 50.0) is None


def test_far_above_the_glass_transition_wlf_is_not_extrapolated():
    from formulate.experts.rheology import _WLF_RANGE_K, wlf_melt_viscosity

    assert wlf_melt_viscosity(373.15 + _WLF_RANGE_K + 1, 373.15, 13.7, 50.0) is None


@requires_rdkit
def test_the_panel_answers_a_melt_viscosity_for_a_tabulated_polymer():
    context = _melt()
    viscosity = context.get("shear_viscosity")
    assert viscosity is not None
    assert viscosity.expert_id == "melt_wlf"
    assert viscosity.quantity.to("Pa*s").value > 1.0


@requires_rdkit
def test_an_untabulated_polymer_is_refused_rather_than_given_universal_constants():
    from formulate.experts.rheology import MeltViscosityExpert

    prediction = MeltViscosityExpert().predict(
        PredictionRequest(
            candidate=_polymer(unit="[*]CC([*])Cl"),  # PVC, not in the table
            properties=frozenset({"shear_viscosity"}),
            conditions=_polymer().conditions,
        )
    )[0]
    assert prediction.quantity is None
    assert "no measured WLF constants" in prediction.notes[0]


@requires_rdkit
def test_a_melt_thinner_than_water_is_refused_rather_than_reported():
    """Staying inside the 120 K window is not enough on its own.

    A design run over the polymer catalogue put poly(methyl methacrylate)
    first on 3.4e-5 Pa s at 180 C - thinner than water by a factor of thirty,
    for a polymer whose real melt viscosity there is of order 10^4, and 75 K
    above its transition so well inside the fitted window. The defect is in the
    constant pair: c1 is the number of decades between the reference
    temperature and the high-temperature asymptote, and 34.0 puts that
    asymptote at 10^-22 Pa s, which is not a viscosity. The pair cannot be
    referenced to the glass transition, and the table cannot tell.
    """
    from formulate.experts.rheology import MeltViscosityExpert, wlf_melt_viscosity

    # The arithmetic is unchanged; it is the expert that declines to report it.
    assert wlf_melt_viscosity(453.15, 378.0, 34.0, 80.0) == pytest.approx(3.4e-5, rel=0.05)

    context = _melt(unit="[*]CC([*])(C)C(=O)OC")
    assert context.get("shear_viscosity") is None

    transition = context["glass_transition_temperature"]
    prediction = MeltViscosityExpert().predict(
        PredictionRequest(
            candidate=_polymer(unit="[*]CC([*])(C)C(=O)OC"),
            properties=frozenset({"shear_viscosity"}),
            conditions=_polymer().conditions,
            context={"glass_transition_temperature": transition},
        )
    )[0]
    assert prediction.quantity is None
    reason = " ".join(prediction.notes)
    assert "below the 0.001 Pa s of water" in reason
    assert "cannot be referenced to this polymer's glass transition" in reason


@requires_rdkit
def test_the_floor_does_not_touch_the_polymer_the_expert_was_built_for():
    """Polystyrene at 180 C comes out at 3.7e3 Pa s and is unaffected: the
    guard removes a demonstrably wrong answer, not a merely uncertain one."""
    from formulate.experts.rheology import _MELT_VISCOSITY_FLOOR

    value = _melt()["shear_viscosity"].quantity.to("Pa*s").value
    assert value > 1.0e3
    assert value > _MELT_VISCOSITY_FLOOR


@requires_rdkit
def test_the_table_is_keyed_so_a_canonical_lookup_actually_hits_it():
    """RDKit canonicalises [*] to *, so a lookup on the table as written misses
    every polymer in it."""
    from formulate.experts.rheology import WLF_CONSTANTS, _canonical_wlf

    assert len(_canonical_wlf()) == len(WLF_CONSTANTS)
    assert _melt().get("shear_viscosity") is not None


@requires_rdkit
def test_the_melt_result_says_it_is_the_weakest_expert_and_an_upper_bound():
    notes = " ".join(_melt()["shear_viscosity"].notes)
    assert "zero-shear" in notes
    assert "upper bound" in notes
    assert "weakest expert" in notes
