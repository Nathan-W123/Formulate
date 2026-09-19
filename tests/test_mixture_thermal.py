"""Thermal and chemical properties of a formulation, one rule per property.

Two thirds of these tests are refusals, and that is deliberate.  The whole
argument of this expert is that the mixing rule is different for every property
and that some properties have no mixing rule at all - a formulation has no
partition coefficient, and a formulation that boils over sixty kelvin has no
boiling point.  A refusal here is a feature, so it is pinned as hard as a
number is.
"""

from __future__ import annotations

import math

import pytest
from conftest import requires_rdkit

from formulate.core.candidate import (
    Candidate,
    ComponentRole,
    FractionBasis,
    MaterialClass,
    MixtureComponent,
    MixtureSpec,
    MoleculeSpec,
    MonomerUnit,
    PhaseAssumption,
    PolymerSpec,
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import PredictionStatus
from formulate.core.properties import get_property
from formulate.core.quantity import Quantity, Uncertainty, UncertaintyKind
from formulate.experts.activity import _gammas, unifac_available, unifac_groups
from formulate.experts.base import PredictionRequest
from formulate.experts.mixture import MixtureExpert
from formulate.experts.mixture_thermal import (
    MAX_BOILING_RANGE_K,
    ONE_ATM_PA,
    Component,
    MixtureThermalExpert,
    bubble_temperature,
    dew_temperature,
    unifac_model,
    vapour_pressure,
)

pytestmark = [
    requires_rdkit,
    pytest.mark.skipif(not unifac_available(), reason="modified UNIFAC is unavailable"),
]

ROOM = Conditions(
    temperature=Quantity(value=298.15, unit="K"), pressure=Quantity(value=1.0, unit="atm")
)

WATER = "O"
ETHANOL = "CCO"
METHANOL = "CO"
ACETONE = "CC(C)=O"
BENZENE = "c1ccccc1"
TOLUENE = "Cc1ccccc1"
HEXANE = "CCCCCC"
DECANE = "CCCCCCCCCC"
HEXADECANE = "CCCCCCCCCCCCCCCC"
ETHYL_ACETATE = "CCOC(C)=O"
PROPAN_2_OL = "CC(C)O"
M_XYLENE = "Cc1cccc(C)c1"
P_XYLENE = "Cc1ccc(C)cc1"
#: Isophorone. It has a CAS but the DDBST compilation carries no modified-UNIFAC
#: assignment for it, which is exactly the gap the boiling point refuses on.
NO_UNIFAC_GROUPS = "CC1=CC(=O)C(C)(C)CC1"

#: One expert instance across the module: it caches the molecular panel per
#: component, and rebuilding it per test turns a two-second suite into a minute.
EXPERT = MixtureThermalExpert()


def blend(pairs, basis=FractionBasis.MOLE, conditions=ROOM, phase=PhaseAssumption.UNKNOWN):
    roles = (ComponentRole.SOLVENT, ComponentRole.CO_SOLVENT, ComponentRole.ADDITIVE)
    components = tuple(
        MixtureComponent(
            role=roles[index % len(roles)],
            fraction=fraction,
            molecule=MoleculeSpec(smiles=smiles),
        )
        for index, (smiles, fraction) in enumerate(pairs)
    )
    return Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(components=components, basis=basis, phase_assumption=phase),
        conditions=conditions,
    )


def predict(candidate, properties=None):
    wanted = frozenset(properties or EXPERT.supported_properties)
    request = PredictionRequest(
        candidate=candidate, properties=wanted, conditions=candidate.conditions
    )
    return {p.property: p for p in EXPERT.predict(request)}


def one(candidate, prop):
    return predict(candidate, [prop])[prop]


# --------------------------------------------------------------------------
# The measured claim: five held-out azeotropes
# --------------------------------------------------------------------------

#: Literature azeotrope and narrow-blend boiling temperatures at 1 atm. Nothing
#: in this module is fitted to them, so they are held out by construction; they
#: are the numbers the module docstring quotes.
AZEOTROPES = [
    ((ACETONE, 0.80), (METHANOL, 0.20), 328.65),
    ((ETHYL_ACETATE, 0.69), (ETHANOL, 0.31), 344.95),
    ((BENZENE, 0.55), (ETHANOL, 0.45), 341.05),
    ((PROPAN_2_OL, 0.685), (WATER, 0.315), 353.55),
    ((M_XYLENE, 0.5), (P_XYLENE, 0.5), 411.9),
]


@pytest.mark.parametrize("first,second,reference", AZEOTROPES)
def test_bubble_point_reproduces_a_literature_azeotrope(first, second, reference):
    prediction = one(blend([first, second]), "normal_boiling_point")
    assert prediction.is_usable, prediction.notes
    value = prediction.quantity.to("K").value
    assert abs(value - reference) < 3.0, f"{value} vs {reference}"
    # An error bar that does not contain a held-out measurement is not an error
    # bar; this is the check that keeps the propagation honest.
    assert abs(value - reference) <= prediction.uncertainty.std


def test_held_out_mean_absolute_error_is_the_number_the_docstring_quotes():
    errors = []
    for first, second, reference in AZEOTROPES:
        prediction = one(blend([first, second]), "normal_boiling_point")
        assert prediction.is_usable, prediction.notes
        errors.append(abs(prediction.quantity.to("K").value - reference))
    assert sum(errors) / len(errors) < 0.5


# --------------------------------------------------------------------------
# The refusals
# --------------------------------------------------------------------------


def test_a_wide_boiling_blend_refuses_and_names_both_temperatures():
    """The refusal this expert exists for.

    Hexane and decane boil 105 K apart and the blend boils over sixty. A single
    number under the name ``normal_boiling_point`` would let it satisfy a hard
    window that most of its mass never meets.
    """
    prediction = one(blend([(HEXANE, 0.5), (DECANE, 0.5)]), "normal_boiling_point")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    reason = prediction.notes[0]
    assert "364" in reason and "424" in reason, reason
    assert "range" in reason


def test_a_wide_boiling_blend_also_refuses_the_enthalpy_of_vaporisation():
    """Same gate, because it is the same missing thing: one temperature."""
    prediction = one(blend([(HEXANE, 0.5), (DECANE, 0.5)]), "enthalpy_vaporization")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "boils over a range" in prediction.notes[0]


def test_benzene_and_toluene_are_refused_even_though_they_are_near_ideal():
    """Near-ideal is not narrow-boiling, and this is the distinction.

    Benzene and toluene mix almost ideally - the activity coefficients are
    within a per cent of one - and the blend still boils over about seven
    kelvin because their pure boiling points are thirty apart. Ideality and a
    single boiling temperature are different questions.
    """
    prediction = one(blend([(BENZENE, 0.5), (TOLUENE, 0.5)]), "normal_boiling_point")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "boils over a range" in prediction.notes[0]


def test_an_azeotrope_beats_the_pure_boiling_point_spread_proxy():
    """Why the gate is the computed range and not a cheap proxy.

    Ethanol and water boil 22 K apart, so any rule based on the spread of the
    pure-component boiling points refuses them. The mixture boils over about
    four kelvin, because the azeotrope pulls bubble and dew together, and it
    really does have a boiling temperature.
    """
    prediction = one(blend([(ETHANOL, 0.5), (WATER, 0.5)]), "normal_boiling_point")
    assert prediction.is_usable, prediction.notes
    assert 350.0 < prediction.quantity.to("K").value < 356.0


def test_a_liquid_liquid_split_refuses_the_boiling_point():
    prediction = one(blend([(HEXANE, 0.5), (WATER, 0.5)]), "normal_boiling_point")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "unstable" in prediction.notes[0]


def test_a_component_with_no_unifac_assignment_refuses_only_the_vle_properties():
    """The refusal has to be surgical.

    Isophorone has no DDBST group assignment, so no activity coefficient and no
    bubble point. That says nothing about a mole-fraction average of critical
    constants, which must still come back.
    """
    out = predict(blend([(BENZENE, 0.5), (NO_UNIFAC_GROUPS, 0.5)]))
    for prop in ("normal_boiling_point", "enthalpy_vaporization"):
        assert out[prop].status is PredictionStatus.UNSUPPORTED
        assert "DDBST" in out[prop].notes[0]
    for prop in ("critical_temperature", "critical_volume", "heat_capacity_gas"):
        assert out[prop].is_usable, out[prop].notes


def test_logp_of_a_formulation_is_refused():
    prediction = one(blend([(ETHANOL, 0.5), (HEXANE, 0.5)]), "logp")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "partition coefficient" in prediction.notes[0]


def test_logp_of_a_one_component_formulation_is_the_pure_substance_value():
    """Refusing a pure substance's partition coefficient would be pedantry."""
    prediction = one(blend([(ETHANOL, 1.0)]), "logp")
    assert prediction.is_usable, prediction.notes
    panel = EXPERT.component_panel(ETHANOL, ROOM)
    assert prediction.quantity.value == pytest.approx(panel["logp"].quantity.value)


def test_a_polymer_component_refuses_every_property():
    """Rule seven: a polymer is not its monomer, and it has no critical point."""
    candidate = Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=(
                MixtureComponent(
                    role=ComponentRole.SOLVENT, fraction=0.5, molecule=MoleculeSpec(smiles=TOLUENE)
                ),
                MixtureComponent(
                    role=ComponentRole.BINDER,
                    fraction=0.5,
                    polymer=PolymerSpec(monomers=(MonomerUnit(smiles="[*]CC([*])c1ccccc1"),)),
                ),
            ),
            basis=FractionBasis.MASS,
        ),
        conditions=ROOM,
    )
    out = predict(candidate)
    assert out, "the expert produced nothing at all"
    for prop, prediction in out.items():
        assert prediction.status is PredictionStatus.UNSUPPORTED, prop
        assert "polymer" in prediction.notes[0]
    assert not EXPERT.assess_domain(candidate).in_domain


def test_an_ionic_component_refuses_every_property():
    candidate = Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=(
                MixtureComponent(
                    role=ComponentRole.SOLVENT, fraction=0.9, molecule=MoleculeSpec(smiles=WATER)
                ),
                MixtureComponent(
                    role=ComponentRole.ADDITIVE,
                    fraction=0.1,
                    molecule=MoleculeSpec(smiles="CC(=O)[O-]", charge=-1),
                ),
            ),
            basis=FractionBasis.MOLE,
        ),
        conditions=ROOM,
    )
    for prediction in predict(candidate).values():
        assert prediction.status is PredictionStatus.UNSUPPORTED
        assert "formal charge" in prediction.notes[0]


def test_a_missing_component_property_refuses_that_property_only():
    """Water defeats Joback, so a water blend has no critical pressure.

    The point is that it refuses rather than averaging over the components it
    happens to have, which would describe a different formulation.
    """
    out = predict(blend([(ETHANOL, 0.5), (WATER, 0.5)]))
    assert out["critical_pressure"].status is PredictionStatus.UNSUPPORTED
    assert "'O'" in out["critical_pressure"].notes[0]
    assert out["molar_refractivity"].is_usable


def test_kays_rule_refuses_without_the_rule_it_is_checked_against():
    """No independent check means no defensible error bar, so no number.

    Kay's disagreement with Li's rule is the only evidence this module has
    about Kay's model form. Water has no critical volume from the panel, so
    Li's rule cannot be evaluated, and a Kay temperature would then be carrying
    an uncertainty with nothing behind it.
    """
    prediction = one(blend([(ETHANOL, 0.5), (WATER, 0.5)]), "critical_temperature")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "critical volume" in prediction.notes[0]


def test_an_emulsion_is_out_of_domain():
    candidate = blend([(WATER, 0.5), (HEXANE, 0.5)], phase=PhaseAssumption.EMULSION)
    domain = EXPERT.assess_domain(candidate)
    assert not domain.in_domain
    assert domain.score == pytest.approx(0.3)


def test_a_recipe_that_cannot_reach_a_mole_basis_refuses_everything():
    """Every rule here is mole-fraction weighted; a mass fraction is not one.

    Isophorone has no liquid density from the panel, so a volume-basis recipe
    containing it cannot be converted, and using the volume fractions as mole
    fractions would be an error of tens of per cent applied silently to every
    property at once.
    """
    candidate = blend(
        [(BENZENE, 0.5), (NO_UNIFAC_GROUPS, 0.5)], basis=FractionBasis.VOLUME
    )
    out = predict(candidate)
    for prop, prediction in out.items():
        assert prediction.status is PredictionStatus.UNSUPPORTED, prop
        assert "mole basis" in prediction.notes[0], prop


# --------------------------------------------------------------------------
# One rule per property, and each one says which
# --------------------------------------------------------------------------


def test_every_property_carries_its_own_mixing_rule_note():
    """A single generic note across all of them would be a lie for half."""
    out = predict(blend([(ACETONE, 0.8), (METHANOL, 0.2)]))
    usable = {k: v for k, v in out.items() if v.is_usable}
    assert len(usable) >= 8
    first_notes = [v.notes[0] for v in usable.values()]
    assert len(set(first_notes)) == len(first_notes)


def test_heat_capacity_is_exactly_additive_and_carries_no_rule_error():
    """Not "nearly" exact: an ideal-gas mixture has no interactions at all."""
    prediction = one(blend([(BENZENE, 0.5), (TOLUENE, 0.5)]), "heat_capacity_gas")
    assert prediction.is_usable, prediction.notes
    panel = [EXPERT.component_panel(s, ROOM) for s in (BENZENE, TOLUENE)]
    expected = sum(
        0.5 * p["heat_capacity_gas"].quantity.to("J/mol/K").value for p in panel
    )
    assert prediction.quantity.to("J/mol/K").value == pytest.approx(expected, rel=1e-12)
    # The whole bar is the components' own, so it equals their weighted sum.
    component_sum = sum(0.5 * p["heat_capacity_gas"].uncertainty.std for p in panel)
    assert prediction.uncertainty.std == pytest.approx(component_sum, rel=1e-12)
    assert "exact" in prediction.notes[0]


def test_synthetic_accessibility_is_the_maximum_not_the_mean():
    """Every component has to be made, so the hardest one gates the blend."""
    easy, hard = BENZENE, "O=C(C)Oc1ccccc1C(=O)O"
    scores = [
        EXPERT.component_panel(s, ROOM)["synthetic_accessibility"].quantity.value
        for s in (easy, hard)
    ]
    prediction = one(blend([(easy, 0.9), (hard, 0.1)]), "synthetic_accessibility")
    assert prediction.is_usable, prediction.notes
    assert prediction.quantity.value == pytest.approx(max(scores))
    assert prediction.quantity.value > sum(scores) / 2
    assert "MAXIMUM" in prediction.notes[0]


def test_molar_refractivity_is_additive_and_its_bar_is_the_components():
    """Lorentz-Lorenz additivity is tight; the Crippen inputs are not."""
    prediction = one(blend([(BENZENE, 0.4), (TOLUENE, 0.6)]), "molar_refractivity")
    assert prediction.is_usable, prediction.notes
    panel = [EXPERT.component_panel(s, ROOM) for s in (BENZENE, TOLUENE)]
    expected = sum(
        x * p["molar_refractivity"].quantity.to("cm^3/mol").value
        for x, p in zip((0.4, 0.6), panel)
    )
    assert prediction.quantity.to("cm^3/mol").value == pytest.approx(expected, rel=1e-12)
    # The rule contributes half a per cent; the components contribute 2.5.
    assert prediction.uncertainty.std < 1.02 * 2.5


def test_the_enthalpy_is_reported_at_the_bubble_point_not_a_component_boiling_point():
    prediction = one(blend([(ACETONE, 0.8), (METHANOL, 0.2)]), "enthalpy_vaporization")
    assert prediction.is_usable, prediction.notes
    reference = prediction.conditions.temperature_k
    boiling = one(blend([(ACETONE, 0.8), (METHANOL, 0.2)]), "normal_boiling_point")
    assert reference == pytest.approx(boiling.quantity.to("K").value, abs=1e-6)
    assert "bubble point" in prediction.notes[0]


def test_the_enthalpy_is_not_a_plain_average_of_the_tabulated_values():
    """Watson correction plus the excess enthalpy, and both have to move it."""
    pairs = [(ACETONE, 0.5), ("ClC(Cl)Cl", 0.5)]
    prediction = one(blend(pairs), "enthalpy_vaporization")
    assert prediction.is_usable, prediction.notes
    naive = sum(
        x * EXPERT.component_panel(s, ROOM)["enthalpy_vaporization"].quantity.to("J/mol").value
        for s, x in pairs
    )
    value = prediction.quantity.to("J/mol").value
    assert abs(value - naive) > 500.0, (value, naive)


# --------------------------------------------------------------------------
# Kay's rule, and the disagreement it is measured against
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "first,second",
    [(BENZENE, TOLUENE), (BENZENE, METHANOL), (HEXANE, ETHANOL)],
)
def test_the_kay_error_bar_actually_contains_the_alternative_rule(first, second):
    """The stated uncertainty has to contain the evidence it is built from."""
    candidate = blend([(first, 0.5), (second, 0.5)])
    prediction = one(candidate, "critical_temperature")
    assert prediction.is_usable, prediction.notes
    components = EXPERT.components(candidate.mixture, PredictionRequest(
        candidate=candidate, properties=frozenset(), conditions=ROOM
    ))
    alternative = EXPERT._kay_alternative("critical_temperature", components, [0.5, 0.5])
    assert alternative is not None
    gap = abs(prediction.quantity.to("K").value - alternative)
    assert gap <= prediction.uncertainty.std + 1e-9


def test_kay_and_li_disagreeing_badly_marks_the_constant_out_of_domain():
    """Methanol and hexadecane differ in size by a factor of five.

    That is exactly where a mole-fraction pseudo-critical stops describing the
    mixture, and the two published rules disagree by thirteen per cent. A wide
    bar is not enough; the prediction is marked out of domain.
    """
    prediction = one(blend([(METHANOL, 0.5), (HEXADECANE, 0.5)]), "critical_temperature")
    assert prediction.status is PredictionStatus.OUT_OF_DOMAIN
    assert prediction.quantity is not None
    assert any("disagree" in w for w in prediction.applicability.warnings)


def test_similar_components_stay_in_domain():
    prediction = one(blend([(BENZENE, 0.5), (TOLUENE, 0.5)]), "critical_temperature")
    assert prediction.status is PredictionStatus.OK


# --------------------------------------------------------------------------
# Cross-expert consistency and the contract
# --------------------------------------------------------------------------


def test_this_molar_volume_and_the_mixture_experts_density_describe_one_liquid():
    """Both are volume additivity over the same component densities.

    If they disagreed, a formulation would have two mutually exclusive
    densities in one report, and nothing in the type system would notice.
    """
    pairs = [(ETHANOL, 0.4), (WATER, 0.6)]
    candidate = blend(pairs)
    volume = one(candidate, "molar_volume_liquid").quantity.to("m^3/mol").value

    request = PredictionRequest(
        candidate=candidate, properties=frozenset({"liquid_density"}), conditions=ROOM
    )
    blended = MixtureExpert().predict(request)[0]
    assert blended.is_usable, blended.notes

    molar_mass = sum(
        x * EXPERT.component_panel(s, ROOM)["molar_mass"].quantity.to("g/mol").value
        for s, x in pairs
    )
    implied = (molar_mass * 1e-3) / volume
    assert implied == pytest.approx(blended.quantity.to("kg/m^3").value, rel=1e-9)


def test_a_one_component_formulation_reproduces_the_pure_substance_exactly():
    """Including the uncertainty: there is no mixing, so no mixing error."""
    panel = EXPERT.component_panel(ETHANOL, ROOM)
    out = predict(blend([(ETHANOL, 1.0)]))
    for prop in ("critical_temperature", "critical_pressure", "critical_volume",
                 "heat_capacity_gas", "molar_refractivity", "normal_boiling_point"):
        prediction = out[prop]
        assert prediction.is_usable, (prop, prediction.notes)
        unit = get_property(prop).canonical_unit
        assert prediction.quantity.to(unit).value == pytest.approx(
            panel[prop].quantity.to(unit).value, rel=1e-12
        )
        assert prediction.uncertainty.converted(
            prediction.quantity.unit, unit
        ).std == pytest.approx(
            panel[prop].uncertainty.converted(panel[prop].quantity.unit, unit).std, rel=1e-12
        )


def test_every_prediction_converts_to_its_canonical_unit_and_carries_a_spread():
    out = predict(blend([(ACETONE, 0.8), (METHANOL, 0.2)]))
    assert set(out) == set(EXPERT.supported_properties)
    for prop, prediction in out.items():
        if not prediction.is_usable:
            assert prediction.quantity is None
            assert prediction.notes and len(prediction.notes[0]) > 40
            continue
        canonical = get_property(prop).canonical_unit
        assert prediction.quantity.to(canonical) is not None
        assert prediction.uncertainty.std is not None and prediction.uncertainty.std > 0.0
        assert prediction.uncertainty.basis


def test_the_unit_conversion_is_pinned_not_assumed():
    """A mole-fraction sum in one unit has to survive being read in another.

    The registry's canonical unit for molar refractivity is cm^3/mol and for
    the critical volume it is m^3/mol; a rule written in the wrong one is a
    factor of a million and would sail past every other test here.
    """
    candidate = blend([(BENZENE, 0.5), (TOLUENE, 0.5)])

    refractivity = one(candidate, "molar_refractivity")
    in_cm3 = refractivity.quantity.to("cm^3/mol").value
    in_m3 = refractivity.quantity.to("m^3/mol").value
    assert in_m3 == pytest.approx(in_cm3 * 1e-6, rel=1e-12)
    assert 20.0 < in_cm3 < 40.0  # benzene is 26 cm^3/mol, toluene 31

    volume = one(candidate, "critical_volume")
    assert 1e-4 < volume.quantity.to("m^3/mol").value < 5e-4
    assert volume.quantity.to("cm^3/mol").value == pytest.approx(
        volume.quantity.to("m^3/mol").value * 1e6, rel=1e-12
    )

    boiling = one(blend([(ACETONE, 0.8), (METHANOL, 0.2)]), "normal_boiling_point")
    kelvin = boiling.quantity.to("K")
    celsius = boiling.quantity.to("degC")
    assert celsius.value == pytest.approx(kelvin.value - 273.15, abs=1e-6)
    # Difference semantics: a 2 K spread stays 2 in degrees Celsius.
    assert boiling.uncertainty.converted("K", "degC").std == pytest.approx(
        boiling.uncertainty.std, rel=1e-12
    )


def test_the_declared_basis_changes_the_answer():
    """A mass fraction is not a mole fraction, and the conversion is real."""
    pairs = [(ETHANOL, 0.5), (WATER, 0.5)]
    by_mole = one(blend(pairs, FractionBasis.MOLE), "molar_volume_liquid")
    by_mass = one(blend(pairs, FractionBasis.MASS), "molar_volume_liquid")
    assert by_mole.is_usable and by_mass.is_usable
    mole_value = by_mole.quantity.to("m^3/mol").value
    mass_value = by_mass.quantity.to("m^3/mol").value
    assert abs(mole_value - mass_value) / mole_value > 0.2


# --------------------------------------------------------------------------
# The machinery underneath
# --------------------------------------------------------------------------


def test_this_modules_unifac_constructor_agrees_with_the_activity_experts():
    """Two constructors of one model is how two modules come to disagree."""
    groups = [unifac_groups(s) for s in (ETHANOL, WATER)]
    fractions = [0.3, 0.7]
    mine = list(unifac_model(groups, fractions, 340.0).gammas())
    theirs = _gammas(groups, fractions, 340.0)
    assert mine == pytest.approx(theirs, rel=1e-12)


def test_the_bubble_point_is_below_the_dew_point_and_both_bracket_the_components():
    groups = [unifac_groups(s) for s in (HEXANE, DECANE)]
    pressures = [
        vapour_pressure(
            s,
            EXPERT.component_panel(s, ROOM)["normal_boiling_point"].quantity.to("K").value,
            EXPERT.component_panel(s, ROOM)["critical_temperature"].quantity.to("K").value,
            EXPERT.component_panel(s, ROOM)["critical_pressure"].quantity.to("Pa").value,
        )
        for s in (HEXANE, DECANE)
    ]
    ceiling = 500.0
    bubble = bubble_temperature(groups, pressures, [0.5, 0.5], ONE_ATM_PA, ceiling)
    dew = dew_temperature(groups, pressures, [0.5, 0.5], ONE_ATM_PA, ceiling)
    assert bubble is not None and dew is not None
    assert bubble < dew
    assert dew - bubble > MAX_BOILING_RANGE_K
    # The range has to sit inside the pure boiling points, 342 K and 447 K.
    assert 341.0 < bubble and dew < 448.0


def test_a_wider_component_boiling_point_widens_the_bubble_point_bar():
    """The propagation is not decorative: it responds to the panel's own bar.

    A component whose boiling point came from Joback carries 12.9 K rather than
    the half kelvin a measured value does, and that has to reach the answer.
    Substituted here rather than hunted for, because a molecule UNIFAC can
    assign groups to almost always has a measured boiling point too - which is
    itself worth knowing about this expert's coverage.
    """
    smiles = [ACETONE, METHANOL]
    fractions = [0.8, 0.2]

    def components(std):
        out = []
        for name, fraction in zip(smiles, fractions):
            panel = dict(EXPERT.component_panel(name, ROOM))
            boiling = panel["normal_boiling_point"]
            panel["normal_boiling_point"] = boiling.model_copy(
                update={
                    "uncertainty": Uncertainty(
                        std=std, kind=UncertaintyKind.EPISTEMIC, basis="substituted in a test"
                    )
                }
            )
            out.append(
                Component(smiles=name, role="solvent", fraction=fraction, predictions=panel)
            )
        return out

    groups = [unifac_groups(s) for s in smiles]
    reference = components(0.5)
    pressures = [
        vapour_pressure(
            c.smiles,
            c.value("normal_boiling_point"),
            c.value("critical_temperature"),
            c.value("critical_pressure"),
        )
        for c in reference
    ]
    ceiling = min(c.value("critical_temperature") for c in reference) * 0.999
    bubble = bubble_temperature(groups, pressures, fractions, ONE_ATM_PA, ceiling)

    narrow, _ = EXPERT._bubble_uncertainty(
        groups, pressures, fractions, reference, bubble, ceiling
    )
    wide, _ = EXPERT._bubble_uncertainty(
        groups, pressures, fractions, components(12.9), bubble, ceiling
    )
    assert 1.5 < narrow < 3.0
    assert wide > 4 * narrow
    assert math.isfinite(wide)


def test_the_expert_declares_no_dependencies_on_purpose():
    """Everything it needs belongs to the components, not to the candidate."""
    assert MixtureThermalExpert.dependencies == frozenset()
    assert MixtureThermalExpert.supported_classes == frozenset({MaterialClass.MIXTURE})
    assert len(MixtureThermalExpert.supported_properties) == 10


# --------------------------------------------------------------------------
# The boiling range does not stop existing because it passed the gate
# --------------------------------------------------------------------------


def _vle_setup(pairs):
    """``(groups, pressures, components, fractions, ceiling)`` for a blend.

    The same objects ``_vle_property`` assembles, so a test can ask what the
    model-error propagation alone would have said and compare it against what
    the expert actually reports.
    """
    candidate = blend(pairs)
    request = PredictionRequest(
        candidate=candidate, properties=frozenset(), conditions=ROOM
    )
    components = EXPERT.components(candidate.mixture, request)
    groups = [unifac_groups(c.smiles) for c in components]
    pressures = [
        vapour_pressure(
            c.smiles,
            c.value("normal_boiling_point"),
            c.value("critical_temperature"),
            c.value("critical_pressure"),
        )
        for c in components
    ]
    fractions = [c.fraction for c in components]
    ceiling = min(c.value("critical_temperature") for c in components) * 0.999
    return groups, pressures, components, fractions, ceiling


def _dew_from_note(prediction):
    for note in prediction.notes:
        if "The dew point is" in note:
            return float(note.split("The dew point is")[1].split("K")[0])
    raise AssertionError("no dew point in the notes")


def test_the_reported_bar_carries_the_boiling_range_that_survived_the_gate():
    """A range under five kelvin is small enough to summarise, not absent.

    Reporting the bubble point with only the model error on it told a ranker
    that ethanol/water boils at 353.0 +/- 1.8 K when this module had itself
    computed that it goes on boiling to 357.4 K.  The range is evidence in
    hand, so it belongs in the bar.
    """
    pairs = [(ETHANOL, 0.5), (WATER, 0.5)]
    prediction = one(blend(pairs), "normal_boiling_point")
    assert prediction.is_usable, prediction.notes

    groups, pressures, components, fractions, ceiling = _vle_setup(pairs)
    bubble = bubble_temperature(groups, pressures, fractions, ONE_ATM_PA, ceiling)
    dew = dew_temperature(groups, pressures, fractions, ONE_ATM_PA, ceiling)
    model_only, _ = EXPERT._bubble_uncertainty(
        groups, pressures, fractions, components, bubble, ceiling
    )

    expected = math.hypot(model_only, (dew - bubble) / math.sqrt(12.0))
    assert prediction.uncertainty.std == pytest.approx(expected, rel=1e-9)
    assert prediction.uncertainty.std > model_only
    assert "boiling range" in prediction.uncertainty.basis


def test_an_azeotrope_keeps_the_tight_bar_it_has_earned():
    """The range term has to vanish where the range really does.

    m- and p-xylene boil within a kelvin of each other and the blend boils
    over a twentieth of one.  A range term that charged such a mixture
    anything would be punishing it for a spread it does not have.
    """
    pairs = [(M_XYLENE, 0.5), (P_XYLENE, 0.5)]
    prediction = one(blend(pairs), "normal_boiling_point")
    assert prediction.is_usable, prediction.notes

    groups, pressures, components, fractions, ceiling = _vle_setup(pairs)
    bubble = bubble_temperature(groups, pressures, fractions, ONE_ATM_PA, ceiling)
    dew = dew_temperature(groups, pressures, fractions, ONE_ATM_PA, ceiling)
    assert dew - bubble < 0.2
    model_only, _ = EXPERT._bubble_uncertainty(
        groups, pressures, fractions, components, bubble, ceiling
    )
    assert prediction.uncertainty.std == pytest.approx(model_only, rel=1e-3)


@pytest.mark.parametrize(
    "pairs",
    [
        [(ETHANOL, 0.5), (WATER, 0.5)],
        [(ACETONE, 0.5), (BENZENE, 0.5)],
        [(ETHYL_ACETATE, 0.5), ("CCCCCCC", 0.5)],
        [(ACETONE, 0.8), (METHANOL, 0.2)],
    ],
)
def test_the_dew_point_lies_inside_about_two_sigma_of_the_reported_value(pairs):
    """The whole point of the widened bar, checked where it is tightest.

    These are the widest ranges the five-kelvin gate admits.  Two sigma is not
    one, because the value is the bubble point and sits at the bottom of the
    range rather than in its middle - which is why the note names the dew point
    as well.
    """
    prediction = one(blend(pairs), "normal_boiling_point")
    assert prediction.is_usable, prediction.notes
    value = prediction.quantity.to("K").value
    dew = _dew_from_note(prediction)
    assert dew - value <= 2.05 * prediction.uncertainty.std


def test_the_enthalpy_carries_the_uncertainty_of_the_temperature_it_is_stated_at():
    """It is Watson-corrected to the bubble point, so that bar is an input.

    The bubble-point spread used to be computed, handed to the enthalpy and
    dropped on the floor.
    """
    prediction = one(blend([(ACETONE, 0.8), (METHANOL, 0.2)]), "enthalpy_vaporization")
    assert prediction.is_usable, prediction.notes
    assert "bubble point this enthalpy is reported at" in prediction.uncertainty.basis

    # Re-derive it: the Watson exponent is 0.38, so d(ln H)/dT is
    # -0.38 / (Tc - T), and the term is that times the bubble-point bar.
    boiling = one(blend([(ACETONE, 0.8), (METHANOL, 0.2)]), "normal_boiling_point")
    bubble = boiling.quantity.to("K").value
    spread = boiling.uncertainty.std
    panel = [EXPERT.component_panel(s, ROOM) for s in (ACETONE, METHANOL)]
    expected = abs(
        sum(
            x * p["enthalpy_vaporization"].quantity.to("J/mol").value
            * 0.38 * spread / (p["critical_temperature"].quantity.to("K").value - bubble)
            for x, p in zip((0.8, 0.2), panel)
        )
    )
    # Within a factor of two of the linearised estimate: the real term also
    # moves H^E, which the hand derivative above ignores.
    term = float(prediction.uncertainty.basis.split("and with ")[1].split(" J/mol")[0])
    assert 0.5 * expected < term < 2.0 * expected


# --------------------------------------------------------------------------
# A pure substance is not a blend, and an absent bar is not a zero one
# --------------------------------------------------------------------------


def test_a_one_component_formulation_keeps_a_constant_that_needs_no_mixing_rule():
    """Kay's rule over one component is the identity.

    Water has no critical volume from the panel, so Li's rule cannot be
    evaluated - but Li's rule is only there to measure Kay's model-form error,
    and a formulation with one component has had no mixing rule applied to it
    at all.  Refusing here was refusing a pure substance's critical temperature
    over an input that no rule in the chain needs.
    """
    prediction = one(blend([(WATER, 1.0)]), "critical_temperature")
    assert prediction.is_usable, prediction.notes
    panel = EXPERT.component_panel(WATER, ROOM)
    assert prediction.quantity.to("K").value == pytest.approx(
        panel["critical_temperature"].quantity.to("K").value, rel=1e-12
    )
    assert prediction.uncertainty.std == pytest.approx(
        panel["critical_temperature"].uncertainty.std, rel=1e-12
    )
    assert "one component" in prediction.uncertainty.basis

    # And the two-component blend it was confused with still refuses.
    assert (
        one(blend([(ETHANOL, 0.5), (WATER, 0.5)]), "critical_temperature").status
        is PredictionStatus.UNSUPPORTED
    )


def _panel_without_uncertainty(smiles, prop):
    panel = dict(EXPERT.component_panel(smiles, ROOM))
    panel[prop] = panel[prop].model_copy(
        update={"uncertainty": Uncertainty.unknown("stripped in a test")}
    )
    return panel


def test_a_density_with_no_uncertainty_refuses_the_molar_volume():
    """Reading an absent bar as zero would hand this blend the tightest bar in
    the run, on the one input that carries the whole component term."""
    candidate = blend([(ETHANOL, 0.4), (WATER, 0.6)])
    request = PredictionRequest(
        candidate=candidate, properties=frozenset(), conditions=ROOM
    )
    domain = EXPERT.assess_domain(candidate)
    components = [
        Component(
            smiles=s,
            role="solvent",
            fraction=x,
            predictions=_panel_without_uncertainty(s, "liquid_density")
            if s == WATER
            else EXPERT.component_panel(s, ROOM),
        )
        for s, x in ((ETHANOL, 0.4), (WATER, 0.6))
    ]
    prediction = EXPERT._molar_volume(
        "molar_volume_liquid", components, [0.4, 0.6], request, domain
    )
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "liquid density" in prediction.notes[0]
    assert "ranked as if it were exact" in prediction.notes[0]


def test_an_accessibility_with_no_uncertainty_refuses_rather_than_inventing_one():
    """The basis string promises the selected component's own number."""
    candidate = blend([(BENZENE, 0.5), (ETHANOL, 0.5)])
    request = PredictionRequest(
        candidate=candidate, properties=frozenset(), conditions=ROOM
    )
    domain = EXPERT.assess_domain(candidate)
    components = [
        Component(
            smiles=s,
            role="solvent",
            fraction=0.5,
            predictions=_panel_without_uncertainty(s, "synthetic_accessibility"),
        )
        for s in (BENZENE, ETHANOL)
    ]
    prediction = EXPERT._accessibility(
        "synthetic_accessibility", components, request, domain
    )
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "carries no uncertainty" in prediction.notes[0]


def test_raoult_sits_above_modified_unifac_for_the_pairs_the_docstring_names():
    """The direction, not only the size.

    The docstring used to quote these four shifts with every sign inverted,
    which is the kind of error that survives review because the magnitudes are
    right.  Ideality *over*-predicts the boiling point of a positive-deviation
    blend and under-predicts a negative-deviation one.
    """
    from scipy.optimize import brentq

    def raoult(pairs):
        groups, pressures, components, fractions, ceiling = _vle_setup(pairs)
        residual = lambda t: sum(  # noqa: E731
            x * vp(t) for x, vp in zip(fractions, pressures)
        ) - ONE_ATM_PA
        return brentq(residual, 200.0, ceiling, xtol=1e-8)

    for pairs, expected in [
        ([(HEXANE, 0.5), (ETHANOL, 0.5)], +15.3),
        ([(ETHANOL, 0.5), (WATER, 0.5)], +7.0),
        ([(ACETONE, 0.5), ("ClC(Cl)Cl", 0.5)], -5.2),
        ([(ACETONE, 0.8), (METHANOL, 0.2)], +2.5),
    ]:
        groups, pressures, components, fractions, ceiling = _vle_setup(pairs)
        bubble = bubble_temperature(groups, pressures, fractions, ONE_ATM_PA, ceiling)
        assert raoult(pairs) - bubble == pytest.approx(expected, abs=0.1)
