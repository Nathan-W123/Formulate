"""Cohesive energy density, and the bulk modulus it implies.

Three things are tested as hard as the predictions themselves.

The **unit conversion**, because getting Pa^0.5 and MPa^0.5 the wrong way round
is a documented past bug in this repository - ``dissolution.py`` carries the
scar - and squaring the parameter turns a factor of a thousand into a factor of
a million while the answer still looks like a cohesive energy density.

The **two constants**, because the brief this module was written against said
``n`` is about eight to eleven for liquids and polymers alike, and it is not:
liquids measure 2.8. A test here breaks on anyone who later "simplifies" the
two constants back into one.

The **refusals**, because in this repository a refusal is a feature. Every gate
below exists because a measurement said so, and the measurement is in the test.
"""

from __future__ import annotations

import math

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import (
    Candidate,
    ComponentRole,
    Conditions,
    MaterialClass,
    MixtureComponent,
    MixtureSpec,
    MoleculeSpec,
    MonomerUnit,
    PolymerSpec,
    molecule_candidate,
    polymer_candidate,
)
from formulate.core.prediction import Prediction, PredictionStatus
from formulate.core.properties import get_property
from formulate.core.quantity import Quantity, Uncertainty, UncertaintyKind
from formulate.experts.base import PredictionRequest
from formulate.experts.cohesion import (
    BULK_MODULUS_T_OUTER,
    EXCLUDED_POLYMERS,
    HYDROGEN_BOND_FRACTION_LIMIT,
    LIQUID_REFERENCES,
    PA_ROOT_PER_MPA_ROOT,
    POLYMER_REFERENCES,
    CohesionExpert,
    ced_from_hansen,
    ced_from_hildebrand,
    cohesion_model,
    fluorine_fraction,
    geometric_mean,
    hydrogen_bond_fraction,
    worst_leave_one_out,
)

TOLUENE = "Cc1ccccc1"
HEXANE = "CCCCCC"
WATER = "O"
DMSO = "CS(C)=O"
PS_UNIT = "[*]CC(c1ccccc1)[*]"
PTFE_UNIT = "[*]C(F)(F)C(F)(F)[*]"

ROOM = Conditions(
    temperature=Quantity(value=298.15, unit="K"), pressure=Quantity(value=1.0, unit="atm")
)

BOTH = frozenset({"cohesive_energy_density", "bulk_modulus"})
CED = frozenset({"cohesive_energy_density"})
BULK = frozenset({"bulk_modulus"})


# -- scaffolding -----------------------------------------------------------


def _upstream(prop: str, value: float, unit: str, std: float | None = None) -> Prediction:
    """A usable upstream prediction, as the dispatcher would hand one over."""
    return Prediction(
        property=prop,
        quantity=Quantity(value=value, unit=unit),
        uncertainty=Uncertainty(std=std, kind=UncertaintyKind.EPISTEMIC, basis="test fixture"),
        status=PredictionStatus.OK,
        expert_id="test",
    )


def _hansen_context(dd: float, dp: float, dh: float, unit: str = "MPa^0.5", std=None):
    return {
        "hansen_dispersion": _upstream("hansen_dispersion", dd, unit, std),
        "hansen_polar": _upstream("hansen_polar", dp, unit, std),
        "hansen_hydrogen_bonding": _upstream("hansen_hydrogen_bonding", dh, unit, std),
    }


def _predict(candidate, properties=BOTH, context=None, conditions=ROOM):
    """Every prediction this expert makes for one candidate, keyed by property."""
    expert = CohesionExpert()
    out = expert.predict(
        PredictionRequest(
            candidate=candidate,
            properties=properties,
            conditions=conditions,
            context=context or {},
        )
    )
    return {p.property: p for p in out}


def _toluene(**kwargs):
    """Toluene with its measured Hildebrand parameter and Hansen triple."""
    context = {
        "hildebrand_solubility_parameter": _upstream(
            "hildebrand_solubility_parameter", 18.243, "MPa^0.5", 0.5
        ),
        **_hansen_context(18.0, 1.4, 2.0),
    }
    context.update(kwargs)
    return _predict(molecule_candidate(TOLUENE, conditions=ROOM), context=context)


# ==========================================================================
# The unit conversion, three ways
# ==========================================================================


@requires_rdkit
def test_the_same_parameter_in_two_units_gives_the_same_cohesive_energy_density():
    """18.24 MPa^0.5 and 18240 Pa^0.5 are one number, not two.

    This is the test that would have caught the ``dissolution.py`` bug: the
    factor between Pa^0.5 and MPa^0.5 is a thousand, and squaring it is a
    million.
    """
    assert PA_ROOT_PER_MPA_ROOT == 1000.0
    in_mpa = _predict(
        molecule_candidate(TOLUENE, conditions=ROOM),
        properties=CED,
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 18.24, "MPa^0.5", 0.5
            )
        },
    )["cohesive_energy_density"]
    in_pa = _predict(
        molecule_candidate(TOLUENE, conditions=ROOM),
        properties=CED,
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 18240.0, "Pa^0.5", 500.0
            )
        },
    )["cohesive_energy_density"]

    a = in_mpa.quantity.to("Pa").value
    b = in_pa.quantity.to("Pa").value
    assert a == pytest.approx(b, rel=1e-9)
    assert in_mpa.uncertainty.std == pytest.approx(in_pa.uncertainty.std, rel=1e-9)


@requires_rdkit
def test_tolueness_cohesive_energy_density_is_a_third_of_a_gigapascal():
    """delta = 18.243 MPa^0.5, so CED = 3.328e8 Pa. Not 332.8, not 3.33e14."""
    prediction = _toluene()["cohesive_energy_density"]
    value = prediction.quantity.to("Pa").value

    assert value == pytest.approx(3.328e8, rel=0.02)
    # A factor of 1e6 in either direction fails loudly rather than returning
    # something that still reads like a cohesive energy density.
    assert 1e8 < value < 1e9
    assert not math.isclose(value, 332.8, rel_tol=0.1)
    assert not math.isclose(value, 3.328e14, rel_tol=0.1)


@requires_rdkit
def test_the_emitted_quantity_converts_to_the_registry_canonical_unit():
    """A joule per cubic metre is a pascal, and the registry says so."""
    prediction = _toluene()["cohesive_energy_density"]
    canonical = get_property("cohesive_energy_density").canonical_unit
    assert prediction.quantity.to(canonical).value == pytest.approx(
        prediction.quantity.to("Pa").value, rel=1e-12
    )
    assert prediction.quantity.to("J/m^3").value == pytest.approx(
        prediction.quantity.to("Pa").value, rel=1e-12
    )


def test_the_two_identities_agree_on_a_solvent_whose_triple_sums_to_its_total():
    """Toluene: 18.0, 1.4, 2.0 sums to 18.17 against a measured 18.24."""
    from_total, _ = ced_from_hildebrand(18.243 * PA_ROOT_PER_MPA_ROOT, 0.0)
    from_triple, _ = ced_from_hansen(
        [(18.0 * PA_ROOT_PER_MPA_ROOT, 0.0), (1.4e3, 0.0), (2.0e3, 0.0)]
    )
    assert from_triple == pytest.approx(from_total, rel=0.01)


# ==========================================================================
# The two constants, and the measurement that says there are two
# ==========================================================================


def test_the_liquid_constant_is_not_the_polymer_one():
    """The headline finding: 2.8 against 9.1, a factor of 3.2."""
    model = cohesion_model()
    assert model.liquid_constant == pytest.approx(2.81, abs=0.02)
    assert model.polymer_constant == pytest.approx(9.11, abs=0.05)
    assert model.polymer_constant / model.liquid_constant > 3.0


def test_a_single_constant_would_be_wrong_by_more_than_threefold_on_a_solvent():
    """The regression guard on the finding above.

    If anyone later collapses the two constants into one, hexane's bulk modulus
    moves by a factor of three in the stiff direction and stays in a plausible
    gigapascal range, so nothing else in the pipeline would notice. This test
    is what notices.
    """
    model = cohesion_model()
    hexane = next(p for p in LIQUID_REFERENCES if p.name == "hexane")
    with_polymer_constant = model.polymer_constant * hexane.ced_pa
    measured = hexane.bulk_modulus_gpa * 1e9
    assert with_polymer_constant / measured > 3.0
    # And the right constant lands on it.
    assert model.liquid_constant * hexane.ced_pa == pytest.approx(measured, rel=0.15)


def test_every_gated_liquid_sits_inside_the_bar_the_module_carries():
    model = cohesion_model()
    gated = [p for p in LIQUID_REFERENCES if p.hydrogen_fraction <= HYDROGEN_BOND_FRACTION_LIMIT]
    assert len(gated) == model.liquid_n == 27
    for point in gated:
        predicted = model.liquid_constant * point.ced_pa
        measured = point.bulk_modulus_gpa * 1e9
        ratio = max(predicted / measured, measured / predicted)
        assert ratio <= model.carried_ratio("liquid"), point.name


def test_every_polymer_sits_inside_the_bar_the_module_carries():
    model = cohesion_model()
    assert model.polymer_n == len(POLYMER_REFERENCES) == 11
    for point in POLYMER_REFERENCES:
        predicted = model.polymer_constant * point.ced_pa
        measured = point.bulk_modulus_gpa * 1e9
        ratio = max(predicted / measured, measured / predicted)
        assert ratio <= model.carried_ratio("polymer"), point.name


def test_the_quoted_spreads_are_held_out_numbers_and_cannot_drift():
    """The numbers in the docstring are computed, not written down.

    Every point is scored against a constant fitted without it, so these are
    held-out ratios rather than the in-sample ones a fit reproduces for free.
    """
    model = cohesion_model()
    assert model.liquid_held_out == pytest.approx(1.77, abs=0.01)
    assert model.polymer_held_out == pytest.approx(1.27, abs=0.01)
    assert model.liquid_span == pytest.approx((1.62, 4.01), abs=0.01)
    assert model.polymer_span == pytest.approx((7.35, 11.08), abs=0.01)


def test_the_polymer_bar_is_widened_past_what_the_fit_would_claim():
    """1.27 is the model's own number and is not what is carried.

    Published room-temperature bulk moduli for one polymer disagree by more
    than that - polystyrene runs 2.4 GPa isothermal to 4.2 GPa adiabatic - so
    the references, not the model, set the floor.
    """
    model = cohesion_model()
    assert model.carried_ratio("polymer") == 2.0
    assert model.carried_ratio("polymer") > model.polymer_held_out
    # The liquid branch carries exactly what it measured, with no widening.
    assert model.carried_ratio("liquid") == model.liquid_held_out


def test_leave_one_out_scores_each_point_against_a_constant_that_never_saw_it():
    """A constant fitted on a point reproduces that point for free."""
    values = [1.0, 1.0, 1.0, 4.0]
    assert geometric_mean([1.0, 1.0, 1.0]) == pytest.approx(1.0)
    # In sample the outlier is 4 / geometric_mean(all) = 4 / 1.414 = 2.83;
    # held out it is 4 / 1.0 = 4.0, which is the honest number.
    assert worst_leave_one_out(values) == pytest.approx(4.0, rel=1e-9)
    assert worst_leave_one_out(values) > max(v / geometric_mean(values) for v in values)


# ==========================================================================
# The gates, and the measurements behind them
# ==========================================================================


def test_the_hydrogen_bonding_gate_is_what_buys_the_liquid_bar():
    """Without it the held-out ratio is 2.79 rather than 1.77.

    The nine liquids past the line run 0.93 (methanol) to 4.20 (glycerol), and
    the three worst are all low: cohesive energy density counts hydrogen bonds
    the bulk modulus does not feel.
    """
    model = cohesion_model()
    assert model.ungated_held_out == pytest.approx(2.79, abs=0.02)
    assert model.ungated_held_out > 1.5 * model.liquid_held_out
    worst = min(LIQUID_REFERENCES, key=lambda p: p.ratio)
    assert worst.name == "methanol"
    assert worst.ratio == pytest.approx(0.93, abs=0.02)
    assert worst.hydrogen_fraction > HYDROGEN_BOND_FRACTION_LIMIT


@requires_rdkit
def test_the_gate_is_not_the_hydrogen_bond_donor_count():
    """RDKit reports zero donors for water, the worst case in the whole set.

    A gate built on the donor count would let it straight through, which is why
    the gate is on dH/delta and the module must never be "simplified" back.
    """
    from formulate import chem

    assert chem.descriptors(WATER)["hbd"] == 0.0
    water_triple = [(15500.0, 0.0), (16000.0, 0.0), (42300.0, 0.0)]
    assert hydrogen_bond_fraction(water_triple) == pytest.approx(0.885, abs=0.005)
    assert hydrogen_bond_fraction(water_triple) > HYDROGEN_BOND_FRACTION_LIMIT


def test_the_fluorine_gate_has_one_measured_case_behind_it():
    """PTFE measures 23.6 against 9.11, a 2.6x miss in one direction."""
    model = cohesion_model()
    name, delta, bulk, _ = EXCLUDED_POLYMERS[0]
    assert name == "polytetrafluoroethylene"
    ratio = bulk * 1e9 / (delta * PA_ROOT_PER_MPA_ROOT) ** 2
    assert ratio == pytest.approx(23.6, abs=0.2)
    assert ratio / model.polymer_constant > 2.5


@requires_rdkit
def test_fluorine_fraction_reads_a_repeat_unit_without_counting_its_attachments():
    assert fluorine_fraction(PTFE_UNIT) == pytest.approx(4.0 / 6.0)
    assert fluorine_fraction(PS_UNIT) == 0.0
    assert fluorine_fraction("not a molecule at all") is None


# ==========================================================================
# What it predicts
# ==========================================================================


@requires_rdkit
def test_toluene_gets_a_bulk_modulus_that_contains_the_measured_one():
    """Measured 1.087 GPa; predicted 0.93 GPa with a bar that spans it."""
    prediction = _toluene()["bulk_modulus"]
    assert prediction.status is PredictionStatus.OK
    value = prediction.quantity.to("Pa").value
    assert value == pytest.approx(0.93e9, rel=0.05)
    assert abs(value - 1.087e9) < prediction.uncertainty.std
    # An honest wide bar: a factor of 1.77, not a flattering few per cent.
    assert prediction.uncertainty.std / value > 0.7


@requires_rdkit
def test_a_polymer_is_answered_on_the_polymer_branch_not_its_monomers():
    """Polystyrene: delta 18.6 MPa^0.5 gives 3.2 GPa against a measured 3.1."""
    predictions = _predict(
        polymer_candidate(PS_UNIT, conditions=ROOM),
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 18.6, "MPa^0.5", 0.5
            )
        },
    )
    bulk = predictions["bulk_modulus"]
    assert bulk.status is PredictionStatus.OK
    assert bulk.quantity.to("Pa").value == pytest.approx(3.15e9, rel=0.05)
    assert abs(bulk.quantity.to("Pa").value - 3.1e9) < bulk.uncertainty.std
    assert bulk.provenance.parameters["regime"] == "polymer"
    assert bulk.provenance.parameters["constant"] == pytest.approx(9.11, abs=0.02)


@requires_rdkit
def test_the_polymer_branch_says_in_its_notes_why_the_liquid_one_would_be_wrong():
    predictions = _predict(
        molecule_candidate(HEXANE, conditions=ROOM),
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 14.87, "MPa^0.5", 0.5
            ),
            **_hansen_context(14.9, 0.0, 0.0),
        },
    )
    notes = " ".join(predictions["bulk_modulus"].notes)
    assert "eight-to-eleven" in notes
    assert predictions["bulk_modulus"].quantity.to("Pa").value == pytest.approx(
        0.62e9, rel=0.05
    )


@requires_rdkit
def test_a_hansen_triple_alone_is_enough_for_a_cohesive_energy_density():
    """No square root is taken and then squared; the energies sum directly."""
    predictions = _predict(
        molecule_candidate(TOLUENE, conditions=ROOM),
        context=_hansen_context(18.0, 1.4, 2.0),
    )
    ced = predictions["cohesive_energy_density"]
    assert ced.status is PredictionStatus.OK
    expected = (18.0e3) ** 2 + (1.4e3) ** 2 + (2.0e3) ** 2
    assert ced.quantity.to("Pa").value == pytest.approx(expected, rel=1e-9)
    # The route penalty is on, because this is not the vaporisation measurement.
    assert ced.uncertainty.std / ced.quantity.to("Pa").value > 0.05


# ==========================================================================
# Uncertainty
# ==========================================================================


@requires_rdkit
def test_the_spread_is_the_upstream_spread_transformed_exactly():
    """sigma_CED = 2 delta sigma, which is half of [(d-s)^2, (d+s)^2]."""
    delta, sigma = 18243.0, 400.0
    single = _predict(
        molecule_candidate(TOLUENE, conditions=ROOM),
        properties=CED,
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", delta, "Pa^0.5", sigma
            )
        },
    )["cohesive_energy_density"]
    doubled = _predict(
        molecule_candidate(TOLUENE, conditions=ROOM),
        properties=CED,
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", delta, "Pa^0.5", 2 * sigma
            )
        },
    )["cohesive_energy_density"]

    assert single.uncertainty.std == pytest.approx(2.0 * delta * sigma, rel=1e-9)
    assert doubled.uncertainty.std == pytest.approx(2.0 * single.uncertainty.std, rel=1e-9)
    half_width = ((delta + sigma) ** 2 - (delta - sigma) ** 2) / 2.0
    assert single.uncertainty.std == pytest.approx(half_width, rel=1e-12)


@requires_rdkit
def test_an_upstream_with_no_stated_spread_does_not_become_a_certain_answer():
    prediction = _predict(
        molecule_candidate(TOLUENE, conditions=ROOM),
        properties=CED,
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 18.243, "MPa^0.5", None
            )
        },
    )["cohesive_energy_density"]
    assert prediction.uncertainty.std is not None
    assert prediction.uncertainty.std > 0.0


@requires_rdkit
def test_two_routes_that_disagree_widen_the_bar_and_say_so():
    """A 40% disagreement is recorded, not quietly resolved in favour of one."""
    context = {
        "hildebrand_solubility_parameter": _upstream(
            "hildebrand_solubility_parameter", 18.243, "MPa^0.5", 0.2
        ),
        **_hansen_context(24.0, 5.0, 3.0, std=0.2),
    }
    prediction = _predict(
        molecule_candidate(TOLUENE, conditions=ROOM), properties=CED, context=context
    )["cohesive_energy_density"]
    hansen_ced = (24.0e3) ** 2 + (5.0e3) ** 2 + (3.0e3) ** 2
    value = prediction.quantity.to("Pa").value
    assert value == pytest.approx(3.328e8, rel=0.02)  # the Hildebrand route wins
    assert value + prediction.uncertainty.std >= hansen_ced
    assert any("disagreement" in note for note in prediction.notes)


# ==========================================================================
# Refusals. Each one is a feature and must not regress.
# ==========================================================================


@requires_rdkit
def test_no_cohesion_upstream_means_no_answer_at_all():
    predictions = _predict(molecule_candidate(TOLUENE, conditions=ROOM), context={})
    for prop in ("cohesive_energy_density", "bulk_modulus"):
        assert predictions[prop].status is PredictionStatus.UNSUPPORTED
        assert predictions[prop].quantity is None
    reason = " ".join(predictions["cohesive_energy_density"].notes)
    assert "hildebrand_solubility_parameter" in reason
    assert "hansen_dispersion" in reason


@requires_rdkit
def test_two_of_three_hansen_components_is_not_a_cohesive_energy_density():
    """Summing two squares understates the energy by construction."""
    partial = _hansen_context(18.0, 1.4, 2.0)
    del partial["hansen_hydrogen_bonding"]
    predictions = _predict(molecule_candidate(TOLUENE, conditions=ROOM), context=partial)
    assert predictions["cohesive_energy_density"].status is PredictionStatus.UNSUPPORTED
    assert predictions["bulk_modulus"].status is PredictionStatus.UNSUPPORTED


@requires_rdkit
def test_no_temperature_means_no_answer():
    """Both properties are condition dependent: the answer moves with T."""
    predictions = _predict(
        molecule_candidate(TOLUENE),
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 18.243, "MPa^0.5", 0.5
            )
        },
        conditions=Conditions(),
    )
    for prop in BOTH:
        assert predictions[prop].status is PredictionStatus.UNSUPPORTED
        assert "temperature" in " ".join(predictions[prop].notes)


@requires_rdkit
def test_far_from_room_temperature_refuses_the_modulus_but_not_the_cohesion():
    """CED is temperature-correct through the upstream parameter; B/CED is not.

    B/CED runs as 1/(alpha T) and every validation point is at 298 K, so this is
    a systematic error an error bar cannot rescue.
    """
    hot = Conditions(temperature=Quantity(value=400.0, unit="K"))
    predictions = _predict(
        molecule_candidate(DMSO, conditions=hot),
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 22.0, "MPa^0.5", 1.0
            ),
            **_hansen_context(18.4, 16.4, 10.2),
        },
        conditions=hot,
    )
    assert predictions["cohesive_energy_density"].status.has_value
    assert predictions["bulk_modulus"].status is PredictionStatus.UNSUPPORTED
    assert str(int(BULK_MODULUS_T_OUTER[1])) in " ".join(predictions["bulk_modulus"].notes)
    # And the cohesion keeps a clean domain: the 298 K window belongs to the
    # ratio, not to an identity on a parameter that arrived at 400 K already.
    assert predictions["cohesive_energy_density"].applicability.score == 1.0


@requires_rdkit
def test_a_hansen_only_cohesion_is_refused_away_from_the_temperature_it_is_tabulated_at():
    """The compilations are 25 C values and the registry does not mark them
    condition dependent, so a 400 K label on one would be a lie."""
    hot = Conditions(temperature=Quantity(value=400.0, unit="K"))
    predictions = _predict(
        molecule_candidate(DMSO, conditions=hot),
        context=_hansen_context(18.4, 16.4, 10.2),
        conditions=hot,
    )
    assert predictions["cohesive_energy_density"].status is PredictionStatus.UNSUPPORTED
    assert "25 C" in " ".join(predictions["cohesive_energy_density"].notes)


@requires_rdkit
def test_a_strongly_hydrogen_bonded_liquid_gets_no_bulk_modulus():
    """Water measures 0.96 against a constant of 2.81 - a systematic miss."""
    predictions = _predict(
        molecule_candidate(WATER, conditions=ROOM),
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 47.93, "MPa^0.5", 1.0
            ),
            **_hansen_context(15.5, 16.0, 42.3),
        },
    )
    assert predictions["cohesive_energy_density"].status.has_value
    bulk = predictions["bulk_modulus"]
    assert bulk.status is PredictionStatus.UNSUPPORTED
    assert "hydrogen bonding" in " ".join(bulk.notes)


@requires_rdkit
def test_without_a_hansen_triple_a_liquid_bulk_modulus_has_no_gate_and_is_refused():
    """The Hildebrand total alone cannot say how much of it is hydrogen bonds."""
    predictions = _predict(
        molecule_candidate(TOLUENE, conditions=ROOM),
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 18.243, "MPa^0.5", 0.5
            )
        },
    )
    assert predictions["cohesive_energy_density"].status.has_value
    assert predictions["bulk_modulus"].status is PredictionStatus.UNSUPPORTED
    assert "donors for water" in " ".join(predictions["bulk_modulus"].notes)


@requires_rdkit
def test_a_fluorine_rich_repeat_unit_gets_no_bulk_modulus():
    predictions = _predict(
        polymer_candidate(PTFE_UNIT, conditions=ROOM),
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 12.7, "MPa^0.5", 0.5
            )
        },
    )
    assert predictions["cohesive_energy_density"].status.has_value
    bulk = predictions["bulk_modulus"]
    assert bulk.status is PredictionStatus.UNSUPPORTED
    assert "fluorine" in " ".join(bulk.notes)


@requires_rdkit
def test_a_part_polymer_part_solvent_formulation_is_refused_outright():
    """A mixture is not its major component, and the two regimes differ by 3x."""
    mixture = Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=(
                MixtureComponent(
                    role=ComponentRole.SOLVENT,
                    fraction=0.7,
                    molecule=MoleculeSpec(smiles=TOLUENE),
                ),
                MixtureComponent(
                    role=ComponentRole.BINDER,
                    fraction=0.3,
                    polymer=PolymerSpec(monomers=(MonomerUnit(smiles=PS_UNIT),)),
                ),
            )
        ),
        conditions=ROOM,
    )
    predictions = _predict(
        mixture,
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 18.4, "MPa^0.5", 0.5
            ),
            **_hansen_context(18.1, 3.9, 3.1),
        },
    )
    for prop in BOTH:
        assert predictions[prop].status is PredictionStatus.UNSUPPORTED
        assert "part small molecule and part polymer" in " ".join(predictions[prop].notes)


@requires_rdkit
def test_an_all_solvent_formulation_is_answered_on_the_liquid_branch():
    mixture = Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=(
                MixtureComponent(
                    role=ComponentRole.SOLVENT,
                    fraction=0.5,
                    molecule=MoleculeSpec(smiles=TOLUENE),
                ),
                MixtureComponent(
                    role=ComponentRole.CO_SOLVENT,
                    fraction=0.5,
                    molecule=MoleculeSpec(smiles=HEXANE),
                ),
            )
        ),
        conditions=ROOM,
    )
    predictions = _predict(mixture, context=_hansen_context(16.5, 0.7, 1.0))
    bulk = predictions["bulk_modulus"]
    assert bulk.status.has_value
    assert bulk.provenance.parameters["regime"] == "liquid"


@requires_rdkit
def test_a_substance_that_is_not_a_liquid_has_no_cohesive_energy_density():
    """A gas has no cohesion to report, and hexane boils at 342 K."""
    hot = Conditions(temperature=Quantity(value=345.0, unit="K"))
    predictions = _predict(
        molecule_candidate(HEXANE, conditions=hot),
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 14.87, "MPa^0.5", 0.5
            )
        },
        conditions=hot,
    )
    notes = " ".join(predictions["cohesive_energy_density"].notes)
    if predictions["cohesive_energy_density"].status is PredictionStatus.UNSUPPORTED:
        assert "boils" in notes
    else:  # the compiled boiling point is optional in a bare environment
        pytest.skip("no compiled boiling point for hexane in this environment")


# ==========================================================================
# Domain
# ==========================================================================


@requires_rdkit
def test_a_candidate_with_no_parseable_structure_is_outside_the_domain():
    candidate = molecule_candidate("this is not a smiles string", conditions=ROOM)
    domain = CohesionExpert().assess_domain(candidate)
    assert domain.in_domain is False
    assert "parse" in domain.basis + " ".join(domain.warnings)


def _polystyrene_at(conditions):
    """Polystyrene, evaluated at whatever conditions the *request* states."""
    return _predict(
        polymer_candidate(PS_UNIT, conditions=conditions),
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 18.6, "MPa^0.5", 0.5
            )
        },
        conditions=conditions,
    )["bulk_modulus"]


@requires_rdkit
def test_a_temperature_far_from_the_validation_point_lowers_the_domain_score():
    warm = Conditions(temperature=Quantity(value=345.0, unit="K"))
    at_room = _polystyrene_at(ROOM)
    away = _polystyrene_at(warm)
    assert at_room.applicability.score == 1.0
    assert away.applicability.score < at_room.applicability.score
    assert any("298 K" in w for w in away.applicability.warnings)


@requires_rdkit
def test_the_domain_is_assessed_at_the_temperature_the_answer_is_for():
    """The candidate's conditions and the request's are not the same object.

    ``_conditions_for`` in the evaluation engine hands an expert the conditions
    a *requirement* asks for, which may differ from the ones the candidate was
    proposed at. Assessing the domain on the candidate's copy reported a 340 K
    bulk modulus with a domain score of 1.0 and no warning, because the
    candidate still said 298 K. This is the regression guard on that.
    """
    warm = Conditions(temperature=Quantity(value=340.0, unit="K"))
    stale = _predict(
        polymer_candidate(PS_UNIT, conditions=ROOM),  # candidate says 298 K
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 18.6, "MPa^0.5", 0.5
            )
        },
        conditions=warm,  # the answer is for 340 K
    )["bulk_modulus"]
    assert stale.status.has_value
    assert stale.applicability.score < 1.0
    assert any("340 K" in w for w in stale.applicability.warnings)
    # And the domain assessment does not invent a temperature of its own.
    assert CohesionExpert().assess_domain(
        polymer_candidate(PS_UNIT, conditions=warm)
    ).warnings == ()


@requires_rdkit
def test_a_polymer_far_above_its_glass_transition_is_flagged_and_widened():
    """PDMS is the case: Tg 150 K, and it measures 4.4 against 9.11."""
    context = {
        "hildebrand_solubility_parameter": _upstream(
            "hildebrand_solubility_parameter", 15.0, "MPa^0.5", 0.5
        ),
        "glass_transition_temperature": _upstream(
            "glass_transition_temperature", 150.0, "K", 10.0
        ),
    }
    loose = _predict(
        polymer_candidate("[*][Si](C)(C)O[*]", conditions=ROOM), context=context
    )["bulk_modulus"]
    tight = _predict(
        polymer_candidate(PS_UNIT, conditions=ROOM),
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 15.0, "MPa^0.5", 0.5
            ),
            "glass_transition_temperature": _upstream(
                "glass_transition_temperature", 373.0, "K", 10.0
            ),
        },
    )["bulk_modulus"]
    assert loose.status.has_value
    assert loose.uncertainty.std > tight.uncertainty.std
    assert any("glass transition" in note for note in loose.notes)
    assert loose.applicability.score < 1.0


@requires_rdkit
def test_the_expert_declares_what_it_needs_and_what_it_covers():
    expert = CohesionExpert()
    assert expert.id == "cohesion"
    assert expert.supported_properties == BOTH
    assert MaterialClass.COMPOSITE not in expert.supported_classes
    for prop in (
        "hildebrand_solubility_parameter",
        "hansen_dispersion",
        "hansen_polar",
        "hansen_hydrogen_bonding",
    ):
        assert prop in expert.dependencies
    assert expert.covers("bulk_modulus", MaterialClass.POLYMER)
    assert not expert.covers("bulk_modulus", MaterialClass.COMPOSITE)


@requires_rdkit
def test_a_structure_that_does_not_parse_gets_no_bulk_modulus():
    """The fluorine gate is evaluated on the structure, so an unreadable one
    cannot be shown to be inside it."""
    predictions = _predict(
        molecule_candidate("this is not a smiles string", conditions=ROOM),
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 18.243, "MPa^0.5", 0.5
            ),
            **_hansen_context(18.0, 1.4, 2.0),
        },
    )
    bulk = predictions["bulk_modulus"]
    assert bulk.status is PredictionStatus.UNSUPPORTED
    assert "does not parse" in " ".join(bulk.notes)
    # The cohesive energy density is an identity on the parameter and survives,
    # but it is flagged out of domain rather than reported as a clean answer.
    assert predictions["cohesive_energy_density"].status is PredictionStatus.OUT_OF_DOMAIN


@requires_rdkit
def test_a_non_positive_cohesive_energy_density_is_not_scaled_into_a_modulus():
    """And is not reported as a value either, because it arrives certain.

    ``sigma = 2 delta sigma_delta`` is exactly zero when delta is zero, so
    emitting the cohesion would put a zero-width error bar on the claim that
    nothing holds the phase together - the one spread this repository never
    allows.
    """
    predictions = _predict(
        molecule_candidate(TOLUENE, conditions=ROOM),
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 0.0, "MPa^0.5", 0.5
            ),
            **_hansen_context(0.0, 0.0, 0.0),
        },
    )
    for prop in BOTH:
        assert predictions[prop].status is PredictionStatus.UNSUPPORTED
        assert predictions[prop].quantity is None
        assert "condensed phase" in " ".join(predictions[prop].notes)
    # The refusal names the expert to go and look at.
    assert "hildebrand_solubility_parameter" in " ".join(
        predictions["cohesive_energy_density"].notes
    )


@requires_rdkit
def test_no_prediction_this_expert_makes_carries_a_zero_width_error_bar():
    """A spread of zero is a claim of certainty, and nothing here can earn one."""
    cases = [
        _toluene(),
        _predict(
            molecule_candidate(TOLUENE, conditions=ROOM),
            context=_hansen_context(18.0, 1.4, 2.0),
        ),
        _predict(
            polymer_candidate(PS_UNIT, conditions=ROOM),
            context={
                "hildebrand_solubility_parameter": _upstream(
                    "hildebrand_solubility_parameter", 18.6, "MPa^0.5", None
                )
            },
        ),
    ]
    seen = 0
    for predictions in cases:
        for prediction in predictions.values():
            if not prediction.status.has_value:
                continue
            seen += 1
            assert prediction.uncertainty.std is not None, prediction.property
            assert prediction.uncertainty.std > 0.0, prediction.property
            assert prediction.uncertainty.basis
    assert seen >= 4


@requires_rdkit
def test_the_reference_tables_are_real_structures():
    """A reference table with an unreadable structure in it is not a reference."""
    from formulate import chem

    for point in LIQUID_REFERENCES:
        assert chem.mol_from_smiles(point.smiles) is not None, point.name
        assert 0.0 <= point.hydrogen_fraction <= 1.0
        assert point.delta_mpa_root > 0 and point.bulk_modulus_gpa > 0
    for point in POLYMER_REFERENCES:
        mol = chem.mol_from_smiles(point.repeat_unit)
        assert mol is not None, point.name
        stars = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
        assert stars == 2, f"{point.name} is a repeat unit, not a molecule"
        assert point.source, "every reference bulk modulus names where it came from"


def test_the_hydrogen_bonding_fractions_match_the_compilation():
    """Spot checks against the Hansen compilation this column was built from."""
    water = next(p for p in LIQUID_REFERENCES if p.name == "water")
    hexane = next(p for p in LIQUID_REFERENCES if p.name == "hexane")
    assert water.hydrogen_fraction == pytest.approx(42.3 / 47.8, abs=0.01)
    assert hexane.hydrogen_fraction == 0.0


def test_the_method_string_quotes_the_constants_the_module_actually_uses():
    """A number written in prose is a number that eventually disagrees with
    the code, so the prose is pinned to the code here."""
    model = cohesion_model()
    method = CohesionExpert.method
    assert f"{model.liquid_constant:.2f}" in method
    assert f"{model.polymer_constant:.2f}" in method


# ==========================================================================
# Holes found by adversarial review, each with the case that exposed it
# ==========================================================================


def test_a_hansen_triple_with_no_magnitude_is_absence_not_zero_hydrogen_bonding():
    """(0, 0, 0) is not a measurement that a liquid has no hydrogen bonds."""
    assert hydrogen_bond_fraction([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]) is None
    assert hydrogen_bond_fraction([(15500.0, 0.0), (16000.0, 0.0), (42300.0, 0.0)]) is not None


@requires_rdkit
def test_a_zero_hansen_triple_does_not_smuggle_water_past_the_gate():
    """Water is the compound the gate exists for.

    With its real triple it is refused. With a triple of zeros alongside the
    same Hildebrand parameter it used to be answered, because the fraction came
    out 0.0 and 0.0 is inside the limit: absence read as the most favourable
    possible evidence, which is the one thing this repository forbids.
    """
    predictions = _predict(
        molecule_candidate(WATER, conditions=ROOM),
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 47.93, "MPa^0.5", 1.0
            ),
            **_hansen_context(0.0, 0.0, 0.0),
        },
    )
    bulk = predictions["bulk_modulus"]
    assert bulk.status is PredictionStatus.UNSUPPORTED
    assert bulk.quantity is None
    assert "no magnitude" in " ".join(bulk.notes)
    # The cohesive energy density is still an identity on the parameter.
    assert predictions["cohesive_energy_density"].status.has_value


@requires_rdkit
def test_a_polymer_with_no_glass_transition_says_the_melt_check_was_not_run():
    """The module's own known hole must not be silent.

    A low-Tg polymer arriving without a Tg is indistinguishable here from a
    glass, and the prediction used to carry nothing at all about that - while
    the exactly analogous missing-Hansen case did carry a note.
    """
    without = _predict(
        polymer_candidate("[*][Si](C)(C)O[*]", conditions=ROOM),
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 15.0, "MPa^0.5", 0.5
            )
        },
    )["bulk_modulus"]
    assert without.status.has_value
    assert any("no glass_transition_temperature arrived" in n for n in without.notes)


@requires_rdkit
def test_far_above_tg_does_not_claim_a_direction_two_materials_disagree_about():
    """Polyethylene is in the fit set and is 150 K above its own Tg.

    It measures 10.7 - stiffer than the constant, not softer - so the note that
    fires for PDMS must not tell a chemist the material is closer to a melt.
    """
    polyethylene = _predict(
        polymer_candidate("[*]CC[*]", conditions=ROOM),
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 16.2, "MPa^0.5", 0.5
            ),
            "glass_transition_temperature": _upstream(
                "glass_transition_temperature", 148.0, "K", 10.0
            ),
        },
    )["bulk_modulus"]
    notes = " ".join(polyethylene.notes)
    assert "opposite directions" in notes and "polyethylene 10.7" in notes
    assert "closer to a melt" not in notes
    # And the widening is symmetric: polyethylene's own miss is the stiff one
    # and sits inside the bar just as PDMS's soft one does.
    assert abs(polyethylene.quantity.to("Pa").value - 2.8e9) < polyethylene.uncertainty.std


@requires_rdkit
def test_a_disputed_cohesive_energy_says_in_its_basis_what_the_spread_really_is():
    """The carried number stopped being the propagation; the basis must say so.

    When the two routes disagree by more than their combined spread the bar is
    the disagreement - here 83% of the value against a propagated 2% - and a
    basis still reciting ``sigma = 2 delta sigma_delta`` would be describing a
    number the prediction is not carrying.
    """
    prediction = _predict(
        molecule_candidate(TOLUENE, conditions=ROOM),
        properties=CED,
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 18.243, "MPa^0.5", 0.2
            ),
            **_hansen_context(24.0, 5.0, 3.0, std=0.2),
        },
    )["cohesive_energy_density"]
    relative = prediction.uncertainty.std / prediction.quantity.to("Pa").value
    assert relative > 0.5  # nothing like the propagated 2%
    assert "disagrees with the Hildebrand parameter" in prediction.uncertainty.basis
    assert f"{relative:.0%}" in prediction.uncertainty.basis
    assert prediction.provenance.parameters["route_disagreement"] == pytest.approx(
        relative, rel=1e-9
    )
    # An undisputed prediction says nothing of the kind.
    plain = _toluene()["cohesive_energy_density"]
    assert "disagrees with the Hildebrand parameter" not in plain.uncertainty.basis
    assert plain.provenance.parameters["route_disagreement"] == 0.0


@requires_rdkit
def test_a_solid_is_refused_with_a_reason_that_says_what_to_do_instead():
    """Naphthalene melts at 353 K, so at 298 K there is no liquid delta to square."""
    predictions = _predict(
        molecule_candidate("c1ccc2ccccc2c1", conditions=ROOM),
        context={
            "hildebrand_solubility_parameter": _upstream(
                "hildebrand_solubility_parameter", 20.3, "MPa^0.5", 0.5
            ),
            **_hansen_context(19.2, 2.0, 5.9),
        },
    )
    ced = predictions["cohesive_energy_density"]
    if ced.status is not PredictionStatus.UNSUPPORTED:
        pytest.skip("no compiled melting point for naphthalene in this environment")
    reason = " ".join(ced.notes)
    assert "solid" in reason
    assert "lattice-energy" in reason  # tells a chemist where to get one instead


@requires_rdkit
def test_a_formulation_says_that_mixing_a_parameter_is_not_mixing_a_cohesion():
    """CED is quadratic in delta, so a linear mixing rule upstream is not linear here."""
    mixture = Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=(
                MixtureComponent(
                    role=ComponentRole.SOLVENT,
                    fraction=0.5,
                    molecule=MoleculeSpec(smiles=TOLUENE),
                ),
                MixtureComponent(
                    role=ComponentRole.CO_SOLVENT,
                    fraction=0.5,
                    molecule=MoleculeSpec(smiles=HEXANE),
                ),
            )
        ),
        conditions=ROOM,
    )
    ced = _predict(mixture, context=_hansen_context(16.5, 0.7, 1.0))[
        "cohesive_energy_density"
    ]
    notes = " ".join(ced.notes)
    assert "quadratic in delta" in notes
    assert "does not fix a mixture's" in notes
