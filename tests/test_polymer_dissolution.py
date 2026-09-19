"""Dissolving a polymer candidate, and the far longer list of times it refuses.

Three things are pinned here, and the last two matter at least as much as the
first.

*The numbers.*  Seven polymer/solvent pairs whose answer a polymer chemist
knows, to two decimal places, plus the unit conversion underneath them - a
hand-written factor of a thousand on the square root of a pressure is the
mistake ``dissolution.py`` had to document, and this module reaches MPa^0.5
twice by two different routes, so the two are asserted to agree.

*The design decisions.*  This expert refuses to predict an interaction radius
and refuses to publish a Hansen distance, and both refusals rest on numbers
measured over the tabulated spheres rather than on taste.  Those measurements
are re-run here, so that extending the sphere table rechecks the claim instead
of inheriting it.

*The refusals.*  Every path that declines to answer is tested as hard as the
paths that do, because the cheapest way to make this expert look better is to
let one of them through with a mid-range guess.  The dependency path is tested
hardest of all: it has to refuse cleanly rather than crash when no upstream
Hansen triple exists at all.
"""

from __future__ import annotations

import itertools
import math

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import (
    Candidate,
    ComponentRole,
    MaterialClass,
    MixtureComponent,
    MixtureSpec,
    MoleculeSpec,
    MonomerRole,
    MonomerUnit,
    PolymerSpec,
    PolymerTopology,
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import Prediction, PredictionStatus
from formulate.core.properties import PROPERTY_REGISTRY, get_property
from formulate.core.quantity import Quantity, Uncertainty, UncertaintyKind
from formulate.experts.base import PredictionRequest
from formulate.experts.dissolution import SOLUBILITY_SPHERES, relative_energy_difference
from formulate.experts.hansen import hansen_distance, hansen_triple
from formulate.experts.polymer_dissolution import (
    DISPLACEMENT_FRACTION,
    compilation_would_have_used,
    names_a_polymer,
    repeat_unit_smiles,
    PROPERTY_FOR_CLASS,
    RADIUS_SPREAD_FRACTION,
    SPHERE_REPEAT_UNITS,
    TEMPERATURE_WINDOW_K,
    PolymerDissolutionExpert,
    solvent_triple_from_name,
    sphere_for_repeat_unit,
)

pytestmark = requires_rdkit

RED = "solubility_red"
MIX_RED = "relative_energy_difference"
CENTRE = ("hansen_dispersion", "hansen_polar", "hansen_hydrogen_bonding")

#: The repeat-unit triples ``polymer_hansen`` actually produces for the
#: tabulated spheres, in MPa^0.5, measured once by running that expert over
#: this engine's own predicted densities.  They are injected as stubs rather
#: than recomputed so that every test here is deterministic and so that the
#: refusal path still has something to refuse when ``polymer_hansen`` is not
#: registered.
PREDICTED_CENTRES = {
    "polystyrene": (18.36, 1.13, 0.0),
    "poly(methyl methacrylate)": (16.31, 5.59, 8.93),
    "poly(vinyl chloride)": (17.71, 12.17, 2.98),
    "polycarbonate": (17.85, 3.06, 6.85),
    "poly(vinyl acetate)": (15.99, 6.76, 9.82),
}

#: ``polymer_hansen``'s own shipped one-sigma spreads, per component.
UPSTREAM_STD_MPA_SQRT = (3.91, 6.05, 4.30)


# -- fixtures --------------------------------------------------------------


def _context(triple_mpa, std=UPSTREAM_STD_MPA_SQRT) -> dict[str, Prediction]:
    """An upstream Hansen triple as the engine would hand it over."""
    return {
        prop: Prediction(
            property=prop,
            quantity=Quantity(value=value, unit="MPa^0.5"),
            uncertainty=Uncertainty(
                std=sigma, kind=UncertaintyKind.EPISTEMIC, basis="stub for test"
            ),
            expert_id="polymer_hansen",
        )
        for prop, value, sigma in zip(CENTRE, triple_mpa, std)
    }


def _polymer(unit: str, **spec_kwargs) -> Candidate:
    monomers = spec_kwargs.pop("monomers", (MonomerUnit(smiles=unit),))
    return Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(monomers=monomers, **spec_kwargs),
        conditions=Conditions.standard(),
    )


def _in(solvent: str | None, **extra) -> Conditions:
    return Conditions.standard().model_copy(update={"environment": solvent, **extra})


def _predict(candidate, conditions, context=None, prop=None) -> Prediction:
    prop = prop or PROPERTY_FOR_CLASS[candidate.material_class]
    out = PolymerDissolutionExpert().predict(
        PredictionRequest(
            candidate=candidate,
            properties=frozenset({prop}),
            conditions=conditions,
            context=context or {},
        )
    )
    assert len(out) == 1, f"expected one prediction for {prop}, got {out}"
    return out[0]


def _polymer_red(name: str, solvent: str, **conditions) -> Prediction:
    return _predict(
        _polymer(SPHERE_REPEAT_UNITS[name]),
        _in(solvent, **conditions),
        _context(PREDICTED_CENTRES[name]),
    )


def _formulation(unit: str, solvent_smiles: str, fraction: float = 0.1, **roles) -> Candidate:
    return Candidate(
        material_class=MaterialClass.MIXTURE,
        conditions=Conditions.standard(),
        mixture=MixtureSpec(
            components=(
                MixtureComponent(
                    role=roles.get("polymer_role", ComponentRole.SOLUTE),
                    fraction=fraction,
                    polymer=PolymerSpec(monomers=(MonomerUnit(smiles=unit),)),
                ),
                MixtureComponent(
                    role=roles.get("solvent_role", ComponentRole.SOLVENT),
                    fraction=1.0 - fraction,
                    molecule=MoleculeSpec(smiles=solvent_smiles),
                ),
            )
        ),
    )


# --------------------------------------------------------------------------
# The design decisions, re-measured
# --------------------------------------------------------------------------


def test_the_interaction_radius_does_not_follow_from_the_polymers_own_cohesion():
    """The measurement that justifies refusing to predict R0, re-run here.

    If someone extends ``SOLUBILITY_SPHERES``, the claim behind the refusal is
    rechecked rather than inherited from a docstring.
    """
    spheres = [SOLUBILITY_SPHERES[k] for k in sorted(SOLUBILITY_SPHERES)]
    totals = [
        math.sqrt(s.dispersion**2 + s.polar**2 + s.hydrogen_bonding**2) for s in spheres
    ]
    radii = [s.radius for s in spheres]
    ratios = [r / t for r, t in zip(radii, totals)]

    assert max(ratios) / min(ratios) > 3.0, (
        "R0/delta_total no longer spans more than 3x; if it has become a "
        "constant, predicting R0 from structure may now be defensible"
    )

    n = len(radii)
    mean_t, mean_r = sum(totals) / n, sum(radii) / n
    cov = sum((t - mean_t) * (r - mean_r) for t, r in zip(totals, radii))
    var_t = sum((t - mean_t) ** 2 for t in totals)
    var_r = sum((r - mean_r) ** 2 for r in radii)
    r_squared = cov**2 / (var_t * var_r)
    assert r_squared < 0.2, f"R0 now tracks delta_total (r^2 = {r_squared:.3f})"


def test_hansen_distance_ranks_polymers_backwards_and_is_never_returned():
    """Ra orders polymers the opposite way to RED, so it is not published.

    At a fixed polymer Ra and RED are the same ordering.  At a fixed *solvent*,
    which is the direction this expert works in, they are not: toluene is the
    textbook case, where the nearest centre belongs to polycarbonate and the
    polymer that actually dissolves is polystyrene.
    """
    toluene = tuple(v / 1000.0 for v in hansen_triple("Cc1ccccc1"))
    ps, pc = SOLUBILITY_SPHERES["polystyrene"], SOLUBILITY_SPHERES["polycarbonate"]

    ra_ps, ra_pc = hansen_distance(toluene, ps.centre), hansen_distance(toluene, pc.centre)
    assert ra_pc < ra_ps, "the distance no longer prefers polycarbonate for toluene"
    assert ra_ps / ps.radius < ra_pc / pc.radius, "RED no longer prefers polystyrene"

    # And the disagreement is general, not one anecdote: over these solvents a
    # large minority of polymer pairs come out in the opposite order.
    solvents = ["Cc1ccccc1", "CC(C)=O", "CCCCCC", "O", "CO", "ClC(Cl)Cl", "C1CCOC1"]
    spheres = [SOLUBILITY_SPHERES[k] for k in sorted(SOLUBILITY_SPHERES)]
    backwards = total = 0
    for smiles in solvents:
        triple = tuple(v / 1000.0 for v in hansen_triple(smiles))
        distances = [hansen_distance(triple, s.centre) for s in spheres]
        reds = [d / s.radius for d, s in zip(distances, spheres)]
        for i, j in itertools.combinations(range(len(spheres)), 2):
            total += 1
            if (distances[i] - distances[j]) * (reds[i] - reds[j]) < 0:
                backwards += 1
    assert backwards / total > 0.25, (
        "Ra and RED now agree on how to order polymers; the reason for withholding "
        "the distance would need rechecking"
    )


def test_no_property_this_expert_returns_is_a_distance():
    """Ra appears in the prose and never as a value under any property name."""
    expert = PolymerDissolutionExpert()
    assert "hansen_distance" not in expert.supported_properties
    for prop in expert.supported_properties:
        assert get_property(prop).canonical_unit == "dimensionless"

    prediction = _polymer_red("polystyrene", "toluene")
    assert prediction.quantity.to_canonical().unit == "dimensionless"
    assert prediction.quantity.value == prediction.canonical.value
    assert any("Ra" in note for note in prediction.notes)


# --------------------------------------------------------------------------
# The numbers
# --------------------------------------------------------------------------


#: Pairs a polymer chemist can check by hand, with the RED the handbook sphere
#: gives and the behaviour it has to reproduce.
KNOWN = [
    ("polystyrene", "toluene", 0.65, True),
    ("polystyrene", "hexane", 1.16, False),
    ("poly(methyl methacrylate)", "acetone", 0.72, True),
    ("poly(vinyl acetate)", "acetone", 0.82, True),
    ("polycarbonate", "chloroform", 0.41, True),
    ("polycarbonate", "dichloromethane", 0.35, True),
    ("poly(methyl methacrylate)", "tetrahydrofuran", 0.70, True),
]


@pytest.mark.parametrize("name,solvent,expected,dissolves", KNOWN)
def test_known_polymer_solvent_pairs(name, solvent, expected, dissolves):
    prediction = _polymer_red(name, solvent)
    assert prediction.status is PredictionStatus.OK
    assert prediction.quantity.value == pytest.approx(expected, abs=0.02)
    assert (prediction.quantity.value < 1.0) is dissolves


def test_the_unit_conversion_is_the_registrys_and_both_routes_agree():
    """MPa^0.5 is reached twice here; a factor of 1000 apart would be invisible.

    ``dissolution.py`` carries the Pa^0.5 -> MPa^0.5 factor as a constant and
    this module converts through the shared unit registry.  If those ever
    disagree the RED stays plausible-looking and every answer is wrong, which
    is exactly what happened the one time the factor was written by hand.
    """
    assert Quantity(value=1.0, unit="MPa^0.5").to("Pa^0.5").value == pytest.approx(1000.0)
    assert Quantity(value=18360.0, unit="Pa^0.5").to("MPa^0.5").value == pytest.approx(18.36)

    sphere = SOLUBILITY_SPHERES["polystyrene"]
    toluene_pa = hansen_triple("Cc1ccccc1")
    expected = relative_energy_difference(toluene_pa, sphere)

    prediction = _polymer_red("polystyrene", "toluene")
    assert prediction.quantity.value == pytest.approx(expected, rel=1e-9)

    # And the Ra printed in the notes is that RED times the radius, not a
    # separately computed distance that could drift away from it.
    ra = expected * sphere.radius
    assert any(f"Ra {ra:.2f}" in note for note in prediction.notes)


def test_a_solvent_sitting_on_the_centre_scores_zero():
    """The degenerate case the derivative-based error bar has to survive."""
    sphere = SOLUBILITY_SPHERES["polystyrene"]
    triple = solvent_triple_from_name("toluene")
    assert triple is not None
    assert hansen_distance(tuple(v / 1000.0 for v in triple), sphere.centre) > 0


def test_a_repeat_unit_is_matched_by_structure_not_by_spelling():
    """Polystyrene written backwards is still polystyrene."""
    assert sphere_for_repeat_unit("[*]CC(c1ccccc1)[*]")[0] == "polystyrene"
    assert sphere_for_repeat_unit("c1ccccc1C([*])C[*]")[0] == "polystyrene"
    assert sphere_for_repeat_unit("[*]CC[*]") is None


# --------------------------------------------------------------------------
# The refusals
# --------------------------------------------------------------------------


def _refused(prediction: Prediction, *fragments: str) -> None:
    assert prediction.status is PredictionStatus.UNSUPPORTED, prediction
    assert prediction.quantity is None
    text = " ".join(prediction.notes).lower()
    for fragment in fragments:
        assert fragment.lower() in text, f"{fragment!r} missing from {text!r}"


def test_no_upstream_triple_refuses_cleanly_rather_than_crashing():
    """The path that has to work if ``polymer_hansen`` never ships."""
    prediction = _predict(_polymer(SPHERE_REPEAT_UNITS["polystyrene"]), _in("toluene"))
    _refused(prediction, "no predicted hansen triple", "hansen_dispersion")


def test_a_partial_upstream_triple_is_not_completed_by_guessing():
    context = _context(PREDICTED_CENTRES["polystyrene"])
    del context["hansen_polar"]
    prediction = _predict(
        _polymer(SPHERE_REPEAT_UNITS["polystyrene"]), _in("toluene"), context
    )
    _refused(prediction, "hansen_polar")


def test_an_unusable_upstream_prediction_counts_as_missing():
    context = _context(PREDICTED_CENTRES["polystyrene"])
    context["hansen_hydrogen_bonding"] = Prediction.unsupported(
        "hansen_hydrogen_bonding", "polymer_hansen", "no increment for this environment"
    )
    prediction = _predict(
        _polymer(SPHERE_REPEAT_UNITS["polystyrene"]), _in("toluene"), context
    )
    _refused(prediction, "hansen_hydrogen_bonding")


def test_an_untabulated_repeat_unit_is_refused_with_the_distance_and_the_evidence():
    """No radius is invented, and the reader is told what was measured instead."""
    prediction = _predict(
        _polymer("[*]CC[*]"), _in("toluene"), _context((17.5, 0.0, 0.0))
    )
    _refused(
        prediction,
        "no fitted interaction radius",
        "Ra = ",
        "0.164 to 0.565",
        "leave-one-out",
        "1.91",
    )


def test_a_copolymer_has_no_sphere():
    monomers = (
        MonomerUnit(smiles="[*]CC(c1ccccc1)[*]", mole_fraction=0.7),
        MonomerUnit(smiles="[*]CC(C#N)[*]", mole_fraction=0.3, role=MonomerRole.COMONOMER),
    )
    prediction = _predict(
        _polymer("", monomers=monomers),
        _in("toluene"),
        _context(PREDICTED_CENTRES["polystyrene"]),
    )
    _refused(prediction, "copolymer", "has never been measured")


def test_a_network_swells_rather_than_dissolving():
    prediction = _predict(
        _polymer(SPHERE_REPEAT_UNITS["polystyrene"], topology=PolymerTopology.NETWORK),
        _in("toluene"),
        _context(PREDICTED_CENTRES["polystyrene"]),
    )
    _refused(prediction, "does not dissolve at any red", "swells")


def test_a_stated_crosslink_density_is_refused_even_on_a_linear_topology():
    prediction = _predict(
        _polymer(
            SPHERE_REPEAT_UNITS["polystyrene"],
            crosslink_density=Quantity(value=50.0, unit="mol/m^3"),
        ),
        _in("toluene"),
        _context(PREDICTED_CENTRES["polystyrene"]),
    )
    _refused(prediction, "crosslink density", "swells")


def test_no_solvent_named_is_refused_with_the_instruction_to_name_one():
    prediction = _predict(
        _polymer(SPHERE_REPEAT_UNITS["polystyrene"]),
        _in(None),
        _context(PREDICTED_CENTRES["polystyrene"]),
    )
    _refused(prediction, "no solvent was named", "environment: 'acetone'")


@pytest.mark.parametrize("environment", ["air", "vacuum"])
def test_an_environment_that_is_not_a_compound_is_refused(environment):
    prediction = _predict(
        _polymer(SPHERE_REPEAT_UNITS["polystyrene"]),
        _in(environment),
        _context(PREDICTED_CENTRES["polystyrene"]),
    )
    _refused(prediction, "resolves to no hansen triple")


def test_a_compound_with_no_hansen_triple_is_refused_rather_than_estimated():
    """Nitrogen resolves to a real CAS number and has no measured triple."""
    assert solvent_triple_from_name("nitrogen") is None
    prediction = _predict(
        _polymer(SPHERE_REPEAT_UNITS["polystyrene"]),
        _in("nitrogen"),
        _context(PREDICTED_CENTRES["polystyrene"]),
    )
    _refused(prediction, "resolves to no hansen triple", "no triple is estimated")


def test_a_predicted_centre_outside_the_fitted_sphere_is_refused():
    """Poly(vinyl chloride): displacement 7.15 against an interaction radius of 3.5.

    This is the identity gate doing its job.  The two descriptions of PVC are
    not of the same material, nothing here can say which is right, and a RED
    from a sphere that does not contain its own polymer is meaningless rather
    than imprecise.
    """
    sphere = SOLUBILITY_SPHERES["poly(vinyl chloride)"]
    displacement = hansen_distance(PREDICTED_CENTRES["poly(vinyl chloride)"], sphere.centre)
    assert displacement > sphere.radius

    prediction = _polymer_red("poly(vinyl chloride)", "tetrahydrofuran")
    _refused(prediction, "further than its own", "7.15", "3.50")


def test_the_gate_lets_a_centre_inside_the_radius_through():
    """The gate is a gate, not a blanket refusal."""
    for name in ("polystyrene", "poly(methyl methacrylate)", "polycarbonate",
                 "poly(vinyl acetate)"):
        sphere = SOLUBILITY_SPHERES[name]
        assert hansen_distance(PREDICTED_CENTRES[name], sphere.centre) < sphere.radius
        assert _polymer_red(name, "acetone").is_usable


# --------------------------------------------------------------------------
# Dimensions and uncertainty
# --------------------------------------------------------------------------


def test_both_properties_are_dimensionless_and_round_trip():
    for prop in PROPERTY_FOR_CLASS.values():
        assert prop in PROPERTY_REGISTRY
        assert get_property(prop).canonical_unit == "dimensionless"

    prediction = _polymer_red("polystyrene", "toluene")
    assert prediction.quantity.value > 0.0
    assert prediction.canonical.value == pytest.approx(prediction.quantity.value)
    assert prediction.canonical_uncertainty.std == pytest.approx(prediction.uncertainty.std)


def test_every_answer_carries_a_positive_spread_with_a_stated_basis():
    for name in ("polystyrene", "poly(methyl methacrylate)", "polycarbonate",
                 "poly(vinyl acetate)"):
        for solvent in ("toluene", "acetone", "hexane", "water"):
            prediction = _polymer_red(name, solvent)
            assert prediction.uncertainty.std > 0.0
            assert prediction.uncertainty.kind is UncertaintyKind.COMBINED
            assert "quadrature" in prediction.uncertainty.basis


def test_the_radius_spread_term_is_present_and_scales_with_red():
    """With no other term moving, doubling RED doubles the radius contribution."""
    near = _polymer_red("polycarbonate", "chloroform")
    far = _polymer_red("polycarbonate", "water")
    assert far.uncertainty.std > near.uncertainty.std
    # The radius term alone is 0.10 * RED, so it can never be smaller than that.
    for prediction in (near, far):
        assert prediction.uncertainty.std >= RADIUS_SPREAD_FRACTION * prediction.quantity.value


def test_a_larger_centre_displacement_widens_the_bar():
    """Polycarbonate agrees with its sphere; poly(vinyl acetate) barely does."""
    spheres = {n: SOLUBILITY_SPHERES[n] for n in ("polycarbonate", "poly(vinyl acetate)")}
    close, far = (
        hansen_distance(PREDICTED_CENTRES[n], spheres[n].centre) / spheres[n].radius
        for n in ("polycarbonate", "poly(vinyl acetate)")
    )
    assert close < far

    tight = _polymer_red("polycarbonate", "acetone")
    wide = _polymer_red("poly(vinyl acetate)", "acetone")
    assert tight.uncertainty.std < wide.uncertainty.std
    assert wide.uncertainty.std > DISPLACEMENT_FRACTION * far * 0.9


def test_the_one_textbook_miss_is_inside_its_own_error_bar():
    """Methanol dissolves poly(vinyl acetate); the handbook sphere says RED 1.29.

    The bar has to be honest about that rather than flattering.  Dropping the
    displacement term would leave this at 1.15 - a confident wrong answer.
    """
    prediction = _polymer_red("poly(vinyl acetate)", "methanol")
    assert prediction.quantity.value > 1.0
    assert prediction.quantity.value - prediction.uncertainty.std < 1.0


def test_a_wider_upstream_spread_does_not_silently_narrow_anything():
    """The upstream bar is reported, not folded in; it must not shrink the answer."""
    narrow = _predict(
        _polymer(SPHERE_REPEAT_UNITS["polystyrene"]),
        _in("toluene"),
        _context(PREDICTED_CENTRES["polystyrene"], std=(0.1, 0.1, 0.1)),
    )
    wide = _predict(
        _polymer(SPHERE_REPEAT_UNITS["polystyrene"]),
        _in("toluene"),
        _context(PREDICTED_CENTRES["polystyrene"], std=(20.0, 20.0, 20.0)),
    )
    assert wide.uncertainty.std == pytest.approx(narrow.uncertainty.std)
    assert any("coarser than the sphere" in note for note in wide.notes)
    assert not any("coarser than the sphere" in note for note in narrow.notes)
    assert wide.applicability.score < narrow.applicability.score


def test_temperature_enters_the_bar_but_only_second_order():
    cold = _polymer_red(
        "polystyrene", "toluene", temperature=Quantity(value=298.15, unit="K")
    )
    warm = _polymer_red(
        "polystyrene", "toluene", temperature=Quantity(value=320.0, unit="K")
    )
    assert warm.uncertainty.std > cold.uncertainty.std
    assert warm.uncertainty.std - cold.uncertainty.std < 0.05


def test_an_unstated_temperature_is_recorded_rather_than_assumed():
    prediction = _predict(
        _polymer(SPHERE_REPEAT_UNITS["polystyrene"]),
        Conditions(environment="toluene"),
        _context(PREDICTED_CENTRES["polystyrene"]),
    )
    assert prediction.is_usable
    assert any("no temperature was stated" in note for note in prediction.notes)


# --------------------------------------------------------------------------
# Applicability domain
# --------------------------------------------------------------------------


def test_a_tabulated_polymer_at_room_temperature_is_in_domain():
    prediction = _polymer_red("polystyrene", "toluene")
    assert prediction.status is PredictionStatus.OK
    assert prediction.applicability.in_domain


def test_elevated_temperature_is_out_of_domain_because_the_radius_grows():
    prediction = _polymer_red(
        "polystyrene", "toluene", temperature=Quantity(value=400.0, unit="K")
    )
    assert prediction.status is PredictionStatus.OUT_OF_DOMAIN
    assert prediction.quantity is not None
    low, high = TEMPERATURE_WINDOW_K
    assert any(f"{low:.0f}-{high:.0f} K" in w for w in prediction.applicability.warnings)
    assert any("R0 grows with temperature" in note for note in prediction.notes)


def test_assess_domain_marks_an_untabulated_repeat_unit_outside():
    expert = PolymerDissolutionExpert()
    outside = expert.assess_domain(_polymer("[*]CC[*]"))
    assert not outside.in_domain
    assert any("no fitted solubility sphere" in w for w in outside.warnings)

    inside = expert.assess_domain(_polymer(SPHERE_REPEAT_UNITS["polystyrene"]))
    assert inside.in_domain


def test_a_regular_backbone_warns_about_crystallinity_without_refusing():
    """Bisphenol-A polycarbonate has no backbone stereocentres."""
    domain = PolymerDissolutionExpert().assess_domain(
        _polymer(SPHERE_REPEAT_UNITS["polycarbonate"])
    )
    assert domain.in_domain
    assert any("crystallise" in w for w in domain.warnings)
    assert _polymer_red("polycarbonate", "chloroform").is_usable


def test_an_oligomer_warns_without_refusing():
    domain = PolymerDissolutionExpert().assess_domain(
        _polymer(
            SPHERE_REPEAT_UNITS["polystyrene"],
            number_average_molar_mass=Quantity(value=2000.0, unit="g/mol"),
        )
    )
    assert domain.in_domain
    assert any("oligomer" in w for w in domain.warnings)
    assert domain.score < 1.0


# --------------------------------------------------------------------------
# Formulations
# --------------------------------------------------------------------------


def test_a_polymer_in_a_solvent_gives_the_same_number_as_the_polymer_alone():
    formulation = _formulation(SPHERE_REPEAT_UNITS["poly(methyl methacrylate)"], "CC(C)=O")
    mixed = _predict(formulation, Conditions.standard())
    assert mixed.status is PredictionStatus.OK
    assert mixed.quantity.value == pytest.approx(0.72, abs=0.02)

    alone = _polymer_red("poly(methyl methacrylate)", "acetone")
    assert mixed.quantity.value == pytest.approx(alone.quantity.value, rel=1e-9)


def test_the_mixture_path_never_uses_the_formulations_averaged_triple():
    """The blend-average trap, asserted rather than hoped for.

    For a MIXTURE candidate the engine's dependency closure puts MixtureExpert's
    volume-fraction-averaged ``hansen_*`` into the context under exactly the
    names this expert declares.  That average already contains the solvent, so
    using it as the polymer's own centre would measure the solvent's distance
    from a point the solvent helped define.  A deliberately absurd average is
    injected here; the answer must not move.
    """
    formulation = _formulation(SPHERE_REPEAT_UNITS["poly(methyl methacrylate)"], "CC(C)=O")
    clean = _predict(formulation, Conditions.standard())
    poisoned = _predict(
        formulation, Conditions.standard(), _context((1.0, 1.0, 1.0), std=(0.1, 0.1, 0.1))
    )
    assert poisoned.status is clean.status
    assert poisoned.quantity.value == pytest.approx(clean.quantity.value, rel=1e-12)
    assert poisoned.uncertainty.std == pytest.approx(clean.uncertainty.std, rel=1e-12)


def test_two_solvents_in_a_formulation_are_refused_rather_than_averaged():
    formulation = Candidate(
        material_class=MaterialClass.MIXTURE,
        conditions=Conditions.standard(),
        mixture=MixtureSpec(
            components=(
                MixtureComponent(
                    role=ComponentRole.SOLUTE,
                    fraction=0.1,
                    polymer=PolymerSpec(
                        monomers=(MonomerUnit(smiles=SPHERE_REPEAT_UNITS["polystyrene"]),)
                    ),
                ),
                MixtureComponent(
                    role=ComponentRole.SOLVENT,
                    fraction=0.5,
                    molecule=MoleculeSpec(smiles="CC(C)=O"),
                ),
                MixtureComponent(
                    role=ComponentRole.CO_SOLVENT,
                    fraction=0.4,
                    molecule=MoleculeSpec(smiles="Cc1ccccc1"),
                ),
            )
        ),
    )
    _refused(_predict(formulation, Conditions.standard()), "2 solvent-role", "volume average")


def test_two_polymers_in_a_formulation_are_refused():
    formulation = Candidate(
        material_class=MaterialClass.MIXTURE,
        conditions=Conditions.standard(),
        mixture=MixtureSpec(
            components=(
                MixtureComponent(
                    role=ComponentRole.SOLUTE,
                    fraction=0.1,
                    polymer=PolymerSpec(
                        monomers=(MonomerUnit(smiles=SPHERE_REPEAT_UNITS["polystyrene"]),)
                    ),
                ),
                MixtureComponent(
                    role=ComponentRole.SOLUTE,
                    fraction=0.1,
                    polymer=PolymerSpec(
                        monomers=(
                            MonomerUnit(
                                smiles=SPHERE_REPEAT_UNITS["poly(methyl methacrylate)"]
                            ),
                        )
                    ),
                ),
                MixtureComponent(
                    role=ComponentRole.SOLVENT,
                    fraction=0.8,
                    molecule=MoleculeSpec(smiles="CC(C)=O"),
                ),
            )
        ),
    )
    _refused(
        _predict(formulation, Conditions.standard()),
        "2 polymer components",
        "no single sphere",
    )


def test_a_formulation_with_no_solvent_role_is_refused():
    formulation = _formulation(
        SPHERE_REPEAT_UNITS["polystyrene"], "CC(C)=O", solvent_role=ComponentRole.PLASTICIZER
    )
    _refused(_predict(formulation, Conditions.standard()), "0 solvent-role")


def test_a_formulation_whose_polymer_has_no_predictable_triple_is_refused():
    """Nylon-6,6: the upstream group table has no amide increment."""
    formulation = _formulation("[*]NCCCCCCNC(=O)CCCCC(=O)[*]", "O")
    _refused(
        _predict(formulation, Conditions.standard()),
        "could not be obtained from the polymer panel",
        "not substituted",
    )


def test_the_two_property_names_are_split_by_class_so_coverage_stays_honest():
    """One number must not enter a Pareto ranking twice under two names."""
    expert = PolymerDissolutionExpert()
    assert expert.covers(RED, MaterialClass.POLYMER)
    assert not expert.covers(MIX_RED, MaterialClass.POLYMER)
    assert expert.covers(MIX_RED, MaterialClass.MIXTURE)
    assert not expert.covers(RED, MaterialClass.MIXTURE)
    assert not expert.covers(RED, MaterialClass.MOLECULE)

    formulation = _formulation(SPHERE_REPEAT_UNITS["poly(methyl methacrylate)"], "CC(C)=O")
    assert (
        expert.predict(
            PredictionRequest(
                candidate=formulation,
                properties=frozenset({RED}),
                conditions=Conditions.standard(),
            )
        )
        == []
    )


# --------------------------------------------------------------------------
# Nothing crashes
# --------------------------------------------------------------------------


def test_empty_conditions_refuse_rather_than_raise():
    prediction = _predict(
        _polymer(SPHERE_REPEAT_UNITS["polystyrene"]),
        Conditions(),
        _context(PREDICTED_CENTRES["polystyrene"]),
    )
    _refused(prediction, "no solvent was named")


def test_an_unparseable_repeat_unit_refuses_rather_than_raises():
    prediction = _predict(
        _polymer("[*]QQ[*]"), _in("toluene"), _context((17.0, 3.0, 3.0))
    )
    _refused(prediction, "no fitted interaction radius")


def test_availability_and_its_reason_agree():
    expert = PolymerDissolutionExpert()
    assert expert.is_available() == (expert.unavailable_reason() == "")


def test_the_expert_declares_what_it_depends_on():
    expert = PolymerDissolutionExpert()
    assert expert.dependencies == frozenset(CENTRE)
    assert expert.supported_classes == frozenset(
        {MaterialClass.POLYMER, MaterialClass.MIXTURE}
    )


# --------------------------------------------------------------------------
# The headline accuracy claim
# --------------------------------------------------------------------------


#: Polymer/solvent behaviour a polymer chemist can state without looking it up.
#: Poly(vinyl chloride) is absent because the identity gate refuses it, so it
#: has no answer to be right or wrong about.
TEXTBOOK = [
    ("polystyrene", "toluene", True),
    ("polystyrene", "benzene", True),
    ("polystyrene", "tetrahydrofuran", True),
    ("polystyrene", "chloroform", True),
    ("polystyrene", "cyclohexanone", True),
    ("polystyrene", "hexane", False),
    ("polystyrene", "methanol", False),
    ("polystyrene", "water", False),
    ("polystyrene", "acetonitrile", False),
    ("polystyrene", "ethanol", False),
    ("poly(methyl methacrylate)", "acetone", True),
    ("poly(methyl methacrylate)", "tetrahydrofuran", True),
    ("poly(methyl methacrylate)", "chloroform", True),
    ("poly(methyl methacrylate)", "dichloromethane", True),
    ("poly(methyl methacrylate)", "hexane", False),
    ("poly(methyl methacrylate)", "water", False),
    ("poly(methyl methacrylate)", "cyclohexane", False),
    ("polycarbonate", "dichloromethane", True),
    ("polycarbonate", "chloroform", True),
    ("polycarbonate", "tetrahydrofuran", True),
    ("polycarbonate", "hexane", False),
    ("polycarbonate", "water", False),
    ("polycarbonate", "methanol", False),
    ("polycarbonate", "ethanol", False),
    ("poly(vinyl acetate)", "acetone", True),
    ("poly(vinyl acetate)", "methanol", True),
    ("poly(vinyl acetate)", "tetrahydrofuran", True),
    ("poly(vinyl acetate)", "ethyl acetate", True),
    ("poly(vinyl acetate)", "hexane", False),
    ("poly(vinyl acetate)", "water", False),
    ("poly(vinyl acetate)", "cyclohexane", False),
]


def test_thirty_of_thirty_one_textbook_calls_and_every_one_inside_its_own_bar():
    """The accuracy this module's docstring claims, measured rather than quoted.

    Two separate assertions, and the second is the one ``formulate calibrate``
    exists to protect: a model with a real error claiming a small one is more
    dangerous to a ranking than a model with a large error that says so.  Every
    call this expert gets wrong has to be a call its own error bar admits it
    could get wrong.
    """
    assert len(TEXTBOOK) == 31

    right = 0
    uncovered = []
    for name, solvent, dissolves in TEXTBOOK:
        prediction = _polymer_red(name, solvent)
        assert prediction.is_usable, f"{name} in {solvent} was refused"
        red, std = prediction.quantity.value, prediction.uncertainty.std
        if (red < 1.0) is dissolves:
            right += 1
            continue
        # Wrong side of the boundary: one sigma must reach across it.
        inside = (red - std < 1.0) if dissolves else (red + std > 1.0)
        if not inside:
            uncovered.append(f"{name}/{solvent}: RED {red:.2f} +- {std:.2f}")

    assert right >= 30, f"only {right}/31 textbook calls correct"
    assert not uncovered, f"wrong calls outside their own one-sigma bound: {uncovered}"


def test_an_empty_polymer_panel_refuses_the_mixture_path_rather_than_crashing():
    """If nothing can supply a repeat-unit triple, there is no identity gate."""
    from formulate.experts.registry import ExpertRegistry

    expert = PolymerDissolutionExpert(panel=ExpertRegistry())
    formulation = _formulation(SPHERE_REPEAT_UNITS["polystyrene"], "Cc1ccccc1")
    out = expert.predict(
        PredictionRequest(
            candidate=formulation,
            properties=frozenset({MIX_RED}),
            conditions=Conditions.standard(),
        )
    )
    _refused(out[0], "could not be obtained from the polymer panel")


def test_two_domain_warnings_do_not_erase_one_another():
    """A coarse upstream bar and an out-of-window temperature must both survive."""
    prediction = _predict(
        _polymer(SPHERE_REPEAT_UNITS["poly(methyl methacrylate)"]),
        _in("acetone", temperature=Quantity(value=400.0, unit="K")),
        _context(PREDICTED_CENTRES["poly(methyl methacrylate)"], std=(20.0, 20.0, 20.0)),
    )
    assert prediction.status is PredictionStatus.OUT_OF_DOMAIN
    warnings = " ".join(prediction.applicability.warnings)
    assert "coarser than the sphere" in warnings
    assert "outside the" in warnings


# --------------------------------------------------------------------------
# Names that resolve to the wrong material
# --------------------------------------------------------------------------


#: What the installed ``chemicals`` compilation hands back for each of these
#: polymer names.  Every one carries a full Hansen triple, so reading
#: ``Conditions.environment`` straight into the compilation returned a RED
#: against the wrong material with no warning anywhere - "polystyrene dissolves
#: polystyrene, RED 0.57, inside the sphere, so a solvent".  Eight of the nine
#: are the monomer, which this repository forbids substituting for its polymer;
#: ``poly(vinyl acetate)`` is not even that, but sec-butyl acetate.
POLYMER_NAMES_THE_COMPILATION_ANSWERS = [
    ("polystyrene", "styrene"),
    ("polyethylene", "ethene"),
    ("polypropylene", "propene"),
    ("poly(vinyl chloride)", "ethene, chloro-"),
    ("polyacrylonitrile", "acrylonitrile"),
    ("polyoxymethylene", "formaldehyde"),
    ("polyethylene glycol", "ethylene glycol"),
    ("poly(vinyl acetate)", "sec-butyl acetate"),
]

#: ``polyurethane`` resolves to ethylurea and ethylurea has no tabulated
#: triple, so the compilation refuses it anyway.  It is kept out of the list
#: above - which asserts that a triple *is* returned - and tested here, because
#: the guard must not depend on the compilation happening to have a gap.
POLYMER_NAME_WITH_NO_TRIPLE = ("polyurethane", "ethylurea")


@pytest.mark.parametrize("name,resolves_to", POLYMER_NAMES_THE_COMPILATION_ANSWERS)
def test_the_compilation_really_does_answer_a_polymer_name_with_another_compound(
    name, resolves_to
):
    """The trap, measured against the installed package rather than asserted.

    If a future ``chemicals`` stops doing this the guard becomes belt and
    braces rather than load-bearing, and this test is where that shows up.
    """
    assert solvent_triple_from_name(name) is not None, (
        f"{name!r} no longer resolves; the guard below is now redundant"
    )
    assert resolves_to in compilation_would_have_used(name)


@pytest.mark.parametrize("name,resolves_to", POLYMER_NAMES_THE_COMPILATION_ANSWERS)
def test_a_polymer_named_as_the_environment_is_refused_not_answered_with_its_monomer(
    name, resolves_to
):
    """A polymer is not its monomer, and the refusal has to say what it caught."""
    assert names_a_polymer(name)
    prediction = _predict(
        _polymer(SPHERE_REPEAT_UNITS["polystyrene"]),
        _in(name),
        _context(PREDICTED_CENTRES["polystyrene"]),
    )
    _refused(prediction, "names a polymer", resolves_to, "not its monomer", "environment: 'acetone'")


def test_a_polymer_name_is_refused_as_a_polymer_even_when_no_triple_follows():
    """The refusal must be about the name, not about a lucky gap in the table."""
    name, resolves_to = POLYMER_NAME_WITH_NO_TRIPLE
    assert solvent_triple_from_name(name) is None
    assert resolves_to in compilation_would_have_used(name)
    _refused(
        _predict(
            _polymer(SPHERE_REPEAT_UNITS["polystyrene"]),
            _in(name),
            _context(PREDICTED_CENTRES["polystyrene"]),
        ),
        "names a polymer",
        "not its monomer",
    )


def test_the_abbreviations_this_repository_already_treats_as_polymers_are_caught_too():
    """``PS`` resolves to a CAS number of its own; ``PMMA`` and ``PC`` do not."""
    for alias in ("PS", "PMMA", "PVC", "PC", "acrylic", "nylon-66", "cellulose acetate"):
        assert names_a_polymer(alias), alias


def test_the_guard_does_not_fire_on_the_liquids_this_expert_exists_for():
    """A leading ``poly`` is the test, and no monomeric solvent is named that way."""
    for solvent in (
        "toluene", "acetone", "hexane", "water", "methanol", "chloroform",
        "tetrahydrofuran", "dimethylformamide", "propylene carbonate",
        "methyl ethyl ketone", "ethyl acetate", "dimethyl sulfoxide",
    ):
        assert not names_a_polymer(solvent), solvent
        assert solvent_triple_from_name(solvent) is not None, solvent

    assert _polymer_red("polystyrene", "toluene").quantity.value == pytest.approx(0.65, abs=0.02)


# --------------------------------------------------------------------------
# The repeat unit is the chain's, not whatever is listed first
# --------------------------------------------------------------------------


def test_an_end_group_listed_first_does_not_hide_the_repeat_unit():
    """``PolymerSpec`` puts end groups in the same tuple, in any order.

    Matching on ``monomers[0]`` refused an end-capped polystyrene and printed
    the distance to the *chain's* predicted centre while calling it the end
    group's, which is a refusal built on a number about a different molecule.
    """
    monomers = (
        MonomerUnit(smiles="CCCC", role=MonomerRole.END_GROUP),
        MonomerUnit(smiles=SPHERE_REPEAT_UNITS["polystyrene"]),
    )
    capped = _polymer("", monomers=monomers)

    assert repeat_unit_smiles(capped.polymer) == SPHERE_REPEAT_UNITS["polystyrene"]

    prediction = _predict(capped, _in("toluene"), _context(PREDICTED_CENTRES["polystyrene"]))
    assert prediction.status is PredictionStatus.OK
    assert prediction.quantity.value == pytest.approx(
        _polymer_red("polystyrene", "toluene").quantity.value, rel=1e-12
    )
    assert PolymerDissolutionExpert().assess_domain(capped).in_domain


def test_an_end_capped_copolymer_is_still_a_copolymer():
    """The end-group filter must not also swallow the copolymer refusal."""
    monomers = (
        MonomerUnit(smiles="CCCC", role=MonomerRole.END_GROUP),
        MonomerUnit(smiles="[*]CC(c1ccccc1)[*]", mole_fraction=0.6),
        MonomerUnit(smiles="[*]CC(C#N)[*]", mole_fraction=0.4, role=MonomerRole.COMONOMER),
    )
    _refused(
        _predict(
            _polymer("", monomers=monomers),
            _in("toluene"),
            _context(PREDICTED_CENTRES["polystyrene"]),
        ),
        "copolymer",
    )


# --------------------------------------------------------------------------
# A formulation's other molecular components
# --------------------------------------------------------------------------


def test_a_second_liquid_under_another_role_is_refused_rather_than_ignored():
    """10% PMMA, 45% acetone, 45% water reported RED 0.72 and said "dissolves".

    The role label is not what makes a molecule part of the liquid.  The
    co-solvent pair was already refused because their effective triple is a
    volume average this expert does not form; the same average moves when the
    second liquid is labelled ``additive``, and half the liquid disappearing
    from the answer is worse than a refusal.
    """
    formulation = Candidate(
        material_class=MaterialClass.MIXTURE,
        conditions=Conditions.standard(),
        mixture=MixtureSpec(
            components=(
                MixtureComponent(
                    role=ComponentRole.SOLUTE,
                    fraction=0.1,
                    polymer=PolymerSpec(
                        monomers=(
                            MonomerUnit(
                                smiles=SPHERE_REPEAT_UNITS["poly(methyl methacrylate)"]
                            ),
                        )
                    ),
                ),
                MixtureComponent(
                    role=ComponentRole.SOLVENT, fraction=0.45,
                    molecule=MoleculeSpec(smiles="CC(C)=O"),
                ),
                MixtureComponent(
                    role=ComponentRole.ADDITIVE, fraction=0.45,
                    molecule=MoleculeSpec(smiles="O"),
                ),
            )
        ),
    )
    _refused(
        _predict(formulation, Conditions.standard()),
        "1 molecular component besides the named solvent",
        "as additive",
        "state the mixed solvent as a single component",
    )


def test_the_two_component_formulation_still_answers():
    """The new refusal must not swallow the case the expert exists for."""
    formulation = _formulation(SPHERE_REPEAT_UNITS["polystyrene"], "Cc1ccccc1")
    prediction = _predict(formulation, Conditions.standard())
    assert prediction.status is PredictionStatus.OK
    assert prediction.quantity.value == pytest.approx(0.65, abs=0.02)


# --------------------------------------------------------------------------
# The identity gate says how well it can see
# --------------------------------------------------------------------------


def test_the_gate_refusal_admits_when_the_margin_is_inside_its_own_noise():
    """Poly(vinyl chloride) clears its radius by less than the spread it is measured with.

    The refusal is still the conservative call, but stating it as a
    demonstrated mismatch would claim a resolution the predicted centre does
    not have - and this repository ranks on error bars precisely so that a
    confident number cannot rest on an unresolved difference.  The refusal has
    to say so, and has to leave the chemist somewhere to go.
    """
    sphere = SOLUBILITY_SPHERES["poly(vinyl chloride)"]
    displacement = hansen_distance(PREDICTED_CENTRES["poly(vinyl chloride)"], sphere.centre)
    gate_sigma = math.sqrt(
        4.0 * UPSTREAM_STD_MPA_SQRT[0] ** 2
        + UPSTREAM_STD_MPA_SQRT[1] ** 2
        + UPSTREAM_STD_MPA_SQRT[2] ** 2
    )
    assert displacement - sphere.radius < gate_sigma

    _refused(
        _polymer_red("poly(vinyl chloride)", "tetrahydrofuran"),
        "further than its own",
        "not resolved",
        "conservative reading",
        f"{gate_sigma:.1f}",
        "solutes: ('poly(vinyl chloride)',)",
    )


def test_a_resolved_mismatch_is_not_hedged():
    """A precise upstream triple that still misses the sphere is a real mismatch."""
    prediction = _predict(
        _polymer(SPHERE_REPEAT_UNITS["poly(vinyl chloride)"]),
        _in("tetrahydrofuran"),
        _context(PREDICTED_CENTRES["poly(vinyl chloride)"], std=(0.05, 0.05, 0.05)),
    )
    _refused(prediction, "further than its own", "disagreement is resolved")
    assert "not resolved" not in " ".join(prediction.notes)


# --------------------------------------------------------------------------
# The constant the error bar rests on
# --------------------------------------------------------------------------


#: The twenty solvents :data:`DISPLACEMENT_FRACTION` was measured over, pinned
#: so the constant can be rechecked rather than believed.  They span the
#: Hansen space the spheres sit in: alkanes, aromatics, chlorinated, ethers,
#: ketones, esters, a nitrile, two dipolar aprotics, four alcohols and water.
DISPLACEMENT_SOLVENTS = [
    "toluene", "benzene", "hexane", "cyclohexane", "chloroform", "dichloromethane",
    "tetrahydrofuran", "1,4-dioxane", "acetone", "2-butanone", "cyclohexanone",
    "ethyl acetate", "acetonitrile", "dimethylformamide", "dimethyl sulfoxide",
    "methanol", "ethanol", "2-propanol", "1-butanol", "water",
]


def test_the_displacement_fraction_is_the_typical_realised_error_not_a_round_number():
    """Re-measure the largest term in the error bar over real pairs.

    ``DISPLACEMENT_FRACTION`` is what makes this expert's bar wide enough to
    admit the one call it gets wrong, so it is the number most worth being
    suspicious of.  It is the mean realised ``|dRa| / displacement`` over the
    hundred pairs below, not a fraction someone liked the look of, and it is
    nowhere near the triangle-inequality cap of 1.0.
    """
    ratios = []
    for name, centre in PREDICTED_CENTRES.items():
        sphere = SOLUBILITY_SPHERES[name]
        displacement = hansen_distance(centre, sphere.centre)
        for solvent in DISPLACEMENT_SOLVENTS:
            triple = solvent_triple_from_name(solvent)
            assert triple is not None, solvent
            mpa = tuple(v / 1000.0 for v in triple)
            moved = abs(hansen_distance(mpa, centre) - hansen_distance(mpa, sphere.centre))
            ratios.append(moved / displacement)

    assert len(ratios) == 100
    mean = sum(ratios) / len(ratios)
    ordered = sorted(ratios)
    median = 0.5 * (ordered[49] + ordered[50])

    assert mean == pytest.approx(0.428, abs=0.02), f"mean is now {mean:.3f}"
    assert median == pytest.approx(0.417, abs=0.02), f"median is now {median:.3f}"
    assert DISPLACEMENT_FRACTION == pytest.approx(mean, abs=0.03)

    # Typical, not worst case: the cap really is attained by some pair, and the
    # constant is not it.  If this ever flipped, the docstring's claim that the
    # term is the typical realised error would have become false.
    assert max(ratios) > 0.9
    above = sum(1 for r in ratios if r > DISPLACEMENT_FRACTION)
    assert 30 <= above <= 60, f"{above}/100 pairs above the constant"


def test_a_partly_covering_polymer_panel_still_gets_a_hansen_expert(monkeypatch):
    """One of the three properties covered is not coverage.

    ``_panel`` used the truthiness of ``registry.coverage``, which is non-empty
    as soon as *any* of the triple is supplied, so a panel carrying one
    component would have left the gap in place and refused every formulation
    for a reason that was not the real one.
    """
    import formulate.experts as experts_pkg
    from formulate.core.properties import PropertyFamily
    from formulate.experts.base import Expert
    from formulate.experts.polymer import PolymerDensityExpert
    from formulate.experts.registry import ExpertRegistry

    class OnlyDispersion(Expert):
        id = "only_dispersion"
        version = "0"
        family = PropertyFamily.INTERFACIAL
        supported_properties = frozenset({"hansen_dispersion"})
        supported_classes = frozenset({MaterialClass.POLYMER})

        def _predict_one(self, prop, request, domain):  # pragma: no cover - never reached
            return None

    # Carries the density the Hansen triple rests on, so the only thing missing
    # is the triple itself - which is the situation being tested.
    partial = ExpertRegistry([OnlyDispersion(), PolymerDensityExpert()])
    assert partial.coverage(CENTRE, MaterialClass.POLYMER), "premise: coverage is non-empty"
    assert partial.uncovered(CENTRE, MaterialClass.POLYMER), "premise: the triple is incomplete"

    monkeypatch.setattr(experts_pkg, "polymer_registry", lambda: partial)
    formulation = _formulation(SPHERE_REPEAT_UNITS["polystyrene"], "Cc1ccccc1")
    prediction = PolymerDissolutionExpert().predict(
        PredictionRequest(
            candidate=formulation,
            properties=frozenset({MIX_RED}),
            conditions=Conditions.standard(),
        )
    )[0]
    assert prediction.status is PredictionStatus.OK
    assert prediction.quantity.value == pytest.approx(0.65, abs=0.02)


def test_an_out_of_domain_upstream_triple_penalises_the_mixture_answer():
    """An out-of-domain prediction is *usable*, which is why it has to be carried.

    The mixture path dispatches its own sub-run, so the sub-prediction's domain
    never reaches the engine on its own.  Dropping it there would let a triple
    ``polymer_hansen`` itself distrusts pass the identity gate and set the error
    bar with no trace, which is the silent substitution this expert is built
    around refusing.
    """
    from formulate.core.properties import PropertyFamily
    from formulate.core.quantity import ApplicabilityDomain
    from formulate.experts.base import Expert
    from formulate.experts.registry import ExpertRegistry

    class ShakyTriple(Expert):
        id = "shaky_triple"
        version = "0"
        family = PropertyFamily.INTERFACIAL
        supported_properties = frozenset(CENTRE)
        supported_classes = frozenset({MaterialClass.POLYMER})

        def assess_domain(self, candidate):
            return ApplicabilityDomain.outside(
                "the packing factor was never fitted to this element", score=0.2
            )

        def _predict_one(self, prop, request, domain):
            value = dict(zip(CENTRE, PREDICTED_CENTRES["polystyrene"]))[prop]
            return self._make(prop, value, "MPa^0.5", request, domain, std=3.9, basis="stub")

    expert = PolymerDissolutionExpert(panel=ExpertRegistry([ShakyTriple()]))
    formulation = _formulation(SPHERE_REPEAT_UNITS["polystyrene"], "Cc1ccccc1")
    prediction = expert.predict(
        PredictionRequest(
            candidate=formulation,
            properties=frozenset({MIX_RED}),
            conditions=Conditions.standard(),
        )
    )[0]

    # Penalised, not refused: the number is still the fitted sphere's.
    assert prediction.quantity.value == pytest.approx(0.65, abs=0.02)
    assert prediction.applicability.score < 1.0
    assert any(
        "predicted triple is itself out of domain" in w
        for w in prediction.applicability.warnings
    )
