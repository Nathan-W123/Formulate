"""Moduli, entanglement, and the strength that is not predicted.

The negative is as much the subject as the positives. Tensile and shear
strength are set by the largest flaw in a specimen, which is a property of how
it was made, so they are absent from the property registry and a test here
keeps them absent.
"""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import polymer_candidate
from formulate.core.conditions import Conditions
from formulate.core.prediction import PredictionStatus
from formulate.core.properties import PROPERTY_REGISTRY
from formulate.core.quantity import Quantity
from formulate.experts.mechanical import (
    CHAIN_DIMENSIONS,
    entanglement_model,
    is_tough,
    packing_length,
    shear_from_young,
    theoretical_strength,
    youngs_modulus,
)

PS = "[*]CC(c1ccccc1)[*]"
PI = "[*]CC(C)=CC[*]"
PE = "[*]CC[*]"
ROOM = Conditions(
    temperature=Quantity(value=298.15, unit="K"), pressure=Quantity(value=1.0, unit="atm")
)


# -- what is not built ----------------------------------------------------


def test_tensile_and_shear_strength_are_not_properties_of_a_structure():
    """Griffith: a specimen fails at its largest flaw, which is how it was made.

    Two bars of identical polymer differ two- or three-fold on processing. There
    is no honest way to register these, so they are not registered, and this
    test is what keeps a well-meaning future change from adding them.
    """
    assert "tensile_strength" not in PROPERTY_REGISTRY
    assert "shear_strength" not in PROPERTY_REGISTRY
    assert "elongation_at_break" not in PROPERTY_REGISTRY


def test_the_bound_that_is_built_is_named_apart_from_the_thing_it_is_not():
    definition = PROPERTY_REGISTRY["theoretical_strength"]
    assert "not a tensile strength" in definition.description.lower()


# -- packing length and entanglement --------------------------------------


def test_packing_length_reproduces_the_published_values():
    """1.69 A for polyethylene and 3.92 for polystyrene, which is what makes
    the cube law below a check rather than a fit."""
    assert packing_length(1.25, 0.784) == pytest.approx(1.69, abs=0.02)
    assert packing_length(0.437, 0.969) == pytest.approx(3.92, abs=0.02)


def test_the_entanglement_constant_transfers_to_withheld_polymers():
    model = entanglement_model()
    assert model.n_fit >= 4
    assert model.validation_n >= 3
    # Entanglement mass spans sixteen-fold across the set; reproducing withheld
    # polymers to under a factor of two is what makes it usable for classing a
    # polymer brittle or tough.
    assert model.validation_spread < 2.0


def test_a_repeat_unit_with_no_measured_chain_dimension_is_refused():
    """It comes from scattering or an RIS calculation, not from the structure."""
    from formulate.experts.mechanical import PolymerMechanicalExpert

    expert = PolymerMechanicalExpert()
    domain = expert.assess_domain(polymer_candidate("[*]CC(Cl)[*]"))  # PVC, untabulated
    assert not domain.in_domain


# -- the branch that decides everything -----------------------------------


def test_the_glass_transition_moves_the_modulus_by_three_orders_of_magnitude():
    glassy, branch_g = youngs_modulus(298.15, 373.0, 1.05, 18100.0)
    rubbery, branch_r = youngs_modulus(298.15, 200.0, 0.91, 6200.0)
    assert branch_g == "glassy" and branch_r == "rubbery"
    assert glassy / rubbery > 1000.0


def test_the_rubbery_branch_reproduces_a_measured_rubber():
    """cis-polyisoprene measures about 1.5 MPa at room temperature."""
    modulus, branch = youngs_modulus(298.15, 200.0, 0.91, 6200.0)
    assert branch == "rubbery"
    assert 0.5e6 < modulus < 3.0e6


def test_the_glassy_branch_lands_in_the_measured_band():
    """Nine amorphous polymers span 2.0 to 3.5 GPa."""
    modulus, _ = youngs_modulus(298.15, 373.0, 1.05, 18100.0)
    assert 2.0e9 <= modulus <= 3.5e9


def test_a_rubbery_modulus_without_an_entanglement_mass_is_an_error_not_a_guess():
    with pytest.raises(ValueError):
        youngs_modulus(298.15, 200.0, 0.91, None)


def test_shear_follows_young_and_a_rubber_is_nearly_incompressible():
    assert shear_from_young(3.0e9, "glassy") == pytest.approx(3.0e9 / 2.7, rel=1e-6)
    assert shear_from_young(1.5e6, "rubbery") == pytest.approx(1.5e6 / 3.0, rel=1e-3)


def test_theoretical_strength_is_a_tenth_of_the_modulus():
    assert theoretical_strength(3.0e9) == pytest.approx(3.0e8)


def test_toughness_is_decided_by_chain_length_against_entanglement():
    assert is_tough(100_000.0, 18_100.0) is True     # many entanglements
    assert is_tough(20_000.0, 18_100.0) is False     # barely one, so brittle
    assert is_tough(None, 18_100.0) is None          # unstated is not assumed


# -- through the engine ---------------------------------------------------




@requires_rdkit
@pytest.mark.parametrize(
    "smiles,prop,low,high",
    [
        (PS, "youngs_modulus", 2.0e9, 3.5e9),
        (PI, "youngs_modulus", 0.5e6, 3.0e6),
        (PS, "entanglement_molar_mass", 9_000.0, 30_000.0),
        (PI, "entanglement_molar_mass", 3_000.0, 12_000.0),
    ],
)
def test_predictions_through_the_engine_land_where_measurements_do(smiles, prop, low, high):
    from formulate.evaluation.engine import EvaluationEngine
    from formulate.experts import default_registry
    from formulate.targets.spec import TargetSpec

    spec = TargetSpec.from_dict(
        {
            "name": "mechanical",
            "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
            "requirements": [
                {
                    "property": prop,
                    "direction": "maximize",
                    "lower": f"{low / 10:g} {'g/mol' if 'molar' in prop else 'Pa'}",
                    "upper": f"{high * 10:g} {'g/mol' if 'molar' in prop else 'Pa'}",
                }
            ],
        }
    )
    candidate = polymer_candidate(smiles)
    predictions, _ = EvaluationEngine(default_registry()).predict([candidate], spec)
    match = [p for p in predictions[0] if p.property == prop and p.quantity is not None]
    assert match, [p.status for p in predictions[0] if p.property == prop]
    unit = "g/mol" if "molar" in prop else "Pa"
    assert low <= match[0].quantity.to(unit).value <= high


@requires_rdkit
def test_a_semicrystalline_polymer_says_the_answer_is_for_the_amorphous_phase():
    """Polyethylene is above its transition, so the rubbery branch fires and
    returns about 8 MPa, where a real bar is nearer 800 because it crystallises."""
    from formulate.evaluation.engine import EvaluationEngine
    from formulate.experts import default_registry
    from formulate.targets.spec import TargetSpec

    spec = TargetSpec.from_dict(
        {
            "name": "pe",
            "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
            "requirements": [
                {"property": "youngs_modulus", "direction": "maximize",
                 "lower": "1e5 Pa", "upper": "1e10 Pa"}
            ],
        }
    )
    predictions, _ = EvaluationEngine(default_registry()).predict([polymer_candidate(PE)], spec)
    match = [p for p in predictions[0] if p.property == "youngs_modulus" and p.quantity]
    assert match
    assert any("crystallis" in note for note in match[0].notes)


@requires_rdkit
def test_a_polymer_with_no_glass_transition_gets_no_modulus():
    """Choosing the wrong branch is a factor of two thousand, so it is refused.

    Poly(dimethylsiloxane) has no Joback group, so its transition is declined,
    and the mechanical expert declines in turn rather than picking a branch.
    """
    from formulate.evaluation.engine import EvaluationEngine
    from formulate.experts import default_registry
    from formulate.targets.spec import TargetSpec

    spec = TargetSpec.from_dict(
        {
            "name": "pdms",
            "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
            "requirements": [
                {"property": "youngs_modulus", "direction": "maximize",
                 "lower": "1e5 Pa", "upper": "1e10 Pa"}
            ],
        }
    )
    predictions, _ = EvaluationEngine(default_registry()).predict(
        [polymer_candidate("[*][Si](C)(C)O[*]")], spec
    )
    match = [p for p in predictions[0] if p.property == "youngs_modulus"]
    assert match
    assert match[0].status is PredictionStatus.UNSUPPORTED


def test_every_tabulated_chain_states_where_it_came_from():
    for entry in CHAIN_DIMENSIONS.values():
        assert entry.source
        assert entry.r2_per_mass > 0
        assert entry.split in {"fit", "validation"}
