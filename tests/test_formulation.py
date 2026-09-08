"""Phase 4: Hansen parameters, measured lookups and formulation mixing rules."""

from __future__ import annotations

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
    PhaseAssumption,
    molecule_candidate,
)
from formulate.core.conditions import Conditions
from formulate.experts import default_registry
from formulate.experts.base import PredictionRequest
from formulate.experts.hansen import (
    HansenSolubilityExpert,
    hansen_distance,
    hansen_triple,
    resolve_cas,
    total_solubility_parameter,
)
from formulate.experts.measured import MeasuredPropertyExpert, measured_value
from formulate.experts.mixture import MixtureExpert

pytestmark = requires_rdkit

hansen_only = pytest.mark.skipif(
    not HansenSolubilityExpert().is_available(), reason="chemicals is not installed"
)


def _blend(pairs, basis=FractionBasis.VOLUME, phase=PhaseAssumption.SINGLE_PHASE):
    return Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=tuple(
                MixtureComponent(
                    role=ComponentRole.SOLVENT, fraction=f, molecule=MoleculeSpec(smiles=s)
                )
                for s, f in pairs
            ),
            basis=basis,
            phase_assumption=phase,
        ),
    )


def _predict(expert, candidate, properties=None):
    request = PredictionRequest(
        candidate=candidate,
        properties=frozenset(properties or expert.supported_properties),
        conditions=Conditions.standard(),
    )
    return {p.property: p for p in expert.predict(request)}


# -- structure resolution --------------------------------------------------


@hansen_only
def test_structures_resolve_through_inchikey_not_smiles():
    """A SMILES lookup fails on benzene, toluene, acetone and DMSO."""
    for smiles in ("CCO", "c1ccccc1", "Cc1ccccc1", "CC(C)=O", "CS(C)=O", "O"):
        assert resolve_cas(smiles) is not None, smiles


@hansen_only
def test_an_unknown_structure_resolves_to_nothing_rather_than_guessing():
    assert resolve_cas("CC(C)(C)C(C)(C)C(C)(C)C(C)(C)C(C)(C)N1CCCCC1") is None


# -- Hansen parameters -----------------------------------------------------


@hansen_only
@pytest.mark.parametrize(
    "smiles,name,expected",
    [
        ("CCO", "ethanol", (15.8, 8.8, 19.4)),
        ("Cc1ccccc1", "toluene", (18.0, 1.4, 2.0)),
        ("O", "water", (15.5, 16.0, 42.3)),
        ("CC(C)=O", "acetone", (15.5, 10.4, 7.0)),
    ],
)
def test_hansen_parameters_match_the_published_values(smiles, name, expected):
    triple = hansen_triple(smiles)
    assert triple is not None
    for got, want in zip(triple, expected):
        assert got / 1000.0 == pytest.approx(want, abs=0.3), name


@hansen_only
def test_the_three_components_reconstruct_the_hildebrand_parameter():
    """delta^2 = dD^2 + dP^2 + dH^2 is what makes this a decomposition."""
    for smiles, hildebrand in [("c1ccccc1", 18.7), ("CCO", 26.5), ("CCCCCC", 14.9)]:
        total = total_solubility_parameter(hansen_triple(smiles)) / 1000.0
        assert total == pytest.approx(hildebrand, abs=0.5), smiles


@hansen_only
def test_hansen_distance_orders_pairs_by_chemical_similarity():
    toluene, hexane, water = (hansen_triple(s) for s in ("Cc1ccccc1", "CCCCCC", "O"))
    assert hansen_distance(toluene, hexane) < hansen_distance(toluene, water)
    assert hansen_distance(toluene, toluene) == pytest.approx(0.0)


@hansen_only
def test_the_dispersion_term_carries_a_factor_of_four():
    """Dropping it distorts every distance; this pins the convention."""
    a = (10_000.0, 0.0, 0.0)
    b = (11_000.0, 0.0, 0.0)
    assert hansen_distance(a, b) == pytest.approx(2000.0)  # 2 * 1000, not 1000


@hansen_only
def test_a_compound_outside_the_compilation_gets_no_estimate():
    """Three components cannot be recovered from one total parameter."""
    expert = HansenSolubilityExpert()
    exotic = molecule_candidate("CC(C)(C)C(C)(C)C(C)(C)C(C)(C)C(C)(C)N1CCCCC1")
    for prediction in _predict(expert, exotic).values():
        assert not prediction.is_usable
        assert "not in the Hansen compilation" in prediction.notes[0]


# -- measured lookups ------------------------------------------------------


@hansen_only
def test_measured_values_match_experiment():
    for smiles, kelvin in [("CCO", 351.6), ("Cc1ccccc1", 383.7), ("O", 373.1)]:
        assert measured_value("normal_boiling_point", smiles) == pytest.approx(kelvin, abs=0.5)


@hansen_only
def test_the_panel_prefers_a_measurement_over_an_estimate():
    """No precedence rule: the measurement wins on uncertainty alone."""
    from formulate.evaluation.engine import prefer

    registry = default_registry()
    candidate = molecule_candidate("Cc1ccccc1")
    measured = _predict(
        MeasuredPropertyExpert(), candidate, ["normal_boiling_point"]
    )["normal_boiling_point"]
    joback = _predict(registry.get("joback"), candidate, ["normal_boiling_point"])[
        "normal_boiling_point"
    ]
    assert measured.is_usable and joback.is_usable
    assert prefer(measured, joback)
    assert not prefer(joback, measured)


@hansen_only
def test_joback_uses_a_measured_boiling_point_for_the_critical_temperature():
    """Joback and Reid intend the correlation to take a measured Tb when there is one."""
    registry = default_registry()
    order = [
        e.id
        for e in registry.resolution_order(
            registry.experts_for(
                ["critical_temperature", "normal_boiling_point"], MaterialClass.MOLECULE
            )
        )
    ]
    assert order.index("measured") < order.index("joback")


@hansen_only
def test_a_substance_with_no_compiled_measurement_says_so():
    expert = MeasuredPropertyExpert()
    exotic = molecule_candidate("CC(C)(C)C(C)(C)C(C)(C)C(C)(C)C(C)(C)N1CCCCC1")
    for prediction in _predict(expert, exotic).values():
        assert not prediction.is_usable


# -- formulation mixing ----------------------------------------------------


@hansen_only
def test_a_blend_density_lies_between_its_components():
    expert = MixtureExpert()
    result = _predict(expert, _blend([("Cc1ccccc1", 0.5), ("CCCCCC", 0.5)]), ["liquid_density"])
    density = result["liquid_density"]
    assert density.is_usable
    grams_per_cm3 = density.quantity.to("g/cm^3").value
    # Toluene 0.862, hexane 0.655; an ideal half-and-half blend is about 0.76.
    assert 0.62 < grams_per_cm3 < 0.90


@hansen_only
def test_blend_hansen_parameters_lie_between_the_components():
    expert = MixtureExpert()
    result = _predict(
        expert, _blend([("Cc1ccccc1", 0.5), ("CC(C)=O", 0.5)]), ["hansen_polar"]
    )
    polar = result["hansen_polar"].quantity.to("Pa^0.5").value / 1000.0
    # Toluene 1.4, acetone 10.4.
    assert 1.4 < polar < 10.4


@hansen_only
def test_the_worst_pair_drives_the_reported_hansen_distance():
    expert = MixtureExpert()
    compatible = _predict(
        expert, _blend([("Cc1ccccc1", 0.5), ("CCCCCC", 0.5)]), ["hansen_distance"]
    )["hansen_distance"]
    incompatible = _predict(
        expert, _blend([("Cc1ccccc1", 0.5), ("O", 0.5)]), ["hansen_distance"]
    )["hansen_distance"]
    assert incompatible.quantity.to("Pa^0.5").value > compatible.quantity.to("Pa^0.5").value


@hansen_only
def test_no_relative_energy_difference_is_invented_without_an_interaction_radius():
    """RED needs a measured R0; without one it cannot be formed."""
    expert = MixtureExpert()
    result = _predict(
        expert, _blend([("Cc1ccccc1", 0.5), ("CCCCCC", 0.5)]), ["hansen_distance"]
    )["hansen_distance"]
    assert any("interaction radius" in note for note in result.notes)
    assert "relative_energy_difference" not in expert.supported_properties


@hansen_only
def test_a_component_without_hansen_data_blocks_the_blend_average():
    """Averaging over a partial set would describe a different formulation.

    The example is benzoyl peroxide, and it had to change: this used to be a
    hindered amine, which the group-contribution route now covers. A component
    is only uncoverable if nothing can place it at all, and a peroxide is -
    there is no -O-O- group in the table and it is refused rather than summed
    without one.
    """
    expert = MixtureExpert()
    result = _predict(
        expert,
        _blend([("Cc1ccccc1", 0.5), ("O=C(OOC(=O)c1ccccc1)c1ccccc1", 0.5)]),
        ["hansen_dispersion"],
    )["hansen_dispersion"]
    assert not result.is_usable


def test_an_estimated_component_no_longer_blocks_the_blend():
    """The counterpart: a structure off the compilation is placeable now.

    Hexanediol diacrylate is in no Hansen compilation, and before the group
    route existed a blend containing it lost all three components at once.
    """
    expert = MixtureExpert()
    result = _predict(
        expert,
        _blend([("Cc1ccccc1", 0.5), ("C=CC(=O)OCCCCCCOC(=O)C=C", 0.5)]),
        ["hansen_dispersion"],
    )["hansen_dispersion"]
    assert result.is_usable
    assert 15.0 < result.quantity.to("MPa^0.5").value < 22.0


@hansen_only
def test_volume_and_mass_bases_give_different_blends():
    """A mass fraction used as a volume fraction is a silent error of tens of percent."""
    expert = MixtureExpert()
    pairs = [("Cc1ccccc1", 0.5), ("CCCCCC", 0.5)]
    by_volume = _predict(expert, _blend(pairs, FractionBasis.VOLUME), ["liquid_density"])
    by_mass = _predict(expert, _blend(pairs, FractionBasis.MASS), ["liquid_density"])
    assert by_volume["liquid_density"].quantity.value != pytest.approx(
        by_mass["liquid_density"].quantity.value
    )


@hansen_only
def test_a_declared_dispersion_is_flagged_as_not_a_single_phase():
    expert = MixtureExpert()
    candidate = _blend(
        [("Cc1ccccc1", 0.5), ("O", 0.5)], phase=PhaseAssumption.EMULSION
    )
    domain = expert.assess_domain(candidate)
    assert not domain.in_domain
    assert any("not a single homogeneous phase" in w for w in domain.warnings)


def test_the_mixture_expert_covers_the_mixture_class():
    registry = default_registry()
    covered = {
        p
        for e in registry
        for p in e.supported_properties
        if MaterialClass.MIXTURE in e.supported_classes
    }
    assert "hansen_distance" in covered
    assert "liquid_density" in covered


# --------------------------------------------------------------------------
# The component that used to take the whole blend down with it
# --------------------------------------------------------------------------


@requires_rdkit
def test_a_blend_containing_water_gets_a_density_and_hansen_parameters():
    """Water broke every one of these, and nothing said so in physical terms.

    The chain was: blend density needs component densities, which came from a
    corresponding-states correlation, which needs a critical temperature, which
    came from Joback group contribution, which raises "zero matching groups"
    on a molecule with no carbon. So a blend containing the commonest
    formulation solvent there is declined its density and, because volume
    fractions could not be formed without it, all three volume-weighted Hansen
    parameters as well.
    """
    blend = _blend([("CCO", 0.5), ("O", 0.5)])
    predictions = _predict(MixtureExpert(), blend)

    for prop in ("liquid_density", "hansen_dispersion", "hansen_polar",
                 "hansen_hydrogen_bonding"):
        assert predictions[prop].quantity is not None, f"{prop} declined on a water blend"


@requires_rdkit
def test_the_blend_density_is_the_ideal_volume_average_of_its_components():
    """Stated rather than inferred, because the rule is an assumption.

    Half and half by volume of ethanol at 785 and water at 997 kg/m^3 is 891 on
    ideal mixing. The real value is nearer 914: ethanol and water contract on
    mixing by about two and a half per cent, which is exactly the excess volume
    an ideal rule cannot know about. The prediction's uncertainty has to be
    wide enough to cover that or the number is a trap.
    """
    blend = _blend([("CCO", 0.5), ("O", 0.5)])
    density = _predict(MixtureExpert(), blend, ["liquid_density"])["liquid_density"]

    assert density.quantity.value == pytest.approx(891.0, abs=2.0)
    spread = density.uncertainty.std
    assert spread is not None
    assert density.quantity.value + 2 * spread >= 914.0, (
        "the stated uncertainty does not reach the measured density, so an ideal "
        "mixing rule is being presented as if excess volume did not exist"
    )


@requires_rdkit
def test_a_measured_density_is_preferred_over_the_correlation_that_needs_joback():
    from formulate.experts.measured import measured_value

    # Methanol is where the correlation is worst: it is out by a third of a
    # gram per cubic centimetre, which is forty per cent.
    assert measured_value("liquid_density", "CO") == pytest.approx(786.6, abs=2.0)
    assert measured_value("critical_temperature", "CO") == pytest.approx(512.5, abs=1.0)
