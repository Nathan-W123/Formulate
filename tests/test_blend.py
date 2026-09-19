"""Polymer blends: the mixing rules, and what they refuse to mix.

A blend exists to break a coupling. One polymer sets its melt viscosity and
its strength with the same knob, chain length; a blend of two chain lengths
sets them with two. The test that matters most is the one at the bottom, which
checks that the blend actually buys that - a longer load-bearing chain at the
same melt viscosity - rather than just producing numbers.
"""

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
    MonomerUnit,
    PolymerSpec,
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import PredictionStatus
from formulate.core.quantity import Quantity
from formulate.experts.base import PredictionRequest
from formulate.experts.blend import (
    MELT_PROPERTIES,
    SOLID_PROPERTIES,
    PolymerBlendMeltExpert,
    PolymerBlendSolidExpert,
    fox_glass_transition,
    log_additive,
    voigt_reuss,
)

PE, PS = "[*]CC[*]", "[*]CC(c1ccccc1)[*]"


def _blend(*parts, molecule_part=False):
    """parts: (repeat_unit, molar_mass_kg_mol, fraction)."""
    components = [
        MixtureComponent(
            role=ComponentRole.MATRIX if i == 0 else ComponentRole.PLASTICIZER,
            fraction=fraction,
            polymer=PolymerSpec(
                monomers=(MonomerUnit(smiles=repeat),),
                number_average_molar_mass=Quantity(value=mass, unit="kg/mol"),
            ),
        )
        for i, (repeat, mass, fraction) in enumerate(parts)
    ]
    if molecule_part:
        components[-1] = MixtureComponent(
            role=ComponentRole.SOLVENT,
            fraction=components[-1].fraction,
            molecule=MoleculeSpec(smiles="CCCCCC"),
        )
    return Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(components=tuple(components), basis=FractionBasis.MASS),
        conditions=Conditions.standard(),
    )


def _predict(candidate, prop, temperature_k):
    expert = (
        PolymerBlendMeltExpert() if prop in MELT_PROPERTIES else PolymerBlendSolidExpert()
    )
    request = PredictionRequest(
        candidate=candidate,
        properties=frozenset({prop}),
        conditions=Conditions(temperature=Quantity(value=temperature_k, unit="K")),
    )
    return next(p for p in expert.predict(request) if p.property == prop)


# -- the rules themselves --------------------------------------------------


def test_log_additive_viscosity_is_the_geometric_mean_at_equal_parts():
    assert log_additive([1.0, 100.0], [0.5, 0.5]) == pytest.approx(10.0)


def test_voigt_bounds_reuss_from_above_and_they_meet_when_identical():
    voigt, reuss = voigt_reuss([1e9, 1e6], [0.5, 0.5])
    assert voigt > reuss
    same_voigt, same_reuss = voigt_reuss([1e9, 1e9], [0.3, 0.7])
    assert same_voigt == pytest.approx(same_reuss)


def test_fox_lands_between_the_two_transitions():
    tg = fox_glass_transition([200.0, 400.0], [0.5, 0.5])
    assert 200.0 < tg < 400.0


def test_the_melt_and_solid_property_sets_do_not_overlap():
    """They are two experts precisely because they want different conditions."""
    assert not (MELT_PROPERTIES & SOLID_PROPERTIES)


# -- what the expert refuses ----------------------------------------------


@requires_rdkit
def test_an_immiscible_pair_is_refused_rather_than_averaged():
    """Most polymer pairs are immiscible: a chain gains almost no mixing
    entropy, so a blend forms two phases and tracks the continuous one."""
    prediction = _predict(_blend((PE, 20.0, 0.7), (PS, 5.0, 0.3)), "youngs_modulus", 298.15)
    assert prediction.status is PredictionStatus.UNSUPPORTED
    joined = " ".join(prediction.notes)
    assert "Hansen space" in joined or "immiscible far" in joined


@requires_rdkit
def test_a_small_molecule_component_is_refused():
    prediction = _predict(
        _blend((PE, 20.0, 0.7), (PE, 5.0, 0.3), molecule_part=True),
        "youngs_modulus", 298.15,
    )
    assert prediction.status is PredictionStatus.UNSUPPORTED


@requires_rdkit
def test_a_missing_component_value_is_refused_not_averaged_over_one_thing():
    """Polystyrene has no melting point, so neither does a blend containing it."""
    prediction = _predict(_blend((PS, 50.0, 0.7), (PS, 5.0, 0.3)), "melting_point", 298.15)
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "average of one thing" in " ".join(prediction.notes)


# -- what it produces ------------------------------------------------------


@requires_rdkit
def test_a_bimodal_blend_is_scored_on_every_property():
    candidate = _blend((PE, 20.0, 0.7), (PE, 0.8, 0.3))
    for prop in ("shear_viscosity", "surface_tension"):
        assert _predict(candidate, prop, 473.15).quantity is not None
    for prop in ("youngs_modulus", "melting_point", "amorphous_density"):
        assert _predict(candidate, prop, 298.15).quantity is not None


@requires_rdkit
def test_the_modulus_uncertainty_spans_the_bounds_and_is_never_negative():
    """Identical components make the bounds coincide, and float error there
    once produced a negative standard deviation."""
    prediction = _predict(_blend((PE, 20.0, 0.7), (PE, 20.1, 0.3)), "youngs_modulus", 298.15)
    assert prediction.uncertainty.std >= 0.0


@requires_rdkit
def test_more_wax_thins_the_melt():
    thin = _predict(_blend((PE, 20.0, 0.55), (PE, 0.8, 0.45)), "shear_viscosity", 473.15)
    thick = _predict(_blend((PE, 20.0, 0.85), (PE, 0.8, 0.15)), "shear_viscosity", 473.15)
    assert thin.quantity.value < thick.quantity.value


def _single_polymer_viscosity(mass_kg_mol: float, temperature_k: float) -> float:
    """The same polymer alone, through the same panel the blend expert uses."""
    from formulate.experts import polymer_registry
    from formulate.experts.base import PredictionRequest as Req

    registry = polymer_registry()
    candidate = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(
            monomers=(MonomerUnit(smiles=PE),),
            number_average_molar_mass=Quantity(value=mass_kg_mol, unit="kg/mol"),
        ),
        conditions=Conditions.standard(),
    )
    wanted = frozenset(
        {"shear_viscosity", "glass_transition_temperature",
         "entanglement_molar_mass", "amorphous_density"}
    )
    context = {}
    for expert in registry.resolution_order(
        registry.experts_for(wanted, MaterialClass.POLYMER)
    ):
        melt_state = "shear_viscosity" in expert.supported_properties
        conditions = (
            Conditions(temperature=Quantity(value=temperature_k, unit="K"))
            if melt_state
            else candidate.conditions
        )
        request = Req(
            candidate=candidate,
            properties=wanted,
            conditions=conditions,
            context=dict(context),
        )
        for prediction in expert.predict(request):
            if prediction.is_usable:
                context.setdefault(prediction.property, prediction)
    return context["shear_viscosity"].quantity.to_canonical().value


@requires_rdkit
def test_blending_buys_a_longer_backbone_at_the_same_melt_viscosity():
    """The whole point of the module.

    A 20 kg/mol polyethylene is far too thick to push through the nozzle. The
    same 20 kg/mol backbone, blended with 30% of a wax, lands inside the
    window - so the blend carries a chain two and a half times longer than the
    single polymer that fits, which is the coupling broken.
    """
    blended = _predict(_blend((PE, 20.0, 0.7), (PE, 0.8, 0.3)), "shear_viscosity", 473.15)
    alone = _single_polymer_viscosity(20.0, 473.15)

    assert blended.quantity.value < alone / 3.0, "the wax must actually thin it"
    assert 1.0 <= blended.quantity.value <= 20.0, "and land inside the nozzle window"
    assert alone > 20.0, "while the same backbone alone does not"


# -- the explorer ----------------------------------------------------------


def test_the_explorer_proposes_bimodal_blends_of_one_chemistry():
    from formulate.exploration import PolymerBlendExplorer
    from formulate.targets.spec import TargetSpec

    spec = TargetSpec.from_dict(
        {
            "material_classes": ["mixture"],
            "requirements": [
                {"property": "shear_viscosity", "direction": "minimize",
                 "lower": "1 Pa*s", "upper": "20 Pa*s"}
            ],
        }
    )
    proposals = PolymerBlendExplorer(cross_polymer=False).propose(spec, 6)
    assert len(proposals) == 6
    for candidate in proposals:
        assert candidate.material_class is MaterialClass.MIXTURE
        components = candidate.mixture.components
        assert len(components) == 2
        masses = {c.polymer.number_average_molar_mass.value for c in components}
        assert len(masses) == 2, "bimodal means two different chain lengths"


def test_the_explorer_declines_a_spec_that_did_not_ask_for_mixtures():
    from formulate.exploration import PolymerBlendExplorer
    from formulate.targets.spec import TargetSpec

    spec = TargetSpec.from_dict(
        {
            "material_classes": ["polymer"],
            "requirements": [
                {"property": "shear_viscosity", "direction": "minimize",
                 "lower": "1 Pa*s", "upper": "20 Pa*s"}
            ],
        }
    )
    assert PolymerBlendExplorer().propose(spec, 5) == []
