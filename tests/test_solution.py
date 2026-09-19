"""A polymer dissolved in a solvent: the spinning dope.

The material a web shooter actually holds, and the one thing the mixture
experts could not score. The checks that matter here are the two branches
joining where both hold, the refusals, and the fact that a dope comes out in
the viscosity band a dope actually occupies.
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
from formulate.experts.solution import (
    ENTANGLED_EXPONENT,
    HUGGINS_COEFFICIENT,
    INTRINSIC_VISCOSITY_SPREAD,
    PolymerSolutionExpert,
    intrinsic_viscosity,
    overlap_concentration,
    solution_viscosity,
)

PS = "[*]CC(c1ccccc1)[*]"
PE = "[*]CC[*]"
TOLUENE, ACETONE = "Cc1ccccc1", "CC(C)=O"


def _polymer_component(repeat, kg_mol, fraction):
    return MixtureComponent(
        role=ComponentRole.SOLUTE,
        fraction=fraction,
        polymer=PolymerSpec(
            monomers=(MonomerUnit(smiles=repeat),),
            number_average_molar_mass=(
                Quantity(value=kg_mol, unit="kg/mol") if kg_mol is not None else None
            ),
        ),
    )


def _solvent_component(smiles, fraction, role=ComponentRole.SOLVENT):
    return MixtureComponent(role=role, fraction=fraction, molecule=MoleculeSpec(smiles=smiles))


def _candidate(*components):
    return Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(components=tuple(components), basis=FractionBasis.MASS),
        conditions=Conditions.standard(),
    )


def _dope(repeat=PS, kg_mol=200.0, fraction=0.20, solvent=ACETONE):
    return _candidate(
        _polymer_component(repeat, kg_mol, fraction),
        _solvent_component(solvent, 1.0 - fraction),
    )


def _predict(candidate, temperature_k=298.15):
    request = PredictionRequest(
        candidate=candidate,
        properties=frozenset({"shear_viscosity"}),
        conditions=Conditions(temperature=Quantity(value=temperature_k, unit="K")),
    )
    return PolymerSolutionExpert().predict(request)[0]


# -- the physics, without the engine around it -----------------------------


def test_flory_fox_reproduces_mark_houwink_intrinsic_viscosities():
    """The one place a solution viscosity is normally a table.

    Mark-Houwink K and a are quoted per polymer, per solvent and per
    temperature. Going through the chain dimension instead means any repeat
    unit can be asked, and this is what that costs.
    """
    from formulate.experts.mechanical import predicted_chain_dimension

    # Theta-state values computed from published Mark-Houwink constants.
    cases = [
        (PE, 100_000, 0.85),
        (PS, 100_000, 0.259),
        (PS, 200_000, 0.367),
        ("[*]CC(C)(C(=O)OC)[*]", 200_000, 0.243),
        ("[*]CC(C)[*]", 200_000, 0.42),
    ]
    worst = 0.0
    for repeat, mass, measured in cases:
        r2 = predicted_chain_dimension(repeat)
        assert r2 is not None, repeat
        predicted = intrinsic_viscosity(r2, mass)
        ratio = max(predicted / measured, measured / predicted)
        worst = max(worst, ratio)
    assert worst <= INTRINSIC_VISCOSITY_SPREAD


def test_the_two_concentration_branches_join_where_both_hold():
    """Rouse joins reptation at the critical mass in melt.py for the same
    reason: an extrapolation of one law into the other's regime is not a
    model of anything."""
    intrinsic = 0.5
    just_below = solution_viscosity(1e-3, intrinsic, (1.0 - 1e-9) / intrinsic)
    just_above = solution_viscosity(1e-3, intrinsic, (1.0 + 1e-9) / intrinsic)
    assert just_below[1] == "dilute" and just_above[1] == "entangled"
    assert just_above[0] == pytest.approx(just_below[0], rel=1e-6)


def test_the_dilute_branch_is_huggins():
    intrinsic, concentration = 0.4, 0.5
    value, branch = solution_viscosity(1.0, intrinsic, concentration)
    overlap = intrinsic * concentration
    assert branch == "dilute"
    assert value == pytest.approx(1.0 + overlap + HUGGINS_COEFFICIENT * overlap**2)


def test_the_entangled_exponent_is_not_the_melt_exponent():
    """3.4 is the exponent in CHAIN LENGTH at fixed concentration. Here the
    variable is concentration, which adds entanglements and shrinks the
    screening length at once, so the exponent is higher."""
    from formulate.experts.melt import REPTATION_EXPONENT

    assert ENTANGLED_EXPONENT > REPTATION_EXPONENT


def test_overlap_concentration_is_the_reciprocal_intrinsic_viscosity():
    assert overlap_concentration(0.5) == pytest.approx(2.0)


# -- through the expert ----------------------------------------------------


@requires_rdkit
def test_a_dope_lands_in_the_band_a_dope_occupies():
    """A spinning dope is roughly 1 to 100 Pa s. That is not a coincidence -
    it is what makes one drawable, and a model that puts one at a millipascal
    second has not described a dope."""
    prediction = _predict(_dope(fraction=0.20, kg_mol=200.0))
    assert prediction.quantity is not None
    assert 0.1 < prediction.quantity.to("Pa*s").value < 100.0


@requires_rdkit
def test_more_polymer_is_thicker():
    thin = _predict(_dope(fraction=0.05)).quantity.to("Pa*s").value
    thick = _predict(_dope(fraction=0.25)).quantity.to("Pa*s").value
    assert thick > 10.0 * thin


@requires_rdkit
def test_a_longer_chain_is_thicker_at_the_same_concentration():
    short = _predict(_dope(kg_mol=20.0)).quantity.to("Pa*s").value
    long = _predict(_dope(kg_mol=400.0)).quantity.to("Pa*s").value
    assert long > short


@requires_rdkit
def test_a_dilute_solution_is_barely_thicker_than_its_solvent():
    """The sanity check at the other end: at a half per cent the polymer is
    a perturbation, and a model that says otherwise is wrong about the
    dilute limit rather than merely imprecise."""
    prediction = _predict(_dope(fraction=0.005, solvent=TOLUENE))
    assert prediction.quantity is not None
    assert prediction.quantity.to("Pa*s").value < 5.0e-3


# -- the refusals, which are the point -------------------------------------


@requires_rdkit
def test_an_all_polymer_formulation_is_refused_as_a_blend():
    """A blend is polymer_blend_melt's question, and answering it here with a
    solution law would be the silent substitution this repository forbids."""
    blend = _candidate(
        _polymer_component(PS, 200.0, 0.5), _polymer_component(PS, 2.0, 0.5)
    )
    prediction = _predict(blend)
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "blend" in " ".join(prediction.notes)


@requires_rdkit
def test_an_all_molecular_formulation_is_refused_as_a_solvent_mixture():
    solvents = _candidate(
        _solvent_component(TOLUENE, 0.5), _solvent_component(ACETONE, 0.5)
    )
    prediction = _predict(solvents)
    assert prediction.status is PredictionStatus.UNSUPPORTED


@requires_rdkit
def test_a_dissolved_polymer_without_a_chain_length_is_refused_not_guessed():
    """[eta] goes as the half power of chain length at theta and higher in a
    good solvent, so the same repeat unit spans orders of magnitude."""
    prediction = _predict(_candidate(
        _polymer_component(PS, None, 0.2), _solvent_component(ACETONE, 0.8)
    ))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "molar mass" in " ".join(prediction.notes)


@requires_rdkit
def test_without_a_temperature_it_refuses():
    request = PredictionRequest(
        candidate=_dope(),
        properties=frozenset({"shear_viscosity"}),
        conditions=Conditions(),
    )
    prediction = PolymerSolutionExpert().predict(request)[0]
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "temperature" in " ".join(prediction.notes)


@requires_rdkit
def test_every_prediction_says_it_assumed_the_polymer_dissolves():
    """Nothing here checks solubility, and a viscosity for a suspension of
    undissolved polymer is a number about a different material."""
    prediction = _predict(_dope())
    joined = " ".join(prediction.notes)
    assert "ASSUMES THE POLYMER DISSOLVES" in joined
    assert "polymer_dissolution" in joined


@requires_rdkit
def test_the_theta_assumption_is_declared_rather_than_buried():
    prediction = _predict(_dope())
    assert "THETA value" in " ".join(prediction.notes)


@requires_rdkit
def test_the_entangled_branch_is_doubted_more_than_the_dilute_one():
    """Above overlap the intrinsic viscosity is raised to a power, so its
    error is amplified and the bar has to say so."""
    dilute = _predict(_dope(fraction=0.005, solvent=TOLUENE))
    entangled = _predict(_dope(fraction=0.25))
    assert dilute.quantity is not None and entangled.quantity is not None
    relative = lambda p: p.uncertainty.std / p.quantity.to("Pa*s").value
    assert relative(entangled) > relative(dilute)


@requires_rdkit
def test_a_copolymer_is_refused_rather_than_averaged():
    copolymer = MixtureComponent(
        role=ComponentRole.SOLUTE,
        fraction=0.2,
        polymer=PolymerSpec(
            monomers=(
                MonomerUnit(smiles=PS, mole_fraction=0.5),
                MonomerUnit(smiles=PE, mole_fraction=0.5),
            ),
            number_average_molar_mass=Quantity(value=200.0, unit="kg/mol"),
        ),
    )
    prediction = _predict(_candidate(copolymer, _solvent_component(ACETONE, 0.8)))
    assert prediction.status is not PredictionStatus.OK


# -- proposing a dope, which nothing could do ------------------------------


def _mixture_spec():
    from formulate.targets import TargetSpec

    return TargetSpec.model_validate(
        {
            "name": "dope",
            "material_classes": ["mixture"],
            "requirements": [
                {
                    "property": "shear_viscosity",
                    "direction": "minimize",
                    "lower": {"value": 1.0, "unit": "Pa*s"},
                    "upper": {"value": 1.0e6, "unit": "Pa*s"},
                }
            ],
        }
    )


@requires_rdkit
def test_a_dope_can_be_proposed_at_all():
    """PolymerSolutionExpert could score one and nothing could propose one,
    so a specification asking for a dope drew an empty pool and reported
    '0 of 0 satisfy every hard constraint' - a search that never ran, reading
    as a search that found nothing."""
    from formulate.exploration.solutions import PolymerSolutionExplorer

    proposed = PolymerSolutionExplorer().propose(_mixture_spec(), 40)
    assert len(proposed) == 40
    for candidate in proposed:
        components = candidate.mixture.components
        assert sum(1 for c in components if c.polymer is not None) == 1
        assert sum(1 for c in components if c.molecule is not None) == 1


@requires_rdkit
def test_a_truncated_draw_spans_every_axis():
    """The bug this replaced, and it produced a whole false finding.

    The first version nested the loops - concentration, then solvent, then
    chain length, then chemistry - with fifty solvents in the second position
    and over five hundred repeat units in the fourth. A draw of two and a half
    THOUSAND never reached a second solvent: every candidate was in water,
    which the Orrick-Erbar viscosity correlation refuses outright because it is
    built on a carbon count and water has no carbon. The run came back
    reporting 2500 candidates whose viscosity could not be predicted, which
    reads as a finding about polymers and was a finding about loop order.

    The product is now walked diagonally first, so a truncated draw spans
    solvents, concentrations, chain lengths and chemistries together.
    """
    from formulate.exploration.solutions import PolymerSolutionExplorer

    proposed = PolymerSolutionExplorer().propose(_mixture_spec(), 60)

    solvents, units, masses, fractions = set(), set(), set(), set()
    for candidate in proposed:
        for component in candidate.mixture.components:
            if component.polymer is not None:
                units.add(component.polymer.monomers[0].smiles)
                masses.add(component.polymer.number_average_molar_mass.to("kg/mol").value)
                fractions.add(component.fraction)
            else:
                solvents.add(component.molecule.smiles)

    # Every axis moves inside sixty draws, and the solvent axis - the one that
    # was stuck - moves most, because it is the longest.
    assert len(solvents) >= 20
    assert len(units) >= 10
    assert len(masses) == 4
    assert len(fractions) == 4


@requires_rdkit
def test_the_spread_ordering_drops_nothing():
    """Reordering a product must not shrink it."""
    from itertools import product

    from formulate.exploration.solutions import _spread

    axes = (("a", "b"), (1, 2, 3), ("x", "y", "z", "w"))
    spread = list(_spread(*axes))
    assert len(spread) == len(set(spread)) == 2 * 3 * 4
    assert set(spread) == set(product(*axes))


@requires_rdkit
def test_a_molecule_specification_draws_nothing_from_it():
    from formulate.exploration.solutions import PolymerSolutionExplorer
    from formulate.targets import TargetSpec

    molecules = TargetSpec.model_validate(
        {
            "name": "m",
            "material_classes": ["molecule"],
            "requirements": [
                {
                    "property": "logp",
                    "direction": "minimize",
                    "lower": {"value": -5.0, "unit": ""},
                    "upper": {"value": 5.0, "unit": ""},
                }
            ],
        }
    )
    assert PolymerSolutionExplorer().propose(molecules, 10) == []


@requires_rdkit
def test_chain_lengths_reach_beyond_what_a_melt_could_push():
    """A solution decouples chain length from processing viscosity, which is
    the whole reason gel spinning exists."""
    from formulate.exploration.polymers import MOLAR_MASSES as MELT_MASSES
    from formulate.exploration.solutions import MOLAR_MASSES as DOPE_MASSES

    assert max(DOPE_MASSES) > max(MELT_MASSES)


@requires_rdkit
def test_nothing_is_proposed_twice():
    from formulate.exploration.solutions import PolymerSolutionExplorer

    proposed = PolymerSolutionExplorer().propose(_mixture_spec(), 300)
    ids = [c.structure_id for c in proposed]
    assert len(ids) == len(set(ids))
