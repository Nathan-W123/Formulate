"""Naming a solute, and the phase a target asks for.

Two things this file guards. That the engine can be handed a behaviour -
"dissolve polystyrene" - rather than the answer to it, which before meant
writing polystyrene's Hansen window into the target by hand. And that a target
asking for a liquid is not offered a solid, which is how a solvent search came
to return naphthalene.
"""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import Candidate, MaterialClass, MoleculeSpec, molecule_candidate
from formulate.core.conditions import Conditions, Phase
from formulate.experts.base import PredictionRequest
from formulate.experts.dissolution import (
    SOLUBILITY_SPHERES,
    DissolutionExpert,
    relative_energy_difference,
    resolve_solute,
)
from formulate.experts.hansen import hansen_triple

pytestmark = requires_rdkit


def _conditions(**kwargs):
    return Conditions.standard().model_copy(update=kwargs)


def _predict(smiles, conditions):
    expert = DissolutionExpert()
    request = PredictionRequest(
        candidate=molecule_candidate(smiles),
        properties=frozenset({"solubility_red"}),
        conditions=conditions,
    )
    return next(iter(expert.predict(request)))


# -- the physics -----------------------------------------------------------


#: Textbook behaviour towards polystyrene. The solvents are the ones a
#: polymer chemist would name; the non-solvents are the ones a naive polarity
#: heuristic gets wrong, which is why they are here.
_POLYSTYRENE = [
    ("Cc1ccccc1", True),        # toluene
    ("C1CCOC1", True),          # tetrahydrofuran
    ("ClC(Cl)Cl", True),        # chloroform
    ("CCOC(C)=O", True),        # ethyl acetate
    ("c1ccccc1", True),         # benzene
    ("O=C1CCCCC1", True),       # cyclohexanone
    ("Cc1ccc(C)cc1", True),     # p-xylene
    ("Clc1ccccc1", True),       # chlorobenzene
    ("CCCCCC", False),          # hexane
    ("CO", False),              # methanol
    ("CCO", False),             # ethanol
    ("O", False),               # water
    ("OCC(O)CO", False),        # glycerol
    ("CCCCO", False),           # 1-butanol
    ("CC#N", False),            # acetonitrile
]


@pytest.mark.parametrize(("smiles", "dissolves"), _POLYSTYRENE)
def test_the_polystyrene_sphere_agrees_with_known_chemistry(smiles, dissolves):
    triple = hansen_triple(smiles)
    if triple is None:
        pytest.skip("no compiled Hansen parameters for this structure")
    red = relative_energy_difference(triple, SOLUBILITY_SPHERES["polystyrene"])
    assert (red < 1.0) is dissolves, f"RED {red:.2f}"


def test_acetone_lands_on_the_boundary_where_it_belongs():
    """It swells polystyrene and does not dissolve it, and RED says so."""
    triple = hansen_triple("CC(C)=O")
    red = relative_energy_difference(triple, SOLUBILITY_SPHERES["polystyrene"])
    assert 0.95 < red < 1.10


def test_the_solvent_parameters_are_converted_out_of_pascals():
    """The panel carries Hansen parameters in Pa^0.5, the spheres in MPa^0.5.

    These are square roots of a pressure, so the factor is a thousand rather
    than a million - the sort of conversion that is wrong by three orders of
    magnitude while still looking like a solubility parameter. Getting it
    wrong scored every real solvent for polystyrene as a non-solvent.
    """
    red = relative_energy_difference(
        hansen_triple("Cc1ccccc1"), SOLUBILITY_SPHERES["polystyrene"]
    )
    assert red == pytest.approx(0.65, abs=0.02)


def test_aliases_resolve_and_unknown_polymers_do_not():
    assert resolve_solute("PS") == "polystyrene"
    assert resolve_solute("PMMA") == "poly(methyl methacrylate)"
    assert resolve_solute("Polystyrene") == "polystyrene"
    assert resolve_solute("polyethylene") is None


# -- what the expert refuses ----------------------------------------------


def test_no_solute_named_means_no_answer():
    """Whether something dissolves is a question about a pair."""
    prediction = _predict("Cc1ccccc1", _conditions())
    assert prediction.quantity is None
    assert "no solute was named" in " ".join(prediction.notes)


def test_a_polymer_with_no_published_sphere_is_refused_by_name():
    prediction = _predict("Cc1ccccc1", _conditions(solutes=("polyethylene",)))
    assert prediction.quantity is None
    joined = " ".join(prediction.notes)
    assert "polyethylene" in joined
    assert "polystyrene" in joined  # it says what it does have


def test_two_solutes_are_refused_rather_than_averaged():
    """A solvent good for one polymer is routinely a non-solvent for another."""
    prediction = _predict(
        "Cc1ccccc1", _conditions(solutes=("polystyrene", "polyamide 66"))
    )
    assert prediction.quantity is None
    assert "averaging them would hide" in " ".join(prediction.notes)


def test_a_named_solute_produces_a_number_with_the_sphere_in_its_notes():
    prediction = _predict("Cc1ccccc1", _conditions(solutes=("polystyrene",)))
    assert prediction.quantity is not None
    assert prediction.quantity.value == pytest.approx(0.65, abs=0.02)
    joined = " ".join(prediction.notes)
    assert "radius 12.7" in joined
    assert "so a solvent" in joined


def test_the_solute_reaches_the_content_address():
    """Two runs differing only in the solute are different questions."""
    a = _conditions(solutes=("polystyrene",)).identity_payload()
    b = _conditions(solutes=("polycarbonate",)).identity_payload()
    assert a != b
    assert "solutes" in a


# -- the phase a target asks for -------------------------------------------


def _liquid_target():
    from formulate.targets.spec import StructuralConstraints

    return StructuralConstraints()


def test_a_solid_is_filtered_out_when_the_target_asks_for_a_liquid():
    from formulate.exploration.filters import CandidateFilter

    conditions = _conditions(phase=Phase.LIQUID)
    naphthalene = CandidateFilter(_liquid_target(), conditions).check(
        molecule_candidate("c1ccc2ccccc2c1")
    )
    assert not naphthalene.passed
    assert "asks for a liquid" in " ".join(naphthalene.reasons)
    assert "melting at 80" in " ".join(naphthalene.reasons)

    toluene = CandidateFilter(_liquid_target(), conditions).check(
        molecule_candidate("Cc1ccccc1")
    )
    assert toluene.passed


def test_the_phase_filter_is_silent_when_nothing_asked_for_one():
    from formulate.exploration.filters import CandidateFilter

    for conditions in (None, _conditions(), _conditions(phase=Phase.SOLID)):
        result = CandidateFilter(_liquid_target(), conditions).check(
            molecule_candidate("c1ccc2ccccc2c1")
        )
        assert result.passed


def test_the_phase_filter_needs_a_temperature_to_judge_by():
    from formulate.exploration.filters import CandidateFilter

    no_temperature = Conditions(phase=Phase.LIQUID)
    result = CandidateFilter(_liquid_target(), no_temperature).check(
        molecule_candidate("c1ccc2ccccc2c1")
    )
    assert result.passed


def test_every_component_of_a_blend_has_to_be_liquid():
    from formulate.core.candidate import (
        ComponentRole,
        FractionBasis,
        MixtureComponent,
        MixtureSpec,
    )
    from formulate.exploration.filters import CandidateFilter

    blend = Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=(
                MixtureComponent(role=ComponentRole.SOLVENT, fraction=0.5,
                                 molecule=MoleculeSpec(smiles="Cc1ccccc1")),
                MixtureComponent(role=ComponentRole.SOLVENT, fraction=0.5,
                                 molecule=MoleculeSpec(smiles="c1ccc2ccccc2c1")),
            ),
            basis=FractionBasis.VOLUME,
        ),
    )
    result = CandidateFilter(_liquid_target(), _conditions(phase=Phase.LIQUID)).check(blend)
    assert not result.passed
    assert "melting at 80" in " ".join(result.reasons)


# -- the whole thing, from a behaviour -------------------------------------


def test_a_target_that_names_a_polymer_finds_its_solvents():
    """The demonstration: no Hansen number appears in this target.

    It names the polymer and asks for something that boils off. Everything
    else - the sphere, its radius, which solvents fall inside it - the engine
    supplies. Before this, the same search meant writing polystyrene's Hansen
    window into the requirements by hand, which is stating the answer.
    """
    from formulate.coordination.coordinator import DeterministicCoordinator, RunConfig
    from formulate.exploration.database import ReferenceDatabaseExplorer
    from formulate.targets.spec import TargetSpec

    spec = TargetSpec.from_dict(
        {
            "name": "dissolves polystyrene and flashes off",
            "material_classes": ["molecule"],
            "conditions": {
                "temperature": "25 degC",
                "pressure": "1 atm",
                "phase": "liquid",
                "solutes": ["polystyrene"],
            },
            "requirements": [
                {"property": "solubility_red", "direction": "minimize", "weight": 3.0},
                {"property": "normal_boiling_point", "direction": "in_range",
                 "lower": "50 degC", "upper": "130 degC", "weight": 1.0, "hard": True},
            ],
        }
    )
    run = DeterministicCoordinator(
        explorers=[ReferenceDatabaseExplorer()], config=RunConfig(pool_size=50, seed=0)
    ).run(spec)

    top = [entry.candidate.label for entry in run.ranking.ranked[:6]]
    known_solvents = {
        "toluene", "tetrahydrofuran", "chloroform", "benzene", "chlorobenzene",
        "1,2-dichloroethane", "cyclohexanone", "pyridine", "styrene", "p-xylene",
        "1,4-dioxane", "carbon tetrachloride", "ethyl acetate", "anisole",
        "nitrobenzene", "aniline", "mesitylene", "dichloromethane", "limonene",
    }
    assert set(top) <= known_solvents, f"a non-solvent reached the top six: {top}"

    # And nothing that is not a liquid at 25 C survived the filter.
    assert "naphthalene" not in top
    assert "phenol" not in top
