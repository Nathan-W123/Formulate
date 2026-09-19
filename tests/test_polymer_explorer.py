"""Single polymers have to be proposable.

Before this explorer existed a specification saying ``material_classes:
[polymer]`` drew nothing at all - not "nothing feasible", which is an answer,
but an empty pool, which looks like one. Every polymer expert in the
repository was unreachable except through a blend.
"""

from __future__ import annotations

from formulate.core.candidate import MaterialClass, Tacticity
from formulate.exploration.polymers import MOLAR_MASSES, PolymerLibraryExplorer
from formulate.targets.spec import TargetSpec


def _spec(*classes: str) -> TargetSpec:
    return TargetSpec.model_validate(
        {
            "name": "test",
            "material_classes": list(classes),
            "requirements": [
                {
                    "property": "amorphous_density",
                    "direction": "minimize",
                    "lower": {"value": 900.0, "unit": "kg/m^3"},
                    "upper": {"value": 1500.0, "unit": "kg/m^3"},
                }
            ],
        }
    )


def test_a_polymer_specification_draws_polymers():
    proposed = PolymerLibraryExplorer().propose(_spec("polymer"), 50)
    assert len(proposed) == 50
    assert all(c.material_class is MaterialClass.POLYMER for c in proposed)
    assert all(c.polymer is not None for c in proposed)


def test_a_molecule_specification_draws_nothing_from_it():
    """Explorers stay in their class rather than filling a pool with the
    wrong kind of candidate."""
    assert PolymerLibraryExplorer().propose(_spec("molecule"), 50) == []


def test_every_proposal_carries_a_chain_length():
    """A repeat unit without one has no melt viscosity and no modulus; the
    library stores repeat units, so the explorer is what supplies the rest."""
    for candidate in PolymerLibraryExplorer().propose(_spec("polymer"), 30):
        mass = candidate.polymer.number_average_molar_mass
        assert mass is not None
        assert mass.to("kg/mol").value in MOLAR_MASSES


def test_a_truncated_draw_still_spans_the_library():
    """Chain length is the outer loop, so asking for twenty does not return
    twenty grades of the first polymer."""
    proposed = PolymerLibraryExplorer().propose(_spec("polymer"), 20)
    units = {c.polymer.monomers[0].smiles for c in proposed}
    masses = {c.polymer.number_average_molar_mass.to("kg/mol").value for c in proposed}
    assert masses == {MOLAR_MASSES[0]}
    # Each repeat unit contributes one proposal, or two where tacticity has to
    # be swept, so twenty draws reach at least ten of the fifty-seven.
    assert len(units) >= 10


def test_both_tacticities_are_proposed_where_tacticity_decides_crystallinity():
    """Atactic polypropylene does not melt and isotactic does. Proposing only
    the unspecified form leaves the melt expert with nothing it can answer."""
    repeat = "[*]CC(C)[*]"  # polypropylene: isotactic melts, atactic does not
    proposed = PolymerLibraryExplorer(molar_masses=(50.0,)).propose(_spec("polymer"), 400)
    tacticities = {
        c.polymer.tacticity
        for c in proposed
        if c.polymer.monomers[0].smiles == repeat
    }
    assert tacticities == {Tacticity.ISOTACTIC, Tacticity.ATACTIC}


def test_nothing_is_proposed_twice():
    proposed = PolymerLibraryExplorer().propose(_spec("polymer"), 400)
    ids = [c.structure_id for c in proposed]
    assert len(ids) == len(set(ids))


def test_what_is_already_scored_is_not_proposed_again():
    explorer = PolymerLibraryExplorer()
    first = explorer.propose(_spec("polymer"), 20)
    second = explorer.propose(_spec("polymer"), 20, scored=first)
    assert not {c.structure_id for c in first} & {c.structure_id for c in second}
