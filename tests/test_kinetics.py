"""Reaction kinetics: the first rate in the property registry.

The arithmetic is three lines and the tests that matter are not about it.
They are about the two refusals, because those are what make a rate usable in
a design run: a cure time quoted without an initiation regime is meaningless,
and a cure time for a monomer whose polymer is a rubber at the cure
temperature is a time at which a liquid becomes a gum.
"""

from __future__ import annotations

import pytest

from formulate.core.candidate import molecule_candidate
from formulate.core.conditions import Conditions
from formulate.core.quantity import Quantity
from formulate.experts.base import PredictionRequest
from formulate.experts.kinetics import (
    INITIATION,
    MONOMERS,
    TERMINATION,
    FreeRadicalCureExpert,
    PropagationExpert,
    cure_time,
    propagation_constant,
)

REDOX = ("redox-peroxide-amine",)


def _request(smiles: str, prop: str, *, processing=REDOX, temperature=298.15):
    return PredictionRequest(
        candidate=molecule_candidate(smiles),
        properties=frozenset({prop}),
        conditions=Conditions(
            temperature=Quantity(value=temperature, unit="K"), processing=processing
        ),
    )


def test_every_monomer_has_kinetics_and_a_structure():
    from formulate.experts.kinetics import _SMILES

    assert set(MONOMERS) == set(TERMINATION) == set(_SMILES)


def test_propagation_is_faster_for_acrylates_than_methacrylates():
    """The best-known qualitative fact about these coefficients.

    An acrylate radical is secondary and much more reactive than the tertiary
    radical a methacrylate leaves, so k_p is one to two orders larger. If the
    Arrhenius pairs were ever transcribed wrongly this is what would break.
    """
    assert propagation_constant("methyl acrylate", 298.15) > 10.0 * propagation_constant(
        "methyl methacrylate", 298.15
    )
    assert propagation_constant("butyl acrylate", 298.15) > 10.0 * propagation_constant(
        "butyl methacrylate", 298.15
    )


def test_cure_time_scales_as_the_inverse_square_root_of_the_initiation_rate():
    slow = cure_time("methyl methacrylate", 298.15, 1.0e-6)
    fast = cure_time("methyl methacrylate", 298.15, 1.0e-4)
    assert slow / fast == pytest.approx(10.0, rel=1e-9)


def test_a_cure_time_without_an_initiation_regime_is_refused():
    """It is a property of the recipe, and defaulting would hide the choice."""
    prediction = FreeRadicalCureExpert().predict(
        _request("C=C(C)C(=O)OC", "cure_time", processing=())
    )[0]
    assert prediction.quantity is None
    assert "initiation regime" in prediction.notes[0]
    assert "redox-peroxide-amine" in prediction.notes[0]


def test_two_initiation_regimes_at_once_are_refused_rather_than_summed():
    prediction = FreeRadicalCureExpert().predict(
        _request("C=C(C)C(=O)OC", "cure_time", processing=("thermal-aibn", "photo-uv"))
    )[0]
    assert prediction.quantity is None
    assert "one at a time" in prediction.notes[0]


def test_a_monomer_whose_polymer_is_a_rubber_at_the_cure_temperature_is_refused():
    """Butyl acrylate cures fastest of all seven and is useless for a solid."""
    prediction = FreeRadicalCureExpert().predict(_request("C=CC(=O)OCCCC", "cure_time"))[0]
    assert prediction.quantity is None
    assert "219 K" in prediction.notes[0]
    # And it is refused despite being the fastest: the point is that speed
    # alone does not make a cure.
    assert cure_time("butyl acrylate", 298.15, INITIATION["redox-peroxide-amine"][0]) < cure_time(
        "methyl methacrylate", 298.15, INITIATION["redox-peroxide-amine"][0]
    )


def test_the_same_monomer_is_accepted_where_its_polymer_is_glassy():
    """Cool the cure below poly(butyl methacrylate)'s 293 K and it is fine."""
    expert = FreeRadicalCureExpert()
    warm = expert.predict(_request("C=C(C)C(=O)OCCCC", "cure_time", temperature=298.15))[0]
    cold = expert.predict(_request("C=C(C)C(=O)OCCCC", "cure_time", temperature=273.15))[0]
    assert warm.quantity is None
    assert cold.quantity is not None


def test_a_structure_outside_the_benchmark_set_is_refused_not_extrapolated():
    for expert in (PropagationExpert(), FreeRadicalCureExpert()):
        prop = next(iter(expert.supported_properties))
        prediction = expert.predict(_request("CCO", prop))[0]
        assert prediction.quantity is None
        assert "benchmark set" in prediction.notes[0]


def test_no_free_radical_monomer_cures_in_a_second_at_ambient():
    """The finding the expert was written to test, kept as a regression.

    Every regime, every monomer that produces a solid: the fastest is vinyl
    acetate under a redox pair at about eighty seconds, which is two orders
    away from a jet that has to set in flight. If a data change ever makes
    this pass, the data change is what needs checking.
    """
    solid = [m for m, row in MONOMERS.items() if row[3] > 298.15]
    assert solid  # the test would be vacuous otherwise
    best = min(
        cure_time(m, 298.15, rate)
        for m in solid
        for rate, _ in INITIATION.values()
    )
    assert best > 10.0
