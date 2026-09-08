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
    FreeRadicalCureExpert,
    PropagationExpert,
    cure_time,
    functionality,
    gel_conversion,
    kinetic_chain_length,
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

    assert set(MONOMERS) == set(_SMILES)


def test_the_tabulated_functionality_agrees_with_the_structure():
    """The table records it; the expert reads it off the molecule.

    Two sources for the same fact is a defect waiting to happen, so the one
    that is derived is the one used and this checks they never diverge.
    """
    from formulate.experts.kinetics import _SMILES

    for name, row in MONOMERS.items():
        assert functionality(_SMILES[name]) == row.functionality, name


def test_a_monofunctional_monomer_has_no_gel_point():
    """It makes linear chains. There is nothing for a network to form from."""
    assert gel_conversion("methyl methacrylate", 298.15, 1.0e-4) is None
    assert gel_conversion("vinyl acetate", 298.15, 1.0e-4) is None


def test_a_crosslinker_gels_far_below_the_conversion_a_linear_polymer_vitrifies_at():
    """The correction that moved the answer by two orders of magnitude.

    Vitrification needs half the monomer converted; gelation needs one
    crosslink per primary chain, and the chains are hundreds of units long.
    """
    gel = gel_conversion("1,6-hexanediol diacrylate", 298.15, 1.0e-4)
    assert gel is not None
    assert gel < 0.01
    fast = cure_time("1,6-hexanediol diacrylate", 298.15, 1.0e-4)
    slow = cure_time("1,6-hexanediol diacrylate", 298.15, 1.0e-4, conversion=0.5)
    assert slow / fast > 100.0


def test_a_higher_functionality_gels_sooner():
    rate = INITIATION["redox-peroxide-amine"][0]
    assert cure_time("trimethylolpropane triacrylate", 298.15, rate) < cure_time(
        "1,6-hexanediol diacrylate", 298.15, rate
    )


def test_a_faster_initiator_shortens_the_chains_and_so_delays_the_gel_point():
    """The trade-off that stops a stronger initiator being the answer.

    Raising the radical flux speeds propagation as its square root and cuts
    the kinetic chain length in proportion, and short chains need more
    conversion before one crosslink per chain exists.
    """
    slow = gel_conversion("1,6-hexanediol diacrylate", 298.15, 1.0e-6)
    fast = gel_conversion("1,6-hexanediol diacrylate", 298.15, 1.0e-2)
    assert kinetic_chain_length("1,6-hexanediol diacrylate", 298.15, 1.0e-2) < (
        kinetic_chain_length("1,6-hexanediol diacrylate", 298.15, 1.0e-6)
    )
    assert fast > slow


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


def test_a_crosslinker_is_not_judged_on_the_glass_transition_of_a_linear_polymer():
    """The refusal that was turning away the only chemistry fast enough.

    A crosslinked network is rigid because it is one molecule, not because
    its chains are stiff, so the linear analogue's glass transition says
    nothing about it. Hexanediol diacrylate is an acrylate, and every linear
    acrylate here is a rubber at ambient temperature.
    """
    prediction = FreeRadicalCureExpert().predict(
        _request("C=CC(=O)OCCCCCCOC(=O)C=C", "cure_time")
    )[0]
    assert prediction.quantity is not None
    assert prediction.quantity.value < 2.0
    assert MONOMERS["1,6-hexanediol diacrylate"].linear_tg_k is None


def test_a_monomer_whose_polymer_is_a_rubber_at_the_cure_temperature_is_refused():
    """Butyl acrylate is among the fastest and is useless for a solid."""
    prediction = FreeRadicalCureExpert().predict(_request("C=CC(=O)OCCCC", "cure_time"))[0]
    assert prediction.quantity is None
    assert "219 K" in prediction.notes[0]
    assert "monofunctional" in prediction.notes[0]
    # And it is refused despite being fast: speed alone does not make a cure.
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


def test_no_linear_monomer_cures_in_a_second_at_ambient():
    """Why a crosslinker was necessary and a faster initiator was not enough.

    Over every regime and every monofunctional monomer that produces a solid,
    the fastest is vinyl acetate under a redox pair at about eighty seconds.
    Nothing that vitrifies to solidify gets near a jet that has to set in
    flight; only gelation does.
    """
    solid = [
        m
        for m, row in MONOMERS.items()
        if row.functionality == 1 and row.linear_tg_k and row.linear_tg_k > 298.15
    ]
    assert solid  # the test would be vacuous otherwise
    best = min(
        cure_time(m, 298.15, rate) for m in solid for rate, _ in INITIATION.values()
    )
    assert best > 10.0
