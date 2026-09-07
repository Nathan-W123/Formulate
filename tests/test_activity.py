"""Mixture phase behaviour from UNIFAC activity coefficients.

The assertions that matter here are the refusals. A group-contribution activity
model is only as good as its interaction table, and the underlying
implementation reads an absent parameter as zero - the value for two groups
that mix perfectly. So the failure mode is not an error, it is a confident
prediction of miscibility for compounds that do not mix, and that is precisely
what specification section 13 exists to catch.
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
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import PredictionStatus
from formulate.core.quantity import Quantity
from formulate.experts.activity import (
    UNIFACActivityExpert,
    evaluate,
    group_name,
    missing_interactions,
    unifac_available,
    unifac_groups,
)
from formulate.experts.base import PredictionRequest

pytestmark = [
    requires_rdkit,
    pytest.mark.skipif(not unifac_available(), reason="modified UNIFAC is unavailable"),
]

ROOM = Conditions(
    temperature=Quantity(value=298.15, unit="K"), pressure=Quantity(value=1.0, unit="atm")
)

WATER = "O"
ETHANOL = "CCO"
TOLUENE = "Cc1ccccc1"
HEXANE = "CCCCCC"
BUTANOL = "CCCCO"
TRICHLOROETHYLENE = "ClC=C(Cl)Cl"


def _blend(a, b, fraction=0.5, basis=FractionBasis.MOLE):
    return Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=(
                MixtureComponent(
                    role=ComponentRole.SOLVENT, fraction=fraction, molecule=MoleculeSpec(smiles=a)
                ),
                MixtureComponent(
                    role=ComponentRole.CO_SOLVENT,
                    fraction=1.0 - fraction,
                    molecule=MoleculeSpec(smiles=b),
                ),
            ),
            basis=basis,
        ),
        conditions=ROOM,
    )


def _predict(candidate, prop="mixing_stability"):
    expert = UNIFACActivityExpert()
    request = PredictionRequest(
        candidate=candidate, properties=frozenset({prop}), conditions=ROOM
    )
    return expert.predict(request)[0]


# -- the model gets the chemistry right ------------------------------------


@pytest.mark.parametrize(
    "a,b,separates",
    [
        (WATER, ETHANOL, False),   # fully miscible
        (WATER, BUTANOL, True),    # about 8 per cent by mass, then two layers
        (WATER, TOLUENE, True),    # immiscible
        (HEXANE, "CCCCCCC", False),  # essentially ideal
        (TOLUENE, HEXANE, False),  # miscible in all proportions
    ],
)
def test_phase_separation_is_predicted_from_the_spinodal_condition(a, b, separates):
    prediction = _predict(_blend(a, b))
    assert prediction.quantity is not None, prediction.notes
    assert (prediction.quantity.value < 0) is separates


def test_an_ideal_mixture_has_almost_no_excess_energy():
    prediction = _predict(_blend(HEXANE, "CCCCCCC"), "excess_gibbs_energy")
    assert abs(prediction.quantity.to("J/mol").value) < 50.0


def test_water_and_toluene_are_far_less_ideal_than_water_and_ethanol():
    hard = _predict(_blend(WATER, TOLUENE), "excess_gibbs_energy").quantity.to("J/mol").value
    easy = _predict(_blend(WATER, ETHANOL), "excess_gibbs_energy").quantity.to("J/mol").value
    assert hard > 3 * easy > 0


# -- the refusals ----------------------------------------------------------


def test_a_missing_interaction_parameter_is_refused_not_read_as_zero():
    """The whole reason this module checks the table before it computes.

    Trichloroethylene and water have no tabulated parameter. Asking anyway
    returns an infinite-dilution activity coefficient near 3, for a pair that
    is in truth about four orders of magnitude worse - a confident prediction
    of miscibility for two liquids that separate on sight.
    """
    water = unifac_groups(WATER)
    tce = unifac_groups(TRICHLOROETHYLENE)
    assert water and tce
    missing = missing_interactions([water, tce])
    assert missing
    assert "H2O" in " ".join(group_name(a) + group_name(b) for a, b in missing)

    prediction = _predict(_blend(WATER, TRICHLOROETHYLENE))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    assert "read as zero" in " ".join(prediction.notes) or "read as zero" in str(prediction)


def test_evaluate_returns_nothing_rather_than_a_number_when_a_pair_is_missing():
    water = unifac_groups(WATER)
    tce = unifac_groups(TRICHLOROETHYLENE)
    assert evaluate([water, tce], [0.5, 0.5], 298.15) is None


def test_a_covered_pair_reports_no_missing_interactions():
    assert missing_interactions([unifac_groups(WATER), unifac_groups(ETHANOL)]) == ()


def test_a_polymer_component_is_out_of_domain():
    from formulate.core.candidate import MonomerUnit, PolymerSpec

    candidate = Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=(
                MixtureComponent(
                    role=ComponentRole.SOLVENT, fraction=0.5, molecule=MoleculeSpec(smiles=TOLUENE)
                ),
                MixtureComponent(
                    role=ComponentRole.BINDER,
                    fraction=0.5,
                    polymer=PolymerSpec(monomers=(MonomerUnit(smiles="[*]CC(c1ccccc1)[*]"),)),
                ),
            ),
            basis=FractionBasis.MOLE,
        ),
        conditions=ROOM,
    )
    prediction = _predict(candidate)
    assert prediction.status in (PredictionStatus.OUT_OF_DOMAIN, PredictionStatus.UNSUPPORTED)


def test_no_cloud_point_is_derived():
    """UNIFAC's temperature dependence cannot support a demixing temperature."""
    from formulate.core.properties import PROPERTY_REGISTRY

    assert "cloud_point" not in PROPERTY_REGISTRY
    prediction = _predict(_blend(WATER, BUTANOL))
    assert any("cloud point" in note for note in prediction.notes)


def test_temperature_is_required():
    prediction = UNIFACActivityExpert().predict(
        PredictionRequest(
            candidate=_blend(WATER, ETHANOL),
            properties=frozenset({"mixing_stability"}),
            conditions=Conditions(),
        )
    )[0]
    assert prediction.status is PredictionStatus.UNSUPPORTED


# -- composition bases -----------------------------------------------------


@pytest.mark.parametrize(
    "basis", [FractionBasis.MOLE, FractionBasis.MASS, FractionBasis.VOLUME]
)
def test_every_composition_basis_is_converted_rather_than_assumed(basis):
    """A solvent blend is usually written by volume, which needs densities."""
    prediction = _predict(_blend(TOLUENE, HEXANE, basis=basis), "excess_gibbs_energy")
    assert prediction.quantity is not None, prediction.notes


def test_the_registry_covers_the_new_mixture_properties(registry):
    coverage = registry.coverage(
        ["excess_gibbs_energy", "mixing_stability"], MaterialClass.MIXTURE
    )
    assert coverage["excess_gibbs_energy"] == ["unifac"]
    assert coverage["mixing_stability"] == ["unifac"]
    # And not offered for a single molecule, where they are meaningless.
    assert registry.uncovered(["mixing_stability"], MaterialClass.MOLECULE) == [
        "mixing_stability"
    ]
