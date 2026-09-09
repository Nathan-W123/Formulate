"""The GHS lookup, and the three ways it is allowed to say nothing.

The uncertainty on a classification is zero because the lookup is exact, which
is a weaker claim than it looks and would be a dangerous one if it were not
paired with a refusal.  So most of what is asserted here is coverage and
refusal: that every structure the search can propose has a row, that a
structure without one is declined by name rather than guessed at, and that the
three failure modes of the acrylate recipe are visible as numbers.
"""

from __future__ import annotations


from conftest import requires_rdkit

from formulate.core.candidate import (
    Candidate,
    ComponentRole,
    FractionBasis,
    MaterialClass,
    MixtureComponent,
    MixtureSpec,
    MoleculeSpec,
    molecule_candidate,
    polymer_candidate,
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import PredictionStatus
from formulate.core.properties import get_property
from formulate.core.quantity import Quantity
from formulate.experts.base import PredictionRequest
from formulate.experts.hazard import (
    HAZARDS,
    GHSHazardExpert,
    hazard_row,
)
from formulate.exploration.database import catalogue, load_polymers

pytestmark = requires_rdkit

ROOM = Conditions(temperature=Quantity(value=298.15, unit="K"))

HDDA = "C=CC(=O)OCCCCCCOC(=O)C=C"
BPO = "O=C(OOC(=O)c1ccccc1)c1ccccc1"
DMPT = "CN(C)c1ccc(C)cc1"
BENZENE = "c1ccccc1"


def _predict(candidate: Candidate, prop: str):
    expert = GHSHazardExpert()
    request = PredictionRequest(
        candidate=candidate, properties=frozenset({prop}), conditions=ROOM
    )
    return next(p for p in expert.predict(request) if p.property == prop)


def _value(smiles: str, prop: str) -> float:
    prediction = _predict(molecule_candidate(smiles, conditions=ROOM), prop)
    assert prediction.is_usable, prediction.notes
    return prediction.quantity.value


# ---------------------------------------------------------------------------
# Coverage: the table has to cover everything the search can reach
# ---------------------------------------------------------------------------


def test_every_structure_the_search_can_propose_has_a_row():
    """A refusal is the right answer for a structure nobody classified, and a
    wrong answer for one the bundled explorer proposes every run."""
    missing = [r["name"] for r in catalogue() if hazard_row(r["smiles"]) is None]
    missing += [
        f"{p['name']} / {u['smiles']}"
        for p in load_polymers()
        for u in p["repeat_units"]
        if hazard_row(u["smiles"]) is None
    ]
    assert missing == []


def test_every_row_states_a_source():
    assert all(row.source for row in HAZARDS.values())


def test_a_repeat_unit_written_with_attachment_points_is_still_found():
    """RDKit canonicalises ``[*]`` to ``*``, so a lookup against the table as
    written misses every polymer in it."""
    assert hazard_row("[*]CC([*])c1ccccc1") is not None
    assert hazard_row("*CC(*)c1ccccc1") is not None


# ---------------------------------------------------------------------------
# The three components of the recipe the engine had chosen
# ---------------------------------------------------------------------------


def test_the_diacrylate_base_is_a_sensitiser():
    """The classification that matters most for a two-part system mixed at a
    nozzle: the person holding it meets the uncured monomer."""
    assert _value(HDDA, "skin_sensitiser") == 1.0
    prediction = _predict(molecule_candidate(HDDA, conditions=ROOM), "skin_sensitiser")
    assert "H317" in " ".join(prediction.notes)


def test_the_peroxide_initiator_is_a_sensitiser_and_an_organic_peroxide():
    assert _value(BPO, "skin_sensitiser") == 1.0
    prediction = _predict(molecule_candidate(BPO, conditions=ROOM), "skin_sensitiser")
    codes = " ".join(prediction.notes)
    assert "H242" in codes
    # H242 is not turned into a property, and the note has to say so rather than
    # leaving a hazard visible in the codes and invisible in the numbers.
    assert "organic peroxide" in codes


def test_the_amine_accelerator_is_acutely_toxic_by_all_three_routes():
    assert _value(DMPT, "acute_toxicity_category") == 3.0
    prediction = _predict(
        molecule_candidate(DMPT, conditions=ROOM), "acute_toxicity_category"
    )
    codes = " ".join(prediction.notes)
    assert "H301" in codes and "H311" in codes and "H331" in codes


def test_every_acrylate_and_methacrylate_in_the_catalogue_is_a_sensitiser():
    """Not a structural inference drawn here - it is what each one's own
    classification says, and it is the standard occupational injury of acrylate
    chemistry."""
    acrylates = [
        "C=CC(=O)OC",
        "C=CC(=O)OCC",
        "C=CC(=O)OCCCC",
        "C=C(C)C(=O)OC",
        "C=C(C)C(=O)OCCCC",
        HDDA,
        "C=C(C)C(=O)OCCOC(=O)C(=C)C",
        "C=CC(=O)OCC(CC)(COC(=O)C=C)COC(=O)C=C",
    ]
    for smiles in acrylates:
        assert _value(smiles, "skin_sensitiser") == 1.0, smiles


# ---------------------------------------------------------------------------
# What a screen carrying only two endpoints would have said
# ---------------------------------------------------------------------------


def test_benzene_passes_both_endpoints_a_two_property_screen_would_have_carried():
    """The reason the third endpoint exists.

    Benzene is not a sensitiser and is not acutely toxic at any GHS category.
    A hazard expert returning two clean numbers for a category 1A carcinogen
    would be worse than no hazard expert at all, because a clean bill is acted
    on and a missing one is not.
    """
    assert _value(BENZENE, "skin_sensitiser") == 0.0
    assert _value(BENZENE, "acute_toxicity_category") == 5.0
    assert _value(BENZENE, "carcinogen_category") == 1.0


def test_every_prediction_says_which_endpoints_the_screen_does_not_carry():
    for prop in sorted(GHSHazardExpert.supported_properties):
        prediction = _predict(molecule_candidate("CCO", conditions=ROOM), prop)
        notes = " ".join(prediction.notes).lower()
        assert "reproductive toxicity" in notes
        assert "not a clean bill" in notes


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_structure_outside_the_table_is_refused_by_name():
    """Caffeine is not in the table and there is no route to a classification
    from a structure that is worth putting a person's skin behind."""
    caffeine = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
    candidate = molecule_candidate(caffeine, conditions=ROOM)
    expert = GHSHazardExpert()
    assert not expert.assess_domain(candidate).in_domain
    for prop in sorted(expert.supported_properties):
        prediction = _predict(candidate, prop)
        assert prediction.status is PredictionStatus.UNSUPPORTED
        assert "no recorded GHS classification" in " ".join(prediction.notes)


def test_one_unknown_component_refuses_the_whole_formulation():
    """A mixture whose worst component was not looked at has not been
    screened."""
    mixture = Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=(
                MixtureComponent(
                    role=ComponentRole.SOLVENT,
                    fraction=0.9,
                    molecule=MoleculeSpec(smiles="CCO"),
                ),
                MixtureComponent(
                    role=ComponentRole.ADDITIVE,
                    fraction=0.1,
                    molecule=MoleculeSpec(smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
                ),
            ),
            basis=FractionBasis.MASS,
        ),
        conditions=ROOM,
    )
    prediction = _predict(mixture, "skin_sensitiser")
    assert prediction.status is PredictionStatus.UNSUPPORTED


def test_an_unrecorded_carcinogenicity_is_a_refusal_and_not_a_pass():
    """The one endpoint a row is allowed to leave blank, and blank means
    declined rather than clean."""
    blank = [name for name, row in HAZARDS.items() if row.carcinogen_category is None]
    assert blank, "no row abstains, so the refusal path is unexercised"
    # The accelerator is one of them: several aromatic amines are carcinogens,
    # and recording a "not classified" for this one would assert a screening
    # nobody did.
    assert hazard_row(DMPT).carcinogen_category is None
    prediction = _predict(molecule_candidate(DMPT, conditions=ROOM), "carcinogen_category")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "not a pass" in " ".join(prediction.notes)
    # And the other two endpoints still answer: an abstention on one endpoint
    # is not an abstention on the substance.
    assert _value(DMPT, "acute_toxicity_category") == 3.0


# ---------------------------------------------------------------------------
# The worst component, not the average one
# ---------------------------------------------------------------------------


def test_a_formulation_takes_the_worst_component_rather_than_an_average():
    """One per cent of a sensitiser sensitises, and a composition-weighted mean
    would let ninety-nine per cent of an inert carrier buy it a pass."""
    mixture = Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=(
                MixtureComponent(
                    role=ComponentRole.SOLVENT,
                    fraction=0.99,
                    molecule=MoleculeSpec(smiles="CCO"),
                ),
                MixtureComponent(
                    role=ComponentRole.CROSSLINKER,
                    fraction=0.01,
                    molecule=MoleculeSpec(smiles=HDDA),
                ),
            ),
            basis=FractionBasis.MASS,
        ),
        conditions=ROOM,
    )
    assert _predict(mixture, "skin_sensitiser").quantity.value == 1.0
    assert _predict(mixture, "acute_toxicity_category").quantity.value == 5.0


def test_a_polymer_is_screened_on_its_repeat_units():
    """A high polymer is not classified as such - it is not absorbed - and the
    row says so with a source behind it rather than by the table finding
    nothing."""
    prediction = _predict(polymer_candidate("[*]NCCCCCC(=O)[*]", conditions=ROOM), "skin_sensitiser")
    assert prediction.quantity.value == 0.0
    assert prediction.applicability.in_domain


def test_the_cured_acrylate_network_inherits_the_monomer_classification():
    """Recorded as a decision in the row rather than derived from the network's
    structure, and the row states both reasons for it."""
    network = polymer_candidate(
        "[*]CC([*])C(=O)OCCCCCCOC(=O)C([*])C[*]", conditions=ROOM
    )
    prediction = _predict(network, "skin_sensitiser")
    assert prediction.quantity.value == 1.0
    notes = " ".join(prediction.notes)
    assert "residual monomer" in notes
    assert "mixed at the point of use" in notes


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


def test_the_endpoints_are_separate_properties_rather_than_one_score():
    """Averaging would let a strong pass on one endpoint pay for a failure on
    another, and hazards do not trade off that way."""
    for name in ("skin_sensitiser", "acute_toxicity_category", "carcinogen_category"):
        definition = get_property(name)
        assert definition.canonical_unit == "dimensionless"
        assert definition.multiplicative_error is False
    assert get_property("skin_sensitiser").bounds == (0.0, 1.0)
    assert get_property("acute_toxicity_category").bounds == (1.0, 5.0)
    assert get_property("carcinogen_category").bounds == (1.0, 4.0)


def test_the_uncertainty_is_zero_and_says_what_the_zero_means():
    prediction = _predict(molecule_candidate(HDDA, conditions=ROOM), "skin_sensitiser")
    assert prediction.uncertainty.std == 0.0
    assert "the lookup being exact" in prediction.uncertainty.basis
