"""The heat of fusion of a polymer crystal, and the crystal that is not there.

Half of this file is refusals, on purpose. An amorphous polymer has no
enthalpy of fusion and a polymer of unstated tacticity has not been asked a
well-posed question, and those are two different answers - so they are tested
as hard as the numbers, and the test that they read *differently* is one of
them.

The other half recomputes every fitted statistic from the shipped table, so a
constant cannot quietly drift away from the data it claims to come from.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MonomerRole,
    MonomerUnit,
    PolymerSpec,
    PolymerTopology,
    Tacticity,
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import PredictionStatus
from formulate.core.properties import get_property
from formulate.core.quantity import Quantity
from formulate.experts.base import PredictionRequest
from formulate.experts.melt import MELTING_POINTS
from formulate.experts.polymer_thermal import (
    BACKBONE_COUNT_OFFSET,
    BACKBONE_UNIT_ENTHALPY,
    DISPUTED_RTOL,
    FUSION_ENTHALPIES,
    MODEL_RTOL,
    OUT_OF_DOMAIN_RTOL,
    TABULATED_RTOL,
    PolymerThermalExpert,
    model_enthalpy_fusion,
    repeat_unit_shape,
    tabulated_enthalpy,
)

PROP = "enthalpy_fusion"

PE = "[*]CC[*]"
PP = "[*]CC(C)[*]"
PS = "[*]CC(c1ccccc1)[*]"
PEO = "[*]CCO[*]"
PET = "[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]"
NYLON6 = "[*]NCCCCCC(=O)[*]"
PDMS = "[*][Si](C)(C)O[*]"
#: Not in the table: a siloxane the PDMS measurement does not license.
METHYLPHENYLSILOXANE = "[*][Si](C)(c1ccccc1)O[*]"
#: Not in the table: an unsaturated backbone the polybutadiene/polyisoprene
#: pair does not license either, because they disagree with each other.
POLYPENTENAMER = "[*]CC=CCC[*]"
PVC = "[*]CC(Cl)[*]"
#: Crystallises only under strain: no crystal at rest.
POLYISOBUTYLENE = "[*]CC(C)(C)[*]"
#: Has a crystal and cyclises below the temperature that would melt it.
PAN = "[*]CC(C#N)[*]"
#: A 2,3-linked naphthalene backbone: two atoms on the shortest path between
#: the attachment points, ten in the rigid unit they are supposed to count.
NAPHTHALENE_2_3_DIYL = "[*]c1cc2ccccc2cc1[*]"
#: The same trap one ring further out, and the one that used to be read as a
#: polymer with an eight-atom pendant hanging off two backbone atoms.
ANTHRACENE_2_3_DIYL = "c1ccc2cc3cc([*])c([*])cc3cc2c1"
#: Pendant of nine heavy atoms.
HEXYL_METHACRYLATE = "[*]CC(C)(C(=O)OCCCCCC)[*]"
#: Pendant of four heavy atoms: answered, and marked out of domain.
PMP = "[*]CC(CC(C)C)[*]"


def _polymer(*repeat_units: str, tacticity=Tacticity.UNSPECIFIED, **kwargs) -> Candidate:
    monomers = tuple(
        MonomerUnit(smiles=s, mole_fraction=1.0 / len(repeat_units)) for s in repeat_units
    )
    return Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(monomers=monomers, tacticity=tacticity, **kwargs),
        conditions=Conditions.standard(),
    )


def _predict(candidate: Candidate):
    request = PredictionRequest(
        candidate=candidate,
        properties=frozenset({PROP}),
        conditions=Conditions.standard(),
    )
    return PolymerThermalExpert().predict(request)[0]


def _mass(repeat_unit: str) -> float:
    from formulate.experts.polymer import repeat_unit_mass

    return repeat_unit_mass(repeat_unit)


# --------------------------------------------------------------------------
# A. The measured values, and the per-gram / per-mole trap
# --------------------------------------------------------------------------


@requires_rdkit
@pytest.mark.parametrize(
    "repeat_unit,tacticity,j_per_gram",
    [
        (PE, Tacticity.UNSPECIFIED, 293.0),
        (NYLON6, Tacticity.UNSPECIFIED, 230.0),
        (PET, Tacticity.UNSPECIFIED, 140.0),
        (PP, Tacticity.ISOTACTIC, 207.0),
        (PEO, Tacticity.UNSPECIFIED, 197.0),
    ],
)
def test_reference_values_convert_to_the_quoted_per_gram_figures(
    repeat_unit, tacticity, j_per_gram
):
    """The five values the brief names, converted rather than asserted twice."""
    prediction = _predict(_polymer(repeat_unit, tacticity=tacticity))
    assert prediction.status is PredictionStatus.OK
    j_per_mole = prediction.quantity.to("J/mol").value
    assert j_per_mole / _mass(repeat_unit) == pytest.approx(j_per_gram, rel=0.02)


@requires_rdkit
def test_polyethylene_is_per_repeat_unit_not_per_methylene():
    """4.11 kJ/mol per CH2 is the textbook number; this repeat unit is C2H4.

    Reporting 4.11 would be right per CH2 and wrong per repeat unit, and the
    repeat unit is what the property is defined per.
    """
    prediction = _predict(_polymer(PE))
    per_repeat = prediction.quantity.to("kJ/mol").value
    assert per_repeat == pytest.approx(8.22, rel=0.02)
    assert per_repeat / 2.0 == pytest.approx(4.11, rel=0.02)
    assert _mass(PE) == pytest.approx(28.05, rel=0.01)


@requires_rdkit
def test_the_unit_is_joules_per_mole_and_converts():
    prediction = _predict(_polymer(PET))
    from formulate.core.units import are_compatible, parse_unit

    canonical = get_property(PROP).canonical_unit
    assert are_compatible(canonical, "J/mol")
    assert parse_unit(canonical) == parse_unit("J/mol")
    assert parse_unit(prediction.quantity.to_canonical().unit) == parse_unit("J/mol")
    assert prediction.quantity.to("kJ/mol").value == pytest.approx(26.9, rel=0.01)
    assert prediction.quantity.to("J/mol").value == pytest.approx(26900.0, rel=0.01)


@requires_rdkit
def test_a_disputed_measurement_carries_the_wider_bar():
    """Polyoxymethylene is quoted at 250 and at 326 J/g; 10% would be a lie."""
    prediction = _predict(_polymer("[*]CO[*]"))
    value = prediction.quantity.to("J/mol").value
    assert prediction.uncertainty.std / value == pytest.approx(DISPUTED_RTOL, rel=1e-6)
    polyethylene = _predict(_polymer(PE))
    assert polyethylene.uncertainty.std / polyethylene.quantity.to(
        "J/mol"
    ).value == pytest.approx(TABULATED_RTOL, rel=1e-6)


# --------------------------------------------------------------------------
# B. Every fitted number, recomputed from the shipped table
# --------------------------------------------------------------------------


def _entries(role: str) -> list[tuple[str, float]]:
    out = []
    for repeat_unit, entries in FUSION_ENTHALPIES.items():
        for entry in entries:
            if entry.role == role:
                out.append((repeat_unit, entry.value))
    return sorted(out)


def _relative_least_squares(x: list[float], y: list[float]) -> float:
    """a minimising sum of squared fractional residuals of ``a x`` against ``y``."""
    return sum(xi / yi for xi, yi in zip(x, y)) / sum((xi / yi) ** 2 for xi, yi in zip(x, y))


def _rms_relative(predicted: list[float], observed: list[float]) -> float:
    return math.sqrt(
        sum(((p - o) / o) ** 2 for p, o in zip(predicted, observed)) / len(observed)
    )


def _counts(entries: list[tuple[str, float]]) -> list[float]:
    """What the model multiplies its coefficient by: backbone atoms + offset."""
    return [repeat_unit_shape(r).backbone_atoms + BACKBONE_COUNT_OFFSET for r, _ in entries]


@requires_rdkit
def test_the_shipped_coefficient_is_the_one_the_fit_split_implies():
    fit = _entries("fit")
    assert len(fit) == 13
    y = [v for _, v in fit]
    assert _relative_least_squares(_counts(fit), y) == pytest.approx(
        BACKBONE_UNIT_ENTHALPY, abs=1.0
    )


def _leave_one_out(entries: list[tuple[str, float]], offset: int) -> list[float]:
    """Fractional residual of each fit entry against a coefficient it did not inform."""
    x = [repeat_unit_shape(r).backbone_atoms + offset for r, _ in entries]
    y = [v for _, v in entries]
    residuals = []
    for i in range(len(entries)):
        others = [j for j in range(len(entries)) if j != i]
        coefficient = _relative_least_squares([x[j] for j in others], [y[j] for j in others])
        residuals.append((coefficient * x[i] - y[i]) / y[i])
    return residuals


@requires_rdkit
def test_the_stated_uncertainty_is_the_pessimistic_held_out_one():
    fit = _entries("fit")
    y = [v for _, v in fit]
    loo = math.sqrt(
        sum(r**2 for r in _leave_one_out(fit, BACKBONE_COUNT_OFFSET)) / len(fit)
    )

    validation = _entries("validation")
    assert len(validation) == 7
    xv = [repeat_unit_shape(r).backbone_atoms for r, _ in validation]
    yv = [v for _, v in validation]
    held_out = _rms_relative([model_enthalpy_fusion(n) for n in xv], yv)

    in_sample = _rms_relative(
        [model_enthalpy_fusion(repeat_unit_shape(r).backbone_atoms) for r, _ in fit], y
    )

    # The measured figures the module docstring quotes.
    assert in_sample == pytest.approx(0.213, abs=0.005)
    assert loo == pytest.approx(0.234, abs=0.005)
    assert held_out == pytest.approx(0.100, abs=0.005)
    # In sample flatters, which is exactly why it is not what is quoted.
    assert in_sample < loo
    assert MODEL_RTOL >= max(loo, held_out)
    assert MODEL_RTOL < max(loo, held_out) + 0.02  # pessimistic, not padded


@requires_rdkit
def test_the_stated_uncertainty_covers_what_a_one_sigma_bar_should():
    """A bar that 13 of 20 held-out polymers fall outside would be a lie.

    68 per cent of twenty is fourteen. Sixteen is wide by two, which is the
    direction this repository wants to err in; fewer than fourteen would mean
    the ranker is being handed a confidence the model has not earned.
    """
    fit, validation = _entries("fit"), _entries("validation")
    residuals = _leave_one_out(fit, BACKBONE_COUNT_OFFSET) + [
        (model_enthalpy_fusion(repeat_unit_shape(r).backbone_atoms) - v) / v
        for r, v in validation
    ]
    assert len(residuals) == 20
    assert sum(1 for r in residuals if abs(r) <= MODEL_RTOL) == 16


@requires_rdkit
def test_the_counting_offset_earns_itself_and_is_not_a_free_parameter():
    """Through the origin the residual is ordered in chain length, not scattered.

    This is the defect the offset exists to fix, so it is pinned from the
    shipped table rather than asserted in prose: the strict proportionality is
    worse left out, worse on the validation seven, and - the part an error bar
    cannot describe - biased low on exactly the two-backbone-atom repeat units
    that are almost everything the model route is ever asked about.
    """
    fit, validation = _entries("fit"), _entries("validation")
    y = [v for _, v in fit]

    def scores(offset: int) -> tuple[float, float, float]:
        loo = _leave_one_out(fit, offset)
        coefficient = _relative_least_squares(
            [repeat_unit_shape(r).backbone_atoms + offset for r, _ in fit], y
        )
        held = [
            (coefficient * (repeat_unit_shape(r).backbone_atoms + offset) - v) / v
            for r, v in validation
        ]
        small = [
            residual
            for residual, (r, _) in zip(loo, fit)
            if repeat_unit_shape(r).backbone_atoms <= 3
        ] + [
            residual
            for residual, (r, _) in zip(held, validation)
            if repeat_unit_shape(r).backbone_atoms <= 3
        ]
        return (
            math.sqrt(sum(r**2 for r in loo) / len(loo)),
            math.sqrt(sum(r**2 for r in held) / len(held)),
            sum(small) / len(small),
        )

    through_origin = scores(0)
    shipped = scores(BACKBONE_COUNT_OFFSET)

    assert through_origin[0] == pytest.approx(0.265, abs=0.005)
    assert through_origin[1] == pytest.approx(0.147, abs=0.005)
    # Better held out both ways, not one traded against the other.
    assert shipped[0] < through_origin[0]
    assert shipped[1] < through_origin[1]
    # And the bias it was introduced to remove is removed.
    assert through_origin[2] < -0.10
    assert abs(shipped[2]) < 0.05


@requires_rdkit
def test_a_free_intercept_lands_on_the_offset_rather_than_beating_it():
    """The offset is one backbone atom because a two-parameter fit says 0.994.

    If that ever stops being true the offset is a fitted parameter in disguise
    and has to be quoted as one, so it is checked rather than remembered.
    """
    numpy = pytest.importorskip("numpy")
    fit = _entries("fit")
    design = numpy.array(
        [[1.0, repeat_unit_shape(r).backbone_atoms] for r, _ in fit], float
    )
    target = numpy.array([v for _, v in fit], float)
    weighted = design / target[:, None]
    intercept, slope = numpy.linalg.lstsq(
        weighted, numpy.ones(len(fit)), rcond=None
    )[0]
    assert intercept / slope == pytest.approx(BACKBONE_COUNT_OFFSET, abs=0.05)


@requires_rdkit
def test_the_joback_group_scheme_stays_rejected_on_the_number_that_rejected_it():
    """Eleven groups over thirteen points transfer worse, not better.

    Leave-one-out flatters a group scheme, so the test is the seven polymers
    no coefficient saw. If the table ever changes, this is what has to stay
    true for the docstring's rejection to stay honest.
    """
    numpy = pytest.importorskip("numpy")
    from formulate.experts.polymer import repeat_unit_groups

    fit = _entries("fit")
    decompositions = [repeat_unit_groups(r) for r, _ in fit]
    if any(d is None for d in decompositions):
        pytest.skip("the Joback fragmenter could not decompose the fit split")
    keys = sorted({k for d in decompositions for k in d})
    target = numpy.array([v for _, v in fit], float)
    design = numpy.array([[d.get(k, 0) for k in keys] for d in decompositions], float)

    weighted = design / target[:, None]
    scale = (weighted**2).sum() / weighted.shape[0]
    coefficients = numpy.linalg.solve(
        weighted.T @ weighted + 1e-4 * scale * numpy.eye(len(keys)),
        weighted.T @ numpy.ones(len(fit)),
    )

    validation = _entries("validation")
    group_predictions, backbone_predictions, observed = [], [], []
    for repeat_unit, value in validation:
        decomposition = repeat_unit_groups(repeat_unit)
        if decomposition is None:
            continue
        group_predictions.append(
            float(numpy.array([decomposition.get(k, 0) for k in keys]) @ coefficients)
        )
        backbone_predictions.append(
            model_enthalpy_fusion(repeat_unit_shape(repeat_unit).backbone_atoms)
        )
        observed.append(value)

    group_error = _rms_relative(group_predictions, observed)
    backbone_error = _rms_relative(backbone_predictions, observed)
    assert group_error >= 0.28
    assert group_error > 4.0 * backbone_error


# --------------------------------------------------------------------------
# C. A cross-check against a table this module does not own
# --------------------------------------------------------------------------


@requires_rdkit
def test_implied_entropy_of_fusion_sits_in_wunderlichs_bead_band():
    """dHm / (Tm x N) should land near 7-12 J/(K mol) per mobile backbone unit.

    The melting points come from ``melt.py``, which knows nothing about this
    module, so a wrong table entry here shows up as an entropy of fusion no
    flexible macromolecule has.
    """
    from formulate import chem

    checked = 0
    for repeat_unit, entries in FUSION_ENTHALPIES.items():
        canonical = chem.canonical_smiles(repeat_unit)
        for key, melting in MELTING_POINTS.items():
            if chem.canonical_smiles(key) != canonical:
                continue
            backbone = repeat_unit_shape(repeat_unit).backbone_atoms
            for entry in entries:
                entropy = entry.value / (melting.temperature * backbone)
                assert 4.0 <= entropy <= 14.0, f"{repeat_unit}: {entropy:.2f} J/(K mol)"
                checked += 1
    assert checked >= 13


# --------------------------------------------------------------------------
# D. The refusals
# --------------------------------------------------------------------------


@requires_rdkit
def test_an_amorphous_polymer_is_refused_rather_than_given_zero():
    prediction = _predict(_polymer(PP, tacticity=Tacticity.ATACTIC))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    reason = " ".join(prediction.notes)
    assert "no crystal to melt" in reason
    assert "zero" in reason.lower()


@requires_rdkit
def test_an_unstated_tacticity_is_refused_differently():
    prediction = _predict(_polymer(PP, tacticity=Tacticity.UNSPECIFIED))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    reason = " ".join(prediction.notes)
    assert "undecided" in reason
    assert "tacticity" in reason


@requires_rdkit
def test_a_polymer_that_decomposes_first_is_not_told_it_has_no_crystal():
    """Polyacrylonitrile has a crystal. It never gets to melt it.

    ``melt.crystallinity`` returns the same ``False`` for a chain that cannot
    pack and for one that packs and then cyclises below its melting point, and
    ``melt.py``'s own comment on that table says a stereoregular sample of the
    second kind *does* have a crystalline phase. A refusal that tells a
    chemist otherwise is a refusal they would hand straight back.
    """
    prediction = _predict(_polymer(PAN, tacticity=Tacticity.ISOTACTIC))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    reason = " ".join(prediction.notes)
    assert "does have a crystalline phase" in reason
    assert "no crystal to melt" not in reason
    assert "decomposes" in reason
    # And it is a different answer from the one an amorphous polymer gets.
    amorphous = " ".join(_predict(_polymer(PP, tacticity=Tacticity.ATACTIC)).notes)
    assert reason != amorphous


@requires_rdkit
def test_a_polymer_that_cannot_pack_at_all_still_says_so():
    """The other ``False``: polyisobutylene orders only under strain."""
    reason = " ".join(_predict(_polymer(POLYISOBUTYLENE)).notes)
    assert "no crystal to melt" in reason
    assert "under strain" in reason


@requires_rdkit
def test_the_two_refusals_are_not_the_same_answer_dressed_up():
    """Undecided is not the same as decided against, and must not read as it."""
    amorphous = " ".join(_predict(_polymer(PP, tacticity=Tacticity.ATACTIC)).notes)
    open_question = " ".join(_predict(_polymer(PP, tacticity=Tacticity.UNSPECIFIED)).notes)
    assert amorphous != open_question
    assert amorphous not in open_question
    assert open_question not in amorphous


@requires_rdkit
def test_a_copolymer_is_refused():
    prediction = _predict(_polymer(PE, PEO))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "copolymer" in " ".join(prediction.notes)


@requires_rdkit
def test_a_candidate_with_no_polymer_is_refused_rather_than_crashing():
    bare = Candidate.model_construct(
        material_class=MaterialClass.POLYMER,
        molecule=None,
        polymer=None,
        mixture=None,
        conditions=Conditions.standard(),
    )
    request = PredictionRequest(
        candidate=bare, properties=frozenset({PROP}), conditions=Conditions.standard()
    )
    prediction = PolymerThermalExpert().predict(request)[0]
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    assert "no polymer" in " ".join(prediction.notes)


@requires_rdkit
@pytest.mark.parametrize(
    "repeat_unit,tacticity,fragment",
    [
        (METHYLPHENYLSILOXANE, Tacticity.ISOTACTIC, "Si"),
        (POLYPENTENAMER, Tacticity.UNSPECIFIED, "double or triple bond"),
        (PVC, Tacticity.ISOTACTIC, "Cl"),
        (HEXYL_METHACRYLATE, Tacticity.ISOTACTIC, "heavy atoms"),
    ],
)
def test_the_model_refuses_outside_its_domain_with_the_measurement_behind_it(
    repeat_unit, tacticity, fragment
):
    prediction = _predict(_polymer(repeat_unit, tacticity=tacticity))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    reason = " ".join(prediction.notes)
    assert fragment in reason
    # Every refusal names a number - a measured miss or a checked bound -
    # rather than a preference.
    assert any(character.isdigit() for character in reason)


@requires_rdkit
def test_a_measurement_answers_where_the_model_would_be_refused():
    """PDMS is the evidence for the siloxane refusal and still answers for itself.

    Refusing a polymer whose heat of fusion has been measured, on the grounds
    that a model fitted to other polymers would be wrong about it, would be
    refusing to read the answer off the page.
    """
    measured = _predict(_polymer(PDMS))
    assert measured.status is PredictionStatus.OK
    assert measured.quantity.to("J/mol").value == pytest.approx(2750.0)
    assert "outside" in " ".join(measured.notes)

    unlicensed = _predict(_polymer(METHYLPHENYLSILOXANE))
    assert unlicensed.status is PredictionStatus.UNSUPPORTED


@requires_rdkit
def test_the_reference_entries_never_touched_the_coefficient():
    """A reference measurement documents the boundary; it must not move the fit."""
    assert {r for r, _ in _entries("reference")} == {
        PDMS,
        "[*]CC=CC[*]",
        "[*]CC(C)=CC[*]",
        PS,
    }
    fit = _entries("fit")
    assert not ({r for r, _ in fit} & {r for r, _ in _entries("reference")})
    assert not ({r for r, _ in fit} & {r for r, _ in _entries("validation")})


# --------------------------------------------------------------------------
# E. Never zero, never silent
# --------------------------------------------------------------------------


@requires_rdkit
def test_no_polymer_in_the_reference_library_is_ever_answered_with_zero():
    """The whole point of the gate: absence is a refusal, not a zero."""
    payload = json.loads(
        (
            Path(__file__).resolve().parent.parent
            / "src"
            / "formulate"
            / "data"
            / "reference_polymers.json"
        ).read_text()
    )
    answered = refused = 0
    for record in payload["polymers"]:
        for tacticity in Tacticity:
            prediction = _predict(
                _polymer(record["repeat_unit"], tacticity=tacticity)
            )
            if prediction.status.has_value:
                assert prediction.quantity is not None
                assert prediction.quantity.to("J/mol").value > 0.0
                assert prediction.uncertainty.std is not None
                answered += 1
            else:
                assert prediction.quantity is None
                assert any(note.strip() for note in prediction.notes)
                refused += 1
    assert answered > 0 and refused > 0


@requires_rdkit
def test_every_tabulated_value_is_positive_and_labelled():
    for repeat_unit, entries in FUSION_ENTHALPIES.items():
        assert repeat_unit_shape(repeat_unit) is not None, repeat_unit
        for entry in entries:
            assert entry.value > 0.0
            assert entry.role in {"fit", "validation", "reference"}
            assert entry.source
            assert entry.rtol in {TABULATED_RTOL, DISPUTED_RTOL}


# --------------------------------------------------------------------------
# F. Out-of-domain routing
# --------------------------------------------------------------------------


@requires_rdkit
def test_a_bulky_pendant_is_answered_out_of_domain_with_the_wider_bar():
    prediction = _predict(_polymer(PMP, tacticity=Tacticity.ISOTACTIC))
    assert prediction.status is PredictionStatus.OUT_OF_DOMAIN
    value = prediction.quantity.to("J/mol").value
    ratio = prediction.uncertainty.std / value
    assert ratio == pytest.approx(OUT_OF_DOMAIN_RTOL, rel=1e-6)
    assert ratio == pytest.approx(0.34, abs=0.005)
    assert ratio > MODEL_RTOL
    assert "1.62" in prediction.uncertainty.basis
    assert "polystyrene" in prediction.uncertainty.basis


@requires_rdkit
def test_a_single_atom_pendant_stays_in_domain_with_the_narrow_bar():
    """A repeat unit the model was built for gets the model's own error, not more."""
    prediction = _predict(_polymer("[*]OCCCCCCOC(=O)[*]"))
    assert prediction.status is PredictionStatus.OK
    value = prediction.quantity.to("J/mol").value
    assert prediction.uncertainty.std / value == pytest.approx(MODEL_RTOL, rel=1e-6)
    assert value == pytest.approx(model_enthalpy_fusion(9), rel=1e-9)


@requires_rdkit
def test_an_unchecked_repeat_unit_length_is_marked_rather_than_refused():
    long_polyester = "[*]OCCCCCCCCCCCCCCCCCCOC(=O)[*]"
    prediction = _predict(_polymer(long_polyester))
    assert prediction.status is PredictionStatus.OUT_OF_DOMAIN
    assert any("backbone atoms is beyond" in note for note in prediction.notes)


# --------------------------------------------------------------------------
# G. The contract
# --------------------------------------------------------------------------


def test_the_expert_declares_itself_honestly():
    expert = PolymerThermalExpert()
    assert expert.id == "polymer_thermal"
    assert expert.supported_properties == frozenset({PROP})
    assert expert.supported_classes == frozenset({MaterialClass.POLYMER})
    assert expert.dependencies == frozenset()
    assert expert.family.value == "thermal"
    assert not expert.covers(PROP, MaterialClass.MOLECULE)
    assert not expert.covers(PROP, MaterialClass.MIXTURE)
    assert expert.covers(PROP, MaterialClass.POLYMER)
    assert "100% crystalline" in expert.method
    assert "repeat unit" in expert.method


@requires_rdkit
@pytest.mark.parametrize("repeat_unit", [PE, "[*]OCCCCCCOC(=O)[*]"])
def test_every_answer_says_which_quantity_it_is(repeat_unit):
    """Per gram and per mole of chain are both plausible readings of this name."""
    prediction = _predict(_polymer(repeat_unit))
    notes = " ".join(prediction.notes)
    assert "per mole of REPEAT UNIT" in notes
    assert "100% CRYSTALLINE" in notes
    assert "degree of crystallinity" in notes
    assert "J/g of repeat unit" in notes
    assert "Joback" in notes or "measured" in notes


@requires_rdkit
def test_a_short_chain_is_marked_and_widened_not_just_mentioned():
    """A note the ranker cannot read is not a caveat, it is a comment.

    The value is the infinite-chain limit whatever the candidate says, so a
    900 g/mol chain gets the tabulated number - and it has to get it out of
    domain and with a bar that admits the chain-end deficit, because the
    ranker scores one sigma into the unfavourable direction and would
    otherwise be told this oligomer is as certain as high polyethylene.
    """
    long_chain = _predict(_polymer(PE))
    short = _predict(
        _polymer(PE, number_average_molar_mass=Quantity(value=900.0, unit="g/mol"))
    )
    assert short.status is PredictionStatus.OUT_OF_DOMAIN
    assert not short.applicability.in_domain
    assert any("infinite-chain limit" in note for note in short.notes)
    # 900 g/mol of a 28 g/mol repeat unit is 32 units; one lost at each end is
    # 6%, added in quadrature to the 10% on the measurement.
    value = short.quantity.to("J/mol").value
    assert short.quantity.to("J/mol").value == pytest.approx(
        long_chain.quantity.to("J/mol").value
    )
    assert short.uncertainty.std / value == pytest.approx(
        math.hypot(TABULATED_RTOL, 2.0 / (900.0 / _mass(PE))), rel=1e-6
    )
    assert short.uncertainty.std > long_chain.uncertainty.std


@requires_rdkit
def test_an_oligomer_carries_a_far_wider_bar_than_a_polymer():
    """At ten repeat units the chain ends are a fifth of the material."""
    oligomer = _predict(
        _polymer(PE, number_average_molar_mass=Quantity(value=300.0, unit="g/mol"))
    )
    ratio = oligomer.uncertainty.std / oligomer.quantity.to("J/mol").value
    assert ratio > 0.19
    assert oligomer.status is PredictionStatus.OUT_OF_DOMAIN


@requires_rdkit
@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        (
            {
                "topology": PolymerTopology.NETWORK,
                "crosslink_density": Quantity(value=1000.0, unit="mol/m^3"),
            },
            "network polymer",
        ),
        (
            {"crosslink_density": Quantity(value=1000.0, unit="mol/m^3")},
            "crosslink density is stated",
        ),
    ],
)
def test_an_architecture_the_table_does_not_describe_is_marked_out_of_domain(
    kwargs, fragment
):
    """A thermoset is not the linear chain whose crystal was measured.

    ``polymer.py`` puts a network and a stated crosslink density outside the
    domain of its repeat-unit correlations; a measured perfect-crystal
    enthalpy has no more authority over a crosslinked candidate than a fitted
    one does, so the measurement route must not escape the same judgement.
    """
    prediction = _predict(_polymer(PE, **kwargs))
    assert prediction.status is PredictionStatus.OUT_OF_DOMAIN
    assert not prediction.applicability.in_domain
    assert prediction.applicability.score < 1.0
    assert any(fragment in warning for warning in prediction.applicability.warnings)
    assert any(fragment in note for note in prediction.notes)


@requires_rdkit
def test_an_end_group_does_not_make_a_homopolymer_a_copolymer():
    candidate = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(
            monomers=(
                MonomerUnit(smiles=PE, mole_fraction=1.0),
                MonomerUnit(smiles="[*]C", mole_fraction=0.0, role=MonomerRole.END_GROUP),
            )
        ),
        conditions=Conditions.standard(),
    )
    prediction = _predict(candidate)
    assert prediction.status is PredictionStatus.OK
    assert prediction.quantity.to("J/mol").value == pytest.approx(8220.0)


@requires_rdkit
def test_tacticity_selects_between_two_crystals_of_one_repeat_unit():
    """Isotactic and syndiotactic polystyrene are 1.62x apart; both are measured."""
    isotactic = tabulated_enthalpy(PS, Tacticity.ISOTACTIC)
    syndiotactic = tabulated_enthalpy(PS, Tacticity.SYNDIOTACTIC)
    assert isotactic.value / syndiotactic.value == pytest.approx(1.62, abs=0.01)
    assert _predict(
        _polymer(PS, tacticity=Tacticity.ISOTACTIC)
    ).quantity.to("J/mol").value == pytest.approx(8960.0)
    assert _predict(
        _polymer(PS, tacticity=Tacticity.SYNDIOTACTIC)
    ).quantity.to("J/mol").value == pytest.approx(5520.0)
    # And neither is reachable without stating which one it is.
    assert (
        _predict(_polymer(PS, tacticity=Tacticity.UNSPECIFIED)).status
        is PredictionStatus.UNSUPPORTED
    )


@requires_rdkit
def test_the_out_of_domain_widening_is_the_polymorph_spread_in_quadrature():
    isotactic = tabulated_enthalpy(PS, Tacticity.ISOTACTIC).value
    syndiotactic = tabulated_enthalpy(PS, Tacticity.SYNDIOTACTIC).value
    half_spread = (isotactic - syndiotactic) / (isotactic + syndiotactic)
    assert OUT_OF_DOMAIN_RTOL == pytest.approx(
        math.hypot(MODEL_RTOL, half_spread), abs=0.005
    )


@requires_rdkit
def test_repeat_unit_shape_reads_a_backbone_ring_as_backbone():
    """A terephthalate ring is part of the chain, not a substituent on it."""
    shape = repeat_unit_shape(PET)
    assert shape.backbone_atoms == 10
    assert shape.pendant_sizes == (1, 1)  # the two carbonyl oxygens
    assert shape.backbone_elements == frozenset({"C", "O"})
    assert not shape.unsaturated_backbone
    # Para-linked: four of the ring's six atoms are on the path, two are not.
    assert (shape.ring_atoms_off_path, shape.backbone_rings) == (2, 1)

    styrene = repeat_unit_shape(PS)
    assert styrene.backbone_atoms == 2
    assert styrene.pendant_sizes == (6,)  # the phenyl, which is not backbone
    assert (styrene.ring_atoms_off_path, styrene.backbone_rings) == (0, 0)


@requires_rdkit
def test_a_ring_fused_to_a_backbone_ring_is_backbone_too():
    """One pass over the ring list absorbs a fused ring only if it is lucky.

    The absorption walks ``AtomRings()`` once, so a ring that meets the core
    only after a later ring has been absorbed is left behind - and then the
    outer two thirds of an anthracene backbone are read as an eight-atom
    pendant hanging off two backbone atoms, which is not what the docstring
    promises and not what the gates are then deciding about.
    """
    shape = repeat_unit_shape(ANTHRACENE_2_3_DIYL)
    assert shape.pendant_sizes == ()
    assert shape.backbone_rings == 3
    assert shape.ring_atoms_off_path == 12
    assert shape.backbone_elements == frozenset({"C"})


@requires_rdkit
@pytest.mark.parametrize("repeat_unit", [NAPHTHALENE_2_3_DIYL, ANTHRACENE_2_3_DIYL])
def test_a_corner_cut_backbone_ring_is_refused_rather_than_undercounted(repeat_unit):
    """Two counted atoms for a ten-atom rigid unit is a different count.

    Every backbone ring in the thirteen is para-linked and hides two of its
    six atoms. A 2,3-linkage hides eight of ten, so the coefficient would be
    multiplied by a number that no longer means what it meant when it was
    fitted - and the answer would be wrong by a factor, not a percentage.
    """
    prediction = _predict(_polymer(repeat_unit))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    reason = " ".join(prediction.notes)
    assert "off that path" in reason
    assert any(character.isdigit() for character in reason)


@requires_rdkit
def test_a_para_linked_ring_backbone_is_still_answered():
    """The gate has to bite on the fused case without taking PET with it."""
    for repeat_unit in (PET, "[*]OCCCCOC(=O)c1ccc(cc1)C(=O)[*]"):
        assert _predict(_polymer(repeat_unit)).status.has_value


@requires_rdkit
def test_a_repeat_unit_mass_that_cannot_be_built_does_not_fail_a_measurement():
    """The per-gram restatement is a note, and a note must not sink a value."""
    from formulate.experts.polymer_thermal import _safe_repeat_unit_mass

    assert _safe_repeat_unit_mass(PE) == pytest.approx(28.05, rel=0.01)
    assert _safe_repeat_unit_mass("[*]CCC") is None  # one attachment point, not two


@requires_rdkit
def test_repeat_unit_shape_refuses_what_is_not_a_chain_segment():
    assert repeat_unit_shape("CCO") is None  # no attachment points
    assert repeat_unit_shape("[*]CCC") is None  # only one
    assert repeat_unit_shape("not a smiles") is None
