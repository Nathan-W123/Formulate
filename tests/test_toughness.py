"""Elongation at break, and the architecture that decides it.

Two of these tests exist to keep a claim honest rather than to check a number.
:func:`test_the_proxy_states_the_hit_rate_it_actually_earns` recomputes the
leave-one-out calibration from the catalogue by an independent route and fails
if the module's docstring has drifted from it, and
:func:`test_a_measured_elongation_outranks_the_proxy_that_was_fitted_to_it`
guards the mistake that was actually made: the class mean over five rubbers is
a tighter number than one handbook range's half-width, so a proxy quoting only
its own fit error outranked every measurement it had been fitted to.
"""

from __future__ import annotations

import math
import statistics

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MonomerRole,
    MonomerUnit,
    PolymerSpec,
    PolymerTopology,
    polymer_candidate,
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import PredictionStatus, best_prediction, prefer
from formulate.core.properties import get_property
from formulate.core.quantity import Quantity
from formulate.experts.base import PredictionRequest
from formulate.experts.toughness import (
    ChainToughnessExpert,
    MeasuredPolymerExpert,
    PolymerArchitectureExpert,
    catalogue_record,
    classify,
    elongation_source_spread,
    polymer_records,
    strength_utilisation,
    toughness_proxy,
)
from formulate.exploration.database import load_polymers, polymer_spec

pytestmark = requires_rdkit

ROOM = Conditions(temperature=Quantity(value=298.15, unit="K"))

PS = "[*]CC([*])c1ccccc1"
PA6 = "[*]NCCCCCC(=O)[*]"
PE = "[*]CC[*]"
HDDA_NETWORK = "[*]CC([*])C(=O)OCCCCCCOC(=O)C([*])C[*]"


def _candidate(name: str) -> Candidate:
    """The catalogue row with this abbreviation, as the explorer would build it."""
    record = next(r for r in load_polymers() if r["abbreviation"] == name)
    return Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=polymer_spec(record),
        conditions=ROOM,
        label=record["name"],
    )


def _predict(expert, candidate, prop, *, context=None, conditions=ROOM):
    request = PredictionRequest(
        candidate=candidate,
        properties=frozenset({prop}),
        conditions=conditions,
        context=context or {},
    )
    return next(p for p in expert.predict(request) if p.property == prop)


def _tg_context(value_k: float, std: float = 7.0):
    """A stand-in upstream transition, so the proxy can be tested on its own."""
    expert = MeasuredPolymerExpert()
    prediction = _predict(expert, _candidate("PA6"), "glass_transition_temperature")
    return {
        "glass_transition_temperature": prediction.model_copy(
            update={"quantity": Quantity(value=value_k, unit="K")}
        )
    }


# ---------------------------------------------------------------------------
# The registry entries
# ---------------------------------------------------------------------------


def test_elongation_is_registered_as_a_multiplicative_property():
    """Two per cent and twelve hundred are both elongations at break.

    Comparing two predictions of that on absolute spread hands the decision to
    whichever expert predicted the brittler material, because a small
    prediction carries a small absolute error however wrong it is.
    """
    assert get_property("elongation_at_break").multiplicative_error is True


def test_crosslink_density_is_not_multiplicative():
    """It is zero for every thermoplastic, and a relative spread on zero is not
    a quantity."""
    assert get_property("crosslink_density").multiplicative_error is False


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------


def test_every_catalogue_row_parses_and_sums_to_one():
    for record in load_polymers():
        spec = polymer_spec(record)
        backbone = [m for m in spec.monomers if m.role is not MonomerRole.END_GROUP]
        assert backbone
        assert math.isclose(sum(m.mole_fraction for m in backbone), 1.0, abs_tol=1e-6)


def test_every_catalogue_row_is_found_by_the_expert_that_reads_it():
    """Matching is on structure and architecture, so a row that cannot be
    matched is a row nothing can ever use."""
    for record in load_polymers():
        assert catalogue_record(polymer_spec(record)) is not None, record["name"]


def test_the_two_polyethylenes_are_told_apart_by_architecture_alone():
    """They share a repeat unit and differ by a factor of two in strength.

    Matching on structure alone would collapse them into one candidate and
    report the wrong one's properties for both.
    """
    hdpe, ldpe = _candidate("HDPE"), _candidate("LDPE")
    assert hdpe.polymer.monomers[0].smiles == ldpe.polymer.monomers[0].smiles
    assert hdpe.structure_id != ldpe.structure_id
    assert catalogue_record(hdpe.polymer).abbreviation == "HDPE"
    assert catalogue_record(ldpe.polymer).abbreviation == "LDPE"


def test_nothing_was_added_to_the_held_out_calibration_set():
    """Every measured uncertainty in this repository that says "over the fifty
    reference compounds" is invalidated by a fifty-first."""
    from formulate.exploration.database import load_reference_compounds

    assert len(load_reference_compounds()) == 50


# ---------------------------------------------------------------------------
# The measured route
# ---------------------------------------------------------------------------


def test_the_measured_spread_is_half_the_quoted_range_width():
    """Recomputed here from the file rather than read from the module."""
    widths = [
        math.log(r["elongation_at_break"][1] / r["elongation_at_break"][0])
        for r in load_polymers()
        if r.get("elongation_at_break")
    ]
    expected = math.exp(statistics.fmean(widths) / 2.0)
    assert elongation_source_spread() == pytest.approx(expected)
    # The claim in the module docstring: a quoted elongation is worth about a
    # factor of two either way.
    assert 1.8 < expected < 2.2


def test_the_measured_route_reports_the_geometric_centre_of_the_range():
    prediction = _predict(MeasuredPolymerExpert(), _candidate("PS"), "elongation_at_break")
    assert prediction.quantity.value == pytest.approx(math.sqrt(0.012 * 0.025))
    assert prediction.applicability.in_domain
    assert any("0.012 to 0.025" in note for note in prediction.notes)


def test_the_measured_route_refuses_where_the_catalogue_is_silent():
    """Poly(vinyl acetate) has a transition and a density and nothing
    mechanical, and the refusal says which of the three is missing."""
    prediction = _predict(MeasuredPolymerExpert(), _candidate("PVAc"), "elongation_at_break")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "no measured elongation" in " ".join(prediction.notes)
    transition = _predict(
        MeasuredPolymerExpert(), _candidate("PVAc"), "glass_transition_temperature"
    )
    assert transition.status is PredictionStatus.OK


def test_the_measured_route_refuses_a_polymer_outside_the_catalogue():
    """Polycarbonate is a real polymer and is not in this file, and the answer
    to that is nothing rather than a nearby row."""
    candidate = polymer_candidate(
        "[*]OC(=O)Oc1ccc(cc1)C(C)(C)c1ccc(cc1)[*]", conditions=ROOM
    )
    prediction = _predict(MeasuredPolymerExpert(), candidate, "elongation_at_break")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "not in the bundled polymer catalogue" in " ".join(prediction.notes)


def test_a_measured_transition_beats_the_group_contribution_estimate():
    """The whole reason the measured route exists.

    A specification asking for a transition inside an eighty-kelvin window is
    asking a question 22 K of held-out error cannot answer and 7 K can.
    """
    from formulate.experts.polymer import PolymerGlassTransitionExpert

    candidate = _candidate("PA6")
    measured = _predict(MeasuredPolymerExpert(), candidate, "glass_transition_temperature")
    estimated = _predict(
        PolymerGlassTransitionExpert(), candidate, "glass_transition_temperature"
    )
    assert measured.uncertainty.std < estimated.uncertainty.std
    assert prefer(measured, estimated)
    assert best_prediction([estimated, measured], "glass_transition_temperature") is measured


# ---------------------------------------------------------------------------
# The proxy, and what it is worth
# ---------------------------------------------------------------------------


def test_the_proxy_states_the_hit_rate_it_actually_earns():
    """Recompute the leave-one-out calibration independently of the module.

    The numbers quoted in the docstring and in every prediction's basis string
    are these, and a change to the catalogue that moves them should fail here
    rather than silently restate an error that is no longer true.
    """
    ambient = 298.15
    members: dict[str, list[float]] = {}
    for record in load_polymers():
        elongation = record.get("elongation_at_break")
        if not elongation:
            continue
        name = classify(
            is_network=record["topology"] in ("network", "dendritic")
            or bool(record.get("crosslink_density_mol_m3")),
            glass_transition_k=record.get("glass_transition_k"),
            temperature_k=ambient,
        )
        if name is None:
            continue
        members.setdefault(name, []).append(
            math.log(math.sqrt(elongation[0] * elongation[1]))
        )

    hits = trials = 0
    for logs in members.values():
        if len(logs) < 2:
            continue
        for index, value in enumerate(logs):
            others = [logs[j] for j in range(len(logs)) if j != index]
            trials += 1
            hits += abs(value - statistics.fmean(others)) <= statistics.stdev(others)

    proxy = toughness_proxy()
    assert (proxy.hits, proxy.trials) == (hits, trials)
    # The claim in the docstring, and it is not a flattering one.
    assert (hits, trials) == (7, 11)
    assert math.exp(proxy.classes["rubbery"][1]) == pytest.approx(1.40, abs=0.01)
    assert math.exp(proxy.classes["glassy"][1]) == pytest.approx(6.64, abs=0.01)


def test_the_glassy_class_is_the_one_that_does_not_work():
    """Polystyrene at two per cent and polyamide 6 at seventy-seven are both
    glasses at ambient, and the ductile one is the closer to its transition.

    Asserted so that a future tightening of the quoted spread has to explain
    this pair, which distance above Tg does not separate in either magnitude or
    direction.
    """
    proxy = toughness_proxy()
    glassy = math.exp(proxy.classes["glassy"][1])
    rubbery = math.exp(proxy.classes["rubbery"][1])
    assert glassy > 4.0 * rubbery

    ps = next(r for r in polymer_records() if r.abbreviation == "PS")
    pa6 = next(r for r in polymer_records() if r.abbreviation == "PA6")
    assert ps.glass_transition_k - 298.15 > pa6.glass_transition_k - 298.15
    assert ps.elongation < pa6.elongation / 20.0


def test_a_measured_elongation_outranks_the_proxy_that_was_fitted_to_it():
    """The mistake this guards was made and shipped in a run.

    The class mean over five rubbers is reproducible to a factor of 1.40 and
    one handbook range's half-width is a factor of 1.97, so a proxy quoting
    only its own fit error is the *tighter* prediction and ``prefer`` chose it
    over every measurement it had been fitted to.  Both routes predict what a
    bar of this polymer will do, so both have to carry the specimen-to-specimen
    spread; the proxy carries its class error on top.
    """
    candidate = _candidate("PP")
    measured = _predict(MeasuredPolymerExpert(), candidate, "elongation_at_break")
    proxy = _predict(
        ChainToughnessExpert(),
        candidate,
        "elongation_at_break",
        context=_tg_context(267.0),
    )
    assert proxy.status is PredictionStatus.OK
    relative_measured = measured.uncertainty.std / measured.quantity.value
    relative_proxy = proxy.uncertainty.std / proxy.quantity.value
    assert relative_proxy > relative_measured
    assert prefer(measured, proxy)
    assert best_prediction([proxy, measured], "elongation_at_break") is measured


def test_a_network_is_classified_brittle_without_being_measured():
    """The one thing the proxy does well, and it does it from architecture
    alone: a single molecule cannot draw."""
    candidate = _candidate("poly(HDDA)")
    prediction = _predict(ChainToughnessExpert(), candidate, "elongation_at_break")
    assert prediction.status is PredictionStatus.OK
    assert prediction.quantity.value < 0.05
    assert "network" in " ".join(prediction.notes)
    # No upstream transition was supplied and none was needed.
    assert prediction.provenance.parameters["classification"] == "network"


def test_the_proxy_refuses_without_a_temperature():
    candidate = _candidate("PS")
    prediction = _predict(
        ChainToughnessExpert(),
        candidate,
        "elongation_at_break",
        context=_tg_context(373.0),
        conditions=Conditions(),
    )
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "no temperature" in " ".join(prediction.notes)


def test_the_proxy_refuses_without_an_upstream_transition():
    """It classifies on where the temperature sits relative to Tg, and a chain
    with no transition available has no class."""
    prediction = _predict(ChainToughnessExpert(), _candidate("PS"), "elongation_at_break")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "no upstream expert supplied one" in " ".join(prediction.notes)


def test_the_proxy_refuses_far_above_the_window_it_was_fitted_in():
    """A rigid engineering plastic three hundred kelvin above ambient is a
    different material from every glass in the catalogue."""
    prediction = _predict(
        ChainToughnessExpert(),
        _candidate("PS"),
        "elongation_at_break",
        context=_tg_context(600.0),
    )
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "beyond the 100 K the catalogue covers" in " ".join(prediction.notes)


def test_the_proxy_declines_a_repeat_unit_that_is_neither_chain_nor_network():
    """Three attachment points on an uncrosslinked repeat unit is a branch
    point, and this classification is about linear chains and stated networks."""
    candidate = polymer_candidate("[*]CC([*])C[*]", conditions=ROOM)
    expert = ChainToughnessExpert()
    assert not expert.assess_domain(candidate).in_domain
    prediction = _predict(
        expert, candidate, "elongation_at_break", context=_tg_context(350.0)
    )
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "attachment points" in " ".join(prediction.notes)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


def test_a_linear_chain_has_no_crosslinks_by_definition():
    prediction = _predict(PolymerArchitectureExpert(), _candidate("PA6"), "crosslink_density")
    assert prediction.quantity.value == 0.0
    assert prediction.uncertainty.std == 0.0
    assert "definition" in prediction.uncertainty.basis


def test_a_branched_chain_still_has_no_crosslinks():
    """Branching is not crosslinking: low-density polyethylene is remeltable and
    a diacrylate network is not."""
    prediction = _predict(PolymerArchitectureExpert(), _candidate("LDPE"), "crosslink_density")
    assert prediction.quantity.value == 0.0


def test_a_network_reports_what_it_states():
    prediction = _predict(
        PolymerArchitectureExpert(), _candidate("poly(HDDA)"), "crosslink_density"
    )
    assert prediction.quantity.value == pytest.approx(9700.0)


def test_a_network_with_no_stated_density_is_refused_rather_than_called_zero():
    """Reporting zero would say it is a thermoplastic, which is the opposite of
    what its topology says."""
    spec = PolymerSpec(
        monomers=(MonomerUnit(smiles=PS, mole_fraction=1.0),),
        topology=PolymerTopology.NETWORK,
    )
    candidate = Candidate(
        material_class=MaterialClass.POLYMER, polymer=spec, conditions=ROOM
    )
    prediction = _predict(PolymerArchitectureExpert(), candidate, "crosslink_density")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "under-specified" in " ".join(prediction.notes)


# ---------------------------------------------------------------------------
# What the theoretical strength bound is worth
# ---------------------------------------------------------------------------


def _strength_bounds() -> dict[str, float]:
    """The panel's flaw-free bound for every catalogue polymer it can reach."""
    from formulate.experts.mechanical import PolymerMechanicalExpert

    expert = PolymerMechanicalExpert()
    table = MeasuredPolymerExpert()
    bounds: dict[str, float] = {}
    for record in polymer_records():
        candidate = _candidate(record.abbreviation)
        density = _predict(table, candidate, "amorphous_density")
        transition = _predict(table, candidate, "glass_transition_temperature")
        if not density.is_usable or not transition.is_usable:
            continue
        prediction = _predict(
            expert,
            candidate,
            "theoretical_strength",
            context={
                "amorphous_density": density,
                "glass_transition_temperature": transition,
            },
        )
        if prediction.is_usable:
            bounds[record.abbreviation] = prediction.quantity.to("Pa").value
    return bounds


def test_the_strength_bound_holds_for_the_glasses_it_was_argued_for():
    """``theoretical_strength`` is introduced as a flaw-free bound that a real
    specimen never reaches, and this is the first data in the repository that
    can check that claim rather than restate it.

    Polystyrene delivers 15 per cent of it and poly(methyl methacrylate) 21,
    which is the order of magnitude the module claims: a moulded bar fails at
    its largest flaw, one to three orders below the flawless solid.
    """
    utilisation = strength_utilisation(_strength_bounds())
    glasses = {
        name: ratio for name, ratio in utilisation.items() if name in ("PS", "PMMA")
    }
    assert set(glasses) == {"PS", "PMMA"}
    assert all(0.05 < ratio < 0.5 for ratio in glasses.values()), glasses


def test_the_bound_is_broken_by_every_semicrystalline_polymer_and_says_why():
    """The finding, not a defect in the bound.

    The two polyethylenes and polypropylene are above their glass transitions
    at ambient, so the panel computes their modulus on the rubber-elastic
    branch - three times rho R T over the entanglement mass, which for
    polyethylene is a megapascal.  They are measured at 8 to 41 MPa, so the
    "flaw-free upper bound" is exceeded by a factor of 17 for low-density
    polyethylene, 39 for high-density, and 339 for polypropylene - which is the
    worst because the rubbery modulus goes as the reciprocal of the
    entanglement mass and the panel puts polypropylene's at 6,014 g/mol against
    polyethylene's 948, so its rubber plateau is six times the lower while its
    measured strength is the higher.

    Nothing is wrong with the bound.  What is wrong is the material it was
    computed for: high-density polyethylene at 25 C is most of the way to
    crystalline and carries its load in crystals, and no property in this
    registry holds a degree of crystallinity.  The panel is answering for the
    amorphous polymer, which is a different material with the same repeat unit.
    This test fixes the size of that gap so it cannot be quietly closed by
    tuning the bound instead of by adding the crystallinity.
    """
    utilisation = strength_utilisation(_strength_bounds())
    semicrystalline = {
        name: ratio for name, ratio in utilisation.items() if name in ("HDPE", "LDPE", "PP")
    }
    assert set(semicrystalline) == {"HDPE", "LDPE", "PP"}
    assert all(ratio > 5.0 for ratio in semicrystalline.values()), semicrystalline
    assert max(semicrystalline.values()) < 500.0, semicrystalline

    from formulate.core.properties import PROPERTY_REGISTRY

    assert not [name for name in PROPERTY_REGISTRY if "crystall" in name]


def test_no_tensile_strength_was_added_to_the_registry():
    """The catalogue carries measured tensile strengths and they are
    deliberately not exposed as a property: a real strength is set by the
    largest flaw in the specimen, which is a property of how it was made."""
    from formulate.core.properties import PROPERTY_REGISTRY

    assert "tensile_strength" not in PROPERTY_REGISTRY


# ---------------------------------------------------------------------------
# Every prediction states an uncertainty and every refusal states a reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expert", [MeasuredPolymerExpert(), ChainToughnessExpert(), PolymerArchitectureExpert()]
)
def test_every_answer_carries_a_number_and_every_refusal_carries_a_reason(expert):
    for record in polymer_records():
        candidate = _candidate(record.abbreviation)
        for prop in sorted(expert.supported_properties):
            prediction = _predict(
                expert, candidate, prop, context=_tg_context(record.glass_transition_k or 350.0)
            )
            if prediction.is_usable:
                assert prediction.uncertainty.std is not None
                assert prediction.uncertainty.basis
                assert math.isfinite(prediction.uncertainty.std)
            else:
                assert " ".join(prediction.notes).strip()
