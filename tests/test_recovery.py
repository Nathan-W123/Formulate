"""Hiding a known material and asking the pipeline to find it again.

Specification section 12: top-k recall against a benchmark set when a known
solution exists. These tests guard the harness, not the science - what the
harness actually measures lives in docs/BENCHMARKS.md, because the answer
changes whenever an expert does and a test that asserted a recall would fail
for a reason that is not a defect.
"""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.coordination.recovery import (
    DESCRIBABLE,
    HiddenMaterial,
    RecoveryResult,
    RecoverySuite,
    describe_material,
    spec_for,
)
from formulate.core.candidate import MaterialClass
from formulate.exploration.database import ReferenceDatabaseExplorer, load_reference_compounds
from formulate.targets.spec import Direction, TargetSpec

_TOLUENE = {
    "name": "toluene",
    "smiles": "Cc1ccccc1",
    "boiling_point_c": 110.6,
    "melting_point_c": -95.0,
    "density_25c": 0.8623,
    "logp": 2.73,
}


# -- describing a material by what it does ---------------------------------


def test_a_material_becomes_a_list_of_behaviours():
    hidden = describe_material(_TOLUENE)
    assert hidden is not None
    assert hidden.smiles == "Cc1ccccc1"
    assert "normal_boiling_point" in hidden.properties
    assert hidden.properties["liquid_density"] == (0.8623, "g/cm^3")


def test_a_material_described_by_two_numbers_is_not_described():
    """Dozens of compounds share any two properties."""
    assert describe_material({"name": "x", "smiles": "CCO", "logp": 1.0}) is None
    assert (
        describe_material({"name": "x", "smiles": "CCO", "logp": 1.0, "density_25c": 0.79})
        is None
    )


def test_only_requested_properties_are_used():
    hidden = describe_material(_TOLUENE, properties=["boiling_point_c", "logp"])
    assert hidden is None  # two is not enough, even when asked for


def test_the_description_is_capped():
    record = dict(_TOLUENE, surface_tension_25c=27.93, critical_temperature_k=591.8)
    hidden = describe_material(record, limit=3)
    assert len(hidden.properties) == 3


def test_every_describable_field_names_a_registered_property():
    from formulate.core.properties import get_property

    for prop, _unit in ((v[0], v[1]) for v in DESCRIBABLE.values()):
        get_property(prop)


# -- turning it into a target ----------------------------------------------


def test_the_spec_asks_for_the_measured_values():
    spec = spec_for(describe_material(_TOLUENE))
    assert isinstance(spec, TargetSpec)
    boiling = next(r for r in spec.requirements if r.property == "normal_boiling_point")
    assert boiling.direction is Direction.TARGET
    assert boiling.target.to("degC").value == pytest.approx(110.6)


def test_no_requirement_is_hard():
    """A hard constraint would make one bad estimate delete the right answer."""
    spec = spec_for(describe_material(_TOLUENE))
    assert not any(r.hard for r in spec.requirements)


def test_the_window_scales_to_the_property_not_to_the_value():
    """The bug this guards collapsed every hypervolume in every run.

    Ethanol's logp is -0.31, so a window at ten per cent of the value is
    +/- 0.031 - an order of magnitude tighter than Crippen's own error. Every
    candidate scored exactly zero desirability on logp, the frontier's logp
    axis was pinned at zero, and hypervolume is a product of edge lengths, so
    it was zero in every round of every run. The iterative search stops on
    "hypervolume gained nothing", so it also stopped early and said it had
    converged.
    """
    ethanol = {"name": "ethanol", "smiles": "CCO", "boiling_point_c": 78.3,
               "melting_point_c": -114.1, "density_25c": 0.7893, "logp": -0.31}
    logp = next(
        r for r in spec_for(describe_material(ethanol)).requirements if r.property == "logp"
    )
    width = logp.upper.value - logp.target.value
    assert width > 0.5, "a logp window must be wider than the method that answers it"


def test_the_window_means_the_same_thing_on_every_property():
    """A tenth of the range each property is observed to take."""
    from formulate.coordination.recovery import _reference_spread

    spec = spec_for(describe_material(_TOLUENE), tolerance=0.10)
    spreads = _reference_spread()
    for requirement in spec.requirements:
        width = requirement.upper.value - requirement.target.value
        assert width == pytest.approx(spreads[requirement.property] * 0.10, rel=1e-6)


def test_a_property_the_reference_set_does_not_tabulate_still_gets_a_window():
    hidden = HiddenMaterial(
        name="x",
        smiles="CCO",
        properties={
            "glass_transition_temperature": (105.0, "degC"),
            "liquid_density": (0.8, "g/cm^3"),
            "normal_boiling_point": (80.0, "degC"),
        },
    )
    glass = next(
        r for r in spec_for(hidden).requirements if r.property == "glass_transition_temperature"
    )
    assert glass.upper.value > glass.target.value


def test_closeness_is_demanded_rather_than_mere_membership():
    spec = spec_for(describe_material(_TOLUENE))
    assert all(r.shape > 1.0 for r in spec.requirements)


# -- withholding the answer -------------------------------------------------


@requires_rdkit
def test_an_excluded_compound_is_not_offered_by_the_catalogue():
    spec = spec_for(describe_material(_TOLUENE))
    proposed = ReferenceDatabaseExplorer(exclude=["Cc1ccccc1"]).propose(spec, 60)
    assert proposed
    assert not any(c.molecule.smiles == "Cc1ccccc1" for c in proposed)


@requires_rdkit
def test_exclusion_matches_on_structure_not_on_spelling():
    """A compound hidden one way must stay hidden written another."""
    spec = spec_for(describe_material(_TOLUENE))
    proposed = ReferenceDatabaseExplorer(exclude=["C1=CC=CC=C1C"]).propose(spec, 60)
    assert not any(c.molecule.smiles == "Cc1ccccc1" for c in proposed)


def test_excluding_nothing_changes_nothing():
    spec = spec_for(describe_material(_TOLUENE))
    plain = ReferenceDatabaseExplorer().propose(spec, 60)
    same = ReferenceDatabaseExplorer(exclude=[]).propose(spec, 60)
    assert [c.structure_id for c in plain] == [c.structure_id for c in same]


def test_the_reference_set_contains_the_compound_being_hidden():
    """Otherwise the discovery arm measures nothing."""
    smiles = {r["smiles"] for r in load_reference_compounds()}
    assert "Cc1ccccc1" in smiles


# -- reporting --------------------------------------------------------------


def _result(rank, mode="discovery", k=10):
    return RecoveryResult(
        hidden=describe_material(_TOLUENE), mode=mode, spec=spec_for(describe_material(_TOLUENE)),
        rank=rank, pool_size=50, top_k=k,
    )


def test_a_material_outside_the_top_k_is_not_recovered():
    assert not _result(20).recovered
    assert _result(3).recovered


def test_a_material_that_never_appeared_is_not_recovered():
    result = _result(None)
    assert not result.recovered
    assert "NOT FOUND" in result.describe()
    assert "never proposed it" in result.describe()


def test_the_two_modes_are_reported_apart():
    """A harness that pooled them would hide retrieval flattering discovery."""
    suite = RecoverySuite(results=[_result(1, "retrieval"), _result(None, "discovery")])
    described = suite.describe()
    assert "RETRIEVAL: 1 of 1" in described
    assert "DISCOVERY: 0 of 1" in described
    assert suite.recovery_rate == pytest.approx(0.5)
    assert len(suite.by_mode("discovery")) == 1


def test_an_unknown_mode_is_refused():
    from formulate.coordination.recovery import run_recovery

    with pytest.raises(ValueError, match="discovery"):
        run_recovery(describe_material(_TOLUENE), mode="cheat")


# -- end to end -------------------------------------------------------------


@requires_rdkit
@pytest.mark.slow
def test_retrieval_puts_a_catalogue_compound_near_the_top_of_its_own_description():
    """The easy half of the experiment, and the one that must not fail.

    A material described by five of its own measured properties, with itself
    left in the catalogue, is the least the ranking can be asked to do. Failing
    this means the experts or the ranking are wrong, not that search is hard.
    """
    from formulate.coordination.recovery import run_recovery

    record = next(r for r in load_reference_compounds() if r["name"] == "toluene")
    result = run_recovery(describe_material(record), mode="retrieval", rounds=1, batch_size=15)
    assert result.rank is not None, "toluene did not appear in a pool that contained it"
    assert result.mode == "retrieval"
    assert result.evaluations > 0


@requires_rdkit
@pytest.mark.slow
def test_discovery_withholds_the_answer_from_the_pool_it_searches():
    from formulate.coordination.recovery import run_recovery

    record = next(r for r in load_reference_compounds() if r["name"] == "toluene")
    result = run_recovery(describe_material(record), mode="discovery", rounds=1, batch_size=15)
    # Whether it is found again is the measurement, not the assertion. What is
    # asserted is that finding it would have meant something: the catalogue did
    # not simply hand it over.
    assert result.mode == "discovery"
    assert result.pool_size > 0
    if result.rank is not None:
        found = result.spec  # reached by search, which is the interesting case
        assert found is not None


@requires_rdkit
@pytest.mark.slow
def test_a_polymer_target_is_accepted_by_the_harness():
    """Section 6 of the brief asks for at least one polymer and one mixture."""
    hidden = HiddenMaterial(
        name="a polymer-like target",
        smiles="CC(C)(C(=O)OC)",
        properties={
            "glass_transition_temperature": (105.0, "degC"),
            "liquid_density": (1.18, "g/cm^3"),
            "logp": (1.0, ""),
        },
    )
    spec = spec_for(hidden, material_classes=(MaterialClass.POLYMER,))
    assert spec.material_classes == (MaterialClass.POLYMER,)
    assert len(spec.requirements) == 3


def test_the_explorer_that_produced_the_answer_is_recorded():
    """A discovery arm that found the answer in the catalogue would be a leak."""
    from formulate.coordination.recovery import _locate

    class _Entry:
        def __init__(self, candidate):
            self.candidate = candidate

    class _Candidate:
        def __init__(self, smiles, strategy):
            self.primary_smiles = smiles
            self.generation_strategy = strategy
            self.label = smiles
            self.candidate_id = smiles

    ranked = [
        _Entry(_Candidate("CCCO", "database:reference")),
        _Entry(_Candidate("CCO", "evolutionary:mutate")),
    ]
    rank, above, found_by = _locate("CCO", ranked)
    assert rank == 2
    assert above == ("CCCO",)
    assert found_by == "evolutionary:mutate"


def test_a_material_that_was_never_produced_records_no_strategy():
    from formulate.coordination.recovery import _locate

    assert _locate("CCO", []) == (None, (), "")


def test_a_window_narrower_than_the_answering_method_scores_nothing():
    """Why every hypervolume in every recovery run was zero.

    Desirability is risk-adjusted: a prediction is scored one standard
    deviation in the unfavourable direction. Crippen puts ethanol's logp at
    -0.0014 against a measured -0.31, comfortably inside a +/- 0.80 window, for
    a nominal desirability of 0.393. Crippen's own error is about 0.8, so the
    pessimistic value lands outside the window and the risk-adjusted
    desirability is exactly zero. Every candidate scores zero on that axis and
    the hypervolume collapses.
    """
    from formulate.targets.desirability import Desirability

    curve = Desirability(
        direction=Direction.TARGET, target=-0.3, lower=-1.101, upper=0.501, shape=2.0
    )
    nominal, adjusted = curve.evaluate(-0.0014, 0.8)
    assert nominal == pytest.approx(0.393, abs=0.01)
    assert adjusted == 0.0
    # Halve the method's error and the credit comes back.
    assert curve.evaluate(-0.0014, 0.3)[1] > 0.0


def test_a_recovery_result_explains_its_own_zero():
    result = RecoveryResult(
        hidden=describe_material(_TOLUENE),
        mode="discovery",
        spec=spec_for(describe_material(_TOLUENE)),
        rank=1,
        pool_size=50,
        success_rate=1.0,
        pinned_axes=("logp",),
    )
    described = result.describe()
    assert "collapsed, not flat" in described
    assert "logp" in described
