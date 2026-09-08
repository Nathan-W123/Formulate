"""Selective physics validation: what gets validated, by what, and what changes."""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.coordination import DeterministicCoordinator, RunConfig
from formulate.coordination.validation import (
    DYNAMICS_PROTOCOLS,
    NOT_VALIDATABLE_REASONS,
    VALIDATABLE,
    Disagreement,
    ValidationMethod,
    ValidationPolicy,
    ValidationReport,
    merge_prediction,
    physics_prediction,
    select_targets,
    validatable_properties,
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import Prediction
from formulate.core.quantity import Quantity, Uncertainty, UncertaintyKind
from formulate.targets.spec import TargetSpec

_SPEC = TargetSpec.from_dict(
    {
        "name": "validation test",
        "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
        "requirements": [
            {
                "property": "normal_boiling_point",
                "direction": "in_range",
                "lower": "60 degC",
                "upper": "170 degC",
                "hard": True,
            },
            {"property": "homo_lumo_gap", "direction": "maximize", "lower": 4, "upper": 9},
            {"property": "logp", "direction": "target", "target": 2.0, "lower": 0, "upper": 4},
        ],
    }
)


# -- what physics may be asked ---------------------------------------------


def test_the_permitted_mapping_splits_requested_properties():
    permitted, refused = validatable_properties(_SPEC)
    assert permitted["homo_lumo_gap"] is ValidationMethod.QUANTUM
    assert "normal_boiling_point" in refused
    assert "logp" in refused


@pytest.mark.parametrize(
    "prop", ["normal_boiling_point", "logp", "synthetic_accessibility", "electronic_energy"]
)
def test_properties_outside_reach_are_refused_with_a_physical_reason(prop):
    """A refusal must say why, not merely that it is unsupported."""
    assert prop not in VALIDATABLE
    reason = NOT_VALIDATABLE_REASONS[prop]
    assert len(reason) > 40


def test_quantum_is_never_offered_a_bulk_property():
    for prop in ("liquid_density", "self_diffusion_coefficient", "work_of_separation"):
        assert ValidationMethod.QUANTUM not in VALIDATABLE[prop]
    # And a property with no workflow at all is offered to neither method.
    # Shear viscosity used to be the example here and no longer is: a
    # Green-Kubo integral became reachable once the pressure tensor did.
    # A partition coefficient is not, and will not be for the same reason it
    # never was - it is a free-energy difference between two solvents.
    assert "logp" not in VALIDATABLE
    assert "logp" in NOT_VALIDATABLE_REASONS


def test_dynamics_is_never_offered_an_electronic_property():
    for prop in ("homo_lumo_gap", "dipole_moment"):
        assert ValidationMethod.DYNAMICS not in VALIDATABLE[prop]


def test_absolute_electronic_energy_is_not_a_rankable_objective():
    """It is monotone in electron count, so ranking on it sorts by size.

    HF/STO-3G gives -75 Hartree for water, -152 for ethanol and -228 for
    benzene. A cross-candidate objective over those numbers measures how big
    each molecule is and nothing else.
    """
    assert "electronic_energy" not in VALIDATABLE
    assert "monotone" in NOT_VALIDATABLE_REASONS["electronic_energy"]


# -- selection -------------------------------------------------------------


@requires_rdkit
def test_selection_prefers_a_property_no_expert_could_supply():
    run = DeterministicCoordinator(config=RunConfig(pool_size=12)).run(_SPEC)
    ranked = [entry.candidate for entry in run.ranking.ranked]
    targets, _ = select_targets(ranked, _SPEC, ValidationPolicy(max_candidates=3))
    assert targets
    assert all(t.property == "homo_lumo_gap" for t in targets)
    assert all(t.method is ValidationMethod.QUANTUM for t in targets)


@requires_rdkit
def test_selection_respects_its_budget():
    run = DeterministicCoordinator(config=RunConfig(pool_size=20)).run(_SPEC)
    ranked = [entry.candidate for entry in run.ranking.ranked]
    targets, _ = select_targets(ranked, _SPEC, ValidationPolicy(max_candidates=2))
    assert len(targets) <= 2


@requires_rdkit
def test_infeasible_candidates_are_not_validated():
    """Confirming a property of a candidate that breaks a hard constraint changes nothing."""
    run = DeterministicCoordinator(config=RunConfig(pool_size=20)).run(_SPEC)
    ranked = [entry.candidate for entry in run.ranking.ranked]
    targets, _ = select_targets(ranked, _SPEC, ValidationPolicy(max_candidates=10))
    for target in targets:
        assert target.candidate.results.feasible


def test_a_spec_with_nothing_validatable_selects_nothing():
    spec = TargetSpec.from_dict(
        {"requirements": [{"property": "logp", "direction": "minimize", "lower": 0, "upper": 5}]}
    )
    targets, refused = select_targets([], spec, ValidationPolicy())
    assert targets == []
    assert "logp" in refused


# -- merging and disagreement ----------------------------------------------


def _prediction(value, std, expert_id="expert", prop="homo_lumo_gap"):
    return Prediction(
        property=prop,
        quantity=Quantity(value=value, unit="eV"),
        uncertainty=Uncertainty(std=std, kind=UncertaintyKind.EPISTEMIC, basis="test"),
        expert_id=expert_id,
        conditions=Conditions.standard(),
    )


def test_inverse_variance_combination_favours_the_tighter_estimate():
    expert = _prediction(5.0, 1.0, "expert")
    physics = _prediction(7.0, 0.25, "qm")
    merged, disagreement = merge_prediction(expert, physics)
    combined = merged.quantity.to("eV").value
    # The physics estimate is sixteen times more precise, so the combination
    # must sit close to it rather than halfway.
    assert 6.5 < combined < 7.0
    assert merged.uncertainty.std < 0.25
    assert disagreement is not None


def test_physics_does_not_automatically_replace_the_expert():
    """A converged calculation with a known systematic error is not obviously better."""
    expert = _prediction(5.0, 0.1, "expert")
    physics = _prediction(9.0, 3.0, "qm")
    merged, _ = merge_prediction(expert, physics)
    combined = merged.quantity.to("eV").value
    assert combined == pytest.approx(5.0, abs=0.1)


def test_a_missing_uncertainty_forces_replacement_and_says_so():
    expert = _prediction(5.0, None, "expert")
    physics = _prediction(7.0, 0.5, "qm")
    merged, _ = merge_prediction(expert, physics)
    assert merged.quantity.to("eV").value == pytest.approx(7.0)
    assert any("could not be combined" in note for note in merged.notes)


def test_significant_disagreement_is_flagged_in_the_merged_notes():
    expert = _prediction(5.0, 0.1, "expert")
    physics = _prediction(9.0, 0.1, "qm")
    merged, disagreement = merge_prediction(expert, physics)
    assert disagreement.is_significant()
    assert abs(disagreement.z_score) > 20
    assert any("overconfident" in note for note in merged.notes)


def test_a_z_score_is_undefined_without_any_stated_uncertainty():
    disagreement = Disagreement(
        property="homo_lumo_gap",
        expert_id="e",
        expert_value=5.0,
        expert_std=None,
        physics_value=9.0,
        physics_std=None,
        unit="eV",
    )
    assert disagreement.z_score is None
    assert not disagreement.is_significant()
    assert "undefined" in disagreement.describe()


def test_consistent_values_are_not_flagged():
    expert = _prediction(5.0, 1.0)
    physics = _prediction(5.3, 1.0, "qm")
    _, disagreement = merge_prediction(expert, physics)
    assert not disagreement.is_significant()


def test_a_physics_prediction_uses_the_same_record_as_an_expert_one():
    """One record type is what lets the ranker treat both identically."""
    prediction = physics_prediction(
        "homo_lumo_gap",
        Quantity(value=7.0, unit="eV"),
        Uncertainty(std=2.0, kind=UncertaintyKind.EPISTEMIC, basis="method error"),
        backend="qm:pyscf",
        method="b3lyp/6-31g",
        conditions=Conditions.standard(),
    )
    assert isinstance(prediction, Prediction)
    assert prediction.is_usable
    assert prediction.expert_id == "qm:pyscf"


# -- end to end ------------------------------------------------------------


@requires_rdkit
@pytest.mark.slow
def test_validated_values_reach_the_objective_vector_and_the_ranking():
    """Merging a prediction is not enough; the objective vector must be rebuilt."""
    from formulate.coordination import (
        IterationConfig,
        default_validating_coordinator,
    )

    coordinator = default_validating_coordinator(
        RunConfig(pool_size=12),
        IterationConfig(max_rounds=1, batch_size=6),
        ValidationPolicy(max_candidates=2, max_seconds=120),
    )
    run = coordinator.run_validated(_SPEC)

    validated = [
        entry.candidate
        for entry in run.final.ranking.ranked
        if entry.candidate.results.prediction_for("homo_lumo_gap") is not None
    ]
    assert validated, "no candidate received a validated gap"
    for candidate in validated:
        assert "homo_lumo_gap" in candidate.results.objective_vector
        assert candidate.results.objective_vector["homo_lumo_gap"] > 0.0
        assert candidate.results.simulation_ids

    assert "PHYSICS VALIDATION" in run.report(top_k=2)


def test_quantum_results_carry_the_conditions_they_were_computed_at():
    """A gas-phase result at zero Kelvin is not a result at 25 degrees.

    Stamping it with the requested conditions would make the section 11
    condition check pass on exactly the claim it was written to catch.
    """
    from formulate.coordination.validation import VACUUM_ZERO_KELVIN

    assert VACUUM_ZERO_KELVIN.temperature_k == 0.0
    assert VACUUM_ZERO_KELVIN.environment == "vacuum"


def test_a_condition_independent_property_is_not_rejected_on_conditions():
    """A frontier gap has no temperature to disagree about."""
    from formulate.coordination.validation import VACUUM_ZERO_KELVIN
    from formulate.evaluation.engine import EvaluationConfig
    from formulate.evaluation.scoring import OutcomeStatus, score_requirement

    requirement = next(r for r in _SPEC.requirements if r.property == "homo_lumo_gap")
    prediction = Prediction(
        property="homo_lumo_gap",
        quantity=Quantity(value=7.0, unit="eV"),
        uncertainty=Uncertainty(std=2.0, kind=UncertaintyKind.EPISTEMIC, basis="method"),
        expert_id="qm:pyscf",
        conditions=VACUUM_ZERO_KELVIN,
    )
    outcome = score_requirement(
        requirement,
        _SPEC,
        [prediction],
        requirement.desirability(),
        EvaluationConfig(),
    )
    assert outcome.status is not OutcomeStatus.CONDITION_MISMATCH
    assert outcome.effective_utility is not None


def test_a_condition_dependent_property_is_still_rejected_on_conditions():
    """The exemption must not leak to properties that genuinely vary."""
    from formulate.core.conditions import Conditions
    from formulate.evaluation.engine import EvaluationConfig
    from formulate.evaluation.scoring import OutcomeStatus, score_requirement
    from formulate.targets.spec import TargetSpec

    spec = TargetSpec.from_dict(
        {
            "conditions": {"temperature": "200 degC"},
            "requirements": [
                {
                    "property": "surface_tension",
                    "direction": "minimize",
                    "lower": 0.01,
                    "upper": 0.05,
                }
            ],
        }
    )
    prediction = Prediction(
        property="surface_tension",
        quantity=Quantity(value=0.02, unit="N/m"),
        expert_id="test",
        conditions=Conditions.standard(),
    )
    outcome = score_requirement(
        spec.requirements[0],
        spec,
        [prediction],
        spec.requirements[0].desirability(),
        EvaluationConfig(),
    )
    assert outcome.status is OutcomeStatus.CONDITION_MISMATCH


# -- every dynamics property must name its own protocol --------------------


def test_every_dynamics_property_has_a_protocol():
    """The invariant whose absence let a density be answered with an energy.

    Protocol selection used to be an if-else ending in a default, so every
    property but the radius of gyration ran the cohesive-energy workflow and
    came back in joules per mole. The dimensionality check caught it, but as an
    uncaught exception out of the middle of a design run rather than as a
    refusal.
    """
    from formulate.coordination.validation import CONDENSED_PROTOCOLS

    dynamics = {
        prop for prop, methods in VALIDATABLE.items() if ValidationMethod.DYNAMICS in methods
    }
    assert dynamics
    # Two routes now: a single-molecule or cluster trajectory, and a periodic
    # condensed phase. Every offered property must be on exactly one of them,
    # and nothing may be mapped that is not offered.
    routed = set(DYNAMICS_PROTOCOLS) | set(CONDENSED_PROTOCOLS)
    assert dynamics <= routed, sorted(dynamics - routed)
    assert routed <= dynamics
    assert not (set(DYNAMICS_PROTOCOLS) & set(CONDENSED_PROTOCOLS))


def test_every_protocol_name_resolves_to_a_real_workflow():
    from formulate.physics.md import MDProtocol

    for prop, name in DYNAMICS_PROTOCOLS.items():
        assert MDProtocol(name), prop


def test_a_property_with_no_workflow_is_refused_with_a_physical_reason():
    spec = TargetSpec.from_dict(
        {
            "name": "not obtainable here",
            "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
            "requirements": [
                {
                    "property": "logp",
                    "direction": "minimize",
                    "lower": "-2",
                    "upper": "4",
                },
            ],
        }
    )
    permitted, refused = validatable_properties(spec)
    assert permitted == {}
    # Not a vague "unsupported": the refusal names the calculation that would
    # be needed, which is what stops someone wiring up whichever protocol
    # happens to be nearest. Shear viscosity used to be the example here, and
    # naming its calculation - a Green-Kubo integral - is what eventually got
    # it implemented.
    assert "free-energy perturbation" in refused["logp"]


@requires_rdkit
@pytest.mark.parametrize("prop", ["work_of_separation"])
def test_a_bulk_property_declines_with_its_cost_rather_than_raising(prop):
    """The dynamics module was written to refuse these. It must get the chance.

    Only work of separation is left on this path: density and self-diffusion
    now have a periodic route through MMFF94 and OpenMM, and refuse on the
    validation budget instead. An interface needs two slabs and is still beyond
    what this installation will pay for.
    """
    from formulate.coordination.validation import (
        PhysicsValidator,
        ValidationTarget,
    )
    from formulate.core.candidate import molecule_candidate
    from formulate.physics.md.base import REQUIREMENTS, MDProtocol

    target = ValidationTarget(
        candidate=molecule_candidate("CCO"),
        property=prop,
        method=ValidationMethod.DYNAMICS,
        value_score=1.0,
        rationale="test",
    )
    prediction, reason = PhysicsValidator(policy=ValidationPolicy())._run_target(target, _SPEC, ValidationReport())

    assert prediction is None
    requirement = REQUIREMENTS[MDProtocol(DYNAMICS_PROTOCOLS[prop])]
    assert str(requirement.min_molecules) in reason
    assert "periodic" in reason


# -- the periodic condensed phase -----------------------------------------


def test_every_condensed_property_has_a_protocol_and_a_stated_cost():
    from formulate.coordination.validation import (
        CONDENSED_COST_SECONDS,
        CONDENSED_PROTOCOLS,
    )

    assert set(CONDENSED_PROTOCOLS) <= set(VALIDATABLE)
    assert set(CONDENSED_PROTOCOLS.values()) <= set(CONDENSED_COST_SECONDS)
    for prop, methods in VALIDATABLE.items():
        if prop in CONDENSED_PROTOCOLS:
            assert ValidationMethod.DYNAMICS in methods


@requires_rdkit
@pytest.mark.parametrize(
    "prop", ["liquid_density", "cohesive_energy_density", "self_diffusion_coefficient"]
)
def test_a_condensed_run_the_budget_cannot_afford_says_so_and_costs_nothing(prop):
    """A shortened condensed run does not fail, it answers wrong.

    A box that has not equilibrated returns a density thirty per cent low with
    an error bar that does not cover the gap, so the budget check has to refuse
    rather than trim the run.
    """
    import time

    from formulate.coordination.validation import PhysicsValidator, ValidationTarget
    from formulate.core.candidate import molecule_candidate

    target = ValidationTarget(
        candidate=molecule_candidate("CCO"),
        property=prop,
        method=ValidationMethod.DYNAMICS,
        value_score=1.0,
        rationale="test",
    )
    started = time.perf_counter()
    prediction, reason = PhysicsValidator(
        policy=ValidationPolicy(max_seconds=600.0)
    )._run_target(target, _SPEC, ValidationReport())
    assert prediction is None
    assert "minutes" in reason and "budget" in reason
    assert time.perf_counter() - started < 60.0


def test_cohesive_energy_density_is_validatable_again_now_a_bulk_route_exists():
    """It was refused while the only route was a finite cluster in the wrong unit."""
    assert "cohesive_energy_density" in VALIDATABLE
    assert "cohesive_energy_density" not in NOT_VALIDATABLE_REASONS
    # Shear viscosity was refused alongside it and is now reachable too, by
    # the same kind of move: the missing thing was a pressure tensor, not
    # physics. What stays refused is what no amount of tooling reaches from a
    # single-phase trajectory.
    assert "shear_viscosity" in VALIDATABLE
    assert "logp" not in VALIDATABLE
    assert "logp" in NOT_VALIDATABLE_REASONS


# -- energies defined as a difference -------------------------------------


def test_free_atom_multiplicities_are_tabulated_not_defaulted():
    """A carbon atom computed as a closed-shell singlet converges and is wrong.

    That is the failure mode this table exists to prevent: the calculation does
    not complain, and the atomization energy comes out hundreds of kJ/mol off
    with nothing to show for it.
    """
    from formulate.physics.qm.thermochemistry import ATOMIC_MULTIPLICITY

    assert ATOMIC_MULTIPLICITY["C"] == 3   # triplet ground state
    assert ATOMIC_MULTIPLICITY["N"] == 4   # quartet
    assert ATOMIC_MULTIPLICITY["O"] == 3   # triplet
    assert ATOMIC_MULTIPLICITY["H"] == 2   # doublet
    assert ATOMIC_MULTIPLICITY["He"] == 1  # closed shell


def test_an_element_with_no_atomic_reference_is_refused():
    from formulate.physics.qm.thermochemistry import unsupported_elements

    assert unsupported_elements(("C", "H", "O")) == ()
    assert unsupported_elements(("C", "Fe")) == ("Fe",)


@requires_rdkit
def test_atomization_declines_an_element_it_has_no_reference_for():
    from formulate.coordination.validation import PhysicsValidator, ValidationTarget
    from formulate.core.candidate import molecule_candidate

    target = ValidationTarget(
        candidate=molecule_candidate("[Fe](Cl)(Cl)Cl"),
        property="atomization_energy",
        method=ValidationMethod.QUANTUM,
        value_score=1.0,
        rationale="test",
    )
    prediction, reason = PhysicsValidator(policy=ValidationPolicy())._run_target(target, _SPEC, ValidationReport())
    assert prediction is None
    assert "Fe" in reason or "geometry" in reason


def test_interaction_energy_states_the_two_errors_it_does_not_remove():
    """Both are one-sided, so a caller that forgets them is biased, not noisy."""
    import inspect

    from formulate.physics.qm import thermochemistry

    source = inspect.getsource(thermochemistry.interaction_energy)
    assert "superposition" in source
    assert "minim" in source  # the geometry is relaxed, not searched


# --------------------------------------------------------------------------
# Which force field runs a condensed phase, and what its answer is worth
# --------------------------------------------------------------------------


def _embed(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


@requires_rdkit
def test_a_liquid_fitted_force_field_is_preferred_where_it_types_the_molecule():
    from formulate.coordination.validation import PhysicsValidator
    from formulate.physics.md import opls

    if not opls.available():
        pytest.skip("foyer/OPLS-AA is not installed")
    field, reason = PhysicsValidator._condensed_force_field(_embed("CCO"))
    assert field == "opls-aa"
    assert reason == ""


@requires_rdkit
def test_falling_back_to_mmff94_carries_the_reason_it_fell_back():
    """"OPLS-AA was unavailable" would be true and useless.

    Chloroform is refused because the types OPLS-AA assigns it do not add up to
    a neutral molecule, which is a fact about this structure. Whoever reads the
    prediction should see that rather than a generic unavailability.
    """
    from formulate.coordination.validation import PhysicsValidator
    from formulate.physics.md import opls

    if not opls.available():
        pytest.skip("foyer/OPLS-AA is not installed")
    field, reason = PhysicsValidator._condensed_force_field(_embed("ClC(Cl)Cl"))
    assert field == "mmff94"
    assert "net charge" in reason


@pytest.mark.parametrize(
    ("force_field", "prop", "fraction"),
    [
        ("opls-aa", "liquid_density", 0.012),
        ("mmff94", "liquid_density", 0.26),
        ("opls-aa", "cohesive_energy_density", 0.039),
        ("mmff94", "cohesive_energy_density", 0.30),
        ("opls-aa", "enthalpy_vaporization", 0.037),
        ("opls-aa", "molar_volume_liquid", 0.012),
        ("opls-aa", "hildebrand_solubility_parameter", 0.020),
    ],
)
def test_the_force_fields_measured_error_dominates_the_sampling_error(
    force_field, prop, fraction
):
    """A block average says a wrong density is a certain one unless this runs."""
    from formulate.coordination.validation import _widen_for_force_field

    value = Quantity(value=0.8, unit="g/cm^3")
    sampling = Uncertainty(std=0.002, kind=UncertaintyKind.SAMPLING, basis="block average")

    widened = _widen_for_force_field(sampling, value, force_field, prop)
    assert widened.std == pytest.approx((0.002**2 + (0.8 * fraction) ** 2) ** 0.5)
    assert widened.std > sampling.std
    assert widened.kind is UncertaintyKind.EPISTEMIC
    assert "block average" in widened.basis
    assert f"{fraction * 100:.1f} per cent" in widened.basis


def test_an_unmeasured_systematic_error_is_not_invented():
    """Self-diffusion spans orders of magnitude and was never measured here.

    Interpolating a systematic error for it from a density would be a made-up
    number wearing the same units as a measured one.
    """
    from formulate.coordination.validation import (
        CONDENSED_SYSTEMATIC,
        _widen_for_force_field,
    )

    for row in CONDENSED_SYSTEMATIC.values():
        assert "self_diffusion" not in row

    sampling = Uncertainty(std=1e-11, kind=UncertaintyKind.SAMPLING, basis="fit error")
    unchanged = _widen_for_force_field(
        sampling, Quantity(value=2e-9, unit="m^2/s"), "opls-aa", "self_diffusion"
    )
    assert unchanged is sampling


def test_the_liquid_fitted_force_field_claims_a_tighter_error_than_the_gas_fitted_one():
    """The whole point of the switch, asserted rather than assumed."""
    from formulate.coordination.validation import CONDENSED_SYSTEMATIC

    for prop in CONDENSED_SYSTEMATIC["opls-aa"]:
        assert (
            CONDENSED_SYSTEMATIC["opls-aa"][prop] < CONDENSED_SYSTEMATIC["mmff94"][prop]
        ), prop


def test_every_condensed_property_that_runs_has_a_stated_systematic_error():
    """Except the ones named as unmeasured, whose runs say so instead."""
    from formulate.coordination.validation import (
        CONDENSED_PROTOCOLS,
        CONDENSED_SYSTEMATIC,
        UNMEASURED_SYSTEMATIC,
    )

    for field in CONDENSED_SYSTEMATIC.values():
        assert set(field) == set(CONDENSED_PROTOCOLS) - UNMEASURED_SYSTEMATIC
    # The exemption list is not a place to hide a property: everything on it
    # has to be a property this system actually runs.
    assert UNMEASURED_SYSTEMATIC <= set(CONDENSED_PROTOCOLS)


def test_a_square_root_halves_the_relative_error_it_inherits():
    """Hildebrand is the square root of the cohesive energy density.

    Quoting the energy density's own fraction on it would overstate the
    uncertainty by a factor of two, which is the kind of conservatism that
    looks harmless and stops a good candidate being distinguished from a bad
    one.
    """
    from formulate.coordination.validation import CONDENSED_SYSTEMATIC

    for field in CONDENSED_SYSTEMATIC.values():
        assert field["hildebrand_solubility_parameter"] == pytest.approx(
            0.5 * field["cohesive_energy_density"], rel=0.05
        )


def test_a_molar_volume_carries_the_density_error_it_is_derived_from():
    """The molar mass is exact, so inverting a density preserves its error."""
    from formulate.coordination.validation import CONDENSED_SYSTEMATIC

    for field in CONDENSED_SYSTEMATIC.values():
        assert field["molar_volume_liquid"] == field["liquid_density"]


def test_an_absolute_electronic_energy_stays_out_even_though_it_is_free():
    """Every quantum run produces one and the reader for it already exists.

    It is still not a validation target: an absolute electronic energy is
    monotone in electron count, so ranking candidates on it sorts them by size.
    Cheap to compute is not the same as meaningful to compare.
    """
    from formulate.coordination.validation import _QM_OBSERVABLES

    assert "electronic_energy" in _QM_OBSERVABLES
    assert "electronic_energy" not in VALIDATABLE


# --------------------------------------------------------------------------
# A run that validated nothing has to say so
# --------------------------------------------------------------------------


def test_a_run_that_bought_no_physics_says_so_before_anything_else():
    """The default budget cannot afford the cheapest protocol.

    Six hundred seconds against three thousand six hundred, so a default run
    validates nothing. It used to report that as a list of per-property skip
    messages: each accurate, none of them saying the thing that matters, which
    is that the ranking in front of the reader rests on correlations alone.
    """
    from formulate.coordination.validation import ValidationReport

    report = ValidationReport(
        skipped={
            "liquid_density": (
                "a density run needs roughly 60 minutes of wall clock and the validation "
                "budget is 10; raise it to buy this one"
            )
        }
    )
    described = report.describe()
    assert described.startswith("NO PHYSICS RAN")
    assert "opt-in" in described
    assert report.priced_out == report.skipped


def test_a_physical_refusal_is_not_dressed_up_as_a_budget_problem():
    """Raising the budget would not buy a protocol that does not exist."""
    from formulate.coordination.validation import ValidationReport

    report = ValidationReport(
        skipped={"logp": "no dynamics protocol in this system produces logp"}
    )
    described = report.describe()
    assert "NO PHYSICS RAN" not in described
    assert "physical rather than budgetary" in described
    assert report.priced_out == {}


def test_the_notice_disappears_once_physics_actually_ran():
    from formulate.coordination.validation import ValidationReport

    report = ValidationReport(calls_made=1, seconds_spent=3601.0)
    assert "NO PHYSICS RAN" not in report.describe()


def test_the_default_budget_cannot_afford_the_cheapest_protocol():
    """The fact the notice exists for. If this ever changes, the wording must."""
    from formulate.coordination.validation import CONDENSED_COST_SECONDS, ValidationPolicy

    assert min(CONDENSED_COST_SECONDS.values()) > ValidationPolicy().max_seconds


# --------------------------------------------------------------------------
# The multi-fidelity rung (specification section 8)
# --------------------------------------------------------------------------


def _atomization_target(smiles: str):
    from formulate.coordination.validation import ValidationTarget
    from formulate.core.candidate import molecule_candidate

    return ValidationTarget(
        candidate=molecule_candidate(smiles),
        property="atomization_energy",
        method=ValidationMethod.QUANTUM,
        value_score=1.0,
        rationale="test",
    )


@requires_rdkit
def test_a_gap_is_never_answered_by_a_potential_that_only_knows_energies():
    """The escalation must be unconditional, not merely likely."""
    from formulate.coordination.validation import PhysicsValidator, ValidationTarget
    from formulate.core.candidate import molecule_candidate
    from formulate.physics.geometry import geometry_from_smiles

    target = ValidationTarget(
        candidate=molecule_candidate("CCO"),
        property="homo_lumo_gap",
        method=ValidationMethod.QUANTUM,
        value_score=1.0,
        rationale="test",
    )
    report = ValidationReport()
    validator = PhysicsValidator(policy=ValidationPolicy())
    settled, _ = validator._cheap_pass(
        target, geometry_from_smiles("CCO", n_conformers=1), report
    )
    assert settled is None
    assert report.cheap_calls == 0
    assert report.escalations and "escalate to QM" in report.escalations[0][1]


@requires_rdkit
def test_an_element_the_table_does_not_know_declines_instead_of_raising():
    """A KeyError from the stage whose job is to decline gracefully.

    Iron has no atomic number in the symbol table the potential is addressed
    through, and looking it up crashed the whole validation stage rather than
    recording a refusal.
    """
    from formulate.coordination.validation import PhysicsValidator
    from formulate.physics.geometry import geometry_from_smiles

    report = ValidationReport()
    settled, decision = PhysicsValidator(policy=ValidationPolicy())._cheap_pass(
        _atomization_target("[Fe](Cl)(Cl)Cl"),
        geometry_from_smiles("[Fe](Cl)(Cl)Cl", n_conformers=1, optimize=False),
        report,
    )
    assert settled is None
    assert decision is not None and decision.escalate
    assert "Fe" in decision.describe()


@requires_rdkit
def test_turning_the_ladder_off_sends_everything_to_the_solver():
    from formulate.coordination.validation import PhysicsValidator
    from formulate.physics.geometry import geometry_from_smiles

    report = ValidationReport()
    validator = PhysicsValidator(policy=ValidationPolicy(multi_fidelity=False))
    validator._run_quantum(
        _atomization_target("CCO"),
        geometry_from_smiles("CCO", n_conformers=1),
        _SPEC,
        report,
    )
    assert report.escalations == []
    assert report.cheap_calls == 0


@requires_rdkit
@pytest.mark.slow
def test_the_potential_answers_an_atomization_energy_and_the_solver_is_not_bought():
    """Section 8's whole point, and the measurement that inverted it.

    MACE-OFF23-small sits 13.0 kJ/mol from seven experimental atomization
    energies; the B3LYP/6-31G calculation this would otherwise escalate to sits
    78.7 kJ/mol from the same seven, under-binding every one. So the cheap rung
    is kept not because it is good enough but because it is the better answer,
    and the expensive call is avoided rather than deferred.
    """
    from formulate.coordination.validation import PhysicsValidator
    from formulate.physics.geometry import geometry_from_smiles
    from formulate.physics.potentials import MacePotential

    if not MacePotential().is_available():
        pytest.skip("MACE is not installed")

    report = ValidationReport()
    prediction, reason = PhysicsValidator(policy=ValidationPolicy())._run_quantum(
        _atomization_target("CO"),
        geometry_from_smiles("CO", n_conformers=3),
        _SPEC,
        report,
    )
    assert prediction is not None, reason
    assert report.cheap_calls == 1
    assert report.calls_avoided == 1
    assert prediction.expert_id.startswith("potential:")
    # Methanol's electronic atomization energy is about 2172 kJ/mol.
    assert prediction.quantity.to("J/mol").value / 1000.0 == pytest.approx(2172.0, abs=60.0)
    assert any("zero-point" in note for note in prediction.notes)


@requires_rdkit
@pytest.mark.slow
def test_silicon_escalates_because_an_extrapolating_network_is_worse():
    from formulate.coordination.validation import PhysicsValidator
    from formulate.physics.geometry import geometry_from_smiles
    from formulate.physics.potentials import MacePotential

    if not MacePotential().is_available():
        pytest.skip("MACE is not installed")

    report = ValidationReport()
    settled, decision = PhysicsValidator(policy=ValidationPolicy())._cheap_pass(
        _atomization_target("C[SiH3]"),
        geometry_from_smiles("C[SiH3]", n_conformers=1),
        report,
    )
    assert settled is None
    assert decision.escalate
    assert "14" in decision.describe()
    assert report.cheap_calls == 0
