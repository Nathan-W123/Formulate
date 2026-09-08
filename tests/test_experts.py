"""Expert interface, registry and the Phase 1 panel."""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import MaterialClass, molecule_candidate
from formulate.core.conditions import Conditions
from formulate.core.prediction import PredictionStatus
from formulate.core.properties import get_property
from formulate.core.quantity import Quantity
from formulate.experts.base import Expert, PredictionRequest
from formulate.experts.interfacial import (
    association_factor,
    brock_bird_surface_tension,
    hildebrand_parameter,
    propagate,
    rackett_molar_volume,
)


def _request(smiles, expert, conditions=None):
    return PredictionRequest(
        candidate=molecule_candidate(smiles),
        properties=frozenset(expert.supported_properties),
        conditions=conditions or Conditions.standard(),
    )


def test_registry_reports_coverage_and_gaps(registry):
    coverage = registry.coverage(["normal_boiling_point", "logp"], MaterialClass.MOLECULE)
    # Three experts cover the boiling point: a compiled measurement, a
    # group-contribution estimate, and a fitted model for the structures group
    # contribution has no groups to match. Coverage lists all three; which one
    # answers is decided per candidate by the evaluation engine, not here, and
    # they are deliberately ordered worst-to-best rather than by preference.
    assert coverage["normal_boiling_point"] == [
        "joback", "learned_boiling_point", "measured",
    ]
    assert coverage["logp"] == ["crippen"]
    assert registry.uncovered(
        ["normal_boiling_point"], MaterialClass.POLYMER
    ) == ["normal_boiling_point"]


def test_registry_rejects_duplicate_ids(registry):
    with pytest.raises(ValueError):
        registry.register(registry.get("joback"))


def test_resolution_order_puts_dependencies_first(registry):
    properties = ["surface_tension", "critical_temperature", "molar_mass"]
    experts = registry.experts_for(properties, MaterialClass.MOLECULE)
    order = [e.id for e in registry.resolution_order(experts)]
    assert order.index("joback") < order.index("interfacial")


def test_every_expert_declares_registered_properties(registry):
    for expert in registry:
        for prop in expert.supported_properties:
            get_property(prop)


@requires_rdkit
def test_every_prediction_carries_its_full_context(registry):
    """Section 4 requires value, units, uncertainty, domain, version, conditions."""
    candidate = molecule_candidate("CCO")
    context: dict = {}
    for expert in registry.resolution_order(list(registry)):
        request = PredictionRequest(
            candidate=candidate,
            properties=frozenset(
                p for e in registry for p in e.supported_properties
            ),
            conditions=Conditions.standard(),
            context=dict(context),
        )
        for prediction in expert.predict(request):
            assert prediction.expert_id == expert.id
            assert prediction.expert_version
            if prediction.is_usable:
                assert prediction.quantity is not None
                assert prediction.method
                assert prediction.provenance is not None
                assert prediction.applicability is not None
                context[prediction.property] = prediction


@requires_rdkit
def test_joback_is_within_its_stated_error_for_common_compounds(registry):
    expert = registry.get("joback")
    for smiles, measured_k in [("CCO", 351.4), ("c1ccccc1", 353.2), ("CC(C)=O", 329.2)]:
        prediction = [
            p
            for p in expert.predict(_request(smiles, expert))
            if p.property == "normal_boiling_point"
        ][0]
        error = abs(prediction.quantity.to("K").value - measured_k)
        assert error < 2 * prediction.uncertainty.std


@requires_rdkit
def test_joback_reports_failure_for_unmatched_atoms(registry):
    expert = registry.get("joback")
    predictions = expert.predict(_request("[Fe]C", expert))
    assert all(p.status is PredictionStatus.FAILED for p in predictions)


@requires_rdkit
def test_an_expert_failure_does_not_raise(registry):
    """A broken candidate must degrade to a recorded failure, not an exception."""
    for expert in registry:
        predictions = expert.predict(_request("C[Xx]C", expert))
        assert all(not p.is_usable or p.quantity is not None for p in predictions)


@requires_rdkit
def test_condition_dependent_property_refuses_without_a_temperature(registry):
    expert = registry.get("joback")
    request = PredictionRequest(
        candidate=molecule_candidate("CCO"),
        properties=frozenset({"heat_capacity_gas"}),
        conditions=Conditions(),
    )
    assert expert.predict(request)[0].status is PredictionStatus.FAILED


@requires_rdkit
def test_esol_flags_saturated_hydrocarbons_as_out_of_domain(registry):
    """The compounds ESOL predicts worst must be the ones it flags."""
    expert = registry.get("esol")
    flagged = expert.assess_domain(molecule_candidate("CCCCCCCC"))
    clean = expert.assess_domain(molecule_candidate("CC(=O)Oc1ccccc1C(=O)O"))
    assert flagged.warnings and flagged.score < clean.score
    assert clean.in_domain


@requires_rdkit
def test_interfacial_refuses_without_its_dependencies(registry):
    expert = registry.get("interfacial")
    predictions = expert.predict(_request("CCO", expert))
    assert all(p.status is PredictionStatus.UNSUPPORTED for p in predictions)
    assert "upstream" in predictions[0].notes[0]


@requires_rdkit
def test_interfacial_refuses_above_the_critical_temperature(registry):
    joback, interfacial = registry.get("joback"), registry.get("interfacial")
    hot = Conditions(temperature=Quantity(value=2000.0, unit="K"))
    context = {
        p.property: p for p in joback.predict(_request("CCO", joback, hot)) if p.is_usable
    }
    request = PredictionRequest(
        candidate=molecule_candidate("CCO"),
        properties=frozenset({"surface_tension"}),
        conditions=hot,
        context=context,
    )
    prediction = interfacial.predict(request)[0]
    assert prediction.status is PredictionStatus.FAILED
    assert "critical temperature" in prediction.notes[0]


def test_correlations_reproduce_measured_benzene():
    """Guards the correlations themselves, independent of any estimated input."""
    tc, pc, tb, vc, hvap = 562.05, 48.95e5, 353.24, 2.56e-4, 30720.0
    assert brock_bird_surface_tension(tc, pc, tb, 298.15) * 1e3 == pytest.approx(28.2, abs=1.5)
    assert rackett_molar_volume(tc, pc, vc, 298.15) * 1e6 == pytest.approx(89.4, abs=2.0)
    # Pa^0.5 -> MPa^0.5 divides by 1e3: the square root halves the exponent.
    assert hildebrand_parameter(hvap, tb, tc, vc, pc, 298.15) / 1e3 == pytest.approx(
        18.7, abs=0.5
    )


def test_uncertainty_propagation_matches_an_analytic_derivative():
    """For f(x) = 3x the propagated spread must be 3 sigma."""
    value, std = propagate(lambda v: 3.0 * v["x"], {"x": (2.0, 0.5)})
    assert value == pytest.approx(6.0)
    assert std == pytest.approx(1.5)


def test_propagation_combines_independent_inputs_in_quadrature():
    _, std = propagate(lambda v: v["a"] + v["b"], {"a": (1.0, 3.0), "b": (1.0, 4.0)})
    assert std == pytest.approx(5.0)


def test_association_factor_grows_with_hydrogen_bonding():
    assert association_factor("surface_tension", 0) == 1.0
    assert association_factor("surface_tension", 1) > 1.0
    assert association_factor("surface_tension", 3) >= association_factor("surface_tension", 1)


class _BrokenExpert(Expert):
    id = "broken"
    supported_properties = frozenset({"logp"})

    def _predict_one(self, prop, request, domain):
        raise RuntimeError("backend exploded")


def test_an_exception_inside_an_expert_becomes_a_recorded_failure():
    expert = _BrokenExpert()
    prediction = expert.predict(_request("CCO", expert))[0]
    assert prediction.status is PredictionStatus.FAILED
    assert "backend exploded" in prediction.notes[0]


# --------------------------------------------------------------------------
# A liquid property for something that is not a liquid
# --------------------------------------------------------------------------


@requires_rdkit
def test_a_solid_is_refused_a_liquid_density_by_every_route():
    """Naphthalene melts at 80 C. Both routes answered anyway.

    A corresponding-states correlation is a smooth function of reduced
    temperature and does not know where the substance freezes, so below the
    melting point it keeps returning a liquid density. The compiled route was
    worse: it returned 1020 kg/m^3 flagged in-domain, labelled "measured, not
    estimated", with an error bar of one kilogram per cubic metre.

    Nothing caught this. It surfaced by reading the top of a ranking, where a
    request for a coating solvent blend had been answered with toluene and
    solid naphthalene.
    """
    from formulate.core.candidate import molecule_candidate
    from formulate.core.conditions import Conditions
    from formulate.experts import default_registry
    from formulate.experts.base import PredictionRequest

    registry = default_registry()
    for smiles in ("c1ccc2ccccc2c1", "Oc1ccccc1"):  # naphthalene, phenol
        candidate = molecule_candidate(smiles)
        answered = []
        for expert in registry:
            if "liquid_density" not in expert.supported_properties:
                continue
            if MaterialClass.MOLECULE not in expert.supported_classes:
                continue
            for prediction in expert.predict(
                PredictionRequest(
                    candidate=candidate,
                    properties=frozenset({"liquid_density"}),
                    conditions=Conditions.standard(),
                )
            ):
                if prediction.quantity is not None:
                    answered.append((expert.id, prediction.quantity.value))
        assert not answered, f"{smiles} was given a liquid density: {answered}"


@requires_rdkit
def test_the_refusal_names_the_melting_point_rather_than_being_generic():
    from formulate.experts.measured import not_liquid_at

    reason = not_liquid_at("c1ccc2ccccc2c1", 298.15)
    assert reason is not None
    assert "solid at 25" in reason
    assert "melting at 80" in reason


@requires_rdkit
def test_a_liquid_is_untouched_by_the_phase_gate():
    from formulate.experts.measured import measured_value, not_liquid_at

    for smiles in ("Cc1ccccc1", "O", "CCO", "CCCCCC"):
        assert not_liquid_at(smiles, 298.15) is None
        assert measured_value("liquid_density", smiles) is not None


@requires_rdkit
def test_an_unknown_melting_point_does_not_cause_a_refusal():
    """Silence about the phase is not evidence of the wrong one.

    Refusing whenever a melting point is missing would decline half the panel
    on no evidence at all.
    """
    from formulate.experts.measured import not_liquid_at

    assert not_liquid_at("CC(C)(C)c1ccc(cc1)C(C)(C)CCC(C)(C)C", 298.15) is None


@requires_rdkit
def test_a_temperature_above_the_boiling_point_is_refused_too():
    from formulate.experts.measured import not_liquid_at

    reason = not_liquid_at("CCO", 400.0)  # ethanol boils at 78 C
    assert reason is not None
    assert "boils at 78" in reason
