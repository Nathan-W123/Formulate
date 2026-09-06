"""Expert interface, registry and the Phase 1 panel."""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import MaterialClass, molecule_candidate
from formulate.core.conditions import Conditions
from formulate.core.prediction import PredictionStatus
from formulate.core.properties import get_property
from formulate.core.quantity import Quantity
from formulate.experts import default_registry
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
    assert coverage["normal_boiling_point"] == ["joback"]
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
