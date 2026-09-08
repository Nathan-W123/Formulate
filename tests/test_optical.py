"""Refractive index by two routes that disagree about which is better.

Section 4's mixture of experts, with no precedence rule anywhere: the physics
route and the learned route both state their spreads and ``prefer()`` chooses.
These tests check that the spreads are honest, because that is the only thing
making the choice correct.
"""

from __future__ import annotations

import math

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import MaterialClass, molecule_candidate
from formulate.core.conditions import Conditions
from formulate.core.prediction import PredictionStatus
from formulate.core.quantity import Quantity
from formulate.evaluation.engine import prefer
from formulate.experts import default_registry
from formulate.experts.base import PredictionRequest
from formulate.experts.optical import LorentzLorenzExpert, lorentz_lorenz

_UPSTREAM = frozenset(
    {
        "refractive_index", "liquid_density", "molar_mass", "molar_refractivity",
        "critical_temperature", "critical_pressure", "critical_volume",
        "normal_boiling_point",
    }
)


def _resolve(smiles: str, registry=None):
    """Run the panel in dependency order and return the winning predictions."""
    registry = registry or default_registry()
    candidate = molecule_candidate(smiles)
    context: dict = {}
    offers: list = []
    for expert in registry.resolution_order(
        registry.experts_for(_UPSTREAM, MaterialClass.MOLECULE)
    ):
        request = PredictionRequest(
            candidate=candidate,
            properties=_UPSTREAM,
            conditions=Conditions.standard(),
            context=dict(context),
        )
        for prediction in expert.predict(request):
            if prediction.property == "refractive_index":
                offers.append(prediction)
            if not prediction.is_usable:
                continue
            incumbent = context.get(prediction.property)
            if incumbent is None or prefer(prediction, incumbent):
                context[prediction.property] = prediction
    return context, offers


# -- the equation ----------------------------------------------------------


def test_the_equation_reproduces_benzene():
    """Molar refraction 26.44 cm^3/mol, molar volume 89.1, measured n 1.5011."""
    assert lorentz_lorenz(26.44, 89.1) == pytest.approx(1.5011, abs=0.01)


def test_the_equation_reproduces_water_from_its_own_constants():
    """Molar refraction 3.71 cm^3/mol, molar volume 18.07, measured n 1.333."""
    assert lorentz_lorenz(3.71, 18.07) == pytest.approx(1.333, abs=0.01)


def test_a_denser_medium_refracts_more():
    assert lorentz_lorenz(26.44, 80.0) > lorentz_lorenz(26.44, 100.0)


def test_the_pole_is_refused_rather_than_returned():
    """A molar volume at or below the molar refraction is two estimates
    disagreeing, not a medium. Past the pole the equation returns the square
    root of a negative number or a huge one, and either looks like an answer."""
    with pytest.raises(ValueError, match="pole"):
        lorentz_lorenz(30.0, 30.0)
    with pytest.raises(ValueError, match="pole"):
        lorentz_lorenz(30.0, 20.0)
    with pytest.raises(ValueError, match="positive"):
        lorentz_lorenz(30.0, 0.0)


def test_a_vanishing_polarisability_gives_a_vacuum():
    assert lorentz_lorenz(1e-9, 100.0) == pytest.approx(1.0, abs=1e-6)


# -- the expert ------------------------------------------------------------


@requires_rdkit
def test_it_refuses_without_a_density_rather_than_inventing_one():
    expert = LorentzLorenzExpert()
    prediction = expert.predict(
        PredictionRequest(
            candidate=molecule_candidate("CCO"),
            properties=frozenset({"refractive_index"}),
            conditions=Conditions.standard(),
        )
    )[0]
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "liquid_density" in prediction.notes[0]


@requires_rdkit
def test_it_refuses_without_a_temperature():
    expert = LorentzLorenzExpert()
    prediction = expert.predict(
        PredictionRequest(
            candidate=molecule_candidate("CCO"),
            properties=frozenset({"refractive_index"}),
            conditions=Conditions(),
        )
    )[0]
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "temperature" in prediction.notes[0]


@requires_rdkit
def test_it_refuses_a_solid():
    """A liquid refractive index is a property of the liquid."""
    _, offers = _resolve("c1ccc2ccccc2c1")  # naphthalene, melts at 80 C
    physics = [o for o in offers if o.expert_id == "lorentz_lorenz"]
    assert physics
    assert all(o.quantity is None for o in physics)


@requires_rdkit
@pytest.mark.parametrize(
    "smiles,measured",
    [("Cc1ccccc1", 1.4941), ("CCO", 1.3611), ("c1ccccc1", 1.5011), ("CCCCCC", 1.3749)],
)
def test_the_physics_route_is_accurate_when_the_density_is_measured(smiles, measured):
    _, offers = _resolve(smiles)
    physics = next(o for o in offers if o.expert_id == "lorentz_lorenz")
    assert physics.quantity is not None
    assert physics.quantity.value == pytest.approx(measured, abs=0.02)


@requires_rdkit
def test_the_stated_spread_widens_when_the_density_is_only_estimated():
    """The whole design rests on this: the physics route has to say when its
    input is weak, or prefer() has no way to choose against it."""
    _, measured_offers = _resolve("Cc1ccccc1")  # toluene has a tabulated density
    _, estimated_offers = _resolve("CC(C)(C)c1ccccc1")  # tert-butylbenzene does not
    tight = next(o for o in measured_offers if o.expert_id == "lorentz_lorenz")
    wide = next(o for o in estimated_offers if o.expert_id == "lorentz_lorenz")
    assert wide.uncertainty.std > 2 * tight.uncertainty.std


@requires_rdkit
def test_the_molar_refraction_error_is_not_counted_twice():
    """It is already inside the measured 0.0123, and propagating Crippen's own
    2.5 cm^3/mol on top made this expert quote +/- 0.049 on an answer that was
    within 0.0012 of the measurement, so the ranking preferred a route that was
    fourteen times further out."""
    _, offers = _resolve("Cc1ccccc1")
    physics = next(o for o in offers if o.expert_id == "lorentz_lorenz")
    assert physics.uncertainty.std < 0.02
    assert "already includes the molar refraction" in physics.uncertainty.basis


@requires_rdkit
def test_the_physics_route_wins_when_it_has_a_real_density():
    context, _ = _resolve("Cc1ccccc1")
    assert context["refractive_index"].expert_id == "lorentz_lorenz"


@requires_rdkit
def test_the_learned_route_wins_when_the_density_is_estimated():
    context, offers = _resolve("CC(C)(C)c1ccccc1")
    assert context["liquid_density"].expert_id == "interfacial"
    assert context["refractive_index"].expert_id == "learned_refractive_index"
    # And the physics route did answer; it simply said so less confidently.
    assert any(o.expert_id == "lorentz_lorenz" and o.quantity for o in offers)


@requires_rdkit
def test_neither_expert_branches_on_which_one_it_is_competing_with():
    """No precedence rule: the choice is made by prefer() on stated spreads.

    Their docstrings cite the measurement that compares them, which is the
    point; what neither may do is read the other's identity and behave
    differently, because then the spreads stop being the thing being compared.
    """
    import inspect

    from formulate.experts import learned, optical

    pairs = [
        (optical.LorentzLorenzExpert, "learned_refractive_index"),
        (learned.LearnedRefractiveIndexExpert, "lorentz_lorenz"),
    ]
    for cls, rival in pairs:
        source = inspect.getsource(cls)
        body = source[source.index('"""', source.index('"""') + 3) + 3:]
        assert rival not in body, f"{cls.__name__} names {rival} outside its docstring"
        assert "expert_id ==" not in body


# -- the dependency order this depends on ----------------------------------


@requires_rdkit
def test_a_consumer_runs_after_every_supplier_not_merely_the_first():
    """Both `measured` and `interfacial` supply a liquid density.

    Ordering on the first supplier made the refractive index consumer ready as
    soon as `measured` had run, so for any molecule `measured` had no density
    for it reported that none was available while `interfacial` was still
    queued behind it.
    """
    registry = default_registry()
    order = [
        e.id
        for e in registry.resolution_order(
            registry.experts_for(_UPSTREAM, MaterialClass.MOLECULE)
        )
    ]
    suppliers = [
        e.id
        for e in registry
        if "liquid_density" in e.supported_properties and e.id in order
    ]
    for supplier in suppliers:
        assert order.index(supplier) < order.index("lorentz_lorenz"), supplier


@requires_rdkit
def test_an_estimated_density_still_reaches_the_physics_route():
    _, offers = _resolve("CC(C)(C)c1ccccc1")
    physics = next(o for o in offers if o.expert_id == "lorentz_lorenz")
    assert physics.quantity is not None, physics.notes


def test_a_quantity_with_no_unit_is_what_a_refractive_index_is():
    assert Quantity(value=1.5, unit="").to("").value == 1.5


def test_the_equation_is_monotone_in_polarisability():
    values = [lorentz_lorenz(r, 100.0) for r in (10.0, 20.0, 30.0, 40.0)]
    assert values == sorted(values)
    assert all(math.isfinite(v) for v in values)
