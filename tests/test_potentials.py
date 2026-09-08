"""The fast rung of the multi-fidelity ladder, and when it hands over.

Section 8. Two things are worth separating here: the potential itself, which
needs torch and downloaded weights and is marked slow, and the escalation
policy, which is arithmetic over stated uncertainties and must be testable with
nothing installed at all. Most of the decisions this module makes are the
second kind.
"""

from __future__ import annotations

import numpy as np
import pytest

from formulate.physics.potentials import (
    EscalationDecision,
    EscalationPolicy,
    InteratomicPotential,
    MacePotential,
    PotentialCache,
    PotentialInfo,
    PotentialResult,
    UnsupportedSystem,
    available_potentials,
)

requires_mace = pytest.mark.skipif(
    not MacePotential().is_available(), reason="MACE or torch is not installed"
)


class _FakePotential(InteratomicPotential):
    """A deterministic stand-in, so the ladder is testable without weights."""

    def __init__(self, elements=frozenset({1, 6, 8})) -> None:
        self.info = PotentialInfo(
            identifier="fake-1",
            description="a constant per atom",
            reference="this test",
            supported_elements=elements,
        )
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def compute(self, atomic_numbers, positions, *, forces=True):
        refusal = self.refusal(atomic_numbers)
        if refusal:
            raise UnsupportedSystem(refusal)
        self.calls += 1
        positions = np.asarray(positions, dtype=float)
        return PotentialResult(
            energy_ev=-float(sum(atomic_numbers)),
            forces_ev_per_angstrom=np.zeros_like(positions) if forces else None,
            info=self.info,
        )


# --------------------------------------------------------------------------
# The element domain
# --------------------------------------------------------------------------


def test_an_element_outside_the_training_set_is_refused_not_extrapolated():
    potential = _FakePotential()
    reason = potential.refusal([1, 6, 14])
    assert reason is not None
    assert "[14]" in reason
    with pytest.raises(UnsupportedSystem):
        potential.compute([1, 6, 14], np.zeros((3, 3)))


def test_a_supported_molecule_is_not_refused():
    assert _FakePotential().refusal([1, 1, 8]) is None


def test_the_refusal_names_what_is_supported():
    """A bare "unsupported" leaves the caller unable to act on it."""
    reason = _FakePotential().refusal([35])
    assert "[1, 6, 8]" in reason


# --------------------------------------------------------------------------
# Content addressing
# --------------------------------------------------------------------------


def test_the_same_geometry_is_never_computed_twice():
    potential, cache = _FakePotential(), PotentialCache()
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]])
    first = cache.compute(potential, [1, 1], positions, forces=False)
    second = cache.compute(potential, [1, 1], positions, forces=False)
    assert potential.calls == 1
    assert first is second
    assert (cache.hits, cache.misses) == (1, 1)


def test_a_position_that_differs_below_the_rounding_is_the_same_point():
    potential, cache = _FakePotential(), PotentialCache()
    a = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]])
    b = a + 1e-9
    cache.compute(potential, [1, 1], a, forces=False)
    cache.compute(potential, [1, 1], b, forces=False)
    assert potential.calls == 1


def test_a_moved_atom_is_a_different_point():
    potential, cache = _FakePotential(), PotentialCache()
    a = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]])
    cache.compute(potential, [1, 1], a, forces=False)
    cache.compute(potential, [1, 1], a + 0.01, forces=False)
    assert potential.calls == 2


def test_asking_for_forces_is_a_different_ask():
    potential, cache = _FakePotential(), PotentialCache()
    positions = np.zeros((2, 3))
    cache.compute(potential, [1, 1], positions, forces=False)
    cache.compute(potential, [1, 1], positions, forces=True)
    assert potential.calls == 2


def test_two_potentials_do_not_share_an_address():
    a, b = _FakePotential(), _FakePotential()
    b.info = PotentialInfo(
        identifier="fake-2", description="", reference="", supported_elements=frozenset({1})
    )
    cache = PotentialCache()
    cache.compute(a, [1, 1], np.zeros((2, 3)), forces=False)
    cache.compute(b, [1, 1], np.zeros((2, 3)), forces=False)
    assert (a.calls, b.calls) == (1, 1)


def test_a_refusal_is_not_cached():
    """A widened element domain must take effect without clearing anything."""
    potential, cache = _FakePotential(), PotentialCache()
    with pytest.raises(UnsupportedSystem):
        cache.compute(potential, [14], np.zeros((1, 3)))
    assert len(cache) == 0


# --------------------------------------------------------------------------
# Escalation
# --------------------------------------------------------------------------


def test_nothing_wrong_means_nothing_bought():
    decision = EscalationPolicy().decide(value=100.0, uncertainty=1.0)
    assert not decision.escalate
    assert "unchallenged" in decision.describe()


def test_out_of_domain_always_escalates():
    decision = EscalationPolicy().decide(out_of_domain_reason="contains silicon")
    assert decision.escalate
    assert "silicon" in decision.describe()


def test_a_wide_stated_uncertainty_escalates():
    decision = EscalationPolicy(uncertainty_fraction=0.10).decide(value=100.0, uncertainty=25.0)
    assert decision.escalate
    assert "25%" in decision.describe()


def test_agreement_within_the_combined_error_does_not_escalate():
    decision = EscalationPolicy().decide(
        value=100.0, uncertainty=5.0, other_value=104.0, other_uncertainty=5.0
    )
    assert not decision.escalate


def test_disagreement_beyond_the_combined_error_escalates():
    decision = EscalationPolicy().decide(
        value=100.0, uncertainty=1.0, other_value=140.0, other_uncertainty=1.0
    )
    assert decision.escalate
    assert "at least one of them is wrong" in decision.describe()


def test_disagreement_with_no_stated_error_is_not_silently_accepted():
    """Two methods that differ and neither says how much it could be off."""
    decision = EscalationPolicy().decide(value=100.0, other_value=140.0)
    assert decision.escalate
    assert "cannot be attributed" in decision.describe()


def test_a_candidate_that_cannot_change_the_answer_is_not_bought():
    decision = EscalationPolicy(rank_horizon=5).decide(
        out_of_domain_reason="contains silicon", rank=40, changes_ranking=False
    )
    assert not decision.escalate


def test_a_cost_skip_still_reports_the_concern():
    """The bug this guards: a doubted number reported as untroubled.

    ``describe`` had one branch for escalation and one for silence, so a
    candidate skipped on cost - which is to say, one whose value was doubted
    and the doubt left unresolved - printed "inside its domain and
    unchallenged". That is the opposite of what happened.
    """
    decision = EscalationPolicy(rank_horizon=5).decide(
        out_of_domain_reason="contains silicon", rank=40, changes_ranking=False
    )
    described = decision.describe()
    assert "unchallenged" not in described
    assert "not escalated" in described
    assert "beyond the 5" in described


def test_a_low_ranked_candidate_that_would_change_the_ranking_is_still_bought():
    decision = EscalationPolicy(rank_horizon=5).decide(
        out_of_domain_reason="contains silicon", rank=40, changes_ranking=True
    )
    assert decision.escalate


def test_an_empty_decision_describes_itself_without_reasons():
    assert "no escalation" in EscalationDecision(False).describe()


# --------------------------------------------------------------------------
# The real backend
# --------------------------------------------------------------------------


@requires_mace
@pytest.mark.slow
def test_mace_declares_its_elements_from_the_weights_not_from_a_literal():
    elements = MacePotential().info.supported_elements
    assert {1, 6, 7, 8} <= elements
    assert 14 not in elements  # silicon is not in MACE-OFF23


@requires_mace
@pytest.mark.slow
def test_mace_states_that_it_has_no_uncertainty_estimator():
    """Absent rather than invented; the escalation leans on other signals."""
    potential = MacePotential()
    assert potential.info.provides_uncertainty is False
    result = potential.compute([1, 8, 1], _WATER, forces=False)
    assert result.uncertainty_ev is None
    assert any("no uncertainty estimator" in d for d in result.diagnostics)


@requires_mace
@pytest.mark.slow
def test_mace_is_deterministic_for_the_same_geometry():
    potential = MacePotential()
    first = potential.compute([1, 8, 1], _WATER, forces=False).energy_ev
    second = potential.compute([1, 8, 1], _WATER, forces=False).energy_ev
    assert first == pytest.approx(second, abs=1e-9)


@requires_mace
@pytest.mark.slow
def test_mace_forces_have_one_row_per_atom():
    result = MacePotential().compute([1, 8, 1], _WATER)
    assert np.asarray(result.forces_ev_per_angstrom).shape == (3, 3)


@requires_mace
@pytest.mark.slow
def test_mace_refuses_silicon_rather_than_extrapolating():
    with pytest.raises(UnsupportedSystem, match="14"):
        MacePotential().compute([14, 14], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.3]]))


@requires_mace
@pytest.mark.slow
def test_the_atomization_energy_of_water_is_positive_and_near_its_measured_value():
    """Electronic, so it is compared against D_e rather than the tabulated D_0.

    Water's D_0 is 917.8 kJ/mol and its zero-point energy 55.4 kJ/mol, so the
    electronic well depth is about 973. Comparing against the 917.8 instead
    would look like a 50 kJ/mol model error that is entirely the missing
    vibrational term.
    """
    value_ev = MacePotential().atomization_energy_ev([1, 8, 1], _WATER)
    assert value_ev * 96.48533212331 == pytest.approx(973.2, abs=30.0)


@requires_mace
@pytest.mark.slow
def test_available_potentials_lists_mace_when_it_is_installed():
    assert any(p.info.identifier.startswith("mace") for p in available_potentials())


_WATER = np.array(
    [[0.7575, 0.5871, 0.0], [0.0, 0.0, 0.0], [-0.7575, 0.5871, 0.0]]
)
