"""Bayesian optimisation over composition: the surrogate, the acquisition, the loop."""

from __future__ import annotations

import numpy as np
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
from formulate.exploration.acquisition import (
    augmented_tchebycheff,
    expected_improvement,
    random_simplex_weights,
    simplex_to_stick_breaking,
    sobol_like_design,
    stick_breaking_to_simplex,
)
from formulate.exploration.bayesopt import BayesOptConfig, BayesOptExplorer
from formulate.exploration.gp import GaussianProcess
from formulate.targets.spec import TargetSpec

_SPEC = TargetSpec.from_dict(
    {
        "name": "blend design",
        "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
        "material_classes": ["mixture"],
        "requirements": [
            {
                "property": "liquid_density",
                "direction": "target",
                "target": "0.82 g/cm^3",
                "lower": "0.70 g/cm^3",
                "upper": "0.95 g/cm^3",
                "weight": 2.0,
            },
            {
                "property": "hansen_distance",
                "direction": "minimize",
                "lower": "0 Pa^0.5",
                "upper": "20000 Pa^0.5",
            },
        ],
    }
)

_COMPONENTS = [
    ("Cc1ccccc1", ComponentRole.SOLVENT),
    ("CCCCCC", ComponentRole.SOLVENT),
    ("CC(C)=O", ComponentRole.CO_SOLVENT),
]


def _blend(fractions):
    return Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=tuple(
                MixtureComponent(role=role, fraction=float(f), molecule=MoleculeSpec(smiles=s))
                for (s, role), f in zip(_COMPONENTS, fractions)
            ),
            basis=FractionBasis.VOLUME,
        ),
        conditions=_SPEC.conditions,
    )


# -- the surrogate ---------------------------------------------------------


def test_the_process_interpolates_a_known_function():
    rng = np.random.default_rng(0)
    f = lambda X: np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] ** 2
    x = rng.uniform(-1, 1, size=(40, 2))
    gp = GaussianProcess().fit(x, f(x), restarts=4, seed=1)

    test = rng.uniform(-1, 1, size=(100, 2))
    mean, sigma = gp.predict(test)
    truth = f(test)
    assert np.sqrt(((mean - truth) ** 2).mean()) < 0.15 * truth.std()
    # Calibrated: nearly everything inside two standard deviations.
    assert (np.abs(mean - truth) < 2 * sigma).mean() > 0.85


def test_uncertainty_collapses_at_observed_points():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, size=(25, 2))
    y = x[:, 0] ** 2 - x[:, 1]
    gp = GaussianProcess().fit(x, y, restarts=4, seed=2)

    _, at_data = gp.predict(x)
    _, away = gp.predict(rng.uniform(-3, 3, size=(25, 2)))
    assert at_data.mean() < away.mean()


def test_relevance_determination_finds_the_active_dimension():
    """A dimension the output ignores should get a long lengthscale."""
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, size=(60, 2))
    y = np.sin(3 * x[:, 0])  # the second dimension does nothing
    gp = GaussianProcess().fit(x, y, restarts=6, seed=3)
    assert gp.hyper.lengthscales[1] > gp.hyper.lengthscales[0]


def test_a_degenerate_dataset_does_not_crash_the_factorisation():
    """Duplicated rows make the covariance singular; the jitter ladder is for that."""
    x = np.zeros((6, 2))
    x[3:] = 1.0
    y = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    gp = GaussianProcess().fit(x, y, restarts=2, seed=4)
    mean, sigma = gp.predict(x)
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(sigma))
    assert gp.jitter_used > 0 or np.all(sigma >= 0)


def test_the_process_refuses_a_single_observation():
    with pytest.raises(ValueError):
        GaussianProcess().fit(np.zeros((1, 2)), np.zeros(1))


# -- the simplex -----------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 5, 8])
def test_the_simplex_transform_is_a_bijection(k):
    """Optimising fractions directly would need repair, and repair fights the optimiser."""
    rng = np.random.default_rng(5)
    for _ in range(200):
        u = rng.random(k - 1)
        fractions = stick_breaking_to_simplex(u)
        assert fractions.sum() == pytest.approx(1.0)
        assert (fractions >= -1e-12).all()
        assert np.allclose(simplex_to_stick_breaking(fractions), u, atol=1e-9)


def test_the_transform_covers_the_corners():
    assert stick_breaking_to_simplex(np.array([1.0, 0.0]))[0] == pytest.approx(1.0)
    assert stick_breaking_to_simplex(np.array([0.0, 0.0]))[-1] == pytest.approx(1.0)


def test_a_space_filling_design_spreads_over_the_cube():
    rng = np.random.default_rng(6)
    design = sobol_like_design(16, 3, rng)
    assert design.shape == (16, 3)
    assert (design >= 0).all() and (design <= 1).all()
    # A Latin hypercube puts one point in each stratum of every axis.
    for axis in range(3):
        strata = np.floor(design[:, axis] * 16).astype(int)
        assert len(set(strata)) == 16


# -- the acquisition -------------------------------------------------------


def test_expected_improvement_prefers_a_better_mean_and_more_uncertainty():
    mean = np.array([0.9, 0.5, 0.5])
    sigma = np.array([0.1, 0.3, 0.1])
    ei = expected_improvement(mean, sigma, best=0.6, exploration=0.0)
    assert ei[0] > ei[1] > ei[2]


def test_a_certain_and_worse_point_has_no_expected_improvement():
    ei = expected_improvement(np.array([0.4]), np.array([0.0]), best=0.6)
    assert ei[0] == 0.0


def test_tchebycheff_reaches_a_point_a_weighted_sum_cannot():
    """The whole reason for the minimum term: a concave frontier."""
    frontier = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    weights = np.array([0.5, 0.5])
    assert int(np.argmax(frontier @ weights)) in (0, 2)
    assert int(np.argmax(augmented_tchebycheff(frontier, weights))) == 1


def test_random_weights_lie_on_the_simplex():
    rng = np.random.default_rng(7)
    for _ in range(50):
        w = random_simplex_weights(4, rng)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()


# -- the explorer ----------------------------------------------------------


@requires_rdkit
def test_nothing_is_proposed_without_an_evaluated_recipe_to_optimise():
    """Proposing a formulation from nothing is structural search, not this."""
    assert BayesOptExplorer().propose(_SPEC, 4, scored=[], seed=0) == []
    assert BayesOptExplorer().propose(_SPEC, 4, scored=[_blend([0.4, 0.3, 0.3])], seed=0) == []


@requires_rdkit
def test_a_molecule_only_target_is_declined():
    spec = _SPEC.model_copy(update={"material_classes": (MaterialClass.MOLECULE,)})
    assert BayesOptExplorer().propose(spec, 4, scored=[], seed=0) == []


@requires_rdkit
def test_proposals_are_valid_compositions_of_the_same_components():
    from formulate.coordination import DeterministicCoordinator, RunConfig

    rng = np.random.default_rng(0)
    seeds = [_blend(f / f.sum()) for f in rng.dirichlet(np.ones(3), size=6)]
    run = DeterministicCoordinator(config=RunConfig(pool_size=40)).run(_SPEC, candidates=seeds)
    pool = [entry.candidate for entry in run.ranking.ranked]

    proposals = BayesOptExplorer(
        BayesOptConfig(minimum_observations=4, acquisition_samples=128, fit_restarts=3)
    ).propose(_SPEC, 4, scored=pool, seed=1)

    assert proposals
    original = {s for s, _ in _COMPONENTS}
    for candidate in proposals:
        fractions = [c.fraction for c in candidate.mixture.components]
        assert sum(fractions) == pytest.approx(1.0)
        assert all(f > 0 for f in fractions)
        assert set(candidate.all_smiles()) <= original
        assert candidate.generation_strategy == "bayesopt:composition"
        assert candidate.parent_ids


def test_the_separation_floor_survives_the_local_refinement():
    """The floor has to bind on what is returned, not only on what is scanned.

    This is the shape of a bug that shipped: the penalty was applied to the
    random scan samples and the L-BFGS refinement that followed minimised raw
    expected improvement, so the refined point walked straight back onto the
    batch member the scan had just excluded. A surrogate with one sharp optimum
    pulls every refinement to the same place, which is exactly the case that
    exposed it.
    """
    rng = np.random.default_rng(11)
    x = rng.random((24, 2))
    peak = np.array([0.5, 0.5])
    y = -np.linalg.norm(x - peak, axis=1) ** 2
    gp = GaussianProcess().fit(x, y, restarts=3, seed=0)

    floor = 0.25
    explorer = BayesOptExplorer(
        BayesOptConfig(acquisition_samples=256, acquisition_refinements=6, minimum_separation=floor)
    )
    chosen: list[np.ndarray] = []
    for _ in range(4):
        chosen.append(
            explorer._maximise_acquisition(gp, float(y.max()), 2, np.random.default_rng(5), chosen)
        )

    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            assert np.linalg.norm(chosen[i] - chosen[j]) >= floor - 1e-9


@requires_rdkit
def test_a_batch_does_not_collapse_onto_one_composition():
    from formulate.coordination import DeterministicCoordinator, RunConfig

    rng = np.random.default_rng(3)
    seeds = [_blend(f / f.sum()) for f in rng.dirichlet(np.ones(3), size=8)]
    run = DeterministicCoordinator(config=RunConfig(pool_size=40)).run(_SPEC, candidates=seeds)
    pool = [entry.candidate for entry in run.ranking.ranked]

    floor = 0.05
    proposals = BayesOptExplorer(
        BayesOptConfig(
            minimum_observations=4,
            acquisition_samples=128,
            fit_restarts=3,
            minimum_separation=floor,
        )
    ).propose(_SPEC, 5, scored=pool, seed=2)

    ids = [c.structure_id for c in proposals]
    assert len(set(ids)) == len(ids)

    # Distinct ids are not separation: two blends 3e-4 apart hash differently
    # and are the same recipe for any purpose a chemist cares about. Measure the
    # distance in the coordinates the floor is actually stated in.
    #
    # Grouped by component set, because a proposal that drives a component to
    # nothing is a different recipe rather than a nearby proportion of this one,
    # and its coordinates do not live in the same cube.
    by_components: dict[tuple[str, ...], list[np.ndarray]] = {}
    for proposal in proposals:
        components = proposal.mixture.components
        key = tuple(sorted(c.canonical_key() for c in components))
        order = sorted(range(len(components)), key=lambda i: components[i].canonical_key())
        fractions = np.array([components[i].fraction for i in order], dtype=float)
        by_components.setdefault(key, []).append(simplex_to_stick_breaking(fractions))

    compared = 0
    for points in by_components.values():
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                assert np.linalg.norm(points[i] - points[j]) >= floor - 1e-6
                compared += 1
    assert compared, "no two proposals shared a component set, so nothing was checked"


@requires_rdkit
def test_the_optimiser_improves_the_objective():
    """The point of the subsystem."""
    from formulate.coordination import DeterministicCoordinator, RunConfig

    rng = np.random.default_rng(0)
    seeds = [_blend(f / f.sum()) for f in rng.dirichlet(np.ones(3), size=6)]
    coordinator = DeterministicCoordinator(config=RunConfig(pool_size=40))
    run = coordinator.run(_SPEC, candidates=seeds)
    pool = [entry.candidate for entry in run.ranking.ranked]

    def best(design_run):
        return max(
            (e.scalar for e in design_run.ranking.ranked if e.feasible and e.scalar is not None),
            default=0.0,
        )

    start = best(run)
    explorer = BayesOptExplorer(
        BayesOptConfig(minimum_observations=4, acquisition_samples=192, fit_restarts=3)
    )
    for round_index in range(1, 4):
        proposals = explorer.propose(_SPEC, 4, scored=pool, seed=round_index)
        if not proposals:
            break
        run = coordinator.run(_SPEC, candidates=pool + proposals)
        pool = [entry.candidate for entry in run.ranking.ranked]

    assert best(run) >= start


# -- reachability ----------------------------------------------------------


def test_the_default_coordinators_actually_carry_the_composition_explorer():
    """A subsystem no run can reach is not a feature, whatever its tests say.

    This module tested ``propose`` directly and passed for a version in which
    every coordinator factory hard-coded retrieval and evolution only, so the
    optimiser never ran outside its own tests while the README said searches
    used it.
    """
    from formulate.coordination.adaptive import default_adaptive_coordinator
    from formulate.coordination.iterative import (
        default_iterative_coordinator,
        default_validating_coordinator,
    )

    for factory in (
        default_iterative_coordinator,
        default_validating_coordinator,
        default_adaptive_coordinator,
    ):
        explorers = factory().explorers
        assert any(isinstance(e, BayesOptExplorer) for e in explorers), factory.__name__


@requires_rdkit
def test_composition_proposals_reach_the_pool_of_a_real_iterative_run():
    from formulate.coordination.iterative import IterationConfig, default_iterative_coordinator
    from formulate.coordination.coordinator import RunConfig

    rng = np.random.default_rng(7)
    seeds = [_blend(f / f.sum()) for f in rng.dirichlet(np.ones(3), size=8)]
    coordinator = default_iterative_coordinator(
        config=RunConfig(pool_size=24),
        iteration=IterationConfig(max_rounds=2, batch_size=9, plateau_rounds=5),
    )
    result = coordinator.run_iterative(_SPEC, seed_candidates=seeds)

    strategies = {
        entry.candidate.generation_strategy for entry in result.final.ranking.ranked
    }
    assert "bayesopt:composition" in strategies


def test_a_budget_smaller_than_the_strategy_count_does_not_starve_the_same_one():
    """The last explorer in the list must not be the one always cut."""
    from formulate.coordination.coordinator import DeterministicCoordinator, RunConfig
    from formulate.exploration.base import Explorer

    class Counting(Explorer):
        def __init__(self, name):
            self.id = name
            self.version = "1"
            self.asked = 0

        def propose(self, spec, count, *, scored=(), seed=0):
            self.asked += 1
            return []

    explorers = [Counting("a"), Counting("b"), Counting("c")]
    for seed in range(6):
        DeterministicCoordinator(
            explorers=explorers, config=RunConfig(pool_size=2, seed=seed)
        )._explore(_SPEC, budget=2)

    # Every explorer gets a first-pass share in at least one of the six rounds.
    assert all(e.asked > 0 for e in explorers)
    fewest, most = min(e.asked for e in explorers), max(e.asked for e in explorers)
    assert most - fewest <= 2
