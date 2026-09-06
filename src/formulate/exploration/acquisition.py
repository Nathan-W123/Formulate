"""Acquisition functions and the multi-objective scalarisation.

Specification section 3 asks for Bayesian optimisation over recipe variables,
and section 4 keeps a Pareto frontier rather than a single scalar.  Those two
have to be reconciled: a Gaussian process models one output, and the ranking
is over several.

ParEGO reconciles them by drawing a fresh random weighting on every proposal
and optimising a scalarisation of the objectives under it.  Over a batch the
weights sweep the frontier, so the proposals spread along it rather than
piling onto whichever corner one fixed weighting happens to favour.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def expected_improvement(
    mean: np.ndarray, sigma: np.ndarray, best: float, exploration: float = 0.01
) -> np.ndarray:
    """Expected improvement over ``best``, for a maximised objective.

    Where the posterior is certain the improvement is deterministic, so the
    formula degenerates; those points are handled separately rather than
    dividing by a standard deviation of zero.
    """
    mean = np.atleast_1d(mean)
    sigma = np.atleast_1d(sigma)
    improvement = mean - best - exploration

    out = np.zeros_like(improvement)
    confident = sigma < 1e-12
    out[confident] = np.maximum(improvement[confident], 0.0)

    uncertain = ~confident
    if uncertain.any():
        z = improvement[uncertain] / sigma[uncertain]
        out[uncertain] = improvement[uncertain] * norm.cdf(z) + sigma[uncertain] * norm.pdf(z)
    return out


def augmented_tchebycheff(
    objectives: np.ndarray, weights: np.ndarray, rho: float = 0.05
) -> np.ndarray:
    """Scalarise several maximised objectives under one weighting.

    s(y) = min_i (w_i y_i) + rho * sum_i (w_i y_i)

    The minimum term is what makes the scalarisation able to reach points on a
    non-convex part of the frontier, which a plain weighted sum cannot. The
    small linear term breaks the ties the minimum leaves behind, which would
    otherwise make whole regions of the frontier indistinguishable.
    """
    objectives = np.atleast_2d(objectives)
    weighted = objectives * weights[None, :]
    return weighted.min(axis=1) + rho * weighted.sum(axis=1)


def random_simplex_weights(count: int, rng: np.random.Generator) -> np.ndarray:
    """A weighting drawn uniformly from the simplex.

    Drawn from a Dirichlet rather than by normalising uniforms: normalising
    concentrates the draws near the centre and the frontier's extremes then
    never get their turn.
    """
    return rng.dirichlet(np.ones(count))


def stick_breaking_to_simplex(u: np.ndarray) -> np.ndarray:
    """Map the unit cube [0,1]^(K-1) onto the K-simplex.

    Optimising fractions directly cannot work: every proposal would have to be
    repaired back onto the constraint surface, and the repair would undo
    whatever the optimiser was trying to do. Stick-breaking makes every point
    of the cube a valid composition by construction, so the optimiser works in
    a space where the constraint cannot be violated.
    """
    u = np.clip(np.atleast_1d(np.asarray(u, dtype=float)), 0.0, 1.0)
    fractions = np.empty(u.size + 1)
    remaining = 1.0
    for index, value in enumerate(u):
        piece = remaining * value
        fractions[index] = piece
        remaining -= piece
    fractions[-1] = remaining
    return fractions


def simplex_to_stick_breaking(fractions: np.ndarray) -> np.ndarray:
    """Invert :func:`stick_breaking_to_simplex`.

    Needed so that a composition already observed can be placed back into the
    optimiser's coordinates and used as training data.
    """
    fractions = np.asarray(fractions, dtype=float)
    u = np.empty(fractions.size - 1)
    remaining = 1.0
    for index in range(fractions.size - 1):
        if remaining <= 1e-12:
            u[index] = 0.0
            continue
        u[index] = np.clip(fractions[index] / remaining, 0.0, 1.0)
        remaining -= fractions[index]
    return u


def sobol_like_design(count: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    """A space-filling design on the unit cube.

    A Latin hypercube rather than independent uniforms: with the handful of
    points a cold start can afford, independent draws leave visible gaps, and
    the surrogate's first lengthscale estimate comes from whatever it saw.
    """
    if count <= 0 or dimensions <= 0:
        return np.zeros((0, max(dimensions, 0)))
    design = np.empty((count, dimensions))
    for axis in range(dimensions):
        strata = (np.arange(count) + rng.random(count)) / count
        design[:, axis] = rng.permutation(strata)
    return design
