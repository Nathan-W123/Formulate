"""Pareto dominance, non-dominated sorting and hypervolume.

Specification section 4: "Combine utilities with explicit weights only for a
scalar baseline; maintain a Pareto frontier for conflicting objectives."

The frontier is the primary ranking.  A weighted sum answers "best under one
particular exchange rate between properties"; the frontier answers "which
candidates are not beaten on every property at once", which is the question a
multi-objective design request actually asks.

All objectives here are utilities in [0, 1] and are maximised.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

#: Two utilities within this distance are treated as equal, so that
#: floating-point noise does not manufacture dominance.
DOMINANCE_EPSILON = 1e-9


def dominates(a: Sequence[float], b: Sequence[float], *, epsilon: float = DOMINANCE_EPSILON) -> bool:
    """True when ``a`` is at least as good as ``b`` everywhere and better somewhere."""
    if len(a) != len(b):
        raise ValueError("Objective vectors must have the same length to be compared.")
    at_least_as_good = all(x >= y - epsilon for x, y in zip(a, b))
    strictly_better = any(x > y + epsilon for x, y in zip(a, b))
    return at_least_as_good and strictly_better


def non_dominated_sort(
    objectives: Sequence[Sequence[float]], *, epsilon: float = DOMINANCE_EPSILON
) -> list[int]:
    """Assign each row a front index: 0 for non-dominated, 1 for the next, and so on.

    This is the fast non-dominated sort of NSGA-II (Deb et al., 2002).
    """
    count = len(objectives)
    if count == 0:
        return []

    dominated_by: list[list[int]] = [[] for _ in range(count)]
    domination_count = [0] * count
    fronts: list[list[int]] = [[]]

    for i in range(count):
        for j in range(i + 1, count):
            if dominates(objectives[i], objectives[j], epsilon=epsilon):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif dominates(objectives[j], objectives[i], epsilon=epsilon):
                dominated_by[j].append(i)
                domination_count[i] += 1

    for i in range(count):
        if domination_count[i] == 0:
            fronts[0].append(i)

    assignment = [0] * count
    current = 0
    while current < len(fronts) and fronts[current]:
        nxt: list[int] = []
        for i in fronts[current]:
            assignment[i] = current
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    nxt.append(j)
        fronts.append(nxt)
        current += 1
    return assignment


def crowding_distance(objectives: Sequence[Sequence[float]]) -> list[float]:
    """NSGA-II crowding distance within one front.

    Boundary solutions get infinite distance so that the extremes of a
    trade-off are never crowded out.
    """
    count = len(objectives)
    if count == 0:
        return []
    if count <= 2:
        return [float("inf")] * count

    matrix = np.asarray(objectives, dtype=float)
    distance = np.zeros(count)
    for axis in range(matrix.shape[1]):
        order = np.argsort(matrix[:, axis])
        column = matrix[order, axis]
        spread = column[-1] - column[0]
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        if spread <= 0:
            continue
        distance[order[1:-1]] += (column[2:] - column[:-2]) / spread
    return [float(d) for d in distance]


def hypervolume(points: Sequence[Sequence[float]], reference: Sequence[float] | None = None) -> float:
    """Exact hypervolume dominated by ``points``, maximising every objective.

    ``reference`` defaults to the origin, which is the natural nadir for
    utilities in [0, 1].  Section 12 uses this as a measure of Pareto
    improvement per evaluation budget.
    """
    if not points:
        return 0.0
    dimensions = len(points[0])
    ref = list(reference) if reference is not None else [0.0] * dimensions
    if any(len(p) != dimensions for p in points):
        raise ValueError("All points must have the same number of objectives.")

    # Convert maximisation-above-reference into minimisation-below-origin.
    transformed = [
        tuple(r - v for v, r in zip(point, ref))
        for point in points
        if all(v >= r for v, r in zip(point, ref))
    ]
    if not transformed:
        return 0.0
    return _hv_min(_filter_non_dominated_min(transformed), [0.0] * dimensions)


def _filter_non_dominated_min(points: Sequence[tuple[float, ...]]) -> list[tuple[float, ...]]:
    """Keep only points not dominated under minimisation."""
    keep: list[tuple[float, ...]] = []
    for i, p in enumerate(points):
        if any(
            i != j
            and all(q <= x for q, x in zip(other, p))
            and any(q < x for q, x in zip(other, p))
            for j, other in enumerate(points)
        ):
            continue
        if p not in keep:
            keep.append(p)
    return keep


def _hv_min(points: Sequence[tuple[float, ...]], reference: Sequence[float]) -> float:
    """Hypervolume under minimisation, by slicing along the first objective."""
    if not points:
        return 0.0
    if len(reference) == 1:
        return max(0.0, reference[0] - min(p[0] for p in points))

    volume = 0.0
    boundaries = sorted({p[0] for p in points})
    for index, lower in enumerate(boundaries):
        upper = boundaries[index + 1] if index + 1 < len(boundaries) else reference[0]
        depth = upper - lower
        if depth <= 0:
            continue
        slab = [p[1:] for p in points if p[0] <= lower]
        if not slab:
            continue
        volume += depth * _hv_min(_filter_non_dominated_min(slab), list(reference[1:]))
    return volume
