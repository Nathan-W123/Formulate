"""Structural diversity.

Specification section 3 reserves exploration budget for distinct chemical
regions and warns against "one family monopolizing the population"; section 12
measures "chemical diversity per evaluation budget".

Diversity here is structural (Tanimoto distance over Morgan fingerprints), not
property diversity.  Two molecules with similar predicted properties may be
chemically unrelated, and for a design report that difference matters: it is
the difference between one idea and two.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from formulate.core.candidate import Candidate


def distance_matrix(candidates: Sequence[Candidate], *, radius: int = 2, n_bits: int = 2048):
    """Pairwise structural distance (1 - Tanimoto) as a square array.

    Pairs whose structures cannot be fingerprinted are recorded as ``nan``
    rather than as maximally distant, so an unparseable structure cannot pose
    as a diverse one.
    """
    from formulate import chem

    count = len(candidates)
    matrix = np.full((count, count), np.nan)
    if count == 0 or not chem.rdkit_available():
        return matrix

    smiles = [c.primary_smiles for c in candidates]
    for i in range(count):
        matrix[i, i] = 0.0
        if smiles[i] is None:
            continue
        sims = chem.bulk_tanimoto(
            smiles[i],
            [s if s is not None else "" for s in smiles],
            radius=radius,
            n_bits=n_bits,
        )
        for j, sim in enumerate(sims):
            if i == j or sim is None:
                continue
            matrix[i, j] = 1.0 - sim
    return matrix


def mean_pairwise_distance(candidates: Sequence[Candidate]) -> float | None:
    """Mean structural distance across the pool, or ``None`` if undefined."""
    if len(candidates) < 2:
        return None
    matrix = distance_matrix(candidates)
    upper = matrix[np.triu_indices(len(candidates), k=1)]
    finite = upper[~np.isnan(upper)]
    return float(finite.mean()) if finite.size else None


def max_min_selection(
    candidates: Sequence[Candidate],
    count: int,
    *,
    seed_index: int = 0,
    order: Sequence[int] | None = None,
) -> list[int]:
    """Pick ``count`` indices that are structurally spread out.

    Standard MaxMin: start from ``seed_index`` (normally the best-ranked
    candidate) and repeatedly add whichever remaining candidate is furthest
    from everything already chosen.  ``order`` breaks ties by preferring
    earlier entries, so a tie between equally distant candidates goes to the
    better-ranked one rather than to whichever numpy happened to list first.
    """
    total = len(candidates)
    if count <= 0 or total == 0:
        return []
    if count >= total:
        return list(range(total))

    matrix = distance_matrix(candidates)
    if np.all(np.isnan(matrix)):
        # Without fingerprints there is no structural signal; fall back to
        # rank order rather than pretending the selection is diverse.
        return list(order[:count]) if order is not None else list(range(count))

    rank_of = {index: position for position, index in enumerate(order or range(total))}

    chosen = [seed_index]
    remaining = [i for i in range(total) if i != seed_index]

    while len(chosen) < count and remaining:
        best_index: int | None = None
        best_key: tuple[float, int] | None = None
        for i in remaining:
            distances = [matrix[i, j] for j in chosen if not np.isnan(matrix[i, j])]
            nearest = min(distances) if distances else 0.0
            key = (-nearest, rank_of.get(i, i))
            if best_key is None or key < best_key:
                best_key, best_index = key, i
        if best_index is None:
            break
        chosen.append(best_index)
        remaining.remove(best_index)
    return chosen
