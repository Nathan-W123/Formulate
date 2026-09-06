"""Bayesian optimisation over formulation composition.

Specification section 3, "Optimization: Bayesian/black-box optimization over
continuous recipe variables such as ratios, molecular weight, crosslink
density, and process conditions."

This explorer works on the ratios.  Given a formulation whose components are
fixed, the question of what proportions to use is exactly the setting Bayesian
optimisation is for: the objective is expensive relative to the surrogate,
smooth in the fractions, and continuous.  Structural search remains the
evolutionary explorer's job; the two answer different questions and are kept
apart for that reason.

The compositions live on a simplex, which is handled by optimising in
stick-breaking coordinates rather than by repairing proposals afterwards.
Repair would fight the optimiser: every step it took off the constraint
surface would be undone before the objective saw it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from formulate.core.candidate import Candidate, MaterialClass, MixtureComponent
from formulate.core.provenance import ProvenanceKind, ProvenanceRecord
from formulate.targets.spec import TargetSpec

from .acquisition import (
    augmented_tchebycheff,
    expected_improvement,
    random_simplex_weights,
    simplex_to_stick_breaking,
    sobol_like_design,
    stick_breaking_to_simplex,
)
from .base import Explorer
from .gp import GaussianProcess


@dataclass(frozen=True, slots=True)
class BayesOptConfig:
    """Knobs for the composition optimiser."""

    #: Observations of one recipe template before the surrogate is trusted.
    #: Below this the explorer lays down a space-filling design instead: a
    #: Gaussian process fitted to three points mostly reports its prior.
    minimum_observations: int = 5
    #: Restarts when fitting kernel hyperparameters.
    fit_restarts: int = 6
    #: Random points scanned before the acquisition is refined locally.
    acquisition_samples: int = 512
    #: Local refinements from the best scanned points.
    acquisition_refinements: int = 3
    #: Exploration margin in the expected-improvement formula.
    exploration: float = 0.01
    #: Smallest separation, in cube coordinates, between two proposals in a
    #: batch. Without it every member of a batch collapses onto one argmax.
    minimum_separation: float = 0.02


class BayesOptExplorer(Explorer):
    """Proposes new component ratios for formulations already being evaluated."""

    id = "bayesopt"
    version = "1"

    def __init__(self, config: BayesOptConfig | None = None) -> None:
        self.config = config or BayesOptConfig()

    def is_available(self) -> bool:
        return True

    def propose(
        self,
        spec: TargetSpec,
        count: int,
        *,
        scored: Sequence[Candidate] = (),
        seed: int = 0,
    ) -> list[Candidate]:
        if count <= 0 or MaterialClass.MIXTURE not in spec.material_classes:
            return []

        templates = self._group_by_template(scored)
        if not templates:
            # Nothing to optimise the composition *of*. Proposing a formulation
            # from nothing would be structural search, which belongs to the
            # evolutionary and retrieval explorers.
            return []

        rng = np.random.default_rng(seed)
        # Work on the template with the most observations: a surrogate fitted
        # to the best-sampled recipe is the one most likely to be right.
        key = max(templates, key=lambda k: len(templates[k]))
        observations = templates[key]
        template = observations[0][0]

        axes = [p for p in spec.properties]
        x, y = self._observation_matrices(observations, axes)

        proposals: list[np.ndarray] = []
        if x is None or x.shape[0] < self.config.minimum_observations:
            proposals = list(
                sobol_like_design(count, self._dimensions(template), rng)
            )
        else:
            proposals = self._optimise(x, y, count, rng)

        out: list[Candidate] = []
        seen = {c.structure_id for c in scored}
        for u in proposals:
            candidate = self._build(template, u, spec)
            if candidate is None or candidate.structure_id in seen:
                continue
            seen.add(candidate.structure_id)
            out.append(candidate)
            if len(out) >= count:
                break
        return out

    # -- observations ------------------------------------------------------

    def _group_by_template(
        self, scored: Sequence[Candidate]
    ) -> dict[tuple[str, ...], list[tuple[Candidate, np.ndarray]]]:
        """Group evaluated mixtures by which components they contain.

        Two recipes with the same components at different proportions are
        points on one surface; two with different components are not, and
        pooling them would ask the surrogate to interpolate across a
        discontinuity.
        """
        groups: dict[tuple[str, ...], list[tuple[Candidate, np.ndarray]]] = {}
        for candidate in scored:
            if candidate.mixture is None or candidate.results is None:
                continue
            components = candidate.mixture.components
            if len(components) < 2:
                continue
            key = tuple(sorted(c.canonical_key() for c in components))
            order = sorted(range(len(components)), key=lambda i: components[i].canonical_key())
            fractions = np.array([components[i].fraction for i in order], dtype=float)
            groups.setdefault(key, []).append((candidate, fractions))
        return groups

    def _observation_matrices(
        self, observations: list[tuple[Candidate, np.ndarray]], axes: list[str]
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        rows: list[np.ndarray] = []
        values: list[list[float]] = []
        for candidate, fractions in observations:
            vector = candidate.results.objective_vector  # type: ignore[union-attr]
            usable = [a for a in axes if a in vector]
            if not usable:
                continue
            rows.append(simplex_to_stick_breaking(fractions))
            values.append([vector.get(a, 0.0) for a in axes])
        if not rows:
            return None, None
        return np.vstack(rows), np.array(values, dtype=float)

    def _dimensions(self, template: Candidate) -> int:
        return len(template.mixture.components) - 1  # type: ignore[union-attr]

    # -- the optimisation --------------------------------------------------

    def _optimise(
        self, x: np.ndarray, y: np.ndarray, count: int, rng: np.random.Generator
    ) -> list[np.ndarray]:
        """One ParEGO round per proposal.

        A fresh weighting per proposal is what spreads a batch along the
        frontier. Reusing one weighting would fit the same surrogate every time
        and return the same argmax, so the batch would be one point repeated.
        """
        config = self.config
        proposals: list[np.ndarray] = []
        dimensions = x.shape[1]

        for _ in range(count):
            weights = random_simplex_weights(y.shape[1], rng)
            scalar = augmented_tchebycheff(y, weights)
            try:
                gp = GaussianProcess().fit(
                    x, scalar, restarts=config.fit_restarts, seed=int(rng.integers(1 << 30))
                )
            except Exception:
                # A surrogate that will not fit is not a reason to stop: fall
                # back to a space-filling point, which is what the cold start
                # would have done anyway.
                proposals.append(rng.random(dimensions))
                continue

            best = float(scalar.max())
            candidate_u = self._maximise_acquisition(gp, best, dimensions, rng, proposals)
            proposals.append(candidate_u)
        return proposals

    def _maximise_acquisition(
        self,
        gp: GaussianProcess,
        best: float,
        dimensions: int,
        rng: np.random.Generator,
        already: list[np.ndarray],
    ) -> np.ndarray:
        """Scan the cube, then refine the best points locally."""
        from scipy.optimize import minimize

        config = self.config
        samples = rng.random((config.acquisition_samples, dimensions))
        mean, sigma = gp.predict(samples)
        scores = expected_improvement(mean, sigma, best, config.exploration)

        # Push the acquisition down near points already chosen in this batch so
        # the next member cannot land on the same argmax.
        for chosen in already:
            distance = np.linalg.norm(samples - chosen[None, :], axis=1)
            scores = np.where(distance < config.minimum_separation, 0.0, scores)

        order = np.argsort(-scores)[: config.acquisition_refinements]
        best_u, best_score = samples[order[0]], float(scores[order[0]])

        for index in order:
            def negative(u: np.ndarray) -> float:
                m, s = gp.predict(np.atleast_2d(u))
                return -float(expected_improvement(m, s, best, config.exploration)[0])

            try:
                result = minimize(
                    negative,
                    samples[index],
                    method="L-BFGS-B",
                    bounds=[(0.0, 1.0)] * dimensions,
                    options={"maxiter": 60},
                )
            except Exception:
                continue
            if np.isfinite(result.fun) and -result.fun > best_score:
                best_score, best_u = -float(result.fun), np.clip(result.x, 0.0, 1.0)
        return best_u

    # -- construction ------------------------------------------------------

    def _build(
        self, template: Candidate, u: np.ndarray, spec: TargetSpec
    ) -> Candidate | None:
        """Turn cube coordinates into a candidate with those proportions."""
        mixture = template.mixture
        if mixture is None:
            return None

        components = list(mixture.components)
        order = sorted(range(len(components)), key=lambda i: components[i].canonical_key())
        fractions = stick_breaking_to_simplex(np.asarray(u, dtype=float))

        updated: list[MixtureComponent] = list(components)
        for position, index in enumerate(order):
            updated[index] = components[index].model_copy(
                update={"fraction": float(fractions[position])}
            )

        # A component driven to nothing is a different recipe, not a proportion
        # of this one, so it is dropped and the rest renormalised.
        kept = [c for c in updated if c.fraction > 1e-6]
        if len(kept) < 2:
            return None
        total = sum(c.fraction for c in kept)
        kept = [c.model_copy(update={"fraction": c.fraction / total}) for c in kept]

        try:
            new_mixture = mixture.model_copy(update={"components": tuple(kept)})
            return template.model_copy(
                update={
                    "mixture": new_mixture,
                    "conditions": spec.conditions,
                    "results": None,
                    "label": "",
                    "generation_strategy": f"{self.id}:composition",
                    "parent_ids": (template.candidate_id,),
                    "provenance": ProvenanceRecord(
                        kind=ProvenanceKind.GENERATION,
                        producer=self.id,
                        producer_version=self.version,
                        input_ids=(template.candidate_id,),
                        parameters={"strategy": "expected improvement over composition"},
                    ),
                }
            )
        except Exception:
            # The schema validates fraction sums; a composition it rejects is
            # simply not proposed.
            return None
