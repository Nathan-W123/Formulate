"""Seeding the first formulations, which nothing else would do.

Specification section 3 asks exploration to produce a diverse candidate pool
and keeps exploration separate from evaluation. For mixtures that separation
had a hole in it. Three explorers could each handle a formulation once one
existed and none would make the first one, each deferring to the others in a
circle:

* the reference-database explorer returns molecules and nothing else;
* the evolutionary explorer mutates composition and swaps components, but only
  on a parent that is already a mixture, and says in as many words that seeding
  the first round belongs to the database or optimisation explorers;
* the Bayesian explorer optimises the composition of a recipe template it has
  observations for, and says that proposing a formulation from nothing is
  structural search belonging to the evolutionary and retrieval explorers.

So a target asking for a blend produced an empty pool, silently. Everything
downstream - the UNIFAC phase check, the Hansen blending, the composition
surrogate - was built and tested and had nothing to act on.

This explorer only makes the first move. It picks component *sets* and one
composition apiece; refining that composition is what the Bayesian explorer is
for, and it does it properly with a surrogate. Proposing a dense ladder of
compositions here would spend the budget on a search that is already better
solved one stage later.

Two things it deliberately does not do.

It does not judge whether the components will mix. An explorer that dropped
hexane and water before proposing them would be scoring, which section 3
reserves for evaluation, and the panel already answers that question properly:
modified UNIFAC returns a stability of -15 for that pair against +1.6 for
ethanol and water. Filtering here would move a decision out of the expert that
can justify it and into one that would be guessing.

It does not order pairs by how *different* the components are. That looks like
the right prior for covering solvent space and is the wrong one for a single
phase, because the most dissimilar pairs are exactly the ones that separate.
Ordering instead maximises how many distinct components appear early, which is
a statement about budget rather than about chemistry, and is the only kind of
statement this class is entitled to make.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Iterator, Sequence

from formulate.core.candidate import (
    Candidate,
    ComponentRole,
    FractionBasis,
    MaterialClass,
    MixtureComponent,
    MixtureSpec,
    MoleculeSpec,
    PhaseAssumption,
)
from formulate.core.provenance import ProvenanceKind, ProvenanceRecord
from formulate.targets.spec import TargetSpec

from .base import Explorer
from .database import load_reference_compounds


@dataclass(frozen=True, slots=True)
class MixtureSeedConfig:
    """How the first formulations are shaped."""

    #: Components per blend. Binary by default: there are 1225 pairs among the
    #: fifty reference compounds and 19600 triples, and no budget reaches the
    #: second number, so offering it by default would only mean covering a
    #: smaller fraction of a larger space.
    components_per_blend: int = 2
    #: Compositions proposed for each component set. One is the default because
    #: the Bayesian explorer optimises composition from observations, and a
    #: ladder here would duplicate that with a worse method.
    compositions: tuple[tuple[float, ...], ...] = ((0.5, 0.5),)
    basis: FractionBasis = FractionBasis.VOLUME
    phase_assumption: PhaseAssumption = PhaseAssumption.SINGLE_PHASE
    role: ComponentRole = ComponentRole.SOLVENT
    #: Cap on the component pool. Pairs grow as the square of this.
    max_components: int = 40


class MixtureSeedExplorer(Explorer):
    """Proposes the first formulations, from molecules that are already known."""

    id = "mixture:seed"
    version = "1"

    def __init__(self, config: MixtureSeedConfig | None = None) -> None:
        self.config = config or MixtureSeedConfig()
        for weights in self.config.compositions:
            if len(weights) != self.config.components_per_blend:
                raise ValueError(
                    f"a composition of {len(weights)} fractions cannot describe a blend of "
                    f"{self.config.components_per_blend} components"
                )
            if abs(sum(weights) - 1.0) > 1e-9:
                raise ValueError(f"composition {weights} does not sum to one")

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

        pool = self._component_pool(spec, scored)
        if len(pool) < self.config.components_per_blend:
            return []

        seen = {c.structure_id for c in scored}
        out: list[Candidate] = []
        for combination in self._combinations(pool, seed):
            for weights in self.config.compositions:
                candidate = self._build(combination, weights, spec)
                if candidate.structure_id in seen:
                    continue
                seen.add(candidate.structure_id)
                out.append(candidate)
                if len(out) >= count:
                    return out
        return out

    # -- the component pool ------------------------------------------------

    def _component_pool(
        self, spec: TargetSpec, scored: Sequence[Candidate]
    ) -> list[tuple[str, str]]:
        """(SMILES, label) for each usable component, best-known first.

        Molecules this run has already evaluated come first when there are
        enough of them: blending compounds the panel has scored is better
        founded than blending the reference set blindly, and it is also how a
        mixed molecule-and-mixture target gets formulations built out of its
        own winners rather than out of a fixed list.
        """
        from_run: list[tuple[str, str]] = []
        for candidate in scored:
            if candidate.material_class is not MaterialClass.MOLECULE:
                continue
            if candidate.molecule is None or candidate.results is None:
                continue
            from_run.append((candidate.molecule.smiles, candidate.label or ""))

        pool = from_run if len(from_run) >= self.config.components_per_blend else [
            (record["smiles"], record["name"]) for record in load_reference_compounds()
        ]

        allowed = set(spec.structural.allowed_elements) | {"H"}
        forbidden = set(spec.structural.forbidden_elements)
        if spec.structural.allowed_elements or forbidden:
            pool = [entry for entry in pool if self._elements_pass(entry[0], allowed, forbidden)]

        deduped: list[tuple[str, str]] = []
        seen_smiles: set[str] = set()
        for smiles, label in pool:
            if smiles in seen_smiles:
                continue
            seen_smiles.add(smiles)
            deduped.append((smiles, label))
        return deduped[: self.config.max_components]

    @staticmethod
    def _elements_pass(smiles: str, allowed: set[str], forbidden: set[str]) -> bool:
        from formulate import chem

        if not chem.rdkit_available():
            return True
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
        if forbidden & elements:
            return False
        if allowed != {"H"} and not elements <= allowed:
            return False
        return True

    # -- which component sets, in which order ------------------------------

    def _combinations(
        self, pool: Sequence[tuple[str, str]], seed: int
    ) -> Iterator[tuple[tuple[str, str], ...]]:
        """Component sets in rounds, no component repeating inside a round.

        Enumerating pairs in index order spends the whole budget on the first
        compound: (0,1), (0,2), (0,3) and so on. Sorting by the largest index
        instead is barely better - it exhausts every pair within a growing
        prefix, so the first ten proposals were ten pairs drawn from five
        compounds.

        Dealing them into rounds fixes it. Each pass takes combinations whose
        components are all still unused in that round and defers the rest, so a
        round is a matching over the pool and ten binary proposals touch twenty
        distinct compounds. Deferred combinations form the next round, and
        every combination is emitted exactly once, so a large budget still
        enumerates the whole space. The shuffle is seeded, which keeps a run
        reproducible without letting the pool's own order decide what gets
        tried first.
        """
        rng = random.Random(seed)
        remaining = list(
            itertools.combinations(range(len(pool)), self.config.components_per_blend)
        )
        rng.shuffle(remaining)
        while remaining:
            used: set[int] = set()
            deferred: list[tuple[int, ...]] = []
            for combination in remaining:
                if used.isdisjoint(combination):
                    used.update(combination)
                    yield tuple(pool[i] for i in combination)
                else:
                    deferred.append(combination)
            remaining = deferred

    # -- building ----------------------------------------------------------

    def _build(
        self,
        combination: Sequence[tuple[str, str]],
        weights: Sequence[float],
        spec: TargetSpec,
    ) -> Candidate:
        components = tuple(
            MixtureComponent(
                role=self.config.role,
                fraction=fraction,
                molecule=MoleculeSpec(smiles=smiles),
            )
            for (smiles, _), fraction in zip(combination, weights)
        )
        labels = [label or smiles for smiles, label in combination]
        percentages = ":".join(f"{f * 100:.0f}" for f in weights)
        return Candidate(
            material_class=MaterialClass.MIXTURE,
            mixture=MixtureSpec(
                components=components,
                basis=self.config.basis,
                phase_assumption=self.config.phase_assumption,
            ),
            conditions=spec.conditions,
            generation_strategy=self.id,
            label=f"{' / '.join(labels)} {percentages}",
            provenance=ProvenanceRecord(
                kind=ProvenanceKind.GENERATION,
                producer=self.id,
                producer_version=self.version,
                parameters={
                    "components": [smiles for smiles, _ in combination],
                    "fractions": list(weights),
                    "basis": self.config.basis.value,
                },
            ),
        )
