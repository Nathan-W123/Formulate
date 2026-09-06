"""Evolutionary search over structures and recipes.

Specification section 3: "Evolutionary search: mutate/crossover valid
structures or recipes; retain Pareto-competitive and diverse parents", and
"Diversity: reserve exploration budget for structurally/chemically distinct
regions; prevent one family from monopolizing the population."

This explorer proposes and nothing more.  It reads the scores already attached
to evaluated candidates in order to choose parents, but it never scores, ranks
or filters - that separation is a section 11 invariant, and it is what keeps a
generator from quietly becoming a validator.

Parent selection reuses the non-dominated sort and crowding distance in
:mod:`formulate.ranking.pareto` rather than carrying a second implementation
that could disagree with the ranker about what dominates what.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from formulate.core.candidate import (
    Candidate,
    ComponentRole,
    MaterialClass,
    MixtureComponent,
    MixtureSpec,
    MoleculeSpec,
    MonomerRole,
    MonomerUnit,
    PolymerSpec,
)
from formulate.core.provenance import ProvenanceKind, ProvenanceRecord
from formulate.core.quantity import Quantity
from formulate.ranking.pareto import crowding_distance, non_dominated_sort
from formulate.targets.spec import TargetSpec

from .base import Explorer
from .operators import DEFAULT_OPERATORS, MutationOperator, brics_crossover


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    """Knobs for the evolutionary search."""

    #: Probability that an offspring comes from mutation rather than crossover.
    mutation_probability: float = 0.7
    #: Candidates compared in each selection tournament.
    tournament_size: int = 3
    #: Offspring allowed to share one Bemis-Murcko scaffold in a single round.
    #: This is the mechanism behind section 3's requirement that no one family
    #: monopolises the population; without it a single good scaffold takes over
    #: within a couple of rounds and the search stops exploring.
    max_per_scaffold: int = 3
    #: Fraction of ranked parents eligible for selection at all.
    elite_fraction: float = 0.5
    #: Proposals requested from a single operator before moving on.
    proposals_per_operator: int = 3
    operators: tuple[MutationOperator, ...] = field(default=DEFAULT_OPERATORS)
    #: Relative amount of composition perturbation applied to a mixture.
    fraction_step: float = 0.15


class EvolutionaryExplorer(Explorer):
    """Mutates and recombines evaluated candidates into new proposals."""

    id = "evolutionary"
    version = "1"

    def __init__(self, config: EvolutionConfig | None = None) -> None:
        self.config = config or EvolutionConfig()

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "RDKit is required to mutate structures"

    # -- proposal ----------------------------------------------------------

    def propose(
        self,
        spec: TargetSpec,
        count: int,
        *,
        scored: Sequence[Candidate] = (),
        seed: int = 0,
    ) -> list[Candidate]:
        if count <= 0 or not self.is_available():
            return []

        parents = self._rank_parents(spec, scored)
        if not parents:
            # Nothing evaluated yet: an evolutionary step has no parents to work
            # from. Returning empty lets the database or optimisation explorers
            # seed the first round rather than inventing an arbitrary origin.
            return []

        rng = random.Random(seed)
        seen = {c.structure_id for c in scored}
        scaffold_counts: dict[str, int] = {}
        out: list[Candidate] = []

        # Bounded attempts: operators frequently return nothing for a given
        # parent, and an unbounded loop would spin on an unmutatable pool.
        attempts = 0
        max_attempts = max(40, count * 12)

        while len(out) < count and attempts < max_attempts:
            attempts += 1
            parent = self._tournament(parents, rng)
            use_crossover = (
                len(parents) > 1 and rng.random() > self.config.mutation_probability
            )
            if use_crossover:
                other = self._tournament([p for p in parents if p is not parent] or parents, rng)
                children = self._crossover(parent, other, rng)
                lineage = [parent, other]
            else:
                children = self._mutate(parent, rng)
                lineage = [parent]

            for child in children:
                if len(out) >= count:
                    break
                if child.structure_id in seen:
                    continue
                scaffold = self._scaffold_key(child)
                if scaffold_counts.get(scaffold, 0) >= self.config.max_per_scaffold:
                    continue
                seen.add(child.structure_id)
                scaffold_counts[scaffold] = scaffold_counts.get(scaffold, 0) + 1
                out.append(self._finalise(child, spec, lineage))
        return out

    # -- parent selection --------------------------------------------------

    def _rank_parents(self, spec: TargetSpec, scored: Sequence[Candidate]) -> list[Candidate]:
        """Order evaluated candidates by Pareto front, then crowding distance.

        Feasible candidates come first: breeding from a candidate that breaches
        a hard constraint wastes budget on a region already known to be ruled
        out. Crowding distance breaks ties toward the sparse parts of the
        frontier, which is where new ground is.
        """
        evaluated = [c for c in scored if c.results is not None]
        if not evaluated:
            return []
        # Stable order regardless of the caller's ordering, so a run reproduces.
        evaluated.sort(key=lambda c: c.candidate_id)

        feasible = [c for c in evaluated if c.results.feasible]
        pool = feasible or evaluated

        axes = [p for p in spec.properties if all(p in c.results.objective_vector for c in pool)]
        if not axes:
            return pool

        vectors = [[c.results.objective_vector.get(a, 0.0) for a in axes] for c in pool]
        fronts = non_dominated_sort(vectors)
        crowding = [0.0] * len(pool)
        for front_index in set(fronts):
            members = [i for i, f in enumerate(fronts) if f == front_index]
            distances = crowding_distance([vectors[i] for i in members])
            for position, i in enumerate(members):
                crowding[i] = distances[position]

        order = sorted(
            range(len(pool)),
            key=lambda i: (fronts[i], -_finite(crowding[i]), pool[i].candidate_id),
        )
        ranked = [pool[i] for i in order]
        keep = max(2, int(len(ranked) * self.config.elite_fraction))
        return ranked[:keep]

    def _tournament(self, parents: Sequence[Candidate], rng: random.Random) -> Candidate:
        """Pick the best of a small random sample.

        ``parents`` arrives already sorted best-first, so the winner is simply
        the lowest index drawn.
        """
        size = min(self.config.tournament_size, len(parents))
        picks = rng.sample(range(len(parents)), size)
        return parents[min(picks)]

    # -- variation ---------------------------------------------------------

    def _mutate(self, parent: Candidate, rng: random.Random) -> list[Candidate]:
        if parent.material_class is MaterialClass.MOLECULE:
            return self._mutate_molecule(parent, rng)
        if parent.material_class is MaterialClass.POLYMER:
            return self._mutate_polymer(parent, rng)
        if parent.mixture is not None:
            return self._mutate_mixture(parent, rng)
        return []

    def _mutate_molecule(self, parent: Candidate, rng: random.Random) -> list[Candidate]:
        smiles = parent.molecule.smiles  # type: ignore[union-attr]
        operators = list(self.config.operators)
        rng.shuffle(operators)
        for operator in operators:
            proposals = operator.propose(smiles, rng, self.config.proposals_per_operator)
            if proposals:
                return [
                    parent.model_copy(
                        update={"molecule": MoleculeSpec(smiles=s), "results": None, "label": ""}
                    )
                    for s in proposals
                ]
        return []

    def _mutate_polymer(self, parent: Candidate, rng: random.Random) -> list[Candidate]:
        """Alter monomer identity, comonomer fraction, chain length or architecture.

        Specification section 3 names exactly these axes for polymer mutation.
        """
        polymer = parent.polymer
        if polymer is None:
            return []
        out: list[Candidate] = []

        chain = [m for m in polymer.monomers if m.role is not MonomerRole.END_GROUP]
        others = [m for m in polymer.monomers if m.role is MonomerRole.END_GROUP]

        # Monomer identity: mutate one repeat unit's structure.
        target = rng.randrange(len(chain))
        operators = list(self.config.operators)
        rng.shuffle(operators)
        for operator in operators:
            proposals = operator.propose(chain[target].smiles, rng, 2)
            if not proposals:
                continue
            for smiles in proposals:
                monomers = list(chain)
                monomers[target] = MonomerUnit(
                    smiles=smiles,
                    mole_fraction=chain[target].mole_fraction,
                    role=chain[target].role,
                )
                out.append(
                    self._respec_polymer(parent, polymer, tuple(monomers) + tuple(others))
                )
            break

        # Comonomer fraction: shift composition between two repeat units.
        if len(chain) >= 2:
            i, j = rng.sample(range(len(chain)), 2)
            step = min(self.config.fraction_step, chain[i].mole_fraction)
            if step > 1e-9:
                monomers = list(chain)
                monomers[i] = _with_fraction(monomers[i], monomers[i].mole_fraction - step)
                monomers[j] = _with_fraction(monomers[j], monomers[j].mole_fraction + step)
                out.append(
                    self._respec_polymer(parent, polymer, tuple(monomers) + tuple(others))
                )

        # Chain length target.
        if polymer.number_average_molar_mass is not None:
            current = polymer.number_average_molar_mass.to("g/mol").value
            scale = rng.choice([0.5, 0.75, 1.5, 2.0])
            updated = polymer.model_copy(
                update={
                    "number_average_molar_mass": Quantity(
                        value=max(200.0, current * scale), unit="g/mol"
                    )
                }
            )
            out.append(
                parent.model_copy(update={"polymer": updated, "results": None, "label": ""})
            )
        return out

    def _respec_polymer(
        self, parent: Candidate, polymer: PolymerSpec, monomers: tuple[MonomerUnit, ...]
    ) -> Candidate:
        updated = polymer.model_copy(update={"monomers": monomers})
        return parent.model_copy(update={"polymer": updated, "results": None, "label": ""})

    def _mutate_mixture(self, parent: Candidate, rng: random.Random) -> list[Candidate]:
        """Perturb composition or swap a component.

        Fractions are renormalised to sum to one because MixtureSpec validates
        that invariant; a mutation that broke it would raise rather than
        produce a candidate.
        """
        mixture = parent.mixture
        if mixture is None or not mixture.components:
            return []
        out: list[Candidate] = []
        components = list(mixture.components)

        # Composition shift between two components.
        if len(components) >= 2:
            i, j = rng.sample(range(len(components)), 2)
            step = min(self.config.fraction_step, components[i].fraction)
            if step > 1e-9:
                shifted = list(components)
                shifted[i] = components[i].model_copy(
                    update={"fraction": components[i].fraction - step}
                )
                shifted[j] = components[j].model_copy(
                    update={"fraction": components[j].fraction + step}
                )
                out.append(self._respec_mixture(parent, mixture, _renormalised(shifted)))

        # Structural mutation of one component.
        index = rng.randrange(len(components))
        component = components[index]
        if component.molecule is not None:
            operators = list(self.config.operators)
            rng.shuffle(operators)
            for operator in operators:
                proposals = operator.propose(component.molecule.smiles, rng, 1)
                if proposals:
                    swapped = list(components)
                    swapped[index] = component.model_copy(
                        update={"molecule": MoleculeSpec(smiles=proposals[0])}
                    )
                    out.append(self._respec_mixture(parent, mixture, tuple(swapped)))
                    break
        return out

    def _respec_mixture(
        self, parent: Candidate, mixture: MixtureSpec, components: tuple[MixtureComponent, ...]
    ) -> Candidate:
        updated = mixture.model_copy(update={"components": components})
        return parent.model_copy(update={"mixture": updated, "results": None, "label": ""})

    def _crossover(
        self, parent_a: Candidate, parent_b: Candidate, rng: random.Random
    ) -> list[Candidate]:
        if parent_a.molecule is None or parent_b.molecule is None:
            # Recipe crossover: take components from both parents.
            return self._recipe_crossover(parent_a, parent_b, rng)
        proposals = brics_crossover(
            parent_a.molecule.smiles, parent_b.molecule.smiles, rng, limit=3
        )
        return [
            parent_a.model_copy(
                update={"molecule": MoleculeSpec(smiles=s), "results": None, "label": ""}
            )
            for s in proposals
        ]

    def _recipe_crossover(
        self, parent_a: Candidate, parent_b: Candidate, rng: random.Random
    ) -> list[Candidate]:
        if parent_a.mixture is None or parent_b.mixture is None:
            return []
        pool = list(parent_a.mixture.components) + list(parent_b.mixture.components)
        wanted = len(parent_a.mixture.components)
        by_key: dict[str, MixtureComponent] = {}
        for component in pool:
            by_key.setdefault(component.canonical_key(), component)
        unique = sorted(by_key.values(), key=lambda c: c.canonical_key())
        if len(unique) < wanted:
            return []
        chosen = rng.sample(unique, wanted)
        return [self._respec_mixture(parent_a, parent_a.mixture, _renormalised(chosen))]

    # -- bookkeeping -------------------------------------------------------

    def _scaffold_key(self, candidate: Candidate) -> str:
        """Bemis-Murcko scaffold, used to cap one family's share of a round."""
        from formulate import chem

        smiles = candidate.primary_smiles
        if smiles is None or not chem.rdkit_available():
            return candidate.material_class.value
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold

            mol = chem.mol_from_smiles(smiles)
            if mol is None:
                return smiles
            scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
            # Acyclic molecules have an empty scaffold; group them by formula so
            # they are not all treated as one enormous family.
            return scaffold or f"acyclic:{chem.formula(smiles)}"
        except Exception:
            return smiles

    def _finalise(
        self, child: Candidate, spec: TargetSpec, lineage: Sequence[Candidate]
    ) -> Candidate:
        parent_ids = tuple(sorted({p.candidate_id for p in lineage}))
        strategy = f"{self.id}:crossover" if len(parent_ids) > 1 else f"{self.id}:mutate"
        return child.model_copy(
            update={
                "conditions": spec.conditions,
                "generation_strategy": strategy,
                "parent_ids": parent_ids,
                "results": None,
                "provenance": ProvenanceRecord(
                    kind=ProvenanceKind.GENERATION,
                    producer=self.id,
                    producer_version=self.version,
                    input_ids=parent_ids,
                    parameters={"strategy": strategy},
                ),
            }
        )


def _with_fraction(monomer: MonomerUnit, fraction: float) -> MonomerUnit:
    return monomer.model_copy(update={"mole_fraction": max(0.0, min(1.0, fraction))})


def _renormalised(components: Sequence[MixtureComponent]) -> tuple[MixtureComponent, ...]:
    """Rescale fractions to sum to one, dropping vanishing components."""
    kept = [c for c in components if c.fraction > 1e-9]
    total = sum(c.fraction for c in kept)
    if not kept or total <= 0:
        return tuple(components)
    return tuple(c.model_copy(update={"fraction": c.fraction / total}) for c in kept)


def _finite(value: float) -> float:
    """Map an infinite crowding distance onto a large finite sort key."""
    return 1e12 if value == float("inf") else value
