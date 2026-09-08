"""Generative exploration: propose structures, never judge them.

Specification section 3 lists a learned generator as an optional exploration
strategy, and the roadmap has carried it as the one Phase 2 item never built.
The other explorers all work from what already exists - a reference database,
mutations of scored parents, compositions of a known recipe - so the pool can
only ever contain the catalogue and its neighbours.

Two providers, and the difference between them is the point.

:class:`SafeGptProvider` is a genuinely pretrained generative model: SAFE-GPT,
a transformer over the SAFE fragment representation, released under Apache 2.0
by Valence Labs. It loads pretrained weights and samples from a learned
distribution over chemical space.

:class:`SelfiesMutationProvider` is not a model and does not claim to be. It
mutates a SELFIES string, a representation whose grammar makes every token
sequence decode to a valid molecule, so mutation cannot produce a broken
structure the way SMILES mutation can. It is a heuristic. It is labelled a
heuristic in its identifier, its description and the provenance of everything
it proposes, because the one thing worse than having no generative model is
having a random walk that reports itself as one.

What this module does not do is decide anything. A generator proposes; the
filter rejects what violates the target's structural constraints, the experts
predict, the ranker ranks. Structural constraints are applied here only as an
early reject, before any expert is called, which is the same courtesy the
filter performs later rather than a judgement about quality.

A warning worth keeping in view. SAFE-GPT is trained on drug-like chemistry,
so left to itself it proposes sulfonamides and chloropyridines - reasonable
molecules, poor solvents. Seeding it from the run's own best candidates
through ``super_structure`` keeps it near the chemistry the target is actually
about, and the benchmark in ``benchmark_exploration`` measures whether that
helps rather than assuming it.
"""

from __future__ import annotations

import abc
import random
from dataclasses import dataclass
from typing import Sequence

from formulate.core.candidate import Candidate, MaterialClass, MoleculeSpec
from formulate.core.provenance import ProvenanceKind, ProvenanceRecord
from formulate.targets.spec import TargetSpec

from .base import Explorer
from .filters import CandidateFilter


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    """What a provider is, in terms a reader can check."""

    identifier: str
    #: True only for a model with trained weights. A heuristic must say False.
    pretrained: bool
    description: str
    reference: str = ""

    def as_parameters(self) -> dict:
        return {
            "provider": self.identifier,
            "pretrained": self.pretrained,
            "description": self.description,
            "reference": self.reference,
        }


class GenerativeProvider(abc.ABC):
    """Proposes SMILES strings, optionally conditioned on seed structures."""

    info: ProviderInfo

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Whether this provider can run here, checked rather than assumed."""

    def unavailable_reason(self) -> str:
        return ""

    @abc.abstractmethod
    def propose(
        self, seeds: Sequence[str], count: int, *, seed: int = 0
    ) -> list[str]:
        """Up to ``count`` SMILES. Deterministic given ``seed`` and ``seeds``."""


class SelfiesMutationProvider(GenerativeProvider):
    """Mutates a SELFIES string. A heuristic, and says so everywhere.

    SELFIES is a molecular string representation whose grammar is constructed
    so that every syntactically valid token sequence maps to a chemically valid
    molecule. Mutating one therefore cannot produce an unparseable structure,
    which is what makes a random walk over it worth anything at all: the
    equivalent walk over SMILES spends most of its proposals on strings that do
    not decode.

    That is a property of the representation, not intelligence. This provider
    has no model, no weights and no learned distribution over chemistry.
    """

    info = ProviderInfo(
        identifier="selfies-mutation",
        pretrained=False,
        description=(
            "random token mutation of a SELFIES string; a heuristic that "
            "guarantees valid output, not a trained generative model"
        ),
        reference="Krenn et al., Mach. Learn.: Sci. Technol. 1 (2020) 045024",
    )

    def is_available(self) -> bool:
        try:
            import selfies  # noqa: F401
        except Exception:
            return False
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        try:
            import selfies  # noqa: F401
        except Exception as exc:
            return f"the selfies package is not installed ({exc})"
        from formulate import chem

        if not chem.rdkit_available():
            return "RDKit is required to canonicalise the decoded structure"
        return ""

    def propose(self, seeds: Sequence[str], count: int, *, seed: int = 0) -> list[str]:
        import selfies as sf
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        if not seeds:
            return []
        alphabet = sorted(sf.get_semantic_robust_alphabet())
        rng = random.Random(seed)
        out: list[str] = []
        attempts = 0
        while len(out) < count and attempts < count * 25:
            attempts += 1
            parent = seeds[rng.randrange(len(seeds))]
            try:
                tokens = list(sf.split_selfies(sf.encoder(parent)))
            except Exception:
                continue
            if not tokens:
                continue
            action = rng.random()
            index = rng.randrange(len(tokens))
            if action < 0.45:
                tokens[index] = rng.choice(alphabet)
            elif action < 0.80:
                tokens.insert(index, rng.choice(alphabet))
            elif len(tokens) > 2:
                tokens.pop(index)
            else:
                tokens[index] = rng.choice(alphabet)
            try:
                smiles = sf.decoder("".join(tokens))
            except Exception:
                continue
            canonical = acceptable_structure(smiles)
            if canonical is None:
                continue
            out.append(canonical)
        return out


class SafeGptProvider(GenerativeProvider):
    """SAFE-GPT: a transformer over a fragment representation, pretrained.

    Sampling is conditioned on the run's own best candidates through
    ``super_structure`` where seeds are available, and falls back to de novo
    sampling only when there are none. That distinction matters for materials
    work: the model's training distribution is drug-like, so unconditioned it
    proposes pharmaceutical scaffolds, which are valid molecules and generally
    poor solvents.
    """

    info = ProviderInfo(
        identifier="safe-gpt",
        pretrained=True,
        description=(
            "SAFE-GPT, a GPT-2 scale transformer over the SAFE fragment "
            "representation, sampled conditionally on seed structures where "
            "available; trained on drug-like chemistry"
        ),
        reference="Noutahi et al., Digital Discovery 3 (2024) 796; Apache-2.0",
    )

    def __init__(self, max_new_tokens: int = 48) -> None:
        self.max_new_tokens = max_new_tokens
        self._designer = None
        #: Failures from the last call, kept so a silent empty batch can be
        #: explained rather than guessed at.
        self.last_errors: list[str] = []

    def is_available(self) -> bool:
        try:
            import safe  # noqa: F401
            import torch  # noqa: F401
        except Exception:
            return False
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        try:
            import torch  # noqa: F401
        except Exception as exc:
            return f"torch is not installed ({exc}); install formulate[generative]"
        try:
            import safe  # noqa: F401
        except Exception as exc:
            return f"the safe-mol package is not installed ({exc})"
        from formulate import chem

        if not chem.rdkit_available():
            return "RDKit is required to canonicalise generated structures"
        return ""

    def _load(self):
        if self._designer is None:
            from safe import SAFEDesign

            self._designer = SAFEDesign.load_default(verbose=False)
        return self._designer

    def propose(self, seeds: Sequence[str], count: int, *, seed: int = 0) -> list[str]:
        import torch
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        designer = self._load()
        produced: list[str] = []
        rng = random.Random(seed)
        # de_novo_generation samples through the transformers generate() call
        # and takes no seed of its own, so reproducibility comes from the
        # global torch generator. super_structure does take one and is given it
        # explicitly, so both paths are deterministic for a given seed.
        torch.manual_seed(seed)
        self.last_errors = []

        if seeds:
            # One core at a time, so a core the model cannot fragment costs
            # that core's proposals rather than the whole batch.
            per_core = max(2, count // max(1, min(len(seeds), 4)))
            for core in list(seeds)[:4]:
                if len(produced) >= count:
                    break
                try:
                    produced.extend(
                        designer.super_structure(
                            core=core,
                            n_samples_per_trial=per_core,
                            n_trials=1,
                            sanitize=True,
                            random_seed=rng.randrange(1 << 30),
                        )
                    )
                except Exception as exc:
                    # Recorded rather than swallowed: a core the model cannot
                    # fragment is ordinary, a wrong call signature is not, and
                    # silence makes the two indistinguishable.
                    self.last_errors.append(f"super_structure({core[:24]}): {exc}")
        if len(produced) < count:
            try:
                produced.extend(
                    designer.de_novo_generation(
                        sanitize=True,
                        n_samples_per_trial=count - len(produced),
                        max_length=self.max_new_tokens,
                    )
                )
            except Exception as exc:
                self.last_errors.append(f"de_novo_generation: {exc}")

        out: list[str] = []
        for smiles in produced:
            canonical = acceptable_structure(smiles)
            if canonical is not None:
                out.append(canonical)
        return out[:count]


def acceptable_structure(smiles: str) -> str | None:
    """The canonical SMILES of a usable single-molecule proposal, or None.

    Both providers emit structures that parse and are still not candidates.
    SAFE-GPT returns salts and co-crystals - ``N.O=Cc1ccc...`` is two species
    separated by a dot - and a molecule candidate carries one structure, so a
    blend belongs in a MixtureSpec with fractions rather than smuggled into a
    SMILES string. Both providers can also emit open-shell fragments such as
    ``[CH]``, which are radicals rather than bottleable compounds.

    Rejecting these here rather than in the filter keeps the rule where the
    behaviour is: the filter's job is the target's structural constraints, and
    no other explorer produces either kind of output.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        return None
    return Chem.MolToSmiles(mol)


def available_providers() -> list[GenerativeProvider]:
    """Every provider that can run here, best first.

    A pretrained model is preferred over a heuristic, which is the ordering the
    specification's model policy asks for. If neither can run, the explorer
    returns nothing and says why rather than silently degrading.
    """
    ordered: list[GenerativeProvider] = [SafeGptProvider(), SelfiesMutationProvider()]
    return [p for p in ordered if p.is_available()]


@dataclass
class GenerativeConfig:
    """How the generative explorer is run."""

    #: Seed structures taken from the best scored candidates, most recent run
    #: first. Zero means unconditioned sampling, which for a drug-trained model
    #: means drug-like proposals.
    seed_candidates: int = 4
    #: Proposals requested per accepted candidate, to absorb rejection by the
    #: structural filter and by deduplication.
    oversample: float = 3.0
    #: Cap on proposals asked of the provider in one call.
    max_proposals: int = 200
    #: Restrict to a named provider, by identifier. None takes the best
    #: available.
    provider: str | None = None


class GenerativeExplorer(Explorer):
    """Proposes structures from a generative model, or from nothing at all."""

    id = "generative"
    version = "1"

    def __init__(
        self,
        config: GenerativeConfig | None = None,
        providers: Sequence[GenerativeProvider] | None = None,
    ) -> None:
        self.config = config or GenerativeConfig()
        self._providers = list(providers) if providers is not None else None

    def providers(self) -> list[GenerativeProvider]:
        if self._providers is not None:
            return [p for p in self._providers if p.is_available()]
        return available_providers()

    def selected_provider(self) -> GenerativeProvider | None:
        for provider in self.providers():
            if self.config.provider in (None, provider.info.identifier):
                return provider
        return None

    def is_available(self) -> bool:
        return self.selected_provider() is not None

    def unavailable_reason(self) -> str:
        if self.is_available():
            return ""
        reasons = []
        for provider in (SafeGptProvider(), SelfiesMutationProvider()):
            reason = provider.unavailable_reason()
            if reason:
                reasons.append(f"{provider.info.identifier}: {reason}")
        return "; ".join(reasons) or "no generative provider is configured"

    def propose(
        self,
        spec: TargetSpec,
        count: int,
        *,
        scored: Sequence[Candidate] = (),
        seed: int = 0,
    ) -> list[Candidate]:
        if count <= 0 or MaterialClass.MOLECULE not in spec.material_classes:
            return []
        provider = self.selected_provider()
        if provider is None:
            return []

        seeds = self._seed_structures(scored)
        requested = min(
            self.config.max_proposals, max(count, int(count * self.config.oversample))
        )
        proposals = provider.propose(seeds, requested, seed=seed)

        # Structural constraints are applied before anything expensive, which
        # is a courtesy to the budget rather than a judgement: the filter
        # applies the same rules again on the way into evaluation.
        gate = CandidateFilter(spec.structural, spec.conditions)
        seen = {c.structure_id for c in scored}
        out: list[Candidate] = []
        for smiles in proposals:
            candidate = self._build(smiles, spec, provider, seeds, seed)
            if candidate.structure_id in seen:
                continue
            if not gate.check(candidate).passed:
                continue
            seen.add(candidate.structure_id)
            out.append(candidate)
            if len(out) >= count:
                break
        return out

    def _seed_structures(self, scored: Sequence[Candidate]) -> list[str]:
        """SMILES of the best scored molecules, in the order given."""
        seeds: list[str] = []
        for candidate in scored:
            if candidate.material_class is not MaterialClass.MOLECULE:
                continue
            if candidate.molecule is None or candidate.results is None:
                continue
            seeds.append(candidate.molecule.smiles)
            if len(seeds) >= self.config.seed_candidates:
                break
        return seeds

    def _build(
        self,
        smiles: str,
        spec: TargetSpec,
        provider: GenerativeProvider,
        seeds: Sequence[str],
        seed: int,
    ) -> Candidate:
        parameters = provider.info.as_parameters()
        parameters.update(
            {
                "seed": seed,
                "seed_structures": list(seeds),
                "oversample": self.config.oversample,
            }
        )
        return Candidate(
            material_class=MaterialClass.MOLECULE,
            molecule=MoleculeSpec(smiles=smiles),
            conditions=spec.conditions,
            generation_strategy=f"{self.id}:{provider.info.identifier}",
            provenance=ProvenanceRecord(
                kind=ProvenanceKind.GENERATION,
                producer=f"{self.id}:{provider.info.identifier}",
                producer_version=self.version,
                parameters=parameters,
            ),
        )
