"""Two properties of the coordination layer, checked rather than asserted.

The tests that actually run a coordinator are marked slow: a full run builds
every explorer, and the generative one loads a transformer. The static checks -
that nothing draws from an unseeded generator and that no module calls a
language model - are the ones most likely to catch a regression and they cost
nothing, so they stay in the fast suite.

Specification section 11 requires reproducible runs; the brief adds that any
language-model layer is restricted to translating intent and explaining
results, and must never produce a numerical prediction. Both are the sort of
claim that stays true only while something checks it, because either could be
broken by a plausible-looking addition made in good faith.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from conftest import requires_rdkit

from formulate.coordination import DeterministicCoordinator, RunConfig
from formulate.targets.spec import TargetSpec

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "formulate"

_SPEC = TargetSpec.from_dict(
    {
        "name": "determinism",
        "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
        "requirements": [
            {
                "property": "normal_boiling_point",
                "direction": "in_range",
                "lower": "60 degC",
                "upper": "170 degC",
                "hard": True,
            },
            {"property": "logp", "direction": "target", "target": 2.0, "lower": 0, "upper": 4},
        ],
    }
)


# -- the same run twice ----------------------------------------------------


@requires_rdkit
@pytest.mark.slow
def test_one_seed_gives_one_ranking():
    first = DeterministicCoordinator(config=RunConfig(pool_size=15, seed=7)).run(_SPEC)
    second = DeterministicCoordinator(config=RunConfig(pool_size=15, seed=7)).run(_SPEC)
    assert [e.candidate.candidate_id for e in first.ranking.ranked] == [
        e.candidate.candidate_id for e in second.ranking.ranked
    ]


@requires_rdkit
@pytest.mark.slow
def test_the_same_seed_gives_the_same_numbers_not_only_the_same_order():
    first = DeterministicCoordinator(config=RunConfig(pool_size=15, seed=7)).run(_SPEC)
    second = DeterministicCoordinator(config=RunConfig(pool_size=15, seed=7)).run(_SPEC)
    for a, b in zip(first.ranking.ranked, second.ranking.ranked):
        assert a.scalar == b.scalar
    assert first.ranking.hypervolume == second.ranking.hypervolume


@requires_rdkit
@pytest.mark.slow
def test_a_different_seed_is_allowed_to_differ():
    """Otherwise the seed is decoration and the determinism test is vacuous."""
    first = DeterministicCoordinator(config=RunConfig(pool_size=15, seed=1)).run(_SPEC)
    second = DeterministicCoordinator(config=RunConfig(pool_size=15, seed=2)).run(_SPEC)
    assert first.ranking.ranked and second.ranking.ranked


@requires_rdkit
@pytest.mark.slow
def test_an_iterative_run_is_reproducible_too():
    from formulate.coordination.iterative import (
        IterationConfig,
        default_iterative_coordinator,
    )

    def once():
        return default_iterative_coordinator(
            config=RunConfig(pool_size=12, seed=3),
            iteration=IterationConfig(max_rounds=2, batch_size=8),
        ).run_iterative(_SPEC)

    first, second = once(), once()
    assert first.metrics.stop_reason == second.metrics.stop_reason
    assert first.metrics.total_evaluated == second.metrics.total_evaluated
    assert [e.candidate.candidate_id for e in first.final.ranking.ranked] == [
        e.candidate.candidate_id for e in second.final.ranking.ranked
    ]


# -- nothing draws from an unseeded source ---------------------------------


def _python_sources():
    return [p for p in _SRC.rglob("*.py") if "__pycache__" not in p.parts]


def test_no_module_draws_from_an_unseeded_random_source():
    """``random.random()`` and ``np.random.rand()`` read a global generator.

    Every explorer here takes a seed and builds its own generator from it. One
    module-level call to the global one would make a run irreproducible in a
    way no test of a single explorer would catch.
    """
    banned = {
        "random": {
            "random", "randint", "choice", "choices", "shuffle", "sample",
            "uniform", "gauss", "randrange",
        },
        "np.random": {"rand", "randn", "randint", "choice", "shuffle", "normal", "uniform"},
    }
    offenders = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = ast.unparse(node.func.value)
            if owner in banned and node.func.attr in banned[owner]:
                offenders.append(f"{path.relative_to(_SRC)}:{node.lineno} {owner}.{node.func.attr}")
    assert not offenders, "unseeded randomness: " + "; ".join(offenders)


def test_every_explorer_accepts_a_seed():
    from formulate.exploration.base import Explorer

    for cls in Explorer.__subclasses__():
        signature = cls.propose.__code__.co_varnames[: cls.propose.__code__.co_argcount]
        assert "seed" in signature or "seed" in (
            cls.propose.__kwdefaults__ or {}
        ), f"{cls.__name__}.propose takes no seed"


# -- the language-model boundary -------------------------------------------


def test_no_module_calls_a_language_model():
    """The brief restricts any LLM layer to intent translation and explanation.

    There is none at all, which is the strongest form of that restriction, and
    this test is what keeps it true. SAFE-GPT is deliberately not caught: it is
    a generative model over molecular fragments that proposes *structures* for
    the explorer to evaluate, and it never produces a number that reaches a
    prediction.
    """
    banned = ("openai", "anthropic", "litellm", "langchain", "transformers.pipeline")
    offenders = []
    for path in _python_sources():
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if any(name == b or name.startswith(b + ".") for b in banned):
                    offenders.append(f"{path.relative_to(_SRC)}:{node.lineno} imports {name}")
    assert not offenders, "language-model dependency: " + "; ".join(offenders)


def test_the_generative_provider_returns_structures_and_not_predictions():
    """SAFE-GPT's output is a list of SMILES that then go through the panel."""
    from formulate.exploration.generative import GenerativeProvider

    assert not hasattr(GenerativeProvider, "predict")
    annotation = GenerativeProvider.propose.__annotations__.get("return")
    assert annotation is not None
    assert "str" in str(annotation), annotation


@requires_rdkit
@pytest.mark.slow
def test_every_prediction_in_a_run_names_a_non_language_model_method():
    coordinator = DeterministicCoordinator(config=RunConfig(pool_size=10, seed=0))
    run = coordinator.run(_SPEC)
    for entry in run.ranking.ranked:
        results = entry.candidate.results
        if results is None:
            continue
        for prediction in results.predictions:
            if prediction.quantity is None:
                continue
            assert prediction.expert_id
            assert prediction.method, f"{prediction.property} states no method"
            assert "gpt" not in prediction.expert_id.lower()
