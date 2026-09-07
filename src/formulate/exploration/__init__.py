"""Candidate generation. Explorers propose; they never score or select."""

from .acquisition import (
    augmented_tchebycheff,
    expected_improvement,
    simplex_to_stick_breaking,
    stick_breaking_to_simplex,
)
from .base import Explorer
from .bayesopt import BayesOptConfig, BayesOptExplorer
from .database import ReferenceDatabaseExplorer, load_reference_compounds
from .mixtures import MixtureSeedConfig, MixtureSeedExplorer
from .filters import CandidateFilter, FilterReport, FilterResult
from .gp import GaussianProcess

__all__ = [
    "BayesOptConfig", "BayesOptExplorer", "GaussianProcess", "augmented_tchebycheff",
    "expected_improvement", "simplex_to_stick_breaking", "stick_breaking_to_simplex",
    "CandidateFilter", "Explorer", "FilterReport", "FilterResult",
    "MixtureSeedConfig", "MixtureSeedExplorer",
    "ReferenceDatabaseExplorer", "load_reference_compounds",
]
