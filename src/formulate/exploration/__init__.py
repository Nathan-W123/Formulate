"""Candidate generation. Explorers propose; they never score or select."""

from .acquisition import (
    augmented_tchebycheff,
    expected_improvement,
    simplex_to_stick_breaking,
    stick_breaking_to_simplex,
)
from .base import Explorer
from .bayesopt import BayesOptConfig, BayesOptExplorer
from .blends import PolymerBlendExplorer
from .database import ReferenceDatabaseExplorer, load_reference_compounds
from .mixtures import MixtureSeedConfig, MixtureSeedExplorer
from .polymers import MOLAR_MASSES, PolymerLibraryExplorer
from .solutions import PolymerSolutionExplorer
from .filters import CandidateFilter, FilterReport, FilterResult
from .gp import GaussianProcess

__all__ = [
    "BayesOptConfig", "BayesOptExplorer", "PolymerBlendExplorer", "GaussianProcess", "augmented_tchebycheff",
    "expected_improvement", "simplex_to_stick_breaking", "stick_breaking_to_simplex",
    "CandidateFilter", "Explorer", "FilterReport", "FilterResult",
    "MixtureSeedConfig", "MixtureSeedExplorer",
    "MOLAR_MASSES", "PolymerLibraryExplorer", "PolymerSolutionExplorer",
    "ReferenceDatabaseExplorer", "load_reference_compounds",
]
