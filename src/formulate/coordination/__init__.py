"""Coordination: routing, budgets, feasibility diagnosis and reporting."""

from .coordinator import DeterministicCoordinator, DesignRun, RunConfig
from .feasibility import ConstraintDiagnosis, FeasibilityAnalysis, analyze
from .iterative import (
    IterationConfig,
    IterativeCoordinator,
    IterativeRun,
    default_iterative_coordinator,
)
from .metrics import RoundRecord, SearchMetrics, top_k_recall
from .report import render_report

__all__ = [
    "ConstraintDiagnosis", "DesignRun", "DeterministicCoordinator", "FeasibilityAnalysis",
    "IterationConfig", "IterativeCoordinator", "IterativeRun", "RoundRecord", "RunConfig",
    "SearchMetrics", "analyze", "default_iterative_coordinator", "render_report",
    "top_k_recall",
]
