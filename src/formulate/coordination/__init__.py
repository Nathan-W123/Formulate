"""Coordination: routing, budgets, feasibility diagnosis and reporting."""

from .coordinator import DeterministicCoordinator, DesignRun, RunConfig
from .feasibility import ConstraintDiagnosis, FeasibilityAnalysis, analyze
from .iterative import (
    IterationConfig,
    IterativeCoordinator,
    IterativeRun,
    ValidatedRun,
    ValidatingCoordinator,
    default_iterative_coordinator,
    default_validating_coordinator,
)
from .metrics import RoundRecord, SearchMetrics, top_k_recall
from .report import render_report
from .validation import (
    Disagreement,
    PhysicsValidator,
    ValidationMethod,
    ValidationPolicy,
    ValidationReport,
    select_targets,
    validatable_properties,
)

__all__ = [
    "ConstraintDiagnosis", "DesignRun", "DeterministicCoordinator", "FeasibilityAnalysis",
    "IterationConfig", "IterativeCoordinator", "IterativeRun", "RoundRecord", "RunConfig",
    "SearchMetrics", "analyze", "default_iterative_coordinator", "render_report",
    "top_k_recall", "Disagreement", "PhysicsValidator", "ValidatedRun",
    "ValidatingCoordinator", "ValidationMethod", "ValidationPolicy", "ValidationReport",
    "default_validating_coordinator", "select_targets", "validatable_properties",
]
