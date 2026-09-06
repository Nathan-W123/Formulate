"""Coordination: routing, budgets, feasibility diagnosis and reporting."""

from .adaptive import (
    Action,
    ActionEstimate,
    AdaptiveConfig,
    AdaptiveCoordinator,
    AdaptiveRun,
    default_adaptive_coordinator,
)
from .benchmark import (
    ArmResult,
    BenchmarkResult,
    IncomparableTarget,
    check_comparable,
    run_benchmark,
)
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
    "Action", "ActionEstimate", "ArmResult", "BenchmarkResult", "IncomparableTarget",
    "check_comparable", "run_benchmark", "AdaptiveConfig", "AdaptiveCoordinator", "AdaptiveRun",
    "default_adaptive_coordinator", "ConstraintDiagnosis", "DesignRun", "DeterministicCoordinator", "FeasibilityAnalysis",
    "IterationConfig", "IterativeCoordinator", "IterativeRun", "RoundRecord", "RunConfig",
    "SearchMetrics", "analyze", "default_iterative_coordinator", "render_report",
    "top_k_recall", "Disagreement", "PhysicsValidator", "ValidatedRun",
    "ValidatingCoordinator", "ValidationMethod", "ValidationPolicy", "ValidationReport",
    "default_validating_coordinator", "select_targets", "validatable_properties",
]
