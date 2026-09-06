"""Mixture-of-experts property prediction.

Specification section 4: all applicable experts score the same candidates, so
that trade-offs between properties are comparable.  Experts never select
candidates; they only predict.
"""

from __future__ import annotations

from .base import Expert, PredictionRequest
from .feasibility import SynthesisFeasibilityExpert
from .interfacial import InterfacialCorrelationExpert
from .joback import JobackThermalExpert
from .lipophilicity import CrippenLipophilicityExpert
from .registry import ExpertRegistry
from .solubility import ESOLSolubilityExpert
from .structural import StructuralDescriptorExpert

#: The Phase 1 expert panel (specification section 10): a small set of
#: reliable, published, deterministic predictors spanning four expert families.
#: Mechanical and electrical experts are deliberately absent - they require the
#: polymer preparation of Phase 4 or the quantum module of Phase 3, and section
#: 13 forbids implying coverage the system does not have.
PHASE1_EXPERTS = (
    JobackThermalExpert,
    CrippenLipophilicityExpert,
    ESOLSolubilityExpert,
    InterfacialCorrelationExpert,
    SynthesisFeasibilityExpert,
    StructuralDescriptorExpert,
)


def default_registry() -> ExpertRegistry:
    """A registry populated with the Phase 1 expert panel."""
    return ExpertRegistry(cls() for cls in PHASE1_EXPERTS)


__all__ = [
    "PHASE1_EXPERTS", "CrippenLipophilicityExpert", "ESOLSolubilityExpert", "Expert",
    "ExpertRegistry", "InterfacialCorrelationExpert", "JobackThermalExpert",
    "PredictionRequest", "StructuralDescriptorExpert", "SynthesisFeasibilityExpert",
    "default_registry",
]
