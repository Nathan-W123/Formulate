"""Mixture-of-experts property prediction.

Specification section 4: all applicable experts score the same candidates, so
that trade-offs between properties are comparable.  Experts never select
candidates; they only predict.
"""

from __future__ import annotations

from .base import Expert, PredictionRequest
from .feasibility import SynthesisFeasibilityExpert
from .hansen import HansenSolubilityExpert
from .interfacial import InterfacialCorrelationExpert
from .joback import JobackThermalExpert
from .lipophilicity import CrippenLipophilicityExpert
from .measured import MeasuredPropertyExpert
from .mixture import MixtureExpert
from .polymer import PolymerDensityExpert, PolymerGlassTransitionExpert
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
    HansenSolubilityExpert,
    MeasuredPropertyExpert,
)

#: Phase 4 adds formulation and polymer coverage. A mixture expert delegates
#: to the molecular panel for its components, so the two sets are kept
#: separate to make that dependency explicit rather than circular. The polymer
#: experts delegate to nothing: a repeat unit is not a molecule, and section 13
#: forbids answering for the bulk polymer with the monomer's properties.
PHASE4_EXPERTS = (MixtureExpert, PolymerGlassTransitionExpert, PolymerDensityExpert)


def molecular_registry() -> ExpertRegistry:
    """The molecule-only panel, used by formulation experts for components."""
    return ExpertRegistry(cls() for cls in PHASE1_EXPERTS)


def default_registry() -> ExpertRegistry:
    """Every expert: the molecular panel plus formulation coverage."""
    return ExpertRegistry(cls() for cls in PHASE1_EXPERTS + PHASE4_EXPERTS)


__all__ = [
    "PHASE1_EXPERTS", "PHASE4_EXPERTS", "HansenSolubilityExpert", "MeasuredPropertyExpert", "MixtureExpert",
    "PolymerDensityExpert", "PolymerGlassTransitionExpert",
    "molecular_registry", "CrippenLipophilicityExpert", "ESOLSolubilityExpert", "Expert",
    "ExpertRegistry", "InterfacialCorrelationExpert", "JobackThermalExpert",
    "PredictionRequest", "StructuralDescriptorExpert", "SynthesisFeasibilityExpert",
    "default_registry",
]
