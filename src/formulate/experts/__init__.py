"""Mixture-of-experts property prediction.

Specification section 4: all applicable experts score the same candidates, so
that trade-offs between properties are comparable.  Experts never select
candidates; they only predict.
"""

from __future__ import annotations

from .activity import UNIFACActivityExpert
from .adhesion import AdhesionExpert
from .blend import PolymerBlendMeltExpert, PolymerBlendSolidExpert
from .cohesion import CohesionExpert
from .conformation import ConformationExpert
from .electronic import QuantumElectronicExpert
from .flammability import FlashPointExpert
from .mixture_thermal import MixtureThermalExpert
from .polymer_dissolution import PolymerDissolutionExpert
from .polymer_feasibility import PolymerFeasibilityExpert
from .polymer_hansen import PolymerHansenExpert
from .polymer_structural import PolymerStructuralExpert
from .polymer_thermal import PolymerThermalExpert
from .semiempirical import SemiempiricalElectronicExpert
from .transport import LiquidTransportExpert
from .melt import PolymerMeltExpert
from .spinline import SpinlineExpert
from .base import Expert, PredictionRequest
from .critical import AtomicCriticalExpert
from .dissolution import DissolutionExpert
from .feasibility import SynthesisFeasibilityExpert
from .hansen import HansenSolubilityExpert
from .interfacial import InterfacialCorrelationExpert
from .joback import JobackThermalExpert
from .learned import LearnedBoilingPointExpert
from .lipophilicity import CrippenLipophilicityExpert
from .measured import MeasuredPropertyExpert
from .mechanical import PolymerMechanicalExpert
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
PHASE4_EXPERTS = (
    MixtureExpert,
    AtomicCriticalExpert,
    LearnedBoilingPointExpert,
    DissolutionExpert,
    AdhesionExpert,
    UNIFACActivityExpert,
    PolymerGlassTransitionExpert,
    PolymerDensityExpert,
    PolymerMechanicalExpert,
    PolymerMeltExpert,
    PolymerBlendMeltExpert,
    PolymerBlendSolidExpert,
    SpinlineExpert,
)

#: Phase 6 closes the coverage holes the registry had been reporting: the
#: electrical family was empty for every material class, a polymer carried no
#: solubility parameter and no structural descriptor, a molecule had no
#: viscosity, and a formulation had nothing but density and Hansen parameters.
#:
#: Ordering within the tuple does not matter - the registry topologically sorts
#: on declared dependencies - but the dependency edges are worth naming, because
#: they are why these are Phase 6 and not Phase 1: CohesionExpert needs a
#: Hildebrand parameter or a Hansen triple, PolymerHansenExpert needs an
#: amorphous density, LiquidTransportExpert needs critical constants, and
#: PolymerDissolutionExpert needs the Hansen triple PolymerHansenExpert makes.
#:
#: MixtureThermalExpert delegates to the molecular panel for its components in
#: the same way MixtureExpert does, so it must not itself be a member of that
#: panel; that would be circular.
PHASE6_EXPERTS = (
    QuantumElectronicExpert,
    SemiempiricalElectronicExpert,
    CohesionExpert,
    ConformationExpert,
    FlashPointExpert,
    LiquidTransportExpert,
    MixtureThermalExpert,
    PolymerHansenExpert,
    PolymerStructuralExpert,
    PolymerThermalExpert,
    PolymerDissolutionExpert,
    PolymerFeasibilityExpert,
)


#: The polymer-only panel, used by the blend expert for its components. Kept
#: separate for the same reason the molecular one is: a blend delegates to it,
#: and including the blend expert would make that circular.
POLYMER_EXPERTS = (
    PolymerGlassTransitionExpert,
    PolymerDensityExpert,
    PolymerMechanicalExpert,
    PolymerMeltExpert,
    PolymerHansenExpert,
    PolymerStructuralExpert,
    PolymerThermalExpert,
)


def polymer_registry() -> ExpertRegistry:
    """The polymer-only panel, used by the blend expert for its components."""
    return ExpertRegistry(cls() for cls in POLYMER_EXPERTS)


def molecular_registry() -> ExpertRegistry:
    """The molecule-only panel, used by formulation experts for components."""
    return ExpertRegistry(cls() for cls in PHASE1_EXPERTS)


def default_registry() -> ExpertRegistry:
    """Every expert: the molecular panel plus formulation coverage."""
    return ExpertRegistry(cls() for cls in PHASE1_EXPERTS + PHASE4_EXPERTS + PHASE6_EXPERTS)


__all__ = [
    "PHASE1_EXPERTS", "PHASE4_EXPERTS", "PHASE6_EXPERTS", "POLYMER_EXPERTS", "polymer_registry",
    "CohesionExpert", "ConformationExpert", "FlashPointExpert", "LiquidTransportExpert",
    "MixtureThermalExpert", "PolymerDissolutionExpert", "PolymerFeasibilityExpert",
    "PolymerHansenExpert", "PolymerStructuralExpert", "PolymerThermalExpert",
    "QuantumElectronicExpert", "SemiempiricalElectronicExpert", "HansenSolubilityExpert", "MeasuredPropertyExpert", "MixtureExpert",
    "AdhesionExpert", "AtomicCriticalExpert", "DissolutionExpert", "LearnedBoilingPointExpert", "PolymerDensityExpert", "PolymerGlassTransitionExpert",
    "PolymerMechanicalExpert", "PolymerMeltExpert", "PolymerBlendMeltExpert", "PolymerBlendSolidExpert", "SpinlineExpert",
    "UNIFACActivityExpert",
    "molecular_registry", "CrippenLipophilicityExpert", "ESOLSolubilityExpert", "Expert",
    "ExpertRegistry", "InterfacialCorrelationExpert", "JobackThermalExpert",
    "PredictionRequest", "StructuralDescriptorExpert", "SynthesisFeasibilityExpert",
    "default_registry",
]
