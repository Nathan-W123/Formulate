"""Mixture-of-experts property prediction.

Specification section 4: all applicable experts score the same candidates, so
that trade-offs between properties are comparable.  Experts never select
candidates; they only predict.
"""

from __future__ import annotations

from .activity import UNIFACActivityExpert
from .adhesion import AdhesionExpert, DahlquistTackExpert
from .base import Expert, PredictionRequest
from .critical import AtomicCriticalExpert
from .dissolution import DissolutionExpert
from .feasibility import SynthesisFeasibilityExpert
from .hansen import GroupContributionHansenExpert, HansenSolubilityExpert
from .hazard import GHSHazardExpert
from .interfacial import InterfacialCorrelationExpert
from .joback import JobackThermalExpert
from .learned import LearnedBoilingPointExpert, LearnedRefractiveIndexExpert
from .lipophilicity import CrippenLipophilicityExpert
from .measured import MeasuredPropertyExpert
from .optical import LorentzLorenzExpert
from .kinetics import FreeRadicalCureExpert, PropagationExpert
from .rheology import (
    CorrespondingStatesViscosityExpert,
    JobackViscosityExpert,
    MeasuredViscosityExpert,
    MeltViscosityExpert,
    TroutonExtensionalExpert,
)
from .mechanical import PolymerMechanicalExpert
from .mixture import MixtureExpert
from .polymer import PolymerDensityExpert, PolymerGlassTransitionExpert
from .registry import ExpertRegistry
from .solubility import ESOLSolubilityExpert
from .structural import StructuralDescriptorExpert
from .toughness import (
    ChainToughnessExpert,
    MeasuredPolymerExpert,
    PolymerArchitectureExpert,
)

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
    # Estimates a Hansen triple where the compilation has none, which is
    # every component of a designed formulation that is not an ordinary
    # solvent. It belongs in the molecule-only panel because that is the
    # panel a mixture expert runs over its own components.
    GroupContributionHansenExpert,
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
    LearnedRefractiveIndexExpert,
    LorentzLorenzExpert,
    MeasuredViscosityExpert,
    JobackViscosityExpert,
    CorrespondingStatesViscosityExpert,
    MeltViscosityExpert,
    TroutonExtensionalExpert,
    DissolutionExpert,
    AdhesionExpert,
    # Whether a material sticks on contact, which the Owens-Wendt expert beside
    # it cannot answer: tack is a ceiling on a modulus, not a match of surface
    # energies, and the two functions a strand needs - hold and stick - want
    # moduli three and a half orders of magnitude apart.
    DahlquistTackExpert,
    UNIFACActivityExpert,
    PolymerGlassTransitionExpert,
    PolymerDensityExpert,
    PolymerMechanicalExpert,
    # Reaction kinetics. Everything above predicts what a material *is*; these
    # two predict how fast it changes, which is what a material chosen to
    # solidify in place has to be selected on.
    PropagationExpert,
    FreeRadicalCureExpert,
    # What a material does to the person holding it, and whether it draws or
    # snaps. Both were missing when a brittle sensitising thermoset came out as
    # the correct answer to five successive specifications: a failure mode that
    # is not a registered property cannot cost a candidate a single point.
    MeasuredPolymerExpert,
    ChainToughnessExpert,
    PolymerArchitectureExpert,
    GHSHazardExpert,
)


def molecular_registry() -> ExpertRegistry:
    """The molecule-only panel, used by formulation experts for components."""
    return ExpertRegistry(cls() for cls in PHASE1_EXPERTS)


def default_registry() -> ExpertRegistry:
    """Every expert: the molecular panel plus formulation coverage."""
    return ExpertRegistry(cls() for cls in PHASE1_EXPERTS + PHASE4_EXPERTS)


__all__ = [
    "PHASE1_EXPERTS", "PHASE4_EXPERTS", "HansenSolubilityExpert", "MeasuredPropertyExpert", "MixtureExpert",
    "AdhesionExpert", "AtomicCriticalExpert", "DahlquistTackExpert",
    "DissolutionExpert",
    "CorrespondingStatesViscosityExpert", "JobackViscosityExpert",
    "LearnedBoilingPointExpert", "LearnedRefractiveIndexExpert", "LorentzLorenzExpert",
    "MeasuredViscosityExpert", "MeltViscosityExpert", "TroutonExtensionalExpert",
    "PropagationExpert", "FreeRadicalCureExpert", "GroupContributionHansenExpert",
    "ChainToughnessExpert", "GHSHazardExpert", "MeasuredPolymerExpert",
    "PolymerArchitectureExpert",
    "PolymerDensityExpert", "PolymerGlassTransitionExpert",
    "PolymerMechanicalExpert",
    "UNIFACActivityExpert",
    "molecular_registry", "CrippenLipophilicityExpert", "ESOLSolubilityExpert", "Expert",
    "ExpertRegistry", "InterfacialCorrelationExpert", "JobackThermalExpert",
    "PredictionRequest", "StructuralDescriptorExpert", "SynthesisFeasibilityExpert",
    "default_registry",
]
