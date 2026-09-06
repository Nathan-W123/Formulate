"""Core primitives shared by every module.

These types encode the invariants of specification section 11: canonical
units, first-class uncertainty, explicit conditions, content addressing and
provenance.  Everything else in the system is built on them.
"""

from .conditions import ConditionMatch, Conditions, Phase, match_conditions
from .errors import (
    BackendUnavailableError,
    BudgetExhaustedError,
    CandidateError,
    ConditionMismatchError,
    DimensionalityError,
    ExpertError,
    FormulateError,
    UnitError,
    UnknownPropertyError,
)
from .hashing import cache_key, canonical_json, content_hash
from .prediction import Prediction, PredictionStatus
from .properties import (
    PROPERTY_REGISTRY,
    PropertyDef,
    PropertyFamily,
    get_property,
    properties_in_family,
    validate_unit_for,
)
from .provenance import ProvenanceKind, ProvenanceRecord, SoftwareEnvironment
from .quantity import ApplicabilityDomain, Quantity, Uncertainty, UncertaintyKind
from .units import canonical_unit, convert, convert_delta, dimensionality, registry

__all__ = [
    "PROPERTY_REGISTRY", "ApplicabilityDomain", "BackendUnavailableError",
    "BudgetExhaustedError", "CandidateError", "ConditionMatch", "ConditionMismatchError",
    "Conditions", "DimensionalityError", "ExpertError", "FormulateError", "Phase",
    "Prediction", "PredictionStatus", "PropertyDef", "PropertyFamily", "ProvenanceKind",
    "ProvenanceRecord", "Quantity", "SoftwareEnvironment", "Uncertainty", "UncertaintyKind",
    "UnitError", "UnknownPropertyError", "cache_key", "canonical_json", "canonical_unit",
    "content_hash", "convert", "convert_delta", "dimensionality", "get_property",
    "match_conditions", "properties_in_family", "registry", "validate_unit_for",
]
