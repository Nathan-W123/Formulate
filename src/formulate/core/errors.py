"""Exception hierarchy.

Errors are deliberately specific: a dimensional mismatch and an out-of-domain
prediction are different failures with different remedies, and collapsing them
into a generic exception is how silent unit bugs enter a ranking pipeline.
"""

from __future__ import annotations


class FormulateError(Exception):
    """Base class for all Formulate errors."""


class UnitError(FormulateError):
    """A unit string could not be parsed, or units were incompatible."""


class DimensionalityError(UnitError):
    """A quantity was supplied with the wrong physical dimension.

    Raised before ranking; specification section 11 forbids ranking prior to
    dimensional validation.
    """


class UnknownPropertyError(FormulateError):
    """A property name is not present in the canonical property registry."""


class ConditionMismatchError(FormulateError):
    """A prediction was requested or used outside its stated conditions."""


class CandidateError(FormulateError):
    """A candidate is structurally invalid or internally inconsistent."""


class ExpertError(FormulateError):
    """An expert failed to produce a prediction."""


class BackendUnavailableError(ExpertError):
    """An optional scientific backend (RDKit, thermo, ...) is not installed."""


class BudgetExhaustedError(FormulateError):
    """The coordinator's evaluation budget was exhausted."""
