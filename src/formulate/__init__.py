"""Formulate: behavior-driven inverse materials design.

Given a target specification of desired *behavior* (properties, conditions,
constraints), search chemical/material design space and return feasible
candidates.  The system does not attempt to invert a single property model;
it explores, predicts, ranks, and selectively validates.

Module separation is a scientific invariant, not a style choice
(specification section 11):

    exploration  proposes candidates
    experts      predict properties
    ranking      optimizes over objectives
    physics      validates (QM / MD)
    coordination routes and explains

No module silently substitutes for another.
"""

__version__ = "0.1.0"

SPEC_VERSION = "inverse-materials-design/compact-1"

__all__ = ["__version__", "SPEC_VERSION"]
