"""Provenance records.

Specification section 11: "every generated structure, prediction, simulation,
transformation, and score is versioned and reproducible."

A provenance record names what produced a result, at which version, from which
inputs, with which parameters.  Records form a DAG through ``input_ids``, so a
final ranking can be traced back to the generator that proposed the structure.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .hashing import content_hash


class ProvenanceKind(str, Enum):
    GENERATION = "generation"
    FILTER = "filter"
    PREDICTION = "prediction"
    SIMULATION = "simulation"
    TRANSFORM = "transform"
    SCORE = "score"
    RETRIEVAL = "retrieval"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SoftwareEnvironment(BaseModel):
    """Versions of the software that produced a result."""

    model_config = ConfigDict(frozen=True)

    formulate_version: str = ""
    python_version: str = Field(default_factory=lambda: sys.version.split()[0])
    platform: str = Field(default_factory=platform.platform)
    backends: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def capture(cls, **backends: str) -> "SoftwareEnvironment":
        from formulate import __version__

        return cls(formulate_version=__version__, backends=dict(backends))


class ProvenanceRecord(BaseModel):
    """An immutable statement of how one artefact came to exist."""

    model_config = ConfigDict(frozen=True)

    kind: ProvenanceKind
    #: Identifier of the module/expert/generator that produced the artefact.
    producer: str
    producer_version: str = "0"
    #: Content hashes of the artefacts this one was derived from.
    input_ids: tuple[str, ...] = ()
    #: Parameters that, with the inputs, should reproduce the artefact.
    parameters: dict[str, Any] = Field(default_factory=dict)
    software: SoftwareEnvironment = Field(default_factory=SoftwareEnvironment.capture)
    created_at: datetime = Field(default_factory=_utcnow)
    notes: tuple[str, ...] = ()

    @property
    def record_id(self) -> str:
        """Content hash over the fields that determine the result.

        The line drawn here is between what computed the number and what merely
        described the host that ran it.

        Included: the producer and its version, the inputs, the parameters, the
        version of this package, and the versions of the backends named in the
        record. A backend version is not decoration - RDKit's conformer
        generation and PySCF's defaults both change between releases, so a
        value computed against a different one is a different value, and giving
        the two the same provenance id says they are interchangeable when they
        are not. The docstring here used to claim only ``software.platform``
        was excluded while the code excluded the whole ``software`` field.

        Excluded: ``created_at``, ``software.platform`` and
        ``software.python_version``. Re-running the same computation tomorrow
        on another machine should produce the same record id; the operating
        system and the interpreter patch level are properties of that machine
        rather than of the method.
        """
        return content_hash(
            {
                "kind": self.kind.value,
                "producer": self.producer,
                "producer_version": self.producer_version,
                "input_ids": list(self.input_ids),
                "parameters": self.parameters,
                "formulate_version": self.software.formulate_version,
                "backends": dict(sorted(self.software.backends.items())),
            },
            prefix="prov",
        )

    def derive(self, **overrides: Any) -> "ProvenanceRecord":
        """Return a copy with fields replaced."""
        return self.model_copy(update=overrides)
