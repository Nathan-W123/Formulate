"""Database retrieval.

Specification section 3: "Database retrieval: retrieve known molecules /
materials near target property regions or structural motifs."

The bundled reference set is small and deliberately so - it exists to make the
pipeline runnable and to give the Phase 1 calibration check a held-out set of
measured values.  A production deployment replaces this explorer with one
querying a real structure database; the interface is what matters.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any, Sequence

from formulate.core.candidate import Candidate, MaterialClass, MoleculeSpec
from formulate.core.provenance import ProvenanceKind, ProvenanceRecord
from formulate.targets.spec import TargetSpec

from .base import Explorer


@lru_cache(maxsize=1)
def load_reference_compounds() -> tuple[dict[str, Any], ...]:
    """The bundled reference compounds, with their measured values."""
    text = (
        resources.files("formulate.data")
        .joinpath("reference_compounds.json")
        .read_text(encoding="utf-8")
    )
    return tuple(json.loads(text)["compounds"])


@lru_cache(maxsize=1)
def reference_provenance_note() -> str:
    text = (
        resources.files("formulate.data")
        .joinpath("reference_compounds.json")
        .read_text(encoding="utf-8")
    )
    return str(json.loads(text)["provenance"])


class ReferenceDatabaseExplorer(Explorer):
    """Proposes candidates from the bundled reference compound set."""

    id = "database:reference"
    version = "1"

    def __init__(self, exclude: Sequence[str] = ()) -> None:
        """``exclude`` hides structures from the catalogue.

        A recovery experiment that leaves the answer in the pool measures
        whether the ranking can find a compound it was handed, which is a
        different and much easier question than whether the search can reach
        one it was not. Names are matched on canonical structure, so hiding a
        compound written one way hides it written any other way.
        """
        self._excluded = frozenset(
            key for key in (_canonical(smiles) for smiles in exclude) if key
        )

    def propose(
        self,
        spec: TargetSpec,
        count: int,
        *,
        scored: Sequence[Candidate] = (),
        seed: int = 0,
    ) -> list[Candidate]:
        if MaterialClass.MOLECULE not in spec.material_classes:
            return []

        seen = {c.structure_id for c in scored}
        out: list[Candidate] = []
        for record in load_reference_compounds():
            if len(out) >= count:
                break
            if _canonical(record["smiles"]) in self._excluded:
                continue
            candidate = Candidate(
                material_class=MaterialClass.MOLECULE,
                molecule=MoleculeSpec(smiles=record["smiles"]),
                conditions=spec.conditions,
                generation_strategy=self.id,
                label=record["name"],
                provenance=ProvenanceRecord(
                    kind=ProvenanceKind.RETRIEVAL,
                    producer=self.id,
                    producer_version=self.version,
                    parameters={"source": "formulate bundled reference set", "name": record["name"]},
                ),
            )
            if candidate.structure_id in seen:
                continue
            seen.add(candidate.structure_id)
            out.append(candidate)
        return out


def _canonical(smiles: str) -> str:
    """Canonical form where RDKit is present, the string itself where it is not."""
    from formulate import chem

    if not chem.rdkit_available():
        return smiles
    return chem.canonical_smiles(smiles) or smiles
