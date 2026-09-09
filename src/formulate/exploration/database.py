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
from typing import Any, Iterator, Sequence

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MoleculeSpec,
    MonomerRole,
    MonomerUnit,
    PolymerSpec,
    PolymerTopology,
    Tacticity,
)
from formulate.core.provenance import ProvenanceKind, ProvenanceRecord
from formulate.core.quantity import Quantity
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
def load_monomers() -> tuple[dict[str, Any], ...]:
    """The polymerisable monomers, kept in a file of their own.

    They are catalogue entries and not calibration data, and the separation is
    load-bearing rather than tidy. ``reference_compounds.json`` is the held-out
    set that every measured uncertainty in this repository was calibrated
    against - "mean absolute error over the fifty reference compounds" appears
    in a dozen expert docstrings - so adding structures to it silently
    invalidates all of them without recomputing any. These are proposed to the
    search and never scored against.

    The catalogue had one monomer in it, styrene, which meant a target selected
    on a polymerisation rate could not be answered by anything: the property
    existed, the expert existed, and the pool had nothing to apply them to.
    """
    text = (
        resources.files("formulate.data")
        .joinpath("monomers.json")
        .read_text(encoding="utf-8")
    )
    return tuple(json.loads(text)["compounds"])


@lru_cache(maxsize=1)
def catalogue() -> tuple[dict[str, Any], ...]:
    """Everything the search may propose, reference compounds first.

    Deduplicated on structure: styrene is a solvent in one file and a monomer
    in the other, and it is one candidate.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in load_reference_compounds() + load_monomers():
        key = _canonical(record["smiles"])
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return tuple(out)


@lru_cache(maxsize=1)
def load_polymers() -> tuple[dict[str, Any], ...]:
    """The polymer candidate pool, kept in a file of its own.

    Same separation, and for the same reason, as :func:`load_monomers`: these
    are catalogue entries proposed to the search, never scored against.  They
    are kept apart from ``reference_polymers.json`` as well, which is the fit
    and validation data behind the group-contribution glass transition - a
    candidate that ends up in the fitting set would be reported with an
    in-sample error, which is not an error at all.

    The pool exists because a target stating ``material_classes: [polymer]``
    had nothing to propose.  Every polymer expert in the panel was reachable
    only by a caller who constructed a :class:`PolymerSpec` by hand, which
    meant the search could not answer a polymer question at all: the properties
    existed, the experts existed, and the explorer returned an empty list.
    """
    text = (
        resources.files("formulate.data")
        .joinpath("polymers.json")
        .read_text(encoding="utf-8")
    )
    return tuple(json.loads(text)["polymers"])


@lru_cache(maxsize=1)
def polymer_provenance_note() -> str:
    text = (
        resources.files("formulate.data")
        .joinpath("polymers.json")
        .read_text(encoding="utf-8")
    )
    return str(json.loads(text)["provenance"])


def polymer_spec(record: dict[str, Any]) -> PolymerSpec:
    """Turn one catalogue row into the payload the polymer experts read.

    Every field the row states is carried through, including the ones that
    make a model refuse.  A network's topology and crosslink density are what
    the mechanical and transition experts consult before declining, and an
    isotactic tacticity is what puts a polymer outside a correlation fitted on
    atactic ones; dropping them here to get more answers out of the panel
    would be manufacturing the answers.
    """
    monomers = tuple(
        MonomerUnit(
            smiles=unit["smiles"],
            mole_fraction=unit["mole_fraction"],
            role=MonomerRole(unit.get("role", MonomerRole.BACKBONE.value)),
        )
        for unit in record["repeat_units"]
    )
    mn = record.get("number_average_molar_mass_g_mol")
    crosslink = record.get("crosslink_density_mol_m3")
    return PolymerSpec(
        monomers=monomers,
        topology=PolymerTopology(record.get("topology", PolymerTopology.LINEAR.value)),
        tacticity=Tacticity(record.get("tacticity", Tacticity.UNSPECIFIED.value)),
        number_average_molar_mass=None if mn is None else Quantity(value=mn, unit="g/mol"),
        crosslink_density=(
            None if crosslink is None else Quantity(value=crosslink, unit="mol/m^3")
        ),
    )


@lru_cache(maxsize=1)
def reference_provenance_note() -> str:
    text = (
        resources.files("formulate.data")
        .joinpath("reference_compounds.json")
        .read_text(encoding="utf-8")
    )
    return str(json.loads(text)["provenance"])


class ReferenceDatabaseExplorer(Explorer):
    """Proposes candidates from the bundled reference sets.

    Molecules come from the compound and monomer catalogues, polymers from the
    polymer catalogue, and which of the two a target gets is decided by the
    material classes it declares rather than by anything this explorer prefers.
    A spec asking for both gets both, in one pool, competing on the same
    requirements - which is the only way a thermoplastic and a curable monomer
    can be compared at all.
    """

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
        seen = {c.structure_id for c in scored}
        out: list[Candidate] = []

        if MaterialClass.MOLECULE in spec.material_classes:
            self._extend(out, self._molecules(spec), count, seen)
        if MaterialClass.POLYMER in spec.material_classes:
            self._extend(out, self._polymers(spec), count, seen)
        return out

    @staticmethod
    def _extend(
        out: list[Candidate],
        source: Iterator[Candidate],
        count: int,
        seen: set[str],
    ) -> None:
        for candidate in source:
            if len(out) >= count:
                return
            if candidate.structure_id in seen:
                continue
            seen.add(candidate.structure_id)
            out.append(candidate)

    def _molecules(self, spec: TargetSpec) -> Iterator[Candidate]:
        for record in catalogue():
            if _canonical(record["smiles"]) in self._excluded:
                continue
            yield Candidate(
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

    def _polymers(self, spec: TargetSpec) -> Iterator[Candidate]:
        """Polymer candidates, excluded on any repeat unit they are built from.

        Two polymers in the catalogue share a repeat unit and differ only in
        architecture - the two polyethylenes - so excluding by structure hides
        both, which is the right behaviour for a recovery experiment: leaving
        the branched one in the pool while hiding the linear one would leave
        the answer half in.
        """
        for record in load_polymers():
            units = [unit["smiles"] for unit in record["repeat_units"]]
            if any(_canonical(smiles) in self._excluded for smiles in units):
                continue
            yield Candidate(
                material_class=MaterialClass.POLYMER,
                polymer=polymer_spec(record),
                conditions=spec.conditions,
                generation_strategy=self.id,
                label=record["name"],
                provenance=ProvenanceRecord(
                    kind=ProvenanceKind.RETRIEVAL,
                    producer=self.id,
                    producer_version=self.version,
                    parameters={
                        "source": "formulate bundled polymer catalogue",
                        "name": record["name"],
                        "role": record.get("role", ""),
                    },
                ),
            )


def _canonical(smiles: str) -> str:
    """Canonical form where RDKit is present, the string itself where it is not."""
    from formulate import chem

    if not chem.rdkit_available():
        return smiles
    return chem.canonical_smiles(smiles) or smiles
