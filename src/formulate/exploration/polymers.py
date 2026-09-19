"""Proposing single polymers.

There was a hole here. ``ReferenceDatabaseExplorer`` proposes molecules and
``PolymerBlendExplorer`` proposes mixtures, so a specification that asked for
``material_classes: [polymer]`` - the one class every polymer expert in the
repository is registered against - drew *nothing*. Not "nothing feasible",
which is an answer; nothing proposed, which is silence. The polymer experts
could only ever be reached through a blend, which is the wrong way round: a
blend is a formulation of polymers, so the polymers have to come first.

The axis this explorer sweeps is chain length, because that is the one
variable a repeat unit does not fix and the one that moves melt behaviour by
orders of magnitude. The same repeat unit at 2 kg/mol is a wax and at 200
kg/mol is unprocessable; the viscosity between them spans six decades. A
library of repeat units with no molar mass attached is not a library of
polymers, so each repeat unit is proposed at several lengths and the ranker
decides which length the requirements actually wanted.

Tacticity is swept for the repeat units where it decides whether the polymer
crystallises at all, and left unspecified otherwise. The melt expert refuses
an unspecified tacticity for exactly those repeat units, so proposing only the
unspecified form would make them permanently unanswerable.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Sequence

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MonomerUnit,
    PolymerSpec,
    Tacticity,
)
from formulate.core.provenance import ProvenanceKind, ProvenanceRecord
from formulate.core.quantity import Quantity
from formulate.targets.spec import TargetSpec

from .base import Explorer

#: Chain lengths proposed for every repeat unit, kg/mol.
#:
#: These are decade-ish steps across the range that matters, and they are
#: chosen around the transitions rather than evenly: 2 is below the
#: entanglement mass of almost everything (a wax), 20 and 50 bracket ordinary
#: commercial grades, 120 is a high grade, and 400 is where melt spinning
#: starts to be impossible for pressure reasons - which is a conclusion the
#: engine should be able to reach rather than one the explorer assumes.
MOLAR_MASSES: tuple[float, ...] = (2.0, 8.0, 20.0, 50.0, 120.0, 400.0)


class PolymerLibraryExplorer(Explorer):
    """Proposes single polymers from the bundled repeat-unit library."""

    id = "library:polymer"
    version = "1"

    def __init__(self, *, molar_masses: Sequence[float] = MOLAR_MASSES) -> None:
        self.molar_masses = tuple(molar_masses)

    @staticmethod
    def _records() -> list[dict]:
        text = (
            resources.files("formulate.data")
            .joinpath("reference_polymers.json")
            .read_text(encoding="utf-8")
        )
        return list(json.loads(text)["polymers"])

    @staticmethod
    def _tacticities(repeat: str) -> tuple[Tacticity, ...]:
        from formulate.experts.melt import TACTICITY_DECIDES_CRYSTALLINITY

        if repeat in TACTICITY_DECIDES_CRYSTALLINITY:
            return (Tacticity.ISOTACTIC, Tacticity.ATACTIC)
        return (Tacticity.UNSPECIFIED,)

    def propose(
        self,
        spec: TargetSpec,
        count: int,
        *,
        scored: Sequence[Candidate] = (),
        seed: int = 0,
    ) -> list[Candidate]:
        if MaterialClass.POLYMER not in spec.material_classes:
            return []

        seen = {c.structure_id for c in scored}
        out: list[Candidate] = []

        # Chain length is the outer loop so that a truncated draw still spans
        # the whole library rather than returning six grades of polyethylene.
        for mass in self.molar_masses:
            for record in self._records():
                repeat = record["repeat_unit"]
                for tacticity in self._tacticities(repeat):
                    if len(out) >= count:
                        return out
                    label = f"{record['name']} {mass:g} kg/mol"
                    if tacticity is not Tacticity.UNSPECIFIED:
                        label = f"{label}, {tacticity.value}"
                    candidate = Candidate(
                        material_class=MaterialClass.POLYMER,
                        polymer=PolymerSpec(
                            monomers=(MonomerUnit(smiles=repeat),),
                            tacticity=tacticity,
                            number_average_molar_mass=Quantity(value=mass, unit="kg/mol"),
                        ),
                        conditions=spec.conditions,
                        generation_strategy=self.id,
                        label=label,
                        provenance=ProvenanceRecord(
                            kind=ProvenanceKind.RETRIEVAL,
                            producer=self.id,
                            producer_version=self.version,
                            parameters={
                                "source": "formulate bundled polymer reference set",
                                "name": record["name"],
                                "number_average_molar_mass_kg_mol": mass,
                                "tacticity": tacticity.value,
                            },
                        ),
                    )
                    if candidate.structure_id in seen:
                        continue
                    seen.add(candidate.structure_id)
                    out.append(candidate)
        return out
