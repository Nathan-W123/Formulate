"""Score the bundled reference polymers against the web-shooter strand spec.

This script exists because ``formulate run examples/web_shooter/strand.yaml``
proposes zero candidates. The engine has two polymer experts and 57 reference
polymers with measured Tg and density, but no explorer that puts a polymer into
the pool: ``ReferenceDatabaseExplorer`` returns early unless the spec asks for
MaterialClass.MOLECULE, and ``EvolutionaryExplorer`` can only mutate a polymer
that is already there. So the polymer branch of the pipeline is unreachable
from the CLI, and the gap is exactly one explorer wide - which is what the
class below demonstrates by filling it in twenty lines.

Everything downstream is the real pipeline: the same coordinator, filters,
evaluation engine, feasibility analysis and ranker the CLI uses.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Sequence

from formulate.coordination import DeterministicCoordinator
from formulate.core.candidate import Candidate, MaterialClass, MonomerUnit, PolymerSpec
from formulate.core.quantity import Quantity
from formulate.core.provenance import ProvenanceKind, ProvenanceRecord
from formulate.exploration.base import Explorer
from formulate.targets import TargetSpec


class ReferencePolymerExplorer(Explorer):
    """Proposes the bundled reference polymers, one candidate per repeat unit."""

    id = "database:reference-polymers"
    version = "2"

    #: Melt viscosity goes as the 3.4 power of chain length, so a candidate
    #: with no stated molar mass has no melt viscosity and the expert says so.
    #: 50 kg/mol is an ordinary commercial grade and is stated, not measured -
    #: it is recorded in provenance so the run does not read it as a property
    #: of the polymer.
    number_average_molar_mass_kg_mol = 50.0

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

        text = (
            resources.files("formulate.data")
            .joinpath("reference_polymers.json")
            .read_text(encoding="utf-8")
        )
        records = json.loads(text)["polymers"]

        seen = {c.structure_id for c in scored}
        out: list[Candidate] = []
        for record in records:
            if len(out) >= count:
                break
            candidate = Candidate(
                material_class=MaterialClass.POLYMER,
                polymer=PolymerSpec(
                    monomers=(MonomerUnit(smiles=record["repeat_unit"]),),
                    number_average_molar_mass=Quantity(
                        value=self.number_average_molar_mass_kg_mol, unit="kg/mol"
                    ),
                ),
                conditions=spec.conditions,
                generation_strategy=self.id,
                label=record["name"],
                provenance=ProvenanceRecord(
                    kind=ProvenanceKind.RETRIEVAL,
                    producer=self.id,
                    producer_version=self.version,
                    parameters={
                        "source": "formulate bundled reference polymers",
                        "name": record["name"],
                        "split": record.get("split", ""),
                        "number_average_molar_mass_kg_mol":
                            self.number_average_molar_mass_kg_mol,
                    },
                ),
            )
            if candidate.structure_id in seen:
                continue
            seen.add(candidate.structure_id)
            out.append(candidate)
        return out


if __name__ == "__main__":
    import sys

    spec = TargetSpec.from_file(
        sys.argv[1] if len(sys.argv) > 1 else "examples/web_shooter/strand.yaml"
    )
    coordinator = DeterministicCoordinator(explorers=[ReferencePolymerExplorer()])
    run = coordinator.run(spec)
    print(run.report(top_k=int(sys.argv[2]) if len(sys.argv) > 2 else 5))
