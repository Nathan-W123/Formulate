"""Evidence store.

Specification section 11 requires every prediction, simulation, transformation
and score to be reproducible; section 8 has validated physics results feeding
"the evidence store" and becoming training data later.

A run is written as one JSON document holding the target, the candidates, their
predictions with provenance, and the ranking.  That document is the record: it
should be sufficient to explain, months later, why a particular recipe was
recommended and on what evidence.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from formulate import SPEC_VERSION, __version__
from formulate.core.hashing import content_hash

if TYPE_CHECKING:  # pragma: no cover
    from formulate.coordination.coordinator import DesignRun


class EvidenceStore:
    """Writes and reads run records under a directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def record(self, run: "DesignRun") -> dict[str, Any]:
        """Build the serialisable record for a run."""
        entries = []
        for entry in run.ranking.ranked:
            candidate = entry.candidate
            entries.append(
                {
                    "rank": entry.rank,
                    "front": entry.front,
                    "feasible": entry.feasible,
                    "scalar_baseline": entry.scalar,
                    "candidate": candidate.model_dump(mode="json"),
                    "candidate_id": candidate.candidate_id,
                    "structure_id": candidate.structure_id,
                }
            )

        document = {
            "schema": "formulate/design-run/1",
            "formulate_version": __version__,
            "spec_version": SPEC_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target": run.spec.model_dump(mode="json"),
            "pipeline": {
                "proposed": run.proposed,
                "evaluated": run.evaluated,
                "filtered_out": len(run.filters.rejected),
                "rejection_reasons": run.filters.rejection_counts,
                "experts_run": list(run.dispatch.experts_run),
                "uncovered_properties": list(run.dispatch.uncovered_properties),
                "unavailable_experts": run.dispatch.unavailable_experts,
            },
            "ranking": {
                "axes": list(run.ranking.axes),
                "dropped_axes": list(run.ranking.dropped_axes),
                "hypervolume": run.ranking.hypervolume,
                "feasible_count": run.ranking.feasible_count,
                "infeasible_count": run.ranking.infeasible_count,
                "mean_structural_distance": run.ranking.mean_diversity,
                "pool_relative_properties": list(run.pool_relative_properties),
            },
            "feasibility": {
                "feasible_count": run.feasibility.feasible_count,
                "evaluated": run.feasibility.evaluated,
                "diagnoses": [
                    {
                        "property": d.property,
                        "requirement": d.requirement,
                        "eliminated": d.eliminated,
                        "would_admit": d.would_admit,
                        "suggested_bound": d.suggested_bound,
                        "observed": list(d.observed) if d.observed else None,
                    }
                    for d in run.feasibility.diagnoses
                ],
            },
            "candidates": entries,
        }
        document["record_id"] = content_hash(
            {k: v for k, v in document.items() if k != "created_at"}, prefix="run"
        )
        return document

    def write(self, run: "DesignRun", *, name: str | None = None) -> Path:
        """Persist a run and return the path written."""
        document = self.record(run)
        self.root.mkdir(parents=True, exist_ok=True)
        filename = f"{name or document['record_id']}.json"
        path = self.root / filename
        path.write_text(json.dumps(document, indent=2, sort_keys=False), encoding="utf-8")
        return path

    def read(self, path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def list_runs(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(self.root.glob("*.json"))
