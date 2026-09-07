"""Content-addressed result cache.

Specification section 11: "content-address candidates + method + conditions so
identical calculations are never repeated."

The key covers the candidate, the expert and its version, the conditions, and
the properties the expert was asked for. Bumping an expert's ``version``
therefore invalidates exactly its own entries and nothing else, which is what
makes it safe to change a correlation without hand-clearing a cache.

The property set belongs in the key because the value stored under it is what
the expert returned *for that ask*: an expert answers the intersection of its
coverage with the request, and sees only the upstream predictions that request
produced. Without it, a cache persisted from one target specification would
serve a later one a partial result.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable

from formulate.core.conditions import Conditions
from formulate.core.hashing import cache_key
from formulate.core.prediction import Prediction


class PredictionCache:
    """In-memory cache with optional append-only persistence.

    Thread-safe: the evaluation engine dispatches experts across candidates
    concurrently and all of them share one cache.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._entries: dict[str, list[Prediction]] = {}
        self._lock = threading.Lock()
        self._path = Path(path) if path is not None else None
        self.hits = 0
        self.misses = 0
        if self._path is not None and self._path.exists():
            self._load()

    @staticmethod
    def key(
        candidate_id: str,
        expert_id: str,
        expert_version: str,
        conditions: Conditions,
        properties: Iterable[str] = (),
    ) -> str:
        return cache_key(
            candidate_hash=candidate_id,
            method=expert_id,
            method_version=expert_version,
            conditions=conditions.identity_payload(),
            properties=properties,
        )

    def get(self, key: str) -> list[Prediction] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self.hits += 1
            return list(entry)

    def put(self, key: str, predictions: Iterable[Prediction]) -> None:
        items = list(predictions)
        with self._lock:
            self._entries[key] = items
            if self._path is not None:
                self._append(key, items)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    # -- persistence -------------------------------------------------------

    def _append(self, key: str, predictions: list[Prediction]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {"key": key, "predictions": [p.model_dump(mode="json") for p in predictions]}
        )
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _load(self) -> None:
        for raw in self._path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
                self._entries[record["key"]] = [
                    Prediction.model_validate(p) for p in record["predictions"]
                ]
            except Exception:
                # A corrupt line loses one cache entry, not the whole run.
                continue
