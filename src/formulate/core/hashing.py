"""Content addressing.

Specification section 11: "content-address candidates + method + conditions so
identical calculations are never repeated", and "every generated structure,
prediction, simulation, transformation, and score is versioned and
reproducible".

Hashes must be stable across processes and machines, so floats are serialised
at fixed precision and mappings are key-sorted.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

#: Significant digits retained when hashing floats.  Enough to distinguish
#: genuinely different compositions, coarse enough that last-bit drift between
#: platforms does not produce a cache miss.
FLOAT_PRECISION = 12

_DIGEST_SIZE = 16


def _normalize(obj: Any) -> Any:
    """Recursively convert ``obj`` into a canonical, JSON-safe structure."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "__nan__"
        if math.isinf(obj):
            return "__inf__" if obj > 0 else "__-inf__"
        # repr of a rounded float keeps 1.0 and 1 distinguishable from "1".
        return float(f"%.{FLOAT_PRECISION}g" % obj)
    if isinstance(obj, dict):
        return {str(k): _normalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_normalize(v) for v in obj)
    if hasattr(obj, "model_dump"):  # pydantic
        return _normalize(obj.model_dump(mode="json"))
    if hasattr(obj, "value") and hasattr(obj, "name"):  # enum
        return _normalize(obj.value)
    return str(obj)


def canonical_json(obj: Any) -> str:
    """Serialise ``obj`` deterministically."""
    return json.dumps(_normalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(obj: Any, *, prefix: str = "") -> str:
    """Return a stable content hash for ``obj``.

    ``prefix`` namespaces the hash so that, for example, a candidate and a
    prediction over the same payload cannot collide.
    """
    payload = canonical_json(obj).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=_DIGEST_SIZE).hexdigest()
    return f"{prefix}-{digest}" if prefix else digest


def cache_key(
    *,
    candidate_hash: str,
    method: str,
    method_version: str,
    conditions: Any,
    properties: Iterable[str] = (),
) -> str:
    """Key for a computed result: candidate + method + version + conditions + ask.

    ``properties`` is part of the key because what a method returns depends on
    what it was asked for. An expert answers the intersection of its coverage
    with the request, and it sees only the upstream predictions that request
    produced, so the same expert on the same candidate at the same conditions
    returns a different list for a different ask. Leaving it out let a cache
    persisted from one target specification serve a later one a result computed
    for a question nobody asked this time.
    """
    return content_hash(
        {
            "candidate": candidate_hash,
            "method": method,
            "method_version": method_version,
            "conditions": conditions,
            "properties": sorted(properties),
        },
        prefix="calc",
    )
