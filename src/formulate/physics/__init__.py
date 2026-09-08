"""Physics validation: quantum and molecular-dynamics evidence.

Specification section 9 stages 8 and 9: run QM/MD on a small high-value or
uncertain subset, then replace or augment the expert predictions with what the
physics found and quantify the disagreement.

Nothing here reimplements a production solver (section 13); the modules
integrate established open packages and contribute the candidate preparation,
protocol, uncertainty and provenance handling around them.
"""

from .qmmm import (
    EmbeddingMode,
    LinkAtom,
    QMMMCalculator,
    QMMMRegion,
    QMMMResult,
    UnsupportedPartition,
    partition,
)
from .backends import (
    BackendCapability,
    BackendKind,
    available_backends,
    describe,
    get_backend_capability,
    periodic_backends,
    probe_backends,
)

__all__ = [
    "BackendCapability", "BackendKind", "EmbeddingMode", "LinkAtom", "QMMMCalculator",
    "QMMMRegion", "QMMMResult", "UnsupportedPartition", "available_backends",
    "describe", "get_backend_capability", "partition", "periodic_backends",
    "probe_backends",
]
