"""The dynamics contract and what each protocol legitimately needs.

Specification section 6 lists the inputs ("atomic topology/coordinates, force
field or potential, boundary conditions, ensemble, temperature/pressure,
timestep, equilibration and sampling plans") and the property workflows.

The part that carries the most weight is the feasibility check.  Section 6
insists that "candidate preparation must construct representative polymer
chains, mixture boxes, interfaces/surfaces, or condensed phases; a single
isolated molecule is insufficient for many bulk properties", and section 13
forbids validating bulk behaviour from an isolated molecule.  A protocol
therefore declares what system and how much sampling it needs, and the engine
refuses to return a number when the installation cannot supply it.

Refusing is the whole point.  A density computed from eight molecules for a
picosecond is not an approximate density; it is a number with no relationship
to the quantity requested, and returning it with a wide error bar would still
be wrong, because the error is systematic and not sampling noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from formulate.core.provenance import ProvenanceRecord
from formulate.core.quantity import Quantity

from ..geometry import MolecularGeometry
from .statistics import BlockAverage


class Ensemble(str, Enum):
    NVE = "nve"
    NVT = "nvt"
    NPT = "npt"


class MDProtocol(str, Enum):
    """Property workflows from specification section 6."""

    #: Torsional and conformational sampling of one molecule.
    CONFORMATIONAL_ENSEMBLE = "conformational_ensemble"
    #: Binding energy per molecule of a finite cluster.
    COHESIVE_ENERGY = "cohesive_energy"
    #: Bulk mass density. Needs a periodic condensed phase.
    DENSITY = "density"
    #: Self-diffusion from the Einstein relation. Needs a periodic phase and
    #: enough time to reach the diffusive regime.
    SELF_DIFFUSION = "self_diffusion"
    #: Work of separation across an interface.
    WORK_OF_SEPARATION = "work_of_separation"


@dataclass(frozen=True, slots=True)
class ProtocolRequirement:
    """What a protocol needs before its output means anything."""

    needs_periodic: bool
    #: Molecules a condensed-phase estimate needs before finite-size error is
    #: smaller than the quantity being measured.
    min_molecules: int
    #: Production sampling, in picoseconds, for the observable to converge.
    min_production_ps: float
    rationale: str


#: Requirements are literature norms for each observable, not this sandbox's
#: budget. They are what makes an infeasibility verdict a scientific statement
#: rather than a complaint about hardware.
REQUIREMENTS: dict[MDProtocol, ProtocolRequirement] = {
    MDProtocol.CONFORMATIONAL_ENSEMBLE: ProtocolRequirement(
        needs_periodic=False,
        min_molecules=1,
        min_production_ps=100.0,
        rationale=(
            "torsional barriers of a few kcal/mol are crossed on a nanosecond timescale, "
            "so a short run samples one well rather than the ensemble"
        ),
    ),
    MDProtocol.COHESIVE_ENERGY: ProtocolRequirement(
        needs_periodic=False,
        min_molecules=8,
        min_production_ps=10.0,
        rationale=(
            "a cluster estimate converges toward the bulk value only as the surface "
            "fraction falls, so the number of molecules sets a systematic error that "
            "longer sampling cannot remove"
        ),
    ),
    MDProtocol.DENSITY: ProtocolRequirement(
        needs_periodic=True,
        min_molecules=250,
        min_production_ps=500.0,
        rationale=(
            "a liquid density needs a periodic box under pressure control; a finite "
            "cluster has a surface, and its average density depends on where the "
            "boundary is drawn"
        ),
    ),
    MDProtocol.SELF_DIFFUSION: ProtocolRequirement(
        needs_periodic=True,
        min_molecules=250,
        min_production_ps=1000.0,
        rationale=(
            "the Einstein relation applies only in the diffusive regime, which follows "
            "a ballistic and a cage-rattling regime; fitting before it is reached "
            "returns a number that is not a diffusion coefficient"
        ),
    ),
    MDProtocol.WORK_OF_SEPARATION: ProtocolRequirement(
        needs_periodic=True,
        min_molecules=400,
        min_production_ps=500.0,
        rationale=(
            "an interface needs two slabs thick enough that each has a bulk-like "
            "interior, plus enough sampling to relax the contact"
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class FeasibilityVerdict:
    """Whether a protocol can be run well enough to mean anything here."""

    feasible: bool
    reason: str
    #: What a scientifically adequate run would cost, in seconds of wall time.
    adequate_cost_seconds: float | None = None
    #: What the installation can actually afford, for contrast.
    affordable_description: str = ""

    def describe(self) -> str:
        if self.feasible:
            return f"feasible: {self.reason}"
        cost = ""
        if self.adequate_cost_seconds is not None:
            cost = f" An adequate run would take about {_duration(self.adequate_cost_seconds)}."
        return f"not feasible here: {self.reason}{cost}"


def _duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f} seconds"
    if seconds < 7200:
        return f"{seconds / 60:.0f} minutes"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.0f} days"


class MDRequest(BaseModel):
    """A dynamics run to perform."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    geometry: MolecularGeometry
    protocol: MDProtocol = MDProtocol.CONFORMATIONAL_ENSEMBLE
    ensemble: Ensemble = Ensemble.NVT
    temperature_k: float = 298.15
    pressure_pa: float | None = None
    timestep_fs: float = 1.0
    equilibration_steps: int = 2000
    production_steps: int = 10000
    sample_interval: int = 10
    #: Force provider label, or None to take the cheapest available.
    calculator: str | None = None
    #: Molecules to place in a cluster, for the protocols that use one.
    n_molecules: int = 1
    friction: float = 0.02
    seed: int = 0
    #: Run even when the feasibility check says the result cannot mean anything.
    #: The result is still marked unusable; this exists for method development,
    #: not for producing numbers.
    force_run: bool = False

    @property
    def production_ps(self) -> float:
        return self.production_steps * self.timestep_fs / 1000.0


class MDResult(BaseModel):
    """What a dynamics run produced, or why it produced nothing."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    protocol: MDProtocol
    calculator: str
    feasible: bool
    feasibility_reason: str
    #: Named observables with block-averaged uncertainty.
    observables: dict[str, Any] = {}
    #: The primary quantity, when the protocol produced one it stands behind.
    value: Quantity | None = None
    sampling: BlockAverage | None = None
    #: Frames discarded as equilibration.
    discarded_frames: int = 0
    production_ps: float = 0.0
    n_molecules: int = 1
    wall_time_seconds: float | None = None
    diagnostics: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    provenance: ProvenanceRecord | None = None

    @property
    def usable(self) -> bool:
        """True only when the run was feasible and produced a converged value."""
        return self.feasible and self.value is not None

    def describe(self) -> str:
        if not self.feasible:
            return f"{self.protocol.value}: {self.feasibility_reason}"
        if self.value is None:
            return f"{self.protocol.value}: no value ({'; '.join(self.diagnostics)})"
        sampling = f"  {self.sampling.describe()}" if self.sampling else ""
        return (
            f"{self.protocol.value} [{self.calculator}]: {self.value}"
            f"  over {self.production_ps:.1f} ps{sampling}"
        )
