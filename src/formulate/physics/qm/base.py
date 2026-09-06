"""The quantum backend contract.

Specification section 7 fixes both ends of it.

Inputs: "geometry, composition/charge/spin, method, basis/pseudopotential,
convergence criteria, boundary/periodic settings."

Outputs: "total energy, optimized geometry, forces/gradients, electronic
observables, convergence diagnostics, method metadata, uncertainty/known
limitations."

Two things this contract refuses to do.

It does not attach a standard deviation to a total electronic energy. An
absolute total energy is method-dependent and not comparable between methods
at all; only differences computed the same way mean anything. A sigma there
would invite exactly the comparison that is invalid.

It does not return a number for an unconverged SCF. A non-converged
calculation has an energy in memory, and reporting it as a result is how a
meaningless number reaches a ranking.
"""

from __future__ import annotations

import abc
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from formulate.core.provenance import ProvenanceRecord
from formulate.core.quantity import Quantity, Uncertainty

from ..geometry import MolecularGeometry


class QMMethod(str, Enum):
    """Electronic-structure methods, named rather than ranked.

    Section 7 is explicit: "Do not assume Hartree-Fock is a universally cheaper
    substitute for DFT or that one method is uniformly more accurate. Select
    method by system/property." So this is a set of options, not a ladder, and
    nothing in the code orders it by accuracy.
    """

    HARTREE_FOCK = "hf"
    DFT = "dft"
    #: Semi-empirical tight binding, for screening.
    GFN2_XTB = "gfn2-xtb"
    #: The teaching implementation. Never a production authority.
    MINIMAL_HF = "minimal-hf"


class ConvergenceStatus(str, Enum):
    CONVERGED = "converged"
    NOT_CONVERGED = "not_converged"
    FAILED = "failed"


class QMRequest(BaseModel):
    """What to compute, and to what standard."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    geometry: MolecularGeometry
    method: QMMethod = QMMethod.DFT
    #: Gaussian basis set name, for the methods that take one.
    basis: str = "6-31g"
    #: Exchange-correlation functional, for DFT.
    xc: str = "b3lyp"
    #: SCF energy convergence threshold, in Hartree.
    convergence: float = 1e-9
    max_scf_cycles: int = 100
    #: Relax the geometry before reporting observables.
    optimize_geometry: bool = False
    max_optimization_steps: int = 50
    #: Force convergence for the optimiser, in eV/Angstrom.
    force_threshold: float = 0.05
    #: Ask for nuclear gradients. They cost roughly another energy evaluation.
    compute_forces: bool = False

    def method_signature(self) -> str:
        """Short label identifying the level of theory."""
        if self.method is QMMethod.DFT:
            return f"{self.xc}/{self.basis}"
        if self.method in (QMMethod.HARTREE_FOCK, QMMethod.MINIMAL_HF):
            return f"HF/{self.basis}"
        return self.method.value

    def cache_payload(self) -> dict[str, Any]:
        return {
            "geometry": self.geometry.identity_payload(),
            "method": self.method.value,
            "basis": self.basis,
            "xc": self.xc,
            "convergence": self.convergence,
            "optimize_geometry": self.optimize_geometry,
            "compute_forces": self.compute_forces,
        }


class QMResult(BaseModel):
    """Everything a quantum calculation produced, including how it went."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    status: ConvergenceStatus
    method_signature: str
    backend: str
    backend_version: str = ""

    #: Total electronic energy. Comparable only against the same method and basis.
    total_energy: Quantity | None = None
    homo_lumo_gap: Quantity | None = None
    dipole_moment: Quantity | None = None
    #: Nuclear gradients in eV/Angstrom, shape (n_atoms, 3), when requested.
    forces: Any = None
    #: The relaxed structure, when optimisation was asked for and succeeded.
    optimized_geometry: MolecularGeometry | None = None
    #: Largest residual force after optimisation, eV/Angstrom.
    max_residual_force: float | None = None

    scf_cycles: int | None = None
    wall_time_seconds: float | None = None
    #: What went wrong, or what the caller should know went nearly wrong.
    diagnostics: tuple[str, ...] = ()
    #: Statements about what this level of theory can and cannot support.
    limitations: tuple[str, ...] = ()
    provenance: ProvenanceRecord | None = None

    @property
    def usable(self) -> bool:
        return self.status is ConvergenceStatus.CONVERGED and self.total_energy is not None

    def energy_uncertainty(self) -> Uncertainty:
        """Deliberately unquantified, with the reason recorded.

        A total electronic energy has no meaningful error bar in isolation: it
        is not an estimate of a measurable quantity. Reporting one would imply
        cross-method comparability that does not exist.
        """
        return Uncertainty(
            basis=(
                "absolute electronic energies are method- and basis-dependent and are not "
                "comparable across levels of theory; only differences computed at the same "
                "level carry meaning, so no interval is quoted here"
            )
        )

    def describe(self) -> str:
        if not self.usable:
            return (
                f"{self.method_signature} on {self.backend}: {self.status.value}"
                + (f" - {'; '.join(self.diagnostics)}" if self.diagnostics else "")
            )
        bits = [f"{self.method_signature} ({self.backend})", f"E = {self.total_energy}"]
        if self.homo_lumo_gap is not None:
            bits.append(f"gap = {self.homo_lumo_gap}")
        if self.dipole_moment is not None:
            bits.append(f"dipole = {self.dipole_moment}")
        if self.wall_time_seconds is not None:
            bits.append(f"{self.wall_time_seconds:.1f} s")
        return "  ".join(bits)


class QMBackend(abc.ABC):
    """A quantum-chemistry engine behind a uniform interface."""

    id: str = "qm"
    version: str = "0"
    #: False for anything that must never be treated as a scientific authority.
    production: bool = True

    @abc.abstractmethod
    def is_available(self) -> bool: ...

    def unavailable_reason(self) -> str:
        return ""

    @abc.abstractmethod
    def supported_methods(self) -> frozenset[QMMethod]: ...

    @abc.abstractmethod
    def run(self, request: QMRequest) -> QMResult:
        """Execute the calculation, returning a result even on failure."""

    def supports(self, request: QMRequest) -> bool:
        return self.is_available() and request.method in self.supported_methods()

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.id}>"
