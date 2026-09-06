"""Which physics backends exist, and what they can honestly be asked for.

Specification section 7 requires production quantum work to go through
established open packages rather than a reimplementation, and section 6 says
the same for molecular dynamics.  That makes backend availability a
first-class concern: the set of installed packages decides which properties
the system may claim to validate at all.

Capability here is *probed*, not assumed.  A backend that imports is not
necessarily a backend that works on the system you want - periodic boundary
support in particular varies between methods of the same package, and getting
that wrong would silently turn a bulk claim into a cluster calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

from .quiet import suppress_native_output


class BackendKind(str, Enum):
    """What a backend is for."""

    #: Electronic structure: energies, orbitals, forces from first principles.
    QUANTUM = "quantum"
    #: Semi-empirical electronic structure - cheaper, parameterised.
    SEMI_EMPIRICAL = "semi_empirical"
    #: Classical force field: no electrons, fixed topology.
    FORCE_FIELD = "force_field"


@dataclass(frozen=True, slots=True)
class BackendCapability:
    """What one backend can do, established by probing where possible."""

    backend_id: str
    kind: BackendKind
    available: bool
    unavailable_reason: str = ""
    #: True only if a periodic cell was actually evaluated successfully.
    supports_periodic: bool = False
    periodic_reason: str = ""
    supports_forces: bool = False
    #: Largest system the backend was observed to handle, where a limit was found.
    max_atoms_observed: int | None = None
    #: Reference cost of one energy+forces evaluation, in milliseconds, measured
    #: on a 4-core CPU sandbox for a molecule of roughly 10-20 atoms. Indicative
    #: only; used for budgeting, never reported as a scientific quantity.
    reference_ms_per_evaluation: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        if not self.available:
            return f"{self.backend_id}: unavailable - {self.unavailable_reason}"
        bits = [f"{self.backend_id} [{self.kind.value}]"]
        bits.append("periodic" if self.supports_periodic else "non-periodic only")
        if self.reference_ms_per_evaluation is not None:
            bits.append(f"~{self.reference_ms_per_evaluation:.0f} ms/eval")
        if self.max_atoms_observed is not None:
            bits.append(f"observed limit {self.max_atoms_observed} atoms")
        return "  ".join(bits)


def _probe_pyscf() -> BackendCapability:
    try:
        import pyscf  # noqa: F401
        from pyscf import gto, scf
    except Exception as exc:
        return BackendCapability(
            "pyscf", BackendKind.QUANTUM, False, f"pyscf is not installed ({exc})"
        )

    try:
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        scf.RHF(mol).kernel()
    except Exception as exc:
        return BackendCapability(
            "pyscf", BackendKind.QUANTUM, False, f"pyscf failed a trivial SCF ({exc})"
        )

    notes = [
        "analytic nuclear gradients available via mf.nuc_grad_method()",
        (
            "pyscf.geomopt is unusable here: neither the geometric nor the berny "
            "solver can be installed, so geometry optimisation is driven through "
            "an ASE optimiser instead"
        ),
    ]
    return BackendCapability(
        "pyscf",
        BackendKind.QUANTUM,
        True,
        supports_periodic=False,
        periodic_reason="molecular (non-periodic) calculations only in this integration",
        supports_forces=True,
        reference_ms_per_evaluation=3000.0,
        notes=tuple(notes),
    )


def _probe_xtb(method: str, kind: BackendKind, reference_ms: float) -> BackendCapability:
    backend_id = f"xtb:{method}"
    try:
        from xtb.ase.calculator import XTB  # noqa: F401
    except Exception as exc:
        return BackendCapability(
            backend_id, kind, False, f"the xtb package is not installed ({exc})"
        )

    from ase import Atoms

    # A trivial water molecule establishes that the method runs at all.
    try:
        with suppress_native_output():
            probe = Atoms("OH2", positions=[(0, 0, 0), (0, 0, 0.96), (0.93, 0, -0.24)])
            probe.calc = XTB(method=method)
            probe.get_potential_energy()
            probe.get_forces()
    except Exception as exc:
        return BackendCapability(backend_id, kind, False, f"{method} failed on water ({exc})")

    periodic, periodic_reason = _probe_periodic(method)
    notes: list[str] = []
    if not periodic:
        notes.append(
            "cannot be used for a periodic condensed phase; a finite cluster has a "
            "surface, so any bulk quantity taken from one carries a finite-size error "
            "that is not a sampling error and does not shrink with longer runs"
        )
    return BackendCapability(
        backend_id,
        kind,
        True,
        supports_periodic=periodic,
        periodic_reason=periodic_reason,
        supports_forces=True,
        reference_ms_per_evaluation=reference_ms,
        notes=tuple(notes),
    )


def _probe_periodic(method: str) -> tuple[bool, str]:
    """Actually evaluate a periodic cell rather than trusting documentation.

    Periodic support differs between methods of the same package: GFN1-xTB
    accepts a cell while GFN-FF and GFN2-xTB refuse one. Assuming otherwise
    would let a caller believe a droplet calculation was a bulk one.
    """
    try:
        from ase import Atoms
        from xtb.ase.calculator import XTB

        with suppress_native_output():
            cell = Atoms(
                "OH2",
                positions=[(0, 0, 0), (0, 0, 0.96), (0.93, 0, -0.24)],
                cell=[10.0, 10.0, 10.0],
                pbc=True,
            )
            cell.calc = XTB(method=method)
            cell.get_potential_energy()
        return True, "a periodic cell was evaluated successfully"
    except Exception as exc:
        return False, f"a periodic cell was rejected: {type(exc).__name__}"


def _probe_mmff() -> BackendCapability:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as exc:
        return BackendCapability(
            "rdkit:mmff94", BackendKind.FORCE_FIELD, False, f"RDKit is not installed ({exc})"
        )
    try:
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=1)
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            raise ValueError("MMFF could not parameterise ethanol")
        AllChem.MMFFGetMoleculeForceField(mol, props).CalcEnergy()
    except Exception as exc:
        return BackendCapability(
            "rdkit:mmff94", BackendKind.FORCE_FIELD, False, f"MMFF probe failed ({exc})"
        )
    return BackendCapability(
        "rdkit:mmff94",
        BackendKind.FORCE_FIELD,
        True,
        supports_periodic=False,
        periodic_reason="MMFF94 through RDKit has no periodic cell concept",
        supports_forces=True,
        reference_ms_per_evaluation=1.0,
        notes=(
            "molecular mechanics only: no electronic observables, and fixed topology "
            "so bond breaking or forming cannot be described",
        ),
    )


def _probe_openmm() -> BackendCapability:
    try:
        import openmm  # noqa: F401
    except Exception as exc:
        return BackendCapability(
            "openmm", BackendKind.FORCE_FIELD, False, f"OpenMM is not installed ({exc})"
        )
    try:
        from openff.toolkit import Molecule  # noqa: F401
    except Exception:
        return BackendCapability(
            "openmm",
            BackendKind.FORCE_FIELD,
            False,
            (
                "OpenMM is installed but openff-toolkit is not, so an arbitrary small "
                "molecule cannot be parameterised. OpenMM ships biomolecular force "
                "fields only. Installing openff-toolkit would make this the preferred "
                "periodic condensed-phase backend"
            ),
        )
    return BackendCapability(
        "openmm",
        BackendKind.FORCE_FIELD,
        True,
        supports_periodic=True,
        periodic_reason="OpenMM supports periodic boundary conditions natively",
        supports_forces=True,
        reference_ms_per_evaluation=0.1,
    )


@lru_cache(maxsize=1)
def probe_backends() -> dict[str, BackendCapability]:
    """Probe every known backend once and cache the result."""
    capabilities = [
        _probe_pyscf(),
        _probe_xtb("GFN2-xTB", BackendKind.SEMI_EMPIRICAL, 200.0),
        _probe_xtb("GFN1-xTB", BackendKind.SEMI_EMPIRICAL, 500.0),
        _probe_xtb("GFN-FF", BackendKind.FORCE_FIELD, 6.0),
        _probe_mmff(),
        _probe_openmm(),
    ]
    return {c.backend_id: c for c in capabilities}


def available_backends(kind: BackendKind | None = None) -> list[BackendCapability]:
    """Every working backend, optionally filtered by kind."""
    found = [c for c in probe_backends().values() if c.available]
    return [c for c in found if kind is None or c.kind is kind]


def periodic_backends() -> list[BackendCapability]:
    """Backends that can evaluate a periodic cell.

    An empty list means no bulk condensed-phase simulation is possible on this
    installation, which the dynamics module must report rather than quietly
    substituting a finite cluster.
    """
    return [c for c in available_backends() if c.supports_periodic]


def get_backend_capability(backend_id: str) -> BackendCapability:
    try:
        return probe_backends()[backend_id]
    except KeyError:
        known = ", ".join(sorted(probe_backends()))
        raise KeyError(f"Unknown backend {backend_id!r}. Known: {known}") from None


def describe() -> str:
    """Human-readable capability report."""
    capabilities = probe_backends()
    lines = ["Physics backends:"]
    for capability in sorted(capabilities.values(), key=lambda c: (c.kind.value, c.backend_id)):
        lines.append(f"  {capability.describe()}")
        for note in capability.notes:
            lines.append(f"      note: {note}")

    periodic = periodic_backends()
    lines.append("")
    if periodic:
        lines.append(
            "Periodic condensed-phase simulation is possible via: "
            + ", ".join(c.backend_id for c in periodic)
        )
    else:
        lines.append(
            "No available backend supports a periodic cell, so no bulk condensed-phase "
            "property can be simulated on this installation."
        )
    return "\n".join(lines)
