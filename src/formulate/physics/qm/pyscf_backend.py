"""PySCF production backend.

Specification section 7: "Production backends: integrate established open
quantum-chemistry/DFT packages rather than reimplementing production DFT."
This module is integration only - the SCF, the integrals and the gradients are
PySCF's. What is added here is the unit discipline, the convergence
diagnostics, and the statements of what each level of theory cannot support.

Geometry optimisation goes through ASE rather than ``pyscf.geomopt``, which
needs the geometric or berny solver and neither can be installed here. The
adapter is the delicate part: PySCF works in Hartree and Bohr, ASE in
electronvolts and Angstrom, and a gradient is not a force. Both conversions
are applied in one place for that reason.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from formulate.core.provenance import ProvenanceKind, ProvenanceRecord, SoftwareEnvironment
from formulate.core.quantity import Quantity

from ..geometry import (
    HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
    HARTREE_TO_EV,
    HARTREE_TO_J_PER_MOL,
    MolecularGeometry,
    geometry_from_ase,
)
from .base import ConvergenceStatus, QMBackend, QMMethod, QMRequest, QMResult

try:  # pragma: no cover
    import pyscf
    from pyscf import dft, gto, scf

    _HAVE_PYSCF = True
    _IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    _HAVE_PYSCF = False
    _IMPORT_ERROR = str(exc)


#: Statements attached to every result at a given level of theory. These are
#: not hedging: each names a physical effect the method omits, and a caller
#: comparing against experiment needs them to interpret the difference.
_LIMITATIONS: dict[str, tuple[str, ...]] = {
    "hf": (
        "Hartree-Fock omits electron correlation entirely, so bond energies are "
        "systematically underbound and dispersion is absent",
    ),
    "dft": (
        "a semi-local or hybrid functional without a dispersion correction does not "
        "describe London forces, which matter for any weakly bound complex",
        "hybrid functionals such as B3LYP systematically underestimate frontier orbital "
        "gaps; the reported gap is an orbital-energy difference, not an optical or "
        "fundamental gap",
    ),
}

_COMMON_LIMITATIONS = (
    "computed for an isolated molecule in vacuum at zero Kelvin, with no zero-point "
    "energy, thermal, entropic or solvent contribution",
    "computed for one conformer; a flexible molecule's measured property is an ensemble "
    "average over many",
)


class PySCFBackend(QMBackend):
    """Hartree-Fock and density-functional theory through PySCF."""

    id = "pyscf"
    version = "1"
    production = True

    def is_available(self) -> bool:
        return _HAVE_PYSCF

    def unavailable_reason(self) -> str:
        return "" if _HAVE_PYSCF else f"pyscf is not installed ({_IMPORT_ERROR})"

    def supported_methods(self) -> frozenset[QMMethod]:
        return frozenset({QMMethod.HARTREE_FOCK, QMMethod.DFT})

    # -- execution ---------------------------------------------------------

    def run(self, request: QMRequest) -> QMResult:
        if not self.is_available():
            return self._failure(request, self.unavailable_reason())

        started = time.perf_counter()
        geometry = request.geometry
        diagnostics: list[str] = []

        try:
            if request.optimize_geometry:
                geometry, opt_diagnostics, residual = self._optimize(request)
                diagnostics.extend(opt_diagnostics)
            else:
                residual = None

            mol = self._build_molecule(geometry, request)
            mf = self._build_scf(mol, request)
            energy_hartree = float(mf.kernel())

            if not mf.converged:
                # A non-converged SCF still leaves a number in memory. Returning
                # it is how a meaningless energy reaches a ranking.
                return QMResult(
                    status=ConvergenceStatus.NOT_CONVERGED,
                    method_signature=request.method_signature(),
                    backend=self.id,
                    backend_version=pyscf.__version__,
                    scf_cycles=request.max_scf_cycles,
                    wall_time_seconds=time.perf_counter() - started,
                    diagnostics=tuple(
                        diagnostics
                        + [
                            f"the SCF did not converge within {request.max_scf_cycles} cycles, "
                            "so no energy is reported"
                        ]
                    ),
                    limitations=self._limitations(request),
                    provenance=self._provenance(request),
                )

            gap = self._homo_lumo_gap(mf, diagnostics)
            dipole = self._dipole(mf, diagnostics)
            forces = self._forces(mf, request, diagnostics)

            return QMResult(
                status=ConvergenceStatus.CONVERGED,
                method_signature=request.method_signature(),
                backend=self.id,
                backend_version=pyscf.__version__,
                total_energy=Quantity(
                    value=energy_hartree * HARTREE_TO_J_PER_MOL, unit="J/mol"
                ),
                homo_lumo_gap=gap,
                dipole_moment=dipole,
                forces=forces,
                optimized_geometry=geometry if request.optimize_geometry else None,
                max_residual_force=residual,
                wall_time_seconds=time.perf_counter() - started,
                diagnostics=tuple(diagnostics),
                limitations=self._limitations(request),
                provenance=self._provenance(request),
            )
        except Exception as exc:
            return self._failure(request, f"{type(exc).__name__}: {exc}", started)

    # -- construction ------------------------------------------------------

    def _build_molecule(self, geometry: MolecularGeometry, request: QMRequest) -> Any:
        return gto.M(
            atom=geometry.xyz_block(),
            basis=request.basis,
            charge=geometry.charge,
            # PySCF's `spin` is the number of unpaired electrons (2S), not the
            # multiplicity (2S+1). Passing a multiplicity here silently changes
            # the electronic state being solved.
            spin=geometry.spin_multiplicity - 1,
            unit="Angstrom",
            verbose=0,
        )

    def _build_scf(self, mol: Any, request: QMRequest) -> Any:
        open_shell = mol.spin != 0
        if request.method is QMMethod.DFT:
            mf = dft.UKS(mol) if open_shell else dft.RKS(mol)
            mf.xc = request.xc
        else:
            mf = scf.UHF(mol) if open_shell else scf.RHF(mol)
        mf.conv_tol = request.convergence
        mf.max_cycle = request.max_scf_cycles
        mf.verbose = 0
        return mf

    # -- observables -------------------------------------------------------

    def _homo_lumo_gap(self, mf: Any, diagnostics: list[str]) -> Quantity | None:
        try:
            energies = np.asarray(mf.mo_energy)
            occupations = np.asarray(mf.mo_occ)
            if energies.ndim == 2:  # unrestricted: one set of orbitals per spin
                energies = np.concatenate(energies)
                occupations = np.concatenate(occupations)
            occupied = energies[occupations > 0]
            virtual = energies[occupations == 0]
            if occupied.size == 0 or virtual.size == 0:
                diagnostics.append("no frontier gap: the basis has no virtual orbitals")
                return None
            gap_hartree = float(virtual.min() - occupied.max())
            if gap_hartree < 0:
                diagnostics.append(
                    "the highest occupied orbital lies above the lowest virtual one, which "
                    "indicates the SCF converged to an unstable solution"
                )
                return None
            return Quantity(value=gap_hartree * HARTREE_TO_EV, unit="eV")
        except Exception as exc:
            diagnostics.append(f"frontier gap unavailable ({exc})")
            return None

    def _dipole(self, mf: Any, diagnostics: list[str]) -> Quantity | None:
        try:
            vector = np.asarray(mf.dip_moment(unit="Debye", verbose=0), dtype=float)
            return Quantity(value=float(np.linalg.norm(vector)), unit="debye")
        except Exception as exc:
            diagnostics.append(f"dipole unavailable ({exc})")
            return None

    def _forces(self, mf: Any, request: QMRequest, diagnostics: list[str]) -> Any:
        if not request.compute_forces:
            return None
        try:
            gradient = np.asarray(mf.nuc_grad_method().kernel(), dtype=float)
            # PySCF returns dE/dR in Hartree/Bohr; a force is its negative.
            return (-gradient * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM).tolist()
        except Exception as exc:
            diagnostics.append(f"analytic gradients unavailable ({exc})")
            return None

    # -- geometry optimisation --------------------------------------------

    def _optimize(self, request: QMRequest) -> tuple[MolecularGeometry, list[str], float | None]:
        """Relax the structure with an ASE optimiser driven by PySCF gradients."""
        from ase.optimize import BFGS

        diagnostics: list[str] = []
        atoms = request.geometry.to_ase()
        atoms.calc = PySCFCalculator(request)

        optimizer = BFGS(atoms, logfile=None)
        try:
            optimizer.run(fmax=request.force_threshold, steps=request.max_optimization_steps)
        except Exception as exc:
            diagnostics.append(f"geometry optimisation stopped early ({exc})")

        forces = np.asarray(atoms.get_forces(), dtype=float)
        residual = float(np.abs(forces).max())
        if residual > request.force_threshold:
            diagnostics.append(
                f"geometry optimisation did not reach the force threshold: residual "
                f"{residual:.4f} eV/Angstrom against a target of {request.force_threshold:.4f}, "
                f"after {request.max_optimization_steps} steps"
            )
        relaxed = geometry_from_ase(atoms, template=request.geometry)
        return (
            relaxed.with_positions(
                relaxed.positions, source=f"{request.method_signature()} relaxed (ASE BFGS)"
            ),
            diagnostics,
            residual,
        )

    # -- bookkeeping -------------------------------------------------------

    def _limitations(self, request: QMRequest) -> tuple[str, ...]:
        key = "dft" if request.method is QMMethod.DFT else "hf"
        return _LIMITATIONS.get(key, ()) + _COMMON_LIMITATIONS

    def _provenance(self, request: QMRequest) -> ProvenanceRecord:
        return ProvenanceRecord(
            kind=ProvenanceKind.SIMULATION,
            producer=self.id,
            producer_version=self.version,
            parameters={
                "method": request.method.value,
                "basis": request.basis,
                "xc": request.xc if request.method is QMMethod.DFT else "",
                "convergence": request.convergence,
                "optimized": request.optimize_geometry,
                "geometry": request.geometry.identity_payload(),
            },
            software=SoftwareEnvironment.capture(
                pyscf=pyscf.__version__ if _HAVE_PYSCF else ""
            ),
        )

    def _failure(
        self, request: QMRequest, reason: str, started: float | None = None
    ) -> QMResult:
        return QMResult(
            status=ConvergenceStatus.FAILED,
            method_signature=request.method_signature(),
            backend=self.id,
            backend_version=pyscf.__version__ if _HAVE_PYSCF else "",
            wall_time_seconds=None if started is None else time.perf_counter() - started,
            diagnostics=(reason,),
            limitations=self._limitations(request),
        )


class PySCFCalculator:
    """An ASE calculator that evaluates energies and forces with PySCF.

    Written by hand rather than imported because ASE ships no PySCF interface.
    It exists so ASE's optimisers can drive a PySCF surface, which is the only
    route to geometry optimisation here: ``pyscf.geomopt`` requires the
    geometric or berny solver and neither can be installed.

    The unit boundary is the whole job. PySCF speaks Hartree and Bohr, ASE
    speaks electronvolts and Angstrom, and PySCF returns a gradient where ASE
    expects a force. Each conversion happens exactly once, here.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, request: QMRequest) -> None:
        self.request = request
        self.results: dict[str, Any] = {}
        self.atoms = None
        self._backend = PySCFBackend()
        self.evaluations = 0

    # ASE's optimisers use this small part of the Calculator protocol.
    def get_potential_energy(self, atoms=None, force_consistent: bool = False) -> float:
        self._ensure(atoms)
        return self.results["energy"]

    def get_forces(self, atoms=None) -> np.ndarray:
        self._ensure(atoms)
        return self.results["forces"]

    def get_property(self, name: str, atoms=None, allow_calculation: bool = True):
        self._ensure(atoms)
        return self.results[name]

    def calculation_required(self, atoms, properties) -> bool:
        return self.atoms is None or not np.allclose(
            np.asarray(atoms.get_positions()), np.asarray(self.atoms)
        )

    def check_state(self, atoms, tol: float = 1e-12) -> list[str]:
        return [] if not self.calculation_required(atoms, ["energy"]) else ["positions"]

    def _ensure(self, atoms) -> None:
        if atoms is None:
            raise ValueError("PySCFCalculator needs an Atoms object.")
        positions = np.asarray(atoms.get_positions(), dtype=float)
        if self.atoms is not None and np.allclose(positions, self.atoms) and self.results:
            return

        geometry = self.request.geometry.with_positions(positions)
        mol = self._backend._build_molecule(geometry, self.request)
        mf = self._backend._build_scf(mol, self.request)
        energy_hartree = float(mf.kernel())
        self.evaluations += 1
        if not mf.converged:
            raise RuntimeError(
                "the SCF failed to converge during geometry optimisation, so the "
                "gradient at this step is meaningless"
            )
        gradient = np.asarray(mf.nuc_grad_method().kernel(), dtype=float)

        self.results = {
            "energy": energy_hartree * HARTREE_TO_EV,
            "forces": -gradient * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
        }
        self.atoms = positions.copy()
