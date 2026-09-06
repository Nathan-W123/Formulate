"""GFN2-xTB semi-empirical backend.

Specification section 7 asks for lower-cost electronic methods to screen with
before escalating, and section 7's fidelity ladder warns against assuming a
single accuracy ordering. GFN2-xTB is the screening rung here: roughly a
hundred times faster than a small-basis DFT calculation on the same molecule,
parameterised rather than derived, and reliable for geometries and relative
conformer energies while being unsuitable for the electronic observables
people most often want from it.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from formulate.core.provenance import ProvenanceKind, ProvenanceRecord, SoftwareEnvironment
from formulate.core.quantity import Quantity

from ..geometry import EV_TO_J_PER_MOL, geometry_from_ase
from ..quiet import suppress_native_output
from .base import ConvergenceStatus, QMBackend, QMMethod, QMRequest, QMResult

_LIMITATIONS = (
    "GFN2-xTB is parameterised, not derived: it reproduces geometries and relative "
    "conformer energies well, but its absolute energies are not comparable with any "
    "ab initio result and its orbital energies are not a substitute for a DFT gap",
    "parameterised for the main-group elements; results outside that set are extrapolation",
    "computed for an isolated molecule in vacuum at zero Kelvin, with no zero-point "
    "energy, thermal, entropic or solvent contribution",
)


class XTBBackend(QMBackend):
    """Semi-empirical tight binding for screening."""

    id = "xtb"
    version = "1"
    production = True

    def is_available(self) -> bool:
        try:
            from xtb.ase.calculator import XTB  # noqa: F401
        except Exception:
            return False
        return True

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "the xtb package is not installed"

    def supported_methods(self) -> frozenset[QMMethod]:
        return frozenset({QMMethod.GFN2_XTB})

    def run(self, request: QMRequest) -> QMResult:
        if not self.is_available():
            return self._failure(request, self.unavailable_reason())
        if request.method is not QMMethod.GFN2_XTB:
            return self._failure(
                request, f"{self.id} does not implement {request.method.value}"
            )

        started = time.perf_counter()
        diagnostics: list[str] = []
        try:
            from xtb.ase.calculator import XTB

            atoms = request.geometry.to_ase()
            with suppress_native_output():
                atoms.calc = XTB(method="GFN2-xTB")
                geometry = request.geometry
                residual: float | None = None

                if request.optimize_geometry:
                    from ase.optimize import BFGS

                    BFGS(atoms, logfile=None).run(
                        fmax=request.force_threshold, steps=request.max_optimization_steps
                    )
                    residual = float(np.abs(atoms.get_forces()).max())
                    geometry = geometry_from_ase(atoms, template=request.geometry).with_positions(
                        np.asarray(atoms.get_positions()), source="GFN2-xTB relaxed (ASE BFGS)"
                    )

                energy_ev = float(atoms.get_potential_energy())
                forces = (
                    np.asarray(atoms.get_forces(), dtype=float).tolist()
                    if request.compute_forces
                    else None
                )
                dipole = self._dipole(atoms, diagnostics)

            if residual is not None and residual > request.force_threshold:
                diagnostics.append(
                    f"geometry optimisation did not reach the force threshold: residual "
                    f"{residual:.4f} eV/Angstrom against {request.force_threshold:.4f}"
                )

            return QMResult(
                status=ConvergenceStatus.CONVERGED,
                method_signature="GFN2-xTB",
                backend=self.id,
                total_energy=Quantity(value=energy_ev * EV_TO_J_PER_MOL, unit="J/mol"),
                dipole_moment=dipole,
                forces=forces,
                optimized_geometry=geometry if request.optimize_geometry else None,
                max_residual_force=residual,
                wall_time_seconds=time.perf_counter() - started,
                diagnostics=tuple(
                    diagnostics
                    + [
                        "no frontier gap is reported: a tight-binding orbital energy "
                        "difference is not comparable with a DFT or experimental gap, and "
                        "reporting it under the same name would invite that comparison"
                    ]
                ),
                limitations=_LIMITATIONS,
                provenance=self._provenance(request),
            )
        except Exception as exc:
            return self._failure(request, f"{type(exc).__name__}: {exc}", started)

    def _dipole(self, atoms: Any, diagnostics: list[str]) -> Quantity | None:
        try:
            vector = np.asarray(atoms.get_dipole_moment(), dtype=float)
            # ASE reports a dipole in electron-Angstrom; 1 e*A = 4.803205 debye.
            return Quantity(value=float(np.linalg.norm(vector)) * 4.803204544, unit="debye")
        except Exception as exc:
            diagnostics.append(f"dipole unavailable ({exc})")
            return None

    def _provenance(self, request: QMRequest) -> ProvenanceRecord:
        return ProvenanceRecord(
            kind=ProvenanceKind.SIMULATION,
            producer=self.id,
            producer_version=self.version,
            parameters={
                "method": "GFN2-xTB",
                "optimized": request.optimize_geometry,
                "geometry": request.geometry.identity_payload(),
            },
            software=SoftwareEnvironment.capture(xtb="22.1"),
        )

    def _failure(
        self, request: QMRequest, reason: str, started: float | None = None
    ) -> QMResult:
        return QMResult(
            status=ConvergenceStatus.FAILED,
            method_signature="GFN2-xTB",
            backend=self.id,
            wall_time_seconds=None if started is None else time.perf_counter() - started,
            diagnostics=(reason,),
            limitations=_LIMITATIONS,
        )
