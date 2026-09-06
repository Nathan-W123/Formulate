"""Quantum-mechanical validation (specification section 7)."""

from .base import ConvergenceStatus, QMBackend, QMMethod, QMRequest, QMResult
from .pyscf_backend import PySCFBackend, PySCFCalculator
from .xtb_backend import XTBBackend

#: Backends in no particular order of accuracy. Section 7 forbids implying one:
#: "Do not assume Hartree-Fock is a universally cheaper substitute for DFT or
#: that one method is uniformly more accurate. Select method by system/property."
QM_BACKENDS = (PySCFBackend, XTBBackend)


def available_qm_backends() -> list[QMBackend]:
    return [backend for backend in (cls() for cls in QM_BACKENDS) if backend.is_available()]


def backend_for(method: QMMethod) -> QMBackend | None:
    """The available backend implementing ``method``, or None."""
    for backend in available_qm_backends():
        if method in backend.supported_methods():
            return backend
    return None


__all__ = [
    "QM_BACKENDS", "ConvergenceStatus", "PySCFBackend", "PySCFCalculator", "QMBackend",
    "QMMethod", "QMRequest", "QMResult", "XTBBackend", "available_qm_backends",
    "backend_for",
]
