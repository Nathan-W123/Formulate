"""A Hartree-Fock implementation with nothing hidden inside it.

Specification section 7: "optional minimal Hartree-Fock implementation to
expose basis functions, SCF, energies, density, gradients, and the origin of
nuclear forces." Every one of those is a visible object here rather than a
number that comes back from a library.

This is a teaching backend and is labelled as one everywhere it can be:
``production = False``, ``PRODUCTION_WARNING`` on every result, and a refusal
of anything it cannot do exactly. It exists to be read, not to be trusted with
a result - :mod:`formulate.physics.qm.pyscf_backend` is what answers real
questions, and every number below is checked against it.

What it covers, and the boundary
    s-type contracted Gaussians only, which in the STO-3G basis means hydrogen
    and helium exactly and nothing else at all. Lithium onwards needs p
    functions, and p-function integrals are a different piece of mathematics -
    the Obara-Saika recursions - not a longer version of this one. Adding
    s-only integrals for a carbon atom would silently drop its 2p shell and
    return a number that looks like an energy, so carbon is refused.

    Within that boundary the integrals are analytic and exact, not quadrature:
    the Gaussian product theorem for overlap, kinetic and two-electron
    integrals, and the Boys function for nuclear attraction.

The origin of nuclear forces
    :func:`hellmann_feynman_forces` computes the classical electrostatic force
    on each nucleus from the converged electron density and the other nuclei.
    That is the whole force only for an exact wavefunction in a complete basis.
    In a finite basis the basis functions ride on the nuclei, so moving a
    nucleus changes the basis itself, and the difference between the true
    gradient and the Hellmann-Feynman force is the Pulay force.
    :func:`forces` reports both, because the gap between them is the lesson.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

#: Shown on every result this module produces.
PRODUCTION_WARNING = (
    "EDUCATIONAL BACKEND: this Hartree-Fock implementation exists to be read, not "
    "to be relied on. Use the PySCF backend for any result that matters."
)

#: STO-3G contracted 1s functions: (exponent, contraction coefficient) pairs.
#: Published values, checked against PySCF's own STO-3G in the test suite so a
#: transcription error cannot survive.
STO3G: dict[str, tuple[tuple[float, float], ...]] = {
    "H": (
        (3.42525091, 0.15432897),
        (0.62391373, 0.53532814),
        (0.16885540, 0.44463454),
    ),
    "He": (
        (6.36242139, 0.15432897),
        (1.15892300, 0.53532814),
        (0.31364979, 0.44463454),
    ),
}

ATOMIC_NUMBER = {"H": 1, "He": 2}

#: Bohr per Angstrom.
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903


class UnsupportedSystem(Exception):
    """Outside what this backend can do exactly, with the reason why."""


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContractedGaussian:
    """One basis function: a fixed sum of normalised s-type primitives.

    ``phi(r) = sum_i c_i * N(a_i) * exp(-a_i |r - centre|^2)``

    The normalisation ``N`` is folded into the stored coefficients at
    construction, so every integral below can treat the primitives as already
    normalised.
    """

    symbol: str
    centre: tuple[float, float, float]
    exponents: tuple[float, ...]
    coefficients: tuple[float, ...]

    @staticmethod
    def sto3g(symbol: str, centre: Sequence[float]) -> "ContractedGaussian":
        if symbol not in STO3G:
            raise UnsupportedSystem(
                f"{symbol} needs p functions in STO-3G, and this backend implements "
                f"s-type integrals only; it covers {sorted(STO3G)} exactly and refuses "
                "everything else rather than dropping a shell and returning a number "
                "that looks like an energy"
            )
        exponents, raw = zip(*STO3G[symbol])
        # Primitive normalisation for an s Gaussian: (2a/pi)^(3/4).
        folded = tuple(
            c * (2.0 * a / math.pi) ** 0.75 for a, c in zip(exponents, raw)
        )
        return ContractedGaussian(
            symbol=symbol,
            centre=tuple(float(x) for x in centre),
            exponents=tuple(float(a) for a in exponents),
            coefficients=folded,
        )

    def __call__(self, point: Sequence[float]) -> float:
        """The value of this basis function in space, for plotting or checking."""
        r2 = sum((float(p) - c) ** 2 for p, c in zip(point, self.centre))
        return sum(
            c * math.exp(-a * r2) for a, c in zip(self.exponents, self.coefficients)
        )

    def describe(self) -> str:
        terms = "  ".join(
            f"{c:+.6f} exp(-{a:.6f} r^2)"
            for a, c in zip(self.exponents, self.coefficients)
        )
        return f"{self.symbol} 1s at {self.centre}: {terms}"


@dataclass(frozen=True, slots=True)
class Molecule:
    """Nuclei and the basis functions sitting on them, in atomic units."""

    symbols: tuple[str, ...]
    #: Shape (n_atoms, 3), in Bohr. Everything below is in atomic units.
    positions: np.ndarray
    basis: tuple[ContractedGaussian, ...]

    @staticmethod
    def build(
        symbols: Sequence[str], positions_angstrom: np.ndarray
    ) -> "Molecule":
        positions = np.asarray(positions_angstrom, dtype=float) * BOHR_PER_ANGSTROM
        symbols = tuple(symbols)
        unsupported = sorted({s for s in symbols if s not in STO3G})
        if unsupported:
            raise UnsupportedSystem(
                f"{', '.join(unsupported)} need p functions in STO-3G, and this "
                f"backend implements s-type integrals only; it covers {sorted(STO3G)} "
                "exactly and refuses everything else rather than dropping a shell and "
                "returning a number that looks like an energy"
            )
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                if float(np.linalg.norm(positions[i] - positions[j])) < 1e-8:
                    raise UnsupportedSystem(
                        f"atoms {i} and {j} are at the same point, so their nuclear "
                        "repulsion is infinite and their basis functions are identical; "
                        "this is a broken geometry rather than a molecule"
                    )
        return Molecule(
            symbols=symbols,
            positions=positions,
            basis=tuple(
                ContractedGaussian.sto3g(symbol, position)
                for symbol, position in zip(symbols, positions)
            ),
        )

    @property
    def charges(self) -> np.ndarray:
        return np.array([ATOMIC_NUMBER[s] for s in self.symbols], dtype=float)

    @property
    def n_electrons(self) -> int:
        return int(self.charges.sum())

    def moved(self, index: int, axis: int, delta: float) -> "Molecule":
        """A copy with one nucleus displaced, for finite differences.

        The basis moves with the nucleus, which is exactly the fact that makes
        the Pulay force nonzero.
        """
        positions = self.positions.copy()
        positions[index, axis] += delta
        return Molecule(
            symbols=self.symbols,
            positions=positions,
            basis=tuple(
                ContractedGaussian.sto3g(symbol, position)
                for symbol, position in zip(self.symbols, positions)
            ),
        )


# ---------------------------------------------------------------------------
# Integrals, all analytic
# ---------------------------------------------------------------------------


def boys_f0(x: float) -> float:
    """``F0(x) = integral_0^1 exp(-x t^2) dt``, the s-type Boys function.

    Equal to ``sqrt(pi/4x) erf(sqrt(x))``, which is 1 at x = 0 by the limit
    rather than by the formula, so the limit is taken explicitly. A series is
    used for small x because the closed form is a ratio of two quantities that
    both go to zero there and loses precision long before x reaches machine
    epsilon.
    """
    if x < 1e-8:
        return 1.0 - x / 3.0 + x * x / 10.0
    return 0.5 * math.sqrt(math.pi / x) * math.erf(math.sqrt(x))


def _product(a: float, A: np.ndarray, b: float, B: np.ndarray):
    """Gaussian product theorem: two s Gaussians make one, times a prefactor."""
    p = a + b
    P = (a * A + b * B) / p
    separation = float(np.dot(A - B, A - B))
    K = math.exp(-a * b / p * separation)
    return p, P, K, separation


def overlap_matrix(basis: Sequence[ContractedGaussian]) -> np.ndarray:
    n = len(basis)
    S = np.zeros((n, n))
    for i, u in enumerate(basis):
        for j, v in enumerate(basis):
            A, B = np.array(u.centre), np.array(v.centre)
            total = 0.0
            for a, ca in zip(u.exponents, u.coefficients):
                for b, cb in zip(v.exponents, v.coefficients):
                    p, _, K, _ = _product(a, A, b, B)
                    total += ca * cb * (math.pi / p) ** 1.5 * K
            S[i, j] = total
    return S


def kinetic_matrix(basis: Sequence[ContractedGaussian]) -> np.ndarray:
    n = len(basis)
    T = np.zeros((n, n))
    for i, u in enumerate(basis):
        for j, v in enumerate(basis):
            A, B = np.array(u.centre), np.array(v.centre)
            total = 0.0
            for a, ca in zip(u.exponents, u.coefficients):
                for b, cb in zip(v.exponents, v.coefficients):
                    p, _, K, separation = _product(a, A, b, B)
                    reduced = a * b / p
                    total += (
                        ca
                        * cb
                        * reduced
                        * (3.0 - 2.0 * reduced * separation)
                        * (math.pi / p) ** 1.5
                        * K
                    )
            T[i, j] = total
    return T


def nuclear_attraction_matrix(
    basis: Sequence[ContractedGaussian], positions: np.ndarray, charges: np.ndarray
) -> np.ndarray:
    n = len(basis)
    V = np.zeros((n, n))
    for i, u in enumerate(basis):
        for j, v in enumerate(basis):
            A, B = np.array(u.centre), np.array(v.centre)
            total = 0.0
            for a, ca in zip(u.exponents, u.coefficients):
                for b, cb in zip(v.exponents, v.coefficients):
                    p, P, K, _ = _product(a, A, b, B)
                    for centre, Z in zip(positions, charges):
                        gap = float(np.dot(P - centre, P - centre))
                        total += (
                            -ca * cb * Z * 2.0 * math.pi / p * K * boys_f0(p * gap)
                        )
            V[i, j] = total
    return V


def electron_repulsion_tensor(basis: Sequence[ContractedGaussian]) -> np.ndarray:
    """``(ij|kl)`` in chemists' notation, over s-type contracted Gaussians."""
    n = len(basis)
    eri = np.zeros((n, n, n, n))
    centres = [np.array(f.centre) for f in basis]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for m in range(n):
                    total = 0.0
                    for a, ca in zip(basis[i].exponents, basis[i].coefficients):
                        for b, cb in zip(basis[j].exponents, basis[j].coefficients):
                            p, P, Kab, _ = _product(a, centres[i], b, centres[j])
                            for c, cc in zip(
                                basis[k].exponents, basis[k].coefficients
                            ):
                                for d, cd in zip(
                                    basis[m].exponents, basis[m].coefficients
                                ):
                                    q, Q, Kcd, _ = _product(
                                        c, centres[k], d, centres[m]
                                    )
                                    gap = float(np.dot(P - Q, P - Q))
                                    total += (
                                        ca
                                        * cb
                                        * cc
                                        * cd
                                        * 2.0
                                        * math.pi**2.5
                                        / (p * q * math.sqrt(p + q))
                                        * Kab
                                        * Kcd
                                        * boys_f0(p * q / (p + q) * gap)
                                    )
                    eri[i, j, k, m] = total
    return eri


def nuclear_repulsion(positions: np.ndarray, charges: np.ndarray) -> float:
    total = 0.0
    for i in range(len(charges)):
        for j in range(i + 1, len(charges)):
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            total += charges[i] * charges[j] / distance
    return total


# ---------------------------------------------------------------------------
# The self-consistent field
# ---------------------------------------------------------------------------


@dataclass
class SCFResult:
    """Everything the field converged to, not only its energy."""

    converged: bool
    energy: float
    electronic_energy: float
    nuclear_repulsion: float
    #: Molecular orbital coefficients, columns are orbitals.
    orbitals: np.ndarray
    orbital_energies: np.ndarray
    #: Closed-shell density matrix, ``2 * sum_occupied c c^T``.
    density: np.ndarray
    overlap: np.ndarray
    core_hamiltonian: np.ndarray
    fock: np.ndarray
    iterations: int = 0
    #: Energy change at each iteration, so the convergence is inspectable.
    history: list[float] = field(default_factory=list)
    warnings: tuple[str, ...] = (PRODUCTION_WARNING,)

    @property
    def n_electrons_from_density(self) -> float:
        """``tr(P S)``, which must come back as the electron count."""
        return float(np.trace(self.density @ self.overlap))

    @property
    def homo_energy(self) -> float | None:
        occupied = len(self.orbital_energies)
        return None if occupied == 0 else None

    def describe(self) -> str:
        lines = [
            PRODUCTION_WARNING,
            f"RHF/STO-3G  E = {self.energy:.10f} Hartree"
            + ("" if self.converged else "   NOT CONVERGED"),
            f"  electronic {self.electronic_energy:.10f}   "
            f"nuclear repulsion {self.nuclear_repulsion:.10f}",
            f"  {self.iterations} iteration(s); tr(PS) = "
            f"{self.n_electrons_from_density:.10f} electrons",
            "  orbital energies: "
            + "  ".join(f"{e:+.6f}" for e in self.orbital_energies),
        ]
        return "\n".join(lines)


def scf(
    molecule: Molecule,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> SCFResult:
    """Roothaan's equations, solved by repeated diagonalisation.

    Closed shell only: the density is built by doubly occupying the lowest
    ``n_electrons / 2`` orbitals, which is what restricted Hartree-Fock means
    and is why an odd electron count is refused rather than rounded.
    """
    if molecule.n_electrons % 2:
        raise UnsupportedSystem(
            f"{molecule.n_electrons} electrons is open shell, and this is a restricted "
            "Hartree-Fock implementation: it doubly occupies orbitals, so half an "
            "electron has nowhere to go. Unrestricted HF is a different method, not a "
            "flag on this one"
        )

    S = overlap_matrix(molecule.basis)
    H = kinetic_matrix(molecule.basis) + nuclear_attraction_matrix(
        molecule.basis, molecule.positions, molecule.charges
    )
    eri = electron_repulsion_tensor(molecule.basis)
    repulsion = nuclear_repulsion(molecule.positions, molecule.charges)
    occupied = molecule.n_electrons // 2

    # Symmetric orthogonalisation: X = S^(-1/2), so the transformed problem is
    # an ordinary eigenvalue problem rather than a generalised one.
    values, vectors = np.linalg.eigh(S)
    if values.min() <= 0:
        raise UnsupportedSystem(
            "the overlap matrix is not positive definite, which means the basis "
            "functions are linearly dependent - two nuclei are on top of each other"
        )
    X = vectors @ np.diag(values ** -0.5) @ vectors.T

    density = np.zeros_like(S)
    energy = previous = 0.0
    history: list[float] = []
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        # G[i,j] = sum_kl P[k,l] ( (ij|kl) - 0.5 (ik|jl) ): Coulomb minus the
        # exchange that comes from antisymmetry, which is the whole difference
        # between Hartree-Fock and a classical electrostatic model.
        coulomb = np.einsum("kl,ijkl->ij", density, eri)
        exchange = np.einsum("kl,ikjl->ij", density, eri)
        F = H + coulomb - 0.5 * exchange

        orbital_energies, coefficients = np.linalg.eigh(X.T @ F @ X)
        coefficients = X @ coefficients
        density = 2.0 * coefficients[:, :occupied] @ coefficients[:, :occupied].T

        electronic = 0.5 * float(np.sum(density * (H + F)))
        energy = electronic + repulsion
        history.append(energy - previous)
        if abs(energy - previous) < tolerance:
            converged = True
            break
        previous = energy

    return SCFResult(
        converged=converged,
        energy=energy,
        electronic_energy=energy - repulsion,
        nuclear_repulsion=repulsion,
        orbitals=coefficients,
        orbital_energies=orbital_energies,
        density=density,
        overlap=S,
        core_hamiltonian=H,
        fock=F,
        iterations=iteration,
        history=history,
    )


# ---------------------------------------------------------------------------
# Where nuclear forces come from
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ForceAnalysis:
    """The force on each nucleus, split into the two things it is made of."""

    #: Shape (n_atoms, 3), Hartree/Bohr. The classical electrostatic force on
    #: each nucleus from the converged electron density and the other nuclei.
    hellmann_feynman: np.ndarray
    #: The true gradient of the energy, by central differences.
    total: np.ndarray
    #: total - hellmann_feynman: the part that exists only because the basis
    #: functions are attached to the nuclei and move when they do.
    pulay: np.ndarray

    def describe(self) -> str:
        lines = [PRODUCTION_WARNING, "Force on each nucleus, Hartree/Bohr:"]
        for index in range(len(self.total)):
            lines.append(
                f"  atom {index}: total {_vector(self.total[index])}"
            )
            lines.append(
                f"           Hellmann-Feynman {_vector(self.hellmann_feynman[index])}"
            )
            lines.append(f"           Pulay {_vector(self.pulay[index])}")
        lines.append(
            "  The Hellmann-Feynman force is the whole force only for an exact "
            "wavefunction in a complete basis. Here the basis functions ride on the "
            "nuclei, so moving one changes the basis itself, and what is left over is "
            "the Pulay force."
        )
        return "\n".join(lines)


def _vector(v: np.ndarray) -> str:
    return "[" + " ".join(f"{x:+.6f}" for x in v) + "]"


def hellmann_feynman_forces(molecule: Molecule, result: SCFResult) -> np.ndarray:
    """The classical electrostatic force on each nucleus.

    Two contributions, both ordinary Coulomb: the other nuclei push, and the
    electron density pulls. The second is an integral of the density against
    the field of a point charge, which for s-type Gaussians has the same closed
    form the nuclear attraction integrals use, differentiated with respect to
    the nuclear position.
    """
    n_atoms = len(molecule.symbols)
    forces = np.zeros((n_atoms, 3))
    charges = molecule.charges
    positions = molecule.positions

    for index in range(n_atoms):
        # Nucleus-nucleus repulsion.
        for other in range(n_atoms):
            if other == index:
                continue
            delta = positions[index] - positions[other]
            distance = float(np.linalg.norm(delta))
            forces[index] += charges[index] * charges[other] * delta / distance**3

        # Electron-nucleus attraction, from the converged density.
        forces[index] += _density_field(molecule, result.density, index)
    return forces


def _density_field(molecule: Molecule, density: np.ndarray, index: int) -> np.ndarray:
    """Force on nucleus ``index`` from the electron density, in Hartree/Bohr.

    The nuclear attraction integral is ``V = -Z * 2pi/p * K * F0(p |P-C|^2)``
    and the energy contains ``sum_ij P_ij V_ij``, so the force is
    ``-d/dC sum_ij P_ij V_ij`` - the outer minus sign matters and was missing
    once. Chaining ``dF0/dx = -F1(x)`` and ``dx/dC = -2p (P - C)`` leaves
    ``+Z * 2pi/p * K * 2p (P - C) * F1``, and the electron density between two
    nuclei then pulls each of them inward, as it must. With the sign dropped it
    pushed them apart and made the Hellmann-Feynman force thirty-six times the
    total force on H2, which is how the error was noticed.
    """
    Z = molecule.charges[index]
    C = molecule.positions[index]
    basis = molecule.basis
    force = np.zeros(3)

    for i, u in enumerate(basis):
        for j, v in enumerate(basis):
            weight = density[i, j]
            if weight == 0.0:
                continue
            A, B = np.array(u.centre), np.array(v.centre)
            for a, ca in zip(u.exponents, u.coefficients):
                for b, cb in zip(v.exponents, v.coefficients):
                    p, P, K, _ = _product(a, A, b, B)
                    separation = P - C
                    x = p * float(np.dot(separation, separation))
                    # d/dC of F0(p|P-C|^2) = 2 p (P - C) F1(p|P-C|^2)
                    force += (
                        weight
                        * ca
                        * cb
                        * Z
                        * 2.0
                        * math.pi
                        / p
                        * K
                        * 2.0
                        * p
                        * separation
                        * boys_f1(x)
                    )
    return force


def boys_f1(x: float) -> float:
    """``F1(x) = integral_0^1 t^2 exp(-x t^2) dt``, with ``F1 = -dF0/dx``.

    From the recursion ``F1(x) = (F0(x) - exp(-x)) / (2x)``, which is a
    difference of two nearly equal numbers for small x and is therefore
    replaced by its series there.
    """
    if x < 1e-6:
        return 1.0 / 3.0 - x / 5.0 + x * x / 14.0
    return (boys_f0(x) - math.exp(-x)) / (2.0 * x)


def forces(molecule: Molecule, *, step: float = 1e-4) -> ForceAnalysis:
    """The total force by central differences, and its two components.

    The total is a finite difference rather than an analytic gradient, and that
    is deliberate: the analytic Pulay term needs derivatives of every integral
    with respect to the basis-function centres, which is more machinery than a
    module written to be read should carry. What matters pedagogically is that
    the Hellmann-Feynman force is *not* the whole force, and a finite
    difference demonstrates that without asserting it.
    """
    result = scf(molecule)
    if not result.converged:
        raise UnsupportedSystem(
            "the field did not converge, so there is no density to take a force from"
        )

    hf = hellmann_feynman_forces(molecule, result)
    total = np.zeros_like(hf)
    for atom in range(len(molecule.symbols)):
        for axis in range(3):
            plus = scf(molecule.moved(atom, axis, step)).energy
            minus = scf(molecule.moved(atom, axis, -step)).energy
            total[atom, axis] = -(plus - minus) / (2.0 * step)
    return ForceAnalysis(hellmann_feynman=hf, total=total, pulay=total - hf)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def run(
    symbols: Sequence[str], positions_angstrom: np.ndarray, **kwargs
) -> SCFResult:
    """Build the molecule and converge the field, in one call."""
    return scf(Molecule.build(symbols, positions_angstrom), **kwargs)


# ---------------------------------------------------------------------------
# The uniform backend interface
# ---------------------------------------------------------------------------


class MinimalHFBackend:
    """This module behind :class:`~formulate.physics.qm.base.QMBackend`.

    ``production = False``, which is not decoration: :func:`backend_for` refuses
    to hand a non-production backend to anything that asked for a real method,
    so wiring this in cannot accidentally make it answer a design question.
    """

    id = "minimal-hf"
    version = "1"
    production = False

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return ""

    def supported_methods(self):
        from .base import QMMethod

        return frozenset({QMMethod.MINIMAL_HF})

    def supports(self, request) -> bool:
        from .base import QMMethod

        return request.method is QMMethod.MINIMAL_HF

    def run(self, request):
        from formulate.core.quantity import Quantity

        from .base import ConvergenceStatus, QMMethod, QMResult

        if request.method is not QMMethod.MINIMAL_HF:
            return QMResult(
                status=ConvergenceStatus.FAILED,
                method_signature=str(request.method.value),
                backend=self.id,
                backend_version=self.version,
                diagnostics=(
                    f"this backend implements {QMMethod.MINIMAL_HF.value} only and will "
                    f"not stand in for {request.method.value}",
                ),
            )
        if request.optimize_geometry:
            return QMResult(
                status=ConvergenceStatus.FAILED,
                method_signature="RHF/STO-3G (educational)",
                backend=self.id,
                backend_version=self.version,
                diagnostics=(
                    "this backend has no geometry optimiser; it computes one point",
                ),
            )

        geometry = request.geometry
        try:
            result = scf(
                Molecule.build(geometry.symbols, geometry.positions),
                max_iterations=request.max_scf_cycles,
                tolerance=max(request.convergence, 1e-12),
            )
        except UnsupportedSystem as exc:
            return QMResult(
                status=ConvergenceStatus.FAILED,
                method_signature="RHF/STO-3G (educational)",
                backend=self.id,
                backend_version=self.version,
                diagnostics=(str(exc),),
                limitations=(PRODUCTION_WARNING,),
            )

        return QMResult(
            status=(
                ConvergenceStatus.CONVERGED
                if result.converged
                else ConvergenceStatus.NOT_CONVERGED
            ),
            method_signature="RHF/STO-3G (educational)",
            backend=self.id,
            backend_version=self.version,
            total_energy=Quantity(
                value=result.energy * 2625499.639479828, unit="J/mol"
            ),
            scf_cycles=result.iterations,
            limitations=(PRODUCTION_WARNING,),
        )
