"""The pressure tensor, which OpenMM does not have.

OpenMM reports a scalar potential energy and a box volume. It has no pressure
at all in its public interface: the barostats compute what they need inside
their own Monte Carlo move and never publish it. That single absence is what
put ``surface_tension`` and ``shear_viscosity`` in this package's
``NOT_VALIDATABLE_REASONS`` list, because both are made entirely of components
of a tensor that could not be read - a surface tension from the difference
between the normal and lateral diagonal terms, a shear viscosity from the
autocorrelation of an off-diagonal one.

This module recovers the tensor by finite difference. Deforming the cell by a
strain ``eps`` in component ``(alpha, beta)`` and carrying the contents along
affinely,

    P_alpha_beta = (1/V) * (kinetic_alpha_beta - dU/deps_alpha_beta)

which is the virial theorem written so that the derivative is something a
molecular dynamics program will actually evaluate: two single-point energies.

Three details decide whether the number that comes out is the pressure or
merely an arithmetic result.

**The strain moves molecules, not atoms.**
    Scaling every atom stretches bonds. Bonds to hydrogen are *constrained* in
    every protocol here, so an atomic affine strain asks the potential for an
    energy at a geometry the integrator forbids, and the constraint forces'
    contribution to the virial is lost with no sign that anything happened.
    Translating each molecule rigidly by ``eps`` times its centre of mass
    leaves every intramolecular term invariant - bonds, angles, torsions, the
    reciprocal-space exclusion corrections, the Ewald self-energy - so the
    difference is purely intermolecular and constraints never enter it.

    This is the *molecular* virial rather than the atomic one. They give
    different instantaneous tensors and the same surface tension and the same
    zero-frequency shear viscosity, which is classical (Walton, Tildesley,
    Rowlinson and Henderson, 1983); the two differ by a term that is a total
    time derivative, which integrates to nothing. The molecular route is used
    here because it is the one that survives constraints.

**Image shifts do not matter.**
    A molecule whose centre of mass lies outside the cell is displaced by
    ``eps`` times a centre that differs from the in-cell one by a lattice
    vector. That is not an error: the cell's own vectors are strained by the
    same ``eps``, so the displaced molecule lands exactly on the image of where
    the in-cell one would have gone and the energy is identical. Nothing here
    wraps coordinates, and nothing needs to.

**Only three of the six off-diagonal strains are expressible.**
    OpenMM requires periodic box vectors in reduced form, with ``a`` along x
    and ``b`` in the xy plane. A strain that tilts ``a`` towards y cannot be
    represented. The tensor is symmetric, so this costs nothing: the three
    components ``(x,y)``, ``(x,z)`` and ``(y,z)`` are the three independent
    shears, and each is obtained by displacing the first coordinate in
    proportion to the second.

The finite difference is checked against a case with a closed form - two
Lennard-Jones particles at fixed separation, whose molecular virial is
``-r U'(r)`` along the separation and exactly zero across it - in
``tests/test_stress.py``, and end to end against a barostat holding a known
pressure.
"""

from __future__ import annotations

import numpy as np

_AVOGADRO = 6.02214076e23

#: One kJ/mol per cubic nanometre, in pascals. Daltons times nm^2/ps^2 is also
#: kJ/mol, so the kinetic and configurational terms need no separate factor.
KJ_PER_MOL_NM3_IN_PA = 1.0e3 / (_AVOGADRO * 1.0e-27)

#: Relative strain for the finite difference. Small enough that the cubic term
#: a central difference does not cancel stays below the sampling noise, large
#: enough that the energy difference clears the round-off of a single-precision
#: platform: at 1e-5 the two energies of a few thousand atoms agree to within
#: their own truncation and the derivative is noise.
STRAIN = 2.0e-4

#: The three independent shears, as (displaced coordinate, coordinate it is
#: displaced in proportion to). The reverse of each is not a separate strain -
#: the tensor is symmetric - and would need a box vector OpenMM cannot hold.
SHEARS = ((0, 1), (0, 2), (1, 2))

DIAGONAL = ((0, 0), (1, 1), (2, 2))


def molecule_owner(n_molecules: int, n_atoms_per_molecule: int) -> np.ndarray:
    """Which molecule each atom belongs to, for the atom-major layout used here."""
    return np.repeat(np.arange(n_molecules), n_atoms_per_molecule)


def centres_of_mass(
    positions: np.ndarray, masses: np.ndarray, owner: np.ndarray, n_molecules: int
) -> np.ndarray:
    weighted = np.zeros((n_molecules, 3))
    np.add.at(weighted, owner, positions * masses[:, None])
    total = np.zeros(n_molecules)
    np.add.at(total, owner, masses)
    return weighted / total[:, None]


def molecular_momenta(
    velocities: np.ndarray, masses: np.ndarray, owner: np.ndarray, n_molecules: int
) -> tuple[np.ndarray, np.ndarray]:
    """Each molecule's total momentum and mass, in daltons nm/ps and daltons."""
    momentum = np.zeros((n_molecules, 3))
    np.add.at(momentum, owner, velocities * masses[:, None])
    total = np.zeros(n_molecules)
    np.add.at(total, owner, masses)
    return momentum, total


def kinetic_stress(
    momentum: np.ndarray, molecular_mass: np.ndarray, volume_nm3: float
) -> np.ndarray:
    """Molecular kinetic contribution to the pressure tensor, kJ/mol/nm^3.

    Isotropic on average with a mean of ``N k T / V`` on each diagonal entry
    and zero off it, so it cancels out of a surface tension; it is computed
    anyway because that cancellation is an assumption worth checking rather
    than relying on.
    """
    return (momentum[:, :, None] * momentum[:, None, :] / molecular_mass[:, None, None]).sum(
        axis=0
    ) / volume_nm3


def configurational_stress(
    context,
    positions: np.ndarray,
    box: np.ndarray,
    centres: np.ndarray,
    owner: np.ndarray,
    volume_nm3: float,
    components: tuple[tuple[int, int], ...] = DIAGONAL,
    *,
    strain: float = STRAIN,
) -> np.ndarray:
    """``-(1/V) dU/deps`` for each requested component, in kJ/mol/nm^3.

    ``context`` is left holding ``positions`` and ``box`` again on return, so
    the caller's trajectory is undisturbed and the integrator can be stepped
    straight afterwards.
    """
    import openmm.unit as u

    out = np.zeros(len(components))
    for index, (alpha, beta) in enumerate(components):
        energies = []
        for sign in (+1.0, -1.0):
            eps = sign * strain
            shifted = positions.copy()
            shifted[:, alpha] += eps * centres[owner, beta]
            vectors = box.copy()
            vectors[beta, alpha] += eps * box[beta, beta]
            context.setPeriodicBoxVectors(*vectors)
            context.setPositions(shifted)
            energies.append(
                context.getState(energy=True)
                .getPotentialEnergy()
                .value_in_unit(u.kilojoule_per_mole)
            )
        out[index] = -(energies[0] - energies[1]) / (2.0 * strain) / volume_nm3

    context.setPeriodicBoxVectors(*box)
    context.setPositions(positions)
    return out
