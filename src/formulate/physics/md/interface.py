"""Surface tension from a liquid slab.

:mod:`condensed` runs a homogeneous box and reads its volume, which is enough
for a density and a cohesive energy. A surface tension is not a property of a
homogeneous box at all. It is the excess free energy carried by an interface,
and the only way to make an interface in a periodic cell is to stop filling it:
put a slab of liquid in the middle of an elongated box and leave vacuum at both
ends. The liquid then has two surfaces, and

    gamma = (Lz / 2) * <P_zz - (P_xx + P_yy) / 2>

is the Kirkwood-Buff expression for what they cost. The factor of two is the
two surfaces; the bracket is a normal stress that is exactly zero in a bulk
liquid and nonzero only because the box is inhomogeneous along z.

Everything hard about the bracket is in :mod:`stress`, which recovers a
pressure tensor OpenMM does not expose. What is specific to a slab is here,
and it is three decisions.

**Constant volume, not constant pressure.** A barostat acting on the lateral
edges would squeeze the interface being measured, and one acting on z would
push against vacuum. The box is set by the density the caller states and then
held, so that density must be the experimental one rather than a guess to be
relaxed - which in turn means a slab tension is only as good as the density
fed to it, and the density is measured separately by
:func:`~formulate.physics.md.condensed.run_npt`.

**The dispersion correction is off.** The analytic long-range van der Waals
correction assumes bulk structure everywhere beyond the cutoff, which a slab
does not have anywhere outside the liquid. Worse, it is isotropic, so it
cancels out of the bracket above and would leave no trace of its own wrongness
in the answer while corrupting the pressures it was computed from. It is
disabled, and the cutoff is instead made long enough that what it truncates is
small.

**Two biases are stated rather than discovered.** A slab of finite lateral
extent cannot carry capillary waves longer than its own edge, and its tension
comes out low by a few per cent at three nanometres. And three-dimensional
Ewald summation is only correct on a slab geometry when the cell carries no
net dipole along the normal, which a symmetric two-surface slab satisfies on
average but not instantaneously; the dipole is measured during production and
reported so that the second can be checked instead of assumed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .stress import (
    DIAGONAL,
    KJ_PER_MOL_NM3_IN_PA,
    centres_of_mass,
    configurational_stress,
    kinetic_stress,
    molecular_momenta,
    molecule_owner,
)

_AVOGADRO = 6.02214076e23


@dataclass(frozen=True, slots=True)
class Slab:
    """A liquid slab centred in an elongated periodic box with vacuum ends."""

    #: Positions in nanometres, atom-major, molecules concatenated in order.
    positions: np.ndarray
    #: Box edges (x, y, z) in nanometres. The slab is normal to z.
    box_nm: tuple[float, float, float]
    n_molecules: int
    n_atoms_per_molecule: int
    #: Molar mass of one molecule, g/mol.
    molar_mass: float
    #: Thickness of the liquid region as packed, nanometres.
    liquid_nm: float

    @property
    def vacuum_nm(self) -> float:
        """Vacuum between one face of the slab and the image of the other."""
        return self.box_nm[2] - self.liquid_nm


def pack_slab(
    mol,
    n_molecules: int,
    density_g_cm3: float,
    *,
    lateral_nm: float,
    vacuum_nm: float,
    seed: int = 0,
) -> Slab:
    """Lay ``n_molecules`` on a lattice filling a slab, vacuum above and below.

    The lateral edges are fixed by the caller and held fixed for the whole run:
    a slab is simulated at constant volume, not constant pressure, because a
    barostat acting on x and y would squeeze the interface it is meant to
    measure. That makes the packing density the density the liquid will have,
    so it should be the experimental one rather than a guess to be relaxed.
    """
    from rdkit.Chem import Descriptors

    from .condensed import _random_rotation

    conformer = mol.GetConformer()
    reference = np.array(
        [list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    ) * 0.1
    reference -= reference.mean(axis=0)
    molar_mass = float(Descriptors.MolWt(mol))

    volume_nm3 = n_molecules * molar_mass / (_AVOGADRO * density_g_cm3 * 1e-21)
    liquid_nm = volume_nm3 / (lateral_nm * lateral_nm)
    box = (lateral_nm, lateral_nm, liquid_nm + 2.0 * vacuum_nm)

    # A lattice whose cells are as near cubic as the slab's aspect ratio
    # allows, shrunk until it has room for every molecule.
    spacing = (volume_nm3 / n_molecules) ** (1.0 / 3.0)
    while True:
        nx, ny, nz = (
            max(1, int(lateral_nm // spacing)),
            max(1, int(lateral_nm // spacing)),
            max(1, int(liquid_nm // spacing)),
        )
        if nx * ny * nz >= n_molecules:
            break
        spacing *= 0.98

    rng = np.random.default_rng(seed)
    origin = np.array([0.0, 0.0, (box[2] - liquid_nm) / 2.0])
    step = np.array([lateral_nm / nx, lateral_nm / ny, liquid_nm / nz])

    positions = []
    placed = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if placed >= n_molecules:
                    break
                centre = origin + (np.array([i, j, k]) + 0.5) * step
                positions.append(reference @ _random_rotation(rng).T + centre)
                placed += 1
    return Slab(
        positions=np.vstack(positions),
        box_nm=box,
        n_molecules=n_molecules,
        n_atoms_per_molecule=mol.GetNumAtoms(),
        molar_mass=molar_mass,
        liquid_nm=liquid_nm,
    )


@dataclass(frozen=True, slots=True)
class SurfaceTensionResult:
    """A surface tension and the evidence for how far to trust it."""

    #: Mean surface tension over production, mN/m.
    surface_tension_mn_m: float
    #: Standard error of that mean from block averaging, mN/m.
    surface_tension_error: float
    #: The three diagonal pressures, bar, as a record of what went into it.
    pressure_bar: tuple[float, float, float]
    #: Kinetic anisotropy, bar: the largest pairwise difference among the three
    #: molecular kinetic pressures. It is zero for an equilibrated slab and is
    #: reported because it is the term assumed to cancel out of the tension.
    kinetic_anisotropy_bar: float
    #: RMS dipole of the cell along the slab normal, in elementary charge
    #: nanometres. Three-dimensional Ewald on a slab is exact only at zero.
    dipole_z_rms: float
    production_ps: float
    n_molecules: int
    box_nm: tuple[float, float, float]
    liquid_nm: float
    wall_seconds: float
    diagnostics: tuple[str, ...] = ()


def surface_tension(
    mol,
    n_molecules: int,
    temperature_k: float,
    density_g_cm3: float,
    *,
    lateral_nm: float = 3.2,
    vacuum_nm: float = 4.0,
    cutoff_nm: float = 1.2,
    equilibration_ps: float = 150.0,
    production_ps: float = 600.0,
    timestep_fs: float = 2.0,
    sample_interval_ps: float = 1.0,
    seed: int = 0,
    platform: str = "CPU",
) -> SurfaceTensionResult:
    """Surface tension of a liquid slab under OPLS-AA, in mN/m.

    Constant volume, not constant pressure: the box is set by ``density_g_cm3``
    and the lateral edges, and nothing is allowed to move it. The liquid's own
    equation of state then decides the normal pressure, and the difference
    between that and the lateral pressure is the tension.
    """
    import openmm as mm
    import openmm.unit as u

    from . import opls

    if cutoff_nm * 2.0 >= lateral_nm:
        raise ValueError(
            f"a {cutoff_nm:.2f} nm cutoff does not fit in a {lateral_nm:.2f} nm "
            "lateral edge; the slab needs every edge to exceed twice the cutoff"
        )
    if vacuum_nm < 2.0 * cutoff_nm:
        raise ValueError(
            f"{vacuum_nm:.2f} nm of vacuum is less than twice the {cutoff_nm:.2f} nm "
            "cutoff, so the two surfaces of the slab interact through it"
        )

    params = opls.extract_parameters(mol)
    slab = pack_slab(
        mol,
        n_molecules,
        density_g_cm3,
        lateral_nm=lateral_nm,
        vacuum_nm=vacuum_nm,
        seed=seed,
    )
    if slab.liquid_nm < 4.0 * cutoff_nm:
        raise ValueError(
            f"a {slab.liquid_nm:.2f} nm slab is thinner than four times the "
            f"{cutoff_nm:.2f} nm cutoff, so its two surfaces overlap and there is "
            "no bulk liquid between them to have a surface"
        )

    system, _ = opls.build_system(
        params,
        n_molecules,
        box_nm=slab.box_nm,
        cutoff_nm=cutoff_nm,
        dispersion_correction=False,
    )
    integrator = mm.LangevinMiddleIntegrator(
        temperature_k * u.kelvin, 1.0 / u.picosecond, timestep_fs * u.femtosecond
    )
    integrator.setRandomNumberSeed(seed)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform))
    context.setPositions(slab.positions)

    started = time.perf_counter()
    mm.LocalEnergyMinimizer.minimize(context, tolerance=10.0, maxIterations=2000)
    context.setVelocitiesToTemperature(temperature_k * u.kelvin, seed)

    steps_per_ps = int(round(1000.0 / timestep_fs))
    integrator.step(int(equilibration_ps * steps_per_ps))

    masses = np.array(
        [
            system.getParticleMass(i).value_in_unit(u.dalton)
            for i in range(system.getNumParticles())
        ]
    )
    owner = molecule_owner(n_molecules, slab.n_atoms_per_molecule)
    charges = np.array(params.charges * n_molecules)
    volume_nm3 = float(np.prod(slab.box_nm))
    box = np.diag(np.array(slab.box_nm))

    sample_steps = max(1, int(sample_interval_ps * steps_per_ps))
    n_samples = max(2, int(production_ps / sample_interval_ps))

    tensions, pressures, kinetics, dipoles = [], [], [], []
    for _ in range(n_samples):
        integrator.step(sample_steps)
        state = context.getState(
            positions=True, velocities=True, enforcePeriodicBox=False
        )
        positions = state.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        velocities = state.getVelocities(asNumpy=True).value_in_unit(
            u.nanometer / u.picosecond
        )
        centres = centres_of_mass(positions, masses, owner, n_molecules)
        momentum, molecular_mass = molecular_momenta(
            velocities, masses, owner, n_molecules
        )
        kinetic = np.diag(kinetic_stress(momentum, molecular_mass, volume_nm3))
        configurational = configurational_stress(
            context, positions, box, centres, owner, volume_nm3, DIAGONAL
        )
        diagonal = (kinetic + configurational) * KJ_PER_MOL_NM3_IN_PA

        # Two surfaces, hence the half; nm to m and N/m to mN/m are the rest.
        tensions.append(
            0.5
            * slab.box_nm[2]
            * 1e-9
            * (diagonal[2] - 0.5 * (diagonal[0] + diagonal[1]))
            * 1e3
        )
        pressures.append(diagonal / 1e5)
        kinetics.append(kinetic * KJ_PER_MOL_NM3_IN_PA / 1e5)
        dipoles.append(float(charges @ (positions[:, 2] - positions[:, 2].mean())))

    from .condensed import _block_error

    tensions = np.array(tensions)
    pressures = np.array(pressures)
    kinetics = np.array(kinetics)

    diagnostics: list[str] = []
    error = _block_error(tensions)
    if production_ps < 400.0:
        diagnostics.append(
            f"{production_ps:.0f} ps of production; a slab tension normally needs "
            "several hundred picoseconds before its block error stops shrinking"
        )
    if not math.isnan(error) and error > 0.15 * abs(tensions.mean()):
        diagnostics.append(
            f"the sampling error is {error / abs(tensions.mean()) * 100:.0f} per cent "
            "of the mean, which is too large to separate a force field error from noise"
        )
    if lateral_nm < 4.0:
        diagnostics.append(
            f"a {lateral_nm:.1f} nm lateral edge suppresses capillary waves longer "
            "than itself, which biases the tension low by a few per cent"
        )
    anisotropy = float(
        max(abs(kinetics.mean(axis=0)[i] - kinetics.mean(axis=0)[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
    )

    return SurfaceTensionResult(
        surface_tension_mn_m=float(tensions.mean()),
        surface_tension_error=error,
        pressure_bar=tuple(float(v) for v in pressures.mean(axis=0)),
        kinetic_anisotropy_bar=anisotropy,
        dipole_z_rms=float(np.sqrt(np.mean(np.square(dipoles)))),
        production_ps=production_ps,
        n_molecules=n_molecules,
        box_nm=slab.box_nm,
        liquid_nm=slab.liquid_nm,
        wall_seconds=time.perf_counter() - started,
        diagnostics=tuple(diagnostics),
    )
