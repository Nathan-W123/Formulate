"""Periodic condensed-phase simulation, and the bulk observables it unlocks.

Specification section 6 asks for density, self-diffusion and cohesive energy
density; section 13 forbids answering any of them from a finite cluster,
because a cluster has a surface and its answer depends on where the boundary is
drawn. Both statements were true and unreachable here until :mod:`mmff` made
MMFF94 available as an OpenMM system, which is what a periodic box needs.

What this module adds on top is the box itself and three protocols:

density
    Constant pressure, and the volume is the observable. This is the one that
    could not previously be attempted at all, because no ensemble here coupled
    the system to a pressure.

cohesive energy density
    The liquid's potential energy per molecule against an isolated molecule of
    the same species at the same temperature, divided by the molar volume. The
    isolated reference is *sampled*, not minimised: comparing a thermally
    excited liquid against a relaxed single molecule attributes the thermal
    energy of the molecule to cohesion, which is how a cohesive energy comes
    out with the wrong sign.

self-diffusion
    The Einstein relation, fitted only over the interval where the mean squared
    displacement is actually linear in time. The ballistic regime at short time
    and the poorly sampled tail at long time both have slopes, and fitting
    through either returns a number that is not a diffusion coefficient.

What this is worth, measured rather than assumed. The engine is exact: the
force field it evaluates reproduces RDKit's MMFF94 to 1e-11 kcal/mol, term by
term. The force field is not. MMFF94 was parameterised against gas-phase
geometries and conformational energies, not against liquids, and it under-binds
a condensed phase.

The size of that under-binding took three attempts to measure, and the first
two answers were wrong. Recorded here because each failed in a way that
returned a plausible number rather than an error.

Potential energy of vaporisation at 298 K, against experiment:

    toluene   -11 per cent   (31.2, 31.5 and 31.9 kJ/mol from three separate
                              equilibration protocols against 35.5; they agree
                              to 0.7, so this one is converged)
    ethanol   -15 per cent   (33.9 against 39.8)
    hexane     -3 per cent   (27.9 against 28.9)

An earlier note in this file put hexane at -34.6 per cent and blamed the
dispersion term on that basis. It does not reproduce: that run minimised for
2000 iterations, which leaves a lattice of rigid molecules in a state tens of
picoseconds of dynamics cannot relax, and the liquid energy it reported was set
by the starting configuration rather than by the force field. Five thousand
iterations, or a melt-and-cool, moves toluene from an apparent -29 per cent to
a stable -11.

The second wrong answer was worse, because it broke a monotonicity that physics
guarantees. Scaling the van der Waals well depth up by 30 per cent *lowered*
the computed vaporisation energy, from 31.2 to 22.9 kJ/mol for toluene, which
cannot happen: a stronger attraction cannot make a liquid easier to evaporate.
The cause was the reference state. A molecule has van der Waals interactions
with itself, and :func:`sample_isolated_energy` was building its reference at
the unscaled force field while the liquid ran at the scaled one, so the
difference in intramolecular energy was being charged to cohesion. A rigid ring
has enough internal contact area for that artefact to swamp the effect. It now
takes the same scale, and the argument is documented rather than defaulted.

So the deficit is real but modest, around ten to fifteen per cent in cohesive
energy, and largest for the associating liquids rather than the dispersive ones
- the opposite of what the retracted hexane figure suggested. The density error
is larger than the energy error because a density sits where attraction
balances repulsion and responds more than proportionally.

The fix is a force field fitted to liquids, and half of what this paragraph
used to say about that was wrong. GAFF and the OpenFF line do need
openff-toolkit or AmberTools, both conda-only and neither installable here,
because they assign partial charges per molecule with AM1-BCC. OPLS-AA does
not: its charges come out of the atom-type table, so it needs a SMARTS typing
engine and nothing else. :mod:`opls` supplies one. Pass ``force_field`` to any
protocol here to run under it instead, and read that module first - OPLS-AA
covers thirty-seven of the fifty reference compounds and one of the ways it
fails is silent.

MMFF94 remains the default because it types anything RDKit can parse, and its
bias is at least shared across candidates, which is what a ranking needs. A
single fitted scale on the MMFF94 well depth was the obvious stopgap and was
abandoned: hexane returned 18.9, 20.7 and 27.9 kJ/mol from three runs of the
same protocol, so there was nothing stable to fit to. Values from MMFF94 are
useful for ranking candidates against each other and are not quantitative;
every prediction built on them says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Avogadro's number.
_AVOGADRO = 6.02214076e23
#: Ideal-gas Boltzmann constant in kJ/(mol K).
_R_KJ = 0.00831446261815324


# --------------------------------------------------------------------------
# Which force field a protocol runs under
# --------------------------------------------------------------------------

#: The force fields these protocols can run under.
FORCE_FIELDS = ("mmff94", "opls-aa")


@dataclass(frozen=True, slots=True)
class _Prepared:
    """One molecule's parameters and the two systems a protocol needs from them.

    The seam exists because the two force fields are reached differently and
    both have to end up at the same place: an OpenMM system whose particles run
    molecule by molecule in input order, with bonds to hydrogen constrained.
    MMFF94 comes out of RDKit and its constraints are added here from the bond
    list; OPLS-AA comes out of foyer as a ParmEd structure that builds its own
    system and applies ``HBonds`` itself. Everything downstream - the barostat,
    the equilibration test, the sampling - is identical, which is the point.
    """

    name: str
    n_atoms: int
    masses: tuple[float, ...]
    _make: object

    def system(self, copies: int, *, box_nm: float | None, cutoff_nm: float = 1.0):
        return self._make(copies, box_nm, cutoff_nm)


def _prepare(mol, force_field: str, *, dispersion_scale: float = 1.0) -> _Prepared:
    if force_field == "mmff94":
        from .mmff import build_system, extract_parameters

        params = extract_parameters(mol)
        if params is None:
            raise ValueError("MMFF94 has no parameters for this molecule")

        hydrogen = {i for i, mass in enumerate(params.masses) if mass < 2.0}

        def make(copies, box_nm, cutoff_nm):
            system = build_system(
                params,
                copies,
                exact=box_nm is None,
                box_nm=box_nm,
                cutoff_nm=cutoff_nm,
                dispersion_scale=dispersion_scale,
            )
            # Bonds to hydrogen have a period under ten femtoseconds, so
            # without this the timestep is set by a motion no bulk property
            # depends on.
            for offset in range(0, copies * params.n_atoms, params.n_atoms):
                for i, j, _, r0 in params.bonds:
                    if i in hydrogen or j in hydrogen:
                        system.addConstraint(i + offset, j + offset, r0 * 0.1)
            return system

        return _Prepared("mmff94", params.n_atoms, tuple(params.masses), make)

    if force_field == "opls-aa":
        if dispersion_scale != 1.0:
            raise ValueError(
                "dispersion_scale is an MMFF94 stopgap for a force field that was "
                "not fitted to liquids; OPLS-AA was, so scaling its well depths "
                "would move it away from its own parameterisation"
            )
        from . import opls

        params = opls.extract_parameters(mol)

        def make(copies, box_nm, cutoff_nm):
            system, _ = opls.build_system(
                params, copies, box_nm=box_nm, cutoff_nm=cutoff_nm
            )
            return system

        return _Prepared("opls-aa", params.n_atoms, params.masses, make)

    raise ValueError(f"unknown force field {force_field!r}; expected one of {FORCE_FIELDS}")


@dataclass(frozen=True, slots=True)
class Box:
    """A periodic box of identical molecules, ready to simulate."""

    #: Positions in nanometres, atom-major, molecules concatenated in order.
    positions: np.ndarray
    #: Cubic box edge, nanometres.
    edge_nm: float
    n_molecules: int
    n_atoms_per_molecule: int
    #: Molar mass of one molecule, g/mol.
    molar_mass: float

    @property
    def density_g_cm3(self) -> float:
        volume_cm3 = (self.edge_nm * 1e-7) ** 3
        return self.n_molecules * self.molar_mass / (_AVOGADRO * volume_cm3)


def edge_for_density(n_molecules: int, molar_mass: float, density_g_cm3: float) -> float:
    """Cubic edge in nanometres holding ``n_molecules`` at a target density."""
    volume_cm3 = n_molecules * molar_mass / (_AVOGADRO * density_g_cm3)
    return (volume_cm3 ** (1.0 / 3.0)) * 1e7


def minimum_molecules(molar_mass: float, density_g_cm3: float, cutoff_nm: float) -> int:
    """Fewest molecules whose box still obeys the minimum image convention.

    A periodic cell must exceed twice the non-bonded cutoff, or a molecule
    interacts with two images of the same neighbour and the energy is wrong in
    a way no amount of sampling detects. The barostat compresses the box during
    equilibration, so the count has to be checked against the *equilibrium*
    density rather than the expanded lattice the run starts from.
    """
    edge = 2.0 * cutoff_nm
    volume_cm3 = (edge * 1e-7) ** 3
    return int(math.ceil(volume_cm3 * _AVOGADRO * density_g_cm3 / molar_mass))


def pack_box(
    mol,
    n_molecules: int,
    density_g_cm3: float,
    *,
    seed: int = 0,
    expansion: float = 1.08,
) -> Box:
    """Lay out ``n_molecules`` copies on a lattice with random orientations.

    The lattice starts slightly expanded and is compressed by the barostat.
    Packing to the target density directly puts atoms inside each other's
    repulsive wall, and the minimiser then has to climb out of an energy that
    can exceed anything the force field was fitted for. Expanding too far is
    the opposite mistake and the more expensive one: a first attempt started at
    an edge 1.3 times the target, which is less than half the target density,
    and sixty picoseconds later the box had reached 0.55 g/cm^3 against an
    experimental 0.79 and was still visibly compressing.
    """
    from rdkit.Chem import Descriptors

    conformer = mol.GetConformer()
    reference = np.array(
        [list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    ) * 0.1
    reference -= reference.mean(axis=0)
    molar_mass = float(Descriptors.MolWt(mol))

    edge = edge_for_density(n_molecules, molar_mass, density_g_cm3) * expansion
    per_side = math.ceil(n_molecules ** (1.0 / 3.0))
    spacing = edge / per_side

    rng = np.random.default_rng(seed)
    positions = []
    placed = 0
    for i in range(per_side):
        for j in range(per_side):
            for k in range(per_side):
                if placed >= n_molecules:
                    break
                centre = (np.array([i, j, k]) + 0.5) * spacing
                positions.append(reference @ _random_rotation(rng).T + centre)
                placed += 1
    return Box(
        positions=np.vstack(positions),
        edge_nm=edge,
        n_molecules=n_molecules,
        n_atoms_per_molecule=mol.GetNumAtoms(),
        molar_mass=molar_mass,
    )


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random rotation, from a random unit quaternion."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# --------------------------------------------------------------------------
# Running a condensed phase
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CondensedResult:
    """What a periodic run produced, and how well converged it is."""

    #: Mean density over production, g/cm^3.
    density_g_cm3: float
    #: Standard error of that mean from block averaging.
    density_error: float
    #: Mean potential energy per molecule over production, kJ/mol.
    energy_per_molecule: float
    energy_error: float
    #: Mean box edge over production, nm.
    edge_nm: float
    production_ps: float
    n_molecules: int
    wall_seconds: float
    diagnostics: tuple[str, ...] = ()

    @property
    def molar_volume_cm3(self) -> float:
        return self.molar_mass_g / self.density_g_cm3 if self.density_g_cm3 else float("nan")

    molar_mass_g: float = 0.0


def _block_error(values: np.ndarray, blocks: int = 5) -> float:
    """Standard error from block averaging, which correlated data needs.

    A naive standard error over a trajectory treats successive frames as
    independent samples when they are not, and understates the uncertainty by
    the square root of the correlation length.
    """
    if values.size < blocks * 2:
        return float("nan")
    size = values.size // blocks
    means = np.array([values[i * size : (i + 1) * size].mean() for i in range(blocks)])
    return float(means.std(ddof=1) / math.sqrt(blocks))


def run_npt(
    mol,
    n_molecules: int,
    temperature_k: float,
    *,
    initial_density: float = 0.8,
    pressure_bar: float = 1.0,
    equilibration_ps: float = 200.0,
    production_ps: float = 200.0,
    timestep_fs: float = 2.0,
    cutoff_nm: float = 1.0,
    seed: int = 0,
    sample_interval_ps: float = 1.0,
    platform: str = "CPU",
    force_field: str = "mmff94",
) -> CondensedResult:
    """Equilibrate a periodic box at constant pressure and sample its volume.

    The volume is the observable, which is why this needs a barostat and why it
    could not be attempted before. Hydrogen bonds are constrained so a two
    femtosecond step is stable; that removes the fastest motion in the system
    and nothing that a density depends on.
    """
    import time

    import openmm as mm
    import openmm.unit as u

    prepared = _prepare(mol, force_field)

    from rdkit.Chem import Descriptors

    # Checked against a density well above the guess, because the barostat is
    # free to compress past it and the failure mode is an exception a hundred
    # picoseconds in rather than at setup.
    molar_mass = float(Descriptors.MolWt(mol))
    needed = minimum_molecules(molar_mass, initial_density * 1.5, cutoff_nm)
    if n_molecules < needed:
        raise ValueError(
            f"{n_molecules} molecules of molar mass {molar_mass:.1f} g/mol cannot fill a "
            f"box larger than twice a {cutoff_nm:.2f} nm cutoff once the barostat has "
            f"compressed it; at least {needed} are needed, or the cutoff must come down"
        )

    box = pack_box(mol, n_molecules, initial_density, seed=seed)
    system = prepared.system(n_molecules, box_nm=box.edge_nm, cutoff_nm=cutoff_nm)

    system.addForce(
        mm.MonteCarloBarostat(pressure_bar * u.bar, temperature_k * u.kelvin, 25)
    )
    integrator = mm.LangevinMiddleIntegrator(
        temperature_k * u.kelvin, 1.0 / u.picosecond, timestep_fs * u.femtosecond
    )
    integrator.setRandomNumberSeed(seed)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform))
    context.setPositions(box.positions)

    started = time.perf_counter()
    mm.LocalEnergyMinimizer.minimize(context, tolerance=10.0, maxIterations=2000)
    context.setVelocitiesToTemperature(temperature_k * u.kelvin, seed)

    steps_per_ps = int(round(1000.0 / timestep_fs))

    def density_now() -> float:
        vectors = context.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(
            u.nanometer
        )
        return n_molecules * box.molar_mass / (_AVOGADRO * float(np.linalg.det(vectors)) * 1e-21)

    # Equilibration ends when the density stops moving, not when a guessed
    # number of picoseconds runs out. A fixed budget is either wasted on a box
    # that settled early or, worse, spent entirely on a box that has not: the
    # first attempt here ran a fixed 30 ps and produced a density 30 per cent
    # low, still climbing, with nothing but the diagnostics to say so.
    chunk_ps = max(2.0, equilibration_ps / 10.0)
    history = [density_now()]
    equilibrated = False
    elapsed_ps = 0.0
    while elapsed_ps < equilibration_ps:
        integrator.step(int(chunk_ps * steps_per_ps))
        elapsed_ps += chunk_ps
        history.append(density_now())
        if len(history) >= 5:
            recent = np.array(history[-4:])
            slope = float(np.polyfit(np.arange(recent.size), recent, 1)[0])
            if abs(slope) < 0.002 * max(recent.mean(), 1e-9):
                equilibrated = True
                break

    sample_steps = max(1, int(sample_interval_ps * steps_per_ps))
    n_samples = max(2, int(production_ps / sample_interval_ps))
    volumes, energies = [], []
    for _ in range(n_samples):
        integrator.step(sample_steps)
        state = context.getState(energy=True)
        vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer)
        volumes.append(float(np.linalg.det(vectors)))
        energies.append(
            state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole) / n_molecules
        )

    volumes = np.array(volumes)
    energies = np.array(energies)
    densities = n_molecules * box.molar_mass / (_AVOGADRO * (volumes * 1e-21))

    diagnostics: list[str] = []
    if not equilibrated:
        diagnostics.append(
            f"the density was still moving when the {equilibration_ps:.0f} ps "
            f"equilibration budget ran out (it went {history[0]:.3f} -> {history[-1]:.3f} "
            "g/cm^3); production started from a box that had not settled"
        )
    if production_ps < 100.0:
        diagnostics.append(
            f"{production_ps:.0f} ps of production against the 100 ps a density "
            "normally needs to lose its memory of the starting lattice"
        )
    drift = np.polyfit(np.arange(densities.size), densities, 1)[0] * densities.size
    if abs(drift) > 3.0 * max(_block_error(densities), 1e-9):
        diagnostics.append(
            f"the density drifted by {drift:+.4f} g/cm^3 across production, which is "
            "larger than its own sampling error: the box is still equilibrating"
        )

    return CondensedResult(
        density_g_cm3=float(densities.mean()),
        density_error=_block_error(densities),
        energy_per_molecule=float(energies.mean()),
        energy_error=_block_error(energies),
        edge_nm=float(np.cbrt(volumes.mean())),
        production_ps=production_ps,
        n_molecules=n_molecules,
        wall_seconds=time.perf_counter() - started,
        diagnostics=tuple(diagnostics),
        molar_mass_g=box.molar_mass,
    )


# --------------------------------------------------------------------------
# Observables that need the condensed phase and an isolated reference
# --------------------------------------------------------------------------


def sample_isolated_energy(
    mol,
    temperature_k: float,
    *,
    production_ps: float = 100.0,
    timestep_fs: float = 1.0,
    seed: int = 0,
    platform: str = "CPU",
    dispersion_scale: float = 1.0,
    force_field: str = "mmff94",
) -> tuple[float, float]:
    """Mean potential energy of one molecule in vacuum at ``temperature_k``.

    Sampled at temperature rather than minimised. A liquid is thermally
    excited; subtracting a relaxed single-molecule energy from it charges the
    molecule's own vibrational and torsional energy to cohesion, which is large
    enough to reverse the sign of a cohesive energy.

    ``dispersion_scale`` must match whatever the liquid was run with, and this
    argument exists because getting it wrong is silent. A molecule has van der
    Waals interactions with itself, and scaling them in the liquid but not in
    the reference charges that difference to cohesion. Measured on toluene: at
    a scale of 1.3 against an unscaled reference the vaporisation energy came
    out at 22.9 kJ/mol, *below* the 31.2 at a scale of 1.0, so strengthening
    the attraction appeared to weaken the liquid. A rigid ring has enough
    internal contact area for the artefact to swamp the effect being measured.
    """
    import openmm as mm
    import openmm.unit as u

    prepared = _prepare(mol, force_field, dispersion_scale=dispersion_scale)
    system = prepared.system(1, box_nm=None)

    integrator = mm.LangevinMiddleIntegrator(
        temperature_k * u.kelvin, 1.0 / u.picosecond, timestep_fs * u.femtosecond
    )
    integrator.setRandomNumberSeed(seed)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform))
    conformer = mol.GetConformer()
    context.setPositions(
        np.array([list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]) * 0.1
    )
    mm.LocalEnergyMinimizer.minimize(context)
    context.setVelocitiesToTemperature(temperature_k * u.kelvin, seed)

    steps_per_ps = int(round(1000.0 / timestep_fs))
    integrator.step(20 * steps_per_ps)  # let it thermalise
    samples = []
    for _ in range(max(2, int(production_ps))):
        integrator.step(steps_per_ps)
        samples.append(
            context.getState(energy=True).getPotentialEnergy().value_in_unit(
                u.kilojoule_per_mole
            )
        )
    values = np.array(samples)
    return float(values.mean()), _block_error(values)


@dataclass(frozen=True, slots=True)
class CohesiveResult:
    """Cohesive energy density, and the vaporisation energy behind it."""

    cohesive_energy_density_pa: float
    error_pa: float
    #: Potential energy of vaporisation per mole, kJ/mol.
    vaporisation_energy: float
    molar_volume_cm3: float
    diagnostics: tuple[str, ...] = ()


def cohesive_energy_density(
    liquid: CondensedResult, gas_energy: float, gas_error: float
) -> CohesiveResult:
    """Turn a liquid run and an isolated reference into an energy density.

    ``(U_gas - U_liquid) / V_molar``. The difference is the potential energy of
    vaporisation, which is the enthalpy of vaporisation less RT; quoting it
    alongside the density is what lets the result be checked against a measured
    enthalpy rather than taken on trust.
    """
    delta = gas_energy - liquid.energy_per_molecule
    molar_volume = liquid.molar_mass_g / liquid.density_g_cm3
    value = delta * 1e9 / molar_volume

    relative = math.hypot(
        liquid.energy_error / abs(delta) if delta else float("inf"),
        gas_error / abs(delta) if delta else float("inf"),
        liquid.density_error / liquid.density_g_cm3,
    )
    diagnostics = list(liquid.diagnostics)
    if delta <= 0:
        diagnostics.append(
            f"the liquid came out {-delta:.1f} kJ/mol above the isolated molecule, which "
            "is not a bound phase: the box has not equilibrated or the reference was not "
            "sampled at the same temperature"
        )
    return CohesiveResult(
        cohesive_energy_density_pa=value,
        error_pa=abs(value) * relative,
        vaporisation_energy=delta,
        molar_volume_cm3=molar_volume,
        diagnostics=tuple(diagnostics),
    )


@dataclass(frozen=True, slots=True)
class DiffusionResult:
    """Self-diffusion coefficient from the Einstein relation."""

    coefficient_m2_s: float
    error_m2_s: float
    #: Log-log slope of the mean squared displacement over the fitted window.
    #: One is the diffusive regime; below one the trajectory has not reached it.
    log_log_slope: float
    fitted_window_ps: tuple[float, float]
    wall_seconds: float = 0.0
    diagnostics: tuple[str, ...] = ()


def run_self_diffusion(
    mol,
    n_molecules: int,
    temperature_k: float,
    *,
    density_g_cm3: float,
    equilibration_ps: float = 100.0,
    production_ps: float = 500.0,
    timestep_fs: float = 2.0,
    cutoff_nm: float = 1.0,
    sample_interval_ps: float = 1.0,
    seed: int = 0,
    platform: str = "CPU",
    force_field: str = "mmff94",
) -> DiffusionResult:
    """Self-diffusion from the Einstein relation, fitted only where it applies.

    A mean squared displacement has three regimes: ballistic at short time,
    where it grows as t squared; diffusive in the middle, where it grows as t;
    and a noisy tail where too few independent displacements remain. Only the
    middle one is a diffusion coefficient. The fit is restricted to the window
    whose log-log slope is nearest one, and that slope is reported, so a
    trajectory that never reached the diffusive regime says so instead of
    returning the slope of whatever it did reach.
    """
    import time

    import openmm as mm
    import openmm.unit as u

    prepared = _prepare(mol, force_field)

    needed = minimum_molecules(sum(prepared.masses), density_g_cm3, cutoff_nm)
    if n_molecules < needed:
        raise ValueError(f"at least {needed} molecules are needed for a {cutoff_nm} nm cutoff")

    box = pack_box(mol, n_molecules, density_g_cm3, seed=seed, expansion=1.0)
    system = prepared.system(n_molecules, box_nm=box.edge_nm, cutoff_nm=cutoff_nm)

    # Constant volume, not constant pressure: a fluctuating box makes an
    # unwrapped displacement ambiguous, and the density is an input here rather
    # than the thing being measured.
    integrator = mm.LangevinMiddleIntegrator(
        temperature_k * u.kelvin, 1.0 / u.picosecond, timestep_fs * u.femtosecond
    )
    integrator.setRandomNumberSeed(seed)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform))
    context.setPositions(box.positions)

    started = time.perf_counter()
    mm.LocalEnergyMinimizer.minimize(context, tolerance=10.0, maxIterations=2000)
    context.setVelocitiesToTemperature(temperature_k * u.kelvin, seed)
    steps_per_ps = int(round(1000.0 / timestep_fs))
    integrator.step(int(equilibration_ps * steps_per_ps))

    masses = np.array(prepared.masses)
    weights = masses / masses.sum()
    edge = box.edge_nm

    def centres() -> np.ndarray:
        positions = context.getState(positions=True).getPositions(asNumpy=True).value_in_unit(
            u.nanometer
        )
        grouped = positions.reshape(n_molecules, prepared.n_atoms, 3)
        return np.einsum("mai,a->mi", grouped, weights)

    sample_steps = max(1, int(sample_interval_ps * steps_per_ps))
    n_frames = max(8, int(production_ps / sample_interval_ps))
    previous = centres()
    unwrapped = previous.copy()
    trajectory = [unwrapped.copy()]
    for _ in range(n_frames):
        integrator.step(sample_steps)
        current = centres()
        # Unwrap by hand: a jump of more than half the box between frames is a
        # molecule crossing the boundary, not a displacement.
        step = current - previous
        step -= edge * np.round(step / edge)
        unwrapped = unwrapped + step
        trajectory.append(unwrapped.copy())
        previous = current

    frames = np.array(trajectory)
    lags = np.arange(1, max(2, frames.shape[0] // 2))
    msd = np.array(
        [np.mean(np.sum((frames[lag:] - frames[:-lag]) ** 2, axis=2)) for lag in lags]
    )
    times = lags * sample_interval_ps

    # Pick the decade-wide window whose log-log slope is closest to one.
    best = None
    span = max(3, lags.size // 3)
    for start in range(0, lags.size - span):
        window = slice(start, start + span)
        slope = float(
            np.polyfit(np.log(times[window]), np.log(np.maximum(msd[window], 1e-12)), 1)[0]
        )
        if best is None or abs(slope - 1.0) < abs(best[0] - 1.0):
            best = (slope, window)
    slope, window = best

    # MSD = 6 D t, with nm^2/ps -> m^2/s being a factor of 1e-6.
    fit = np.polyfit(times[window], msd[window], 1)
    coefficient = float(fit[0]) / 6.0 * 1e-6
    residual = msd[window] - np.polyval(fit, times[window])
    error = float(np.std(residual) / max(np.mean(msd[window]), 1e-12)) * abs(coefficient)

    diagnostics: list[str] = []
    if abs(slope - 1.0) > 0.15:
        diagnostics.append(
            f"the mean squared displacement grows as t^{slope:.2f} over the best window, "
            "not t^1: the trajectory has not reached the diffusive regime and this is "
            "not a diffusion coefficient"
        )
    if production_ps < 500.0:
        diagnostics.append(
            f"{production_ps:.0f} ps of production against the 500 ps a self-diffusion "
            "coefficient normally needs"
        )
    return DiffusionResult(
        coefficient_m2_s=coefficient,
        error_m2_s=error,
        log_log_slope=slope,
        fitted_window_ps=(float(times[window][0]), float(times[window][-1])),
        wall_seconds=time.perf_counter() - started,
        diagnostics=tuple(diagnostics),
    )


# --------------------------------------------------------------------------
# Shear viscosity
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ViscosityResult:
    """Shear viscosity from the Green-Kubo integral, and how flat its plateau is."""

    viscosity_pa_s: float
    #: Spread across the three independent shear components, which are three
    #: statistically independent estimates of the same number.
    error_pa_s: float
    #: Where the running integral was read, picoseconds.
    plateau_window_ps: tuple[float, float]
    #: Fractional drift of the running integral across that window. A Green-Kubo
    #: integral that has converged is flat; one that is still climbing is a
    #: lower bound, and this says which it was.
    plateau_drift: float
    #: Decay time of the stress autocorrelation, picoseconds.
    correlation_time_ps: float
    #: Root mean square of the instantaneous configurational shear stress, bar.
    stress_rms_bar: float
    #: Root mean square of the *kinetic* shear stress as a fraction of that.
    #: The kinetic term is deliberately left out of the integral - see the
    #: protocol's note on why - and this is how much was left out.
    kinetic_fraction: float
    sampled_ps: float
    n_molecules: int
    wall_seconds: float
    diagnostics: tuple[str, ...] = ()


def _autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Unbiased autocorrelation of a single stress component, by FFT.

    No mean is subtracted. An off-diagonal stress has a mean of exactly zero in
    an isotropic liquid, and subtracting the sample mean instead would remove a
    little of the signal along with it - the slowly varying part is precisely
    what the Green-Kubo integral is made of. Zero padding to twice the length
    turns the circular correlation the transform computes into a linear one,
    and dividing by the number of overlapping pairs at each lag makes the
    estimator unbiased rather than tapered.
    """
    n = series.size
    padded = np.zeros(2 * n)
    padded[:n] = series
    spectrum = np.fft.rfft(padded)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum))[:max_lag]
    return correlation / (n - np.arange(max_lag))


def run_shear_viscosity(
    mol,
    n_molecules: int,
    temperature_k: float,
    *,
    density_g_cm3: float,
    equilibration_ps: float = 100.0,
    production_ps: float = 200.0,
    stress_interval_fs: float = 10.0,
    correlation_ps: float = 4.0,
    timestep_fs: float = 2.0,
    cutoff_nm: float = 1.0,
    seed: int = 0,
    platform: str = "CPU",
    force_field: str = "opls-aa",
) -> ViscosityResult:
    """Shear viscosity from the stress autocorrelation of a periodic box.

        eta = (V / kT) * integral over t of <sigma_xy(0) sigma_xy(t)>

    averaged over the three independent off-diagonal components, which
    :mod:`stress` obtains by finite difference because OpenMM publishes no
    pressure tensor.

    This is the expensive protocol in the package and the reason is structural
    rather than incidental. A density needs the volume once a picosecond; a
    Green-Kubo integral needs the stress every few femtoseconds, because the
    autocorrelation it integrates loses most of its amplitude within a hundred,
    and each of those samples costs two single-point energies per component.
    The finite difference, not the dynamics, is what the wall clock goes on.

    Constant volume: the stress tensor of a box whose volume is being moved by
    a barostat is not the stress tensor of the liquid at that density.
    """
    import time

    import openmm as mm
    import openmm.unit as u

    from .stress import (
        KJ_PER_MOL_NM3_IN_PA,
        SHEARS,
        centres_of_mass,
        configurational_stress,
        kinetic_stress,
        molecular_momenta,
        molecule_owner,
    )

    prepared = _prepare(mol, force_field)
    needed = minimum_molecules(sum(prepared.masses), density_g_cm3, cutoff_nm)
    if n_molecules < needed:
        raise ValueError(f"at least {needed} molecules are needed for a {cutoff_nm} nm cutoff")

    steps_per_sample = stress_interval_fs / timestep_fs
    if abs(steps_per_sample - round(steps_per_sample)) > 1e-9 or steps_per_sample < 1:
        raise ValueError(
            f"a {stress_interval_fs:.1f} fs stress interval is not a whole number of "
            f"{timestep_fs:.1f} fs steps"
        )
    steps_per_sample = int(round(steps_per_sample))

    box = pack_box(mol, n_molecules, density_g_cm3, seed=seed, expansion=1.0)
    system = prepared.system(n_molecules, box_nm=box.edge_nm, cutoff_nm=cutoff_nm)
    integrator = mm.LangevinMiddleIntegrator(
        temperature_k * u.kelvin, 1.0 / u.picosecond, timestep_fs * u.femtosecond
    )
    integrator.setRandomNumberSeed(seed)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform))
    context.setPositions(box.positions)

    started = time.perf_counter()
    mm.LocalEnergyMinimizer.minimize(context, tolerance=10.0, maxIterations=2000)
    context.setVelocitiesToTemperature(temperature_k * u.kelvin, seed)
    integrator.step(int(equilibration_ps * 1000.0 / timestep_fs))

    masses = np.array(prepared.masses * n_molecules)
    owner = molecule_owner(n_molecules, prepared.n_atoms)
    edge = box.edge_nm
    volume_nm3 = edge**3
    vectors = np.diag([edge, edge, edge])

    # The kinetic term is dropped from the off-diagonal stress on purpose. Its
    # correlation decays within a few femtoseconds - it is the free flight of
    # the molecules between collisions - and at the sampling interval this
    # protocol can afford it is aliased rather than resolved. For a liquid well
    # below its critical point it carries under a per cent of the integral;
    # including it badly sampled is worse than leaving it out knowingly.
    n_samples = int(production_ps * 1000.0 / stress_interval_fs)
    stress = np.zeros((n_samples, len(SHEARS)))
    kinetic = np.zeros((n_samples, len(SHEARS)))
    for index in range(n_samples):
        integrator.step(steps_per_sample)
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
        tensor = kinetic_stress(momentum, molecular_mass, volume_nm3)
        kinetic[index] = [tensor[a, b] for a, b in SHEARS]
        stress[index] = configurational_stress(
            context, positions, vectors, centres, owner, volume_nm3, SHEARS
        )
    stress *= KJ_PER_MOL_NM3_IN_PA  # to pascals
    kinetic *= KJ_PER_MOL_NM3_IN_PA

    dt_ps = stress_interval_fs / 1000.0
    max_lag = max(4, int(correlation_ps / dt_ps))
    if max_lag > n_samples // 2:
        max_lag = n_samples // 2

    boltzmann = 1.380649e-23
    prefactor = (volume_nm3 * 1e-27) / (boltzmann * temperature_k)
    lags = np.arange(max_lag) * dt_ps

    integrals = []
    correlations = []
    for component in range(len(SHEARS)):
        acf = _autocorrelation(stress[:, component], max_lag)
        correlations.append(acf)
        # Trapezoidal running integral, picoseconds to seconds.
        running = np.concatenate(([0.0], np.cumsum((acf[1:] + acf[:-1]) * 0.5 * dt_ps)))
        integrals.append(running * prefactor * 1e-12)
    integrals = np.array(integrals)
    mean_acf = np.mean(correlations, axis=0)

    # Read the plateau over the last third of the correlation window, which is
    # far enough out that the fast collisional decay is finished and near
    # enough in that the tail's own noise has not taken over.
    plateau = slice(int(max_lag * 2 // 3), max_lag)
    per_component = integrals[:, plateau].mean(axis=1)
    viscosity = float(per_component.mean())
    error = float(per_component.std(ddof=1) / math.sqrt(len(SHEARS)))

    window = integrals.mean(axis=0)[plateau]
    drift = float(
        np.polyfit(np.arange(window.size), window, 1)[0] * window.size / max(abs(viscosity), 1e-30)
    )

    positive = mean_acf > 0
    decay = float(
        lags[np.argmax(mean_acf < mean_acf[0] / math.e)] if np.any(~positive) else lags[-1]
    )

    diagnostics: list[str] = []
    if abs(drift) > 0.2:
        diagnostics.append(
            f"the running integral drifts by {drift * 100:+.0f} per cent across the "
            "plateau window, so it has not converged and this is a lower bound rather "
            "than a viscosity"
        )
    if error > 0.25 * abs(viscosity):
        diagnostics.append(
            f"the three shear components disagree by {error / abs(viscosity) * 100:.0f} "
            "per cent of the mean; more sampling is needed before a force field error "
            "can be told from noise"
        )
    kinetic_fraction = float(
        np.sqrt(np.mean(kinetic**2)) / max(np.sqrt(np.mean(stress**2)), 1e-30)
    )
    if kinetic_fraction > 0.1:
        diagnostics.append(
            f"the kinetic shear stress is {kinetic_fraction * 100:.0f} per cent of the "
            "configurational one by root mean square, and it is not in the integral; "
            "this liquid is dilute enough that leaving it out is a real omission"
        )
    if decay > correlation_ps / 3.0:
        diagnostics.append(
            f"the stress correlation is still {decay:.2f} ps wide against a "
            f"{correlation_ps:.1f} ps window, so the integral is being cut off before "
            "the tail has decayed"
        )

    return ViscosityResult(
        viscosity_pa_s=viscosity,
        error_pa_s=error,
        plateau_window_ps=(float(lags[plateau.start]), float(lags[max_lag - 1])),
        plateau_drift=drift,
        correlation_time_ps=decay,
        stress_rms_bar=float(np.sqrt(np.mean(stress**2)) / 1e5),
        kinetic_fraction=float(
            np.sqrt(np.mean(kinetic**2)) / max(np.sqrt(np.mean(stress**2)), 1e-30)
        ),
        sampled_ps=production_ps,
        n_molecules=n_molecules,
        wall_seconds=time.perf_counter() - started,
        diagnostics=tuple(diagnostics),
    )
