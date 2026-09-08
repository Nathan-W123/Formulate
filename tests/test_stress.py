"""The pressure tensor recovered by finite difference, against closed forms.

Every claim in :mod:`formulate.physics.md.stress` that can be checked without
a trajectory is checked here: the value of the molecular virial for a pair
whose energy has a derivative anyone can write down, the invariance under
image shifts that lets the finite difference skip wrapping, and the promise
that the context comes back holding what it was given.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")

from formulate.physics.md.stress import (  # noqa: E402
    DIAGONAL,
    KJ_PER_MOL_NM3_IN_PA,
    SHEARS,
    centres_of_mass,
    configurational_stress,
    kinetic_stress,
    molecular_momenta,
    molecule_owner,
)

SIGMA = 0.35
EPSILON = 1.2
EDGE = 6.0


def _pair_context():
    """Two uncharged Lennard-Jones particles in a periodic box, one per molecule."""
    import openmm as mm
    import openmm.unit as u

    system = mm.System()
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(EDGE, 0, 0), mm.Vec3(0, EDGE, 0), mm.Vec3(0, 0, EDGE)
    )
    force = mm.NonbondedForce()
    force.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(2.0 * u.nanometer)
    force.setUseDispersionCorrection(False)
    for _ in range(2):
        system.addParticle(1.0)
        force.addParticle(0.0, SIGMA, EPSILON)
    system.addForce(force)
    return mm.Context(
        system, mm.VerletIntegrator(1.0e-6), mm.Platform.getPlatformByName("Reference")
    )


def _exact(separation: np.ndarray) -> np.ndarray:
    """The molecular virial of one pair: ``-(U'(r)/r) * d_alpha d_beta / V``."""
    r = float(np.linalg.norm(separation))
    du = 4.0 * EPSILON * (-12.0 * SIGMA**12 / r**13 + 6.0 * SIGMA**6 / r**7)
    outer = np.outer(separation, separation)
    return -du / r * outer / EDGE**3


def test_pair_virial_matches_the_derivative_of_the_potential():
    context = _pair_context()
    separation = np.array([0.30, 0.18, 0.22])
    positions = np.array([[1.5, 1.5, 1.5], [1.5, 1.5, 1.5] + separation])
    box = np.diag([EDGE, EDGE, EDGE])
    owner = np.array([0, 1])

    got = configurational_stress(
        context, positions, box, positions.copy(), owner, EDGE**3, DIAGONAL + SHEARS
    )
    exact = _exact(separation)
    expected = np.array([exact[a, b] for a, b in DIAGONAL + SHEARS])

    # The residual is the finite difference's own truncation error, which is
    # quadratic in the strain: at 2e-4 it lands five orders below the value.
    assert np.abs(got - expected).max() < 1e-6 * np.abs(expected).max()


def test_a_separation_along_one_axis_leaves_the_other_two_pressures_zero():
    context = _pair_context()
    positions = np.array([[1.5, 1.5, 1.5], [1.95, 1.5, 1.5]])
    box = np.diag([EDGE, EDGE, EDGE])
    got = configurational_stress(
        context, positions, box, positions.copy(), np.array([0, 1]), EDGE**3, DIAGONAL
    )
    assert got[0] != 0.0
    assert abs(got[1]) < 1e-9 * abs(got[0])
    assert abs(got[2]) < 1e-9 * abs(got[0])


def test_shifting_a_molecule_by_a_box_vector_does_not_change_the_stress():
    """The claim that lets the finite difference skip wrapping coordinates.

    A centre of mass outside the cell is scaled by a strain applied to a
    centre that differs from the in-cell one by a lattice vector. That is only
    harmless because the cell's own vectors are strained by the same amount,
    so the displaced molecule lands on the image of where the in-cell one
    would have gone.
    """
    context = _pair_context()
    box = np.diag([EDGE, EDGE, EDGE])
    owner = np.array([0, 1])

    inside = np.array([[1.5, 1.5, 1.5], [1.8, 1.68, 1.72]])
    shifted = inside + np.array([[0.0, 0.0, 0.0], [EDGE, -EDGE, 0.0]])

    a = configurational_stress(
        context, inside, box, inside.copy(), owner, EDGE**3, DIAGONAL + SHEARS
    )
    b = configurational_stress(
        context, shifted, box, shifted.copy(), owner, EDGE**3, DIAGONAL + SHEARS
    )
    assert np.abs(a - b).max() < 1e-6 * np.abs(a).max()


def test_the_context_is_left_holding_what_it_was_given():
    import openmm.unit as u

    context = _pair_context()
    positions = np.array([[1.5, 1.5, 1.5], [1.8, 1.68, 1.72]])
    box = np.diag([EDGE, EDGE, EDGE])
    context.setPositions(positions + np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]))
    elsewhere = (
        context.getState(energy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    )
    context.setPositions(positions)
    expected = (
        context.getState(energy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    )
    configurational_stress(
        context, positions, box, positions.copy(), np.array([0, 1]), EDGE**3, SHEARS
    )
    after = (
        context.getState(energy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    )
    assert after == pytest.approx(expected, abs=1e-9)
    assert elsewhere != expected  # the test would pass vacuously otherwise


def test_the_kinetic_stress_of_an_ideal_gas_is_n_k_t_over_v():
    """Molecular kinetic term against the ideal gas law it has to reproduce.

    Velocities are drawn from a Maxwell-Boltzmann distribution at 300 K for a
    molecular mass of 72 daltons, in nm/ps, and the trace of the resulting
    kinetic stress must be ``3 N k T / V`` to within the sampling error of the
    draw.
    """
    rng = np.random.default_rng(11)
    n_molecules, temperature, volume = 4000, 300.0, 125.0
    mass = 72.0
    # kT in kJ/mol is R T; the velocity variance per component is kT/m.
    gas_constant = 8.31446261815324e-3
    sigma = math.sqrt(gas_constant * temperature / mass)
    velocities = rng.normal(scale=sigma, size=(n_molecules, 3))
    masses = np.full(n_molecules, mass)
    owner = molecule_owner(n_molecules, 1)
    momentum, molecular_mass = molecular_momenta(velocities, masses, owner, n_molecules)

    tensor = kinetic_stress(momentum, molecular_mass, volume)
    ideal = n_molecules * gas_constant * temperature / volume
    assert np.trace(tensor) / 3.0 == pytest.approx(ideal, rel=0.03)
    off = max(abs(tensor[a, b]) for a, b in SHEARS)
    assert off < 0.05 * ideal

    pressure_bar = ideal * KJ_PER_MOL_NM3_IN_PA / 1e5
    assert pressure_bar == pytest.approx(1329.0, rel=0.02)


def test_centres_of_mass_weight_by_mass_and_not_by_count():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    masses = np.array([3.0, 1.0])
    centres = centres_of_mass(positions, masses, molecule_owner(1, 2), 1)
    assert centres[0] == pytest.approx([0.25, 0.0, 0.0])


def test_the_finite_difference_matches_the_force_based_virial_on_a_real_force_field():
    """The pair test uses one interaction; this uses all of OPLS-AA at once.

    An aperiodic cluster has a molecular virial with an exact closed form.
    Displacing every molecule by ``eps`` times its centre of mass changes the
    energy by ``-eps * sum_I R_I . F_I`` to first order, because the
    displacement is affine and the force is minus the gradient - and both
    sides of that are things OpenMM will report. It holds term by term, so it
    checks the finite difference against bonds, angles, torsions, the
    fourteen-scaled exclusions and the full uncut non-bonded sum together,
    which the two-particle case cannot.
    """
    import openmm as mm
    import openmm.unit as u

    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    # Imported before foyer is reachable at all: this module installs the
    # simtk alias that a 2021 foyer still imports under.
    from formulate.physics.md import opls
    from formulate.physics.md.condensed import pack_box
    from formulate.physics.md.stress import STRAIN

    if not opls.available():
        pytest.skip("OPLS-AA is not installed")

    mol = Chem.AddHs(Chem.MolFromSmiles("CCC(C)=O"))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    AllChem.MMFFOptimizeMolecule(mol)
    params = opls.extract_parameters(mol)

    n_molecules = 8
    box = pack_box(mol, n_molecules, 0.75, seed=5)
    system, _ = opls.build_system(params, n_molecules, box_nm=None)
    context = mm.Context(
        system, mm.VerletIntegrator(1.0e-6), mm.Platform.getPlatformByName("Reference")
    )
    positions = box.positions - box.positions.mean(axis=0)
    context.setPositions(positions)
    forces = (
        context.getState(forces=True)
        .getForces(asNumpy=True)
        .value_in_unit(u.kilojoule_per_mole / u.nanometer)
    )

    masses = np.array(params.masses * n_molecules)
    owner = molecule_owner(n_molecules, params.n_atoms)
    centres = centres_of_mass(positions, masses, owner, n_molecules)
    molecular_force = np.zeros((n_molecules, 3))
    np.add.at(molecular_force, owner, forces)

    def energy(shift: np.ndarray) -> float:
        context.setPositions(positions + shift)
        return (
            context.getState(energy=True)
            .getPotentialEnergy()
            .value_in_unit(u.kilojoule_per_mole)
        )

    for alpha, beta in DIAGONAL + SHEARS:
        shift = np.zeros_like(positions)
        shift[:, alpha] = STRAIN * centres[owner, beta]
        derivative = (energy(shift) - energy(-shift)) / (2.0 * STRAIN)
        exact = -float(centres[:, beta] @ molecular_force[:, alpha])
        assert derivative == pytest.approx(exact, rel=1e-4)


def test_the_autocorrelation_recovers_a_known_correlation_time():
    """The estimator the Green-Kubo integral is built on, against two cases.

    White noise has all its correlation at lag zero and none after it. An
    Ornstein-Uhlenbeck process has a single exponential whose time constant
    and whose integral are both known in advance, and the integral is what a
    viscosity actually is.
    """
    from formulate.physics.md.condensed import _autocorrelation

    rng = np.random.default_rng(0)

    white = _autocorrelation(rng.normal(size=200_000), 5)
    assert white[0] == pytest.approx(1.0, abs=0.02)
    assert np.abs(white[1:]).max() < 0.02

    tau, n = 20.0, 400_000
    decay = math.exp(-1.0 / tau)
    kick = math.sqrt(1.0 - decay * decay)
    noise = rng.normal(size=n)
    series = np.zeros(n)
    for i in range(1, n):
        series[i] = decay * series[i - 1] + kick * noise[i]

    acf = _autocorrelation(series, 80)
    lags = np.arange(80)
    fitted = -1.0 / np.polyfit(lags[:60], np.log(acf[:60]), 1)[0]
    assert fitted == pytest.approx(tau, rel=0.05)
    # The integral is the correlation time, less the exponential tail beyond
    # the window and half a step at the origin from the trapezoidal rule.
    assert np.trapezoid(acf, lags) == pytest.approx(tau, rel=0.06)
