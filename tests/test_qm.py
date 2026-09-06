"""Quantum backends: unit discipline, convergence honesty, known values."""

from __future__ import annotations

import numpy as np
import pytest

from formulate.physics.geometry import (
    BOHR_TO_ANGSTROM,
    HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
    HARTREE_TO_EV,
    HARTREE_TO_J_PER_MOL,
    MolecularGeometry,
    geometry_from_smiles,
    rmsd,
)
from formulate.physics.qm import (
    ConvergenceStatus,
    PySCFBackend,
    QMMethod,
    QMRequest,
    XTBBackend,
    available_qm_backends,
    backend_for,
)

pyscf_only = pytest.mark.skipif(
    not PySCFBackend().is_available(), reason="pyscf is not installed"
)
xtb_only = pytest.mark.skipif(not XTBBackend().is_available(), reason="xtb is not installed")

HARTREE_PER_JMOL = 1.0 / HARTREE_TO_J_PER_MOL


def _water(distorted: bool = False) -> MolecularGeometry:
    positions = (
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.20], [1.10, 0.0, -0.30]])
        if distorted
        else np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.958], [0.927, 0.0, -0.240]])
    )
    return MolecularGeometry(symbols=("O", "H", "H"), positions=positions)


# -- conversion factors ----------------------------------------------------


def test_conversion_factors_match_codata():
    """A wrong factor gives a plausible number wrong by a constant - the worst kind."""
    assert HARTREE_TO_EV == pytest.approx(27.2113862, abs=1e-6)
    assert BOHR_TO_ANGSTROM == pytest.approx(0.5291772, abs=1e-6)
    assert HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM == pytest.approx(51.4220675, abs=1e-5)
    assert HARTREE_TO_J_PER_MOL == pytest.approx(2625499.64, abs=1.0)


# -- geometry --------------------------------------------------------------


def test_geometry_validates_its_own_shape():
    with pytest.raises(ValueError):
        MolecularGeometry(symbols=("O", "H"), positions=np.zeros((3, 3)))


def test_geometry_counts_electrons_and_formula():
    water = _water()
    assert water.n_electrons == 10
    assert water.formula == "H2O"
    assert MolecularGeometry(
        symbols=("O", "H", "H"), positions=water.positions, charge=-1
    ).n_electrons == 11


@pytest.mark.skipif(
    not __import__("formulate").chem.rdkit_available(), reason="RDKit is not installed"
)
def test_embedding_records_the_conformer_ensemble_it_chose_from():
    """A property from one arbitrary conformer is not a property of the molecule."""
    flexible = geometry_from_smiles("CCCCCCCC", n_conformers=8)
    assert flexible.conformers_considered > 1
    assert flexible.conformer_energy_spread is not None
    assert "lowest of" in flexible.source


@pytest.mark.skipif(
    not __import__("formulate").chem.rdkit_available(), reason="RDKit is not installed"
)
def test_a_single_conformer_says_so():
    single = geometry_from_smiles("O", n_conformers=1)
    assert any("only one conformer" in n for n in single.notes)


def test_rmsd_requires_matching_atoms():
    with pytest.raises(ValueError):
        rmsd(_water(), MolecularGeometry(symbols=("O", "H", "F"), positions=_water().positions))


# -- backend contract ------------------------------------------------------


def test_registry_routes_a_method_to_a_backend():
    assert available_qm_backends()
    if PySCFBackend().is_available():
        assert backend_for(QMMethod.DFT).id == "pyscf"
    if XTBBackend().is_available():
        assert backend_for(QMMethod.GFN2_XTB).id == "xtb"


def test_a_total_energy_carries_no_error_bar_and_says_why():
    from formulate.physics.qm.base import QMResult

    result = QMResult(status=ConvergenceStatus.FAILED, method_signature="x", backend="y")
    uncertainty = result.energy_uncertainty()
    assert uncertainty.std is None
    assert "not comparable across levels of theory" in uncertainty.basis


# -- PySCF -----------------------------------------------------------------


@pyscf_only
def test_water_energy_matches_the_known_value():
    result = PySCFBackend().run(
        QMRequest(geometry=_water(), method=QMMethod.DFT, basis="6-31g", xc="b3lyp")
    )
    assert result.status is ConvergenceStatus.CONVERGED
    hartree = result.total_energy.value * HARTREE_PER_JMOL
    assert hartree == pytest.approx(-76.38, abs=0.05)


@pyscf_only
def test_hartree_fock_reproduces_the_textbook_h2_energy():
    h2 = MolecularGeometry(
        symbols=("H", "H"), positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    )
    result = PySCFBackend().run(
        QMRequest(geometry=h2, method=QMMethod.HARTREE_FOCK, basis="sto-3g")
    )
    hartree = result.total_energy.value * HARTREE_PER_JMOL
    assert hartree == pytest.approx(-1.1167, abs=1e-3)


@pyscf_only
def test_geometry_optimisation_reaches_the_literature_structure():
    """The whole ASE/PySCF unit boundary is validated by this one number.

    A wrong Hartree/Bohr to eV/Angstrom factor, or a missing gradient sign,
    still optimises - to the wrong geometry.
    """
    result = PySCFBackend().run(
        QMRequest(
            geometry=_water(distorted=True),
            method=QMMethod.HARTREE_FOCK,
            basis="sto-3g",
            optimize_geometry=True,
            max_optimization_steps=60,
            force_threshold=0.02,
        )
    )
    assert result.status is ConvergenceStatus.CONVERGED
    relaxed = result.optimized_geometry
    assert relaxed is not None

    positions = relaxed.positions
    bond_a = float(np.linalg.norm(positions[1] - positions[0]))
    bond_b = float(np.linalg.norm(positions[2] - positions[0]))
    unit_a = (positions[1] - positions[0]) / bond_a
    unit_b = (positions[2] - positions[0]) / bond_b
    angle = float(np.degrees(np.arccos(np.clip(unit_a @ unit_b, -1, 1))))

    # HF/STO-3G water: O-H 0.989 Angstrom, H-O-H 100.0 degrees.
    assert bond_a == pytest.approx(0.989, abs=0.01)
    assert bond_b == pytest.approx(0.989, abs=0.01)
    assert angle == pytest.approx(100.0, abs=1.5)
    assert result.max_residual_force < 0.02


@pyscf_only
def test_forces_are_returned_as_forces_not_gradients():
    """At a stretched bond the force must pull the atoms together."""
    stretched = MolecularGeometry(
        symbols=("H", "H"), positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.20]])
    )
    result = PySCFBackend().run(
        QMRequest(
            geometry=stretched,
            method=QMMethod.HARTREE_FOCK,
            basis="sto-3g",
            compute_forces=True,
        )
    )
    forces = np.asarray(result.forces)
    assert forces.shape == (2, 3)
    # Atom 1 sits at +z of atom 0, so a restoring force on it points to -z.
    assert forces[1][2] < 0
    assert forces[0][2] > 0


@pyscf_only
def test_an_unconverged_scf_reports_no_energy_at_all():
    """A non-converged SCF leaves a number in memory; returning it is the bug."""
    result = PySCFBackend().run(
        QMRequest(
            geometry=_water(),
            method=QMMethod.DFT,
            basis="6-31g",
            convergence=1e-14,
            max_scf_cycles=1,
        )
    )
    assert result.status is ConvergenceStatus.NOT_CONVERGED
    assert result.total_energy is None
    assert not result.usable
    assert any("did not converge" in d for d in result.diagnostics)


@pyscf_only
def test_every_result_states_what_the_method_cannot_support():
    result = PySCFBackend().run(
        QMRequest(geometry=_water(), method=QMMethod.DFT, basis="sto-3g")
    )
    joined = " ".join(result.limitations)
    assert "dispersion" in joined or "London" in joined
    assert "vacuum" in joined
    assert "conformer" in joined


@pyscf_only
def test_a_bad_basis_fails_without_raising():
    result = PySCFBackend().run(
        QMRequest(geometry=_water(), method=QMMethod.HARTREE_FOCK, basis="not-a-basis")
    )
    assert result.status is ConvergenceStatus.FAILED
    assert result.diagnostics


@pyscf_only
def test_the_multiplicity_convention_is_converted_for_pyscf():
    """PySCF's `spin` is 2S, not 2S+1; passing a multiplicity changes the state."""
    backend = PySCFBackend()
    doublet = MolecularGeometry(
        symbols=("O", "H"),
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]]),
        spin_multiplicity=2,
    )
    mol = backend._build_molecule(doublet, QMRequest(geometry=doublet, basis="sto-3g"))
    assert mol.spin == 1


# -- xtb -------------------------------------------------------------------


@xtb_only
def test_xtb_runs_and_refuses_to_report_a_dft_comparable_gap():
    result = XTBBackend().run(
        QMRequest(geometry=_water(), method=QMMethod.GFN2_XTB)
    )
    assert result.status is ConvergenceStatus.CONVERGED
    assert result.total_energy is not None
    assert result.homo_lumo_gap is None
    assert any("not comparable" in d for d in result.diagnostics)


@xtb_only
def test_xtb_declines_a_method_it_does_not_implement():
    result = XTBBackend().run(QMRequest(geometry=_water(), method=QMMethod.DFT))
    assert result.status is ConvergenceStatus.FAILED


@xtb_only
def test_xtb_optimisation_produces_a_sane_water_geometry():
    result = XTBBackend().run(
        QMRequest(
            geometry=_water(distorted=True),
            method=QMMethod.GFN2_XTB,
            optimize_geometry=True,
            max_optimization_steps=80,
            force_threshold=0.05,
        )
    )
    positions = result.optimized_geometry.positions
    bond = float(np.linalg.norm(positions[1] - positions[0]))
    assert bond == pytest.approx(0.96, abs=0.05)
