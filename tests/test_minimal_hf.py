"""A Hartree-Fock implementation with nothing hidden inside it.

Specification section 7's optional teaching backend. Every number it produces
is checked against PySCF, because a teaching implementation that is subtly
wrong teaches the wrong thing, and "it looks like the right sort of number" is
how that happens.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from formulate.physics.qm.minimal_hf import (
    PRODUCTION_WARNING,
    STO3G,
    ContractedGaussian,
    Molecule,
    UnsupportedSystem,
    boys_f0,
    boys_f1,
    electron_repulsion_tensor,
    forces,
    hellmann_feynman_forces,
    kinetic_matrix,
    nuclear_attraction_matrix,
    nuclear_repulsion,
    overlap_matrix,
    run,
    scf,
)

_H2 = (["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]))


def _pyscf(symbols, positions, basis="sto-3g"):
    from pyscf import gto, scf as pyscf_scf

    atom = "\n".join(
        f"{s} {x} {y} {z}" for s, (x, y, z) in zip(symbols, positions)
    )
    return pyscf_scf.RHF(gto.M(atom=atom, basis=basis, verbose=0)).run()


# -- basis functions --------------------------------------------------------


def test_the_stored_sto3g_parameters_are_the_published_ones():
    """A transcription error in the table would be invisible in the energy."""
    pytest.importorskip("pyscf")
    from pyscf import gto

    for symbol in STO3G:
        shells = gto.basis.load("sto-3g", symbol)
        assert len(shells) == 1, f"{symbol} has more than one shell in STO-3G"
        assert shells[0][0] == 0, f"{symbol} has a non-s shell"
        for mine, theirs in zip(STO3G[symbol], shells[0][1:]):
            assert mine[0] == pytest.approx(theirs[0], rel=1e-7)
            assert mine[1] == pytest.approx(theirs[1], rel=1e-7)


def test_a_basis_function_is_a_function_that_can_be_evaluated():
    """Section 7 asks the basis functions to be exposed, not merely used."""
    function = ContractedGaussian.sto3g("H", (0.0, 0.0, 0.0))
    assert function((0.0, 0.0, 0.0)) > 0
    # It decays but never reaches zero: the most diffuse STO-3G primitive still
    # contributes about 4e-9 at ten Bohr, so the test is that it falls off, not
    # that it vanishes.
    assert function((0.0, 0.0, 10.0)) < 1e-8
    assert function((0.0, 0.0, 2.0)) < function((0.0, 0.0, 1.0)) < function((0.0, 0.0, 0.0))
    # An s function is spherically symmetric.
    assert function((0.0, 0.0, 0.5)) == pytest.approx(function((0.5, 0.0, 0.0)))


def test_a_normalised_basis_function_overlaps_itself_to_one():
    function = ContractedGaussian.sto3g("H", (0.0, 0.0, 0.0))
    assert overlap_matrix([function])[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_an_element_needing_p_functions_is_refused_rather_than_truncated():
    """Dropping carbon's 2p shell returns a number that looks like an energy."""
    with pytest.raises(UnsupportedSystem, match="p functions"):
        ContractedGaussian.sto3g("C", (0.0, 0.0, 0.0))
    with pytest.raises(UnsupportedSystem, match="p functions"):
        Molecule.build(["C", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.09]]))


# -- the Boys function ------------------------------------------------------


def test_the_boys_function_is_one_at_the_origin():
    """Its closed form is 0/0 there, so the limit has to be taken explicitly."""
    assert boys_f0(0.0) == pytest.approx(1.0, abs=1e-12)
    assert boys_f1(0.0) == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_the_boys_series_and_closed_form_agree_where_they_meet():
    """The series exists because the closed form loses precision for small x."""
    for x in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
        closed = 0.5 * math.sqrt(math.pi / x) * math.erf(math.sqrt(x))
        assert boys_f0(x) == pytest.approx(closed, rel=1e-9)


def test_f1_is_minus_the_derivative_of_f0():
    for x in (0.1, 1.0, 5.0, 20.0):
        h = 1e-6
        derivative = (boys_f0(x + h) - boys_f0(x - h)) / (2 * h)
        assert boys_f1(x) == pytest.approx(-derivative, rel=1e-5)


def test_the_boys_function_decays():
    assert boys_f0(0.5) > boys_f0(5.0) > boys_f0(50.0) > 0.0


# -- integrals against a reference -----------------------------------------


def test_the_overlap_and_core_hamiltonian_match_pyscf():
    pytest.importorskip("pyscf")
    from pyscf import gto

    symbols, positions = _H2
    molecule = Molecule.build(symbols, positions)
    reference = gto.M(atom="H 0 0 0\nH 0 0 0.74", basis="sto-3g", verbose=0)

    S = overlap_matrix(molecule.basis)
    H = kinetic_matrix(molecule.basis) + nuclear_attraction_matrix(
        molecule.basis, molecule.positions, molecule.charges
    )
    # 1e-7 rather than machine precision: the stored contraction coefficients
    # are the published eight-figure values, not PySCF's internal ones.
    assert np.abs(S - reference.intor("int1e_ovlp")).max() < 1e-7
    assert np.abs(H - _pyscf(*_H2).get_hcore()).max() < 1e-7


def test_the_two_electron_integrals_match_pyscf():
    pytest.importorskip("pyscf")
    from pyscf import gto

    symbols, positions = _H2
    molecule = Molecule.build(symbols, positions)
    reference = gto.M(atom="H 0 0 0\nH 0 0 0.74", basis="sto-3g", verbose=0)
    assert np.abs(
        electron_repulsion_tensor(molecule.basis) - reference.intor("int2e")
    ).max() < 1e-7


def test_the_two_electron_integrals_have_the_symmetries_they_must():
    molecule = Molecule.build(*_H2)
    eri = electron_repulsion_tensor(molecule.basis)
    assert np.allclose(eri, np.transpose(eri, (1, 0, 2, 3)))  # (ij|kl) = (ji|kl)
    assert np.allclose(eri, np.transpose(eri, (0, 1, 3, 2)))  # (ij|kl) = (ij|lk)
    assert np.allclose(eri, np.transpose(eri, (2, 3, 0, 1)))  # (ij|kl) = (kl|ij)


def test_nuclear_repulsion_is_the_textbook_sum():
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    assert nuclear_repulsion(positions, np.array([1.0, 1.0])) == pytest.approx(0.5)


# -- the self-consistent field ---------------------------------------------


@pytest.mark.parametrize(
    "symbols,positions",
    [
        (["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        (["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.20]]),
        (["He"], [[0.0, 0.0, 0.0]]),
        (["He", "He"], [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]),
    ],
)
def test_the_energy_matches_pyscf(symbols, positions):
    pytest.importorskip("pyscf")
    positions = np.array(positions)
    mine = run(symbols, positions)
    assert mine.converged
    assert mine.energy == pytest.approx(_pyscf(symbols, positions).e_tot, abs=1e-8)


def test_the_orbital_energies_match_pyscf():
    pytest.importorskip("pyscf")
    mine = run(*_H2)
    assert np.allclose(mine.orbital_energies, _pyscf(*_H2).mo_energy, atol=1e-7)


def test_the_density_integrates_to_the_electron_count():
    """tr(PS) is the number of electrons, and a wrong occupation shows here."""
    assert run(*_H2).n_electrons_from_density == pytest.approx(2.0, abs=1e-9)
    assert run(["He"], np.zeros((1, 3))).n_electrons_from_density == pytest.approx(
        2.0, abs=1e-9
    )


def test_the_convergence_history_is_available_rather_than_only_the_answer():
    result = run(*_H2)
    assert result.iterations >= 2
    assert len(result.history) == result.iterations
    assert abs(result.history[-1]) < 1e-9


def test_the_energy_splits_into_the_two_parts_it_is_made_of():
    result = run(*_H2)
    assert result.energy == pytest.approx(
        result.electronic_energy + result.nuclear_repulsion, abs=1e-12
    )
    assert result.nuclear_repulsion > 0


def test_an_open_shell_system_is_refused_rather_than_rounded():
    """Restricted Hartree-Fock doubly occupies orbitals; half an electron has
    nowhere to go."""
    with pytest.raises(UnsupportedSystem, match="open shell"):
        run(["H"], np.zeros((1, 3)))


def test_two_nuclei_on_top_of_each_other_are_refused():
    """Caught at construction, before the infinite nuclear repulsion is computed.

    The overlap matrix's linear-dependence guard would also catch it, but only
    after a divide-by-zero had already happened, and it would blame the basis
    for what is a broken geometry.
    """
    with pytest.raises(UnsupportedSystem, match="same point"):
        run(["H", "H"], np.zeros((2, 3)))


def test_every_result_says_it_is_not_for_production():
    result = run(*_H2)
    assert PRODUCTION_WARNING in result.warnings
    assert "EDUCATIONAL" in result.describe()


# -- where nuclear forces come from ----------------------------------------


def test_the_total_force_matches_pyscfs_analytic_gradient():
    pytest.importorskip("pyscf")
    analysis = forces(Molecule.build(*_H2))
    reference = -_pyscf(*_H2).nuc_grad_method().kernel()
    assert np.abs(analysis.total - reference).max() < 1e-5


@pytest.mark.parametrize("distance", [0.60, 0.74, 1.20])
def test_the_hellmann_feynman_force_matches_its_own_definition(distance):
    """Move only the point charge, holding the density and the basis fixed.

    That is what the Hellmann-Feynman term *is*, so the analytic formula must
    reproduce it. The check caught a dropped minus sign: the energy contains
    sum_ij P_ij V_ij and the force is minus its derivative, and without the
    outer sign the electron density pushed the nuclei apart instead of pulling
    them together, making the term thirty-six times the total force on H2.
    """
    molecule = Molecule.build(["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, distance]]))
    converged = scf(molecule)
    analytic = hellmann_feynman_forces(molecule, converged)

    step = 1e-5
    numeric = np.zeros((2, 3))
    for atom in range(2):
        for axis in range(3):
            energies = []
            for sign in (1, -1):
                shifted = molecule.positions.copy()
                shifted[atom, axis] += sign * step
                # The basis stays where it was: only the charge moves.
                V = nuclear_attraction_matrix(molecule.basis, shifted, molecule.charges)
                energies.append(
                    float(np.sum(converged.density * V))
                    + nuclear_repulsion(shifted, molecule.charges)
                )
            numeric[atom, axis] = -(energies[0] - energies[1]) / (2 * step)
    assert np.abs(analytic - numeric).max() < 1e-8


def test_the_pulay_force_is_what_is_left_over():
    analysis = forces(Molecule.build(*_H2))
    assert np.allclose(analysis.pulay, analysis.total - analysis.hellmann_feynman)


def test_the_hellmann_feynman_force_is_not_the_whole_force():
    """The lesson: in a finite basis it is not even close.

    At H2's equilibrium the total force is about +0.028 Hartree/Bohr and the
    Hellmann-Feynman part is about -0.051 - the wrong sign, because the basis
    functions ride on the nuclei and moving one changes the basis itself.
    """
    analysis = forces(Molecule.build(*_H2))
    assert abs(analysis.pulay[0, 2]) > abs(analysis.total[0, 2])
    assert np.sign(analysis.hellmann_feynman[0, 2]) != np.sign(analysis.total[0, 2])


def test_the_forces_on_a_diatomic_are_equal_and_opposite():
    """Newton's third law, which a sign error in either term would break."""
    analysis = forces(Molecule.build(*_H2))
    assert np.allclose(analysis.total[0], -analysis.total[1], atol=1e-8)
    assert np.allclose(
        analysis.hellmann_feynman[0], -analysis.hellmann_feynman[1], atol=1e-8
    )


def test_the_force_analysis_explains_the_gap_rather_than_only_printing_it():
    described = forces(Molecule.build(*_H2)).describe()
    assert "Pulay" in described
    assert "ride on the nuclei" in described
    assert "EDUCATIONAL" in described


def test_a_stretched_bond_pulls_together_and_a_compressed_one_pushes_apart():
    """The sign of the force must track the equilibrium, near 0.735 A here."""
    stretched = forces(
        Molecule.build(["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.10]]))
    )
    compressed = forces(
        Molecule.build(["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.55]]))
    )
    assert stretched.total[0, 2] > 0  # atom 0 pulled toward atom 1
    assert compressed.total[0, 2] < 0  # atom 0 pushed away


# -- the uniform backend interface ------------------------------------------


def test_the_teaching_backend_is_never_selected_for_a_real_method():
    """It reproduces PySCF to twelve decimals on H and He, which is exactly
    what would make it dangerous to reach for: it looks right until carbon."""
    from formulate.physics.qm import MinimalHFBackend, backend_for
    from formulate.physics.qm.base import QMMethod

    assert MinimalHFBackend.production is False
    assert backend_for(QMMethod.MINIMAL_HF).id == "minimal-hf"
    for method in (QMMethod.HARTREE_FOCK, QMMethod.DFT):
        chosen = backend_for(method)
        assert chosen is None or chosen.production


def test_the_backend_refuses_to_stand_in_for_another_method():
    from formulate.physics.qm import MinimalHFBackend
    from formulate.physics.qm.base import ConvergenceStatus, QMMethod, QMRequest
    from formulate.physics.geometry import MolecularGeometry

    geometry = MolecularGeometry(
        symbols=("H", "H"), positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    )
    result = MinimalHFBackend().run(
        QMRequest(geometry=geometry, method=QMMethod.DFT, optimize_geometry=False)
    )
    assert result.status is ConvergenceStatus.FAILED
    assert "will not stand in" in result.diagnostics[0]


def test_the_backend_reproduces_the_module_energy_through_the_uniform_interface():
    pytest.importorskip("pyscf")
    from formulate.physics.geometry import MolecularGeometry
    from formulate.physics.qm import MinimalHFBackend
    from formulate.physics.qm.base import QMMethod, QMRequest

    geometry = MolecularGeometry(
        symbols=("H", "H"), positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    )
    result = MinimalHFBackend().run(
        QMRequest(geometry=geometry, method=QMMethod.MINIMAL_HF, optimize_geometry=False)
    )
    assert result.usable
    hartree = result.total_energy.to("J/mol").value / 2625499.639479828
    assert hartree == pytest.approx(_pyscf(*_H2).e_tot, abs=1e-8)
    assert PRODUCTION_WARNING in result.limitations


def test_the_backend_refuses_carbon_through_the_interface_too():
    from formulate.physics.geometry import MolecularGeometry
    from formulate.physics.qm import MinimalHFBackend
    from formulate.physics.qm.base import ConvergenceStatus, QMMethod, QMRequest

    geometry = MolecularGeometry(
        symbols=("C", "H"), positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.09]])
    )
    result = MinimalHFBackend().run(
        QMRequest(geometry=geometry, method=QMMethod.MINIMAL_HF, optimize_geometry=False)
    )
    assert result.status is ConvergenceStatus.FAILED
    assert "p functions" in result.diagnostics[0]


def test_the_backend_declines_a_geometry_optimisation_it_cannot_do():
    from formulate.physics.geometry import MolecularGeometry
    from formulate.physics.qm import MinimalHFBackend
    from formulate.physics.qm.base import ConvergenceStatus, QMMethod, QMRequest

    geometry = MolecularGeometry(
        symbols=("H", "H"), positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    )
    result = MinimalHFBackend().run(
        QMRequest(geometry=geometry, method=QMMethod.MINIMAL_HF, optimize_geometry=True)
    )
    assert result.status is ConvergenceStatus.FAILED
    assert "no geometry optimiser" in result.diagnostics[0]
