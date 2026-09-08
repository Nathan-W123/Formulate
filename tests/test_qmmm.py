"""A quantum region embedded in a classical one (specification section 8).

Most of these tests need no quantum calculation: where the boundary is, what
happens to the bonds it cuts, and what is refused are all decidable from the
graph, and those are the parts that go wrong silently. The calculations
themselves are marked slow.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_rdkit

from formulate.physics.qmmm import (
    EmbeddingMode,
    LinkAtom,
    QMMMCalculator,
    QMMMRegion,
    QMMMResult,
    UnsupportedPartition,
    partition,
)


def _build(smiles: str, seed: int = 7):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _with_hydrogens(mol, heavy):
    """A region is only well posed if each hydrogen goes with its heavy atom."""
    keep = set(heavy)
    return sorted(
        keep
        | {
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 1 and atom.GetNeighbors()[0].GetIdx() in keep
        }
    )


@pytest.fixture
def hexanol():
    return _build("CCCCCCO")


@pytest.fixture
def hexanol_region(hexanol):
    """QM on the alcohol end, MM on the tail, cutting one C-C single bond."""
    return partition(hexanol, _with_hydrogens(hexanol, [4, 5, 6]))


# -- the partition ---------------------------------------------------------


@requires_rdkit
def test_the_boundary_produces_one_link_atom_per_cut_bond(hexanol_region):
    assert len(hexanol_region.link_atoms) == 1
    link = hexanol_region.link_atoms[0]
    assert link.qm_index == 4
    assert link.mm_index == 3


@requires_rdkit
def test_the_capping_hydrogen_sits_on_the_bond_it_replaces(hexanol_region):
    """Along the original bond vector, not at an arbitrary point."""
    region = hexanol_region
    link = region.link_atoms[0]
    bond = region.positions[link.mm_index] - region.positions[link.qm_index]
    placed = np.asarray(link.position) - region.positions[link.qm_index]
    cosine = float(
        np.dot(bond, placed) / (np.linalg.norm(bond) * np.linalg.norm(placed))
    )
    assert cosine == pytest.approx(1.0, abs=1e-9)


@requires_rdkit
def test_the_capping_hydrogen_is_a_carbon_hydrogen_bond_away(hexanol_region):
    link = hexanol_region.link_atoms[0]
    distance = float(
        np.linalg.norm(np.asarray(link.position) - hexanol_region.positions[link.qm_index])
    )
    assert distance == pytest.approx(1.09, abs=0.06)


@requires_rdkit
def test_the_placement_scales_with_the_bond_rather_than_being_fixed(hexanol_region):
    """A stretched boundary bond must not put the hydrogen inside the QM atom."""
    assert hexanol_region.link_atoms[0].scale == pytest.approx(1.09 / 1.53, rel=1e-6)


@requires_rdkit
def test_the_frontier_charge_is_shifted_rather_than_left_under_the_link_atom(
    hexanol_region,
):
    """A full atomic charge a bond length from the hydrogen representing it
    over-polarises the quantum region."""
    region = hexanol_region
    frontier = region.link_atoms[0].mm_index
    position = region.mm_indices.index(frontier)
    assert region.mm_charges[position] == 0.0
    assert region.link_atoms[0].shifted_charge != 0.0


@requires_rdkit
def test_shifting_the_charge_conserves_the_classical_region_total(hexanol_region):
    """Deleting it instead would change the net charge of the system."""
    assert hexanol_region.mm_net_charge == pytest.approx(
        -sum(
            _opls(hexanol_region)[i] for i in hexanol_region.qm_indices
        ),
        abs=1e-6,
    )


def _opls(region: QMMMRegion) -> dict[int, float]:
    """The original per-atom charges, reconstructed from the region."""
    charges = {i: 0.0 for i in range(len(region.symbols))}
    for index, value in zip(region.mm_indices, region.mm_charges):
        charges[index] = value
    for link in region.link_atoms:
        charges[link.mm_index] = link.shifted_charge
    return charges


@requires_rdkit
def test_the_quantum_region_includes_its_capping_atoms(hexanol_region):
    assert hexanol_region.n_qm == len(hexanol_region.qm_indices) + 1
    assert len(hexanol_region.qm_xyz_block().splitlines()) == hexanol_region.n_qm


@requires_rdkit
def test_a_partition_with_no_cut_bonds_needs_no_link_atom():
    """Two separate molecules: the boundary crosses nothing."""
    from rdkit import Chem

    pair = Chem.AddHs(Chem.MolFromSmiles("CCO.CCO"))
    from rdkit.Chem import AllChem

    AllChem.EmbedMolecule(pair, randomSeed=3)
    fragments = Chem.GetMolFrags(pair)
    region = partition(pair, list(fragments[0]), charges=[0.0] * pair.GetNumAtoms())
    assert region.link_atoms == ()


# -- what is refused -------------------------------------------------------


@requires_rdkit
def test_polarizable_embedding_refuses_and_names_both_halves_of_the_blocker():
    mol = _build("CCCCCCO")
    with pytest.raises(UnsupportedPartition) as caught:
        partition(mol, _with_hydrogens(mol, [4, 5, 6]), embedding=EmbeddingMode.POLARIZABLE)
    message = str(caught.value)
    assert "polarizabilities" in message
    assert "OPLS-AA" in message
    assert "PySCF" in message


@requires_rdkit
def test_cutting_a_double_bond_is_refused():
    mol = _build("CCC=CCC")
    with pytest.raises(UnsupportedPartition, match="double"):
        partition(mol, _with_hydrogens(mol, [0, 1, 2]))


@requires_rdkit
def test_cutting_into_an_aromatic_ring_names_the_delocalisation():
    """Not "one hydrogen cannot cap it", which is true of the bond order and
    silent about what actually makes the partition wrong."""
    mol = _build("Cc1ccccc1")
    with pytest.raises(UnsupportedPartition) as caught:
        partition(mol, _with_hydrogens(mol, [1, 2, 3]))
    assert "delocalisation" in str(caught.value)


@requires_rdkit
def test_cutting_a_polar_bond_is_refused():
    mol = _build("CCCCCCO")
    with pytest.raises(UnsupportedPartition, match="polar C-O"):
        partition(mol, [6])


@requires_rdkit
def test_separating_a_hydrogen_from_its_own_atom_is_refused():
    """The link atom would be a hydrogen capping where a hydrogen already was."""
    mol = _build("CCCCCCO")
    with pytest.raises(UnsupportedPartition, match="C-H bond"):
        partition(mol, [4, 5, 6])


@requires_rdkit
def test_an_empty_or_total_quantum_region_is_refused():
    mol = _build("CCCCCCO")
    with pytest.raises(UnsupportedPartition, match="empty"):
        partition(mol, [])
    with pytest.raises(UnsupportedPartition, match="nothing to embed it in"):
        partition(mol, list(range(mol.GetNumAtoms())))


@requires_rdkit
def test_an_index_outside_the_molecule_is_refused():
    mol = _build("CCCCCCO")
    with pytest.raises(UnsupportedPartition, match="must lie in"):
        partition(mol, [0, 999])


@requires_rdkit
def test_a_wrong_length_charge_list_is_refused():
    mol = _build("CCCCCCO")
    with pytest.raises(UnsupportedPartition, match="one per atom"):
        partition(mol, _with_hydrogens(mol, [4, 5, 6]), charges=[0.0, 0.0])


@requires_rdkit
def test_a_molecule_the_force_field_cannot_type_is_refused_rather_than_guessed():
    """Water has no OPLS-AA typing in foyer's distribution, and inventing
    charges here would make the whole calculation fiction.

    Two separate waters, so the partition itself is clean - no bond is cut -
    and the only thing left to fail is the charge assignment.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    pair = Chem.AddHs(Chem.MolFromSmiles("O.O"))
    AllChem.EmbedMolecule(pair, randomSeed=3)
    first = list(Chem.GetMolFrags(pair)[0])
    with pytest.raises(UnsupportedPartition, match="no MM charges"):
        partition(pair, first)


# -- capability reporting --------------------------------------------------


def test_the_backend_states_what_it_can_and_cannot_do():
    capabilities = QMMMCalculator().capabilities()
    assert "electrostatic" in capabilities["embedding_modes"]
    assert "mechanical" in capabilities["embedding_modes"]
    assert "polarizable" not in capabilities["embedding_modes"]
    assert "polarizable" in capabilities["unsupported_embedding_modes"]
    assert capabilities["gradients"] is False
    assert "OPLS" in capabilities["mm_charge_source"]


def test_the_result_reports_the_embedding_energy_rather_than_an_interaction_energy():
    region = QMMMRegion(
        symbols=("C",), positions=np.zeros((1, 3)), qm_indices=(0,),
        mm_indices=(), mm_charges=(),
    )
    result = QMMMResult(
        total_energy_hartree=-1.0, vacuum_energy_hartree=-0.99, region=region,
        method_signature="QM/MM HF/sto-3g",
    )
    assert result.embedding_energy_kj_per_mol == pytest.approx(-26.25, abs=0.1)


def test_a_link_atom_describes_where_it_went_and_what_it_took():
    described = LinkAtom(
        qm_index=4, mm_index=3, position=(0.0, 0.0, 1.09), scale=0.712,
        shifted_charge=-0.12,
    ).describe()
    assert "QM atom 4" in described and "MM atom 3" in described
    assert "-0.1200 e shifted" in described


# -- the calculation -------------------------------------------------------


@requires_rdkit
@pytest.mark.slow
def test_electrostatic_embedding_shifts_the_energy_and_mechanical_does_not(
    hexanol, hexanol_region
):
    calculator = QMMMCalculator(basis="sto-3g", use_dft=False)
    if not calculator.is_available():
        pytest.skip(calculator.unavailable_reason())

    electrostatic = calculator.run(hexanol_region)
    mechanical = calculator.run(
        partition(
            hexanol,
            _with_hydrogens(hexanol, [4, 5, 6]),
            embedding=EmbeddingMode.MECHANICAL,
        )
    )
    assert electrostatic.converged
    assert electrostatic.embedding_energy_kj_per_mol != 0.0
    # Mechanical embedding puts nothing in the Hamiltonian, so reporting a
    # shift would mean a calculation had happened that had not.
    assert mechanical.embedding_energy_kj_per_mol == 0.0
    assert any("unpolarised" in limit for limit in mechanical.limitations)


@requires_rdkit
@pytest.mark.slow
def test_every_result_states_what_the_number_leaves_out(hexanol_region):
    calculator = QMMMCalculator(basis="sto-3g", use_dft=False)
    if not calculator.is_available():
        pytest.skip(calculator.unavailable_reason())
    result = calculator.run(hexanol_region)
    joined = " ".join(result.limitations)
    assert "one-way" in joined
    assert "dispersion" in joined
    assert "link atom" in joined


@pytest.mark.slow
def test_the_embedding_falls_off_like_a_charge_dipole_interaction():
    """Physics rather than plumbing: a wrong sign or a misplaced charge fails.

    A neutral water molecule with a point charge on its dipole axis is a
    charge-dipole interaction, which goes as 1/r^2. Doubling the distance must
    therefore quarter the energy.
    """
    pytest.importorskip("pyscf")
    from pyscf import gto, qmmm, scf

    water = (
        "O 0.0000 0.0000 0.1173\n"
        "H 0.0000 0.7572 -0.4692\n"
        "H 0.0000 -0.7572 -0.4692"
    )

    def shift(distance: float) -> float:
        molecule = gto.M(atom=water, basis="6-31g", verbose=0)
        bare = scf.RHF(molecule).run().e_tot
        embedded = qmmm.add_mm_charges(
            scf.RHF(molecule), np.array([[0.0, 0.0, distance]]), np.array([1.0])
        ).run().e_tot
        return (embedded - bare) * 2625.5

    near, far = shift(8.0), shift(16.0)
    assert near < 0 and far < 0
    assert near / far == pytest.approx(4.0, rel=0.05)


@pytest.mark.slow
def test_reversing_the_charge_reverses_the_sign_and_leaves_induction_behind():
    """The residue after cancelling the sign is polarisation, always stabilising."""
    pytest.importorskip("pyscf")
    from pyscf import gto, qmmm, scf

    water = (
        "O 0.0000 0.0000 0.1173\n"
        "H 0.0000 0.7572 -0.4692\n"
        "H 0.0000 -0.7572 -0.4692"
    )

    def shift(charge: float) -> float:
        molecule = gto.M(atom=water, basis="6-31g", verbose=0)
        bare = scf.RHF(molecule).run().e_tot
        embedded = qmmm.add_mm_charges(
            scf.RHF(molecule), np.array([[0.0, 0.0, 4.0]]), np.array([charge])
        ).run().e_tot
        return (embedded - bare) * 2625.5

    positive, negative = shift(1.0), shift(-1.0)
    assert positive < 0 < negative
    assert positive + negative < 0
