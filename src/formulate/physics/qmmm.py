"""QM/MM: a quantum region embedded in a classical one.

Specification section 8. The point of a QM/MM partition is that most of a large
system does not need quantum mechanics - a reaction centre or a chromophore
does, and the protein or solvent around it can be point charges. What makes it
a method rather than a convenience is that the three hard parts are handled
explicitly: where the boundary is, what happens to the bonds it cuts, and
whether the classical region is allowed to polarise the quantum one.

What is implemented
    Mechanical and electrostatic embedding, both real. Electrostatic embedding
    puts the MM point charges into the QM one-electron Hamiltonian through
    PySCF's ``qmmm.add_mm_charges``, so the MM region polarises the QM density;
    mechanical embedding does not, and the QM region is computed in vacuum with
    the electrostatics left to the force field. MM charges come from OPLS-AA
    through :mod:`formulate.physics.md.opls`, which types them with foyer and
    refuses molecules it has no parameters for, so a charge here is a published
    force-field charge rather than something assigned in this file.

    Covalent boundaries are capped with link atoms, and the frontier MM charge
    is shifted rather than left sitting under the capping hydrogen.

What is refused, and why
    Cutting anything but a single bond between two carbons. A double bond
    cannot be capped by one hydrogen; a bond into an aromatic ring cuts a
    delocalised system that a hydrogen does not restore; a polar bond puts the
    link atom where the charge distribution it is meant to imitate is not the
    one it produces. These are refusals rather than warnings because each of
    them silently produces a number.

The concrete blocker
    Polarizable embedding, where the MM region responds to the QM density
    rather than only acting on it, is not implemented and cannot be with what
    is installed. It needs MM polarizabilities - a Drude or AMOEBA-style force
    field - and OPLS-AA is a fixed-charge force field with no polarizabilities
    to read. PySCF has no polarizable-embedding driver in this build either.
    :class:`EmbeddingMode.POLARIZABLE` therefore exists and refuses, naming
    both halves of the blocker, rather than being silently absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np

#: Bohr per Angstrom, for handing coordinates to PySCF.
_ANGSTROM_PER_BOHR = 0.529177210903
#: Hartree to kJ/mol.
_KJ_PER_MOL_PER_HARTREE = 2625.499639479828

#: Equilibrium bond lengths in Angstrom, for placing a capping hydrogen along
#: the bond it replaces. The link atom sits at a fraction of the original bond
#: length rather than at a fixed distance, so a stretched boundary bond does not
#: put the hydrogen inside the QM atom.
_BOND_LENGTH = {("C", "C"): 1.53, ("C", "H"): 1.09}


class EmbeddingMode(str, Enum):
    """How much the classical region is allowed to do to the quantum one."""

    #: MM charges do not enter the QM Hamiltonian. The QM region is computed in
    #: vacuum and the electrostatics between regions are the force field's.
    MECHANICAL = "mechanical"
    #: MM point charges enter the QM one-electron Hamiltonian, so the MM region
    #: polarises the QM density. One-way: the QM density does not act back.
    ELECTROSTATIC = "electrostatic"
    #: Mutual polarisation. Not implemented; see the module docstring.
    POLARIZABLE = "polarizable"


class UnsupportedPartition(Exception):
    """This partition cannot be computed correctly, with the reason why."""


@dataclass(frozen=True, slots=True)
class LinkAtom:
    """A hydrogen capping a covalent bond the boundary cut.

    Placed along the original bond vector at ``scale`` of its length, where
    ``scale`` is the ratio of the equilibrium C-H bond to the equilibrium bond
    being replaced. Placing it at a fixed 1.09 Angstrom instead would move it
    relative to the QM atom whenever the boundary bond is stretched, which is
    exactly when the geometry is interesting.
    """

    #: Index, in the whole system, of the QM atom that keeps the bond.
    qm_index: int
    #: Index of the MM atom the bond went to, now replaced.
    mm_index: int
    position: tuple[float, float, float]
    scale: float
    #: Charge lifted off the frontier MM atom and spread over its neighbours.
    shifted_charge: float = 0.0

    def describe(self) -> str:
        return (
            f"link atom capping the bond from QM atom {self.qm_index} to MM atom "
            f"{self.mm_index}, placed at {self.scale:.3f} of the bond length; "
            f"{self.shifted_charge:+.4f} e shifted off the frontier atom"
        )


@dataclass(frozen=True, slots=True)
class QMMMRegion:
    """A partitioned system: what is quantum, what is classical, what caps it."""

    symbols: tuple[str, ...]
    positions: Any
    qm_indices: tuple[int, ...]
    mm_indices: tuple[int, ...]
    #: MM point charges, one per entry of ``mm_indices``, after any shift.
    mm_charges: tuple[float, ...]
    link_atoms: tuple[LinkAtom, ...] = ()
    embedding: EmbeddingMode = EmbeddingMode.ELECTROSTATIC
    notes: tuple[str, ...] = ()

    @property
    def n_qm(self) -> int:
        """Quantum atoms including the capping hydrogens."""
        return len(self.qm_indices) + len(self.link_atoms)

    def qm_xyz_block(self) -> str:
        """The quantum region, capping hydrogens last, in PySCF's format."""
        lines = [
            f"{self.symbols[i]} {self.positions[i][0]:.10f} "
            f"{self.positions[i][1]:.10f} {self.positions[i][2]:.10f}"
            for i in self.qm_indices
        ]
        lines.extend(
            f"H {link.position[0]:.10f} {link.position[1]:.10f} {link.position[2]:.10f}"
            for link in self.link_atoms
        )
        return "\n".join(lines)

    def mm_positions(self) -> Any:
        return np.asarray([self.positions[i] for i in self.mm_indices], dtype=float)

    @property
    def mm_net_charge(self) -> float:
        return float(sum(self.mm_charges))

    def describe(self) -> str:
        lines = [
            f"{self.n_qm} quantum atom(s) ({len(self.link_atoms)} of them capping), "
            f"{len(self.mm_indices)} classical, {self.embedding.value} embedding",
            f"MM net charge {self.mm_net_charge:+.4f} e",
        ]
        lines.extend(f"  {link.describe()}" for link in self.link_atoms)
        lines.extend(f"  {note}" for note in self.notes)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class QMMMResult:
    """The energy of an embedded region, and what it does and does not include."""

    total_energy_hartree: float
    #: The same region computed with no MM charges present. The difference is
    #: the electrostatic embedding energy and nothing else - it is not an
    #: interaction energy, because no dispersion or repulsion is in it.
    vacuum_energy_hartree: float
    region: QMMMRegion
    method_signature: str = ""
    converged: bool = True
    limitations: tuple[str, ...] = ()

    @property
    def embedding_energy_kj_per_mol(self) -> float:
        return (
            self.total_energy_hartree - self.vacuum_energy_hartree
        ) * _KJ_PER_MOL_PER_HARTREE

    def describe(self) -> str:
        lines = [
            f"{self.method_signature}: E = {self.total_energy_hartree:.8f} Hartree"
            + ("" if self.converged else "  NOT CONVERGED"),
            f"electrostatic embedding contributes "
            f"{self.embedding_energy_kj_per_mol:+.2f} kJ/mol",
            self.region.describe(),
        ]
        lines.extend(f"  {limit}" for limit in self.limitations)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Building a partition
# ---------------------------------------------------------------------------


def partition(
    mol: Any,
    qm_atoms: Sequence[int],
    *,
    embedding: EmbeddingMode = EmbeddingMode.ELECTROSTATIC,
    charges: Sequence[float] | None = None,
    conformer: int = -1,
) -> QMMMRegion:
    """Split an RDKit molecule into a quantum region and a classical one.

    ``qm_atoms`` are atom indices. ``charges`` defaults to OPLS-AA charges
    assigned by foyer, which refuses molecules it has no parameters for rather
    than guessing - a QM/MM calculation on invented charges is not a QM/MM
    calculation.

    Raises :class:`UnsupportedPartition` for a boundary that cannot be capped
    correctly. Every one of those cases otherwise returns a number.
    """
    if embedding is EmbeddingMode.POLARIZABLE:
        raise UnsupportedPartition(
            "polarizable embedding needs MM polarizabilities, and the only force "
            "field wired up here is OPLS-AA, which is fixed-charge and has none; "
            "PySCF in this build also has no polarizable-embedding driver. Use "
            "electrostatic embedding, which polarises the QM region but not the MM one"
        )

    qm = sorted(set(int(i) for i in qm_atoms))
    n_atoms = mol.GetNumAtoms()
    if not qm:
        raise UnsupportedPartition("the quantum region is empty")
    if qm[0] < 0 or qm[-1] >= n_atoms:
        raise UnsupportedPartition(
            f"quantum atom indices must lie in [0, {n_atoms - 1}], got {qm}"
        )
    if len(qm) == n_atoms:
        raise UnsupportedPartition(
            "every atom is in the quantum region, so there is nothing to embed it in; "
            "run a plain quantum calculation instead"
        )

    positions = np.asarray(mol.GetConformer(conformer).GetPositions(), dtype=float)
    symbols = tuple(atom.GetSymbol() for atom in mol.GetAtoms())
    qm_set = set(qm)

    boundary = [
        bond
        for bond in mol.GetBonds()
        if (bond.GetBeginAtomIdx() in qm_set) != (bond.GetEndAtomIdx() in qm_set)
    ]
    for bond in boundary:
        _check_cuttable(bond, mol)

    if charges is None:
        charges = _opls_charges(mol)
    charges = [float(c) for c in charges]
    if len(charges) != n_atoms:
        raise UnsupportedPartition(
            f"expected {n_atoms} charges, one per atom, got {len(charges)}"
        )

    links, charges = _cap(boundary, qm_set, symbols, positions, charges, mol)

    mm = tuple(i for i in range(n_atoms) if i not in qm_set)
    notes: list[str] = []
    if embedding is EmbeddingMode.MECHANICAL:
        notes.append(
            "mechanical embedding: the MM charges are not in the QM Hamiltonian, so "
            "the quantum region is unpolarised and the electrostatics between regions "
            "are the force field's"
        )
    return QMMMRegion(
        symbols=symbols,
        positions=positions,
        qm_indices=tuple(qm),
        mm_indices=mm,
        mm_charges=tuple(charges[i] for i in mm),
        link_atoms=tuple(links),
        embedding=embedding,
        notes=tuple(notes),
    )


def _check_cuttable(bond: Any, mol: Any) -> None:
    """Refuse a boundary bond a hydrogen cannot honestly replace.

    Checked most-specific first, so the reason reported is the true one. An
    earlier version tested the bond order before aromaticity, and since RDKit
    types an aromatic bond as AROMATIC rather than SINGLE, every cut into a
    benzene ring was refused with "one hydrogen cannot cap it" - accurate about
    the bond order and silent about the delocalisation, which is the thing that
    actually makes the partition wrong.
    """
    from rdkit.Chem import BondType

    begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
    pair = f"{begin.GetSymbol()}-{end.GetSymbol()}"

    if "H" in (begin.GetSymbol(), end.GetSymbol()):
        raise UnsupportedPartition(
            f"the boundary cuts a {pair} bond, so the link atom would be a hydrogen "
            "capping the place a hydrogen already was; put that hydrogen in the same "
            "region as the atom it is bonded to"
        )
    if bond.GetIsAromatic() or (begin.GetIsAromatic() and end.GetIsAromatic()):
        raise UnsupportedPartition(
            f"the boundary cuts into an aromatic system at a {pair} bond; a capping "
            "hydrogen does not restore the delocalisation that was cut"
        )
    if bond.IsInRing():
        raise UnsupportedPartition(
            f"the boundary cuts a ring at a {pair} bond, which needs two link atoms "
            "and leaves the ring open in the quantum region"
        )
    if bond.GetBondType() is not BondType.SINGLE:
        raise UnsupportedPartition(
            f"the boundary cuts a {bond.GetBondType().name.lower()} {pair} bond; one "
            "hydrogen cannot cap it, and capping it anyway changes the bond order of "
            "the atom that keeps it"
        )
    if {begin.GetSymbol(), end.GetSymbol()} != {"C"}:
        raise UnsupportedPartition(
            f"the boundary cuts a polar {pair} bond; a link atom is only defensible "
            "across a nonpolar C-C single bond, because it has to imitate a charge "
            "distribution it does not reproduce"
        )


def _cap(
    boundary: Sequence[Any],
    qm_set: set[int],
    symbols: Sequence[str],
    positions: Any,
    charges: list[float],
    mol: Any,
) -> tuple[list[LinkAtom], list[float]]:
    """Place a capping hydrogen per cut bond and move the frontier charge off it.

    The frontier MM atom is the one the link atom now sits near. Leaving its
    point charge in place puts a full atomic charge a bond length from a
    hydrogen that is already representing that atom, which over-polarises the
    quantum region. The charge is therefore lifted and spread evenly over the
    remaining MM atoms, which keeps the classical region's total charge
    unchanged; that is the charge-shift scheme, and it is chosen over simply
    deleting the charge because deletion changes the net charge of the system.
    """
    charges = list(charges)
    links: list[LinkAtom] = []
    frontier: list[int] = []

    for bond in boundary:
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        qm_index, mm_index = (begin, end) if begin in qm_set else (end, begin)
        vector = positions[mm_index] - positions[qm_index]
        length = float(np.linalg.norm(vector))
        if length <= 1e-6:
            raise UnsupportedPartition(
                f"atoms {qm_index} and {mm_index} sit on top of each other, so the "
                "boundary bond has no direction to place a link atom along"
            )
        original = _BOND_LENGTH.get(
            tuple(sorted((symbols[qm_index], symbols[mm_index]))), length
        )
        scale = _BOND_LENGTH[("C", "H")] / original
        links.append(
            LinkAtom(
                qm_index=qm_index,
                mm_index=mm_index,
                position=tuple(positions[qm_index] + vector * scale),
                scale=scale,
                shifted_charge=charges[mm_index],
            )
        )
        frontier.append(mm_index)

    receivers = [
        i for i in range(len(symbols)) if i not in qm_set and i not in set(frontier)
    ]
    if frontier and not receivers:
        raise UnsupportedPartition(
            "every classical atom is on the boundary, so there is nowhere to shift "
            "the frontier charge to; make the classical region larger"
        )
    for index in frontier:
        share = charges[index] / len(receivers)
        charges[index] = 0.0
        for receiver in receivers:
            charges[receiver] += share
    return links, charges


def _opls_charges(mol: Any) -> tuple[float, ...]:
    """OPLS-AA partial charges, or the force field's own refusal."""
    from formulate.physics.md.opls import UnsupportedMolecule, extract_parameters

    try:
        return extract_parameters(mol).charges
    except UnsupportedMolecule as exc:
        raise UnsupportedPartition(
            f"no MM charges are available for this molecule: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------


@dataclass
class QMMMCalculator:
    """Runs an embedded quantum calculation, or says why it cannot."""

    basis: str = "6-31g"
    xc: str = "b3lyp"
    #: Hartree-Fock when empty, density functional theory otherwise.
    use_dft: bool = True
    max_cycle: int = 100
    limitations: tuple[str, ...] = field(default_factory=tuple)

    def is_available(self) -> bool:
        try:
            from pyscf import gto, qmmm, scf  # noqa: F401
        except Exception:
            return False
        return True

    def unavailable_reason(self) -> str:
        try:
            import pyscf  # noqa: F401
        except Exception as exc:
            return f"PySCF is not installed ({exc}); install formulate[physics]"
        try:
            from pyscf import qmmm  # noqa: F401
        except Exception as exc:
            return f"this PySCF build has no qmmm module ({exc})"
        return ""

    def capabilities(self) -> dict[str, Any]:
        """What this backend can and cannot do, for a caller to check first."""
        return {
            "available": self.is_available(),
            "unavailable_reason": self.unavailable_reason(),
            "embedding_modes": [
                EmbeddingMode.MECHANICAL.value,
                EmbeddingMode.ELECTROSTATIC.value,
            ],
            "unsupported_embedding_modes": {
                EmbeddingMode.POLARIZABLE.value: (
                    "needs MM polarizabilities; OPLS-AA is fixed-charge and this "
                    "PySCF build has no polarizable-embedding driver"
                )
            },
            "link_atoms": "hydrogen, across nonpolar C-C single bonds only",
            "boundary_charge_scheme": "charge shift off the frontier atom",
            "mm_charge_source": "OPLS-AA via foyer",
            "gradients": False,
        }

    def run(self, region: QMMMRegion) -> QMMMResult:
        from pyscf import gto, qmmm

        if not self.is_available():
            raise UnsupportedPartition(self.unavailable_reason())

        molecule = gto.M(atom=region.qm_xyz_block(), basis=self.basis, verbose=0)
        vacuum = self._method(molecule)
        vacuum.max_cycle = self.max_cycle
        vacuum_energy = float(vacuum.kernel())

        if region.embedding is EmbeddingMode.MECHANICAL:
            # Nothing is added to the Hamiltonian, so the embedded energy is
            # the vacuum one. Reporting it as though a QM/MM calculation had
            # happened would be the fabrication this mode exists to avoid.
            return QMMMResult(
                total_energy_hartree=vacuum_energy,
                vacuum_energy_hartree=vacuum_energy,
                region=region,
                method_signature=self._signature(),
                converged=bool(vacuum.converged),
                limitations=self.limitations
                + (
                    "mechanical embedding: the quantum region is unpolarised, and the "
                    "electrostatics between the regions are not in this number at all",
                ),
            )

        embedded = qmmm.add_mm_charges(
            self._method(molecule),
            region.mm_positions(),
            np.asarray(region.mm_charges, dtype=float),
        )
        embedded.max_cycle = self.max_cycle
        energy = float(embedded.kernel())

        limitations = self.limitations + (
            "electrostatic embedding is one-way: the MM charges polarise the QM "
            "density and the QM density does not act back on them",
            "no dispersion or exchange repulsion between the regions is included; "
            "those belong to the force field's non-bonded terms",
        )
        if region.link_atoms:
            limitations = limitations + (
                f"{len(region.link_atoms)} link atom(s): the capping hydrogens are not "
                "the atoms they replace, and the energy is not comparable with one "
                "computed on a different partition of the same molecule",
            )
        return QMMMResult(
            total_energy_hartree=energy,
            vacuum_energy_hartree=vacuum_energy,
            region=region,
            method_signature=self._signature(),
            converged=bool(embedded.converged),
            limitations=limitations,
        )

    def _method(self, molecule: Any) -> Any:
        from pyscf import dft, scf

        if self.use_dft:
            method = dft.RKS(molecule)
            method.xc = self.xc
            return method
        return scf.RHF(molecule)

    def _signature(self) -> str:
        level = f"{self.xc}/{self.basis}" if self.use_dft else f"HF/{self.basis}"
        return f"QM/MM {level}"
