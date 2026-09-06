"""Three-dimensional structures for physics backends.

A SMILES string is a graph; every quantum and dynamics method needs
coordinates.  Turning one into the other is a modelling decision, not a
formatting step, and section 7 makes the danger explicit: results depend on
which conformer you happened to embed.

So a geometry here always records how it was produced and how many
alternatives were considered.  A property computed on one arbitrary conformer
of a flexible molecule is not a property of the molecule, and a result that
cannot say which conformer it used cannot be judged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from formulate.core.errors import BackendUnavailableError

#: CODATA 2018 conversion factors. These appear in every backend boundary and
#: getting one wrong produces a plausible number that is wrong by a constant,
#: which is the hardest kind of error to notice, so they live in one place.
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM
HARTREE_TO_J_PER_MOL = 2625499.639479828  # Hartree -> J/mol
EV_TO_J_PER_MOL = 96485.33212331001


@dataclass(frozen=True, slots=True)
class MolecularGeometry:
    """Atomic positions in Angstrom, with the electronic state they belong to."""

    symbols: tuple[str, ...]
    #: Shape (n_atoms, 3), in Angstrom.
    positions: np.ndarray
    charge: int = 0
    #: 2S+1. A closed-shell singlet is 1.
    spin_multiplicity: int = 1
    #: How this geometry was produced, e.g. "ETKDG + MMFF94, lowest of 10".
    source: str = ""
    #: Conformers generated before this one was chosen.
    conformers_considered: int = 1
    #: Spread of the conformer ensemble, kcal/mol, when more than one was made.
    conformer_energy_spread: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.positions.shape != (len(self.symbols), 3):
            raise ValueError(
                f"positions must be ({len(self.symbols)}, 3), got {self.positions.shape}"
            )
        if self.spin_multiplicity < 1:
            raise ValueError("spin multiplicity must be at least 1")

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def n_electrons(self) -> int:
        return sum(_ATOMIC_NUMBER[s] for s in self.symbols) - self.charge

    @property
    def formula(self) -> str:
        counts: dict[str, int] = {}
        for symbol in self.symbols:
            counts[symbol] = counts.get(symbol, 0) + 1
        return "".join(f"{s}{counts[s] if counts[s] > 1 else ''}" for s in sorted(counts))

    def xyz_block(self) -> str:
        """Coordinates in the ``symbol x y z`` form PySCF accepts."""
        return "\n".join(
            f"{symbol} {x:.10f} {y:.10f} {z:.10f}"
            for symbol, (x, y, z) in zip(self.symbols, self.positions)
        )

    def to_ase(self) -> Any:
        """An ``ase.Atoms`` with the same positions, in Angstrom."""
        try:
            from ase import Atoms
        except Exception as exc:  # pragma: no cover
            raise BackendUnavailableError(f"ASE is required for this conversion ({exc})") from exc
        return Atoms(symbols=list(self.symbols), positions=np.asarray(self.positions, dtype=float))

    def with_positions(self, positions: np.ndarray, *, source: str = "") -> "MolecularGeometry":
        return MolecularGeometry(
            symbols=self.symbols,
            positions=np.asarray(positions, dtype=float),
            charge=self.charge,
            spin_multiplicity=self.spin_multiplicity,
            source=source or self.source,
            conformers_considered=self.conformers_considered,
            conformer_energy_spread=self.conformer_energy_spread,
            notes=self.notes,
        )

    def identity_payload(self) -> dict[str, Any]:
        """Content-addressing payload.

        Coordinates are rounded to 1e-6 Angstrom: a geometry that differs only
        in the last bits of a float is the same geometry, and treating it as a
        different one would defeat the cache.
        """
        return {
            "symbols": list(self.symbols),
            "positions": [[round(float(v), 6) for v in row] for row in self.positions],
            "charge": self.charge,
            "spin_multiplicity": self.spin_multiplicity,
        }

    def describe(self) -> str:
        bits = [f"{self.formula} ({self.n_atoms} atoms)"]
        if self.charge:
            bits.append(f"charge {self.charge:+d}")
        if self.spin_multiplicity != 1:
            bits.append(f"multiplicity {self.spin_multiplicity}")
        if self.source:
            bits.append(self.source)
        return ", ".join(bits)


_ATOMIC_NUMBER = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Br": 35, "I": 53,
}


def geometry_from_smiles(
    smiles: str,
    *,
    n_conformers: int = 10,
    seed: int = 0xF00D,
    optimize: bool = True,
) -> MolecularGeometry:
    """Embed a SMILES string in three dimensions.

    Several conformers are generated and the lowest-energy one kept, and the
    spread of the ensemble is recorded on the result. That spread is the point:
    if it is large, any property computed from the single chosen conformer is
    a property of that conformer and not of the molecule, and a caller can see
    that from the geometry rather than having to guess.
    """
    from formulate import chem

    if not chem.rdkit_available():
        raise BackendUnavailableError("RDKit is required to embed a SMILES string")

    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = chem.mol_from_smiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES {smiles!r}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    conformer_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=max(1, n_conformers), params=params))
    if not conformer_ids:
        # Fall back to a single unconstrained embedding: better a poor geometry
        # that is declared than a silent failure.
        if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
            raise ValueError(f"Could not generate any 3D geometry for {smiles!r}")
        conformer_ids = [0]

    energies: list[tuple[float, int]] = []
    spread: float | None = None
    notes: list[str] = []

    if optimize:
        try:
            results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
            energies = [
                (energy, conformer_ids[i])
                for i, (converged, energy) in enumerate(results)
                if i < len(conformer_ids)
            ]
            unconverged = sum(1 for converged, _ in results if converged != 0)
            if unconverged:
                notes.append(
                    f"{unconverged} of {len(results)} conformer optimisations did not converge"
                )
        except Exception as exc:
            notes.append(f"MMFF optimisation unavailable ({exc}); using embedded geometry")

    if energies:
        energies.sort()
        best_id = energies[0][1]
        if len(energies) > 1:
            spread = energies[-1][0] - energies[0][0]
    else:
        best_id = conformer_ids[0]

    conformer = mol.GetConformer(best_id)
    positions = np.array(
        [list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float
    )
    symbols = tuple(atom.GetSymbol() for atom in mol.GetAtoms())

    if spread is not None and spread > 5.0:
        notes.append(
            f"conformers span {spread:.1f} kcal/mol, so any property computed from the "
            "single lowest-energy structure describes that conformer rather than the "
            "molecule; a Boltzmann-weighted ensemble would be needed for a property average"
        )

    source = (
        f"ETKDGv3{' + MMFF94' if optimize else ''}, lowest of {len(conformer_ids)}"
        if len(conformer_ids) > 1
        else f"ETKDGv3{' + MMFF94' if optimize else ''}, single conformer"
    )
    if len(conformer_ids) == 1:
        notes.append(
            "only one conformer was generated, so conformational spread is unknown"
        )

    return MolecularGeometry(
        symbols=symbols,
        positions=positions,
        charge=Chem.GetFormalCharge(mol),
        spin_multiplicity=1,
        source=source,
        conformers_considered=len(conformer_ids),
        conformer_energy_spread=spread,
        notes=tuple(notes),
    )


def geometry_from_ase(atoms: Any, template: MolecularGeometry | None = None) -> MolecularGeometry:
    """Convert an ``ase.Atoms`` back into a geometry, keeping the template's state."""
    positions = np.asarray(atoms.get_positions(), dtype=float)
    symbols = tuple(atoms.get_chemical_symbols())
    if template is not None:
        return template.with_positions(positions)
    return MolecularGeometry(symbols=symbols, positions=positions)


def rmsd(a: MolecularGeometry, b: MolecularGeometry) -> float:
    """Root-mean-square deviation in Angstrom, without realignment."""
    if a.symbols != b.symbols:
        raise ValueError("Cannot compare geometries with different atoms.")
    diff = np.asarray(a.positions) - np.asarray(b.positions)
    return float(np.sqrt((diff**2).sum(axis=1).mean()))
