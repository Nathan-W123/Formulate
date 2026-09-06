"""Structural mutation and crossover operators.

Specification section 3: "Evolutionary search: mutate/crossover valid
structures or recipes" and "For polymers, mutations may alter monomer identity,
comonomer fraction, topology, chain-length target, tacticity, crosslinker, or
cure/process variables."

Every operator returns *proposals*, never a judgement.  An operator may emit a
structure that is chemically absurd; rejecting it is the filter's job
(section 3, "Use validity filters before expert inference"), and duplicating
that logic here would give two places to disagree about what is valid.

What operators must guarantee is only that their output parses and sanitises.
An unsanitisable molecule is not a bad candidate, it is a broken one, and
letting it through would push a parsing error downstream into an expert.
"""

from __future__ import annotations

import abc
import random
from typing import Sequence

_ELEMENT_SWAPS: dict[str, tuple[str, ...]] = {
    "C": ("N", "O", "S"),
    "N": ("C", "O"),
    "O": ("N", "S", "C"),
    "S": ("O", "N", "C"),
    "F": ("Cl", "Br", "I"),
    "Cl": ("F", "Br", "I"),
    "Br": ("F", "Cl", "I"),
    "I": ("F", "Cl", "Br"),
}

#: Substituents appended at a site with spare valence. The first atom of each
#: is the attachment point.
_FRAGMENTS: tuple[str, ...] = (
    "C", "CC", "CCC", "C(C)C", "O", "OC", "N", "NC", "N(C)C",
    "F", "Cl", "Br", "C#N", "C(F)(F)F", "C(=O)O", "C(=O)C", "C(=O)N",
    "c1ccccc1", "C1CCCCC1", "S(=O)(=O)C", "[N+](=O)[O-]",
)

_ATOMIC_NUMBERS = {
    "C": 6, "N": 7, "O": 8, "S": 16, "F": 9, "Cl": 17, "Br": 35, "I": 53, "P": 15,
}


def _sanitized_smiles(mol) -> str | None:
    """Canonical SMILES if the edited molecule is chemically well formed.

    Importing the chem facade first is deliberate: it silences RDKit's
    valence logger, and these operators generate invalid intermediates by
    design, so an un-silenced RDKit would print a warning for every rejected
    proposal.
    """
    from rdkit import Chem

    from formulate import chem as _chem  # noqa: F401  (silences RDKit logging)

    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
    except Exception:
        return None
    if not smiles:
        return None
    # Round-trip: a SMILES that will not re-parse is useless downstream.
    return smiles if Chem.MolFromSmiles(smiles) is not None else None


class MutationOperator(abc.ABC):
    """Proposes structural variants of one molecule."""

    id: str = "operator"

    @abc.abstractmethod
    def propose(self, smiles: str, rng: random.Random, limit: int) -> list[str]:
        """Return up to ``limit`` variant SMILES, possibly empty."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.id}>"


class AtomSubstitution(MutationOperator):
    """Swap one heavy atom for a compatible element."""

    id = "atom_substitution"

    def propose(self, smiles: str, rng: random.Random, limit: int) -> list[str]:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        sites = [
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetSymbol() in _ELEMENT_SWAPS and not a.GetIsAromatic()
        ]
        rng.shuffle(sites)
        out: list[str] = []
        for idx in sites:
            symbol = mol.GetAtomWithIdx(idx).GetSymbol()
            for replacement in _shuffled(_ELEMENT_SWAPS[symbol], rng):
                edit = Chem.RWMol(mol)
                atom = edit.GetAtomWithIdx(idx)
                atom.SetAtomicNum(_ATOMIC_NUMBERS[replacement])
                atom.SetNoImplicit(False)
                atom.SetNumExplicitHs(0)
                candidate = _sanitized_smiles(edit)
                if candidate and candidate != smiles and candidate not in out:
                    out.append(candidate)
                if len(out) >= limit:
                    return out
        return out


class FragmentAppend(MutationOperator):
    """Attach a substituent at an atom with spare valence."""

    id = "fragment_append"

    def __init__(self, fragments: Sequence[str] = _FRAGMENTS) -> None:
        self.fragments = tuple(fragments)

    def propose(self, smiles: str, rng: random.Random, limit: int) -> list[str]:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        sites = [a.GetIdx() for a in mol.GetAtoms() if a.GetTotalNumHs() > 0]
        if not sites:
            return []
        rng.shuffle(sites)
        out: list[str] = []
        for idx in sites:
            for fragment_smiles in _shuffled(self.fragments, rng):
                fragment = Chem.MolFromSmiles(fragment_smiles)
                if fragment is None:
                    continue
                combined = Chem.RWMol(Chem.CombineMols(mol, fragment))
                try:
                    combined.AddBond(idx, mol.GetNumAtoms(), Chem.BondType.SINGLE)
                except Exception:
                    continue
                candidate = _sanitized_smiles(combined)
                if candidate and candidate != smiles and candidate not in out:
                    out.append(candidate)
                if len(out) >= limit:
                    return out
        return out


class AtomDeletion(MutationOperator):
    """Remove a terminal heavy atom, shrinking the structure."""

    id = "atom_deletion"

    def propose(self, smiles: str, rng: random.Random, limit: int) -> list[str]:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() <= 2:
            return []
        sites = [a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() == 1 and not a.IsInRing()]
        rng.shuffle(sites)
        out: list[str] = []
        for idx in sites:
            edit = Chem.RWMol(mol)
            edit.RemoveAtom(idx)
            candidate = _sanitized_smiles(edit)
            if candidate and candidate != smiles and candidate not in out:
                out.append(candidate)
            if len(out) >= limit:
                return out
        return out


class BondOrderChange(MutationOperator):
    """Promote or demote an acyclic bond between single and double."""

    id = "bond_order_change"

    def propose(self, smiles: str, rng: random.Random, limit: int) -> list[str]:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        bonds = [
            b.GetIdx()
            for b in mol.GetBonds()
            if not b.IsInRing()
            and not b.GetIsAromatic()
            and b.GetBondType() in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE)
        ]
        rng.shuffle(bonds)
        out: list[str] = []
        for idx in bonds:
            edit = Chem.RWMol(mol)
            bond = edit.GetBondWithIdx(idx)
            flipped = (
                Chem.BondType.DOUBLE
                if bond.GetBondType() == Chem.BondType.SINGLE
                else Chem.BondType.SINGLE
            )
            bond.SetBondType(flipped)
            for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
                atom.SetNoImplicit(False)
                atom.SetNumExplicitHs(0)
            candidate = _sanitized_smiles(edit)
            if candidate and candidate != smiles and candidate not in out:
                out.append(candidate)
            if len(out) >= limit:
                return out
        return out


class RingFusion(MutationOperator):
    """Close a ring between two atoms three or four bonds apart."""

    id = "ring_closure"

    def propose(self, smiles: str, rng: random.Random, limit: int) -> list[str]:
        from rdkit import Chem
        from rdkit.Chem import rdmolops

        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() < 5:
            return []
        distances = rdmolops.GetDistanceMatrix(mol)
        # A new bond between atoms d bonds apart closes a ring of size d+1.
        # Restricting d to 4-6 gives five- to seven-membered rings, the sizes
        # that are actually common and unstrained. Allowing d=3 produced
        # four-membered rings, and bridging two atoms already inside a ring
        # produced fused bicyclics such as c1cc2ccc1-2 - structures that
        # sanitise cleanly but could not be made.
        pairs = [
            (i, j)
            for i in range(mol.GetNumAtoms())
            for j in range(i + 1, mol.GetNumAtoms())
            if 4 <= distances[i][j] <= 6
            and mol.GetAtomWithIdx(i).GetTotalNumHs() > 0
            and mol.GetAtomWithIdx(j).GetTotalNumHs() > 0
            and not (mol.GetAtomWithIdx(i).IsInRing() and mol.GetAtomWithIdx(j).IsInRing())
        ]
        rng.shuffle(pairs)
        out: list[str] = []
        for i, j in pairs:
            edit = Chem.RWMol(mol)
            try:
                edit.AddBond(i, j, Chem.BondType.SINGLE)
            except Exception:
                continue
            candidate = _sanitized_smiles(edit)
            if candidate and candidate != smiles and candidate not in out:
                out.append(candidate)
            if len(out) >= limit:
                return out
        return out


DEFAULT_OPERATORS: tuple[MutationOperator, ...] = (
    AtomSubstitution(),
    FragmentAppend(),
    AtomDeletion(),
    BondOrderChange(),
    RingFusion(),
)


def brics_crossover(
    parent_a: str, parent_b: str, rng: random.Random, limit: int = 4
) -> list[str]:
    """Recombine two molecules by exchanging BRICS fragments.

    BRICS cuts at bonds that correspond to real retrosynthetic disconnections,
    so the recombined products tend to be chemistry a synthetic chemist would
    recognise rather than arbitrary graph splices. Products are still only
    proposals and go through the same validity filters as anything else.
    """
    from rdkit import Chem
    from rdkit.Chem import BRICS

    molecules = [Chem.MolFromSmiles(s) for s in (parent_a, parent_b)]
    if any(m is None for m in molecules):
        return []

    fragments: set[str] = set()
    for mol in molecules:
        try:
            fragments |= set(BRICS.BRICSDecompose(mol))
        except Exception:
            continue
    if len(fragments) < 2:
        return []

    fragment_mols = [Chem.MolFromSmiles(f) for f in sorted(fragments)]
    fragment_mols = [m for m in fragment_mols if m is not None]
    if len(fragment_mols) < 2:
        return []

    # Shuffle here with the seeded generator rather than letting BRICSBuild
    # scramble internally: its scrambleReagents draws from the global random
    # module, which would make a run unreproducible, and reproducibility is a
    # section 11 invariant.
    rng.shuffle(fragment_mols)

    parents = {parent_a, parent_b}
    out: list[str] = []
    # BRICSBuild enumerates a combinatorial space; take a bounded slice rather
    # than exhausting it. Only per-product failures are caught - a bad call
    # signature is a programming error and must surface, not be swallowed into
    # an empty result.
    builder = BRICS.BRICSBuild(fragment_mols, scrambleReagents=False)
    for index, product in enumerate(builder):
        if index >= limit * 8 or len(out) >= limit:
            break
        try:
            candidate = _sanitized_smiles(Chem.RWMol(product))
        except Exception:
            continue
        if candidate and candidate not in parents and candidate not in out:
            out.append(candidate)
    return out


def _shuffled(items: Sequence[str], rng: random.Random) -> list[str]:
    out = list(items)
    rng.shuffle(out)
    return out
