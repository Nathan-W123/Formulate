"""Thin facade over the chemistry toolkit.

RDKit is an optional dependency.  Isolating every import behind this module
keeps :mod:`formulate.core`, :mod:`formulate.targets` and
:mod:`formulate.ranking` importable and testable without it, and gives one
place to swap toolkits later.

Nothing here makes scientific decisions; it parses, canonicalises and computes
structural descriptors.  Property prediction lives in :mod:`formulate.experts`.
"""

from __future__ import annotations

import functools
from typing import Any, Iterable

from formulate.core.errors import BackendUnavailableError

_RDKIT_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - exercised by whichever branch the environment takes
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors
    from rdkit.DataStructs import BulkTanimotoSimilarity, TanimotoSimilarity

    RDLogger.DisableLog("rdApp.*")  # RDKit logs parse failures we handle ourselves
    _HAVE_RDKIT = True
except Exception as exc:  # pragma: no cover
    _RDKIT_IMPORT_ERROR = exc
    _HAVE_RDKIT = False


def rdkit_available() -> bool:
    """True when RDKit could be imported."""
    return _HAVE_RDKIT


def require_rdkit() -> None:
    """Raise a clear error when a chemistry operation needs RDKit."""
    if not _HAVE_RDKIT:
        raise BackendUnavailableError(
            "RDKit is required for this operation but is not installed. "
            "Install the chemistry extra: pip install 'formulate[chem]'. "
            f"Original import error: {_RDKIT_IMPORT_ERROR}"
        )


def rdkit_version() -> str:
    if not _HAVE_RDKIT:
        return ""
    import rdkit

    return rdkit.__version__


@functools.lru_cache(maxsize=4096)
def mol_from_smiles(smiles: str, *, sanitize: bool = True) -> Any | None:
    """Parse SMILES into a molecule, or ``None`` if invalid."""
    require_rdkit()
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    except Exception:
        return None
    return mol


@functools.lru_cache(maxsize=4096)
def canonical_smiles(smiles: str) -> str | None:
    """Return the canonical SMILES, or ``None`` when the input is invalid.

    Used for content addressing, so two spellings of the same structure share
    a cache entry.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


@functools.lru_cache(maxsize=4096)
def inchi_key(smiles: str) -> str | None:
    """Standard InChIKey, or ``None`` when unavailable."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def is_valid_smiles(smiles: str) -> bool:
    return rdkit_available() and mol_from_smiles(smiles) is not None


@functools.lru_cache(maxsize=4096)
def formula(smiles: str) -> str | None:
    mol = mol_from_smiles(smiles)
    return None if mol is None else rdMolDescriptors.CalcMolFormula(mol)


@functools.lru_cache(maxsize=4096)
def descriptors(smiles: str) -> dict[str, float]:
    """Structural descriptors used by filters, ESOL and diversity selection."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return {}
    heavy = mol.GetNumHeavyAtoms()
    aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    return {
        "molar_mass": float(Descriptors.MolWt(mol)),
        "heavy_atom_count": float(heavy),
        "rotatable_bond_count": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "topological_polar_surface_area": float(rdMolDescriptors.CalcTPSA(mol)),
        "aromatic_atom_fraction": float(aromatic) / heavy if heavy else 0.0,
        "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
        "hbd": float(rdMolDescriptors.CalcNumHBD(mol)),
        "hba": float(rdMolDescriptors.CalcNumHBA(mol)),
        "formal_charge": float(Chem.GetFormalCharge(mol)),
        "crippen_logp": float(Crippen.MolLogP(mol)),
        "crippen_mr": float(Crippen.MolMR(mol)),
    }


def elements(smiles: str) -> set[str]:
    """Set of element symbols present (heavy atoms and explicit hydrogens)."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return set()
    return {a.GetSymbol() for a in mol.GetAtoms()}


@functools.lru_cache(maxsize=4096)
def _fingerprint(smiles: str, radius: int, n_bits: int) -> Any | None:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return gen.GetFingerprint(mol)


def fingerprint(smiles: str, *, radius: int = 2, n_bits: int = 2048) -> Any | None:
    """Morgan fingerprint used for structural similarity."""
    require_rdkit()
    return _fingerprint(smiles, radius, n_bits)


def tanimoto(smiles_a: str, smiles_b: str, *, radius: int = 2, n_bits: int = 2048) -> float | None:
    """Tanimoto similarity in [0, 1], or ``None`` if either input is invalid."""
    fa = fingerprint(smiles_a, radius=radius, n_bits=n_bits)
    fb = fingerprint(smiles_b, radius=radius, n_bits=n_bits)
    if fa is None or fb is None:
        return None
    return float(TanimotoSimilarity(fa, fb))


def bulk_tanimoto(query: str, others: Iterable[str], *, radius: int = 2, n_bits: int = 2048):
    """Similarity of ``query`` against many SMILES; ``None`` for invalid entries."""
    fq = fingerprint(query, radius=radius, n_bits=n_bits)
    others = list(others)
    if fq is None:
        return [None] * len(others)
    fps = [fingerprint(s, radius=radius, n_bits=n_bits) for s in others]
    valid = [(i, f) for i, f in enumerate(fps) if f is not None]
    out: list[float | None] = [None] * len(others)
    if valid:
        sims = BulkTanimotoSimilarity(fq, [f for _, f in valid])
        for (i, _), s in zip(valid, sims):
            out[i] = float(s)
    return out


def has_substructure(smiles: str, smarts: str) -> bool:
    """True when ``smiles`` contains the ``smarts`` pattern."""
    mol = mol_from_smiles(smiles)
    patt = _smarts(smarts)
    if mol is None or patt is None:
        return False
    return mol.HasSubstructMatch(patt)


@functools.lru_cache(maxsize=1024)
def _smarts(smarts: str) -> Any | None:
    require_rdkit()
    try:
        return Chem.MolFromSmarts(smarts)
    except Exception:
        return None
