"""The fast rung against the quantum rung, on conformer energy differences.

Absolute energies from a neural potential and from a quantum calculation are
not on the same scale and never will be: MACE learns a shifted, size-extensive
surface. What a ranking consumes is differences, so differences are what is
compared.

Molecules with real rotatable bonds, and conformers pruned by RMSD. A first
pass used ethanol and methanol and reported a mean error of 0.1 meV, which was
not agreement: RDKit had returned the same geometry three times and both
methods correctly said the difference was zero.

Run with ``python bench/conformer_energies.py``. Needs formulate[potentials]
and PySCF. Results are recorded in docs/BENCHMARKS.md.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from formulate.physics.geometry import MolecularGeometry  # noqa: E402
from formulate.physics.potentials import MacePotential  # noqa: E402
from formulate.physics.qm import backend_for  # noqa: E402
from formulate.physics.qm.base import QMMethod, QMRequest  # noqa: E402

EV_PER_J_MOL = 96485.33212331001
SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"}
MOLECULES = [
    ("butane", "CCCC"),
    ("pentane", "CCCCC"),
    ("1-propanol", "CCCO"),
    ("ethylene glycol", "OCCO"),
    ("methyl acetate", "COC(C)=O"),
    ("propanal", "CCC=O"),
    ("2-methoxyethanol", "COCCO"),
    ("butanenitrile", "CCCC#N"),
]
#: Embed generously, keep a few, and require them to be genuinely different.
N_EMBED, N_KEEP, RMSD_MIN = 20, 4, 0.5


def distinct_conformers(smiles: str):
    """Embed, optimise, and prune by heavy-atom RMSD to distinct geometries."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, rdMolAlign

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMultipleConfs(mol, numConfs=N_EMBED, randomSeed=0xC0FFEE)
    AllChem.MMFFOptimizeMoleculeConfs(mol)

    kept: list[int] = []
    for index in range(mol.GetNumConformers()):
        probe = Chem.Mol(mol)
        if all(
            rdMolAlign.GetBestRMS(probe, Chem.Mol(mol), prbId=index, refId=other) > RMSD_MIN
            for other in kept
        ):
            kept.append(index)
        if len(kept) == N_KEEP:
            break
    numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return numbers, [np.array(mol.GetConformer(i).GetPositions()) for i in kept]


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Agreement between two orderings, +1 identical and -1 reversed."""
    n = len(a)
    concordant = sum(
        np.sign(a[i] - a[j]) == np.sign(b[i] - b[j])
        for i in range(n)
        for j in range(i + 1, n)
    )
    return (2.0 * concordant / (n * (n - 1) // 2)) - 1.0


def main() -> None:
    potential = MacePotential()
    backend = backend_for(QMMethod.DFT)
    if backend is None:
        raise SystemExit("no DFT backend is installed")
    print("QM backend:", type(backend).__name__)

    energy_errors, force_errors, taus, spreads = [], [], [], []
    potential_seconds = quantum_seconds = 0.0
    points = 0

    for name, smiles in MOLECULES:
        numbers, conformers = distinct_conformers(smiles)
        if len(conformers) < 3:
            print(f"{name:18s} only {len(conformers)} distinct conformer(s), skipped")
            continue
        symbols = tuple(SYMBOL[n] for n in numbers)
        fast_e, slow_e, fast_f, slow_f = [], [], [], []
        usable = True

        for positions in conformers:
            started = time.perf_counter()
            result = potential.compute(numbers, positions, forces=True)
            potential_seconds += time.perf_counter() - started
            fast_e.append(result.energy_ev)
            fast_f.append(np.asarray(result.forces_ev_per_angstrom))

            geometry = MolecularGeometry(
                symbols=symbols, positions=np.asarray(positions, dtype=float)
            )
            started = time.perf_counter()
            quantum = backend.run(
                QMRequest(
                    geometry=geometry,
                    method=QMMethod.DFT,
                    basis="6-31g",
                    xc="b3lyp",
                    optimize_geometry=False,
                    compute_forces=True,
                )
            )
            quantum_seconds += time.perf_counter() - started
            points += 1
            if not quantum.usable:
                usable = False
                break
            slow_e.append(quantum.total_energy.to("J/mol").value / EV_PER_J_MOL)
            # Already eV/Angstrom: the backend's ASE calculator converts, and
            # converting again here inflated the error by a factor of 51.
            slow_f.append(
                None if quantum.forces is None else np.asarray(quantum.forces, dtype=float)
            )

        if not usable:
            print(f"{name:18s} QM unusable")
            continue

        fast_rel = np.array(fast_e) - fast_e[0]
        slow_rel = np.array(slow_e) - slow_e[0]
        error_mev = float(np.abs(fast_rel - slow_rel).mean() * 1000.0)
        energy_errors.append(error_mev)
        spreads.append(float(np.ptp(slow_rel) * 1000.0))
        taus.append(kendall_tau(fast_rel, slow_rel))

        per_conformer = [
            float(np.abs(a - b).mean()) for a, b in zip(fast_f, slow_f) if b is not None
        ]
        if per_conformer:
            force_errors.append(float(np.mean(per_conformer)))

        print(
            f"{name:18s} {len(fast_rel)} confs  dE err {error_mev:6.1f} meV  "
            f"(spread {spreads[-1]:6.1f} meV)  tau {taus[-1]:+.2f}"
            + (f"  |dF| {force_errors[-1]:.3f} eV/A" if per_conformer else ""),
            flush=True,
        )

    if not energy_errors:
        return
    mean_error = float(np.mean(energy_errors))
    print(
        f"\nmean |conformer dE error|   {mean_error:8.1f} meV "
        f"({mean_error * EV_PER_J_MOL / 1e6:.2f} kJ/mol)"
    )
    print(f"mean DFT conformer spread   {np.mean(spreads):8.1f} meV  (the signal being resolved)")
    if force_errors:
        print(f"mean |force error|          {np.mean(force_errors):8.3f} eV/Angstrom")
    print(f"mean ranking agreement      {np.mean(taus):+8.2f} (Kendall tau, 1.0 = same order)")
    # A molecule whose whole conformer spread is below the measured error has
    # no ordering to agree with; reporting its tau alongside the others reads
    # as a disagreement about chemistry rather than the ordering of noise.
    resolved = [t for t, s in zip(taus, spreads) if s > mean_error]
    if resolved:
        print(
            f"  restricted to the {len(resolved)} molecule(s) whose spread exceeds that "
            f"error: {np.mean(resolved):+.2f}"
        )
    print(
        f"MACE wall {potential_seconds:6.2f}s over {points} points | "
        f"QM wall {quantum_seconds:7.2f}s | "
        f"speedup {quantum_seconds / max(potential_seconds, 1e-9):.0f}x"
    )


if __name__ == "__main__":
    main()
