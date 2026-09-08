"""Is the expensive rung actually the more accurate one?

Section 8 describes a multi-fidelity ladder and assumes it is. For atomization
energy at the basis this pipeline runs, that is testable, so it is tested.

Both columns are electronic well depths: molecule minus ground-state free
atoms, with no zero-point correction on either side. Reference values are
D_0 + ZPE, both commonly tabulated handbook numbers rather than a curated
benchmark - the same standing this repository gives its other reference
compounds. Comparing against D_0 without adding the zero-point energy back
produces an apparent model error of about 100 kJ/mol that is entirely the
missing vibrational term.

Run with ``python bench/atomization_ladder.py``. Needs formulate[potentials]
and PySCF. Results are recorded in docs/BENCHMARKS.md.
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from formulate.physics.geometry import geometry_from_smiles  # noqa: E402
from formulate.physics.potentials import MacePotential  # noqa: E402
from formulate.physics.qm import backend_for  # noqa: E402
from formulate.physics.qm.base import QMMethod, QMRequest  # noqa: E402
from formulate.physics.qm.thermochemistry import atomization_energy  # noqa: E402

ATOMIC_NUMBER = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
J_PER_MOL_PER_EV = 96485.33212331001

#: name, SMILES, D_0 (kJ/mol), zero-point energy (kJ/mol).
REFERENCES = [
    ("methane", "C", 1642.2, 116.8),
    ("water", "O", 917.8, 55.4),
    ("ammonia", "N", 1158.0, 88.7),
    ("methanol", "CO", 2037.1, 135.3),
    ("formaldehyde", "C=O", 1494.1, 69.3),
    ("ethane", "CC", 2823.9, 194.6),
    ("ethene", "C=C", 2225.9, 132.2),
]


def main() -> None:
    potential = MacePotential()
    backend = backend_for(QMMethod.DFT)
    if backend is None:
        raise SystemExit("no DFT backend is installed")
    atom_references = potential.atomic_reference_energies()

    fast_errors, slow_errors = [], []
    print(f"{'molecule':14s} {'De(exp)':>9s} {'MACE':>9s} {'err':>7s} {'DFT':>9s} {'err':>7s}")
    for name, smiles, d0, zpe in REFERENCES:
        well_depth = d0 + zpe
        geometry = geometry_from_smiles(smiles, n_conformers=3)
        numbers = [ATOMIC_NUMBER[s] for s in geometry.symbols]

        energy = potential.compute(numbers, geometry.positions, forces=False).energy_ev
        fast = (sum(atom_references[n] for n in numbers) - energy) * J_PER_MOL_PER_EV / 1000.0

        result = atomization_energy(
            backend,
            geometry,
            QMRequest(
                geometry=geometry,
                method=QMMethod.DFT,
                basis="6-31g",
                xc="b3lyp",
                optimize_geometry=False,
            ),
        )
        if result is None:
            print(f"{name:14s} the quantum calculation did not converge")
            continue
        slow = result.value / 1000.0

        fast_errors.append(fast - well_depth)
        slow_errors.append(slow - well_depth)
        print(
            f"{name:14s} {well_depth:9.1f} {fast:9.1f} {fast - well_depth:+7.1f} "
            f"{slow:9.1f} {slow - well_depth:+7.1f}",
            flush=True,
        )

    fast_errors, slow_errors = np.array(fast_errors), np.array(slow_errors)
    print(
        f"\n{potential.info.identifier:18s} MAE {np.abs(fast_errors).mean():6.1f}  "
        f"bias {fast_errors.mean():+6.1f} kJ/mol"
    )
    print(
        f"{'B3LYP/6-31G':18s} MAE {np.abs(slow_errors).mean():6.1f}  "
        f"bias {slow_errors.mean():+6.1f} kJ/mol"
    )
    if np.abs(fast_errors).mean() < np.abs(slow_errors).mean():
        print(
            "\nThe cheaper rung is the more accurate one for this property at this basis, "
            "so escalating to the solver would spend wall clock to get further from the "
            "truth. See _POTENTIAL_IS_THE_BETTER_RUNG in coordination/validation.py."
        )
    else:
        print("\nThe expensive rung is the more accurate one, as section 8 assumes.")


if __name__ == "__main__":
    main()
