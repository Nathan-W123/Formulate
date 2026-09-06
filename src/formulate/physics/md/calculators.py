"""Force providers for molecular dynamics.

Specification section 6: "Use mature classical MD backends for production
validation."  ASE supplies the integrators; this module supplies the forces,
behind one interface so a protocol never needs to know which engine is under
it.

The periodic question is settled here rather than left to the caller, because
it decides what a run can legitimately claim.  A finite cluster has a surface,
and a bulk property measured on one carries a finite-size error that does not
shrink with longer sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from formulate.core.errors import BackendUnavailableError

from ..backends import get_backend_capability
from ..quiet import suppress_native_output


@dataclass(frozen=True, slots=True)
class CalculatorChoice:
    """A force provider, with what it can be trusted to do."""

    backend_id: str
    label: str
    supports_periodic: bool
    reference_ms_per_evaluation: float
    electronic: bool
    notes: tuple[str, ...] = ()


#: Ordered cheapest first. Cost matters more than accuracy for the sampling
#: lengths reachable here: a more accurate potential sampled for a tenth as
#: long is usually the worse answer.
CALCULATORS: tuple[CalculatorChoice, ...] = (
    CalculatorChoice(
        "rdkit:mmff94", "MMFF94", False, 1.0, False,
        ("molecular mechanics with fixed topology: no electrons, no bond breaking",),
    ),
    CalculatorChoice(
        "xtb:GFN-FF", "GFN-FF", False, 6.0, False,
        ("a general force field covering arbitrary organic structures",),
    ),
    CalculatorChoice(
        "xtb:GFN1-xTB", "GFN1-xTB", True, 500.0, True,
        ("the only periodic-capable option available here, and roughly a hundred times "
         "more expensive per step than GFN-FF",),
    ),
    CalculatorChoice(
        "xtb:GFN2-xTB", "GFN2-xTB", False, 200.0, True,
        ("more accurate than GFN-FF for conformer energetics, and far slower",),
    ),
)


def available_calculators(periodic: bool = False) -> list[CalculatorChoice]:
    """Working calculators, optionally restricted to periodic-capable ones."""
    out = []
    for choice in CALCULATORS:
        capability = get_backend_capability(choice.backend_id)
        if not capability.available:
            continue
        if periodic and not capability.supports_periodic:
            continue
        out.append(choice)
    return out


def choose_calculator(
    periodic: bool = False, prefer: str | None = None
) -> CalculatorChoice | None:
    """Pick a force provider, cheapest first unless one is named."""
    options = available_calculators(periodic=periodic)
    if not options:
        return None
    if prefer:
        for choice in options:
            if choice.backend_id == prefer or choice.label == prefer:
                return choice
    return min(options, key=lambda c: c.reference_ms_per_evaluation)


def attach_calculator(atoms: Any, choice: CalculatorChoice, rdkit_mol: Any = None) -> None:
    """Attach the chosen force provider to an ``ase.Atoms``."""
    if choice.backend_id == "rdkit:mmff94":
        if rdkit_mol is None:
            raise BackendUnavailableError(
                "MMFF94 needs the RDKit molecule it was parameterised from; "
                "atoms alone do not carry bond orders"
            )
        atoms.calc = MMFFCalculator(rdkit_mol)
        return

    if choice.backend_id.startswith("xtb:"):
        from xtb.ase.calculator import XTB

        method = choice.backend_id.split(":", 1)[1]
        with suppress_native_output():
            atoms.calc = XTB(method=method)
        return

    raise BackendUnavailableError(f"No adapter for calculator {choice.backend_id!r}")


class MMFFCalculator:
    """An ASE calculator backed by RDKit's MMFF94 implementation.

    ASE ships no MMFF interface, so this bridges them. It is the cheapest way
    to move a molecule on a physically reasonable surface, at roughly a
    millisecond per evaluation.

    Two conversions matter and are done once, here: RDKit reports energies in
    kcal/mol where ASE expects electronvolts, and returns a gradient where ASE
    expects a force.
    """

    implemented_properties = ["energy", "forces"]
    #: kcal/mol -> eV
    KCAL_PER_MOL_TO_EV = 0.04336410390059322

    def __init__(self, rdkit_mol: Any) -> None:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        self.mol = Chem.Mol(rdkit_mol)
        if self.mol.GetNumConformers() == 0:
            raise ValueError("MMFFCalculator needs a molecule with a conformer")
        self.properties = AllChem.MMFFGetMoleculeProperties(self.mol)
        if self.properties is None:
            raise BackendUnavailableError(
                "MMFF94 has no parameters for this molecule; choose another calculator"
            )
        self.results: dict[str, Any] = {}
        self._positions: np.ndarray | None = None
        self.evaluations = 0

    def get_potential_energy(self, atoms=None, force_consistent: bool = False) -> float:
        self._ensure(atoms)
        return self.results["energy"]

    def get_forces(self, atoms=None) -> np.ndarray:
        self._ensure(atoms)
        return self.results["forces"]

    def get_property(self, name: str, atoms=None, allow_calculation: bool = True):
        self._ensure(atoms)
        return self.results[name]

    def get_stress(self, atoms=None):
        raise NotImplementedError(
            "MMFF94 through RDKit has no periodic cell, so there is no stress tensor"
        )

    def calculation_required(self, atoms, properties) -> bool:
        return self._positions is None or not np.allclose(
            np.asarray(atoms.get_positions()), self._positions
        )

    def check_state(self, atoms, tol: float = 1e-12) -> list[str]:
        return [] if not self.calculation_required(atoms, ["energy"]) else ["positions"]

    def _ensure(self, atoms) -> None:
        if atoms is None:
            raise ValueError("MMFFCalculator needs an Atoms object.")
        positions = np.asarray(atoms.get_positions(), dtype=float)
        if self._positions is not None and np.allclose(positions, self._positions):
            return

        from rdkit.Chem import AllChem

        conformer = self.mol.GetConformer()
        for index, position in enumerate(positions):
            conformer.SetAtomPosition(index, position.tolist())
        field = AllChem.MMFFGetMoleculeForceField(self.mol, self.properties)

        energy_kcal = field.CalcEnergy()
        gradient_kcal = np.asarray(field.CalcGrad(), dtype=float).reshape(-1, 3)
        self.results = {
            "energy": energy_kcal * self.KCAL_PER_MOL_TO_EV,
            # RDKit returns dE/dx; a force is its negative.
            "forces": -gradient_kcal * self.KCAL_PER_MOL_TO_EV,
        }
        self._positions = positions.copy()
        self.evaluations += 1
