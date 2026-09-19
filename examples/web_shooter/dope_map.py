"""Where a solution dope has to sit before it will strain-harden.

The melt route came out noise-limited: every feasible candidate sat at a
Weissenberg number of 0.5 to 0.8 against a hard bound of 0.5, which is to say
it passed inside its own error bar. A solution has two knobs the melt does not
- concentration and chain length, independently - and this maps them.
"""

from __future__ import annotations

import yaml

from formulate.core.candidate import (
    Candidate,
    ComponentRole,
    FractionBasis,
    MaterialClass,
    MixtureComponent,
    MixtureSpec,
    MoleculeSpec,
    MonomerUnit,
    PolymerSpec,
)
from formulate.core.conditions import Conditions
from formulate.core.quantity import Quantity
from formulate.evaluation import EvaluationEngine
from formulate.experts import default_registry
from formulate.targets import TargetSpec

PROPS = [
    "shear_viscosity",
    "extensional_strain_hardening",
    "extrusion_pressure",
    "entanglement_molar_mass",
]
#: The threshold below which a thread beads up instead of thinning.
ONSET = 0.5
POLYSTYRENE = "[*]CC(c1ccccc1)[*]"
ACETONE = "CC(C)=O"


def _spec(path: str = "/tmp/_dope_map.yaml") -> TargetSpec:
    document = {
        "name": "dope map",
        "conditions": {
            "temperature": "25 degC",
            "pressure": "1 atm",
            "spinline": {
                "die_diameter": "1000 um",
                "die_land": "0.2 mm",
                "draw_ratio": 100.0,
                "line_speed": "20 m/s",
                "filament_count": 200,
            },
        },
        "material_classes": ["mixture"],
        "requirements": [
            {"property": p, "direction": "maximize", "lower": -1e12, "upper": 1e12}
            for p in PROPS
        ],
    }
    with open(path, "w") as handle:
        yaml.safe_dump(document, handle)
    return TargetSpec.from_file(path)


def _dope(mass_fraction: float, kg_per_mol: float) -> Candidate:
    return Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=(
                MixtureComponent(
                    role=ComponentRole.SOLUTE,
                    fraction=mass_fraction,
                    polymer=PolymerSpec(
                        monomers=(MonomerUnit(smiles=POLYSTYRENE),),
                        number_average_molar_mass=Quantity(value=kg_per_mol, unit="kg/mol"),
                    ),
                ),
                MixtureComponent(
                    role=ComponentRole.SOLVENT,
                    fraction=1.0 - mass_fraction,
                    molecule=MoleculeSpec(smiles=ACETONE),
                ),
            ),
            basis=FractionBasis.MASS,
        ),
        conditions=Conditions.standard(),
    )


def main() -> None:
    spec = _spec()
    grid = [(w, m) for w in (0.20, 0.35, 0.50, 0.65) for m in (200.0, 1000.0, 4000.0)]
    per_candidate, _ = EvaluationEngine(default_registry()).predict(
        [_dope(w, m) for w, m in grid], spec
    )

    print(f"{'wt%':>5} {'kg/mol':>8} {'eta Pa.s':>11} {'Wi':>9} {'bar':>9} {'Me kg/mol':>10}  spins?")
    for (fraction, mass), predictions in zip(grid, per_candidate):
        found = {p.property: p for p in predictions}

        def value(name: str, unit: str) -> float:
            prediction = found.get(name)
            if prediction is None or prediction.quantity is None:
                return float("nan")
            return prediction.quantity.to(unit).value

        weissenberg = value("extensional_strain_hardening", "dimensionless")
        print(
            f"{fraction*100:5.0f} {mass:8.0f} {value('shear_viscosity', 'Pa*s'):11.4g} "
            f"{weissenberg:9.3g} {value('extrusion_pressure', 'bar'):9.3g} "
            f"{value('entanglement_molar_mass', 'kg/mol'):10.3g}"
            f"  {'YES' if weissenberg >= ONSET else 'no'}"
        )


if __name__ == "__main__":
    main()
