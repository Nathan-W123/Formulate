"""The three questions a wrist-mounted web fluid has to answer at once.

Every earlier run in this directory asked one or two of them and reported a
winner. The three are:

1. Does it SPIN?     extensional_strain_hardening at or above about a half,
                     or the thread beads up instead of thinning
2. Can it be FIRED?  extrusion_pressure inside what a cartridge on a person
                     can supply
3. Does it DISSOLVE? relative_energy_difference below one, at a temperature a
                     person can wear

The third is the one that keeps getting skipped, and skipping it is not a
small error: a viscosity computed for a polymer that will not dissolve is a
number about a suspension, not a dope. The solution expert prints
"THIS ASSUMES THE POLYMER DISSOLVES" on every prediction for exactly this
reason, and the assumption still went unchecked through several rounds of
this analysis.

It also does not always have an answer. The Hansen interaction radius is a
MEASURED quantity - this repository established that it is not derivable from
structure, over seven tabulated spheres whose R0/delta ratios span 3.45x with
r^2 = 0.11 - so a polymer without a fitted sphere gets "no fitted interaction
radius" rather than a number. That is a refusal, not a pass, and it is why
poly(4-methylstyrene) is absent from the answer despite outscoring everything
on the first two questions.
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
    Tacticity,
)
from formulate.core.conditions import Conditions
from formulate.core.quantity import Quantity
from formulate.evaluation import EvaluationEngine
from formulate.experts import default_registry
from formulate.targets import TargetSpec

#: Wearable. Warm to the touch, no burn, and the temperature everything below
#: is asked at - including the solubility question, which is what ruled out
#: gel-spun polyethylene: it needs about 130 C before its crystals let go.
TEMPERATURE_C = 40

#: Below this a thread beads up under its own surface tension however viscous.
SPIN_ONSET = 0.5

#: What a compressed-air cylinder of the kind paintball hardware carries can
#: supply on a wrist.
PRESSURE_CEILING_BAR = 250.0

#: Below one, Hansen's relative energy difference says the solvent dissolves it.
DISSOLVES_BELOW = 1.0

POLYMERS = {
    "polystyrene": "[*]CC(c1ccccc1)[*]",
    "PMMA": "[*]CC(C)(C(=O)OC)[*]",
    "poly(vinyl acetate)": "[*]CC(OC(C)=O)[*]",
    # Outscores every one of the above on spinnability and pressure, and is
    # absent from the answer because its solubility cannot be computed.
    "poly(4-methylstyrene)": "[*]CC(c1ccc(C)cc1)[*]",
}

#: Flash points from the flammability expert, rounded. A wrist device wants a
#: solvent that does not flash BELOW room temperature, which rules out the
#: obvious ones: acetone at -24 C and 2-butanone at -10.
SOLVENTS = {
    "2-heptanone": ("CCCCCC(C)=O", 37),
    "butyl acetate": ("CC(=O)OCCCC", 22),
    "o-xylene": ("Cc1ccccc1C", 28),
    "toluene": ("Cc1ccccc1", 5),
}

GRID = [(0.20, 1500.0), (0.15, 4000.0)]


def _spec(properties, path):
    document = {
        "name": "wrist candidate",
        "conditions": {
            "temperature": f"{TEMPERATURE_C} degC",
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
            {
                "property": p,
                "direction": "maximize",
                "lower": -1e12,
                "upper": 1e12,
                "conditions": {"temperature": f"{TEMPERATURE_C} degC"},
            }
            for p in properties
        ],
    }
    with open(path, "w") as handle:
        yaml.safe_dump(document, handle)
    return TargetSpec.from_file(path)


def _dope(repeat, solvent, fraction, molar_mass):
    return Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=(
                MixtureComponent(
                    role=ComponentRole.SOLUTE,
                    fraction=fraction,
                    polymer=PolymerSpec(
                        monomers=(MonomerUnit(smiles=repeat),),
                        tacticity=Tacticity.ATACTIC,
                        number_average_molar_mass=Quantity(value=molar_mass, unit="kg/mol"),
                    ),
                ),
                MixtureComponent(
                    role=ComponentRole.SOLVENT,
                    fraction=1.0 - fraction,
                    molecule=MoleculeSpec(smiles=solvent),
                ),
            ),
            basis=FractionBasis.MASS,
        ),
        conditions=Conditions.standard(),
    )


def main() -> None:
    engine = EvaluationEngine(default_registry())
    process = _spec(
        ["extensional_strain_hardening", "extrusion_pressure", "shear_viscosity"],
        "/tmp/_wrist_process.yaml",
    )
    solubility = _spec(["relative_energy_difference"], "/tmp/_wrist_solubility.yaml")

    rows = []
    for polymer, repeat in POLYMERS.items():
        for solvent, (smiles, flash) in SOLVENTS.items():
            dopes = [_dope(repeat, smiles, w, m) for w, m in GRID]
            spun, _ = engine.predict(dopes, process)
            wet, _ = engine.predict(dopes, solubility)
            for (fraction, molar_mass), made, soluble in zip(GRID, spun, wet):
                found = {p.property: p for p in made}
                found.update({p.property: p for p in soluble})

                def value(name, unit):
                    prediction = found.get(name)
                    if prediction is None or prediction.quantity is None:
                        return None
                    return prediction.quantity.to(unit).value

                weissenberg = value("extensional_strain_hardening", "dimensionless")
                bar = value("extrusion_pressure", "bar")
                red = value("relative_energy_difference", "dimensionless")
                if weissenberg is None or bar is None:
                    continue
                spins = weissenberg >= SPIN_ONSET and bar <= PRESSURE_CEILING_BAR
                dissolves = red is not None and red < DISSOLVES_BELOW
                rows.append(
                    (bar, weissenberg, red, polymer, solvent, flash, fraction, molar_mass,
                     spins and dissolves)
                )

    confirmed = sorted(r for r in rows if r[-1])
    print(
        f"{len(confirmed)} of {len(rows)} combinations answer ALL THREE questions "
        f"at {TEMPERATURE_C} C\n"
    )
    print(f"{'bar':>7} {'Wi':>6} {'RED':>6}  polymer / solvent (flash) / wt% / kg per mol")
    for bar, weissenberg, red, polymer, solvent, flash, fraction, molar_mass, _ in confirmed:
        print(
            f"{bar:7.1f} {weissenberg:6.2f} {red:6.2f}  {polymer} in {solvent} "
            f"(fp {flash} C), {fraction:.0%}, {molar_mass:g}"
        )

    unanswerable = [r for r in rows if r[2] is None]
    if unanswerable:
        names = sorted({r[3] for r in unanswerable})
        print(
            f"\n{len(unanswerable)} combinations could not be answered at all: "
            + ", ".join(names)
            + ". No fitted Hansen interaction radius, which this repository has "
            "measured to be underivable from structure. A solubility that cannot be "
            "computed is a refusal, not a pass."
        )


if __name__ == "__main__":
    main()
