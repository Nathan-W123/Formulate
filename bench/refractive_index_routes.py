"""Lorentz-Lorenz against a learned forest, on the same compounds.

A first pass compared 4327 compounds for the physics route against 883 for the
forest, on different populations, with the forest's training set overlapping
the physics route's test set. It reported the forest six times better and was
measuring the wrong thing: the physics route had been fed a Joback-then-Rackett
molar volume, and almost all of its error was the density estimate rather than
the equation.

This one holds the compounds fixed - the 385 that carry a tabulated liquid
density - and refits the forest with every one of them removed, so all three
rows below are on the same molecules and none of them was seen in training.

Run with ``python bench/refractive_index_routes.py``. Needs formulate[learned].
Results are recorded in docs/BENCHMARKS.md.
"""

from __future__ import annotations

import json
import math
import warnings
from importlib import resources

import numpy as np

warnings.filterwarnings("ignore")

from formulate.experts.interfacial import rackett_molar_volume  # noqa: E402
from formulate.experts.learned import featurise  # noqa: E402
from formulate.experts.measured import measured_value  # noqa: E402


def measurements() -> list[tuple[str, float]]:
    with resources.files("formulate.data").joinpath(
        "refractive_index_measurements.json"
    ).open() as handle:
        return [tuple(row) for row in json.load(handle)["rows"]]


def lorentz_lorenz(molar_refraction: float, molar_volume: float) -> float | None:
    ratio = molar_refraction / molar_volume
    if not 0.0 < ratio < 1.0:
        return None
    return math.sqrt((1.0 + 2.0 * ratio) / (1.0 - ratio))


def estimated_molar_volume(mol, temperature: float = 293.15) -> float | None:
    """A Joback-then-Rackett molar volume, in cm^3/mol: what a novel molecule gets."""
    from rdkit import Chem
    from thermo.group_contribution import Joback

    try:
        joback = Joback(Chem.MolToSmiles(mol))
        tc = joback.Tc(joback.counts)
        pc = joback.Pc(joback.counts, joback.atom_count)
        vc = joback.Vc(joback.counts)
    except Exception:
        return None
    if not all(v and v > 0 for v in (tc, pc, vc)) or temperature >= tc:
        return None
    try:
        return rackett_molar_volume(tc, pc, vc, temperature) * 1e6
    except Exception:
        return None


def main() -> None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors
    from sklearn.ensemble import RandomForestRegressor

    RDLogger.DisableLog("rdApp.*")

    held, rest = [], []
    for smiles, value in measurements():
        density = measured_value("liquid_density", smiles)  # kg/m^3
        (held if density else rest).append((smiles, value, density))
    print(f"with a tabulated density: {len(held)}   without: {len(rest)}")

    features, targets = [], []
    for smiles, value, _ in rest:
        vector = featurise(smiles)
        if vector is not None:
            features.append(vector)
            targets.append(value)
    forest = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1).fit(
        np.array(features), np.array(targets)
    )
    print(f"forest fitted on {len(targets)}, none of them in the comparison set")

    with_measured, with_estimated, learned, truth = [], [], [], []
    for smiles, value, density in held:
        mol = Chem.MolFromSmiles(smiles)
        vector = featurise(smiles)
        if mol is None or vector is None:
            continue
        refraction = Crippen.MolMR(mol)
        measured_volume = Descriptors.MolWt(mol) / (density / 1000.0)
        physics = lorentz_lorenz(refraction, measured_volume)
        if physics is None:
            continue
        estimated_volume = estimated_molar_volume(mol)
        with_measured.append(physics)
        with_estimated.append(
            lorentz_lorenz(refraction, estimated_volume) if estimated_volume else None
        )
        learned.append(float(forest.predict([vector])[0]))
        truth.append(value)

    reference = np.array(truth)

    def report(label: str, values: list[float | None]) -> None:
        array = np.array([np.nan if v is None else v for v in values], dtype=float)
        answered = ~np.isnan(array)
        error = array[answered] - reference[answered]
        print(
            f"  {label:38s} answers {answered.mean():4.0%}  "
            f"MAE {np.abs(error).mean():.4f}  bias {error.mean():+.4f}  "
            f"RMSE {np.sqrt((error**2).mean()):.4f}"
        )

    print(f"\nOn the same {len(reference)} compounds:")
    report("Lorentz-Lorenz, measured density", with_measured)
    report("Lorentz-Lorenz, estimated density", with_estimated)
    report("learned forest (never saw these)", learned)
    print(
        "\nNeither route wins outright, so neither has precedence in the code: the "
        "physics route propagates its density's uncertainty and states a narrow spread "
        "on a measured density and a wide one on an estimate, and prefer() chooses."
    )


if __name__ == "__main__":
    main()
