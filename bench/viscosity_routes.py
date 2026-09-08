"""Three routes to a liquid shear viscosity, against 273 measured values.

Viscosity spans orders of magnitude, so error is judged on log10: a method that
puts 1 mPa s at 3 is wrong by the same amount as one that puts 100 at 300, and
an absolute error in Pa s would say otherwise.

Run with ``python bench/viscosity_routes.py``. Results are recorded in
docs/BENCHMARKS.md.
"""

from __future__ import annotations

import hashlib
import math
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from formulate.experts.rheology import (  # noqa: E402
    acentric_factor,
    joback_viscosity,
    letsou_stiel_viscosity,
)

T = 298.15


def measured_set() -> list[tuple[str, float]]:
    """Structures and measured viscosities at 298 K from the DIPPR compilation."""
    from chemicals.identifiers import search_chemical
    import chemicals.viscosity as viscosity
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    viscosity._load_mu_data()
    rows = []
    for cas, row in viscosity.mu_data_Perrys_8E_2_313.iterrows():
        # Only where the correlation is validated. Evaluating a DIPPR fit
        # outside its stated range is how a "measured" value becomes an
        # extrapolation, and it moved the benchmark by 0.02 log units when it
        # was left out.
        low, high = row.get("Tmin"), row.get("Tmax")
        if low and T < low - 5:
            continue
        if high and T > high + 5:
            continue
        try:
            mu = math.exp(
                row["C1"] + row["C2"] / T + row["C3"] * math.log(T)
                + row["C4"] * T ** row["C5"]
            )
        except Exception:
            continue
        if not 1e-5 < mu < 10.0:
            continue
        try:
            smiles = search_chemical(cas).smiles
        except Exception:
            continue
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None or mol.GetNumAtoms() < 2:
            continue
        canonical = Chem.MolToSmiles(mol)
        if "." in canonical or Chem.GetFormalCharge(mol) != 0:
            continue
        if not any(a.GetSymbol() == "C" for a in mol.GetAtoms()):
            continue
        rows.append((canonical, mu))
    return rows


def corresponding_states(smiles: str, *, corrected: bool) -> float | None:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from thermo.group_contribution import Joback

    try:
        joback = Joback(smiles)
        tb = joback.Tb(joback.counts)
        tc = joback.Tc(joback.counts)
        pc = joback.Pc(joback.counts, joback.atom_count)
    except Exception:
        return None
    if not all(x and x > 0 for x in (tb, tc, pc)):
        return None
    omega = acentric_factor(tb, tc, pc)
    if omega is None:
        return None
    mass = Descriptors.MolWt(Chem.MolFromSmiles(smiles))
    return letsou_stiel_viscosity(T, mass, tc, pc, omega, corrected=corrected)


def report(name: str, pairs: list[tuple[float, float]], total: int) -> None:
    errors = np.array([math.log10(p / m) for m, p in pairs])
    print(
        f"  {name:38s} answers {len(pairs):3d}/{total}  "
        f"MAE(log10) {np.abs(errors).mean():.3f}  "
        f"median factor {10 ** np.median(np.abs(errors)):.2f}x  "
        f"bias {errors.mean():+.3f}  "
        f"within 2x {np.mean(np.abs(errors) < math.log10(2)):.0%}"
    )


def main() -> None:
    data = measured_set()
    print(f"{len(data)} compounds with a measured liquid viscosity at 298 K\n")

    joback = [(m, v) for s, m in data if (v := joback_viscosity(s, T)) is not None]
    published = [
        (m, v) for s, m in data
        if (v := corresponding_states(s, corrected=False)) is not None
    ]
    report("Joback group contribution", joback, len(data))
    report("Letsou-Stiel as published", published, len(data))

    # The offset is fitted on half and measured on the other half, split on a
    # hash of the structure so it does not move between runs.
    def bucket(smiles: str) -> int:
        return int(hashlib.sha256(smiles.encode()).hexdigest(), 16) % 2

    rows = [
        (s, m, v) for s, m in data
        if (v := corresponding_states(s, corrected=False)) is not None
    ]
    fit = [r for r in rows if bucket(r[0]) == 0]
    held = [r for r in rows if bucket(r[0]) == 1]
    offset = float(np.mean([math.log10(v / m) for _, m, v in fit]))
    print(
        f"\n  offset fitted on {len(fit)} compounds: {offset:+.4f} log10 "
        f"(a factor of {10 ** -offset:.2f})"
    )
    report(
        "Letsou-Stiel + offset, held out",
        [(m, v * 10.0**-offset) for _, m, v in held],
        len(held),
    )

    overlap = [
        (m, j, corresponding_states(s, corrected=True))
        for s, m in data
        if (j := joback_viscosity(s, T)) is not None
    ]
    overlap = [(m, j, ls) for m, j, ls in overlap if ls is not None]
    if overlap:
        ej = np.array([abs(math.log10(j / m)) for m, j, _ in overlap])
        el = np.array([abs(math.log10(ls / m)) for m, _, ls in overlap])
        print(
            f"\n  head to head on the {len(overlap)} both answer: "
            f"Joback {ej.mean():.3f} against corresponding states {el.mean():.3f}; "
            f"Joback wins {np.mean(ej < el):.0%}"
        )
    print(
        "\nNeither has precedence in the code: each states its own spread and "
        "prefer() chooses,\nwhich puts the measured route first, Joback second and "
        "corresponding states last."
    )


if __name__ == "__main__":
    main()
