"""The minimum cartridge pressure a spinnable dope needs, at the best geometry.

Reproduces the number in the README's "So how much pressure does it need?"
section: lift only the extrusion_pressure cap and see what the pool can do.
"""
import math
import sys

import yaml

from formulate.coordination import DeterministicCoordinator
from formulate.coordination.coordinator import RunConfig
from formulate.targets import TargetSpec

SPEC = "examples/web_shooter/dope.yaml"
STRENGTH_PA = 300e6  # a drawn, oriented fibre. Stated, not predicted: see the README.


def main(die_um=1000, land_mm=0.2, speed=20, draw=100, count=200, out="/tmp/_p.yaml"):
    base = yaml.safe_load(open(SPEC))
    base["conditions"]["spinline"].update(
        {
            "die_diameter": f"{die_um} um",
            "die_land": f"{land_mm} mm",
            "line_speed": f"{speed} m/s",
            "draw_ratio": float(draw),
            "filament_count": int(count),
        }
    )
    for requirement in base["requirements"]:
        if requirement["property"] == "extrusion_pressure":
            requirement["upper"] = "100000 bar"
    yaml.safe_dump(base, open(out, "w"))

    run = DeterministicCoordinator(config=RunConfig(pool_size=900)).run(TargetSpec.from_file(out))
    rows = []
    for entry in run.ranking.ranked:
        if [o for o in run.outcomes_for(entry.candidate) if o.violation is not None]:
            continue
        have = {p.property: p for p in entry.candidate.results.predictions}
        rows.append(
            (
                have["extrusion_pressure"].quantity.to("bar").value,
                entry.candidate.label,
                have["shear_viscosity"].quantity.to("Pa*s").value,
                have["extensional_strain_hardening"].quantity.value,
                have["shear_thinning_ratio"].quantity.value,
            )
        )
    rows.sort()

    final_um = die_um / math.sqrt(draw)
    area = count * math.pi / 4 * (final_um * 1e-6) ** 2
    print(f"{len(rows)} candidates pass every hard constraint once the pressure cap is lifted")
    print(
        f"{final_um:.0f} um filament x {count} = {area*1e6:.2f} mm^2 of bundle; "
        f"at {STRENGTH_PA/1e6:.0f} MPa that holds {area*STRENGTH_PA/4.448:.0f} lb"
    )
    print(f"\n{'bar':>9}  {'eta Pa.s':>10} {'Wi':>7} {'thin':>7}  polymer")
    seen = set()
    for bar, label, eta, wi, thin in rows:
        stem = label.split(",")[0]
        if stem in seen:
            continue
        seen.add(stem)
        print(f"{bar:9.0f}  {eta:10.3g} {wi:7.1f} {thin:7.1f}  {stem}")
        if len(seen) >= 12:
            break


if __name__ == "__main__":
    main(*(int(float(a)) if i != 1 else float(a) for i, a in enumerate(sys.argv[1:])))
