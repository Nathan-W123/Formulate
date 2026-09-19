"""Walk the spinline geometry, since the engine ranks materials at one.

The geometry is a condition, not a candidate, so a run answers "which material
at this spinline". Choosing the spinline is the operator's job - the same way
choosing a strand diameter was - and this walks it.

The lever worth knowing about: a finer filament costs pressure as one over the
die radius squared, but the *final* diameter is the die divided by the square
root of the draw ratio. So opening the die and drawing harder gives the same
filament at a quarter of the pressure. Pressure is bought with draw, not with
a narrower hole.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from score_strand import ReferencePolymerExplorer  # noqa: E402

from formulate.coordination import DeterministicCoordinator  # noqa: E402
from formulate.targets import TargetSpec  # noqa: E402

BASE = Path(__file__).parent / "spinline.yaml"


def run_at(die_um: float, draw: float, mn_kg_mol: float, land_mm: float | None = None):
    data = yaml.safe_load(BASE.read_text())
    spinline = data["conditions"]["spinline"]
    spinline["die_diameter"] = f"{die_um} um"
    spinline["draw_ratio"] = draw
    spinline["die_land"] = f"{land_mm if land_mm else die_um / 500.0} mm"
    explorer = ReferencePolymerExplorer()
    explorer.number_average_molar_mass_kg_mol = mn_kg_mol
    run = DeterministicCoordinator(explorers=[explorer]).run(TargetSpec.from_dict(data))
    winners = [t.candidate.label for t in run.ranking.ranked if t.feasible]
    return run, winners


if __name__ == "__main__":
    print("=" * 74)
    print("SPINLINE SWEEP - die, draw and chain length")
    print("=" * 74)
    print("\n  final filament is die / sqrt(draw); all of these land near 60 um\n")
    print(f"{'die':>7}{'draw':>7}{'final':>8}{'Mn':>7}{'feasible':>10}   material")
    hits = []
    for die, draw in ((300, 25), (600, 100), (900, 225), (1200, 400)):
        for mn in (60, 100, 150, 250):
            run, winners = run_at(die, draw, mn)
            final = die / draw**0.5
            if winners:
                hits.append((die, draw, mn, winners))
            print(f"{die:>5}um{draw:>7.0f}{final:>6.0f}um{mn:>6}k"
                  f"{run.ranking.feasible_count:>10}   {', '.join(winners[:3]) or '-'}")

    if hits:
        die, draw, mn, winners = hits[0]
        print(f"\n  first feasible: {die} um die, drawn {draw:.0f}x, {mn} kg/mol -> {winners}")
    else:
        print("\n  nothing feasible anywhere in this sweep")
