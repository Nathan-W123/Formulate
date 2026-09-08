"""Hide a known material, describe it only by what it does, and look for it.

Section 12 asks for top-k recall against a benchmark set when a known solution
exists. Every other benchmark here measures one part - an expert against a
handbook value, a potential against a solver. This measures the system.

Two arms, reported apart because pooling them flatters the result:

**Retrieval** leaves the answer in the catalogue and asks where the ranking
puts it. It is close to tautological - the material is described by its own
measurements and a lookup expert holds exactly those measurements - and it is
run anyway, because failing it would mean the ranking is broken rather than
that search is hard.

**Discovery** removes the answer from the catalogue. Anything found there was
built by an explorer, and the strategy that built it is printed.

Run with ``python bench/recover_known_materials.py``. Results are recorded in
docs/BENCHMARKS.md.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from formulate.coordination.recovery import (  # noqa: E402
    RecoverySuite,
    describe_material,
    run_recovery,
)
from formulate.exploration.database import load_reference_compounds  # noqa: E402

#: Four common solvents spanning the reference set's chemistry: an aromatic, a
#: hydrogen-bonding alcohol, a polar aprotic ketone and a chlorinated one.
HIDE = ("toluene", "ethanol", "acetone", "chloroform")
ROUNDS, BATCH = 3, 20

#: Acceptance windows, as fractions of each property's observed spread. The
#: second is not a fudge to make the numbers look better: desirability is
#: risk-adjusted, so a window narrower than about two standard deviations of
#: the answering method scores every candidate zero on that axis and collapses
#: the hypervolume. Running both is how that diagnosis is checked rather than
#: asserted - if it is right, the wider window has a live hypervolume and a
#: similar recall; if the recall moves a lot too, the windows were doing more
#: work than intended and the tighter number is the honest one.
TOLERANCES = (0.10, 0.35)


def main() -> None:
    records = {r["name"]: r for r in load_reference_compounds()}

    for tolerance in TOLERANCES:
        print("=" * 72)
        print(f"acceptance window: {tolerance:.0%} of each property's observed spread")
        print("=" * 72)
        suite = RecoverySuite()

        for name in HIDE:
            record = records.get(name)
            if record is None:
                print(f"{name}: not in the reference set, skipped")
                continue
            hidden = describe_material(record)
            if hidden is None:
                print(f"{name}: fewer than three tabulated properties, skipped")
                continue
            print(f">> {hidden.describe()}", flush=True)

            for mode in ("retrieval", "discovery"):
                result = run_recovery(
                    hidden,
                    mode=mode,
                    rounds=ROUNDS,
                    batch_size=BATCH,
                    tolerance=tolerance,
                )
                suite.results.append(result)
                print(result.describe(), flush=True)

        print()
        print(suite.describe())
        print()


if __name__ == "__main__":
    main()
