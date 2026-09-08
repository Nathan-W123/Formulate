"""The adaptive coordinator against the fixed pipeline, paired by seed.

Section 10 and the brief both say the same thing: keep the adaptive path only
if it improves on the fixed one at equal compute. Both halves matter, and the
second is the one a benchmark quietly drops - an arm that evaluates three times
as many candidates and ends slightly ahead has shown nothing about its policy.

Run with ``python bench/adaptive_vs_fixed.py``. Results are recorded in
docs/BENCHMARKS.md.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from formulate.coordination.benchmark import run_benchmark  # noqa: E402
from formulate.targets.spec import TargetSpec  # noqa: E402

SPEC = TargetSpec.from_dict(
    {
        "name": "coating solvent",
        "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
        "requirements": [
            {
                "property": "normal_boiling_point",
                "direction": "in_range",
                "lower": "60 degC",
                "upper": "170 degC",
                "hard": True,
            },
            {
                "property": "liquid_density",
                "direction": "target",
                "target": "0.85 g/cm^3",
                "lower": "0.6 g/cm^3",
                "upper": "1.1 g/cm^3",
            },
            {"property": "logp", "direction": "target", "target": 2.0, "lower": 0, "upper": 4},
            {
                "property": "surface_tension",
                "direction": "minimize",
                "lower": "15 mN/m",
                "upper": "45 mN/m",
            },
        ],
    }
)


def main() -> None:
    result = run_benchmark(
        SPEC,
        seeds=(0, 1, 2, 3, 4),
        budget_seconds=90.0,
        pool_size=20,
        batch_size=12,
        validation_candidates=0,
    )
    print(result.describe())


if __name__ == "__main__":
    main()
