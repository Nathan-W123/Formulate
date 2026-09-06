"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from formulate import __version__

_EXAMPLE_SPEC = """\
# A Formulate target specification.
#
# Every requirement states a property, a direction, and the bounds that define
# where desirability reaches 0 and 1. Quantities may be written as bare numbers
# (read in the property's canonical unit) or as "value unit" strings.
name: low-VOC coating solvent

conditions:
  temperature: 25 degC
  pressure: 1 atm

material_classes: [molecule]

structural:
  allowed_elements: [C, H, O]
  allow_charged: false
  max_heavy_atoms: 20

requirements:
  # A hard requirement eliminates a candidate that breaches it.
  - property: normal_boiling_point
    direction: in_range
    lower: 140 degC
    upper: 200 degC
    hard: true
    rationale: slow enough to level, fast enough to cure

  - property: surface_tension
    direction: minimize
    lower: 0.020 N/m
    upper: 0.035 N/m
    weight: 2.0
    rationale: wetting of a low-energy substrate

  - property: synthetic_accessibility
    direction: minimize
    lower: 1
    upper: 5

# Anything the request did not say, that the system had to decide, belongs here.
assumptions:
  - statement: >-
      Low-VOC was read as a boiling-point window rather than a regulatory
      VOC definition.
    source: inferred
    made_by: operator
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="formulate",
        description=(
            "Behavior-driven inverse materials design: map desired material "
            "behavior to candidate molecules, polymers, blends and formulations."
        ),
    )
    parser.add_argument("--version", action="version", version=f"formulate {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a design query from a target specification")
    run.add_argument("spec", help="path to a target specification (.yaml or .json)")
    run.add_argument("--top", type=int, default=5, help="how many candidates to detail")
    run.add_argument("--pool", type=int, default=200, help="maximum candidates to draw")
    run.add_argument("--workers", type=int, default=4, help="concurrent evaluation threads")
    run.add_argument(
        "--no-risk-adjust",
        action="store_true",
        help="rank on predicted values rather than on uncertainty-adjusted ones",
    )
    run.add_argument(
        "--rounds",
        type=int,
        default=0,
        help=(
            "search iteratively for this many rounds after the seeding round. "
            "Evolutionary search only contributes when scores are fed back, so "
            "it does nothing at the default of 0"
        ),
    )
    run.add_argument("--batch", type=int, default=25, help="new candidates proposed per round")
    run.add_argument(
        "--max-evaluations", type=int, default=None, help="ceiling on candidates evaluated"
    )
    run.add_argument("--save", metavar="DIR", help="write the run record to this directory")
    run.add_argument("--json", action="store_true", help="emit the run record instead of a report")

    subparsers.add_parser("experts", help="list registered experts and their coverage")
    subparsers.add_parser("properties", help="list the canonical property registry")
    subparsers.add_parser("example", help="print an example target specification")

    calibrate = subparsers.add_parser(
        "calibrate", help="check expert accuracy and uncertainty against reference compounds"
    )
    calibrate.add_argument("--json", action="store_true", help="emit machine-readable results")
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    from formulate.coordination import DeterministicCoordinator, RunConfig
    from formulate.evaluation import EvaluationConfig
    from formulate.store.evidence import EvidenceStore
    from formulate.targets import TargetSpec

    path = Path(args.spec)
    if not path.exists():
        print(f"error: no such specification: {path}", file=sys.stderr)
        return 2
    try:
        spec = TargetSpec.from_file(path)
    except Exception as exc:
        print(f"error: could not read {path}: {exc}", file=sys.stderr)
        return 2

    config = RunConfig(
        evaluation=EvaluationConfig(
            max_workers=max(1, args.workers),
            use_risk_adjusted=not args.no_risk_adjust,
        ),
        pool_size=args.pool,
        top_k=args.top,
    )

    if args.rounds > 0:
        from formulate.coordination import IterationConfig, default_iterative_coordinator

        coordinator = default_iterative_coordinator(
            config,
            IterationConfig(
                max_rounds=args.rounds,
                batch_size=args.batch,
                max_evaluations=args.max_evaluations,
            ),
        )
        iterative = coordinator.run_iterative(spec)
        run = iterative.final
        rendered = iterative.report(top_k=args.top)
    else:
        run = DeterministicCoordinator(config=config).run(spec)
        rendered = run.report(top_k=args.top)

    if args.json:
        print(json.dumps(EvidenceStore(".").record(run), indent=2))
    else:
        print(rendered)

    if args.save:
        written = EvidenceStore(args.save).write(run)
        print(f"\nRun record written to {written}", file=sys.stderr)

    # A run that found nothing feasible is a legitimate scientific answer, not
    # a crash, but it is worth a distinct exit status for scripting.
    return 0 if run.ranking.feasible_count else 1


def _cmd_experts(_: argparse.Namespace) -> int:
    from formulate.experts import default_registry

    print(default_registry().describe())
    return 0


def _cmd_properties(_: argparse.Namespace) -> int:
    from formulate.core.properties import PROPERTY_REGISTRY

    width = max(len(n) for n in PROPERTY_REGISTRY)
    by_family: dict[str, list] = {}
    for definition in PROPERTY_REGISTRY.values():
        by_family.setdefault(definition.family.value, []).append(definition)
    for family in sorted(by_family):
        print(f"\n{family}")
        for definition in sorted(by_family[family], key=lambda d: d.name):
            flag = " (condition dependent)" if definition.condition_dependent else ""
            unit = "-" if definition.canonical_unit == "dimensionless" else definition.canonical_unit
            print(f"  {definition.name:<{width}}  {unit:<22} {definition.description}{flag}")
    return 0


def _cmd_example(_: argparse.Namespace) -> int:
    print(_EXAMPLE_SPEC, end="")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from formulate.evaluation.calibration import calibrate, describe
    from formulate.experts import default_registry
    from formulate.exploration import load_reference_compounds

    results = calibrate(default_registry(), load_reference_compounds())
    if args.json:
        print(
            json.dumps(
                {
                    name: {
                        "count": entry.count,
                        "mean_absolute_error": entry.mean_absolute_error,
                        "rmse": entry.rmse,
                        "bias": entry.bias,
                        "unit": entry.unit,
                        "within_one_sigma": entry.within_one_sigma,
                        "within_two_sigma": entry.within_two_sigma,
                        "verdict": entry.verdict(),
                    }
                    for name, entry in results.items()
                },
                indent=2,
            )
        )
    else:
        print(describe(results))
    return 0


_COMMANDS = {
    "run": _cmd_run,
    "experts": _cmd_experts,
    "properties": _cmd_properties,
    "example": _cmd_example,
    "calibrate": _cmd_calibrate,
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
