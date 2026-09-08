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
    run.add_argument(
        "--validate",
        type=int,
        default=0,
        metavar="N",
        help=(
            "after searching, spend quantum or dynamics calculations on the N most "
            "informative candidate-property pairs and re-rank on the result. Only "
            "properties physics can legitimately produce are eligible"
        ),
    )
    run.add_argument(
        "--validate-seconds",
        type=float,
        default=600.0,
        help="wall-clock ceiling for the validation stage",
    )
    run.add_argument("--save", metavar="DIR", help="write the run record to this directory")
    run.add_argument("--json", action="store_true", help="emit the run record instead of a report")

    subparsers.add_parser("experts", help="list registered experts and their coverage")
    subparsers.add_parser("properties", help="list the canonical property registry")
    subparsers.add_parser("example", help="print an example target specification")
    subparsers.add_parser(
        "physics", help="report which quantum and dynamics backends are usable here"
    )

    bench = subparsers.add_parser(
        "benchmark",
        help=(
            "compare the adaptive coordinator against the fixed pipeline at equal "
            "wall-clock budget, weigh the evaluations each spent, and say which to use"
        ),
    )
    bench.add_argument("spec", help="path to a target specification")
    bench.add_argument("--seeds", type=int, default=3, help="paired runs to perform")
    bench.add_argument(
        "--budget", type=float, default=60.0, help="seconds each arm may spend per run"
    )
    bench.add_argument("--pool", type=int, default=20, help="initial pool size")

    calibrate = subparsers.add_parser(
        "calibrate", help="check expert accuracy and uncertainty against reference compounds"
    )
    calibrate.add_argument("--json", action="store_true", help="emit machine-readable results")
    calibrate.add_argument(
        "--expert",
        help=(
            "restrict the comparison to one expert, by id. Without it a compiled "
            "measurement answers wherever it has a value, and the result is the "
            "database agreeing with itself rather than an estimator being tested"
        ),
    )
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

    if args.validate > 0:
        from formulate.coordination import (
            IterationConfig,
            ValidationPolicy,
            default_validating_coordinator,
        )

        coordinator = default_validating_coordinator(
            config,
            IterationConfig(
                max_rounds=max(0, args.rounds),
                batch_size=args.batch,
                max_evaluations=args.max_evaluations,
            ),
            ValidationPolicy(
                max_candidates=args.validate, max_seconds=args.validate_seconds
            ),
        )
        validated = coordinator.run_validated(spec)
        run = validated.final
        rendered = validated.report(top_k=args.top)
    elif args.rounds > 0:
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


def _cmd_physics(_: argparse.Namespace) -> int:
    from formulate.physics import describe as describe_backends
    from formulate.physics.md import REQUIREMENTS

    print(describe_backends())
    print()
    print("What each dynamics protocol needs before its output means anything:")
    for protocol, requirement in REQUIREMENTS.items():
        periodic = "a periodic cell" if requirement.needs_periodic else "no periodic cell"
        print(
            f"  {protocol.value:24s} {periodic}, at least {requirement.min_molecules} "
            f"molecules, {requirement.min_production_ps:.0f} ps"
        )
        print(f"      {requirement.rationale}")

    # The section 8 rungs. They are not dynamics backends and do not belong in
    # the table above, but this command is where someone looks to find out what
    # physics is available, so leaving them out made them undiscoverable.
    from formulate.physics.potentials import available_potentials
    from formulate.physics.qmmm import QMMMCalculator

    print()
    print("Fast interatomic potentials (single point only):")
    potentials = available_potentials()
    if not potentials:
        from formulate.physics.potentials import MacePotential

        print(f"  none installed - {MacePotential().unavailable_reason()}")
    for potential in potentials:
        info = potential.info
        print(f"  {info.identifier}  elements {sorted(info.supported_elements)}")
        print(f"      trained at {info.training_level or 'an unstated level'}")
        print(
            "      uncertainty estimator: "
            + ("yes" if info.provides_uncertainty else "none, so escalation uses the "
               "element domain and cross-method disagreement instead")
        )

    print()
    print("QM/MM embedding:")
    capabilities = QMMMCalculator().capabilities()
    if not capabilities["available"]:
        print(f"  unavailable - {capabilities['unavailable_reason']}")
    else:
        print(f"  modes: {', '.join(capabilities['embedding_modes'])}")
        for mode, reason in capabilities["unsupported_embedding_modes"].items():
            print(f"  {mode}: unsupported - {reason}")
        print(f"  link atoms: {capabilities['link_atoms']}")
        print(f"  boundary charges: {capabilities['boundary_charge_scheme']}")
        print(f"  MM charges from: {capabilities['mm_charge_source']}")
        print(f"  gradients: {'yes' if capabilities['gradients'] else 'not implemented'}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from formulate.coordination import IncomparableTarget, run_benchmark
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

    try:
        result = run_benchmark(
            spec,
            seeds=tuple(range(max(1, args.seeds))),
            budget_seconds=args.budget,
            pool_size=args.pool,
        )
    except IncomparableTarget as exc:
        print(f"error: this target cannot be benchmarked.\n{exc}", file=sys.stderr)
        return 2

    print(result.describe())
    # A negative result is a legitimate finding, not a failure, so it does not
    # set a failing exit status; only an inability to decide does.
    return 0 if result.pairs else 1


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from formulate.evaluation.calibration import calibrate, describe
    from formulate.experts import default_registry
    from formulate.exploration import load_reference_compounds

    results = calibrate(
        default_registry(), load_reference_compounds(), expert_id=args.expert
    )
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
                        "answered_by": entry.answered_by,
                        "self_comparison": entry.self_comparison,
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
    "physics": _cmd_physics,
    "benchmark": _cmd_benchmark,
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
