"""Is the web shooter ranking decided by chemistry, or by error bars?

Section 12 of the specification asks that uncertainty be ranked on rather than
merely reported, and this is the script that makes the distinction pay. Every
requirement is scored twice - at the predicted value, and one standard
deviation into the unfavourable direction - so a candidate that clears a bound
only inside its own error bar is visible as such.

Run it on the full specification and the answer is stark: twelve candidates are
feasible and none of them is feasible robustly.
"""

from __future__ import annotations

import collections

from formulate.coordination import DeterministicCoordinator
from formulate.coordination.coordinator import RunConfig
from formulate.targets import TargetSpec

SPEC = "examples/web_shooter/optimal.yaml"


def main() -> None:
    spec = TargetSpec.from_file(SPEC)
    run = DeterministicCoordinator(config=RunConfig(pool_size=900)).run(spec)

    feasible = robust = 0
    marginal: collections.Counter[str] = collections.Counter()
    top = None
    for entry in run.ranking.ranked:
        outcomes = run.outcomes_for(entry.candidate)
        if [o for o in outcomes if o.violation is not None]:
            continue
        feasible += 1
        if top is None:
            top = entry.candidate
        at_risk = [o.requirement.property for o in outcomes if o.constraint_at_risk]
        if not at_risk:
            robust += 1
        marginal.update(at_risk)

    print(f"{feasible} feasible of {len(run.ranking.ranked)} evaluated")
    print(f"{robust} clear every hard bound robustly, i.e. one sigma the wrong way too")
    print("\nrequirements that hold only inside their own error bar:")
    for prop, count in marginal.most_common():
        print(f"   {prop:32s} {count:3d} / {feasible}")

    if top is None:
        return
    print(f"\ntop candidate: {top.label}")
    print("relative one-sigma, largest first:")
    rows = []
    for pred in top.results.predictions:
        if pred.quantity is None or pred.uncertainty is None or pred.uncertainty.std is None:
            continue
        # std is carried in the prediction's own unit, so compare it against the
        # value in that same unit rather than the canonical one.
        value = abs(pred.quantity.value)
        if value > 0:
            rows.append((pred.uncertainty.std / value * 100, pred.property, pred.expert_id))
    for percent, prop, expert in sorted(rows, reverse=True):
        print(f"   {prop:32s} {percent:8.0f}%   via {expert}")


if __name__ == "__main__":
    main()
