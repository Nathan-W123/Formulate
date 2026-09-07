# Formulate

Behavior-driven inverse materials design: an engine that maps desired material
*behavior* to candidate molecules, polymers, blends and formulations.

Given a target specification — properties, operating conditions, tolerances,
priorities and constraints — Formulate searches chemical design space and
returns ranked, feasible candidates with their predicted properties,
uncertainties and constraint violations.

It does **not** try to invert a property model. Inverse design is many-to-many:
a request may have several answers or none. The system explores, predicts,
ranks, and (from Phase 3) selectively validates with physics.

## Install

```bash
pip install -e '.[chem,dev]'
```

The chemistry extra pulls in RDKit and `thermo`. Without it, `formulate.core`,
`formulate.targets` and `formulate.ranking` still import and test cleanly; the
experts report themselves unavailable rather than crashing.

## Quick start

```bash
formulate example > solvent.yaml            # a commented specification to edit
formulate run solvent.yaml --top 5          # search, rank and report
formulate run solvent.yaml --rounds 5       # iterative search with evolution
formulate run solvent.yaml --validate 3     # add selective QM/MD validation
formulate experts                           # what the panel can predict
formulate properties                        # the canonical property registry
formulate physics                           # which QM/MD backends are usable here
formulate benchmark solvent.yaml            # adaptive vs fixed, and which to use
formulate calibrate                         # accuracy and uncertainty vs. reference data
```

A specification states what you want, in units:

```yaml
name: low-VOC coating solvent
conditions:
  temperature: 25 degC
  pressure: 1 atm
structural:
  allowed_elements: [C, H, O]
requirements:
  - property: normal_boiling_point
    direction: in_range
    lower: 140 degC
    upper: 200 degC
    hard: true                    # eliminates a candidate that breaches it
  - property: surface_tension
    direction: minimize
    lower: 0.020 N/m              # anchors where desirability hits 1 and 0
    upper: 0.035 N/m
    weight: 2.0
assumptions:
  - statement: Low-VOC was read as a boiling-point window, not a regulatory definition.
    source: inferred
```

Or from Python:

```python
from formulate.coordination import DeterministicCoordinator
from formulate.targets import TargetSpec

run = DeterministicCoordinator().run(TargetSpec.from_file("solvent.yaml"))
print(run.report(top_k=5))
for entry in run.ranking.frontier:
    print(entry.candidate.label, entry.candidate.results.objective_vector)
```

## How it is put together

Module separation is a scientific invariant, not a style preference. The
generator proposes, experts predict, the ranker optimises, physics validates,
and the reasoning layer explains. No module silently substitutes for another.

| Package | Responsibility |
|---|---|
| `core` | units, quantities, uncertainty, conditions, the canonical `Candidate`, content addressing, provenance |
| `targets` | `TargetSpec`: properties, directions, tolerances, priorities, hard/soft status, surfaced assumptions |
| `experts` | model-agnostic property prediction with declared dependencies and applicability domains |
| `evaluation` | expert dispatch, dimensional and condition validation, raw values to dimensionless utilities |
| `ranking` | constraints, Pareto frontier, hypervolume, structural diversity, scalar baseline |
| `exploration` | candidate generation (retrieval, evolution) and cheap validity filters |
| `physics` | quantum and molecular-dynamics validation behind established packages |
| `coordination` | the pipeline, the iterate loop, selective validation, reporting |
| `store` | content-addressed prediction cache and the evidence store |

### Four rules that shape the code

**Nothing is ranked before its dimensions agree.** Every value carries a unit
parsed by one shared registry. Uncertainties convert with difference semantics,
so a 12.9 K error bar stays 12.9 in degrees Celsius instead of becoming −260.

**Absence is never a neutral score.** A prediction that could not be made
carries an explicit status and no number, so downstream code cannot read it as
a mid-range result. A hard requirement that could not be *checked* makes a
candidate infeasible rather than letting it pass by default.

**Uncertainty is ranked on, not just reported.** Each requirement is scored
twice: at the predicted value, and at a pessimistic bound one standard
deviation into the unfavourable direction. A candidate that meets a bound only
within its own error bar is marked as such.

**A failed search is an answer.** When nothing is feasible, the run reports
which constraint eliminated what, which single relaxation would admit
candidates, how far the nearest miss is, and which constraints were never
satisfied together.

**Physics refuses what it cannot support.** A bulk density needs a periodic
condensed phase, and on this installation no potential can provide one at the
required size. Rather than returning a number from eight molecules with a wide
error bar — the error is systematic, so the error bar would not rescue it —
the run refuses and estimates what an adequate calculation would cost.

## What is implemented

Phases 1, 2, 4 and 5 complete; most of Phase 3. See
[docs/ROADMAP.md](docs/ROADMAP.md) for what is deliberately not built and why.

Phase 5's deliverable is a measurement, not a coordinator: the adaptive policy
is benchmarked against the fixed pipeline at equal wall-clock, and currently
**does not** earn its complexity, which the benchmark reports plainly.

Search iterates: retrieval seeds a population, evolution mutates and
recombines it, Bayesian optimisation tunes the continuous composition of the
blends it finds, and scores feed back each round. Physics validation then spends
a bounded budget on the few candidate-property pairs where uncertainty could
still reorder the ranking, and refuses any property quantum chemistry or
molecular dynamics cannot legitimately produce.

The expert panel spans six families, reusing published open implementations
where they exist:

| Expert | Family | Class | Predicts |
|---|---|---|---|
| `joback` | thermal | molecule | boiling and melting point, critical constants, enthalpies, ideal-gas heat capacity |
| `measured` | thermal | molecule | compiled experimental boiling point, melting point, molar mass |
| `crippen` | chemical | molecule | partition coefficient, molar refractivity |
| `esol` | chemical | molecule | aqueous solubility |
| `interfacial` | interfacial | molecule | surface tension, liquid density, molar volume, Hildebrand parameter |
| `hansen` | interfacial | molecule | Hansen dispersion, polar and hydrogen-bonding components |
| `feasibility` | feasibility | molecule | synthetic accessibility with structural alerts |
| `structural` | structural | molecule | exact graph descriptors |
| `mixture` | interfacial | mixture | density and Hansen parameters by mixing rules, plus compatibility distance |
| `polymer_tg` | thermal | polymer | glass transition from an additive molar function over repeat-unit groups |
| `polymer_density` | mechanical | polymer | amorphous density from van der Waals volume and a fitted packing factor |

The two polymer experts are the only ones whose coefficients are fitted in this
repository rather than published elsewhere, so they carry their own validation
set: ten polymers withheld from the fit and from the choice of descriptors,
against which the glass transition comes out at 22 K RMSE and the density at
3.7%. In-sample agreement is not reported as evidence, because a fitted model
reproduces what it was fitted to by construction.

`formulate calibrate` measures both accuracy and whether the stated uncertainty
is honest — a model with a 3 K error claiming 1 K is more dangerous to a
ranking than one with a 15 K error claiming 15 K. Against the bundled reference
compounds all four calibrated properties currently report one-sigma coverage
between 59% and 84%, against the ~68% a correct estimate implies.

## What this does not establish

These are in the specification's non-goals, and they are printed on every run
report next to the numbers they qualify:

- Requested properties do not uniquely determine a structure.
- A high model score is not experimental proof, and not evidence that a
  compound can be synthesised.
- Bulk, formulation and processing behaviour is not established by
  single-molecule correlations.
- No quantum or molecular-dynamics validation runs in this phase, so no result
  yet carries physics evidence.

Electrical experts, and the mechanical properties beyond density, are
deliberately absent rather than stubbed. `formulate experts` reports them as
uncovered for the material classes they would apply to.

## Development

```bash
python -m pytest          # 416 tests
python -m pytest --cov=formulate
```
