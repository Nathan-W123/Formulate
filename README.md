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

**Physics refuses what it cannot support, and the refusals are audited.** A
bulk property needs a periodic condensed phase, which this installation now
reaches through OPLS-AA and OpenMM: density, cohesive energy, self-diffusion,
surface tension and shear viscosity all run. The last two arrived late, and the
reasons they had been refused for turned out to be wrong — both are components
of a pressure tensor, and what was missing was not system size or physics but
an accessor, since OpenMM publishes no pressure tensor at all. Recovering it by
finite difference retired both entries at once, which is why a stated refusal
is worth writing down precisely: a vague one cannot be found to be false.

What is still refused says what it would take. A normal boiling point needs a
saturated vapour, and the slab that measures a surface tension is a
liquid-vapour coexistence — but at 298 K a compound of ordinary volatility puts
well under one molecule in the vacuum gap, so the refusal is now a counted
number rather than an assertion. Rather than returning a value from a box too
small to hold the phenomenon — the error would be systematic, so an error bar
would not rescue it — the run refuses and estimates what an adequate
calculation would cost.

## What is implemented

All five phases are implemented. See [docs/ROADMAP.md](docs/ROADMAP.md) for
what is deliberately not built and why, and [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
for every number this repository has measured about itself, including the ones
that came out badly.

Phase 5's deliverable is a measurement, not a coordinator: the adaptive policy
is benchmarked against the fixed pipeline and currently **does not** earn its
complexity. On the most recent run it won five of five paired seeds at p = 0.06
— and made 2.7 times as many expert evaluations to do it, which is a bigger
budget rather than a better policy. The benchmark now weighs what each arm
spent and says so.

Search iterates: retrieval seeds a population, evolution mutates and
recombines it, Bayesian optimisation tunes the continuous composition of the
blends it finds, and scores feed back each round. Physics validation then spends
a bounded budget on the few candidate-property pairs where uncertainty could
still reorder the ranking, and refuses any property quantum chemistry or
molecular dynamics cannot legitimately produce.

The expert panel spans eight families, reusing published open implementations
where they exist:

| Expert | Family | Class | Predicts |
|---|---|---|---|
| `joback` | thermal | molecule | boiling and melting point, critical constants, enthalpies, ideal-gas heat capacity |
| `measured` | thermal | molecule | compiled experimental boiling point, melting point, molar mass, surface tension |
| `crippen` | chemical | molecule | partition coefficient, molar refractivity |
| `esol` | chemical | molecule | aqueous solubility |
| `interfacial` | interfacial | molecule | surface tension, liquid density, molar volume, Hildebrand parameter |
| `hansen` | interfacial | molecule | Hansen dispersion, polar and hydrogen-bonding components |
| `feasibility` | feasibility | molecule | synthetic accessibility with structural alerts |
| `structural` | structural | molecule | exact graph descriptors |
| `mixture` | interfacial | mixture | density and Hansen parameters by mixing rules, plus compatibility distance |
| `unifac` | chemical | mixture | excess Gibbs energy and phase stability from modified UNIFAC |
| `adhesion` | interfacial | molecule | work of separation on a named substrate, Owens-Wendt |
| `polymer_tg` | thermal | polymer | glass transition from an additive molar function over repeat-unit groups |
| `polymer_density` | mechanical | polymer | amorphous density from van der Waals volume and a fitted packing factor |
| `polymer_mechanical` | mechanical | polymer | Young's and shear modulus, entanglement molar mass, flaw-free strength bound |
| `critical_atomic` | thermal | molecule | critical constants from atom counts, where group contribution has no groups |
| `learned_boiling_point` | thermal | molecule | fitted boiling point, for structures no group table covers |
| `lorentz_lorenz` | electrical | molecule | refractive index from a density and a molar refraction |
| `learned_refractive_index` | electrical | molecule | fitted refractive index, for when no density is available |
| `dissolution` | interfacial | molecule | solubility of a named solute, from Hansen distance |
| `viscosity_measured` | interfacial | molecule | tabulated liquid viscosity, DIPPR and VDI data methods only |
| `viscosity_joback` | interfacial | molecule | liquid viscosity by group contribution |
| `viscosity_corresponding_states` | interfacial | molecule | liquid viscosity from critical constants, for structures no group table covers |
| `melt_wlf` | interfacial | polymer | melt viscosity by WLF from the polymer's own glass transition |
| `trouton` | interfacial | molecule | extensional viscosity, exact for a Newtonian liquid and refused for a polymer |
| `kinetics_propagation` | chemical | molecule | propagation rate coefficient from the IUPAC pulsed-laser benchmark, seven monomers plus three crosslinkers on borrowed family values |
| `cure_free_radical` | chemical | molecule | time for a reacting liquid to stop flowing: gel point by Flory–Stockmayer for a crosslinker, half conversion for a linear polymer; needs an initiation regime named in the conditions and refuses a monomer whose homopolymer is a rubber at the cure temperature |
| `hansen_group_contribution` | interfacial | molecule | Hansen triple by Hoftyzer–Van Krevelen where the compilation has none, calibrated at 0.89 / 0.80 / 1.18 MPa^0.5; refuses on any uncovered group |
| `polymer_measured` | mechanical | polymer | tabulated glass transition, amorphous density and elongation at break, from the bundled polymer catalogue |
| `toughness_proxy` | mechanical | polymer | elongation at break by network/glassy/rubbery classification, leave-one-out to a factor of 1.40 among rubbers and 6.64 among glasses |
| `polymer_architecture` | structural | polymer | crosslink density, read off the specification and refused for a network that states none |
| `hazard_ghs` | specialized | molecule, polymer, mixture | skin sensitisation, acute toxicity and carcinogenicity, by curated lookup only; refuses any structure not in the table |

The two polymer experts are the only ones whose coefficients are fitted in this
repository rather than published elsewhere, so they carry their own validation
set: ten polymers withheld from the fit and from the choice of descriptors,
against which the glass transition comes out at 22 K RMSE and the density at
3.7%. In-sample agreement is not reported as evidence, because a fitted model
reproduces what it was fitted to by construction.

`hazard_ghs` is the one expert in the panel that never estimates. Every value is
a lookup in a curated GHS table and a structure outside it is refused by name,
because the structural alerts for skin sensitisation are wrong often enough in
both directions to matter and a false negative there is somebody's hands. It
carries three endpoints rather than two: benzene is not a sensitiser and is not
acutely toxic at any category, so a screen without carcinogenicity would return
two clean numbers for a category 1A carcinogen. What the table does not carry -
reproductive and target-organ toxicity, mutagenicity, aspiration, flammability,
environmental hazard, every exposure limit - is named in each prediction's own
notes, because the screen can screen out and cannot screen in.

The two refractive index experts are a worked example of section 4's design:
neither has precedence over the other anywhere in the code. Measured on the 385
reference compounds that carry a tabulated density, with the learned model
refitted without them, Lorentz–Lorenz is at 0.0123 mean absolute error and the
forest at 0.0165 — but the same equation fed an *estimated* density goes to
0.198, because it has a pole. So the physics expert propagates its density's
uncertainty, states ±0.015 on a tabulated density and ±0.05 on an estimate, and
`prefer()` picks on the tightest stated spread.

`formulate calibrate` measures both accuracy and whether the stated uncertainty
is honest — a model with a 3 K error claiming 1 K is more dangerous to a
ranking than one with a 15 K error claiming 15 K. It also refuses to flatter
itself: for the five properties where a compiled measurement can answer, the
default run was comparing that lookup against the table it was compiled from,
reporting a critical temperature accurate to exactly zero. Those rows are now
labelled *not an accuracy measurement*, and `--expert joback` measures the
estimator instead — 14.8 °C on the boiling point, 27.8 K on the critical
temperature, with one-sigma coverage between 59% and 80% against the ~68% a
correct estimate implies.

## What this does not establish

These are in the specification's non-goals, and they are printed on every run
report next to the numbers they qualify:

- Requested properties do not uniquely determine a structure.
- A high model score is not experimental proof, and not evidence that a
  compound can be synthesised.
- Bulk, formulation and processing behaviour is not established by
  single-molecule correlations.
- Physics validation is opt-in and bounded. A default run's budget does not
  buy a condensed-phase simulation, and the report says so in those words
  rather than leaving the absence to be inferred.
- Inverse design here reaches a near neighbour rather than the answer. Hiding
  four known solvents and describing them only by their measured properties,
  the search recovered one of the four; what it returned instead was
  chemically sensible in every case.

A degree of crystallinity is deliberately absent and is the largest thing
missing from the polymer side. It is measurable and the panel cannot use it:
high-density polyethylene at 25 C is above its glass transition, so the
mechanical expert computes a rubber-elastic modulus for a material that carries
its load in crystals, and the resulting flaw-free strength bound comes out 39
times *below* the measured tensile strength. Alongside it, no polymer melting
point: a semicrystalline hot melt is useful up to its melting point and can only
be asked for here on its glass transition, which understates it by a hundred
kelvin and more.

The mechanical properties beyond density are deliberately absent rather than
stubbed, and so is a dielectric constant: a model for it was trained on 1212
measured liquids and rejected, because its confidence gate admitted only
nonpolar molecules and so answered the question nobody needed answering.
`formulate experts` reports uncovered properties for the material classes they
would apply to.

## Development

```bash
python -m pytest -m "not slow"    # the fast suite
python -m pytest                  # everything, including real MD and QM
python -m pytest --cov=formulate
```

`bench/` holds the scripts behind every number in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md); each one is runnable on its own and
prints what it measured.

`formulate.processing` is separate from the experts and from the physics, and
deliberately so. A jet's fate depends on the nozzle and the velocity as much as
on the liquid, so nothing in it is a property: it does not enter the registry,
does not reach the ranking, and answers questions about what happens to a
material when it is made into something. It currently covers jet break-up,
drying, and setting by reaction — including the exotherm that comes with it.
