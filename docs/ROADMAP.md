# Roadmap

Build order follows section 10 of the engineering specification. Each phase is
gated on the previous one being measurably good, not merely present.

## Phase 1 — the evaluator ✅

*"3-5 reliable expert models, known single-molecule dataset, canonical units
and conditions, uncertainty, normalized utilities, Pareto/scalar ranking.
Verify rankings against held-out known materials."*

Complete. Six experts across four families; canonical unit registry with
dimensional validation; first-class uncertainty distinguishing model error from
sampling error; Derringer-Suich utilities with risk adjustment; feasibility,
Pareto frontier, hypervolume and structural diversity; content-addressed
caching and provenance; infeasibility diagnosis; an evidence store; a CLI.

Verification is `formulate calibrate`, which reports accuracy *and* uncertainty
calibration against the bundled reference compounds.

## Phase 2 — inverse search ◐

*"Add database retrieval + evolutionary generator; iterate generation and
evaluation; measure hit rate, diversity, sample efficiency, and failure modes."*

**Done.** Database retrieval, an evolutionary explorer (mutation, BRICS
crossover, Pareto-archive parent selection, scaffold-capped diversity, growth
control), the iterate loop that feeds scores back to the explorers, and the
section 12 metrics: inverse-design success rate over *generated* candidates,
hypervolume per evaluation, sample efficiency, diversity per round, top-k
recall, and failure modes aggregated from filter rejections.

Measured on a 50-candidate seed pool, hypervolume rises 0.1405 → 0.2112 over
five rounds and reaches 90% of its final value in 80 of 140 evaluations.

**Not done.**

- **Bayesian optimisation over continuous recipe variables.** Designed in
  detail (numpy/scipy Matern-5/2 GP, ParEGO scalarisation, stick-breaking
  simplex transform) but not implemented. Its value is currently limited
  anyway: no registered expert covers a mixture, so there is nothing to
  optimise a recipe *against* until Phase 4.
- **A generative explorer.** Optional in section 3, and section 3 requires it
  never be treated as a validator.

## Phase 3 — physics ◐

*"Add selective QM and classical MD validation for properties each method can
legitimately test; compare expert predictions with physics/experiment."*

**Done.**

- **Quantum** (`formulate.physics.qm`): PySCF for Hartree-Fock and DFT with
  analytic gradients, dipole and frontier gap; GFN2-xTB as the screening rung.
  Geometry optimisation runs through an ASE optimiser over a hand-written
  PySCF calculator, because `pyscf.geomopt` needs a solver that cannot be
  installed here. Validated against a literature structure: a distorted water
  relaxes to O–H 0.9894 Å and 100.03°, against HF/STO-3G values of 0.989 Å and
  100.0°.
- **Dynamics** (`formulate.physics.md`): ASE integrators over pluggable
  calculators (MMFF94, GFN-FF, GFN1/GFN2-xTB), density-sized cluster
  construction, equilibration detection, and block-averaged sampling error.
- **Selective validation** (`formulate.coordination.validation`): chooses the
  few candidate-property pairs where uncertainty could still reorder the
  ranking, refuses any property physics cannot legitimately produce, combines
  a validated value with the expert prediction by inverse variance, quantifies
  disagreement as a z-score, re-scores and re-ranks.

**What this installation cannot do, and says so.** Every bulk protocol in
section 6 needs a periodic condensed phase. The only periodic-capable
potential here (GFN1-xTB) fails above roughly seventy atoms, and OpenMM cannot
parameterise an arbitrary small molecule without openff-toolkit. So density,
self-diffusion and work of separation are **refused**, each with the system
size and sampling time the observable needs and an estimate of what an
adequate run would cost. Run `formulate physics` to see the capability report.

**Not done.**

- **The section 8 multi-fidelity layer**: ML interatomic potential with
  active-learning escalation to QM, and QM/MM embedding. torch is unavailable,
  and a committee-of-cheap-regressors stand-in would be an uncertainty signal
  rather than a usable potential; building it as though it were the real thing
  would misrepresent what it is.
- **The optional minimal Hartree-Fock teaching backend** of section 7.

## Phase 4 — polymers and formulations ◐

*"Extend candidate schema, generators, preparation, and experts to
composition, architecture, chain statistics, interfaces, and processing."*

The schema and the generators were already there from earlier phases: the
candidate model describes polymers and mixtures, and the evolutionary explorer
mutates both. The gap was that **no expert covered either**. A mixture had five
properties, all graph descriptors; a polymer had none.

**Done — formulations.**

- **Hansen solubility parameters** from a curated published compilation, not a
  correlation. A single Hildebrand parameter cannot answer a compatibility
  question: ethanol and nitromethane share a cohesive energy density and are
  poor substitutes, because one holds it in hydrogen bonds and the other in
  dipolar interactions. Structures resolve through their InChIKey, because a
  SMILES lookup fails on benzene, toluene, acetone and DMSO.
- **Mixture expert** evaluating each component through the molecular panel and
  applying mixing rules to what comes back, so a component's density is the
  same number it would have had as a candidate in its own right. Volume
  fractions are formed explicitly, since a mass fraction used where a volume
  fraction belongs is a silent error of tens of percent.
- **Compatibility** via the Hansen distance between the least compatible pair.
  No relative energy difference is reported, because that needs a measured
  interaction radius which is defined per polymer and not for a solvent pair.
- **Measured pure-component lookups** for boiling and melting point. This needs
  no precedence rule: the engine already prefers the in-domain prediction with
  the tighter spread, so a measurement wins because it is better. Panel
  accuracy on the reference set improved from 13.7 to 0.2 degrees on boiling
  point and 25.6 to 0.9 on melting point.

Mixture coverage went from 5 properties to 10.

**Not done — polymers.** `glass_transition_temperature`, polymer density and
polymer Hansen parameters remain uncovered, and the registry reports them as
uncovered rather than returning a single-molecule number in their place. These
need van Krevelen and Hoftyzer-Van Krevelen group-contribution tables, and a
wrong group value produces a plausible result that is wrong by a constant —
the hardest kind of error to notice and one no test of the code's logic would
catch. Shipping tables recalled rather than verified would be the wrong trade.

## Phase 5 — adaptive coordination ✅ (with a negative result)

*"Compare fixed pipeline against uncertainty/expected-value-driven compute
allocation. The coordinator earns complexity only if it improves quality per
unit compute."*

Both halves are built: the coordinator, and the benchmark that judges it.

**The coordinator** scores every action — generate, diversify, validate, stop —
by the ranking movement it is expected to buy divided by what it costs, and
takes the best ratio. The expected-value model is built from what the run
measures (observed hypervolume gain per candidate, observed cost of a physics
call) rather than from constants chosen to make the policy look good. No
scientific decision routes through a language model; given a seed it is
deterministic.

Three defects were found by running it rather than by reading it:

- A plain mean over observations let one unlucky zero disqualify an action
  permanently. The first version spent its whole budget on validations that
  returned nothing. Estimates are now shrunk toward a prior.
- Hypervolume is a degenerate progress signal while any objective is
  uncovered: an identically zero axis gives zero volume however good the rest
  is, so the policy could not see that acquiring the missing axis was the only
  thing worth doing. Coverage now dominates until everything is measured.
- Generation destroyed validated evidence, because re-evaluation rebuilds each
  candidate's results from expert predictions alone. Physics results are now
  held outside the candidate and restored.

**The benchmark** holds wall-clock equal — the only currency both arms spend
from the same purse — refuses targets whose utility axes come from the pool
rather than the request, gives each arm its own cache so neither free-rides on
the other, and compares paired runs with an exact sign test.

**The measured answer is that adaptive does not currently earn its
complexity.** On a three-seed paired benchmark, both arms reached identical
hypervolume (0.4796) on every seed, while the adaptive arm evaluated 138
candidates to the fixed arm's 38 and spent roughly three times the wall clock.
The benchmark reports `KEEP THE FIXED PIPELINE`, which section 10 treats as
the expected outcome rather than a failure.

That result also exposes a limitation of the benchmark itself: this target
saturates, so both arms max out and every pair ties. A discriminating target
suite — including one where the frontier is genuinely hard to reach — is what
would let the comparison say something stronger than "no difference here".
