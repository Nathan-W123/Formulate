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

## Phase 4 — polymers and formulations

The candidate schema already covers polymers (monomers, topology, tacticity,
chain-length targets, crosslink density) and mixtures (components, roles,
fraction basis, phase assumptions), and the evolutionary explorer mutates
both. What is missing is the experts: no registered expert covers a polymer or
a mixture, and `formulate` reports those properties as uncovered rather than
returning a single-molecule number in their place.

## Phase 5 — adaptive coordination

Not started, by design. The deterministic and iterative coordinators are the
baseline that comparison needs. An adaptive coordinator would choose the next
*action* — generate more, diversify, call another expert, raise fidelity, run
physics, stop — by expected information gain, and would have to demonstrably
beat the baseline per unit compute before replacing it.

The validation stage already contains the first piece of that machinery: its
selection policy scores candidate-property pairs by how much a validated value
could move the ranking, which is expected-information-gain reasoning applied
to one decision.
