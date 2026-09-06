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

## Phase 2 — inverse search

*"Add database retrieval + evolutionary generator; iterate generation and
evaluation; measure hit rate, diversity, sample efficiency, and failure modes."*

Database retrieval exists over a bundled reference set, and the `Explorer`
interface and validity filters are in place. Still to build:

- An evolutionary explorer: mutation and crossover over valid structures, with
  a Pareto archive retaining diverse parents.
- Bayesian optimisation over continuous recipe variables.
- The iterate loop in the coordinator: generate, filter, score, select, update,
  repeat until budget or convergence.
- Metrics: hit rate, sample efficiency, hypervolume per evaluation.
- A generative explorer, which section 3 requires never be treated as a
  validator.

## Phase 3 — physics

*"Add selective QM and classical MD validation for properties each method can
legitimately test; compare expert predictions with physics/experiment."*

Not started. Section 7 is explicit that production DFT and MD solvers must not
be reimplemented; the work is integration with established open packages, plus
candidate preparation — building representative chains, mixture boxes and
interfaces, since section 13 forbids validating bulk behaviour from an isolated
molecule. Validated results feed the evidence store.

## Phase 4 — polymers and formulations

*"Extend candidate schema, generators, preparation, and experts to composition,
architecture, chain statistics, interfaces, and processing."*

The candidate schema already covers polymers (monomers, topology, tacticity,
chain-length targets, crosslink density) and mixtures (components, roles,
fraction basis, phase assumptions), with fraction sums and duplicate components
validated. What is missing is the experts: no registered expert currently
covers a polymer, and `formulate` says so rather than returning a
single-molecule number in its place.

## Phase 5 — adaptive coordination

*"Compare fixed pipeline against uncertainty/expected-value-driven compute
allocation. The coordinator earns complexity only if it improves quality per
unit compute."*

Not started, by design. The deterministic coordinator is the baseline that
comparison needs. An adaptive coordinator would choose the next *action* —
generate more, diversify, call another expert, raise fidelity, run physics,
stop — by expected information gain, and would have to demonstrably beat the
baseline per unit compute before replacing it.
