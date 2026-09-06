# Behavior-Driven Inverse Materials Design

Compact engineering specification for an engine that maps desired material
behavior to candidate molecules, polymers, blends and formulations. This is the
source requirement document the codebase is built against; section numbers
referenced in code comments point here.

**Input** — desired properties, constraints, operating conditions, acceptable trade-offs.
**Core loop** — explore candidates → expert prediction → multi-objective ranking → selective physics validation → iterate.
**Output** — ranked material recipes with predicted properties, uncertainty, validation evidence, and constraint violations.

## 1. System Objective

Given a target specification, search chemical/material design space and return
feasible candidates rather than attempting to invert a single property model.
The system must support single molecules first, then polymers and
multicomponent formulations without changing the high-level pipeline.

- Primary objective: maximize satisfaction of multiple requested properties under hard constraints.
- Secondary objectives: diversity, synthesizability, stability, uncertainty reduction, and computational cost.
- Failure behavior: identify incompatible constraints and estimate which constraint relaxations create feasible regions.
- Research baseline: fixed generation + normalized expert scores. Advanced system adds adaptive compute allocation and active learning.

## 2. Shared Material Representation

Every module reads/writes one canonical candidate object. Representation is
class-dependent but exposes a common interface.

- Identity: candidate ID, material class, provenance, parent candidates, generation strategy.
- Small molecule: graph/SMILES, stereochemistry, charge, conformer set.
- Polymer: monomer(s), repeat unit, topology/architecture, composition, molecular-weight distribution targets, tacticity, crosslink specification.
- Mixture/formulation: components, roles, mole/mass/volume fractions, phase assumptions, additives, polymer components where applicable.
- Conditions: temperature, pressure, environment, surfaces/interfaces, concentration, processing/cure history.
- Results: expert predictions, uncertainty, applicability-domain flags, simulation results, objective vector, constraint violations, aggregate rank.

## 3. Exploration Module

**Purpose** — generate a diverse candidate pool with enough coverage that the
evaluator is not forced to rank only locally similar or poor candidates.
Exploration and evaluation remain separate.

**Search portfolio**
- Database retrieval: retrieve known molecules/materials near target property regions or structural motifs.
- Evolutionary search: mutate/crossover valid structures or recipes; retain Pareto-competitive and diverse parents.
- Generative search: optional learned generator for novel structures/recipes; never treated as a validator.
- Optimization: Bayesian/black-box optimization over continuous recipe variables such as ratios, molecular weight, crosslink density, and process conditions.
- Diversity: reserve exploration budget for structurally/chemically distinct regions; prevent one family from monopolizing the population.

**Hierarchical generation**
- Choose/maintain candidate material classes: molecule, polymer, blend/formulation, later composite.
- Generate class-specific structure first, then composition/architecture, then processing and operating conditions.
- For mixtures, enforce fraction sums, component compatibility rules, allowed component count, concentration bounds, and chemically meaningful roles.
- For polymers, mutations may alter monomer identity, comonomer fraction, topology, chain-length target, tacticity, crosslinker, or cure/process variables.
- Use validity filters before expert inference: valence, charge, duplicate detection, forbidden chemistry, composition bounds, basic stability/synthesis heuristics.

**Iteration** — generate batch → cheap validity filters → expert scoring →
selection/Pareto archive → update search populations/models → repeat until
budget, convergence, or target satisfaction. Multiple search strategies operate
in parallel and merge into one deduplicated candidate pool.

## 4. Mixture-of-Experts Evaluation

**Role** — experts do not independently choose their favorite materials. All
applicable experts score the same candidates so trade-offs are comparable.

**Expert families**
- Thermal/phase: melting/transition behavior, thermal stability, heat-related properties.
- Mechanical: modulus, strength-related proxies, toughness/elastic response where trained data support them.
- Interfacial: adhesion/interface affinity, surface-related behavior.
- Electrical/electronic: conductivity-related or electronic properties; route first-principles quantities to QM when required.
- Chemical/stability: reactivity/stability, solubility/compatibility, degradation-related predictors.
- Specialized learned properties: biological/sensory properties such as odor where first-principles prediction is not sufficient.
- Feasibility: synthesis/manufacturing heuristics, data-domain detection, and optional cost/environmental constraints.

**Model policy**
- Reuse open pretrained predictors when scientifically adequate; fine-tune where domain data permit; train new niche experts only when coverage is missing.
- Experts may be GNNs, 3D equivariant networks, descriptor models, empirical correlations, or deterministic physics models; the interface is model-agnostic.
- Every prediction returns value/distribution, units, uncertainty/confidence, applicability-domain score, model/version, conditions, and provenance.
- Never treat normalized scores as physical probabilities unless the underlying model is calibrated as such.

**Scoring** — convert requested properties into dimensionless utility/constraint
functions. Preserve raw physical predictions. Combine utilities with explicit
weights only for a scalar baseline; maintain a Pareto frontier for conflicting
objectives. Hard constraints eliminate or heavily penalize candidates.
Missing/uncertain predictions must not silently become neutral scores.

## 5. Coordination Module

**Baseline** — the MVP coordinator is deterministic: dispatch candidates to
applicable experts, validate units/conditions, normalize objectives, enforce
constraints, deduplicate results, compute Pareto/scalar rankings, and control
iteration budgets.

**Advanced adaptive coordinator**
- Choose the next action rather than merely the next candidate: generate more, diversify, call another expert, increase model fidelity, run QM/MD, or stop.
- Allocate expensive compute where expected information gain or ranking impact is high.
- Escalate candidates with high utility but high uncertainty; avoid expensive validation of clearly dominated candidates.
- Track disagreement between experts/physics and trigger re-evaluation or model-domain warnings.
- Use an LLM/reasoning layer only for translating vague user intent, decomposing niche requests, tool routing explanations, and human-readable rationale; keep scientific ranking and constraints reproducible.

**Target translation** — natural-language goals must become explicit properties,
units, conditions, objective direction, tolerance, priority, and hard/soft
status. Ambiguous targets require assumptions to be surfaced rather than hidden.

## 6. Molecular Dynamics Module

**Purpose** — validate or estimate properties arising from finite-temperature
structure, motion, interfaces, and bulk behavior. MD is not expected to provide
electronic/sensory properties.

**Initial implementation**
- Use mature classical MD backends for production validation; a minimal in-house engine may be built for learning, not as the production scientific authority.
- Inputs: atomic topology/coordinates, force field or potential, boundary conditions, ensemble, temperature/pressure, timestep, equilibration and sampling plans.
- Core integration: compute forces → integrate nuclear motion → apply ensemble controls/boundaries → sample observables.
- Candidate preparation must construct representative polymer chains, mixture boxes, interfaces/surfaces, or condensed phases; a single isolated molecule is insufficient for many bulk properties.

**Property workflows**
- Adhesion: construct interface; obtain interfacial/free-energy or work-of-separation metrics under defined surfaces and conditions.
- Phase/thermal: sample across temperature/pressure and detect structural/thermodynamic transitions; finite-size/time uncertainty must be reported.
- Mechanical: equilibrate bulk model; apply controlled deformation/stress protocol and derive response metrics.
- Transport/dynamics: derive diffusion/viscosity-like quantities from appropriate trajectory observables where model validity permits.

**Reactive/high-fidelity** — classical fixed-topology MD cannot generally
describe bond breaking/forming. Use reactive potentials, ML interatomic
potentials, or ab initio MD when chemistry changes are essential.

## 7. Quantum-Mechanical Module

**Purpose** — provide electronic structure, energetics, optimized geometries,
reaction/interaction energetics, and forces for high-fidelity validation or
training data.

- Production backends: integrate established open quantum-chemistry/DFT packages rather than reimplementing production DFT.
- Learning backend: optional minimal Hartree-Fock implementation to expose basis functions, SCF, energies, density, gradients, and the origin of nuclear forces.
- Inputs: geometry, composition/charge/spin, method, basis/pseudopotential, convergence criteria, boundary/periodic settings.
- Outputs: total energy, optimized geometry, forces/gradients, electronic observables, convergence diagnostics, method metadata, uncertainty/known limitations.

**Fidelity ladder** — do not assume Hartree-Fock is a universally cheaper
substitute for DFT or that one method is uniformly more accurate. Select method
by system/property. Use lower-cost electronic methods for screening where
validated, then escalate only candidates whose ranking depends on higher fidelity.

## 8. Hybrid QM/MD and Multi-Fidelity Loop

Ab initio MD evaluates electronic structure at nuclear configurations to obtain
forces, then advances nuclei. It is expensive because forces must remain
consistent with changing geometry. The platform should support multi-fidelity
alternatives without pretending intermittent QM forces are automatically valid.

- Preferred scalable path: train/use an ML interatomic potential on QM data, run fast MD with it, and query QM when uncertainty is high (active learning).
- Alternative: QM/MM for localized chemically active regions embedded in a cheaper environment.
- A fixed "QM every N timesteps" scheme is an experimental approximation and requires a mathematically defined force propagation/interpolation method plus error validation; it is not the default.
- Validated QM/MD results feed the evidence store and can become new training data for predictors/potentials.

## 9. End-to-End Workflow

| # | Stage | Specification |
|---|---|---|
| 1 | Specify | User provides desired behavior/properties, conditions, tolerances, priorities, and constraints. |
| 2 | Formalize | Translate goals into machine-readable objectives and determine relevant material classes and expert coverage. |
| 3 | Explore | Parallel database, evolutionary, optimization, and optional generative strategies propose candidates/recipes. |
| 4 | Filter | Apply structural validity, composition, duplicate, feasibility, and cheap-domain filters. |
| 5 | Predict | Applicable experts score the same candidates and return raw values plus uncertainty. |
| 6 | Rank | Compute constraint satisfaction, normalized utilities, Pareto frontier, diversity, and scalar baseline if requested. |
| 7 | Iterate | Use rankings to update exploration; adaptive coordinator may request more information or fidelity. |
| 8 | Validate | Run QM/MD only on a small high-value/uncertain subset using property-specific protocols. |
| 9 | Re-rank | Replace/augment predictions with validated results; quantify disagreement and confidence. |
| 10 | Return | Report top recipes, conditions, evidence, uncertainties, trade-offs, and infeasible constraints. |

## 10. MVP Sequence

- **Phase 1 — evaluator**: 3-5 reliable expert models, known single-molecule dataset, canonical units/conditions, uncertainty, normalized utilities, Pareto/scalar ranking. Verify rankings against held-out known materials.
- **Phase 2 — inverse search**: add database retrieval + evolutionary generator; iterate generation/evaluation; measure hit rate, diversity, sample efficiency, and failure modes.
- **Phase 3 — physics**: add selective QM and classical MD validation for properties each method can legitimately test; compare expert predictions with physics/experiment.
- **Phase 4 — polymers/formulations**: extend candidate schema, generators, preparation, and experts to composition, architecture, chain statistics, interfaces, and processing.
- **Phase 5 — adaptive coordination**: compare fixed pipeline against uncertainty/expected-value-driven compute allocation. The coordinator earns complexity only if it improves quality per unit compute.

## 11. Core Interfaces and Invariants

- Units: canonical unit registry; no ranking before dimensional validation and condition matching.
- Provenance: every generated structure, prediction, simulation, transformation, and score is versioned and reproducible.
- Uncertainty: first-class field throughout; distinguish aleatoric/model uncertainty, applicability-domain warnings, and simulation sampling error where available.
- Conditions: predictions are invalid outside stated temperature/pressure/environment unless explicitly modeled.
- Asynchrony: experts and physics jobs run independently/parallel; coordinator operates on partial results without corrupting rank semantics.
- Caching: content-address candidates + method + conditions so identical calculations are never repeated.
- Scientific separation: generator proposes; experts predict; ranker optimizes; QM/MD validate; reasoning layer explains/routes. No module silently substitutes for another.

## 12. Success Metrics

- Property prediction error/calibration on held-out materials.
- Inverse-design success rate: fraction of generated candidates satisfying all hard constraints after validation.
- Hypervolume/Pareto improvement and chemical diversity per evaluation budget.
- Top-k recall against benchmark candidate sets when a known solution exists.
- Compute efficiency: validated quality per GPU/CPU hour and number of QM/MD calls avoided.
- Physics disagreement rate and ability to identify out-of-domain expert predictions.
- For formulations: validity of recipes, phase/compatibility failures, sensitivity to composition/process variables.

## 13. Explicit Non-Goals / Scientific Guardrails

- Do not claim arbitrary requested properties uniquely determine a chemical structure; inverse design is many-to-many and may be infeasible.
- Do not claim MD + QM derive every material property. Sensory/biological, synthesis, manufacturing, aging, and many macroscale behaviors require data-driven or higher-scale models/experiments.
- Do not validate bulk polymer/formulation behavior from an isolated molecule alone.
- Do not treat a high ML score as experimental proof or synthesizability.
- Do not build custom production DFT/MD solvers unless solver development itself becomes the research question.
