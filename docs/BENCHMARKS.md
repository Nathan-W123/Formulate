# Measured performance

Numbers here were produced by the scripts in `bench/`, on the machine that ran
them, and are quoted with the wall clock that bought them. Nothing in this file
is cited from a paper; where a published figure would have been convenient, it
was measured instead, because a published figure describes the authors'
hardware and the authors' molecules.

Reproduce with:

```
python bench/conformer_energies.py     # the fast rung against the quantum rung
python bench/atomization_ladder.py     # both rungs against experiment
```

Both need `formulate[potentials]` (torch, mace-torch) and PySCF.

---

## Multi-fidelity: MACE-OFF23-small against B3LYP/6-31G

### Conformer energy differences

Absolute energies from a neural potential and from a quantum calculation are
not on the same scale and never will be — MACE learns a shifted, size-extensive
surface — so what is compared is differences, which is also what a ranking
consumes. Conformers are embedded with ETKDG, optimised with MMFF94 and pruned
to distinct geometries by heavy-atom RMSD above 0.5 Å.

| molecule | conformers | mean \|ΔE error\| | DFT spread | Kendall τ | mean \|force error\| |
|---|---|---|---|---|---|
| butane | 3 | 3.3 meV | 27.0 meV | +0.33 | 0.109 eV/Å |
| pentane | 4 | 7.8 meV | 53.3 meV | +1.00 | 0.111 eV/Å |
| 1-propanol | 3 | 0.3 meV | 5.6 meV | +0.33 | 0.206 eV/Å |
| ethylene glycol | 4 | 35.1 meV | 126.6 meV | +0.67 | 0.349 eV/Å |
| propanal | 3 | 14.0 meV | 61.4 meV | +0.33 | 0.343 eV/Å |
| 2-methoxyethanol | 4 | 20.2 meV | 189.7 meV | +0.67 | 0.299 eV/Å |
| butanenitrile | 3 | 2.0 meV | 4.7 meV | −1.00 | 0.324 eV/Å |

- **Energy error** 11.8 meV (1.14 kJ/mol) mean absolute, against a mean DFT
  conformer spread of 66.9 meV. The potential resolves the signal with about
  six times the margin.
- **Force error** 0.249 eV/Å mean absolute per component.
- **Ranking agreement** +0.33 Kendall τ over all seven. That number is
  misleading on its own: butanenitrile's entire DFT spread is 4.7 meV, less
  than half the potential's own 11.8 meV error, so its −1.00 is the ordering of
  noise rather than a disagreement about chemistry. Over the five molecules
  whose spread exceeds the measured error, τ is **+0.60**. Neither figure is
  good enough to rank conformers on without care.
- **Wall clock** 6.25 s against 185.8 s over 24 single points, a **30×**
  reduction, forces included on both sides. A second run of the same script on
  a machine also running the test suite gave 17.0 s against 270.3 s — 16× — with
  every energy, force and τ identical. The speedup is contention-sensitive and
  the accuracy is not; treat 16–30× as the range on four cores rather than 30×
  as the number.

Two bugs were fixed before these numbers meant anything. A first pass reported
0.1 meV agreement on ethanol and methanol: RDKit had returned the same geometry
three times and both methods correctly said the difference was zero. A second
reported a 10 eV/Å force error, which was the backend's forces — already in
eV/Å — being converted from Hartree/Bohr a second time, a factor of 51.4.

### Atomization energy, and the rung that turned out to be worse

Section 8 describes a ladder and assumes the expensive rung is the accurate
one. That is testable here, so it was tested. Both columns are electronic well
depths: molecule minus ground-state free atoms, no zero-point correction on
either side. Reference values are D₀ + ZPE, both commonly tabulated handbook
numbers rather than a curated benchmark — the same standing this repository
gives its other reference compounds.

| molecule | D_e (exp) | MACE | error | B3LYP/6-31G | error |
|---|---|---|---|---|---|
| methane | 1759.0 | 1756.3 | −2.7 | 1748.3 | −10.7 |
| water | 973.2 | 968.5 | −4.7 | 857.5 | −115.7 |
| ammonia | 1246.7 | 1252.1 | +5.4 | 1167.8 | −78.9 |
| methanol | 2172.4 | 2140.8 | −31.6 | 2032.2 | −140.2 |
| formaldehyde | 1563.4 | 1558.8 | −4.6 | 1466.4 | −97.0 |
| ethane | 3018.5 | 2979.5 | −39.0 | 2948.9 | −69.6 |
| ethene | 2358.1 | 2354.7 | −3.4 | 2319.5 | −38.6 |

All values kJ/mol.

|  | mean absolute error | bias |
|---|---|---|
| MACE-OFF23-small | **13.0** | −11.5 |
| B3LYP/6-31G | **78.7** | −78.7 |

The small basis under-binds every one of the seven, which is the textbook
failure of 6-31G on bond energies. The network was fitted to
wB97M-D3(BJ)/def2-TZVPPD, a level this pipeline cannot afford to run. So for
atomization energy the potential is not a cheap approximation to the quantum
answer — it is the better answer, six times over, at a sixth of the wall clock.

**Consequence for the code.** `_POTENTIAL_IS_THE_BETTER_RUNG` in
`formulate/coordination/validation.py` names the properties where escalation
would make the answer worse. For those, escalation fires on one trigger only:
the molecule falling outside the network's element domain, where an
extrapolating neural network is worse than an under-binding basis. The stated
one-sigma on the potential's atomization energy is 16.3 kJ/mol, from the 13.0
above under the σ = 1.253 × MAE conversion used elsewhere in this repository.

An earlier version quoted the potential's uncertainty as its spread against
B3LYP/6-31G — 73.4 kJ/mol. That number is real but it is almost entirely the
basis's error, and stating it would have made the better method look like the
worse one and hidden it behind the `prefer()` rule, which selects on the
tightest stated in-domain uncertainty.

### What was not measured

- Anything condensed-phase. MACE-OFF costs 535 ms per force evaluation on a
  576-atom box on this hardware, which puts 100 ps at fifteen hours; it is a
  single-point engine here and the module docstring says so.
- Any element outside {H, C, N, O, F, P, S, Cl, Br, I}, which the model refuses.
- Charged or open-shell species. MACE-OFF23 is fitted to neutral closed-shell
  organics and nothing here checks that a candidate is one.

---

## End-to-end inverse design: hide a known material and look for it

Four common solvents, each described only by its own measured properties —
boiling point, melting point, density, surface tension, logp — with the answer
withheld. Two arms, reported apart:

- **Retrieval** leaves the material in the catalogue. It is close to
  tautological: the material is described by its own measurements and a lookup
  expert holds exactly those measurements. It is run anyway, because failing it
  would mean the ranking is broken rather than that search is hard.
- **Discovery** removes it. Anything found there was built by an explorer, and
  the harness records which one, so a catalogue leak cannot be mistaken for a
  result.

Three rounds, batches of 20, one seed. `python bench/recover_known_materials.py`.

| material | retrieval | discovery | produced by |
|---|---|---|---|
| toluene | rank 1 of 160 | not proposed | — |
| ethanol | rank 1 of 140 | **rank 1 of 159** | `evolutionary:mutate` |
| acetone | rank 1 of 140 | not proposed | — |
| chloroform | rank 1 of 140 | not proposed | — |

**Retrieval 4 of 4. Discovery 1 of 4.** Identical at both window widths below,
so the recall is a property of the search rather than of how the target was
phrased.

Ethanol is a real recovery: the catalogue withheld it and the mutation operator
rebuilt it. It is also the easiest of the four — two heavy atoms, reachable from
methanol or 1-propanol in one edit. The three that failed were never *proposed*,
not ranked badly; what the search returned instead was chemically sensible in
every case (ethylbenzene and propyl acetate for toluene, 2-butanone and methanol
for acetone, carbon tetrachloride and dichloromethane for chloroform). Inverse
design here means "reaches a near neighbour", not "reaches the answer".

Frontier diversity 0.92–0.94 throughout. No physics ran: none of these five
properties is on the validatable list, so the ranking is the expert panel's.

### The hypervolume was collapsed, not flat

Every run reported a hypervolume gain of exactly zero, and the iterative
coordinator stops on "hypervolume gained nothing" — so every search also
stopped early and reported convergence that had not happened. Both were one
mechanism.

Desirability is risk-adjusted: a prediction is scored one standard deviation in
the unfavourable direction. Crippen puts ethanol's logp at −0.0014 against a
measured −0.31, comfortably inside a ±0.80 window, for a nominal desirability
of 0.393. Crippen's own error is about 0.8, so the pessimistic value lands
outside the window and the risk-adjusted desirability is exactly 0.000. Every
candidate scores zero on that axis; hypervolume is a product of edge lengths
from the origin, so one zero edge zeroes it whatever the other four objectives
did.

**A target window narrower than roughly two standard deviations of the method
answering it produces an objective nothing can score.** That is correct,
conservative behaviour by the desirability layer, and it is invisible in the
number it produces.

Three changes followed. `RankingResult.pinned_axes` names the axes where every
frontier point sits at zero, and the report says a zero hypervolume is a
collapsed measure rather than an unimproved one. The plateau rule abstains
instead of reading the collapse as convergence. `SearchMetrics.hypervolume_per_evaluation`
returns None rather than 0.0, because zero reads as "bought no improvement",
which is a different claim from "bought none that this measure can see".

Widening the windows checks the diagnosis rather than asserting it:

| window (fraction of each property's observed spread) | hypervolume gain | retrieval | discovery |
|---|---|---|---|
| 10% | 0 in every run, `logp` pinned | 4/4 | 1/4 |
| 35% | +0.015 to +0.13 | 4/4 | 1/4 |

The measure comes back to life and the recall does not move, which is what the
diagnosis predicts. Had the recall moved too, the windows would have been doing
more work than intended and the tighter number would be the honest one.

### What was not measured

- **Uncertainty calibration during search.** `evaluation/calibration.py` measures
  it against the reference set; nothing here measures whether the stated
  uncertainties stay honest on structures the generators invent, which is
  where they are least likely to be.
- **Polymers and mixtures end to end.** `spec_for` accepts either material
  class and the harness runs, but the reference set the answers are hidden from
  contains only molecules, so there is nothing to hide. A polymer or mixture
  recovery experiment needs a reference set of polymers or mixtures first.
- **More than one seed.** Discovery is a stochastic search and 1 of 4 on one
  seed is an observation, not a rate.

---

## QM/MM embedding

`python -m pytest tests/test_qmmm.py` — the physical checks are marked slow and
need PySCF.

The plumbing is easy to get wrong in ways that still return numbers, so the
embedding is checked against electrostatics rather than against itself. A
neutral water molecule (HF/6-31G) with a +1 point charge on its dipole axis:

| what | expected | measured |
|---|---|---|
| falls off as a charge–dipole interaction | E(8 Å)/E(16 Å) = 4 | **4.05** |
| decays toward zero with distance | → 0 | −31.8, −7.8, −1.9, −0.5 kJ/mol at 5, 10, 20, 40 Å |
| reverses sign with the charge | +1 e negative, −1 e positive | −50.4 and +46.8 kJ/mol at 4 Å |
| the residue after cancelling sign is induction | always stabilising | −3.60 kJ/mol at 4 Å, −0.71 at 6 Å |

The last row is the sharpest: induction is second order in the field, so it
survives when the first-order term cancels, and it must be negative at every
distance and shrink faster than the first-order term. It does.

A worked partition — 1-hexanol with the alcohol end quantum and the hexyl tail
classical, cutting one C–C single bond:

```
9 quantum atoms (1 of them capping), 13 classical, electrostatic embedding
MM net charge +0.0000 e
  link atom capping the bond from QM atom 4 to MM atom 3,
  placed at 0.712 of the bond length; -0.1200 e shifted off the frontier atom
QM/MM HF/sto-3g: E = -152.12894385 Hartree
electrostatic embedding contributes +1.67 kJ/mol
```

The MM net charge is zero after the shift, which is the point of shifting
rather than deleting the frontier charge.

### What is refused

Each of these otherwise returns a number, so all six raise:

| partition | why |
|---|---|
| cuts a multiple bond | one hydrogen cannot cap it, and capping it changes the bond order of the atom that keeps it |
| cuts into an aromatic system | a capping hydrogen does not restore the delocalisation |
| cuts a ring | needs two link atoms and leaves the ring open in the QM region |
| cuts a polar bond | a link atom has to imitate a charge distribution it does not reproduce |
| separates a hydrogen from its heavy atom | the link atom would cap where a hydrogen already was |
| asks for polarizable embedding | see the blocker below |

The checks run most-specific first. An earlier ordering tested bond order
before aromaticity, and since RDKit types an aromatic bond as AROMATIC rather
than SINGLE, every cut into a benzene ring was refused with "one hydrogen
cannot cap it" — true about the bond order and silent about the thing that
actually makes the partition wrong.

### The blocker

Polarizable embedding, where the MM region responds to the QM density rather
than only acting on it, is not implemented and cannot be with what is
installed. It needs MM polarizabilities — a Drude or AMOEBA-style force field —
and OPLS-AA, the only force field wired up here, is fixed-charge and has none.
This PySCF build ships no polarizable-embedding driver either.
`EmbeddingMode.POLARIZABLE` exists and refuses, naming both halves, rather than
being silently absent. `QMMMCalculator.capabilities()` reports it too, so a
caller can check before building a partition.

Gradients are also not implemented: PySCF exposes `qmmm.add_mm_charges_grad`,
but nothing here consumes QM/MM forces, and an untested gradient path would be
a capability claim with no evidence behind it.

---

## The educational Hartree–Fock backend

`python -m pytest tests/test_minimal_hf.py`

Section 7 asks for an optional minimal implementation exposing "basis
functions, SCF, energies, density, gradients, and the origin of nuclear
forces". Each of those is a visible object in `physics/qm/minimal_hf.py` rather
than a number returned by a library. It is labelled educational everywhere:
`production = False`, a warning on every result, and `backend_for()` refuses to
return a non-production backend for any method other than the one it teaches.

**Its boundary.** s-type contracted Gaussians only, which in STO-3G means
hydrogen and helium exactly and nothing else. Lithium onwards needs p
functions, which are a different piece of mathematics (the Obara–Saika
recursions) rather than a longer version of this one. Carbon is refused,
because s-only integrals for carbon would silently drop its 2p shell and return
a number that looks like an energy.

**Checked against PySCF**, since a teaching implementation that is subtly wrong
teaches the wrong thing:

| quantity | agreement with PySCF/STO-3G |
|---|---|
| H₂ at 0.74 Å | −1.1167593074 vs −1.1167593074 Ha (1.2e-12) |
| H₂ at 1.20 Å | −1.0051067066 vs −1.0051067066 Ha (1.2e-11) |
| He atom | −2.8077839575 vs −2.8077839575 Ha (1.8e-15) |
| orbital energies (H₂) | −0.57855386, +0.67114349 — exact match |
| overlap, core Hamiltonian, two-electron integrals | < 1e-7 |
| tr(PS) | 2.0000000000 |

The 1e-7 on the integrals is the stored STO-3G contraction coefficients being
the published eight-figure values rather than PySCF's internal ones; the test
suite checks the stored table against `pyscf.gto.basis.load` so a transcription
error cannot survive.

### Where nuclear forces come from

The pedagogical payload. For H₂ at 0.74 Å:

| | force on atom 0, z (Hartree/Bohr) |
|---|---|
| total (central differences) | **+0.027680** |
| PySCF analytic gradient | +0.0276796 |
| Hellmann–Feynman part | −0.050516 |
| Pulay remainder | +0.078195 |

The Hellmann–Feynman force — the classical electrostatic force on each nucleus
from the converged density and the other nuclei — is not merely inaccurate
here, it has **the wrong sign**. It is the whole force only for an exact
wavefunction in a complete basis; here the basis functions ride on the nuclei,
so moving one changes the basis itself, and what is left over is the Pulay
force. In this minimal basis the leftover is nearly three times the total.

The analytic Hellmann–Feynman term is checked against its own definition —
displace the point charge while holding the density *and* the basis-function
centres fixed — at three bond lengths, agreeing to 1e-10. That check caught a
dropped minus sign: the energy contains Σ P_ij V_ij and the force is minus its
derivative, and without the outer sign the electron density pushed the nuclei
apart instead of pulling them together, making the term 36× the total force. It
looked like a dramatic illustration of the Pulay effect. It was a bug.

### What it does not do

Geometry optimisation (declined explicitly), open-shell systems (restricted HF
doubly occupies orbitals, so an odd electron count is refused rather than
rounded), any basis but STO-3G, and any element but H and He. The total force
is a central difference rather than an analytic gradient — the analytic Pulay
term needs derivatives of every integral with respect to the basis-function
centres, which is more machinery than a module written to be read should carry,
and a finite difference demonstrates the point without asserting it.

---

## Refractive index: two routes, neither with precedence

`python bench/refractive_index_routes.py` — needs `formulate[learned]`.

The brief asks the learned layer to reach electronic and electrical properties.
Refractive index was chosen because it is one of the few where a closed-form
relation genuinely works, so the learned route has something real to beat:

    (n² − 1)/(n² + 2) = R_m / V_m

with R_m the molar refraction (Crippen's atomic contributions) and V_m the
molar volume. That makes the question "is a model better than the physics
here?" answerable rather than rhetorical.

**The comparison had to be redone once.** A first pass gave the forest 0.0140
against Lorentz–Lorenz's 0.0867 and looked decisive. It was measuring the wrong
thing: 4327 compounds for the physics route against 883 for the forest, on
different populations, with the forest's training set overlapping the physics
route's test set, and the physics route fed a Joback-then-Rackett molar volume
so that nearly all of its error was the density estimate. Redone on the 385
compounds that carry a tabulated liquid density, with the forest refitted with
all 385 removed:

| route | answers | MAE | bias | RMSE |
|---|---|---|---|---|
| Lorentz–Lorenz, measured density | 100% | **0.0123** | −0.0060 | 0.0378 |
| Lorentz–Lorenz, estimated density | 99% | 0.1979 | +0.1679 | 2.5500 |
| learned forest (never saw these) | 100% | 0.0165 | +0.0081 | **0.0342** |

The physics wins on MAE when it has a real density. The forest wins on RMSE —
fewer catastrophic outliers. And the middle row is why: the equation has a pole
at R_m = V_m, so a molar volume slightly too small sends n toward infinity, and
its RMSE is sixty-seven times its own MAE.

**Both ship, and neither has precedence written into it.** The physics expert
propagates its density's uncertainty, so it states ±0.0154 on a tabulated
density and ±0.04–0.06 on a Rackett estimate; `prefer()` selects on the
tightest stated in-domain uncertainty and picks correctly in all three regimes:

| the density is | Lorentz–Lorenz states | learned states | chosen | who was actually closer |
|---|---|---|---|---|
| tabulated (toluene) | ±0.0154, err +0.0012 | ±0.0192, err +0.0174 | physics | physics |
| Rackett-estimated (tert-butylbenzene) | ±0.0546, err −0.0033 | ±0.0073, err +0.0028 | learned | learned, just |
| unavailable (dodecyl benzoate) | abstains | ±0.0122 | learned | — |

One honest edge: on dodecane the forest states ±0.0045 against the physics
route's ±0.0154 and wins, while being slightly further out (+0.0053 against
−0.0040). The forest is a little overconfident on molecules inside its training
set. The mechanism did what it was told; it is reported rather than tuned away.

### Two defects this found

**The molar refraction error was being counted twice.** The 0.0123 above was
measured *with* Crippen molar refractions, so it already contains their error.
Propagating Crippen's own stated 2.5 cm³/mol on top made the physics expert
quote ±0.049 on an answer within 0.0012 of the measurement, and the ranking
preferred a route fourteen times further out that said so more confidently.
Only the density's uncertainty is propagated now.

**A consumer ran after the first supplier of its dependency, not after all of
them.** Both `measured` and `interfacial` supply a liquid density, and
`measured` runs early, so the refractive index consumer became ready as soon as
`measured` had run — and for any molecule `measured` had no density for, it
reported that none was available while `interfacial` was still queued behind it
and went on to produce one. `resolution_order` now waits for every expert that
could supply a dependency. This affected the existing panel too, not only the
new expert.

### The learned layer's abstraction had never been used twice

`LEARNABLE` was three positional strings, and every boiling-point-shaped
assumption in the base class turned out to be hard-coded rather than declared:
a noise floor of one kelvin, a relative-spread denominator floored at one
kelvin, and the letter K in three messages. On a refractive index of 1.36 the
noise floor alone forced a stated spread of 1.0, the gate read that as the
trees disagreeing by 73 per cent, and the expert declined every molecule it was
asked about. It is now a `Learnable` record declaring unit, noise floor, scale
floor, provenance and the measured comparison against the non-learned route.

### Relative permittivity: trained, and not shipped

1212 measured liquids, the same forest, the same gates. Ungated it is useless —
MAE 3.65 on a property whose values run 1.9 to 104, RMSE 8.59. Gated at the
usual 15% it looks excellent at MAE 0.08, and that number is an artefact:

| | admitted | refused |
|---|---|---|
| count (of 243 test compounds) | 34 | 209 |
| permittivity range | 1.89 – 6.54 | 1.94 – 104.0 |
| median | 2.16 | 7.15 |
| fraction below ε = 3 | 68% | — (21% across the whole test set) |

The admitted list is heptane, isooctane, octyl bromide, stearic acid. The model
has learned to recognise nonpolar molecules and report that they are nonpolar,
and declines every molecule whose dielectric constant is actually in doubt.
Unlike the melting point there is no group-contribution alternative, so this
leaves the property uncovered — which is the honest state rather than a bad
answer wearing a tight error bar. Recorded in `learned.py` so the experiment is
not repeated.

### Polymer glass transition: trained, and not shipped either

The brief's other named gap for the learned layer was higher-value polymer
properties. Measured rather than assumed, on the bundled reference set's ten
held-out polymers:

| route | held-out MAE | RMSE |
|---|---|---|
| group contribution (the shipped expert) | **18.5 K** | 22.1 K |
| random forest over 217 descriptors | 36.2 K | 41.8 K |
| predicting the training mean | 65.6 K | 72.9 K |

The forest is learning something — it halves the trivial baseline — and is
still twice as wrong as the expert already fitted to the same data. With 47
fitting polymers, a handful of van Krevelen group parameters is the right
amount of structure to impose and 217 free descriptors is not.

The limit here is the corpus, not the method: 57 handbook polymers is a
regression guard, and no polymer property database is installed offline. This
stays rejected until there is data to learn from.

---

## Adaptive against fixed coordination

`python bench/adaptive_vs_fixed.py` — five seeds, paired, 90 s budget each.

Section 10 and the brief both require the same thing: keep the adaptive path
only if it improves on the fixed one **at equal compute**. The benchmark was
measuring only the first half.

Two runs of the same script, on the same target:

| run | adaptive wins | mean Δ hypervolume | sign test | adaptive's evaluations |
|---|---|---|---|---|
| first | 4 of 5 | +0.0647 | p = 0.375 | 2.31× the fixed pipeline's |
| second | 5 of 5 | +0.1181 | p = 0.062 | 2.70× |

Both keep the fixed pipeline, but only the second one is interesting. At
p = 0.062 with a positive mean the old verdict rule said **ADOPT THE ADAPTIVE
COORDINATOR** — and adaptive had made 2.7 times as many expert evaluations to
get there. That is a bigger budget, not a better policy, and the benchmark
reported none of it.

`BenchmarkResult` now carries `evaluation_ratio` and the verdict refuses to
adopt on a lead bought with more than 10% extra compute. The rule still adopts
a genuine win at equal compute, which a test checks, so the guard cannot make
adoption impossible.

The gap between the two runs is itself a result: five paired seeds move a sign
test from p = 0.375 to p = 0.062 on the same code and the same target. Neither
number should be read as an estimate of anything. The decision here rests on
the compute ratio, which is stable across both runs, rather than on the p-value,
which is not.

**Verdict: the fixed pipeline stays the default.** The adaptive coordinator
remains available and is not deleted — it may well be better at equal compute,
and nothing here has shown that either way.

---

## Determinism and the language-model boundary

`python -m pytest tests/test_determinism.py`

Two claims the brief makes that stay true only while something checks them.

**Reproducibility.** The same seed gives the same ranking, the same scalar
scores and the same hypervolume, for a single-shot run and for an iterative
one; a different seed is allowed to differ, so the check is not vacuous. A
static pass over every module asserts that nothing calls `random.random()`,
`np.random.rand()` or their siblings on the global generator — every explorer
takes a seed and builds its own.

**No language model produces a number.** There is no LLM in the codebase at
all, which is the strongest form of the brief's restriction, and an AST pass
over every module keeps it that way by refusing imports of `openai`,
`anthropic`, `litellm` and `langchain`. SAFE-GPT is deliberately not caught: it
is a generative model over molecular fragments that proposes *structures* for
the expert panel to evaluate, and it never produces a value that reaches a
prediction. Every prediction in a completed run is asserted to name a method
and an expert.

---

## Shear viscosity: three routes to a property that had none

`python bench/viscosity_routes.py`

`shear_viscosity` had been in the property registry since Phase 1 with nothing
behind it, and section 6 records why physics could not supply it — a viscosity
comes from a Green–Kubo integral or non-equilibrium shear, and this system runs
neither. It was reachable from the correlation side the whole time.

Against 273 compounds with a measured liquid viscosity at 298 K, restricted to
where the DIPPR correlation is validated:

| route | answers | MAE (log₁₀) | median factor | bias | within 2× |
|---|---|---|---|---|---|
| Joback group contribution | 195/273 | **0.108** | 1.17× | −0.053 | 93% |
| Letsou–Stiel as published | 273/273 | 0.358 | 1.64× | −0.337 | 60% |
| Letsou–Stiel + fitted offset, held out | 131/131 | 0.296 | 1.63× | −0.028 | 66% |

Error is judged on log₁₀ because viscosity spans orders of magnitude: putting
1 mPa·s at 3 is the same error as putting 100 at 300, and an absolute spread in
Pa·s would say otherwise. Joback wins 81% of the 195 both answer.

**The inputs were not the problem this time.** Letsou–Stiel fed *measured*
critical constants scores 0.351; fed Joback-estimated ones it scores 0.358. That
is the opposite of the refractive index result, where the equation was fine and
the density was the whole error — so the check is worth running rather than
assuming either way.

The published correlation under-predicts these liquids by a consistent factor of
2.11. The offset is fitted on half the compounds, split on a hash of the
structure, and the figure above is measured on the other half. It removes the
bias (−0.352 → −0.028 held out) and barely touches the spread: the method's
problem is scatter, and only its offset is correctable.

A measured route sits above both, restricted to DIPPR and VDI data methods for
the same reason the density route is. `VISWANATH_NATARAJAN_2E` is in the same
table and returns **7470 Pa·s for 2-butanone** — seven orders of magnitude high,
and it looks exactly like a number. A plausibility bound of 100 Pa·s catches it;
a first attempt used 10⁴ Pa·s, which a molten polymer can reach, and let it
straight through.

On seven common solvents the panel lands at factor 1.00–1.04 via the measured
route, and `prefer()` orders the three correctly by stated spread without any
precedence rule: measured (±3%), Joback (±31%), corresponding states (±85%).

### Extensional viscosity, and what it refuses

For an incompressible Newtonian liquid in uniaxial extension the Trouton ratio
is exactly three. That is a result, not a correlation, so `TroutonExtensionalExpert`
adds no error of its own — the whole spread is the shear viscosity's, scaled.

For a polymer solution it is false, and the expert refuses rather than returning
three times something. A spinning dope strain-hardens: its extensional viscosity
rises by orders of magnitude as chains stretch and depends on strain rate and
strain history, so it is not one number and 3η is not an approximation to it.

---

## Does the jet become a fibre?

`python bench/web_shooter_jet.py`

A Newtonian jet always breaks up — surface tension amplifies any disturbance
longer than the circumference, on the Rayleigh scale √(ρR³/σ). A polymer
solution can resist: once the thinning rate exceeds the inverse chain relaxation
time the chains stretch instead of relaxing, and the elastic stress holds the
filament together. The competition is a ratio of those two times, the Deborah
number, and the transition sits near De ≈ 1.

Polystyrene in 2-butanone, with μ, ρ and σ from the panel (0.395 mPa·s,
799 kg/m³, 24.0 mN/m — viscosity via the measured route):

| M (kDa) | c (wt%) | c/c* | λ (ms) | De @ 0.3 mm | verdict |
|---|---|---|---|---|---|
| 150 | 10 | 3.2 | 0.038 | 0.11 | spray, not entangled |
| 150 | 20 | 6.6 | 0.645 | 1.88 | **filament** at 0.3 mm, marginal at 0.5, spray at 1.0 |
| 150 | 30 | 10.1 | 3.50 | 10.1 | filament at every nozzle |
| 500 | 10 | 6.5 | 4.00 | 11.8 | filament at every nozzle |
| 2000 | 20 | 29.6 | 14093 | 41000 | filament — but a gel, not pumpable |

**Molar mass is the strong lever, not concentration.** Going 150 → 500 kDa at
fixed 10 wt% moves De from 0.11 to 11.8 and turns a spray into a filament;
tripling the concentration at 150 kDa is needed to achieve the same. The
practical window is **≥500 kDa polystyrene at ≥10 wt%**, or 150 kDa at ≥20 wt%
through a nozzle no wider than 0.3 mm.

At 2 MDa the relaxation time exceeds a minute. The Deborah number is then
enormous and means nothing about break-up: the dope is a rubbery gel that will
melt-fracture at the nozzle rather than flow, and the assessment says so instead
of reporting a clean filament.

### What this does not decide

It is a regime, not a fibre, and every assessment carries that on it:

- **Whether the filament draws down to a good fibre.** Surviving capillary
  break-up is necessary, not sufficient; draw ratio and chain alignment decide
  the strength and neither is here.
- **The concentrated-solution relaxation time** is a reptation scaling from the
  dilute Zimm value, not a measurement. A factor of two moves De by the same
  factor — which matters only near the threshold, and the marginal band exists
  for that reason.
- **Solvent leaving the filament.** Nothing here models drying, so it does not
  say whether the thread solidifies before it lands. That was the original
  question about a web shooter and it remains unanswered.
- **Mark–Houwink constants** are tabulated per polymer, solvent and temperature.
  An untabulated pair is refused, because the exponent carries solvent quality
  and borrowing one across solvents changes the intrinsic viscosity threefold at
  high molar mass.

---

## Does the filament dry before it lands?

`python bench/web_shooter_drying.py`

Surviving capillary break-up leaves a solution, not a fibre. Two resistances in
series decide whether it becomes one in flight: getting vapour away from the
surface, and getting solvent from the core to the surface.

2-butanone (ρ 799 kg/m³, M 72.11 g/mol from the panel; Pₛₐₜ 12.1 kPa; Fuller
air-side diffusivity 9.18 mm²/s against a literature ~9), 80 wt% solvent, a
10 m shot at 20 m/s — **500 ms of flight**:

| radius | D = 10⁻¹⁰ m²/s | D = 10⁻¹¹ | D = 10⁻¹² |
|---|---|---|---|
| 5 µm | **dry** (43 ms) | **dry** (432 ms) | skinned |
| 20 µm | skinned (692 ms) | skinned | skinned |
| 100 µm | skinned (17 s) | skinned | skinned |
| 500 µm | wet | wet | wet |
| 2.1 mm | wet | wet | wet |

**Evaporation is never the problem.** It takes 1.8 ms at 5 µm and 179 ms at
100 µm — comfortably inside the flight. Diffusion out of the core is what
fails, and it fails by orders of magnitude, because the skin vitrifies first and
becomes the barrier for everything behind it.

The radius where diffusion just keeps up with a half-second flight:

| D (m²/s) | radius | diameter |
|---|---|---|
| 10⁻¹⁰ | 17.0 µm | 34 µm |
| 10⁻¹¹ | 5.4 µm | 10.8 µm |
| 10⁻¹² | 1.7 µm | 3.4 µm |
| 10⁻¹³ | 0.5 µm | 1.1 µm |

**Spider dragline silk is 3–5 µm in diameter.** That is not a coincidence: a
strand spun in air and expected to set before it lands has to be a few microns
across, and a spider is at the value this calculation lands on.

### The contradiction the calculation exists to produce

Holding an 800 N adult at a well-drawn 58 MPa needs a **4.2 mm** strand. Drying
in flight needs one under **11 µm**. The two requirements are 400× apart in
diameter, and a bundle of the thin ones would need:

| D (m²/s) | filaments needed |
|---|---|
| 10⁻¹⁰ | 15,184 |
| 10⁻¹¹ | 151,836 |

Which is, of course, what a real web is: many fine strands, not one thick one.
A single-orifice shooter producing one monofilament cannot do both — it lands
tacky-cored if it is thick enough to hold you, and holds ~4 N if it is thin
enough to dry.

### What this does not decide

- **The diffusion coefficient is an input, not a prediction**, and it is the
  dominant uncertainty by a wide margin. It falls four to six orders of
  magnitude as the surface vitrifies. Nothing here estimates it, because a
  concentration- and temperature-dependent mutual diffusivity in a vitrifying
  polymer solution is not something a correlation over a structure supplies, and
  inventing one would put the whole verdict on a fabricated number.
- **The evaporation time is a lower bound.** The mass-transfer coefficient is a
  crossflow correlation on a cylinder and the filament moves along its own axis,
  where the boundary layer grows down the length instead of being renewed. The
  surface is also held at saturation, which overstates the driving force once it
  starts to dry. Both errors point the same way, which is why the conclusion
  "evaporation is not the problem" is safe.
- **Nothing models the moving boundary.** A real drying filament has a
  concentration profile and a shrinking radius, not a single diffusion time.

---

## Melt viscosity, and why the hot-melt route also fails as a direct shot

`python -m pytest tests/test_rheology.py -k melt`

The drying calculation showed that setting by cooling beats setting by drying
by four orders of magnitude in diffusivity, so the obvious move is a hot melt
with no solvent at all. `MeltViscosityExpert` was built to test that, and it
does not survive contact with the nozzle.

**The construction.** WLF anchored at 10¹² Pa·s at the glass transition, which
is the rheological definition of Tg rather than a fitted parameter, with
measured WLF constants per polymer and the polymer's own Tg from the panel.

**Universal constants are not good enough, by three orders of magnitude.**
Anchored at Tg, the universal pair (17.44, 51.6) puts polystyrene at 3 Pa·s at
200 °C against a real melt viscosity of order 10³–10⁴. The polymer-specific pair
(13.7, 50.0) puts it at 740. So the table carries measured constants for three
polymers and an untabulated one is **refused** rather than answered with the
universal pair.

This is the weakest expert in the panel and says so on every prediction: one
order of magnitude, three polymers, a 120 K window above Tg, and zero-shear, so
it is an upper bound on what a spinning flow actually sees.

### Pressure at the nozzle

Hagen–Poiseuille through a 0.5 mm nozzle with a 10 mm land, polystyrene melt
from the panel:

| T (°C) | η (Pa·s) | at 0.5 m/s | at 20 m/s |
|---|---|---|---|
| 130 | 6.9 × 10⁶ | 4.4 × 10⁷ bar | 1.8 × 10⁹ bar |
| 170 | 1.0 × 10⁴ | 6.4 × 10⁴ bar | 2.6 × 10⁶ bar |
| 190 | 1.5 × 10³ | 9,819 bar | 3.9 × 10⁵ bar |
| 210 | 377 | **2,410 bar** | 96,417 bar |

A hydraulic hand tool reaches about 700 bar; a grease gun about 100. At 210 °C —
already near where polystyrene begins to degrade — a direct 20 m/s shot needs
**96,000 bar**. Shear-thinning at spinning rates is worth one to two orders of
magnitude, which closes the slow column and not the fast one.

**So the melt cannot be shot. It can only be extruded slowly and drawn down** —
which is melt-blowing, and is what the rest of the numbers had been pointing at:

| filament | vitrifies in | flight needed |
|---|---|---|
| 5 µm | 10.8 µs | 0.22 mm |
| 20 µm | 173 µs | 3.5 mm |
| 100 µm | 4.3 ms | 86 mm |

A melt-blown filament sets essentially instantly, and melt-blowing produces
1–10 µm fibres — the same diameter the drying calculation demanded and the same
diameter as spider dragline silk. Holding 800 N then needs **700,000 filaments
at 5 µm**, or 44,000 at 20 µm.

Which is the same conclusion the drying calculation reached by a different
route: the thing that works is a bundle of many fine fibres formed by
aerodynamic draw-down, not one thick strand from one orifice.

### Crystallisation kinetics: not added, and not for want of trying

The polymers in play — polystyrene, PMMA, poly(vinyl acetate) — are **amorphous**.
They do not crystallise; they vitrify, which is the thermal calculation above.
The bundled reference set carries amorphous density and glass transition and
nothing else: no melting point, no heat of fusion, no crystallisation half-time
for the semicrystalline polymers where Avrami kinetics would apply. There is
nothing to fit and nothing to validate against, so an Avrami model here would
be a fabricated capability rather than an approximate one.

---

## Keeping the shot

`python bench/ballistic_jet.py`

The melt route died at 96,512 bar. Both halves of that number were wrong for
this problem: it forced a *polymer melt* through a *half-millimetre* nozzle,
and the load calculation had already established that the strand has to be
4.2 mm. Pressure carries viscosity linearly and the radius squared, so both
choices were expensive and neither was required.

A thick jet is also a different break-up problem. `assess_jet` asks whether a
thin filament being drawn down resists capillary thinning, which is about
elasticity. A thick jet flying ballistically is asked something else: how far
it gets before capillarity closes on it. The Weber number of a 4.2 mm jet at
20 m/s is 59,000, and the answer is "a long way" — provided it leaves the
nozzle laminar.

**Viscosity decides that, and there is a floor:**

| µ (mPa·s) | Re | regime | break-up length | pressure to shoot |
|---|---|---|---|---|
| 2 | 44,100 | turbulent | **atomises** | — |
| 30 | 2,940 | turbulent | **atomises** | — |
| 50 | 1,764 | laminar | 27 m | **0.36 bar** |
| 100 | 882 | laminar | 33 m | 0.73 bar |
| 500 | 176 | laminar | 80 m | 3.63 bar |
| 2000 | 44 | laminar | 226 m | 14.5 bar |

Below about 40 mPa·s the jet is turbulent when it leaves and atomises at the
orifice — which is why a fire hose makes spray rather than a rod of water. Above
it, the jet is coherent for tens of metres and the pressure is under a bar.

**0.36 bar against 96,512.** A factor of 270,000, from two changes that cost
nothing: a thicker nozzle, which the load requirement already demanded, and a
resin instead of a melt.

### What a shot costs

277 mL/s. A 10 m shot lasts 0.50 s, uses **139 mL and 145 g** of resin, and
produces **5.8 N of thrust** while it fires — noticeable, not staggering.

### Why the cure has to be chemical

| mechanism | time to set a 4.2 mm strand |
|---|---|
| drying (solvent out) | 21 hours |
| cooling (heat out) | 7.6 s |
| **chemical reaction** | **radius-independent** |

Both physical mechanisms are diffusion in a cylinder, so both scale as R², and
a strand thick enough to hold a person is thick enough to defeat them. A
polymerisation is not: it proceeds through the volume at once, so a 4.2 mm
strand cures in the same time as a 5 µm one. That is the only setting mechanism
in this whole analysis whose rate does not depend on how thick the thing is.

So the shootable configuration is a **reactive resin of 50–500 mPa·s through a
4 mm nozzle**, cured chemically rather than by drying or cooling.

### What is still not established

- **No cure chemistry has been chosen or modelled.** Nothing in this repository
  predicts a cure rate, a pot life or an exotherm, and a two-part system mixed
  at the nozzle has a mixing problem this says nothing about. The claim here is
  only that the *mechanism* has the right scaling, not that a specific resin
  exists that does it in the time available.
- **The break-up correlation is Grant and Middleman's laminar one**, and the
  50 mPa·s row sits at Re 1764, close enough to the 2000 limit that the
  transition is not sharply resolved there.
- **Nothing has been said about what 145 g of curing resin does on landing**,
  or to a bystander, or to the person carrying 5.8 N of thrust on one wrist.
