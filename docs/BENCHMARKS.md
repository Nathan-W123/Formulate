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

---

## The three things the shot left open

`python bench/web_shooter_cure.py`

### 1. Cure chemistry: the exotherm sets the recipe

Chain polymerisation of a vinyl monomer releases 50–90 kJ per mole of double
bonds, and a mole of small monomer is about 100 g. Divided by a specific heat
near 1700 J/(kg·K), that is a few hundred kelvin if nothing escapes — and a
4.2 mm strand has a 7.6 s thermal time, so a 2 s cure keeps **79%** of it.

Methyl methacrylate, cured in 2 s in a 4.2 mm strand:

| reactive fraction | adiabatic rise | peak | verdict |
|---|---|---|---|
| 100% (neat) | 340 K | 294 °C | **cooks itself** — MMA boils at 100 |
| 30% | 102 K | 106 °C | burns |
| 20% | 68 K | 79 °C | burns |
| **15%** | 51 K | **65 °C** | **usable** |
| 10% | 34 K | 52 °C | usable |

**So at most ~15% of the resin may actually react.** The rest must be
pre-formed polymer or filler carrying heat capacity only — which is exactly
what acrylic bone cement does, for exactly this reason, and it raises the
viscosity into the range the coherent jet needs anyway. One choice satisfies
two constraints.

The cure-time window is **0.5–5 s**: faster than 0.5 s and a solid rod arrives
and bounces with no anchor; slower than ~5 s and it runs off before it holds.

### 2. Laminar margin

| µ (mPa·s) | Re | margin to 2000 | break-up length |
|---|---|---|---|
| 50 | 1764 | 1.1× | 27 m |
| **90** | 980 | **2.0×** | 32 m |
| 150 | 588 | 3.4× | 39 m |
| 300 | 294 | 6.8× | 57 m |

The earlier 50 mPa·s row sat at Re 1764, too close to the transition to rely
on. **90 mPa·s gives a 2× margin** and 150 gives 3.4×, both comfortably inside
what a filled resin is anyway.

### 3. What arrives

| | | |
|---|---|---|
| shot volume | 139 mL | 145 g |
| impact energy | **29 J** | paintball ~10 J, .22 air rifle ~20 J |
| thrust while firing | 5.8 N | 0.6 kgf — trivial |
| hanging load after | **800 N** | 82 kgf through one wrist |

29 J is squarely in eye-injury territory and is not lethal. The thrust is
nothing. **The hazard is the 800 N afterwards**, which is inherent to the idea
rather than to the chemistry, and a curing mass at 65 °C against skin is
uncomfortable rather than injurious — 15% reactive was chosen to put it there.

### Still not established

Cure *kinetics*. The cure time is an input to this calculation and nothing in
this repository derives it from an initiator, a temperature or a recipe. The
retained-heat fraction is a lumped one-parameter model, correct at both limits
and monotone between, not a solution of the coupled problem — where a hotter
core reacts faster and releases its heat sooner.

---

## A quantum validation that failed

`python bench/polymerisation_enthalpy_qm.py`

The web-shooter analysis rests on correlations and dimensionless groups, and
almost none of it is reachable by the physics stage — of the eight properties
involved, seven are on the not-validatable list with a stated physical reason
and only `liquid_density` is validatable at all, by dynamics that a default
budget does not buy.

One number is different. The whole "at most 15% of the resin may react"
conclusion rests on an enthalpy of polymerisation taken from a table — 57.8
kJ/mol for methyl methacrylate — and a bond-energy difference is exactly what a
quantum calculation is for. So it was run, as the propagation step a carbon
radical adding across the double bond, at B3LYP/6-31G* with the open-shell path
in the PySCF backend.

**It did not reproduce the tabulated values.**

| system | QM | published | difference |
|---|---|---|---|
| ethylene (control) | −118.9 kJ/mol | −93.0 | **−25.9** |
| ethylene, geometries optimised | −120.3 kJ/mol | −93.0 | **−27.3** |
| methyl methacrylate | −149.5 kJ/mol | −57.8 | **−91.7** |

Optimising the geometries moved the control by 1.4 kJ/mol and did not close the
gap, so the unrelaxed geometry was not the problem. What is left is that **the
calculation is not the quantity**: an electronic energy difference for one
radical adding to one monomer is missing the zero-point and thermal corrections
that turn it into a 298 K enthalpy, and a single addition does not feel the
chain crowding that makes methyl methacrylate's polymerisation enthalpy so much
smaller than ethylene's in the first place. The 92 kJ/mol gap on MMA is
substantially that steric effect, which is precisely what the model omits.

So the tabulated 57.8 kJ/mol stands, and this calculation does not support it.
It is recorded because the attempt is informative in the other direction: every
error here points the same way, toward **more** heat rather than less, so the
conclusion that the resin must be heavily diluted survives and strengthens. At
149 kJ/mol the allowed reactive fraction would fall from 15% to about 6%.

**Nothing else in the web-shooter chain has been validated by QM or MD, and the
engine would refuse most of it.** It is a chain of correlations, and several
links are weak by their own account: the melt viscosity expert is one order of
magnitude over three polymers, the cure time is an input rather than a
prediction, the diffusion coefficient in the drying calculation is an input, and
the Hansen route got polymer–polymer miscibility wrong for a well-understood
reason. It is a feasibility argument, not a validated design.

## Physical validation, and what it took

The paragraph above said nothing in the web-shooter chain had been validated by
physics and the engine would refuse most of it. Both halves were true, and both
had the same cause, which was not the one the refusals gave.

### The refusals were wrong about why

`surface_tension` was in `NOT_VALIDATABLE_REASONS` for needing "a periodic
condensed phase larger than the available potentials support"; `shear_viscosity`
for having "no dynamics workflow." What neither had was a pressure tensor. A
tension is a difference between diagonal components, a viscosity is the
autocorrelation of an off-diagonal one, and **OpenMM publishes no pressure
tensor at all** — its barostats compute one inside their own Monte Carlo move and
never expose it. Both properties were refused for an absent accessor.

`physics/md/stress.py` recovers it by finite difference: strain the cell, carry
the contents along affinely, take dU/dε from two single-point energies. Three
choices decide whether that is a pressure or arithmetic. The strain translates
whole molecules rather than scaling atoms, because bonds to hydrogen are
constrained and an atomic strain silently loses the constraint virial — so this
is the molecular virial, which gives the same tension and the same
zero-frequency viscosity as the atomic one. Image shifts cancel, because the
cell's vectors strain by the same ε. And only three of six shears are
expressible, since OpenMM needs box vectors in reduced form; the tensor is
symmetric, so that costs nothing.

Verified against four closed forms before any wall clock was spent on it:

| check | agreement |
|---|---|
| two Lennard-Jones particles, `−U′(r) dₐd_b / V`, all six components | 1 part in 10⁶ |
| full OPLS-AA cluster against the force identity `−Σ R_I·F_I` | few parts in 10⁶ |
| Ewald's exact 1/L scaling under isotropic strain | 8 parts in 10⁶ |
| ideal-gas kinetic term against `NkT/V` | 1329 vs 1325 bar |

And once end to end: a Monte Carlo barostat holding **250 bar** on 200 molecules
of 2-butanone, read back by finite difference at **258 ± 23 bar**, with an
instantaneous spread of 277 bar. The target was buried in fluctuations eleven
times its size and came back within a sigma.

### What MD then said about the carrier

`python bench/web_shooter_md.py bulk` — 400 molecules of 2-butanone under
OPLS-AA, 200 ps at constant pressure plus one isolated molecule at the same
temperature. 4.5 hours of contended CPU.

| | MD | measured | deviation |
|---|---|---|---|
| liquid density | 0.8077 ± 0.0005 g/cm³ | 0.7995 | **+1.0 %** |
| Hildebrand parameter | 19.63 ± 0.13 MPa^0.5 | 19.05 | **+3.0 %** |
| enthalpy of vaporisation | 36.9 kJ/mol | 34.8 | +6 % |

That is OPLS-AA's known record (1.2 % on density, 3.7 % on vaporisation over
five other liquids), from a force field that saw none of these numbers. The run
flagged itself: density drifted +0.0023 g/cm³ across production, larger than
its own sampling error, so the ±0.0005 is optimistic and the honest error is
nearer the drift. The Hildebrand parameter is the number that matters — it is
the total behind the Hansen sphere the solvent selection ran on, and MD
reproduces it independently from cohesive energy.

The surface tension slab was still running when this was written. 2-butanone
has since left the recipe (below), so these validate the engine rather than the
material; they are the first MD-validated numbers in the chain regardless.

### The base could not be searched for, because no rate existed

The reactive jet's majority component was specified as "a monomer that cures in
flight" and never selected, because the property registry held forty-nine
properties and **not one rate**. A material chosen for what it becomes cannot be
searched on what it is.

`experts/kinetics.py` adds propagation from the IUPAC pulsed-laser benchmark
(good to ~10 %), an order-of-magnitude termination coefficient, and an
initiation regime named in `Conditions.processing` — refused if absent, because
a cure time without one is a property of nothing. Seven monomers went into a
catalogue file of their own, **not** into `reference_compounds.json`, which is
the held-out calibration set a dozen docstrings cite by size.

`examples/cured_jet_base.yaml`, two-second ceiling: **nothing feasible in 56.**
Fastest solid-former was vinyl acetate at 79 s. The four faster monomers were
refused because their homopolymers are rubbers at ambient — a time at which a
liquid becomes a gum is not a cure. That disconfirmed the hand-picked base by
40×.

### Two orders of magnitude were in the endpoint, not the chemistry

The 79 s was time to *vitrification* at half conversion. A crosslinker does not
vitrify to solidify; it **gels**, at one crosslink per primary chain, which for
chains hundreds of units long is a fraction of a per cent of conversion.
Functionality is now read from the structure, gelation comes from
Flory–Stockmayer, and the glass-transition refusal applies only to linear
polymers — it had been turning away every diacrylate, the exact chemistry fast
enough to work.

`examples/cured_jet_base_gelled.yaml`, two seconds: **hexanediol diacrylate gels
in 0.064 s, trimethylolpropane triacrylate in 0.026 s.** Both stated as lower
bounds on conversion (cyclisation) and upper bounds on time (Trommsdorff), with
a borrowed family k_p since no pulsed-laser measurement exists for a monomer
that gels during the experiment.

### Three defects the search turned up, all silent

**`prefer()` compared absolute uncertainty.** On a property spanning orders of
magnitude that hands the win to whichever expert predicts the smallest number.
HDDA's viscosity: Joback 4.25 mPa·s at ±31 %, corresponding states 0.175 at
±85 % — and the second won, on 1.5×10⁻⁴ against 1.3×10⁻³ Pa·s of spread. Every
jet number was wrong by fifty-fold until chased. Nine properties now declare
`multiplicative_error` and compare on relative spread; a boiling point keeps the
absolute comparison, and a test asserts both halves.

**Joback estimates arrived labelled "measured."** The compilation's default
accessor returns a group contribution when it holds nothing else and does not
say so; it came through the route stamped *"measured, not estimated"* with a
1 K error bar, and the panel called both winning monomers solids at 25 °C.
Joback's melting point is its weakest correlation (53 K low on MMA where a
measurement exists). Filtered to compilation methods; zero of the fifty
reference compounds affected.

**A vacuum boiling point served as a normal one.** HDDA's compiled 403 K is
where it distils at a few mmHg; at one atmosphere it is near 570. It passed a
hard constraint it should have failed and fed a Brock–Bird surface tension of
8.4 mN/m against a real 34. Caught by a 181 K disagreement with the group
estimate (median 7 K over the set) and arbitrated by reduced boiling point
Tb/Tc, 0.663 ± 0.034 over 58 compounds: the compiled value's 0.527 is the worst
outlier in the set. The same tiebreaker sides with the compilation for
γ-butyrolactone (0.654 vs Joback's 0.494), so it settles both contradicted
compounds in opposite directions with neither chosen by hand. With the boiling
point right, the panel's surface tension for HDDA is **35.2 mN/m**.

### Closing the jet, and where it closed

`python bench/web_shooter_gelled_jet.py` runs the three constraints of the
reactive jet against each other for the first time. Thin jets stay cool and
laminar and break up in milliseconds; thick ones survive and cook (409 K of
adiabatic rise in a diacrylate, 121 °C at 4.2 mm); at the resin's own viscosity
there is no radius that works. Viscosity opens the window — which is what the
thickener was always for. At 100 mPa·s and 1 mm radius: Reynolds 404, break-up
11.5 m against a 0.064 s gel, peak 32 °C, 0.8 bar at the nozzle.

`bench/web_shooter_strength.py` puts a bound under the assumed 40 MPa. The
engine's own `theoretical_strength` (E/10, 286 MPa on the glassy plateau) is
knocked down by a factor measured on the two glassy polymers it can predict:
0.14 for polystyrene, 0.245 for PMMA. **Realised: 40–70 MPa, 4–7 filaments of
1 mm.** The aligned-chain ceiling is 33 GPa and unreachable, because nothing
draws a ballistic jet — which is why spiders pull silk rather than shoot it.

### The formulation, validated as a mixture

Asked to score the blend, the panel returned nothing for nine properties: the
Hansen compilation covers ordinary solvents, and every component here is
something else. `GroupContributionHansenExpert` fills it by Hoftyzer–Van
Krevelen, calibrated against the compilation over the forty reference structures
it accepts — **0.89, 0.80, 1.18 MPa^0.5** on dispersion, polar and hydrogen
bonding — with a test that recomputes those and fails if the expert ever claims
better. It refuses on any uncovered group, on a polyhalogenated carbon, and
(after being caught typing benzoyl peroxide as two esters) on a peroxyester.
Polymer components now resolve through the polymer experts with their `[*]`
attachment points intact.

| | |
|---|---|
| polystyrene in HDDA | **RED 0.83** — dissolves (2-butanone 0.87, hexane 1.16 as controls) |
| blend Hansen triple, polymer included | 16.6 / 3.6 / 8.4 MPa^0.5 |
| blend density | 1170 kg/m³ |

`examples/web_fluid_formulation.yaml`: 96.6 % HDDA, 1.5 % polystyrene 500 kDa,
1.2 % benzoyl peroxide (part A), 0.67 % N,N-dimethyl-p-toluidine (part B).
The thickener loading is stated as `c[η] = 1.81`, which is exact, and split into
0.7–3.1 wt% across the measured polystyrene pairs, because no Mark–Houwink
constants exist for it in a diacrylate and borrowing one is refused.

### What passes, and why it is still the wrong thing to build

Everything the engine can compute passes with margin. What it cannot compute is
exactly what decides the device: **MD is impossible on the base** (OPLS-AA has
no ester carbonyl type and refuses every acrylate), **strength is
flaw-controlled** and no structure route exists, **adhesion is untouched**,
and a Mc ≈ 113 g/mol network is brittle — the literature confirms sub-second
acrylate cure gives brittle specimens. Add a 64 ms pot life that must clear
2.8 mL of mixhead in 6.4 ms, a 409 K exotherm in anything trapped, and three
components that are sensitisers or oxidisers, and the reactive jet is the
scientifically interesting answer and the wrong material.

A polyamide hot melt sidesteps every one of those: single part, no pot life, no
exotherm, tough rather than brittle, benign pellets, and 33–53 MPa published —
the same band as the acrylate without any of its failure modes. The engine could
not select it, because toughness, hazard and a polymer candidate pool were not
in it. All three are in it now, and the section below is the run: the engine
reaches a copolyamide on its own evidence, and refuses the acrylate on three
hard requirements it could not previously see.

---

## The thermoplastic run: making the acrylate lose

```
python -m formulate.cli.main run examples/web_fluid_thermoplastic.yaml
python -m formulate.cli.main run examples/web_fluid_thermoplastic_relaxed.yaml
python -m pytest tests/test_toughness.py tests/test_hazard.py -q
```

Five runs converged on 1,6-hexanediol diacrylate with a benzoyl peroxide /
N,N-dimethyl-p-toluidine redox pair, gelling in 0.064 s. That is the correct
answer to the specifications that were written, and it is the wrong material to
build: a tightly crosslinked network is brittle, a 64 ms pot life clogs, the
diacrylate is a skin sensitiser and the peroxide is both a sensitiser and an
organic peroxide, and a thermoset nozzle jam is permanent.

**The engine did not miss any of that. None of it was a property.** A failure
mode the registry does not carry cannot cost a candidate a single point, and
that is a statement about the registry rather than about the ranking.

So this run adds what was missing — an elongation at break, a crosslink
density, and three GHS endpoints — puts the cured network into the candidate
pool alongside sixteen commodity thermoplastics, and asks the panel to choose.

### What the panel is worth on each of the new columns

| route | measured against | what it is worth |
|---|---|---|
| tabulated elongation | geometric width of the quoted `[low, high]` ranges, 16 polymers | **factor 1.97** |
| toughness proxy, rubbery class | leave-one-out, 5 polymers | factor 1.40 (2.14 with specimen spread) |
| toughness proxy, glassy class | leave-one-out, 6 polymers | **factor 6.64** (7.47 with specimen spread) |
| toughness proxy, overall | leave-one-out hit rate | **7 / 11**, typical miss 3.53× |
| tabulated glass transition | stated compilation-to-compilation spread | 7 K |
| tabulated amorphous density | worst disagreement between the two tables bundled here | 0.02 g/cm³ |
| crosslink density | read off the specification | exact, or refused |
| GHS endpoints | curated lookup | exact, or refused |

**The proxy's glassy class does not work and the number says so.** Polystyrene
elongates 1.7% and polyamide 6 draws to 77%, forty-five times as far; both are
glasses at ambient and the *ductile* one is the closer to its transition, 25 K against 75.
Distance above Tg does not separate them in magnitude or even in direction.
What does is whether the glass shear-yields before it crazes, which is the
entanglement density of the chains in the glass — a quantity
`polymer_mechanical` already computes from the packing length, and holds the
chain dimension for nine repeat units, none of them a polyamide or a polyester.
The better proxy is one measurement per polymer away and is not reachable from
the structure.

**A defect the calibration caught before the run shipped.** The class mean over
five rubbers is reproducible to 1.40 and one handbook range's half-width is
1.97, so a proxy quoting only its own fit error is the *tighter* prediction and
`prefer()` chose it over every measurement it had been fitted to. Both routes
predict what a bar of this polymer will do, so both carry the
specimen-to-specimen spread and the proxy carries its class error on top.
`test_a_measured_elongation_outranks_the_proxy_that_was_fitted_to_it` holds it.

### Run 1, as specified: nothing feasible, and the blocker is the panel

Seventeen candidates, **zero feasible**. `shear_viscosity` at 180 °C eliminated
all seventeen — sixteen because no value could be produced at all, and
polystyrene because 3,657 Pa·s is 366 times the ceiling.

That is the correct behaviour and it is not a chemistry result. The panel's only
route to a melt viscosity is WLF referenced to the glass transition with
measured constants for three polymers, and for a semicrystalline hot melt that
route is not approximately right but **inapplicable**: above its melting point
the melt follows Arrhenius and has left the window WLF describes. A hard
requirement that cannot be verified is a hard violation, so every candidate died
on a gap in the panel and the six requirements that *can* be checked never got
to decide anything.

**One wrong number was found here rather than by reading.** Poly(methyl
methacrylate) came first at 3.4 × 10⁻⁵ Pa·s — thinner than water by thirty
times, for a polymer whose real melt viscosity at 180 °C is of order 10⁴, and
75 K above its transition so well inside the existing 120 K guard. The defect is
in the constant pair: `c1` is the number of decades between the reference
temperature and the high-temperature asymptote, so a pair referenced to Tg
cannot have a `c1` that puts the asymptote below any liquid that exists. 13.7
leaves polystyrene asymptotic to 0.02 Pa·s; **34.0 leaves PMMA asymptotic to
10⁻²² Pa·s**, which is not a viscosity. The pair is very likely quoted against a
reference temperature of its own and the table cannot tell. The expert now
refuses below the viscosity of water and names the `c1` that caused it.
Polystyrene is untouched.

### Run 2, melt viscosity soft: a copolyamide hot melt

Four of seventeen feasible. Ranked:

| # | candidate | Tg / K | elongation | crosslinks | H317 | ρ / kg m⁻³ | verdict |
|---|---|---|---|---|---|---|---|
| 1 | **copolyamide 6/66/12** | 320 ± 34 *(est.)* | **4.58** *(meas.)* | 0 | no | 1042 *(est.)* | feasible |
| 2 | **polyamide 6** | **323 ± 7** *(meas.)* | 0.77 *(meas.)* | 0 | no | 1084 *(meas.)* | feasible |
| 3 | poly(butylene terephthalate) | 315 *(meas.)* | 1.22 *(meas.)* | 0 | no | 1260 *(meas.)* | feasible |
| 4 | poly(ethylene terephthalate) | 342 *(meas.)* | 0.95 *(meas.)* | 0 | no | 1335 *(meas.)* | feasible |
| 7 | polystyrene | 373 *(meas.)* | **0.017** *(meas.)* | 0 | no | 1050 | **brittle** |
| 12 | poly(methyl methacrylate) | 378 *(meas.)* | **0.045** *(meas.)* | 0 | no | 1170 | **brittle** |
| 15 | polylactide | 328 *(meas.)* | **0.045** *(meas.)* | 0 | no | 1248 | **brittle** |
| 5,6,10 | PP, HDPE, LDPE | 267, 195, 195 | 2.4–3.6 | 0 | no | 850–855 | **Tg too low** |
| 8,9 | EVA-18, EVA-40 | 214, 233 *(est.)* | 7.3, 8.4 *(meas.)* | 0 | no | — | **Tg too low** |
| 11 | polyamide 12 | 310 *(meas.)* | 2.65 *(meas.)* | 0 | no | 990 | **Tg 3 K short** |
| 17 | **cured poly(HDDA)** | refused | **0.022** *(meas.)* | **9,700** | **yes** | refused | **four hard failures** |

**The engine converges on a copolyamide hot melt**, which is the material a
person who has built one would have named. It gets there on stated evidence and
not by assertion:

- **copolyamide first** on a measured elongation of 4.58 — it draws six times
  further than polyamide 6 — with a transition it has to *estimate*, at ±34 K,
  which is why its transition utility is zero on the risk-adjusted scale even
  though the nominal value is inside the window. It wins on toughness and pays
  for it in certainty.
- **polyamide 6 second**, and it is the more solid answer: every number in its
  row is measured, its transition is inside the window by 10 K against a 7 K
  spread, and it is the lightest of the four feasible.
- The **polyesters** are feasible and lose on density alone: PET is 1,335
  kg/m³ against polyamide 6's 1,084, and mass on a wrist is the one soft
  objective that discriminated here.

**The acrylate finishes last, refused on four hard requirements at once**, three
of them by the machinery this run added:

| requirement | poly(HDDA) | source |
|---|---|---|
| `elongation_at_break` ≥ 0.05 | **0.022** | measured range 1–5% for a neat cured diacrylate network |
| `crosslink_density` = 0 | **9,700 mol/m³** | calculated from complete conversion; exact in sign |
| `skin_sensitiser` = false | **true (H317)** | curated GHS table, inherited from the monomer by a recorded decision |
| `glass_transition_temperature` in 40–120 °C | refused | network topology is outside every repeat-unit correlation |

That last row is the shape of the whole result: the network is refused rather
than scored, by two experts independently, each naming its reason. Nothing here
argued the acrylate down. It was proposed, evaluated by the same panel at the
same conditions as everything else, and lost.

### What the run does not establish, and what it exposed

**The melt viscosity was never verified for any winner.** The soft run's top
four all score zero on it, which is the absence of a measurement and not a bad
measurement. A copolyamide hot melt's real melt viscosity at 180 °C is of order
2–50 Pa·s and would pass the ceiling comfortably; this engine cannot say so, and
saying so here would be asserting exactly the kind of number the rest of the
repository refuses to assert.

**No polymer melting point exists in the registry, and this is now the largest
gap on the polymer side.** Every semicrystalline candidate in the pool — the
polyethylenes, polypropylene, both polyamides, both polyesters,
polycaprolactone, both EVA grades — is useful in service up to its *melting*
point and was judged here on its *glass* transition, which understates it by a
hundred kelvin and more. The requirement that eliminated ten of seventeen
candidates is therefore the wrong requirement, correctly applied. It is exactly
why polyamide 12 lost by 3 K and why both EVA grades and the polyethylenes lost
at all — a hot melt is *designed* to have a low transition.

**A gate that refused three properties for needing a number they never read.**
`polymer_mechanical` required a tabulated chain dimension before answering
anything, and for the modulus below the glass transition that requirement is
spurious: the glassy branch returns a constant fitted across nine measured
amorphous polymers and never consults the entanglement mass, and the strength
bound is a tenth of that modulus. The chain dimension is tabulated for nine
repeat units, none of them a polyamide or a polyester — so the gate was
declining exactly the materials in this run's feasible set, for a calculation
that would not have used it. It now gates the entanglement mass and the rubbery
branch only, which is where the number is actually read.

The ranking is unchanged by the fix — the top four are the same four in the
same order, because all of them gain the same value — but the column is no
longer empty:

| | E (GPa) | G (GPa) | flaw-free bound |
|---|---|---|---|
| copolyamide, PA6, PA12, PET, PBT, PS, PMMA, PLA, TPU | **2.86 ± 0.50** | **1.06 ± 0.19** | **286 ± 143 MPa** |

**That is one number, not nine.** Every polymer above is a glass at ambient and
the glassy plateau barely depends on structure — 2.0 to 3.5 GPa across the nine
measured polymers behind it, which is less scatter than there is between
compilations for one polymer. The modulus therefore does not discriminate
between any of these candidates and did not contribute to the ranking. What
separates them is the elongation, which is exactly what the run ranked on.

**What is still refused is the thing that decides brittle from tough.** The
entanglement molar mass needs the chain dimension, so `is_tough` cannot run for
any polyamide or polyester, and every one of those predictions carries a note
saying so: *"no tabulated chain dimension for this repeat unit, so whether the
chains entangle into a load-bearing network is unknown; that, not the modulus,
is what decides brittle from tough."* A modulus is not a strength.

**No degree of crystallinity either, and it costs a number that can be shown.**
With the gate corrected, the catalogue's measured tensile column can now check
what `theoretical_strength` is worth against twelve polymers rather than five:

| polymer | measured tensile | flaw-free bound | delivered |
|---|---|---|---|
| copolyamide 6/66/12 | 28 MPa | 286 MPa | **9.9%** |
| thermoplastic polyurethane | 39 MPa | 286 MPa | **14%** |
| polystyrene | 44 MPa | 286 MPa | **15%** |
| polyamide 12 | 47 MPa | 286 MPa | **16%** |
| poly(butylene terephthalate) | 55 MPa | 286 MPa | **19%** |
| polylactide | 59 MPa | 286 MPa | **21%** |
| poly(ethylene terephthalate) | 60 MPa | 286 MPa | **21%** |
| poly(methyl methacrylate) | 60 MPa | 286 MPa | **21%** |
| polyamide 6 | 77 MPa | 286 MPa | **27%** |
| low-density polyethylene | 12 MPa | 0.7 MPa | **17×over** |
| high-density polyethylene | 26 MPa | 0.7 MPa | **39×over** |
| polypropylene | 36 MPa | 0.1 MPa | **339×over** |

The bound holds for all nine glasses, at the one-to-three-orders-below the
mechanical module claims, and the spread across them — 9.9% to 27% — is the
real content: it is a measure of how much of the possible each material
delivers, which is what the bound was introduced to say. It is *broken* by every
semicrystalline polymer, and not because the bound is wrong: those three are
above their glass transitions at ambient, so the panel computes a rubber-elastic
modulus — `3ρRT/M_e`, about a megapascal — for materials that carry their load
in crystals. Polypropylene is the worst because the rubbery modulus goes as
`1/M_e` and the panel puts its entanglement mass at 6,014 g/mol against
polyethylene's 948. The panel is answering correctly for the amorphous polymer,
which is a different material with the same repeat unit.

**Adhesion had no route at any fidelity**, and closing it turned out not to be
a surface-chemistry problem at all. That is the next section.

**The hazard screen screens out and cannot screen in.** Skin sensitisation,
acute toxicity and carcinogenicity, from a 76-row curated table covering every
structure the bundled explorer can propose. Reproductive and target-organ
toxicity, mutagenicity, aspiration, flammability, environmental hazard and every
exposure limit are outside it, and three of the 76 rows abstain on
carcinogenicity rather than record a "not classified" nobody established. The initiator system shows
what that costs: benzoyl peroxide's H242 appears in the codes attached to every
prediction and in none of the numbers, so a specification screening on
sensitisation and acute toxicity alone sees only half of what is wrong with
putting an organic peroxide in a hand-held cartridge.

**Six of the seventeen candidates carry an estimated transition rather than a
measured one** — the copolyamide, both EVA grades, the polyurethane, and the two
that were refused. For the copolyamide that is a ±34 K estimate deciding a hard
80 K window, and it is the one number in the winning row that a bench would
settle in an afternoon.

**Nothing about the two-part pot life was tested here.** The 64 ms gel time is a
processing property, not a material one, and it is the reason a thermoplastic
route is being asked for at all — but a specification stating
`material_classes: [polymer]` cannot see it, because a cure time belongs to a
monomer and a monomer is not a polymer. The two halves of the argument still do
not meet inside one run.

---

## Tack: why the strand cannot also be the glue

```
python -m formulate.cli.main run examples/web_fluid_tip.yaml
python -m pytest tests/test_adhesion.py -q
```

Run 7 selected a load-bearing filament and left the thing a web needs most
unanswered: the strand has to stick to what it hits, and an ordinary
nylon-like copolyamide does not. The obvious move was to extend the
Owens–Wendt expert to polymers. That would have answered the wrong question.

**Surface energy decides whether a liquid wets. It does not decide whether a
solid sticks.** A poly(tetrafluoroethylene) film and a poly(tetrafluoroethylene)
grease have identical surface chemistry and only one of them is an adhesive.
What separates them is stiffness: Dahlquist's criterion puts pressure-sensitive
tack behind a storage modulus below about **10⁵–3 × 10⁵ Pa** at 1 Hz, because a
stiffer material cannot deform into the asperities of a real surface in the
second it is pressed there.

### The contradiction, in numbers the panel already had

| | shear modulus at 298 K | vs the 3 × 10⁵ Pa ceiling |
|---|---|---|
| copolyamide 6/66/12 | 1.06 GPa | **3,531× too stiff** |
| PA6, PET, PBT, PS, PMMA, PLA, PA12, TPU, PVAc | 1.06 GPa | 3,531× too stiff |
| HDPE / LDPE, amorphous phase | 2.23 MPa | 7.4× too stiff |
| polypropylene, amorphous phase | 0.35 MPa | **1.2× — inside the band** |

A filament that carries 80 kg is a glass and a glass is ~10⁹ Pa; tack needs
10⁵. **The two functions want moduli three and a half orders of magnitude
apart, so no single material occupies both.** Separating them into two
materials is not a preference between architectures — it is the only available
one, and it is now a result the panel states rather than an assertion in a
notes block. Surface treatment of the filament does not escape it either:
coating a 1 GPa strand leaves a 1 GPa strand.

### The band is the uncertainty, and inside it the expert declines

The criterion is quoted as 10⁵ Pa as often as 3 × 10⁵, and that is not sloppy
citation — "sticks" depends on how rough the surface is, how hard it was
pressed and for how long. Both ends are carried, the modulus's own spread is
read as a **factor** rather than an amount (a rubbery modulus carries a
relative spread of one, so a plus-or-minus would reach zero and call every
rubber tacky), and a material whose interval straddles the band is **refused**.

Amorphous polypropylene is the standing case: 3.5 × 10⁵ Pa, factor of two
either way, so 1.75 × 10⁵ to 7 × 10⁵ — the whole band. It is refused, and it is
the *highest-ranked* candidate in the run. The nearest thing to an answer is
precisely the material the expert declines to classify, which is the correct
outcome: inside the band the decision turns on the difference between the
storage modulus at 1 Hz the criterion states and the static plateau this panel
produces, and that difference is not modelled.

### Run 8: nothing is tacky

`examples/web_fluid_tip.yaml` asks for the second material — tacky,
thermoplastic, not a sensitiser, transition below ambient. **0 feasible of 17.**
`tack` eliminated every candidate: twelve confidently too stiff, one straddling
the band, four with no modulus at all.

| outcome | count | which |
|---|---|---|
| confidently **not** tacky | 12 | every glass, plus both polyethylenes |
| refused, straddles the band | 1 | polypropylene (amorphous phase) |
| refused, no modulus available | 4 | EVA-18, EVA-40, PCL, cured poly(HDDA) |
| **tacky** | **0** | — |

**The two most useful candidates are the two that cannot be scored.** EVA at
both vinyl-acetate levels is the classic hot-melt tackifier base, and neither
can be given a modulus at all: `polymer_mechanical` needs a tabulated chain
dimension and there is none for a copolymer repeat unit.

### What this does not establish, and it is most of it

**This is a precondition, not a measurement.** A probe-tack or loop-tack test
reports a force; this reports whether the material could collect one. Bond
strength is dominated by viscoelastic dissipation during separation and exceeds
the thermodynamic work of adhesion by two to three orders of magnitude —
nothing in this panel computes it, and the Owens–Wendt expert beside this one
would not have either.

**The tackifier is outside what the panel can score, not merely absent from the
catalogue.** A real tackifying resin — a rosin ester, a hydrocarbon resin — is a
low-molar-mass oligomer, not an entangled polymer, and every modulus model here
is the high-polymer limit: a glassy plateau fitted on nine amorphous polymers,
or `3ρRT/M_e` for a rubber. Adding tackifier rows to the catalogue would
produce more refusals, not more answers. Reaching them needs a modulus model
below the entanglement limit, which does not exist here.

**Nothing here specifies the architecture.** A tip bead, a coaxial sheath and a
surface treatment are three different products, and the run chooses between
none of them. Nor is there anything about how the two materials bond to each
other, which for a polyamide/polyolefin pair is its own hard problem.
