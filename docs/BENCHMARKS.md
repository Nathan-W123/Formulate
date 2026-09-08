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
