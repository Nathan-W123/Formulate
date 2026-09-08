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
