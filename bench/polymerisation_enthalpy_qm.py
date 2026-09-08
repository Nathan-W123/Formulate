"""The one number in the web-shooter chain that quantum mechanics can check.

The whole "at most fifteen per cent may react" conclusion rests on an
enthalpy of polymerisation taken from a table: 57.8 kJ/mol for methyl
methacrylate. That is a bond-energy difference, which is exactly what a
quantum calculation is for.

Modelled as the propagation step, a carbon radical adding across the double
bond, which is what actually happens and is what the tabulated enthalpy means:

    CH3. + CH2=CR1R2  ->  CH3-CH2-C.R1R2

Ethylene first, where the answer is known (-93 kJ/mol), as a check on the
method before it is trusted on the monomer that matters.
"""
from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore")
from formulate.physics.geometry import geometry_from_smiles  # noqa: E402
from formulate.physics.qm import backend_for  # noqa: E402
from formulate.physics.qm.base import QMMethod, QMRequest  # noqa: E402

backend = backend_for(QMMethod.DFT)

def energy(smiles, multiplicity, label):
    g = geometry_from_smiles(smiles, n_conformers=5)
    g = g.__class__(symbols=g.symbols, positions=g.positions,
                    charge=0, spin_multiplicity=multiplicity, source=g.source)
    t0 = time.time()
    r = backend.run(QMRequest(geometry=g, method=QMMethod.DFT, basis="6-31g*",
                              xc="b3lyp", optimize_geometry=False))
    if not r.usable:
        print(f"   {label:34s} FAILED: {'; '.join(r.diagnostics)[:70]}")
        return None
    e = r.total_energy.to("J/mol").value
    print(f"   {label:34s} {e/1000:16.1f} kJ/mol   ({time.time()-t0:.0f} s, "
          f"{len(g.symbols)} atoms)")
    return e

CASES = [
    ("ethylene", "[CH3]", "C=C", "CC[CH2]", -93.0),
    ("methyl methacrylate", "[CH3]", "C=C(C)C(=O)OC", "CC[C](C)C(=O)OC", -57.8),
]
for name, radical, monomer, adduct, published in CASES:
    print(f"\n{name}:")
    e_r = energy(radical, 2, "methyl radical (doublet)")
    e_m = energy(monomer, 1, "monomer (singlet)")
    e_a = energy(adduct, 2, "adduct radical (doublet)")
    if None in (e_r, e_m, e_a):
        continue
    dh = (e_a - e_r - e_m) / 1000.0
    print(f"   {'-> propagation enthalpy':34s} {dh:16.1f} kJ/mol")
    print(f"   {'   published':34s} {published:16.1f} kJ/mol"
          f"   (difference {dh - published:+.1f})")
