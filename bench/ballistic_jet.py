"""Keep the shot. What has to change?

The 96,000 bar came from forcing a polymer melt through a half-millimetre
nozzle. Both halves of that were wrong for this problem: the load calculation
already wanted a thick strand, and pressure goes as 1/R^2.

Run with ``python bench/ballistic_jet.py``. Results are in docs/BENCHMARKS.md.
"""

from __future__ import annotations

import math
import warnings

warnings.filterwarnings("ignore")
from formulate.processing import (  # noqa: E402
    breakup_length,
    extrusion_pressure,
    reynolds_number,
)

rho, v, sigma, L_nozzle = 1050.0, 20.0, 0.030, 0.02
D_LOAD = 4.2e-3   # what holding 800 N at 58 MPa needs

print("pressure to shoot a 4.2 mm jet at 20 m/s (20 mm nozzle land)\n")
print(f"{'mu /mPa.s':>10s} {'Re':>7s} {'break-up':>10s} {'pressure':>12s}")
for mu in (50e-3, 100e-3, 500e-3, 2.0, 10.0):
    L = breakup_length(rho, v, D_LOAD, sigma, mu)
    p = extrusion_pressure(mu, D_LOAD/2, v, L_nozzle)
    print(f"{mu*1000:10.0f} {reynolds_number(rho,v,D_LOAD,mu):7.0f} "
          f"{('atomises' if L is None else f'{L:5.0f} m'):>10s} {p/1e5:11.2f} bar")

print("\nfor contrast, the melt through the fine nozzle that killed it:")
print(f"   377 Pa.s through 0.5 mm at 20 m/s: "
      f"{extrusion_pressure(377.0, 2.5e-4, 20.0, 0.01)/1e5:,.0f} bar")

print("\nwhat a shot costs")
q = math.pi*(D_LOAD/2)**2 * v
print(f"   flow {q*1e6:.0f} mL/s; a 10 m shot lasts {10/v:.2f} s and uses {q*(10/v)*1e6:.0f} mL")
mass = q*(10/v)*rho              # kg of resin per shot
thrust = rho*q*v                 # momentum flux, N, while the jet is firing
print(f"   that is {mass*1000:.0f} g of resin, and {thrust:.1f} N of thrust "
      f"while it fires")

print("\nwhy the cure has to be chemical")
alpha, D_solvent = 1.0e-7, 1e-11
for label, coeff in (("cooling (thermal)", alpha), ("drying (solvent)", D_solvent)):
    t = (D_LOAD/2)**2/(2.404826**2 * coeff)
    print(f"   set a 4.2 mm strand by {label:20s} {t:10.1f} s")
print(f"   set it by {'reaction':20s}        radius-independent: a 4.2 mm strand "
      "cures in the same time as a 5 um one")
