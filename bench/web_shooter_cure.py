"""The three things the ballistic-jet result left open.

How much of the resin may react before the exotherm burns, how much laminar
margin the jet has, and what 145 g arriving at 20 m/s actually does.

Run with ``python bench/web_shooter_cure.py``. Results are in docs/BENCHMARKS.md.
"""

from __future__ import annotations

import math
import warnings

warnings.filterwarnings("ignore")
from formulate.processing import (  # noqa: E402
    CureConditions,
    assess_cure,
    breakup_length,
    impact_energy,
    jet_thrust,
    reynolds_number,
    thermal_time,
)

R, V, RHO, SIGMA = 2.1e-3, 20.0, 1050.0, 0.030
FLIGHT = 0.5

print("== 1. cure chemistry: how much of the resin may react? ==\n")
print(f"{'reactive':>9s} {'rise':>7s} {'retained':>9s} {'peak':>7s}  verdict")
for frac in (1.00, 0.50, 0.30, 0.20, 0.15, 0.10):
    a = assess_cure(CureConditions(monomer="methyl methacrylate", reactive_fraction=frac,
        cure_time=2.0, radius=R, flight_time=FLIGHT))
    flag = "  BURNS" if a.burns_on_contact else "  ok"
    print(f"{frac:9.0%} {a.adiabatic_rise_k:6.0f}K {a.retained_fraction:9.0%} "
          f"{a.peak_temperature_c:6.0f}C  {a.verdict.value}{flag}")
print(f"\nthermal time of a {R*2e3:.1f} mm strand: {thermal_time(R):.1f} s "
      f"- a 2 s cure keeps most of its heat")

print("\n== cure timing window ==")
for t in (0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
    a = assess_cure(CureConditions(monomer="methyl methacrylate", reactive_fraction=0.15,
        cure_time=t, radius=R, flight_time=FLIGHT))
    print(f"   cure in {t:5.1f} s -> {a.verdict.value:9s}  peak {a.peak_temperature_c:5.0f} C")

print("\n== 2. laminar margin ==\n")
print(f"{'mu /mPa.s':>10s} {'Re':>7s} {'margin to 2000':>15s} {'break-up':>10s}")
for mu in (50e-3, 90e-3, 150e-3, 300e-3):
    Re = reynolds_number(RHO, V, 2*R, mu)
    L = breakup_length(RHO, V, 2*R, SIGMA, mu)
    print(f"{mu*1000:10.0f} {Re:7.0f} {2000/Re:14.1f}x {L:9.0f} m")

print("\n== 3. what it does on arrival ==\n")
vol = math.pi*R**2*V*FLIGHT
print(f"   shot volume            {vol*1e6:8.0f} mL   ({vol*RHO*1000:.0f} g)")
print(f"   impact energy          {impact_energy(vol, RHO, V):8.0f} J    "
      f"(paintball ~10 J, .22 air rifle ~20 J)")
print(f"   thrust while firing    {jet_thrust(RHO, R, V):8.1f} N    "
      f"({jet_thrust(RHO,R,V)/9.81:.1f} kgf on the wrist)")
print(f"   hanging load afterward {800:8.0f} N    ({800/9.81:.0f} kgf through the same wrist)")
