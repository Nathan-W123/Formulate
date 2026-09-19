"""Hardware for the drawn-filament device, and whether it plugs.

The drawn architecture changes the hardware out of recognition, and the reason
is that the die runs slow. Mass is conserved, so a filament drawn 625x leaves
the die at one six-hundred-and-twenty-fifth of the line speed. The melt creeps;
the speed is bought afterwards, for free, by pulling.

That collapses everything downstream of it - the flow rate, the reservoir, the
pressure, the shot mass - by the same factor.
"""

from __future__ import annotations

import math

LB = 4.4482216152605
F50 = 50 * LB

DIE_UM, DRAW, LINE_SPEED = 1500.0, 625.0, 9.0
FINAL_UM = DIE_UM / math.sqrt(DRAW)
RHO, ALPHA = 910.0, 1.0e-7
SPAN = 3.0


def filaments_for(stress_mpa: float) -> float:
    area = F50 / (stress_mpa * 1e6)
    return area / (math.pi * (FINAL_UM / 2e6) ** 2)


print("=" * 70)
print("1. HOW MANY FILAMENTS, AND WHAT A SHOT WEIGHS")
print("=" * 70)
print(f"\n  die {DIE_UM:.0f} um drawn {DRAW:.0f}x -> {FINAL_UM:.0f} um filament\n")
print(f"{'drawn strength':>16}{'filaments':>12}{'3 m shot':>12}")
for stress in (300, 500, 800, 1200):
    n = filaments_for(stress)
    mass = n * math.pi * (FINAL_UM / 2e6) ** 2 * SPAN * RHO
    print(f"{stress:>13} MPa{n:>12.0f}{mass*1e3:>10.2f} g")
print("""
  Melt-spun drawn polyethylene runs 300-800 MPa; gel-spun reaches thousands.
  Take 500 MPa and 160 filaments, and a three-metre shot is under two grams.
  The 45 g of the fat-strand design is gone, and with it the heated tube, the
  big battery and most of the burn hazard.""")

print("=" * 70)
print("2. FLOW, AND WHY THE PUMP IS TINY")
print("=" * 70)
n = filaments_for(500)
die_area = n * math.pi * (DIE_UM / 2e6) ** 2
die_speed = LINE_SPEED / DRAW
flow = die_area * die_speed
print(f"\n  melt creeps through the die at {die_speed*1e3:.1f} mm/s, not {LINE_SPEED:.0f} m/s")
print(f"  total flow {flow*1e6:.2f} mL/s over {n:.0f} holes")
print(f"  a {SPAN:.0f} m shot lasts {SPAN/LINE_SPEED*1e3:.0f} ms and uses "
      f"{flow*SPAN/LINE_SPEED*1e6:.2f} mL")
print(f"  a 20 mL reservoir is {20/(flow*SPAN/LINE_SPEED*1e6):.0f} shots")

print("\n" + "=" * 70)
print("3. DOES IT CLOG? YES, IN SECONDS, AND ONLY ONE FIX WORKS")
print("=" * 70)
print("\n  the die hole freezes on the same clock everything else ran on:\n")
for d_um in (300, 600, 1500, 3000):
    print(f"    {d_um:>5} um hole freezes solid in {(d_um/2e6)**2/ALPHA:>6.1f} s")
print(f"""
  A {DIE_UM:.0f} um hole is solid {(DIE_UM/2e6)**2/ALPHA:.0f} seconds after the heat goes off, and a
  frozen die is not cleared by pressure - it is drilled. So the die cannot be
  allowed to cool between shots, which makes this a heated device that idles
  hot rather than one that warms up on demand.

  That is the real cost of the whole architecture, and it is not chemistry.""")

MASS_AL, CP_AL, CP_PE, LATENT = 0.10, 900.0, 2000.0, 100e3
DT = 145.0
warm = MASS_AL * CP_AL * DT + 0.02 * (CP_PE * DT + LATENT)
print(f"\n  warm-up from cold: {warm/1e3:.1f} kJ = {warm/3600:.1f} Wh")
print(f"    at 100 W that is {warm/100/60:.1f} minutes before the first shot")
for loss in (5, 10, 20):
    print(f"  idling at {loss:>2} W of loss: a 20 Wh pack holds it hot for "
          f"{20/loss:.1f} hours")
print("""
  The second clog is slower and worse. Polyethylene held at 165 C in air
  oxidises, and the gel it forms plugs a 1500 um hole from the inside over
  hours. Commercial polyethylene is sold with antioxidant already in it, which
  is what makes this hours rather than minutes; a nitrogen blanket over the
  melt removes it. Neither is exotic - both are what a real extruder does.""")
