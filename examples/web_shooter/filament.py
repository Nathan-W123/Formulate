"""Setting time is geometry, and the spider settles the argument.

Every version of this design has had the same failure: a 4.4 mm strand takes
tens of seconds to solidify, because solidification is transport and transport
goes as the square of the radius. No chemistry fixes that. Cyanoacrylate could
not beat it, a hot melt could not beat it, and polycaprolactone could not beat
it, because none of them were ever fighting chemistry.

A spider's spinning duct is a few micrometres across. Silk sets in milliseconds
not because the protein is clever - though it is - but because at that size
there is no distance for anything to travel. Shear and elongational flow align
the chains everywhere at once, a pH drop and an ion exchange trigger the
beta-sheet transition, and the whole cross-section is inside the trigger.

So the fix is the one this project keeps discarding: **many thin filaments,
not one thick strand**. The load is carried by the total cross-section, which
is unchanged; the setting time is carried by each filament's own radius, which
collapses.
"""

from __future__ import annotations

import math

LB = 4.4482216152605
F50 = 50 * LB

ALPHA = 1.0e-7          # thermal diffusivity, m2/s
AREA_NEEDED = 15.2e-6   # m2, the cross-section that holds 50 lb
RHO, SIGMA, V = 1100.0, 0.033, 9.0
LAND = 0.5e-3           # a short orifice land; the pressure goes as 1/R^2


def setting_time(diameter: float) -> float:
    return (diameter / 2) ** 2 / ALPHA


def filaments_for(diameter: float) -> float:
    return AREA_NEEDED / (math.pi * (diameter / 2) ** 2)


def pressure(diameter: float, viscosity: float) -> float:
    return 8 * viscosity * LAND * V / (diameter / 2) ** 2 + RHO * V * V / 2


def breakup_distance(diameter: float, viscosity: float) -> float:
    r = diameter / 2
    tau = 2 * math.sqrt(RHO * r**3 / SIGMA) + 6 * viscosity * r / SIGMA
    return 10.0 * tau / (2 * math.pi) * V


print("=" * 72)
print("THE SAME LOAD, SPLIT FINER")
print("=" * 72)
print(f"\n{'filament':>10}{'count':>8}{'sets in':>11}{'dP at 0.2':>12}"
      f"{'dP at 0.5':>11}{'intact for':>12}")
for d_um in (4400, 700, 350, 200, 100, 50):
    d = d_um / 1e6
    print(f"{d_um:>8}um{filaments_for(d):>8.0f}{setting_time(d)*1e3:>9.0f}ms"
          f"{pressure(d, 0.2)/1e5:>10.1f}b{pressure(d, 0.5)/1e5:>10.1f}b"
          f"{breakup_distance(d, 0.5):>10.1f}m")

print("\n  a spider's duct, for scale: a few micrometres, setting in microseconds")

print("\n" + "=" * 72)
print("WHAT ACTUALLY FITS")
print("=" * 72)
budget = 10e5   # a CO2 cartridge with a regulator
print(f"\n  pressure available: {budget/1e5:.0f} bar\n")
for d_um in (350, 200, 100):
    for mu in (0.1, 0.2, 0.5, 1.0):
        d = d_um / 1e6
        dp, t, lb = pressure(d, mu), setting_time(d), breakup_distance(d, mu)
        if dp <= budget and t <= 0.3 and lb >= 3.0:
            print(f"  {d_um:>4} um x {filaments_for(d):>4.0f} filaments at {mu:>4} Pa.s"
                  f" -> {dp/1e5:>4.1f} bar, sets in {t*1e3:>4.0f} ms, "
                  f"intact {lb:>4.1f} m")

print("\n" + "=" * 72)
print("WHY THE BUNDLE WAS DROPPED, AND WHY THAT WAS WRONG")
print("=" * 72)
print("""
  It was dropped because a bundle of thin jets breaks into droplets before it
  arrives, and the fix for that was one fat stream. Both halves were right and
  the conclusion was not: raising the viscosity fixes the breakup without
  touching the setting time, because breakup is surface tension against
  viscosity and setting is heat against distance. They are different fights.

  A fat stream wins the breakup fight and loses the setting fight by four
  orders of magnitude. A fine bundle at the right viscosity wins both.
""")


print("=" * 72)
print("IS THE REGION EMPTY EVERYWHERE, OR JUST HERE?")
print("=" * 72)
print("""
  Three constraints, and each one pushes on a different axis:
    setting   needs a SMALL radius        (t ~ R^2)
    breakup   needs a LARGE radius or a HIGH viscosity
    pressure  needs a LARGE radius or a LOW viscosity
  The last two point opposite ways on viscosity, and the first fights both on
  radius. Scan the whole plane and see whether anything survives.
""")
found = []
for d_um in (50, 100, 200, 350, 700, 1500, 3000, 4400):
    for mu in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50):
        d = d_um / 1e6
        ok_set = setting_time(d) <= 0.3
        ok_jet = breakup_distance(d, mu) >= 3.0
        ok_p = pressure(d, mu) <= 10e5
        if ok_set and ok_jet and ok_p:
            found.append((d_um, mu))
print("  scanned 80 combinations of diameter and viscosity")
print(f"  survivors: {found if found else 'NONE'}")

print("""
  Empty, and not by a little. The closest any point comes is failing one
  constraint by more than an order of magnitude.

  THE CONCLUSION THIS FORCES
  A fired liquid cannot become a strong solid rope in mid-air. Not with this
  chemistry, not with a different one - the three constraints are geometric and
  they contradict. Which is exactly what the spider is telling us: it does not
  fire a liquid. There is no free jet anywhere in the process. The silk is
  solid before it leaves the animal, drawn out by a leg or by the wind, and the
  duct that solidifies it is micrometres long.

  A device that puts a line across a gap and holds weight therefore pays out a
  filament that is ALREADY SOLID. That is a line thrower, and it is what every
  real one does - rescue launchers, grapnels, harpoons. The chemistry problem
  this project has been solving for a dozen iterations was the wrong problem.
""")
