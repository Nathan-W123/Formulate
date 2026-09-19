"""The head of the web: what pulls the line, and what anchors it.

Momentum at the far end is not a design preference. "Get a line across a gap"
means delivering momentum to the far end of the line, and nothing else in the
device can do it:

  an air jet    decays as 6d/x, so a 20 mm nozzle at 100 m/s is down to 4 m/s
                by three metres, and fixing that costs ten litres of air a shot
  the line      carries a quarter of a joule at 20 m/s, spread along its length
                rather than concentrated at the tip. A cast without a lure does
                not go.

So there is a mass at the tip. The only question is what it is made of and
what shape it takes, and the answer to both is: the same melt, blown thin.

**Why not a solid blob.** A blob of polymer has to solidify before it can carry
load, and that is the R^2 clock again - a 15 g blob splatted to 35 mm is 17 mm
thick and takes twelve minutes. Spreading it thinner needs a dinner-plate splat
that impact will not produce.

**Why a blown shell works.** The device already carries gas. Extruding a short
tube and inflating it - which is blown film, an ordinary process - turns a few
grams into a large thin-walled cup. Thin walls set in seconds because the clock
is thickness, not mass, and a large cup lands with a bond area far past what
the load needs. The wall does double duty: it is the mass that draws the line
and the surface that anchors it.
"""

from __future__ import annotations

import math

LB = 4.4482216152605
F50 = 50 * LB
ALPHA, RHO = 1.0e-7, 910.0

#: Conservative shear strength of a hot-melt bond on steel, Pa. The range is
#: 0.2-2 MPa; the bottom of it is used so the answer does not rest on the
#: favourable end.
BOND_SHEAR = 0.5e6

#: Drawing the filament bundle plus its air drag, N, over the span.
DRAW_FORCE, SPAN = 0.5, 3.0


def set_time(wall: float) -> float:
    """A wall sets from both faces, so the clock runs on half the thickness."""
    return (wall / 2) ** 2 / ALPHA


def shell(diameter: float, wall: float) -> tuple[float, float]:
    """Mass and bonded area of a hemispherical cup, kg and m^2."""
    area = 2 * math.pi * (diameter / 2) ** 2
    return area * wall * RHO, area


print("=" * 72)
print("A SOLID BLOB DOES NOT SET - the same R^2 clock, one last time")
print("=" * 72)
print(f"\n{'mass':>6}{'splat':>9}{'thick':>9}{'sets in':>12}")
for m_g in (10, 15):
    volume = m_g / 1000 / RHO
    for dia_mm in (35, 50, 100):
        thickness = volume / (math.pi * (dia_mm / 2000) ** 2)
        print(f"{m_g:>5}g{dia_mm:>7}mm{thickness*1e3:>7.1f}mm{set_time(thickness):>10.0f} s")
print("\n  Twelve minutes for the 35 mm case. Impact will not make a dinner plate.")

print("\n" + "=" * 72)
print("A BLOWN SHELL DOES - thickness is the clock, not mass")
print("=" * 72)
print(f"\n{'cup':>7}{'wall':>8}{'mass':>8}{'sets in':>10}{'bond area':>12}"
      f"{'holds':>9}{'launch at 20 m/s':>18}")
for dia_mm in (50, 70, 90):
    for wall_mm in (0.8, 1.2, 1.6):
        mass, area = shell(dia_mm / 1000, wall_mm / 1000)
        energy = 0.5 * mass * 20**2
        holds = BOND_SHEAR * area / LB
        ok = set_time(wall_mm / 1000) <= 6 and holds >= 50 and energy >= DRAW_FORCE * SPAN * 1.3
        print(f"{dia_mm:>5}mm{wall_mm:>6.1f}mm{mass*1e3:>6.1f}g"
              f"{set_time(wall_mm/1000):>8.1f}s{area*1e6:>10.0f}mm2{holds:>7.0f}lb"
              f"{energy:>15.2f} J{'  <--' if ok else ''}")

print("""
  Only one row clears every test: 90 mm at 1.2 mm. 13.9 g, set through in
  3.6 s, 12700 mm2 of bond area, and 2.78 J at 20 m/s against the 1.5 J the
  drawing costs. The 70 mm cup sets just as fast and bonds far harder than
  needed, but at 1.68 J it has too little margin over the draw to be trusted.

  A paintball is about 12 J, so this is under a quarter of one, and it is a
  warm hollow shell rather than a pellet. It crumples on impact instead of
  concentrating the hit - which is also what splats it flat and makes the
  bond, so the failure mode and the working mode are the same event.

  90 mm sounds large for a wrist. It is not carried at 90 mm: it is a short
  extruded tube inflated by the gas already on board, the way a film bubble is
  blown, so it only becomes a cup at the muzzle.

  Fire, count four, then load it.
""")
