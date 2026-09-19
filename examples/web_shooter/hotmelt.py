"""Reopening the route that was dismissed for the wrong reason.

Hot melt was ruled out early on "the hardware has no heater", which is a
choice rather than a constraint - a heater is the cheapest part on the list.
The real reason to revisit it is an asymmetry that the chemical route does
not have.

A curing strand has to react all the way through before any of it is solid,
so its clock is the bulk clock, R^2/alpha, and that is 48 s on a 4.4 mm
strand. A cooling strand does not. Its surface reaches air temperature almost
at once and a solid skin starts growing inward as sqrt(alpha*t) - and a solid
skin cannot be pinched by surface tension. Breakup stops the moment the skin
forms, not when the strand is solid.

So the question is whether a bought glue stick skins fast enough to survive
the flight, and whether the skin alone carries the strand's own weight until
the core catches up.
"""

from __future__ import annotations

import math

LB = 4.4482216152605
F50 = 50 * LB

R_JET = 2.2e-3
A_JET = math.pi * R_JET**2          # 15.2 mm2, the same orifice as before
ALPHA = 1.0e-7                      # thermal diffusivity, m2/s
RHO = 980.0
FLIGHT = 0.333


def skin(t):
    """Solidified depth, m. Stefan-like growth, the usual sqrt(alpha*t)."""
    return min(math.sqrt(ALPHA * t), R_JET)


def load_bearing_area(t):
    d = skin(t)
    return math.pi * (R_JET**2 - (R_JET - d) ** 2)


print("=" * 72)
print("1. DOES THE SKIN FORM IN TIME? (it is the skin that stops pinch-off)")
print("=" * 72)
print(f"\n{'time':>9}{'skin':>10}{'solid area':>13}{'holds':>10}   ")
for t in (0.01, 0.05, FLIGHT, 1.0, 3.0, 10.0, 30.0, 48.0):
    d, a = skin(t), load_bearing_area(t)
    note = "  <- lands here" if abs(t - FLIGHT) < 1e-9 else ""
    print(f"{t:>8.2f}s{d*1e6:>8.0f}um{a*1e6:>11.1f}mm2"
          f"{25e6*a/LB:>9.0f}lb{note}")
print("\n  it needs to carry only its own weight on landing - 3 m of strand is")
print(f"  50 g, which is {0.050*9.81:.2f} N, or {0.050*9.81/LB:.2f} lb. The 180 um skin at")
print("  touchdown is three orders of magnitude past that.")
print("  Full payload takes ~10 s of standing. Fire, wait, then load it.")

print("\n" + "=" * 72)
print("2. IS A BOUGHT GLUE STICK STRONG ENOUGH?")
print("=" * 72)
print(f"\n{'hot melt grade':<40}{'UTS':>8}{'holds on 15.2 mm2':>20}")
for name, uts, temp in (
        ("EVA craft glue stick (low temp)", 6, "120 C"),
        ("EVA craft glue stick (high temp)", 10, "190 C"),
        ("polyolefin / APAO", 12, "170 C"),
        ("polyamide (3M Jet-melt, Macromelt)", 28, "200 C"),
        ("thermoplastic polyurethane hot melt", 40, "180 C")):
    holds = uts * 1e6 * A_JET / LB
    flag = "  MEETS 50 lb" if holds >= 50 else ""
    print(f"  {name:<38}{uts:>5} MPa{holds:>13.0f} lb{flag}   applied at {temp}")

print("\n" + "=" * 72)
print("3. THE HARDWARE DELTA")
print("=" * 72)
Q = A_JET * 9.0
shot_g = A_JET * 3.0 * RHO * 1e3
melt_J = shot_g * (2.0 * 170 + 100)         # sensible + latent
for mu in (2.0, 10.0, 20.0):
    dP = 8 * mu * 0.002 * 9.0 / R_JET**2 + RHO * 81.0 / 2
    print(f"  melt at {mu:>4} Pa.s -> {dP/1e5:>5.2f} bar   "
          f"{'fits the 6 bar cartridge' if dP < 6e5 else 'too stiff'}")
print(f"\n  flow {Q*1e6:.0f} mL/s, a 3 m shot is {shot_g:.0f} g in {3.0/9.0*1e3:.0f} ms")
print("  that is far past what a glue gun melts on demand, so the reservoir has")
print(f"  to be pre-melted: {melt_J/1e3:.0f} kJ to bring {shot_g:.0f} g up from cold,")
print(f"  = {melt_J/3600:.1f} Wh, or {melt_J/300:.0f} W over five minutes.")
print("  a 100 W cartridge heater in an aluminium tube, and a small LiPo.")

print("\n" + "=" * 72)
print("WHAT THIS BUYS, AND WHAT IT COSTS")
print("=" * 72)
print("  buys:  every part is a catalogue item. No commissioned resin, no")
print("         two-part metering, no static mixer, no pot life, no exotherm,")
print("         and it bonds on contact because it lands molten.")
print("  costs: a heater and a battery; a nozzle that must stay hot or it")
print("         plugs solid; a burn hazard at 200 C; and ~10 s of standing")
print("         before the strand carries its rated load.")
