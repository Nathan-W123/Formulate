"""The bench test that settles the jet question, with no chemistry in it.

Whether a 4.4 mm stream flies 3 m as a rope or breaks into drops is set by
viscosity, surface tension, density and speed - not by what the fluid is made
of. So it can be tested with corn syrup for the price of a bottle of corn
syrup, and the answer transfers to any resin at the same viscosity.

Run this to get the numbers; run the protocol it prints to get the answer.
"""

from __future__ import annotations

import math

RHO_S, SIGMA_S = 1400.0, 0.078      # corn syrup
RHO_W, SIGMA_W = 1000.0, 0.072      # water, the control
D, V, LAND = 4.4e-3, 9.0, 0.002
LN = 10.0


def breakup_distance(d, mu, rho, sigma, v):
    R = d / 2
    tau = 2 * math.sqrt(rho * R**3 / sigma) + 6 * mu * R / sigma
    return LN * tau / (2 * math.pi) * v


def drive_pressure(mu, rho):
    return 8 * mu * LAND * V / (D / 2) ** 2 + rho * V * V / 2


def ball_fall_time(mu, rho_f, d_ball=5e-3, drop=0.10, rho_s=7800.0):
    """Stokes terminal velocity - how you check the syrup is at 10 Pa.s."""
    r = d_ball / 2
    v = 2.0 / 9.0 * (rho_s - rho_f) * 9.81 * r**2 / mu
    return drop / v, v


print("=" * 70)
print("PREDICTION - what each fluid should do")
print("=" * 70)
print(f"\n{'fluid':<26}{'viscosity':>11}{'intact for':>13}{'verdict':>22}")
for name, mu, rho, sigma in (
        ("water (the control)", 0.001, RHO_W, SIGMA_W),
        ("thin syrup", 1.0, RHO_S, SIGMA_S),
        ("corn syrup at target", 10.0, RHO_S, SIGMA_S)):
    L = breakup_distance(D, mu, rho, sigma, V)
    v = "breaks up almost at once" if L < 0.5 else (
        "marginal" if L < 3.0 else "rope all the way")
    print(f"  {name:<24}{mu:>8} Pa.s{L:>11.1f} m  {v}")

print("\n  The control matters more than the test. Water MUST break up inside")
print("  half a metre. If it does not, the rig is wrong and the syrup result")
print("  means nothing.")

print("\n" + "=" * 70)
print("PROTOCOL")
print("=" * 70)
mu_t = 10.0
t_ball, v_ball = ball_fall_time(mu_t, RHO_S)
print(f"""
  1. Orifice. Drill 4.4 mm through a bottle cap or a short tube. Keep the
     straight section about 2 mm long - a long bore costs pressure.

  2. Fluid. Corn syrup, thinned with water until a 5 mm steel ball
     falls 100 mm in about {t_ball:.0f} seconds ({v_ball*1e3:.1f} mm/s). That is
     10 Pa.s and it is the only measurement that has to be careful.

  3. Drive. {drive_pressure(mu_t, RHO_S)/1e5:.1f} bar behind it gives {V:.0f} m/s at the orifice.
     A 2 L PET bottle and a bike pump. Aim along a corridor, horizontally,
     about 1.5 m off the floor.

  4. Film at 240 fps (any recent phone). Mark the floor at 0.5 m intervals.

  5. Run water first. It should shatter into drops within half a metre.
     If it flies, stop - the rig is not doing what you think.

  6. Run the syrup. Measure where the stream first breaks.

  READING THE RESULT
    intact past 3 m      the jet model holds. Everything downstream of it
                         in this project - the strand, the hot melt, the
                         sizing - stands on ground that has been checked.
    breaks at 1-3 m      the model is optimistic by a factor of a few.
                         Raise viscosity or shorten the range and re-run.
    breaks under 1 m     the model is wrong by an order of magnitude and
                         firing a liquid rope does not work at this scale.
                         That is worth knowing for the price of a bottle
                         of syrup.
""")
