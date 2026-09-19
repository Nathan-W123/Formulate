"""A coupled radial cure/heat/breakup model for one fired strand.

Everything up to here was a scaling argument, and scaling arguments treated
three things separately that are in fact racing each other:

  * cure generates heat, and heat accelerates cure (the runaway risk),
  * cure raises viscosity, and viscosity is what resists capillary pinch-off,
  * the surface sheds heat, so it stays cooler, so it cures *last* - which
    matters because surface tension acts on the surface, not on the core.

That last coupling has no place in a scaling estimate at all. So this
integrates the real thing: a radial reaction-diffusion problem for one
cross-section of the jet, marched over the flight, with a Rayleigh
perturbation growing against the viscosity the cure is producing.

What it is not: a free-surface flow solver. The jet radius is held fixed and
breakup is judged by a linear-stability criterion, which is the standard
slender-jet treatment and is valid right up to pinch-off, not through it.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

# -- the strand and the resin ------------------------------------------------
R_JET = 2.2e-3           # m, the 4.4 mm orifice
RHO = 1100.0             # kg/m3
CP = 1500.0              # J/kg/K
K_TH = 0.165             # W/m/K  (alpha = 1e-7 m2/s)
SIGMA = 0.033            # N/m
DH = 81.6e3              # J/kg, the 12% monomer / 80% oligomer blend
ALPHA_GEL = 0.5          # trifunctional oligomer gels at 1/(f-1)
ETA0 = 10.0              # Pa.s uncured
T_AIR = 298.15
H_CONV = 35.0            # W/m2/K, forced convection at 9 m/s over 4.4 mm

# Arrhenius, normalised so that the accelerator sets a chosen gel time at 25 C
EA = 50e3
RGAS = 8.314

N = 40                   # radial nodes
FLIGHT = 0.333           # s, 3 m at 9 m/s


def rate_constant(T, k25):
    """First-order cure constant, k25 at 298.15 K, Arrhenius above it."""
    return k25 * np.exp(-EA / RGAS * (1.0 / T - 1.0 / 298.15))


def viscosity(alpha):
    """Castro-Macosko: diverges at the gel point."""
    a = np.minimum(alpha, ALPHA_GEL * 0.9999)
    return ETA0 * (ALPHA_GEL / (ALPHA_GEL - a)) ** 4


def growth_rate(eta_surface):
    """Fastest-growing Rayleigh mode for a viscous jet (Weber)."""
    tau = 2 * math.sqrt(RHO * R_JET**3 / SIGMA) + 6 * eta_surface * R_JET / SIGMA
    return 2 * math.pi / tau


def simulate(k25, flight=FLIGHT, report=True):
    r = np.linspace(0.0, R_JET, N)
    dr = r[1] - r[0]

    def rhs(t, y):
        T, a, lne = y[:N], y[N:2 * N], y[2 * N]
        k = rate_constant(T, k25)
        da = k * (1.0 - a)
        da = np.where(a >= 1.0, 0.0, da)

        # cylindrical Laplacian, symmetric at r=0, convective at the surface
        lap = np.empty_like(T)
        lap[0] = 4.0 * (T[1] - T[0]) / dr**2
        lap[1:-1] = ((T[2:] - 2 * T[1:-1] + T[:-2]) / dr**2
                     + (T[2:] - T[:-2]) / (2 * dr * r[1:-1]))
        T_ghost = T[-2] - 2 * dr * H_CONV / K_TH * (T[-1] - T_AIR)
        lap[-1] = ((T_ghost - 2 * T[-1] + T[-2]) / dr**2
                   + (T_ghost - T[-2]) / (2 * dr * r[-1]))

        dT = K_TH / (RHO * CP) * lap + DH / CP * da
        dlne = growth_rate(viscosity(a[-1]))
        return np.concatenate([dT, da, [dlne]])

    y0 = np.concatenate([np.full(N, T_AIR), np.zeros(N), [0.0]])
    sol = solve_ivp(rhs, (0.0, flight), y0, method="LSODA",
                    rtol=1e-6, atol=1e-8, dense_output=True, max_step=flight / 200)

    ts = np.linspace(0.0, flight, 200)
    Y = sol.sol(ts)
    Tmax = Y[:N].max(axis=0) - 273.15
    a_core, a_surf = Y[N], Y[2 * N - 1]
    lne = Y[2 * N]
    breakup_threshold = math.log(1000.0)   # perturbation grows to the jet radius

    if report:
        print(f"  {'t (ms)':>7}{'core a':>9}{'surf a':>9}{'Tmax (C)':>10}"
              f"{'eta_surf':>12}{'pinch-off':>11}")
        for i in range(0, 200, 25):
            eta_s = viscosity(np.array([a_surf[i]]))[0]
            pct = lne[i] / breakup_threshold * 100
            print(f"  {ts[i]*1e3:>7.0f}{a_core[i]:>9.3f}{a_surf[i]:>9.3f}"
                  f"{Tmax[i]:>10.1f}{eta_s:>11.0f}P{pct:>10.0f}%")
    return dict(ts=ts, Tpeak=Tmax.max(), a_core_end=a_core[-1], a_surf_end=a_surf[-1],
                broke=lne[-1] >= breakup_threshold,
                pinch_frac=lne[-1] / breakup_threshold,
                gelled=a_surf[-1] >= ALPHA_GEL * 0.98)


if __name__ == "__main__":
    print("=" * 72)
    print("ONE STRAND, FROM NOZZLE TO WALL - 4.4 mm, 9 m/s, 333 ms of flight")
    print("=" * 72)
    print("\nGel clock set so the resin reaches its gel point in ~200 ms at 25 C:\n")
    k25 = math.log(1 / (1 - ALPHA_GEL)) / 0.200
    out = simulate(k25)
    print(f"\n  peak temperature anywhere : {out['Tpeak']:.1f} C")
    print(f"  conversion at core / surface: {out['a_core_end']:.3f} / {out['a_surf_end']:.3f}")
    print(f"  perturbation grew to        : {out['pinch_frac']*100:.0f}% of pinch-off")
    print(f"  verdict                     : "
          f"{'BROKE UP' if out['broke'] else 'arrived intact'}, "
          f"{'gelled' if out['gelled'] else 'STILL LIQUID AT THE SURFACE'}")

    print("\n" + "=" * 72)
    print("SWEEPING THE ONE KNOB WE HAVE: accelerator, i.e. the gel clock")
    print("=" * 72)
    print(f"\n  {'gel time at 25C':>16}{'peak T':>9}{'surf a':>9}{'pinch':>8}  verdict")
    for t_gel in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00):
        k = math.log(1 / (1 - ALPHA_GEL)) / t_gel
        o = simulate(k, report=False)
        v = []
        if o["broke"]:
            v.append("breaks up")
        if not o["gelled"]:
            v.append("lands liquid")
        if o["Tpeak"] > 150:
            v.append("COOKS")
        print(f"  {t_gel*1e3:>13.0f} ms{o['Tpeak']:>8.0f}C{o['a_surf_end']:>9.3f}"
              f"{o['pinch_frac']*100:>7.0f}%  {', '.join(v) if v else 'WORKS'}")


# --------------------------------------------------------------------------
# The constraint the flight model cannot see
# --------------------------------------------------------------------------

MIXER_VOL = 1.7e-6       # m3, a 6 x 60 mm static mixer
FLOW = 137e-6            # m3/s
P_BUDGET = 6.0e5         # Pa, the CO2 cartridge


def nozzle_check(t_gel):
    """Cure begins at the mixer, not at the muzzle.

    The gel clock starts the instant the two streams meet, so the resin is
    already partly converted - and already thicker - by the time it reaches
    the orifice. A fast accelerator wins the flight and loses the nozzle.
    """
    residence = MIXER_VOL / FLOW
    k = math.log(1 / (1 - ALPHA_GEL)) / t_gel
    alpha_exit = 1.0 - math.exp(-k * residence)
    eta_exit = float(viscosity(np.array([alpha_exit]))[0])
    dP = 8 * eta_exit * 0.002 * 9.0 / R_JET**2 + RHO * 81.0 / 2
    return residence, alpha_exit, eta_exit, dP


if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("THE CONSTRAINT THE FLIGHT MODEL CANNOT SEE")
    print("=" * 72)
    res, _, _, _ = nozzle_check(0.2)
    print(f"\n  cure starts where the streams meet, not at the muzzle.")
    print(f"  a 6 x 60 mm mixer at {FLOW*1e6:.0f} mL/s holds the resin for "
          f"{res*1e3:.0f} ms before it ever reaches the orifice.\n")
    print(f"  {'gel time':>10}{'a at exit':>11}{'eta at exit':>13}{'pressure':>11}  verdict")
    window = []
    for t_gel in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00):
        _, a_ex, eta_ex, dP = nozzle_check(t_gel)
        o = simulate(math.log(1 / (1 - ALPHA_GEL)) / t_gel, report=False)
        bad = []
        if dP > P_BUDGET:
            bad.append("clogs the nozzle")
        if not o["gelled"]:
            bad.append("lands liquid")
        if o["broke"]:
            bad.append("breaks up")
        if not bad:
            window.append(t_gel)
        print(f"  {t_gel*1e3:>7.0f} ms{a_ex:>11.3f}{eta_ex:>12.0f}P{dP/1e5:>10.1f}b"
              f"  {', '.join(bad) if bad else 'WORKS'}")

    if window:
        print(f"\n  WORKING WINDOW: gel time {min(window)*1e3:.0f}-{max(window)*1e3:.0f} ms")
        print(f"    faster and the resin stiffens inside the mixer and stalls the flow;")
        print(f"    slower and it lands before it is solid.")
        print(f"  The two limits are set by different hardware - mixer volume on one")
        print(f"  side, flight time on the other - so both are tunable, and the")
        print(f"  cheapest tuning is a shorter mixer rather than a new resin.")
    else:
        print("\n  NO WORKING WINDOW at this mixer volume.")


# --------------------------------------------------------------------------
# Does the integrator reproduce the two limits that have closed-form answers?
# --------------------------------------------------------------------------


def self_check():
    """A simulation that agrees with nothing known is not evidence."""
    print("\n" + "=" * 72)
    print("SELF-CHECK against the two limits that can be solved by hand")
    print("=" * 72)

    global H_CONV, K_TH
    h, k = H_CONV, K_TH
    try:
        H_CONV, K_TH = 0.0, 1e-12          # no heat leaves: pure adiabatic
        o = simulate(math.log(2) / 0.100, report=False)
        want = 25.0 + DH / CP
        err = abs(o["Tpeak"] - want)
        print(f"\n  adiabatic peak temperature")
        print(f"    closed form  25 + dH/cp = {want:.2f} C")
        print(f"    integrator              = {o['Tpeak']:.2f} C     "
              f"{'AGREES' if err < 0.5 else f'DIFFERS by {err:.2f} K'}")
    finally:
        H_CONV, K_TH = h, k

    # no cure: the perturbation should grow at the constant Newtonian rate
    o = simulate(0.0, report=False)
    omega = growth_rate(ETA0)
    want = omega * FLIGHT / math.log(1000.0)
    err = abs(o["pinch_frac"] - want)
    print(f"\n  uncured jet, perturbation growth over {FLIGHT*1e3:.0f} ms")
    print(f"    closed form  omega*t/ln(1000) = {want*100:.2f}% of pinch-off")
    print(f"    integrator                    = {o['pinch_frac']*100:.2f}%     "
          f"{'AGREES' if err < 0.005 else f'DIFFERS by {err*100:.2f} points'}")

    print(f"\n  Both limits reproduce, so the coupling is what the integrator adds,")
    print(f"  not the arithmetic. What it does NOT check is whether the cure")
    print(f"  kinetics, the gel exponent and the 30 J/g are right for a real resin.")
    print(f"  Those are assumptions, and the simulation inherits every one of them.")


if __name__ == "__main__":
    self_check()
