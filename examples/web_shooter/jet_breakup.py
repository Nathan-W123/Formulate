import math
rho, sigma, v = 1100.0, 0.033, 5.0
rho_air = 1.2
LN = 10.0   # e-foldings from a ~0.1% initial perturbation to pinch-off

def breakup(d, mu, v=v):
    R = d/2
    # Weber's viscous correction: inertia term + viscous term
    tau = 2*math.sqrt(rho*R**3/sigma) + 6*mu*R/sigma
    t = LN*tau/ (2*math.pi)     # fastest mode, order-of-magnitude
    return t, t*v

print("Which regime are we even in?")
for d_mm in (0.7, 4.4):
    d = d_mm/1e3
    We_l = rho*v*v*d/sigma
    We_g = rho_air*v*v*d/sigma
    print(f"  d={d_mm} mm: liquid We={We_l:.0f}, gas We={We_g:.2f} "
          f"({'aerodynamic stripping matters' if We_g>1 else 'gas is not the problem; capillarity is'})")

print(f"\nCapillary breakup, single jet, at {v} m/s")
print(f"{'jet':>8}{'viscosity':>12}{'Oh':>7}{'breaks after':>14}{'= distance':>13}{'':>4}")
for d_mm in (0.7, 4.4):
    for mu in (0.01, 0.1, 1.0, 5.0, 20.0):
        d = d_mm/1e3
        Oh = mu/math.sqrt(rho*sigma*d)
        t, L = breakup(d, mu)
        verdict = "reaches 3 m" if L >= 3.0 else ""
        print(f"{d_mm:>6}mm{mu:>10}Pa.s{Oh:>7.1f}{t*1e3:>12.0f}ms{L:>11.2f} m  {verdict}")

print("\nHow fast must it cure to beat its own breakup?")
for d_mm, mu in ((4.4,1.0),(4.4,5.0),(4.4,20.0)):
    t,L = breakup(d_mm/1e3, mu)
    print(f"  {d_mm} mm at {mu} Pa.s: must gel within {t*1e3:.0f} ms "
          f"(it travels {L:.2f} m first)")

print("\nWhat a spider does instead:")
print("  it does not shoot silk. It anchors the thread and pulls it out with a leg,")
print("  or lets the wind draw it. Drawing a filament from a reservoir has no")
print("  Rayleigh-Plateau limit at all, because the thread is solid before it is long.")

print("\n" + "="*64)
print("CORRECTION: one fat orifice beats the bundle on BOTH counts")
print("="*64)
print("  I earlier said many small holes beat one big one on pressure.")
print("  That is backwards - dP = 8*mu*L*v/R^2, so a SMALLER R costs MORE.")
print("  The bundle was only ever justified by drying, and a reactive cure")
print("  removes that reason entirely.\n")
Q_target = 76e-6      # m3/s, same throughput as before
for label, n, d_mm in (("bundle  40 x 0.7 mm", 40, 0.7), ("single  4.4 mm", 1, 4.4)):
    R = d_mm/2e3
    A = n*math.pi*R**2
    v_ = Q_target/A
    for mu in (1.0, 5.0):
        dP = 8*mu*0.002*v_/R**2
        t, L = breakup(d_mm/1e3, mu, v_)
        print(f"  {label:<22} mu={mu:>4} Pa.s  v={v_:.1f} m/s  "
              f"dP={dP/1e5:>6.2f} bar   intact to {L:>5.1f} m")
print("\n  and the range you can buy with the 13 bar you have:")
R = 2.2e-3
for v_ in (5, 10, 15, 20):
    dP = 8*5.0*0.002*v_/R**2
    t, L = breakup(4.4e-3, 5.0, v_)
    print(f"    {v_:>2} m/s: {dP/1e5:.2f} bar, jet intact to {L:.0f} m, "
          f"ballistic reach {v_*v_*math.sin(math.radians(90))/9.81:.0f} m")
