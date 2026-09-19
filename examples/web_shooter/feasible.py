import math
LB=4.4482216152605; F=50*LB
rho, sigma, alpha = 1100.0, 0.033, 1.0e-7
A_NEED = 15.2e-6        # m2, for 50 lb at SF 1.7 against 25 MPa
L_SPAN, V, P_AVAIL, LAND = 3.0, 9.0, 3.0e5, 0.002

print("="*70)
print("FIRST: can a LIQUID rope span 3 m at all? (I claimed it could)")
print("="*70)
mass = A_NEED*L_SPAN*rho
stress = mass*9.81/A_NEED
print(f"  3 m of rope weighs {mass*1e3:.0f} g, hanging from its own top: {stress/1e3:.0f} kPa")
print(f"  a viscous fluid at that stress extends at sigma/(3*eta):\n")
for eta in (5, 1e3, 1e5, 1e8, 1e9):
    rate = stress/(3*eta)
    print(f"    eta = {eta:>8.0e} Pa.s -> strain rate {rate:>10.2e} /s "
          f"= {rate*100:>10.1e} % per second   {'OK' if rate < 0.01 else 'it falls'}")
print(f"\n  -> needs ~1e8 Pa.s, which is a SOLID. 'Arrives liquid, cures on the")
print(f"     wall' was wrong: it would run down the wall and drip. It has to be")
print(f"     solid when it lands, so it must fully cure IN FLIGHT.")

print("\n" + "="*70)
print("SO: full cure in flight, which means shedding the exotherm in flight")
print("="*70)
t_flight = L_SPAN/V
print(f"  flight time {t_flight*1e3:.0f} ms. A filament sheds heat in R^2/alpha,")
print(f"  so it must be thin enough that R^2/alpha <= {t_flight*1e3:.0f} ms:\n")
d_thermal = 2*math.sqrt(alpha*t_flight)
print(f"    THERMAL CEILING: filament diameter <= {d_thermal*1e3:.2f} mm\n")

print("="*70)
print("THE FEASIBLE REGION, over filament diameter and viscosity")
print("="*70)
print("  every row keeps total area at 15.2 mm2 by using n filaments\n")
print(f"{'d':>7}{'eta':>9}{'n':>7}{'dP':>10}{'breakup':>10}{'sheds heat':>12}  verdict")
LN=10.0
def breakup_len(d, mu, v):
    R=d/2
    tau = 2*math.sqrt(rho*R**3/sigma) + 6*mu*R/sigma
    return LN*tau/(2*math.pi)*v

ok_any=False
for d_mm in (0.2, 0.3, 0.5, 1.0, 2.0, 4.4):
    d=d_mm/1e3; R=d/2
    n = A_NEED/(math.pi*R**2)
    for mu in (0.1, 1.0, 5.0, 20.0):
        dP = 8*mu*LAND*V/R**2 + rho*V*V/2
        Lb = breakup_len(d, mu, V)
        t_th = R**2/alpha
        c1 = dP <= P_AVAIL
        c2 = Lb >= L_SPAN
        c3 = t_th <= t_flight
        good = c1 and c2 and c3
        ok_any = ok_any or good
        flags = ("P" if not c1 else ".")+("B" if not c2 else ".")+("T" if not c3 else ".")
        print(f"{d_mm:>6}mm{mu:>7}Pa.s{n:>7.0f}{dP/1e5:>9.1f}b{Lb:>9.1f}m{t_th*1e3:>10.0f}ms"
              f"  {'FEASIBLE' if good else 'fails '+flags}")
print(f"\n  key: P = over pressure budget, B = breaks up early, T = too thick to")
print(f"       shed its own cure heat")
print(f"\n  ANY FEASIBLE COMBINATION? {'yes' if ok_any else 'NO - the region is empty'}")

print("\n" + "="*70)
print("WHICH CONSTRAINT IS CHEMISTRY, AND WHICH IS GEOMETRY?")
print("="*70)
print("  pressure and breakup are geometry+viscosity: the 4.4 mm single stream")
print("  already passes both at 1-5 Pa.s. Look at the table - it fails ONLY on T.")
print("  T is the one that is chemistry, and it has two terms:")
print("     shed the heat (needs a thin filament), OR")
print("     make less heat (needs a different feedstock)\n")

cp = 1.5
dH_mono = 480.0   # J/g for neat cyanoacrylate monomer
print(f"  adiabatic rise = heat/gram / cp. Neat CA monomer: "
      f"{dH_mono:.0f}/{cp} = {dH_mono/cp:.0f} K -> cooks.")
print(f"  tolerable rise is ~100 K, so we need heat/gram below {100*cp:.0f} J/g,")
print(f"  i.e. a reactive mass fraction under {100*cp/dH_mono:.0%}.\n")

print("  That is what a PREPOLYMER is: most of the mass already polymerised, so")
print("  only the remainder reacts and only that remainder makes heat.\n")
print(f"{'reactive frac':>14}{'heat/g':>9}{'adiabatic rise':>16}{'reaches':>10}")
for f in (1.00, 0.70, 0.50, 0.30, 0.20):
    h = f*dH_mono; dT = h/cp
    v = "cooks" if dT>175 else ("hot" if dT>75 else "fine")
    print(f"{f:>13.0%}{h:>8.0f}J{dT:>14.0f} K{25+dT:>8.0f} C   {v}")

print("\n  and does a 70% oligomer loading still extrude? Viscosity from")
print("  semidilute scaling, oligomer Mn ~ 3000 so c* is high (~10% w/v):")
mu_m = 0.003
for Mn, c_star in ((3000,10.0),(10000,3.0)):
    for c in (50.0, 70.0):
        ratio = c/c_star
        eta = mu_m*(1+ratio**3.9)
        print(f"    Mn={Mn:>6}  {c:>3.0f}% loading -> c/c* = {ratio:>4.1f}, "
              f"eta ~ {eta:>8.1f} Pa.s")

print("\n" + "="*70)
print("RE-TEST: 4.4 mm single stream, 30% reactive oligomer resin")
print("="*70)
import math
d, mu = 4.4e-3, 5.0
R=d/2
dP = 8*mu*LAND*V/R**2 + rho*V*V/2
Lb = breakup_len(d, mu, V)
dT = 0.30*dH_mono/cp
print(f"  pressure   {dP/1e5:>6.2f} bar   vs {P_AVAIL/1e5:.0f} available     {'PASS' if dP<=P_AVAIL else 'FAIL'}")
print(f"  breakup    {Lb:>6.1f} m     vs {L_SPAN:.0f} needed        {'PASS' if Lb>=L_SPAN else 'FAIL'}")
print(f"  exotherm   +{dT:>5.0f} K     reaching {25+dT:.0f} C        {'PASS' if dT<=100 else 'FAIL'}")
print(f"  area       {math.pi*R**2*1e6:>6.1f} mm2   vs {A_NEED*1e6:.1f} needed     "
      f"{'PASS' if math.pi*R**2>=A_NEED*0.99 else 'FAIL'}")
print(f"\n  the region is no longer empty. The fix was never a different reaction -")
print(f"  it was reacting less of the mass, by starting from an oligomer.")
