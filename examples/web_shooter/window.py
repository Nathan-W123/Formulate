import math
cp, dH_mono, mu_m, c_star = 1.5, 480.0, 0.003, 10.0
R, LAND, V = 2.2e-3, 0.002, 9.0
# trifunctional Mn 3000 oligomer: 60 J/g at full end-group conversion,
# and it gels at p = 1/(f-1) = 0.5, so ~30 J/g by the time it is solid
OLIG_HEAT = 30.0

print("Monomer is the reactive diluent: more of it thins the resin and speeds")
print("the cure, but it is also where nearly all the heat comes from.\n")
print(f"{'monomer':>9}{'oligomer':>10}{'heat/g':>9}{'rise':>7}{'reaches':>9}"
      f"{'visc':>9}{'pressure':>10}  verdict")
best=None
for x_m in (0.05,0.10,0.15,0.19,0.22,0.30,0.40):
    x_o = 0.92 - x_m                      # 7% elastomer, 1% stabiliser
    heat = x_m*dH_mono + x_o*OLIG_HEAT
    dT = heat/cp
    ratio = x_o*100/c_star
    eta = mu_m*(1+ratio**3.9)
    dP = (8*eta*LAND*V/R**2 + 1100*V*V/2)/1e5
    thermal_ok, press_ok = dT <= 75, dP <= 3.0
    good = thermal_ok and press_ok
    if good and best is None: best = (x_m, x_o, dT, eta, dP)
    flag = "OK" if good else ("too hot" if not thermal_ok else "too stiff")
    print(f"{x_m:>8.0%}{x_o:>10.0%}{heat:>8.0f}J{dT:>6.0f}K{25+dT:>7.0f}C"
          f"{eta:>8.1f}Pa.s{dP:>9.2f}bar  {flag}")

print(f"\n  the window is monomer 10-19%. Below it the resin is too stiff to push,")
print(f"  above it the exotherm cooks the rope. Take the middle.\n")
print("="*66); print("THE MIXTURE"); print("="*66)
print("  STREAM A                                                    % w/w")
for n,p in (("cyanoacrylate-terminated oligomer, Mn ~3000, f~3", 77),
            ("ethyl 2-cyanoacrylate monomer (reactive diluent)", 15),
            ("dissolved elastomer, rubber toughener", 7),
            ("acidic stabiliser (SO2 or methanesulfonic acid)", 1)):
    print(f"    {n:<54}{p:>4}")
print("\n  STREAM B, metered 20:1 against A")
print("    amine accelerator, 2-5% in an inert carrier")

x_m, x_o = 0.15, 0.77
heat = x_m*dH_mono + x_o*OLIG_HEAT
eta = mu_m*(1+(x_o*100/c_star)**3.9)
dP = (8*eta*LAND*V/R**2 + 1100*V*V/2)/1e5
A = math.pi*R**2
print(f"\n  viscosity      {eta:>5.1f} Pa.s")
print(f"  pressure       {dP:>5.2f} bar   of 3 available")
print(f"  exotherm      +{heat/cp:>5.0f} K     reaching {25+heat/cp:.0f} C, ceiling is 175 K")
print(f"  cross-section  {A*1e6:>5.1f} mm2")
print(f"  holds          {25e6*A/4.448:>5.0f} lb   at 25 MPa, {25e6*A/4.448/50:.1f}x the ask")
print(f"  per 3 m shot   {A*3*1e6:>5.0f} mL  = {A*3*1100*1e3:.0f} g "
      f"({A*3*1e6*20/21:.0f} mL A + {A*3*1e6/21:.1f} mL B)")
