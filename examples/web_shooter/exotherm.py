print("="*66)
print("CAN ACCELERATED CA GEL A 4.4 mm ROPE IN 333 ms?")
print("="*66)
print("Anionic CA polymerisation is initiation-limited and essentially")
print("non-terminating, so doubling accelerator halves the set time. The")
print("chemistry allows it. The question is whether the heat does.\n")

# ethyl 2-cyanoacrylate
M   = 125.13      # g/mol
dH  = 60e3        # J/mol, vinyl/acrylate propagation enthalpy (60-80 typical)
cp  = 1.5         # J/g/K
alpha = 1.0e-7    # m2/s thermal diffusivity
rho = 1100.0

dT_full = dH/(M*cp)
print(f"heat of polymerisation      {dH/1e3:.0f} kJ/mol over {M:.0f} g/mol = {dH/M:.0f} J/g")
print(f"ADIABATIC temperature rise  {dT_full:.0f} K at full conversion\n")

print("Is a curing rope adiabatic? Compare cure time to heat-escape time:")
for d_mm, label in ((0.1,"a glue bond line"),(0.5,""),(4.4,"our rope")):
    R = d_mm/2e3
    t_th = R**2/alpha
    print(f"  {d_mm:>5} mm {label:<18} heat escapes in {t_th*1e3:>9.1f} ms")
print("\n  -> a 100 um bond line dumps its heat in 25 ms, so it never heats up.")
print("     THAT is why superglue works in a joint and smokes on a cotton ball.")
print("     Our 4.4 mm rope needs 48 s to shed heat and would cure in 0.33 s,")
print("     so it is fully adiabatic. Every joule stays in.\n")

print("Temperature the rope reaches, vs conversion:")
print(f"{'conversion':>11}{'dT':>8}{'final T':>10}   verdict")
for x in (0.05,0.10,0.15,0.20,0.30,0.50,1.00):
    dT = x*dT_full; T = 25+dT
    if T > 200: v = "DECOMPOSES - polycyanoacrylate unzips above ~200 C"
    elif T > 100: v = "boils the monomer, foams the rope"
    elif T > 60: v = "hot but survivable"
    else: v = "fine"
    print(f"{x:>10.0%}{dT:>7.0f}K{T:>9.0f}C   {v}")

x_safe = (200-25)/dT_full
print(f"\n  ceiling: {x_safe:.0%} conversion before it cooks itself.")
print("  full cure in flight is not slow chemistry - it is thermally impossible.")

print("\n" + "="*66)
print("SO WHAT IS ACTUALLY AVAILABLE")
print("="*66)
print("It does not need to cure. It needs to stop flowing. A chain-growth")
print("polymer reaches high molar mass at low conversion, so a few percent")
print("converted is already high-MW polymer dissolved in its own monomer:\n")
for x in (0.05, 0.10, 0.15):
    dT = x*dT_full
    print(f"  {x:>4.0%} conversion: +{dT:>3.0f} K -> {25+dT:>3.0f} C, "
          f"and ~{x*100:.0f}% w/w polymer in monomer")
print("\n  10-15% of a high-MW polymer in its own monomer is a thick gel - enough")
print("  to resist sag and to stop Rayleigh pinch-off, which is all flight needs.")
print("  It then finishes curing on the wall over minutes, shedding heat slowly.")
print("\n  This is a calculation, not an experiment. What it establishes is that")
print("  the strong claim is dead and the weak one is not: full cure in 333 ms")
print("  is ruled out by thermodynamics, partial gelation is not.")

print("\n" + "="*66)
print("IS 10-15% ACTUALLY THICK ENOUGH? (the weak claim, checked)")
print("="*66)
# overlap concentration from intrinsic viscosity, c* ~ 1/[eta]
mu_mono = 0.003          # Pa.s, CA monomer is thin
for Mw, eta_int in ((1e5, 100.0), (5e5, 300.0), (1e6, 500.0)):
    c_star = 1.0/eta_int*100          # g/100mL -> % w/v
    for c in (5.0, 10.0, 15.0):
        ratio = c/c_star
        # entangled semidilute scaling, eta_sp ~ (c/c*)^3.9
        eta = mu_mono*(1+ratio**3.9)
        print(f"  Mw={Mw:.0e}  c*={c_star:.2f}%   at {c:>4.0f}% -> c/c* = {ratio:>5.1f}, "
              f"viscosity ~ {eta:>9.0f} Pa.s")
print("\n  5% is marginal: at the low end of molar mass it gives ~2 Pa.s, under the")
print("  ~5 Pa.s the Rayleigh calculation wanted. 10% clears it for every molar")
print("  mass here, by a factor of five at worst.")
print("\n  OPERATING WINDOW: gel to ~10% conversion in flight.")
print("    thick enough  - 24 Pa.s at worst, against 5 Pa.s needed")
print("    cool enough   - +32 K, reaching 57 C, against a 55% ceiling")
print("  The window is real but it is not wide, and only an experiment closes it.")
