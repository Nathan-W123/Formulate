import math
cp, dH_mono, dH_bond = 1.5, 480.0, 60e3

print("="*68)
print("FIRST: does an INERT oligomer work? (what I implied last time)")
print("="*68)
print("  75% inert filler + 25% reactive binder. The filler is Mn 3000, which")
print("  is far below PMMA's entanglement mass of 9200 - so it cannot carry")
print("  load through a network. It is a plasticiser, not a reinforcement.\n")
for f_react in (0.25, 0.30, 0.50):
    # crude: strength scales with the load-bearing (networked) fraction
    print(f"    {f_react:.0%} reactive -> roughly {25*f_react:>4.1f} MPa of the 25 MPa needed"
          f"   {'FAILS' if 25*f_react < 25 else 'ok'}")
print("\n  -> an inert oligomer buys thermal headroom by throwing away strength.")
print("     The oligomer has to become part of the network. It must be")
print("     FUNCTIONALISED - reactive end groups on an already-long chain.\n")

print("="*68)
print("TELECHELIC / MULTIFUNCTIONAL OLIGOMER: heat per gram")
print("="*68)
print("  only the end groups react, but the whole chain joins the network,")
print("  so the heat falls with Mn while the strength does not.\n")
print(f"{'Mn':>8}{'f':>4}{'heat/g':>10}{'full-cure rise':>16}{'at gel point':>15}")
for Mn in (1000, 3000, 6000):
    for f in (2, 3, 4):
        h = f*dH_bond/Mn                 # J/g at full conversion of end groups
        dT = h/cp
        p_gel = 1.0/(f-1) if f > 2 else 1.0
        print(f"{Mn:>8}{f:>4}{h:>9.0f}J{dT:>14.0f} K{p_gel*dT:>13.0f} K")
print("\n  neat monomer, for comparison:      480 J/g        320 K")
print("  a trifunctional Mn 3000 oligomer gels at 50% of its end groups,")
print("  which is a 20 K rise. The thermal problem simply disappears.")

print("\n" + "="*68)
print("THE MIXTURE")
print("="*68)
rows = [
 ("cyanoacrylate-functional oligomer, Mn ~3000, f~3", 70, "the load-bearing network; low heat because only ends react"),
 ("ethyl 2-cyanoacrylate monomer",                    22, "reactive diluent - cuts viscosity, sets cure speed"),
 ("dissolved elastomer (rubber toughener)",            7, "stops the cured rope being brittle"),
 ("acidic stabiliser (SO2 / methanesulfonic)",         1, "keeps stream A from setting in the barrel"),
]
print("  STREAM A                                              % w/w")
for n,p,why in rows:
    print(f"    {n:<52}{p:>4}")
    print(f"        {why}")
print("\n  STREAM B (metered ~20:1 against A)")
print("    amine accelerator, 2-5% in an inert carrier")
print("        sets the gel clock; this is the knob you tune against flight time")

f_react_mass = 0.22 + 0.70*(3*125/3000)
print(f"\n  reactive mass fraction: {f_react_mass:.0%}  ->  "
      f"heat {f_react_mass*dH_mono:.0f} J/g  ->  +{f_react_mass*dH_mono/cp:.0f} K, reaching "
      f"{25+f_react_mass*dH_mono/cp:.0f} C")

print("\n" + "="*68)
print("PER SHOT, 3 m of 4.4 mm")
print("="*68)
A = math.pi*(2.2e-3)**2
V = A*3.0
print(f"  volume {V*1e6:.0f} mL, mass {V*1100*1e3:.0f} g")
print(f"  stream A {V*1e6*20/21:.0f} mL, stream B {V*1e6/21:.1f} mL")
print(f"  holds {25e6*A/4.448:.0f} lb at 25 MPa, {25e6*A/4.448/50:.1f}x the 50 lb asked")
