import math
cp, dH_mono, mu_m = 1.5, 480.0, 0.003
R, LAND, V = 2.2e-3, 0.002, 9.0
ME_PMMA = 9200.0          # PMMA entanglement molar mass, g/mol

print("="*72)
print("WHY THE EARLIER 'INERT OLIGOMER FAILS' ARGUMENT WAS TOO BROAD")
print("="*72)
print("  It fails BELOW the entanglement mass. Above it, a dissolved polymer")
print("  is an entangled, load-bearing network even without a covalent bond")
print("  to the cyanoacrylate - the two chains simply thread each other.")
print(f"  PMMA entangles at {ME_PMMA:.0f} g/mol, so anything past ~18-20k qualifies.\n")

# Mark-Houwink for PMMA, [eta] in mL/g
K, A_MH = 0.0096, 0.69
def intrinsic(M):  return K * M**A_MH
def c_star(M):     return 100.0 / intrinsic(M)          # g/100 mL
def viscosity(M, c_pct):
    return mu_m * (1.0 + (c_pct / c_star(M))**3.9)

print("="*72)
print("THE TRADE: higher molar mass needs less of it, which leaves more monomer")
print("="*72)
print(f"\n{'PMMA Mw':>9}{'x Me':>7}{'c* ':>8}{'loading for':>13}{'monomer':>9}"
      f"{'rise':>7}{'reaches':>9}  verdict")
print(f"{'':>9}{'':>7}{'':>8}{'~10 Pa.s':>13}{'left':>9}")
best = None
for Mw in (15000, 20000, 25000, 30000, 40000, 60000, 100000):
    cs = c_star(Mw)
    load = cs * (3300.0)**(1/3.9)            # loading giving ~10 Pa.s
    if load > 85: load = 85.0
    monomer = max(0.0, 100 - load - 7 - 1)   # 7% rubber, 1% stabiliser
    dT = (monomer/100 * dH_mono) / cp
    ent = Mw / ME_PMMA
    ok = dT <= 90 and ent >= 2.0 and load <= 85
    if ok and best is None: best = (Mw, load, monomer, dT, ent)
    why = "OK" if ok else ("brittle, under 2x Me" if ent < 2.0 else "too hot")
    print(f"{Mw:>9}{ent:>6.1f}x{cs:>7.1f}%{load:>12.0f}%{monomer:>8.0f}%"
          f"{dT:>6.0f}K{25+dT:>8.0f}C  {why}")

Mw, load, monomer, dT, ent = best
eta = viscosity(Mw, load)
dP = (8*eta*LAND*V/R**2 + 1100*V*V/2)/1e5
print(f"\n  best off-the-shelf point: PMMA Mw {Mw:,} at {load:.0f}%")
print(f"    viscosity {eta:.1f} Pa.s, pressure {dP:.2f} bar of 6, "
      f"+{dT:.0f} K to {25+dT:.0f} C, {ent:.1f}x entangled")

print("\n" + "="*72)
print("COST, against the custom route")
print("="*72)
A = math.pi*R**2; shot_g = A*3.0*1100*1e3
for label, price in (("PMMA resin (Elvacite / Degalan / Paraloid grade)", 25),
                     ("ethyl cyanoacrylate, bulk", 50),
                     ("blended, roughly", 30)):
    print(f"  {label:<48} ~${price}/kg")
print(f"\n  a {shot_g:.0f} g shot costs about ${shot_g/1000*30:.2f}")
print(f"  a custom trifunctional CA-terminated oligomer is a five-figure")
print(f"  synthesis before you have a gram, and CA esters are made by")
print(f"  Knoevenagel condensation then cracked back from the polymer -")
print(f"  putting that end group on an oligomer backbone is a research project.")
