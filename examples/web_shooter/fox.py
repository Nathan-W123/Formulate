print("="*72)
print("THE CHECK THE VISCOSITY SCALING MISSES: is the blend even a liquid?")
print("="*72)
print("  A concentrated polymer solution is only pourable if its glass")
print("  transition is well below room temperature. Fox: 1/Tg = w1/Tg1 + w2/Tg2\n")
TG_PMMA, TG_MONO = 378.0, 180.0     # K; the monomer acts as a plasticiser

def fox(w_poly, tg_poly=TG_PMMA):
    return 1.0 / (w_poly/tg_poly + (1-w_poly)/TG_MONO)

print(f"{'PMMA %':>8}{'blend Tg':>11}{'at 25 C it is':>17}{'monomer left':>14}{'rise':>7}")
for w in (0.30, 0.45, 0.53, 0.65, 0.77, 0.85):
    tg = fox(w)
    mono = max(0.0, 1 - w - 0.08)
    state = "a pourable liquid" if tg < 250 else ("a thick syrup" if tg < 285 else "A SOLID")
    print(f"{w*100:>7.0f}%{tg-273.15:>9.0f}C{state:>18}{mono*100:>12.0f}%"
          f"{mono*480/1.5:>6.0f}K")

print("\n  -> 85% PMMA has a Tg of 51 C. It is not an 8 Pa.s liquid, it is a solid.")
print("     The semidilute viscosity scaling does not know about Tg and happily")
print("     extrapolated into the glass. Everything above ~53% was fiction.")
print("     And at 53% there is 39% monomer left, which is a 125 K rise. Dead.\n")

print("="*72)
print("SO THE BACKBONE MATTERS, AND I HAD GLOSSED IT")
print("="*72)
print("  An acrylic oligomer is still glassy at Mn 3000 (Fox-Flory:")
print(f"  378 - 2e5/3000 = {378 - 2e5/3000 - 273.15:.0f} C). A polyether or polyester")
print("  backbone is not - polypropylene glycol at Mn 3000 has a Tg near -70 C")
print("  and pours at room temperature.\n")
print("  That is exactly what a urethane-acrylate oligomer is, and those are")
print("  sold by the drum:")
for name, mn, f, eta, price in (
        ("aliphatic urethane acrylate (Sartomer CN-series)", "1-5k", "2-6", "2-50 Pa.s", "$15-40/kg"),
        ("polyester acrylate (Allnex Ebecryl)",              "1-3k", "2-6", "1-20 Pa.s", "$10-30/kg"),
        ("epoxy acrylate",                                   "0.5-2k", "2",   "5-100 Pa.s", "$10-25/kg")):
    print(f"    {name:<50}")
    print(f"      Mn {mn}, f = {f}, {eta} neat, {price}")
print("\n  Every one of them has the right molar mass, the right functionality,")
print("  the right viscosity and the right price. The oligomer was never the")
print("  problem.")

print("\n" + "="*72)
print("WHAT IS THE PROBLEM")
print("="*72)
print(f"  {'chemistry':<42}{'gel time':>14}{'vs 333 ms':>12}")
for name, t, note in (
        ("cyanoacrylate + amine accelerator", 0.3, "only one fast enough"),
        ("thiol-Michael, strongly base-catalysed", 5.0, ""),
        ("acrylic redox (peroxide/amine, 'SGA')", 40.0, ""),
        ("fast polyurethane, tin-catalysed", 8.0, ""),
        ("epoxy-amine", 300.0, "")):
    print(f"  {name:<42}{t:>11.0f} s{t/0.333:>10.0f}x  {note}")
print("\n  Cyanoacrylate is the only chemistry that gels in a third of a second,")
print("  and it is sold only as the monomer - which is where all the heat is.")
print("  The oligomers that are sold cure one to three orders of magnitude too")
print("  slowly. That is the wall, and it is a kinetics wall, not a catalogue one.")
