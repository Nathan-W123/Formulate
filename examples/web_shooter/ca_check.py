import math
LB=4.4482216152605; F=50*LB

print("=== does the spinneret we already have carry 50 lb? ===")
print("   (100% solids, so no shrinkage: dry strand = orifice)")
for n, d_mm in ((40,0.7),(46,0.7),(40,0.8),(60,0.7)):
    A = n*math.pi*(d_mm/2e3)**2
    print(f"  {n} holes x {d_mm} mm -> {A*1e6:5.2f} mm2, stress {F/A/1e6:5.1f} MPa, "
          f"SF vs 25 MPa CA = {25/(F/A/1e6):.2f}")

print("\n=== can 13 bar push it through? (Hagen-Poiseuille) ===")
print(f"{'visc':>6}{'land':>7}{'v':>7}{'dP':>10}{'flow':>10}")
for mu in (0.1, 0.5, 1.0, 5.0):
    for L_mm in (2.0, 10.0):
        for v in (5.0,):
            R = 0.35e-3
            dP = 8*mu*(L_mm/1e3)*v/R**2
            Q = 40*math.pi*R**2*v
            ok = "  <-- fits 13 bar" if dP < 13e5 else ""
            print(f"{mu:>5}Pa.s{L_mm:>6}mm{v:>6}m/s{dP/1e5:>9.1f}bar{Q*1e6:>8.0f}mL/s{ok}")

print("\n=== what a shot costs, with no solvent to carry ===")
A = 40*math.pi*(0.35e-3)**2
for L in (2.0, 3.0, 5.0):
    V = A*L; m = V*1100
    print(f"  {L} m of web: {V*1e6:5.1f} mL, {m*1e3:5.0f} g   "
          f"(the 25% dope version was {m*1e3/0.25:.0f} g)")

print("\n=== cure: why chemistry beats transport ===")
print("  solvent loss from a 700 um filament, diffusion-limited: "
      f"{(0.35e-3)**2/1e-10:.0f} s")
print("  cooling of a 700 um hot melt:                          "
      f"{(0.35e-3)**2/1e-7:.2f} s")
print("  CA + amine accelerator, anionic chain growth:           ~1 s, and it is")
print("    a bulk reaction - every point cures at once, so it does not scale")
print("    with radius squared at all. That is the whole reason it works.")
