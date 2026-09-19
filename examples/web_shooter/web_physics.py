import math
LB = 4.4482216152605

print("="*66)
print("1. STRENGTH: how much cross-section does 50 lb actually need?")
print("="*66)
F50, F30 = 50*LB, 30*LB
mats = [("unvulcanised natural rubber (rubber cement)", 1.5, 940, "creeps - no network"),
        ("SBS block copolymer (contact cement)",       20.0, 940, "glassy PS domains crosslink it"),
        ("cast PMMA",                                  70.0, 1180, "brittle"),
        ("nylon-6,6 drawn fibre",                     700.0, 1140, "needs melt drawing")]
SF = 2.0
print(f"{'material':<44}{'UTS':>7}{'area@SF2':>10}{'d(1 strand)':>13}")
for n, uts, rho, note in mats:
    A = F50/(uts*1e6/SF)
    print(f"{n:<44}{uts:>6.0f}M{A*1e6:>9.1f}mm2{math.sqrt(4*A/math.pi)*1e3:>11.1f}mm")

print("\n  ...and what 3 m of that weighs (the real showstopper):")
for n, uts, rho, note in mats:
    A = F50/(uts*1e6/SF); V = A*3.0
    dry = V*rho
    print(f"    {n:<44}{dry*1e3:>7.0f} g dry, {dry*1e3/0.25:>6.0f} g of dope at 25% solids")

print("\n" + "="*66)
print("2. DRYING: can a filament lose its solvent in 0.6 s of flight?")
print("="*66)
# Stage A: how fast can vapour leave the surface (evaporation-limited)?
# Stage B: how fast can solvent reach the surface from inside (diffusion-limited)?
D_air, M, psat, T, R = 8.0e-6, 0.086, 20000.0, 298.0, 8.314   # n-heptane-ish
c_sat = psat*M/(R*T)                       # kg/m3 at the surface
rho_air, mu_air, v = 1.2, 1.8e-5, 5.0
D_poly = 1e-10                             # solvent in a concentrated rubber solution
rho_s, w = 700.0, 0.25                     # solvent density, solids mass fraction

print(f"  saturated heptane vapour at the surface: {c_sat:.2f} kg/m3")
print(f"\n{'filament':>10}{'Re':>7}{'evap-limited':>15}{'diffusion-limited':>20}{'flight 0.6s?':>14}")
for d_um in (50, 100, 300, 1000, 3000):
    d = d_um*1e-6; Rf = d/2
    Re = rho_air*v*d/mu_air
    Sh = 0.3 + 0.62*math.sqrt(Re)*(0.7**(1/3))      # cylinder in crossflow
    delta = d/max(Sh, 1.0)
    flux = D_air*c_sat/delta                         # kg/m2/s
    # solvent mass per unit length / surface area per unit length
    m_solv = math.pi*Rf**2*rho_s*(1-w)
    t_evap = m_solv/(2*math.pi*Rf*flux)
    t_diff = Rf**2/D_poly
    verdict = "YES" if max(t_evap, t_diff) < 0.6 else "no"
    print(f"{d_um:>8}um{Re:>7.0f}{t_evap:>13.2f} s{t_diff:>17.0f} s{verdict:>14}")

print("\n  (diffusion wins: the surface skins in milliseconds and the core stays wet)")

print("\n" + "="*66)
print("3. CLOGGING: how long before the nozzle skins over between shots?")
print("="*66)
for d_um in (300, 700):
    Rf = d_um/2*1e-6
    # a stagnant meniscus loses solvent only by evaporation into still air
    delta = 1e-3                                  # still-air boundary layer, ~1 mm
    flux = D_air*c_sat/delta
    # time to dry a skin one tenth of the orifice radius deep
    skin = 0.1*Rf
    t = skin*rho_s*(1-w)/flux
    print(f"  {d_um} um orifice, still air: a {skin*1e6:.0f} um skin forms in {t:.1f} s")

print("\n" + "="*66)
print("4. THE SOLVENTLESS ALTERNATIVE: does a hot melt freeze in flight?")
print("="*66)
print("  heat diffuses ~1000x faster than solvent does, so cooling beats drying")
alpha = 1.0e-7          # thermal diffusivity of a polymer melt, m2/s
D_poly = 1e-10
for d_um in (100, 300, 1000, 3000, 5300):
    Rf = d_um/2*1e-6
    t_cool = Rf**2/alpha
    t_dry  = Rf**2/D_poly
    print(f"  {d_um:>5} um: freezes in {t_cool:>7.2f} s   (vs {t_dry:>8.0f} s to dry)"
          f"   {'OK in 0.6 s flight' if t_cool < 0.6 else ''}")

print("\n  filaments needed for 22 mm2 of SBS-strength cross-section:")
import math
for d_um in (100, 300, 1000):
    a = math.pi*(d_um/2e6)**2
    print(f"    {d_um:>5} um -> {22.2e-6/a:>8.0f} filaments")
