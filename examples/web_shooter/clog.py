import math
print("="*64); print("A. UNMIXED CA SITTING IN THE ORIFICE"); print("="*64)
print("  CA cures where moisture reaches it. Water diffuses into the liquid")
print("  at ~1e-9 m2/s, so the cured plug grows as sqrt(D*t):\n")
D = 1e-9
for label, t in (("30 s",30),("2 min",120),("10 min",600),("1 hour",3600),("8 hours",28800)):
    x = math.sqrt(D*t)
    frac = x/2.2e-3
    print(f"  {label:>8}: plug {x*1e3:5.2f} mm deep  "
          f"({frac*100:4.0f}% of the 4.4 mm orifice radius)  "
          f"{'blows out on the next shot' if frac < 0.25 else 'needs clearing'}")

print("\n  -> minutes are fine. Leave it an hour and you are drilling it out.")
print("     A sealed cap with a desiccant pellet removes this entirely.")

print("\n"+"="*64); print("B. THE STATIC MIXER"); print("="*64)
print("  This one clogs every single time, by design: the whole point is that")
print("  CA + accelerator gels in about a second, and what is in the mixer")
print("  when you release the trigger gels there.")
V = math.pi*(3e-3)**2*0.06        # a 6 mm bore, 60 mm long mixer
print(f"  a typical 6 x 60 mm mixer holds {V*1e6:.1f} mL - lost per shot")
print("  mixer tips are a consumable, about 30 cents each. Budget one per shot.")

print("\n"+"="*64); print("C. THE PROBLEM NEITHER OF THOSE IS"); print("="*64)
v, d = 9.0, 4.4e-3
print(f"  at {v} m/s the jet is in the air for {3.0/v*1000:.0f} ms over 3 m.")
print(f"  accelerated CA gels in roughly 1-5 s. So it does NOT set in flight:")
print(f"  it arrives liquid, bonds on contact (which CA does superbly), and")
print(f"  then cures while hanging - during which it sags and cannot hold load.")
print(f"\n  UV was the obvious escape and it fails on exposure time:")
ring_mm = 20
expo = ring_mm/1e3/v
print(f"    a {ring_mm} mm LED ring at {v} m/s gives {expo*1e3:.1f} ms of light.")
print(f"    a UV acrylate needs ~0.3-1 J/cm2, so that demands "
      f"{0.5/expo/1e4*1e4:.0f} W/cm2 - a few hundred times a cheap LED array.")
