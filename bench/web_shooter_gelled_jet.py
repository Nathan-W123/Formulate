"""Is there a jet that gels before it breaks up, without cooking itself?

The reactive-jet route has three constraints that pull against each other and
were never checked together. A thin jet stays laminar and sheds its reaction
heat, but Rayleigh instability breaks it into drops in milliseconds. A thick
jet survives long enough to gel and holds its own heat, and a diacrylate
curing to completion has four hundred kelvin of adiabatic rise in it. And the
whole thing has to leave a nozzle at a pressure a wrist can carry.

The free variable is viscosity, which is what the thickener was always for.
Raising it drops the Reynolds number - keeping the jet laminar at a radius
that would otherwise atomise - and raises the Ohnesorge number, which
lengthens the break-up. It does not change the cure at all, because a
fractional conversion rate does not depend on how thick the medium is.

Nothing here is new physics. Every function called is one the earlier
web shooter benchmarks already used; what was missing was running them on the
same design at the same time.

Run with ``python bench/web_shooter_gelled_jet.py``. Results are in
docs/BENCHMARKS.md.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from formulate.experts.kinetics import INITIATION, cure_time  # noqa: E402
from formulate.processing.curing import (  # noqa: E402
    _BURN_THRESHOLD_C,
    adiabatic_temperature_rise,
    retained_fraction,
    thermal_time,
)
from formulate.processing.spinning import (  # noqa: E402
    breakup_length,
    extrusion_pressure,
    reynolds_number,
)

BASE = "1,6-hexanediol diacrylate"
#: Neat resin at 25 C: density and surface tension of a hexanediol diacrylate.
DENSITY = 1010.0
SURFACE_TENSION = 0.034
#: Jet speed and shot length the earlier runs were costed at.
VELOCITY = 20.0
SHOT_M = 10.0
#: Nozzle land length, m.
NOZZLE_LENGTH = 5.0e-3
#: Time to full conversion, over which the reaction heat is released. The gel
#: point arrives in a fraction of a second; the remaining ninety-nine per cent
#: of the enthalpy comes out over roughly this long.
FULL_CONVERSION_S = 100.0
#: Eighty kilograms, and a tensile strength for a crosslinked acrylate glass.
LOAD_N = 785.0
STRENGTH_PA = 40.0e6


def main() -> None:
    gel = cure_time(BASE, 298.15, INITIATION["redox-peroxide-amine"][0])
    flight = SHOT_M / VELOCITY
    print(f"{BASE}, redox initiation at 25 C")
    print(f"  gels in {gel:.3f} s; {SHOT_M:.0f} m at {VELOCITY:.0f} m/s is {flight:.2f} s of flight")
    print(f"  adiabatic rise at full conversion: "
          f"{adiabatic_temperature_rise(BASE, 1.0):.0f} K\n")

    header = (
        f"{'viscosity':>10s} {'radius':>8s} {'Re':>7s} {'L_break':>9s} {'t_break':>9s} "
        f"{'peak T':>7s} {'nozzle':>9s}  verdict"
    )
    print(header)
    print("-" * len(header))
    for viscosity in (0.00425, 0.02, 0.05, 0.1, 0.3):
        for radius in (2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3):
            diameter = 2.0 * radius
            reynolds = reynolds_number(DENSITY, VELOCITY, diameter, viscosity)
            length = breakup_length(
                DENSITY, VELOCITY, diameter, SURFACE_TENSION, viscosity
            )
            peak = 25.0 + adiabatic_temperature_rise(BASE, 1.0) * retained_fraction(
                FULL_CONVERSION_S, thermal_time(radius)
            )
            nozzle = extrusion_pressure(viscosity, radius, VELOCITY, NOZZLE_LENGTH) / 1e5
            row = (
                f"{viscosity * 1000:9.1f}m {radius * 1e3:7.2f}mm {reynolds:7.0f} "
            )
            if length is None:
                print(row + f"{'turbulent':>9s} {'--':>9s} {peak:6.0f}C {nozzle:8.2f}b  "
                      "turbulent, atomises")
                continue
            breakup = length / VELOCITY
            verdict = []
            if gel >= breakup:
                verdict.append("breaks up before it gels")
            if peak >= _BURN_THRESHOLD_C:
                verdict.append(f"exceeds the {_BURN_THRESHOLD_C:.0f} C burn threshold")
            print(
                row + f"{length:8.2f}m {breakup:8.3f}s {peak:6.0f}C {nozzle:8.2f}b  "
                + ("; ".join(verdict) if verdict else f"WORKS ({breakup / gel:.0f}x margin to break-up)")
            )

    area = LOAD_N / STRENGTH_PA
    print(f"\n{LOAD_N:.0f} N at {STRENGTH_PA / 1e6:.0f} MPa needs {area * 1e6:.1f} mm^2 of cross-section:")
    for radius in (5.0e-4, 1.0e-3, 2.0e-3):
        import math

        per = math.pi * radius**2
        print(f"  {math.ceil(area / per):3d} filaments of {radius * 1e3:.1f} mm radius")


if __name__ == "__main__":
    main()
