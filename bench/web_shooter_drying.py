"""Does the filament dry before it lands, and how thin would it have to be?

Solvent density and molar mass come from the expert panel; vapour pressure from
the same compilations the panel uses; the air-side diffusivity from Fuller. The
diffusion coefficient inside the drying filament is swept rather than predicted,
because it falls four to six orders of magnitude as the surface vitrifies and
nothing here can supply it.

Run with ``python bench/web_shooter_drying.py``. Results are in
docs/BENCHMARKS.md.
"""

from __future__ import annotations

import math
import warnings

warnings.filterwarnings("ignore")

from formulate.core.candidate import MaterialClass, molecule_candidate  # noqa: E402
from formulate.core.conditions import Conditions  # noqa: E402
from formulate.evaluation.engine import prefer  # noqa: E402
from formulate.experts import default_registry  # noqa: E402
from formulate.experts.base import PredictionRequest  # noqa: E402
from formulate.processing import (  # noqa: E402
    DryingConditions,
    assess_drying,
    fuller_diffusion_volume,
)
from formulate.processing.drying import critical_radius  # noqa: E402

CARRIER = "CCC(C)=O"  # 2-butanone
CARRIER_CAS = "78-93-3"
WANTED = frozenset(
    {
        "liquid_density", "molar_mass", "surface_tension", "shear_viscosity",
        "critical_temperature", "critical_pressure", "normal_boiling_point",
    }
)
#: A ten metre shot at twenty metres per second.
DISTANCE, VELOCITY = 10.0, 20.0
#: Well-drawn polystyrene, twenty per cent of the theoretical strength bound.
PRACTICAL_STRENGTH_PA = 58e6
#: An adult, before any swing loading.
LOAD_N = 800.0


def panel(smiles: str) -> dict:
    registry = default_registry()
    context: dict = {}
    for expert in registry.resolution_order(
        registry.experts_for(WANTED, MaterialClass.MOLECULE)
    ):
        request = PredictionRequest(
            candidate=molecule_candidate(smiles), properties=WANTED,
            conditions=Conditions.standard(), context=dict(context),
        )
        for prediction in expert.predict(request):
            if not prediction.is_usable:
                continue
            incumbent = context.get(prediction.property)
            if incumbent is None or prefer(prediction, incumbent):
                context[prediction.property] = prediction
    return context


def main() -> None:
    from thermo import VaporPressure

    context = panel(CARRIER)
    density = context["liquid_density"].quantity.to("kg/m^3").value
    molar_mass = context["molar_mass"].quantity.to("g/mol").value
    vapour_pressure = VaporPressure(CASRN=CARRIER_CAS).T_dependent_property(298.15)
    volume = fuller_diffusion_volume(CARRIER)
    print(
        f"2-butanone: rho {density:.0f} kg/m3, M {molar_mass:.2f} g/mol, "
        f"Psat {vapour_pressure / 1000:.1f} kPa, Fuller V {volume:.1f} cm3/mol\n"
    )

    print(
        f"{'radius um':>10s} {'D m2/s':>9s} {'flight':>9s} {'evap':>10s} "
        f"{'diffusion':>12s}  regime"
    )
    for diffusivity in (1e-10, 1e-11, 1e-12):
        for radius_um in (5, 20, 100, 500, 2100):
            result = assess_drying(
                DryingConditions(
                    radius=radius_um * 1e-6, velocity=VELOCITY, distance=DISTANCE,
                    vapour_pressure=vapour_pressure, solvent_molar_mass=molar_mass,
                    solvent_diffusion_volume=volume, density=density * 1.1,
                    solvent_fraction=0.80, diffusivity_in_filament=diffusivity,
                )
            )
            print(
                f"{radius_um:10d} {diffusivity:9.0e} "
                f"{result.flight_time * 1e3:8.0f}ms {result.evaporation_time * 1e3:9.1f}ms "
                f"{result.diffusion_time * 1e3:11.1f}ms  {result.regime.value}"
            )
        print()

    flight = DISTANCE / VELOCITY
    print(f"radius at which diffusion just keeps up with a {flight:.1f} s flight:")
    for diffusivity in (1e-10, 1e-11, 1e-12, 1e-13):
        radius = critical_radius(flight, diffusivity)
        print(
            f"   D = {diffusivity:.0e} m2/s  ->  {radius * 1e6:7.1f} um radius "
            f"({radius * 2e6:.1f} um diameter)"
        )

    area = LOAD_N / PRACTICAL_STRENGTH_PA
    load_radius = math.sqrt(area / math.pi)
    print(
        f"\nto hold {LOAD_N:.0f} N at {PRACTICAL_STRENGTH_PA / 1e6:.0f} MPa: "
        f"{load_radius * 1e6:.0f} um radius ({load_radius * 2e3:.1f} mm diameter)"
    )
    for diffusivity in (1e-10, 1e-11):
        radius = critical_radius(flight, diffusivity)
        print(
            f"   at D = {diffusivity:.0e} it dries only below {radius * 1e6:.1f} um, "
            f"so a bundle needs {(load_radius / radius) ** 2:,.0f} filaments"
        )


if __name__ == "__main__":
    main()
