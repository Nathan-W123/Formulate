"""Does a polystyrene dope in 2-butanone leave the nozzle as a fibre or a spray?

Solvent viscosity, density and surface tension come from the expert panel; the
criterion comes from :mod:`formulate.processing.spinning`. The sweep is over the
two things a designer controls - polymer molar mass and concentration - plus
nozzle diameter.

Run with ``python bench/web_shooter_jet.py``. Results are recorded in
docs/BENCHMARKS.md.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from formulate.core.candidate import MaterialClass, molecule_candidate  # noqa: E402
from formulate.core.conditions import Conditions  # noqa: E402
from formulate.evaluation.engine import prefer  # noqa: E402
from formulate.experts import default_registry  # noqa: E402
from formulate.experts.base import PredictionRequest  # noqa: E402
from formulate.processing import SpinningConditions, assess_jet  # noqa: E402

WANTED = frozenset(
    {
        "shear_viscosity", "liquid_density", "surface_tension",
        "critical_temperature", "critical_pressure", "normal_boiling_point",
        "molar_mass",
    }
)
CARRIER = "CCC(C)=O"  # 2-butanone
POLYMER_DENSITY = 1.05  # g/cm^3, polystyrene


def panel(smiles: str) -> dict:
    """Run the molecular panel and keep the preferred prediction per property."""
    registry = default_registry()
    context: dict = {}
    for expert in registry.resolution_order(
        registry.experts_for(WANTED, MaterialClass.MOLECULE)
    ):
        request = PredictionRequest(
            candidate=molecule_candidate(smiles),
            properties=WANTED,
            conditions=Conditions.standard(),
            context=dict(context),
        )
        for prediction in expert.predict(request):
            if not prediction.is_usable:
                continue
            incumbent = context.get(prediction.property)
            if incumbent is None or prefer(prediction, incumbent):
                context[prediction.property] = prediction
    return context


def main() -> None:
    context = panel(CARRIER)
    viscosity = context["shear_viscosity"].quantity.to("Pa*s").value
    density = context["liquid_density"].quantity.to("kg/m^3").value
    surface_tension = context["surface_tension"].quantity.to("N/m").value
    print(
        f"2-butanone from the panel: mu {viscosity * 1000:.3f} mPa s, "
        f"rho {density:.0f} kg/m3, sigma {surface_tension * 1000:.1f} mN/m "
        f"(viscosity via {context['shear_viscosity'].expert_id})\n"
    )

    print(
        f"{'M (kDa)':>8s} {'c (wt%)':>8s} {'nozzle mm':>10s} {'c/c*':>6s} "
        f"{'lambda ms':>11s} {'De':>10s}  regime"
    )
    for molar_mass in (150e3, 500e3, 2.0e6):
        for weight_fraction in (0.10, 0.20, 0.30):
            solvent_density = density / 1000.0
            concentration = (
                weight_fraction
                * solvent_density
                / (
                    1 - weight_fraction
                    + weight_fraction * solvent_density / POLYMER_DENSITY
                )
            )
            for diameter_mm in (0.3, 0.5, 1.0):
                result = assess_jet(
                    SpinningConditions(
                        nozzle_radius=diameter_mm * 1e-3 / 2,
                        velocity=20.0,
                        solvent_viscosity=viscosity,
                        density=density * (1 + 0.25 * weight_fraction),
                        surface_tension=surface_tension,
                        molar_mass=molar_mass,
                        concentration=concentration,
                        polymer="polystyrene",
                        solvent="2-butanone",
                    )
                )
                flags = "" if result.entangled else "  (not entangled)"
                if result.relaxation_time > 1.0:
                    flags += "  (gel, not pumpable)"
                print(
                    f"{molar_mass / 1000:8.0f} {weight_fraction * 100:8.0f} "
                    f"{diameter_mm:10.1f} {result.concentration_ratio:6.1f} "
                    f"{result.relaxation_time * 1e3:11.3f} {result.deborah:10.2f}  "
                    f"{result.regime.value}{flags}"
                )


if __name__ == "__main__":
    main()
