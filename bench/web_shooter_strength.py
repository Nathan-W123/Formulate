"""How strong can the cured strand be, and how strong will it actually be?

The mechanical expert refuses a crosslinked network outright - it is an
entanglement model, and a network has no entanglement modulus - so the
strength that sets the filament count in the jet design was an assumption.
This puts a bound and a calibrated estimate under it.

Three ceilings, from the physics that is loosest to the physics that is
tightest, and then the knockdown that turns a ceiling into a number:

* chain scission, every backbone aligned with the load - what the C-C bond
  could carry, and what a drawn fibre reaches a tenth of
* Orowan, ``sqrt(E * gamma / a0)`` - a flawless isotropic solid
* the engine's own ``theoretical_strength``, E/10 - the same bound, cruder

A real cast thermoset reaches a fraction of the isotropic ceiling, and the
fraction is measured here on the two glassy polymers the engine can predict a
modulus for and a handbook gives a strength for. That fraction, not the
ceiling, is the number the design should carry.

Run with ``python bench/web_shooter_strength.py``. Results in
docs/BENCHMARKS.md.
"""

from __future__ import annotations

import math
import warnings

warnings.filterwarnings("ignore")

from formulate.core.candidate import (  # noqa: E402
    Candidate,
    MaterialClass,
    MonomerUnit,
    PolymerSpec,
)
from formulate.core.conditions import Conditions  # noqa: E402
from formulate.core.prediction import prefer  # noqa: E402
from formulate.core.quantity import Quantity  # noqa: E402
from formulate.experts import default_registry  # noqa: E402
from formulate.experts.base import PredictionRequest  # noqa: E402

#: Surface energy of the cured diacrylate, from the surface tension the panel
#: gives the monomer once its boiling point is right (35.2 mN/m), J/m^2.
SURFACE_ENERGY = 0.0352
#: Interchain van der Waals spacing, m.
SPACING = 3.0e-10
#: Force to rupture a C-C bond and the cross-section one extended chain
#: occupies, for the aligned ceiling.
BOND_FORCE = 6.0e-9
CHAIN_AREA = 1.8e-19

#: Handbook tensile strength of the bulk glassy polymer at 25 C, MPa.
REAL_STRENGTH = {
    "[*]CC(c1ccccc1)[*]": (40.0, "polystyrene"),
    "[*]CC(C)(C(=O)OC)[*]": (70.0, "PMMA"),
}
WANT = frozenset(
    {"youngs_modulus", "theoretical_strength", "glass_transition_temperature", "amorphous_density"}
)
LOAD_N = 785.0
FILAMENT_RADIUS = 1.0e-3


def panel(repeat_unit: str) -> dict:
    registry = default_registry()
    spec = PolymerSpec(
        monomers=(MonomerUnit(smiles=repeat_unit),),
        number_average_molar_mass=Quantity(value=2.0e5, unit="g/mol"),
    )
    candidate = Candidate(
        material_class=MaterialClass.POLYMER, polymer=spec, conditions=Conditions.standard()
    )
    context: dict = {}
    for expert in registry.resolution_order(registry.experts_for(WANT, MaterialClass.POLYMER)):
        request = PredictionRequest(
            candidate=candidate, properties=WANT, conditions=Conditions.standard(),
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
    print("Knockdown from the engine's theoretical strength to a measured one:")
    knockdowns = []
    modulus = None
    for unit, (real, name) in REAL_STRENGTH.items():
        context = panel(unit)
        theoretical = context["theoretical_strength"].quantity.value / 1e6
        modulus = context["youngs_modulus"].quantity.value
        knockdowns.append(real / theoretical)
        print(f"  {name:12s} theoretical {theoretical:6.1f} MPa   real {real:5.1f}   "
              f"knockdown {real / theoretical:.3f}")
    low, high = min(knockdowns), max(knockdowns)

    orowan = math.sqrt(modulus * SURFACE_ENERGY / SPACING)
    aligned = BOND_FORCE / CHAIN_AREA
    print(f"\nCeilings for the cured diacrylate, E = {modulus / 1e9:.2f} GPa (glassy plateau):")
    print(f"  chain scission, aligned        {aligned / 1e9:7.1f} GPa   unreachable without drawing")
    print(f"  Orowan, flawless isotropic     {orowan / 1e6:7.0f} MPa")
    print(f"  engine theoretical, E/10       {modulus / 10 / 1e6:7.0f} MPa")

    print(f"\nRealised, at the measured knockdown of {low:.2f}-{high:.2f}:")
    for label, ceiling in (("from E/10", modulus / 10), ("from Orowan", orowan)):
        lo, hi = ceiling * low / 1e6, ceiling * high / 1e6
        area = math.pi * FILAMENT_RADIUS**2
        print(f"  {label:12s} {lo:5.0f}-{hi:5.0f} MPa  ->  {LOAD_N:.0f} N needs "
              f"{math.ceil(LOAD_N / (hi * 1e6) / area):2d}-{math.ceil(LOAD_N / (lo * 1e6) / area):2d} "
              f"filaments of {FILAMENT_RADIUS * 1e3:.0f} mm radius")
    print("\n  The knockdown rests on two polymers. It is a bracket, not a distribution.")


if __name__ == "__main__":
    main()
