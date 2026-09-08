"""Does molecular dynamics reproduce the web shooter's carrier solvent?

The web shooter chain in docs/BENCHMARKS.md rests on four numbers for
2-butanone, and every one of them was read out of a table: density, surface
tension, shear viscosity and the Hansen parameters. The design was never
physically validated, because three of those four properties sat in
``NOT_VALIDATABLE_REASONS``. :mod:`formulate.physics.md.stress` retired the
reason two of them shared - OpenMM publishes no pressure tensor - so they can
now be recomputed from a force field that never saw any of them.

Nothing here is a prediction of an unknown. All four experimental values are
in the panel already. The question is whether an independent route agrees, and
what the disagreement is when it does not.

Each stage is hours of wall clock, so they are run one at a time::

    python bench/web_shooter_md.py bulk
    python bench/web_shooter_md.py tension
    python bench/web_shooter_md.py viscosity

Results are recorded in docs/BENCHMARKS.md.
"""

from __future__ import annotations

import sys
import time
import warnings

warnings.filterwarnings("ignore")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from formulate.core.candidate import MaterialClass, molecule_candidate  # noqa: E402
from formulate.core.conditions import Conditions  # noqa: E402
from formulate.evaluation.engine import prefer  # noqa: E402
from formulate.experts import default_registry  # noqa: E402
from formulate.experts.base import PredictionRequest  # noqa: E402

CARRIER = "CCC(C)=O"  # 2-butanone
TEMPERATURE_K = 298.15

#: Molecules per box. Two hundred is the smallest box whose viscosity is not
#: obviously size-limited, and the viscosity is the one protocol where the
#: finite difference rather than the dynamics sets the cost, so it pays to keep
#: the box small. The slab is sized instead by what it has to contain: three
#: hundred and fifty molecules across a 3.2 nm face is 5.1 nm of liquid, which
#: is just over four cutoffs and so just thick enough to have an interior.
BULK_MOLECULES = 400
VISCOSITY_MOLECULES = 200
SLAB_MOLECULES = 350

WANTED = frozenset(
    {
        "liquid_density",
        "surface_tension",
        "shear_viscosity",
        "molar_mass",
        "hansen_dispersion",
        "hansen_polar",
        "hansen_hydrogen_bonding",
    }
)


def carrier():
    mol = Chem.AddHs(Chem.MolFromSmiles(CARRIER))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def panel() -> dict:
    """What the expert panel holds for the carrier, to compare against."""
    registry = default_registry()
    context: dict = {}
    for expert in registry.resolution_order(
        registry.experts_for(WANTED, MaterialClass.MOLECULE)
    ):
        request = PredictionRequest(
            candidate=molecule_candidate(CARRIER),
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


def _compare(name: str, simulated: float, error: float, measured: float, unit: str) -> None:
    deviation = (simulated - measured) / measured * 100.0
    sigma = abs(simulated - measured) / error if error > 0 else float("inf")
    print(f"\n  {name}")
    print(f"    molecular dynamics   {simulated:12.5g} +/- {error:<10.4g} {unit}")
    print(f"    experiment (panel)   {measured:12.5g} {'':<10s} {unit}")
    print(f"    deviation            {deviation:+11.1f} per cent, {sigma:.1f} sampling sigma")


def stage_bulk() -> None:
    """Density, enthalpy of vaporisation and Hildebrand parameter, from one run.

    All three come out of the same constant-pressure box plus one isolated
    molecule sampled at the same temperature. Splitting them into separate
    stages would run the expensive part twice for nothing.
    """
    from formulate.physics.md.condensed import (
        cohesive_energy_density,
        run_npt,
        sample_isolated_energy,
    )

    context = panel()
    measured_density = context["liquid_density"].quantity.to("g/cm^3").value
    hansen = [
        context[f"hansen_{axis}"].quantity.to("MPa^0.5").value
        for axis in ("dispersion", "polar", "hydrogen_bonding")
    ]
    measured_hildebrand = sum(value**2 for value in hansen) ** 0.5

    liquid = run_npt(
        carrier(),
        BULK_MOLECULES,
        TEMPERATURE_K,
        initial_density=0.80,
        pressure_bar=1.0,
        equilibration_ps=200.0,
        production_ps=200.0,
        cutoff_nm=1.0,
        seed=7,
        force_field="opls-aa",
    )
    _compare(
        "liquid density at 298 K",
        liquid.density_g_cm3,
        liquid.density_error,
        measured_density,
        "g/cm^3",
    )
    print(f"    {liquid.production_ps:.0f} ps production, {liquid.wall_seconds / 3600:.2f} h")
    for note in liquid.diagnostics:
        print(f"    ! {note}")

    gas_energy, gas_error = sample_isolated_energy(
        carrier(), TEMPERATURE_K, production_ps=200.0, seed=7, force_field="opls-aa"
    )
    cohesive = cohesive_energy_density(liquid, gas_energy, gas_error)

    # The enthalpy of vaporisation is the potential energy difference plus RT.
    enthalpy = cohesive.vaporisation_energy + 0.00831446261815324 * TEMPERATURE_K
    hildebrand = cohesive.cohesive_energy_density_pa**0.5 / 1e3
    error = 0.5 * cohesive.error_pa / cohesive.cohesive_energy_density_pa * hildebrand
    _compare(
        "Hildebrand parameter at 298 K",
        hildebrand,
        error,
        measured_hildebrand,
        "MPa^0.5",
    )
    print(f"    against the panel's Hansen sphere: dD {hansen[0]:.1f}, dP {hansen[1]:.1f}, "
          f"dH {hansen[2]:.1f} MPa^0.5")
    print(f"    enthalpy of vaporisation {enthalpy:.1f} kJ/mol "
          "(experiment 34.8 at 298 K, Majer & Svoboda)")
    print(f"    molar volume {cohesive.molar_volume_cm3:.2f} cm^3/mol")
    for note in cohesive.diagnostics:
        print(f"    ! {note}")


def stage_tension() -> None:
    from formulate.physics.md.interface import surface_tension

    context = panel()
    measured = context["surface_tension"].quantity.to("mN/m").value
    density = context["liquid_density"].quantity.to("g/cm^3").value
    result = surface_tension(
        carrier(),
        SLAB_MOLECULES,
        TEMPERATURE_K,
        density,
        lateral_nm=3.2,
        vacuum_nm=3.0,
        cutoff_nm=1.2,
        equilibration_ps=150.0,
        production_ps=400.0,
        seed=11,
    )
    _compare(
        "surface tension at 298 K",
        result.surface_tension_mn_m,
        result.surface_tension_error,
        measured,
        "mN/m",
    )
    box = result.box_nm
    print(f"    box {box[0]:.2f} x {box[1]:.2f} x {box[2]:.2f} nm, liquid {result.liquid_nm:.2f} nm")
    print(f"    pressures {result.pressure_bar[0]:+.0f} / {result.pressure_bar[1]:+.0f} / "
          f"{result.pressure_bar[2]:+.0f} bar (xx / yy / zz)")
    print(f"    kinetic anisotropy {result.kinetic_anisotropy_bar:.1f} bar, "
          f"dipole along z {result.dipole_z_rms:.2f} e nm")
    print(f"    vapour {result.vapour_molecules:.2f} molecules in the gap, "
          f"implying {result.vapour_pressure_pa / 1000.0:.1f} kPa")
    print(f"    {result.production_ps:.0f} ps production, {result.wall_seconds / 3600:.2f} h")
    for note in result.notes:
        print(f"    - {note}")
    for note in result.diagnostics:
        print(f"    ! {note}")


def stage_viscosity() -> None:
    from formulate.physics.md.condensed import run_shear_viscosity

    context = panel()
    measured = context["shear_viscosity"].quantity.to("Pa*s").value
    density = context["liquid_density"].quantity.to("g/cm^3").value
    result = run_shear_viscosity(
        carrier(),
        VISCOSITY_MOLECULES,
        TEMPERATURE_K,
        density_g_cm3=density,
        equilibration_ps=100.0,
        production_ps=200.0,
        stress_interval_fs=10.0,
        correlation_ps=4.0,
        cutoff_nm=1.0,
        seed=13,
    )
    _compare(
        "shear viscosity at 298 K",
        result.viscosity_pa_s * 1000.0,
        result.error_pa_s * 1000.0,
        measured * 1000.0,
        "mPa s",
    )
    print(f"    plateau read over {result.plateau_window_ps[0]:.2f}-"
          f"{result.plateau_window_ps[1]:.2f} ps, drift {result.plateau_drift * 100:+.0f} per cent")
    print(f"    stress correlation time {result.correlation_time_ps:.3f} ps, "
          f"rms shear stress {result.stress_rms_bar:.0f} bar")
    print(f"    {result.sampled_ps:.0f} ps production, {result.wall_seconds / 3600:.2f} h")
    for note in result.diagnostics:
        print(f"    ! {note}")


STAGES = {
    "bulk": stage_bulk,
    "tension": stage_tension,
    "viscosity": stage_viscosity,
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        print(f"usage: python bench/web_shooter_md.py {{{'|'.join(STAGES)}}}")
        raise SystemExit(2)
    name = sys.argv[1]
    print(f"2-butanone under OPLS-AA: {name}")
    started = time.perf_counter()
    STAGES[name]()
    print(f"\n  stage wall clock {(time.perf_counter() - started) / 3600:.2f} h")


if __name__ == "__main__":
    main()
