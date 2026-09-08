"""Does the filament dry before it lands?

A jet that survives capillary break-up is still a solution. It becomes a fibre
only when enough solvent has left, and that is a race against the flight time.

Two resistances in series, and which one dominates changes during the flight:

**Getting the vapour away from the surface.** Set by the solvent's vapour
pressure and by how fast air carries it off. For a volatile solvent on a thin
fast-moving filament this is quick - tens of milliseconds - and it is not
usually what decides the outcome.

**Getting solvent from the core to the surface.** Set by diffusion through the
filament itself, which takes a time of order ``R^2 / D``. The trap is that ``D``
is not a constant: as the surface loses solvent it passes through its glass
transition, and the diffusion coefficient of a small molecule in a glassy
polymer is four to six orders of magnitude below its value in the dilute
solution. The skin that forms first is the barrier for everything behind it.

So the honest calculation is a comparison of three times - flight, evaporation
and diffusion - and a verdict of dry, skinned or wet. A skinned filament is not
a failure mode this module invents: it is the normal outcome of dry-spinning
too fast, and it is why industrial columns are metres long.

The diffusion coefficient is the dominant uncertainty by a wide margin and is
an input rather than a prediction. Nothing here estimates it, because a
concentration- and temperature-dependent mutual diffusion coefficient in a
vitrifying polymer solution is not something a correlation over a structure can
supply, and inventing one would put the whole verdict on a fabricated number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

_GAS_CONSTANT = 8.31446261815324

#: Fuller atomic diffusion volumes, cm^3/mol. Fuller, Schettler & Giddings
#: (1966), as tabulated in Poling, Prausnitz & O'Connell.
_FULLER_VOLUMES = {
    "C": 15.9, "H": 2.31, "O": 6.11, "N": 4.54, "S": 22.9,
    "F": 14.7, "Cl": 21.0, "Br": 21.9, "I": 29.8,
}
#: Correction for each aromatic or heterocyclic ring.
_FULLER_RING = -18.3
#: Air, as a single species.
_FULLER_AIR = 19.7
_MOLAR_MASS_AIR = 28.96

#: Air at 25 degrees Celsius.
_AIR_DENSITY = 1.184
_AIR_VISCOSITY = 1.84e-5

#: First zero of the Bessel function J0, squared: the decay constant of the
#: slowest radial mode in a cylinder. Using R^2/D instead would understate the
#: diffusion time by a factor of nearly six.
_CYLINDER_MODE = 2.404826**2


class DryingRegime(str, Enum):
    """What state the filament is in when it lands."""

    #: Solvent has left throughout: a solid fibre.
    DRY = "dry"
    #: The surface has vitrified and the core has not: a tacky-cored strand.
    SKINNED = "skinned"
    #: Not even the surface has dried: it lands as a liquid.
    WET = "wet"


@dataclass(frozen=True, slots=True)
class DryingConditions:
    """The filament, the solvent and the flight, in SI units."""

    #: Filament radius after draw-down, m. This is the strong variable.
    radius: float
    #: Flight speed, m/s.
    velocity: float
    #: Distance to the target, m.
    distance: float
    #: Solvent saturation vapour pressure at the filament temperature, Pa.
    vapour_pressure: float
    #: Solvent molar mass, g/mol.
    solvent_molar_mass: float
    #: Fuller diffusion volume of the solvent, cm^3/mol.
    solvent_diffusion_volume: float
    #: Solution density, kg/m^3.
    density: float
    #: Mass fraction of the solution that is solvent.
    solvent_fraction: float
    #: Mutual diffusion coefficient of solvent in the drying filament, m^2/s.
    #: An input, not a prediction: see the module docstring.
    diffusivity_in_filament: float
    temperature: float = 298.15


@dataclass(frozen=True, slots=True)
class DryingAssessment:
    """The three times, the verdict, and the radius that would change it."""

    regime: DryingRegime
    flight_time: float
    evaporation_time: float
    diffusion_time: float
    mass_transfer_coefficient: float
    air_diffusivity: float
    #: Filament radius at which diffusion would just keep up with the flight.
    critical_radius: float
    limitations: tuple[str, ...] = ()

    @property
    def limiting_step(self) -> str:
        return "diffusion out of the core" if self.diffusion_time > self.evaporation_time \
            else "evaporation from the surface"

    def describe(self) -> str:
        lines = [
            f"regime: {self.regime.value.upper()}   (limited by {self.limiting_step})",
            f"  flight       {self.flight_time * 1e3:10.1f} ms",
            f"  evaporation  {self.evaporation_time * 1e3:10.1f} ms",
            f"  diffusion    {self.diffusion_time * 1e3:10.1f} ms",
            f"  k_c {self.mass_transfer_coefficient:.3f} m/s, "
            f"D(air) {self.air_diffusivity * 1e6:.2f} mm^2/s",
            f"  a filament would have to be under {self.critical_radius * 1e6:.1f} um "
            "in radius for diffusion to keep up",
        ]
        lines.extend(f"  {limit}" for limit in self.limitations)
        return "\n".join(lines)


def fuller_diffusion_volume(smiles: str) -> float | None:
    """Fuller diffusion volume in cm^3/mol from a structure, or None."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdMolDescriptors

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    except Exception:
        return None
    if mol is None:
        return None
    total = 0.0
    for atom in mol.GetAtoms():
        contribution = _FULLER_VOLUMES.get(atom.GetSymbol())
        if contribution is None:
            return None
        total += contribution
    total += _FULLER_RING * rdMolDescriptors.CalcNumAromaticRings(mol)
    return total


def fuller_diffusivity(
    temperature: float,
    pressure: float,
    molar_mass: float,
    diffusion_volume: float,
) -> float:
    """Binary gas diffusivity of a vapour in air, m^2/s.

    Fuller, Schettler & Giddings. The published form gives cm^2/s for pressure
    in bar, so both are converted here rather than at the call site.
    """
    combined = 2.0 / (1.0 / molar_mass + 1.0 / _MOLAR_MASS_AIR)
    denominator = (
        (pressure / 1e5)
        * math.sqrt(combined)
        * (diffusion_volume ** (1 / 3) + _FULLER_AIR ** (1 / 3)) ** 2
    )
    return 0.00143 * temperature**1.75 / denominator * 1e-4


def mass_transfer_coefficient(
    radius: float, velocity: float, air_diffusivity: float
) -> float:
    """``k_c`` in m/s for a filament moving through still air.

    A crossflow correlation on a cylinder, which is not the geometry: the
    filament moves along its own axis, where the boundary layer grows down the
    length instead of being renewed at every point. Crossflow therefore
    overstates the transfer, so the evaporation time below is a lower bound.
    That direction is deliberate - if evaporation is slow even on the optimistic
    correlation, the conclusion is safe.
    """
    diameter = 2.0 * radius
    reynolds = _AIR_DENSITY * velocity * diameter / _AIR_VISCOSITY
    schmidt = _AIR_VISCOSITY / (_AIR_DENSITY * air_diffusivity)
    sherwood = 0.3 + 0.62 * math.sqrt(reynolds) * schmidt ** (1 / 3)
    return sherwood * air_diffusivity / diameter


def evaporation_time(conditions: DryingConditions, coefficient: float) -> float:
    """Time to carry the solvent away from the surface, seconds.

    Mass of solvent per unit length over the flux per unit length, with the
    surface held at saturation. Holding it there is optimistic: as the surface
    dries its activity falls and the driving force with it.
    """
    saturation = (
        conditions.vapour_pressure
        * conditions.solvent_molar_mass
        * 1e-3
        / (_GAS_CONSTANT * conditions.temperature)
    )
    per_length = (
        math.pi * conditions.radius**2 * conditions.density * conditions.solvent_fraction
    )
    flux_per_length = 2.0 * math.pi * conditions.radius * coefficient * saturation
    return per_length / flux_per_length


def diffusion_time(radius: float, diffusivity: float) -> float:
    """Time for the slowest radial mode to decay in a cylinder, seconds."""
    return radius**2 / (_CYLINDER_MODE * diffusivity)


def critical_radius(flight_time: float, diffusivity: float) -> float:
    """Radius at which diffusion just keeps up with the flight."""
    return math.sqrt(_CYLINDER_MODE * diffusivity * flight_time)


def assess_drying(conditions: DryingConditions) -> DryingAssessment:
    """Compare the three times and say what lands."""
    air_diffusivity = fuller_diffusivity(
        conditions.temperature,
        101325.0,
        conditions.solvent_molar_mass,
        conditions.solvent_diffusion_volume,
    )
    coefficient = mass_transfer_coefficient(
        conditions.radius, conditions.velocity, air_diffusivity
    )
    flight = conditions.distance / conditions.velocity
    evaporation = evaporation_time(conditions, coefficient)
    diffusion = diffusion_time(conditions.radius, conditions.diffusivity_in_filament)

    if flight >= max(evaporation, diffusion):
        regime = DryingRegime.DRY
    elif flight >= evaporation:
        regime = DryingRegime.SKINNED
    else:
        regime = DryingRegime.WET

    limitations = [
        "the mass transfer coefficient is a crossflow correlation on a cylinder and "
        "the filament moves along its own axis, so the evaporation time is a lower "
        "bound rather than an estimate",
        "the surface is held at saturation, which overstates the driving force once "
        "it starts to dry",
        "the diffusion coefficient is an input, not a prediction: it falls by four to "
        "six orders of magnitude as the surface vitrifies, and the verdict moves with it",
    ]
    if regime is DryingRegime.SKINNED:
        limitations.append(
            "a skin over a wet core is not a fibre: it lands tacky, it will keep "
            "shrinking and voiding as the core dries, and its strength is not the "
            "strength of the solid polymer"
        )
    return DryingAssessment(
        regime=regime,
        flight_time=flight,
        evaporation_time=evaporation,
        diffusion_time=diffusion,
        mass_transfer_coefficient=coefficient,
        air_diffusivity=air_diffusivity,
        critical_radius=critical_radius(flight, conditions.diffusivity_in_filament),
        limitations=tuple(limitations),
    )
