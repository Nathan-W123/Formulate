"""Setting by reaction, and the heat that comes with it.

Drying and cooling are diffusion in a cylinder and both scale as the radius
squared, which is why a strand thick enough to hold a person is thick enough to
defeat them. A polymerisation does not: it proceeds through the volume at once,
so a four-millimetre strand cures in the same time as a five-micron one.

That is the whole reason to reach for it, and it comes with a bill. Chain
polymerisation of a vinyl monomer releases 50 to 90 kJ for every mole of double
bonds it consumes, and a mole of small monomer is about a tenth of a kilogram.
Divided by a specific heat of around 1700 J/(kg K), that is a temperature rise
of a few hundred kelvin if none of the heat escapes.

Whether it escapes is the same competition as before, in the other direction:
heat leaves a cylinder on the thermal diffusion time, and a cure that finishes
faster than that is effectively adiabatic. A thick strand has a long thermal
time, so the very geometry that makes reaction the only workable mechanism is
the geometry that traps its heat. The way out is dilution - carry pre-formed
polymer or filler so that less of the mass is reacting - which is what acrylic
bone cement does for exactly this reason, and it raises the viscosity into the
range a coherent jet needs anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

#: Standard enthalpies of polymerisation, kJ **per mole of double bond**, with
#: molar masses in g/mol and double bonds per monomer. Negative in the
#: thermodynamic convention; stored positive because what matters here is how
#: much heat comes out.
#:
#: Per double bond rather than per monomer, because that is the quantity that
#: is nearly constant - opening a vinyl double bond releases fifty to ninety
#: kilojoules whatever it is attached to - and because storing it per monomer
#: put a diacrylate's 157 kJ next to an acrylate's 79 with nothing to say they
#: were different units of the same thing. A test asserts the range, and it is
#: what caught that.
POLYMERISATION: dict[str, tuple[float, float, int]] = {
    "methyl methacrylate": (57.8, 100.12, 1),
    "styrene": (69.9, 104.15, 1),
    "methyl acrylate": (78.7, 86.09, 1),
    "vinyl acetate": (88.0, 86.09, 1),
    # Crosslinkers, at their family's enthalpy per double bond. A diacrylate
    # opens twice as many bonds per molecule and weighs less than twice as
    # much, so per kilogram it is the hotter resin - which is the price of the
    # gel point that makes it fast.
    "1,6-hexanediol diacrylate": (78.7, 226.27, 2),
    "trimethylolpropane triacrylate": (78.7, 296.32, 3),
}

#: Specific heat of a typical acrylic or vinyl liquid, J/(kg K).
_SPECIFIC_HEAT = 1700.0

#: Thermal diffusivity of a filled organic resin, m^2/s.
_THERMAL_DIFFUSIVITY = 1.0e-7

#: First zero of J0 squared: the slowest radial mode in a cylinder.
_CYLINDER_MODE = 2.404826**2

#: Above this a burn is immediate rather than a matter of exposure time. Bone
#: cement causes thermal necrosis around here, which is the closest well
#: documented analogue to a curing mass held against tissue.
_BURN_THRESHOLD_C = 70.0


class CureVerdict(str, Enum):
    """Whether the cure lands where it needs to."""

    #: Sets before it arrives: a solid rod hits the target and does not stick.
    TOO_FAST = "too fast"
    #: Lands liquid, sets shortly after: what is wanted.
    USABLE = "usable"
    #: Still liquid long after landing: it runs off before it holds anything.
    TOO_SLOW = "too slow"


@dataclass(frozen=True, slots=True)
class CureConditions:
    """The resin, the strand and the shot."""

    monomer: str
    #: Mass fraction of the resin that actually polymerises. The rest is
    #: pre-formed polymer or filler and contributes heat capacity only.
    reactive_fraction: float
    #: Time for the bulk to reach gel and carry load, s.
    cure_time: float
    #: Strand radius, m.
    radius: float
    #: Flight time before it lands, s.
    flight_time: float
    ambient_c: float = 25.0
    specific_heat: float = _SPECIFIC_HEAT


@dataclass(frozen=True, slots=True)
class CureAssessment:
    """The timing verdict and the temperature it reaches getting there."""

    verdict: CureVerdict
    adiabatic_rise_k: float
    retained_fraction: float
    peak_temperature_c: float
    thermal_time: float
    burns_on_contact: bool
    limitations: tuple[str, ...] = ()

    def describe(self) -> str:
        lines = [
            f"cure: {self.verdict.value.upper()}",
            f"  adiabatic rise      {self.adiabatic_rise_k:8.0f} K",
            f"  heat retained       {self.retained_fraction:8.0%}  "
            f"(cure against a {self.thermal_time:.1f} s thermal time)",
            f"  peak temperature    {self.peak_temperature_c:8.0f} C"
            + ("   BURNS ON CONTACT" if self.burns_on_contact else ""),
        ]
        lines.extend(f"  {limit}" for limit in self.limitations)
        return "\n".join(lines)


def adiabatic_temperature_rise(
    monomer: str, reactive_fraction: float, specific_heat: float = _SPECIFIC_HEAT
) -> float:
    """Temperature rise in kelvin if none of the reaction heat escapes."""
    if monomer not in POLYMERISATION:
        raise KeyError(
            f"no polymerisation enthalpy for {monomer}; the table has "
            f"{sorted(POLYMERISATION)}"
        )
    enthalpy_kj, molar_mass, double_bonds = POLYMERISATION[monomer]
    #: J per kilogram of monomer: every double bond in it opens.
    per_kilogram = enthalpy_kj * double_bonds * 1000.0 / (molar_mass * 1e-3)
    return reactive_fraction * per_kilogram / specific_heat


def thermal_time(radius: float, diffusivity: float = _THERMAL_DIFFUSIVITY) -> float:
    """Time for the slowest radial thermal mode to decay, s."""
    return radius**2 / (_CYLINDER_MODE * diffusivity)


def retained_fraction(cure_time: float, thermal: float) -> float:
    """Share of the reaction heat still in the strand when the cure finishes.

    A lumped one-parameter model, ``1 / (1 + t_cure / t_thermal)``: heat has
    the whole cure to escape and escapes on the thermal time, so a cure much
    faster than that keeps nearly all of it and a cure much slower keeps
    little. It is not a solution of the coupled problem, which would need the
    reaction rate as a function of temperature and would feed back on itself.
    It is right at both limits and monotone between them, which is what the
    verdict needs.
    """
    return 1.0 / (1.0 + cure_time / thermal)


def assess_cure(conditions: CureConditions) -> CureAssessment:
    """Does it set at the right moment, and how hot does it get doing so?"""
    rise = adiabatic_temperature_rise(
        conditions.monomer, conditions.reactive_fraction, conditions.specific_heat
    )
    thermal = thermal_time(conditions.radius)
    retained = retained_fraction(conditions.cure_time, thermal)
    peak = conditions.ambient_c + rise * retained

    if conditions.cure_time < conditions.flight_time:
        verdict = CureVerdict.TOO_FAST
    elif conditions.cure_time > 10.0 * conditions.flight_time:
        verdict = CureVerdict.TOO_SLOW
    else:
        verdict = CureVerdict.USABLE

    limitations = [
        "the retained fraction is a lumped one-parameter model, not a solution of "
        "the coupled heat-and-reaction problem, which would feed back on itself: a "
        "hotter core reacts faster and releases its heat sooner",
        "no cure kinetics are predicted here. The cure time is an input, and nothing "
        "in this repository derives it from an initiator, a temperature or a recipe",
    ]
    if verdict is CureVerdict.TOO_FAST:
        limitations.append(
            "setting before it arrives means a solid rod hits the target: it will "
            "bounce rather than bond, and the strand has no anchor"
        )
    burns = peak > _BURN_THRESHOLD_C
    if burns:
        limitations.append(
            f"peak is above {_BURN_THRESHOLD_C:.0f} C, where a curing mass held "
            "against skin causes thermal injury rather than discomfort"
        )
    return CureAssessment(
        verdict=verdict,
        adiabatic_rise_k=rise,
        retained_fraction=retained,
        peak_temperature_c=peak,
        thermal_time=thermal,
        burns_on_contact=burns,
        limitations=tuple(limitations),
    )


def impact_energy(volume_m3: float, density: float, velocity: float) -> float:
    """Kinetic energy of the shot when it arrives, joules."""
    return 0.5 * volume_m3 * density * velocity**2


def jet_thrust(density: float, radius: float, velocity: float) -> float:
    """Reaction force on the shooter while the jet fires, newtons."""
    return density * math.pi * radius**2 * velocity**2
