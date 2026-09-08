"""Does a jet of this liquid become a fibre, or break into drops?

A Newtonian jet always breaks up. Surface tension amplifies any disturbance
whose wavelength exceeds the circumference, and the jet becomes drops in a time
set by the Rayleigh scale ``sqrt(rho R^3 / sigma)``. Nothing about the
viscosity prevents it; viscosity only slows it down.

A polymer solution can resist. As the neck thins, the extension rate rises, and
once it exceeds the inverse of the chain relaxation time the chains stretch
rather than relax. Stretched chains raise the extensional viscosity by orders
of magnitude, the neck stops thinning at the rate capillarity wants, and the
filament survives as a thread - the beads-on-string shape - long enough for
solvent to leave and the thread to solidify.

That competition is a ratio of two times, and the ratio has a name: the Deborah
number, chain relaxation time over Rayleigh time. Above about one the filament
persists; below it the jet is effectively Newtonian and makes drops.

What this module is not
    It is not a simulation. It is the standard dimensionless grouping, which
    decides the *regime* and not the fibre. Whether a persistent filament
    becomes a good fibre depends on the draw ratio, on how fast solvent
    actually leaves, and on whether the skin vitrifies before the core - none of
    which is here, and the assessment says so on every result rather than
    leaving it to be assumed.

The relaxation time needs an intrinsic viscosity, which needs Mark-Houwink
constants, which are specific to a polymer, a solvent and a temperature. Those
are tabulated below and a pair that is not in the table is refused, because the
alternative is to borrow a constant from a different solvent and report the
answer as though it meant something.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

#: Mark-Houwink constants ``[eta] = K M^a`` with ``[eta]`` in cm^3/g and ``M``
#: in g/mol, at 25 degrees Celsius. Brandrup & Immergut, Polymer Handbook.
#:
#: Keyed by (polymer, solvent). The exponent carries the solvent quality: 0.5 is
#: a theta solvent where the coil is ideal, and 0.8 is a good one where it is
#: swollen. Two solvents for the same polymer give intrinsic viscosities that
#: differ by a factor of three at high molar mass, which is why borrowing across
#: the table is refused rather than approximated.
MARK_HOUWINK: dict[tuple[str, str], tuple[float, float]] = {
    ("polystyrene", "2-butanone"): (3.9e-2, 0.58),
    ("polystyrene", "toluene"): (1.1e-2, 0.725),
    ("polystyrene", "tetrahydrofuran"): (1.6e-2, 0.706),
    ("poly(methyl methacrylate)", "2-butanone"): (6.8e-3, 0.72),
    ("poly(methyl methacrylate)", "tetrahydrofuran"): (1.28e-2, 0.69),
    ("poly(vinyl acetate)", "2-butanone"): (4.2e-2, 0.62),
}

#: Above this the elastic stress resists capillary thinning and the filament
#: persists; below it the jet behaves as a Newtonian one and makes drops. The
#: transition is not sharp, so a band either side is reported as marginal
#: rather than being rounded to a verdict.
_DEBORAH_FILAMENT = 1.0
_DEBORAH_MARGIN = 0.3

#: Prefactor for the longest Zimm mode in a good solvent.
_ZIMM_PREFACTOR = 0.422

#: A relaxation time above this is longer than the shot itself, and the dope has
#: stopped being a liquid you can push through a nozzle: it is a rubbery gel
#: that will fracture rather than flow. The criterion below still returns a
#: Deborah number there - a very large one - and it no longer means the filament
#: survives, so it is flagged rather than reported as a clean answer.
_PUMPABLE_RELAXATION_SECONDS = 1.0

_GAS_CONSTANT = 8.31446261815324


class JetRegime(str, Enum):
    """What the grouping says happens to the jet."""

    #: Capillary break-up wins: the jet becomes drops.
    SPRAY = "spray"
    #: Neither dominates within the uncertainty of the criterion.
    MARGINAL = "marginal"
    #: Elastic stress resists thinning: a filament persists.
    FILAMENT = "filament"


@dataclass(frozen=True, slots=True)
class SpinningConditions:
    """The nozzle and the dope, in SI units."""

    #: Nozzle radius, m.
    nozzle_radius: float
    #: Jet velocity at the nozzle, m/s.
    velocity: float
    #: Solvent shear viscosity, Pa s - from the panel.
    solvent_viscosity: float
    #: Solution density, kg/m^3 - from the panel.
    density: float
    #: Surface tension, N/m - from the panel.
    surface_tension: float
    #: Polymer number-average molar mass, g/mol.
    molar_mass: float
    #: Polymer concentration, g/cm^3 of solution.
    concentration: float
    polymer: str
    solvent: str
    temperature: float = 298.15


@dataclass(frozen=True, slots=True)
class SpinningAssessment:
    """The regime, the numbers behind it, and what it does not decide."""

    regime: JetRegime
    deborah: float
    ohnesorge: float
    rayleigh_time: float
    relaxation_time: float
    intrinsic_viscosity: float
    overlap_concentration: float
    concentration_ratio: float
    entangled: bool
    limitations: tuple[str, ...] = ()

    def describe(self) -> str:
        lines = [
            f"regime: {self.regime.value.upper()}",
            f"  Deborah   {self.deborah:8.2f}  "
            f"(relaxation {self.relaxation_time * 1e3:.3f} ms over Rayleigh "
            f"{self.rayleigh_time * 1e3:.3f} ms)",
            f"  Ohnesorge {self.ohnesorge:8.3f}",
            f"  c/c*      {self.concentration_ratio:8.2f}  "
            + ("entangled" if self.entangled else "NOT entangled: isolated coils, so a "
               "filament would thin to nothing rather than draw"),
            f"  [eta]     {self.intrinsic_viscosity:8.1f} cm^3/g, "
            f"c* {self.overlap_concentration * 1000:.1f} g/L",
        ]
        lines.extend(f"  {limit}" for limit in self.limitations)
        return "\n".join(lines)


class UnknownPair(Exception):
    """No Mark-Houwink constants for this polymer in this solvent."""


def intrinsic_viscosity(polymer: str, solvent: str, molar_mass: float) -> float:
    """``[eta]`` in cm^3/g from Mark-Houwink, or a refusal."""
    key = (polymer.strip().lower(), solvent.strip().lower())
    if key not in MARK_HOUWINK:
        raise UnknownPair(
            f"no Mark-Houwink constants for {polymer} in {solvent}. The exponent "
            "carries the solvent quality and differs enough between solvents to change "
            "the intrinsic viscosity threefold at high molar mass, so one is not "
            f"borrowed from another; the table has {sorted(MARK_HOUWINK)}"
        )
    k, a = MARK_HOUWINK[key]
    return k * molar_mass**a


def overlap_concentration(intrinsic: float) -> float:
    """``c* = 1/[eta]`` in g/cm^3: where coils begin to touch."""
    return 1.0 / intrinsic


#: Huggins coefficient for a flexible coil in a good solvent. It varies from
#: about 0.3 to 0.5 and only matters below the overlap concentration, where
#: the term it multiplies is small.
_HUGGINS = 0.35

#: Exponent for the viscosity of an entangled semi-dilute solution in a good
#: solvent, ``eta/eta_s ~ (c[eta])^n``. Reptation with excluded volume gives
#: 3.9; a theta solvent gives 4.7. The lower value is used because a thickener
#: is only worth adding in a solvent that dissolves it well, and it is the
#: conservative choice: a smaller exponent needs *more* polymer to reach a
#: target viscosity, so a formulation sized on it is not short of thickener.
_ENTANGLED_EXPONENT = 3.9


def solution_viscosity(
    intrinsic: float, concentration: float, solvent_viscosity: float
) -> float:
    """Viscosity of a polymer solution, Pa s, from ``c`` in g/cm^3.

    Two regimes joined at the overlap concentration, because one expression
    does not cover both and using either alone is wrong by orders of
    magnitude at the other end. Below overlap the coils are separate and
    Huggins applies, ``eta_sp = c[eta] + k (c[eta])^2``. Above it they
    interpenetrate and entangle, and the viscosity climbs as a power of the
    same product.

    The whole dependence is on ``c[eta]``, the coil overlap parameter, which
    is what makes a formulation quotable without knowing the polymer's
    Mark-Houwink constants in the particular solvent: the *product* is fixed
    by the target viscosity, and only the split between concentration and
    intrinsic viscosity needs the constants.
    """
    overlap = concentration * intrinsic
    if overlap <= 1.0:
        relative = 1.0 + overlap + _HUGGINS * overlap**2
    else:
        # Continuous at the crossover: Huggins evaluated at c[eta] = 1.
        relative = (1.0 + 1.0 + _HUGGINS) * overlap**_ENTANGLED_EXPONENT
    return solvent_viscosity * relative


def overlap_for_viscosity(target: float, solvent_viscosity: float) -> float:
    """The ``c[eta]`` a solution needs to reach ``target`` viscosity, Pa s.

    The inverse of :func:`solution_viscosity`, and the quantity a formulation
    is actually specified by. Turning it into a weight per cent needs the
    intrinsic viscosity, which needs Mark-Houwink constants for that exact
    polymer in that exact solvent; leaving the answer here is what lets a
    recipe be stated when those do not exist.
    """
    relative = target / solvent_viscosity
    if relative <= 2.0 + _HUGGINS:
        # Huggins branch: solve k x^2 + x + 1 - relative = 0 for x.
        discriminant = 1.0 + 4.0 * _HUGGINS * (relative - 1.0)
        return (-1.0 + math.sqrt(max(discriminant, 0.0))) / (2.0 * _HUGGINS)
    return (relative / (2.0 + _HUGGINS)) ** (1.0 / _ENTANGLED_EXPONENT)


def zimm_relaxation_time(
    intrinsic: float, molar_mass: float, solvent_viscosity: float, temperature: float
) -> float:
    """Longest Zimm mode of an isolated coil, in seconds.

    ``[eta]`` arrives in cm^3/g and is converted to m^3/kg, which is the same
    number divided by a thousand - a conversion worth writing out, because
    getting it wrong moves the relaxation time by three orders of magnitude and
    the answer is still a plausible-looking time.
    """
    intrinsic_si = intrinsic * 1e-3
    mass_si = molar_mass * 1e-3
    return (
        _ZIMM_PREFACTOR
        * intrinsic_si
        * mass_si
        * solvent_viscosity
        / (_GAS_CONSTANT * temperature)
    )


def rayleigh_time(density: float, radius: float, surface_tension: float) -> float:
    """``sqrt(rho R^3 / sigma)``: the inertio-capillary break-up scale."""
    return math.sqrt(density * radius**3 / surface_tension)


def ohnesorge_number(
    viscosity: float, density: float, radius: float, surface_tension: float
) -> float:
    """Viscous against inertial-capillary. Large means viscosity slows break-up."""
    return viscosity / math.sqrt(density * surface_tension * radius)


def deborah_number(relaxation: float, rayleigh: float) -> float:
    return relaxation / rayleigh


def reynolds_number(
    density: float, velocity: float, diameter: float, viscosity: float
) -> float:
    """Inertia against viscosity in the nozzle. Above about 2000 the jet is
    turbulent when it leaves, and a turbulent jet atomises immediately rather
    than breaking up in an orderly way further downstream."""
    return density * velocity * diameter / viscosity


def weber_number(
    density: float, velocity: float, diameter: float, surface_tension: float
) -> float:
    """Inertia against surface tension. Large means the jet carries far before
    capillarity can gather it into drops."""
    return density * velocity**2 * diameter / surface_tension


def breakup_length(
    density: float,
    velocity: float,
    diameter: float,
    surface_tension: float,
    viscosity: float,
) -> float | None:
    """How far a coherent laminar jet travels before it becomes drops, m.

    Grant and Middleman's correlation for the laminar Rayleigh regime,
    ``L/d = 19.5 We^0.5 (1 + 3 Oh)^0.85``. Returns None above the laminar
    limit, where the jet atomises at the orifice and a break-up *length* is not
    the right description of what happens to it.

    This is a different question from the one :func:`assess_jet` answers. That
    one is about a thin filament being drawn down, where what matters is
    whether elasticity outruns capillary thinning. This one is about a thick
    jet flying ballistically, where what matters is how far it gets before
    capillarity closes on it - and a thick fast jet gets a long way, because
    the Weber number is enormous.
    """
    if reynolds_number(density, velocity, diameter, viscosity) > 2000.0:
        return None
    weber = weber_number(density, velocity, diameter, surface_tension)
    ohnesorge = viscosity / math.sqrt(density * surface_tension * diameter)
    return diameter * 19.5 * math.sqrt(weber) * (1.0 + 3.0 * ohnesorge) ** 0.85


def extrusion_pressure(
    viscosity: float, radius: float, velocity: float, length: float
) -> float:
    """Pressure to drive a melt or dope down a nozzle, Pa.

    Hagen-Poiseuille for a Newtonian fluid in a round channel, written in terms
    of the mean velocity rather than the volumetric flow. It scales as the
    inverse square of the radius, which is why a fine nozzle is expensive: at
    fixed velocity, halving the radius quadruples the pressure.

    A melt shear-thins hard, so feeding it a zero-shear viscosity gives an upper
    bound rather than an estimate. That is the useful direction here - if the
    bound is achievable the design is safe, and if it is absurd by four orders
    of magnitude the shear-thinning will not rescue it.
    """
    return 8.0 * viscosity * length * velocity / radius**2


def assess_jet(conditions: SpinningConditions) -> SpinningAssessment:
    """Which regime this dope and nozzle fall in."""
    intrinsic = intrinsic_viscosity(
        conditions.polymer, conditions.solvent, conditions.molar_mass
    )
    c_star = overlap_concentration(intrinsic)
    ratio = conditions.concentration / c_star

    zimm = zimm_relaxation_time(
        intrinsic, conditions.molar_mass, conditions.solvent_viscosity,
        conditions.temperature,
    )
    # Above the overlap concentration the chains interpenetrate and the
    # terminal time grows with concentration. The exponent is the reptation
    # scaling for a good solvent, 3/(3v-1) with v = 0.588.
    relaxation = zimm * ratio ** (3.0 / (3.0 * 0.588 - 1.0)) if ratio > 1.0 else zimm

    t_rayleigh = rayleigh_time(
        conditions.density, conditions.nozzle_radius, conditions.surface_tension
    )
    deborah = deborah_number(relaxation, t_rayleigh)
    ohnesorge = ohnesorge_number(
        conditions.solvent_viscosity, conditions.density,
        conditions.nozzle_radius, conditions.surface_tension,
    )

    if deborah > _DEBORAH_FILAMENT + _DEBORAH_MARGIN:
        regime = JetRegime.FILAMENT
    elif deborah < _DEBORAH_FILAMENT - _DEBORAH_MARGIN:
        regime = JetRegime.SPRAY
    else:
        regime = JetRegime.MARGINAL

    # Entanglement is roughly an order of magnitude above overlap. Below it the
    # chains stretch but cannot transmit stress along the filament.
    entangled = ratio > 6.0

    limitations = [
        "a regime, not a fibre: this says the filament survives capillary break-up, "
        "not that it draws down to a good one",
        "the concentrated-solution relaxation time is a scaling from the dilute Zimm "
        "value, not a measurement; a factor of two here moves the Deborah number by "
        "the same factor",
        "nothing here models solvent leaving the filament, so it does not say whether "
        "the thread solidifies before it lands",
    ]
    if relaxation > _PUMPABLE_RELAXATION_SECONDS:
        limitations.append(
            f"the relaxation time is {relaxation:.1f} s, longer than the shot itself: "
            "this dope is a rubbery gel rather than a pumpable liquid, it will melt "
            "fracture at the nozzle instead of flowing, and the Deborah number above "
            "no longer means what it means in the other rows"
        )
    if not entangled:
        limitations.append(
            f"c/c* is {ratio:.1f}: below the entanglement threshold the criterion is "
            "about isolated stretched coils and overstates what the filament can carry"
        )
    return SpinningAssessment(
        regime=regime,
        deborah=deborah,
        ohnesorge=ohnesorge,
        rayleigh_time=t_rayleigh,
        relaxation_time=relaxation,
        intrinsic_viscosity=intrinsic,
        overlap_concentration=c_star,
        concentration_ratio=ratio,
        entangled=entangled,
        limitations=tuple(limitations),
    )
