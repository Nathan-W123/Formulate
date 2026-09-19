"""The filament, which is where every web-shooter dead end actually lived.

``examples/web_shooter`` ran a dozen iterations of the wrong search. It asked
what material sets fast enough, and no material does, because setting is
transport and transport goes as the square of the radius. A 4.4 mm strand
takes forty-eight seconds to solidify whether it is cyanoacrylate, a hot melt
or polycaprolactone. The chemistry was never the variable.

None of that was visible to the engine. A candidate carried what a material
*is* - its modulus, its melting point, its viscosity - and nothing about the
shape it is made into, and the three quantities that decided the design are
not properties of a material at all:

* how long a filament takes to solidify, which is radius squared over the
  thermal diffusivity,
* what it costs to push it through a die, which goes as one over radius
  squared,
* and whether it survives its own surface tension long enough to do either.

Those are properties of a material *at a geometry*, so the geometry is a
condition - ``Conditions.spinline`` - and this expert reads it exactly as the
adhesion expert reads a substrate.

**Why a drawn filament and not a jet.** The earlier analysis proved a fired
liquid cannot become a strong rope in mid-air: setting wants a small radius,
capillary breakup wants a large one or a viscous melt, and pressure wants a
large one or a thin melt. The three contradict and the feasible region is
empty. That result stands, and it is about a *free* jet.

A drawn filament is not a free jet. It is under axial tension, and axial
tension suppresses the Rayleigh instability rather than merely slowing it: a
perturbation that would grow into a bead is stretched out along the filament
faster than it can grow. That is why melt spinning works at all, why a
spider's dragline is drawn rather than squirted, and why the impossibility
result does not close the problem - it closes one architecture.

The stabilisation enters here through the draw ratio. Below a draw ratio of
one the filament is a free jet and the old result applies; above it the
filament thins as it travels, which both stabilises it and cuts the
solidification time it needs.
"""

from __future__ import annotations

import math

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest

#: Thermal diffusivity of an organic polymer, m^2/s. This is the one input
#: that is nearly a constant across the whole class: polymers run 0.9-1.3e-7
#: whatever they are made of, because conduction is through the same kind of
#: bonds at the same kind of density. The spread is carried as uncertainty.
THERMAL_DIFFUSIVITY = 1.0e-7
THERMAL_DIFFUSIVITY_RTOL = 0.20

#: Fraction of the radius that must solidify before a filament holds its own
#: shape. A skin does not need to reach the centre to stop it beading.
SKIN_FRACTION_FOR_SHAPE = 0.5

#: Thermal conductivity of a polymer melt, W/m/K, and of air.
POLYMER_CONDUCTIVITY = 0.25
AIR_CONDUCTIVITY = 0.030
AIR_DENSITY = 1.0
AIR_VISCOSITY = 2.2e-5
POLYMER_HEAT_CAPACITY = 2000.0

#: Temperature ratio a filament must fall through before it is set: from a melt
#: about 145 K above ambient to about 80 K above it, which for polyethylene is
#: roughly 165 C down to the 100 C where crystallisation runs fast.
COOLING_RATIO = 145.0 / 80.0

#: Gas constant, J/(mol K).
GAS_CONSTANT = 8.31446261815324

#: Cross-model power-law index. A flexible polymer melt shear-thins with an
#: index near 0.3, so the apparent viscosity falls as (lambda gamma)^0.7 once
#: the shear rate passes the inverse relaxation time. Generic across flexible
#: chains, and not fitted here.
POWER_LAW_INDEX = 0.3

#: Weissenberg number above which chains stretch faster than they relax, so the
#: extensional viscosity climbs and the thread resists thinning. Below it a jet
#: beads up however viscous it is - which is the property a spinning dope lives
#: or dies by, and the one the registry had no name for.
STRAIN_HARDENING_ONSET = 0.5

#: How much axial drawing suppresses the capillary instability. A filament
#: under tension has its perturbations stretched along it faster than they
#: grow, and the stabilisation scales with how hard it is being drawn. Taken
#: as linear in the draw ratio, which is the conservative reading of it.
DRAW_STABILISATION = 1.0


def heat_transfer_coefficient(diameter: float, speed: float) -> float:
    """Forced convection onto a fine cylinder in crossflow, W/m^2/K."""
    reynolds = AIR_DENSITY * speed * diameter / AIR_VISCOSITY
    nusselt = 0.3 + 0.62 * math.sqrt(max(reynolds, 1e-9)) * 0.7 ** (1.0 / 3.0)
    return nusselt * AIR_CONDUCTIVITY / diameter


def solidification_time(
    diameter: float,
    speed: float = 1.0,
    diffusivity: float = THERMAL_DIFFUSIVITY,
) -> float:
    """Seconds for a filament to cool to where it sets.

    Two regimes, and which one applies is the Biot number ``h R / k``: whether
    the bottleneck is getting heat out through the air film or through the
    polymer.

    The first version of this took the conduction case unconditionally, which
    is the ``Bi >> 1`` limit and is wrong for anything fine. A 60 um filament
    has ``Bi = 0.18``: the air film is the bottleneck, and the honest answer is
    five times slower than the conduction model gave. Conduction only takes
    over above about 330 um at spinning speeds, which is to say for the fat
    strands this project started with and abandoned.

    Both regimes still go as a power of the radius, so the structural finding -
    that setting is geometry and not chemistry - is unchanged. Only the number
    moves, and it moves the wrong way for the design, which is why it is worth
    getting right.
    """
    radius = diameter / 2.0
    h = heat_transfer_coefficient(diameter, speed)
    biot = h * radius / POLYMER_CONDUCTIVITY
    conduction = (SKIN_FRACTION_FOR_SHAPE * radius) ** 2 / diffusivity
    # Lumped capacitance: rho c V / (h A), with V/A = R/2 for a cylinder.
    convection = (
        AIR_DENSITY * 0.0 + 910.0
    ) * POLYMER_HEAT_CAPACITY * (radius / 2.0) / h * math.log(COOLING_RATIO)
    return max(conduction, convection) if biot >= 1.0 else convection


#: Bagley end correction, in die radii.
#:
#: The first version of this function was Hagen-Poiseuille through the land
#: plus the kinetic head, and it had an escape hatch: shorten the land and the
#: pressure goes to nothing. A search given that hatch walks straight into it,
#: and this one did - it proposed a 50 um orifice plate and reported a
#: comfortable pressure, which is not a design but a division by a length that
#: was allowed to reach zero.
#:
#: A real die costs pressure at its entrance whatever its land is. The melt
#: converges into the hole, and the extensional work of that convergence plus
#: the entry vortex is the Bagley end correction: the total drop is
#: ``2 tau_w (L/R + e)``, which is the same as adding ``e`` radii to the land.
#: For polymer melts ``e`` runs from about 2 at low rates to 10 or more for an
#: elastic melt at high rate; 5 is the usual working figure and is used here.
#:
#: For a 400 um hole that is a millimetre of equivalent land, which is longer
#: than the land the search wanted. So this is not a refinement - it is the
#: term that decides the answer for any short die.
BAGLEY_END_CORRECTION = 5.0


def extrusion_pressure(
    diameter: float, land: float, speed: float, viscosity: float, density: float
) -> float:
    """Hagen-Poiseuille through the land and entrance, plus the kinetic head."""
    radius = diameter / 2.0
    effective = land + BAGLEY_END_CORRECTION * radius
    return 8.0 * viscosity * effective * speed / radius**2 + density * speed**2 / 2.0


def breakup_length(
    diameter: float, speed: float, viscosity: float, density: float, surface_tension: float,
    draw_ratio: float = 1.0,
) -> float:
    """Distance before capillary forces pinch the filament off.

    The Weber growth rate for a viscous thread, carried over ten e-foldings,
    and then divided by the draw stabilisation: a filament being pulled has
    its perturbations stretched out along it as fast as they grow.
    """
    radius = diameter / 2.0
    tau = 2.0 * math.sqrt(density * radius**3 / surface_tension) + (
        6.0 * viscosity * radius / surface_tension
    )
    free_jet = 10.0 * tau / (2.0 * math.pi) * speed
    return free_jet * (1.0 + DRAW_STABILISATION * max(0.0, draw_ratio - 1.0))


def plateau_modulus(density: float, entanglement: float, temperature: float) -> float:
    """``G_N = rho R T / M_e``, Pa. The stiffness of the entangled network."""
    return density * GAS_CONSTANT * temperature / entanglement


def relaxation_time(viscosity: float, plateau: float) -> float:
    """``lambda = eta0 / G_N``, s. How long a stretched chain takes to forget."""
    return viscosity / plateau


def cross_viscosity(zero_shear: float, lam: float, shear_rate: float) -> float:
    """Cross model: the apparent viscosity at a shear rate."""
    return zero_shear / (1.0 + (lam * shear_rate) ** (1.0 - POWER_LAW_INDEX))


class SpinlineExpert(Expert):
    """Solidification, extrusion pressure and filament stability at a geometry."""

    id = "spinline"
    version = "1"
    method = (
        "radius-squared solidification, Hagen-Poiseuille extrusion pressure, and a "
        "draw-stabilised capillary breakup length, over a stated spinline geometry"
    )
    family = PropertyFamily.MECHANICAL
    supported_classes = frozenset({MaterialClass.POLYMER, MaterialClass.MIXTURE})
    supported_properties = frozenset(
        {
            "solidification_time",
            "extrusion_pressure",
            "filament_stability",
            "terminal_relaxation_time",
            "extensional_strain_hardening",
            "shear_thinning_ratio",
        }
    )
    #: Either density will do, and which one arrives says what is being spun.
    #: A melt or a blend has an amorphous density; a polymer dissolved in a
    #: solvent has a liquid one. Asking only for the first is what made a
    #: spinning DOPE - the thing a dry-spinning line actually holds - come back
    #: with every property refused for a density nobody could supply, on a
    #: candidate whose density is perfectly well defined.
    dependencies = frozenset(
        {
            "shear_viscosity",
            "surface_tension",
            "amorphous_density",
            "liquid_density",
            "entanglement_molar_mass",
        }
    )

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        basis = "spinline geometry stated in the conditions"
        return ApplicabilityDomain(basis=basis)

    def _geometry(self, request: PredictionRequest):
        spinline = request.conditions.spinline
        if spinline is None:
            return None, (
                "a filament property is defined at a geometry, and none was stated. "
                "Put one in the conditions, for example spinline: {die_diameter: 300 um, "
                "draw_ratio: 25, line_speed: 9 m/s}"
            )
        if spinline.die_diameter is None:
            return None, "the spinline states no die diameter, which every filament property rests on"
        return spinline, ""

    def _predict_one(self, prop, request: PredictionRequest, domain):
        spinline, reason = self._geometry(request)
        if spinline is None:
            return Prediction.unsupported(prop, self.id, reason)

        diameter = spinline.final_diameter.to("m").value
        draw = spinline.draw_ratio or 1.0

        if prop == "solidification_time":
            line_speed = (
                spinline.line_speed.to("m/s").value
                if spinline.line_speed is not None
                else 1.0
            )
            value = solidification_time(diameter, line_speed)
            return self._make(
                prop, value, "second", request, domain,
                std=value * 2.0 * THERMAL_DIFFUSIVITY_RTOL,
                kind=UncertaintyKind.EPISTEMIC,
                basis=(
                    "the thermal diffusivity of an organic polymer is 0.9-1.3e-7 m^2/s "
                    "whatever it is made of, and the time goes as its inverse"
                ),
                notes=(
                    f"die {spinline.die_diameter.to('m').value*1e6:.0f} um drawn "
                    f"{draw:g}x to {diameter*1e6:.1f} um",
                    "the air film is the bottleneck at this size, not conduction through "
                    "the polymer; both go as a power of the radius, so this is still a "
                    "property of the geometry rather than of the material",
                    "this is cooling only. Crystallisation has its own clock that nothing "
                    "here models - quiescent polyethylene takes tens of milliseconds and "
                    "spinline stress cuts that to a few, which is the mechanism the design "
                    "rests on and the largest thing taken on trust",
                ),
                conditions=request.conditions,
            )

        speed = (
            spinline.line_speed.to("m/s").value if spinline.line_speed is not None else None
        )
        if speed is None:
            return Prediction.unsupported(
                prop, self.id, "the spinline states no line speed"
            )
        viscosity = request.dependency_value("shear_viscosity", "Pa*s")
        density = request.dependency_value("amorphous_density", "kg/m^3")
        if density is None:
            density = request.dependency_value("liquid_density", "kg/m^3")
        if viscosity is None or density is None:
            missing = "melt viscosity" if viscosity is None else "density"
            return Prediction.unsupported(
                prop, self.id, f"the {missing} this rests on was not available"
            )

        if prop in ("terminal_relaxation_time", "extensional_strain_hardening",
                    "shear_thinning_ratio"):
            entanglement = request.dependency_value("entanglement_molar_mass", "kg/mol")
            if entanglement is None:
                return Prediction.unsupported(
                    prop, self.id,
                    "the entanglement molar mass this rests on was not available, and the "
                    "plateau modulus is what a relaxation time is measured against",
                )
            temperature = request.conditions.temperature
            t_kelvin = temperature.to("K").value if temperature is not None else 298.15
            plateau = plateau_modulus(density, entanglement, t_kelvin)
            lam = relaxation_time(viscosity, plateau)

            if prop == "terminal_relaxation_time":
                return self._make(
                    prop, lam, "second", request, domain,
                    std=lam * 1.0, kind=UncertaintyKind.EPISTEMIC,
                    basis="the melt viscosity behind it carries a decade of its own",
                    notes=(
                        f"plateau modulus {plateau/1e6:.2f} MPa from rho R T / M_e",
                        "the clock a flow has to beat: stretch the chain faster than this "
                        "and it stays stretched",
                    ),
                    conditions=request.conditions,
                )

            if prop == "shear_thinning_ratio":
                die_d = spinline.die_diameter.to("m").value
                die_speed = speed / draw
                shear_rate = 8.0 * die_speed / die_d
                apparent = cross_viscosity(viscosity, lam, shear_rate)
                value = viscosity / apparent
                return self._make(
                    prop, value, "dimensionless", request, domain,
                    std=value * 0.5, kind=UncertaintyKind.EPISTEMIC,
                    basis="a generic power-law index of 0.3 for a flexible chain, not fitted",
                    notes=(
                        f"die shear rate {shear_rate:.0f} /s against a relaxation time of "
                        f"{lam*1e3:.2f} ms, so Weissenberg {lam*shear_rate:.2f}",
                        f"apparent viscosity {apparent:.4g} Pa.s against {viscosity:.4g} at "
                        "rest: this is what lets a long chain be pushed at all",
                    ),
                    conditions=request.conditions,
                )

            # extensional_strain_hardening: Hencky strain over the draw time.
            solid = solidification_time(diameter, speed)
            strain_rate = math.log(max(draw, 1.0000001)) / solid
            value = lam * strain_rate
            return self._make(
                prop, value, "dimensionless", request, domain,
                std=value * 0.6, kind=UncertaintyKind.EPISTEMIC,
                basis="relaxation time and draw time each carry their own spread",
                notes=(
                    f"Hencky strain {math.log(max(draw,1.0000001)):.1f} over {solid*1e3:.1f} ms "
                    f"is {strain_rate:.0f} /s, against a relaxation time of {lam*1e3:.2f} ms",
                    (
                        "above the onset, so the chains stretch faster than they relax and "
                        "the thread resists thinning"
                        if value >= STRAIN_HARDENING_ONSET
                        else "below the onset: the chains relax as fast as the flow stretches "
                        "them, so this jet beads up rather than drawing"
                    ),
                ),
                conditions=request.conditions,
            )

        if prop == "extrusion_pressure":
            land = (
                spinline.die_land.to("m").value
                if spinline.die_land is not None
                else 2.0 * spinline.die_diameter.to("m").value
            )
            # The melt does NOT move at the line speed while it is in the die.
            # Mass is conserved, so the velocity ratio is the area ratio, which
            # is the draw ratio: a filament drawn a hundredfold leaves the die
            # a hundred times slower than it arrives at the far end. Using the
            # line speed here put a 600 um die at 5.9e10 Pa and made every
            # geometry look impossible. It is the point of drawing - the die
            # runs slow and wide, and the speed is bought afterwards, for free,
            # by pulling.
            die_speed = speed / draw
            value = extrusion_pressure(
                spinline.die_diameter.to("m").value, land, die_speed, viscosity, density
            )
            return self._make(
                prop, value, "pascal", request, domain,
                std=value * 0.5, kind=UncertaintyKind.EPISTEMIC,
                basis="dominated by the melt viscosity, which carries a decade of its own",
                notes=(
                    f"die {spinline.die_diameter.to('m').value*1e6:.0f} um, land "
                    f"{land*1e3:.2f} mm, melt moving {die_speed*1e3:.2f} mm/s in the die "
                    f"against {speed:g} m/s on the line, {viscosity:.3g} Pa.s",
                    f"plus a Bagley end correction of {BAGLEY_END_CORRECTION:.0f} radii "
                    f"= {BAGLEY_END_CORRECTION * spinline.die_diameter.to('m').value / 2 * 1e3:.2f} "
                    "mm of equivalent land, which is what the melt pays converging into "
                    "the hole. Shortening the land does not remove it",
                    "one over radius squared: a finer die costs pressure exactly as fast "
                    "as it buys solidification time",
                ),
                conditions=request.conditions,
            )

        surface_tension = request.dependency_value("surface_tension", "N/m")
        if surface_tension is None:
            return Prediction.unsupported(
                prop, self.id, "the melt surface tension this rests on was not available"
            )
        distance = breakup_length(diameter, speed, viscosity, density, surface_tension, draw)
        needed = solidification_time(diameter, speed) * speed
        value = distance / needed if needed > 0 else 0.0
        return self._make(
            prop, value, "dimensionless", request, domain,
            std=value * 0.6, kind=UncertaintyKind.EPISTEMIC,
            basis="the breakup length and the solidification time each carry their own spread",
            notes=(
                f"survives {distance*1e3:.1f} mm, needs {needed*1e3:.1f} mm to solidify",
                (
                    f"drawn {draw:g}x, which stabilises it: a filament under tension has its "
                    "perturbations stretched out along it as fast as they grow"
                    if draw > 1.0
                    else "undrawn, so this is a free jet and gets no stabilisation"
                ),
                "above one the filament solidifies before it beads; below one it does not",
            ),
            conditions=request.conditions,
        )
