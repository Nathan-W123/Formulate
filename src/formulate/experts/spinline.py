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

#: How much axial drawing suppresses the capillary instability. A filament
#: under tension has its perturbations stretched along it faster than they
#: grow, and the stabilisation scales with how hard it is being drawn. Taken
#: as linear in the draw ratio, which is the conservative reading of it.
DRAW_STABILISATION = 1.0


def solidification_time(diameter: float, diffusivity: float = THERMAL_DIFFUSIVITY) -> float:
    """Seconds for a skin to reach the depth that holds the shape.

    ``t = (f R)^2 / alpha``. The square is the whole story: halving the
    diameter quarters the time, and no chemistry does anything comparable.
    """
    return (SKIN_FRACTION_FOR_SHAPE * diameter / 2.0) ** 2 / diffusivity


def extrusion_pressure(
    diameter: float, land: float, speed: float, viscosity: float, density: float
) -> float:
    """Hagen-Poiseuille through the die land, plus the kinetic head."""
    radius = diameter / 2.0
    return 8.0 * viscosity * land * speed / radius**2 + density * speed**2 / 2.0


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
        {"solidification_time", "extrusion_pressure", "filament_stability"}
    )
    dependencies = frozenset({"shear_viscosity", "surface_tension", "amorphous_density"})

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
            value = solidification_time(diameter)
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
                    "radius squared: this is a property of the geometry, and the material "
                    "barely enters it",
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
        if viscosity is None or density is None:
            missing = "melt viscosity" if viscosity is None else "density"
            return Prediction.unsupported(
                prop, self.id, f"the {missing} this rests on was not available"
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
        needed = solidification_time(diameter) * speed
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
