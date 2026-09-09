"""Thermodynamic work of adhesion between a liquid and a named substrate.

Specification section 12 counts a formulation's "phase/compatibility failures"
as a thing to be judged on, and adhesion is the one a coating or an adhesive
lives or dies by. The quantity here is the reversible work to separate a
wetted interface into two free surfaces, which the registry already carries as
``work_of_separation``: the same property the molecular-dynamics route refuses
because it needs two slabs of four hundred molecules.

The substrate is not part of the candidate, and should not be. A coating is a
candidate; the aluminium it is painted onto is a condition of use. It is read
from ``Conditions.surfaces``, which the schema has carried since the start for
exactly this.

Owens and Wendt's two-component form is what is computed:

    W = 2 sqrt(gd_1 gd_2) + 2 sqrt(gp_1 gp_2)

splitting each surface energy into a dispersive and a polar part, because a
single total cannot explain why water beads on poly(tetrafluoroethylene) and
wets glass: those two solids differ far more in the polar term than in the
total.

Both components come from tabulated measurements rather than from structure,
and the reason is worth recording. The obvious route was to split a measured
surface tension using the ratio of Hansen parameters already in the repository,
which would have covered any compound in the Hansen table. It does not work:
against published Owens-Wendt splits it puts water's dispersive component at
11.3 mN/m instead of 21.8, ethylene glycol's at 17.6 instead of 29.3, and
ethanol's at 10.2 instead of 18.8, while getting the non-polar liquids right
for the trivial reason that there is nothing to split. Hansen's decomposition
and Owens-Wendt's are different decompositions of different quantities, and the
one is not a rescaling of the other. Beerbower's published relation was tried
next and misses hexane's total by a factor of four and a half, which means the
constant or the exponent in the form recalled here is wrong; fitting one until
it matched would have produced a correlation with no provenance.

So this module refuses any liquid or substrate it does not have a measurement
for. That is a real limit on coverage and it is the same limit the Hansen
expert already works under.

What it does NOT produce is practical adhesion: peel strength, lap shear, or
anything a test method reports. Those exceed the thermodynamic work by one to
three orders of magnitude, because almost all the energy in peeling a real
joint goes into deforming the adherends rather than into creating surface.

For the same reason it does not answer whether something is *tacky*, and that
question turns out not to be a surface-chemical one at all.
:class:`DahlquistTackExpert`, at the bottom of this module, answers it from a
modulus: a material only sticks on contact if it is soft enough to deform into
the asperities of a real surface while it is pressed there, and no surface
energy substitutes for that. The two experts here therefore answer different
questions and neither is a refinement of the other.
Section 13 puts them outside what this system may claim, and the two are not
related by a constant that could be applied here.
"""

from __future__ import annotations

from dataclasses import dataclass

#: mN/m is the same number as mJ/m^2; both are used below and they are equal.
#: Values are the dispersive and polar components in the Owens-Wendt
#: convention at room temperature.


@dataclass(frozen=True, slots=True)
class SurfaceEnergy:
    """Dispersive and polar components of a surface energy, mJ/m^2."""

    dispersive: float
    polar: float
    #: Free text on what was measured. For a solid this decides the answer.
    basis: str
    #: One-sigma spread on the total, mJ/m^2.
    spread: float = 2.0

    @property
    def total(self) -> float:
        return self.dispersive + self.polar


#: Test liquids and common solvents whose components are tabulated.
#:
#: Keyed by canonical SMILES. These are the liquids the contact-angle
#: literature actually uses, which is why their splits are known at all.
LIQUIDS: dict[str, SurfaceEnergy] = {
    "O": SurfaceEnergy(21.8, 51.0, "water, the reference polar test liquid", 1.0),
    "OCCO": SurfaceEnergy(29.3, 19.0, "ethylene glycol, a standard test liquid", 1.5),
    "OCC(O)CO": SurfaceEnergy(34.0, 30.0, "glycerol, a standard test liquid", 2.0),
    "NC=O": SurfaceEnergy(39.5, 18.7, "formamide, a standard test liquid", 2.0),
    "ICI": SurfaceEnergy(50.8, 0.0, "diiodomethane, the reference dispersive test liquid", 1.0),
    "CCO": SurfaceEnergy(18.8, 3.2, "ethanol", 1.5),
    "CCCCO": SurfaceEnergy(20.2, 6.0, "1-butanol", 2.0),
    "Cc1ccccc1": SurfaceEnergy(27.9, 0.0, "toluene, essentially non-polar", 1.0),
    "CCCCCC": SurfaceEnergy(17.9, 0.0, "hexane, purely dispersive", 0.5),
    "ClC(Cl)Cl": SurfaceEnergy(27.2, 0.8, "chloroform", 1.5),
}

#: Substrates, named as they would be written in ``Conditions.surfaces``.
#:
#: The polymers are reliable: a clean polymer surface is reproducible and the
#: values agree between compilations to about a millijoule. The inorganics are
#: not, and the spread quoted for them is not a measurement uncertainty but a
#: statement that the number depends on what is adsorbed on the surface. A
#: freshly cleaved oxide in vacuum has a surface energy in the hundreds; the
#: same oxide in laboratory air, covered in water and hydrocarbon, is in the
#: tens. Which one applies is a question about the process, not the material,
#: so the value here is the ambient one and the uncertainty says so.
SUBSTRATES: dict[str, SurfaceEnergy] = {
    "ptfe": SurfaceEnergy(18.6, 0.5, "poly(tetrafluoroethylene), clean", 1.0),
    "polyethylene": SurfaceEnergy(33.0, 0.0, "polyethylene, clean and untreated", 1.5),
    "polypropylene": SurfaceEnergy(30.5, 0.0, "polypropylene, clean and untreated", 1.5),
    "polystyrene": SurfaceEnergy(41.4, 0.6, "polystyrene, clean", 1.5),
    "pvc": SurfaceEnergy(39.5, 1.5, "poly(vinyl chloride), clean", 2.0),
    "pmma": SurfaceEnergy(35.9, 4.3, "poly(methyl methacrylate), clean", 1.5),
    "pet": SurfaceEnergy(37.8, 3.5, "poly(ethylene terephthalate), clean", 1.5),
    "nylon-6,6": SurfaceEnergy(35.9, 5.3, "nylon-6,6, clean", 2.0),
    "glass": SurfaceEnergy(33.0, 45.0, "soda-lime glass in laboratory air", 15.0),
    "aluminium-oxide": SurfaceEnergy(35.0, 15.0, "aluminium with its native oxide, in air", 15.0),
    "steel": SurfaceEnergy(30.0, 15.0, "steel in air, with an adsorbed layer", 15.0),
}

#: Aliases, so a recipe may say what a person would say.
SUBSTRATE_ALIASES: dict[str, str] = {
    "teflon": "ptfe",
    "pe": "polyethylene",
    "pp": "polypropylene",
    "ps": "polystyrene",
    "aluminium": "aluminium-oxide",
    "aluminum": "aluminium-oxide",
    "alumina": "aluminium-oxide",
    "soda-lime-glass": "glass",
    "stainless-steel": "steel",
}


def resolve_substrate(name: str) -> tuple[str, SurfaceEnergy] | None:
    """Look up a substrate by the name a recipe would use."""
    key = name.strip().lower().replace("_", "-").replace(" ", "-")
    key = SUBSTRATE_ALIASES.get(key, key)
    entry = SUBSTRATES.get(key)
    return (key, entry) if entry is not None else None


def liquid_energy(smiles: str) -> SurfaceEnergy | None:
    """Tabulated components for a liquid, or None.

    None is not a failure to try harder. The components cannot be derived from
    a total surface tension, from Hansen parameters, or from structure by any
    route this module could verify, so a liquid outside the table has no
    answer here.
    """
    from formulate import chem

    if chem.rdkit_available():
        canonical = chem.canonical_smiles(smiles)
        if canonical is not None:
            for key, value in LIQUIDS.items():
                if chem.canonical_smiles(key) == canonical:
                    return value
    return LIQUIDS.get(smiles)


def work_of_adhesion(liquid: SurfaceEnergy, solid: SurfaceEnergy) -> float:
    """Owens-Wendt work of adhesion, mJ/m^2.

    Geometric means of the like components. The polar term is what separates a
    surface water wets from one it beads on, and dropping it - as the
    single-component Girifalco-Good form does - makes poly(tetrafluoroethylene)
    and poly(methyl methacrylate) look far more alike to water than they are.
    """
    import math

    return 2.0 * math.sqrt(liquid.dispersive * solid.dispersive) + 2.0 * math.sqrt(
        liquid.polar * solid.polar
    )


def spreading_coefficient(liquid: SurfaceEnergy, solid: SurfaceEnergy) -> float:
    """Work of adhesion less the liquid's own cohesion, mJ/m^2.

    Positive means the liquid spreads on the solid of its own accord; negative
    means it beads and the contact angle is finite. This is the quantity a
    coating formulator is usually after, and it is a difference of two similar
    numbers, so it carries the sum of both uncertainties rather than either.
    """
    return work_of_adhesion(liquid, solid) - 2.0 * liquid.total


# --------------------------------------------------------------------------
# The expert
# --------------------------------------------------------------------------


from formulate.core.candidate import Candidate, MaterialClass  # noqa: E402
from formulate.core.prediction import Prediction  # noqa: E402
from formulate.core.properties import PropertyFamily  # noqa: E402
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind  # noqa: E402

from .base import Expert  # noqa: E402


class AdhesionExpert(Expert):
    """Work of separation for a liquid on a named substrate.

    The same property the dynamics route declines. That one would compute it
    from two slabs and four hundred molecules, which this installation cannot
    afford; this one reads it off measured surface energies in microseconds and
    is right about which way water runs. Both are legitimate answers at
    different fidelity, and the evaluation engine decides between them on
    uncertainty rather than on preference.
    """

    id = "adhesion"
    version = "1"
    method = "Owens-Wendt two-component work of adhesion from tabulated surface energies"
    family = PropertyFamily.INTERFACIAL
    supported_classes = frozenset({MaterialClass.MOLECULE})
    supported_properties = frozenset({"work_of_separation"})

    def _substrate(self, request):
        surfaces = request.conditions.surfaces
        if not surfaces:
            return None, (
                "adhesion is a property of an interface, and no substrate was named. "
                "State one in the conditions, for example surfaces: ['aluminium-oxide']"
            )
        if len(surfaces) > 1:
            return None, (
                f"{len(surfaces)} surfaces were named and a work of adhesion is defined "
                "for one interface at a time"
            )
        found = resolve_substrate(surfaces[0])
        if found is None:
            return None, (
                f"no measured surface energy is tabulated for {surfaces[0]!r}. Known "
                "substrates: " + ", ".join(sorted(SUBSTRATES))
            )
        return found, ""

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        basis = "measured dispersive and polar surface energies, Owens-Wendt convention"
        if candidate.molecule is None:
            return ApplicabilityDomain.outside("candidate carries no molecule", basis=basis)
        if liquid_energy(candidate.molecule.smiles) is None:
            return ApplicabilityDomain.outside(
                "this liquid's dispersive and polar components are not tabulated, and "
                "they cannot be derived from a total surface tension or from Hansen "
                "parameters",
                basis=basis,
            )
        return ApplicabilityDomain(basis=basis)

    def _predict_one(self, prop, request, domain):
        molecule = request.candidate.molecule
        if molecule is None:
            return Prediction.unsupported(prop, self.id, "candidate carries no molecule")

        liquid = liquid_energy(molecule.smiles)
        if liquid is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "no measured dispersive/polar split is tabulated for this liquid; the "
                "split cannot be derived from a total surface tension, and the Hansen "
                "route was tested and puts water's dispersive part at half its measured "
                "value",
            )

        found, reason = self._substrate(request)
        if found is None:
            return Prediction.unsupported(prop, self.id, reason)
        name, solid = found

        work = work_of_adhesion(liquid, solid)
        spreading = spreading_coefficient(liquid, solid)
        # A geometric mean of two uncertain numbers; the relative errors add in
        # quadrature and the solid usually dominates.
        relative = (
            (liquid.spread / max(liquid.total, 1e-9)) ** 2
            + (solid.spread / max(solid.total, 1e-9)) ** 2
        ) ** 0.5

        notes = [
            f"substrate {name}: {solid.basis}",
            f"liquid components {liquid.dispersive:.1f} dispersive, "
            f"{liquid.polar:.1f} polar mJ/m^2",
            (
                f"spreading coefficient {spreading:+.1f} mJ/m^2: the liquid "
                + ("spreads" if spreading > 0 else "beads and shows a finite contact angle")
            ),
            "this is the reversible thermodynamic work, not a peel strength or a lap "
            "shear: a real joint dissipates one to three orders of magnitude more energy "
            "deforming the adherends, and the two are not related by a constant",
        ]
        if solid.spread >= 10.0:
            notes.append(
                "the substrate is an inorganic surface whose energy is set by what is "
                "adsorbed on it rather than by the bulk material; the value used is for "
                "laboratory air, and a cleaned or plasma-treated surface is a different "
                "material for this purpose"
            )

        return self._make(
            prop,
            work / 1000.0,  # mJ/m^2 -> J/m^2, which is N/m
            "N/m",
            request,
            domain,
            std=work * relative / 1000.0,
            kind=UncertaintyKind.COMBINED,
            basis=(
                "propagated from the spreads on both surface energies; for an inorganic "
                "substrate that spread is surface state, not measurement precision"
            ),
            notes=tuple(notes),
            substrate=name,
        )


# ---------------------------------------------------------------------------
# Tack, which is a rheological question and not a surface-chemical one
# ---------------------------------------------------------------------------

#: The Dahlquist criterion, in Pa: the storage modulus at 1 Hz and use
#: temperature below which a material wets a rough surface under thumb pressure
#: in the contact time available.
#:
#: It is quoted as 1e5 Pa as often as 3e5, and the disagreement is not sloppy
#: citation - it is the criterion having a soft edge, because "sticks" depends
#: on how rough the surface is, how hard it was pressed and for how long. Both
#: ends are carried rather than one, and a material inside the band is refused
#: instead of being called tacky or not: the width IS the uncertainty, and
#: collapsing it to a single threshold would manufacture a decision the
#: criterion does not support.
DAHLQUIST_LOW = 1.0e5
DAHLQUIST_HIGH = 3.0e5


class DahlquistTackExpert(Expert):
    """Whether a polymer sticks on contact, from its modulus alone.

    The module above computes a thermodynamic work of adhesion, and this expert
    exists because that quantity does not answer the question people ask it.
    Surface energy decides whether a liquid *wets*; it does not decide whether
    a solid *sticks*, and the gap between the two is the whole of pressure
    sensitive adhesion. A poly(tetrafluoroethylene) film and a
    poly(tetrafluoroethylene) grease have identical surface chemistry and only
    one of them is an adhesive.

    What decides it is stiffness. Dahlquist's observation is that a material
    only adheres on contact if its storage modulus is below about 10^5 to
    3x10^5 Pa at 1 Hz, because a stiffer material cannot deform into the
    asperities of a real surface in the second or so it is pressed there. No
    amount of favourable surface energy substitutes for that, which is why this
    expert reads a modulus and not a surface tension.

    The consequence for a load-bearing strand is three and a half orders of
    magnitude: a filament that carries load is a glass at about 10^9 Pa, and
    tack needs 10^5. **At one temperature no single material occupies both.**

    That qualifier is load-bearing and an earlier version of this docstring
    left it out, which made the claim false. A hot melt occupies both *at
    different temperatures*, and that is precisely what a hot melt is: molten
    it wets anything, and cold it carries load. The same numbers say so.
    Under a thumb pressure of 10^5 Pa held for one second, a glass at 1.06 GPa
    reaches a strain of 9.4e-5 and conforms to nothing, while a polystyrene
    melt at 180 C - 3657 Pa s from this panel's own melt expert - flows to a
    strain of 27 and conforms to everything. Same material, 290,000 times the
    deformation, and the only difference is temperature.

    So the criterion here is evaluated *at the temperature it is asked about*
    and says nothing about any other. What closes the one-material escape for a
    thrown strand is not materials science but the clock: this repository's own
    cooling calculation puts a 20 micron filament's vitrification at 173
    microseconds against a flight of half a second, so it arrives some three
    thousand times too late to be molten. A hot melt that could stick on
    contact would have to reach the wall still hot, and it does not.

    **Three things this does not do.** It does not measure tack - a probe-tack
    or loop-tack test reports a force, and this reports whether the
    precondition for one is met. It does not predict bond strength, which is
    dominated by viscoelastic dissipation during separation and routinely
    exceeds the thermodynamic work by two to three orders of magnitude. And it
    reads a static plateau modulus where the criterion is stated for the
    storage modulus at 1 Hz; for a glass three thousand times over the line
    that substitution changes nothing, and near the line it decides the answer,
    which is why near the line the expert declines.
    """

    id = "tack_dahlquist"
    version = "1"
    method = "Dahlquist modulus criterion for pressure-sensitive tack"
    family = PropertyFamily.MECHANICAL
    supported_classes = frozenset({MaterialClass.POLYMER})
    supported_properties = frozenset({"tack"})
    dependencies = frozenset({"shear_modulus"})

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        basis = (
            f"a modulus ceiling of {DAHLQUIST_LOW:.0e} to {DAHLQUIST_HIGH:.0e} Pa, which "
            "is a criterion rather than a correlation"
        )
        if candidate.polymer is None:
            return ApplicabilityDomain.outside("candidate carries no polymer", basis=basis)
        return ApplicabilityDomain(basis=basis)

    def _predict_one(self, prop, request, domain) -> Prediction | None:
        upstream = request.dependency("shear_modulus")
        if upstream is None or upstream.quantity is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "tack is a statement about a modulus and no upstream expert supplied "
                "one; surface energy does not substitute for it, which is the whole "
                "point of this expert sitting beside the Owens-Wendt one",
            )
        modulus = upstream.quantity.to("Pa").value
        std = upstream.uncertainty.converted(upstream.quantity.unit, "Pa").std

        # The modulus is registered as a multiplicative-error property, so its
        # spread is a factor rather than an amount and the interval has to be
        # built that way: a plus-or-minus on a rubbery modulus whose relative
        # spread is one would otherwise reach zero.
        factor = 1.0 + (std / modulus if std and modulus > 0 else 0.0)
        low, high = modulus / factor, modulus * factor

        temperature = request.conditions.temperature_k
        where = "" if temperature is None else f" at {temperature:.0f} K"
        interval = (
            f"the shear modulus{where} is {modulus:.3g} Pa, a factor of {factor:.2f} "
            f"either way, so {low:.3g} to {high:.3g} Pa"
        )

        if high < DAHLQUIST_LOW:
            value, verdict = 1.0, (
                f"clear of the criterion's lower edge of {DAHLQUIST_LOW:.0e} Pa: soft "
                "enough to wet a rough surface under thumb pressure"
            )
        elif low > DAHLQUIST_HIGH:
            value, verdict = 0.0, (
                f"clear of the criterion's upper edge of {DAHLQUIST_HIGH:.0e} Pa by a "
                f"factor of {low / DAHLQUIST_HIGH:.3g}: too stiff to deform into the "
                "asperities of a real surface in the time it is pressed there"
            )
        else:
            return Prediction.unsupported(
                prop,
                self.id,
                f"{interval}, which straddles the criterion's own "
                f"{DAHLQUIST_LOW:.0e} to {DAHLQUIST_HIGH:.0e} Pa band. Inside that band "
                "the answer is decided by the difference between the storage modulus at "
                "1 Hz that Dahlquist states and the static plateau this panel produces, "
                "and by how rough the surface is and how hard it was pressed - none of "
                "which is modelled here. Calling it tacky or not would be inventing the "
                "decision rather than making it",
            )

        notes = [
            interval,
            verdict,
            f"modulus via {upstream.expert_id}",
            "a precondition, not a measurement: a probe-tack test reports a force, and "
            "bond strength is dominated by viscoelastic dissipation during separation, "
            "which exceeds the thermodynamic work of adhesion by two to three orders of "
            "magnitude and is not computed anywhere in this panel",
            "Dahlquist states the storage modulus at 1 Hz and this is a static plateau; "
            "the substitution is harmless this far from the band and would not be near it",
        ]
        if value == 0.0 and modulus > 1.0e8:
            notes.append(
                "this is a glass rather than a near miss: about 10^9 Pa against the "
                "10^5 tack needs. At THIS temperature no material is both, but the same "
                "material molten is - a hot melt wets while liquid and carries load once "
                "cold - so this is a verdict about a temperature and not about a "
                "substance. Whether the hot route is open depends on whether the "
                "material is still molten when it makes contact, which is a cooling "
                "question this panel does not answer"
            )

        return self._make(
            prop,
            value,
            "",
            request,
            domain,
            # Zero because the classification is only issued once the modulus
            # and its own spread sit clear of the criterion's band; inside it
            # nothing is issued at all. The zero is that construction, not a
            # claim that tack is a certain thing.
            std=0.0,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                "issued only when the modulus interval clears the criterion's own "
                f"{DAHLQUIST_LOW:.0e}-{DAHLQUIST_HIGH:.0e} Pa band; inside the band the "
                "expert refuses rather than reporting a zero or a one"
            ),
            notes=tuple(notes),
            shear_modulus_pa=round(modulus, 3),
        )
