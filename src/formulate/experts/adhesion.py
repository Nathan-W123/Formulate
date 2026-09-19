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
    "peo": SurfaceEnergy(
        30.0, 13.0, "poly(ethylene oxide), clean; the polar term is the ether oxygen", 3.0
    ),
    "glass": SurfaceEnergy(33.0, 45.0, "soda-lime glass in laboratory air", 15.0),
    "aluminium-oxide": SurfaceEnergy(35.0, 15.0, "aluminium with its native oxide, in air", 15.0),
    "steel": SurfaceEnergy(30.0, 15.0, "steel in air, with an adsorbed layer", 15.0),
}

#: The same table read from the other side, keyed by repeat-unit SMILES.
#:
#: Eight of the substrates above are polymers, and a polymer is a thing this
#: engine proposes as well as a thing it sticks to. Nothing new is measured
#: here: a clean polystyrene surface has one surface energy whether the
#: polystyrene is the wall or the web. The entry exists so that a polymer
#: candidate can be asked the question, which until now only a liquid could be.
#:
#: Which side of the interface matters. For a dry-spun strand the liquid wets
#: the target for the instant before the solvent leaves, and what is left
#: holding is the polymer. So the solvent's work of adhesion describes the
#: contact and the polymer's describes the joint, and it is the second one that
#: decides whether the web holds.
POLYMER_SURFACES: dict[str, str] = {
    "[*]CC[*]": "polyethylene",
    "[*]CC(C)[*]": "polypropylene",
    "[*]CC(c1ccccc1)[*]": "polystyrene",
    "[*]CC(Cl)[*]": "pvc",
    "[*]CC(C)(C(=O)OC)[*]": "pmma",
    "[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]": "pet",
    "[*]NCCCCCCNC(=O)CCCCC(=O)[*]": "nylon-6,6",
    "[*]CCO[*]": "peo",
    "[*]C(F)(F)C(F)(F)[*]": "ptfe",
}

#: Aliases, so a recipe may say what a person would say.
SUBSTRATE_ALIASES: dict[str, str] = {
    "teflon": "ptfe",
    "poly(ethylene-oxide)": "peo",
    "peg": "peo",
    "pe": "polyethylene",
    "pp": "polypropylene",
    "ps": "polystyrene",
    "aluminium": "aluminium-oxide",
    "aluminum": "aluminium-oxide",
    "alumina": "aluminium-oxide",
    "soda-lime-glass": "glass",
    "stainless-steel": "steel",
}


#: Polar fraction of a surface energy, from the polar fraction of the surface.
#:
#: This is what turns adhesion from a nineteen-substance table into something
#: any structure can be asked. The Owens-Wendt split needs two numbers, and the
#: total is already available from the Sugden parachor; what was missing was
#: how that total divides.
#:
#: It divides the way the surface does. RDKit's topological polar surface area
#: counts the area contributed by nitrogen and oxygen; Labute's approximate
#: surface area counts all of it. Their ratio is how much of the molecule's
#: outside is polar, and the polar fraction of the surface ENERGY follows it,
#: saturating rather than running past one:
#:
#:     x_polar = 1 - exp(-k * TPSA / ASA)
#:
#: One fitted constant, k = 0.2505, over the nineteen liquids and polymers
#: whose measured splits this module already held. Left out one at a time the
#: RMSE is 0.068 in polar fraction and the worst miss is 0.155, on formamide.
#: In the work of adhesion that error is damped further, because the polar
#: terms enter as a geometric mean rather than directly.
#:
#: What it misses: TPSA counts nitrogen and oxygen and not halogen, so PVC and
#: PTFE come out purely dispersive when they measure a few percent polar. That
#: is inside the stated spread, and it is why the spread is stated.
POLAR_FRACTION_K = 0.2505

#: One-sigma on a predicted polar fraction, from the held-out fit above.
POLAR_FRACTION_SPREAD = 0.068


def polar_fraction(smiles: str) -> float | None:
    """How much of a surface energy is polar, from structure alone."""
    import math

    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    area = rdMolDescriptors.CalcLabuteASA(mol)
    if area <= 0:
        return None
    ratio = rdMolDescriptors.CalcTPSA(mol) / area
    return 1.0 - math.exp(-POLAR_FRACTION_K * ratio)


#: Worst ratio of the parachor against measured liquid surface tensions.
#: Over the ten test liquids above it runs 0.94x to 1.19x, which is tighter
#: than on polymers - the parachor was devised for liquids and only borrowed
#: for repeat units.
LIQUID_PARACHOR_SPREAD = 1.19


def predicted_surface_energy(
    smiles: str, density_g_cm3: float, *, is_polymer: bool = False
) -> SurfaceEnergy | None:
    """Owens-Wendt components from structure, mJ/m^2.

    The total comes from the Sugden parachor and the split from the polar
    fraction of the surface. Both are additive over structure, so this answers
    for any molecule or repeat unit RDKit can read rather than for the
    nineteen that have been measured.
    """
    from .melt import (
        PARACHOR_SPREAD,
        molecular_surface_tension,
        parachor_surface_tension,
    )

    if is_polymer:
        total = parachor_surface_tension(smiles, density_g_cm3)
        ratio_spread = PARACHOR_SPREAD
        structure = smiles.replace("[*]", "[H]")
    else:
        total = molecular_surface_tension(smiles, density_g_cm3)
        ratio_spread = LIQUID_PARACHOR_SPREAD
        structure = smiles
    if total is None:
        return None
    fraction = polar_fraction(structure)
    if fraction is None:
        return None

    # Two independent errors: how big the surface energy is, and how it
    # divides. Both are carried, and the spread is quoted on the total.
    spread = total * (ratio_spread - 1.0) + total * POLAR_FRACTION_SPREAD
    return SurfaceEnergy(
        dispersive=total * (1.0 - fraction),
        polar=total * fraction,
        basis=(
            f"predicted: Sugden parachor at {density_g_cm3:.3f} g/cm^3, split "
            f"{fraction:.0%} polar from the polar fraction of the surface area"
        ),
        spread=spread,
    )


def resolve_substrate(name: str) -> tuple[str, SurfaceEnergy] | None:
    """A substrate by the name a recipe would use, or by its repeat unit.

    The named table is a catalogue of real surfaces and stays one - what glass
    or steel is worth in laboratory air is a measurement about the surface,
    not something to derive from a formula. What is new is that a recipe may
    also name a *polymer* substrate by its repeat-unit SMILES, and get the
    same parachor-and-polar-area treatment as a candidate. A web that has to
    stick to a plastic nobody tabulated is the ordinary case, not the exotic
    one.
    """
    key = name.strip().lower().replace("_", "-").replace(" ", "-")
    key = SUBSTRATE_ALIASES.get(key, key)
    entry = SUBSTRATES.get(key)
    if entry is not None:
        return key, entry

    raw = name.strip()
    if "[*]" not in raw:
        return None
    from formulate.experts.polymer import packing_model

    try:
        density = packing_model().density(raw)
    except Exception:
        return None
    if not density or density <= 0:
        return None
    predicted = predicted_surface_energy(raw, density, is_polymer=True)
    return (raw, predicted) if predicted is not None else None


def polymer_energy(
    candidate: "Candidate", density_g_cm3: float | None = None
) -> tuple[str, SurfaceEnergy] | None:
    """Owens-Wendt components for a homopolymer candidate, or None.

    Only a single-repeat-unit chain is answered. A copolymer's surface is not
    the mole-weighted average of its components' - the lower-energy unit
    enriches at the surface, by a factor that depends on block length and on
    how the sample was cooled - so a mixing rule here would be a guess with a
    systematic direction to its error.
    """
    from formulate import chem

    polymer = candidate.polymer
    if polymer is None:
        return None
    chain = [m for m in polymer.monomers if m.role is not MonomerRole.END_GROUP]
    if len(chain) != 1:
        return None

    smiles = chain[0].smiles
    key = None
    if chem.rdkit_available():
        canonical = chem.canonical_smiles(smiles)
        if canonical is not None:
            for repeat, name in POLYMER_SURFACES.items():
                if chem.canonical_smiles(repeat) == canonical:
                    key = name
                    break
    if key is None:
        key = POLYMER_SURFACES.get(smiles)
    if key is not None:
        return key, SUBSTRATES[key]
    if density_g_cm3 is None:
        return None
    predicted = predicted_surface_energy(smiles, density_g_cm3, is_polymer=True)
    return (smiles, predicted) if predicted is not None else None


def liquid_energy(
    smiles: str, density_g_cm3: float | None = None
) -> SurfaceEnergy | None:
    """Owens-Wendt components for a liquid, or None.

    Measured where the contact-angle literature has measured them, and
    predicted from the parachor and the polar surface fraction otherwise. The
    Hansen route was tried and refused - it puts water's dispersive part at
    half its measured value - and the structural route replacing it is checked
    against these same ten liquids.
    """
    from formulate import chem

    if chem.rdkit_available():
        canonical = chem.canonical_smiles(smiles)
        if canonical is not None:
            for key, value in LIQUIDS.items():
                if chem.canonical_smiles(key) == canonical:
                    return value
    tabulated = LIQUIDS.get(smiles)
    if tabulated is not None:
        return tabulated
    if density_g_cm3 is None:
        return None
    return predicted_surface_energy(smiles, density_g_cm3, is_polymer=False)


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


from formulate.core.candidate import Candidate, MaterialClass, MonomerRole  # noqa: E402
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
    version = "2"
    method = (
        "Owens-Wendt two-component work of adhesion, from measured surface energies "
        "where they exist and from the Sugden parachor split by polar surface area "
        "otherwise"
    )
    family = PropertyFamily.INTERFACIAL
    supported_classes = frozenset({MaterialClass.MOLECULE, MaterialClass.POLYMER})
    supported_properties = frozenset({"work_of_separation"})
    #: Either one, depending on the class. The parachor needs a density, and
    #: neither density is this expert's to predict.
    dependencies = frozenset({"amorphous_density", "liquid_density"})

    @staticmethod
    def _density(request) -> float | None:
        for prop in ("amorphous_density", "liquid_density"):
            value = request.dependency_value(prop, "g/cm^3")
            if value is not None and value > 0:
                return value
        return None

    def _adherend(
        self, candidate: Candidate, density_g_cm3: float | None = None
    ) -> tuple[str, SurfaceEnergy] | None:
        """The candidate's own surface energy, whichever class it is."""
        if candidate.material_class is MaterialClass.POLYMER:
            return polymer_energy(candidate, density_g_cm3)
        if candidate.molecule is None:
            return None
        found = liquid_energy(candidate.molecule.smiles, density_g_cm3)
        return (candidate.molecule.smiles, found) if found is not None else None

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
                f"no measured surface energy is tabulated for {surfaces[0]!r}, and it is "
                "not a repeat unit this module can read. Name one of "
                + ", ".join(sorted(SUBSTRATES))
                + ", or give a polymer substrate as its repeat-unit SMILES"
            )
        return found, ""

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        basis = (
            "measured dispersive and polar surface energies where they exist, "
            "Owens-Wendt convention; parachor and polar surface area otherwise"
        )
        # A nominal density only answers "can this structure be read at all";
        # the real one arrives with the request and decides the number.
        if self._adherend(candidate, 1.0) is not None:
            return ApplicabilityDomain(basis=basis)
        if candidate.material_class is MaterialClass.POLYMER:
            if candidate.polymer is None:
                return ApplicabilityDomain.outside("candidate carries no polymer", basis=basis)
            return ApplicabilityDomain.outside(
                "this candidate is a copolymer, whose surface is not the average of its "
                "components' because the lower-energy unit enriches there - or its "
                "repeat unit carries an element the parachor has no increment for",
                basis=basis,
            )
        if candidate.molecule is None:
            return ApplicabilityDomain.outside("candidate carries no molecule", basis=basis)
        return ApplicabilityDomain.outside(
            "this molecule carries an element the parachor has no increment for, and "
            "its dispersive/polar split is not tabulated either",
            basis=basis,
        )

    def _predict_one(self, prop, request, domain):
        candidate = request.candidate
        is_polymer = candidate.material_class is MaterialClass.POLYMER

        density = self._density(request)
        adherend = self._adherend(candidate, density)
        if adherend is None:
            if is_polymer and candidate.polymer is None:
                reason = "candidate carries no polymer"
            elif not is_polymer and candidate.molecule is None:
                reason = "candidate carries no molecule"
            elif density is None:
                reason = (
                    "no measured surface energy is tabulated for this "
                    + ("polymer" if is_polymer else "liquid")
                    + ", and the parachor that would replace it needs a density, which "
                    "no upstream expert could supply"
                )
            elif is_polymer:
                reason = (
                    "this candidate is a copolymer, whose surface is not the mole-weighted "
                    "average of its components' because the lower-energy unit enriches "
                    "there by an amount that depends on block length and thermal history "
                    "- or its repeat unit carries an element the parachor has no "
                    "increment for"
                )
            else:
                reason = (
                    "no measured dispersive/polar split is tabulated for this liquid, and "
                    "it carries an element the parachor has no increment for"
                )
            return Prediction.unsupported(prop, self.id, reason)
        adherend_name, phase = adherend

        found, reason = self._substrate(request)
        if found is None:
            return Prediction.unsupported(prop, self.id, reason)
        name, solid = found

        work = work_of_adhesion(phase, solid)
        spreading = spreading_coefficient(phase, solid)
        # A geometric mean of two uncertain numbers; the relative errors add in
        # quadrature and the solid usually dominates.
        relative = (
            (phase.spread / max(phase.total, 1e-9)) ** 2
            + (solid.spread / max(solid.total, 1e-9)) ** 2
        ) ** 0.5

        notes = [
            f"substrate {name}: {solid.basis}",
            f"{'polymer' if is_polymer else 'liquid'} components {phase.dispersive:.1f} "
            f"dispersive, {phase.polar:.1f} polar mJ/m^2",
        ]
        if is_polymer:
            notes.append(
                f"the adherend is the solid polymer ({phase.basis}), so this is the joint "
                "that remains after any carrier solvent has left, not the wetting of the "
                "target at the moment of contact"
            )
            notes.append(
                "a solid does not spread, so no contact angle follows from this; the "
                "number says how much reversible work the finished interface is worth"
            )
        else:
            notes.append(
                f"spreading coefficient {spreading:+.1f} mJ/m^2: the liquid "
                + ("spreads" if spreading > 0 else "beads and shows a finite contact angle")
            )
        notes.append(
            "this is the reversible thermodynamic work, not a peel strength or a lap "
            "shear: a real joint dissipates one to three orders of magnitude more energy "
            "deforming the adherends, and the two are not related by a constant"
        )
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
            adherend=adherend_name,
        )
