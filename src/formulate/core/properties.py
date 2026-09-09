"""Canonical property registry.

A property name is meaningless without a dimension, a family, and a statement
of whether it depends on conditions.  Registering them centrally is what lets
:mod:`formulate.evaluation` reject a prediction whose units do not match the
target, and lets the coordinator decide which expert families are applicable
to a request.

The ``family`` values follow the expert families of specification section 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from .errors import DimensionalityError, UnknownPropertyError
from .units import canonical_unit, dimensionality


class PropertyFamily(str, Enum):
    """Expert families defined by specification section 4."""

    THERMAL = "thermal"
    MECHANICAL = "mechanical"
    INTERFACIAL = "interfacial"
    ELECTRICAL = "electrical"
    CHEMICAL = "chemical"
    SPECIALIZED = "specialized"
    FEASIBILITY = "feasibility"
    STRUCTURAL = "structural"


@dataclass(frozen=True, slots=True)
class PropertyDef:
    """Definition of a single physical or derived property."""

    name: str
    canonical_unit: str
    family: PropertyFamily
    description: str
    condition_dependent: bool = False
    #: Physically admissible range in canonical units, used as a sanity gate.
    bounds: tuple[float | None, float | None] = (None, None)
    #: True when this property's errors are multiplicative rather than
    #: additive, so two predictions of it are compared on relative spread.
    #:
    #: The distinction decides which of two experts is believed. A boiling
    #: point is additive: half a kelvin is better than five kelvin whatever
    #: the value, and the absolute spread is the right comparison. A viscosity
    #: is not. It runs from a tenth of a millipascal second to a megapascal
    #: second, its error is naturally a factor rather than a difference - the
    #: benchmarks in this repository already score it on log10 for that reason
    #: - and comparing absolute spreads on it hands the decision to whichever
    #: expert predicts the smallest number, because a small prediction carries
    #: a small absolute error even when it is a wild guess.
    #:
    #: That is not hypothetical. Asked for the viscosity of hexanediol
    #: diacrylate, the group method gave 4.25 mPa s and called itself 31 per
    #: cent uncertain while corresponding states gave 0.175 and called itself
    #: 85 per cent uncertain, and the absolute comparison chose the second: it
    #: was a factor of twenty-four lower and admitted to being nearly three
    #: times less sure, and won on 0.15 against 1.32 millipascal seconds of
    #: spread. The selection rule preferred the answer that was more wrong for
    #: being smaller.
    multiplicative_error: bool = False

    @property
    def dimensionality(self) -> str:
        return dimensionality(self.canonical_unit)


def _p(
    name: str,
    unit: str,
    family: PropertyFamily,
    description: str,
    *,
    condition_dependent: bool = False,
    bounds: tuple[float | None, float | None] = (None, None),
    multiplicative_error: bool = False,
) -> PropertyDef:
    return PropertyDef(
        name=name,
        canonical_unit=canonical_unit(unit),
        family=family,
        description=description,
        condition_dependent=condition_dependent,
        bounds=bounds,
        multiplicative_error=multiplicative_error,
    )


_T = PropertyFamily.THERMAL
_C = PropertyFamily.CHEMICAL
_I = PropertyFamily.INTERFACIAL
_F = PropertyFamily.FEASIBILITY
_S = PropertyFamily.STRUCTURAL
_E = PropertyFamily.ELECTRICAL
_M = PropertyFamily.MECHANICAL

#: The properties Phase 1 can predict, plus structural descriptors used by
#: filters and diversity selection.  Phases 3-4 extend this with mechanical,
#: electrical and transport properties once QM/MD and polymer preparation land.
PROPERTY_REGISTRY: Final[dict[str, PropertyDef]] = {
    d.name: d
    for d in [
        _p("normal_boiling_point", "K", _T, "Boiling point at 1 atm.", bounds=(0.0, None)),
        _p("melting_point", "K", _T, "Normal melting point.", bounds=(0.0, None)),
        _p("critical_temperature", "K", _T, "Critical temperature.", bounds=(0.0, None)),
        _p("critical_pressure", "Pa", _T, "Critical pressure.", bounds=(0.0, None)),
        _p("critical_volume", "m^3/mol", _T, "Critical molar volume.", bounds=(0.0, None)),
        _p(
            "enthalpy_vaporization",
            "J/mol",
            _T,
            "Enthalpy of vaporization at the normal boiling point.",
            bounds=(0.0, None),
        ),
        _p("enthalpy_fusion", "J/mol", _T, "Enthalpy of fusion.", bounds=(0.0, None)),
        _p(
            "heat_capacity_gas",
            "J/mol/K",
            _T,
            "Ideal-gas heat capacity.",
            condition_dependent=True,
            bounds=(0.0, None),
        ),
        _p(
            "flash_point",
            "K",
            _T,
            "Closed-cup flash point.",
            bounds=(0.0, None),
        ),
        _p("logp", "", _C, "Octanol/water partition coefficient, log10.", bounds=(-15.0, 20.0)),
        _p("molar_refractivity", "cm^3/mol", _C, "Molar refractivity.", bounds=(0.0, None)),
        _p(
            "aqueous_solubility_logs",
            "",
            _C,
            "Aqueous solubility as log10(S / (mol/L)) at 298 K.",
            condition_dependent=True,
            bounds=(-15.0, 2.0),
        ),
        _p(
            "surface_tension",
            "N/m",
            _I,
            "Liquid/vapour surface tension.",
            condition_dependent=True,
            bounds=(0.0, None),
        ),
        _p(
            "hildebrand_solubility_parameter",
            "Pa^0.5",
            _I,
            "Hildebrand solubility parameter, sqrt(cohesive energy density).",
            condition_dependent=True,
            bounds=(0.0, None),
        ),
        _p(
            "liquid_density",
            "kg/m^3",
            _I,
            "Saturated liquid density.",
            condition_dependent=True,
            bounds=(0.0, None),
        ),
        _p(
            "molar_volume_liquid",
            "m^3/mol",
            _I,
            "Saturated liquid molar volume.",
            condition_dependent=True,
            bounds=(0.0, None),
        ),
        _p(
            "synthetic_accessibility",
            "",
            _F,
            "Ertl-Schuffenhauer synthetic accessibility score, 1 (easy) to 10 (hard).",
            bounds=(1.0, 10.0),
        ),
        _p("molar_mass", "g/mol", _S, "Molar mass.", bounds=(0.0, None)),
        _p("heavy_atom_count", "", _S, "Number of non-hydrogen atoms.", bounds=(0.0, None)),
        _p(
            "rotatable_bond_count", "", _S, "Number of rotatable bonds.", bounds=(0.0, None)
        ),
        _p("topological_polar_surface_area", "angstrom^2", _S, "TPSA.", bounds=(0.0, None)),
        _p("aromatic_atom_fraction", "", _S, "Fraction of heavy atoms that are aromatic.",
           bounds=(0.0, 1.0)),
        # --- Quantum-derived observables (specification section 7 outputs). ---
        # Registered so a target can request them and so the coordinator can
        # route them to QM; a property with no expert is reported as uncovered
        # rather than silently omitted.
        _p(
            "electronic_energy",
            "J/mol",
            _E,
            "Total electronic energy. Only differences between consistent "
            "calculations are meaningful; the absolute value is method-dependent.",
        ),
        _p(
            "homo_lumo_gap",
            "eV",
            _E,
            "Frontier orbital energy gap. An orbital-energy difference, not an "
            "optical or fundamental gap.",
            bounds=(0.0, None),
        ),
        _p("dipole_moment", "debye", _E, "Electric dipole moment.", bounds=(0.0, None)),
        _p(
            "refractive_index",
            "",
            _E,
            "Refractive index of the liquid, sodium D line unless stated otherwise.",
            condition_dependent=True,
            # Below one is not physical for a transparent liquid at optical
            # frequencies; nothing organic reaches four.
            bounds=(1.0, 4.0),
        ),
        _p(
            "interaction_energy",
            "J/mol",
            _C,
            "Binding energy of an assembly relative to its separated parts.",
            condition_dependent=True,
        ),
        _p(
            "atomization_energy",
            "J/mol",
            _C,
            "Energy to separate a molecule into free atoms.",
        ),
        # --- Dynamics-derived observables (specification section 6 workflows). ---
        _p(
            "self_diffusion_coefficient",
            "m^2/s",
            _I,
            "Self-diffusion coefficient from the Einstein relation.",
            condition_dependent=True,
            bounds=(0.0, None),
         multiplicative_error=True),
        _p(
            "shear_viscosity",
            "Pa*s",
            _I,
            "Shear viscosity.",
            condition_dependent=True,
            bounds=(0.0, None),
         multiplicative_error=True),
        _p(
            "extensional_viscosity",
            "Pa*s",
            _I,
            "Uniaxial extensional (Trouton) viscosity. For a Newtonian liquid this is "
            "exactly three times the shear viscosity; for a polymer solution it is a "
            "function of strain rate and strain history rather than a single number.",
            condition_dependent=True,
            bounds=(0.0, None),
         multiplicative_error=True),
        _p(
            "cohesive_energy_density",
            "J/m^3",
            _I,
            "Cohesive energy per unit volume; the square of the Hildebrand parameter.",
            condition_dependent=True,
            bounds=(0.0, None),
        ),
        _p(
            "work_of_separation",
            "J/m^2",
            _I,
            "Reversible work to separate an interface into two free surfaces.",
            condition_dependent=True,
            bounds=(0.0, None),
        ),
        _p(
            "radius_of_gyration",
            "angstrom",
            _S,
            "Mass-weighted radius of gyration, averaged over a conformational ensemble.",
            condition_dependent=True,
            bounds=(0.0, None),
        ),
        # --- Bulk properties requiring condensed-phase models (Phase 4). ---
        _p(
            "glass_transition_temperature",
            "K",
            _T,
            "Glass transition temperature.",
            bounds=(0.0, None),
        ),
        _p(
            "youngs_modulus", "Pa", _M, "Tensile elastic modulus.",
            condition_dependent=True, bounds=(0.0, None),
         multiplicative_error=True),
        _p(
            "bulk_modulus", "Pa", _M, "Isothermal bulk modulus.",
            condition_dependent=True, bounds=(0.0, None),
         multiplicative_error=True),
        _p(
            "shear_modulus", "Pa", _M, "Elastic shear modulus.",
            condition_dependent=True, bounds=(0.0, None),
         multiplicative_error=True),
        # A chain property, not a bulk one, and the thing that actually decides
        # whether a polymer is brittle or tough: below roughly two entanglement
        # lengths a chain cannot form a load-bearing network and the material
        # is brittle whatever its modulus says.
        _p(
            "entanglement_molar_mass",
            "g/mol",
            _M,
            "Molar mass between entanglements, from the chain's packing length.",
            bounds=(0.0, None),
        ),
        # Deliberately NOT called tensile_strength. A real strength is set by
        # the largest flaw in the specimen, which is a property of how it was
        # made; this is the flaw-free limit and exceeds any measured strength
        # by one to three orders of magnitude. Naming it apart is what stops
        # the two being compared.
        _p(
            "theoretical_strength",
            "Pa",
            _M,
            "Flaw-free upper bound on strength. Not a tensile strength: a real "
            "specimen fails at its largest defect, far below this.",
            condition_dependent=True,
            bounds=(0.0, None),
         multiplicative_error=True),
        # Deliberately not "liquid_density": a glassy polymer is not a
        # saturated liquid, and a semicrystalline sample is denser than its
        # amorphous phase by an amount that depends on how it was processed
        # rather than on what it is made of.  Naming the amorphous phase keeps
        # the prediction honest about which number it is.
        _p(
            "amorphous_density",
            "kg/m^3",
            _M,
            "Density of the amorphous phase of a polymer.",
            condition_dependent=True,
            bounds=(0.0, None),
        ),
        # --- Hansen solubility parameters (specification section 4,
        # "solubility/compatibility"; section 12, formulation compatibility).
        # The three components resolve what a single Hildebrand parameter
        # cannot: two liquids can share a total cohesive energy density while
        # being immiscible because one holds it in hydrogen bonds and the other
        # in dispersion forces.
        # --- Mixture phase behaviour (specification section 12, "validity of
        # recipes, phase/compatibility failures"). A Hansen distance says how
        # similar two liquids are; an activity coefficient says what the
        # mixture actually does, including separating into two phases.
        _p(
            "excess_gibbs_energy",
            "J/mol",
            _C,
            "Excess Gibbs energy of mixing, RT * sum(x_i ln gamma_i). Zero for an "
            "ideal mixture, positive when the components prefer their own kind.",
            condition_dependent=True,
        ),
        _p(
            "solubility_red",
            "",
            _C,
            "Hansen relative energy difference between a solvent and a named solute: the "
            "distance to the solute's solubility sphere centre divided by its radius. "
            "Below one dissolves, near one swells, above one does not.",
            condition_dependent=True,
            bounds=(0.0, None),
        ),
        _p(
            "mixing_stability",
            "",
            _C,
            "Minimum curvature of the molar Gibbs energy of mixing over composition, "
            "in units of RT. Negative means the mixture is unstable somewhere and "
            "separates into two phases.",
            condition_dependent=True,
        ),
        # Reaction kinetics. The registry had none, and the gap was not
        # cosmetic: a material that has to solidify in flight is chosen on how
        # fast it reacts, so a design whose deciding property is a rate could
        # not be searched at all. Both of these are chain-growth quantities;
        # a step-growth or ring-opening cure would need its own.
        _p(
            "propagation_rate_constant",
            "m^3/mol/s",
            _C,
            "Propagation rate coefficient of a chain-growth polymerisation, k_p. "
            "How fast one radical adds one monomer.",
            condition_dependent=True,
            bounds=(0.0, None),
         multiplicative_error=True),
        _p(
            "cure_time",
            "s",
            _C,
            "Time for a reacting liquid to stop flowing under a stated initiation "
            "system, taken as the time to reach half conversion. Not a gel time in "
            "the Flory-Stockmayer sense: that needs a crosslinker and a functionality, "
            "and this is the linear chain-growth case where solidification is "
            "vitrification rather than network formation.",
            condition_dependent=True,
            bounds=(0.0, None),
         multiplicative_error=True),
        # --- Toughness, architecture and hazard: the three things the acrylate
        # run could not be ranked on. ---
        #
        # A cured 1,6-hexanediol diacrylate network was the correct answer to a
        # spec stating cure time, boiling point, viscosity and density, and the
        # wrong material to build: brittle, a skin sensitiser, and a thermoset
        # that cannot be remelted. None of those three is a state property the
        # registry carried, so none of them could lose the candidate a single
        # point. These are the properties that make them rankable.
        _p(
            "elongation_at_break",
            "",
            _M,
            "Tensile strain at fracture, as a fraction rather than a percentage. "
            "The toughness proxy: a material that reaches its strength and then "
            "snaps at two per cent strain absorbs a hundredth of the energy of one "
            "that draws to two hundred, at the same strength.",
            condition_dependent=True,
            bounds=(0.0, None),
            # Handbook elongations for one polymer run from two per cent to
            # twelve hundred, and one compilation's range for a single grade
            # routinely spans a factor of six. That is a multiplicative error by
            # construction, and comparing two of them on absolute spread would
            # hand the decision to whichever expert predicted the brittler
            # material - exactly the failure the flag exists to prevent.
            multiplicative_error=True,
        ),
        # Whether a material sticks on contact, which is a bulk rheological
        # question rather than a surface-chemical one. It sits in the mechanical
        # family for that reason: the Dahlquist criterion is a ceiling on a
        # modulus, and a material stiffer than it does not stick however well
        # matched its surface energy is.
        _p(
            "tack",
            "",
            _M,
            "One if the material meets the Dahlquist criterion for pressure-sensitive "
            "tack - a storage modulus low enough to wet a rough surface under thumb "
            "pressure in the contact time available - and zero if it does not. Not a "
            "peel strength and not a work of adhesion: it is the rheological "
            "precondition without which neither of those can be collected.",
            condition_dependent=True,
            bounds=(0.0, 1.0),
        ),
        _p(
            "crosslink_density",
            "mol/m^3",
            _S,
            "Moles of elastically effective network strands per unit volume. Zero "
            "for a thermoplastic and only for a thermoplastic: a covalent network "
            "cannot be remelted, redissolved or reprocessed, and a spec that needs "
            "a hot melt is stating a requirement on this rather than a preference.",
            bounds=(0.0, None),
        ),
        # Hazard. Two GHS endpoints, read from a curated table and never inferred
        # from structure, because there is no structure-activity model here worth
        # putting a person's skin behind. What the table does not carry, the
        # expert refuses - which is why the endpoints are separate properties
        # rather than one "safe" score that would quietly average an unknown
        # against a zero.
        _p(
            "skin_sensitiser",
            "",
            PropertyFamily.SPECIALIZED,
            "One if the substance carries a GHS skin sensitisation classification "
            "(H317, Skin Sens. 1/1A/1B), zero if it has been screened and does not. "
            "Never inferred from structure: absence of a classification in the table "
            "is a refusal, not a zero.",
            bounds=(0.0, 1.0),
        ),
        _p(
            "acute_toxicity_category",
            "",
            PropertyFamily.SPECIALIZED,
            "Most severe GHS acute toxicity category across the oral, dermal and "
            "inhalation routes: 1 is fatal at the smallest dose, 4 is harmful, and 5 "
            "is reserved here for a substance screened against all three routes and "
            "classified on none. Lower is worse, so a requirement on it is a lower "
            "bound rather than an upper one.",
            bounds=(1.0, 5.0),
        ),
        _p(
            "carcinogen_category",
            "",
            PropertyFamily.SPECIALIZED,
            "GHS carcinogenicity: 1 for category 1A (known), 2 for 1B (presumed), 3 "
            "for category 2 (suspected), 4 for a substance screened and classified "
            "on none. Present because a hazard screen carrying only sensitisation "
            "and acute toxicity would report benzene as unobjectionable, and a false "
            "clean bill is worse than no bill at all.",
            bounds=(1.0, 4.0),
        ),
        _p(
            "hansen_dispersion",
            "Pa^0.5",
            _I,
            "Hansen dispersion component of the solubility parameter.",
            bounds=(0.0, None),
        ),
        _p(
            "hansen_polar",
            "Pa^0.5",
            _I,
            "Hansen polar (dipolar) component of the solubility parameter.",
            bounds=(0.0, None),
        ),
        _p(
            "hansen_hydrogen_bonding",
            "Pa^0.5",
            _I,
            "Hansen hydrogen-bonding component of the solubility parameter.",
            bounds=(0.0, None),
        ),
        _p(
            "hansen_distance",
            "Pa^0.5",
            _I,
            "Hansen distance Ra between the least compatible pair in a formulation.",
            bounds=(0.0, None),
        ),
        _p(
            "relative_energy_difference",
            "",
            _I,
            "Hansen distance divided by the interaction radius; below one predicts "
            "miscibility.",
            bounds=(0.0, None),
        ),
    ]
}


def get_property(name: str) -> PropertyDef:
    """Look up a property definition, raising if unregistered.

    Unregistered properties are an error rather than a warning: an unknown
    property cannot be dimensionally validated, and section 11 forbids ranking
    anything that has not been.
    """
    try:
        return PROPERTY_REGISTRY[name]
    except KeyError:
        raise UnknownPropertyError(
            f"Property {name!r} is not in the canonical registry. "
            f"Known properties: {', '.join(sorted(PROPERTY_REGISTRY))}"
        ) from None


def validate_unit_for(name: str, unit: str) -> None:
    """Raise :class:`DimensionalityError` if ``unit`` cannot express ``name``."""
    prop = get_property(name)
    if dimensionality(unit) != prop.dimensionality:
        raise DimensionalityError(
            f"Property {name!r} has dimension {prop.dimensionality} "
            f"(canonical unit {prop.canonical_unit!r}) but was given unit {unit!r} "
            f"with dimension {dimensionality(unit)}."
        )


def properties_in_family(family: PropertyFamily) -> list[str]:
    """All registered property names belonging to ``family``."""
    return sorted(n for n, d in PROPERTY_REGISTRY.items() if d.family is family)
