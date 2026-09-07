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
) -> PropertyDef:
    return PropertyDef(
        name=name,
        canonical_unit=canonical_unit(unit),
        family=family,
        description=description,
        condition_dependent=condition_dependent,
        bounds=bounds,
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
        ),
        _p(
            "shear_viscosity",
            "Pa*s",
            _I,
            "Shear viscosity.",
            condition_dependent=True,
            bounds=(0.0, None),
        ),
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
        ),
        _p(
            "bulk_modulus", "Pa", _M, "Isothermal bulk modulus.",
            condition_dependent=True, bounds=(0.0, None),
        ),
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
            "mixing_stability",
            "",
            _C,
            "Minimum curvature of the molar Gibbs energy of mixing over composition, "
            "in units of RT. Negative means the mixture is unstable somewhere and "
            "separates into two phases.",
            condition_dependent=True,
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
