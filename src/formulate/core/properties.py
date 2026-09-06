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
