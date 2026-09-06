"""Canonical unit registry.

Specification section 11: "canonical unit registry; no ranking before
dimensional validation and condition matching."

Every physical value in the system carries a unit string parsed by a single
shared :mod:`pint` registry.  Comparison, normalization and ranking all convert
to canonical units first.  Temperature is handled explicitly because offset
units (degC, degF) do not convert by scaling and are a classic source of silent
error in property pipelines.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final

import pint

from .errors import DimensionalityError, UnitError

# A single process-wide registry.  Quantities from different pint registries
# cannot interoperate, so this must not be constructed per-module.
_REGISTRY: Final[pint.UnitRegistry] = pint.UnitRegistry(
    autoconvert_offset_to_baseunit=False,
)

DIMENSIONLESS: Final[str] = "dimensionless"

# Canonical unit per dimensionality.  SI base units are the default; these
# entries pin the cases where the SI base form is unhelpful for reporting.
_CANONICAL_BY_DIMENSIONALITY: Final[dict[str, str]] = {
    "[temperature]": "kelvin",
    "[mass] * [length] ** 2 / [time] ** 2 / [substance]": "joule / mole",
    "[mass] * [length] ** 2 / [time] ** 2 / [substance] / [temperature]": "joule / mole / kelvin",
    "[mass] / [length] / [time] ** 2": "pascal",
    "[mass] / [length] ** 3": "kilogram / meter ** 3",
    "[length] ** 3 / [substance]": "meter ** 3 / mole",
    "[mass] / [substance]": "kilogram / mole",
    "[mass] / [time] ** 2": "newton / meter",
    "[mass] ** 0.5 / [length] ** 0.5 / [time]": "pascal ** 0.5",
    # Bare energy appears only for orbital-energy differences, where the
    # electronvolt is the universal convention; joules would report a gap as
    # 1e-19 and help nobody.
    "[mass] * [length] ** 2 / [time] ** 2": "electron_volt",
    "[mass] / [length] / [time]": "pascal * second",
    # Likewise the debye for dipole moments; the SI coulomb-metre is ~1e-30.
    "[length] * [time] * [current]": "debye",
    "dimensionless": DIMENSIONLESS,
}


def registry() -> pint.UnitRegistry:
    """Return the shared unit registry."""
    return _REGISTRY


@lru_cache(maxsize=512)
def parse_unit(unit: str) -> pint.Unit:
    """Parse a unit string, raising :class:`UnitError` on failure."""
    text = (unit or "").strip()
    if text in ("", "1", "-", "none", "None"):
        text = DIMENSIONLESS
    try:
        return _REGISTRY.parse_units(text)
    except Exception as exc:  # pint raises a wide variety of parse errors
        raise UnitError(f"Cannot parse unit {unit!r}: {exc}") from exc


@lru_cache(maxsize=512)
def dimensionality(unit: str) -> str:
    """Return a stable string key for the dimensionality of ``unit``."""
    return str(parse_unit(unit).dimensionality)


def are_compatible(unit_a: str, unit_b: str) -> bool:
    """True when two units describe the same physical dimension."""
    return dimensionality(unit_a) == dimensionality(unit_b)


@lru_cache(maxsize=512)
def canonical_unit(unit: str) -> str:
    """Return the canonical unit string for the dimension of ``unit``."""
    dim = dimensionality(unit)
    pinned = _CANONICAL_BY_DIMENSIONALITY.get(dim)
    if pinned is not None:
        return pinned
    # Fall back to SI base units for dimensions we have not pinned.
    base = _REGISTRY.Quantity(1.0, parse_unit(unit)).to_base_units().units
    return str(base)


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a magnitude between units of the same dimension.

    Offset temperature units are converted through pint's temperature handling
    rather than by scaling, so 25 degC becomes 298.15 K and not 25 K.
    """
    src, dst = parse_unit(from_unit), parse_unit(to_unit)
    if src.dimensionality != dst.dimensionality:
        raise DimensionalityError(
            f"Cannot convert {from_unit!r} ({src.dimensionality}) to "
            f"{to_unit!r} ({dst.dimensionality}): incompatible dimensions."
        )
    if src == dst:
        return float(value)
    try:
        return float(_REGISTRY.Quantity(value, src).to(dst).magnitude)
    except Exception as exc:
        raise UnitError(f"Conversion {from_unit!r} -> {to_unit!r} failed: {exc}") from exc


def is_offset_unit(unit: str) -> bool:
    """True for units with a non-zero offset (degC, degF).

    Uncertainties and differences in offset units are scale-only quantities;
    callers must not convert a standard deviation with :func:`convert`.
    """
    try:
        return bool(_REGISTRY.Quantity(0.0, parse_unit(unit))._get_non_multiplicative_units())
    except Exception:
        return False


def convert_delta(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a *difference* (e.g. a standard deviation) between units.

    For offset units the offset cancels, so only the scale factor applies.
    """
    src, dst = parse_unit(from_unit), parse_unit(to_unit)
    if src.dimensionality != dst.dimensionality:
        raise DimensionalityError(
            f"Cannot convert delta {from_unit!r} to {to_unit!r}: incompatible dimensions."
        )
    if src == dst:
        return float(value)
    if dimensionality(from_unit) == "[temperature]":
        # Kelvin and Celsius share a degree size; Fahrenheit and Rankine do not.
        scale_src = _REGISTRY.Quantity(1.0, f"delta_{src}" if _is_degree(src) else src)
        scale_dst = f"delta_{dst}" if _is_degree(dst) else str(dst)
        return float(scale_src.to(scale_dst).magnitude * value)
    return float(_REGISTRY.Quantity(value, src).to(dst).magnitude)


def _is_degree(unit: pint.Unit) -> bool:
    return str(unit) in ("degree_Celsius", "degree_Fahrenheit")
