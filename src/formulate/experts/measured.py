"""Measured pure-component properties.

Specification section 4's model policy asks for open sources to be reused
where they are scientifically adequate rather than for new predictors to be
fitted.  A compiled measurement is the strongest case of that: where a value
has been measured, estimating it instead is a choice to be less accurate.

This expert therefore sits alongside the group-contribution and correlation
experts rather than replacing them.  Where a compound has been measured it
supplies the measurement with a small uncertainty; where it has not, it
supplies nothing at all, and the estimating experts answer.  The evaluation
engine already prefers the in-domain prediction with the tighter spread, so
no precedence logic is needed here - the measurement wins because it is
better, not because it is privileged.
"""

from __future__ import annotations

import functools

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest
from .hansen import resolve_cas

try:  # pragma: no cover
    from chemicals import MW, Tb, Tc, Tm

    _HAVE_CHEMICALS = True
    _IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    _HAVE_CHEMICALS = False
    _IMPORT_ERROR = str(exc)


#: Property -> (lookup, unit, one-sigma spread, what the spread represents).
#:
#: The spreads are not measurement precision, which would be far smaller. They
#: are the disagreement between compilations for the same substance, which
#: comes from sample purity, pressure correction and which original source was
#: preferred. Quoting instrument precision here would be the more precise lie.
_LOOKUPS = {
    "normal_boiling_point": (
        "Tb",
        "K",
        0.5,
        "spread between compilations for a measured boiling point, driven by purity and "
        "the pressure correction to one atmosphere rather than by instrument precision",
    ),
    "melting_point": (
        "Tm",
        "K",
        1.0,
        "spread between compilations for a measured melting point; polymorphism and "
        "purity move it more than the measurement does",
    ),
    "molar_mass": (
        "MW",
        "g/mol",
        0.0,
        "computed exactly from the standard atomic weights",
    ),
    "surface_tension": (
        "sigma",
        "N/m",
        0.0005,
        "spread between compilations for a measured surface tension; the liquid must be "
        "clean, because a trace of surfactant lowers it far more than the measurement "
        "uncertainty",
    ),
    "critical_temperature": (
        "Tc",
        "K",
        2.0,
        "spread between compilations for a measured critical temperature; a substance "
        "that decomposes near its critical point is extrapolated rather than measured "
        "and the disagreement is larger than this",
    ),
    "liquid_density": (
        "rho",
        "kg/m^3",
        1.0,
        "measured against the reference set at 25 degrees Celsius: 43 of the 48 tabulated "
        "densities are covered by a data method, mean absolute error 0.07 per cent and "
        "worst case 0.38. One kilogram per cubic metre is about a tenth of a per cent of "
        "a typical organic liquid and is where compilations disagree with each other",
    ),
}

#: Liquid-density methods in ``thermo`` that carry tabulated data or a
#: reference equation of state, as opposed to predicting from structure.
#:
#: The same distinction as for surface tension, and it matters more here
#: because the fallbacks are silent. ``thermo`` will drop through to Rackett,
#: Yen-Woods or Campbell-Thodos, which need only critical constants and return
#: a number for anything; the panel's own corresponding-states route is already
#: that, and the point of this expert is to be better than it or say nothing.
#:
#: ``HTCOSTALDFIT``, ``RACKETTFIT`` and ``MMSNM0FIT`` are deliberately absent
#: even though they are fitted to real densities. They were not needed: every
#: reference compound that resolves to a CAS number is covered by the list
#: below, so admitting a fitted correlation would widen the door without
#: answering a single extra compound.
_MEASURED_DENSITY_METHODS = (
    "HEOS_FIT",
    "COOLPROP",
    "DIPPR_PERRY_8E",
    "VDI_PPDS",
    "VDI_TABULAR",
    "CRC_INORG_L",
    "CRC_INORG_L_CONST",
)

#: Surface-tension methods in ``thermo`` that are fits to measured data rather
#: than predictions from structure.
#:
#: The distinction is the whole point of this expert. ``thermo`` will happily
#: fall back to Brock-Bird or Sastri-Rao, which are corresponding-states
#: correlations for non-associating fluids and are the very things this is
#: meant to displace: the panel's own Brock-Bird route refuses water outright
#: and overestimates ethanol by 63 per cent and ethylene glycol by 65.
_MEASURED_SURFACE_TENSION_METHODS = (
    "IAPWS_SIGMA",
    "REFPROP_FIT",
    "REFPROP",
    "SOMAYAJULU2",
    "SOMAYAJULU",
    "VDI_PPDS",
    "VDI_TABULAR",
    "JASPER",
)


@functools.lru_cache(maxsize=4096)
def measured_value(prop: str, smiles: str) -> float | None:
    """Look up a measured value for a structure, or None if it is not tabulated."""
    if not _HAVE_CHEMICALS:
        return None
    cas = resolve_cas(smiles)
    if cas is None:
        return None
    name = _LOOKUPS[prop][0]
    if name == "sigma":
        return _measured_surface_tension(cas)
    if name == "rho":
        return _measured_liquid_density(cas)
    try:
        value = {"Tb": Tb, "Tm": Tm, "MW": MW, "Tc": Tc}[name](cas)
    except Exception:
        return None
    return None if value is None else float(value)


def _measured_liquid_density(cas: str, temperature: float = 298.15) -> float | None:
    """Density at 25 degrees Celsius in kg/m^3, from a data method only.

    This is what unblocks water, and through water most of the formulation
    panel. Liquid density reached the mixture experts by one route only - a
    corresponding-states correlation that needs a critical temperature, which
    came from Joback group contribution, which cannot type a molecule with no
    carbon in it. So water had no critical temperature, therefore no density,
    therefore no volume fraction, and every blend containing it lost its
    density and all three volume-weighted Hansen parameters at once. Water's
    density is one of the better known numbers in physical chemistry and was
    sitting in a compilation already installed here.
    """
    try:
        from chemicals import MW as molar_mass
        from thermo import VolumeLiquid

        model = VolumeLiquid(CASRN=cas)
        mass = molar_mass(cas)
    except Exception:
        return None
    if not mass:
        return None
    available = set(getattr(model, "all_methods", ()) or ())
    for method in _MEASURED_DENSITY_METHODS:
        if method not in available:
            continue
        try:
            molar_volume = model.calculate(temperature, method)
        except Exception:
            continue
        if molar_volume and molar_volume == molar_volume and molar_volume > 0:
            # g/mol / (m^3/mol) = g/m^3; a thousandth of that is kg/m^3.
            return float(mass / molar_volume / 1000.0)
    return None


def _measured_surface_tension(cas: str, temperature: float = 298.15) -> float | None:
    """Surface tension at 298 K, from a data method only.

    Returns None rather than falling through to a correlation: an estimate is
    what the interfacial expert already provides, and the point of this expert
    is to be better than it or silent.
    """
    try:
        from thermo import SurfaceTension

        model = SurfaceTension(CASRN=cas)
    except Exception:
        return None
    available = set(getattr(model, "all_methods", ()) or ())
    for method in _MEASURED_SURFACE_TENSION_METHODS:
        if method not in available:
            continue
        try:
            value = model.calculate(temperature, method)
        except Exception:
            continue
        if value is not None and value == value and value > 0:
            return float(value)
    return None


class MeasuredPropertyExpert(Expert):
    """Supplies compiled experimental values where they exist."""

    id = "measured"
    version = "1"
    method = "compiled experimental pure-component data (via the chemicals package)"
    family = PropertyFamily.THERMAL
    supported_classes = frozenset({MaterialClass.MOLECULE})
    supported_properties = frozenset(_LOOKUPS)

    def is_available(self) -> bool:
        from formulate import chem

        return _HAVE_CHEMICALS and chem.rdkit_available()

    def unavailable_reason(self) -> str:
        from formulate import chem

        if not _HAVE_CHEMICALS:
            return f"the chemicals package is not installed ({_IMPORT_ERROR})"
        if not chem.rdkit_available():
            return "RDKit is required to resolve a structure to an InChIKey"
        return ""

    def _software(self) -> SoftwareEnvironment:
        import chemicals

        return SoftwareEnvironment.capture(chemicals=chemicals.__version__)

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        smiles = candidate.molecule.smiles if candidate.molecule else None
        if smiles is None:
            return ApplicabilityDomain.outside("candidate carries no molecule")
        if resolve_cas(smiles) is None:
            return ApplicabilityDomain.outside(
                "this structure could not be resolved to a compiled substance",
                basis="substances present in the experimental compilation",
            )
        return ApplicabilityDomain(
            basis="a measured value for this exact substance, not an estimate"
        )

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        smiles = request.candidate.molecule.smiles  # type: ignore[union-attr]
        value = measured_value(prop, smiles)
        if value is None:
            # Silence, not a guess. The estimating experts cover this compound,
            # and a fabricated "measurement" would outrank them on uncertainty.
            return Prediction.unsupported(
                prop,
                self.id,
                "this substance has no compiled measurement for this property",
            )

        _, unit, spread, basis = _LOOKUPS[prop]
        return self._make(
            prop,
            value,
            unit,
            request,
            domain,
            std=spread,
            kind=UncertaintyKind.ALEATORIC,
            basis=basis,
            notes=(
                "measured, not estimated",
                f"CAS {resolve_cas(smiles)}",
            ),
        )
