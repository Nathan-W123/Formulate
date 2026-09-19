"""Thermal and chemical properties of a liquid formulation.

A formulation already gets a density and a Hansen triple from
:mod:`formulate.experts.mixture`, and phase stability from
:mod:`formulate.experts.activity`.  Everything else a solvent blend is
specified on - where it boils, how much heat it takes to boil it, what its
critical constants are, how much of it a mole occupies - was uncovered.

The point of this module is that **the mixing rule is different for every one
of those properties**, and that a single "averaged by mole fraction" sentence
across all of them would be a lie for half.  What is applied, and what each is
worth:

* **Critical temperature and pressure: Kay's rule**, the mole-fraction average
  of the pure-component constants [Kay, Ind. Eng. Chem. 28 (1936) 1014].  Kay's
  pseudo-critical is *not* the mixture's true critical point - a mixture has a
  critical locus, not a critical point - and the rule ignores the binary
  interaction parameter entirely.  Its model-form error is therefore carried as
  direct evidence: the disagreement with an independent published rule, Li's
  for the temperature [Li, Can. J. Chem. Eng. 49 (1971) 709] and Prausnitz-Gunn
  for the pressure [Prausnitz & Gunn, AIChE J. 4 (1958) 430].  Measured here
  with panel constants: 1.4 K (0.2%) benzene/toluene, 10.0 K (1.9%)
  benzene/methanol, 12.7 K (2.3%) hexane/decane on the temperature; 0.7%,
  7.8% and 3.9% on the pressure.  That disagreement is a lower bound on Kay's
  error, not the error.
* **Critical volume, molar volume and molar refractivity: additive in mole
  fraction.**  Refractivity is genuinely additive through the Lorentz-Lorenz
  relation; molar volume is additive only up to the excess volume of mixing,
  which is ignored and carried as an uncertainty instead.
* **Ideal-gas heat capacity: additive in mole fraction, and exact.**  Not
  approximately: an ideal-gas mixture has no interactions by definition, so its
  heat capacity is the mole-fraction average with no model-form error at all.
  Only the component uncertainties are carried.
* **Normal boiling point: the 1 atm bubble point from modified Raoult**,
  ``sum(x_i gamma_i Psat_i) = P`` [Smith, Van Ness & Abbott, "Introduction to
  Chemical Engineering Thermodynamics", 7th ed., Ch. 10], with activity
  coefficients from modified UNIFAC (Dortmund) and vapour pressures from
  ``thermo``.  **Reported only when the mixture is proven narrow-boiling** -
  see below.
* **Enthalpy of vaporisation: mole-fraction additive at one temperature.**  The
  real error in adding component enthalpies is not the mixing rule, it is that
  each component's value is defined at *its own* boiling point; decane's is 25%
  larger at pentane's boiling point than at its own.  So each is
  Watson-corrected to the mixture's bubble point first, and the UNIFAC excess
  enthalpy is subtracted.
* **logP: refused.**  A formulation has no partition coefficient.
* **Synthetic accessibility: the maximum over components.**  You have to make
  every component, so the hardest one gates the formulation.

**The boiling point is the interesting refusal.**  A mixture does not have a
boiling point unless it is azeotropic; it boils over a range.  Reporting the
bubble point alone would let a blend that boils from 365 K to 425 K satisfy a
hard "boils between 360 and 370 K" requirement on the strength of its first
drop.  So the range is computed rather than guessed - the dew point at the same
overall composition and pressure - and a single number is reported only when
``dew - bubble <= 5 K``.  A cheap proxy on the spread of the pure-component
boiling points was built and rejected: ethanol/water 50:50 has a 22 K spread of
pure boiling points and a computed boiling range of 4.4 K, because the
azeotrope pulls bubble and dew together.  The proxy refuses a mixture that
really does boil at one temperature; the VLE solve costs 0.01 s and does not.

Passing the gate is not the same as the range having gone away, so **the range
that survives it is carried in the error bar**, as its width over sqrt(12) -
the standard deviation of a uniform spread between bubble and dew, which is
what the two endpoints alone support.  A true azeotrope keeps the tight bar it
has earned; a blend that squeaks under the gate at 4.9 K pays 1.4 K for it.
This was a real hole: with only the model error in the bar, the dew point this
module had already computed sat more than one sigma above the reported value
for 41% of the common solvent binaries the gate accepts and more than two sigma
for 9% of them.  Ethanol/water read 353.0 +/- 1.8 K while boiling on to
357.4 K; it now reads 353.0 +/- 2.2 K.

Assuming ideality instead of running UNIFAC was rejected on numbers, measured
here: Raoult puts the bubble point +15.3 K above modified UNIFAC for
hexane/ethanol, +7.0 K above for ethanol/water and -5.2 K below for
acetone/chloroform, all at or beyond the 5 K gate, and an ideal calculation
cannot produce an azeotrope at all.  Non-ideality *is* the boiling behaviour of a mixture, so a mixture
UNIFAC cannot cover is refused rather than given a Raoult number.

**Held-out accuracy of the bubble point.**  Five literature azeotrope and
narrow-blend temperatures, none of them fitted or tuned here, reproduced to
-0.23 K (acetone/methanol 0.80/0.20, 328.65 K), +1.26 K (ethyl
acetate/ethanol 0.69/0.31, 344.95 K), +0.10 K (benzene/ethanol 0.55/0.45,
341.05 K), -0.03 K (2-propanol/water 0.685/0.315, 353.55 K) and -0.06 K
(m-/p-xylene 50/50, 411.9 K): mean absolute error 0.34 K over the five.  Those
are blends of well-covered solvents with measured vapour-pressure
correlations.  Over the eighty-six binary blends of twenty common solvents
that the gate accepts at 50:50, the reported one-sigma bar ran 1.8 to 2.6 K
with a median of 2.2 K - 1.8 to 2.4 K of it the activity model's own accuracy
and the rest the surviving boiling range - because in practice a molecule the
DDBST compilation can assign UNIFAC groups to is a molecule the panel also has a
measured boiling point for, at half a kelvin.  Where that does not hold the bar
moves with it: replacing a component's measured boiling point with Joback's
12.9 K widens the acetone/methanol bubble point from 2.0 K to 14.0 K, which
falls out of the propagation rather than being asserted.

**Limitations that no error bar fixes.**  Kay's pseudo-criticals are reported
under the names ``critical_temperature`` and ``critical_pressure``, which do
not say "pseudo"; a mixture's true critical pressure can exceed both
components' and a mole-fraction average never can.  ``heat_capacity_gas`` is
the ideal-gas heat capacity at the *liquid's* composition, not that of the
vapour in equilibrium with it.  ``enthalpy_vaporization`` is reported at the
bubble point while the registry defines it at the normal boiling point; the
prediction states its reference temperature and nothing enforces that a reader
honours it.  No liquid-liquid stability test exists for three or more
components here, so a ternary that splits into two liquid phases would still
receive a bubble point computed from one.

Widening the bar by the boiling range does not centre it.  The value reported
is the bubble point, which sits at the *bottom* of the range, so the dew point
still lands near two sigma above it even now - 2.0 for the widest case the gate
admits, ethanol/water.  A symmetric standard deviation is the only shape the
interface has; a bubble point with an asymmetric interval running up to the dew
point is what the quantity really is, and the note says so in words because the
number cannot.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

from formulate.core.candidate import (
    Candidate,
    FractionBasis,
    MaterialClass,
    MixtureSpec,
    molecule_candidate,
)
from formulate.core.conditions import Conditions
from formulate.core.hashing import content_hash
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, Quantity, UncertaintyKind

from .base import Expert, PredictionRequest
from .interfacial import watson_enthalpy_vaporization

#: Gas constant, J/(mol K).
R_GAS = 8.31446261815324

#: The pressure a *normal* boiling point is defined at. Fixed rather than read
#: from the request: the registry defines ``normal_boiling_point`` as "boiling
#: point at 1 atm", so solving at whatever pressure a specification happened to
#: state would silently return a different property under the same name.
ONE_ATM_PA = 101325.0

#: Largest dew-minus-bubble spread, in kelvin, for which a single number may be
#: reported under the name ``normal_boiling_point``.
#:
#: Five kelvin because that is where this method stops being able to resolve a
#: range at all: the tightest bar it produces on a bubble point is 2-3 K for a
#: blend of well-covered solvents (measured below), so a computed range narrower
#: than about twice that is not distinguishable from a single temperature. It is
#: argued from the method's own resolution, not fitted to anything, and it is a
#: judgement call at the boundary - a 4.9 K range is reported and a 5.1 K range
#: refused, and nothing physical separates them.
MAX_BOILING_RANGE_K = 5.0

#: Multiplier turning the computed dew-minus-bubble range into its contribution
#: to the one-sigma bar on the reported boiling point.
#:
#: The value reported is the bubble point, but what a consumer reads under the
#: name ``normal_boiling_point`` is "the temperature at which this boils", and
#: for a mixture that is an interval and not a point.  Passing the gate does not
#: make the interval go away; it only makes it small enough to summarise.  Given
#: nothing but the two endpoints the maximum-entropy distribution over the
#: interval is the uniform one, whose standard deviation is its width over
#: sqrt(12).  So a true azeotrope, whose range really is zero, keeps the tight
#: bar it has earned, and a blend that squeaks under the gate at 4.9 K carries
#: the 1.4 K the range is worth.
#:
#: Without this the bar was the model error on the bubble point alone, and the
#: dew point this module had already computed sat more than one sigma above the
#: reported value for 41% of the common solvent binaries the gate accepts and
#: more than two sigma for 9% of them - ethanol/water reported 353.0 +/- 1.8 K
#: while boiling on to 357.4 K.  An error bar is ranked on here, so omitting a
#: spread the module has in hand understates exactly where it matters.
BOILING_RANGE_SIGMA_FRACTION = 1.0 / math.sqrt(12.0)

#: Fractional disagreement between Kay's rule and the independent rule it is
#: checked against, beyond which the critical constant is marked out of domain
#: rather than merely wide.
#:
#: Five per cent because that is roughly twice the worst disagreement seen for
#: any pair whose components are chemically alike (2.3%, hexane/decane), so
#: exceeding it means the two rules are not describing the same fluid and the
#: pseudo-critical has stopped being a useful parameter rather than just an
#: imprecise one.
KAY_DISAGREEMENT_LIMIT = 0.05

#: Model-form uncertainty on a mole-fraction-additive critical volume, as a
#: fraction of the value.
#:
#: The weakest number in this module, and flagged as such in the basis string.
#: The critical temperature and pressure each have a second published
#: parameter-free rule to disagree with, which turns their model-form error into
#: a measurement. For the critical volume there is no such rule here - Chueh-
#: Prausnitz needs a fitted binary interaction parameter - so 5% is a stated
#: assumption rather than evidence, chosen to sit alongside the 0.2-2.3%
#: measured on the temperature and 0.7-11.3% on the pressure.
CRITICAL_VOLUME_RULE_FRACTION = 0.05

#: Excess volume of mixing, as a fraction of the ideal molar volume, for a blend
#: in which no component has a hydrogen-bond donor.
#:
#: Anchored on benzene + cyclohexane, whose excess volume peaks near
#: +0.65 cm^3/mol on an ideal 99 cm^3/mol - 0.7%. Two liquids that interact
#: only by dispersion pack close to additively.
EXCESS_VOLUME_FRACTION_INERT = 0.01

#: The same, for a blend in which any component hydrogen bonds.
#:
#: Anchored on ethanol + water, whose excess volume reaches about
#: -1.0 cm^3/mol on an ideal 38 cm^3/mol - 2.6%. The trigger is *any* donor
#: rather than a mismatch between donors and non-donors, because ethanol and
#: water are both donors and are the larger of the two anchors: breaking a
#: liquid's own hydrogen-bond network contracts the mixture whether or not its
#: partner can donate.
EXCESS_VOLUME_FRACTION_ASSOCIATING = 0.03

#: Model-form uncertainty on an additive molar refractivity, as a fraction.
#:
#: Lorentz-Lorenz additivity is good to well under one per cent, and saying so
#: is the point: the component values come from Crippen's atomic contributions
#: at about 2.5 cm^3/mol each, which is roughly ten per cent and dominates this
#: by a factor of twenty. The tight rule does not make the answer tight.
LORENTZ_LORENZ_RULE_FRACTION = 0.005

#: Fraction of the UNIFAC excess enthalpy carried as uncertainty on it.
#:
#: Modified UNIFAC was partly fitted to excess enthalpies and reproduces them
#: far worse than it reproduces activity coefficients; half the magnitude of its
#: own correction is the conservative way to say that the correction may be
#: wrong by as much as itself.
EXCESS_ENTHALPY_RELATIVE_ERROR = 0.5

#: Multiplier applied to every activity coefficient when propagating the
#: bubble-point uncertainty.
#:
#: Seven per cent is the accuracy modified UNIFAC (Dortmund) is usually credited
#: with on an activity coefficient for a well-covered binary. Measured through
#: the bubble-point solve it moves the answer by 1.7 K (ethanol/water) to 2.3 K
#: (benzene/toluene), and because it applies to every blend alike it is what
#: sets the floor on this method's error bar - a measured 1.8 to 2.4 K over
#: forty-seven common solvent pairs.
GAMMA_PERTURBATION = 1.07

#: Bracket for the bubble- and dew-point root, in kelvin. The lower end sits
#: below any ordinary solvent's freezing point; the upper end is narrowed per
#: mixture to just under the lowest component critical temperature, because
#: above that a vapour-pressure correlation is extrapolating into a fluid with
#: no vapour-liquid boundary.
_VLE_T_MIN = 120.0
_VLE_T_MAX = 900.0

#: Component properties pulled from the molecular panel. Every mixing rule here
#: consumes one of them; ``molar_mass`` and ``liquid_density`` additionally
#: convert the recipe's declared basis to the mole basis all the rules need.
_COMPONENT_PROPERTIES = frozenset(
    {
        "molar_mass",
        "liquid_density",
        "normal_boiling_point",
        "critical_temperature",
        "critical_pressure",
        "critical_volume",
        "enthalpy_vaporization",
        "heat_capacity_gas",
        "molar_refractivity",
        "logp",
        "synthetic_accessibility",
    }
)

#: Unit each component property is read in, chosen so the mixing rule can be
#: written in SI without a conversion in the middle of the arithmetic.
_COMPONENT_UNITS = {
    "molar_mass": "g/mol",
    "liquid_density": "kg/m^3",
    "normal_boiling_point": "K",
    "critical_temperature": "K",
    "critical_pressure": "Pa",
    "critical_volume": "m^3/mol",
    "enthalpy_vaporization": "J/mol",
    "heat_capacity_gas": "J/mol/K",
    "molar_refractivity": "cm^3/mol",
    "logp": "",
    "synthetic_accessibility": "",
}

#: Properties that are a plain mole-fraction sum of the same component
#: property, mapped to the unit the sum is reported in.
_ADDITIVE = {
    "critical_temperature": "K",
    "critical_pressure": "Pa",
    "critical_volume": "m^3/mol",
    "molar_refractivity": "cm^3/mol",
    "heat_capacity_gas": "J/mol/K",
}


# --------------------------------------------------------------------------
# Vapour pressure and the VLE solve
# --------------------------------------------------------------------------


def unifac_available() -> bool:
    """True when the modified-UNIFAC tables are importable."""
    from .activity import unifac_available as _available

    return _available()


@functools.lru_cache(maxsize=2048)
def vapour_pressure(
    smiles: str, tb: float | None, tc: float | None, pc: float | None
):
    """A ``thermo.VaporPressure`` for one component, or ``None``.

    Two routes, in order.  A measured correlation keyed by CAS is preferred,
    resolved through the same InChIKey path the Hansen expert uses, because a
    bubble point is only as good as the vapour pressures under it.  Failing
    that, Ambrose-Walton over the panel's own boiling point and critical
    constants, with a Lee-Kesler acentric factor.  The estimated route is
    anchored at the boiling point, so it is far better near 1 atm than at
    ambient: fed the panel's constants it reproduces the panel's own boiling
    point to 0.07 K (ethanol), 0.15 K (hexane), 0.29 K (acetone) and 0.44 K
    (benzene), while its 298 K vapour pressure is out by 0.6% to 7.3%.  The
    catch is upstream rather than here - a Joback boiling point carrying 12.9 K
    of its own maps to roughly a 50% vapour-pressure error, and the propagated
    bar says so.
    """
    from thermo import VaporPressure

    from .hansen import resolve_cas

    cas = resolve_cas(smiles)
    if cas is not None:
        try:
            measured = VaporPressure(CASRN=cas)
            probe = measured(tb if tb is not None else 298.15)
            if probe is not None and probe > 0.0 and math.isfinite(probe):
                return measured
        except Exception:
            pass

    if tb is None or tc is None or pc is None or not (0.0 < tb < tc):
        return None
    try:
        from chemicals.acentric import LK_omega

        omega = LK_omega(tb, tc, pc)
        estimated = VaporPressure(Tb=tb, Tc=tc, Pc=pc, omega=omega)
        probe = estimated(tb)
        if probe is None or not math.isfinite(probe) or probe <= 0.0:
            return None
        return estimated
    except Exception:
        return None


def unifac_model(groups, fractions, temperature: float):
    """The modified-UNIFAC (Dortmund) model for a composition at a temperature.

    A separate constructor from :func:`formulate.experts.activity._gammas`
    because the excess enthalpy is needed here and that helper returns only the
    activity coefficients.  Two constructors of the same model is exactly how
    two parts of a system come to disagree about the same mixture, so a test
    pins this one's coefficients against that one's.
    """
    from thermo.unifac import DOUFIP2016, DOUFSG, UNIFAC

    return UNIFAC.from_subgroups(
        T=temperature,
        xs=list(fractions),
        chemgroups=[dict(g) for g in groups],
        subgroups=DOUFSG,
        interaction_data=DOUFIP2016,
        version=1,
    )


def _bracket(fn, lo: float, hi: float, steps: int = 48):
    """A sign change of ``fn`` inside ``[lo, hi]``, or ``None``.

    ``brentq`` needs a bracket and refuses to guess one.  Scanning for it is
    what turns "no solution in the physically sensible range" into a refusal
    rather than an exception.
    """
    previous_t = lo
    try:
        previous_f = fn(lo)
    except Exception:
        previous_f = None
    for index in range(1, steps + 1):
        t = lo + (hi - lo) * index / steps
        try:
            f = fn(t)
        except Exception:
            f = None
        if previous_f is not None and f is not None and previous_f * f <= 0.0:
            return previous_t, t
        previous_t, previous_f = t, f
    return None


def _solve(fn, lo: float, hi: float) -> float | None:
    from scipy.optimize import brentq

    bracket = _bracket(fn, lo, hi)
    if bracket is None:
        return None
    try:
        return float(brentq(fn, bracket[0], bracket[1], xtol=1e-6))
    except Exception:
        return None


def bubble_temperature(
    groups,
    pressures,
    mole_fractions,
    pressure: float,
    ceiling: float,
    *,
    gamma_scale: float = 1.0,
    psat_scale: tuple[float, ...] | None = None,
) -> float | None:
    """Temperature at which a liquid of this composition first boils.

    Modified Raoult: ``sum(x_i gamma_i Psat_i(T)) = P``.  ``gamma_scale`` and
    ``psat_scale`` exist for the uncertainty propagation, which re-solves the
    same root at perturbed inputs rather than linearising it.
    """
    scale = psat_scale or (1.0,) * len(mole_fractions)

    def residual(t: float) -> float:
        gammas = unifac_model(groups, mole_fractions, t).gammas()
        total = 0.0
        for x, gamma, vp, s in zip(mole_fractions, gammas, pressures, scale):
            total += x * gamma * gamma_scale * vp(t) * s
        return total - pressure

    return _solve(residual, _VLE_T_MIN, ceiling)


def dew_temperature(
    groups, pressures, vapour_fractions, pressure: float, ceiling: float
) -> float | None:
    """Temperature at which a vapour of this composition first condenses.

    Called with the *overall* composition, which makes it the temperature at
    which the last drop of that liquid boils away.  Bubble to dew is therefore
    the range the formulation actually boils over, and it is the number the
    boiling-point refusal is decided on.  The inner loop is the standard
    successive substitution on the liquid composition, which UNIFAC needs
    because its activity coefficients depend on the liquid it is evaluating.
    """

    def residual(t: float) -> float:
        liquid = list(vapour_fractions)
        for _ in range(80):
            gammas = unifac_model(groups, liquid, t).gammas()
            updated = [
                y * pressure / (gamma * vp(t))
                for y, gamma, vp in zip(vapour_fractions, gammas, pressures)
            ]
            total = sum(updated)
            if total <= 0.0:
                return float("nan")
            updated = [v / total for v in updated]
            if max(abs(a - b) for a, b in zip(updated, liquid)) < 1e-10:
                liquid = updated
                break
            liquid = updated
        gammas = unifac_model(groups, liquid, t).gammas()
        return (
            sum(
                y * pressure / (gamma * vp(t))
                for y, gamma, vp in zip(vapour_fractions, gammas, pressures)
            )
            - 1.0
        )

    return _solve(residual, _VLE_T_MIN, ceiling)


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Component:
    """One component of a formulation, evaluated through the molecular panel.

    The panel's :class:`~formulate.core.prediction.Prediction` objects are kept
    rather than reduced to floats, because every rule in this module propagates
    the component's own error bar and a float has thrown it away.
    """

    smiles: str
    role: str
    fraction: float
    predictions: dict[str, Prediction]

    def value(self, prop: str) -> float | None:
        prediction = self.predictions.get(prop)
        if prediction is None or prediction.quantity is None:
            return None
        return prediction.quantity.to(_COMPONENT_UNITS[prop]).value

    def std(self, prop: str) -> float | None:
        prediction = self.predictions.get(prop)
        if prediction is None or prediction.quantity is None:
            return None
        return prediction.uncertainty.converted(
            prediction.quantity.unit, _COMPONENT_UNITS[prop]
        ).std

    @property
    def molar_volume(self) -> float | None:
        """``M / rho`` in m^3/mol.

        Deliberately not the panel's own ``molar_volume_liquid``.  The Rackett
        molar volume for ethanol carries 1.1e-5 on 5.18e-5 - twenty-one per
        cent - while the measured density it could have been built from carries
        1 kg/m^3, which is 0.13%.  Taking it from the density is never worse,
        and it is the same arithmetic ``MixtureExpert`` already does, so this
        expert's molar volume and that expert's density describe one liquid
        rather than two.
        """
        molar_mass = self.value("molar_mass")
        density = self.value("liquid_density")
        if not molar_mass or not density:
            return None
        return (molar_mass * 1e-3) / density


def _has_hydrogen_bond_donor(smiles: str) -> bool:
    from formulate import chem

    return int(chem.descriptors(smiles).get("hbd", 0.0)) > 0


# --------------------------------------------------------------------------
# The expert
# --------------------------------------------------------------------------


class MixtureThermalExpert(Expert):
    """Thermal and chemical properties of a liquid formulation.

    Sits beside :class:`~formulate.experts.mixture.MixtureExpert` rather than
    inside it because the two answer different kinds of question with different
    machinery: that one applies one volume-additive rule to three interfacial
    properties, this one applies a different rule to each of ten thermal and
    chemical ones and solves a vapour-liquid equilibrium for two of them.
    """

    id = "mixture_thermal"
    version = "1"
    method = (
        "per-property mixing rules over the molecular panel: Kay's pseudo-criticals "
        "(Ind. Eng. Chem. 28:1014, 1936), mole-fraction additivity for volume, "
        "refractivity and ideal-gas heat capacity, and a modified-Raoult bubble point "
        "with modified UNIFAC (Dortmund) activity coefficients"
    )
    family = PropertyFamily.THERMAL
    supported_classes = frozenset({MaterialClass.MIXTURE})
    supported_properties = frozenset(
        {
            "normal_boiling_point",
            "critical_temperature",
            "critical_pressure",
            "critical_volume",
            "enthalpy_vaporization",
            "heat_capacity_gas",
            "molar_refractivity",
            "logp",
            "molar_volume_liquid",
            "synthetic_accessibility",
        }
    )
    #: Empty on purpose.  What this expert needs belongs to the *components*,
    #: not to the mixture candidate, and no expert in the panel produces a
    #: mixture-class critical constant, so a declared dependency could never be
    #: resolved and would only make every prediction refuse.  ``mixture.py`` and
    #: ``blend.py`` are empty for the same reason.
    dependencies: frozenset[str] = frozenset()

    #: Cap on the per-instance component cache. A run evaluates many blends of
    #: the same few solvents, so caching the panel is the difference between
    #: one panel call per component and one per component per property.
    _CACHE_LIMIT = 512

    def __init__(self, registry=None) -> None:
        self._registry = registry
        self._cache: dict[tuple, dict[str, Prediction]] = {}

    def registry(self):
        """The molecular panel used to evaluate components.

        Built lazily for the same reason ``MixtureExpert`` does it: the default
        registry imports every expert, including this one.
        """
        if self._registry is None:
            from formulate.experts import molecular_registry

            self._registry = molecular_registry()
        return self._registry

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "RDKit is required to read component structures"

    def _software(self) -> SoftwareEnvironment:
        import thermo

        from formulate import chem

        return SoftwareEnvironment.capture(rdkit=chem.rdkit_version(), thermo=thermo.__version__)

    # -- domain ------------------------------------------------------------

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        basis = (
            "homogeneous liquid blends of neutral small molecules; the mixing rules are "
            "small-molecule liquid-state rules and the vapour-liquid solve is modified "
            "UNIFAC over the DDBST group assignment"
        )
        mixture = candidate.mixture
        if mixture is None:
            return ApplicabilityDomain.outside("candidate carries no mixture", basis=basis)
        if any(component.polymer is not None for component in mixture.components):
            return ApplicabilityDomain.outside(
                "a polymer component is present. A polymer has no critical point and no "
                "boiling point - it decomposes first - so none of these rules has a "
                "quantity to mix for it",
                basis=basis,
            )
        charged = [
            component.molecule.smiles
            for component in mixture.components
            if component.molecule is not None and component.molecule.charge != 0
        ]
        if charged:
            return ApplicabilityDomain.outside(
                f"a component carries a formal charge ({', '.join(charged)}); neither "
                "these corresponding-states rules nor any vapour-pressure correlation "
                "here applies to an ionic species",
                basis=basis,
            )
        if mixture.phase_assumption.value in ("emulsion", "dispersion", "suspension"):
            return ApplicabilityDomain(
                score=0.3,
                in_domain=False,
                warnings=(
                    f"the formulation is declared a {mixture.phase_assumption.value}, which "
                    "is by definition not a single homogeneous phase; these rules describe "
                    "the hypothetical single phase and not the dispersed system",
                ),
                basis=basis,
            )

        warnings: list[str] = []
        score = 1.0
        if len(mixture.components) > 2:
            warnings.append(
                "three or more components: the liquid-liquid stability test used to gate "
                "the boiling point is a binary scan, so a ternary that splits into two "
                "liquid phases would still be given a bubble point solved from one"
            )
            score = min(score, 0.6)
        return ApplicabilityDomain(
            score=score, in_domain=True, warnings=tuple(warnings), basis=basis
        )

    # -- component evaluation ----------------------------------------------

    def component_panel(self, smiles: str, conditions: Conditions) -> dict[str, Prediction]:
        """Every usable molecular prediction for one component structure.

        The walk is the one ``MixtureExpert`` does - grow the dependency
        closure, order the experts, feed each one's output to the next - so a
        component's boiling point here is the same number, from the same method
        with the same uncertainty, it would have had as a candidate in its own
        right.
        """
        # Keyed on every condition, not just temperature and pressure: a panel
        # expert may read the phase or the named solutes, and a cache that
        # ignored them would hand back an answer to a different question.
        key = (smiles, content_hash(conditions.identity_payload()))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        registry = self.registry()
        wanted = set(_COMPONENT_PROPERTIES)
        # Ask the panel for what it needs to answer, not only for what is
        # wanted: a liquid density depends on critical constants from another
        # expert, and requesting the density alone selects only the expert that
        # serves it, whose dependency then arrives empty.
        for _ in range(len(registry) + 1):
            grown = set(wanted)
            for expert in registry:
                if expert.supported_properties & wanted:
                    grown |= expert.dependencies
            if grown == wanted:
                break
            wanted = grown
        frozen = frozenset(wanted)

        sub = molecule_candidate(smiles, conditions=conditions)
        context: dict[str, Prediction] = {}
        for expert in registry.resolution_order(
            registry.experts_for(frozen, MaterialClass.MOLECULE)
        ):
            sub_request = PredictionRequest(
                candidate=sub, properties=frozen, conditions=conditions, context=dict(context)
            )
            for prediction in expert.predict(sub_request):
                if prediction.is_usable:
                    context.setdefault(prediction.property, prediction)

        if len(self._cache) >= self._CACHE_LIMIT:
            self._cache.clear()
        self._cache[key] = context
        return context

    def components(self, mixture: MixtureSpec, request: PredictionRequest) -> list[Component]:
        out: list[Component] = []
        for component in mixture.components:
            smiles = component.molecule.smiles  # type: ignore[union-attr]
            out.append(
                Component(
                    smiles=smiles,
                    role=component.role.value,
                    fraction=component.fraction,
                    predictions=self.component_panel(smiles, request.conditions),
                )
            )
        return out

    @staticmethod
    def mole_fractions(
        components: list[Component], basis: FractionBasis
    ) -> tuple[list[float] | None, str]:
        """Convert the declared fractions to a mole basis.

        Every rule in this module is mole-fraction weighted, and a mass
        fraction used as a mole fraction is a silent error of tens of per cent
        the moment the components differ in molar mass.  A recipe that cannot be
        converted therefore refuses everything rather than being mixed on the
        basis it happens to be written in.
        """
        if basis is FractionBasis.MOLE:
            return [c.fraction for c in components], ""

        moles: list[float] = []
        for component in components:
            molar_mass = component.value("molar_mass")
            if not molar_mass:
                return None, (
                    f"the recipe is stated on a {basis.value} basis and the panel has no "
                    f"molar mass for {component.smiles!r}, so it cannot be converted to "
                    "the mole basis every rule here is weighted on"
                )
            if basis is FractionBasis.MASS:
                moles.append(component.fraction / molar_mass)
            else:
                density = component.value("liquid_density")
                if not density:
                    return None, (
                        "the recipe is stated by volume and the panel has no liquid "
                        f"density for {component.smiles!r}, so it cannot be converted to "
                        "a mole basis"
                    )
                moles.append(component.fraction * density / molar_mass)

        total = sum(moles)
        if total <= 0:
            return None, "the composition came out non-positive"
        return [m / total for m in moles], ""

    # -- prediction --------------------------------------------------------

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        mixture = request.candidate.mixture
        if mixture is None:
            return Prediction.unsupported(prop, self.id, "candidate carries no mixture")

        # Rule 7 of this repository: no silent fallback between material
        # classes. A polymer component is refused rather than given an
        # out-of-domain number, because there is no quantity to mix - a polymer
        # has no critical point and no boiling point.
        if any(component.polymer is not None for component in mixture.components):
            return Prediction.unsupported(
                prop,
                self.id,
                "a component is a polymer. A polymer has no critical point and no boiling "
                "point - it decomposes - and these are small-molecule liquid rules, so "
                "there is nothing to mix for it",
            )
        charged = [
            component.molecule.smiles
            for component in mixture.components
            if component.molecule is not None and component.molecule.charge != 0
        ]
        if charged:
            return Prediction.unsupported(
                prop,
                self.id,
                f"component {charged[0]!r} carries a formal charge; none of these rules, "
                "and no vapour-pressure correlation here, applies to an ionic species",
            )

        components = self.components(mixture, request)
        fractions, reason = self.mole_fractions(components, mixture.basis)
        if fractions is None:
            return Prediction.unsupported(prop, self.id, reason)

        if prop == "logp":
            return self._logp(prop, components, fractions, request, domain)
        if prop == "synthetic_accessibility":
            return self._accessibility(prop, components, request, domain)
        if prop == "molar_volume_liquid":
            return self._molar_volume(prop, components, fractions, request, domain)
        if prop in ("normal_boiling_point", "enthalpy_vaporization"):
            return self._vle_property(prop, components, fractions, request, domain)
        return self._additive(prop, components, fractions, request, domain)

    # -- rule: mole-fraction additive --------------------------------------

    def _additive(
        self, prop, components, fractions, request, domain
    ) -> Prediction | None:
        """Kay's rule and the plain additive constants.

        One code path for five properties, but *five different justifications*,
        which is why each leaves with its own note and its own model-form term
        rather than a shared sentence about mole fractions.
        """
        missing = [c.smiles for c in components if c.value(prop) is None]
        if missing:
            return Prediction.unsupported(
                prop,
                self.id,
                f"the panel has no {prop} for {missing[0]!r}"
                + (f" and {len(missing) - 1} other component(s)" if len(missing) > 1 else "")
                + ". A mixing rule evaluated over the components it happens to have is an "
                "average of a different formulation",
            )

        values = [c.value(prop) for c in components]
        value = sum(x * v for x, v in zip(fractions, values))

        # Component errors are summed rather than added in quadrature. Every
        # component's constant comes from the same group-contribution table, so
        # their errors share a sign more often than not; this is the fully
        # correlated bound and the deliberately conservative choice. It costs
        # this expert ranking ties against anything that assumed independence,
        # and it is the direction the calibration report can detect.
        component_term = 0.0
        for x, c in zip(fractions, components):
            spread = c.std(prop)
            if spread is None:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"the panel's {prop} for {c.smiles!r} carries no uncertainty, so the "
                    "mixture's cannot be stated and the value would be ranked as if it "
                    "were exact",
                )
            component_term += x * spread

        rule_term, rule_basis, note, narrowed = self._model_form(
            prop, components, fractions, value, domain
        )
        if rule_term is None:
            return Prediction.unsupported(prop, self.id, rule_basis)

        std = math.hypot(component_term, rule_term)
        return self._make(
            prop,
            value,
            _ADDITIVE[prop],
            request,
            domain if narrowed is None else narrowed,
            std=std,
            kind=UncertaintyKind.COMBINED,
            basis=(
                f"{component_term:.4g} from the components, summed linearly rather than in "
                "quadrature because they come from one group-contribution table and their "
                f"errors are treated as fully correlated; combined in quadrature with "
                f"{rule_term:.4g} of model-form error - {rule_basis}"
            ),
            notes=(note,),
        )

    def _model_form(self, prop, components, fractions, value, domain):
        """Model-form uncertainty, its justification, and the property's note.

        Returns ``(term, basis, note, narrowed_domain)``.  A ``None`` term is a
        refusal and ``basis`` carries the reason.
        """
        single = len(components) == 1

        if prop == "heat_capacity_gas":
            # Not "nearly" exact. An ideal gas has no intermolecular
            # interactions by definition, so an ideal-gas mixture's heat
            # capacity IS the mole-fraction average and there is no rule error
            # to carry at all.
            return (
                0.0,
                "none: mole-fraction additivity is exact for an ideal-gas mixture, by the "
                "definition of one. The whole bar is the components' own",
                "mole-fraction additive, and exact: an ideal-gas mixture has no "
                "interactions, so this rule introduces no error of its own. It is the "
                "ideal-gas heat capacity at the LIQUID's composition, not that of the "
                "vapour in equilibrium with it, which is enriched in the volatile "
                "component",
                None,
            )

        if prop == "molar_refractivity":
            term = 0.0 if single else LORENTZ_LORENZ_RULE_FRACTION * abs(value)
            return (
                term,
                f"{LORENTZ_LORENZ_RULE_FRACTION:.1%} of the value for the Lorentz-Lorenz "
                "relation, which is genuinely additive in mole fraction rather than "
                "approximately so. It is twenty times smaller than the Crippen atomic "
                "contributions it is combined with, and does not make the answer tight",
                "additive in mole fraction by the Lorentz-Lorenz relation, which makes "
                "refractivity one of the few genuinely additive properties here",
                None,
            )

        if prop == "critical_volume":
            term = 0.0 if single else CRITICAL_VOLUME_RULE_FRACTION * abs(value)
            return (
                term,
                f"{CRITICAL_VOLUME_RULE_FRACTION:.0%} of the value, which is the weakest "
                "number in this module: unlike Kay's temperature and pressure there is no "
                "second parameter-free rule here to disagree with, so this is a stated "
                "assumption rather than a measurement",
                "Kay-type pseudo-critical volume, additive in mole fraction. This is not "
                "the mixture's critical volume - a mixture has a critical locus, not a "
                "critical point",
                None,
            )

        # A one-component formulation is a pure substance, and Kay's rule over
        # one component returns that component's own constant unchanged. There
        # is no pseudo-critical, no binary interaction being ignored and no
        # model form to have an error - so demanding the independent check
        # would refuse a number no rule was applied to. Li's rule and
        # Prausnitz-Gunn both collapse to the same constant here, which is why
        # requiring a critical volume to evaluate them was refusing a pure
        # substance's critical temperature over an input neither rule needs.
        if single:
            return (
                0.0,
                "none: there is one component, so Kay's rule returns its constant "
                "unchanged and no mixing rule was applied. The whole bar is the panel's "
                "own for that pure substance",
                "one component, so this is the pure substance's own critical constant "
                "rather than a pseudo-critical: Kay's rule over a single component is "
                "the identity",
                None,
            )

        # Kay's temperature and pressure, each checked against an independent
        # published rule. The disagreement between two rules is evidence about
        # the model form in a way that an asserted percentage is not.
        alternative = self._kay_alternative(prop, components, fractions)
        if alternative is None:
            missing = [c.smiles for c in components if not c.value("critical_volume")]
            return (
                None,
                f"Kay's rule would give a pseudo-{prop.replace('_', ' ')} for this blend, "
                f"but the panel has no critical volume for {missing[0]!r}, so the "
                "independent rule it is checked against cannot be evaluated. Kay's error "
                "would then have no defensible basis - it is 0.2% for chemically alike "
                "components and tens of per cent for unlike ones, and without the check "
                "there is no way to tell which this is",
                "",
                None,
            )

        gap = abs(value - alternative)
        narrowed = None
        if value and gap / abs(value) > KAY_DISAGREEMENT_LIMIT:
            narrowed = ApplicabilityDomain(
                score=min(domain.score, 0.3),
                in_domain=False,
                warnings=domain.warnings
                + (
                    f"Kay's rule and its independent check disagree by "
                    f"{gap / abs(value):.1%}, above the {KAY_DISAGREEMENT_LIMIT:.0%} at "
                    "which a mole-fraction pseudo-critical stops describing this mixture",
                ),
                basis=domain.basis,
            )

        if prop == "critical_temperature":
            return (
                gap,
                f"the {gap:.4g} K by which Kay's rule and Li's rule (Can. J. Chem. Eng. "
                "49:709, 1971) disagree here. Two published parameter-free rules "
                "disagreeing is direct evidence about the model form, and it is a LOWER "
                "bound on Kay's error rather than the error: measured over reference "
                "pairs it runs 0.2% for benzene/toluene and 2.3% for hexane/decane, while "
                "Kay is known to be tens of per cent low for components of very different "
                "size",
                "Kay's rule: the mole-fraction average of the pure-component critical "
                "temperatures. It ignores the binary interaction parameter entirely, and "
                "it is a pseudo-critical constant rather than the mixture's critical "
                "point, which is a locus and not a point",
                narrowed,
            )
        return (
            gap,
            f"the {gap / abs(value):.1%} by which Kay's rule and the Prausnitz-Gunn rule "
            "(AIChE J. 4:430, 1958) disagree here. Measured over reference pairs it runs "
            "0.7% for benzene/toluene, 7.8% for benzene/methanol and 11.3% for "
            "hexane/ethanol, and it is a lower bound: a real mixture's critical pressure "
            "can exceed BOTH components', which a mole-fraction average can never do",
            "Kay's rule: the mole-fraction average of the pure-component critical "
            "pressures. A bounded average, so it structurally cannot reproduce the "
            "maximum a real binary critical pressure passes through",
            narrowed,
        )

    @staticmethod
    def _kay_alternative(prop, components, fractions) -> float | None:
        """Li's mixture critical temperature, or the Prausnitz-Gunn pressure."""
        temperatures = [c.value("critical_temperature") for c in components]
        volumes = [c.value("critical_volume") for c in components]
        if any(v is None or not v for v in volumes) or any(t is None for t in temperatures):
            return None

        if prop == "critical_temperature":
            from chemicals.critical import Li

            try:
                return float(Li(list(fractions), list(temperatures), list(volumes)))
            except Exception:
                return None

        pressures = [c.value("critical_pressure") for c in components]
        if any(p is None for p in pressures):
            return None
        # Prausnitz-Gunn: Pc_m = R Tc_m sum(z_i Zc_i) / sum(z_i Vc_i), with the
        # pseudo-critical temperature from Kay and each Zc from the component's
        # own constants, so it shares no arithmetic with the Kay pressure it is
        # being compared against.
        kay_temperature = sum(z * t for z, t in zip(fractions, temperatures))
        compressibilities = [
            p * v / (R_GAS * t) for p, v, t in zip(pressures, volumes, temperatures)
        ]
        mixture_volume = sum(z * v for z, v in zip(fractions, volumes))
        if mixture_volume <= 0:
            return None
        return (
            R_GAS
            * kay_temperature
            * sum(z * zc for z, zc in zip(fractions, compressibilities))
            / mixture_volume
        )

    # -- rule: volume-additive ---------------------------------------------

    def _molar_volume(self, prop, components, fractions, request, domain) -> Prediction | None:
        volumes = [c.molar_volume for c in components]
        missing = [c.smiles for c, v in zip(components, volumes) if v is None]
        if missing:
            return Prediction.unsupported(
                prop,
                self.id,
                f"a molar volume needs a molar mass and a liquid density, and the panel "
                f"has no usable pair for {missing[0]!r}. A density is itself temperature "
                "dependent, so an unstated temperature reaches here as a missing component",
            )

        value = sum(x * v for x, v in zip(fractions, volumes))

        component_term = 0.0
        for x, c, v in zip(fractions, components, volumes):
            density = c.value("liquid_density")
            density_std = c.std("liquid_density")
            mass_std = c.std("molar_mass")
            molar_mass = c.value("molar_mass")
            # An absent uncertainty is not a zero one. Reading it as zero would
            # quietly hand this blend the tightest bar in the run - the density
            # is the whole of the component term - and the ranker ranks on that
            # bar, so it would win a tie precisely because nobody measured it.
            if density_std is None or mass_std is None:
                which = "liquid density" if density_std is None else "molar mass"
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"the panel's {which} for {c.smiles!r} carries no uncertainty, so the "
                    "mixture's molar volume cannot state one either and the value would "
                    "be ranked as if it were exact",
                )
            # V = M / rho, so the relative errors add in quadrature within one
            # component (two unrelated measurements), and the components then
            # add linearly for the correlation reason above.
            relative = math.hypot(
                density_std / density, (mass_std / molar_mass) if molar_mass else 0.0
            )
            component_term += x * v * relative

        associating = any(_has_hydrogen_bond_donor(c.smiles) for c in components)
        excess_fraction = (
            EXCESS_VOLUME_FRACTION_ASSOCIATING if associating else EXCESS_VOLUME_FRACTION_INERT
        )
        excess_term = 0.0 if len(components) == 1 else excess_fraction * value
        std = math.hypot(component_term, excess_term)

        return self._make(
            prop,
            value,
            "m^3/mol",
            request,
            domain,
            std=std,
            kind=UncertaintyKind.COMBINED,
            basis=(
                f"{component_term:.4g} from the components' own molar masses and densities, "
                f"summed linearly as fully correlated; combined in quadrature with "
                f"{excess_term:.4g}, being {excess_fraction:.0%} for the excess volume of "
                "mixing this rule ignores - anchored on benzene + cyclohexane at 0.7% and "
                "ethanol + water at 2.6%. The sign of that excess is not predicted, only "
                "its size"
            ),
            notes=(
                "volume-additive in mole fraction, with each component's volume taken as "
                "molar mass over liquid density rather than from a Rackett correlation, so "
                "this molar volume and the mixture expert's blend density are the same "
                "physics and cannot disagree",
            ),
        )

    # -- rule: vapour-liquid equilibrium -----------------------------------

    def _vle_property(self, prop, components, fractions, request, domain) -> Prediction | None:
        """The bubble point, and the enthalpy of vaporisation gated on it.

        Both live behind the same gate because they need the same thing: one
        temperature at which this formulation can be said to boil.  Without it
        the boiling point is a fiction and the enthalpy is a sum of numbers
        belonging to different temperatures.
        """
        # A one-component "mixture" is a pure substance. Solving a VLE for it
        # would return its own boiling point with an extra error bar bolted on;
        # passing the panel's value straight through is both more accurate and
        # more honest about what the number is.
        if len(components) == 1:
            return self._passthrough(prop, components[0], request, domain)

        if not unifac_available():
            return Prediction.unsupported(
                prop, self.id, "the thermo package does not expose modified UNIFAC here"
            )

        from .activity import evaluate, group_name, missing_interactions, unifac_groups

        groups = []
        for c in components:
            assignment = unifac_groups(c.smiles)
            if assignment is None:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"the DDBST compilation has no modified-UNIFAC assignment for "
                    f"{c.smiles!r}. Without activity coefficients this would have to be "
                    "solved as an ideal solution, and ideality moves the bubble point by "
                    "5 to 15 K on ordinary solvent pairs - larger than the range within "
                    "which a mixture may be said to boil at one temperature at all",
                )
            groups.append(assignment)

        missing = missing_interactions(groups)
        if missing:
            pairs = ", ".join(f"{group_name(a)}/{group_name(b)}" for a, b in missing)
            return Prediction.unsupported(
                prop,
                self.id,
                f"no modified-UNIFAC interaction parameter is tabulated for {pairs}. An "
                "absent parameter is read as zero, which is the value for two groups that "
                "mix perfectly, so the missing case and the ideal case would be "
                "indistinguishable in the answer",
            )

        pressures = []
        for c in components:
            vp = vapour_pressure(
                c.smiles,
                c.value("normal_boiling_point"),
                c.value("critical_temperature"),
                c.value("critical_pressure"),
            )
            if vp is None:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"no vapour pressure is available for {c.smiles!r}: it resolves to no "
                    "CAS with a measured correlation, and the panel has no boiling point "
                    "and critical constants to build an Ambrose-Walton estimate from",
                )
            pressures.append(vp)

        criticals = [c.value("critical_temperature") for c in components]
        ceiling = min(
            [t * 0.999 for t in criticals if t is not None] + [_VLE_T_MAX]
        )

        bubble = bubble_temperature(groups, pressures, fractions, ONE_ATM_PA, ceiling)
        if bubble is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "no bubble point exists in the liquid range at 1 atm for this composition: "
                "the modified-Raoult sum never reaches one atmosphere below the lowest "
                "component critical temperature",
            )

        # Stability is tested at the bubble point rather than at the request's
        # temperature, because the question this gate asks is whether the liquid
        # is one phase *where it boils*. Reading the request's temperature - or
        # worse, assuming ambient when none is stated - would answer a different
        # question at a temperature the boiling point does not happen at.
        stability = evaluate(groups, fractions, bubble)
        if stability is not None and stability.stability < 0:
            return Prediction.unsupported(
                prop,
                self.id,
                f"the binary is liquid-liquid unstable on the activity model's composition "
                f"scan at the bubble point (curvature {stability.stability:.2f} RT near "
                f"x = {stability.least_stable_fraction:.2f} at {bubble:.0f} K). A bubble "
                "point solved from a single liquid phase does not describe a mixture that "
                "boils as two. The scan covers the whole composition range rather than the "
                "stated one, so a dilute but genuinely single-phase blend - two per cent "
                "butanol in water scores -2.17 at ambient - is refused here too; that errs "
                "safe",
            )

        dew = dew_temperature(groups, pressures, fractions, ONE_ATM_PA, ceiling)
        if dew is None:
            return Prediction.unsupported(
                prop,
                self.id,
                f"the bubble point is {bubble:.1f} K but the dew point could not be solved, "
                "so the range this formulation boils over is unknown. A bubble point "
                "reported without it would be the temperature of the first drop and "
                "nothing more",
            )

        boiling_range = dew - bubble
        if boiling_range > MAX_BOILING_RANGE_K:
            return Prediction.unsupported(
                prop,
                self.id,
                f"this formulation does not have a boiling point: it boils over a range, "
                f"from a bubble point of {bubble:.1f} K to a dew point of {dew:.1f} K, "
                f"{boiling_range:.1f} K wide. A single number under this name would let the "
                f"blend satisfy a boiling-point window that most of its mass never meets. "
                f"The limit is {MAX_BOILING_RANGE_K:.0f} K, about twice the best bar this "
                "method can put on a bubble point",
            )

        model_spread, spread_basis = self._bubble_uncertainty(
            groups, pressures, fractions, components, bubble, ceiling
        )
        # Passing the gate says the range is small enough to summarise with one
        # number; it does not say the range is not there. The value reported is
        # the bubble point and the blend goes on boiling to the dew point, so
        # the range is an uncertainty on the quantity actually being asked for
        # and it belongs in the bar rather than only in a note. It is computed,
        # not assumed, and it vanishes for a true azeotrope.
        range_term = BOILING_RANGE_SIGMA_FRACTION * boiling_range
        spread = math.hypot(model_spread, range_term)
        spread_basis = (
            f"{model_spread:.3g} K of model error - {spread_basis} - combined in quadrature "
            f"with {range_term:.3g} K, being the {boiling_range:.2f} K computed boiling range "
            "read as a uniform spread between bubble and dew point. The value is the bubble "
            "point, so this bar is symmetric about a temperature that sits at the bottom of "
            "that range rather than in the middle of it"
        )
        at_bubble = Conditions(
            temperature=Quantity(value=bubble, unit="K"),
            pressure=Quantity(value=ONE_ATM_PA, unit="Pa"),
        )

        if prop == "normal_boiling_point":
            return self._make(
                prop,
                bubble,
                "K",
                request,
                domain,
                std=spread,
                kind=UncertaintyKind.COMBINED,
                basis=spread_basis,
                conditions=at_bubble,
                notes=(
                    f"this is the BUBBLE point at 1 atm - the temperature at which the "
                    f"first vapour appears - reported under the name normal_boiling_point "
                    f"only because the computed boiling range is {boiling_range:.1f} K, "
                    f"within the {MAX_BOILING_RANGE_K:.0f} K inside which a mixture may be "
                    f"said to boil at one temperature. The dew point is {dew:.1f} K",
                    "modified Raoult, sum(x_i gamma_i Psat_i) = 1 atm, with gamma from "
                    "modified UNIFAC (Dortmund); held-out mean absolute error 0.34 K over "
                    "five literature azeotropes of well-covered solvents",
                ),
                bubble_point_k=round(bubble, 3),
                dew_point_k=round(dew, 3),
            )

        return self._enthalpy(
            prop,
            components,
            fractions,
            groups,
            bubble,
            dew,
            spread,
            at_bubble,
            request,
            domain,
        )

    def _passthrough(self, prop, component, request, domain) -> Prediction | None:
        """A one-component formulation is a pure substance; hand its value over."""
        value = component.value(prop)
        if value is None:
            return Prediction.unsupported(
                prop,
                self.id,
                f"this formulation has one component and the panel has no {prop} for "
                f"{component.smiles!r}",
            )
        spread = component.std(prop)
        return self._make(
            prop,
            value,
            _COMPONENT_UNITS[prop] or "",
            request,
            domain,
            std=spread,
            kind=UncertaintyKind.COMBINED,
            basis=(
                f"unchanged from the molecular panel's own {prop} for this component, "
                "including its uncertainty; no mixing rule was applied because there is "
                "nothing to mix"
            ),
            notes=(
                "a one-component formulation is a pure substance, so this is the "
                "component's own value rather than the result of any mixing rule",
            ),
        )

    def _bubble_uncertainty(
        self, groups, pressures, fractions, components, bubble, ceiling
    ) -> tuple[float, str]:
        """Re-solve the bubble point at perturbed inputs.

        Propagated numerically rather than linearised, because the root of
        ``sum(x gamma Psat) = P`` has no closed form to differentiate and the
        sensitivity is what a user needs to see.  Two mechanisms: the activity
        model's own accuracy, and each component's panel uncertainty on its
        boiling point carried into a vapour-pressure uncertainty through
        Clausius-Clapeyron, which is what makes a blend of novel molecules come
        back with a visibly wider bar than a blend of known solvents.
        """
        terms: list[str] = []

        shifted = bubble_temperature(
            groups, pressures, fractions, ONE_ATM_PA, ceiling, gamma_scale=GAMMA_PERTURBATION
        )
        gamma_term = abs(shifted - bubble) if shifted is not None else 0.0
        terms.append(
            f"{gamma_term:.3g} K from scaling every activity coefficient by "
            f"{GAMMA_PERTURBATION:.2f}, modified UNIFAC's usual accuracy"
        )

        vapour_term = 0.0
        for index, component in enumerate(components):
            boiling = component.value("normal_boiling_point")
            boiling_std = component.std("normal_boiling_point")
            if not boiling or not boiling_std:
                continue
            step = 0.5
            try:
                high = pressures[index](bubble + step)
                low = pressures[index](bubble - step)
                slope = (math.log(high) - math.log(low)) / (2.0 * step)
            except Exception:
                continue
            factor = math.exp(slope * boiling_std)
            scale = [1.0] * len(components)
            scale[index] = factor
            moved = bubble_temperature(
                groups, pressures, fractions, ONE_ATM_PA, ceiling, psat_scale=tuple(scale)
            )
            if moved is not None:
                vapour_term += abs(moved - bubble)
        terms.append(
            f"{vapour_term:.3g} K from each component's own boiling-point uncertainty, "
            "converted to a vapour-pressure uncertainty through Clausius-Clapeyron and "
            "summed linearly as correlated"
        )

        total = math.hypot(gamma_term, vapour_term)
        return total, (
            "propagated by re-solving the bubble point at perturbed inputs: "
            + "; ".join(terms)
            + ". Measured over forty-seven common solvent binaries this lands between 1.8 "
            "and 2.4 K, almost all of it the activity term, because a molecule UNIFAC can "
            "assign groups to usually has a measured boiling point too. Where it does not "
            "the bar follows: a component carrying Joback's 12.9 K takes the "
            "acetone/methanol bubble point from 2.0 K to 14.0 K"
        )

    def _enthalpy(
        self,
        prop,
        components,
        fractions,
        groups,
        bubble,
        dew,
        bubble_spread,
        at_bubble,
        request,
        domain,
    ) -> Prediction | None:
        """Component enthalpies Watson-corrected to one temperature, less H^E."""
        corrected: list[float] = []
        component_term = 0.0
        for x, c in zip(fractions, components):
            enthalpy = c.value("enthalpy_vaporization")
            boiling = c.value("normal_boiling_point")
            critical = c.value("critical_temperature")
            if enthalpy is None or boiling is None or critical is None:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"the Watson correction of {c.smiles!r} to the mixture's bubble point "
                    "needs its enthalpy of vaporisation, its own boiling point and its "
                    "critical temperature, and the panel does not have all three. Adding "
                    "enthalpies defined at different temperatures is the error this rule "
                    "exists to avoid - decane's is 25% larger at pentane's boiling point "
                    "than at its own",
                )
            if not (bubble < critical and boiling < critical):
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"the mixture boils at {bubble:.1f} K, at or above the critical "
                    f"temperature of {c.smiles!r} ({critical:.1f} K); there is no enthalpy "
                    "of vaporisation to correct",
                )
            value = watson_enthalpy_vaporization(enthalpy, boiling, critical, bubble)
            corrected.append(value)
            spread = c.std("enthalpy_vaporization")
            if spread is None:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"the panel's enthalpy of vaporisation for {c.smiles!r} carries no "
                    "uncertainty",
                )
            component_term += x * spread * (value / enthalpy)

        # H^E is the heat absorbed on mixing the liquids; vaporising the
        # mixture releases it again, so it is subtracted from the ideal sum. Not
        # negligible for every pair: measured here it is +48 J/mol for
        # ethanol/2-propanol (0.13% of the additive sum, far below the noise)
        # but -1749 J/mol for acetone/chloroform, which is 5.9% and comparable
        # to the 2.2 kJ/mol the panel already carries on a single component.
        def at_temperature(temperature: float) -> float | None:
            try:
                heat = float(unifac_model(groups, fractions, temperature).HE())
            except Exception:
                return None
            total = 0.0
            for x, c in zip(fractions, components):
                total += x * watson_enthalpy_vaporization(
                    c.value("enthalpy_vaporization"),
                    c.value("normal_boiling_point"),
                    c.value("critical_temperature"),
                    temperature,
                )
            return total - heat

        value = at_temperature(bubble)
        if value is None:
            return Prediction.unsupported(
                prop, self.id, "the activity model produced no excess enthalpy"
            )
        excess = sum(x * h for x, h in zip(fractions, corrected)) - value

        # The enthalpy is stated AT the bubble point, so the uncertainty on the
        # bubble point is an uncertainty on this number too, not a separate
        # fact about a separate property. Re-evaluated at the shifted
        # temperature rather than linearised, for the same reason the bubble
        # point itself is: the Watson exponent and H^E both move with T and
        # neither is worth differentiating by hand.
        ceiling = min(c.value("critical_temperature") for c in components)
        shifted = bubble + bubble_spread
        if shifted >= ceiling:
            shifted = max(bubble - bubble_spread, _VLE_T_MIN)
        moved = at_temperature(shifted)
        temperature_term = abs(moved - value) if moved is not None else 0.0

        excess_term = EXCESS_ENTHALPY_RELATIVE_ERROR * abs(excess)
        std = math.sqrt(component_term**2 + excess_term**2 + temperature_term**2)

        return self._make(
            prop,
            value,
            "J/mol",
            request,
            domain,
            std=std,
            kind=UncertaintyKind.COMBINED,
            basis=(
                f"{component_term:.4g} J/mol from the components' own enthalpies, Watson-"
                "scaled to the bubble point and summed linearly as fully correlated; "
                f"combined in quadrature with {excess_term:.4g} J/mol, being "
                f"{EXCESS_ENTHALPY_RELATIVE_ERROR:.0%} of the {excess:.0f} J/mol excess "
                "enthalpy subtracted, because modified UNIFAC reproduces an excess "
                f"enthalpy far worse than an activity coefficient, and with "
                f"{temperature_term:.4g} J/mol from re-evaluating the whole rule "
                f"{bubble_spread:.2f} K away, which is the bar on the bubble point this "
                "enthalpy is reported at"
            ),
            conditions=at_bubble,
            notes=(
                f"mole-fraction additive ONLY after each component is Watson-corrected "
                f"from its own boiling point to the mixture's bubble point, {bubble:.1f} K; "
                f"the modified-UNIFAC excess enthalpy ({excess:.0f} J/mol) is then "
                "subtracted. A plain mole-fraction average of tabulated values would be "
                "adding numbers that belong to different temperatures",
                f"reported at {bubble:.1f} K, not at any pure component's normal boiling "
                f"point; the registry defines this property at the normal boiling point "
                f"and for a mixture the bubble point is the closest thing there is. A "
                f"consumer that Watson-corrects it again would double-correct. The boiling "
                f"range is {dew - bubble:.1f} K",
            ),
            bubble_point_k=round(bubble, 3),
        )

    # -- rule: refusal ------------------------------------------------------

    def _logp(self, prop, components, fractions, request, domain) -> Prediction | None:
        if len(components) == 1:
            return self._passthrough(prop, components[0], request, domain)
        return Prediction.unsupported(
            prop,
            self.id,
            "a formulation has no partition coefficient. logP is the ratio in which a "
            "single solute at infinite dilution distributes between two phases; a "
            "formulation is not a solute, each of its components would partition "
            "differently, both phases would change composition as it did, and the "
            "components would act as co-solvents on one another. A mole-fraction average "
            "of logs is the geometric mean of the component partition coefficients, which "
            "corresponds to no experiment anyone can perform. Ask for the components' "
            "values instead",
        )

    # -- rule: worst component ---------------------------------------------

    def _accessibility(self, prop, components, request, domain) -> Prediction | None:
        scored = [(c.value(prop), c) for c in components]
        missing = [c.smiles for value, c in scored if value is None]
        if missing:
            return Prediction.unsupported(
                prop,
                self.id,
                f"no synthetic accessibility for {missing[0]!r}; the maximum over a subset "
                "of the components is not the maximum",
            )

        value, hardest = max(scored, key=lambda pair: pair[0])
        spread = hardest.std(prop)
        if spread is None:
            # The basis string below promises this is the selected component's
            # own number. Substituting a default here would make that sentence
            # false, which is worse than refusing.
            return Prediction.unsupported(
                prop,
                self.id,
                f"the panel's synthetic accessibility for {hardest.smiles!r} - the hardest "
                "component, and the one this rule selects - carries no uncertainty, so the "
                "formulation's cannot be stated",
            )
        return self._make(
            prop,
            value,
            "",
            request,
            domain,
            std=spread,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                f"the selected component's own {spread:.3g}, unchanged: this is one "
                "component's score rather than a combination of several. The maximum of "
                "several noisy estimates is biased upward, so the true hardest component "
                "is likely a little easier than this says"
            ),
            notes=(
                f"the MAXIMUM over components, not an average: every component has to be "
                f"made, so the hardest one gates the formulation - here {hardest.smiles!r} "
                f"at {value:.2f}. An average would let easy solvents hide one impossible "
                f"molecule. It says nothing about whether the components can be blended, "
                f"whether they are purchasable, or that a {len(components)}-component "
                f"recipe is harder to assemble than a one-component one",
            ),
        )
