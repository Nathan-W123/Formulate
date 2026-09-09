"""Shear and extensional viscosity.

``shear_viscosity`` has been in the property registry since Phase 1 with
nothing behind it, and section 6 records why physics cannot supply it: a
viscosity comes from a Green-Kubo integral of the stress autocorrelation or
from non-equilibrium shear, and this system runs neither. That left it
uncovered from both directions, which is what this module fixes - from the
correlation side, where it was always reachable.

Three routes to a shear viscosity, in the order they should be believed:

**Measured**, from the DIPPR and VDI compilations already installed. Restricted
to data methods, for the same reason the density route is: ``thermo`` will drop
through to Letsou-Stiel or Joback without saying so, and a "measured" expert
that is quietly an estimator is worse than no measured expert at all.

**Joback group contribution**, which is the better estimator wherever it can
type a molecule: log10 mean absolute error 0.108 over 195 of 273 compounds with
a measured viscosity, a typical factor of 1.17 and 93 per cent inside a factor
of two.

**Letsou-Stiel corresponding states**, which answers everything and is much
weaker: log10 mean absolute error 0.296 held out, a typical factor of 1.6 and
66 per cent inside a factor of two.

None of the three has precedence written into it. Each states its own spread
and ``prefer()`` chooses, exactly as it does between the two refractive index
routes.

Extensional viscosity is a different kind of claim and is treated as one. For
an incompressible Newtonian liquid in uniaxial extension the Trouton ratio is
exactly three - that is a result, not a correlation, and it carries no error of
its own beyond the shear viscosity it is built from. For a polymer solution it
is false: the extensional viscosity rises by orders of magnitude as chains
stretch, depends on strain rate and on strain history, and is not a single
number. So :class:`TroutonExtensionalExpert` answers for a liquid and refuses
for a polymer, rather than returning three times something and letting the
caller assume it means what it says.
"""

from __future__ import annotations

import functools
import math

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.prediction import Prediction, PredictionStatus
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import ProvenanceKind, ProvenanceRecord
from formulate.core.quantity import ApplicabilityDomain, Quantity, Uncertainty, UncertaintyKind

from .base import Expert, PredictionRequest

#: Exact for an incompressible Newtonian fluid in uniaxial extension.
TROUTON_RATIO = 3.0

#: Nothing liquid at ambient conditions sits outside this, and a correlation
#: evaluated out of range does. ``VISWANATH_NATARAJAN_2E`` returns 7470 Pa s for
#: 2-butanone, which is seven orders of magnitude high and looks like a number.
#:
#: The upper bound is a hundred pascal-seconds. The most viscous pure
#: small-molecule liquid at ambient is around glycerol at 1.4, so this leaves
#: nearly two orders of magnitude of headroom and still rejects the 7470 by a
#: factor of seventy-five. A first attempt used ten thousand, which is a number
#: a molten polymer can reach and let the bad value straight through - this
#: expert answers for small-molecule liquids and the bound should say so.
#: The lower bound sits below any real liquid: pentane and diethyl ether are the
#: least viscous common ones at about 2e-4.
_PLAUSIBLE_RANGE = (1e-6, 1e2)

#: ``thermo`` viscosity methods that carry tabulated data or a fitted reference
#: correlation, as opposed to predicting from structure. The two estimating
#: methods are deliberately absent: this panel has its own routes to both, and
#: reaching them through a "measured" expert would hide which one answered.
_MEASURED_VISCOSITY_METHODS = ("DIPPR_PERRY_8E", "VDI_PPDS")

#: log10 mean absolute error of each estimator against 273 compounds with a
#: measured liquid viscosity at 298 K, from ``bench/viscosity_routes.py``.
_JOBACK_LOG10_MAE = 0.108
_LETSOU_STIEL_LOG10_MAE = 0.296

#: Letsou-Stiel under-predicts these liquids by a consistent factor. The offset
#: is fitted on half the compounds, split deterministically on a hash of the
#: structure, and the figure above is measured on the other half. It removes the
#: bias (-0.352 to -0.028 on the held-out split) and barely touches the spread,
#: which is the honest summary: the method's problem is scatter, and only its
#: offset is correctable.
_LETSOU_STIEL_OFFSET = -0.3243

#: Natural log of ten, for turning a log10 spread into a relative one.
_LN10 = math.log(10.0)


def _multiplicative_uncertainty(
    value: float, log10_mae: float, basis: str
) -> Uncertainty:
    """A spread for a quantity whose error is a factor rather than an amount.

    A viscosity that is wrong by a factor of two is wrong by 0.5 mPa s at one
    millipascal-second and by 50 Pa s at a hundred, so a single absolute spread
    describes neither. ``std`` carries the local linear equivalent because the
    ranking needs one number, and the confidence interval carries the factor
    itself, which is the honest shape.
    """
    sigma_log10 = 1.253 * log10_mae
    factor = 10.0**sigma_log10
    return Uncertainty(
        std=value * _LN10 * sigma_log10,
        ci_low=value / factor,
        ci_high=value * factor,
        kind=UncertaintyKind.EPISTEMIC,
        basis=(
            f"{basis}; the error is multiplicative, so one sigma is a factor of "
            f"{factor:.2f} rather than a fixed amount, and the interval carries that "
            "while std carries its local linear equivalent"
        ),
    )


# ---------------------------------------------------------------------------
# The correlations themselves
# ---------------------------------------------------------------------------


def joback_viscosity(smiles: str, temperature: float) -> float | None:
    """Liquid viscosity in Pa s from Joback's group contribution, or None."""
    try:
        from thermo.group_contribution import Joback

        value = Joback(smiles).mul(temperature)
    except Exception:
        return None
    if value is None or not _PLAUSIBLE_RANGE[0] < value < _PLAUSIBLE_RANGE[1]:
        return None
    return float(value)


def letsou_stiel_viscosity(
    temperature: float,
    molar_mass_g_mol: float,
    critical_temperature: float,
    critical_pressure: float,
    acentric_factor: float,
    *,
    corrected: bool = True,
) -> float | None:
    """Liquid viscosity in Pa s from Letsou-Stiel corresponding states.

    ``corrected`` applies the fitted offset above. Passing False gives the
    correlation as published, which is what the benchmark compares against.
    """
    from chemicals.viscosity import Letsou_Stiel

    if temperature >= critical_temperature:
        return None
    try:
        value = Letsou_Stiel(
            temperature,
            molar_mass_g_mol,
            critical_temperature,
            critical_pressure,
            acentric_factor,
        )
    except Exception:
        return None
    if value is None or not _PLAUSIBLE_RANGE[0] < value < _PLAUSIBLE_RANGE[1]:
        return None
    if corrected:
        value = value * 10.0 ** (-_LETSOU_STIEL_OFFSET)
    return float(value)


def acentric_factor(
    boiling_point: float, critical_temperature: float, critical_pressure: float
) -> float | None:
    """Lee-Kesler acentric factor from constants the panel already produces."""
    from chemicals.acentric import LK_omega

    try:
        value = LK_omega(boiling_point, critical_temperature, critical_pressure)
    except Exception:
        return None
    return None if value is None else float(value)


# ---------------------------------------------------------------------------
# Experts
# ---------------------------------------------------------------------------


class _ViscosityExpert(Expert):
    """Shared plumbing: a liquid, at a stated temperature, in the liquid range."""

    family = PropertyFamily.INTERFACIAL
    supported_classes = frozenset({MaterialClass.MOLECULE})
    supported_properties = frozenset({"shear_viscosity"})

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "RDKit is required to read the structure"

    def _liquid_at(self, request: PredictionRequest) -> tuple[str, float] | Prediction:
        """The structure and temperature, or the refusal that stands in for them."""
        prop = "shear_viscosity"
        if request.candidate.molecule is None:
            return Prediction.unsupported(prop, self.id, "needs a single molecule")
        temperature = request.conditions.temperature_k
        if temperature is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "a liquid viscosity changes by a factor of two over twenty degrees and "
                "no temperature was given",
            )
        smiles = request.candidate.molecule.smiles
        from .measured import not_liquid_at

        wrong_phase = not_liquid_at(smiles, temperature)
        if wrong_phase:
            return Prediction.unsupported(prop, self.id, wrong_phase)
        return smiles, temperature

    def _emit(
        self,
        value: float,
        uncertainty: Uncertainty,
        method: str,
        request: PredictionRequest,
        domain: ApplicabilityDomain,
        notes: tuple[str, ...] = (),
    ) -> Prediction:
        return Prediction(
            property="shear_viscosity",
            quantity=Quantity(value=value, unit="Pa*s"),
            uncertainty=uncertainty,
            applicability=domain,
            status=PredictionStatus.OK,
            expert_id=self.id,
            expert_version=self.version,
            method=method,
            conditions=request.conditions,
            provenance=ProvenanceRecord(
                kind=ProvenanceKind.PREDICTION,
                producer=self.id,
                producer_version=self.version,
                parameters={"temperature_k": request.conditions.temperature_k},
            ),
            notes=notes,
        )


class MeasuredViscosityExpert(_ViscosityExpert):
    """Liquid viscosity from a compilation, where one exists."""

    id = "viscosity_measured"
    version = "1"

    def _predict_one(self, prop, request, domain) -> Prediction | None:
        ready = self._liquid_at(request)
        if isinstance(ready, Prediction):
            return ready
        smiles, temperature = ready

        from .hansen import resolve_cas

        cas = resolve_cas(smiles)
        if cas is None:
            return Prediction.unsupported(
                prop, self.id, "no CAS number resolves for this structure"
            )
        value = _measured_viscosity(cas, temperature)
        if value is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "no tabulated viscosity covers this compound at this temperature; the "
                "estimating routes are separate experts and are not reached from here",
            )
        return self._emit(
            value,
            Uncertainty(
                # Compilations of a measured liquid viscosity agree with each
                # other to a few per cent; the DIPPR and VDI correlations for
                # 2-butanone differ by 0.2 per cent at 298 K.
                std=value * 0.03,
                kind=UncertaintyKind.EPISTEMIC,
                basis=(
                    "spread between the DIPPR and VDI compilations for a measured "
                    "liquid viscosity, about three per cent"
                ),
            ),
            "tabulated liquid viscosity (DIPPR 8E / VDI)",
            request,
            domain,
            notes=("measured, not estimated",),
        )


class JobackViscosityExpert(_ViscosityExpert):
    """Group contribution, the better estimator where it can type the molecule."""

    id = "viscosity_joback"
    version = "1"

    def _predict_one(self, prop, request, domain) -> Prediction | None:
        ready = self._liquid_at(request)
        if isinstance(ready, Prediction):
            return ready
        smiles, temperature = ready

        value = joback_viscosity(smiles, temperature)
        if value is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "Joback has no viscosity groups matching this structure, or the result "
                "fell outside the range any liquid occupies",
            )
        return self._emit(
            value,
            _multiplicative_uncertainty(
                value,
                _JOBACK_LOG10_MAE,
                "Joback group contribution against 195 compounds with a measured "
                "liquid viscosity at 298 K, log10 mean absolute error 0.108",
            ),
            "Joback group contribution",
            request,
            domain,
        )


class CorrespondingStatesViscosityExpert(_ViscosityExpert):
    """Letsou-Stiel, which answers everything and is much weaker."""

    id = "viscosity_corresponding_states"
    version = "1"
    dependencies = frozenset(
        {"critical_temperature", "critical_pressure", "normal_boiling_point", "molar_mass"}
    )

    def _predict_one(self, prop, request, domain) -> Prediction | None:
        ready = self._liquid_at(request)
        if isinstance(ready, Prediction):
            return ready
        _, temperature = ready

        needed = {}
        for name, unit in (
            ("critical_temperature", "K"),
            ("critical_pressure", "Pa"),
            ("normal_boiling_point", "K"),
            ("molar_mass", "g/mol"),
        ):
            upstream = request.dependency(name)
            if upstream is None or upstream.quantity is None:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"needs {name} from an upstream expert, and none was available",
                )
            needed[name] = upstream.quantity.to(unit).value

        omega = acentric_factor(
            needed["normal_boiling_point"],
            needed["critical_temperature"],
            needed["critical_pressure"],
        )
        if omega is None:
            return Prediction.failed(
                prop, self.id, "the Lee-Kesler acentric factor could not be formed"
            )
        value = letsou_stiel_viscosity(
            temperature,
            needed["molar_mass"],
            needed["critical_temperature"],
            needed["critical_pressure"],
            omega,
        )
        if value is None:
            return Prediction.failed(
                prop,
                self.id,
                "Letsou-Stiel returned nothing usable at this temperature",
            )
        return self._emit(
            value,
            _multiplicative_uncertainty(
                value,
                _LETSOU_STIEL_LOG10_MAE,
                "Letsou-Stiel with a fitted offset, measured on 131 compounds held out "
                "of that fit, log10 mean absolute error 0.296",
            ),
            "Letsou-Stiel corresponding states with a fitted offset",
            request,
            domain,
            notes=(
                f"the published correlation under-predicts these liquids by a factor of "
                f"{10 ** -_LETSOU_STIEL_OFFSET:.2f}, and that offset is corrected here; "
                "the offset was fitted on half the compounds and the stated error "
                "measured on the other half",
                "corresponding states, so it answers for structures no group table "
                "covers and is the weaker route wherever one does",
            ),
        )


#: Williams-Landel-Ferry constants referenced to each polymer's own glass
#: transition, from Ferry, Viscoelastic Properties of Polymers. Keyed by
#: canonical repeat unit, because a PolymerSpec carries no name.
#:
#: The universal pair (17.44, 51.6) is deliberately absent. Anchored at Tg it
#: puts polystyrene at 3 Pa s at 200 degrees, against a real melt viscosity of
#: order 10^3 to 10^4 - three to four orders of magnitude out, because the
#: universal constants were never meant to be carried a hundred kelvin above
#: Tg. The polymer-specific pair puts it at 740, which is within the order of
#: magnitude this construction is worth. So an untabulated polymer is refused
#: rather than answered with the universal pair.
WLF_CONSTANTS: dict[str, tuple[float, float]] = {
    "[*]CC([*])c1ccccc1": (13.7, 50.0),          # polystyrene
    "[*]CC([*])OC(C)=O": (15.6, 46.8),           # poly(vinyl acetate)
    "[*]CC([*])(C)C(=O)OC": (34.0, 80.0),        # poly(methyl methacrylate)
}

#: Viscosity at the glass transition, Pa s. This is the rheological definition
#: of Tg rather than a fitted parameter: the transition is where a glass-former
#: reaches about 10^12 Pa s.
_VISCOSITY_AT_TG = 1.0e12

#: What the whole construction is worth. An order of magnitude, because that is
#: the spread between the polymer-specific WLF result and a measured melt
#: viscosity, and claiming better would be claiming the table is a measurement.
_MELT_LOG10_MAE = 0.8

#: WLF is a fit over the range it was measured in, roughly Tg to Tg + 100 K.
#: Beyond that it is an extrapolation of a divergent function and the answer
#: falls apart quietly, so it is bounded rather than trusted.
_WLF_RANGE_K = 120.0

#: A floor below which the answer is not a polymer melt, Pa s.
#:
#: Staying inside the 120 K window is not enough on its own, and a run over the
#: polymer catalogue is what showed it. Asked for poly(methyl methacrylate) at
#: 180 C, which is 75 K above its transition and well inside the window, the
#: tabulated pair (34.0, 80.0) returns 3.4e-5 Pa s - thinner than water by a
#: factor of thirty, for a polymer whose real melt viscosity there is of order
#: 10^4. It came first in a design run on that number.
#:
#: The defect is in the constant pair rather than in the arithmetic. ``c1`` is
#: the number of decades the viscosity falls between the reference temperature
#: and the high-temperature asymptote, so a pair referenced to Tg cannot have a
#: ``c1`` that puts the asymptote below any liquid that exists: 13.7 leaves
#: polystyrene asymptotic to 0.02 Pa s, and 34.0 leaves this one asymptotic to
#: 10^-22, which is not a viscosity. That pair is very likely quoted against a
#: reference temperature of its own rather than against the transition, and the
#: table cannot tell.
#:
#: So the floor is set at the viscosity of water, which nothing entangled has
#: ever been under, and the expert refuses rather than reporting a number it
#: can prove is wrong. Refusing loses poly(methyl methacrylate) as an answer;
#: reporting it wins the run with a lie.
_MELT_VISCOSITY_FLOOR = 1.0e-3


def wlf_melt_viscosity(
    temperature: float, glass_transition: float, c1: float, c2: float
) -> float | None:
    """Zero-shear melt viscosity in Pa s, anchored at 10^12 Pa s at Tg."""
    shift = temperature - glass_transition
    if shift < 0.0 or shift > _WLF_RANGE_K:
        return None
    exponent = math.log10(_VISCOSITY_AT_TG) - c1 * shift / (c2 + shift)
    return 10.0**exponent


class MeltViscosityExpert(Expert):
    """Melt viscosity of an amorphous polymer, by WLF from its own Tg.

    The weakest expert in the panel and labelled as such: one order of
    magnitude, on three tabulated polymers, over a hundred-kelvin window above
    the glass transition. It exists because the alternative for a melt-spinning
    question was no number at all, and because being wrong by a factor of ten
    still settles a question whose answer differs by ten thousand.
    """

    id = "melt_wlf"
    version = "1"
    family = PropertyFamily.INTERFACIAL
    supported_classes = frozenset({MaterialClass.POLYMER})
    supported_properties = frozenset({"shear_viscosity"})
    dependencies = frozenset({"glass_transition_temperature"})

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "RDKit is required to read the repeat unit"

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        unit = _repeat_unit(candidate)
        if unit is None or unit not in _canonical_wlf():
            return ApplicabilityDomain.outside(
                "no measured WLF constants for this repeat unit, and the universal pair "
                "is three orders of magnitude out a hundred kelvin above Tg",
                basis=f"{len(WLF_CONSTANTS)} tabulated polymers",
            )
        return ApplicabilityDomain(
            basis="measured WLF constants for this polymer, referenced to its own Tg"
        )

    def _predict_one(self, prop, request, domain) -> Prediction | None:
        unit = _repeat_unit(request.candidate)
        if unit is None:
            return Prediction.unsupported(
                prop, self.id, "needs a single-repeat-unit polymer"
            )
        constants = _canonical_wlf().get(unit)
        if constants is None:
            return Prediction.unsupported(
                prop,
                self.id,
                f"no measured WLF constants for {unit}. The universal pair puts "
                "polystyrene three orders of magnitude out at a processing "
                "temperature, so it is not substituted here; the table covers "
                f"{sorted(WLF_CONSTANTS)}",
            )
        temperature = request.conditions.temperature_k
        if temperature is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "a melt viscosity changes by orders of magnitude over the processing "
                "window and no temperature was given",
            )
        upstream = request.dependency("glass_transition_temperature")
        if upstream is None or upstream.quantity is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "needs glass_transition_temperature from an upstream expert, and none "
                "was available",
            )
        glass_transition = upstream.quantity.to("K").value
        value = wlf_melt_viscosity(temperature, glass_transition, *constants)
        if value is not None and value < _MELT_VISCOSITY_FLOOR:
            asymptote = math.log10(_VISCOSITY_AT_TG) - constants[0]
            return Prediction.unsupported(
                prop,
                self.id,
                f"the tabulated pair returns {value:.3g} Pa s at {temperature:.0f} K, "
                f"below the {_MELT_VISCOSITY_FLOOR:g} Pa s of water, which no entangled "
                f"melt reaches. Its c1 of {constants[0]:.1f} puts the high-temperature "
                f"asymptote at 10^{asymptote:.0f} Pa s, so the pair cannot be referenced "
                "to this polymer's glass transition even though the table applies it as "
                "though it were. Refusing rather than reporting a number that is "
                "demonstrably wrong",
            )
        if value is None:
            shift = temperature - glass_transition
            return Prediction.unsupported(
                prop,
                self.id,
                f"the requested temperature is {shift:+.0f} K from the glass transition "
                f"and WLF is a fit over roughly 0 to {_WLF_RANGE_K:.0f} K above it; "
                "outside that it is an extrapolation of a divergent function"
                if shift > 0
                else "below the glass transition there is no melt to have a viscosity",
            )
        return Prediction(
            property=prop,
            quantity=Quantity(value=value, unit="Pa*s"),
            uncertainty=_multiplicative_uncertainty(
                value,
                _MELT_LOG10_MAE,
                "WLF anchored at 10^12 Pa s at the glass transition with measured "
                "constants; the construction is worth about an order of magnitude and "
                "has not been validated against melt viscosities in this repository",
            ),
            applicability=domain,
            status=PredictionStatus.OK,
            expert_id=self.id,
            expert_version=self.version,
            method="Williams-Landel-Ferry from the glass transition",
            conditions=request.conditions,
            provenance=ProvenanceRecord(
                kind=ProvenanceKind.PREDICTION,
                producer=self.id,
                producer_version=self.version,
                parameters={"c1": constants[0], "c2": constants[1]},
            ),
            notes=(
                f"{temperature - glass_transition:+.0f} K above the glass transition",
                "zero-shear: a melt shear-thins hard at spinning rates, so this is an "
                "upper bound on what the flow actually sees",
                "the weakest expert in this panel; it settles questions whose answers "
                "differ by orders of magnitude and should not be used for finer ones",
            ),
        )


@functools.lru_cache(maxsize=1)
def _canonical_wlf() -> dict[str, tuple[float, float]]:
    """WLF_CONSTANTS re-keyed on canonical SMILES.

    The table is written with explicit attachment points because that is how a
    repeat unit is read; RDKit canonicalises ``[*]`` to ``*``, so a lookup
    against the table as written silently misses every polymer in it.
    """
    from formulate import chem

    if not chem.rdkit_available():
        return dict(WLF_CONSTANTS)
    out: dict[str, tuple[float, float]] = {}
    for unit, constants in WLF_CONSTANTS.items():
        out[chem.canonical_smiles(unit) or unit] = constants
    return out


def _repeat_unit(candidate: Candidate) -> str | None:
    """The canonical repeat unit of a single-monomer polymer, or None."""
    polymer = candidate.polymer
    if polymer is None or len(polymer.monomers) != 1:
        return None
    from formulate import chem

    smiles = polymer.monomers[0].smiles
    return chem.canonical_smiles(smiles) if chem.rdkit_available() else smiles


class TroutonExtensionalExpert(Expert):
    """Extensional viscosity where it is a single number, and a refusal where it is not."""

    id = "trouton"
    version = "1"
    family = PropertyFamily.INTERFACIAL
    supported_classes = frozenset({MaterialClass.MOLECULE})
    supported_properties = frozenset({"extensional_viscosity"})
    dependencies = frozenset({"shear_viscosity"})

    def is_available(self) -> bool:
        return True

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        return ApplicabilityDomain(
            score=1.0,
            in_domain=True,
            basis=(
                "the Trouton ratio of three is exact for an incompressible Newtonian "
                "liquid in uniaxial extension; the domain question is whether the "
                "liquid is Newtonian, which is why a polymer is refused rather than "
                "scored down"
            ),
        )

    def _predict_one(self, prop, request, domain) -> Prediction | None:
        if request.candidate.material_class is not MaterialClass.MOLECULE:
            return Prediction.unsupported(
                prop,
                self.id,
                "the Trouton ratio is three only for a Newtonian liquid. A polymer "
                "solution or melt strain-hardens: its extensional viscosity rises by "
                "orders of magnitude as chains stretch and depends on strain rate and "
                "on strain history, so it is not one number and three times the shear "
                "viscosity is not an approximation to it",
            )
        upstream = request.dependency("shear_viscosity")
        if upstream is None or upstream.quantity is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "needs shear_viscosity from an upstream expert, and none was available",
            )

        shear = upstream.quantity.to("Pa*s").value
        value = TROUTON_RATIO * shear
        shear_std = upstream.uncertainty.converted(upstream.quantity.unit, "Pa*s").std
        return Prediction(
            property=prop,
            quantity=Quantity(value=value, unit="Pa*s"),
            uncertainty=Uncertainty(
                # Exactly three, so the whole spread is the shear viscosity's,
                # scaled. Nothing is added here because nothing is approximated
                # here.
                std=None if shear_std is None else TROUTON_RATIO * shear_std,
                ci_low=None
                if upstream.uncertainty.ci_low is None
                else TROUTON_RATIO * upstream.uncertainty.ci_low,
                ci_high=None
                if upstream.uncertainty.ci_high is None
                else TROUTON_RATIO * upstream.uncertainty.ci_high,
                kind=upstream.uncertainty.kind,
                basis=(
                    "three times the shear viscosity's own spread; the Trouton ratio is "
                    "exact for a Newtonian liquid and contributes no error of its own. "
                    f"The shear viscosity came from {upstream.expert_id}: "
                    f"{upstream.uncertainty.basis}"
                ),
            ),
            applicability=domain,
            status=PredictionStatus.OK,
            expert_id=self.id,
            expert_version=self.version,
            method="Trouton ratio of three, exact for a Newtonian liquid",
            conditions=request.conditions,
            provenance=ProvenanceRecord(
                kind=ProvenanceKind.PREDICTION,
                producer=self.id,
                producer_version=self.version,
            ),
            notes=(
                f"derived from shear_viscosity via {upstream.expert_id}",
                "a Newtonian result: it says nothing about a polymer solution, which is "
                "what a spinning dope is",
            ),
        )


def _measured_viscosity(cas: str, temperature: float) -> float | None:
    """Tabulated liquid viscosity in Pa s, from a data method only."""
    try:
        from thermo import ViscosityLiquid
    except Exception:
        return None
    try:
        obj = ViscosityLiquid(CASRN=cas)
        for method in _MEASURED_VISCOSITY_METHODS:
            if method not in obj.all_methods:
                continue
            obj.method = method
            value = obj.T_dependent_property(temperature)
            if value is None:
                continue
            value = float(value)
            if _PLAUSIBLE_RANGE[0] < value < _PLAUSIBLE_RANGE[1]:
                return value
    except Exception:
        return None
    return None
