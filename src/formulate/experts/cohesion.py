"""Cohesive energy density, and the bulk modulus that follows from it.

Two identities and one scaling law. Nothing here is fitted from scratch, and
the single constant the scaling law needs is measured against published values
rather than chosen.

**Cohesive energy density** is the defining identity ``CED = delta^2``. It
comes either from an upstream Hildebrand parameter, or - where only the three
Hansen components exist - as the plain sum ``dD^2 + dP^2 + dH^2``, which is
Hansen's own decomposition of the cohesive energy (C. M. Hansen, *Hansen
Solubility Parameters: A User's Handbook*, 2nd ed., CRC Press 2007, ch. 1).
The Hansen route sums energies directly; no square root is taken and then
squared.

**Bulk modulus** is ``B = n * CED``, and that relation is derived rather than
correlated. For a Mie (m, n) pair potential ``u = -A r^-m + B r^-n``, expanding
the energy about its minimum gives ``B_modulus = u''(r0) r0^2 / (9 V0)``, which
reduces to ``(m n / 9) * CED``: eight for a 12-6 Lennard-Jones solid, 10.7 for
a 16-6 one, and the same algebra that gives Kittel's ``(n-1)|U|/(9V)`` for an
ionic crystal. It is the basis of the factor of eight to eleven quoted for
polymers in van Krevelen & te Nijenhuis, *Properties of Polymers*, 4th ed.,
Elsevier 2009, ch. 7, and is equivalent through ``P_internal = alpha T B_T`` to
the internal-pressure route of A. F. M. Barton, *Handbook of Solubility
Parameters and Other Cohesion Parameters*, 2nd ed., CRC Press 1991, ch. 5.

**The constant is not one number, and that is the main finding here.** Measured
at 298.15 K against published isothermal compressibilities and Hvap-based
solubility parameters:

* 27 liquids with ``dH/delta <= 0.5``: ``n = 2.81`` (geometric mean), spanning
  1.62 (acetonitrile) to 4.01 (aniline), worst leave-one-out ratio **1.77**.
* 11 solid and rubbery polymers: ``n = 9.11``, spanning 7.35 (cis-polyisoprene)
  to 11.08 (PMMA), worst leave-one-out ratio 1.27 - but carried at a factor of
  **two**, because the published reference bulk moduli disagree with each other
  by more than that (see below).

So the eight-to-eleven figure is right for a polymer and wrong by a factor of
three for a liquid. Applying ``n = 9.11`` to hexane gives 2.01 GPa against a
measured 0.599 GPa. A single constant is not supportable and this module does
not ship one.

The split is not a fudge. ``B_T = (P_internal / CED) * CED / (alpha T)``, and
``P_internal / CED`` measures 0.66 to 1.17 (mean 0.99) over the same 27
liquids - the published pattern, which is also what confirms the compressibility
table below is right. What differs between the two regimes is ``alpha T``: 0.24
to 0.49, mean 0.355, for these liquids, against roughly 0.06 for a glassy
polymer and 0.18 for a rubber. Same law, different free volume, and 1/0.355 = 2.8
against 1/0.12 = 8.3.

What this does **not** do, and why.

* No bulk modulus for a hydrogen-bonded liquid. Across the nine liquids with
  ``dH/delta > 0.5`` the ratio collapses to 0.93 (methanol) and 0.96 (water):
  cohesive energy density counts hydrogen bonds the bulk modulus does not feel,
  and the error is systematic rather than scatter. Including them moves the
  constant to 2.53 and the worst leave-one-out ratio to 2.79.
* No bulk modulus for a fluorine-rich structure. PTFE measures ``n = 23.6``, a
  2.6x miss in one direction, because fluorine's low polarisability gives a
  small cohesive energy density at high stiffness.
* No answer at all for a mixture that is part molecule and part polymer:
  neither regime was validated on it, and answering with the major component's
  regime is exactly the substitution the house rules forbid.
* The polymer branch is validated only near 298 K on solids and rubbers, and
  what happens outside that window is genuinely not known. PDMS, 150 K above
  its Tg, measures 4.4 - a melt. Polyethylene is 150 K above its Tg too, is in
  the fit set, and measures 10.7, because it is semicrystalline and the
  crystallites carry load the amorphous cohesive energy density never sees. The
  two known points out there miss in opposite directions, so a polymer far
  above its Tg is answered with a symmetrically widened bar and a note saying
  which way it goes depends on crystallinity - which is not in the property
  registry. A known hole, not an oversight, and not a one-sided one.
* Semicrystalline polymers conflate two materials: crystallites raise the
  modulus without raising the amorphous cohesive energy proportionately. That
  shows up as extra spread, because degree of crystallinity is a processing
  variable and is not in the property registry.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

from formulate.core.candidate import Candidate, MaterialClass, MonomerRole
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest

#: Pa^0.5 per MPa^0.5. A thousand, not a million, because these are square
#: roots of a pressure. Getting this wrong is a documented past bug in this
#: repository - see the same constant in ``dissolution.py``, where it made
#: every real solvent for polystyrene score as a non-solvent - and it is worse
#: here, because squaring it turns a factor of a thousand into a factor of a
#: million while the answer still looks like a cohesive energy density.
PA_ROOT_PER_MPA_ROOT = 1000.0

#: Pa per GPa, used only to keep the reference tables readable.
_PA_PER_GPA = 1.0e9

#: Fraction of the total solubility parameter held in hydrogen bonding above
#: which no bulk modulus is produced for a liquid. Measured: the nine liquids
#: above this line run 0.93 to 4.20 against a constant of 2.81, and the three
#: worst (methanol 0.93, water 0.96, ethanol 1.32) are all low, so the error is
#: one-directional rather than scatter.
#:
#: Half is a judgement, not a fit, and the alternatives are recorded so the
#: choice stays visible: dropping the line to 0.25 would exclude acetonitrile
#: and improve the held-out ratio from 1.77 to about 1.50, which is a 35%
#: narrower error bar bought by removing one inconvenient compound. That is
#: boundary tuning and is deliberately not done.
HYDROGEN_BOND_FRACTION_LIMIT = 0.5

#: Fraction of heavy atoms that may be fluorine before the bulk modulus is
#: refused. Evidence is one polymer - PTFE, ``n = 23.6`` against 9.11 - so the
#: threshold errs toward refusing rather than toward a wider bar, because the
#: one miss measured is 2.6x in a single direction and an error bar centred on
#: 9.11 would not contain it.
FLUORINE_FRACTION_LIMIT = 0.25

#: Temperature window for a bulk modulus, in kelvin. Every validation point is
#: at 298.15 K and the ratio runs as ``1/(alpha T)``, so extrapolation is a
#: systematic error that a wider bar cannot rescue. Inside the inner window the
#: prediction is in domain; between the windows it is produced and flagged;
#: outside the outer one it is refused.
BULK_MODULUS_T_INNER = (283.0, 323.0)
BULK_MODULUS_T_OUTER = (250.0, 350.0)

#: Temperature window in which a cohesive energy density may be built from
#: Hansen components. The registry marks ``hansen_*`` as not condition
#: dependent because the compilations are tabulated at 25 C; a CED assembled
#: from them at 350 K would be a 298 K number wearing a 350 K label. The
#: Hildebrand route carries no such window - it is condition dependent upstream
#: and arrives already at the requested temperature.
HANSEN_ROUTE_T_WINDOW = (283.0, 323.0)

#: Floor on the spread of a tabulated solubility parameter, Pa^0.5. Matches the
#: compilation spread ``hansen.py`` carries (0.5 MPa^0.5) and is used only when
#: an upstream prediction states no uncertainty at all: an identity applied to a
#: number with no stated spread still has the spread of that number, and zero is
#: never the honest answer.
TABULATED_PARAMETER_FLOOR_PA_ROOT = 500.0

#: Route penalty on a CED built from a small molecule's Hansen triple rather
#: than from its Hildebrand parameter, as a fraction of the CED. Measured here
#: over 36 solvents: the Hansen sum reproduces the Hvap-based Hildebrand
#: parameter with a mean ratio of 1.003 and an RMS deviation of 2.6% (range
#: 0.951 to 1.121, worst acetic acid). CED goes as the square, so 2 x 2.6%.
HANSEN_ROUTE_CED_RELATIVE_MOLECULE = 0.052

#: The same penalty for a polymer, and it is the honest part of this module. A
#: solvent's tabulated triple is constructed to sum to its measured
#: vaporisation-based parameter; a polymer's triple is a solubility-sphere
#: centre fitted to which solvents dissolve it. Measured against the five
#: spheres in ``dissolution.py`` for which a group-contribution delta is also
#: available, the sphere centres give totals 1.044 to 1.337 times that delta
#: (RMS deviation 20.3%), i.e. cohesive energy densities up to 1.79x high. If
#: whichever expert supplies ``hansen_*`` for a polymer uses sphere centres,
#: every polymer CED here is systematically high and the bulk modulus with it;
#: this term is what keeps the stated bar honest about that.
HANSEN_ROUTE_CED_RELATIVE_POLYMER = 0.406

#: Multiplicative bar carried on a polymer bulk modulus, following the
#: convention of ``mechanical.py`` that a spread over a wide-ranging quantity is
#: a ratio rather than an absolute. The measured leave-one-out ratio is 1.27,
#: and that number is *not* what is carried: published room-temperature bulk
#: moduli for one polymer disagree by more than the model error being validated
#: - polystyrene is quoted from 2.4 GPa (isothermal, from pressure-volume-
#: temperature data) to 4.2 GPa (adiabatic, from ultrasonic sound speed), a
#: factor of 1.75 - so the reference values, not the model, set the floor.
POLYMER_CARRIED_RATIO = 2.0

#: Extra widening for a polymer more than this far above its glass transition.
#: Two points, and they disagree about the direction, which is why the widening
#: is symmetric and why the note this triggers must not claim a direction.
#: PDMS (Tg about 150 K) measures ``n = 4.4`` against 9.11 - 2.05x soft, a melt
#: rather than a rubber. Polyethylene (Tg about 148 K) is equally far above its
#: Tg, is *in* the fit set, and measures 10.7 - 1.17x stiff, because it is
#: semicrystalline and the crystallites carry load the amorphous cohesive
#: energy density knows nothing about. Distance above Tg therefore predicts
#: extra spread, not softness, and the ratio is set by the worse of the two
#: (9.113/4.44 = 2.05).
#:
#: The honest caveat: ``POLYMER_CARRIED_RATIO`` is already 2.0, so this widens
#: the bar by ten per cent and not by the factor of two the raw miss suggests.
#: It is kept because the ratio it names is measured and because the note it
#: attaches is the part that matters; it is not doing much work on its own.
FAR_ABOVE_TG_K = 100.0
FAR_ABOVE_TG_RATIO = 2.1

_HANSEN_PROPERTIES = ("hansen_dispersion", "hansen_polar", "hansen_hydrogen_bonding")


# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LiquidPoint:
    """One solvent at 298.15 K.

    ``delta`` is the Hvap-based Hildebrand parameter in MPa^0.5 as the
    ``thermo`` package computes it from the enthalpy of vaporisation and the
    saturated liquid molar volume. ``bulk_modulus`` is the reciprocal of a
    compiled isothermal compressibility in GPa. ``hydrogen_fraction`` is
    ``dH/delta_total`` from the Hansen compilation, which is what the gate is
    evaluated on.
    """

    name: str
    smiles: str
    delta_mpa_root: float
    bulk_modulus_gpa: float
    hydrogen_fraction: float

    @property
    def ced_pa(self) -> float:
        return (self.delta_mpa_root * PA_ROOT_PER_MPA_ROOT) ** 2

    @property
    def ratio(self) -> float:
        """Measured ``B_T / CED``."""
        return self.bulk_modulus_gpa * _PA_PER_GPA / self.ced_pa


#: Solvents at 298.15 K. Solubility parameters from ``thermo`` (enthalpy of
#: vaporisation over liquid molar volume); isothermal compressibilities from the
#: standard compilations - Y. Marcus, *The Properties of Solvents*, Wiley 1998,
#: and the CRC Handbook - at 298.15 K and 0.1 MPa.
#:
#: The compressibilities are the numbers this module most depends on and they
#: are cross-checked rather than trusted: the internal pressure they imply,
#: ``alpha T / kappa_T``, reproduces the published pattern of ``P_int / CED``
#: (1.02 to 1.17 for the hydrocarbons, 0.43 for ethanol, 0.07 for water). A
#: table with a bad compressibility in it would not do that.
#:
#: The gated-out entries are kept rather than deleted, so that the gate's
#: justification is checkable from the module and cannot drift from the prose.
LIQUID_REFERENCES: tuple[LiquidPoint, ...] = (
    LiquidPoint("water", "O", 47.93, 2.210, 0.885),
    LiquidPoint("glycerol", "OCC(O)CO", 33.66, 4.762, 0.795),
    LiquidPoint("ethylene glycol", "OCCO", 33.72, 2.703, 0.789),
    LiquidPoint("methanol", "CO", 29.30, 0.800, 0.758),
    LiquidPoint("ethanol", "CCO", 26.09, 0.901, 0.731),
    LiquidPoint("1-propanol", "CCCO", 24.59, 1.000, 0.707),
    LiquidPoint("1-butanol", "CCCCO", 23.53, 1.053, 0.681),
    LiquidPoint("acetic acid", "CC(O)=O", 19.06, 1.111, 0.632),
    LiquidPoint("1-hexanol", "CCCCCCO", 22.13, 1.111, 0.594),
    LiquidPoint("aniline", "Nc1ccccc1", 23.55, 2.222, 0.472),
    LiquidPoint("1,4-dioxane", "C1COCCO1", 20.54, 1.351, 0.455),
    LiquidPoint("N,N-dimethylformamide", "CN(C)C=O", 23.96, 1.538, 0.455),
    LiquidPoint("tetrahydrofuran", "C1CCOC1", 18.97, 1.020, 0.411),
    LiquidPoint("ethyl acetate", "CCOC(C)=O", 18.35, 0.885, 0.397),
    LiquidPoint("dimethyl sulfoxide", "CS(C)=O", 26.75, 1.908, 0.382),
    LiquidPoint("dichloromethane", "ClCCl", 20.37, 0.980, 0.358),
    LiquidPoint("acetone", "CC(C)=O", 19.64, 0.794, 0.351),
    LiquidPoint("chloroform", "ClC(Cl)Cl", 18.92, 1.000, 0.301),
    LiquidPoint("diethyl ether", "CCOCC", 15.35, 0.535, 0.297),
    LiquidPoint("pyridine", "c1ccncc1", 21.57, 1.316, 0.271),
    LiquidPoint("2-butanone", "CCC(C)=O", 18.88, 0.847, 0.268),
    LiquidPoint("acetonitrile", "CC#N", 24.05, 0.935, 0.250),
    LiquidPoint("nitromethane", "C[N+](=O)[O-]", 25.70, 1.333, 0.241),
    LiquidPoint("o-xylene", "Cc1ccccc1C", 18.38, 1.205, 0.171),
    LiquidPoint("nitrobenzene", "O=[N+]([O-])c1ccccc1", 22.59, 1.961, 0.136),
    LiquidPoint("toluene", "Cc1ccccc1", 18.24, 1.087, 0.110),
    LiquidPoint("benzene", "c1ccccc1", 18.74, 1.035, 0.108),
    LiquidPoint("carbon tetrachloride", "ClC(Cl)(Cl)Cl", 17.55, 0.943, 0.034),
    LiquidPoint("carbon disulfide", "S=C=S", 20.40, 1.075, 0.030),
    LiquidPoint("cyclohexane", "C1CCCCC1", 16.76, 0.877, 0.012),
    LiquidPoint("pentane", "CCCCC", 14.36, 0.450, 0.000),
    LiquidPoint("hexane", "CCCCCC", 14.87, 0.599, 0.000),
    LiquidPoint("heptane", "CCCCCCC", 15.21, 0.694, 0.000),
    LiquidPoint("octane", "CCCCCCCC", 15.45, 0.781, 0.000),
    LiquidPoint("decane", "CCCCCCCCCC", 15.80, 0.901, 0.000),
    LiquidPoint("dodecane", "CCCCCCCCCCCC", 16.08, 1.020, 0.000),
)


@dataclass(frozen=True, slots=True)
class PolymerPoint:
    """One polymer near 298 K, with where its bulk modulus came from."""

    name: str
    repeat_unit: str
    delta_mpa_root: float
    bulk_modulus_gpa: float
    source: str

    @property
    def ced_pa(self) -> float:
        return (self.delta_mpa_root * PA_ROOT_PER_MPA_ROOT) ** 2

    @property
    def ratio(self) -> float:
        return self.bulk_modulus_gpa * _PA_PER_GPA / self.ced_pa


#: Polymers near 298 K. Solubility parameters are the group-contribution values
#: of van Krevelen & te Nijenhuis ch. 7 and the *Polymer Handbook*, not
#: solubility-sphere centres - see ``HANSEN_ROUTE_CED_RELATIVE_POLYMER`` for why
#: the difference matters.
#:
#: The bulk moduli are the weakest numbers in this module and are labelled with
#: their route, because the route is most of the disagreement. For a glassy
#: polymer ``K = E / (3 (1 - 2 nu))`` from a measured tensile modulus and
#: Poisson ratio is the value most compilations quote; it is ill-conditioned as
#: nu approaches one half, so for rubbers a directly quoted bulk modulus is used
#: instead. Adiabatic values from sound speed run 5-10% high for polymers and 25
#: to 35% high for liquids and are not used.
POLYMER_REFERENCES: tuple[PolymerPoint, ...] = (
    PolymerPoint("polyethylene", "[*]CC[*]", 16.2, 2.8, "compiled compressibility, LDPE"),
    PolymerPoint("polypropylene", "[*]CC(C)[*]", 16.3, 2.9, "compiled compressibility"),
    PolymerPoint("polyisobutylene", "[*]CC(C)(C)[*]", 16.0, 2.0, "quoted bulk modulus, rubber"),
    PolymerPoint(
        "cis-1,4-polyisoprene", "[*]CC(C)=CC[*]", 16.5, 2.0, "quoted bulk modulus, rubber"
    ),
    PolymerPoint(
        "poly(vinyl acetate)", "[*]CC(OC(C)=O)[*]", 19.2, 3.3, "compiled compressibility"
    ),
    PolymerPoint("polystyrene", "[*]CC(c1ccccc1)[*]", 18.6, 3.1, "E 3.2 GPa, nu 0.33"),
    PolymerPoint(
        "poly(methyl methacrylate)",
        "[*]CC(C)(C(=O)OC)[*]",
        19.0,
        4.0,
        "E 3.1 GPa, nu 0.37",
    ),
    PolymerPoint("poly(vinyl chloride)", "[*]CC(Cl)[*]", 19.4, 4.1, "E 3.0 GPa, nu 0.38"),
    PolymerPoint(
        "polycarbonate",
        "[*]OC(=O)Oc1ccc(C(C)(C)c2ccc([*])cc2)cc1",
        19.4,
        3.0,
        "E 2.35 GPa, nu 0.37",
    ),
    PolymerPoint(
        "poly(ethylene terephthalate)",
        "[*]OCCOC(=O)c1ccc(C([*])=O)cc1",
        21.9,
        3.9,
        "E 2.8 GPa, nu 0.38",
    ),
    PolymerPoint("poly(ethylene oxide)", "[*]CCO[*]", 20.2, 3.5, "compiled compressibility"),
)

#: Polymers deliberately outside the fit, kept because they are the evidence
#: for two of the refusals and for the one hole this module admits to. Each
#: entry is (name, delta MPa^0.5, bulk modulus GPa, why it is out).
EXCLUDED_POLYMERS: tuple[tuple[str, float, float, str], ...] = (
    ("polytetrafluoroethylene", 12.7, 3.8, "fluorine-rich: measures 23.6, a 2.6x miss"),
    ("polydimethylsiloxane", 15.0, 1.0, "Tg 150 K, effectively a melt: measures 4.4"),
    ("polyamide 6,6", 27.8, 5.5, "strongly hydrogen bonded: measures 7.1, inside the bar"),
)


# --------------------------------------------------------------------------
# The measured model
# --------------------------------------------------------------------------


def geometric_mean(values: list[float]) -> float:
    """Geometric mean: the right average for a quantity carried as a ratio."""
    return math.exp(sum(math.log(v) for v in values) / len(values))


def worst_leave_one_out(values: list[float]) -> float:
    """Worst ratio between a point and a constant fitted without it.

    The whole set is used for the shipped constant and every point is scored
    against a constant that never saw it, so the quoted spread is a held-out
    number by construction rather than the in-sample one a fit reproduces for
    free.
    """
    worst = 1.0
    for index, value in enumerate(values):
        others = values[:index] + values[index + 1 :]
        constant = geometric_mean(others)
        worst = max(worst, value / constant, constant / value)
    return worst


@dataclass(frozen=True, slots=True)
class CohesionModel:
    """The two constants, and what each is worth on data it did not see."""

    liquid_constant: float
    liquid_held_out: float
    liquid_n: int
    liquid_span: tuple[float, float]
    polymer_constant: float
    polymer_held_out: float
    polymer_n: int
    polymer_span: tuple[float, float]
    #: Same as the liquid fit, but without the hydrogen-bonding gate. Kept so
    #: the gate's value is a measured number rather than a claim.
    ungated_constant: float
    ungated_held_out: float

    def constant_for(self, regime: str) -> float:
        return self.liquid_constant if regime == "liquid" else self.polymer_constant

    def carried_ratio(self, regime: str) -> float:
        """The multiplicative bar actually carried, which is not always the fit's."""
        if regime == "liquid":
            return self.liquid_held_out
        return max(self.polymer_held_out, POLYMER_CARRIED_RATIO)


@functools.lru_cache(maxsize=1)
def cohesion_model() -> CohesionModel:
    """Fit both constants and measure how far each transfers.

    Deliberately computed from the tables rather than written down, so that a
    number quoted in the docstring or in a test cannot drift away from the
    number the module actually uses.
    """
    gated = [
        p.ratio
        for p in LIQUID_REFERENCES
        if p.hydrogen_fraction <= HYDROGEN_BOND_FRACTION_LIMIT
    ]
    ungated = [p.ratio for p in LIQUID_REFERENCES]
    polymers = [p.ratio for p in POLYMER_REFERENCES]
    return CohesionModel(
        liquid_constant=geometric_mean(gated),
        liquid_held_out=worst_leave_one_out(gated),
        liquid_n=len(gated),
        liquid_span=(min(gated), max(gated)),
        polymer_constant=geometric_mean(polymers),
        polymer_held_out=worst_leave_one_out(polymers),
        polymer_n=len(polymers),
        polymer_span=(min(polymers), max(polymers)),
        ungated_constant=geometric_mean(ungated),
        ungated_held_out=worst_leave_one_out(ungated),
    )


# --------------------------------------------------------------------------
# The two identities
# --------------------------------------------------------------------------


def ced_from_hildebrand(delta_pa_root: float, std_pa_root: float) -> tuple[float, float]:
    """``CED = delta^2`` in Pa, with its propagated spread.

    ``sigma_CED = 2 delta sigma`` is exactly half the width of the transformed
    one-sigma interval ``[(delta-s)^2, (delta+s)^2]``, so it is the symmetric
    part of the transform rather than an approximation to it.
    """
    return delta_pa_root**2, 2.0 * delta_pa_root * std_pa_root


def ced_from_hansen(components: list[tuple[float, float]]) -> tuple[float, float]:
    """``CED = dD^2 + dP^2 + dH^2`` in Pa, with its propagated spread.

    Hansen's decomposition is of the cohesive *energy*, so the three squares are
    summed directly. The components are treated as independent, which understates
    the spread when all three come from one compilation entry; the basis string
    says so rather than silently correcting it.
    """
    ced = sum(value**2 for value, _ in components)
    variance = sum((2.0 * value * std) ** 2 for value, std in components)
    return ced, math.sqrt(variance)


def hydrogen_bond_fraction(components: list[tuple[float, float]]) -> float | None:
    """``dH / delta_total``, the gate the liquid branch is refused on, or None.

    Built from the Hansen triple and never from a hydrogen-bond donor count:
    RDKit reports zero donors for water, which would let the single worst case
    in the whole set straight through the gate.

    None when the triple has no magnitude. A ``(0, 0, 0)`` triple says nothing
    about hydrogen bonding, and reading it as "zero per cent" would hand the
    gate its most favourable answer on no evidence - the house rule that
    absence is never a neutral score. It is not hypothetical: an upstream that
    emitted three zeros alongside a real Hildebrand parameter would otherwise
    get water a bulk modulus, which is the one compound the gate exists for.
    """
    total = math.sqrt(sum(value**2 for value, _ in components))
    if total <= 0.0:
        return None
    # Positional, and the position is fixed by _HANSEN_PROPERTIES, which is the
    # order _hansen() builds the list in.
    hydrogen = components[_HANSEN_PROPERTIES.index("hansen_hydrogen_bonding")][0]
    return hydrogen / total


def fluorine_fraction(smiles: str) -> float | None:
    """Fraction of heavy atoms that are fluorine, or None if unparseable."""
    from formulate import chem

    mol = chem.mol_from_smiles(smiles)
    if mol is None:
        return None
    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy:
        return None
    return sum(1 for a in heavy if a.GetAtomicNum() == 9) / len(heavy)


def bulk_modulus_from_ced(ced_pa: float, constant: float) -> float:
    """``B = n CED``. Both in pascals, because J/m^3 is a pascal."""
    return constant * ced_pa


def _ptfe_ratio() -> float:
    """PTFE's measured ``B/CED``, computed rather than quoted.

    It appears in two refusal messages and in the module docstring, and a
    number written out three times is a number that eventually disagrees with
    itself.
    """
    _, delta, bulk, _ = EXCLUDED_POLYMERS[0]
    return bulk * _PA_PER_GPA / (delta * PA_ROOT_PER_MPA_ROOT) ** 2


# --------------------------------------------------------------------------
# The expert
# --------------------------------------------------------------------------


class CohesionExpert(Expert):
    """Cohesive energy density, and the bulk modulus that follows from it.

    Depends on an upstream solubility parameter and produces nothing without
    one. That is the whole design: ``CED = delta^2`` is nearly free once a
    Hildebrand parameter exists, and inventing a group-contribution delta here
    when none arrives would duplicate a better-grounded expert with a worse
    method.
    """

    id = "cohesion"
    version = "1"
    method = (
        "CED = delta^2 (Hansen 2007, ch. 1); bulk modulus B = n CED from the Mie "
        "pair-potential relation, with n measured separately for liquids (2.81) "
        "and polymers (9.11)"
    )
    family = PropertyFamily.MECHANICAL
    supported_classes = frozenset(
        {MaterialClass.MOLECULE, MaterialClass.POLYMER, MaterialClass.MIXTURE}
    )
    supported_properties = frozenset({"cohesive_energy_density", "bulk_modulus"})
    dependencies = frozenset(
        {
            "hildebrand_solubility_parameter",
            *_HANSEN_PROPERTIES,
            "glass_transition_temperature",
        }
    )

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        if self.is_available():
            return ""
        return (
            "RDKit is required to read the structure the fluorine and phase gates "
            "are evaluated on"
        )

    def _software(self) -> SoftwareEnvironment:
        from formulate import chem

        return SoftwareEnvironment.capture(rdkit=chem.rdkit_version())

    # -- structure ---------------------------------------------------------

    def _regime(self, candidate: Candidate) -> str | None:
        """``liquid``, ``polymer``, or None when the two are mixed.

        A polymer is not its monomer and a mixture is not its major component,
        so a formulation that is part small molecule and part polymer gets no
        answer: neither constant was validated on one, and picking the majority
        component's regime is a factor of three.
        """
        if candidate.material_class is MaterialClass.MOLECULE:
            return "liquid"
        if candidate.material_class is MaterialClass.POLYMER:
            return "polymer"
        mixture = candidate.mixture
        if mixture is None or not mixture.components:
            return None
        if all(c.molecule is not None for c in mixture.components):
            return "liquid"
        if all(c.polymer is not None for c in mixture.components):
            return "polymer"
        return None

    def _structures(self, candidate: Candidate) -> list[str]:
        """Every SMILES the structural gates should look at."""
        if candidate.polymer is not None:
            return [
                m.smiles
                for m in candidate.polymer.monomers
                if m.role is not MonomerRole.END_GROUP
            ]
        return candidate.all_smiles()

    def _fluorine_rich(self, candidate: Candidate) -> tuple[str, float] | None:
        for smiles in self._structures(candidate):
            fraction = fluorine_fraction(smiles)
            if fraction is not None and fraction > FLUORINE_FRACTION_LIMIT:
                return smiles, fraction
        return None

    # -- domain ------------------------------------------------------------

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        model = cohesion_model()
        basis = (
            f"{model.liquid_n} solvents and {model.polymer_n} polymers at 298 K; the "
            "cohesive energy density is an identity and inherits its domain from "
            "whichever expert supplied the solubility parameter"
        )

        structures = self._structures(candidate)
        if not structures:
            return ApplicabilityDomain.outside(
                "candidate carries no parseable structure for its class", basis=basis
            )
        from formulate import chem

        if any(chem.mol_from_smiles(s) is None for s in structures):
            return ApplicabilityDomain.outside(
                "at least one structure in this candidate does not parse", basis=basis
            )

        if self._regime(candidate) is None:
            return ApplicabilityDomain.outside(
                "this formulation mixes small molecules and polymers; the bulk modulus "
                "ratio differs threefold between the two and neither constant was "
                "validated on a blend of them",
                basis=basis,
            )

        warnings: list[str] = []
        score = 1.0

        fluorine = self._fluorine_rich(candidate)
        if fluorine is not None:
            warnings.append(
                f"{fluorine[1]:.0%} of the heavy atoms in {fluorine[0]} are fluorine; PTFE "
                f"measures {_ptfe_ratio():.1f} against a polymer constant of "
                f"{model.polymer_constant:.2f}, so no bulk modulus is produced for this "
                "structure. The cohesive energy density is unaffected - it is an identity "
                "on the solubility parameter"
            )
            score = min(score, 0.5)

        # Temperature is deliberately *not* assessed here. A candidate carries the
        # conditions it was proposed at, and the engine evaluates an expert at the
        # conditions its requirements ask for, which are not the same object: see
        # ``_conditions_for`` in evaluation/engine.py, where a requirement stating
        # its own conditions overrides the candidate's. Reading the candidate here
        # produced a 340 K bulk modulus carrying a domain score of 1.0 and no
        # warning at all, because the candidate still said 298 K. The check now
        # lives in ``_temperature_domain`` and runs on the temperature the answer
        # is actually computed at.

        return ApplicabilityDomain(
            score=score,
            in_domain=score > 0.3,
            warnings=tuple(warnings),
            basis=basis,
        )

    def _temperature_domain(self, temperature: float) -> ApplicabilityDomain | None:
        """Domain penalty for answering away from where the ratio was measured.

        Separate from :meth:`assess_domain` for two reasons. Only the request
        knows the temperature the answer is for. And this window belongs to the
        bulk modulus alone: the cohesive energy density is an identity on a
        parameter that already arrived at the requested temperature, so
        flagging it here would be a domain warning about a different property,
        which costs exactly what section 12 measures.
        """
        if BULK_MODULUS_T_INNER[0] <= temperature <= BULK_MODULUS_T_INNER[1]:
            return None
        return ApplicabilityDomain(
            score=0.4,
            warnings=(
                (
                    f"every bulk-modulus validation point is at 298 K and this answer "
                    f"is for {temperature:.0f} K; the ratio runs as 1/(alpha T), so the "
                    "error is systematic with distance from there"
                ),
            ),
        )

    # -- dependencies ------------------------------------------------------

    def _hildebrand(self, request: PredictionRequest) -> tuple[float, float] | None:
        pred = request.dependency("hildebrand_solubility_parameter")
        if pred is None or pred.quantity is None:
            return None
        value = pred.quantity.to("Pa^0.5").value
        std = pred.uncertainty.converted(pred.quantity.unit, "Pa^0.5").std
        return value, TABULATED_PARAMETER_FLOOR_PA_ROOT if not std else std

    def _hansen(self, request: PredictionRequest) -> list[tuple[float, float]] | None:
        """The complete triple, or None.

        Two of three is not accepted. Summing two squares understates the
        cohesive energy by construction, and the shortfall would be invisible:
        the answer still looks like a cohesive energy density.
        """
        out: list[tuple[float, float]] = []
        for prop in _HANSEN_PROPERTIES:
            pred = request.dependency(prop)
            if pred is None or pred.quantity is None:
                return None
            value = pred.quantity.to("Pa^0.5").value
            std = pred.uncertainty.converted(pred.quantity.unit, "Pa^0.5").std
            out.append((value, TABULATED_PARAMETER_FLOOR_PA_ROOT if not std else std))
        return out

    # -- prediction --------------------------------------------------------

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        candidate = request.candidate

        structures = self._structures(candidate)
        if not structures:
            return Prediction.unsupported(
                prop, self.id, "candidate carries no parseable structure for its class"
            )

        regime = self._regime(candidate)
        if regime is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "this formulation is part small molecule and part polymer; the bulk "
                "modulus ratio differs by a factor of three between the two regimes and "
                "neither was validated on a blend, so answering with the major "
                "component's regime would be a guess wearing a number",
            )

        temperature = request.conditions.temperature_k
        if temperature is None:
            return Prediction.unsupported(
                prop,
                self.id,
                f"{prop} is condition dependent and no temperature was given; the answer "
                "is a different number at every temperature",
            )

        # A gas has no cohesive energy density and no bulk modulus. Only checked
        # for a molecule, because that is the only class for which a compiled
        # melting and boiling point exists - and for a mixture a pure-component
        # boiling point does not fix the formulation's phase anyway, which the
        # mixture note says out loud rather than quietly skipping the check.
        if candidate.molecule is not None:
            from .measured import not_liquid_at

            wrong_phase = not_liquid_at(candidate.molecule.smiles, temperature)
            if wrong_phase:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"{wrong_phase}. An upstream Hildebrand parameter is a liquid-phase "
                    "quantity - vaporisation energy over liquid molar volume - so "
                    "squaring it here would report the cohesion of a phase this "
                    "substance does not occupy. Ask again inside the liquid range, or "
                    "supply a solid-state cohesive energy from a lattice-energy "
                    "calculation; this expert will not extrapolate one",
                )

        route = self._cohesive_energy(request, regime, temperature)
        if isinstance(route, str):
            return Prediction.unsupported(prop, self.id, route)
        ced, ced_std, notes, params = route

        if prop == "cohesive_energy_density":
            route_relative = float(params["route_relative"])  # type: ignore[arg-type]
            disagreement = float(params["route_disagreement"])  # type: ignore[arg-type]
            return self._make(
                prop,
                ced,
                "Pa",
                request,
                domain,
                std=ced_std,
                kind=UncertaintyKind.COMBINED,
                basis=(
                    "CED = delta^2 is an identity, so the spread is the upstream spread "
                    "transformed: sigma = 2 delta sigma_delta, which is exactly half the "
                    "width of [(delta-s)^2, (delta+s)^2] rather than an approximation to "
                    "it"
                    + (
                        f", combined in quadrature with a route term of "
                        f"{route_relative:.1%} of the value for assembling the total from "
                        "Hansen components. The three components are treated as "
                        "independent, which understates the spread when all three come "
                        "from one compilation entry"
                        if route_relative
                        else ""
                    )
                    + (
                        f". That propagation is not what is carried here: the Hansen "
                        f"triple disagrees with the Hildebrand parameter by "
                        f"{disagreement:.0%} of the value, more than the two routes' "
                        "combined spread, and the wider disagreement is carried instead "
                        "so the interval spans both answers"
                        if disagreement
                        else ""
                    )
                ),
                notes=tuple(notes),
                **params,
            )

        return self._bulk_modulus(
            request, domain, regime, temperature, ced, ced_std, notes, params
        )

    def _cohesive_energy(
        self, request: PredictionRequest, regime: str, temperature: float
    ) -> tuple[float, float, list[str], dict[str, object]] | str:
        """CED in Pa with its spread, or a refusal reason."""
        hildebrand = self._hildebrand(request)
        hansen = self._hansen(request)

        hansen_in_window = (
            HANSEN_ROUTE_T_WINDOW[0] <= temperature <= HANSEN_ROUTE_T_WINDOW[1]
        )
        if hansen is not None and not hansen_in_window and hildebrand is None:
            return (
                f"the only cohesion available is a Hansen triple, and the compilations are "
                f"tabulated at 25 C while this request is at {temperature:.0f} K. A "
                "cohesive energy density assembled from them here would be a 298 K number "
                "wearing a different label"
            )

        if hildebrand is None and hansen is None:
            return (
                "needs either hildebrand_solubility_parameter or a complete Hansen triple "
                "(hansen_dispersion, hansen_polar, hansen_hydrogen_bonding) from an "
                "upstream expert, and neither was available. No group-contribution "
                "solubility parameter is computed here as a substitute"
            )

        route_relative = (
            HANSEN_ROUTE_CED_RELATIVE_POLYMER
            if regime == "polymer"
            else HANSEN_ROUTE_CED_RELATIVE_MOLECULE
        )
        notes: list[str] = []
        #: Fraction of the value the two routes disagree by, when that
        #: disagreement is what ends up being carried. Zero the rest of the time.
        disagreement = 0.0

        if request.candidate.material_class is MaterialClass.MIXTURE:
            notes.append(
                "this is a formulation and the solubility parameter arrived already "
                "mixed. CED is quadratic in delta, so a linear mixing rule upstream is "
                "not linear here and a strongly non-ideal blend is wrong in a way this "
                "module cannot see. The phase of the formulation is not checked either: "
                "a pure-component boiling point does not fix a mixture's"
            )

        if hildebrand is not None:
            ced, propagated = ced_from_hildebrand(*hildebrand)
            route = "hildebrand"
            # The identity is exact and the upstream parameter is already at the
            # requested temperature, so this route carries propagation only.
            route_term = 0.0
            notes.append(
                f"CED = delta^2 from an upstream Hildebrand parameter of "
                f"{hildebrand[0] / PA_ROOT_PER_MPA_ROOT:.2f} MPa^0.5"
            )
        else:
            assert hansen is not None
            ced, propagated = ced_from_hansen(hansen)
            route = "hansen"
            route_term = route_relative * ced
            notes.append(
                "CED = dD^2 + dP^2 + dH^2, Hansen's own decomposition of the cohesive "
                "energy, summed as energies rather than through a total parameter"
            )
            if regime == "polymer":
                notes.append(
                    "a polymer's Hansen triple is a solubility-sphere centre fitted to "
                    "which solvents dissolve it, not a vaporisation measurement; measured "
                    "against five published spheres those centres give totals 1.04 to "
                    "1.34 times the group-contribution parameter, so this value may be up "
                    "to 1.8x high and the bar says so"
                )

        std = math.hypot(propagated, route_term)

        # Both routes present and disagreeing is information, not noise. The
        # interval is widened to span both rather than quietly reporting one.
        if hildebrand is not None and hansen is not None and hansen_in_window:
            other, other_std = ced_from_hansen(hansen)
            gap = abs(ced - other)
            combined = math.hypot(std, math.hypot(other_std, route_relative * other))
            if gap > combined:
                notes.append(
                    f"the Hildebrand route gives {ced:.4g} Pa and the Hansen triple "
                    f"{other:.4g} Pa, a disagreement larger than their combined spread; "
                    "the interval is widened to span both rather than choosing silently"
                )
                if gap > std:
                    # The carried spread stops being the propagated one here, and
                    # the basis string has to stop saying that it is.
                    disagreement = gap / ced if ced else 0.0
                std = max(std, gap)
                route = "hildebrand, disputed by hansen"
            else:
                route = "hildebrand, hansen agrees within the bar"

        # Refused here rather than in the bulk-modulus branch alone, because a
        # cohesive energy density of zero is not a small number - it is the
        # statement that nothing holds the phase together, and it arrives with a
        # spread of exactly zero attached (sigma = 2 delta sigma_delta, and delta
        # is zero), so it would be reported as a *certain* zero. A certain answer
        # is never what an identity on an upstream estimate produces.
        if ced <= 0.0:
            return (
                f"the upstream cohesion came out at {ced:.4g} Pa, which is not a "
                "condensed phase - nothing holds it together - and the identity "
                "CED = delta^2 hands that back with a spread of exactly zero, a "
                "certainty this expert has no way to earn. Check the expert that "
                "produced hildebrand_solubility_parameter or the Hansen triple; it "
                "reported a solubility parameter of zero"
            )

        params: dict[str, object] = {
            "route": route,
            "route_relative": route_relative if route_term else 0.0,
            "route_disagreement": disagreement,
            "regime": regime,
        }
        return ced, std, notes, params

    def _bulk_modulus(
        self,
        request: PredictionRequest,
        domain: ApplicabilityDomain,
        regime: str,
        temperature: float,
        ced: float,
        ced_std: float,
        notes: list[str],
        params: dict[str, object],
    ) -> Prediction:
        prop = "bulk_modulus"
        model = cohesion_model()

        if not (BULK_MODULUS_T_OUTER[0] <= temperature <= BULK_MODULUS_T_OUTER[1]):
            return Prediction.unsupported(
                prop,
                self.id,
                f"every point this ratio was measured on is at 298 K and the request is "
                f"at {temperature:.0f} K. B/CED runs as 1/(alpha T), so extrapolating it "
                f"is a systematic error rather than scatter, and an error bar would not "
                f"rescue it. The supported window is "
                f"{BULK_MODULUS_T_OUTER[0]:.0f} to {BULK_MODULUS_T_OUTER[1]:.0f} K",
            )

        # Backstop. ``_cohesive_energy`` already refuses a non-positive cohesion
        # for both properties, so this cannot fire today; it stays because the
        # invariant it protects - you cannot scale nothing into a stiffness - is
        # cheaper to keep than to rediscover if the route ever changes.
        if ced <= 0.0:  # pragma: no cover
            return Prediction.unsupported(
                prop,
                self.id,
                "the cohesive energy density came out non-positive, which is not a "
                "condensed phase; there is nothing to scale into a bulk modulus",
            )

        conditions_domain = self._temperature_domain(temperature)
        if conditions_domain is not None:
            domain = domain.merged_with(conditions_domain)

        from formulate import chem

        unparseable = [
            s for s in self._structures(request.candidate) if chem.mol_from_smiles(s) is None
        ]
        if unparseable:
            return Prediction.unsupported(
                prop,
                self.id,
                f"{unparseable[0]!r} does not parse, so the fluorine gate cannot be "
                f"evaluated. The gate exists because PTFE misses by "
                f"{_ptfe_ratio() / model.polymer_constant:.1f}x, and a structure that "
                "cannot be read cannot be shown to be past it",
            )

        fluorine = self._fluorine_rich(request.candidate)
        if fluorine is not None:
            return Prediction.unsupported(
                prop,
                self.id,
                f"{fluorine[1]:.0%} of the heavy atoms in {fluorine[0]} are fluorine. PTFE "
                f"measures B/CED = {_ptfe_ratio():.1f} against the polymer constant "
                f"{model.polymer_constant:.2f} - fluorine's low polarisability gives a "
                f"small cohesive energy density at high stiffness - and the miss is "
                f"{_ptfe_ratio() / model.polymer_constant:.1f}x in one direction, so it is "
                "refused rather than carried as a wider bar",
            )

        hansen = self._hansen(request)
        if hansen is None and regime == "liquid":
            return Prediction.unsupported(
                prop,
                self.id,
                "a bulk modulus for a liquid is only produced when the Hansen triple is "
                "available, because the hydrogen-bonding fraction dH/delta is the gate "
                "the liquid branch is validated behind. The hydrogen-bond donor count is "
                "not accepted as a substitute: RDKit reports zero donors for water, which "
                "is the worst case in the set",
            )

        notes = list(notes)
        fraction = None if hansen is None else hydrogen_bond_fraction(hansen)
        if hansen is not None and fraction is None:
            # A triple of zeros is not a measurement of zero hydrogen bonding.
            if regime == "liquid":
                return Prediction.unsupported(
                    prop,
                    self.id,
                    "the Hansen triple that arrived has no magnitude - all three "
                    "components are zero - so the hydrogen-bonding fraction the liquid "
                    "branch is gated on cannot be evaluated. Reading it as zero per cent "
                    "would hand the gate its most favourable answer on no evidence, and "
                    "water is exactly the compound that would slip through. Supply a "
                    "real hansen_dispersion / hansen_polar / hansen_hydrogen_bonding "
                    "triple, or ask only for cohesive_energy_density",
                )
            notes.append(
                "the Hansen triple that arrived is all zeros, so how much of this "
                "polymer's cohesion is hydrogen bonding is unchecked, exactly as if no "
                "triple had arrived at all"
            )
        if fraction is not None:
            if regime == "liquid" and fraction > HYDROGEN_BOND_FRACTION_LIMIT:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"{fraction:.0%} of this liquid's cohesive energy is hydrogen bonding, "
                    f"above the {HYDROGEN_BOND_FRACTION_LIMIT:.0%} the liquid branch was "
                    "validated behind. Over the nine compiled liquids past that line B/CED "
                    "collapses to 0.93 for methanol and 0.96 for water against a constant "
                    f"of {model.liquid_constant:.2f}: the cohesive energy density counts "
                    "hydrogen bonds the bulk modulus does not feel, and the error is "
                    "systematic rather than scatter",
                )
            if regime == "liquid":
                notes.append(
                    f"{fraction:.0%} of the cohesive energy is hydrogen bonding, inside "
                    f"the {HYDROGEN_BOND_FRACTION_LIMIT:.0%} this branch was validated "
                    "behind"
                )
            if regime == "polymer" and fraction > HYDROGEN_BOND_FRACTION_LIMIT:
                notes.append(
                    f"{fraction:.0%} of this polymer's cohesive energy is hydrogen "
                    "bonding. The liquid branch refuses past this line; the polymer branch "
                    "does not, because the one hydrogen-bonded polymer that can be checked "
                    f"here - nylon 6,6 - measures 7.1 against {model.polymer_constant:.2f}, "
                    "inside the carried bar. "
                    "That is one point, and it is the whole evidence"
                )

        if hansen is None and regime == "polymer":
            notes.append(
                "no Hansen triple arrived, so how much of this polymer's cohesion is "
                "hydrogen bonding could not be checked. The polymer branch does not "
                "refuse on that - the liquid branch does - but it is unchecked here"
            )

        constant = model.constant_for(regime)
        value = bulk_modulus_from_ced(ced, constant)
        carried = model.carried_ratio(regime)

        if regime == "polymer":
            glass_transition = request.dependency_value("glass_transition_temperature", "K")
            if glass_transition is None:
                # Symmetrical with the missing-Hansen note above: the check that
                # would have been run is named, rather than being skipped in
                # silence and leaving a melt indistinguishable from a glass.
                notes.append(
                    "no glass_transition_temperature arrived, so whether this polymer is "
                    "far enough above its Tg to sit outside the solids and rubbers the "
                    f"constant was measured on is unchecked. PDMS, {FAR_ABOVE_TG_K:.0f} K "
                    f"past that line, measures 4.4 against {model.polymer_constant:.2f} - "
                    "a miss this prediction cannot rule out"
                )
            elif temperature > glass_transition + FAR_ABOVE_TG_K:
                carried = max(carried, FAR_ABOVE_TG_RATIO)
                notes.append(
                    f"this polymer is {temperature - glass_transition:.0f} K above its "
                    "glass transition, outside the window the solids and rubbers behind "
                    "the constant sit in. The two materials measured out there miss in "
                    f"opposite directions - PDMS 4.4 and polyethylene 10.7 against "
                    f"{model.polymer_constant:.2f} - so distance above Tg buys extra "
                    "spread, not a softer answer, and the bar is widened symmetrically "
                    f"to {FAR_ABOVE_TG_RATIO:.1f}x rather than the value being adjusted. "
                    "Which of the two this polymer resembles turns on crystallinity, "
                    "which is not in the property registry"
                )
                domain = domain.merged_with(
                    ApplicabilityDomain(
                        score=0.4,
                        warnings=(
                            (
                                "far above the glass transition, where the two materials "
                                "measured disagree about the sign of the error; a known "
                                "hole"
                            ),
                        ),
                        basis="solid and rubbery polymers near 298 K",
                    )
                )

        relative = math.hypot(carried - 1.0, ced_std / ced if ced else 0.0)
        span_low, span_high = (
            model.liquid_span if regime == "liquid" else model.polymer_span
        )

        notes.append(
            f"B = {constant:.2f} x CED. The constant is the geometric mean over "
            f"{model.liquid_n if regime == 'liquid' else model.polymer_n} "
            f"{'liquids' if regime == 'liquid' else 'polymers'} at 298 K, spanning "
            f"{span_low:.2f} to {span_high:.2f}"
        )
        if regime == "liquid":
            notes.append(
                f"the eight-to-eleven figure quoted for polymers does not apply to a "
                f"liquid: it would give {model.polymer_constant * ced / 1e9:.2f} GPa here "
                f"against {value / 1e9:.2f} GPa, a factor of "
                f"{model.polymer_constant / constant:.1f}"
            )
        else:
            notes.append(
                "the published reference bulk moduli are the weakest numbers behind this: "
                "polystyrene is quoted from 2.4 GPa (isothermal, from PVT data) to 4.2 GPa "
                "(adiabatic, from sound speed). The factor-of-two bar is set by that "
                f"disagreement, not by the model's own {model.polymer_held_out:.2f}x"
            )

        return self._make(
            prop,
            value,
            "Pa",
            request,
            domain,
            std=value * relative,
            kind=UncertaintyKind.COMBINED,
            basis=(
                f"multiplicative: a factor of {carried:.2f} on the {regime} branch "
                f"(worst leave-one-out ratio "
                f"{model.liquid_held_out if regime == 'liquid' else model.polymer_held_out:.2f} "
                f"over {model.liquid_n if regime == 'liquid' else model.polymer_n} measured "
                f"materials"
                + (
                    ", widened to two because published reference bulk moduli for one "
                    "polymer disagree by more than that"
                    if regime == "polymer"
                    else ""
                )
                + f"), combined in quadrature with {ced_std / ced:.1%} propagated from the "
                "cohesive energy density"
            ),
            notes=tuple(notes),
            constant=round(constant, 4),
            **params,
        )
