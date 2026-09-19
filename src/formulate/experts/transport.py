"""Liquid transport of a small molecule: how thick it is, and how fast it moves.

Two properties, one chain of reasoning.

**Shear viscosity** - Orrick and Erbar's group-contribution correlation,

    ln(eta / (rho * M)) = A + B / T

with eta in centipoise, rho in g/cm^3, M in g/mol and T in kelvin, and A and B
summed over seventeen structural groups.  Primary source: C. Orrick and
J. H. Erbar (1974), unpublished, tabulated in Reid, Prausnitz and Poling,
"The Properties of Gases and Liquids", 4th ed., McGraw-Hill (1987), Sec. 9-11.
The textbook tabulation is cited because that is genuinely where the table
lives.

**Self-diffusion** - the Stokes-Einstein relation in its *slip* limit,

    D = k_B T / (4 pi eta r),   r = (3 V_m / (4 pi N_A))^(1/3)

(Sutherland 1905; Einstein 1905).  The 4 pi slip form is used rather than the
6 pi stick form: measured on eighteen liquids here, 4 pi reproduces the data
with a bias of x0.78 while 6 pi lands at x0.52, a systematic factor of two.

Why not a corresponding-states route, which would have reused critical
constants this panel already estimates?  Both candidates were evaluated
numerically on *measured* critical constants before this expert was written,
so the comparison tested the correlations rather than their inputs:

* Letsou-Stiel states its own validity as 0.76 < Tr < 0.98.  At 25 degrees
  Celsius every liquid of interest sits at Tr 0.35-0.64, and the method
  returns glycerol a factor of 900 low, ethylene glycol 16x low, ethanol 2.5x
  low.  The gap is not repairable with an error bar, because at one atmosphere
  a liquid essentially never reaches Tr 0.76 - for toluene that is 450 K,
  well above its 384 K boiling point.  A validity window that is unreachable
  at atmospheric pressure is no use to a formulation tool.
* Przedziecki-Sridhar, the standard low-temperature alternative, returns
  *negative* viscosities for water, the alcohols, glycerol and the glycols.
  The `chemicals` package's own docstring says "This function is not
  recommended."

Sastri-Rao was rejected for a different reason: it puts an estimated boiling
point inside an exponent, where Orrick-Erbar puts an estimated density in a
linear prefactor.  In this pipeline a novel molecule's density arrives from
Rackett with roughly ten per cent uncertainty, and a linear prefactor
propagates that gracefully where an exponent does not.

ACCURACY, measured here rather than quoted.  134 liquids at 25 degrees Celsius
with measured densities and measured viscosities (DIPPR, REFPROP, VDI and
Viswanath-Natarajan correlations served through `thermo`), of which 110 pass
this expert's gates and 24 are refused:

    acyclic, 0-1 oxygen-bearing group   N=80   RMS ln-ratio 0.195 (x1.22)   bias x1.03
    aromatic six-ring                   N=22   RMS ln-ratio 0.242 (x1.27)   bias x0.93
    acyclic, 2+ oxygen-bearing groups   N= 8   RMS ln-ratio 0.734 (x2.08)   bias x1.25

Measured one-sigma coverage against the bars this expert reports is 70, 77 and
88 per cent for those three, against the ~68 per cent a correct estimate
implies.  Self-diffusion, end to end with the viscosity predicted too, lands
at bias x0.80, RMS ln-ratio 0.301 and 76 per cent coverage over seventeen
liquids with known NMR self-diffusion coefficients.

The alkane and n-alcohol homologous series reproduce to within 0-5 per cent
from C5 to C16 and from C1 to C11, which is the check that the transcribed
group table is right rather than merely plausible.

LIMITATIONS, all of them enforced as refusals rather than as warnings:

* No nitrogen, sulfur, fluorine, phosphorus or silicon contribution exists in
  the table, so those atoms would contribute a silent zero.  Measured cost of
  not refusing: aniline x0.13, dimethyl sulfoxide x0.13, acetonitrile x0.29,
  pyridine x0.41, nitrobenzene x0.42, perfluorohexane x9.9.  This excludes
  fluorinated solvents entirely, which is a real coverage gap.
* Saturated rings are systematically wrong rather than imprecise: over ten
  ring compounds the bias is x0.59 and every one but tetralin reads low, worst
  cyclohexanol at x0.13.  Cyclohexane, THF, dioxane, cyclic carbonates and
  lactones are therefore refused - common solvents, deliberately given up.
* Water has no carbon and no matchable group at all: x0.05.
* A SMILES that is not one neutral closed-shell molecule - a dotted mixture
  or salt, an inner salt, a radical - is refused rather than summed.  The
  group counter cannot see a fragment boundary: "CCCCCC.CCCCCC" reads as
  dodecane at x3.9 hexane's viscosity, and "C[CH2]" reads as ethane.

References for the rejected alternatives: Letsou, A. and Stiel, L. I.,
AIChE J. 19 (1973) 409; Przedziecki, J. W. and Sridhar, T., AIChE J. 31
(1985) 333; both as tabulated in Reid, Prausnitz and Poling, Sec. 9-12.
"""

from __future__ import annotations

import functools
import math
from typing import Mapping

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest
from .interfacial import propagate

#: Boltzmann's constant, J/K.
BOLTZMANN = 1.380649e-23
#: Avogadro's number, 1/mol.
AVOGADRO = 6.02214076e23

#: Orrick-Erbar group contributions to A and B, from the Table 9-11
#: tabulation in Reid, Prausnitz and Poling, 4th ed.
#:
#: This table was transcribed rather than machine-read, so it was validated
#: forward against 134 measured viscosities *before* the expert was written.
#: Reproducing heptane at x0.99, octane x1.00, decane x0.99, hexadecane x0.96,
#: 1-butanol x0.98, 1-octanol x1.01, 1-decanol x1.00, methyl acetate x0.99,
#: diethyl ether x1.00, ethylbenzene x0.99 and chlorobenzene x1.01 is not
#: something a misremembered table does by accident.  The halogen rows are the
#: thinnest evidence - chloroform x1.27, carbon tetrachloride x1.41 and
#: 1,2-dichloroethane x0.64 are the widest non-refused residuals - so if any
#: single entry is wrong it is most likely one of those.
_GROUP_CONTRIBUTIONS: dict[str, tuple[float, float]] = {
    "branch_ch": (-0.15, 35.0),      # R2CH-, a methine branch point
    "branch_c": (-1.20, 400.0),      # R3C-, a quaternary branch point
    "double_bond": (0.24, -90.0),    # non-aromatic C=C
    "ring5": (0.10, 32.0),           # five-membered saturated ring
    "ring6": (-0.45, 250.0),         # six-membered saturated ring
    "aromatic_ring": (0.00, 20.0),
    "ortho": (-0.12, 100.0),
    "meta": (0.05, -34.0),
    "para": (-0.01, -5.0),
    "chlorine": (-0.61, 220.0),
    "bromine": (-1.25, 365.0),
    "iodine": (-1.75, 400.0),
    "hydroxyl": (-3.00, 1600.0),
    "ester": (-1.00, 420.0),
    "ether": (-0.38, 140.0),
    "carbonyl": (-0.50, 350.0),
    "carboxyl": (-0.90, 770.0),
}

#: The carbon-count backbone of the correlation: every carbon atom contributes
#: to both constants before any functional group is added.
_CARBON_A0, _CARBON_A1 = -6.95, -0.21
_CARBON_B0, _CARBON_B1 = 275.0, 99.0

#: Elements the table has a contribution for.  Everything else is refused
#: rather than given a silent zero - see the module docstring for the measured
#: cost of not doing this.
_SUPPORTED_ELEMENTS = frozenset({"C", "H", "O", "Cl", "Br", "I"})

#: How many oxygen atoms each oxygen-bearing group accounts for.  The mass
#: balance built from this catches two distinct silent failures: an oxygen no
#: pattern matched (a peroxide, an orphan), and an oxygen counted twice (an
#: anhydride, where the ester pattern matches from both sides of a shared
#: bridging oxygen, and a carbonate, likewise).  Measured: acetic anhydride
#: would read x1.15, dimethyl carbonate x1.30 and diethyl carbonate x1.54 if
#: the over-count were allowed through.
_OXYGENS_PER_GROUP = {
    "hydroxyl": 1,
    "ether": 1,
    "carbonyl": 1,
    "ester": 2,
    "carboxyl": 2,
}

#: One-sigma spread of ln(predicted / measured), per structural bucket,
#: measured at 25 degrees Celsius on measured densities.
#:
#: The brief expected hydrogen bonding to be the axis of failure, following
#: the association treatment in :mod:`formulate.experts.interfacial`.  It is
#: not: Orrick-Erbar contains an explicitly fitted -OH group, so mono-alcohols
#: come out as accurate as alkanes (1-butanol x0.98, 1-octanol x1.01).  The
#: axis the data actually picks out is how many oxygen-bearing groups a
#: molecule carries, and separately whether it is aromatic.
#:
#: Following the principle stated in ``interfacial.py`` - choose the widening
#: so one-sigma coverage lands near the 68 per cent a correct estimate
#: implies, erring wide rather than narrow - these give measured coverage of
#: 70 per cent (N=80), 77 per cent (N=22) and 88 per cent (N=8).  The
#: polyoxygenated figure is wider than 68 per cent because with eight
#: compounds there is no finer choice: 0.60 lands at 62 per cent and 0.70 at
#: 88 per cent, and the wide side is the honest one.
_BUCKET_LOG_SIGMA = {
    "acyclic": 0.20,
    "aromatic": 0.22,
    "polyoxygenated": 0.70,
}

#: Extra ln-space uncertainty per kelvin away from the calibration
#: temperature.  The correlation was fitted near ambient and the B/T form is
#: Arrhenius, so the ratio drifts as temperature rises.
#:
#: The drift is *not* one-directional, which an earlier reading of six
#: compounds suggested it was.  Re-measured over the seventy-two reference
#: liquids with fifty kelvin or more of liquid range above ambient,
#: d(ln ratio)/dT runs from -0.0096 per kelvin (diethylene glycol) to +0.0067
#: (2-butanol), median magnitude 0.0013; the alcohols and acids drift the
#: opposite way from the alkanes and ketones.  So this term cannot be an
#: extreme, only a typical, and it is justified by what it delivers rather
#: than by a direction: measured one-sigma coverage against the bars this
#: expert reports is 75 per cent at the calibration temperature, then 83, 94,
#: 95, 98 and 97 per cent at 25 to 125 kelvin above it.  It is already erring
#: wide out there because the structural bucket dominates the quadrature, and
#: raising this constant to the worst compound's 0.0096 would push a
#: 125 kelvin extrapolation to a factor of four and say nothing at all.
_TEMPERATURE_LOG_SIGMA_PER_K = 0.002

#: The correlation's calibration temperature, and the temperature every number
#: in the module docstring was measured at.
_CALIBRATION_TEMPERATURE_K = 298.15

#: Beyond this every prediction carries a note saying it is extrapolated.  At
#: 150 degrees Celsius, on measured densities at that temperature, hexadecane
#: reads x0.78 and ethylene glycol x0.62; the per-kelvin term above covers
#: both, but a user should still be told.  The note is attached in
#: :meth:`LiquidTransportExpert._predict_one` rather than in ``assess_domain``,
#: because a domain is assessed for a candidate and this is a property of the
#: request: the same molecule is in domain at 25 degrees and extrapolated at
#: 150.
_TEMPERATURE_WARNING_K = 373.15

#: One-sigma spread of ln(predicted / measured) for slip Stokes-Einstein,
#: measured on eighteen liquids at 25 degrees Celsius with measured
#: viscosities and measured molar volumes.
#:
#: The residuals are tightly clustered (standard deviation 0.124 in ln) around
#: a *bias* of x0.78 - the method reads systematically 22 per cent low.  The
#: bias is deliberately carried rather than corrected away, so the bar has to
#: cover it: 0.278 is the raw RMS but gives only 39 per cent coverage because
#: the distribution is not centred on the prediction, and 0.35 is the smallest
#: value reaching that (measured 67 per cent against the relation alone on
#: measured viscosities, and 76 per cent end to end over seventeen liquids
#: with the viscosity predicted too).
_STOKES_EINSTEIN_LOG_SIGMA = 0.35

#: Above this viscosity self-diffusion is refused.  Stokes-Einstein was
#: checked here from 0.22 to 16.8 mPa*s, over which the ratio stays inside
#: x0.66-x1.01 with no viscosity trend.  The one point beyond, glycerol at
#: about 1000 mPa*s, falls to x0.42 - the onset of the fractional
#: Stokes-Einstein regime, where diffusion decouples from viscosity as a
#: liquid approaches its glass transition.  The principled gate would be a
#: reduced temperature T/Tg, but glass_transition_temperature exists in this
#: registry only as a polymer property and a polymer property must not be read
#: for a molecule.  So the gate is empirical and sits at the top of the
#: checked range: a liquid at 40 mPa*s will be refused that might have been
#: fine.  It reads the *predicted* viscosity, which is the only one this
#: expert has, so ethylene glycol - 16.8 mPa*s measured but 24 predicted -
#: is refused; the shipped self-diffusion range is therefore 0.22 to
#: 7.3 mPa*s.

_STOKES_EINSTEIN_MAX_VISCOSITY_PAS = 20e-3

#: Molar mass beyond which the self-diffusion residual is extrapolating.  The
#: eighteen-liquid set spans 32 to 154 g/mol, and within it the residual
#: correlates with molar mass at Pearson -0.50, drifting from x1.01 for
#: ethanol to x0.69 for decane.  An inverse-design generator routinely
#: proposes heavier molecules than that, where D will read increasingly low.
_DIFFUSION_MASS_CEILING = 155.0

_DEPENDENCY_UNITS = {
    "liquid_density": "g/cm^3",
    "molar_mass": "g/mol",
    "normal_boiling_point": "K",
    "molar_volume_liquid": "m^3/mol",
}

#: A molecule needs a density and a molar mass for the correlation itself, and
#: a boiling point to establish that there is a liquid at all.  Self-diffusion
#: needs a molar volume on top, for the hydrodynamic radius.  Note that
#: shear_viscosity is deliberately *not* a dependency of
#: self_diffusion_coefficient: an expert that both supplies and depends on the
#: same property makes ``ExpertRegistry.resolution_order`` raise on a cyclic
#: dependency and kills the whole run.  The viscosity is therefore recomputed
#: internally, from the same density this expert was handed, so that one
#: perturbation of that density moves both the correlation and the propagated
#: bar instead of the bar being taken twice from two nominally independent
#: inputs.  The radius does *not* come from that density - it comes from the
#: separately resolved molar volume, which is why :meth:`_diffusion` has to
#: cross-check the two against each other rather than assuming they agree.
_NEEDED: dict[str, tuple[str, ...]] = {
    "shear_viscosity": ("liquid_density", "molar_mass", "normal_boiling_point"),
    "self_diffusion_coefficient": (
        "liquid_density", "molar_mass", "normal_boiling_point", "molar_volume_liquid",
    ),
}

_SMARTS = {
    "carboxyl": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2H0][#6]",
    "hydroxyl": "[OX2H1]",
    "carbonyl": "[CX3]=[OX1]",
    "ether": "[OX2H0]",
}


# -- structure ---------------------------------------------------------------


@functools.lru_cache(maxsize=4096)
def group_counts(smiles: str) -> dict[str, int] | None:
    """Orrick-Erbar group counts for a structure, or ``None`` if unparseable.

    Two keys are not group contributions but bookkeeping: ``carbon`` is the
    carbon count that drives the correlation's backbone term, and
    ``unaccounted_oxygen`` is the oxygen mass-balance residual that the
    refusal gate reads.
    """
    from rdkit import Chem

    from formulate import chem

    mol = chem.mol_from_smiles(smiles)
    if mol is None:
        return None

    counts = {key: 0 for key in _GROUP_CONTRIBUTIONS}
    counts["carbon"] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
    for name, symbol in (("chlorine", "Cl"), ("bromine", "Br"), ("iodine", "I")):
        counts[name] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == symbol)

    patterns = {k: Chem.MolFromSmarts(v) for k, v in _SMARTS.items()}
    # Order matters.  A carboxylic acid contains a hydroxyl and a carbonyl; an
    # ester contains a carbonyl and an ether oxygen.  Claiming the composite
    # groups first and then excluding their atoms is what stops acetic acid
    # being charged for -COOH *and* -OH, which on its own put it ten times too
    # thick before this was fixed.
    claimed_carbons: set[int] = set()
    claimed_oxygens: set[int] = set()
    for key in ("carboxyl", "ester"):
        for match in mol.GetSubstructMatches(patterns[key]):
            counts[key] += 1
            claimed_carbons.add(match[0])
            claimed_oxygens.update(match[1:3])
    for match in mol.GetSubstructMatches(patterns["hydroxyl"]):
        if match[0] not in claimed_oxygens:
            counts["hydroxyl"] += 1
            claimed_oxygens.add(match[0])
    for match in mol.GetSubstructMatches(patterns["carbonyl"]):
        if match[0] not in claimed_carbons:
            counts["carbonyl"] += 1
            claimed_carbons.add(match[0])
            claimed_oxygens.add(match[1])
    for match in mol.GetSubstructMatches(patterns["ether"]):
        if match[0] not in claimed_oxygens:
            counts["ether"] += 1
            claimed_oxygens.add(match[0])

    oxygens = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "O")
    accounted = sum(counts[k] * n for k, n in _OXYGENS_PER_GROUP.items())
    counts["unaccounted_oxygen"] = oxygens - accounted

    # Branch points, counted on the heavy-atom skeleton.  Counting only carbon
    # neighbours instead was tried and is a wash (RMS ln-ratio 0.192 against
    # 0.195 over the same 83 compounds), so the simpler reading is kept.
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C" or atom.GetIsAromatic():
            continue
        if atom.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        hydrogens, degree = atom.GetTotalNumHs(), atom.GetDegree()
        if hydrogens == 1 and degree == 3:
            counts["branch_ch"] += 1
        elif hydrogens == 0 and degree == 4:
            counts["branch_c"] += 1

    counts["double_bond"] = sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
        and not bond.GetIsAromatic()
        and bond.GetBeginAtom().GetSymbol() == "C"
        and bond.GetEndAtom().GetSymbol() == "C"
    )

    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if all(a.GetIsAromatic() for a in atoms):
            counts["aromatic_ring"] += 1
        elif len(ring) == 5:
            counts["ring5"] += 1
        elif len(ring) == 6:
            counts["ring6"] += 1

    # The ortho/meta/para terms apply to a *di*substituted benzene ring, which
    # is what the table specifies.  Without them o-xylene came out at x0.84;
    # with them it is x1.04, m-xylene x1.01 and p-xylene x1.00.
    for ring in ring_info.AtomRings():
        if len(ring) != 6 or not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        substituted = [
            i
            for i in ring
            if any(n.GetIdx() not in ring for n in mol.GetAtomWithIdx(i).GetNeighbors())
        ]
        if len(substituted) != 2:
            continue
        position = {atom: index for index, atom in enumerate(ring)}
        separation = abs(position[substituted[0]] - position[substituted[1]])
        separation = min(separation, 6 - separation)
        counts["ortho" if separation == 1 else "meta" if separation == 2 else "para"] += 1

    return counts


@functools.lru_cache(maxsize=4096)
def structural_refusal(smiles: str) -> str | None:
    """Why Orrick-Erbar cannot be applied to this structure, or ``None``.

    Every gate here is a case where the arithmetic would complete and return a
    confident number that is wrong, which is the failure mode a
    group-contribution method has: a group the table does not contain
    contributes zero and the molecule still looks well formed.
    """
    from rdkit import Chem

    from formulate import chem

    mol = chem.mol_from_smiles(smiles)
    if mol is None:
        return "the candidate carries no parseable molecule"

    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    unsupported = sorted(elements - _SUPPORTED_ELEMENTS)
    if unsupported:
        return (
            f"Orrick-Erbar has no group contribution for {', '.join(unsupported)}, so "
            "those atoms would contribute a silent zero rather than an error; measured "
            "on this gap the method reads aniline 7.6x thin, dimethyl sulfoxide 7.5x "
            "thin and perfluorohexane 9.9x thick. Fluorinated solvents are excluded "
            "entirely, which is a real coverage gap rather than a technicality"
        )

    # A dotted SMILES is not one molecule.  Nothing downstream notices: the
    # group counter happily sums two fragments into one pseudo-molecule, and
    # "CCCCCC.CCCCCC" comes back as 1.18 mPa*s - dodecane's viscosity for a
    # pair of hexanes - while "CCO.CCO" comes back as 32 mPa*s.  A mixture is
    # not its major component and a salt is not its anion, so this is refused
    # here rather than answered, the same way :mod:`formulate.experts.conformation`
    # and :mod:`formulate.experts.semiempirical` refuse it.
    if len(Chem.GetMolFrags(mol)) != 1:
        return (
            "the SMILES is more than one disconnected fragment, and the group count "
            "would silently add them together - two hexanes written as one SMILES "
            "read as dodecane. A salt, a hydrate or a mixture needs its components "
            "predicted separately and combined by a mixture expert"
        )

    # Net charge is not enough.  An inner salt sums to zero and is still ionic,
    # which is the distinction :mod:`formulate.experts.flammability` already
    # draws for the same reason, so the test is per atom.
    if any(atom.GetFormalCharge() for atom in mol.GetAtoms()):
        return (
            "the correlation was fitted to neutral liquids; an ionic species has a "
            "Coulombic contribution to its viscosity that no neutral group table sees, "
            "and a zwitterion sums to zero charge while still being one"
        )

    # A radical has an unfilled valence, so the group table is counting a
    # structure that is not there: "C[CH2]" returns 0.054 mPa*s as though it
    # were ethane.  Radicals are also not bulk liquids anyone formulates with.
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        return (
            "the molecule carries an unpaired electron; the group table was fitted to "
            "closed-shell liquids and a radical has no bulk liquid phase to have a "
            "viscosity. If a valence was left open by accident, complete it"
        )

    counts = group_counts(smiles)
    if counts is None:  # pragma: no cover - mol parsed above
        return "the candidate carries no parseable molecule"

    if counts["carbon"] == 0:
        return (
            "the correlation is built on a carbon count and this molecule has no "
            "carbon, so there is nothing for it to sum over; water predicts "
            "0.04 mPa*s against a measured 0.89"
        )

    if counts["unaccounted_oxygen"] > 0:
        return (
            f"{counts['unaccounted_oxygen']} oxygen atom(s) matched no group in the "
            "table and would contribute nothing at all; a peroxide, an orphan oxygen "
            "or an arrangement the patterns do not recognise cannot be costed"
        )
    if counts["unaccounted_oxygen"] < 0:
        return (
            "the matched groups account for more oxygen than the molecule contains, so "
            "an oxygen is being charged twice - this is what an anhydride or a "
            "carbonate does to the ester pattern, and it reads 15 to 54 per cent thick"
        )

    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        if bond.GetBondType() == Chem.BondType.TRIPLE:
            return "the table has no triple-bond contribution, so an alkyne costs nothing"
        if begin.GetSymbol() == "O" and end.GetSymbol() == "O":
            return (
                "an oxygen-oxygen bond is read by the ether pattern as two independent "
                "ether oxygens, which balances but is not what a peroxide is"
            )

    if counts["ring5"] or counts["ring6"]:
        return (
            "the saturated-ring contributions are systematically wrong rather than "
            "imprecise here: over ten ring compounds the bias is 0.59x and all but one "
            "read low, worst cyclohexanol at 0.13x - a 7.7-fold error. Refusing gives "
            "up cyclohexane, THF, dioxane, cyclic carbonates and lactones, which are "
            "common solvents, but a one-directional miss is a wrong model and no error "
            "bar rescues it"
        )

    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if not all(a.GetIsAromatic() for a in atoms):
            return (
                "this ring is neither aromatic nor a five- or six-membered saturated "
                "ring, and the table has no contribution for it"
            )
        if len(ring) != 6 or any(a.GetSymbol() != "C" for a in atoms):
            return (
                "the aromatic-ring contribution was fitted to benzene rings; a "
                "heteroaromatic or non-six-membered aromatic ring is a different "
                "electronic object and is not covered"
            )

    return None


@functools.lru_cache(maxsize=4096)
def uncertainty_bucket(smiles: str) -> str:
    """Which measured error bucket this structure falls in.

    A molecule carrying two or more oxygen-bearing groups takes the wide
    bucket even when it is also aromatic.  No compound in the validation set
    exercises that combination, so the rule assigns the wider of the two bars
    rather than pretending the aromatic figure has been checked there.
    """
    counts = group_counts(smiles) or {}
    oxygen_groups = sum(counts.get(k, 0) for k in _OXYGENS_PER_GROUP)
    if oxygen_groups >= 2:
        return "polyoxygenated"
    return "aromatic" if counts.get("aromatic_ring", 0) else "acyclic"


# -- correlations ------------------------------------------------------------


def orrick_erbar_constants(counts: Mapping[str, int]) -> tuple[float, float]:
    """The A and B of ``ln(eta / (rho M)) = A + B / T`` for a group count."""
    carbon = counts["carbon"]
    a = _CARBON_A0 + _CARBON_A1 * carbon
    b = _CARBON_B0 + _CARBON_B1 * carbon
    for group, (group_a, group_b) in _GROUP_CONTRIBUTIONS.items():
        n = counts.get(group, 0)
        a += group_a * n
        b += group_b * n
    return a, b


def orrick_erbar_viscosity(
    counts: Mapping[str, int],
    density_g_cm3: float,
    molar_mass: float,
    temperature: float,
) -> float:
    """Liquid shear viscosity in centipoise.

    Centipoise is the correlation's own unit and is returned as such; the
    registry converts it to the canonical Pa*s.  Hand-scaling here would be
    one more place for a factor of a thousand to hide.
    """
    a, b = orrick_erbar_constants(counts)
    return density_g_cm3 * molar_mass * math.exp(a + b / temperature)


def hydrodynamic_radius(molar_volume: float) -> float:
    """Radius in metres of a sphere of the per-molecule liquid volume."""
    return (3.0 * molar_volume / (4.0 * math.pi * AVOGADRO)) ** (1.0 / 3.0)


def stokes_einstein_diffusion(
    temperature: float, viscosity_pa_s: float, radius: float
) -> float:
    """Self-diffusion coefficient in m^2/s, slip boundary condition.

    Stokes-Einstein is derived for a large sphere moving through a structureless
    continuum.  A small molecule diffusing through a liquid of its own size
    violates that outright, which is why the stick prefactor 6 pi is replaced
    by the slip limit 4 pi - and why even then the result carries a bias.
    """
    return BOLTZMANN * temperature / (4.0 * math.pi * viscosity_pa_s * radius)


class LiquidTransportExpert(Expert):
    """How thick a solvent is, and how fast a molecule of it moves."""

    id = "liquid_transport"
    version = "1"
    method = (
        "Orrick-Erbar group contributions for liquid viscosity (Reid, Prausnitz and "
        "Poling, 4th ed., Table 9-11) with slip-limit Stokes-Einstein self-diffusion"
    )
    family = PropertyFamily.INTERFACIAL
    supported_classes = frozenset({MaterialClass.MOLECULE})
    supported_properties = frozenset(_NEEDED)
    dependencies = frozenset(_DEPENDENCY_UNITS)

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "RDKit is not installed"

    def _software(self) -> SoftwareEnvironment:
        from formulate import chem

        return SoftwareEnvironment.capture(rdkit=chem.rdkit_version())

    # -- domain --------------------------------------------------------------

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        basis = (
            "neutral, closed-shell, single-fragment C/H/O/Cl/Br/I liquids without "
            "saturated rings, between their melting and boiling points; validated on "
            "134 measured viscosities at 25 degrees Celsius, of which 110 pass these "
            "gates and 24 are refused"
        )
        smiles = candidate.molecule.smiles if candidate.molecule else None
        if smiles is None:
            return ApplicabilityDomain.outside(
                "candidate carries no molecule", basis=basis
            )

        refusal = structural_refusal(smiles)
        if refusal is not None:
            return ApplicabilityDomain.outside(refusal, basis=basis)

        counts = group_counts(smiles) or {}
        warnings: list[str] = []
        score = 1.0

        bucket = uncertainty_bucket(smiles)
        if bucket == "polyoxygenated":
            warnings.append(
                "two or more oxygen-bearing groups: measured over eight polyols and "
                "glycol ethers the spread is a factor of 2.1 with a bias of 1.25x, "
                "against 1.22x for the acyclic set. Glycol ethers are among the most "
                "important coating solvents there are and this is the bucket with the "
                "least evidence behind its error bar"
            )
            score = min(score, 0.45)

        if counts.get("aromatic_ring", 0) > 1:
            warnings.append(
                "more than one aromatic ring: every aromatic compound in the "
                "validation set is monocyclic, and the ortho/meta/para bookkeeping "
                "reads a ring fusion as a substitution"
            )
            score = min(score, 0.4)

        if counts.get("carbon", 0) > 20:
            warnings.append(
                "more than twenty carbons: the validated range runs to hexadecane, and "
                "the carbon term is linear in both A and B with nothing to bound it"
            )
            score = min(score, 0.5)

        warnings.append(
            "the group table is structural, not exhaustive: an unusual C/H/O "
            "arrangement that happens to balance its oxygens - an enol, an orthoester, "
            "a ketene acetal - passes these gates and would be silently wrong"
        )

        return ApplicabilityDomain(
            score=score,
            in_domain=score > 0.3,
            warnings=tuple(warnings),
            basis=basis,
        )

    # -- prediction ----------------------------------------------------------

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        molecule = request.candidate.molecule
        if molecule is None:
            return Prediction.unsupported(
                prop, self.id, "the candidate carries no molecule"
            )
        smiles = molecule.smiles

        refusal = structural_refusal(smiles)
        if refusal is not None:
            return Prediction.unsupported(prop, self.id, refusal)

        temperature = request.conditions.temperature_k
        if temperature is None:
            return Prediction.unsupported(
                prop,
                self.id,
                f"{prop} is steeply temperature dependent through the B/T term - about "
                "six per cent per kelvin for glycerol - and no temperature was given",
            )

        inputs: dict[str, tuple[float, float | None]] = {}
        for name in _NEEDED[prop]:
            unit = _DEPENDENCY_UNITS[name]
            pred = request.dependency(name)
            if pred is None or pred.quantity is None:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"needs {name} from an upstream expert, and none was available; "
                    "substituting a default would be a guess dressed as a measurement",
                )
            inputs[name] = (
                pred.quantity.to(unit).value,
                pred.uncertainty.converted(pred.quantity.unit, unit).std,
            )

        boiling = inputs["normal_boiling_point"][0]
        if temperature >= boiling:
            return Prediction.unsupported(
                prop,
                self.id,
                f"the requested {temperature - 273.15:.0f} degrees Celsius is at or "
                f"above the estimated boiling point of {boiling - 273.15:.0f}, so there "
                "is no liquid at one atmosphere to have a viscosity",
            )

        # The other end of the range, which a boiling point does not answer: an
        # Arrhenius form extrapolated below the melting point keeps returning a
        # number, and that number describes a supercooled liquid the substance
        # is not.
        from .measured import not_liquid_at

        wrong_phase = not_liquid_at(smiles, temperature)
        if wrong_phase:
            return Prediction.unsupported(prop, self.id, wrong_phase)

        counts = group_counts(smiles)
        if counts is None:  # pragma: no cover - refusal above catches this
            return Prediction.failed(prop, self.id, "group counting failed")

        if prop == "shear_viscosity":
            return self._viscosity(request, domain, smiles, counts, inputs, temperature)
        return self._diffusion(request, domain, smiles, counts, inputs, temperature)

    # -- the two properties --------------------------------------------------

    def _extrapolation_note(self, temperature: float) -> tuple[str, ...]:
        """A note when the request sits well above where the table was fitted.

        This cannot live in ``assess_domain``, which sees a candidate and no
        conditions; the same molecule is in domain at 25 degrees Celsius and
        extrapolated at 150.
        """
        if temperature <= _TEMPERATURE_WARNING_K:
            return ()
        return (
            f"the request is {temperature - 273.15:.0f} degrees Celsius, above the "
            f"{_TEMPERATURE_WARNING_K - 273.15:.0f} where the Orrick-Erbar residuals "
            "stop being flat: on measured densities at 150 degrees Celsius hexadecane "
            "reads 0.78x and ethylene glycol 0.62x. The per-kelvin term widens the bar "
            "to cover that, but this is an extrapolation and not a calibrated point",
        )

    def _log_sigma(self, smiles: str, temperature: float) -> tuple[float, str]:
        """Total ln-space uncertainty of the viscosity, and how it was reached."""
        bucket = uncertainty_bucket(smiles)
        structural = _BUCKET_LOG_SIGMA[bucket]
        drift = _TEMPERATURE_LOG_SIGMA_PER_K * abs(
            temperature - _CALIBRATION_TEMPERATURE_K
        )
        return math.hypot(structural, drift), (
            f"measured spread of ln(predicted/measured) for the {bucket} bucket "
            f"({structural:.2f}), in quadrature with {drift:.2f} for the "
            f"{abs(temperature - _CALIBRATION_TEMPERATURE_K):.0f} K "
            "extrapolation from the 25 degrees Celsius calibration"
        )

    def _viscosity(
        self,
        request: PredictionRequest,
        domain: ApplicabilityDomain,
        smiles: str,
        counts: Mapping[str, int],
        inputs: Mapping[str, tuple[float, float | None]],
        temperature: float,
    ) -> Prediction:
        upstream = {k: inputs[k] for k in ("liquid_density", "molar_mass")}
        try:
            value, propagated = propagate(
                lambda v: orrick_erbar_viscosity(
                    counts, v["liquid_density"], v["molar_mass"], temperature
                ),
                upstream,
            )
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            return Prediction.failed("shear_viscosity", self.id, f"correlation failed: {exc}")

        log_sigma, sigma_basis = self._log_sigma(smiles, temperature)
        relative = math.sqrt(math.exp(log_sigma**2) - 1.0)
        method_std = value * relative
        total = math.hypot(method_std, propagated)
        dominant = "propagated input error" if propagated > method_std else "correlation error"

        return self._make(
            "shear_viscosity",
            value,
            "cP",
            request,
            domain,
            std=total,
            kind=UncertaintyKind.COMBINED,
            basis=(
                f"{sigma_basis}; as a linear spread that is {relative:.0%} of the value "
                f"({method_std:.4g} cP), combined in quadrature with {propagated:.4g} cP "
                f"propagated from the supplied density and molar mass. {dominant} "
                "dominates. The structural spread is the honest one for this bucket: "
                "measured one-sigma coverage is 70 per cent over 80 acyclic compounds, "
                "77 per cent over 22 aromatics and 88 per cent over 8 polyoxygenated"
            ),
            notes=(
                "Orrick-Erbar predicts a Newtonian zero-shear viscosity. The conditions "
                "carry no shear rate, which is harmless for a small molecule but means "
                "this number is not comparable in kind to what a polymer melt expert "
                "returns under the same property name, where shear thinning is the "
                "whole point",
                "derived from an upstream density and molar mass, whatever their own "
                "provenance, not from a measured viscosity",
                *self._extrapolation_note(temperature),
            ),
            temperature_k=temperature,
            bucket=uncertainty_bucket(smiles),
        )

    def _diffusion(
        self,
        request: PredictionRequest,
        domain: ApplicabilityDomain,
        smiles: str,
        counts: Mapping[str, int],
        inputs: Mapping[str, tuple[float, float | None]],
        temperature: float,
    ) -> Prediction:
        density, molar_mass = inputs["liquid_density"][0], inputs["molar_mass"][0]
        viscosity_pa_s = (
            orrick_erbar_viscosity(counts, density, molar_mass, temperature) * 1e-3
        )
        if viscosity_pa_s > _STOKES_EINSTEIN_MAX_VISCOSITY_PAS:
            return Prediction.unsupported(
                "self_diffusion_coefficient",
                self.id,
                f"the liquid is {viscosity_pa_s * 1e3:.0f} mPa*s thick, beyond the "
                f"{_STOKES_EINSTEIN_MAX_VISCOSITY_PAS * 1e3:.0f} mPa*s this relation was "
                "checked to. Diffusion decouples from viscosity as a liquid approaches "
                "its glass transition - glycerol at about 1000 mPa*s reads 0.42x where "
                "everything between 0.2 and 17 mPa*s reads 0.66x to 1.01x - and a "
                "different physical regime is not something an error bar covers",
            )

        upstream = {
            k: inputs[k] for k in ("liquid_density", "molar_mass", "molar_volume_liquid")
        }
        try:
            value, propagated = propagate(
                lambda v: stokes_einstein_diffusion(
                    temperature,
                    orrick_erbar_viscosity(
                        counts, v["liquid_density"], v["molar_mass"], temperature
                    )
                    * 1e-3,
                    hydrodynamic_radius(v["molar_volume_liquid"]),
                ),
                upstream,
            )
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            return Prediction.failed(
                "self_diffusion_coefficient", self.id, f"correlation failed: {exc}"
            )

        viscosity_sigma, sigma_basis = self._log_sigma(smiles, temperature)
        log_sigma = math.hypot(_STOKES_EINSTEIN_LOG_SIGMA, viscosity_sigma)
        relative = math.sqrt(math.exp(log_sigma**2) - 1.0)
        method_std = value * relative
        total = math.hypot(method_std, propagated)

        notes = [
            "Stokes-Einstein is derived for a large sphere in a continuum, and a small "
            "molecule diffusing through a liquid of its own size violates that. The "
            "slip prefactor 4*pi is used rather than the stick 6*pi because 6*pi "
            "measures a systematic factor of two low (bias 0.52x against 0.78x over "
            "eighteen liquids). The remaining 22 per cent bias is carried in the error "
            "bar rather than fitted away",
            "computed from this expert's own viscosity rather than from an upstream "
            "one, so that a perturbation of the supplied density moves the correlation "
            "and its propagated bar together. The radius is a separate matter: it comes "
            "from the molar volume the engine resolved independently, which is why the "
            "two are cross-checked below rather than assumed to agree",
        ]
        notes.extend(self._extrapolation_note(temperature))

        # The molar volume and the density are the same measurement, resolved
        # independently by the engine, so it can hand over a measured density
        # beside an estimated molar volume.  Treating them as independent
        # widens the bar rather than narrowing it, which is safe; the radius
        # being inconsistent with the viscosity it divides is not, so say so.
        implied = molar_mass / (density * 1e6)  # g/mol over g/cm^3 -> m^3/mol
        supplied = inputs["molar_volume_liquid"][0]
        if abs(supplied - implied) > 0.05 * implied:
            notes.append(
                f"the supplied molar volume ({supplied:.4g} m^3/mol) disagrees by "
                f"{abs(supplied - implied) / implied:.0%} with the one implied by the "
                "supplied density and molar mass; the radius and the viscosity are "
                "therefore not describing quite the same liquid"
            )

        if molar_mass > _DIFFUSION_MASS_CEILING:
            notes.append(
                f"at {molar_mass:.0f} g/mol this is above the 155 g/mol top of the "
                "validation set, within which the residual already trends with molar "
                "mass (Pearson -0.50, ethanol 1.01x falling to decane 0.69x), so the "
                "value will read low"
            )

        return self._make(
            "self_diffusion_coefficient",
            value,
            "m^2/s",
            request,
            domain,
            std=total,
            kind=UncertaintyKind.COMBINED,
            basis=(
                f"{_STOKES_EINSTEIN_LOG_SIGMA:.2f} in ln for the Stokes-Einstein "
                "relation itself, measured on eighteen liquids with measured "
                "viscosities and molar volumes and chosen to cover the 0.78x bias "
                "rather than to match the 0.124 residual scatter; in quadrature with "
                f"{sigma_basis}, since D goes as 1/eta. As a linear spread that is "
                f"{relative:.0%} ({method_std:.4g} m^2/s), combined with {propagated:.4g} "
                "propagated from the supplied density, molar mass and molar volume. "
                "End-to-end one-sigma coverage measured at 76 per cent over seventeen "
                "liquids with the viscosity predicted too. A fitted prefactor "
                "of 3.12*pi does better - held-out RMS 0.13 in ln against 0.35 - and is "
                "declined because its residual trends with molar mass, so one constant "
                "fitted over 32 to 154 g/mol would drift silently for the heavier "
                "molecules a generator proposes"
            ),
            notes=tuple(notes),
            temperature_k=temperature,
            viscosity_pa_s=viscosity_pa_s,
        )
