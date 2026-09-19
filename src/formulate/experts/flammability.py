"""Closed-cup flash point from the Catoire-Naudin correlation.

Method: Catoire, L. and Naudin, V., "Estimation of Temperatures of Flash Points
of Organic Compounds", J. Phys. Chem. Ref. Data 33 (2004) 1083-1111::

    Tf = 1.477 * Tb^0.79686 * dHvap(298.15 K)^0.16845 * nC^(-0.05948)

with the temperatures in kelvin, the enthalpy of vaporisation in kJ/mol and
``nC`` the number of carbon atoms.

Why this correlation and not the classic linear one.  A flash point is the
temperature at which the liquid's vapour first reaches its lower flammable
limit, so it is set by a vapour pressure, and a vapour pressure needs two
numbers - where the liquid boils and how strongly it is held.  A relation in
the boiling point alone has only the first.  Four liquids in the panel below
boil within four kelvin of each other - ethyl acetate 350.2 K, ethanol
351.6, benzene 353.2, cyclohexane 353.9 - and flash at 269.2, 285.2, 262.2
and 256.2 K.  A twenty-nine kelvin spread at one boiling point is not noise a
better fit could remove; any ``Tf = a + b*Tb`` relation, and the Patil
quadratic in Tb, must return one number for all four.  The enthalpy of
vaporisation is what separates them, and Catoire-Naudin is the published form
that carries it.

Accuracy, measured here rather than quoted.  Ninety-one compounds with
compiled flash points (``chemicals.safety.T_flash``) and measured boiling
points, critical temperatures and enthalpies of vaporisation, spanning
alkanes, aromatics, alcohols, ethers, esters, ketones, aldehydes, amines,
nitriles, amides, nitro compounds and organosulfur:

    RMSE 6.33 K, mean absolute error 4.47 K, bias -0.67 K, worst -26.7 K
    (furan).  Excluding the four organosulfur compounds, which are handled
    separately below: RMSE 5.93 K, MAE 4.13 K.

On the ten common solvents this module was briefed against - ethanol, toluene,
acetone, hexane, methanol, ethyl acetate, 1-butanol, diethyl ether, xylene,
1-propanol - the RMSE is 2.22 K.  That is better than the whole-panel figure
and should not be read as the accuracy: those ten are the best-characterised
liquids in the set.

Uncertainty is honest in the only sense that can be checked: over the panel,
74.7 per cent of compounds fall inside their own one-sigma bar and 93.4 per
cent inside two sigma, against the 68 and 95 a correct estimate implies.
Driven from Joback-estimated inputs instead of measured ones, the same panel
gives RMSE 11.3 K with 69.4 per cent one-sigma coverage - the accuracy falls
by a factor of two and the error bar follows it, which is the behaviour the
propagation is there to produce.

Those coverage figures are in sample, and saying so matters: ``METHOD_SPREAD_K``
below is 1.253 times the mean absolute error of those same ninety-one
compounds, so measuring coverage over them asks the bar whether it covers the
data it was set from.  The correlation is published and its accuracy on the
panel is a genuine out-of-sample number; the error *bar* is not, and needed
its own held-out check.  So the whole flash-point compilation was driven
through instead - every compound in ``chemicals``' DIPPR-Serat, IEC 60079-20-1
and NFPA 497 tables with a resolvable structure and compiled Tb, Tc and
dHvap(Tb), 426 of them, of which the screens below refuse 59 and the
correlation answers 367.  Two hundred and seventy-nine of those are not in the
ninety-one:

    RMSE 7.08 K, MAE 4.92 K, bias +0.49 K, one-sigma coverage 71.0 per cent,
    two-sigma 90.7 per cent.  Non-sulfur only, which is what the 5.2 K bar is
    for: one-sigma 67.7 per cent against the 68 it claims.

The bar survives its own held-out test, and the accuracy degrades by about
0.7 K rather than by a factor.  Two sigma catching 90.7 rather than 95 per
cent is the one thing that does not hold: the tails are fatter than a normal
distribution, so an outlier here is further out than the bar suggests.  Furan
at -26.7 K is 5.1 sigma, which over 367 compounds a normal distribution would
produce about once in ten thousand panels; it happened here on the first one.

The reference state is load-bearing.  The correlation wants the enthalpy of
vaporisation at 298.15 K; this repository's ``enthalpy_vaporization`` is
defined at the normal boiling point.  Feeding the registry value in
unconverted produces numbers that look entirely reasonable and are
systematically wrong: RMSE 9.75 K with a -7.36 K bias over the same panel,
against 6.33 K with the Watson correction applied.  Nothing in the output
would reveal it, so the source temperature is read from the upstream
prediction's own conditions and the correction can never be applied twice.

Limitations, each of which is a refusal or a widened bar below rather than a
caveat in prose: carboxylic acids dimerise in the vapour and a monomeric
enthalpy of vaporisation is the wrong quantity for them (measured -15 to -29 K
on the short acids); polyols are unpredictable in the unsafe direction
(+17.1 K on ethylene glycol, +11.5 on propylene glycol, while glycerol and
diethylene glycol are within 4 K, and with four compounds there is no telling
which a new polyol will be); organosulfur compounds over-predict, by +11.9 K
over the four in the panel and by +6.8 K over the thirty-two in the wider
compilation, thirty of which err the same way; and a mixture's flash
point is not a property of its components at all.  The correlation is a
one-atmosphere closed-cup number, and closed-cup standards - ASTM D56, D93,
D3278 - do not agree with each other to better than a couple of kelvin on the
same liquid.

Three further refusals are not about accuracy but about being handed an input
that cannot be right.  A critical temperature within ten per cent of the
boiling point makes the Watson correction divide by nearly zero, and the flash
point that comes out is set by that subtraction rather than by chemistry - a
critical temperature half a kelvin above ethanol's boiling point returns
381.7 K, thirty kelvin above where ethanol boils.  A flash point at or above
the boiling point is refused outright, since a liquid whose vapour is already
at one atmosphere passed its lower flammable limit long before.  And a
zwitterion is an inner salt with no vapour pressure, which a net formal charge
does not catch: glycine and the betaines total zero.

Upstream uncertainty is propagated numerically through the correlation, as
:mod:`formulate.experts.interfacial` does, because a flash point derived from
an estimated boiling point cannot honestly claim the accuracy the correlation
shows on measured ones: with measured inputs the propagated term is under
1 K, with Joback-estimated inputs it is the dominant term.
"""

from __future__ import annotations

import math
from typing import Mapping

from formulate.core.candidate import Candidate, MaterialClass, MixtureSpec
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest
from .interfacial import propagate, watson_enthalpy_vaporization

# -- the correlation --------------------------------------------------------

#: Catoire & Naudin (2004), equation 1.  Kept as named constants rather than
#: inlined so that a future edit to any one of them is visible in a diff; the
#: exponent on nC is the small negative correction that keeps a long chain from
#: being over-predicted once its boiling point and enthalpy are already large.
CATOIRE_PREFACTOR = 1.477
CATOIRE_TB_EXPONENT = 0.79686
CATOIRE_HVAP_EXPONENT = 0.16845
CATOIRE_CARBON_EXPONENT = -0.05948

#: The correlation's reference state for the enthalpy of vaporisation.  Not the
#: request's temperature: 298.15 K is baked into the fitted coefficients, so
#: evaluating the enthalpy anywhere else would silently refit the correlation.
#: The Watson relation that moves the enthalpy onto this state is imported from
#: :mod:`formulate.experts.interfacial` rather than re-entered here.
REFERENCE_TEMPERATURE_K = 298.15


def catoire_naudin_flash_point(tb: float, hvap_298: float, carbon_count: int) -> float:
    """Flash point in K from Tb in K, dHvap(298.15 K) in J/mol and nC.

    The enthalpy is taken in J/mol - the registry's canonical unit - and
    converted to the kJ/mol the published coefficients expect inside, so that
    no caller has to remember which convention this correlation uses.
    """
    if tb <= 0.0 or hvap_298 <= 0.0 or carbon_count < 1:
        raise ValueError(
            "the Catoire-Naudin correlation needs a positive boiling point, a positive "
            f"enthalpy of vaporisation and at least one carbon; got {tb}, {hvap_298}, "
            f"{carbon_count}"
        )
    return (
        CATOIRE_PREFACTOR
        * tb**CATOIRE_TB_EXPONENT
        * (hvap_298 / 1000.0) ** CATOIRE_HVAP_EXPONENT
        * carbon_count**CATOIRE_CARBON_EXPONENT
    )


# -- what this expert needs from other experts ------------------------------

#: Declared dependencies and the units they are read in.  The critical
#: temperature earns its place only through the Watson correction, which is an
#: uncomfortable amount of machinery for a reference-state shift - but the
#: alternative is the 9.75 K / -7.36 K bias measured above, and rule 4 of this
#: repository forbids substituting anything for a dependency that is missing.
DEPENDENCY_UNITS = {
    "normal_boiling_point": "K",
    "enthalpy_vaporization": "J/mol",
    "critical_temperature": "K",
}

# -- uncertainty ------------------------------------------------------------

#: One sigma is 1.253 times a mean absolute error for a normal distribution.
#: The same conversion ``joback.py`` uses for its own measured spreads, so the
#: two experts' error bars mean the same thing.
NORMAL_SIGMA_PER_MAE = 1.253

#: Correlation error on *measured* inputs: 1.253 x the 4.13 K mean absolute
#: error measured over the 87 non-sulfur compounds of the panel.
#:
#: It cannot honestly go much lower than this, and not only because of the fit.
#: The ten flash points quoted in the brief this module was written against
#: disagree with the compiled values by 2.47 K RMSE and by 6.85 K on
#: 1-propanol alone, where the brief says 295 K and the compilation 288.15.
#: Closed-cup standards - ASTM D56, D93, D3278 - do not return the same number
#: for the same liquid, and an error bar narrower than the disagreement between
#: two references is a claim about the world that the world does not support.
#: The prediction for 1-propanol, 295.2 K, sits inside that disagreement and
#: agrees with whichever source you happen to be holding.
METHOD_SPREAD_K = 5.2

#: Organosulfur: 1.253 x the 11.9 K mean absolute error over the four sulfur
#: compounds in the panel - dimethyl sulfide +12.2, ethanethiol +12.9, dimethyl
#: disulfide +13.7, dimethyl sulfoxide +8.8.  Four compounds is not a sample,
#: but all four err the same way and by a similar amount, and the direction is
#: the unsafe one: the correlation says the liquid is harder to ignite than it
#: is.  Widening the bar and marking the prediction out of domain says that,
#: where a fitted -12 K offset on four points would be decoration.
#:
#: Four compounds later became thirty-two, when the whole flash-point
#: compilation was driven through for the held-out check above.  The bias is
#: confirmed - thirty of the thirty-two over-predict, mean +6.8 K - and
#: the magnitude is half what four compounds suggested, so 1.253 x that
#: sample's 7.0 K mean absolute error would be 8.8 K rather than 14.9.  The
#: wider number is kept anyway, and this is the one place in the module where
#: a bar is deliberately not calibrated to its own coverage (14.9 K catches
#: 32 of 32 where a correct bar catches about 22, which 8.8 K does: 20).  The error here is a
#: one-directional bias on the unsafe side, and a symmetric bar can only
#: describe it by being wide enough that the pessimistic bound the ranker
#: scores against lands where the compound actually flashes.
ORGANOSULFUR_SPREAD_K = 14.9

#: Largest reduced temperature, Tb/Tc, for which the Watson correction is
#: allowed to run.
#:
#: The correction divides by ``1 - Tb/Tc``, so as a supplied critical
#: temperature approaches the boiling point the enthalpy it produces at 298 K
#: diverges - and it diverges quietly, into a flash point that is merely
#: surprising rather than obviously broken.  A critical temperature half a
#: kelvin above ethanol's boiling point returns 381.7 K, thirty kelvin *above*
#: where ethanol boils.
#:
#: Guldberg's rule puts a real liquid at Tb/Tc near two thirds, and over the
#: 426 compounds of the flash-point compilation with both constants compiled
#: the largest is 0.816 (a hexamethyl siloxane).  A ratio of 0.90 is therefore
#: not an unusual liquid; it is a broken critical temperature, and the honest
#: answer to one of those is that the correction cannot be evaluated rather
#: than a number that came out of dividing by nearly zero.
WATSON_MAX_REDUCED_TEMPERATURE = 0.90

#: Upstream boiling-point spread above which the prediction carries an explicit
#: note.  Two kelvin is roughly four times what a compiled measurement declares
#: (0.5 K) and a sixth of what Joback declares (12.9 K), so it separates "this
#: came from a measurement" from "this came from a group contribution" without
#: needing to ask which expert answered.
ESTIMATED_INPUT_TB_STD_K = 2.0

# -- the screens ------------------------------------------------------------

#: Elements the correlation was developed over.  Organosilicon and
#: organophosphorus liquids are outside it, and organometallics are frequently
#: pyrophoric - for those a flash point describes the wrong hazard entirely.
SUPPORTED_ELEMENTS = frozenset({"C", "H", "N", "O", "S", "F", "Cl", "Br", "I"})

#: GHS's own boundary between a flammable gas and a flammable liquid.  Below
#: it the hazard is governed by the lower flammable limit of a gas, not by a
#: closed-cup flash point, and the closed-cup test is not performed at all.
#: Refuses propane, butane, dimethyl ether, vinyl chloride, ethylene oxide.
FLAMMABLE_LIQUID_TB_FLOOR_K = 293.15

#: Groups that carry their own oxidiser.  For these the governing hazard is
#: decomposition or detonation, and quoting a flash point beside a detonable
#: compound understates it by implying the usual controls apply.  Simple
#: C-nitro compounds are deliberately absent: nitromethane and nitrobenzene
#: predict to within 1.7 K here and are ordinary flammable liquids.
#: A tuple rather than a mapping because the nitrate ester appears twice, once
#: per charge form - RDKit does not treat the neutral pentavalent drawing and
#: the charge-separated one as the same pattern, and both are written in the
#: wild.
SELF_OXIDISING_SMARTS: tuple[tuple[str, str], ...] = (
    ("a peroxide", "[OX2;!$(O=*)][OX2;!$(O=*)]"),
    ("an organic azide", "[#6][NX2]=[NX2+]=[NX1-]"),
    ("a nitrate ester", "[OX2;!$(O=*)][NX3+](=O)[OX1-]"),
    ("a nitrate ester", "[OX2;!$(O=*)][NX3](=O)=O"),
    ("a nitramine", "[NX3]([#6])[NX3+](=O)[OX1-]"),
)

#: A nitro group on carbon, in both drawings RDKit distinguishes.  One of these
#: is an ordinary flammable liquid - nitromethane -1.7 K, nitrobenzene -1.2 K,
#: and the eight mononitro compounds in the reference compilation are predicted
#: to 3.6 K mean absolute error - so the self-oxidising screen above must not
#: reach them.  Two or more on one molecule is a different substance: the
#: oxygen balance is no longer negative enough for the governing hazard to be
#: ignition of a vapour, and the list of dinitro and trinitro aromatics is a
#: list of secondary explosives (2,4-dinitrotoluene, TNT, picric acid).  The
#: same argument the nitrate-ester and nitramine patterns above are there to
#: make, so it is made the same way.  Nothing in the 426-compound compilation
#: this module is measured against carries two, so the screen costs no measured
#: coverage; it closes a gap through which TNT would have been handed a flash
#: point and the usual flammable-liquid controls implied along with it.
C_NITRO_SMARTS = "[#6][$([NX3](=O)=O),$([NX3+](=O)[OX1-])]"

#: Two, because one is nitromethane and two is 2,4-dinitrotoluene.
MAX_C_NITRO_GROUPS = 1

#: Carboxylic acid.  Measured error on the short acids: formic -29.1 K, acetic
#: -24.2, propionic -15.7, butyric -18.1.  All the same sign and the mechanism
#: is understood - these dimerise in the vapour, so a monomeric enthalpy of
#: vaporisation is the wrong quantity to put into a vapour-pressure
#: correlation.  Valeric (-2.2) and benzoic (+4.9) acid are fine, so this
#: refusal costs real coverage; a twenty-kelvin systematic offset on a safety
#: number is not a wide error bar, it is the wrong answer.
CARBOXYLIC_ACID_SMARTS = "[CX3](=O)[OX2H1]"

#: Hydroxyl, counted.  Two or more and the prediction is refused: ethylene
#: glycol +17.1 K and propylene glycol +11.5 K, both over-predicting, against
#: glycerol +0.9 and diethylene glycol -3.7, which are fine.  Four compounds
#: cannot tell those two groups apart - vicinal does not separate them, since
#: glycerol is vicinal too - and the failures are in the unsafe direction and
#: sit either side of the 366 K GHS boundary.  Refusing because the class
#: cannot be split is the honest reading; it is also the expensive one, and it
#: rests on four compounds, which the refusal reason says out loud.
HYDROXYL_SMARTS = "[OX2H1]"


def carbon_count(smiles: str) -> int:
    """Number of carbon atoms, which the correlation takes as a factor."""
    from formulate import chem

    mol = chem.mol_from_smiles(smiles)
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")


def carbon_hydrogen_count(smiles: str) -> int:
    """Hydrogens bonded to carbon.

    Not the total hydrogen count: the hydrogen of a hydroxyl or an amine is not
    what a flame abstracts.  Sustained propagation in a closed cup needs
    abstractable hydrogen *on carbon*.
    """
    from formulate import chem

    mol = chem.mol_from_smiles(smiles)
    if mol is None:
        return 0
    return sum(
        atom.GetTotalNumHs(includeNeighbors=True)
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C"
    )


def halogen_count(smiles: str) -> int:
    from formulate import chem

    mol = chem.mol_from_smiles(smiles)
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in ("F", "Cl", "Br", "I"))


def hydroxyl_count(smiles: str) -> int:
    from rdkit import Chem

    from formulate import chem

    mol = chem.mol_from_smiles(smiles)
    pattern = Chem.MolFromSmarts(HYDROXYL_SMARTS)
    if mol is None or pattern is None:
        return 0
    return len(mol.GetSubstructMatches(pattern))


def c_nitro_count(smiles: str) -> int:
    """Nitro groups bonded to carbon, in either drawing."""
    from rdkit import Chem

    from formulate import chem

    mol = chem.mol_from_smiles(smiles)
    pattern = Chem.MolFromSmarts(C_NITRO_SMARTS)
    if mol is None or pattern is None:
        return 0
    return len(mol.GetSubstructMatches(pattern))


def ionic_reason(smiles: str) -> str | None:
    """Why this molecule is an ion rather than a neutral liquid, or None.

    Two cases, and the second is the one a net charge misses.  A quaternary
    ammonium cation has a non-zero total charge.  An amino acid, a betaine or a
    sulfobetaine has a total charge of zero and is still an inner salt: it does
    not evaporate, so it cannot flash.  What separates it from nitromethane -
    which is also drawn with a plus and a minus, and which is an ordinary
    flammable liquid - is that in a nitro group, an azide or an amine oxide the
    two charges sit on bonded atoms and describe one neutral functional group,
    while in a betaine they sit at opposite ends of the molecule.  So the test
    is per charged atom: a charge with no countercharge next door is a real
    one.
    """
    from rdkit import Chem

    from formulate import chem

    mol = chem.mol_from_smiles(smiles)
    if mol is None:
        return None

    total = Chem.GetFormalCharge(mol)
    if total:
        return (
            f"carries a net formal charge of {total:+d}; the correlation is fitted to "
            "neutral molecular liquids, and a salt or ionic liquid has a vapour pressure "
            "far too low to flash"
        )

    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge == 0:
            continue
        paired = any(
            (neighbour.GetFormalCharge() < 0)
            if charge > 0
            else (neighbour.GetFormalCharge() > 0)
            for neighbour in atom.GetNeighbors()
        )
        if not paired:
            return (
                f"is a zwitterion - the {atom.GetSymbol()}{charge:+d} here has no "
                "countercharge bonded to it, so this is an inner salt rather than a "
                "neutral molecule with a charge-separated functional group. An amino "
                "acid or a betaine has no measurable vapour pressure and so no flash "
                "point; a nitro group, an azide and an amine oxide carry their two "
                "charges on bonded atoms and are not caught here"
            )
    return None


def non_combustible_reason(smiles: str) -> str | None:
    """Why this molecule has no closed-cup flash point, or None.

    Every branch here is a refusal rather than a wide error bar, because each
    one describes a liquid for which the number the correlation would return
    does not exist, rather than one it would get wrong.
    """
    from formulate import chem

    extra = chem.elements(smiles) - SUPPORTED_ELEMENTS
    if extra:
        return (
            f"contains {', '.join(sorted(extra))}, outside the C/H/N/O/S/halogen set the "
            "correlation was developed over; organosilicon and organophosphorus liquids "
            "are not in it, and an organometallic is more likely to be pyrophoric, which "
            "is a different hazard from a flash point"
        )

    carbons = carbon_count(smiles)
    if carbons == 0:
        return (
            "has no carbon atom, so there is nothing here to oxidise and the correlation "
            "- which takes the carbon count as a factor - is not evaluable at all"
        )

    hydrogens = carbon_hydrogen_count(smiles)
    if hydrogens == 0:
        return (
            "has no carbon-bound hydrogen; sustained flame propagation in a closed cup "
            "needs abstractable hydrogen on carbon. This screen also refuses carbon "
            "disulfide, which does flash at 243 K and which the correlation over-predicts "
            "by 12.6 K here, so it errs towards silence rather than towards a number"
        )

    halogens = halogen_count(smiles)
    if halogens >= hydrogens:
        return (
            f"carries {halogens} halogen(s) against {hydrogens} carbon-bound hydrogen(s): "
            "the halogen scavenges the chain carriers faster than the hydrogen feeds them, "
            "and liquids in this range - chloroform, dichloromethane, "
            "1,1,1-trichloroethane - have no closed-cup flash point. The screen counts "
            "atoms rather than weighing them, so it also refuses trichloroethylene, for "
            "which a flash point is tabulated and which this correlation would have "
            "reproduced to within 0.4 K"
        )

    for label, smarts in SELF_OXIDISING_SMARTS:
        if chem.has_substructure(smiles, smarts):
            return (
                f"contains {label}, which carries its own oxidiser. The governing hazard "
                "is decomposition or detonation rather than ignition of a vapour above a "
                "liquid, and a flash point quoted beside it would understate it"
            )

    nitro_groups = c_nitro_count(smiles)
    if nitro_groups > MAX_C_NITRO_GROUPS:
        return (
            f"carries {nitro_groups} nitro groups on carbon. One is a flammable liquid - "
            "nitromethane and nitrobenzene are predicted to within 1.7 K here and are "
            "kept - but a dinitro or trinitro compound carries enough of its own oxidiser "
            "that the governing hazard is decomposition or detonation, as it is for the "
            "nitrate esters and nitramines refused above. Quoting a closed-cup flash "
            "point for 2,4-dinitrotoluene or TNT would imply the controls that go with a "
            "flammable liquid, which are not the controls these need"
        )

    if chem.has_substructure(smiles, CARBOXYLIC_ACID_SMARTS):
        return (
            "is a carboxylic acid. These dimerise in the vapour, so the monomeric enthalpy "
            "of vaporisation this correlation is fed is the wrong quantity, and the error "
            "is systematic rather than scattered: measured here at -29.1 K for formic, "
            "-24.2 for acetic, -15.7 for propionic and -18.1 for butyric acid. A twenty-"
            "kelvin offset that always points the same way is a wrong answer, not a wide one"
        )

    hydroxyls = hydroxyl_count(smiles)
    if hydroxyls >= 2:
        return (
            f"carries {hydroxyls} hydroxyl groups. Measured on the four polyols available "
            "here, the correlation over-predicts ethylene glycol by 17.1 K and propylene "
            "glycol by 11.5 - the unsafe direction, and across the 366 K GHS boundary - "
            "while glycerol and diethylene glycol come out within 4 K. Four compounds "
            "cannot say which group a new polyol falls in, and being unable to tell is "
            "the reason for refusing rather than for guessing"
        )

    return None


def mixture_refusal_reason(mixture: MixtureSpec) -> str:
    """Why a blend's flash point is not this expert's to give.

    Stated per candidate rather than by omitting MIXTURE from the supported
    classes, because the chemistry is the point: the failure mode here is not
    "no expert covers this", it is "the obvious answer is wrong".
    """
    from .measured import measured_value

    volatile = ""
    best: float | None = None
    for component in mixture.components:
        if component.molecule is None:
            continue
        boiling = measured_value("normal_boiling_point", component.molecule.smiles)
        if boiling is not None and (best is None or boiling < best):
            best, volatile = boiling, component.molecule.smiles

    lowest = (
        f" Its most volatile component here is {volatile}, boiling at {best:.0f} K, and a "
        "few per cent of it can decide the answer."
        if best is not None
        else ""
    )
    return (
        "a mixture's flash point is not an average of its components'. It is the "
        "temperature at which the vapour above the blend first reaches its lower "
        "flammable limit, which is Le Chatelier's rule over partial pressures "
        "y_i = x_i * gamma_i * Psat_i(T) / P, so a few per cent of a volatile component "
        "sets it. There is not even a safe bound to fall back on: with positive deviation "
        "from ideality x_i*gamma_i exceeds one and a blend can flash below every one of "
        "its components. The activity coefficients are available from the UNIFAC expert "
        "in experts/activity.py, but neither a vapour pressure nor a lower flammable "
        "limit is a property in this registry, so they could only come from an undeclared "
        "side channel, and there is no mixture flash-point reference data here to hold "
        "the result out against." + lowest
    )


class FlashPointExpert(Expert):
    """Closed-cup flash point of a combustible organic liquid."""

    id = "flash_point"
    version = "1"
    method = "Catoire & Naudin correlation (J. Phys. Chem. Ref. Data 33:1083, 2004)"
    family = PropertyFamily.THERMAL
    # MIXTURE is listed so that a blend gets the chemical refusal above rather
    # than dropping out of dispatch with no explanation. The cost is that
    # flash_point stops appearing in the coverage report's uncovered list for
    # mixtures; the per-candidate reason is worth more than that line.
    supported_classes = frozenset({MaterialClass.MOLECULE, MaterialClass.MIXTURE})
    supported_properties = frozenset({"flash_point"})
    dependencies = frozenset(DEPENDENCY_UNITS)

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "RDKit is required to screen for combustibility"

    # -- domain ------------------------------------------------------------

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        from formulate import chem

        if candidate.material_class is MaterialClass.MIXTURE:
            return ApplicabilityDomain.outside(
                "the correlation is fitted to pure liquids",
                basis="pure organic liquids at one atmosphere",
            )

        smiles = candidate.molecule.smiles if candidate.molecule else None
        if smiles is None or chem.mol_from_smiles(smiles) is None:
            return ApplicabilityDomain.outside("candidate carries no parseable molecule")

        if ionic_reason(smiles) is not None:
            return ApplicabilityDomain.outside(
                "the correlation is fitted to neutral molecular liquids; a salt, a "
                "zwitterion or an ionic liquid has a negligible vapour pressure and "
                "usually no flash point at all",
                basis="neutral organic liquids",
            )

        warnings: list[str] = []
        score = 1.0
        in_domain = True

        elements = chem.elements(smiles)
        if "S" in elements:
            warnings.append(
                "organosulfur: 30 of the 32 sulfur compounds in the flash-point "
                "compilation over-predict, by +6.8 K on average and +13.7 K at worst, "
                "which is the unsafe direction. The error bar is widened to cover it, but "
                "a systematic offset is not really what an error bar describes"
            )
            # Out of domain rather than merely warned about, because the error
            # here is a known bias and not scatter: four out of four the same
            # way. The prediction is still returned, so a search can rank on it,
            # but it is flagged as one this method is not right about.
            score = min(score, 0.3)
            in_domain = False

        if "Br" in elements or "I" in elements:
            warnings.append(
                "brominated or iodinated: bromine inhibits a flame far more strongly per "
                "atom than chlorine does, and the halogen screen here counts atoms rather "
                "than weighing them. The one brominated compound with a reference value, "
                "bromoethane, came out 17.0 K low"
            )
            score = min(score, 0.4)

        carbons = carbon_count(smiles)
        if carbons > 14:
            warnings.append(
                f"{carbons} carbons: the panel this was measured against reaches C12 "
                "(dodecane), and a long-chain liquid's flash point is increasingly set by "
                "the accuracy of its boiling point rather than by this correlation"
            )
            score = min(score, 0.6)

        return ApplicabilityDomain(
            score=score,
            in_domain=in_domain,
            warnings=tuple(warnings),
            basis=(
                "neutral combustible organic liquids of C, H, N, O, S and the lighter "
                "halogens, boiling above 293 K; measured here against 91 compounds with "
                "compiled closed-cup flash points, RMSE 6.33 K"
            ),
        )

    # -- prediction --------------------------------------------------------

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        from formulate import chem

        candidate = request.candidate

        if candidate.material_class is MaterialClass.MIXTURE:
            if candidate.mixture is None:  # pragma: no cover - the model validator forbids it
                return Prediction.unsupported(prop, self.id, "candidate carries no mixture")
            return Prediction.unsupported(prop, self.id, mixture_refusal_reason(candidate.mixture))

        if candidate.molecule is None:
            return Prediction.unsupported(prop, self.id, "candidate carries no molecule")
        smiles = candidate.molecule.smiles
        if chem.mol_from_smiles(smiles) is None:
            return Prediction.unsupported(
                prop, self.id, f"{smiles!r} could not be parsed as a molecule"
            )

        # The structural screens run before the dependencies are read, so that
        # water is refused for not burning rather than for missing a critical
        # temperature it was never going to need.
        ionic = ionic_reason(smiles)
        if ionic is not None:
            return Prediction.unsupported(prop, self.id, f"this substance {ionic}")
        refusal = non_combustible_reason(smiles)
        if refusal is not None:
            return Prediction.unsupported(prop, self.id, f"this substance {refusal}")

        inputs: dict[str, tuple[float, float | None]] = {}
        for name, unit in sorted(DEPENDENCY_UNITS.items()):
            pred = request.dependency(name)
            if pred is None or pred.quantity is None:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"needs {name} from an upstream expert and none was available; the "
                    "Catoire-Naudin correlation takes all three of the boiling point, the "
                    "enthalpy of vaporisation and - for the reference-state correction - "
                    "the critical temperature, and substituting for any of them would "
                    "make the answer untraceable",
                )
            inputs[name] = (
                pred.quantity.to(unit).value,
                pred.uncertainty.converted(pred.quantity.unit, unit).std,
            )

        tb = inputs["normal_boiling_point"][0]
        tc = inputs["critical_temperature"][0]

        if tb <= FLAMMABLE_LIQUID_TB_FLOOR_K:
            return Prediction.unsupported(
                prop,
                self.id,
                f"boils at {tb:.1f} K, at or below the {FLAMMABLE_LIQUID_TB_FLOOR_K:.2f} K "
                "that GHS uses to separate a flammable liquid from a flammable gas. Below "
                "it the hazard is set by the lower flammable limit of the gas and the "
                "closed-cup test is not performed",
            )
        # Which temperature the upstream enthalpy is stated at. The registry
        # defines it at the normal boiling point and Joback stamps that on the
        # prediction; reading the stamp rather than assuming it is what stops
        # the correction being applied twice if some future expert supplies the
        # 298 K value directly. Read before the guards below, because the state
        # the enthalpy arrives in is one of the things they have to check.
        hvap_prediction = request.dependency("enthalpy_vaporization")
        stamped = (
            hvap_prediction.conditions.temperature_k if hvap_prediction is not None else None
        )
        source_t = stamped if stamped is not None else tb

        if tc <= max(tb, source_t, REFERENCE_TEMPERATURE_K):
            return Prediction.unsupported(
                prop,
                self.id,
                f"the supplied critical temperature {tc:.1f} K is not above the boiling "
                f"point {tb:.1f} K, the {source_t:.1f} K the enthalpy of vaporisation is "
                "stated at, and 298.15 K, so the Watson correction that puts the enthalpy "
                "on this correlation's reference state cannot be evaluated",
            )
        # The Watson correction runs from whichever temperature the enthalpy is
        # stated at, so it is that temperature - normally the boiling point -
        # whose distance from the critical point decides whether the correction
        # means anything.
        hottest = max(tb, source_t)
        reduced = hottest / tc
        if reduced >= WATSON_MAX_REDUCED_TEMPERATURE:
            return Prediction.unsupported(
                prop,
                self.id,
                f"the enthalpy of vaporisation is stated at {hottest:.1f} K, which is "
                f"{reduced:.3f} of the supplied critical temperature {tc:.1f} K. The "
                "Watson correction divides by 1 - T/Tc, so at this ratio the enthalpy it "
                "reports at 298 K - and the flash point that follows - is set by how "
                "close those two numbers are rather than by the chemistry. No liquid in "
                "the reference compilation exceeds 0.816 and Guldberg's rule puts a "
                "normal one near 0.67, so what needs fixing is the critical temperature, "
                "not this refusal",
            )
        nc = carbon_count(smiles)

        def correlate(values: Mapping[str, float]) -> float:
            source_t = stamped if stamped is not None else values["normal_boiling_point"]
            critical = values["critical_temperature"]
            if critical <= max(source_t, REFERENCE_TEMPERATURE_K):
                raise ValueError("critical temperature below the reference state")
            hvap_298 = watson_enthalpy_vaporization(
                values["enthalpy_vaporization"], source_t, critical, REFERENCE_TEMPERATURE_K
            )
            return catoire_naudin_flash_point(values["normal_boiling_point"], hvap_298, nc)

        try:
            value, propagated = propagate(correlate, inputs)
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            return Prediction.failed(prop, self.id, f"correlation failed: {exc}")

        # A pure liquid cannot flash above where it boils. At its normal
        # boiling point the vapour above it is pure vapour at one atmosphere,
        # which is a hundred times any lower flammable limit, so the flash
        # point was passed long before. The correlation does not know that, and
        # will return a number above Tb if it is fed an enthalpy large enough:
        # ethanol with a 200 kJ/mol enthalpy comes out at 376 K. Over the 426
        # compounds of the compilation the smallest margin is 55 K, so this
        # costs no measured coverage - it catches an input that is wrong.
        if value >= tb:
            return Prediction.unsupported(
                prop,
                self.id,
                f"the correlation returns {value:.1f} K, at or above the supplied boiling "
                f"point {tb:.1f} K. A pure liquid's vapour reaches one atmosphere at its "
                "boiling point and the lower flammable limit is a hundred times below "
                "that, so a flash point above the boiling point is not a physical answer; "
                "the enthalpy of vaporisation or the critical temperature supplied here "
                "is wrong",
            )

        method_std = (
            ORGANOSULFUR_SPREAD_K if "S" in chem.elements(smiles) else METHOD_SPREAD_K
        )
        total = math.hypot(method_std, propagated)

        tb_std = inputs["normal_boiling_point"][1] or 0.0
        notes = [
            "a one-atmosphere closed-cup value; a real flash point falls by several "
            "kelvin at altitude, which this correlation does not model",
            "derived from the boiling point, enthalpy of vaporisation and critical "
            "temperature supplied by other experts, not from a measured flash point; how "
            "much of that is measurement and how much estimate is in the error bar's "
            "basis rather than in this note",
        ]

        # A note, not a refusal, and this is the one place the design as written
        # was wrong. Refusing a flash point that lands below the compiled melting
        # point sounds right - a closed-cup test measures a liquid - and it
        # throws away benzene, cyclohexane and tert-butanol, whose tabulated
        # flash points are genuinely below their melting points (benzene flashes
        # at 262 K and freezes at 279) and which this correlation reproduces to
        # within 5.6 K. The measurement overruled the argument. Silent when the
        # melting point is unknown, following measured.not_liquid_at.
        from .measured import measured_value

        melting = measured_value("melting_point", smiles)
        if melting is not None and value < melting:
            notes.append(
                f"this flash point is below the compiled melting point {melting:.1f} K. "
                "That is not by itself wrong - benzene, cyclohexane and tert-butanol all "
                "have tabulated flash points below where they freeze - but the closed-cup "
                "test on a substance that is solid at the temperature quoted is a "
                "different measurement from the one on a liquid"
            )
        if tb_std > ESTIMATED_INPUT_TB_STD_K:
            notes.append(
                f"the boiling point supplied carries {tb_std:.1f} K of its own, and this "
                "correlation passes about 0.66 of a boiling-point error straight through: "
                "over the reference panel with Joback-estimated inputs the flash-point "
                "RMSE is 11.3 K and the worst case 36.7 K, against 6.33 K and 26.7 K with "
                "measured ones. Read this as a GHS category rather than as a number"
            )

        dominant = "propagated input error" if propagated > method_std else "correlation error"
        return self._make(
            prop,
            value,
            "K",
            request,
            domain,
            std=total,
            kind=UncertaintyKind.COMBINED,
            basis=(
                f"correlation error {method_std:.1f} K on measured inputs (1.253 x the "
                f"{method_std / NORMAL_SIGMA_PER_MAE:.1f} K mean absolute error measured "
                "over the reference panel), combined in quadrature with "
                f"{propagated:.3g} K propagated from the supplied boiling point, enthalpy "
                f"of vaporisation and critical temperature; {dominant} dominates. Inputs "
                "are treated as independent, which understates the spread when several "
                "come from the same group-contribution fit"
            ),
            notes=tuple(notes),
            carbon_count=nc,
            enthalpy_reference_temperature_k=stamped if stamped is not None else tb,
        )
