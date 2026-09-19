"""Semi-empirical electronic structure: GFN2-xTB dipole moments.

Method: Bannwarth, C., Ehlert, S. and Grimme, S., "GFN2-xTB - An Accurate and
Broadly Parametrized Self-Consistent Tight-Binding Quantum Chemical Method with
Multipole Electrostatics and Density-Dependent Dispersion Contributions",
J. Chem. Theory Comput. 15 (2019) 1652-1671.  Evaluated through the ``xtb``
Python API on a Boltzmann-weighted conformer ensemble embedded with ETKDGv3
(Wang, Witek, Landrum and Riniker, J. Chem. Inf. Model. 60 (2020) 2044) and
relaxed with MMFF94 (Halgren, J. Comput. Chem. 17 (1996) 490).

This is the cheap counterpart to a DFT electronic expert: 68 ms per molecule
here against seconds for a small-basis DFT dipole, so it can score a whole
candidate pool rather than a handful.  It is not trying to win that comparison.
Where a DFT expert also covers ``dipole_moment``, :func:`prefer` arbitrates on
the applicability domain and then on the stated spread, and the spread stated
here is honestly wide.  A 0.1 D DFT dipole *should* displace a 0.5 D
tight-binding one, and nothing in this module is tuned to prevent that.

WHAT WAS MEASURED HERE, AND HOW
-------------------------------
79 compounds with well-established experimental gas-phase dipole moments (CRC
Handbook "Dipole Moments" table; NIST CCCBDB compilation) were run through the
exact protocol this module ships.  Against those measurements:

* mean absolute error 0.430 D, RMSE 0.608 D, mean signed error **+0.341 D**;
* by magnitude: MAE 0.215 D below 1.5 D, 0.379 D from 1.5 to 3 D, 0.779 D
  above 3 D - the error grows with the dipole, so a flat error bar would be
  too wide for a weakly polar molecule and too narrow for a strongly polar one;
* by element class: CHNO 0.346 D (n=59), chlorine 0.181 D (n=8), divalent or
  oxidised sulfur 0.660 D (n=7), bromine/iodine 1.929 D (n=3);
* 68 ms per molecule (median 25 ms, worst 504 ms) with the thread pin in force.

LIMITATIONS, IN THE ORDER THEY WILL BITE
----------------------------------------
1. The method **overestimates**, it does not merely scatter: +0.341 D mean over
   79 compounds, one-sided almost everywhere, and growing to +0.733 D above
   3 D.  A search that maximises dipole is flattered by every candidate; one
   that minimises it is penalised by every candidate.  This is stated rather
   than corrected away - see ``_STD_BASIS``.
2. Strongly polar groups are the accuracy tail, and a dipole-maximising search
   surfaces exactly them.  Nitrobenzene is out by +1.51 D and benzotrifluoride
   by +1.36 D, the two worst CHNOF cases in the set.  Nitro and perfluoroalkyl
   groups are therefore flagged out of domain.
3. Everything here is an isolated molecule in vacuum.  A liquid-phase effective
   dipole is enhanced by the reaction field, typically 20-40% for a polar
   solvent.  If a formulation target means the effective dipole, this expert
   answers a different question with the right units and the right name.
4. The ensemble average uses conformer populations from GFN2 single points on
   MMFF94 geometries, not from GFN2 minima.  Where those are wrong the average
   is wrong: ethylenediamine comes back at 0.95 D against 1.99 D experimental
   because GFN2 over-stabilises the internally hydrogen-bonded gauche conformer
   whose dipole cancels.  The conformer spread (1.17 D there) is what carries
   that into the error bar.  Relaxing every conformer with GFN2 before
   weighting, measured here, moves the ensemble average by -0.21 D (ethylene
   glycol) to +0.28 D (1,2-dimethoxyethane), with ethanol -0.03, 1-butanol
   -0.03 and 1,2-dichloroethane +0.08: inside the stated bar in every case, at
   about a hundred times the cost.
5. Validation is organic.  GFN2-xTB is parameterised to Z <= 86 and will return
   a dipole for a tungsten complex; nothing here says whether it is any good.
6. A structure the candidate leaves ambiguous is refused rather than averaged.
   An unassigned stereogenic double bond is two different compounds -
   (E)-1,2-dichloroethene is 0.00 D here and (Z) 1.85 D - and the conformer
   generator resolves it arbitrarily or mixes both.  Unassigned tetrahedral
   centres are *not* refused; see :data:`_STEREO_REFUSAL` for the measurement
   that separates the two cases.

DELIBERATELY NOT COVERED: ``homo_lumo_gap``
-------------------------------------------
The brief for this expert asked for a HOMO-LUMO gap as well.  It is refused,
and the refusal is expressed as absence from :attr:`supported_properties` so
that ``registry.uncovered()`` keeps reporting the property as uncovered rather
than claiming this expert supplies it.

Measured here against B3LYP/def2-SVP on identical geometries (15 molecules),
the GFN2 gap error flips sign with structure class: **+3.90 eV** mean over six
saturated sigma systems (methane +4.24, water +5.03, ethanol +4.01, cyclohexane
+4.08, ammonia +3.98, 1,4-dioxane +2.06) and **-2.26 eV** over nine
pi-conjugated ones (benzene -1.93, pyridine -2.97, acetonitrile -3.96, acetone
-2.32, naphthalene -1.58).  The whole-set mean is +0.20 eV purely by
cancellation; the mean absolute error is 2.92 eV and the range runs -3.96 to
+5.03 eV.  A standard deviation wide enough to span both classes would be
several eV - most of the property's useful range - and would anyway be
describing a signed, structure-dependent bias rather than a spread.  Comparing
to experiment instead does not rescue it: benzene's GFN2 gap of 4.83 eV sits
next to a 4.90 eV first singlet excitation and looks excellent, while water's
14.16 eV sits next to 7.4 eV.  The apparent agreement for aromatics is
coincidence.  Two further reasons specific to this repository:

* ``formulate.physics.qm.xtb_backend`` already refuses to report a gap from the
  same backend, in writing.  Shipping one here would make two parts of the
  system disagree about the same number from the same code.
* ``prefer()`` resolves two experts for one property on domain then spread.  A
  tight-binding eigenvalue difference and a Kohn-Sham gap are not commensurate
  quantities, and the engine has no way to see that.

Rule 1's ``Prediction.unsupported`` is for a candidate-specific failure.  This
refusal holds for every candidate, always, so it belongs in the capability
declaration instead.
"""

from __future__ import annotations

import ctypes
import math
import pathlib
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest

# --------------------------------------------------------------------------
# Physical constants and unit conversions
# --------------------------------------------------------------------------

#: Bohr radius in angstrom (CODATA 2018).  Load-bearing: ``xtb``'s Calculator
#: takes positions in BOHR, not angstrom, and says so nowhere in the Python
#: signature.  Feeding angstrom does not raise - it silently returns a
#: converged-looking energy and a plausible-looking dipole for a molecule
#: stretched by a factor of 1.89.  Verified against the published GFN2 result
#: for water: -5.0702 Eh on the bohr convention, -3.5007 Eh on angstrom.
ANGSTROM_PER_BOHR = 0.529177210903

#: Debye per atomic unit of dipole moment (e * bohr).  ``Results.get_dipole()``
#: returns atomic units; the registry's canonical unit for dipole_moment is
#: debye.
DEBYE_PER_AU = 2.541746473

#: eV/angstrom per hartree/bohr, for the geometry-quality gradient check.
EV_PER_ANGSTROM_PER_HARTREE_BOHR = 27.211386245988 / ANGSTROM_PER_BOHR

#: Boltzmann constant in hartree/kelvin, so conformer energies straight out of
#: xtb can be Boltzmann-weighted without a round trip through SI.
BOLTZMANN_HARTREE_PER_K = 3.166811563e-6

#: Temperature used when the request does not state one.  The registry marks
#: dipole_moment condition-independent, which is true for a rigid molecule and
#: not true for a flexible one, whose ensemble average moves with temperature.
#: The request's temperature is used when there is one and recorded in
#: provenance either way, so the discrepancy is visible rather than hidden.
DEFAULT_TEMPERATURE_K = 298.15


# --------------------------------------------------------------------------
# Element screening
# --------------------------------------------------------------------------

#: Highest atomic number GFN2-xTB is parameterised for.
#:
#: THIS IS A SAFETY LIMIT, NOT AN ACCURACY ONE.  Scanning Z = 80..94 through
#: this exact code path in one subprocess each: Z <= 86 returns an energy; Z =
#: 87, 89, 90, 91, 92 SEGFAULT (exit 139), Z = 93 aborts (exit 134), and Z =
#: 88 and 94 hit a Fortran "no basis found ... ERROR STOP" (exit 1).  Every one
#: of those kills the interpreter, so ``Expert.predict``'s try/except cannot
#: contain them.  RDKit parses "[Pu]" and "[Fr]" without complaint, so the path
#: is reachable from an ordinary candidate.  The screen therefore runs in pure
#: Python before any xtb object is constructed.
MAX_ATOMIC_NUMBER = 86

#: Elements the 79-compound validation set actually contains.  GFN2 covers far
#: more than this; the claim being made is about what was checked, not about
#: what the method will run on.
VALIDATED_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})

#: Soft, highly polarisable atoms GFN2 over-polarises.  Measured on this set:
#: bromine/iodine mean absolute error 1.929 D (n=3, all +ve - CH3I comes back
#: at 3.67 D against 1.62 D experimental) and sulfur 0.660 D (n=7), against
#: 0.346 D for CHNO.  Chlorine is deliberately NOT in this set: at 0.181 D over
#: eight compounds, with essentially no bias, it does not need the widening and
#: giving it one would flatter chlorinated candidates' error bars.
SOFT_ELEMENTS = frozenset({"S", "Se", "Br", "Te", "I"})

#: Multiplier applied to the error bar when a soft element is present.
#:
#: Seven sulfur and three bromine/iodine compounds are far too few to FIT a
#: multiplier on, so this is a round floor rather than a regression
#: coefficient.  It is set where it is because 2.0 is the smallest round value
#: at which every compound in the 79-compound set falls inside its own
#: two-sigma bound: at 1.0 two-sigma coverage is 92.4%, at 1.5 it is 96.2%, at
#: 2.0 it is 100%.  Bromine and iodine are additionally marked out of domain,
#: because a 1.9 D mean error is not rescued by an error bar.
SOFT_ELEMENT_FACTOR = 2.0


# --------------------------------------------------------------------------
# Conformer sampling
# --------------------------------------------------------------------------

#: Fixed ETKDG seed.  A content-addressed cache stores whatever this expert
#: returns the first time, so the answer has to be a function of the structure
#: alone.  A single MMFF conformer is not: 1,2-dichloroethane returns 2.80,
#: 2.80, 0.00, 2.80, 2.80 D under embedding seeds 1-5 (experiment 1.12), and
#: ethylenediamine returns 2.42, 2.47, 0.24, 0.49, 0.00.  Caching a coin flip
#: is worse than a wide bar.
EMBED_SEED = 0xF00D

#: Conformers requested per torsional degree of freedom, and the cap.
#:
#: Fixed at 16 for everything, the cost is paid on rigid molecules that cannot
#: use it (benzene: sixteen identical single points).  Scaling with the
#: molecule's own flexibility spends the budget where the spread is: measured
#: over the reference set through the shipped code path this costs 68 ms per
#: molecule, median 25 ms, with rigid chlorobenzene at 17 ms and flexible
#: ethylenediamine at 249 ms.
#:
#: The cap is where the ensemble stops being a population and starts being a
#: sample, and it BINDS FROM FOUR ROTORS UPWARD (32 / 8), not at the eight
#: where the domain score falls - so between four and eight rotors the ensemble
#: is truncated silently.  What that truncation costs was measured rather than
#: assumed, by raising the cap and re-running: 1-hexanol (five rotors) moves
#: 0.009 D, diglyme (six) moves 0.078 D and ethylenediamine (five) moves
#: 0.139 D, all below the 0.25 D floor on the error bar, which is why the
#: domain score is left alone until the spread term has more to say.
CONFORMERS_PER_TORSION = 8
MAX_CONFORMERS = 32

#: RMS pruning is switched OFF at the embedding stage, which is not the
#: obvious choice and was made on measurement.  With ``pruneRmsThresh = 0.3``
#: ethylenediamine collapses to three conformers no matter how many are asked
#: for, and *which* three depends on the seed, because the N-H torsions that
#: move its dipole barely move its heavy-atom RMS.  Its ensemble average then
#: varies by 0.876 D (one sigma) across embedding seeds - the very
#: irreproducibility the ensemble exists to remove.  Unpruned, that falls to
#: 0.204 D, and in every flexible case tested the seed-to-seed variation is
#: smaller than the conformer spread the prediction already reports (1,2-
#: dichloroethane 0.226 against a reported 1.118; ethylene glycol 0.103 against
#: 0.159; 1-butanol 0.011 against 0.073; rigid molecules exactly 0.000).
PRUNE_RMS_THRESHOLD = -1.0

#: SMARTS for a torsion that RDKit's rotatable-bond count does not see: the
#: bond from a heavy atom to a terminal O-H, N-H or S-H.  Those hydrogens are
#: invisible to a heavy-atom rotor count and dominate the dipole of an alcohol,
#: an amine or a diol.  Amide N-H is excluded: it is not a free rotor.
#:
#: It matches per HYDROGEN, not per rotor, so an -NH2 counts twice and water
#: counts twice although neither has two rotors (water has none).  That is
#: deliberate and it is a BUDGET heuristic, not a count of degrees of freedom:
#: the molecules it over-counts are exactly the ones whose dipole is most
#: conformer-sensitive, and the extra conformers are spent there.  The price is
#: paid on the rigid end - water gets sixteen identical single points, about
#: 15 ms wasted - and the number must not be reported to a chemist as a torsion
#: count, which is why the domain warning calls them sampled rotors.
TERMINAL_ROTOR_SMARTS = "[OX2,NX3,SX2;!$([NX3]C=O)]-[#1]"


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------

#: Floor on the one-sigma spread, in debye, and the term proportional to the
#: value.  ``std = sqrt(floor^2 + (rel * mu)^2 + spread^2) * soft_factor``.
#:
#: Two shape parameters, calibrated to put roughly 68% of the reference set
#: inside one sigma.  In-sample coverage of a calibrated bar is circular, so
#: the procedure was validated held out: over 3000 random 50/50 splits of the
#: 79 compounds, with the two parameters re-chosen on each training half and
#: coverage measured on the other half, in-sample one-sigma coverage 69.2%
#: falls to a HELD-OUT 68.4% (sd 10.5%), with held-out two-sigma coverage
#: 99.6%.  68.4% is the number to quote.  The fixed shipped pair gives 69.6%
#: one-sigma and 100% two-sigma over all 79 compounds, against the 68% / 95% a
#: correct estimate implies - erring slightly wide, as this repository prefers.
DIPOLE_STD_FLOOR_DEBYE = 0.25
DIPOLE_STD_RELATIVE = 0.15

_STD_BASIS = (
    "GFN2-xTB against 79 experimental gas-phase dipoles (CRC/NIST CCCBDB): MAE 0.430 D, "
    "RMSE 0.608 D. std = sqrt(0.25^2 + (0.15*mu)^2 + conformer_spread^2), doubled when a "
    "soft polarisable element (S, Se, Br, Te, I) is present; held-out one-sigma coverage "
    "68.4% over 3000 random 50/50 splits, 100% two-sigma over the whole set. NOT "
    "bias-corrected: the method overestimates by +0.341 D on average, and by +0.733 D "
    "above 3 D, so a pessimistic bound below the value is the conservative direction"
)

#: Why no bias correction is applied, although one would visibly improve the
#: mean absolute error.  Subtracting +0.341 D would be a correction fitted here
#: to handbook values with no independent test set, in a repository whose rule
#: 5 prefers a published method to an invention.  Worse, the bias is not
#: constant: -0.040 D below 1.5 D, +0.379 D from 1.5 to 3 D, +0.733 D above
#: 3 D, +0.660 D for sulfur and +1.929 D for bromine/iodine.  One subtraction
#: would be wrong in a new way for each class.  The published method's raw
#: value ships, the error bar covers the bias, and the direction of the lean is
#: stated so a ranker reading a pessimistic bound knows which way it leans.
_BIAS_NOTE = (
    "GFN2-xTB overestimates dipole moments: mean signed error +0.341 D over 79 reference "
    "compounds (+0.733 D above 3 D). No bias correction is applied, because the bias is "
    "structure-dependent and correcting it would be a fit to this handbook set"
)

#: Gas phase, recorded on every prediction because nothing downstream can tell.
_PHASE_NOTE = (
    "isolated molecule in vacuum; a condensed-phase effective dipole is enhanced by the "
    "reaction field, typically by 20-40% in a polar liquid"
)


# --------------------------------------------------------------------------
# Geometry quality
# --------------------------------------------------------------------------

#: Maximum GFN2 gradient, in eV/angstrom, an MMFF geometry may show before it
#: is treated as unfit for an electronic property.
#:
#: MMFF failure is silent and RDKit's own flag does not catch it:
#: ``MMFFHasAllMoleculeParams`` returns True for carbon dioxide and MMFF still
#: produces a 1.405 A C=O bond against a true 1.16 A.  Measured through this
#: code path, that CO2 shows a 13.5 eV/A GFN2 gradient, N2 15.9 and CS2 6.8,
#: while the worst of the 79 ordinary organics in the reference set is
#: formaldehyde at 2.43 and the median is well under 1.  The headroom is NOT
#: symmetric and it is worth knowing which side is thin: 3.0 sits only 1.23x
#: above the worst ordinary organic measured here (formaldehyde 2.43, and
#: cyanogen 2.64 outside the reference set) and 2.3x below the cheapest true
#: force-field failure (CS2 6.80, SO2 7.95, CO2 13.46, N2 15.91).  A stiff
#: small molecule with one more unit of gradient would be refused by this
#: threshold, which is the direction this repository prefers to err, but it is
#: a heuristic and not a proof: a geometry wrong enough to shift the dipole but
#: not enough to trip the gradient passes silently.
MAX_GRADIENT_EV_PER_ANGSTROM = 3.0


# --------------------------------------------------------------------------
# Domain
# --------------------------------------------------------------------------

#: Groups whose dipole GFN2's minimal basis over-polarises hardest, with the
#: measured error.  These are the two worst CHNOF cases in the reference set,
#: and a search that ranks on a large dipole preferentially surfaces exactly
#: them, so the method's worst cases concentrate at the top of the ranking.
#:
#: Each entry holds a TUPLE of SMARTS rather than one comma-joined string.
#: ``chem.has_substructure`` returns False for a pattern RDKit cannot compile,
#: so splitting a joined string on "," would silently disable any future
#: pattern that used a comma inside brackets - ``[NX3,NX4+](=O)[O-]`` splits
#: into two fragments that both fail to parse, and the flag would stop firing
#: with nothing to show for it.  ``test_the_overpolarised_patterns_compile``
#: keeps that honest.
_OVERPOLARISED_GROUPS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "nitro",
        ("[NX3](=O)=O", "[NX3+](=O)[O-]"),
        "nitrobenzene is over-predicted by +1.51 D and nitromethane by +0.70 D, the "
        "largest CHNOF errors in the reference set",
    ),
    (
        "perfluoroalkyl",
        ("[CX4]([F])([F])[F]",),
        "benzotrifluoride is over-predicted by +1.36 D; only two fluorinated compounds "
        "were validated, so this is a flag rather than a calibration",
    ),
)

#: Rotor count above which a 32-conformer ensemble is a sample of the conformer
#: population rather than the population.  The cap itself binds four rotors
#: earlier (see :data:`MAX_CONFORMERS`); this is where the truncation starts to
#: matter, measured by lifting the cap and re-running: at five or six rotors
#: the ensemble average moves 0.009 D (1-hexanol), 0.078 D (diglyme) and
#: 0.139 D (ethylenediamine), all inside the 0.25 D floor, while a
#: fourteen-rotor chain is sampling a few per cent of its conformers.
FLEXIBILITY_DOMAIN_LIMIT = 8

#: Fraction of the value the conformer spread may reach before the ensemble
#: average stops meaning much.  When the individual conformers' dipoles are
#: larger than their weighted mean, the mean is a cancellation and depends on
#: relative conformer energies GFN2 is not reliable for - ethylenediamine is
#: the worst case in the reference set and sits at a spread of 1.23 times its
#: own mean.
SPREAD_DOMAIN_LIMIT = 0.75

#: Why an unassigned double-bond configuration is refused and an unassigned
#: tetrahedral centre is not.
#:
#: ETKDG does not decline a double bond whose configuration the SMILES leaves
#: open: it builds one, and asked for several conformers it builds BOTH.  Eight
#: conformers of ``ClC=CCl`` come back as six cis and two trans here, so the
#: "conformer ensemble" would be an average over two different compounds.  The
#: two are 1.85 D apart - (Z)-1,2-dichloroethene 1.85 D against (E) 0.00 D
#: through this code path, and (Z)-crotononitrile 4.13 D against (E) 4.60 D -
#: while the bare SMILES silently returns whichever the seed happens to build
#: (1.85 D and 4.13 D respectively) with a 0.000 D conformer spread, i.e. an
#: error bar asserting the answer is settled.  No bar this expert states covers
#: 1.85 D, so this is a refusal rather than a widening.
#:
#: Unassigned TETRAHEDRAL centres are deliberately not refused.  |mu| is the
#: same for two enantiomers - butan-2-ol comes back at 1.97 D as (R) and 1.95 D
#: as (S), the difference being conformer sampling - and where two centres make
#: genuine diastereomers the measured difference is far inside the spread the
#: ensemble already reports: meso-2,3-butanediol 2.51 D against (R,R) 2.54 D on
#: a reported conformer spread of 1.1 D.  Refusing those would cost real
#: candidates (three of the fifty bundled reference compounds carry one
#: unassigned centre) for a difference the error bar already covers.
_STEREO_REFUSAL = (
    "the SMILES leaves the configuration of a stereogenic double bond unassigned, and E "
    "and Z are different compounds with different dipoles: measured through this code "
    "path, (E)-1,2-dichloroethene is 0.00 D and (Z) 1.85 D. The conformer generator picks "
    "one arbitrarily, or mixes both into one ensemble average, and the spread it reports "
    "does not admit to it. Write the geometry into the SMILES (Cl/C=C/Cl or Cl/C=C\\Cl) "
    "and this expert will answer"
)

#: The measured comparison behind the ``homo_lumo_gap`` refusal, kept next to
#: the code so a later re-addition has to argue with a number.
_GAP_REFUSAL = (
    "this expert deliberately does not report homo_lumo_gap. Measured here on 15 molecules "
    "against B3LYP/def2-SVP on identical geometries, the GFN2 tight-binding gap error "
    "flips sign with structure class: +3.90 eV mean for saturated sigma systems, -2.26 eV "
    "for pi-conjugated ones, whole-set MAE 2.92 eV over a -3.96 to +5.03 eV range. No "
    "single spread describes a signed, structure-dependent bias of that size"
)


# --------------------------------------------------------------------------
# Thread pinning
# --------------------------------------------------------------------------

#: Directory of the shared libraries the xtb wheel bundles.
_XTB_LIB_DIRNAME = "xtb.libs"

#: THREAD PINNING IS LOAD-BEARING, NOT AN OPTIMISATION.
#:
#: Out of the box this wheel runs a benzene single point in 99 ms on an idle
#: four-core box; pinning its *bundled* libgomp and OpenBLAS to one thread runs
#: the same calculation in 8 ms, a factor of twelve.  Over the whole 79-compound
#: reference set the gap is wider still - 5.3 s pinned against 416 s unpinned,
#: 67 ms against 5266 ms per molecule, though that unpinned figure was measured
#: with other work on the same four cores and so flatters the pin.  This expert
#: exists only because it is 100-1000x cheaper than DFT; unpinned, most of that
#: advantage is spent on OpenMP synchronisation over single points too small to
#: need it, and nothing fails - the run simply gets through less of the pool.
#:
#: ``ctypes.CDLL(None)`` does not reach these symbols: the wheel loads its
#: libraries RTLD_LOCAL, so they are not in the global namespace.  Setting
#: ``OMP_NUM_THREADS`` after import is equally useless, because libgomp has
#: already read it.  Re-opening each bundled library by path and calling its
#: own ``omp_set_num_threads``/``openblas_set_num_threads`` is the only route.
#: Verified to change results at the 1e-14 D and 1e-14 Eh level only
#: (summation order), so this is a cost measure with no scientific consequence,
#: and if the libraries cannot be found the expert still answers correctly -
#: just slowly, and it says so in a note.
_THREAD_ENTRY_POINTS = (
    ("omp_set_num_threads", "omp_get_max_threads"),
    ("openblas_set_num_threads", "openblas_get_num_threads"),
)


#: Resolved once. ``ctypes.CDLL`` never calls ``dlclose``, so re-opening the
#: same six libraries on every prediction would leak a dlopen reference per
#: call for the lifetime of the process.
_HANDLE_CACHE: list[ctypes.CDLL] | None = None


def _thread_handles() -> list[ctypes.CDLL]:
    """Every bundled xtb library that exposes a thread-count setter."""
    global _HANDLE_CACHE
    if _HANDLE_CACHE is not None:
        return _HANDLE_CACHE
    _HANDLE_CACHE = _load_thread_handles()
    return _HANDLE_CACHE


def _load_thread_handles() -> list[ctypes.CDLL]:
    try:
        import xtb
    except Exception:  # pragma: no cover - covered by is_available()
        return []
    libdir = pathlib.Path(xtb.__file__).resolve().parent.parent / _XTB_LIB_DIRNAME
    if not libdir.is_dir():
        return []
    handles: list[ctypes.CDLL] = []
    for path in sorted(libdir.iterdir()):
        if ".so" not in path.name:
            continue
        try:
            # RTLD_LOCAL: the library is already loaded, so this only takes a
            # reference and a dlsym handle. Promoting it to RTLD_GLOBAL would
            # change symbol resolution for everything else in the process.
            handle = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
        except OSError:
            continue
        if any(hasattr(handle, setter) for setter, _ in _THREAD_ENTRY_POINTS):
            handles.append(handle)
    return handles


@contextmanager
def single_threaded() -> Iterator[bool]:
    """Pin xtb's bundled OpenMP and BLAS to one thread, then restore.

    Yields True when the pin took effect.  This mutates state shared with
    anything else in the interpreter using those handles: it is saved and
    restored, but it is neither re-entrant nor safe if the engine ever
    dispatches experts concurrently inside one process.  A missed restore costs
    speed elsewhere, never correctness here.
    """
    # EVERY prior is read before ANY count is set, and each underlying setter is
    # touched once.  Both halves are load-bearing, and the reason is ugly:
    # ``dlsym`` searches a library's dependency tree, so once ``xtb.interface``
    # is imported - which this expert always does before pinning - the handle
    # for libxtb resolves ``omp_set_num_threads`` and
    # ``openblas_set_num_threads`` to the SAME addresses as libgomp's and
    # libopenblas's own handles.  Read-and-set in one pass then recorded four
    # entries for two real thread counts, the second pair reading a "prior" of
    # 1 that the first pair had just written, and the restore replayed them in
    # order and left the process pinned at one thread for good: 4, 4, 1, 1.
    # Measured, that is what happened - every interpreter that ran one
    # prediction stayed single-threaded afterwards, which costs pyscf, numpy and
    # OpenMM their threads and this expert nothing.  Deduplicating on the
    # resolved function pointer is what makes the prior a prior.
    entries: list[tuple[ctypes.CDLL, str, int]] = []
    seen: set[object] = set()
    for handle in _thread_handles():
        for setter, getter in _THREAD_ENTRY_POINTS:
            # A library is pinned only if its count can be READ BACK first.
            # Some builds export ``openblas_set_num_threads`` without
            # ``openblas_get_num_threads``; pinning one of those would leave it
            # at one thread for the life of the interpreter, because there is
            # no prior value to restore.
            if not hasattr(handle, setter) or not hasattr(handle, getter):
                continue
            try:
                address = ctypes.cast(getattr(handle, setter), ctypes.c_void_p).value
            except Exception:  # a stub handle in a test is not a ctypes symbol
                address = None
            key: object = address if address is not None else (id(handle), setter)
            if key in seen:
                continue
            try:
                prior = int(getattr(handle, getter)())
            except Exception:  # pragma: no cover - defensive
                continue
            if prior <= 0:  # pragma: no cover - defensive
                continue
            seen.add(key)
            entries.append((handle, setter, prior))

    saved: list[tuple[ctypes.CDLL, str, int]] = []
    for handle, setter, prior in entries:
        try:
            getattr(handle, setter)(1)
        except Exception:  # pragma: no cover - defensive
            continue
        saved.append((handle, setter, prior))
    try:
        # Report what was actually pinned, not what was merely found: the note
        # on the prediction is a claim about how the number was produced.
        yield bool(saved)
    finally:
        for handle, setter, prior in reversed(saved):
            try:
                getattr(handle, setter)(prior)
            except Exception:  # pragma: no cover - defensive
                pass


# --------------------------------------------------------------------------
# The calculation
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    """A Boltzmann-weighted conformer ensemble of GFN2-xTB single points."""

    #: Boltzmann-weighted |mu| over the ensemble, in debye.
    dipole_debye: float
    #: Boltzmann-weighted standard deviation of |mu|, in debye.
    spread_debye: float
    #: Conformers that produced a usable single point.
    conformers: int
    #: Conformers ETKDG was asked for.
    requested: int
    #: Largest GFN2 gradient component over the ensemble, in eV/angstrom.
    max_gradient: float
    #: Lowest total electronic energy, in hartree.
    min_energy_hartree: float
    #: Temperature the Boltzmann weights were formed at.
    temperature_k: float
    #: Whether the thread pin was in force.
    pinned: bool


def torsion_count(mol: object) -> int:
    """Torsional degrees of freedom, including terminal O-H / N-H / S-H rotors.

    RDKit's rotatable-bond count excludes a bond to a heavy atom of degree one,
    which throws away exactly the hydroxyl and amine torsions that dominate a
    small polar molecule's dipole.
    """
    from rdkit import Chem
    from rdkit.Chem import Lipinski

    rotatable = int(Lipinski.NumRotatableBonds(mol))  # type: ignore[arg-type]
    pattern = Chem.MolFromSmarts(TERMINAL_ROTOR_SMARTS)
    with_h = Chem.AddHs(mol)
    terminal = len(with_h.GetSubstructMatches(pattern)) if pattern is not None else 0
    return rotatable + terminal


def conformer_budget(mol: object) -> int:
    """How many conformers this molecule's flexibility earns."""
    torsions = torsion_count(mol)
    if torsions == 0:
        return 1
    return min(CONFORMERS_PER_TORSION * torsions, MAX_CONFORMERS)


def _embed(mol: object, n: int, seed: int) -> tuple[object, list[int]]:
    """ETKDGv3 embedding plus MMFF94 relaxation of every conformer."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    with_h = Chem.AddHs(mol)  # type: ignore[arg-type]
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = PRUNE_RMS_THRESHOLD
    ids = list(AllChem.EmbedMultipleConfs(with_h, numConfs=n, params=params))
    if not ids:
        # Cage-like and highly strained structures defeat distance-geometry
        # embedding from a clean start; random coordinates sometimes succeed.
        params.useRandomCoords = True
        ids = list(AllChem.EmbedMultipleConfs(with_h, numConfs=n, params=params))
    if not ids:
        return with_h, []
    try:
        AllChem.MMFFOptimizeMoleculeConfs(with_h, maxIters=500)
    except Exception:
        # MMFF94 has no parameters for some element combinations. UFF is worse,
        # but the GFN2 gradient check below is what actually decides whether
        # the geometry is usable, so falling back here is not a silent
        # substitution of an unvalidated method.
        try:
            AllChem.UFFOptimizeMoleculeConfs(with_h, maxIters=500)
        except Exception:
            pass
    return with_h, ids


def run_ensemble(
    smiles: str, *, temperature_k: float = DEFAULT_TEMPERATURE_K, seed: int = EMBED_SEED
) -> EnsembleResult:
    """Boltzmann-weighted GFN2-xTB dipole over a conformer ensemble.

    Raises ``ValueError`` when no geometry or no converged single point could
    be produced, and lets ``xtb``'s own ``XTBException`` through for a genuine
    SCF failure.  Callers turn both into a ``Prediction``.
    """
    import numpy as np
    from rdkit import Chem
    from xtb.interface import Calculator
    from xtb.libxtb import VERBOSITY_MUTED
    from xtb.utils import get_method

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse {smiles!r}")
    requested = conformer_budget(mol)
    with_h, ids = _embed(mol, requested, seed)
    if not ids:
        raise ValueError("conformer embedding produced no geometry")

    numbers = np.array([a.GetAtomicNum() for a in with_h.GetAtoms()], dtype=int)
    energies: list[float] = []
    dipoles: list[float] = []
    gradients: list[float] = []

    with single_threaded() as pinned:
        param = get_method("GFN2-xTB")
        for conf_id in ids:
            positions = np.asarray(with_h.GetConformer(conf_id).GetPositions(), dtype=float)
            calc = Calculator(param, numbers, positions / ANGSTROM_PER_BOHR)
            calc.set_verbosity(VERBOSITY_MUTED)
            result = calc.singlepoint()
            energy = float(result.get_energy())
            # xtb does NOT always raise on a nonsense result: two carbons
            # 0.01 A apart return E = +939 Eh and a meaningless dipole with no
            # exception. A converged GFN2 energy is strongly negative, so a
            # non-negative one is a failed calculation, not a high-energy one.
            if not math.isfinite(energy) or energy >= 0.0:
                continue
            energies.append(energy)
            dipoles.append(
                float(np.linalg.norm(np.asarray(result.get_dipole(), dtype=float)))
                * DEBYE_PER_AU
            )
            gradients.append(
                float(np.abs(np.asarray(result.get_gradient(), dtype=float)).max())
                * EV_PER_ANGSTROM_PER_HARTREE_BOHR
            )

    if not energies:
        raise ValueError(
            "no conformer produced a converged GFN2-xTB single point with a negative "
            "total electronic energy"
        )

    e = np.asarray(energies)
    mu = np.asarray(dipoles)
    weights = np.exp(-(e - e.min()) / (BOLTZMANN_HARTREE_PER_K * temperature_k))
    weights /= weights.sum()
    mean = float((weights * mu).sum())
    variance = float((weights * (mu - mean) ** 2).sum())
    return EnsembleResult(
        dipole_debye=mean,
        spread_debye=math.sqrt(max(variance, 0.0)),
        conformers=len(energies),
        requested=requested,
        max_gradient=float(max(gradients)),
        min_energy_hartree=float(e.min()),
        temperature_k=temperature_k,
        pinned=pinned,
    )


# --------------------------------------------------------------------------
# Structural screening
# --------------------------------------------------------------------------


def screen(smiles: str, *, declared_charge: int = 0) -> str | None:
    """Reason this molecule gets no dipole from this expert, or None.

    Pure Python and pure RDKit: nothing here may construct an xtb object,
    because the element check guards against a segmentation fault that no
    ``except`` can catch.

    ``declared_charge`` is :attr:`MoleculeSpec.charge`, which the candidate
    record carries separately from the SMILES.  It is read here because a
    candidate may declare an ion whose SMILES does not spell one, and the
    structure-only checks below would then pass a neutral molecule's dipole off
    as the ion's.  ``joback`` and ``quantum_electronic`` both consult the same
    field; an expert that ignored it would be the one place in the panel where
    a declared charge silently evaporates.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"RDKit could not parse the SMILES {smiles!r}"

    heavy = sorted(
        {a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() > MAX_ATOMIC_NUMBER}
    )
    if heavy:
        return (
            f"GFN2-xTB is parameterised up to atomic number {MAX_ATOMIC_NUMBER} (radon) and "
            f"this candidate contains {', '.join(heavy)}; the backend does not raise for "
            "these, it terminates the process, so the element set is screened before any "
            "calculation is attempted"
        )

    fragments = Chem.GetMolFrags(mol)
    if len(fragments) > 1:
        return (
            f"the candidate is {len(fragments)} disconnected fragments, which is an ion "
            "pair or a co-crystal rather than one molecule; its 'dipole' would scale with "
            "a separation the conformer generator picks arbitrarily"
        )

    charge = Chem.GetFormalCharge(mol)
    if declared_charge != 0 and declared_charge != charge:
        return (
            f"the candidate declares a net charge of {declared_charge:+d} while its SMILES "
            f"{smiles!r} carries {charge:+d}: the record contradicts itself, and a dipole "
            "computed from the structure as written would not be the declared species'. "
            "Spell the charge into the SMILES - and expect a refusal even then, because a "
            "charged species has no origin-independent dipole moment"
        )
    if charge != 0 or declared_charge != 0:
        net = charge or declared_charge
        return (
            f"net formal charge {net:+d}: the dipole moment of a charged species is "
            "origin-dependent and is therefore not a property of the molecule. Measured "
            "through this code path, the acetate anion returns 7.30 D as embedded, 31.31 D "
            "translated 5 A and 103.36 D translated 20 A, at identical energy"
        )

    radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    electrons = sum(a.GetAtomicNum() for a in Chem.AddHs(mol).GetAtoms()) - charge
    if radicals or electrons % 2:
        return (
            "open-shell species: xtb's spin argument is silently ignored here - the methyl "
            "radical returns E = -3.56252221 Eh identically at uhf = 0, 1 and 2 - so the "
            "dipole would come from a spin-restricted, spin-averaged density and would not "
            "be the doublet's dipole"
        )

    # Configuration, unlike conformation, is not something an ensemble may
    # average over: the two configurations are different compounds.
    stereogenic_bonds = {
        Chem.StereoType.Bond_Double,
        Chem.StereoType.Bond_Cumulene_Even,
    }
    for element in Chem.FindPotentialStereo(mol):
        if (
            element.type not in stereogenic_bonds
            or element.specified == Chem.StereoSpecified.Specified
        ):
            continue
        # A CUMULATED double bond is exempt, for the same reason an unassigned
        # tetrahedral centre is: an allene's two forms are enantiomers, not
        # E and Z, and |mu| cannot tell them apart.  Measured here, penta-2,3-
        # diene returns 0.2649 D from either axial form and from the SMILES
        # that specifies neither, and 2,4-difluoropenta-2,3-diene 0.0001 D from
        # both.  RDKit flags these as potential stereo anyway - it flags
        # methylallene, which has no stereoisomers at all - so without this the
        # screen would refuse candidates over a difference of 1e-4 D.
        if _is_cumulated(mol.GetBondWithIdx(int(element.centeredOn))):
            continue
        return _STEREO_REFUSAL
    return None


def _is_cumulated(bond: object) -> bool:
    """True when ``bond`` is one double bond of a cumulated (allene) system."""
    from rdkit import Chem

    for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):  # type: ignore[attr-defined]
        doubles = sum(1 for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
        if doubles > 1:
            return True
    return False


# --------------------------------------------------------------------------
# The expert
# --------------------------------------------------------------------------


class SemiempiricalElectronicExpert(Expert):
    """GFN2-xTB dipole moment from a Boltzmann-weighted conformer ensemble."""

    id = "xtb_electronic"
    version = "1"
    method = (
        "GFN2-xTB self-consistent tight binding with multipole electrostatics "
        "(Bannwarth, Ehlert & Grimme, J. Chem. Theory Comput. 15:1652, 2019), "
        "Boltzmann-averaged over an ETKDGv3/MMFF94 conformer ensemble"
    )
    family = PropertyFamily.ELECTRICAL
    supported_classes = frozenset({MaterialClass.MOLECULE})
    #: dipole_moment only. homo_lumo_gap is refused for every candidate, and
    #: that refusal is expressed here as absence so ``registry.uncovered()``
    #: keeps reporting the gap as uncovered. See the module docstring.
    supported_properties = frozenset({"dipole_moment"})
    dependencies: frozenset[str] = frozenset()

    # -- availability ------------------------------------------------------

    def is_available(self) -> bool:
        from formulate import chem

        if not chem.rdkit_available():
            return False
        try:
            from xtb.interface import Calculator  # noqa: F401
            from xtb.utils import get_method
        except Exception:
            return False
        return get_method("GFN2-xTB") is not None

    def unavailable_reason(self) -> str:
        if self.is_available():
            return ""
        from formulate import chem

        if not chem.rdkit_available():
            return "RDKit is not installed, so no conformer can be generated"
        return "the xtb package is not installed, or it does not expose GFN2-xTB"

    def _software(self) -> SoftwareEnvironment:
        from formulate import chem

        try:
            import xtb

            xtb_version = str(getattr(xtb, "__version__", "unknown"))
        except Exception:  # pragma: no cover - guarded by is_available()
            xtb_version = "unavailable"
        return SoftwareEnvironment.capture(rdkit=chem.rdkit_version(), xtb=xtb_version)

    # -- domain ------------------------------------------------------------

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        from rdkit import Chem

        from formulate import chem

        basis = (
            "GFN2-xTB is parameterised for Z <= 86; the dipole error bar quoted here was "
            "measured on 79 neutral closed-shell small organic molecules of "
            "H/C/N/O/F/P/S/Cl/Br/I with experimental gas-phase dipole moments"
        )

        smiles = candidate.molecule.smiles if candidate.molecule else None
        if smiles is None:
            return ApplicabilityDomain.outside(
                "candidate carries no molecule payload", basis=basis
            )
        mol = chem.mol_from_smiles(smiles)
        if mol is None:
            return ApplicabilityDomain.outside(
                f"RDKit could not parse the SMILES {smiles!r}", basis=basis
            )

        warnings: list[str] = []
        score = 1.0
        in_domain = True

        elements = {a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms()}

        unvalidated = sorted(elements - VALIDATED_ELEMENTS)
        if unvalidated:
            warnings.append(
                f"contains {', '.join(unvalidated)}, outside the neutral-organic set this "
                "expert's accuracy was measured on; GFN2 will return a number and nothing "
                "here says whether it is any good. The lanthanides in particular are "
                "handled f-in-core with one shared parameter set, so La and Lu are treated "
                "as the same element"
            )
            score = min(score, 0.25)
            in_domain = False

        if elements & {"Br", "I"}:
            warnings.append(
                "bromine or iodine present: mean absolute error 1.93 D over the three "
                "such compounds validated (iodomethane 3.67 D against 1.62 D measured). "
                "The doubled error bar does not make this a usable number"
            )
            score = min(score, 0.2)
            in_domain = False
        elif elements & SOFT_ELEMENTS:
            warnings.append(
                "soft polarisable element (S, Se or Te) present: mean absolute error "
                "0.66 D over seven sulfur compounds against 0.35 D for CHNO, which the "
                "doubled error bar covers"
            )
            score = min(score, 0.5)

        for name, patterns, evidence in _OVERPOLARISED_GROUPS:
            if any(chem.has_substructure(smiles, s) for s in patterns):
                warnings.append(f"{name} group: {evidence}")
                score = min(score, 0.5)
                in_domain = False

        torsions = torsion_count(mol)
        if torsions > FLEXIBILITY_DOMAIN_LIMIT:
            warnings.append(
                f"{torsions} sampled rotors (rotatable bonds plus each O-H, N-H and S-H "
                f"hydrogen): a {MAX_CONFORMERS}-conformer "
                "ensemble is a sample of the conformer population rather than the "
                "population, and the Boltzmann average inherits whatever GFN2 gets wrong "
                "about the relative conformer energies"
            )
            score = min(score, 0.5)

        return ApplicabilityDomain(
            score=score, in_domain=in_domain, warnings=tuple(warnings), basis=basis
        )

    # -- prediction --------------------------------------------------------

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        if prop != "dipole_moment":  # pragma: no cover - guarded by covers()
            return None

        candidate = request.candidate
        if candidate.molecule is None:
            return Prediction.unsupported(
                prop, self.id, "candidate carries no molecule payload"
            )
        smiles = candidate.molecule.smiles

        refusal = screen(smiles, declared_charge=int(candidate.molecule.charge))
        if refusal is not None:
            return Prediction.unsupported(prop, self.id, refusal)

        temperature = self._temperature(request)
        try:
            ensemble = run_ensemble(smiles, temperature_k=temperature)
        except ValueError as exc:
            return Prediction.failed(prop, self.id, str(exc))
        except Exception as exc:
            # A genuine SCF failure raises XTBException; report it as a failure
            # with the backend's own message rather than as a number.
            return Prediction.failed(prop, self.id, f"{type(exc).__name__}: {exc}")

        merged = self._geometry_domain(domain, ensemble)
        std = self._std(smiles, ensemble)
        notes = self._notes(ensemble)

        return self._make(
            prop,
            ensemble.dipole_debye,
            "debye",
            request,
            merged,
            std=std,
            kind=UncertaintyKind.COMBINED,
            basis=_STD_BASIS,
            notes=notes,
            conformers=ensemble.conformers,
            conformers_requested=ensemble.requested,
            conformer_spread_debye=round(ensemble.spread_debye, 6),
            boltzmann_temperature_K=temperature,
            embed_seed=EMBED_SEED,
            max_gradient_eV_per_angstrom=round(ensemble.max_gradient, 4),
            single_threaded=ensemble.pinned,
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _temperature(request: PredictionRequest) -> float:
        """Temperature for the Boltzmann weights."""
        stated = request.conditions.temperature
        if stated is None:
            return DEFAULT_TEMPERATURE_K
        value = float(stated.to("K").value)
        return value if value > 0.0 else DEFAULT_TEMPERATURE_K

    def _std(self, smiles: str, ensemble: EnsembleResult) -> float:
        from formulate import chem

        elements = chem.elements(smiles)
        factor = SOFT_ELEMENT_FACTOR if elements & SOFT_ELEMENTS else 1.0
        base = math.sqrt(
            DIPOLE_STD_FLOOR_DEBYE**2
            + (DIPOLE_STD_RELATIVE * ensemble.dipole_debye) ** 2
            + ensemble.spread_debye**2
        )
        return base * factor

    def _geometry_domain(
        self, domain: ApplicabilityDomain, ensemble: EnsembleResult
    ) -> ApplicabilityDomain:
        """Fold in what only the finished calculation can tell us."""
        extra: list[str] = []
        score = 1.0
        in_domain = True

        if ensemble.max_gradient > MAX_GRADIENT_EV_PER_ANGSTROM:
            extra.append(
                f"the force-field geometry is not a GFN2 stationary point: largest GFN2 "
                f"gradient {ensemble.max_gradient:.1f} eV/A against a "
                f"{MAX_GRADIENT_EV_PER_ANGSTROM:.0f} eV/A threshold (ordinary organics sit "
                "below 2.5). The electronic structure was evaluated on a geometry MMFF94 "
                "got wrong"
            )
            score = min(score, 0.2)
            in_domain = False

        if ensemble.spread_debye > SPREAD_DOMAIN_LIMIT * max(ensemble.dipole_debye, 1e-9):
            extra.append(
                f"the conformers' dipoles ({ensemble.spread_debye:.2f} D spread) are large "
                f"against their ensemble average ({ensemble.dipole_debye:.2f} D): the "
                "average is a cancellation, and it depends on relative conformer energies "
                "rather than on any one molecule's dipole"
            )
            score = min(score, 0.4)

        if not extra:
            return domain
        return domain.merged_with(
            ApplicabilityDomain(
                score=score,
                in_domain=in_domain,
                warnings=tuple(extra),
                basis="geometry and conformer-ensemble quality, measured after the fact",
            )
        )

    @staticmethod
    def _notes(ensemble: EnsembleResult) -> Sequence[str]:
        notes = [
            _PHASE_NOTE,
            _BIAS_NOTE,
            f"Boltzmann average of |mu| over {ensemble.conformers} GFN2-xTB single points "
            f"at {ensemble.temperature_k:.2f} K, weighted by GFN2 energies evaluated at "
            "the MMFF94 geometries rather than at GFN2 minima (relaxing every conformer "
            "first moves the average by -0.21 to +0.28 D on the cases measured here, "
            f"inside the stated bar); conformer spread {ensemble.spread_debye:.3f} D",
            _GAP_REFUSAL,
        ]
        if not ensemble.pinned:
            notes.append(
                "xtb's bundled OpenMP/BLAS libraries could not be pinned to one thread, so "
                "this ran roughly an order of magnitude slower than it should have. The "
                "value is unaffected; the throughput of a budgeted run is not"
            )
        return tuple(notes)
