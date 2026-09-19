"""Radius of gyration: a conformer ensemble for a molecule, an ideal chain for a polymer.

Two methods live under one property here and they are never blended.  The base
class dispatches on ``material_class`` and each branch refuses rather than
borrowing the other's answer: a polymer is not its repeat unit embedded in
vacuum, and a molecule has no ``<R^2>/M``.

**MOLECULE.**  ETKDGv3 distance-geometry embedding (Wang, Witek, Landrum and
Riniker, *J. Chem. Inf. Model.* **60** (2020) 2044), MMFF94 optimisation of
every conformer (Halgren, *J. Comput. Chem.* **17** (1996) 490), then a
Boltzmann average of the mass-weighted radius of gyration over the
deduplicated ensemble at the requested temperature with weights
``exp(-E_MMFF / RT)``.  Ensemble size follows the published rule of Ebejer,
Morris and Deane, *J. Chem. Inf. Model.* **52** (2012) 1146.  The Rg is
mass-weighted over every atom including hydrogens - the same definition
``physics/md/engine.py::_radius_of_gyration`` uses, so this expert and the
validator that may later check it compute the same quantity.

**POLYMER.**  The unperturbed (melt / theta) ideal chain, ``Rg^2 = <R^2>/6``
(Flory, *Statistical Mechanics of Chain Molecules*, Interscience, 1969), with
``<R^2> = (<R^2>/M) x M_n``.  ``<R^2>/M`` is read from
:data:`formulate.experts.mechanical.CHAIN_DIMENSIONS` (Fetters, Lohse and
Colby, "Chain Dimensions and Entanglement Spacings", *Physical Properties of
Polymers Handbook*, 2nd ed., Springer 2007, ch. 25) or, where nothing is
tabulated, from :func:`~formulate.experts.mechanical.predicted_chain_dimension`
- through ``PolymerMechanicalExpert._chain``, deliberately, so the two experts
cannot disagree about which chain dimension a repeat unit has.  Departure from
Gaussian statistics at short chain length is quantified with the wormlike-chain
result of Benoit and Doty, *J. Phys. Chem.* **57** (1953) 958.

Accuracy, all measured in this repository (see the module constants for the
runs):

* MMFF94 geometry against ten exactly known gas-phase geometries: 0.74% mean
  and 2.17% worst (CF4) over the eight it is entitled to, and 21.1% (CO2) and
  15.8% (CS2) over the two cumulated double bonds, which are refused.
* MMFF94 against a GFN2-xTB relaxation of its own minimum, over the fifty
  bundled reference compounds: 0.47% mean, 0.99% at the 90th percentile,
  1.73% worst.  Re-run in review, same protocol: 0.49% mean, 1.09% at the
  90th percentile, 1.93% worst (propylene carbonate, whose GFN2 relaxation
  repuckers the ring), with alpha-pinene at 2.15% the worst found outside
  that set.  The 2.2% geometry floor covers all of them and nothing to
  spare, which is why the classes it does *not* cover are refused by
  structure rather than averaged into it.
* Beyond the cumulated double bonds, three more classes where MMFF94 reports
  full parameters and is badly wrong were found in review and are refused:
  N=N and other heteroatom triple bonds (dinitrogen 32.9% high against GFN2
  and 33.0% against its experimental 1.0977 A bond), two-coordinate centres
  at any element rather than only at carbon (sulfur dioxide 14.7% high), and
  two triple-bonded atoms joined to each other (cyanogen 6.3%, diacetylene
  5.3%).  Open-shell species are refused outright: MMFF94 is a closed-shell
  parameterisation, and nitrogen dioxide comes out 10.3% high.
* Boltzmann weights against GFN2-xTB single points on the same geometries,
  over twenty-five molecules spanning zero to six hydrogen-bond donors: 2.6%
  worst with no donor, 0.3% worst with one, 15.7% worst with two or more
  (1,4-butanediol, where MMFF94's intramolecular hydrogen bond decides the
  populations).  That panel had no glyme in it, and a second panel run in
  review found the gauche effect outside it: 1,2-dimethoxyethane moves 8.1%
  and triglyme 12.0% on reweighting, with no hydrogen-bond donor to key it
  on, so a vicinal heteroatom pair now carries its own term.
* Chain dimension: ``Rg/sqrt(M)`` from the tabulated ``<R^2>/M`` reproduces the
  SANS coefficients to 1.9% over five polymers, which is semi-circular because
  both come from the Fetters compilation.  The structural correlation that
  covers untabulated repeat units is out by 8.2% on Rg at worst.
* Sampling: the answer at the production ensemble size, against the same
  calculation at 1000 embeddings, over twenty-nine molecules of 1 to 15
  rotatable bonds and fourteen with none.  The error is 0.00% for every molecule with no rotatable
  bond, including the ring systems whose pucker the ensemble does sample,
  and rises to 3.5% for a chain (octadecane) and 25.5% for tetraethylene
  glycol, where 200 embeddings miss three conformers 0.8 kcal/mol below
  everything they found.  The term the ensemble computes from the minima it
  has understates that distance by up to 2.54x, so it is widened 2.6x; see
  ``_SAMPLING_WIDENING``, which also says why that is a lower bound.

Limitations this method does not remove:

* The ensemble is in vacuum.  Reweighting the same geometries with GFN2-xTB
  plus implicit water moves the answer by up to 1.8% for a diol, and a
  re-relaxation in solvent - which was not run - would move it further.  A
  molecule in a melt or a crystal is not modelled at all.
* Potential-energy weights are not free-energy weights: conformational
  entropy, the vibrational contribution to each well and the multiplicity of
  symmetry-equivalent minima are all absent.
* The polymer number is a number-average unperturbed dimension.  Scattering
  reports a z-average, which for a most-probable dispersity-2 sample is larger
  by sqrt(3); and a good solvent swells the coil by an excluded-volume
  exponent.  Both are in the notes, neither is in the value.
"""

from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass

import numpy as np

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MonomerRole,
    PolymerTopology,
)
from formulate.core.conditions import Phase
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest

#: Gas constant in kcal/(mol K).  MMFF94 energies come back in kcal/mol, so the
#: Boltzmann weights are formed in that unit rather than converting twice.
R_KCAL = 1.987204259e-3

#: Ensemble size against rotatable-bond count, from Ebejer, Morris and Deane
#: (2012), who measured how many ETKDG embeddings are needed before the
#: ensemble stops moving.
#:
#: Their rule was calibrated on reproducing a bioactive conformation, not on
#: converging a Boltzmann average, and at these sizes it does not converge
#: this one.  Measured here at the size each molecule actually runs at,
#: against the same calculation at 1000 embeddings: glycerol (2 rotatable
#: bonds, 50 conformers) 1.7439 against 1.7609, 0.96%; octane (5, 50) 2.7938
#: against 2.8631, 2.4%; dodecane (9, 200) 3.9992 against 4.0976, 2.4%;
#: hexadecane (13, 300) 5.0623 against 5.1204, 1.1%.  Earlier drafts of this
#: comment quoted glycerol at 100 conformers and octane at 300 - neither is
#: the size those two molecules run at, so neither number described the
#: shipped configuration.
#:
#: The sizes are kept and the error bar is made to carry the gap, because the
#: cost is the reason they are what they are: 14 s for hexadecane at 300 and
#: 47 s at 1000, inside a search loop that evaluates dozens of candidates a
#: round.  ``_SAMPLING_WIDENING`` is what makes that honest.
_ENSEMBLE_SIZE = ((7, 50), (12, 200))
_ENSEMBLE_SIZE_MAX = 300

#: Above this many rotatable bonds the expert refuses.  The widening below is
#: measured up to fifteen rotatable bonds, and what evidence exists above that
#: is a five-seed scatter at a monoglyceride's seventeen - which is the
#: measurement this module has since shown cannot see a well that every seed
#: missed.  Beyond twenty there is nothing at all, and a wide bar with nothing
#: behind it would be a guess dressed as a measurement.
_MAX_ROTATABLE_BONDS = 20

#: Deduplication tolerances: two MMFF minima are the same minimum when they
#: agree in energy to this many kcal/mol *and* in Rg to this many angstrom.
#:
#: The textbook criterion is a symmetry-aware heavy-atom RMSD, and it is
#: unaffordable: measured here, ``GetAllConformerBestRMS`` costs 8.6 ms a pair
#: on octane, which is 44850 pairs and 385 s at 300 conformers against the 2 s
#: it took to build that ensemble.
#:
#: Merging on energy and Rg instead merges two genuinely distinct minima that
#: share both.  That does *not* leave the weighted mean where it was, which an
#: earlier version of this comment claimed: a pair of mirror-image gauche
#: conformers is two microstates with one energy and one radius, and giving
#: them one Boltzmann weight between them under-counts the folded populations
#: and biases a chain's radius high.  Measured here by re-weighting each
#: surviving minimum by the number of embeddings that collapsed onto it, which
#: is the cheapest proxy for its multiplicity: octane +0.44%, dodecane +0.56%,
#: glycerol +0.55%, dibutyl ether -0.26%, hexadecane -0.04%, 2,4-dimethylpentane
#: 0.00%.  So the bias is real, is not always in the same direction, and is
#: inside a fifth of the geometry floor - small enough to leave uncorrected and
#: too real to keep describing as zero.
_DEDUPE_ENERGY_KCAL = 0.02
_DEDUPE_RG_ANGSTROM = 0.02

#: Deduplication is not cosmetic.  ETKDG plus MMFF collapses many embeddings
#: onto the same minimum - 1000 embeddings of octane give 60 distinct ones -
#: and duplicates multiply the apparent sample size without moving the mean.
#: Measured on octane at 1000 embeddings the naive standard error claims 0.28%
#: where the deduplicated one says 2.44%; on glycerol, 0.29% against 2.04%.
#: Reporting the first would be exactly the flattering-narrow-bar failure the
#: repository names.

#: Random splits used by the split-half sampling term.  200 is where the RMS
#: difference stopped moving in the third decimal place.
_SPLIT_HALF_TRIALS = 200

#: Widening on the sampling term.
#:
#: The sampling term measures the minima that were found; it cannot measure one
#: that was never found, and a five-seed scatter at a fixed ensemble size
#: cannot either - every seed can miss the same well.  The measurement that
#: does see it is the answer at the production ensemble size against the same
#: calculation at 1000 embeddings, and against that the computed term is too
#: small by up to 2.54x:
#:
#: ======================  ====  ===  ========  ======  =====
#: molecule                rot.    N  distance    term  ratio
#: ======================  ====  ===  ========  ======  =====
#: octadecane                15  300     3.53%   1.39%   2.54
#: methyl laurate            10  200     1.94%   0.96%   2.02
#: nonane                     6   50     2.69%   1.39%   1.94
#: octane                     5   50     2.42%   1.25%   1.93
#: tetraethylene glycol      10  200    25.51%  13.29%   1.92
#: 1-dodecanol               10  200     1.93%   1.08%   1.78
#: dodecane                   9  200     2.40%   1.39%   1.73
#: methyl decanoate           7   50     2.14%   1.56%   1.37
#: dibutyl ether              6   50     2.91%   2.66%   1.10
#: hexadecane                13  300     1.13%   1.38%   0.82
#: heptane                    4   50     0.21%   3.32%   0.06
#: ======================  ====  ===  ========  ======  =====
#:
#: Twenty-nine molecules of 1 to 15 rotatable bonds were measured this way,
#: and fourteen with none.  The first ten flexible ones gave a worst ratio of
#: 1.93 (octane); the nineteen measured afterwards pushed it to 2.54
#: (octadecane), which is the honest way to read this number - it is the worst
#: of everything measured, not a factor that survived a held-out test, and the
#: next molecule could push it again.  2.6 covers all twenty-nine.
#: Tetraethylene glycol is the case that shows what the term is blind to: 200
#: embeddings miss three conformers 0.8 kcal/mol below everything they found,
#: and the answer moves from 2.95 to 3.96 A when they are found.
#:
#: All fourteen molecules with no rotatable bond sit at 0.00% - including
#: cyclohexane, cyclooctane, decalin, morpholine and propylene carbonate, whose
#: ring pucker the ensemble does sample and does converge on - and for benzene
#: the term is zero, so the widening changes nothing for a rigid molecule.
#:
#: This replaces a 1.3x widening above twelve rotatable bonds that was fitted
#: to one molecule's seed scatter and covered neither octadecane nor octane.
#: Two honest cautions.  The 1000-embedding reference is itself not converged
#: for the longest chains (hexadecane moves 1.1% between 300 and 1000), so
#: these distances are lower bounds and so is the factor.  And the factor is a
#: ratio of two measured quantities, both noisy, taken at its worst rather than
#: averaged, which makes it conservative for most molecules: heptane's real
#: error is 0.21% against a widened 8.6% bar.  That asymmetry is deliberate.
_SAMPLING_WIDENING = 2.6

#: Above this many rotatable bonds the ensemble is capped at
#: ``_ENSEMBLE_SIZE_MAX`` and stops growing with the molecule, which is worth
#: saying on the prediction even though the widening above is not keyed on it.
_FLEXIBLE_ROTATABLE_BONDS = 12

#: Error in the Boltzmann *weights* rather than in the geometries, keyed on
#: hydrogen-bond donor count in the manner of ``interfacial._ASSOCIATION_FACTOR``.
#:
#: Measured by reweighting the same MMFF ensemble with GFN2-xTB single points
#: over twenty-five molecules: worst 2.58% at zero donors (diethyl ether),
#: 0.34% at one (propylamine), 15.68% at two or more (1,4-butanediol, where
#: MMFF94 over-stabilises the intramolecular hydrogen bond and holds the chain
#: folded that GFN2 opens out).  The table takes the *worst* in each group, not
#: the mean, because this is a systematic error and averaging it away is what
#: produced the flattering bar the repository forbids.
#:
#: One donor inherits the donor-free figure rather than its own measured
#: 0.34%: seven compounds cannot support the claim that one donor is eight
#: times safer than none, and there is no mechanism that would make it so.
#: Three-or-more inherits the two-donor figure for the same reason in reverse -
#: it measured better (4.13% worst over four compounds) and more hydrogen
#: bonding is not a reason to trust MMFF more.
_ENERGY_ERROR_BY_DONORS = {0: 0.026, 1: 0.026, 2: 0.157}

#: Two electronegative substituents on the ends of an *acyclic* C-C bond: the
#: gauche effect, and the one place where the donor-keyed table above is not
#: enough.  MMFF94 has no hyperconjugative term, so it puts a 1,2-dialkoxyethane
#: in the wrong rotameric state entirely, and the molecule has no hydrogen-bond
#: donor to key that on.
#:
#: Measured the same way as the donor table - the same MMFF ensemble reweighted
#: with GFN2-xTB single points - the shift in the Boltzmann-averaged Rg is 8.1%
#: for 1,2-dimethoxyethane, 8.4% for 1,2-diethoxyethane, 5.3% for diglyme,
#: 12.0% for triglyme, 10.7% for tetraglyme and 5.1% for 1,2-difluoroethane,
#: against the 2.6% the donor-free entry claims.  A glyme is not an exotic
#: candidate for a formulation engine, so this is not a corner.
#:
#: The bond has to be acyclic for the effect to reach the radius: ethylene
#: carbonate has the same O-C-C-O and cannot rotate about it, and its measured
#: shift is 0.00%.  1,3-difluoropropane (0.99%) and dimethoxymethane (0.17%)
#: are not vicinal and are not matched.  Where the molecule also has two
#: hydrogen-bond donors the donor entry is larger and is what applies.
_GAUCHE_SMARTS = "[#7,#8,#9]-[#6X4]-!@[#6X4]-[#7,#8,#9]"
_GAUCHE_ENERGY_ERROR = 0.12

#: Floor on the geometry term, so a rigid molecule never claims a zero bar.
#: Two independent measurements set it.  Against eight exactly known gas-phase
#: geometries MMFF94's Rg is out by 0.74% mean and 2.17% worst (CF4: 1.2485
#: against 1.2220 from r(C-F) = 1.315 A).  Against a GFN2-xTB relaxation of its
#: own minimum over the fifty bundled reference compounds, 0.47% mean, 0.99% at
#: the 90th percentile, 1.73% worst (acetonitrile).  The molecules for which
#: this floor is known to fail are refused rather than covered by it.
_GEOMETRY_ERROR = 0.022

#: A cumulated double bond.  MMFF94 answers for these - RDKit's
#: ``MMFFHasAllMoleculeParams`` returns True for every one tested - and the
#: answer is badly wrong: CO2 comes out 21.1% above its experimental gas-phase
#: geometry and 22.9% above a GFN2-xTB relaxation, CS2 15.8% and 16.0%, methyl
#: isocyanate 7.1% against GFN2 and carbodiimide 4.1%.  The error is a wrong
#: bond length, so it survives any amount of conformer sampling and no width of
#: error bar would rescue it.  Nothing else in this expert catches it.
#:
#: The centre was restricted to carbon and is not any more.  Sulfur dioxide is
#: the same motif at sulfur and MMFF94 puts its radius 14.7% above GFN2 and
#: 14.3% above the experimental 1.4308 A / 119.3 degree geometry, which the
#: carbon-only pattern let through as an in-domain answer with a 3.7% bar.
#: The cost of generalising is the azide group, measured at +2.6% (hydrazoic
#: acid) and -1.5% (methyl azide): at the edge of the geometry floor rather
#: than outside it, and refused with the class rather than carved out of it on
#: two compounds.  No compound in the bundled reference set matches.
_CUMULENE_SMARTS = "[*X2](=*)=*"

#: A triple bond between two atoms that are both not carbon.  MMFF94 has no
#: bond-stretch parameters for one and falls back on its empirical rule, which
#: puts N#N at 1.46 A against an experimental 1.0977: a radius of gyration
#: 32.9% above GFN2 and 33.0% above experiment, reported as in-domain with a
#: 3.4% bar before this guard existed.  It is the worst error found anywhere
#: in this expert, and ``MMFFHasAllMoleculeParams`` is happy with it.
_HETERO_TRIPLE_SMARTS = "[!#6]#[!#6]"

#: Two triple-bonded atoms joined directly to each other: cyanogen, the
#: diacetylenes, dicyanoacetylene, propiolonitrile.  Against GFN2: cyanogen
#: +6.3%, diacetylene +5.3%, triacetylene +6.2%, dicyanoacetylene +6.4% - two
#: to three times the geometry floor, and systematic across the class.  A
#: single triple bond with an sp3 or aromatic neighbour is fine and is not
#: matched: acetonitrile +1.7%, 2-butyne +0.9%, phenylacetylene +0.6%,
#: propargyl alcohol +0.2%.  Malononitrile, where an sp3 carbon separates the
#: two nitriles, is +2.8% and is also not matched; it is inside the floor.
_CONJUGATED_SP_SMARTS = "[$([*]#[*])]-[$([*]#[*])]"

#: Spread between compilations on a *tabulated* ``<R^2>/M``, as a fraction.
#: Fetters' own compilation carries entries that differ by this much between
#: editions and between the scattering and rotational-isomeric-state routes;
#: Rg goes as the square root, so it enters halved.
_TABULATED_CHAIN_SPREAD = 0.05

#: ``|d ln<R^2>/dT|`` per kelvin, used as a bound for extrapolating a chain
#: dimension away from the temperature it was measured at.  Flory (1969)
#: tabulates temperature coefficients between 0.1e-3 and 1.1e-3 per K across
#: the measured polymers; the top of that range is used rather than a mean, and
#: halved on the way to Rg.  It was 1.0e-3 here, which is not the top of the
#: range the same sentence quotes - the bar it set was the only one in this
#: module that did not match the basis printed next to it.
_CHAIN_TEMPERATURE_COEFFICIENT = 1.1e-3

#: Span of the tabulated chain dimensions, used when the reference temperature
#: cannot be read out of the entry's prose source string.  298 to 413 K is what
#: the table actually covers, so the widest honest extrapolation is that span.
_CHAIN_TABLE_T_LOW = 298.0
_CHAIN_TABLE_T_HIGH = 413.0

#: All-trans projection of one backbone bond, in angstrom.  1.54 A of C-C at a
#: 109.5 degree valence angle projects 1.54 sin(54.75 deg) = 1.257.  It is used
#: only for the contour length that sets the Kuhn count, which gates a warning
#: and an uncertainty term and never the value, so the approximation it makes
#: for an Si-O or an aromatic backbone is tolerable.
_BOND_PROJECTION_ANGSTROM = 1.25

#: Kuhn segments below which a chain is not Gaussian enough to call this an
#: unperturbed dimension without saying so.  At ten the wormlike-chain
#: shortfall is 7.0%; at 2.6 - polystyrene at M_n 2000 - it is 22.1%.
_MIN_KUHN_SEGMENTS = 10.0

#: Excluded-volume exponent for a coil in a good solvent, quoted in the notes
#: so a caller cannot read the unperturbed value as a solution dimension:
#: Rg ~ M^0.588 there against M^0.5 here (des Cloizeaux and Jannink,
#: renormalisation group).  No expansion factor is quoted with it, because the
#: exponent alone does not give one - the prefactor is solvent-specific, and
#: the solvent is not part of the candidate, which is why a solution request is
#: refused rather than corrected.
_GOOD_SOLVENT_EXPONENT = 0.588


# ---------------------------------------------------------------------------
# The conformer ensemble
# ---------------------------------------------------------------------------


def ensemble_size(rotatable_bonds: int) -> int:
    """Number of ETKDG embeddings for a molecule with this many rotatable bonds."""
    for limit, size in _ENSEMBLE_SIZE:
        if rotatable_bonds <= limit:
            return size
    return _ENSEMBLE_SIZE_MAX


def geometry_refusal(smiles: str) -> str | None:
    """Why MMFF94's geometry for this molecule cannot be trusted, or ``None``.

    Every class here answers - ``MMFFHasAllMoleculeParams`` returns True for
    all of them - and answers wrongly by more than any conformer sampling or
    error bar could repair, because the error is in a bond length rather than
    in a population.  Each one is a measurement, quoted in the constant it
    comes from, not a precaution.
    """
    from formulate import chem

    if chem.has_substructure(smiles, _CUMULENE_SMARTS):
        return (
            "a cumulated double bond at a two-coordinate centre: MMFF94 puts CO2's "
            "radius of gyration 21.1% above its experimental gas-phase geometry, CS2's "
            "15.8% above it and SO2's 14.3% above it, while reporting that it has full "
            "parameters for all three. The error is a bond length, so no amount of "
            "sampling and no width of error bar repairs it. Use a quantum geometry for "
            "this molecule - the repository's QM validation path will produce one"
        )
    if chem.has_substructure(smiles, _HETERO_TRIPLE_SMARTS):
        return (
            "a triple bond between two atoms that are not carbon: MMFF94 has no "
            "bond-stretch parameters for one and falls back on its empirical rule, "
            "which puts N#N at 1.46 A against the experimental 1.0977 A and its radius "
            "of gyration 33% high. Use a quantum geometry for this molecule"
        )
    if chem.has_substructure(smiles, _CONJUGATED_SP_SMARTS):
        return (
            "two triple-bonded atoms joined directly to each other: MMFF94 is 5.3% to "
            "6.4% high on the radius of gyration of every member of this class measured "
            "here (cyanogen, diacetylene, triacetylene, dicyanoacetylene) against "
            "GFN2-xTB, two to three times the geometry error this expert claims. A "
            "single triple bond with an sp3 or aromatic neighbour is not affected and "
            "is not refused. Use a quantum geometry for this molecule"
        )
    mol = chem.mol_from_smiles(smiles)
    if mol is not None and any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return (
            "an open-shell species: MMFF94 is a closed-shell parameterisation (Halgren "
            "1996 fitted it to closed-shell organic molecules) and RDKit types a radical "
            "centre as its closed-shell neighbour rather than refusing, which puts NO2's "
            "radius of gyration 10.3% above GFN2-xTB. Give the closed-shell species, or "
            "take the geometry from an unrestricted quantum calculation"
        )
    return None


@dataclass(frozen=True, slots=True)
class ConformerEnsemble:
    """A deduplicated set of MMFF94 minima and their mass-weighted radii."""

    #: Mass-weighted Rg of each unique minimum, angstrom.
    radii: tuple[float, ...]
    #: MMFF94 energy of each, kcal/mol.
    energies: tuple[float, ...]
    #: Embeddings requested, before deduplication.
    embedded: int

    def boltzmann(self, temperature_k: float) -> tuple[float, float, float]:
        """Weighted mean Rg, weighted standard deviation, and effective count."""
        rg = np.asarray(self.radii, dtype=float)
        energy = np.asarray(self.energies, dtype=float)
        weights = np.exp(-(energy - energy.min()) / (R_KCAL * temperature_k))
        weights = weights / weights.sum()
        mean = float((weights * rg).sum())
        variance = float((weights * (rg - mean) ** 2).sum())
        n_eff = float(1.0 / float((weights**2).sum()))
        return mean, math.sqrt(max(variance, 0.0)), n_eff

    def split_half(self, temperature_k: float) -> float:
        """RMS difference between two random halves of the ensemble, halved.

        This is the term the weighted standard error cannot supply.  The
        standard error is computed over the minima that were found and says
        nothing about one that was never found; partitioning the ensemble asks
        instead how much the answer depends on *which* minima these are.
        """
        n = len(self.radii)
        if n < 4:
            return 0.0
        rg = np.asarray(self.radii, dtype=float)
        energy = np.asarray(self.energies, dtype=float)
        rng = np.random.default_rng(20240517)
        beta = 1.0 / (R_KCAL * temperature_k)
        diffs = np.empty(_SPLIT_HALF_TRIALS, dtype=float)
        index = np.arange(n)
        for trial in range(_SPLIT_HALF_TRIALS):
            perm = rng.permutation(index)
            half = n // 2
            means = []
            for part in (perm[:half], perm[half:]):
                e = energy[part]
                w = np.exp(-(e - e.min()) * beta)
                means.append(float((w * rg[part]).sum() / w.sum()))
            diffs[trial] = means[0] - means[1]
        return float(np.sqrt(np.mean(diffs**2)) / 2.0)


class EnsembleRefusal(Exception):
    """The ensemble could not be built for a reason a chemist would accept."""


def _mass_weighted_rg(mol: object, conf_id: int) -> float:
    positions = np.asarray(mol.GetConformer(conf_id).GetPositions(), dtype=float)  # type: ignore[attr-defined]
    masses = np.array([a.GetMass() for a in mol.GetAtoms()], dtype=float)  # type: ignore[attr-defined]
    total = masses.sum()
    centre = (positions * masses[:, None]).sum(axis=0) / total
    squared = (((positions - centre) ** 2).sum(axis=1) * masses).sum() / total
    return float(math.sqrt(max(squared, 0.0)))


def _dedupe(radii: np.ndarray, energies: np.ndarray) -> np.ndarray:
    """Indices of the distinct minima, lowest energy first."""
    keep: list[int] = []
    for i in np.argsort(energies):
        if any(
            abs(energies[i] - energies[j]) < _DEDUPE_ENERGY_KCAL
            and abs(radii[i] - radii[j]) < _DEDUPE_RG_ANGSTROM
            for j in keep
        ):
            continue
        keep.append(int(i))
    return np.asarray(keep, dtype=int)


@functools.lru_cache(maxsize=512)
def build_ensemble(smiles: str, n_conformers: int, seed: int = 20240517) -> ConformerEnsemble:
    """Embed, optimise, deduplicate.  Cached: the result is temperature-free.

    Raises :class:`EnsembleRefusal` with a reason rather than returning
    something unusable, so every failure mode reaches the caller as a refusal
    instead of a number.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise EnsembleRefusal(f"RDKit cannot parse the SMILES {smiles!r}")
    # A disconnected SMILES is not one molecule and has no single Rg.  ETKDG
    # does not say so: embedding '[Na+].[Cl-]' puts both ions at the origin and
    # the Rg comes back as exactly 0.0, and 'O.O' returns the radius of one
    # water.  Both pass the registry's (0, None) bounds gate, so this has to be
    # caught here.
    if len(Chem.GetMolFrags(mol)) != 1:
        raise EnsembleRefusal(
            "the SMILES contains more than one disconnected fragment; a salt or a "
            "hydrate has no single radius of gyration, and a vacuum embedding of one "
            "silently superimposes the fragments"
        )
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = -1.0  # deduplication happens after optimisation, not before
    params.useSmallRingTorsions = True
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    if len(ids) == 0:
        raise EnsembleRefusal(
            "ETKDG produced no conformer at all; the distance-geometry bounds could "
            "not be satisfied for this structure"
        )

    if not AllChem.MMFFHasAllMoleculeParams(mol):
        # No UFF fallback.  UFF is a different method with a different error,
        # and swapping it in under the same `method` string would be exactly
        # the silent substitution this repository forbids.
        raise EnsembleRefusal(
            "MMFF94 has no parameters for this molecule, so its conformer energies "
            "cannot be computed and the Boltzmann weights have no basis"
        )

    result = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000, numThreads=0)
    converged = np.array([c for c, _ in result], dtype=int)
    energies = np.array([e for _, e in result], dtype=float)
    # MMFFOptimizeMoleculeConfs signals "I could not do this" by returning
    # (-1, -1.0) rather than raising.  Consumed as an energy, -1.0 everywhere
    # makes every Boltzmann weight equal and yields a plausible unweighted
    # number from a calculation that did not happen.
    if np.all(converged == -1) and np.allclose(energies, -1.0):
        raise EnsembleRefusal(
            "MMFF94 returned its failure sentinel for every conformer; there are no "
            "energies to weight with"
        )
    if not np.isfinite(energies).all():
        raise EnsembleRefusal("MMFF94 returned a non-finite conformer energy")

    radii = np.array([_mass_weighted_rg(mol, int(i)) for i in ids], dtype=float)
    keep = _dedupe(radii, energies)
    return ConformerEnsemble(
        radii=tuple(float(x) for x in radii[keep]),
        energies=tuple(float(x) for x in energies[keep]),
        embedded=len(ids),
    )


# ---------------------------------------------------------------------------
# The ideal chain
# ---------------------------------------------------------------------------


def gaussian_rg_angstrom(r2_per_mass: float, molar_mass_g_mol: float) -> float:
    """``Rg = sqrt(<R^2>/6)`` with ``<R^2> = (<R^2>/M) M_n``, in angstrom.

    Reproduces the SANS coefficients of the Fetters compilation: 0.4564 for
    polyethylene against 0.46, 0.2699 for polystyrene against 0.275, 0.3821 for
    1,4-polybutadiene against 0.380, 0.2661 for PMMA against 0.270 and 0.2652
    for PDMS against 0.270 - which is a consistency check, not an independent
    one, because the tabulated ``<R^2>/M`` and those coefficients come from the
    same compilation.
    """
    return math.sqrt(r2_per_mass * molar_mass_g_mol / 6.0)


def wormlike_ratio(n_kuhn: float) -> float:
    """``Rg_wormlike / Rg_gaussian`` at ``n_kuhn`` Kuhn segments.

    Benoit and Doty (1953): a chain of finite contour length is smaller than
    the Gaussian limit because its first segments cannot curl.  0.994 at 132
    segments, 0.946 at 13.2, 0.776 at 2.6 - so the correction is negligible for
    a real polymer and decisive for an oligomer, which is why the oligomer is
    flagged rather than answered as if it were a chain.
    """
    if n_kuhn <= 0:
        return float("nan")
    ratio_squared = (
        1.0
        - 1.5 / n_kuhn
        + 1.5 / n_kuhn**2
        - (0.75 / n_kuhn**3) * (1.0 - math.exp(-2.0 * n_kuhn))
    )
    return math.sqrt(max(ratio_squared, 0.0))


@functools.lru_cache(maxsize=1)
def chain_dimension_spread() -> tuple[float, float]:
    """Worst held-out and worst overall ratio of predicted to tabulated ``<R^2>/M``.

    Computed from the table rather than written down, so the constant in the
    uncertainty basis cannot drift away from the data it describes.  Currently
    1.054 over the four validation-split polymers and 1.170 over all ten - the
    miss being polyethylene, one of the correlation's own *fit* polymers.  Four
    held-out points cannot exclude a miss the method demonstrably makes, so the
    expert reports the second number and states both.
    """
    from .mechanical import CHAIN_DIMENSIONS, predicted_chain_dimension

    held_out: list[float] = []
    overall: list[float] = []
    for unit, entry in CHAIN_DIMENSIONS.items():
        predicted = predicted_chain_dimension(unit)
        if predicted is None or predicted <= 0:
            continue
        ratio = max(predicted, entry.r2_per_mass) / min(predicted, entry.r2_per_mass)
        overall.append(ratio)
        if entry.split == "validation":
            held_out.append(ratio)
    return (max(held_out) if held_out else float("nan"), max(overall) if overall else float("nan"))


_SOURCE_TEMPERATURE = re.compile(r"at\s+(\d+(?:\.\d+)?)\s*K")


def source_temperature(source: str) -> float | None:
    """Reference temperature parsed out of a ``ChainDimension.source`` string.

    The table states it in prose ("polyethylene, melt at 413 K").  Where the
    prose does not state it, this returns None and the caller widens to the
    whole span of the table, which is the wider answer - a reworded source
    degrades the bar rather than the value.
    """
    match = _SOURCE_TEMPERATURE.search(source or "")
    return float(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------


class ConformationExpert(Expert):
    """Radius of gyration, by conformer ensemble or by ideal chain."""

    id = "conformation"
    version = "1"
    method = (
        "ETKDGv3 + MMFF94 Boltzmann-averaged conformer ensemble for a molecule "
        "(Wang et al., JCIM 60:2044, 2020; Halgren, JCC 17:490, 1996; ensemble size "
        "after Ebejer et al., JCIM 52:1146, 2012); unperturbed ideal chain "
        "Rg^2 = <R^2>/6 for a polymer (Flory 1969) on the Fetters chain dimensions, "
        "with the finite-chain correction of Benoit & Doty (1953)"
    )
    family = PropertyFamily.STRUCTURAL
    supported_classes = frozenset({MaterialClass.MOLECULE, MaterialClass.POLYMER})
    supported_properties = frozenset({"radius_of_gyration"})
    dependencies: frozenset[str] = frozenset()

    # -- availability ------------------------------------------------------

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "RDKit is not installed"

    def _software(self) -> SoftwareEnvironment:
        from formulate import chem

        return SoftwareEnvironment.capture(rdkit=chem.rdkit_version())

    # -- domain ------------------------------------------------------------

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        if candidate.material_class is MaterialClass.POLYMER:
            return self._polymer_domain(candidate)
        return self._molecule_domain(candidate)

    _MOLECULE_BASIS = (
        "neutral, connected organic molecules with at most 20 rotatable bonds, in "
        "vacuum; MMFF94's parameterisation set plus the conformer-convergence range "
        "measured here"
    )
    _POLYMER_BASIS = (
        "linear homopolymers of a stated number-average molar mass in the melt or at "
        "theta conditions; the Fetters chain-dimension compilation, or the side-group "
        "correlation that extends it"
    )

    def _molecule_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        from formulate import chem

        spec = candidate.molecule
        if spec is None:
            return ApplicabilityDomain.outside(
                "candidate carries no molecule", basis=self._MOLECULE_BASIS
            )
        mol = chem.mol_from_smiles(spec.smiles)
        if mol is None:
            return ApplicabilityDomain.outside(
                "candidate carries no parseable molecule", basis=self._MOLECULE_BASIS
            )

        refusal = geometry_refusal(spec.smiles)
        if refusal is not None:
            return ApplicabilityDomain.outside(refusal, basis=self._MOLECULE_BASIS)

        descriptors = chem.descriptors(spec.smiles)
        rotatable = int(descriptors.get("rotatable_bond_count", 0.0))
        warnings: list[str] = []
        score = 1.0

        if rotatable > _MAX_ROTATABLE_BONDS:
            return ApplicabilityDomain.outside(
                f"{rotatable} rotatable bonds: the sampling error of this ensemble was "
                f"measured here up to fifteen and no further, so there is no evidence to "
                f"support any error bar beyond {_MAX_ROTATABLE_BONDS}",
                basis=self._MOLECULE_BASIS,
            )
        if rotatable > _FLEXIBLE_ROTATABLE_BONDS:
            warnings.append(
                f"{rotatable} rotatable bonds against an ensemble capped at "
                f"{_ENSEMBLE_SIZE_MAX} conformers: the ensemble stops growing with the "
                f"molecule here, and a molecule this flexible has more torsional states "
                f"than the ensemble has members. The sampling term is doubled on every "
                f"prediction, which covered every case measured, but what it covers is "
                f"the distance to a 1000-embedding ensemble - not to a complete one"
            )
            score = min(score, 0.6)

        charge = int(descriptors.get("formal_charge", 0.0)) or spec.charge
        if charge:
            warnings.append(
                "non-zero formal charge: there is no gas-phase reference radius of "
                "gyration for an ion, and a vacuum ensemble folds the chain back onto "
                "its own charge with nothing to screen it"
            )
            score = min(score, 0.25)

        return ApplicabilityDomain(
            score=score,
            in_domain=score > 0.3,
            warnings=tuple(warnings),
            basis=self._MOLECULE_BASIS,
        )

    def _polymer_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        spec = candidate.polymer
        if spec is None:
            return ApplicabilityDomain.outside(
                "candidate carries no polymer", basis=self._POLYMER_BASIS
            )
        if spec.topology is not PolymerTopology.LINEAR:
            return ApplicabilityDomain.outside(
                f"{spec.topology.value} topology: a network has no finite chain to have a "
                "radius of gyration, and a branched, star, graft or dendritic chain is "
                "smaller than a linear one of the same mass by a branching factor that "
                "needs the number and length of the arms, which PolymerSpec does not carry",
                basis=self._POLYMER_BASIS,
            )
        backbone = [m for m in spec.monomers if m.role is not MonomerRole.END_GROUP]
        if len(backbone) != 1:
            return ApplicabilityDomain.outside(
                f"{len(backbone)} backbone repeat units: <R^2>/M for a copolymer is not "
                "the mole-weighted mean of its homopolymers' values, and nothing here can "
                "compute it",
                basis=self._POLYMER_BASIS,
            )
        if spec.number_average_molar_mass is None:
            return ApplicabilityDomain.outside(
                "no number-average molar mass: the same repeat unit spans orders of "
                "magnitude, polystyrene being 12 A at M_n 2000 and 85 A at 100000",
                basis=self._POLYMER_BASIS,
            )

        chain = self._chain_dimension(candidate)
        if chain is None:
            return ApplicabilityDomain.outside(
                "no chain dimension for this repeat unit: it is not tabulated, and the "
                "structural correlation needs a repeat unit RDKit can parse carrying "
                "exactly two [*] attachment points",
                basis=self._POLYMER_BASIS,
            )

        degree = self._degree_of_polymerisation(candidate)
        if degree is not None and degree < 1.0:
            return ApplicabilityDomain.outside(
                f"M_n is {degree:.2f} of one repeat unit: there is no chain here to have "
                "an unperturbed dimension",
                basis=self._POLYMER_BASIS,
            )

        warnings: list[str] = []
        score = 1.0
        if chain.split == "predicted":
            held_out, overall = chain_dimension_spread()
            warnings.append(
                f"<R^2>/M is predicted from side-group bulk rather than tabulated; the "
                f"correlation is out by {overall:.3f}x on <R^2>/M at worst over the whole "
                f"table ({math.sqrt(overall) - 1:.1%} on Rg)"
            )
            score = min(score, 0.6)

        n_kuhn = self._kuhn_segments(candidate, chain.r2_per_mass)
        if n_kuhn is not None and n_kuhn < _MIN_KUHN_SEGMENTS:
            warnings.append(
                f"only {n_kuhn:.1f} Kuhn segments: this chain is not Gaussian, and the "
                f"ideal-chain formula overstates its size by "
                f"{1.0 - wormlike_ratio(n_kuhn):.0%}"
            )
            score = min(score, 0.25)

        return ApplicabilityDomain(
            score=score,
            in_domain=score > 0.3,
            warnings=tuple(warnings),
            basis=self._POLYMER_BASIS,
        )

    # -- chain helpers -----------------------------------------------------

    def _chain_dimension(self, candidate: Candidate):
        """The chain dimension another expert would use for this repeat unit.

        Read through ``PolymerMechanicalExpert._chain`` deliberately: two
        experts disagreeing about a polymer's ``<R^2>/M`` would be worse than
        the coupling, because the entanglement mass and the coil size would
        then describe different chains.
        """
        from .mechanical import PolymerMechanicalExpert

        return PolymerMechanicalExpert()._chain(candidate)

    def _degree_of_polymerisation(self, candidate: Candidate) -> float | None:
        """``M_n`` in repeat units, or ``None`` when the repeat unit cannot be read."""
        from .mechanical import backbone_descriptors

        spec = candidate.polymer
        if spec is None or spec.number_average_molar_mass is None:
            return None
        backbone = [m for m in spec.monomers if m.role is not MonomerRole.END_GROUP]
        if len(backbone) != 1:
            return None
        found = backbone_descriptors(backbone[0].smiles)
        if found is None or found[1] <= 0:
            return None
        return spec.number_average_molar_mass.to("g/mol").value / found[1]

    def _kuhn_segments(self, candidate: Candidate, r2_per_mass: float) -> float | None:
        """``N_K = R_max^2 / <R^2>``, from the all-trans contour length."""
        from .mechanical import backbone_descriptors

        spec = candidate.polymer
        if spec is None or spec.number_average_molar_mass is None:
            return None
        backbone = [m for m in spec.monomers if m.role is not MonomerRole.END_GROUP]
        if len(backbone) != 1:
            return None
        found = backbone_descriptors(backbone[0].smiles)
        if found is None:
            return None
        n_bonds, repeat_mass, _ = found
        if repeat_mass <= 0:
            return None
        mn = spec.number_average_molar_mass.to("g/mol").value
        # n_bonds counts both bonds to the attachment points; chaining merges
        # them into one inter-repeat bond, so a repeat contributes n_bonds - 1.
        bonds_per_repeat = max(n_bonds - 1, 1)
        total_bonds = (mn / repeat_mass) * bonds_per_repeat
        r_max = total_bonds * _BOND_PROJECTION_ANGSTROM
        r2 = r2_per_mass * mn
        return r_max**2 / r2 if r2 > 0 else None

    # -- prediction --------------------------------------------------------

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        temperature = request.conditions.temperature_k
        if temperature is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "the radius of gyration is condition dependent and no temperature was "
                "given: the Boltzmann weights need one, and the answer moves 5.2% for "
                "dodecane between 200 and 600 K",
            )
        # Not a pedantic guard.  At T = 0 the Boltzmann exponent divides by zero,
        # numpy returns nan rather than raising, and the molecular branch used to
        # report that as "the ensemble produced a non-positive radius of gyration"
        # - a failure message describing the wrong failure.
        if temperature <= 0.0:
            return Prediction.unsupported(
                prop,
                self.id,
                f"a temperature of {temperature:g} K was requested: a Boltzmann average "
                "is undefined there, and the chain dimensions this expert reads for a "
                "polymer are measured between 298 and 413 K. Ask at a temperature the "
                "material exists at",
            )
        if request.candidate.material_class is MaterialClass.POLYMER:
            return self._predict_polymer(prop, request, domain, temperature)
        return self._predict_molecule(prop, request, domain, temperature)

    def _predict_molecule(
        self,
        prop: str,
        request: PredictionRequest,
        domain: ApplicabilityDomain,
        temperature: float,
    ) -> Prediction:
        from formulate import chem

        spec = request.candidate.molecule
        if spec is None:
            return Prediction.unsupported(prop, self.id, "candidate carries no molecule")
        refusal = geometry_refusal(spec.smiles)
        if refusal is not None:
            return Prediction.unsupported(prop, self.id, refusal)

        descriptors = chem.descriptors(spec.smiles)
        rotatable = int(descriptors.get("rotatable_bond_count", 0.0))
        if rotatable > _MAX_ROTATABLE_BONDS:
            return Prediction.unsupported(
                prop,
                self.id,
                f"{rotatable} rotatable bonds: the sampling error of this ensemble was "
                f"measured here up to fifteen, so beyond {_MAX_ROTATABLE_BONDS} there is "
                f"no measured basis for an error bar and a guessed one would not be one. "
                f"A molecule this flexible needs a dedicated conformer search - the "
                f"repository's molecular-dynamics validation path is where that lives",
            )

        # MoleculeSpec.conformer_count is honoured as a floor and never as a
        # reduction. A caller-supplied size smaller than the rule's is exactly
        # the under-converged ensemble whose error the measured bar understates.
        size = max(ensemble_size(rotatable), min(int(spec.conformer_count or 0), 1000))
        try:
            ens = build_ensemble(spec.smiles, size)
        except EnsembleRefusal as exc:
            return Prediction.unsupported(prop, self.id, str(exc))

        mean, spread, n_eff = ens.boltzmann(temperature)
        if not math.isfinite(mean) or mean <= 0:
            return Prediction.failed(
                prop, self.id, "the ensemble produced a non-positive radius of gyration"
            )

        standard_error = spread / math.sqrt(n_eff) if n_eff > 0 else 0.0
        split = ens.split_half(temperature)
        # Both of these see only the minima that were found.  The factor is the
        # measured distance between what they claim and what the answer moves by
        # when the ensemble is taken to 1000 embeddings.
        sampling = _SAMPLING_WIDENING * max(standard_error, split) / mean
        capped = rotatable > _FLEXIBLE_ROTATABLE_BONDS

        donors = int(descriptors.get("hbd", 0.0))
        energy_error = _ENERGY_ERROR_BY_DONORS[min(donors, 2)]
        gauche = chem.has_substructure(spec.smiles, _GAUCHE_SMARTS)
        if gauche:
            energy_error = max(energy_error, _GAUCHE_ENERGY_ERROR)
        relative = math.sqrt(sampling**2 + energy_error**2 + _GEOMETRY_ERROR**2)

        notes = [
            f"Boltzmann average over {len(ens.radii)} distinct MMFF94 minima from "
            f"{ens.embedded} ETKDG embeddings, at {temperature:.1f} K",
            "gas-phase ensemble: reweighting the same geometries with GFN2-xTB and "
            "implicit water moved the answer by up to 1.8% for a diol here, and a "
            "re-relaxation in solvent was not run; a molecule in a melt or a crystal "
            "is not modelled",
            "the weights are potential-energy weights, not free-energy weights, so "
            "conformational entropy and the multiplicity of symmetry-equivalent minima "
            "are absent; for a long alkane this over-weights the all-trans minimum",
        ]
        if gauche:
            notes.append(
                "a vicinal heteroatom pair on an acyclic C-C bond: MMFF94 has no "
                "hyperconjugative term, so it gets the gauche/anti populations of a "
                "glyme or a 1,2-dihaloethane wrong rather than slightly wrong - "
                "reweighting with GFN2-xTB moves triglyme by 12%, which is what the "
                "energy term carries here"
            )
        if capped:
            notes.append(
                f"{rotatable} rotatable bonds against an ensemble capped at "
                f"{_ENSEMBLE_SIZE_MAX} conformers: the sampling term is widened like "
                "every other, but it is measured against a 1000-embedding ensemble "
                "rather than a complete one, and this molecule has more torsional "
                "states than either"
            )
        phase = request.conditions.phase
        if phase in (Phase.LIQUID, Phase.SOLID, Phase.SOLUTION, Phase.MELT):
            notes.append(
                f"the requested phase is {phase.value}, and this ensemble is in vacuum; "
                "the value is the isolated-molecule dimension"
            )

        return self._make(
            prop,
            mean,
            "angstrom",
            request,
            domain,
            std=mean * relative,
            kind=UncertaintyKind.COMBINED,
            basis=(
                f"quadrature of three measured terms, {relative:.1%} of the value in all. "
                f"Sampling {sampling:.1%}: the larger of the weighted standard error over "
                f"{n_eff:.1f} effective distinct minima and a {_SPLIT_HALF_TRIALS}-split "
                f"half-ensemble difference, widened {_SAMPLING_WIDENING:g}x because both "
                f"measure the minima that were found and neither measures one that was "
                f"never found - against a 1000-embedding ensemble they are short by up to "
                f"2.54x over the twenty-nine flexible molecules measured here"
                + f". Energy {energy_error:.1%}: the worst shift in the Boltzmann-averaged "
                f"Rg when the same ensemble is reweighted with GFN2-xTB single points, "
                f"measured over 25 molecules and keyed on {donors} hydrogen-bond donor(s)"
                + (
                    ", and on the vicinal heteroatom pair on an acyclic C-C bond that "
                    "MMFF94 has no hyperconjugative term for - the gauche effect, 12.0% "
                    "worst over six glymes and 1,2-difluoroethane"
                    if gauche
                    else ""
                )
                + f". "
                f"Geometry {_GEOMETRY_ERROR:.1%}: MMFF94's worst error against ten exactly "
                f"known gas-phase geometries (2.17%, CF4) and against a GFN2-xTB "
                f"relaxation over the fifty reference compounds (1.93% worst, propylene "
                f"carbonate), with the structural classes it fails outright refused rather "
                f"than covered by this number"
            ),
            notes=tuple(notes),
            temperature_k=temperature,
            ensemble_size=size,
            distinct_minima=len(ens.radii),
            rotatable_bonds=rotatable,
        )

    def _predict_polymer(
        self,
        prop: str,
        request: PredictionRequest,
        domain: ApplicabilityDomain,
        temperature: float,
    ) -> Prediction:
        spec = request.candidate.polymer
        if spec is None:
            return Prediction.unsupported(prop, self.id, "candidate carries no polymer")
        if spec.topology is not PolymerTopology.LINEAR:
            return Prediction.unsupported(
                prop,
                self.id,
                f"{spec.topology.value} topology: a network has no finite chain to have a "
                "radius of gyration at all, and a branched, star, graft or dendritic chain "
                "is smaller than a linear one of the same mass by a branching factor "
                "(0.78 for a three-arm star) that needs the number and length of the arms, "
                "which PolymerSpec does not carry",
            )
        backbone = [m for m in spec.monomers if m.role is not MonomerRole.END_GROUP]
        if len(backbone) != 1:
            return Prediction.unsupported(
                prop,
                self.id,
                f"{len(backbone)} backbone repeat units: <R^2>/M for a random copolymer is "
                "not the mole-weighted mean of its homopolymers' values, and nothing here "
                "can compute it",
            )
        if request.conditions.phase is Phase.SOLUTION:
            return Prediction.unsupported(
                prop,
                self.id,
                "the requested phase is solution, and this is the unperturbed melt/theta "
                "dimension. The expansion factor to a good-solvent coil needs the solvent, "
                "which is not part of the candidate",
            )
        if spec.number_average_molar_mass is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "no number-average molar mass is stated: the same repeat unit spans orders "
                "of magnitude, polystyrene being 12 A at M_n 2000 and 85 A at 100000",
            )

        chain = self._chain_dimension(request.candidate)
        if chain is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "no chain dimension for this repeat unit: nothing is tabulated for it and "
                "the structural correlation needs a repeat unit RDKit can parse carrying "
                "exactly two [*] attachment points",
            )

        mn = spec.number_average_molar_mass.to("g/mol").value
        if mn <= 0:
            return Prediction.failed(prop, self.id, "number-average molar mass is not positive")

        # A "chain" lighter than one repeat unit is not a short chain, it is not
        # a chain.  <R^2> = (<R^2>/M) M_n goes on returning a number for it - 1.9 A
        # for polystyrene at M_n 50 - and the wormlike term turns into an 80% error
        # bar rather than a refusal, which is a number where there should be none.
        degree = self._degree_of_polymerisation(request.candidate)
        if degree is not None and degree < 1.0:
            return Prediction.unsupported(
                prop,
                self.id,
                f"M_n = {mn:.0f} g/mol is less than one {backbone[0].smiles} repeat unit "
                f"({degree:.2f} of one): there is no chain here to have an unperturbed "
                f"dimension. If this is the monomer, ask for it as a molecule",
            )
        value = gaussian_rg_angstrom(chain.r2_per_mass, mn)

        # -- uncertainty, all three terms on Rg rather than on <R^2> --------
        held_out, overall = chain_dimension_spread()
        if chain.split == "predicted":
            chain_term = math.sqrt(overall) - 1.0
            chain_basis = (
                f"the side-group correlation is out by {overall:.3f}x on <R^2>/M at worst "
                f"over the whole table ({chain_term:.1%} on Rg); the held-out figure over "
                f"the four validation polymers is a flattering {math.sqrt(held_out) - 1:.1%} "
                f"and is not used, because the worst miss is polyethylene, one of the "
                f"correlation's own fit polymers"
            )
        else:
            chain_term = math.sqrt(1.0 + _TABULATED_CHAIN_SPREAD) - 1.0
            chain_basis = (
                f"{_TABULATED_CHAIN_SPREAD:.0%} on the tabulated <R^2>/M for the spread "
                f"between compilations, halved by the square root to {chain_term:.1%} on Rg"
            )

        reference_t = source_temperature(chain.source)
        if reference_t is None:
            delta_t = max(
                abs(temperature - _CHAIN_TABLE_T_LOW), abs(temperature - _CHAIN_TABLE_T_HIGH)
            )
            temperature_basis = (
                f"the entry's source states no temperature, so the whole "
                f"{_CHAIN_TABLE_T_LOW:.0f}-{_CHAIN_TABLE_T_HIGH:.0f} K span of the table is "
                f"used, {delta_t:.0f} K"
            )
        else:
            delta_t = abs(temperature - reference_t)
            temperature_basis = (
                f"{delta_t:.0f} K from the {reference_t:.0f} K the chain dimension was "
                f"measured at"
            )
        temperature_term = 0.5 * _CHAIN_TEMPERATURE_COEFFICIENT * delta_t

        n_kuhn = self._kuhn_segments(request.candidate, chain.r2_per_mass)
        if n_kuhn is None:
            gaussian_term = 0.0
            gaussian_basis = "no backbone bond count, so no finite-chain term"
        else:
            gaussian_term = max(1.0 - wormlike_ratio(n_kuhn), 0.0)
            gaussian_basis = (
                f"a wormlike chain of {n_kuhn:.1f} Kuhn segments is {gaussian_term:.1%} "
                f"smaller than the Gaussian limit (Benoit & Doty 1953), folded in one-sided"
            )

        relative = math.sqrt(chain_term**2 + temperature_term**2 + gaussian_term**2)

        notes = [
            "this is the unperturbed melt/theta dimension, not the coil in a good "
            f"solvent: there Rg ~ M^{_GOOD_SOLVENT_EXPONENT} rather than M^0.5 and the "
            "coil is larger by an expansion factor that depends on the solvent",
            f"Rg^2 = <R^2>/6 for an ideal chain, with <R^2>/M = {chain.r2_per_mass:.3f} "
            f"A^2 per (g/mol) [{chain.source}] and M_n = {mn:.0f} g/mol",
            "a number-average dimension for a chain of the stated M_n; scattering reports "
            "the z-average, which for a most-probable dispersity-2 sample is larger by "
            "sqrt(3) = 1.73x for reasons that have nothing to do with this model",
            "tacticity is not in the number: an isotactic and a syndiotactic vinyl polymer "
            "have different <R^2>/M and the table carries one value",
        ]
        if n_kuhn is not None:
            notes.append(f"{n_kuhn:.1f} Kuhn segments at this chain length")
        if chain.split == "predicted":
            notes.append(
                "<R^2>/M is predicted from side-group bulk, not tabulated for this repeat "
                "unit"
            )

        return self._make(
            prop,
            value,
            "angstrom",
            request,
            domain,
            std=value * relative,
            kind=UncertaintyKind.COMBINED,
            basis=(
                f"quadrature of three terms, {relative:.1%} of the value in all. "
                f"Chain dimension {chain_term:.1%}: {chain_basis}. "
                f"Temperature {temperature_term:.1%}: |d ln<R^2>/dT| is bounded at "
                f"{_CHAIN_TEMPERATURE_COEFFICIENT:.1e} per K across the polymers Flory "
                f"tabulates, halved for Rg and applied over {temperature_basis}. "
                f"Finite chain {gaussian_term:.1%}: {gaussian_basis}"
            ),
            notes=tuple(notes),
            temperature_k=temperature,
            r2_per_mass=chain.r2_per_mass,
            number_average_molar_mass_g_mol=mn,
            chain_dimension_split=chain.split,
        )
