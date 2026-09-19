"""Quantum electronic structure: B3LYP/6-31G(d,p) on a GFN2-xTB-relaxed ensemble.

The electrical family had no expert at all.  This one fills part of it with a
first-principles calculation rather than a correlation: RDKit ETKDGv3 embeds a
conformer ensemble, MMFF94 pre-optimises and pre-filters it, GFN2-xTB relaxes
the survivors, and PySCF evaluates B3LYP/6-31G(d,p) single points on the
relaxed structures.  Dipole moment is a Boltzmann-weighted ensemble average;
frontier gap and atomization energy come from the lowest conformer, the latter
against free-atom references computed at the same level of theory in their
ground-state multiplicities.

Why this level of theory, each point settled by a measurement made here:

* **DFT rather than Hartree-Fock.**  HF/6-31G(d,p) is seven times cheaper on
  ethanol here (0.8 s against 5.9 s) but omits electron correlation entirely,
  so it cannot produce an atomization energy at all, and the two methods
  disagree by 8.6 eV on the same frontier gap for that molecule: HF Koopmans
  17.86 eV against B3LYP Kohn-Sham 9.26 eV.  B3LYP is also the level whose
  error statistics are published for all three observables, so its uncertainty
  can be defended from literature and then checked locally, which is what
  :data:`VALIDATION` records.
* **p functions on hydrogen.**  Measured here at a fixed GFN2-xTB geometry,
  6-31G(d) to 6-31G(d,p) halves the water atomization error (-63.0 to
  -33.5 kJ/mol, -6.5% to -3.4%) and takes ammonia from -2.4% to -0.4%.  Every
  large outlier was an X-H bond to an electronegative atom, which is exactly
  what the missing p function costs.  It is about 1.8x the work and buys the
  property that was worst behaved.
* **No diffuse functions, deliberately.**  Measured on ethanol at one geometry,
  the lowest virtual orbital sits at +2.14 eV in 6-31G(d), +0.15 eV in
  6-31+G(d) and -0.35 eV in 6-311++G(d,p) while the HOMO moves only 0.53 eV:
  the LUMO is collapsing toward the continuum because a saturated alcohol has
  no bound anion.  A basis without diffuse functions cannot represent a
  continuum state, which forces the lowest virtual to stay valence-like and is
  the only reason a Kohn-Sham "gap" is a reproducible number here.  The same
  absence is why anions are refused.
* **The geometry the QM runs on.**  An ETKDG + MMFF94 geometry is not fit for a
  bond energy and MMFF's own parameter check does not say so: measured here,
  MMFF94 gives carbon dioxide a C=O bond of 1.405 A against the true 1.160 A
  and an atomization energy 369.3 kJ/mol (22.7%) low, while
  ``MMFFHasAllMoleculeParams`` returns True and the optimiser reports
  convergence.  GFN2-xTB fixes the bond to 1.144 A and the error to -1.5%.
  Hydrogen fluoride is the other failure mode: ``MMFFHasAllMoleculeParams`` is
  False, MMFF silently does nothing, and the H-F distance stays at the raw
  embedding's 0.983 A against a true 0.917 A.  A full B3LYP geometry
  optimisation would be better and is unaffordable: the ASE/BFGS driver in
  :mod:`formulate.physics.qm.pyscf_backend` needs 30-50 single points per
  conformer.
* **What each property does when GFN2-xTB is missing, decided per property by
  measuring the same substitution.**  Running the identical B3LYP single point
  on the MMFF94 geometry instead of the xTB one costs, measured here: at most
  0.205 D on a dipole (nitrobenzene; water 0.053, acetonitrile 0.045, acetone
  0.086), 4.16 eV on a frontier gap (carbon dioxide, 7.23 against 11.39 eV) and
  369.3 kJ/mol on an atomization energy (carbon dioxide again).  Against each
  property's own error bar that is a widening for the dipole, and larger than
  the whole bar for the other two - 4.16 eV against a 3.69 eV gap error, 369
  against a 20 kJ/mol floor - so the gap and the atomization energy refuse
  instead.  Crucially the refusal is on whether the xTB relaxation *happened*,
  not on whether the xtb package imports: a GFN2-xTB run that raises or fails
  to converge leaves exactly the MMFF geometry behind.

What the numbers are and are not:

* The frontier gap is a **Kohn-Sham orbital-energy difference**, not an optical
  or a fundamental gap, and a hybrid functional underestimates the fundamental
  gap systematically.  The bias is one-signed and several electronvolts wide;
  the uncertainty says so rather than pretending to precision.
* The atomization energy is the **electronic, bottom-of-well** value.  A
  thermochemical atomization enthalpy is smaller by the zero-point energy
  (117 kJ/mol for methane, about 200 for ethanol).  That is a definitional
  offset with a known sign, not an error, so it is stated rather than buried
  inside a symmetric error bar.
* The dipole is the **isolated-molecule vacuum** value.  A condensed phase
  enhances it by 20-40% for hydrogen-bonding liquids (water is 1.85 D isolated
  and about 2.9 D in the liquid), so this is not a solvent-polarity descriptor.

Two states this expert is not allowed to assume it is in:

* **A closed-shell singlet.**  Every SCF here is restricted, and RDKit reports
  zero radical electrons for dioxygen, so the radical and electron-count
  refusals do not catch a molecule whose ground state is a triplet.  Measured
  here, O=O converges as a closed-shell singlet with a 1.93 eV Kohn-Sham gap -
  comfortably above the static-correlation mark - and returns an atomization
  energy of 355 kJ/mol against a true De of 505, inside a 20 kJ/mol bar.  So a
  small computed gap now buys a second, decisive calculation rather than only a
  domain flag: a UKS triplet single point at the same geometry.  For O=O that
  triplet is 164 kJ/mol *below* the singlet, which settles the question instead
  of estimating it.  It is run only below :data:`TRIPLET_PROBE_GAP_EV`, so the
  molecules that pay for it are the ones that need it.
* **Main-group.**  6-31G(d,p) is defined through zinc, which is not the same as
  being usable there: it carries no 3d-optimised polarisation or semi-core
  correlation set, and B3LYP's errors on open-d-shell systems are a different
  order from its errors on the small organics every constant below was measured
  on.  Left unchecked, chromyl chloride returned a dipole of 0.68 D with the
  0.144 D organic bar attached to it.  The d block is refused by name.

``electronic_energy`` is registered but deliberately **not** covered.  An
absolute electronic energy is monotone in electron count, so ranking on it
sorts candidates by size, and :mod:`formulate.physics.qm.base` already refuses
to attach a standard deviation to one.  Claiming it would make the registry
report the electrical family covered when it is not.

References
----------
Becke, A. D., J. Chem. Phys. 98 (1993) 5648; Lee, Yang and Parr, Phys. Rev. B
37 (1988) 785 (B3LYP).  Hariharan and Pople, Theor. Chim. Acta 28 (1973) 213;
Hehre, Ditchfield and Pople, J. Chem. Phys. 56 (1972) 2257 (6-31G(d,p)).
Bannwarth, Ehlert and Grimme, J. Chem. Theory Comput. 15 (2019) 1652
(GFN2-xTB).  Wang, Witek, Landrum and Riniker, J. Chem. Inf. Model. 60 (2020)
2044 (ETKDGv3).  Halgren, J. Comput. Chem. 17 (1996) 490 (MMFF94).  Sun et
al., J. Chem. Phys. 153 (2020) 024109 (PySCF).  Koopmans, Physica 1 (1934)
104; Perdew and Levy, Phys. Rev. Lett. 51 (1983) 1884 (why an orbital-energy
difference is not a fundamental gap).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.conditions import Conditions
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, Quantity, UncertaintyKind

from .base import Expert, PredictionRequest

#: Exchange-correlation functional and Gaussian basis.  Both appear in the
#: cache keys and in every provenance record, because an atomization energy is
#: a difference between two calculations and the difference is only meaningful
#: while both sides used exactly this pair.
FUNCTIONAL = "b3lyp"
BASIS = "6-31g**"
#: PySCF's spelling of 6-31G(d,p); the two strings build an identical basis,
#: checked here (24 contracted functions for water either way).
BASIS_LABEL = "6-31G(d,p)"

#: Gas constant, J/(mol K), for the Boltzmann weights over the ensemble.
R_GAS = 8.31446261815324

#: Ensemble temperature used when the request states none.  A dipole average
#: needs conformer populations and populations need a temperature; 298.15 K is
#: where the reference dipoles were measured.
DEFAULT_TEMPERATURE_K = 298.15

#: Heavy atoms above which this expert refuses.
#:
#: Measured on these four cores: aspirin, 13 heavy atoms and 222 contracted
#: basis functions, took 38.5 s for one B3LYP/6-31G(d,p) single point;
#: naphthalene at 10 heavy atoms took 19.8 s and benzene at 6 took 6.8 s.  A
#: three-conformer ensemble plus free-atom references at 13 heavy atoms is
#: therefore minutes per candidate, and a search evaluates hundreds.  A
#: molecule that size also carries four or more rotatable bonds, so three
#: conformers sample a few per cent of its rotamers: cost and correctness point
#: the same way.
MAX_HEAVY_ATOMS = 12

#: Conformers embedded, and the most that reach B3LYP.
#:
#: Twenty embeddings cost milliseconds and give the MMFF pre-filter something
#: to choose from; three B3LYP points is what the cost cap above allows.  The
#: gap between the two is not hidden: when more distinct conformers sit inside
#: the thermal window than were computed, the prediction is marked
#: under-sampled and its conformer term widened.
N_EMBED = 20
MAX_CONFORMERS = 3

#: RMSD below which two embeddings are the same conformer, Angstrom.  ETKDG's
#: own pruning threshold; 0.5 A is the usual value and separates rotamers
#: without splitting one minimum into several.
RMSD_PRUNE = 0.5

#: Energy above the MMFF minimum, kJ/mol, within which a conformer is counted
#: as thermally populated.  10 kJ/mol is about 4 RT at room temperature, i.e.
#: a Boltzmann weight of ~2%: below that a conformer cannot move a weighted
#: average enough to matter.
THERMAL_WINDOW_KJ = 10.0

#: A bond length that moved more than this between MMFF94 and GFN2-xTB,
#: Angstrom, means the force field had no idea what the molecule was.
#:
#: Carbon dioxide is the case that earned this guard: MMFF94 puts its C=O at
#: 1.405 A, GFN2-xTB at 1.144 A, a 0.26 A shift, and MMFF reports success
#: throughout.  0.1 A is far outside the 0.01-0.03 A a force field and a
#: tight-binding method normally differ by for a bond they both know.
BOND_SHIFT_ALARM_ANGSTROM = 0.1

#: Elements 6-31G(d,p) is defined for in PySCF, probed here rather than assumed.
#:
#: The probe matters: the basis stops at zinc.  Bromine, selenium and iodine
#: are *not* covered, which contradicts the usual description of 6-31G as a
#: first- and second-row basis extended through the 3d block.  Covering iodine
#: needs a def2 basis with an effective core potential, which would invalidate
#: every error constant measured below, so those elements are refused instead.
BASIS_ELEMENTS = frozenset(
    {
        "H", "He",
        "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
        "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    }
)

#: The part of that set this expert refuses anyway.
#:
#: "Defined in the basis" and "usable at this level of theory" are different
#: claims, and only the first was probed.  6-31G(d,p) reaches zinc as a plain
#: valence basis with no 3d-optimised polarisation and no semi-core correlation
#: functions, and B3LYP on an open d shell has static-correlation errors an
#: order of magnitude beyond anything in the reference sets below - all of
#: which are first- and second-row organics.  Measured here, the gap left
#: chromyl chloride returning a 0.68 D dipole carrying the 0.144 D bar fitted
#: to small organics, and in-domain at score 1.0.  Nothing about that number is
#: supported, so it is refused rather than flagged.
D_BLOCK_ELEMENTS = frozenset(
    {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"}
)

#: SCF cycles before a calculation is declared not converged.  A non-converged
#: SCF still leaves an energy in memory, and the backend contract in
#: physics/qm/base.py refuses to report it; this expert propagates that refusal
#: as FAILED rather than substituting a number.
MAX_SCF_CYCLES = 200

#: Below this Kohn-Sham gap, eV, a nominally closed-shell molecule is showing
#: static correlation that one determinant does not have.  The candidate is
#: marked out of domain for *all three* properties, not just the gap, because
#: the same wavefunction produced all of them.  1.5 eV is roughly where a
#: singlet-triplet instability starts to appear in B3LYP for organic molecules.
SMALL_GAP_EV = 1.5

#: Below this Kohn-Sham gap, eV, the closed-shell assumption stops being
#: assumed and gets tested against a UKS triplet single point at the same
#: geometry.
#:
#: The threshold only decides who pays for the extra SCF; the test itself is
#: decisive, since a triplet that comes out lower is not a tighter estimate but
#: a different electronic state.  3.0 eV is set by measurement from both sides:
#: the lowest gap anywhere in the eight-molecule frontier set is
#: p-benzoquinone's 3.80 eV, so no molecule that validated this expert is
#: probed, while dioxygen - the case the radical and electron-count refusals
#: both miss, because RDKit calls O=O closed shell - computes 1.93 eV and is.
#: Raising it costs time on molecules that do not need it; lowering it risks
#: missing a triplet, which is the expensive direction.
TRIPLET_PROBE_GAP_EV = 3.0

#: kcal/mol -> kJ/mol, for the MMFF energies RDKit reports.
KCAL_TO_KJ = 4.184


# ---------------------------------------------------------------------------
# Measured error constants
# ---------------------------------------------------------------------------
#
# Every number below was produced by running *this* protocol - ETKDGv3, MMFF94
# pre-filter, GFN2-xTB relaxation, B3LYP/6-31G(d,p) single point - against
# experimental reference data, not read out of a paper about a similar method
# and not taken from the SCF convergence threshold.  The SCF converges to
# microhartrees and is wrong by tenths of an electronvolt; quoting convergence
# would be dishonest by orders of magnitude.

#: One-sigma method error on the dipole moment, debye.
#:
#: HELD OUT, and this is the held-out number.  Twenty molecules with gas-phase
#: experimental dipoles (CRC/NIST) were split on RDKit's rotatable-bond count:
#: the thirteen rigid ones set the constant and the seven flexible ones - the
#: ones whose dipole depends on which rotamer you optimised into - tested it
#: and were never used to fit it.  In sample the rigid set gives 0.127 D; the
#: held-out flexible set gives the 0.144 D quoted here, larger, which is the
#: direction an honest split is expected to go.  Quoting the 0.127 would be
#: quoting the number the constant was fitted to reproduce.
DIPOLE_METHOD_STD_DEBYE = 0.144

#: Fraction of the held-out dipole set that fell inside its own one-sigma
#: bound (five of seven), against the ~68% a correct estimate implies.  The
#: same criterion ``formulate calibrate`` applies to the rest of the panel.
#: The two misses are diethyl ether (out by 0.149 D) and nitrobenzene (0.253),
#: so the bar is missed narrowly rather than being systematically too tight.
DIPOLE_HELD_OUT_COVERAGE = 5.0 / 7.0

#: Extra dipole uncertainty, debye, when the MMFF94 geometry has to stand in
#: for the GFN2-xTB one.
#:
#: Measured on this protocol by running the identical B3LYP single point on
#: both geometries.  The polar molecules move 0.053 D (water), 0.045 D
#: (acetonitrile), 0.086 D (acetone) and 0.205 D (nitrobenzene); benzene and
#: carbon dioxide stay at zero by symmetry, which is no evidence either way and
#: is left out.  0.12 D is the r.m.s. over the four that carry information, and
#: it is deliberately not the 0.09 D that the first three alone suggested:
#: nitrobenzene is larger than all of them and a constant fitted to the three
#: smallest polar molecules in the set would have under-claimed it.
#:
#: This is the only one of the three properties where the substitution is
#: survivable.  The same swap costs 4.16 eV on a gap and 369 kJ/mol on a bond
#: energy - in both cases more than the whole error bar - so those two refuse.
MMFF_GEOMETRY_DIPOLE_STD_DEBYE = 0.12

#: What the same substitution costs a frontier gap, eV, and a bond energy,
#: kJ/mol.  Both measured on carbon dioxide, whose C=O bond MMFF94 puts at
#: 1.405 A against GFN2-xTB's 1.144: the B3LYP gap is 7.23 eV on the force
#: field geometry and 11.39 eV on the relaxed one, and the atomization energy
#: is 369.3 kJ/mol (22.7%) low.  Quoted in the refusals so a reader can see the
#: size of what is being declined rather than being asked to take it on trust.
MMFF_GEOMETRY_GAP_SHIFT_EV = 4.16
MMFF_GEOMETRY_ATOMIZATION_SHIFT_KJ_MOL = 369.3

#: One-sigma error on the Kohn-Sham frontier gap against the experimental
#: fundamental gap (photoelectron IP minus electron-transmission EA), eV, and
#: the mean signed deviation that proves the bias is one-signed.
#:
#: Measured over eight conjugated molecules where both an IP and an EA exist:
#: benzene, pyridine, naphthalene, thiophene, pyrrole, styrene, p-benzoquinone
#: and nitrobenzene.  Every one came out low, from -3.46 eV to -4.35 eV.  The
#: signed number is what a reader needs: the true fundamental gap is *larger*
#: than this expert reports by about this much, not merely uncertain by it.
GAP_RMSE_EV = 3.692
GAP_MEAN_SIGNED_EV = -3.681

#: Scatter of the gap deviation about its own mean, eV.
#:
#: Worth separating from the RMSE above, because it says something the RMSE
#: hides: almost all of the 3.69 eV is one constant offset, and the Kohn-Sham
#: gap is reproducible to about a quarter of an electronvolt *relative to other
#: numbers from this expert*.  It is not corrected for, because the registry
#: defines this property as the orbital-energy difference and because an offset
#: fitted to eight molecules is not a law.
GAP_RESIDUAL_SCATTER_EV = 0.279

#: Atomization energy uncertainty: max(floor, fraction x value).
#:
#: Measured against twelve bottom-of-well De references built from ATcT/CODATA
#: enthalpies of formation at 0 K plus literature zero-point energies, so the
#: reference is the same quantity the calculation produces.  The r.m.s. error
#: was 2.19% and 19.1 kJ/mol; the two constants are those rounded outward.
#:
#: The two-part form is not decoration, and the reason is the opposite of the
#: one usually given.  The *largest* errors in this set are on the smallest
#: molecules - hydrogen fluoride -36.7 kJ/mol over one bond, water -33.5 over
#: two, both X-H bonds to a very electronegative atom - while ethane manages
#: +26.6 kJ/mol over seven.  So a purely relative bar collapses to 12 kJ/mol on
#: exactly the molecules that are worst, and the floor is what stops it.  The
#: relative term takes over above about 800 kJ/mol, where per-bond error does
#: start to accumulate.  Coverage over the twelve is 10/12, wide rather than
#: narrow; the two misses are the fluoride and the water.
ATOMIZATION_RELATIVE_STD = 0.025
ATOMIZATION_FLOOR_KJ_MOL = 20.0

#: What the constants above were measured on, kept so the claims cannot rot
#: silently and so a reader can see the size of the sets behind them.  Each
#: entry is (label, n, statistic).
VALIDATION: tuple[tuple[str, int, str], ...] = (
    ("dipole, rigid molecules (in sample)", 13, "RMSE 0.127 D"),
    ("dipole, flexible molecules (held out)", 7, "RMSE 0.144 D, 1-sigma coverage 5/7"),
    (
        "frontier gap vs IP-EA",
        8,
        "RMSE 3.692 eV, mean signed -3.681 eV, scatter about the bias 0.279 eV",
    ),
    (
        "atomization vs bottom-of-well De",
        12,
        "RMS 2.19% / 19.1 kJ/mol, worst -6.2% (HF), 1-sigma coverage 10/12",
    ),
)


# ---------------------------------------------------------------------------
# Ensemble machinery
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Ensemble:
    """Scalars extracted from a conformer ensemble.

    Deliberately holds no PySCF or RDKit object: this is what the module-level
    cache stores, and retaining a ``Mole`` per candidate would defeat the point
    of having a cap on molecule size.
    """

    symbols: tuple[str, ...]
    #: B3LYP total electronic energy per computed conformer, J/mol.
    energies: tuple[float, ...]
    dipoles: tuple[float, ...]
    gaps: tuple[float | None, ...]
    #: Boltzmann weights over ``energies`` at ``temperature_k``, summing to 1.
    weights: tuple[float, ...]
    temperature_k: float
    geometry_source: str
    xtb_used: bool
    mmff_used: bool
    #: Distinct conformers inside the thermal window, and how many reached B3LYP.
    populated: int
    computed: int
    #: Largest bond-length change between MMFF94 and GFN2-xTB, Angstrom.
    max_bond_shift: float | None
    #: E(closed-shell singlet) - E(triplet) at the lowest conformer, J/mol,
    #: when the gap was small enough to earn the probe.  Positive means the
    #: triplet is the lower state and this whole ensemble solved the wrong one.
    #: ``None`` means the question was not asked, not that it was answered.
    triplet_below_singlet_j: float | None = None
    diagnostics: tuple[str, ...] = ()
    failure: str | None = None

    @property
    def wrong_spin_state(self) -> bool:
        return self.triplet_below_singlet_j is not None and self.triplet_below_singlet_j > 0.0

    @property
    def under_sampled(self) -> bool:
        return self.populated > self.computed

    @property
    def lowest(self) -> int:
        return min(range(len(self.energies)), key=lambda i: self.energies[i])

    def mean_dipole(self) -> float:
        """Root-mean-square dipole over the ensemble.

        The r.m.s. rather than the arithmetic mean, because that is what a
        gas-phase dielectric measurement of the molar polarisation returns: the
        Debye equation is linear in ``mu^2``.  A microwave Stark measurement
        instead reports one conformer's dipole, so the two are not
        interchangeable for a flexible molecule - 1,2-dichloroethane is 0 D
        anti and about 3.1 D gauche - which is why the choice is stated rather
        than assumed.
        """
        return math.sqrt(sum(w * mu * mu for w, mu in zip(self.weights, self.dipoles)))

    def dipole_spread(self) -> float:
        """Population-weighted standard deviation of the dipole, debye.

        Exactly zero for a rigid molecule, and the dominant uncertainty term
        for a flexible one whose rotamers differ in polarity.
        """
        mean = sum(w * mu for w, mu in zip(self.weights, self.dipoles))
        var = sum(w * (mu - mean) ** 2 for w, mu in zip(self.weights, self.dipoles))
        return math.sqrt(max(var, 0.0))


def _boltzmann(energies: tuple[float, ...], temperature_k: float) -> tuple[float, ...]:
    if not energies:
        return ()
    lowest = min(energies)
    rt = R_GAS * max(temperature_k, 1.0)
    raw = [math.exp(-(e - lowest) / rt) for e in energies]
    total = sum(raw)
    return tuple(r / total for r in raw)


def _bond_lengths(symbols: tuple[str, ...], positions: Any, bonds: tuple[tuple[int, int], ...]):
    import numpy as np

    arr = np.asarray(positions, dtype=float)
    return [float(np.linalg.norm(arr[i] - arr[j])) for i, j in bonds]


def build_ensemble(
    smiles: str,
    *,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
    max_conformers: int = MAX_CONFORMERS,
    n_embed: int = N_EMBED,
    seed: int = 0xF00D,
) -> Ensemble:
    """Embed, pre-filter, relax and evaluate a conformer ensemble.

    Returns an :class:`Ensemble` whose ``failure`` is set rather than raising,
    so a caller can turn a geometry or SCF problem into a refusal with a reason
    instead of a stack trace.
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from formulate import chem
    from formulate.physics.geometry import MolecularGeometry
    from formulate.physics.qm.base import QMMethod, QMRequest
    from formulate.physics.qm.pyscf_backend import PySCFBackend
    from formulate.physics.qm.xtb_backend import XTBBackend

    empty = dict(
        symbols=(), energies=(), dipoles=(), gaps=(), weights=(),
        temperature_k=temperature_k, geometry_source="", xtb_used=False,
        mmff_used=False, populated=0, computed=0, max_bond_shift=None,
    )

    mol = chem.mol_from_smiles(smiles)
    if mol is None:
        return Ensemble(**empty, failure=f"RDKit cannot parse the SMILES {smiles!r}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    params.pruneRmsThresh = RMSD_PRUNE
    conformer_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=max(1, n_embed), params=params))
    if not conformer_ids:
        if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
            return Ensemble(
                **empty,
                failure=f"ETKDGv3 produced no 3D conformer for {smiles!r}",
            )
        conformer_ids = [0]

    diagnostics: list[str] = []
    mmff_used = bool(AllChem.MMFFHasAllMoleculeParams(mol))
    if mmff_used:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=1000)
        ranked = sorted(
            (float(energy), cid)
            for (_converged, energy), cid in zip(results, conformer_ids)
        )
    else:
        # MMFF silently does nothing when it has no parameters - hydrogen
        # fluoride is the case that showed it - so say so rather than letting a
        # raw embedding pass for an optimised structure.
        diagnostics.append(
            "MMFF94 has no parameters for this molecule, so the conformers are raw "
            "ETKDGv3 embeddings and were not energy-ranked before selection"
        )
        ranked = [(0.0, cid) for cid in conformer_ids]

    floor = ranked[0][0]
    window = [(e, cid) for e, cid in ranked if (e - floor) * KCAL_TO_KJ <= THERMAL_WINDOW_KJ]
    selected = window[: max(1, max_conformers)]

    bonds = tuple(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()
    )
    symbols = tuple(atom.GetSymbol() for atom in mol.GetAtoms())
    charge = Chem.GetFormalCharge(mol)

    xtb = XTBBackend()
    pyscf_backend = PySCFBackend()
    xtb_used = xtb.is_available()

    energies: list[float] = []
    dipoles: list[float] = []
    gaps: list[float | None] = []
    geometries: list[MolecularGeometry] = []
    max_shift: float | None = None

    for rank, (_energy, cid) in enumerate(selected):
        conformer = mol.GetConformer(cid)
        positions = np.array(
            [list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
            dtype=float,
        )
        geometry = MolecularGeometry(
            symbols=symbols,
            positions=positions,
            charge=charge,
            spin_multiplicity=1,
            source=f"ETKDGv3{' + MMFF94' if mmff_used else ''}, conformer {rank + 1}",
            conformers_considered=len(conformer_ids),
        )

        if xtb_used:
            relaxed = xtb.run(
                QMRequest(
                    geometry=geometry,
                    method=QMMethod.GFN2_XTB,
                    optimize_geometry=True,
                    force_threshold=0.05,
                    max_optimization_steps=200,
                )
            )
            if relaxed.usable and relaxed.optimized_geometry is not None:
                before = _bond_lengths(symbols, positions, bonds)
                after = _bond_lengths(symbols, relaxed.optimized_geometry.positions, bonds)
                shift = max((abs(a - b) for a, b in zip(before, after)), default=0.0)
                max_shift = shift if max_shift is None else max(max_shift, shift)
                geometry = relaxed.optimized_geometry
            else:
                xtb_used = False
                diagnostics.append(
                    "GFN2-xTB relaxation failed; the QM ran on the force-field geometry"
                )

        result = pyscf_backend.run(
            QMRequest(
                geometry=geometry,
                method=QMMethod.DFT,
                basis=BASIS,
                xc=FUNCTIONAL,
                optimize_geometry=False,
                max_scf_cycles=MAX_SCF_CYCLES,
            )
        )
        if not result.usable or result.total_energy is None:
            return Ensemble(
                **{**empty, "symbols": symbols, "mmff_used": mmff_used},
                diagnostics=tuple(diagnostics),
                failure=(
                    f"{BASIS_LABEL} B3LYP did not produce a usable result for conformer "
                    f"{rank + 1}: {'; '.join(result.diagnostics) or result.status.value}"
                ),
            )
        geometries.append(geometry)
        energies.append(result.total_energy.to("J/mol").value)
        dipoles.append(
            float(result.dipole_moment.to("debye").value)
            if result.dipole_moment is not None
            else float("nan")
        )
        gaps.append(
            float(result.homo_lumo_gap.to("eV").value)
            if result.homo_lumo_gap is not None
            else None
        )

    if any(math.isnan(mu) for mu in dipoles):
        return Ensemble(
            **{**empty, "symbols": symbols},
            diagnostics=tuple(diagnostics),
            failure="PySCF returned no dipole for at least one conformer",
        )

    if max_shift is not None and max_shift > BOND_SHIFT_ALARM_ANGSTROM:
        pre = "MMFF94" if mmff_used else "the raw ETKDGv3 embedding"
        diagnostics.append(
            f"a bond length moved {max_shift:.2f} A between {pre} and GFN2-xTB, which "
            "is the signature of a force field with no parameters for this bonding "
            "pattern; the xTB geometry was used but the force-field pre-filter that "
            "chose the conformers was working from a distorted surface"
        )

    # Is the closed-shell singlet even the right state?  Nothing upstream can
    # tell: RDKit reports zero radical electrons for O=O and for many carbenes
    # and nitrenes, and the electron count is even in all of them.  A small
    # computed gap is the only cheap signal there is, so below the threshold the
    # question is settled with one UKS triplet single point rather than left to
    # a domain flag.
    triplet_below_singlet: float | None = None
    finite_gaps = [g for g in gaps if g is not None]
    if finite_gaps and min(finite_gaps) < TRIPLET_PROBE_GAP_EV:
        low = min(range(len(energies)), key=lambda i: energies[i])
        reference = geometries[low]
        probe = pyscf_backend.run(
            QMRequest(
                geometry=MolecularGeometry(
                    symbols=reference.symbols,
                    positions=reference.positions,
                    charge=reference.charge,
                    spin_multiplicity=3,
                    source=f"{reference.source}, triplet probe",
                ),
                method=QMMethod.DFT,
                basis=BASIS,
                xc=FUNCTIONAL,
                optimize_geometry=False,
                max_scf_cycles=MAX_SCF_CYCLES,
            )
        )
        if probe.usable and probe.total_energy is not None:
            triplet_below_singlet = energies[low] - probe.total_energy.to("J/mol").value
        else:
            # Not knowing is not the same as knowing it is fine, and the caller
            # ranks on this: say the question went unanswered.
            diagnostics.append(
                f"the Kohn-Sham gap is {min(finite_gaps):.2f} eV, low enough to ask "
                "whether the ground state is a triplet, and the UKS triplet probe did "
                "not converge, so the closed-shell assumption is untested here"
            )

    return Ensemble(
        symbols=symbols,
        energies=tuple(energies),
        dipoles=tuple(dipoles),
        gaps=tuple(gaps),
        weights=_boltzmann(tuple(energies), temperature_k),
        temperature_k=temperature_k,
        geometry_source=(
            f"ETKDGv3{' + MMFF94' if mmff_used else ''}"
            + (" + GFN2-xTB relaxed" if xtb_used else " (no xTB relaxation)")
        ),
        xtb_used=xtb_used,
        mmff_used=mmff_used,
        populated=len(window),
        computed=len(selected),
        max_bond_shift=max_shift,
        triplet_below_singlet_j=triplet_below_singlet,
        diagnostics=tuple(diagnostics),
    )


#: Ensembles already computed, keyed by structure *and* level of theory.
#:
#: The base ``Expert.predict`` loops ``_predict_one`` per property, so without
#: this the SCF would run three times for one candidate.  The key carries the
#: functional, the basis and the conformer count as well as the structure: an
#: atomization energy is a difference between a molecule and its free atoms,
#: and a cache that forgot the level of theory would let those two sides come
#: from different calculations and still look plausible.
_ENSEMBLE_CACHE: dict[tuple[Any, ...], Ensemble] = {}

#: Free-atom energies at this level of theory, J/mol, keyed the same way.
_ATOM_CACHE: dict[tuple[str, str, str], float | None] = {}


def cached_ensemble(
    smiles: str, temperature_k: float, max_conformers: int = MAX_CONFORMERS
) -> Ensemble:
    from formulate import chem

    canonical = chem.canonical_smiles(smiles) or smiles
    # The temperature only enters through the Boltzmann weights, so it is
    # bucketed to 1 K: a 298.15 K and a 298.20 K request are the same ensemble.
    key = (canonical, FUNCTIONAL, BASIS, max_conformers, round(temperature_k))
    hit = _ENSEMBLE_CACHE.get(key)
    if hit is None:
        hit = build_ensemble(
            canonical, temperature_k=temperature_k, max_conformers=max_conformers
        )
        _ENSEMBLE_CACHE[key] = hit
    return hit


def free_atom_energy(symbol: str) -> float | None:
    """Ground-state free-atom energy at this level of theory, J/mol.

    The multiplicity comes from :data:`formulate.physics.qm.thermochemistry.ATOMIC_MULTIPLICITY`
    rather than being defaulted, because a carbon atom forced to be a
    closed-shell singlet converges happily and leaves the atomization energy
    wrong by roughly 400 kJ/mol per carbon.
    """
    import numpy as np

    from formulate.physics.geometry import MolecularGeometry
    from formulate.physics.qm.base import QMMethod, QMRequest
    from formulate.physics.qm.pyscf_backend import PySCFBackend
    from formulate.physics.qm.thermochemistry import ATOMIC_MULTIPLICITY

    key = (symbol, FUNCTIONAL, BASIS)
    if key in _ATOM_CACHE:
        return _ATOM_CACHE[key]

    multiplicity = ATOMIC_MULTIPLICITY.get(symbol)
    if multiplicity is None:
        _ATOM_CACHE[key] = None
        return None

    geometry = MolecularGeometry(
        symbols=(symbol,),
        positions=np.zeros((1, 3)),
        spin_multiplicity=multiplicity,
        source=f"free {symbol} atom, ground-state multiplicity {multiplicity}",
    )
    result = PySCFBackend().run(
        QMRequest(
            geometry=geometry,
            method=QMMethod.DFT,
            basis=BASIS,
            xc=FUNCTIONAL,
            optimize_geometry=False,
            max_scf_cycles=MAX_SCF_CYCLES,
        )
    )
    if not result.usable or result.total_energy is None:
        # Not cached.  A missing multiplicity is a permanent fact about the
        # table above; an SCF that did not converge is a transient fact about
        # one run, and caching it would turn a retry into a second refusal for
        # every later candidate containing this element.
        return None
    value = result.total_energy.to("J/mol").value
    _ATOM_CACHE[key] = value
    return value


# ---------------------------------------------------------------------------
# Structural guards
# ---------------------------------------------------------------------------


def has_valence_acceptor(smiles: str) -> bool:
    """True when the molecule has a valence orbital for an electron to go into.

    A saturated molecule has none: measured on ethanol at one geometry, the
    lowest virtual orbital sits at +2.14 eV in 6-31G(d), +0.15 eV in 6-31+G(d)
    and -0.35 eV in 6-311++G(d,p) while the HOMO moves only 0.53 eV.  That
    orbital is a discretised continuum state whose energy is a property of the
    basis set, not of the molecule, so there is nothing to attach an
    uncertainty to.
    """
    from rdkit import Chem

    from formulate import chem

    mol = chem.mol_from_smiles(smiles)
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            return True
    for bond in mol.GetBonds():
        if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
            if begin.GetAtomicNum() > 1 and end.GetAtomicNum() > 1:
                return True
    return False


def _unpaired_electrons(smiles: str) -> int:
    from formulate import chem

    mol = chem.mol_from_smiles(smiles)
    if mol is None:
        return 0
    return sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())


def _electron_count(smiles: str) -> int:
    from rdkit import Chem

    from formulate import chem

    mol = Chem.AddHs(chem.mol_from_smiles(smiles))
    total = sum(atom.GetAtomicNum() for atom in mol.GetAtoms())
    return total - Chem.GetFormalCharge(mol)


class QuantumElectronicExpert(Expert):
    """Dipole moment, frontier gap and atomization energy from B3LYP/6-31G(d,p)."""

    id = "quantum_electronic"
    version = "1"
    method = (
        "B3LYP/6-31G(d,p) single points (PySCF) on a GFN2-xTB-relaxed ETKDGv3 + MMFF94 "
        "conformer ensemble; dipole Boltzmann-averaged over the ensemble, frontier gap "
        "and atomization energy from the lowest conformer with free-atom references at "
        "the same level of theory. Becke (1993); Lee, Yang & Parr (1988); Hariharan & "
        "Pople (1973); Bannwarth, Ehlert & Grimme (2019). The gap is a Kohn-Sham "
        "orbital-energy difference, not a fundamental or optical gap; the atomization "
        "energy is the electronic bottom-of-well value with no zero-point correction"
    )
    family = PropertyFamily.ELECTRICAL
    supported_classes = frozenset({MaterialClass.MOLECULE})
    supported_properties = frozenset({"dipole_moment", "homo_lumo_gap", "atomization_energy"})
    dependencies: frozenset[str] = frozenset()

    def __init__(self, *, max_heavy_atoms: int = MAX_HEAVY_ATOMS,
                 max_conformers: int = MAX_CONFORMERS) -> None:
        self.max_heavy_atoms = max_heavy_atoms
        self.max_conformers = max_conformers

    # -- availability ------------------------------------------------------

    def is_available(self) -> bool:
        from formulate import chem
        from formulate.physics.qm.pyscf_backend import PySCFBackend

        return chem.rdkit_available() and PySCFBackend().is_available()

    def unavailable_reason(self) -> str:
        from formulate import chem
        from formulate.physics.qm.pyscf_backend import PySCFBackend

        if not chem.rdkit_available():
            return "RDKit is not installed, so no 3D geometry can be built"
        backend = PySCFBackend()
        if not backend.is_available():
            return backend.unavailable_reason() or "pyscf is not importable"
        return ""

    def _software(self) -> SoftwareEnvironment:
        from formulate import chem

        versions = {"rdkit": chem.rdkit_version()}
        try:  # pragma: no cover - version reporting only
            import pyscf

            versions["pyscf"] = pyscf.__version__
        except Exception:
            pass
        return SoftwareEnvironment.capture(**versions)

    # -- domain ------------------------------------------------------------

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        from formulate import chem

        basis = (
            "neutral, closed-shell main-group organic molecules up to "
            f"{self.max_heavy_atoms} heavy atoms, in the elements 6-31G(d,p) covers "
            "(H through Ca; the d block Sc-Zn is inside the basis but outside every "
            "reference set behind this expert's uncertainty, so it is refused); B3LYP "
            "was parameterised on small-molecule thermochemistry"
        )
        smiles = candidate.molecule.smiles if candidate.molecule else None
        if smiles is None or chem.mol_from_smiles(smiles) is None:
            return ApplicabilityDomain.outside(
                "candidate carries no parseable molecule", basis=basis
            )

        descriptors = chem.descriptors(smiles)
        warnings: list[str] = []
        score = 1.0

        heavy = int(descriptors.get("heavy_atom_count", 0.0))
        if heavy > self.max_heavy_atoms:
            return ApplicabilityDomain.outside(
                f"{heavy} heavy atoms is above the {self.max_heavy_atoms} this expert "
                "will spend a B3LYP ensemble on",
                basis=basis,
            )

        unsupported = sorted(chem.elements(smiles) - BASIS_ELEMENTS)
        if unsupported:
            return ApplicabilityDomain.outside(
                f"6-31G(d,p) is not defined for {', '.join(unsupported)}", basis=basis
            )

        metals = sorted(chem.elements(smiles) & D_BLOCK_ELEMENTS)
        if metals:
            return ApplicabilityDomain.outside(
                f"{', '.join(metals)} is a d-block element: 6-31G(d,p) reaches it but "
                "carries no 3d-optimised polarisation, and no reference molecule behind "
                "this expert's error bars contains a transition metal",
                basis=basis,
            )

        if descriptors.get("formal_charge", 0.0):
            return ApplicabilityDomain.outside(
                "a charged species: its dipole moment is origin-dependent and 6-31G(d,p) "
                "has no diffuse functions to bind an extra electron",
                basis=basis,
            )

        if _unpaired_electrons(smiles) or _electron_count(smiles) % 2:
            return ApplicabilityDomain.outside(
                "an open-shell species; this expert solves a restricted closed-shell SCF",
                basis=basis,
            )

        rotatable = int(descriptors.get("rotatable_bond_count", 0.0))
        if rotatable > 3:
            warnings.append(
                f"{rotatable} rotatable bonds against at most {self.max_conformers} "
                "conformers carried to B3LYP: the ensemble samples a small and "
                "energy-biased fraction of the rotamers, and the bias runs low for a "
                "dipole because vacuum electrostatics favour the least polar rotamer"
            )
            score = min(score, 0.5)

        return ApplicabilityDomain(
            score=score, in_domain=score > 0.3, warnings=tuple(warnings), basis=basis
        )

    # -- prediction --------------------------------------------------------

    def _refuse(self, prop: str, request: PredictionRequest) -> Prediction | None:
        """Every reason to decline before a single SCF cycle is spent."""
        from formulate import chem

        molecule = request.candidate.molecule
        if molecule is None:
            return Prediction.unsupported(
                prop, self.id, "this expert reads a single-molecule candidate only"
            )
        smiles = molecule.smiles
        if chem.mol_from_smiles(smiles) is None:
            return Prediction.failed(prop, self.id, f"RDKit cannot parse {smiles!r}")

        descriptors = chem.descriptors(smiles)
        charge = int(descriptors.get("formal_charge", 0.0)) or molecule.charge
        if charge:
            return Prediction.unsupported(
                prop,
                self.id,
                f"net formal charge {charge:+d}: a charged species has no origin-independent "
                "dipole moment, 6-31G(d,p) carries no diffuse functions so an anion's extra "
                "electron is not bound in this basis, and the free-atom references are "
                "neutral so their energy difference would not be an atomization energy. "
                "Submit the neutral conjugate acid or base if that is the species you mean "
                "to rank",
            )

        radicals = _unpaired_electrons(smiles)
        if radicals or _electron_count(smiles) % 2:
            return Prediction.unsupported(
                prop,
                self.id,
                "an open-shell species: this expert solves a restricted closed-shell SCF "
                "only, and a doublet forced closed-shell converges and returns plausible "
                "numbers, which is worse than refusing. Rank the closed-shell parent, or "
                "route this through an unrestricted method whose errors were measured on "
                "radicals",
            )

        heavy = int(descriptors.get("heavy_atom_count", 0.0))
        if heavy > self.max_heavy_atoms:
            return Prediction.unsupported(
                prop,
                self.id,
                f"{heavy} heavy atoms is above this expert's cap of {self.max_heavy_atoms}: "
                "measured on these four cores, one B3LYP/6-31G(d,p) single point takes 6.8 s "
                "at 6 heavy atoms, 19.8 s at 10 and 38.5 s at 13, so an ensemble plus "
                "free-atom references at that size is minutes per candidate. This is a "
                "budget, not a physical limit: construct the expert with a larger "
                "max_heavy_atoms if a screen of this size is worth the wall time",
            )

        unsupported = sorted(chem.elements(smiles) - BASIS_ELEMENTS)
        if unsupported:
            return Prediction.unsupported(
                prop,
                self.id,
                f"6-31G(d,p) is not defined for {', '.join(unsupported)}; covering it needs "
                "an effective-core-potential basis, which would invalidate every error "
                "constant this expert's uncertainty is measured against",
            )

        metals = sorted(chem.elements(smiles) & D_BLOCK_ELEMENTS)
        if metals:
            return Prediction.unsupported(
                prop,
                self.id,
                f"{', '.join(metals)} is a d-block element. 6-31G(d,p) is defined that far "
                "but is a plain valence basis there - no 3d-optimised polarisation, no "
                "semi-core correlation functions - and B3LYP on an open d shell carries "
                "static-correlation errors far outside anything in the first- and "
                "second-row organic sets that set every error bar here; measured, the gap "
                "let chromyl chloride return 0.68 D wearing a 0.144 D bar. Use a "
                "transition-metal level of theory (a def2 basis with a metal-tested "
                "functional, validated against organometallic reference data) rather than "
                "reading this expert's number",
            )
        return None

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        refusal = self._refuse(prop, request)
        if refusal is not None:
            return refusal

        smiles = request.candidate.molecule.smiles  # type: ignore[union-attr]

        if prop == "homo_lumo_gap" and not has_valence_acceptor(smiles):
            return Prediction.unsupported(
                prop,
                self.id,
                "a saturated molecule with no aromatic ring and no multiple bond between "
                "heavy atoms has no valence acceptor orbital: measured on ethanol, the "
                "lowest virtual orbital moves from +2.14 eV in 6-31G(d) to -0.35 eV in "
                "6-311++G(d,p) while the HOMO moves 0.53 eV, so the 'gap' is a property of "
                "the basis set rather than of the molecule. If what you want is how hard "
                "this molecule is to ionise, ask for that; there is no bound acceptor state "
                "here to put an electron into",
            )

        if prop == "atomization_energy":
            missing = self._missing_atomic_references(smiles)
            if missing:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"no tabulated free-atom ground-state multiplicity for "
                    f"{', '.join(missing)}; defaulting an atom to a closed-shell singlet "
                    "converges and is wrong by roughly 400 kJ/mol per carbon, so the "
                    "reference is refused rather than assumed",
                )
        if prop in ("atomization_energy", "homo_lumo_gap"):
            # Cheap half of the geometry check: if xtb will not even import,
            # decline before spending an SCF.  The other half is below, because
            # importing is not running.
            from formulate.physics.qm.xtb_backend import XTBBackend

            if not XTBBackend().is_available():
                return Prediction.unsupported(
                    prop,
                    self.id,
                    self._mmff_geometry_refusal(
                        prop, "GFN2-xTB is not importable, so no relaxation was run"
                    ),
                )

        temperature = request.conditions.temperature_k or DEFAULT_TEMPERATURE_K
        ensemble = cached_ensemble(smiles, temperature, self.max_conformers)
        if ensemble.failure is not None:
            return Prediction.failed(prop, self.id, ensemble.failure)
        if not ensemble.mmff_used and not ensemble.xtb_used:
            return Prediction.unsupported(
                prop,
                self.id,
                "neither MMFF94 nor GFN2-xTB could relax this molecule, so there is no "
                "trustworthy geometry to run the QM on",
            )

        # An importable xtb is not a completed relaxation.  A GFN2-xTB run that
        # raises or fails to converge leaves the MMFF94 geometry in place, and
        # the two properties that cannot survive that substitution have to be
        # refused on what actually happened, not on what was installed.
        if not ensemble.xtb_used and prop in ("atomization_energy", "homo_lumo_gap"):
            return Prediction.unsupported(
                prop,
                self.id,
                self._mmff_geometry_refusal(
                    prop,
                    "GFN2-xTB imported but its relaxation did not complete for this "
                    "molecule, so the force-field geometry is what survived",
                ),
            )

        if ensemble.wrong_spin_state:
            gap = min(g for g in ensemble.gaps if g is not None)
            kj = (ensemble.triplet_below_singlet_j or 0.0) / 1000.0
            return Prediction.unsupported(
                prop,
                self.id,
                f"the closed-shell singlet this expert solves is not the ground state: at "
                f"the same geometry a UKS triplet is {kj:.0f} kJ/mol lower, which the "
                f"{gap:.2f} eV Kohn-Sham gap flagged and the triplet single point then "
                "settled. Nothing upstream catches this - RDKit reports zero radical "
                "electrons for dioxygen and for many carbenes and nitrenes, and the "
                "electron count is even - so the whole wavefunction, and every number "
                "taken from it, belongs to the wrong state. Compute this molecule as an "
                "open-shell triplet, or supply it as one",
            )

        domain = self._with_gap_diagnosis(domain, ensemble)

        if prop == "dipole_moment":
            return self._dipole(request, domain, ensemble)
        if prop == "homo_lumo_gap":
            return self._gap(request, domain, ensemble)
        return self._atomization(request, domain, ensemble)

    # -- the three properties ---------------------------------------------

    def _dipole(
        self, request: PredictionRequest, domain: ApplicabilityDomain, ensemble: Ensemble
    ) -> Prediction:
        value = ensemble.mean_dipole()
        conformer_term = ensemble.dipole_spread()
        method_term = DIPOLE_METHOD_STD_DEBYE

        notes = [
            "a vacuum, isolated-molecule dipole: a condensed phase enhances it by 20-40% "
            "for hydrogen-bonding liquids (water is 1.85 D isolated and about 2.9 D in the "
            "liquid), so this is not a solvent-polarity descriptor",
            f"root-mean-square over {ensemble.computed} Boltzmann-weighted conformer(s) at "
            f"{ensemble.temperature_k:.1f} K, which is what a dielectric measurement of the "
            "molar polarisation returns; a microwave Stark value is one conformer's",
            f"geometry: {ensemble.geometry_source}",
        ]
        notes.extend(ensemble.diagnostics)

        if not ensemble.xtb_used:
            method_term = math.hypot(method_term, MMFF_GEOMETRY_DIPOLE_STD_DEBYE)
            notes.append(
                "no GFN2-xTB relaxation was applied, so this is B3LYP on the MMFF94 "
                "geometry and the method term is widened; measured, that substitution "
                "moves the dipole 0.053 D for water, 0.086 D for acetone and 0.205 D for "
                "nitrobenzene. The frontier gap and the atomization energy refuse in the "
                "same situation, because for them the same swap is worth 4.16 eV and "
                "369 kJ/mol"
            )
        if ensemble.under_sampled:
            # An under-sampled ensemble is biased, not merely noisy: vacuum
            # electrostatics favour the rotamer whose bond dipoles cancel, so
            # the low-energy conformers this keeps are the least polar ones.
            conformer_term = math.hypot(conformer_term, 0.5 * value)
            notes.append(
                f"{ensemble.populated} conformers sit inside the {THERMAL_WINDOW_KJ:.0f} "
                f"kJ/mol thermal window but only {ensemble.computed} reached B3LYP; the "
                "conformer term is widened because the sample is biased toward the "
                "least polar rotamer, not merely small"
            )

        std = math.hypot(method_term, conformer_term)
        dominant = "conformer spread" if conformer_term > method_term else "method error"
        return self._make(
            "dipole_moment",
            value,
            "debye",
            request,
            domain,
            std=std,
            kind=UncertaintyKind.COMBINED,
            basis=(
                f"held-out RMSE {DIPOLE_METHOD_STD_DEBYE:.3f} D of this protocol against "
                f"twenty gas-phase experimental dipoles, split on rotatable-bond count so "
                f"the seven flexible molecules were withheld from the thirteen rigid ones "
                f"that set the constant (in sample 0.127 D, held out 0.144 D, and it is the "
                f"held-out figure quoted); {DIPOLE_HELD_OUT_COVERAGE:.0%} of the held-out "
                f"set fell inside its own one-sigma bound against the ~68% a correct "
                f"estimate implies. Combined in quadrature with a {conformer_term:.3f} D "
                f"population-weighted conformer spread; {dominant} dominates"
            ),
            conditions=self._vacuum(ensemble.temperature_k),
            notes=tuple(notes),
            functional=FUNCTIONAL,
            basis_set=BASIS_LABEL,
            geometry=ensemble.geometry_source,
            conformers=ensemble.computed,
        )

    def _gap(
        self, request: PredictionRequest, domain: ApplicabilityDomain, ensemble: Ensemble
    ) -> Prediction:
        index = ensemble.lowest
        value = ensemble.gaps[index]
        if value is None:
            return Prediction.failed(
                "homo_lumo_gap",
                self.id,
                "the highest occupied orbital came out above the lowest virtual one, or the "
                "basis produced no virtual orbital: the SCF converged to an unstable "
                "solution and the difference is not a gap",
            )

        finite = [g for g in ensemble.gaps if g is not None]
        conformer_term = (max(finite) - min(finite)) / 2.0 if len(finite) > 1 else 0.0
        std = math.hypot(GAP_RMSE_EV, conformer_term)

        return self._make(
            "homo_lumo_gap",
            value,
            "eV",
            request,
            domain,
            std=std,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                f"RMSE {GAP_RMSE_EV:.2f} eV of this protocol's Kohn-Sham gap against the "
                f"experimental fundamental gap (photoelectron IP minus electron-transmission "
                f"EA) over eight conjugated molecules, combined with a {conformer_term:.2f} "
                f"eV conformer spread. The mean signed deviation is "
                f"{GAP_MEAN_SIGNED_EV:+.2f} eV and every molecule in the set came out low, "
                f"so the bias is one-signed: the true fundamental gap is LARGER than this "
                f"value by about that much, not merely uncertain by it. Scatter about that "
                f"bias is only {GAP_RESIDUAL_SCATTER_EV:.2f} eV, so this number is "
                f"reproducible against other numbers from this expert while being wrong "
                f"against experiment"
            ),
            conditions=self._vacuum(0.0),
            notes=(
                "a Kohn-Sham orbital-energy difference, not an optical gap and not a "
                "fundamental gap; it is comparable with other numbers from this expert at "
                "this basis and functional, and with little else",
                f"taken from the lowest of {ensemble.computed} conformer(s); "
                f"geometry: {ensemble.geometry_source}",
            )
            + ensemble.diagnostics,
            functional=FUNCTIONAL,
            basis_set=BASIS_LABEL,
            geometry=ensemble.geometry_source,
        )

    def _atomization(
        self, request: PredictionRequest, domain: ApplicabilityDomain, ensemble: Ensemble
    ) -> Prediction:
        index = ensemble.lowest
        molecular = ensemble.energies[index]

        total = 0.0
        for symbol in sorted(set(ensemble.symbols)):
            atom = free_atom_energy(symbol)
            if atom is None:
                return Prediction.failed(
                    "atomization_energy",
                    self.id,
                    f"the free {symbol} atom reference did not converge at "
                    f"{FUNCTIONAL}/{BASIS_LABEL}, so there is no consistent reference to "
                    "subtract",
                )
            total += atom * ensemble.symbols.count(symbol)

        value_j = total - molecular
        if value_j <= 0.0:
            return Prediction.failed(
                "atomization_energy",
                self.id,
                "the free atoms came out below the molecule, which is not a bound species: "
                "the reference multiplicities or the SCF converged to the wrong state",
            )
        value = value_j / 1000.0  # J/mol -> kJ/mol
        std = max(ATOMIZATION_FLOOR_KJ_MOL, ATOMIZATION_RELATIVE_STD * value)

        return self._make(
            "atomization_energy",
            value,
            "kJ/mol",
            request,
            domain,
            std=std,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                f"max({ATOMIZATION_FLOOR_KJ_MOL:.0f} kJ/mol, "
                f"{ATOMIZATION_RELATIVE_STD:.1%} of the value), both read off this "
                "protocol's measured error against twelve bottom-of-well De references "
                "built from ATcT/CODATA 0 K enthalpies of formation plus literature "
                "zero-point energies: r.m.s. 2.19% and 19.1 kJ/mol, worst -6.2% (hydrogen "
                "fluoride), 10 of 12 inside their own one-sigma bound. The floor exists "
                "because the largest absolute errors in that set are on the SMALLEST "
                "molecules - hydrogen fluoride -36.7 kJ/mol over one bond, water -33.5 over "
                "two, both X-H bonds to a very electronegative atom - where a purely "
                "relative bar would collapse to about 13 kJ/mol; the relative term takes "
                "over above roughly 800 kJ/mol. The zero-point energy is NOT in this bar, "
                "on purpose: it is a one-signed definitional offset, not an error"
            ),
            conditions=self._vacuum(0.0),
            notes=(
                "the ELECTRONIC, bottom-of-well atomization energy. A thermochemical "
                "atomization enthalpy is smaller by the zero-point energy - 117 kJ/mol for "
                "methane, about 200 for ethanol - and the offset grows with molecule size, "
                "so comparing this against a thermochemical table passes candidates that "
                "should fail",
                "free atoms are computed in their own basis while the molecule uses the "
                "full molecular basis, which overbinds by tens of kJ/mol at this basis "
                "size; the geometry not being the B3LYP minimum pushes the other way. Both "
                "are inside the measured error above rather than assumed to cancel",
                f"free-atom references at {FUNCTIONAL}/{BASIS_LABEL} in their ground-state "
                "multiplicities; geometry: " + ensemble.geometry_source,
            )
            + ensemble.diagnostics,
            functional=FUNCTIONAL,
            basis_set=BASIS_LABEL,
            geometry=ensemble.geometry_source,
        )

    # -- helpers -----------------------------------------------------------

    def _mmff_geometry_refusal(self, prop: str, lead: str) -> str:
        """Why an MMFF94 geometry is fatal here but survivable for the dipole.

        Both branches say the same thing and differ only in what went wrong, so
        the wording lives in one place: a reader comparing an "xtb is missing"
        refusal against an "xtb crashed" one should not have to work out
        whether the two are the same decision.
        """
        cost = (
            f"{MMFF_GEOMETRY_GAP_SHIFT_EV:.2f} eV on carbon dioxide's frontier gap "
            f"(7.23 against 11.39 eV), more than this property's whole "
            f"{GAP_RMSE_EV:.2f} eV error bar"
            if prop == "homo_lumo_gap"
            else (
                f"{MMFF_GEOMETRY_ATOMIZATION_SHIFT_KJ_MOL:.0f} kJ/mol (22.7%) on carbon "
                f"dioxide's atomization energy, against a "
                f"{ATOMIZATION_FLOOR_KJ_MOL:.0f} kJ/mol floor"
            )
        )
        return (
            f"{lead}, which leaves B3LYP standing on an MMFF94 geometry. Measured here, "
            f"that substitution costs {cost} - and MMFF reports success throughout, so "
            "there is no diagnostic left to widen the bar with. Install or repair xtb, or "
            "supply a relaxed geometry. The dipole moment is still answered on this same "
            "ensemble, widened rather than refused, because the same swap moves it by at "
            "most 0.21 D"
        )

    def _missing_atomic_references(self, smiles: str) -> list[str]:
        from formulate import chem
        from formulate.physics.qm.thermochemistry import ATOMIC_MULTIPLICITY

        present = set(chem.elements(smiles)) | {"H"}
        return sorted(s for s in present if s not in ATOMIC_MULTIPLICITY)

    def _with_gap_diagnosis(
        self, domain: ApplicabilityDomain, ensemble: Ensemble
    ) -> ApplicabilityDomain:
        """Mark the whole candidate out of domain when the wavefunction is suspect.

        A very small computed gap in a nominally closed-shell molecule means
        static correlation a single determinant does not have.  That is a fault
        in the wavefunction, and the same wavefunction produced the dipole and
        the atomization energy too, so all three are marked - not only the gap.
        """
        gaps = [g for g in ensemble.gaps if g is not None]
        if not gaps or min(gaps) >= SMALL_GAP_EV:
            return domain
        return domain.merged_with(
            ApplicabilityDomain.outside(
                f"the computed Kohn-Sham gap is {min(gaps):.2f} eV, below the "
                f"{SMALL_GAP_EV:.1f} eV at which a nominally closed-shell organic molecule "
                "is showing static correlation that one determinant does not have; the same "
                "wavefunction produced every value here",
                score=0.2,
            )
        )

    def _vacuum(self, temperature_k: float) -> Conditions:
        """The conditions these numbers actually belong to.

        All three properties are condition-independent in the registry, so a
        vacuum stamp answers a request stated at 25 degrees legitimately rather
        than being rejected on a technicality.  The dipole carries the ensemble
        temperature because its conformer populations depend on it; the gap and
        the atomization energy carry 0 K because they come from one structure.
        """
        return Conditions(
            temperature=Quantity(value=temperature_k, unit="kelvin"),
            environment="vacuum",
        )
