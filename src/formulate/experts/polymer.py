"""Polymer bulk properties from repeat-unit structure.

Specification section 3 keeps polymers as a first-class material class, and
section 13 warns that bulk polymer behaviour must not be inferred from an
isolated molecule.  Both are the reason this module exists and the reason it
is shaped the way it is: a polymer's glass transition and bulk density are
properties of a chain, so everything here is computed per *repeat unit* of an
infinite chain rather than from a capped oligomer that happens to look like
the monomer.

Two predictors live here, and they fail in different places on purpose:

``PolymerGlassTransitionExpert``
    van Krevelen's additive molar glass-transition function,
    ``Tg = sum_i n_i Yg_i / M_repeat``.  The group set is Joback's, taken from
    the ``thermo`` package rather than invented here, and the ``Yg``
    coefficients are *fitted in this repository* against the measured
    reference set in ``data/reference_polymers.json``.  They are not a
    transcription of van Krevelen's published table.  That distinction
    matters: a fitted table reproduces the polymers it was fitted to by
    construction, so the only number worth quoting is the error on the
    validation split, which no coefficient ever saw.

``PolymerDensityExpert``
    Amorphous density from the van der Waals volume of the repeat unit and a
    single fitted packing factor, ``rho = M / (k V_w N_A)``.  The van der Waals
    volume is computed geometrically from a 3D structure rather than from a
    group table, so this expert covers repeat units the Joback group set
    cannot express - silicones and carbonates among them.

What the two are worth, measured rather than claimed.  The glass transition
comes out at 34 K RMSE leaving one fitted polymer out at a time and 22 K over
ten polymers withheld from the fit entirely, with a systematic offset of about
+13 K on those ten - the model runs high on structures it has not seen, by
less than its own error bar.  The density comes out at 4.3 per cent on the fit
split and 3.7 per cent on the withheld one.  Neither offset is corrected for,
because correcting to a validation set is how a validation set stops meaning
anything.

Both take the same view of their own limits.  A repeat unit that cannot be
decomposed is refused rather than approximated; a stereoregular or crosslinked
polymer is reported out of domain rather than answered as though it were
atactic and linear; and the uncertainty attached to every value is the
held-out error of the fit, not the residual scatter of the fit itself.
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MonomerRole,
    PolymerSpec,
    PolymerTopology,
    Tacticity,
)
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest

#: Avogadro's number scaled so that a volume in cubic angstroms per repeat
#: unit multiplied by this is a molar volume in cubic centimetres per mole.
_A3_TO_CM3_PER_MOL = 0.6022140761

#: Ridge coefficient on the Yg fit.  Small enough not to bias the
#: coefficients, large enough to keep the normal equations solvable when two
#: groups are collinear across the reference set.
_RIDGE = 1e-5

#: A group must appear in at least this many reference polymers before a
#: prediction leaning on it is treated as in-domain.  Below it the fitted
#: coefficient is carrying one or two polymers' worth of information and
#: cannot be expected to transfer; the held-out error is measurably worse.
_MIN_GROUP_SUPPORT = 3

#: Fox-Flory scale for the chain-end depression of Tg, in K g/mol.  The
#: constant is polymer-specific and is not known here, so this is used only to
#: size the uncertainty of a prediction for a short chain, never to correct it.
_FOX_FLORY_SCALE = 1.0e5

#: Below this number-average molar mass the chain-end depression is comparable
#: to the property itself and the high-polymer limit is not a useful answer.
_MIN_NUMBER_AVERAGE_MOLAR_MASS = 2000.0


def _data_path() -> Path:
    return Path(__file__).resolve().parent.parent.joinpath("data", "reference_polymers.json")


@functools.lru_cache(maxsize=1)
def reference_polymers() -> tuple[dict, ...]:
    """The measured reference set, fit and validation splits together."""
    with _data_path().open() as handle:
        payload = json.load(handle)
    return tuple(payload["polymers"])


def reference_split(split: str) -> tuple[dict, ...]:
    return tuple(p for p in reference_polymers() if p["split"] == split)


# --------------------------------------------------------------------------
# Repeat-unit geometry and decomposition
# --------------------------------------------------------------------------


def _rdkit():
    from rdkit import Chem

    return Chem


def link_repeat_units(unit_smiles: str, count: int):
    """Join ``count`` copies of a repeat unit into a chain, capped with methyl.

    The repeat unit is written with two attachment points, ``[*]``.  Capping
    with methyl rather than hydrogen is not cosmetic: a hydrogen cap turns an
    ester end into an aldehyde and an ether end into an alcohol, which changes
    the group assignment of the terminal unit.  A methyl cap leaves the ends
    as ordinary esters and ethers, and because every quantity here is taken as
    a difference between two chain lengths, the caps cancel exactly.
    """
    Chem = _rdkit()
    template = Chem.MolFromSmiles(unit_smiles)
    if template is None:
        raise ValueError(f"repeat unit {unit_smiles!r} is not a valid SMILES")
    if sum(1 for a in template.GetAtoms() if a.GetAtomicNum() == 0) != 2:
        raise ValueError(
            f"repeat unit {unit_smiles!r} must carry exactly two [*] attachment points"
        )

    chain = Chem.RWMol(template)
    for _ in range(count - 1):
        offset = chain.GetNumAtoms()
        chain.InsertMol(Chem.MolFromSmiles(unit_smiles))
        dummies = [a.GetIdx() for a in chain.GetAtoms() if a.GetAtomicNum() == 0]
        left = [d for d in dummies if d < offset][-1]
        right = [d for d in dummies if d >= offset][0]
        chain.AddBond(
            chain.GetAtomWithIdx(left).GetNeighbors()[0].GetIdx(),
            chain.GetAtomWithIdx(right).GetNeighbors()[0].GetIdx(),
            Chem.BondType.SINGLE,
        )
        for idx in sorted((left, right), reverse=True):
            chain.RemoveAtom(idx)

    for idx in [a.GetIdx() for a in chain.GetAtoms() if a.GetAtomicNum() == 0][::-1]:
        atom = chain.GetAtomWithIdx(idx)
        atom.SetAtomicNum(6)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    molecule = chain.GetMol()
    Chem.SanitizeMol(molecule)
    return molecule


@functools.lru_cache(maxsize=1024)
def repeat_unit_groups(unit_smiles: str) -> dict[int, int] | None:
    """Joback group counts for one interior repeat unit, or None.

    Taking the difference between a trimer and a dimer isolates exactly one
    interior unit: the two chain ends are identical in both, so whatever the
    caps contribute cancels.  Decomposing the bare repeat unit directly is not
    an option, because its dangling valences are not a chemical environment
    any group definition describes.
    """
    from thermo.group_contribution.joback import (
        JOBACK_GROUPS_FOR_FRAGMENTATION,
        smarts_fragment,
    )

    short, ok_short, _ = smarts_fragment(
        JOBACK_GROUPS_FOR_FRAGMENTATION, rdkitmol=link_repeat_units(unit_smiles, 2)
    )
    long, ok_long, _ = smarts_fragment(
        JOBACK_GROUPS_FOR_FRAGMENTATION, rdkitmol=link_repeat_units(unit_smiles, 3)
    )
    if not (ok_short and ok_long):
        return None
    counts = {k: long.get(k, 0) - short.get(k, 0) for k in set(short) | set(long)}
    if any(v < 0 for v in counts.values()):
        # The two chain lengths decomposed inconsistently; the difference is
        # then not a repeat unit and must not be passed off as one.
        return None
    return {k: v for k, v in counts.items() if v}


@functools.lru_cache(maxsize=1024)
def repeat_unit_mass(unit_smiles: str) -> float:
    """Molar mass of one interior repeat unit, in g/mol."""
    from rdkit.Chem import Descriptors

    Chem = _rdkit()
    heavy = Chem.AddHs(link_repeat_units(unit_smiles, 3))
    light = Chem.AddHs(link_repeat_units(unit_smiles, 2))
    return float(Descriptors.MolWt(heavy) - Descriptors.MolWt(light))


def _substituent_key(molecule, root: int, exclude: int) -> str:
    """Canonical description of the branch hanging off ``root``, away from ``exclude``."""
    Chem = _rdkit()
    seen = {exclude}
    stack = [root]
    branch: list[int] = []
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        branch.append(idx)
        stack.extend(n.GetIdx() for n in molecule.GetAtomWithIdx(idx).GetNeighbors())
    sub = Chem.MolFragmentToSmiles(molecule, atomsToUse=sorted(branch), canonical=True)
    return sub


@functools.lru_cache(maxsize=1024)
def symmetric_gem_substitution(unit_smiles: str) -> int:
    """Backbone atoms carrying two identical substituents.

    Polyisobutylene sits 53 K below polypropylene, poly(vinylidene chloride)
    99 K below poly(vinyl chloride), poly(vinylidene fluoride) 81 K below
    poly(vinyl fluoride).  In each case the second substituent is a copy of
    the first, and the resulting local symmetry removes the rotational barrier
    asymmetry that a single substituent creates.  An additive group sum cannot
    see this - Joback's ``>C<`` is the same group in polyisobutylene and in
    poly(methyl methacrylate), where the effect runs the other way - so it is
    supplied as one extra descriptor alongside the group counts.
    """
    Chem = _rdkit()
    molecule = Chem.MolFromSmiles(unit_smiles)
    attachment = [a.GetIdx() for a in molecule.GetAtoms() if a.GetAtomicNum() == 0]
    if len(attachment) != 2:
        return 0
    backbone = set(Chem.GetShortestPath(molecule, attachment[0], attachment[1]))
    total = 0
    for atom in molecule.GetAtoms():
        idx = atom.GetIdx()
        if idx not in backbone or atom.GetAtomicNum() == 0:
            continue
        pendant = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() not in backbone]
        if len(pendant) != 2:
            continue
        if _substituent_key(molecule, pendant[0], idx) == _substituent_key(
            molecule, pendant[1], idx
        ):
            total += 1
    return total


# --------------------------------------------------------------------------
# The fitted glass-transition model
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GlassTransitionModel:
    """Fitted molar glass-transition function and its honest error."""

    group_keys: tuple[int, ...]
    #: Yg per group, in K g/mol, plus one trailing coefficient for the
    #: symmetric-geminal-substitution descriptor.
    coefficients: tuple[float, ...]
    #: How many reference polymers each group appears in.
    support: Mapping[int, int]
    n_fit: int
    #: Leave-one-out RMSE over the fit split, restricted to polymers whose
    #: groups all keep enough support when that polymer is removed.
    held_out_rmse: float
    held_out_mae: float
    #: RMSE over the validation split, which no coefficient and no descriptor
    #: choice ever saw.
    validation_rmse: float
    validation_mae: float
    validation_n: int

    @property
    def sigma(self) -> float:
        """The uncertainty to attach to a prediction.

        The larger of the two held-out figures.  Leave-one-out on the fit
        split reuses polymers that informed the descriptor choice; the
        validation split is small.  Neither alone is trustworthy, so the
        pessimistic one is quoted.
        """
        return max(self.held_out_rmse, self.validation_rmse)

    def design_row(self, groups: Mapping[int, int], gem: int, mass: float) -> np.ndarray:
        row = np.array(
            [groups.get(k, 0) for k in self.group_keys] + [gem], dtype=float
        )
        return row / mass

    def predict(self, groups: Mapping[int, int], gem: int, mass: float) -> float:
        return float(self.design_row(groups, gem, mass) @ np.asarray(self.coefficients))

    def unsupported_groups(self, groups: Mapping[int, int]) -> list[int]:
        """Groups in this repeat unit that the reference set barely constrains."""
        return sorted(
            k for k in groups if self.support.get(k, 0) < _MIN_GROUP_SUPPORT
        )

    def unknown_groups(self, groups: Mapping[int, int]) -> list[int]:
        """Groups no reference polymer contains, so with no fitted coefficient."""
        return sorted(k for k in groups if k not in self.support)


def _solve(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    normal = design.T @ design + _RIDGE * np.eye(design.shape[1])
    return np.linalg.solve(normal, design.T @ target)


def _assemble(records: Sequence[dict]) -> tuple[list[dict], list[dict[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a set of reference polymers, dropping the ones that cannot be."""
    kept, groups, tg, mass, gem = [], [], [], [], []
    for record in records:
        if record.get("glass_transition_k") is None:
            continue
        decomposition = repeat_unit_groups(record["repeat_unit"])
        if decomposition is None:
            continue
        kept.append(record)
        groups.append(decomposition)
        tg.append(float(record["glass_transition_k"]))
        mass.append(repeat_unit_mass(record["repeat_unit"]))
        gem.append(float(symmetric_gem_substitution(record["repeat_unit"])))
    return kept, groups, np.array(tg), np.array(mass), np.array(gem)


@functools.lru_cache(maxsize=1)
def glass_transition_model() -> GlassTransitionModel:
    """Fit the Yg coefficients and measure how well they transfer.

    Two error figures come out of this and they answer different questions.
    Leave-one-out over the fit split answers "how well does this model do on a
    polymer it was not fitted to", but only for polymers whose groups are
    still represented once they are removed - a group that appears in one
    polymer cannot be predicted from the others, and scoring that case would
    be measuring the reference set rather than the model.  The validation
    split answers the harder question, because it took no part in choosing the
    descriptors either.
    """
    fit_records, fit_groups, tg, mass, gem = _assemble(reference_split("fit"))
    keys = tuple(sorted({k for g in fit_groups for k in g}))
    counts = np.array([[g.get(k, 0) for k in keys] for g in fit_groups], dtype=float)
    support = {k: int((counts[:, i] > 0).sum()) for i, k in enumerate(keys)}

    features = np.hstack([counts, gem[:, None]])
    design = features / mass[:, None]
    coefficients = _solve(design, tg)

    residuals = []
    for i in range(len(tg)):
        mask = np.ones(len(tg), dtype=bool)
        mask[i] = False
        present = counts[i] > 0
        if present.any() and (counts[mask][:, present] > 0).sum(axis=0).min() < _MIN_GROUP_SUPPORT:
            continue  # scoring this would measure the reference set, not the model
        residuals.append(tg[i] - design[i] @ _solve(design[mask], tg[mask]))
    held = np.array(residuals) if residuals else np.array([np.nan])

    model = GlassTransitionModel(
        group_keys=keys,
        coefficients=tuple(float(c) for c in coefficients),
        support=support,
        n_fit=len(fit_records),
        held_out_rmse=float(np.sqrt(np.mean(held**2))),
        held_out_mae=float(np.mean(np.abs(held))),
        validation_rmse=0.0,
        validation_mae=0.0,
        validation_n=0,
    )

    errors = []
    for record in reference_split("validation"):
        prediction = predict_glass_transition(record["repeat_unit"], model)
        if prediction is None:
            continue
        errors.append(prediction - float(record["glass_transition_k"]))
    validation = np.array(errors) if errors else np.array([np.nan])

    return GlassTransitionModel(
        group_keys=model.group_keys,
        coefficients=model.coefficients,
        support=model.support,
        n_fit=model.n_fit,
        held_out_rmse=model.held_out_rmse,
        held_out_mae=model.held_out_mae,
        validation_rmse=float(np.sqrt(np.mean(validation**2))),
        validation_mae=float(np.mean(np.abs(validation))),
        validation_n=len(errors),
    )


def predict_glass_transition(
    unit_smiles: str, model: GlassTransitionModel | None = None
) -> float | None:
    """Tg of a homopolymer from its repeat unit, or None if it cannot be decomposed."""
    model = model or glass_transition_model()
    groups = repeat_unit_groups(unit_smiles)
    if groups is None or model.unknown_groups(groups):
        return None
    return model.predict(
        groups, symmetric_gem_substitution(unit_smiles), repeat_unit_mass(unit_smiles)
    )


# --------------------------------------------------------------------------
# Amorphous density from van der Waals volume
# --------------------------------------------------------------------------

#: Chain lengths whose volume difference isolates one interior repeat unit.
#: Two and four rather than three and six because the larger oligomers of the
#: longer repeat units embed unreliably, and the difference is unbiased at any
#: pair of lengths.
_VOLUME_SHORT = 2
_VOLUME_LONG = 4
#: Conformers averaged per oligomer.  A van der Waals volume depends slightly
#: on conformation through non-bonded overlap, and the spread across a handful
#: of conformers is well under a percent.
_VOLUME_CONFORMERS = 3


def _oligomer_volume(molecule, conformers: int) -> float:
    from rdkit.Chem import AllChem

    Chem = _rdkit()
    with_hydrogens = Chem.AddHs(molecule)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D  # the volume must not depend on when it was asked for
    params.useRandomCoords = True
    params.maxIterations = 2000
    ids = list(AllChem.EmbedMultipleConfs(with_hydrogens, numConfs=conformers, params=params))
    if not ids:
        raise ValueError("3D embedding failed for this repeat unit")
    if AllChem.MMFFHasAllMoleculeParams(with_hydrogens):
        AllChem.MMFFOptimizeMoleculeConfs(with_hydrogens, maxIters=1000)
    else:
        # Silicones and other main-group backbones fall outside MMFF94; UFF
        # covers them and a van der Waals volume is insensitive to the small
        # geometry differences between the two.
        AllChem.UFFOptimizeMoleculeConfs(with_hydrogens, maxIters=1000)
    return float(
        np.mean([AllChem.ComputeMolVolume(with_hydrogens, confId=i) for i in ids])
    )


@functools.lru_cache(maxsize=512)
def repeat_unit_vdw_volume(unit_smiles: str, conformers: int = _VOLUME_CONFORMERS) -> float:
    """Van der Waals volume of one interior repeat unit, in cubic angstroms.

    Computed geometrically from 3D structures rather than from a group table,
    and again as a difference between two chain lengths so that the overlap
    between bonded neighbours and the volume of the caps both cancel.  On the
    simple hydrocarbons where Bondi's group volumes are unambiguous this
    reproduces them to within a couple of percent.
    """
    long = _oligomer_volume(link_repeat_units(unit_smiles, _VOLUME_LONG), conformers)
    short = _oligomer_volume(link_repeat_units(unit_smiles, _VOLUME_SHORT), conformers)
    return (long - short) / (_VOLUME_LONG - _VOLUME_SHORT)


@dataclass(frozen=True, slots=True)
class PackingModel:
    """The single fitted constant relating occupied volume to bulk volume."""

    #: Bulk molar volume divided by van der Waals molar volume.
    packing_factor: float
    n_fit: int
    fit_rms_percent: float
    validation_rms_percent: float
    validation_max_percent: float
    validation_n: int

    def density(self, unit_smiles: str) -> float:
        """Amorphous density in g/cm^3."""
        volume = repeat_unit_vdw_volume(unit_smiles) * _A3_TO_CM3_PER_MOL
        return repeat_unit_mass(unit_smiles) / (self.packing_factor * volume)

    @property
    def sigma_percent(self) -> float:
        return max(self.fit_rms_percent, self.validation_rms_percent)


# --------------------------------------------------------------------------
# Reading a PolymerSpec
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChainAnalysis:
    """What a ``PolymerSpec`` amounts to for an additive repeat-unit model."""

    #: Mole-fraction-weighted group counts; fractional for a copolymer.
    groups: Mapping[int, float]
    #: Mole-fraction-weighted repeat-unit mass, g/mol.
    mass: float
    #: Mole-fraction-weighted symmetric geminal substitution count.
    gem: float
    #: Backbone repeat units and their mole fractions, in input order.
    units: tuple[tuple[str, float], ...]
    notes: tuple[str, ...] = ()
    #: Set when the polymer cannot be analysed at all.
    failure: str | None = None


def analyse_chain(spec: PolymerSpec, *, need_groups: bool = True) -> ChainAnalysis:
    """Reduce a polymer specification to weighted repeat-unit quantities."""
    backbone = [m for m in spec.monomers if m.role is not MonomerRole.END_GROUP]
    crosslinkers = [m for m in spec.monomers if m.role is MonomerRole.CROSSLINKER]
    notes: list[str] = []

    if crosslinkers:
        notes.append(
            "a crosslinker is present; the repeat-unit models here describe an "
            "uncrosslinked chain"
        )
    if len(backbone) > 1:
        notes.append(
            "more than one backbone monomer: treated as a random copolymer with "
            "additive repeat units, which is wrong for a block copolymer - a block "
            "copolymer has one transition per block, not the single averaged value "
            "reported here"
        )

    groups: dict[int, float] = {}
    mass = 0.0
    gem = 0.0
    units: list[tuple[str, float]] = []
    for monomer in backbone:
        smiles = monomer.smiles
        fraction = monomer.mole_fraction
        units.append((smiles, fraction))
        try:
            unit_mass = repeat_unit_mass(smiles)
        except ValueError as exc:
            return ChainAnalysis({}, 0.0, 0.0, (), tuple(notes), str(exc))
        mass += fraction * unit_mass
        gem += fraction * symmetric_gem_substitution(smiles)
        if need_groups:
            decomposition = repeat_unit_groups(smiles)
            if decomposition is None:
                return ChainAnalysis(
                    {}, 0.0, 0.0, (), tuple(notes),
                    f"the repeat unit {smiles!r} cannot be decomposed into the Joback "
                    "group set; carbonate, sulfone, siloxane and phosphazene backbones "
                    "have no group in it",
                )
            for key, count in decomposition.items():
                groups[key] = groups.get(key, 0.0) + fraction * count

    if mass <= 0.0:
        return ChainAnalysis({}, 0.0, 0.0, (), tuple(notes), "repeat-unit mass came out non-positive")
    return ChainAnalysis(groups, mass, gem, tuple(units), tuple(notes))


def _architecture_reason(spec: PolymerSpec) -> str | None:
    """Why this architecture sits outside a linear-chain correlation, if it does."""
    if spec.topology in (PolymerTopology.NETWORK, PolymerTopology.DENDRITIC):
        return (
            f"a {spec.topology.value} polymer is not a linear chain; crosslinks and "
            "dense branching change both the transition and the packing by amounts "
            "no repeat-unit sum can see"
        )
    if spec.crosslink_density is not None:
        return "a stated crosslink density puts this outside an uncrosslinked-chain correlation"
    return None


# --------------------------------------------------------------------------
# Experts
# --------------------------------------------------------------------------


class _PolymerExpert(Expert):
    """Shared plumbing: polymers only, and a software stamp that names the fit."""

    supported_classes = frozenset({MaterialClass.POLYMER})

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        from formulate import chem

        return "" if chem.rdkit_available() else "RDKit is required to read a repeat unit"

    def _spec(self, candidate: Candidate) -> PolymerSpec | None:
        return candidate.polymer


class PolymerGlassTransitionExpert(_PolymerExpert):
    """Glass transition from an additive molar function over Joback groups.

    The coefficients are fitted here, against the ``fit`` split of
    ``data/reference_polymers.json``.  Calling that fitted table "van
    Krevelen's" would be a claim about a book this repository does not
    contain; what is van Krevelen's is the *form* of the model, the molar
    glass-transition function ``Tg = sum n_i Yg_i / M``, and what is Joback's
    is the group set.  The numbers are ours and are only as good as the
    reference set behind them, which is why the uncertainty attached to every
    prediction is a held-out error rather than a residual.
    """

    id = "polymer_tg"
    version = "1"
    method = (
        "additive molar glass-transition function (van Krevelen form) over Joback "
        "groups, coefficients fitted in-repository against a measured reference set"
    )
    family = PropertyFamily.THERMAL
    supported_properties = frozenset({"glass_transition_temperature"})

    def _software(self) -> SoftwareEnvironment:
        import rdkit
        import thermo

        return SoftwareEnvironment.capture(rdkit=rdkit.__version__, thermo=thermo.__version__)

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        model = glass_transition_model()
        basis = (
            f"{model.n_fit} homopolymers with measured Tg, decomposed into "
            f"{len(model.group_keys)} Joback groups"
        )
        spec = self._spec(candidate)
        if spec is None:
            return ApplicabilityDomain.outside("candidate carries no polymer", basis=basis)

        architecture = _architecture_reason(spec)
        if architecture is not None:
            return ApplicabilityDomain.outside(architecture, basis=basis)

        if spec.tacticity in (Tacticity.ISOTACTIC, Tacticity.SYNDIOTACTIC):
            return ApplicabilityDomain.outside(
                f"the reference set is atactic or conventionally amorphous; an "
                f"{spec.tacticity.value} polymer can sit tens of kelvin either side of "
                "it, and the direction depends on the polymer",
                basis=basis,
            )

        mn = spec.number_average_molar_mass
        if mn is not None:
            mn_value = mn.to("g/mol").value
            if mn_value < _MIN_NUMBER_AVERAGE_MOLAR_MASS:
                return ApplicabilityDomain.outside(
                    f"Mn = {mn_value:.0f} g/mol is short enough that chain ends, not "
                    "repeat units, set the transition; this model is the high-polymer limit",
                    basis=basis,
                )

        analysis = analyse_chain(spec)
        if analysis.failure is not None:
            return ApplicabilityDomain.outside(analysis.failure, basis=basis)

        integral = {k: int(round(v)) for k, v in analysis.groups.items()}
        unknown = model.unknown_groups(integral)
        if unknown:
            return ApplicabilityDomain.outside(
                "this repeat unit uses groups no reference polymer contains: "
                + ", ".join(_group_names(unknown)),
                basis=basis,
            )
        thin = model.unsupported_groups(integral)
        if thin:
            return ApplicabilityDomain.outside(
                "the coefficient for "
                + ", ".join(_group_names(thin))
                + f" rests on fewer than {_MIN_GROUP_SUPPORT} reference polymers",
                basis=basis,
            )
        return ApplicabilityDomain(basis=basis)

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        spec = self._spec(request.candidate)
        if spec is None:
            return Prediction.unsupported(prop, self.id, "candidate carries no polymer")

        analysis = analyse_chain(spec)
        if analysis.failure is not None:
            return Prediction.unsupported(prop, self.id, analysis.failure)

        model = glass_transition_model()
        if model.unknown_groups({k: int(round(v)) for k, v in analysis.groups.items()}):
            return Prediction.unsupported(
                prop,
                self.id,
                "no coefficient was fitted for at least one group in this repeat unit",
            )

        value = model.predict(analysis.groups, analysis.gem, analysis.mass)
        notes = list(analysis.notes)
        std = model.sigma

        mn = spec.number_average_molar_mass
        if mn is not None:
            mn_value = mn.to("g/mol").value
            depression = _FOX_FLORY_SCALE / mn_value
            if depression > 0.25 * std:
                # The Fox-Flory constant is polymer-specific and unknown here, so
                # the chain-end depression is not corrected for.  Its size is
                # added to the uncertainty instead: the answer is the
                # high-polymer limit and is known to sit above the truth.
                std = math.hypot(std, depression)
                notes.append(
                    f"Mn = {mn_value:.0f} g/mol implies a chain-end depression of order "
                    f"{depression:.0f} K that this high-polymer correlation does not apply"
                )

        return self._make(
            prop,
            value,
            "K",
            request,
            domain,
            std=std,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                f"held-out error of the fit: {model.held_out_rmse:.0f} K leave-one-out over "
                f"{model.n_fit} fitted polymers, {model.validation_rmse:.0f} K over "
                f"{model.validation_n} polymers withheld from the fit entirely"
            ),
            notes=tuple(notes),
            repeat_units=[s for s, _ in analysis.units],
            repeat_unit_mass_g_mol=round(analysis.mass, 4),
        )


def _group_names(keys: Sequence[int]) -> list[str]:
    from thermo.group_contribution.joback import JOBACK_GROUPS

    return [JOBACK_GROUPS[k].group for k in keys]


#: The packing model is fitted at room temperature and carries no temperature
#: dependence of its own, so it is only offered near the temperature it was
#: fitted at.  A polymer's volumetric expansion coefficient is of order
#: 2-7e-4 per kelvin, which over this window moves the density by about a
#: percent - inside the model's own error, and outside it not.
_DENSITY_TEMPERATURE_WINDOW_K = (273.0, 323.0)

#: Elements the packing factor was fitted over.  A single factor asserts that
#: every polymer wastes the same fraction of its bulk volume on free volume,
#: and that is only true among chemistries with comparable intermolecular
#: forces.  Poly(dimethylsiloxane) is the standing counterexample: its
#: measured density is 0.97 g/cm^3 where this model says 1.13, because a
#: siloxane backbone packs far more loosely than any carbon backbone in the
#: reference set.  Rather than quietly extrapolate, elements the fit never
#: saw put the answer out of domain.
_FITTED_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "Cl"})


def repeat_unit_elements(unit_smiles: str) -> frozenset[str]:
    """Element symbols present in a repeat unit, hydrogens included."""
    Chem = _rdkit()
    molecule = Chem.AddHs(link_repeat_units(unit_smiles, 2))
    return frozenset(a.GetSymbol() for a in molecule.GetAtoms())


@functools.lru_cache(maxsize=1)
def _stored_volumes() -> Mapping[str, float]:
    return {
        p["repeat_unit"]: float(p["van_der_waals_volume_a3"])
        for p in reference_polymers()
        if p.get("van_der_waals_volume_a3") is not None
    }


def cached_vdw_volume(unit_smiles: str) -> float:
    """Van der Waals volume of a repeat unit, reusing the stored value if there is one.

    The stored values in ``reference_polymers.json`` were produced by
    :func:`repeat_unit_vdw_volume` and are kept only so that fitting the
    packing factor does not re-embed sixty oligomers in 3D every time the
    module is imported.  A repeat unit that is not in the file is computed.
    """
    stored = _stored_volumes().get(unit_smiles)
    return stored if stored is not None else repeat_unit_vdw_volume(unit_smiles)


@functools.lru_cache(maxsize=1)
def packing_model() -> PackingModel:
    """Fit the one constant that turns occupied volume into bulk volume.

    A glassy polymer and a rubbery one do not pack the same way at 298 K -
    one is below its transition and one is above it - so separate packing
    factors, and a full expansion model about Tg, were both tried here.  Both
    improved the fit split and made the validation split worse, so neither
    survives: the single factor is what this reference set actually supports.
    """
    def parts(split: str) -> tuple[np.ndarray, np.ndarray]:
        records = [
            p for p in reference_split(split)
            if p.get("van_der_waals_volume_a3") is not None
            and p.get("amorphous_density_g_cm3") is not None
        ]
        volume = np.array(
            [p["van_der_waals_volume_a3"] for p in records]
        ) * _A3_TO_CM3_PER_MOL
        mass = np.array([repeat_unit_mass(p["repeat_unit"]) for p in records])
        density = np.array([p["amorphous_density_g_cm3"] for p in records])
        return mass / (density * volume), (mass / volume) / density

    ratios, _ = parts("fit")
    factor = float(np.exp(np.log(ratios).mean()))

    def rms_percent(split: str) -> tuple[float, float, int]:
        r, _ = parts(split)
        error = (r / factor - 1.0) * 100.0
        if error.size == 0:
            return float("nan"), float("nan"), 0
        return (
            float(np.sqrt(np.mean(error**2))),
            float(np.max(np.abs(error))),
            int(error.size),
        )

    fit_rms, _, n_fit = rms_percent("fit")
    val_rms, val_max, n_val = rms_percent("validation")
    return PackingModel(
        packing_factor=factor,
        n_fit=n_fit,
        fit_rms_percent=fit_rms,
        validation_rms_percent=val_rms,
        validation_max_percent=val_max,
        validation_n=n_val,
    )


class PolymerDensityExpert(_PolymerExpert):
    """Amorphous density from van der Waals volume and one packing factor.

    ``rho = M / (k V_w N_A)``.  The van der Waals volume is measured off a 3D
    structure rather than summed from a group table, which is why this expert
    answers for repeat units the glass-transition expert refuses: a siloxane
    or a carbonate has no Joback group but it certainly has a volume.

    The value is the density of the *amorphous* phase.  A semicrystalline
    sample is denser in proportion to its crystallinity, and crystallinity is
    a property of how the sample was made, not of what it is made from, so it
    is not predicted here and not silently folded in.
    """

    id = "polymer_density"
    version = "1"
    method = (
        "amorphous density from the geometric van der Waals volume of the repeat unit "
        "and a single packing factor fitted in-repository"
    )
    family = PropertyFamily.MECHANICAL
    supported_properties = frozenset({"amorphous_density"})

    def _software(self) -> SoftwareEnvironment:
        import rdkit

        return SoftwareEnvironment.capture(rdkit=rdkit.__version__)

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        model = packing_model()
        basis = (
            f"a packing factor of {model.packing_factor:.3f} fitted to {model.n_fit} "
            "amorphous polymer densities at room temperature"
        )
        spec = self._spec(candidate)
        if spec is None:
            return ApplicabilityDomain.outside("candidate carries no polymer", basis=basis)
        architecture = _architecture_reason(spec)
        if architecture is not None:
            return ApplicabilityDomain.outside(architecture, basis=basis)

        foreign: set[str] = set()
        for monomer in spec.monomers:
            if monomer.role is MonomerRole.END_GROUP:
                continue
            try:
                foreign |= repeat_unit_elements(monomer.smiles) - _FITTED_ELEMENTS
            except ValueError as exc:
                return ApplicabilityDomain.outside(str(exc), basis=basis)
        if foreign:
            return ApplicabilityDomain.outside(
                "the packing factor was fitted only over "
                + ", ".join(sorted(_FITTED_ELEMENTS))
                + "; this repeat unit also contains "
                + ", ".join(sorted(foreign))
                + ", whose intermolecular forces set a different packing",
                basis=basis,
            )
        return ApplicabilityDomain(basis=basis)

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        spec = self._spec(request.candidate)
        if spec is None:
            return Prediction.unsupported(prop, self.id, "candidate carries no polymer")

        temperature = request.conditions.temperature_k
        if temperature is None:
            return Prediction.unsupported(
                prop, self.id, "amorphous density is temperature dependent and no temperature was given"
            )
        low, high = _DENSITY_TEMPERATURE_WINDOW_K
        if not low <= temperature <= high:
            return Prediction.unsupported(
                prop,
                self.id,
                f"the packing factor was fitted at room temperature and carries no "
                f"temperature dependence; {temperature:.0f} K is outside the "
                f"{low:.0f}-{high:.0f} K window over which that is defensible",
            )

        analysis = analyse_chain(spec, need_groups=False)
        if analysis.failure is not None:
            return Prediction.unsupported(prop, self.id, analysis.failure)

        model = packing_model()
        try:
            volume = sum(
                fraction * cached_vdw_volume(smiles) for smiles, fraction in analysis.units
            )
        except ValueError as exc:
            return Prediction.unsupported(prop, self.id, str(exc))
        if volume <= 0.0:
            return Prediction.unsupported(prop, self.id, "repeat-unit volume came out non-positive")

        density = analysis.mass / (model.packing_factor * volume * _A3_TO_CM3_PER_MOL)
        density_si = density * 1000.0  # g/cm^3 -> kg/m^3, the canonical unit
        return self._make(
            prop,
            density_si,
            "kg/m^3",
            request,
            domain,
            std=density_si * model.sigma_percent / 100.0,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                f"held-out error of the fit: {model.validation_rms_percent:.1f} per cent over "
                f"{model.validation_n} polymers withheld from it, against "
                f"{model.fit_rms_percent:.1f} per cent on the {model.n_fit} it was fitted to"
            ),
            conditions=request.conditions,
            notes=tuple(analysis.notes) + (
                "amorphous phase only; a semicrystalline sample is denser by an amount "
                "set by processing rather than by chemistry",
            ),
            packing_factor=round(model.packing_factor, 4),
            van_der_waals_volume_a3=round(volume, 3),
        )
