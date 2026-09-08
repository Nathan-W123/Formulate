"""A learned boiling point, for the molecules group contribution cannot type.

Every other expert in this panel is a published deterministic method, and that
was the right default: over the fifty calibration compounds Joback predicts a
boiling point to 14.8 K where a random forest trained here manages 23.5. On
chemistry the incumbent covers, group contribution wins and this expert should
lose.

The reason it exists is a different measurement. A discovery run was asked for
a high-boiling, hydrophobic, low-freezing liquid - a region no reference
compound occupies - and the evolutionary explorer did the right thing: it found
the single feasible compound, propylene carbonate, and began generating
analogues of it. Joback has no group for a cyclic carbonate, so every one of
those came back with no boiling point, no critical constants and no density,
because in this panel the whole chain hangs off the boiling point::

    groups -> boiling point -> Tc, Pc, Vc -> Rackett   -> density
                                          -> Brock-Bird -> surface tension

Thirty-five novel structures were generated and none could be ranked. A
generator that invents chemistry its evaluator refuses to score cannot
discover anything, and no amount of accuracy on catalogue compounds fixes that.

So the comparison that matters is not the one on the reference set. On the six
carbonates that run actually produced, Joback answers none and this expert
answers all six, with its own domain check reporting every one as inside the
training distribution rather than an extrapolation - cyclic carbonates are well
represented among ten thousand measured boiling points, they are simply absent
from a table written in 1987.

Three things this expert has to carry, and they are the ones the day's failures
taught rather than a matter of taste:

it refuses
    A nearest-neighbour distance in standardised descriptor space, with the
    threshold set at the 99th percentile of that distance inside the training
    set. Without this a learned expert answers confidently everywhere, which is
    how a pretrained toxicity model scored benzene as safer than ethanol.

its uncertainty is earned
    :func:`~formulate.evaluation.engine.prefer` selects on the tightest stated
    uncertainty. A model that asserts confidence it has not measured does not
    merely predict badly, it outranks the honest experts and degrades the
    panel. The spread here is the forest's own disagreement, scaled by a factor
    fitted so that about 68 per cent of held-out compounds fall inside it.

it is reproducible
    The repository carries the training data, not a pickled model. A 34 MB
    fitted forest is opaque, breaks across scikit-learn versions and cannot be
    audited; 307 KB of measured boiling points can be read, checked and
    retrained. The model is a build artefact, cached on first use.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import pathlib
from dataclasses import dataclass
from typing import Any

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest

_DATA_DIRECTORY = pathlib.Path(__file__).resolve().parent.parent / "data"

#: Property -> (data file, unit, what the incumbent manages where it applies).
#:
#: Both entries exist for coverage rather than accuracy, and both say so on
#: every prediction. A melting point is the weaker of the two by some way: it is
#: governed by crystal packing, which neither a group table nor a descriptor
#: model represents, and Joback's own figure for it is deliberately inflated to
#: 40 K for that reason. Learning it does not make it good; it makes it exist
#: for the structures that would otherwise have nothing.
@dataclass(frozen=True, slots=True)
class Learnable:
    """What one learnable property needs that the others cannot supply.

    This was three positional strings until a second property was added, and
    every boiling-point-shaped assumption in the class below turned out to be
    hard-coded rather than declared: a noise floor of one kelvin, a relative
    spread measured against a denominator floored at one kelvin, and the letter
    K in three separate messages. On a refractive index of 1.36 the noise floor
    alone forced a stated spread of 1.0, the gate then read that as the trees
    disagreeing by 73 per cent, and the expert declined every molecule it was
    asked about. The abstraction existed; it had simply never been used twice.
    """

    #: File under ``formulate/data`` holding the measurements.
    filename: str
    #: Unit the values are in, empty for a dimensionless property.
    unit: str
    #: How this expert stands against the non-learned route, measured.
    comparison: str
    #: Smallest stated uncertainty that means anything for this property, in
    #: its own unit. Below this the forest is claiming a precision the
    #: measurements it was fitted to do not have.
    noise_floor: float
    #: Denominator floor for the relative-spread gate, in the same unit. It
    #: exists so a value near zero cannot make any spread look infinite.
    scale_floor: float
    #: What the training data is and what was held out of it.
    provenance_note: str

    def format(self, value: float) -> str:
        """A value with its unit, shown to where the noise floor sits.

        A boiling point whose floor is one kelvin is written to the kelvin; a
        refractive index whose floor is 0.0005 is written to four decimals.
        Printing further would claim a precision the property does not have,
        which is the same error as stating one.
        """
        digits = max(0, min(6, -int(math.floor(math.log10(self.noise_floor)))))
        return f"{value:.{digits}f}" + (f" {self.unit}" if self.unit else "")


LEARNABLE: dict[str, Learnable] = {
    "normal_boiling_point": Learnable(
        filename="boiling_point_measurements.json",
        unit="K",
        comparison=(
            "Joback is the better estimate wherever it has groups to match, at 14.8 K "
            "against 23.5 over the calibration set; this expert exists for the "
            "structures it cannot type at all"
        ),
        # A boiling point tabulated to better than a kelvin is unusual, and the
        # compiled sources round to about that.
        noise_floor=1.0,
        scale_floor=1.0,
        provenance_note=(
            "training data excludes every value whose source was Joback, and excludes "
            "the calibration compounds by InChIKey"
        ),
    ),
    "refractive_index": Learnable(
        filename="refractive_index_measurements.json",
        unit="",
        comparison=(
            "Lorentz-Lorenz is the better estimate wherever a real density is "
            "available, at 0.0123 against 0.0165 on 385 compounds this model never "
            "saw; fed an estimated density instead it goes to 0.1995 with an RMSE of "
            "2.55, because the equation has a pole and a molar volume slightly too "
            "small sends the answer to infinity. This expert is what to use when the "
            "density is not known"
        ),
        # Handbook refractive indices are tabulated to four decimal places and
        # differ between sources in the fourth, so a tenth of that is the floor.
        noise_floor=0.0005,
        # Every liquid refractive index is above one, so this floor never binds;
        # it is stated rather than omitted because the gate divides by it.
        scale_floor=1.0,
        provenance_note=(
            "CRC handbook values resolved from CAS; the 385 compounds carrying a "
            "tabulated density are kept in, and the published comparison against "
            "Lorentz-Lorenz was measured on a forest refitted without them"
        ),
    ),
}


#: A melting point was trained and is not shipped, recorded here so the
#: experiment is not repeated.
#:
#: Twenty-six thousand measured melting points, the same forest, the same
#: gates: held-out mean absolute error 52.0 K against Joback's measured 27.8,
#: and the spread gate then refused four of five test molecules including
#: toluene, which Joback answers. A melting point is set by how molecules pack
#: in a crystal, and neither a group table nor a molecular descriptor sees a
#: crystal. Coverage is worthless when the covered answers are twice as wrong
#: as the ones already available and the model declines most of them anyway.
_MELTING_POINT_REJECTED = (
    "trained on 26226 measured values, held-out MAE 52.0 K against Joback's 27.8, "
    "and its own spread gate declined most molecules; not shipped"
)

#: A relative permittivity was trained and is not shipped either, for a
#: different and more interesting reason than the melting point's.
#:
#: 1212 measured liquids, the same forest, the same gates. Ungated it is
#: useless: mean absolute error 3.65 on a property whose values run from 1.9 to
#: 104, with an RMSE of 8.59. Gated at the usual fifteen per cent it looks
#: excellent - 0.08 mean absolute error - and that number is an artefact. The
#: gate admits 34 of 243 test compounds, whose permittivities run 1.89 to 6.54
#: with a median of 2.16, and 68 per cent of them sit below 3 against 21 per
#: cent of the full test set. The admitted list is heptane, isooctane,
#: octyl bromide, stearic acid: the model has learned to recognise nonpolar
#: molecules and report that they are nonpolar.
#:
#: That is not a model that answers one molecule in seven accurately. It is a
#: model that answers the question nobody needed answering, and declines every
#: molecule whose dielectric constant is actually in doubt. Unlike the melting
#: point there is no group-contribution alternative here, so this leaves the
#: property uncovered - which is the honest state rather than a bad answer
#: wearing a tight error bar.
_RELATIVE_PERMITTIVITY_REJECTED = (
    "trained on 1212 measured liquids; ungated MAE 3.65 on a 1.9-104 range, and its "
    "spread gate admits only nonpolar molecules (median permittivity 2.16), so what "
    "it answers well is what needed no model; not shipped"
)

_DATA = _DATA_DIRECTORY / LEARNABLE["normal_boiling_point"].filename

#: Trees in the forest. Three hundred is where held-out error stops improving
#: on this set; more only costs fitting time.
_TREES = 300

#: Fraction of held-out compounds a one-sigma bound should contain.
_TARGET_COVERAGE = 0.68

#: Refuse when the forest's own scaled disagreement exceeds this fraction of
#: the value it predicts.
#:
#: A nearest-neighbour distance is not enough on its own. Water passed that
#: check and came back at 700 K against a true 373, because in a standardised
#: descriptor space a three-atom molecule is not obviously far from anything.
#: The forest knew - it quoted plus or minus 486 K - and the domain test did
#: not listen.
#:
#: The disagreement turns out to predict the error closely, measured over the
#: 2050 held-out compounds::
#:
#:     refuse above    answers    mean absolute error of what it answers
#:     (no gate)          100%    51.8 K
#:     0.20                94%    31.0 K
#:     0.15                91%    26.6 K
#:     0.10                85%    23.4 K
#:
#: Fifteen per cent is where this sits: it declines one molecule in eleven and
#: halves the error on the rest. Water, at 0.69, is refused by any of these.
_MAX_RELATIVE_SPREAD = 0.15

#: Nearest-neighbour distance quantile inside the training set that marks the
#: edge of the domain. A molecule further from the training data than 99 per
#: cent of training molecules are from each other is refused.
_DOMAIN_QUANTILE = 0.99


#: Largest descriptor magnitude a forest will be shown, comfortably inside the
#: float32 range scikit-learn casts to.
_DESCRIPTOR_LIMIT = 1.0e30


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
    except Exception:
        return False
    return True


def _cache_directory() -> pathlib.Path:
    root = os.environ.get("FORMULATE_CACHE") or os.path.expanduser("~/.cache/formulate")
    path = pathlib.Path(root) / "learned"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _descriptor_names() -> list[str]:
    from rdkit.Chem import Descriptors

    return [name for name, _ in Descriptors._descList]


def featurise(smiles: str):
    """RDKit's 2D descriptors for one structure, or None if it will not parse."""
    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.ML.Descriptors import MoleculeDescriptors

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(_descriptor_names())
    values = np.array(calculator.CalcDescriptors(mol), dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    # Some RDKit descriptors are finite and still unusable: Ipc grows roughly
    # exponentially with molecule size and reaches 1e40 on the larger members of
    # a twenty-six thousand compound set, which overflows the float32 a forest
    # fits in. Clipping keeps the descriptor's ordering where it is meaningful
    # and stops the tail from breaking the fit.
    return np.clip(values, -_DESCRIPTOR_LIMIT, _DESCRIPTOR_LIMIT)


@functools.lru_cache(maxsize=len(LEARNABLE))
def learned_model(prop: str = "normal_boiling_point") -> dict[str, Any]:
    """Fit the forest for ``prop``, or load the cached one.

    Deterministic given the data, so the cache key is the data's own digest: a
    changed dataset refits rather than silently serving a model of the old one.
    """
    import joblib
    import numpy as np

    path = _DATA_DIRECTORY / LEARNABLE[prop].filename
    document = json.loads(path.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    cached = _cache_directory() / f"{prop}_{digest}.joblib"
    if cached.exists():
        try:
            return joblib.load(cached)
        except Exception:
            cached.unlink(missing_ok=True)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestNeighbors

    rows, targets = [], []
    for smiles, value in document["rows"]:
        features = featurise(smiles)
        if features is None:
            continue
        rows.append(features)
        targets.append(value)
    x = np.vstack(rows)
    y = np.array(targets)

    x_fit, x_test, y_fit, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    forest = RandomForestRegressor(
        n_estimators=_TREES, min_samples_leaf=2, max_features="sqrt",
        n_jobs=-1, random_state=0,
    )
    forest.fit(x_fit, y_fit)

    predicted = forest.predict(x_test)
    residual = np.abs(predicted - y_test)
    spread = np.std([tree.predict(x_test) for tree in forest.estimators_], axis=0)
    spread[spread <= 0] = 1e-9
    scale = float(np.quantile(residual / spread, _TARGET_COVERAGE))

    mean, deviation = x_fit.mean(0), x_fit.std(0)
    deviation[deviation == 0] = 1.0
    neighbours = NearestNeighbors(n_neighbors=2).fit((x_fit - mean) / deviation)
    inside, _ = neighbours.kneighbors((x_fit - mean) / deviation)
    threshold = float(np.quantile(inside[:, 1], _DOMAIN_QUANTILE))

    model = {
        "forest": forest, "mean": mean, "deviation": deviation,
        "neighbours": neighbours, "scale": scale, "threshold": threshold,
        "held_out_mae": float(np.mean(residual)),
        "held_out_coverage": float(np.mean(residual <= scale * spread)),
        "n_train": int(len(y_fit)), "n_held_out": int(len(y_test)),
        "digest": digest, "property": prop,
    }
    try:
        joblib.dump(model, cached, compress=3)
    except Exception:  # a read-only cache is not a reason to fail
        pass
    return model


def boiling_point_model() -> dict[str, Any]:
    """The boiling point model, by its original name."""
    return learned_model("normal_boiling_point")


class _LearnedExpert(Expert):
    """One fitted property, for structures no group table covers."""

    #: Set by each subclass; the property this expert answers. The base class
    #: validates supported_properties when the subclass is created, so both are
    #: declared there rather than derived here.
    prop: str = ""
    version = "1"

    @property
    def spec(self) -> Learnable:
        """Everything about this property that the base class must not assume."""
        return LEARNABLE[self.prop]

    family = PropertyFamily.THERMAL
    supported_classes = frozenset({MaterialClass.MOLECULE})
    supported_properties: frozenset[str] = frozenset()

    def _data_path(self) -> pathlib.Path:
        return _DATA_DIRECTORY / self.spec.filename

    def is_available(self) -> bool:
        from formulate import chem

        return sklearn_available() and chem.rdkit_available() and self._data_path().exists()

    def unavailable_reason(self) -> str:
        from formulate import chem

        if not sklearn_available():
            return "scikit-learn is not installed; install formulate[learned]"
        if not chem.rdkit_available():
            return "RDKit is required to compute descriptors"
        if not self._data_path().exists():
            return f"the measured {self.prop.replace('_', ' ')} data is missing"
        return ""

    def _software(self) -> SoftwareEnvironment:
        import sklearn

        return SoftwareEnvironment.capture(scikit_learn=sklearn.__version__)

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        if candidate.molecule is None:
            return ApplicabilityDomain.outside("candidate carries no molecule")
        features = featurise(candidate.molecule.smiles)
        if features is None:
            return ApplicabilityDomain.outside("this structure could not be parsed")

        model = learned_model(self.prop)
        distance = self._distance(model, features)
        if distance > model["threshold"]:
            return ApplicabilityDomain.outside(
                f"this structure sits {distance / model['threshold']:.1f} times further "
                "from the training data than the training molecules sit from each other, "
                "so any number here would be an extrapolation rather than a prediction",
                basis=f"{model['n_train']} molecules with a measured "
                f"{self.prop.replace('_', ' ')}",
            )
        return ApplicabilityDomain(
            basis=f"{model['n_train']} molecules with a measured "
            f"{self.prop.replace('_', ' ')}; this one falls inside their descriptor range"
        )

    @staticmethod
    def _distance(model, features):
        scaled = ((features - model["mean"]) / model["deviation"]).reshape(1, -1)
        distance, _ = model["neighbours"].kneighbors(scaled, n_neighbors=1)
        return float(distance[0, 0])

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        import numpy as np

        smiles = request.candidate.molecule.smiles  # type: ignore[union-attr]
        features = featurise(smiles)
        if features is None:
            return Prediction.failed(prop, self.id, "this structure could not be parsed")

        model = learned_model(self.prop)
        row = features.reshape(1, -1)
        value = float(model["forest"].predict(row)[0])
        spread = float(np.std([tree.predict(row)[0] for tree in model["forest"].estimators_]))
        std = max(spread * model["scale"], self.spec.noise_floor)

        # The forest's own disagreement, as a fraction of what it predicts. A
        # nearest-neighbour distance says whether this molecule looks like the
        # training data; this says whether the trees actually agree about it,
        # and the two disagree often enough to need both.
        relative = std / max(abs(value), self.spec.scale_floor)
        if relative > _MAX_RELATIVE_SPREAD:
            return Prediction.unsupported(
                prop,
                self.id,
                f"the trees disagree by {relative:.0%} of the value they predict "
                f"({self.spec.format(value)} plus or minus {self.spec.format(std)}), "
                "which over the held-out set marks a prediction worth about twice the "
                "usual error; declining rather than reporting it",
            )

        return self._make(
            prop,
            value,
            self.spec.unit,
            request,
            domain,
            std=std,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                "the forest's own disagreement, scaled by a factor fitted so that "
                f"{model['held_out_coverage']:.0%} of held-out compounds fall inside one "
                "sigma; the held-out mean absolute error is "
                f"{self.spec.format(model['held_out_mae'])}"
            ),
            notes=(
                f"fitted to {model['n_train']} measured values of "
                f"{self.prop.replace('_', ' ')}, tested on {model['n_held_out']} held out",
                self.spec.comparison,
                self.spec.provenance_note,
            ),
        )


class LearnedBoilingPointExpert(_LearnedExpert):
    """A boiling point for structures no group table covers."""

    id = "learned_boiling_point"
    prop = "normal_boiling_point"
    supported_properties = frozenset({"normal_boiling_point"})
    method = "random forest over RDKit descriptors, fitted to measured boiling points"



class LearnedRefractiveIndexExpert(_LearnedExpert):
    """A refractive index for structures whose density is not known.

    The companion to :class:`~formulate.experts.optical.LorentzLorenzExpert`,
    which is better whenever a real density is available and much worse when it
    is not. Neither has precedence written into it: both state their spreads
    honestly and ``prefer()`` chooses.
    """

    id = "learned_refractive_index"
    prop = "refractive_index"
    family = PropertyFamily.ELECTRICAL
    supported_properties = frozenset({"refractive_index"})
    method = "random forest over RDKit descriptors, fitted to measured refractive indices"
