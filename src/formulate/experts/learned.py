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
import os
import pathlib
from typing import Any

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest

_DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "boiling_point_measurements.json"

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
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


@functools.lru_cache(maxsize=1)
def boiling_point_model() -> dict[str, Any]:
    """Fit the forest, or load the cached one. Deterministic given the data."""
    import joblib
    import numpy as np

    document = json.loads(_DATA.read_text())
    digest = hashlib.sha256(_DATA.read_bytes()).hexdigest()[:16]
    cached = _cache_directory() / f"boiling_point_{digest}.joblib"
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
        "digest": digest,
    }
    try:
        joblib.dump(model, cached, compress=3)
    except Exception:  # a read-only cache is not a reason to fail
        pass
    return model


class LearnedBoilingPointExpert(Expert):
    """A boiling point for structures no group table covers."""

    id = "learned_boiling_point"
    version = "1"
    method = "random forest over RDKit descriptors, fitted to measured boiling points"
    family = PropertyFamily.THERMAL
    supported_classes = frozenset({MaterialClass.MOLECULE})
    supported_properties = frozenset({"normal_boiling_point"})

    def is_available(self) -> bool:
        from formulate import chem

        return sklearn_available() and chem.rdkit_available() and _DATA.exists()

    def unavailable_reason(self) -> str:
        from formulate import chem

        if not sklearn_available():
            return "scikit-learn is not installed; install formulate[learned]"
        if not chem.rdkit_available():
            return "RDKit is required to compute descriptors"
        if not _DATA.exists():
            return "the measured boiling point data is missing from this installation"
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

        model = boiling_point_model()
        distance = self._distance(model, features)
        if distance > model["threshold"]:
            return ApplicabilityDomain.outside(
                f"this structure sits {distance / model['threshold']:.1f} times further "
                "from the training data than the training molecules sit from each other, "
                "so any number here would be an extrapolation rather than a prediction",
                basis=f"{model['n_train']} molecules with measured boiling points",
            )
        return ApplicabilityDomain(
            basis=f"{model['n_train']} molecules with measured boiling points; this one "
            "falls inside their descriptor range"
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

        model = boiling_point_model()
        row = features.reshape(1, -1)
        value = float(model["forest"].predict(row)[0])
        spread = float(np.std([tree.predict(row)[0] for tree in model["forest"].estimators_]))
        std = max(spread * model["scale"], 1.0)

        # The forest's own disagreement, as a fraction of what it predicts. A
        # nearest-neighbour distance says whether this molecule looks like the
        # training data; this says whether the trees actually agree about it,
        # and the two disagree often enough to need both.
        relative = std / max(abs(value), 1.0)
        if relative > _MAX_RELATIVE_SPREAD:
            return Prediction.unsupported(
                prop,
                self.id,
                f"the trees disagree by {relative:.0%} of the value they predict "
                f"({value:.0f} K plus or minus {std:.0f}), which over the held-out set "
                "marks a prediction worth about twice the usual error; declining rather "
                "than reporting it",
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
                "the forest's own disagreement, scaled by a factor fitted so that "
                f"{model['held_out_coverage']:.0%} of held-out compounds fall inside one "
                f"sigma; the held-out mean absolute error is {model['held_out_mae']:.1f} K"
            ),
            notes=(
                f"fitted to {model['n_train']} measured boiling points, tested on "
                f"{model['n_held_out']} held out",
                "Joback is the better estimate wherever it has groups to match, at 14.8 K "
                "against 23.5 over the calibration set; this expert exists for the "
                "structures it cannot type at all",
                "training data excludes every value whose source was Joback, and excludes "
                "the calibration compounds by InChIKey",
            ),
        )
