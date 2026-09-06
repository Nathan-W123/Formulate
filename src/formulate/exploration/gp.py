"""A Gaussian process on numpy and scipy alone.

Specification section 3 asks for "Bayesian/black-box optimization over
continuous recipe variables such as ratios, molecular weight, crosslink
density, and process conditions".  Neither scikit-learn nor botorch is
available here, so the surrogate is written directly.

The parts that matter are the numerical ones.  A Gaussian process is a
Cholesky factorisation of a matrix that becomes singular exactly when the data
are informative - two nearby points make two nearly identical rows - so the
factorisation must be defended rather than attempted and hoped for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve, cholesky
from scipy.optimize import minimize

#: Jitter added to the diagonal, in units of the (standardised) output variance.
#: Tried in order until the factorisation succeeds. Absolute rather than scaled
#: by the signal variance, so that the log-likelihood stays a smooth function of
#: the hyperparameters: jitter proportional to a hyperparameter would put a
#: discontinuity in the very objective being optimised.
_JITTER_LADDER = (1e-10, 1e-8, 1e-6, 1e-4, 1e-2)


@dataclass(frozen=True, slots=True)
class GPHyperparameters:
    """Kernel parameters, stored in log space where they are unconstrained."""

    log_signal_variance: float
    #: One per input dimension: automatic relevance determination.
    log_lengthscales: np.ndarray
    log_noise: float

    @property
    def lengthscales(self) -> np.ndarray:
        return np.exp(self.log_lengthscales)

    def to_vector(self) -> np.ndarray:
        return np.concatenate(
            ([self.log_signal_variance], self.log_lengthscales, [self.log_noise])
        )

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "GPHyperparameters":
        return cls(
            log_signal_variance=float(vector[0]),
            log_lengthscales=np.asarray(vector[1:-1], dtype=float),
            log_noise=float(vector[-1]),
        )


def matern52(a: np.ndarray, b: np.ndarray, hyper: GPHyperparameters) -> np.ndarray:
    """Matern 5/2 kernel with a lengthscale per dimension.

    Matern 5/2 rather than a squared exponential because the latter assumes the
    objective is infinitely differentiable, which no property surface is; the
    resulting over-smoothing makes the posterior variance collapse and the
    acquisition stop exploring.
    """
    scaled_a = a / hyper.lengthscales
    scaled_b = b / hyper.lengthscales
    squared = (
        (scaled_a**2).sum(axis=1)[:, None]
        + (scaled_b**2).sum(axis=1)[None, :]
        - 2.0 * scaled_a @ scaled_b.T
    )
    distance = np.sqrt(np.maximum(squared, 0.0))
    root5 = np.sqrt(5.0) * distance
    return np.exp(hyper.log_signal_variance) * (1.0 + root5 + root5**2 / 3.0) * np.exp(-root5)


class GaussianProcess:
    """A GP regressor with ARD, fitted by marginal likelihood."""

    def __init__(self, jitter_ladder: tuple[float, ...] = _JITTER_LADDER) -> None:
        self.jitter_ladder = jitter_ladder
        self.x: np.ndarray | None = None
        self.y_mean = 0.0
        self.y_scale = 1.0
        self._y_standard: np.ndarray | None = None
        self.hyper: GPHyperparameters | None = None
        self._factor = None
        self._alpha: np.ndarray | None = None
        self.jitter_used = 0.0

    # -- fitting -----------------------------------------------------------

    def fit(self, x: np.ndarray, y: np.ndarray, *, restarts: int = 8, seed: int = 0) -> "GaussianProcess":
        """Fit hyperparameters by maximising the log marginal likelihood.

        The targets are standardised first. Without it the signal variance has
        to absorb the scale of the data, which puts the optimum far out in log
        space and makes the same prior wrong for every new objective.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if x.shape[0] != y.size:
            raise ValueError(f"{x.shape[0]} inputs against {y.size} outputs")
        if x.shape[0] < 2:
            raise ValueError("a Gaussian process needs at least two observations")

        self.x = x
        self.y_mean = float(y.mean())
        self.y_scale = float(y.std()) or 1.0
        self._y_standard = (y - self.y_mean) / self.y_scale

        rng = np.random.default_rng(seed)
        best_value = np.inf
        best_vector = None

        # A lengthscale near the spread of the inputs is the only starting point
        # that is not arbitrary; the restarts explore around it.
        spread = np.maximum(x.max(axis=0) - x.min(axis=0), 1e-6)
        base = np.concatenate(([0.0], np.log(spread), [np.log(1e-2)]))

        for attempt in range(max(1, restarts)):
            start = base if attempt == 0 else base + rng.normal(scale=1.0, size=base.size)
            try:
                result = minimize(
                    self._negative_log_marginal_likelihood,
                    start,
                    method="L-BFGS-B",
                    bounds=[(-8.0, 8.0)] * base.size,
                    options={"maxiter": 200},
                )
            except Exception:
                continue
            if np.isfinite(result.fun) and result.fun < best_value:
                best_value, best_vector = float(result.fun), result.x

        if best_vector is None:
            best_vector = base
        self.hyper = GPHyperparameters.from_vector(best_vector)
        self._factorise(self.hyper)
        return self

    def _negative_log_marginal_likelihood(self, vector: np.ndarray) -> float:
        """Objective for hyperparameter fitting.

        Numerical gradients are left to the optimiser rather than derived
        analytically. The analytic derivative with respect to the log signal
        variance is only correct if the jitter does not itself depend on that
        variance, and a scaled jitter is the usual way to make the
        factorisation robust - so the two conveniences are incompatible. At the
        sizes this surrogate sees, a Cholesky costs under a millisecond and the
        finite differences are cheap enough not to matter.
        """
        try:
            hyper = GPHyperparameters.from_vector(vector)
            factor, _, log_determinant = self._build(hyper)
        except Exception:
            return np.inf
        alpha = cho_solve((factor, True), self._y_standard)
        n = self._y_standard.size
        return float(
            0.5 * self._y_standard @ alpha
            + 0.5 * log_determinant
            + 0.5 * n * np.log(2.0 * np.pi)
        )

    def _build(self, hyper: GPHyperparameters):
        kernel = matern52(self.x, self.x, hyper)
        noise = np.exp(hyper.log_noise)
        n = self.x.shape[0]
        last_error: Exception | None = None
        for jitter in self.jitter_ladder:
            try:
                factor = cholesky(kernel + (noise + jitter) * np.eye(n), lower=True)
                log_determinant = 2.0 * np.log(np.diag(factor)).sum()
                return factor, jitter, log_determinant
            except Exception as exc:  # not positive definite at this jitter
                last_error = exc
        raise RuntimeError(f"the covariance matrix stayed singular at every jitter: {last_error}")

    def _factorise(self, hyper: GPHyperparameters) -> None:
        factor, jitter, _ = self._build(hyper)
        self._factor = factor
        self.jitter_used = jitter
        self._alpha = cho_solve((factor, True), self._y_standard)

    # -- prediction --------------------------------------------------------

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation, in the original units."""
        if self.hyper is None or self._factor is None or self.x is None:
            raise RuntimeError("the process has not been fitted")
        x = np.atleast_2d(np.asarray(x, dtype=float))

        cross = matern52(x, self.x, self.hyper)
        mean = cross @ self._alpha

        solved = cho_solve((self._factor, True), cross.T)
        prior = np.exp(self.hyper.log_signal_variance)
        variance = prior - np.einsum("ij,ji->i", cross, solved)
        # Rounding can drive a variance a hair below zero at an observed point.
        variance = np.maximum(variance, 0.0)

        return mean * self.y_scale + self.y_mean, np.sqrt(variance) * self.y_scale
