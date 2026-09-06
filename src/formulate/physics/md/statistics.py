"""Trajectory statistics.

Specification section 6 requires that "finite-size/time uncertainty must be
reported", and section 11 makes simulation sampling error a distinct kind of
uncertainty from model error.

The reason this module exists rather than a call to ``numpy.std`` is that a
molecular-dynamics trajectory is correlated in time.  Successive frames are
not independent samples, so the standard error of the mean over N frames
understates the true uncertainty by roughly the square root of the number of
frames per correlation time.  Measured on a 1001-frame GFN-FF trajectory of
butanol, the block-averaged error was 7.7 times the naive one.  Reporting the
naive figure would not be a rough estimate; it would be wrong by an order of
magnitude, in the direction that makes results look better than they are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BlockAverage:
    """A mean with an error bar that accounts for serial correlation."""

    mean: float
    #: Standard error of the mean, from block averaging.
    standard_error: float
    #: Standard error ignoring correlation, kept only to expose the difference.
    naive_standard_error: float
    n_samples: int
    n_blocks: int
    block_size: int
    #: standard_error / naive_standard_error. Roughly sqrt of the number of
    #: frames per correlation time; a value near 1 means the samples were
    #: effectively independent.
    inflation: float
    converged: bool
    notes: tuple[str, ...] = ()

    @property
    def effective_samples(self) -> float:
        """Independent samples the trajectory is actually worth."""
        return self.n_samples / max(1.0, self.inflation**2)

    def describe(self, unit: str = "") -> str:
        suffix = f" {unit}" if unit else ""
        return (
            f"{self.mean:.6g} +/- {self.standard_error:.4g}{suffix} "
            f"({self.n_blocks} blocks of {self.block_size}, "
            f"{self.effective_samples:.0f} effective samples, "
            f"error inflated {self.inflation:.1f}x over the naive estimate)"
        )


def block_average(values, min_blocks: int = 5, max_blocks: int = 50) -> BlockAverage:
    """Estimate a mean and its standard error by block averaging.

    The block size is chosen by scanning: as blocks grow past the correlation
    time the estimated error rises and then plateaus, and the plateau is the
    honest value.  Taking the largest available block instead would be noisy
    (few blocks means a poorly determined variance), and taking the smallest
    would reproduce the naive underestimate, so the scan takes the maximum of
    the estimates that still have enough blocks to be meaningful.
    """
    series = np.asarray(list(values), dtype=float)
    series = series[np.isfinite(series)]
    n = series.size

    if n < 2:
        return BlockAverage(
            mean=float(series[0]) if n else float("nan"),
            standard_error=float("nan"),
            naive_standard_error=float("nan"),
            n_samples=n,
            n_blocks=0,
            block_size=0,
            inflation=float("nan"),
            converged=False,
            notes=("too few samples to estimate an uncertainty",),
        )

    mean = float(series.mean())
    naive = float(series.std(ddof=1) / np.sqrt(n))

    best_error = naive
    best_blocks = n
    best_size = 1
    for n_blocks in range(min_blocks, min(max_blocks, n // 2) + 1):
        size = n // n_blocks
        if size < 2:
            continue
        trimmed = series[: size * n_blocks].reshape(n_blocks, size)
        block_means = trimmed.mean(axis=1)
        error = float(block_means.std(ddof=1) / np.sqrt(n_blocks))
        if error > best_error:
            best_error, best_blocks, best_size = error, n_blocks, size

    inflation = best_error / naive if naive > 0 else float("nan")
    notes: list[str] = []
    converged = True

    if best_size <= 2:
        notes.append(
            "no block size longer than the sampling interval increased the error estimate, "
            "which usually means the trajectory is too short to resolve its own "
            "correlation time"
        )
        converged = False
    effective = n / max(1.0, inflation**2) if np.isfinite(inflation) else float("nan")
    if np.isfinite(effective) and effective < 10:
        notes.append(
            f"the trajectory is worth only about {effective:.0f} independent samples; "
            "the mean is not converged and the error bar is itself poorly determined"
        )
        converged = False

    return BlockAverage(
        mean=mean,
        standard_error=best_error,
        naive_standard_error=naive,
        n_samples=n,
        n_blocks=best_blocks,
        block_size=best_size,
        inflation=inflation,
        converged=converged,
        notes=tuple(notes),
    )


def autocorrelation_time(values, max_lag: int | None = None) -> float:
    """Integrated autocorrelation time in units of the sampling interval.

    Summation is truncated at the first non-positive autocorrelation, the
    standard remedy for the fact that the tail of an estimated correlation
    function is dominated by noise and summing all of it diverges.
    """
    series = np.asarray(list(values), dtype=float)
    series = series[np.isfinite(series)]
    n = series.size
    if n < 4:
        return float("nan")

    centred = series - series.mean()
    variance = float((centred**2).mean())
    if variance <= 0:
        return 0.0

    limit = min(max_lag or n // 2, n - 1)
    total = 0.0
    for lag in range(1, limit + 1):
        correlation = float((centred[:-lag] * centred[lag:]).mean() / variance)
        if correlation <= 0.0:
            break
        total += correlation
    return 1.0 + 2.0 * total


def detect_equilibration(values, fraction_step: float = 0.05) -> int:
    """Index after which a series looks stationary.

    Uses the marginal-standard-error rule: discard the leading fraction that
    maximises the number of effectively independent samples in what remains.
    Discarding too little leaves a drifting transient in the average;
    discarding too much throws away signal, and this trades them off rather
    than fixing an arbitrary cut.
    """
    series = np.asarray(list(values), dtype=float)
    series = series[np.isfinite(series)]
    n = series.size
    if n < 10:
        return 0

    best_index = 0
    best_effective = -np.inf
    step = max(1, int(n * fraction_step))
    for start in range(0, n // 2, step):
        remainder = series[start:]
        if remainder.size < 8:
            break
        tau = autocorrelation_time(remainder)
        if not np.isfinite(tau) or tau <= 0:
            continue
        effective = remainder.size / tau
        if effective > best_effective:
            best_effective, best_index = effective, start
    return best_index
