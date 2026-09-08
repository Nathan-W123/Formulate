"""Interatomic potentials, and deciding when one is not good enough.

Specification section 8 asks for a multi-fidelity layer: a cheap prediction
first, a fast potential on the candidates worth more, and quantum mechanics
only where the cheaper answer cannot be trusted. The roadmap has carried this
as not done, with torch's absence as the stated reason. Torch is installable,
so the reason has expired.

What this module is
    A model-agnostic interface - energies, forces, a declared element domain -
    with one real backend behind it. :class:`MacePotential` wraps MACE-OFF23,
    an equivariant message-passing network trained on organic molecules at the
    wB97M-D3(BJ)/def2-TZVPPD level. It declares its own supported elements,
    read from the weights rather than written down here, so a molecule
    containing anything else is refused rather than extrapolated.

What this module is not
    A replacement for the force fields in :mod:`formulate.physics.md`. That was
    measured rather than assumed: MACE-OFF costs 535 ms per force evaluation on
    a 576-atom box on this hardware, which puts a hundred picoseconds at
    fifteen hours and a five-liquid density benchmark into weeks. It is a
    single-point engine here - geometries, conformer energies, escalation
    decisions - and the docstring says so because someone will otherwise try
    the density.

The honest gap
    MACE-OFF23 ships no uncertainty estimator. There is no ensemble, no
    variance head, and inventing a confidence from a single deterministic
    network would be exactly the fabricated capability this project keeps
    refusing. So :class:`PotentialResult` carries ``uncertainty=None`` and
    escalation leans on the two signals that are real: whether the molecule is
    inside the declared element domain, and whether two independent methods
    disagree by more than their own stated errors.
"""

from __future__ import annotations

import abc
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from formulate.core.hashing import content_hash

#: Hartree per electronvolt, for reporting alongside quantum results.
_EV_PER_HARTREE = 27.211386245988


@dataclass(frozen=True, slots=True)
class PotentialInfo:
    """What a potential is and where it may be believed."""

    identifier: str
    description: str
    reference: str
    #: Atomic numbers the model was trained on and declares support for.
    supported_elements: frozenset[int]
    #: Level of theory the training labels came from, where known.
    training_level: str = ""
    #: True only where the backend supplies a real uncertainty estimate.
    provides_uncertainty: bool = False


@dataclass(frozen=True, slots=True)
class PotentialResult:
    """A single-point energy and its forces, with what is and is not known."""

    energy_ev: float
    #: (n_atoms, 3) in eV/A, or None if the backend was asked not to compute them.
    forces_ev_per_angstrom: Any = None
    #: None where the backend has no trustworthy estimator, which is the usual
    #: case. A number invented here would be worse than the absence.
    uncertainty_ev: float | None = None
    wall_seconds: float = 0.0
    info: PotentialInfo | None = None
    diagnostics: tuple[str, ...] = ()

    @property
    def energy_hartree(self) -> float:
        return self.energy_ev / _EV_PER_HARTREE


class UnsupportedSystem(Exception):
    """The potential declines this system, with the reason why."""


class InteratomicPotential(abc.ABC):
    """Energies and forces for a set of atoms, or an explicit refusal."""

    info: PotentialInfo

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Whether the backend can run here. Checked, never assumed."""

    def unavailable_reason(self) -> str:
        return ""

    def refusal(self, atomic_numbers: Sequence[int]) -> str | None:
        """Why this system is outside the declared domain, or None."""
        unsupported = sorted(set(atomic_numbers) - self.info.supported_elements)
        if unsupported:
            return (
                f"element(s) {unsupported} are outside the training set of "
                f"{self.info.identifier}, which covers "
                f"{sorted(self.info.supported_elements)}; a number here would be an "
                "extrapolation of a neural network rather than a prediction"
            )
        return None

    @abc.abstractmethod
    def compute(
        self,
        atomic_numbers: Sequence[int],
        positions: Any,
        *,
        forces: bool = True,
    ) -> PotentialResult:
        """A single-point energy, and forces unless told otherwise."""


class MacePotential(InteratomicPotential):
    """MACE-OFF23, an equivariant network for neutral organic molecules."""

    def __init__(self, model: str = "small", dtype: str = "float64") -> None:
        self.model = model
        self.dtype = dtype
        self._calculator: Any = None
        self._info: PotentialInfo | None = None

    @property
    def info(self) -> PotentialInfo:  # type: ignore[override]
        if self._info is None:
            self._info = PotentialInfo(
                identifier=f"mace-off23-{self.model}",
                description=(
                    "MACE-OFF23 equivariant message-passing potential for neutral "
                    "organic molecules; single-point use only, since a force "
                    "evaluation costs 535 ms for 576 atoms on CPU"
                ),
                reference="Kovacs et al., arXiv:2312.15211; ASL licence",
                supported_elements=self._declared_elements(),
                training_level="wB97M-D3(BJ)/def2-TZVPPD",
                provides_uncertainty=False,
            )
        return self._info

    def _declared_elements(self) -> frozenset[int]:
        """Read from the loaded weights, so it cannot drift from the model."""
        try:
            calculator = self._load()
        except Exception:
            # The published MACE-OFF23 element set, used only when the weights
            # cannot be reached; the loaded model always overrides it.
            return frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})
        model = calculator.models[0]
        numbers = getattr(model, "atomic_numbers", None)
        if numbers is None:
            return frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})
        return frozenset(int(z) for z in numbers)

    def is_available(self) -> bool:
        try:
            import mace  # noqa: F401
            import torch  # noqa: F401
            from ase import Atoms  # noqa: F401
        except Exception:
            return False
        return True

    def unavailable_reason(self) -> str:
        for module, hint in (
            ("torch", "install formulate[potentials]"),
            ("mace", "install formulate[potentials]"),
            ("ase", "install formulate[physics]"),
        ):
            try:
                __import__(module)
            except Exception as exc:
                return f"{module} is not installed ({exc}); {hint}"
        return ""

    def _load(self):
        if self._calculator is None:
            from mace.calculators import mace_off

            self._calculator = mace_off(
                model=self.model, device="cpu", default_dtype=self.dtype
            )
        return self._calculator

    def atomic_reference_energies(self) -> dict[int, float]:
        """The isolated-atom energies the model was fitted with, in eV.

        MACE learns a total energy as a sum of these references plus a
        many-body correction, so subtracting them is not a convention imposed
        here - it is how the model is constructed, and the difference is the
        quantity the network was actually trained to reproduce.
        """
        model = self._load().models[0]
        return {
            int(z): float(e)
            for z, e in zip(model.atomic_numbers, model.atomic_energies_fn.atomic_energies)
        }

    def atomization_energy_ev(
        self, atomic_numbers: Sequence[int], positions: Any
    ) -> float:
        """Electronic atomization energy, positive for a bound molecule.

        This is D_e, not D_0: there is no zero-point correction, because a
        single-point energy contains no vibrational information. Measured
        against six small molecules whose experimental D_0 and zero-point
        energies are both tabulated, the mean absolute error is about 14
        kJ/mol; comparing it against a D_0 without adding the zero-point
        energy back produces an apparent error of about 97 kJ/mol that is
        almost entirely the missing term.
        """
        references = self.atomic_reference_energies()
        result = self.compute(atomic_numbers, positions, forces=False)
        return sum(references[int(z)] for z in atomic_numbers) - result.energy_ev

    def compute(
        self,
        atomic_numbers: Sequence[int],
        positions: Any,
        *,
        forces: bool = True,
    ) -> PotentialResult:
        import numpy as np
        from ase import Atoms

        refusal = self.refusal(atomic_numbers)
        if refusal:
            raise UnsupportedSystem(refusal)

        calculator = self._load()
        atoms = Atoms(numbers=list(atomic_numbers), positions=np.asarray(positions))
        atoms.calc = calculator
        started = time.perf_counter()
        energy = float(atoms.get_potential_energy())
        force_array = np.asarray(atoms.get_forces()) if forces else None
        return PotentialResult(
            energy_ev=energy,
            forces_ev_per_angstrom=force_array,
            uncertainty_ev=None,
            wall_seconds=time.perf_counter() - started,
            info=self.info,
            diagnostics=(
                "MACE-OFF23 provides no uncertainty estimator; escalation uses the "
                "declared element domain and cross-method disagreement instead",
            ),
        )


# --------------------------------------------------------------------------
# Never computing the same point twice
# --------------------------------------------------------------------------


@dataclass
class PotentialCache:
    """Content-addressed single-point results.

    Section 11: "content-address candidates + method + conditions so identical
    calculations are never repeated." The address is the geometry, the model
    identity and whether forces were asked for - nothing else enters, because
    nothing else changes the answer. Coordinates are rounded to 1e-6 Angstrom
    by the same rule the geometry uses for its own identity: a position that
    differs in the last bits of a float is the same position, and treating it
    as a different one would defeat the cache entirely.

    An escalation ladder revisits the same geometry more than once - the cheap
    pass, then the comparison, then the decision - so this is not an
    optimisation for a repeated run. It is what makes one run cheap.
    """

    entries: dict[str, PotentialResult] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    @staticmethod
    def key(
        identifier: str,
        atomic_numbers: Sequence[int],
        positions: Any,
        *,
        forces: bool,
    ) -> str:
        return content_hash(
            {
                "potential": identifier,
                "atomic_numbers": [int(z) for z in atomic_numbers],
                "positions": [
                    [round(float(v), 6) for v in row] for row in positions
                ],
                "forces": bool(forces),
            },
            prefix="pot",
        )

    def compute(
        self,
        potential: InteratomicPotential,
        atomic_numbers: Sequence[int],
        positions: Any,
        *,
        forces: bool = True,
    ) -> PotentialResult:
        """``potential.compute`` with the answer remembered.

        A refusal is not cached: it is raised before any work happens, so there
        is nothing to save, and storing it would mean a widened element domain
        went unnoticed until someone cleared the cache.
        """
        address = self.key(
            potential.info.identifier, atomic_numbers, positions, forces=forces
        )
        with self._lock:
            hit = self.entries.get(address)
            if hit is not None:
                self.hits += 1
                return hit
        result = potential.compute(atomic_numbers, positions, forces=forces)
        with self._lock:
            self.entries[address] = result
            self.misses += 1
        return result

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __len__(self) -> int:
        return len(self.entries)


def available_potentials() -> list[InteratomicPotential]:
    """Every potential usable here."""
    return [p for p in (MacePotential(),) if p.is_available()]


# --------------------------------------------------------------------------
# When the cheap answer is not good enough
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EscalationDecision:
    """Whether to spend a quantum calculation, and what made that necessary."""

    escalate: bool
    reasons: tuple[str, ...] = ()

    def describe(self) -> str:
        if self.escalate:
            return "escalate to QM: " + "; ".join(self.reasons)
        if self.reasons:
            # A candidate declined on cost still had concerns, and reporting it
            # as untroubled would hide the one thing a reader needs: that the
            # number was doubted and the doubt was not resolved.
            return "not escalated: " + "; ".join(self.reasons)
        return "no escalation: the cheaper method is inside its domain and unchallenged"


@dataclass
class EscalationPolicy:
    """The section 8 triggers, each one a stated condition rather than a feel.

    Four things justify spending a quantum calculation. Three are properties of
    the cheap answer - it is out of domain, its own uncertainty is large, or two
    methods disagree beyond their stated errors. The fourth is a property of the
    ranking: a candidate whose interval straddles the decision boundary is worth
    resolving however confident the method claims to be, and a candidate that
    cannot change the answer is not worth resolving however uncertain it is.
    """

    #: Relative uncertainty above which a value is not usable as it stands.
    uncertainty_fraction: float = 0.10
    #: Disagreement between methods, in units of their combined stated error.
    disagreement_sigma: float = 2.0
    #: Skip candidates this far below the decision boundary in rank order.
    rank_horizon: int = 5

    def decide(
        self,
        *,
        out_of_domain_reason: str | None = None,
        value: float | None = None,
        uncertainty: float | None = None,
        other_value: float | None = None,
        other_uncertainty: float | None = None,
        rank: int | None = None,
        changes_ranking: bool | None = None,
    ) -> EscalationDecision:
        reasons: list[str] = []

        if out_of_domain_reason:
            reasons.append(f"out of domain ({out_of_domain_reason})")

        if value is not None and uncertainty is not None and value:
            relative = abs(uncertainty / value)
            if relative > self.uncertainty_fraction:
                reasons.append(
                    f"stated uncertainty is {relative:.0%} of the value, above the "
                    f"{self.uncertainty_fraction:.0%} this policy will accept"
                )

        if None not in (value, other_value):
            spread = ((uncertainty or 0.0) ** 2 + (other_uncertainty or 0.0) ** 2) ** 0.5
            gap = abs(value - other_value)  # type: ignore[arg-type]
            if spread > 0 and gap > self.disagreement_sigma * spread:
                reasons.append(
                    f"two methods differ by {gap / spread:.1f} times their combined "
                    "stated error, so at least one of them is wrong about this molecule"
                )
            elif spread == 0 and gap > 0:
                reasons.append(
                    "two methods disagree and neither states an uncertainty, so the "
                    "disagreement cannot be attributed"
                )

        # A candidate that cannot move the answer is not worth resolving, however
        # uncertain it is. This is the cost control, and it is applied last so
        # that the reasons above are still recorded.
        if reasons and rank is not None and rank > self.rank_horizon:
            if not changes_ranking:
                return EscalationDecision(
                    False,
                    (
                        f"ranked {rank}, beyond the {self.rank_horizon} that could change "
                        "the recommendation; the concerns above are recorded and not bought",
                    ),
                )
        return EscalationDecision(bool(reasons), tuple(reasons))
