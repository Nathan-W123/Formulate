"""Cheap validity filters.

Specification section 3: "Use validity filters before expert inference:
valence, charge, duplicate detection, forbidden chemistry, composition bounds,
basic stability/synthesis heuristics."

The point of running these first is economy: rejecting a nonsensical structure
costs microseconds, while sending it to the expert panel costs milliseconds
and pollutes the pool.  Every rejection records a reason, because a filter
that silently eats candidates is indistinguishable from a broken generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from formulate.core.candidate import Candidate, MaterialClass
from formulate.targets.spec import StructuralConstraints


@dataclass(frozen=True, slots=True)
class FilterResult:
    passed: bool
    reasons: tuple[str, ...] = ()

    @classmethod
    def rejected(cls, *reasons: str) -> "FilterResult":
        return cls(passed=False, reasons=reasons)

    @classmethod
    def accepted(cls) -> "FilterResult":
        return cls(passed=True)


@dataclass(frozen=True, slots=True)
class FilterReport:
    kept: tuple[Candidate, ...] = ()
    rejected: tuple[tuple[Candidate, FilterResult], ...] = ()

    @property
    def rejection_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, result in self.rejected:
            for reason in result.reasons:
                counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def describe(self) -> str:
        if not self.rejected:
            return f"Validity filters: {len(self.kept)} candidates, none rejected."
        lines = [
            f"Validity filters: kept {len(self.kept)}, rejected {len(self.rejected)}.",
        ]
        lines.extend(f"  {count} x {reason}" for reason, count in self.rejection_counts.items())
        return "\n".join(lines)


class CandidateFilter:
    """Applies structural constraints before any expert is called."""

    def __init__(self, constraints: StructuralConstraints | None = None) -> None:
        self.constraints = constraints or StructuralConstraints()

    def check(self, candidate: Candidate) -> FilterResult:
        from formulate import chem

        reasons: list[str] = []
        constraints = self.constraints

        smiles_list = candidate.all_smiles()
        if not smiles_list:
            return FilterResult.rejected("candidate contains no structures")

        if chem.rdkit_available():
            for smiles in smiles_list:
                if chem.mol_from_smiles(smiles) is None:
                    reasons.append("contains a structure that fails valence or parsing checks")
                    break

        if not reasons and chem.rdkit_available():
            reasons.extend(self._chemistry_checks(candidate, smiles_list))

        reasons.extend(self._composition_checks(candidate))
        return FilterResult.accepted() if not reasons else FilterResult(False, tuple(reasons))

    def _chemistry_checks(self, candidate: Candidate, smiles_list: Sequence[str]) -> list[str]:
        from formulate import chem

        constraints = self.constraints
        reasons: list[str] = []

        present: set[str] = set()
        for smiles in smiles_list:
            present |= chem.elements(smiles)

        if constraints.allowed_elements:
            extra = present - set(constraints.allowed_elements)
            if extra:
                reasons.append(f"contains elements outside the allowed set: {', '.join(sorted(extra))}")
        if constraints.forbidden_elements:
            hit = present & set(constraints.forbidden_elements)
            if hit:
                reasons.append(f"contains forbidden elements: {', '.join(sorted(hit))}")

        for smarts in constraints.forbidden_smarts:
            if any(chem.has_substructure(s, smarts) for s in smiles_list):
                reasons.append(f"matches forbidden substructure {smarts!r}")

        if constraints.required_smarts:
            if not any(
                chem.has_substructure(s, smarts)
                for s in smiles_list
                for smarts in constraints.required_smarts
            ):
                reasons.append(
                    "contains none of the required substructures: "
                    + ", ".join(repr(s) for s in constraints.required_smarts)
                )

        if not constraints.allow_charged:
            for smiles in smiles_list:
                if chem.descriptors(smiles).get("formal_charge", 0.0):
                    reasons.append("is charged, and charged species were not permitted")
                    break

        if candidate.material_class is MaterialClass.MOLECULE:
            heavy = chem.descriptors(smiles_list[0]).get("heavy_atom_count", 0.0)
            if constraints.max_heavy_atoms is not None and heavy > constraints.max_heavy_atoms:
                reasons.append(
                    f"has {heavy:.0f} heavy atoms, above the limit of {constraints.max_heavy_atoms}"
                )
            if constraints.min_heavy_atoms is not None and heavy < constraints.min_heavy_atoms:
                reasons.append(
                    f"has {heavy:.0f} heavy atoms, below the minimum of {constraints.min_heavy_atoms}"
                )
        return reasons

    def _composition_checks(self, candidate: Candidate) -> list[str]:
        constraints = self.constraints
        reasons: list[str] = []
        if candidate.mixture is None:
            return reasons
        count = len(candidate.mixture.components)
        if constraints.max_components is not None and count > constraints.max_components:
            reasons.append(
                f"has {count} components, above the limit of {constraints.max_components}"
            )
        if constraints.min_components is not None and count < constraints.min_components:
            reasons.append(
                f"has {count} components, below the minimum of {constraints.min_components}"
            )
        return reasons

    def apply(self, candidates: Sequence[Candidate]) -> FilterReport:
        """Partition a pool into kept and rejected, with reasons."""
        kept: list[Candidate] = []
        rejected: list[tuple[Candidate, FilterResult]] = []
        for candidate in candidates:
            result = self.check(candidate)
            (kept if result.passed else rejected).append(
                candidate if result.passed else (candidate, result)  # type: ignore[arg-type]
            )
        return FilterReport(kept=tuple(kept), rejected=tuple(rejected))
