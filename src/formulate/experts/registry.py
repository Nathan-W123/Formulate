"""Registry of available experts.

The coordinator asks the registry which experts are applicable to a target
rather than importing them directly, so adding an expert never requires
touching the coordinator.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Iterator

from formulate.core.candidate import MaterialClass
from formulate.core.properties import PropertyFamily

from .base import Expert


class ExpertRegistry:
    """A collection of experts, queryable by property and material class."""

    def __init__(self, experts: Iterable[Expert] = ()) -> None:
        self._experts: dict[str, Expert] = {}
        for expert in experts:
            self.register(expert)

    def register(self, expert: Expert, *, replace: bool = False) -> Expert:
        if expert.id in self._experts and not replace:
            raise ValueError(
                f"An expert with id {expert.id!r} is already registered; "
                "pass replace=True to override."
            )
        self._experts[expert.id] = expert
        return expert

    def unregister(self, expert_id: str) -> None:
        self._experts.pop(expert_id, None)

    def get(self, expert_id: str) -> Expert:
        try:
            return self._experts[expert_id]
        except KeyError:
            raise KeyError(f"No expert registered with id {expert_id!r}.") from None

    def __iter__(self) -> Iterator[Expert]:
        return iter(self._experts.values())

    def __len__(self) -> int:
        return len(self._experts)

    def __contains__(self, expert_id: object) -> bool:
        return expert_id in self._experts

    # -- queries -----------------------------------------------------------

    def experts_for(
        self, properties: Iterable[str], material_class: MaterialClass
    ) -> list[Expert]:
        """Every expert covering at least one requested property for this class."""
        wanted = frozenset(properties)
        return [e for e in self._experts.values() if e.applicable_properties(wanted, material_class)]

    def coverage(
        self, properties: Iterable[str], material_class: MaterialClass
    ) -> dict[str, list[str]]:
        """Map each requested property to the ids of experts that can supply it."""
        out: dict[str, list[str]] = defaultdict(list)
        for prop in properties:
            for expert in self._experts.values():
                if expert.covers(prop, material_class):
                    out[prop].append(expert.id)
        return {p: sorted(ids) for p, ids in out.items()}

    def uncovered(
        self, properties: Iterable[str], material_class: MaterialClass
    ) -> list[str]:
        """Requested properties no registered expert can supply.

        Reported to the user rather than silently dropped: a target the system
        cannot evaluate is a finding, not an empty column.
        """
        cov = self.coverage(properties, material_class)
        return sorted(p for p in properties if not cov.get(p))

    def by_family(self, family: PropertyFamily) -> list[Expert]:
        return [e for e in self._experts.values() if e.family is family]

    def resolution_order(self, experts: Iterable[Expert]) -> list[Expert]:
        """Order experts so dependencies are produced before they are needed.

        A dependency cycle is a programming error and raises rather than
        deadlocking or silently dropping an expert.
        """
        pending = list(experts)
        produced: set[str] = set()
        ordered: list[Expert] = []
        while pending:
            ready = [e for e in pending if e.dependencies <= produced]
            if not ready:
                # Dependencies that nothing in this set can supply are external;
                # run the rest anyway so they can report the gap themselves.
                external = [
                    e
                    for e in pending
                    if not any(d in u.supported_properties for d in e.dependencies for u in pending)
                ]
                if external:
                    ready = external
                else:
                    names = ", ".join(sorted(e.id for e in pending))
                    raise ValueError(f"Cyclic expert dependencies among: {names}")
            ready.sort(key=lambda e: e.id)
            for expert in ready:
                ordered.append(expert)
                produced |= expert.supported_properties
                pending.remove(expert)
        return ordered

    def describe(self) -> str:
        lines = [f"{len(self)} experts registered:"]
        for expert in sorted(self._experts.values(), key=lambda e: (e.family.value, e.id)):
            status = "" if expert.is_available() else f"  UNAVAILABLE: {expert.unavailable_reason()}"
            lines.append(
                f"  {expert.id} v{expert.version} [{expert.family.value}] "
                f"-> {', '.join(sorted(expert.supported_properties))}{status}"
            )
        return "\n".join(lines)
