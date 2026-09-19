"""Proposing spinning dopes: a polymer dissolved in a solvent.

``PolymerSolutionExpert`` can score one and nothing could propose one. That is
the same hole the polymer class had before ``PolymerLibraryExplorer`` and the
mixture class had before the coordinator's defaults were widened, and it has
the same symptom: a specification asking for a dope drew an empty pool and the
run reported "0 of 0 satisfy every hard constraint", which reads as a search
that found nothing rather than one that never ran.

The axes are three and they are not interchangeable:

* **which polymer**, which decides what the finished filament is
* **how long its chains are**, which decides whether the dope strain-hardens -
  a dope lives or dies on this and it is the axis a melt cannot vary freely,
  because in a melt chain length also sets the viscosity
* **which solvent and how much**, which decides whether it is a fluid at all

The cross product is walked DIAGONALLY rather than as nested loops, so a
pool that gets cut off still spans every axis - see ``_spread``, and the
two-and-a-half-thousand-candidate run in water that produced it.

Concentration is swept low, because that is where a solution dope lives.
Above about a third the solution is a gel rather than a liquid and the
viscosity climbs out of any pressure a person can carry; below a few per cent
the chains do not overlap and the jet beads up. The engine decides which, but
proposing forty per cent solutions would spend the pool on candidates whose
answer is known.

Solvent choice is left wide rather than filtered by solubility here. Whether
the polymer actually dissolves is ``polymer_dissolution``'s question, it
answers it with a Hansen distance and a crystallinity gate, and an explorer
that pre-filtered on its own guess would hide the cases where the two
disagree - which are exactly the interesting ones.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Sequence

from formulate.core.candidate import (
    Candidate,
    ComponentRole,
    FractionBasis,
    MaterialClass,
    MixtureComponent,
    MixtureSpec,
    MoleculeSpec,
    MonomerUnit,
    PolymerSpec,
    Tacticity,
)
from formulate.core.provenance import ProvenanceKind, ProvenanceRecord
from formulate.core.quantity import Quantity
from formulate.targets.spec import TargetSpec

from .base import Explorer

#: Chain lengths proposed for the dissolved polymer, kg/mol.
#:
#: Reaching much higher than a melt would, and deliberately. A solution
#: decouples chain length from processing viscosity - dilution carries the
#: viscosity down and the chain carries the elasticity up - which is the whole
#: reason gel spinning exists and the reason these run to a value no melt
#: process could push.
MOLAR_MASSES: tuple[float, ...] = (200.0, 600.0, 1500.0, 4000.0)

#: Mass fractions of polymer. A dope lives between overlap and gelation.
MASS_FRACTIONS: tuple[float, ...] = (0.03, 0.08, 0.15, 0.25)


def _solvents() -> list[tuple[str, str]]:
    text = (
        resources.files("formulate.data")
        .joinpath("reference_compounds.json")
        .read_text(encoding="utf-8")
    )
    return [(c["name"], c["smiles"]) for c in json.loads(text)["compounds"]]


def _repeat_units() -> list[tuple[str, str]]:
    """Every repeat unit the engine can propose, enumerated ones included."""
    text = (
        resources.files("formulate.data")
        .joinpath("reference_polymers.json")
        .read_text(encoding="utf-8")
    )
    units = [(p["name"], p["repeat_unit"]) for p in json.loads(text)["polymers"]]
    seen = {u for _, u in units}
    try:
        from .monomers import chain_hetero, condensation, vinyl
    except Exception:  # pragma: no cover - the package ships with them
        return units
    for module in (vinyl, condensation, chain_hetero):
        for name, unit in module.units():
            if unit not in seen:
                seen.add(unit)
                units.append((name, unit))
    return units


def _spread(*axes):
    """The cross product, ordered so a truncated draw spans every axis.

    Nested loops were the first version and they are wrong for a pool that
    gets cut off. With fifty solvents in the second loop and five hundred
    repeat units in the fourth, a draw of two thousand five hundred never
    reached a second solvent: every candidate was in WATER, which the
    Orrick-Erbar viscosity correlation refuses outright because it is built on
    a carbon count and water has no carbon. The run reported two and a half
    thousand candidates whose viscosity could not be predicted, and read like
    a finding about polymers when it was a finding about loop order.

    This walks the product diagonally first - taking index k from every axis at
    once, so after a handful of draws every solvent, every concentration and
    every chain length has appeared - and only then fills in the rest. Nothing
    is dropped; the ordering changes and the set does not.
    """
    lengths = [len(a) for a in axes]
    if not lengths or any(n == 0 for n in lengths):
        return
    seen: set[tuple[int, ...]] = set()
    for step in range(max(lengths)):
        index = tuple(step % n for n in lengths)
        if index in seen:
            continue
        seen.add(index)
        yield tuple(axis[i] for axis, i in zip(axes, index))

    from itertools import product

    for index in product(*(range(n) for n in lengths)):
        if index in seen:
            continue
        seen.add(index)
        yield tuple(axis[i] for axis, i in zip(axes, index))


class PolymerSolutionExplorer(Explorer):
    """Proposes polymer-in-solvent dopes across chemistry, chain length and concentration."""

    id = "solution:polymer"
    version = "1"

    def __init__(
        self,
        *,
        molar_masses: Sequence[float] = MOLAR_MASSES,
        mass_fractions: Sequence[float] = MASS_FRACTIONS,
        solvents: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        self.molar_masses = tuple(molar_masses)
        self.mass_fractions = tuple(mass_fractions)
        self._solvents = list(solvents) if solvents is not None else None

    def solvents(self) -> list[tuple[str, str]]:
        if self._solvents is None:
            self._solvents = _solvents()
        return self._solvents

    @staticmethod
    def _tacticities(repeat: str) -> tuple[Tacticity, ...]:
        from formulate.experts.melt import backbone_stereocentres

        return (
            (Tacticity.ISOTACTIC, Tacticity.ATACTIC)
            if backbone_stereocentres(repeat)
            else (Tacticity.UNSPECIFIED,)
        )

    def propose(
        self,
        spec: TargetSpec,
        count: int,
        *,
        scored: Sequence[Candidate] = (),
        seed: int = 0,
    ) -> list[Candidate]:
        if MaterialClass.MIXTURE not in spec.material_classes:
            return []

        limits = spec.structural
        if limits.max_components is not None and limits.max_components < 2:
            return []

        seen = {c.structure_id for c in scored}
        out: list[Candidate] = []
        units = _repeat_units()
        solvents = self.solvents()

        for fraction, (solvent_name, solvent_smiles), mass, (polymer_name, repeat) in _spread(
            self.mass_fractions, solvents, self.molar_masses, units
        ):
            for tacticity in self._tacticities(repeat):
                if len(out) >= count:
                    return out
                components = (
                    MixtureComponent(
                        role=ComponentRole.SOLUTE,
                        fraction=fraction,
                        polymer=PolymerSpec(
                            monomers=(MonomerUnit(smiles=repeat),),
                            tacticity=tacticity,
                            number_average_molar_mass=Quantity(value=mass, unit="kg/mol"),
                        ),
                    ),
                    MixtureComponent(
                        role=ComponentRole.SOLVENT,
                        fraction=1.0 - fraction,
                        molecule=MoleculeSpec(smiles=solvent_smiles),
                    ),
                )
                label = (
                    f"{fraction:.0%} {polymer_name} {mass:g} kg/mol in {solvent_name}"
                )
                if tacticity is not Tacticity.UNSPECIFIED:
                    label = f"{label}, {tacticity.value}"
                candidate = Candidate(
                    material_class=MaterialClass.MIXTURE,
                    mixture=MixtureSpec(components=components, basis=FractionBasis.MASS),
                    conditions=spec.conditions,
                    generation_strategy=self.id,
                    label=label,
                    provenance=ProvenanceRecord(
                        kind=ProvenanceKind.RETRIEVAL,
                        producer=self.id,
                        producer_version=self.version,
                        parameters={
                            "polymer": polymer_name,
                            "solvent": solvent_name,
                            "mass_fraction": fraction,
                            "number_average_molar_mass_kg_mol": mass,
                            "tacticity": tacticity.value,
                        },
                    ),
                )
                if candidate.structure_id in seen:
                    continue
                seen.add(candidate.structure_id)
                out.append(candidate)
        return out
