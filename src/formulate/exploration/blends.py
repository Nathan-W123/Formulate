"""Proposing polymer blends.

Specification section 3 asks exploration to cover formulations as well as
single materials. It did, for solutions - ``MixtureSeedExplorer`` seeds
molecular formulations - but not for blends of polymers, which is a different
search: the axis that matters is not which two polymers but *what chain lengths
and in what ratio*, because that is the pair of knobs a blend exists to give
you.

A single polymer couples its melt viscosity to its strength through one
variable, chain length. A blend of the same polymer at two chain lengths
breaks that coupling: the long fraction carries load, the short one thins the
melt, and the ratio sets where between them the blend sits. That is bimodal
polyethylene, and it is why every hot-melt adhesive is a formulation rather
than a resin.

So the default proposal here is *bimodal* - one repeat unit, two molar masses -
rather than pairs of different polymers. Two different polymers are usually
immiscible, which the blend expert refuses and which is the physically correct
answer for most pairs; two grades of the same polymer are miscible by
construction. Cross-polymer pairs are proposed too, and mostly get refused,
which is information rather than waste.
"""

from __future__ import annotations

import json
from importlib import resources
from itertools import combinations
from typing import Sequence

from formulate.core.candidate import (
    Candidate,
    ComponentRole,
    FractionBasis,
    MaterialClass,
    MixtureComponent,
    MixtureSpec,
    MonomerUnit,
    PolymerSpec,
)
from formulate.core.provenance import ProvenanceKind, ProvenanceRecord
from formulate.core.quantity import Quantity
from formulate.targets.spec import TargetSpec

from .base import Explorer

#: Chain lengths proposed for the load-bearing fraction, kg/mol. These bracket
#: an ordinary commercial grade and run down to where a chain stops entangling.
LONG_MOLAR_MASSES = (20.0, 50.0, 120.0)

#: And for the thinning fraction. A wax is unentangled by design: it is there
#: to carry viscosity down, and it carries no load, which is exactly why the
#: blend needs the long fraction too.
SHORT_MOLAR_MASSES = (0.8, 2.0, 5.0)

#: Mass fractions of the short component. Below about a tenth it does not move
#: the melt; above about half there is not enough long chain left to carry
#: anything.
SHORT_FRACTIONS = (0.15, 0.30, 0.45)


class PolymerBlendExplorer(Explorer):
    """Proposes binary polymer blends, bimodal by default."""

    id = "blend:polymer"
    version = "1"

    def __init__(self, *, cross_polymer: bool = True) -> None:
        #: Propose pairs of *different* polymers as well as bimodal ones. Most
        #: get refused as immiscible, which is the right answer and is worth
        #: having on the record rather than never asked.
        self.cross_polymer = cross_polymer

    def _reference_units(self) -> list[tuple[str, str]]:
        text = (
            resources.files("formulate.data")
            .joinpath("reference_polymers.json")
            .read_text(encoding="utf-8")
        )
        return [(p["name"], p["repeat_unit"]) for p in json.loads(text)["polymers"]]

    @staticmethod
    def _component(repeat: str, molar_mass: float, fraction: float, role: ComponentRole):
        return MixtureComponent(
            role=role,
            fraction=fraction,
            polymer=PolymerSpec(
                monomers=(MonomerUnit(smiles=repeat),),
                number_average_molar_mass=Quantity(value=molar_mass, unit="kg/mol"),
            ),
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
        units = self._reference_units()
        seen = {c.structure_id for c in scored}
        out: list[Candidate] = []

        def emit(label: str, components, parameters) -> bool:
            if limits.max_components is not None and len(components) > limits.max_components:
                return False
            if limits.min_components is not None and len(components) < limits.min_components:
                return False
            candidate = Candidate(
                material_class=MaterialClass.MIXTURE,
                mixture=MixtureSpec(components=tuple(components), basis=FractionBasis.MASS),
                conditions=spec.conditions,
                generation_strategy=self.id,
                label=label,
                provenance=ProvenanceRecord(
                    kind=ProvenanceKind.RETRIEVAL,
                    producer=self.id,
                    producer_version=self.version,
                    parameters=parameters,
                ),
            )
            if candidate.structure_id in seen:
                return False
            seen.add(candidate.structure_id)
            out.append(candidate)
            return True

        # Bimodal first: one chemistry, two chain lengths, and the ratio.
        for name, repeat in units:
            for long_mass in LONG_MOLAR_MASSES:
                for short_mass in SHORT_MOLAR_MASSES:
                    for short_fraction in SHORT_FRACTIONS:
                        if len(out) >= count:
                            return out
                        emit(
                            f"{name}, bimodal {long_mass:g}/{short_mass:g} kg/mol "
                            f"at {short_fraction:.0%} short",
                            [
                                self._component(
                                    repeat, long_mass, 1.0 - short_fraction,
                                    ComponentRole.MATRIX,
                                ),
                                self._component(
                                    repeat, short_mass, short_fraction,
                                    ComponentRole.PLASTICIZER,
                                ),
                            ],
                            {
                                "mode": "bimodal",
                                "polymer": name,
                                "long_kg_mol": long_mass,
                                "short_kg_mol": short_mass,
                                "short_fraction": short_fraction,
                            },
                        )

        if not self.cross_polymer:
            return out

        for (name_a, unit_a), (name_b, unit_b) in combinations(units, 2):
            for short_fraction in SHORT_FRACTIONS:
                if len(out) >= count:
                    return out
                emit(
                    f"{name_a} + {name_b} at {short_fraction:.0%}",
                    [
                        self._component(
                            unit_a, LONG_MOLAR_MASSES[1], 1.0 - short_fraction,
                            ComponentRole.MATRIX,
                        ),
                        self._component(
                            unit_b, SHORT_MOLAR_MASSES[-1], short_fraction,
                            ComponentRole.PLASTICIZER,
                        ),
                    ],
                    {
                        "mode": "cross-polymer",
                        "matrix": name_a,
                        "modifier": name_b,
                        "modifier_fraction": short_fraction,
                    },
                )
        return out
