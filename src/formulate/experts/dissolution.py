"""Will this solvent dissolve that polymer?

Specification section 1 asks for behaviour-driven design: a target states what
the material has to *do*, and the engine works out what that implies. Solvent
selection was the case where that had not been honoured. Asking for a solvent
for polystyrene meant writing the Hansen window for polystyrene into the target
by hand, which is not stating a behaviour - it is stating the answer and asking
the engine to look it up.

This expert closes that. The target names the solute, in
:attr:`~formulate.core.conditions.Conditions.solutes`, exactly as it already
names a substrate for adhesion. The engine supplies the solubility sphere.

The physics is Hansen's, and it is a sphere rather than a point. Every polymer
has a centre in the three-dimensional space of dispersion, polar and hydrogen
bonding parameters, and an interaction radius around it; a solvent dissolves
the polymer when it falls inside. The distance uses Hansen's factor of two on
the dispersion axis, which is not a fudge - it is what makes the volume a
sphere rather than an ellipsoid, and it comes from the original fits::

    Ra^2 = 4(dD_s - dD_p)^2 + (dP_s - dP_p)^2 + (dH_s - dH_p)^2
    RED  = Ra / R0

RED below one is a solvent, near one is a swelling agent or a marginal solvent,
above one is a non-solvent. Reporting the ratio rather than the raw distance is
what makes the number comparable between polymers with different radii.

The spheres below are published values, not fits made here, and the table is
deliberately short: a polymer whose sphere is not tabulated is refused rather
than approximated from its monomer, because a sphere inferred from repeat-unit
group contributions disagrees with the fitted one by enough to move solvents
across the boundary.

What this does not model. Molar mass, which shifts the boundary for the same
polymer - a low oligomer dissolves in solvents a high polymer only swells in.
Crystallinity, which is why polyethylene and polypropylene resist solvents that
their solubility parameters say should work, and why neither is in the table.
Specific interactions such as acid-base pairing, which Hansen's scheme averages
into the hydrogen bonding term. And kinetics: a solvent can be thermodynamically
good and still take a day to penetrate a thick part.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest
from .hansen import hansen_triple


@dataclass(frozen=True, slots=True)
class SolubilitySphere:
    """A polymer's Hansen centre and interaction radius, in MPa^0.5."""

    dispersion: float
    polar: float
    hydrogen_bonding: float
    radius: float
    source: str

    @property
    def centre(self) -> tuple[float, float, float]:
        return (self.dispersion, self.polar, self.hydrogen_bonding)


#: Published Hansen solubility spheres, keyed by the name a target would use.
#:
#: From Hansen, *Hansen Solubility Parameters: A User's Handbook*, 2nd edition
#: (CRC Press, 2007), which tabulates fitted spheres for common polymers.
#: Values are in MPa^0.5.
SOLUBILITY_SPHERES: dict[str, SolubilitySphere] = {
    "polystyrene": SolubilitySphere(21.3, 5.8, 4.3, 12.7, "Hansen handbook, 2nd ed."),
    "poly(methyl methacrylate)": SolubilitySphere(
        18.6, 10.5, 7.5, 8.6, "Hansen handbook, 2nd ed."
    ),
    "poly(vinyl chloride)": SolubilitySphere(18.2, 7.5, 8.3, 3.5, "Hansen handbook, 2nd ed."),
    "polycarbonate": SolubilitySphere(18.1, 5.9, 6.9, 7.5, "Hansen handbook, 2nd ed."),
    "poly(vinyl acetate)": SolubilitySphere(20.9, 11.3, 9.7, 13.7, "Hansen handbook, 2nd ed."),
    "cellulose acetate": SolubilitySphere(18.6, 12.7, 11.0, 7.6, "Hansen handbook, 2nd ed."),
    "polyamide 66": SolubilitySphere(17.4, 9.9, 14.6, 5.1, "Hansen handbook, 2nd ed."),
}

#: Aliases a target is likely to use for the same material.
SOLUTE_ALIASES = {
    "ps": "polystyrene",
    "pmma": "poly(methyl methacrylate)",
    "acrylic": "poly(methyl methacrylate)",
    "pvc": "poly(vinyl chloride)",
    "pc": "polycarbonate",
    "pvac": "poly(vinyl acetate)",
    "nylon 66": "polyamide 66",
    "nylon-66": "polyamide 66",
}


def resolve_solute(name: str) -> str | None:
    """The tabulated polymer this name refers to, or None."""
    key = name.strip().lower()
    key = SOLUTE_ALIASES.get(key, key)
    return key if key in SOLUBILITY_SPHERES else None


#: The panel carries Hansen parameters in Pa^0.5; the published spheres are in
#: MPa^0.5. The factor is a thousand, not a million, because these are square
#: roots of a pressure - which is exactly the kind of conversion that produces a
#: number wrong by three orders of magnitude while still looking like a
#: solubility parameter. Getting it wrong here made every real solvent for
#: polystyrene score as a non-solvent, and the only reason that was caught is
#: that the known answers were checked before the expert was used.
_PA_ROOT_PER_MPA_ROOT = 1000.0


def relative_energy_difference(
    solvent: tuple[float, float, float], sphere: SolubilitySphere
) -> float:
    """Hansen's RED: below one dissolves, above one does not.

    ``solvent`` is in Pa^0.5, as the rest of the panel carries it.
    """
    dd, dp, dh = (v / _PA_ROOT_PER_MPA_ROOT for v in solvent)
    distance = math.sqrt(
        4.0 * (dd - sphere.dispersion) ** 2
        + (dp - sphere.polar) ** 2
        + (dh - sphere.hydrogen_bonding) ** 2
    )
    return distance / sphere.radius


class DissolutionExpert(Expert):
    """Whether a candidate dissolves the solute the target names."""

    id = "dissolution"
    version = "1"
    method = "Hansen solubility sphere, published radii, relative energy difference"
    family = PropertyFamily.CHEMICAL
    supported_classes = frozenset({MaterialClass.MOLECULE})
    supported_properties = frozenset({"solubility_red"})

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        from formulate import chem

        if not chem.rdkit_available():
            return "RDKit is required to resolve a structure to its Hansen parameters"
        return ""

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        if candidate.molecule is None:
            return ApplicabilityDomain.outside("candidate carries no molecule")
        if hansen_triple(candidate.molecule.smiles) is None:
            return ApplicabilityDomain.outside(
                "this structure has no compiled Hansen parameters",
                basis="solvents present in the Hansen compilation",
            )
        return ApplicabilityDomain(
            basis="a solvent with compiled Hansen parameters, against a polymer with a "
            "published solubility sphere"
        )

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        solutes = request.conditions.solutes
        if not solutes:
            return Prediction.unsupported(
                prop,
                self.id,
                "no solute was named, and whether something dissolves is a question "
                "about a pair rather than about a solvent on its own",
            )
        if len(solutes) > 1:
            return Prediction.unsupported(
                prop,
                self.id,
                f"{len(solutes)} solutes were named and this reports one number; a "
                "solvent good for one polymer is routinely a non-solvent for another, "
                "so averaging them would hide the disagreement",
            )

        name = solutes[0]
        key = resolve_solute(name)
        if key is None:
            return Prediction.unsupported(
                prop,
                self.id,
                f"no published solubility sphere for {name!r}; the tabulated ones are "
                + ", ".join(sorted(SOLUBILITY_SPHERES)),
            )
        sphere = SOLUBILITY_SPHERES[key]

        smiles = request.candidate.molecule.smiles  # type: ignore[union-attr]
        triple = hansen_triple(smiles)
        if triple is None:
            return Prediction.unsupported(
                prop, self.id, "this structure has no compiled Hansen parameters"
            )

        red = relative_energy_difference(triple, sphere)
        verdict = (
            "inside the sphere, so a solvent"
            if red < 0.9
            else "on the boundary, so a swelling agent or a marginal solvent"
            if red < 1.1
            else "outside the sphere, so a non-solvent"
        )
        return self._make(
            prop,
            red,
            "",
            request,
            domain,
            # A tenth of a radius. The spheres are fitted to a finite solvent
            # set and shift by about that much between published fits, which is
            # enough to move a marginal solvent across the boundary and is why
            # the verdict says "marginal" rather than choosing.
            std=0.10,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                "spread between published fits of the same polymer's sphere, which is "
                "larger than any error in the solvent's own Hansen parameters"
            ),
            notes=(
                f"{key}: centre ({sphere.dispersion}, {sphere.polar}, "
                f"{sphere.hydrogen_bonding}) MPa^0.5, radius {sphere.radius}",
                f"RED {red:.2f}, {verdict}",
                sphere.source,
                "molar mass, crystallinity and dissolution kinetics are not modelled; a "
                "high polymer dissolves in less than its oligomer does",
            ),
        )
