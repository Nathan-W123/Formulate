"""Elastic moduli, entanglement, and the strength that cannot be predicted.

Three questions get asked together and only two of them have answers.

**Tensile and shear strength do not.** A specimen fails at its largest flaw -
Griffith gives the fracture stress as the square root of (2 E gamma / pi a),
where ``a`` is the size of that flaw - and the flaw is a property of how the
sample was made, not of what it is made from. Two bars of identical polymer
differ by a factor of two or three on processing alone. That is not a modelling
gap this module could close with more effort; it is what the quantity is. There
is no ``tensile_strength`` in the property registry and there should not be.

**Elastic moduli do**, and the dominant term is not subtle. Measured across
nine amorphous polymers, a glass sits at 2.9 GPa with a standard deviation of
0.5, and four rubbers sit at 1.3 MPa. That is a factor of two thousand, and
which side a polymer falls on is decided by whether the temperature is above or
below its glass transition - a quantity the polymer expert already predicts. So
the model is a switch, not a correlation, and the structure-dependence within
the glassy branch is smaller than the scatter between compilations.

Above the transition the modulus is the rubber-elastic one, ``3 rho R T / M_e``,
which needs the entanglement molar mass.

**Entanglement molar mass does**, from the packing length. The chain's volume
per unit of its own mean-square end-to-end distance sets how far one chain
reaches before another must thread it, and Fetters and co-workers established
that ``M_e`` scales as the cube of that length. Checked here against six
polymers whose ``M_e`` spans sixteen-fold, one constant reproduces them to
within a factor of 1.74.

That factor matters less than it looks, because ``M_e`` is not wanted as a
number. It is wanted because it decides whether a polymer is brittle or tough:
a chain shorter than a couple of entanglement lengths cannot form a
load-bearing network and the material snaps regardless of how stiff it is. A
sixteen-fold range resolved to a factor of two answers that question.

**Theoretical strength** is produced only as a bound, under its own name, and
never as ``tensile_strength``. It is the stress a flawless solid would carry,
around a tenth of the modulus, and it exceeds what a real specimen does by one
to three orders of magnitude - which is the useful part: it says how much of
the possible a given material is actually delivering, and why a drawn fibre
gets closer to it than a moulded bar ever will.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

#: Gas constant, J/(mol K).
_R = 8.31446261815324
#: Avogadro's number.
_AVOGADRO = 6.02214076e23

#: Mean glassy Young's modulus and its spread across measured amorphous
#: polymers, Pa. Nine polymers from polystyrene to polyacrylonitrile span
#: 2.0-3.5 GPa, so the structure-dependence inside the glass is weaker than
#: the disagreement between sources.
_GLASSY_MODULUS = 2.86e9
_GLASSY_SPREAD = 0.50e9

#: Poisson's ratio. A glass is near 0.35; a rubber is nearly incompressible.
_POISSON_GLASSY = 0.35
_POISSON_RUBBERY = 0.4999

#: Theoretical strength as a fraction of the modulus. The Frenkel estimate is
#: between a tenth and a thirtieth; the looser end is used so the bound is not
#: mistaken for a prediction.
_THEORETICAL_STRENGTH_FRACTION = 0.10

#: Chains a polymer needs before it can form a load-bearing entangled network.
#: Below roughly this the material is brittle whatever its modulus.
_ENTANGLEMENTS_FOR_TOUGHNESS = 2.0


@dataclass(frozen=True, slots=True)
class ChainDimension:
    """A chain's mean-square end-to-end distance per unit molar mass.

    In angstrom squared per (g/mol). This is the one input that cannot be got
    from structure here: it comes from rotational-isomeric-state calculations
    or from neutron scattering, and it is tabulated per polymer.
    """

    r2_per_mass: float
    #: Measured entanglement molar mass, g/mol, where one is known.
    measured_entanglement: float | None
    split: str
    source: str


#: Keyed by repeat-unit SMILES, matching the polymer reference set.
CHAIN_DIMENSIONS: dict[str, ChainDimension] = {
    "[*]CC[*]": ChainDimension(1.25, 1150, "fit", "polyethylene, melt at 413 K"),
    "[*]CC(c1ccccc1)[*]": ChainDimension(0.437, 18100, "fit", "polystyrene, melt at 413 K"),
    "[*]CC(C)=CC[*]": ChainDimension(0.679, 6200, "fit", "cis-1,4-polyisoprene at 298 K"),
    "[*]CC(C)(C)[*]": ChainDimension(0.570, 6700, "fit", "polyisobutylene at 298 K"),
    "[*]CC=CC[*]": ChainDimension(0.876, 1850, "validation", "1,4-polybutadiene at 298 K"),
    "[*]CC(C)(C(=O)OC)[*]": ChainDimension(0.425, 9200, "validation", "PMMA, melt at 413 K"),
    "[*][Si](C)(C)O[*]": ChainDimension(0.422, 12000, "validation", "PDMS at 298 K"),
    "[*]CC(C)[*]": ChainDimension(0.678, None, "fit", "atactic polypropylene"),
    "[*]CCO[*]": ChainDimension(0.805, None, "fit", "poly(ethylene oxide)"),
}


def packing_length(r2_per_mass: float, density_g_cm3: float) -> float:
    """Packing length in angstrom: chain volume per unit of its own extent.

    ``p = M / (rho N_A <R^2>)``. Reproduces the published values - 1.69 A for
    polyethylene, 3.92 for polystyrene - which is what makes the cube law below
    checkable rather than merely fitted.
    """
    return 1.0e24 / (density_g_cm3 * _AVOGADRO * r2_per_mass)


@dataclass(frozen=True, slots=True)
class EntanglementModel:
    """The one fitted constant in ``M_e = k rho N_A p^3``, and its error."""

    constant: float
    n_fit: int
    #: Worst ratio of predicted to measured over the fit split.
    fit_spread: float
    #: Same over polymers withheld from the fit.
    validation_spread: float
    validation_n: int

    def entanglement(self, r2_per_mass: float, density_g_cm3: float) -> float:
        p = packing_length(r2_per_mass, density_g_cm3) * 1e-8  # angstrom -> cm
        return self.constant * density_g_cm3 * _AVOGADRO * p**3


@functools.lru_cache(maxsize=1)
def entanglement_model() -> EntanglementModel:
    """Fit the packing constant, and measure how far it transfers.

    The densities used are the ones the measurements were made at, which for
    the melt-state polymers is not room temperature. Using a room-temperature
    density for a melt-state entanglement mass would be an inconsistency of
    about ten per cent, inside the factor of two this model carries but not
    worth introducing for nothing.
    """
    densities = {
        "[*]CC[*]": 0.784,
        "[*]CC(c1ccccc1)[*]": 0.969,
        "[*]CC(C)=CC[*]": 0.900,
        "[*]CC(C)(C)[*]": 0.918,
        "[*]CC=CC[*]": 0.900,
        "[*]CC(C)(C(=O)OC)[*]": 1.13,
        "[*][Si](C)(C)O[*]": 0.970,
    }

    def constants(split: str) -> list[float]:
        out = []
        for unit, entry in CHAIN_DIMENSIONS.items():
            if entry.split != split or entry.measured_entanglement is None:
                continue
            rho = densities[unit]
            p = packing_length(entry.r2_per_mass, rho) * 1e-8
            out.append(entry.measured_entanglement / (rho * _AVOGADRO * p**3))
        return out

    fitted = constants("fit")
    constant = sum(fitted) / len(fitted)
    model = EntanglementModel(constant, len(fitted), max(fitted) / min(fitted), 1.0, 0)

    ratios = []
    for unit, entry in CHAIN_DIMENSIONS.items():
        if entry.split != "validation" or entry.measured_entanglement is None:
            continue
        predicted = model.entanglement(entry.r2_per_mass, densities[unit])
        ratios.append(max(predicted, entry.measured_entanglement)
                      / min(predicted, entry.measured_entanglement))
    return EntanglementModel(
        constant=constant,
        n_fit=len(fitted),
        fit_spread=max(fitted) / min(fitted),
        validation_spread=max(ratios) if ratios else float("nan"),
        validation_n=len(ratios),
    )


def youngs_modulus(temperature_k: float, glass_transition_k: float,
                   density_g_cm3: float, entanglement_g_mol: float | None) -> tuple[float, str]:
    """Young's modulus in Pa, and which branch produced it.

    Below the transition the answer is the glassy plateau, which barely depends
    on structure. Above it the answer is the entangled network, which depends
    on nothing else.
    """
    if temperature_k < glass_transition_k:
        return _GLASSY_MODULUS, "glassy"
    if entanglement_g_mol is None or entanglement_g_mol <= 0:
        raise ValueError("a rubbery modulus needs an entanglement molar mass")
    density_kg_m3 = density_g_cm3 * 1000.0
    return (
        3.0 * density_kg_m3 * _R * temperature_k / (entanglement_g_mol / 1000.0),
        "rubbery",
    )


def shear_from_young(modulus_pa: float, branch: str) -> float:
    """``G = E / (2(1+nu))``. A rubber is nearly incompressible, a glass is not."""
    poisson = _POISSON_GLASSY if branch == "glassy" else _POISSON_RUBBERY
    return modulus_pa / (2.0 * (1.0 + poisson))


def theoretical_strength(modulus_pa: float) -> float:
    """Flaw-free upper bound, Pa. Never a tensile strength."""
    return modulus_pa * _THEORETICAL_STRENGTH_FRACTION


def is_tough(number_average_molar_mass: float | None, entanglement_g_mol: float | None) -> bool | None:
    """Whether the chain is long enough to entangle into a load-bearing network.

    The question ``how strong is it`` usually means ``will it snap`` , and that
    is decided here rather than by the modulus. Returns None when the chain
    length was not stated, because a polymer with no molar mass has no answer.
    """
    if number_average_molar_mass is None or entanglement_g_mol is None:
        return None
    return number_average_molar_mass >= _ENTANGLEMENTS_FOR_TOUGHNESS * entanglement_g_mol


# --------------------------------------------------------------------------
# The expert
# --------------------------------------------------------------------------


from formulate.core.candidate import Candidate, MaterialClass, MonomerRole  # noqa: E402
from formulate.core.prediction import Prediction  # noqa: E402
from formulate.core.properties import PropertyFamily  # noqa: E402
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind  # noqa: E402

from .base import Expert  # noqa: E402


class PolymerMechanicalExpert(Expert):
    """Moduli, entanglement and a strength bound for a linear polymer.

    Depends on the glass transition and the amorphous density, which the
    polymer experts supply. That dependency is the model: which side of the
    transition the temperature falls on changes the modulus by a factor of two
    thousand, so a mechanical answer that did not consult Tg would be wrong far
    more often than it was right.
    """

    id = "polymer_mechanical"
    version = "1"
    method = (
        "glassy plateau or rubber elasticity selected on Tg, with entanglement molar "
        "mass from the chain packing length"
    )
    family = PropertyFamily.MECHANICAL
    supported_classes = frozenset({MaterialClass.POLYMER})
    supported_properties = frozenset(
        {"youngs_modulus", "shear_modulus", "entanglement_molar_mass", "theoretical_strength"}
    )
    dependencies = frozenset({"glass_transition_temperature", "amorphous_density"})

    def _chain(self, candidate: Candidate) -> ChainDimension | None:
        spec = candidate.polymer
        if spec is None:
            return None
        backbone = [m for m in spec.monomers if m.role is not MonomerRole.END_GROUP]
        if len(backbone) != 1:
            return None
        from formulate import chem

        smiles = backbone[0].smiles
        if smiles in CHAIN_DIMENSIONS:
            return CHAIN_DIMENSIONS[smiles]
        if chem.rdkit_available():
            canonical = chem.canonical_smiles(smiles)
            for key, entry in CHAIN_DIMENSIONS.items():
                if chem.canonical_smiles(key) == canonical:
                    return entry
        return None

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        model = entanglement_model()
        basis = (
            f"glassy plateau from nine measured amorphous polymers, and a packing-length "
            f"entanglement constant fitted on {model.n_fit} chains"
        )
        spec = candidate.polymer
        if spec is None:
            return ApplicabilityDomain.outside("candidate carries no polymer", basis=basis)
        if self._chain(candidate) is None:
            return ApplicabilityDomain.outside(
                "no chain dimension is tabulated for this repeat unit. The mean-square "
                "end-to-end distance per unit mass comes from scattering or from a "
                "rotational-isomeric-state calculation, and cannot be read off the "
                "structure here",
                basis=basis,
            )
        return ApplicabilityDomain(basis=basis)

    def _predict_one(self, prop, request, domain):
        candidate = request.candidate
        spec = candidate.polymer
        if spec is None:
            return Prediction.unsupported(prop, self.id, "candidate carries no polymer")

        chain = self._chain(candidate)
        if chain is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "no tabulated chain dimension for this repeat unit; it is a measured "
                "quantity, not one derivable from the structure here",
            )

        density = request.dependency_value("amorphous_density", "g/cm^3")
        if density is None:
            return Prediction.unsupported(
                prop, self.id, "the amorphous density this rests on was not available"
            )

        model = entanglement_model()
        entanglement = model.entanglement(chain.r2_per_mass, density)

        if prop == "entanglement_molar_mass":
            return self._make(
                prop,
                entanglement,
                "g/mol",
                request,
                domain,
                # A ratio, not an absolute: the spread is multiplicative.
                std=entanglement * (model.validation_spread - 1.0),
                kind=UncertaintyKind.EPISTEMIC,
                basis=(
                    f"the packing-length cube law reproduces withheld polymers to a factor "
                    f"of {model.validation_spread:.2f}, against an entanglement mass that "
                    "spans sixteen-fold across the set"
                ),
                notes=(
                    f"chain dimension from {chain.source}",
                    "this is what decides brittle from tough: a chain shorter than about "
                    "two entanglement lengths cannot form a load-bearing network, however "
                    "stiff it is",
                ),
                packing_length_angstrom=round(packing_length(chain.r2_per_mass, density), 3),
            )

        temperature = request.conditions.temperature_k
        if temperature is None:
            return Prediction.unsupported(
                prop, self.id, "a modulus is temperature dependent and none was given"
            )
        glass_transition = request.dependency_value("glass_transition_temperature", "K")
        if glass_transition is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "the glass transition decides which branch applies, and it was not "
                "available; guessing the branch is a factor of two thousand",
            )

        try:
            modulus, branch = youngs_modulus(temperature, glass_transition, density, entanglement)
        except ValueError as exc:
            return Prediction.unsupported(prop, self.id, str(exc))

        if branch == "glassy":
            relative = _GLASSY_SPREAD / _GLASSY_MODULUS
            reason = (
                f"{temperature:.0f} K is below the transition at {glass_transition:.0f} K, "
                "so this is the glassy plateau: 2.9 GPa with a 0.5 GPa spread across nine "
                "measured amorphous polymers, and only weakly dependent on structure"
            )
        else:
            # Rubber elasticity without a front factor lands within a factor of
            # two on the four rubbers checked, which is the honest spread.
            relative = 1.0
            reason = (
                f"{temperature:.0f} K is above the transition at {glass_transition:.0f} K, "
                "so this is the entangled network: 3 rho R T / M_e, which reproduced four "
                "measured rubbers to within a factor of two"
            )

        notes = [reason]
        if branch == "rubbery":
            # Polyethylene is the standing example: 195 K transition, so this
            # branch fires and returns about 8 MPa, while a real polyethylene
            # bar is nearer 800 because most of it is crystalline. The answer
            # here is for the amorphous phase, and crystallinity is an outcome
            # of processing that nothing in this system predicts.
            notes.append(
                "this is the modulus of the amorphous phase. A polymer that crystallises "
                "is far stiffer than this - polyethylene measures about 0.8 GPa against "
                "the 8 MPa its amorphous rubber would give - and the crystalline fraction "
                "is set by processing, which this system does not predict"
            )
        tough = is_tough(
            spec.number_average_molar_mass.to("g/mol").value
            if spec.number_average_molar_mass is not None
            else None,
            entanglement,
        )
        if tough is False:
            notes.append(
                "the stated chain is shorter than two entanglement lengths, so this "
                "material is brittle whatever its modulus: it has no entangled network to "
                "carry load"
            )
        elif tough is None:
            notes.append(
                "no number-average molar mass was stated, so whether the chains entangle "
                "into a load-bearing network is unknown; that, not the modulus, is what "
                "decides brittle from tough"
            )

        if prop == "theoretical_strength":
            value = theoretical_strength(modulus)
            notes.append(
                "an upper bound on a flawless solid, not a tensile strength. A real "
                "specimen fails at its largest defect, one to three orders of magnitude "
                "below this; a drawn fibre approaches it and a moulded bar never will"
            )
            return self._make(
                prop, value, "Pa", request, domain,
                std=value * max(relative, 0.5),
                kind=UncertaintyKind.EPISTEMIC,
                basis="a tenth of the modulus, the loose end of the Frenkel estimate",
                notes=tuple(notes),
                branch=branch,
            )

        if prop == "shear_modulus":
            value = shear_from_young(modulus, branch)
            notes.append(
                f"from Young's modulus at a Poisson ratio of "
                f"{_POISSON_GLASSY if branch == 'glassy' else _POISSON_RUBBERY}"
            )
        else:
            value = modulus

        return self._make(
            prop, value, "Pa", request, domain,
            std=value * relative,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                "spread of the glassy plateau across measured polymers"
                if branch == "glassy"
                else "rubber elasticity without a front factor is good to a factor of two"
            ),
            notes=tuple(notes),
            branch=branch,
        )
