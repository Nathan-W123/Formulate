"""Energies defined as a difference between separate calculations.

An absolute electronic energy is not a property: it is monotone in the number
of electrons, so ranking candidates on it orders them by size. What carries
meaning is a difference between calculations at one level of theory, and the
two the specification names are the atomization energy - a molecule against its
free atoms - and the interaction energy - an assembly against its separated
parts.

Both were listed as quantum-validatable and neither had an observable behind
it, so the validation stage declined them. They are not hard; they are simply
several calculations rather than one, and each has a trap that makes a wrong
answer look right.

Atomization: the free atoms are open shell. Carbon's ground state is a triplet,
nitrogen a quartet, oxygen a triplet. Computing them as closed-shell singlets
is a calculation that converges and returns an atomization energy wrong by
hundreds of kJ/mol, so the multiplicities are stated here rather than defaulted.

Interaction: the answer depends on the geometry of the assembly, which is a
search problem with many minima, and on basis-set superposition error, which
makes any complex look more bound than it is. Neither is solved here. This
module relaxes a complex rather than searching it, and the counterpoise
correction that would remove the superposition error needs ghost-atom basis
functions this backend does not expose. Both are reported on every value as
limitations rather than left for the reader to remember.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Ground-state spin multiplicity (2S+1) of the free atom, by element.
#:
#: Defaulting these to a singlet is the trap this table exists to avoid: an
#: unrestricted calculation on a carbon atom forced to be closed shell still
#: converges, and the atomization energy that follows is wrong by roughly
#: 400 kJ/mol per carbon.
ATOMIC_MULTIPLICITY: dict[str, int] = {
    "H": 2, "He": 1,
    "Li": 2, "Be": 1, "B": 2, "C": 3, "N": 4, "O": 3, "F": 2, "Ne": 1,
    "Na": 2, "Mg": 1, "Al": 2, "Si": 3, "P": 4, "S": 3, "Cl": 2, "Ar": 1,
    "K": 2, "Ca": 1, "Br": 2, "I": 2,
}

#: Hartree -> J/mol.
HARTREE_J_MOL = 2625499.6394798254


@dataclass(frozen=True, slots=True)
class DifferenceResult:
    """An energy difference, with everything needed to judge it."""

    #: J/mol. Positive means the assembled or bonded form is the lower one.
    value: float
    #: The pieces the difference was taken over, for auditing.
    components: tuple[tuple[str, float], ...]
    method_signature: str
    diagnostics: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


def unsupported_elements(symbols: tuple[str, ...]) -> tuple[str, ...]:
    """Elements with no tabulated atomic ground state, so no reference energy."""
    return tuple(sorted({s for s in symbols if s not in ATOMIC_MULTIPLICITY}))


def atomization_energy(backend, geometry, request_template) -> DifferenceResult | None:
    """Energy to separate a molecule into free atoms, in J/mol.

    ``sum(E_atom) - E_molecule``, positive for a bound molecule. Each atomic
    reference is computed at the same level of theory as the molecule and in
    its own ground state; anything else compares two different calculations and
    calls the difference chemistry.
    """
    import numpy as np

    from formulate.physics.geometry import MolecularGeometry

    missing = unsupported_elements(geometry.symbols)
    if missing:
        return None

    molecule = backend.run(request_template.model_copy(update={"geometry": geometry}))
    if not molecule.usable or molecule.total_energy is None:
        return None

    components: list[tuple[str, float]] = [
        ("molecule", molecule.total_energy.to("J/mol").value)
    ]
    total_atoms = 0.0
    for symbol in sorted(set(geometry.symbols)):
        atom = MolecularGeometry(
            symbols=(symbol,),
            positions=np.zeros((1, 3)),
            spin_multiplicity=ATOMIC_MULTIPLICITY[symbol],
            source=f"free {symbol} atom, ground-state multiplicity "
            f"{ATOMIC_MULTIPLICITY[symbol]}",
        )
        result = backend.run(
            request_template.model_copy(update={"geometry": atom, "optimize_geometry": False})
        )
        if not result.usable or result.total_energy is None:
            return None
        energy = result.total_energy.to("J/mol").value
        count = geometry.symbols.count(symbol)
        components.append((f"{count} x {symbol}", energy))
        total_atoms += count * energy

    value = total_atoms - components[0][1]
    diagnostics = []
    if value <= 0:
        diagnostics.append(
            "the free atoms came out below the molecule, which is not a bound species: "
            "the reference multiplicities or the SCF convergence are wrong"
        )
    return DifferenceResult(
        value=value,
        components=tuple(components),
        method_signature=molecule.method_signature,
        diagnostics=tuple(diagnostics),
        limitations=(
            "an atomization energy at a single level of theory carries that method's "
            "systematic error, which for a small basis is tens of kJ/mol per bond",
            "zero-point vibrational energy is not included, so this is the bottom of "
            "the well rather than a thermochemical atomization enthalpy",
        ),
    )


def interaction_energy(backend, complex_geometry, fragments, request_template):
    """Binding energy of an assembly relative to its separated parts, in J/mol.

    ``sum(E_fragment) - E_complex``, positive for a bound assembly.

    Uncorrected for basis-set superposition, because removing it needs ghost
    atoms and this backend does not expose them. That error always inflates the
    binding, and it grows as the basis shrinks - which is exactly the regime a
    screening calculation runs in - so the value carries a known sign of bias
    and says so.
    """
    if len(fragments) < 2:
        return None

    total = backend.run(request_template.model_copy(update={"geometry": complex_geometry}))
    if not total.usable or total.total_energy is None:
        return None

    components: list[tuple[str, float]] = [
        ("complex", total.total_energy.to("J/mol").value)
    ]
    separated = 0.0
    for index, fragment in enumerate(fragments):
        result = backend.run(
            request_template.model_copy(
                update={"geometry": fragment, "optimize_geometry": False}
            )
        )
        if not result.usable or result.total_energy is None:
            return None
        energy = result.total_energy.to("J/mol").value
        components.append((f"fragment {index + 1}", energy))
        separated += energy

    return DifferenceResult(
        value=separated - components[0][1],
        components=tuple(components),
        method_signature=total.method_signature,
        limitations=(
            "the value is for the geometry supplied, not for the global minimum of the "
            "assembly; a complex has many minima and this module relaxes rather than "
            "searches",
            "fragments are computed in their own basis, so basis-set superposition error "
            "is present and inflates the binding; a counterpoise correction needs ghost "
            "atoms, which this backend does not expose",
        ),
    )
