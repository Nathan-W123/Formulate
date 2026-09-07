"""OPLS-AA: a force field that was fitted to liquids.

:mod:`mmff` gives an exact MMFF94 engine, and :mod:`condensed` records what
that engine is worth in a condensed phase: MMFF94 was parameterised against
gas-phase geometries and conformational energies, so it under-binds a liquid by
ten to fifteen per cent in cohesive energy and by considerably more in density.
That file's closing paragraph said the fix was OPLS or GAFF and that both were
conda-only. The second half of that claim was wrong.

OPLS-AA is reachable, and the reason is structural rather than lucky. The
obstacle for every other fitted force field here is charges: GAFF and the
OpenFF line assign partial charges per molecule with AM1-BCC, which needs a
semiempirical quantum code and a toolkit that will not install. OPLS-AA does
not. Its charges come out of the atom-type table, so typing the molecule *is*
parameterising it, and typing is a SMARTS problem that RDKit can already do.

What supplies the types is :mod:`foyer`, whose ``oplsaa.xml`` carries 813 atom
types with SMARTS definitions, charges, sigmas and epsilons. It is a 2021
release, it predates OpenMM 8, and it needs two shims to run against it. Both
are applied at import time here rather than by editing the installed package,
and both are narrow enough to state in one line:

* ``simtk.openmm`` was renamed to ``openmm``. A meta-path finder binds the old
  names to the *same module objects* as the new ones. Aliasing only the
  top-level package is not enough - OpenMM signals nonbonded methods with
  singleton sentinels compared by identity, and a second copy of
  ``forcefield.py`` imported under the old name makes every one of them fail to
  match.
* ``ForceField._SystemData`` gained a required ``topology`` argument, whose
  constructor now does bookkeeping foyer still does by hand afterwards. The
  shim restores the no-argument form by passing an empty topology, which leaves
  those lists empty for foyer to fill exactly as it did under OpenMM 7.

The shim is not assumed to be equivalent to patching foyer's source; it was
checked. Both routes produce identical atom types, identical term counts and
identical potential energies to eight decimal places for ethanol, hexane,
toluene, acetone and diethyl ether.

Coverage is the real limit, and it is not small. Over the fifty reference
compounds, thirty-seven type cleanly and thirteen do not, in three ways:

no type at all
    Esters and lactones. ``oplsaa.xml`` has no SMARTS for an ester carbonyl
    carbon, so ethyl acetate, butyl acetate, ethyl lactate and
    gamma-butyrolactone are refused outright. This failure is loud.

more than one type
    Propylene carbonate matches two definitions with no override between them.
    Also loud.

a type that is silently wrong
    Aniline, anisole, styrene, naphthalene and the chloroalkanes come back
    fully typed and *not electrically neutral* - chloroform by half an
    elementary charge. The aromatic and chloride SMARTS in this file are
    broader than the charge sets behind them, so a substituted ring or a second
    chlorine picks up parameters from a molecule it does not belong to. Nothing
    raises. The only thing that catches it is summing the charges, so this
    module does that on every molecule and refuses on any net charge, which is
    an exact invariant for a neutral species rather than a tolerance.

One more gap is worth naming because it is invisible from the outside and turns
out not to matter. ``oplsaa.xml`` contains no improper torsions at all, and
published OPLS-AA uses them to hold sp2 centres flat. Left to the proper
torsions alone, over 200 ps at 298 K the largest out-of-plane excursion is 7.5
degrees for a benzene ring carbon, 10.2 for nitrobenzene and 13.9 for the
carbonyl carbon of acetone. Those are thermal librations; nothing inverts. The
ring and carbonyl proper torsions carry enough of the barrier on their own for
the molecules this module will be asked about.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Any

#: A neutral molecule's charges must sum to zero. The table stores them to
#: three decimals, so this admits rounding and nothing else.
NET_CHARGE_TOLERANCE = 1.0e-3


class UnsupportedMolecule(Exception):
    """OPLS-AA cannot parameterise this molecule, with the reason why."""


# --------------------------------------------------------------------------
# Making foyer importable
# --------------------------------------------------------------------------


def _install_simtk_alias() -> None:
    """Bind ``simtk.*`` to the identical ``openmm.*`` module objects."""
    if "simtk" in sys.modules:
        return

    from importlib.abc import Loader, MetaPathFinder
    from importlib.machinery import ModuleSpec

    import openmm
    import openmm.unit

    def real_name(name: str) -> str | None:
        for old, new in (("simtk.unit", "openmm.unit"), ("simtk.openmm", "openmm")):
            if name == old:
                return new
            if name.startswith(old + "."):
                return new + name[len(old):]
        return None

    class _AliasLoader(Loader):
        def __init__(self, module):
            self._module = module

        def create_module(self, spec):
            return self._module

        def exec_module(self, module):
            pass

    class _AliasFinder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            target_name = real_name(fullname)
            if target_name is None:
                return None
            module = importlib.import_module(target_name)
            spec = ModuleSpec(fullname, _AliasLoader(module),
                              is_package=hasattr(module, "__path__"))
            if hasattr(module, "__path__"):
                spec.submodule_search_locations = module.__path__
            return spec

    import types

    package = types.ModuleType("simtk")
    package.__path__ = []  # a package, so ``simtk.openmm`` is looked up as a submodule
    package.openmm = openmm
    package.unit = openmm.unit
    sys.modules["simtk"] = package
    sys.meta_path.insert(0, _AliasFinder())


def _restore_empty_system_data() -> None:
    """Let ``_SystemData()`` be constructed with no topology again."""
    from openmm import app

    system_data = app.ForceField._SystemData
    if getattr(system_data.__init__, "_formulate_shim", False):
        return
    original = system_data.__init__

    def __init__(self, topology=None):
        original(self, topology if topology is not None else app.Topology())

    __init__._formulate_shim = True
    system_data.__init__ = __init__


_FORCE_FIELD: Any = None


def available() -> bool:
    """Whether OPLS-AA can be loaded in this environment."""
    try:
        load_forcefield()
    except Exception:
        return False
    return True


def load_forcefield():
    """The foyer OPLS-AA force field, loaded once."""
    global _FORCE_FIELD
    if _FORCE_FIELD is None:
        _install_simtk_alias()
        _restore_empty_system_data()
        import foyer

        _FORCE_FIELD = foyer.Forcefield(name="oplsaa")
    return _FORCE_FIELD


# --------------------------------------------------------------------------
# Typing a molecule
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OplsParameters:
    """One molecule, typed and parameterised under OPLS-AA."""

    structure: Any
    """A ParmEd structure carrying types, charges and every bonded term."""

    atom_types: tuple[str, ...]
    charges: tuple[float, ...]
    masses: tuple[float, ...]

    @property
    def n_atoms(self) -> int:
        return len(self.atom_types)

    @property
    def net_charge(self) -> float:
        return sum(self.charges)


def _to_parmed(mol):
    """A ParmEd structure with this molecule's elements, bonds and conformer."""
    import parmed as pmd

    if mol.GetNumConformers() == 0:
        raise UnsupportedMolecule("typing needs a conformer and the molecule has none")

    conformer = mol.GetConformer()
    structure = pmd.Structure()
    for index, atom in enumerate(mol.GetAtoms()):
        position = conformer.GetAtomPosition(index)
        parmed_atom = pmd.Atom(
            name=f"{atom.GetSymbol()}{index}", atomic_number=atom.GetAtomicNum()
        )
        parmed_atom.xx, parmed_atom.xy, parmed_atom.xz = (
            position.x,
            position.y,
            position.z,
        )
        structure.add_atom(parmed_atom, "RES", 1)
    for bond in mol.GetBonds():
        structure.bonds.append(
            pmd.Bond(
                structure.atoms[bond.GetBeginAtomIdx()],
                structure.atoms[bond.GetEndAtomIdx()],
            )
        )
    return structure


def extract_parameters(mol) -> OplsParameters:
    """Type ``mol`` under OPLS-AA, or refuse it with a reason.

    Raises :class:`UnsupportedMolecule` rather than returning ``None``, because
    every one of the three ways this fails is worth reporting to whoever asked:
    an uncovered functional group, an ambiguous match, and a set of types whose
    charges do not sum to zero are different problems.
    """
    if mol.GetNumAtoms() != sum(1 for _ in mol.GetAtoms()):  # pragma: no cover
        raise UnsupportedMolecule("malformed molecule")

    field = load_forcefield()
    try:
        structure = field.apply(_to_parmed(mol), assert_dihedral_params=False)
    except UnsupportedMolecule:
        raise
    except Exception as error:
        raise UnsupportedMolecule(
            f"OPLS-AA has no consistent typing for this molecule: {error}"
        ) from error

    charges = tuple(float(atom.charge) for atom in structure.atoms)
    net = sum(charges)
    if abs(net) > NET_CHARGE_TOLERANCE:
        raise UnsupportedMolecule(
            f"the assigned OPLS-AA types carry a net charge of {net:+.3f} e on a "
            "neutral molecule, so at least one of them belongs to a different "
            "compound and the parameters cannot be trusted"
        )

    return OplsParameters(
        structure=structure,
        atom_types=tuple(atom.type for atom in structure.atoms),
        charges=charges,
        masses=tuple(float(atom.mass) for atom in structure.atoms),
    )


def refusal(mol) -> str | None:
    """The reason OPLS-AA will not parameterise ``mol``, or None if it will."""
    try:
        extract_parameters(mol)
    except UnsupportedMolecule as error:
        return str(error)
    return None


# --------------------------------------------------------------------------
# Building a system
# --------------------------------------------------------------------------


def build_system(
    params: OplsParameters,
    copies: int = 1,
    *,
    box_nm: float | None = None,
    cutoff_nm: float = 1.0,
    constrain_hydrogens: bool = True,
):
    """An OpenMM system for ``copies`` identical molecules under OPLS-AA.

    Atoms come out molecule by molecule in the input order, which is the layout
    :func:`~formulate.physics.md.condensed.pack_box` produces.

    With no box the system is aperiodic and uncut, for an isolated-molecule
    reference. With a box it is periodic: PME for the charges, a cutoff on van
    der Waals with the analytic long-range correction that a density needs.

    OPLS-AA mixes van der Waals parameters geometrically, not by the
    Lorentz-Berthelot rule an OpenMM ``NonbondedForce`` implements. ParmEd
    honours that by moving the whole van der Waals part into a
    ``CustomNonbondedForce``, which is why the returned system carries one.
    """
    from openmm import app
    from openmm import unit as u

    if box_nm is not None and cutoff_nm * 2.0 >= box_nm:
        raise ValueError(
            f"a {cutoff_nm:.2f} nm cutoff does not fit in a {box_nm:.2f} nm box: the "
            "minimum image convention needs the box to exceed twice the cutoff"
        )

    structure = params.structure * copies if copies != 1 else params.structure
    structure.combining_rule = params.structure.combining_rule
    if box_nm is not None:
        edge = box_nm * 10.0
        structure.box = [edge, edge, edge, 90.0, 90.0, 90.0]

    constraints = app.HBonds if constrain_hydrogens else None
    if box_nm is None:
        system = structure.createSystem(
            nonbondedMethod=app.NoCutoff, constraints=constraints, rigidWater=False
        )
    else:
        system = structure.createSystem(
            nonbondedMethod=app.PME,
            nonbondedCutoff=cutoff_nm * u.nanometer,
            constraints=constraints,
            rigidWater=False,
        )
    return system, structure
