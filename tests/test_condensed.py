"""MMFF94 as an OpenMM system, and the bulk observables it makes possible.

The translation is tested against an oracle rather than against intuition:
RDKit will evaluate any single MMFF94 term on its own, so every force built
here has an exact reference. That is the whole reason this route was worth
taking - a hand-written force field with no oracle would be a plausible-looking
source of wrong numbers, which is precisely what specification section 13 warns
against.

The simulations themselves are slow by nature, so the runs here are short and
the assertions are about machinery and refusals rather than about agreeing with
experiment. The accuracy claim lives in docs/ROADMAP.md next to the run that
measured it.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_rdkit

from formulate.physics.md import condensed
from formulate.physics.md.mmff import (
    KCAL,
    _TERMS,
    build_system,
    extract_parameters,
    openmm_available,
)

requires_openmm = pytest.mark.skipif(not openmm_available(), reason="OpenMM is not installed")
pytestmark = [requires_rdkit, requires_openmm]

MOLECULES = [
    "CCO", "CC(=O)OC", "c1ccccc1C", "CC#N", "CC(C)(C)O",
    "OCC(O)CO", "ClC(Cl)Cl", "C=O", "O=C(N)C", "c1ccncc1", "CS(=O)(=O)C",
]


def _embedded(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _rdkit_energy(mol, only=None):
    """RDKit's own MMFF energy, optionally of a single term."""
    from rdkit.Chem import AllChem

    props = AllChem.MMFFGetMoleculeProperties(mol)
    for term in _TERMS:
        getattr(props, f"SetMMFF{term}Term")(True if only is None else term == only)
    return AllChem.MMFFGetMoleculeForceField(mol, props).CalcEnergy()


def _openmm_energy(mol, params):
    import openmm as mm
    import openmm.unit as u

    system = build_system(params, 1, exact=True)
    context = mm.Context(
        system,
        mm.VerletIntegrator(1.0 * u.femtosecond),
        mm.Platform.getPlatformByName("Reference"),
    )
    conformer = mol.GetConformer()
    context.setPositions(
        np.array([list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]) * 0.1
    )
    joules = context.getState(energy=True).getPotentialEnergy().value_in_unit(
        u.kilojoule_per_mole
    )
    return joules / KCAL


# -- the translation ------------------------------------------------------


@pytest.mark.parametrize("smiles", MOLECULES)
def test_the_translation_reproduces_rdkit_exactly(smiles):
    """Every MMFF94 term, on the same geometry, to machine precision.

    Not 'close enough': the two implementations evaluate the same functional
    form on the same parameters, so anything above rounding is a mistranslation
    hiding somewhere - a factor of two on a stretch-bend, a degree-versus-radian
    slip in a cubic bend, a missing 1-4 scaling.
    """
    mol = _embedded(smiles)
    params = extract_parameters(mol)
    assert params is not None
    assert abs(_openmm_energy(mol, params) - _rdkit_energy(mol)) < 1e-6


def test_a_molecule_outside_mmff_is_refused_rather_than_partly_parameterised():
    from rdkit import Chem

    # A bare metal has no MMFF94 atom type. Half a force field would still run.
    mol = Chem.AddHs(Chem.MolFromSmiles("[Fe]"))
    assert extract_parameters(mol) is None


def test_the_torsion_phase_convention_is_not_guessed():
    """MMFF's V2 term carries a minus sign, which is a phase of pi in OpenMM."""
    mol = _embedded("CCCC")
    params = extract_parameters(mol)
    reference = _rdkit_energy(mol, only="Torsion")
    assert any(abs(v2) > 0.1 for *_, v2, _ in ((0, 0, 0, 0, t[5], t[6]) for t in params.torsions))
    assert abs(reference) > 1e-6  # the case would be vacuous otherwise
    assert abs(_openmm_energy(mol, params) - _rdkit_energy(mol)) < 1e-6


# -- the box --------------------------------------------------------------


def test_a_box_is_built_at_the_density_it_was_asked_for():
    box = condensed.pack_box(_embedded("CCO"), 216, 0.789, seed=0, expansion=1.0)
    assert box.n_molecules == 216
    assert box.density_g_cm3 == pytest.approx(0.789, rel=1e-6)


def test_a_box_too_small_for_the_cutoff_is_refused_before_it_runs():
    """Below twice the cutoff a molecule sees two images of one neighbour.

    The failure without this check arrives a hundred picoseconds in, as an
    OpenMM exception from the barostat, after the compute has been spent.
    """
    minimum = condensed.minimum_molecules(46.07, 0.789, 1.0)
    assert minimum > 1
    with pytest.raises(ValueError, match="at least"):
        condensed.run_npt(_embedded("CCO"), minimum // 4, 298.15)


def test_the_packing_leaves_room_for_the_barostat_to_compress():
    """Packing straight to the target density starts inside the repulsive wall."""
    target = 0.789
    box = condensed.pack_box(_embedded("CCO"), 216, target, seed=0)
    assert box.density_g_cm3 < target
    # But not so far below that equilibration becomes the whole run: an early
    # attempt started at 0.36 g/cm^3 and was still compressing 60 ps later.
    assert box.density_g_cm3 > 0.55 * target


def test_random_orientations_are_actually_random():
    first = condensed.pack_box(_embedded("CCO"), 27, 0.789, seed=1).positions
    second = condensed.pack_box(_embedded("CCO"), 27, 0.789, seed=2).positions
    assert not np.allclose(first, second)


# -- the observables ------------------------------------------------------


def test_cohesive_energy_density_is_a_difference_against_a_sampled_reference():
    liquid = condensed.CondensedResult(
        density_g_cm3=0.789,
        density_error=0.004,
        energy_per_molecule=-40.0,
        energy_error=0.3,
        edge_nm=3.0,
        production_ps=200.0,
        n_molecules=250,
        wall_seconds=1.0,
        molar_mass_g=46.07,
    )
    result = condensed.cohesive_energy_density(liquid, gas_energy=0.0, gas_error=0.2)

    # 40 kJ/mol over 58.4 cm^3/mol is 685 MPa, which is ethanol's cohesive
    # energy density: the square of a Hildebrand parameter of 26 MPa^0.5.
    assert result.vaporisation_energy == pytest.approx(40.0)
    assert result.molar_volume_cm3 == pytest.approx(46.07 / 0.789)
    assert result.cohesive_energy_density_pa == pytest.approx(6.85e8, rel=0.02)
    assert result.error_pa > 0


def test_an_unbound_liquid_is_reported_rather_than_returned_as_a_number():
    liquid = condensed.CondensedResult(
        density_g_cm3=0.789,
        density_error=0.004,
        energy_per_molecule=+5.0,   # above the isolated molecule: not a liquid
        energy_error=0.3,
        edge_nm=3.0,
        production_ps=200.0,
        n_molecules=250,
        wall_seconds=1.0,
        molar_mass_g=46.07,
    )
    result = condensed.cohesive_energy_density(liquid, gas_energy=0.0, gas_error=0.2)
    assert result.cohesive_energy_density_pa < 0
    assert any("not a bound phase" in d for d in result.diagnostics)
