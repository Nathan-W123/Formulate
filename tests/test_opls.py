"""OPLS-AA typing and system construction.

The atom types and charges asserted here are the published OPLS-AA values, not
this run's output frozen into a test. That distinction is the point: a typing
engine that quietly hands back parameters from the wrong compound is the exact
failure this module exists to catch, and a regression test written from its own
output would agree with it.
"""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.physics.md import opls

pytestmark = [
    requires_rdkit,
    pytest.mark.skipif(not opls.available(), reason="foyer/OPLS-AA is not installed"),
]


def embedded(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


# --------------------------------------------------------------------------
# Typing
# --------------------------------------------------------------------------


def test_ethanol_gets_the_published_opls_types_and_charges():
    """Jorgensen's 1996 all-atom ethanol, atom for atom."""
    params = opls.extract_parameters(embedded("CCO"))
    assert params.atom_types == (
        "opls_135",  # CT, methyl carbon
        "opls_157",  # CT, carbon bearing the hydroxyl
        "opls_154",  # OH
        "opls_140",  # HC on the methyl
        "opls_140",
        "opls_140",
        "opls_140",  # HC on the methylene
        "opls_140",
        "opls_155",  # HO
    )
    assert params.charges == pytest.approx(
        (-0.18, 0.145, -0.683, 0.06, 0.06, 0.06, 0.06, 0.06, 0.418)
    )


def test_a_neutral_molecule_types_to_zero_net_charge():
    for smiles in ("CCCCCC", "Cc1ccccc1", "CC(C)=O", "CCOCC", "CO"):
        params = opls.extract_parameters(embedded(smiles))
        assert params.net_charge == pytest.approx(0.0, abs=opls.NET_CHARGE_TOLERANCE)


def test_chloroform_is_refused_because_its_types_do_not_add_up():
    """The loudest of the silent failures, and the reason for the gate.

    ``oplsaa.xml`` types chloroform from the monochloroalkane definitions,
    whose charges only balance for one chlorine. Every term is assigned, no
    exception is raised, and the molecule carries half an elementary charge.
    """
    with pytest.raises(opls.UnsupportedMolecule) as raised:
        opls.extract_parameters(embedded("ClC(Cl)Cl"))
    assert "net charge" in str(raised.value)
    assert "-0.503" in str(raised.value)


def test_an_ester_is_refused_outright():
    """No SMARTS covers the ester carbonyl carbon, and foyer says so."""
    with pytest.raises(opls.UnsupportedMolecule) as raised:
        opls.extract_parameters(embedded("CCOC(C)=O"))
    assert "no consistent typing" in str(raised.value)


def test_refusal_reports_the_reason_without_raising():
    assert opls.refusal(embedded("CCO")) is None
    assert "net charge" in opls.refusal(embedded("ClC(Cl)Cl"))


def test_a_molecule_with_no_conformer_is_refused():
    from rdkit import Chem

    with pytest.raises(opls.UnsupportedMolecule) as raised:
        opls.extract_parameters(Chem.AddHs(Chem.MolFromSmiles("CCO")))
    assert "conformer" in str(raised.value)


# --------------------------------------------------------------------------
# Building a system
# --------------------------------------------------------------------------


def test_copies_are_laid_out_molecule_by_molecule():
    """The layout pack_box produces, so positions and particles line up."""
    params = opls.extract_parameters(embedded("CCO"))
    system, structure = opls.build_system(params, 4, box_nm=3.0)
    assert system.getNumParticles() == 4 * params.n_atoms
    types = [atom.type for atom in structure.atoms]
    assert types == list(params.atom_types) * 4


def test_van_der_waals_is_mixed_geometrically_not_by_lorentz_berthelot():
    """OPLS-AA's combining rule is not the one a NonbondedForce implements.

    ParmEd moves the whole van der Waals part into a CustomNonbondedForce whose
    energy expression multiplies the per-particle parameters. Losing that would
    not raise; it would return a system that is quietly a different force
    field, so the test looks at the expression itself.
    """
    import openmm as mm

    params = opls.extract_parameters(embedded("CCO"))
    assert params.structure.combining_rule == "geometric"
    system, _ = opls.build_system(params, 4, box_nm=3.0)

    custom = [f for f in system.getForces() if isinstance(f, mm.CustomNonbondedForce)]
    assert len(custom) == 1
    assert "sigc=sigma1*sigma2" in custom[0].getEnergyFunction().replace(" ", "")
    assert "epsilon1*epsilon2" in custom[0].getEnergyFunction()

    # The plain NonbondedForce is left carrying only the charges.
    nonbonded = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)]
    assert len(nonbonded) == 1
    for index in range(nonbonded[0].getNumParticles()):
        _, _, epsilon = nonbonded[0].getParticleParameters(index)
        assert epsilon._value == 0.0


def test_a_periodic_box_carries_the_long_range_dispersion_correction():
    """A density is sensitive to it: without the correction the box runs light."""
    import openmm as mm

    params = opls.extract_parameters(embedded("CCCCCC"))
    system, _ = opls.build_system(params, 4, box_nm=3.0)
    for force in system.getForces():
        if isinstance(force, mm.CustomNonbondedForce):
            assert force.getUseLongRangeCorrection()
        if isinstance(force, mm.NonbondedForce):
            assert force.getUseDispersionCorrection()
            assert force.getNonbondedMethod() == mm.NonbondedForce.PME


def test_an_isolated_molecule_is_aperiodic():
    import openmm as mm

    params = opls.extract_parameters(embedded("CCO"))
    system, _ = opls.build_system(params, 1)
    assert not system.usesPeriodicBoundaryConditions()
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            assert force.getNonbondedMethod() == mm.NonbondedForce.NoCutoff


def test_a_cutoff_that_does_not_fit_the_box_is_refused():
    params = opls.extract_parameters(embedded("CCO"))
    with pytest.raises(ValueError, match="minimum image"):
        opls.build_system(params, 4, box_nm=1.8, cutoff_nm=1.0)


def test_one_four_interactions_are_scaled_by_one_half():
    """OPLS-AA's convention, and it is written in the force field header."""
    import openmm as mm

    params = opls.extract_parameters(embedded("CCCCCC"))
    system, _ = opls.build_system(params, 1)
    nonbonded = next(
        f for f in system.getForces() if isinstance(f, mm.NonbondedForce)
    )
    charges = {}
    for index in range(nonbonded.getNumParticles()):
        charges[index] = nonbonded.getParticleParameters(index)[0]._value

    scalings = []
    for index in range(nonbonded.getNumExceptions()):
        i, j, charge_product, _, epsilon = nonbonded.getExceptionParameters(index)
        if epsilon._value == 0.0:
            continue  # a 1-2 or 1-3 pair, excluded outright
        scalings.append(charge_product._value / (charges[i] * charges[j]))
    assert scalings
    assert scalings == pytest.approx([0.5] * len(scalings))
