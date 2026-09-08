"""The slab geometry a surface tension is measured in, and what it refuses.

The tension itself costs hours and is measured by ``bench/surface_tension.py``
rather than here. What is testable in a second is the packing - a slab has to
be at the density it was asked for, centred, with vacuum at both ends - and
the three refusals that stop a geometry from being simulated at all.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openmm")
Chem = pytest.importorskip("rdkit.Chem")

from rdkit.Chem import AllChem  # noqa: E402

from formulate.physics.md.interface import pack_slab, surface_tension  # noqa: E402


@pytest.fixture(scope="module")
def butanone():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCC(C)=O"))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def test_the_liquid_region_is_at_the_density_it_was_asked_for(butanone):
    slab = pack_slab(butanone, 300, 0.80, lateral_nm=3.2, vacuum_nm=4.0)
    volume_cm3 = (slab.box_nm[0] * slab.box_nm[1] * slab.liquid_nm) * 1e-21
    density = 300 * slab.molar_mass / (6.02214076e23 * volume_cm3)
    assert density == pytest.approx(0.80, rel=1e-9)


def test_the_slab_is_centred_with_vacuum_at_both_ends(butanone):
    slab = pack_slab(butanone, 300, 0.80, lateral_nm=3.2, vacuum_nm=4.0)
    z = slab.positions[:, 2]
    centre = slab.box_nm[2] / 2.0
    assert abs(z.mean() - centre) < 0.2
    # Nothing within a nanometre of either face of the box, so the two
    # surfaces cannot see each other through the periodic boundary.
    assert z.min() > 1.0
    assert z.max() < slab.box_nm[2] - 1.0
    assert slab.vacuum_nm == pytest.approx(8.0, rel=1e-9)


def test_every_molecule_is_placed_and_none_is_placed_twice(butanone):
    slab = pack_slab(butanone, 250, 0.80, lateral_nm=3.0, vacuum_nm=3.0, seed=4)
    assert slab.positions.shape == (250 * butanone.GetNumAtoms(), 3)
    centres = slab.positions.reshape(250, butanone.GetNumAtoms(), 3).mean(axis=1)
    separations = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
    np.fill_diagonal(separations, np.inf)
    assert separations.min() > 0.3


def test_a_cutoff_that_does_not_fit_the_lateral_edge_is_refused(butanone):
    with pytest.raises(ValueError, match="lateral edge"):
        surface_tension(butanone, 200, 298.15, 0.80, lateral_nm=2.0, cutoff_nm=1.2)


def test_vacuum_thinner_than_twice_the_cutoff_is_refused(butanone):
    with pytest.raises(ValueError, match="two surfaces of the slab interact"):
        surface_tension(butanone, 200, 298.15, 0.80, lateral_nm=3.2, cutoff_nm=1.2, vacuum_nm=2.0)


def test_a_slab_too_thin_to_have_a_bulk_interior_is_refused(butanone):
    # Fifty molecules across a 3.2 nm face is under a nanometre of liquid, so
    # the two surfaces would be in contact and there is no interior between
    # them for either of them to be the surface of.
    with pytest.raises(ValueError, match="no bulk liquid"):
        surface_tension(butanone, 50, 298.15, 0.80, lateral_nm=3.2, cutoff_nm=1.2)
