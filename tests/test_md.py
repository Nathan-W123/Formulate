"""Molecular dynamics: statistics, feasibility refusals, and cluster physics."""

from __future__ import annotations

import numpy as np
import pytest

from formulate.physics.geometry import geometry_from_smiles
from formulate.physics.md import (
    REQUIREMENTS,
    MDEngine,
    MDProtocol,
    MDRequest,
    autocorrelation_time,
    available_calculators,
    block_average,
    build_cluster,
    choose_calculator,
    detect_equilibration,
)

has_calculator = pytest.mark.skipif(
    not available_calculators(), reason="no MD force provider is installed"
)
has_rdkit = pytest.mark.skipif(
    not __import__("formulate").chem.rdkit_available(), reason="RDKit is not installed"
)


# -- statistics ------------------------------------------------------------


def test_block_averaging_recovers_independence_for_uncorrelated_data():
    rng = np.random.default_rng(0)
    result = block_average(rng.normal(size=2000))
    assert result.inflation < 2.5
    assert result.effective_samples > 500


def test_block_averaging_detects_serial_correlation():
    """The naive standard error understates a correlated series badly."""
    rng = np.random.default_rng(1)
    value, series = 0.0, []
    for _ in range(2000):
        value = 0.95 * value + rng.normal() * 0.31
        series.append(value)
    result = block_average(series)
    assert result.inflation > 4.0
    assert result.standard_error > result.naive_standard_error
    assert result.effective_samples < 200


def test_autocorrelation_time_matches_the_analytic_ar1_value():
    """For AR(1) the integrated time is (1+phi)/(1-phi)."""
    rng = np.random.default_rng(2)
    value, series = 0.0, []
    for _ in range(20000):
        value = 0.9 * value + rng.normal() * 0.436
        series.append(value)
    assert autocorrelation_time(series) == pytest.approx(19.0, rel=0.35)


def test_block_average_of_too_few_samples_reports_rather_than_guesses():
    result = block_average([1.0])
    assert not result.converged
    assert result.notes


def test_equilibration_detection_finds_a_transient():
    rng = np.random.default_rng(3)
    series = np.concatenate([np.linspace(10, 0, 300), rng.normal(size=1200)])
    index = detect_equilibration(series)
    assert 200 <= index <= 450


def test_equilibration_detection_discards_nothing_from_a_stationary_series():
    rng = np.random.default_rng(4)
    assert detect_equilibration(rng.normal(size=800)) < 200


# -- calculators -----------------------------------------------------------


def test_only_periodic_capable_calculators_are_offered_for_periodic_work():
    for choice in available_calculators(periodic=True):
        assert choice.supports_periodic


def test_the_cheapest_calculator_is_chosen_by_default():
    options = available_calculators()
    if not options:
        pytest.skip("no calculators")
    assert choose_calculator().reference_ms_per_evaluation == min(
        c.reference_ms_per_evaluation for c in options
    )


@has_rdkit
def test_the_mmff_adapter_returns_forces_not_gradients():
    from ase import Atoms
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from formulate.physics.md.calculators import MMFFCalculator

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    AllChem.MMFFOptimizeMolecule(mol)
    conformer = mol.GetConformer()
    positions = np.array(
        [list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    )
    atoms = Atoms(
        symbols=[a.GetSymbol() for a in mol.GetAtoms()], positions=positions
    )
    atoms.calc = MMFFCalculator(mol)

    minimum = atoms.get_potential_energy()
    displaced = positions.copy()
    displaced[0] += [0.3, 0.0, 0.0]
    atoms.set_positions(displaced)

    assert atoms.get_potential_energy() > minimum
    # A restoring force opposes the displacement; a gradient would not.
    assert atoms.get_forces()[0][0] < 0


# -- feasibility -----------------------------------------------------------


@has_rdkit
@pytest.mark.parametrize(
    "protocol",
    [MDProtocol.DENSITY, MDProtocol.SELF_DIFFUSION, MDProtocol.WORK_OF_SEPARATION],
)
def test_bulk_protocols_are_refused_with_a_reason_and_a_cost(protocol):
    """A density from eight molecules is not an approximate density."""
    geometry = geometry_from_smiles("CCO", n_conformers=2)
    result = MDEngine().run(MDRequest(geometry=geometry, protocol=protocol))
    assert not result.usable
    assert not result.feasible
    assert result.value is None
    assert "not feasible" in result.feasibility_reason
    assert "adequate run would take" in result.feasibility_reason


@has_rdkit
def test_a_feasible_protocol_is_permitted():
    geometry = geometry_from_smiles("CCO", n_conformers=2)
    verdict = MDEngine().assess(
        MDRequest(geometry=geometry, protocol=MDProtocol.CONFORMATIONAL_ENSEMBLE)
    )
    assert verdict.feasible


def test_every_protocol_declares_what_it_needs():
    for protocol in MDProtocol:
        requirement = REQUIREMENTS[protocol]
        assert requirement.min_molecules >= 1
        assert requirement.min_production_ps > 0
        assert requirement.rationale


# -- cluster construction --------------------------------------------------


@has_rdkit
def test_a_cluster_is_built_at_a_liquid_like_density():
    """Sizing the lattice from the molecule's extent gave a gas, not a liquid."""
    geometry = geometry_from_smiles("CCO", n_conformers=2)
    cluster = build_cluster(geometry, 12, seed=0, target_density_g_cm3=0.85)

    assert len(cluster.symbols) == 12 * len(geometry.symbols)
    extent = cluster.positions.max(axis=0) - cluster.positions.min(axis=0)
    volume_per_molecule = float(np.prod(extent)) / 12
    # Liquid ethanol occupies about 97 cubic Angstrom per molecule.
    assert 50 < volume_per_molecule < 200


@has_rdkit
def test_cluster_density_scales_with_the_requested_density():
    geometry = geometry_from_smiles("CCO", n_conformers=2)
    dense = build_cluster(geometry, 8, target_density_g_cm3=1.2)
    sparse = build_cluster(geometry, 8, target_density_g_cm3=0.4)
    def span(c):
        return float(np.prod(c.positions.max(axis=0) - c.positions.min(axis=0)))

    assert span(dense) < span(sparse)


# -- protocols that run ----------------------------------------------------


@has_rdkit
@has_calculator
@pytest.mark.slow
def test_cohesive_energy_has_the_right_sign_and_scale():
    """Sign and magnitude both depend on a temperature-consistent reference.

    Comparing a minimised isolated molecule against a thermally excited
    cluster charges every molecule's vibrational energy against the binding
    and flips the sign, which is what this guards.
    """
    geometry = geometry_from_smiles("CCO", n_conformers=3)
    result = MDEngine().run(
        MDRequest(
            geometry=geometry,
            protocol=MDProtocol.COHESIVE_ENERGY,
            calculator="GFN-FF",
            n_molecules=8,
            temperature_k=250,
            equilibration_steps=800,
            production_steps=2000,
            sample_interval=25,
        )
    )
    assert result.usable
    kj_per_mol = result.value.value / 1000.0
    # Ethanol's cohesive energy is about 39.5 kJ/mol; an 8-molecule cluster is
    # mostly surface and must land below it, but well above zero.
    assert 5.0 < kj_per_mol < 45.0
    assert result.sampling.standard_error > 0
    assert any("surface" in d for d in result.diagnostics)
    assert any("same" in d and "K" in d for d in result.diagnostics)


@has_rdkit
@has_calculator
@pytest.mark.slow
def test_a_short_run_says_that_it_is_short():
    geometry = geometry_from_smiles("CCCCO", n_conformers=3)
    result = MDEngine().run(
        MDRequest(
            geometry=geometry,
            protocol=MDProtocol.CONFORMATIONAL_ENSEMBLE,
            calculator="GFN-FF",
            equilibration_steps=200,
            production_steps=1000,
            sample_interval=10,
        )
    )
    assert result.usable
    assert any("ps of production sampling against" in d for d in result.diagnostics)
    assert any("finite-size" in limitation for limitation in result.limitations)


# -- the ensemble is not silently substituted ------------------------------


def _request(**kw):
    from formulate.physics.md import MDProtocol, MDRequest

    geometry = kw.pop("geometry", None)
    return MDRequest(
        geometry=geometry,
        protocol=kw.pop("protocol", MDProtocol.CONFORMATIONAL_ENSEMBLE),
        calculator="GFN-FF",
        temperature_k=298.15,
        equilibration_steps=10,
        production_steps=20,
        sample_interval=5,
        **kw,
    )


def test_a_constant_pressure_run_is_refused_rather_than_run_at_constant_volume():
    """No barostat exists here, and NVT wearing an NPT label is not a near miss.

    The volume is the observable a constant-pressure run exists to produce.
    Holding it fixed does not fail loudly; it answers a different question and
    returns a number that looks fine.
    """
    from formulate.physics.geometry import geometry_from_smiles
    from formulate.physics.md.base import Ensemble
    from formulate.physics.md.engine import MDEngine

    geometry = geometry_from_smiles("CCO", n_conformers=2)
    engine = MDEngine()

    assert engine.assess(_request(geometry=geometry)).feasible

    npt = engine.assess(_request(geometry=geometry, ensemble=Ensemble.NPT))
    assert not npt.feasible
    assert "barostat" in npt.reason


def test_a_requested_pressure_is_refused_rather_than_recorded_and_ignored():
    from formulate.physics.geometry import geometry_from_smiles
    from formulate.physics.md.engine import MDEngine

    geometry = geometry_from_smiles("CCO", n_conformers=2)
    verdict = MDEngine().assess(_request(geometry=geometry, pressure_pa=101325.0))
    assert not verdict.feasible
    assert "pressure" in verdict.reason


def test_every_runnable_protocol_names_its_own_workflow():
    """Protocol dispatch is a mapping, so an unimplemented one cannot borrow another."""
    from formulate.physics.md.base import REQUIREMENTS, MDProtocol
    from formulate.physics.md.engine import MDEngine, PROTOCOL_METHODS

    assert set(PROTOCOL_METHODS) <= set(MDProtocol)
    for protocol, method in PROTOCOL_METHODS.items():
        assert callable(getattr(MDEngine, method)), protocol
    # Every protocol still declares what it would need, implemented or not.
    assert set(REQUIREMENTS) == set(MDProtocol)


def test_forcing_an_unimplemented_protocol_does_not_run_a_different_one():
    from formulate.physics.geometry import geometry_from_smiles
    from formulate.physics.md import MDProtocol
    from formulate.physics.md.engine import MDEngine

    geometry = geometry_from_smiles("CCO", n_conformers=2)
    result = MDEngine().run(
        _request(geometry=geometry, protocol=MDProtocol.DENSITY, force_run=True)
    )
    assert not result.usable
    assert result.value is None
