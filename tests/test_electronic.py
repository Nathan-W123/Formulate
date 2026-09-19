"""Tests for the quantum electronic-structure expert.

The refusals are tested at least as hard as the predictions.  In this
repository a refusal is a feature: an expert that quietly returns a mid-range
number for a radical, an anion or a saturated alcohol's "gap" is worse than one
that declines, so every refusal here is pinned with the chemical argument it
has to carry, not only with its status.

The expensive tests are confined to molecules whose whole ensemble is a couple
of seconds - water, ethanol, 1,2-dichloroethane - and they pin numbers that
were measured against experiment in the validation run behind the module's
error constants.
"""

from __future__ import annotations

import math

import pytest

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MixtureComponent,
    MixtureSpec,
    MoleculeSpec,
    molecule_candidate,
    polymer_candidate,
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import PredictionStatus
from formulate.core.properties import PropertyFamily, get_property
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind
from formulate.experts.base import PredictionRequest
from formulate.experts.electronic import (
    ATOMIZATION_FLOOR_KJ_MOL,
    ATOMIZATION_RELATIVE_STD,
    BASIS_ELEMENTS,
    D_BLOCK_ELEMENTS,
    DIPOLE_METHOD_STD_DEBYE,
    GAP_MEAN_SIGNED_EV,
    GAP_RMSE_EV,
    MMFF_GEOMETRY_DIPOLE_STD_DEBYE,
    SMALL_GAP_EV,
    TRIPLET_PROBE_GAP_EV,
    VALIDATION,
    Ensemble,
    QuantumElectronicExpert,
    _boltzmann,
    cached_ensemble,
    free_atom_energy,
    has_valence_acceptor,
)

pytest.importorskip("rdkit")
pyscf = pytest.importorskip("pyscf")


EXPERT = QuantumElectronicExpert()

#: Measured against experiment in the validation run that set this module's
#: error constants.  Pinned so a change of functional, basis or geometry
#: protocol cannot pass silently.
WATER_DIPOLE_EXPERIMENTAL = 1.855
WATER_ATOMIZATION_DE_KJ_MOL = 973.2


def request_for(smiles: str, *props: str, conditions: Conditions | None = None):
    return PredictionRequest(
        candidate=molecule_candidate(smiles),
        properties=frozenset(props),
        conditions=conditions or Conditions.standard(),
    )


def predict(smiles: str, *props: str, expert: QuantumElectronicExpert | None = None):
    exp = expert or EXPERT
    return {p.property: p for p in exp.predict(request_for(smiles, *props))}


@pytest.fixture
def clean_ensemble_cache():
    """Isolate a test that deliberately builds a degraded ensemble.

    The module cache is keyed on structure and level of theory, not on which
    backends happened to work, so a test that breaks GFN2-xTB would otherwise
    hand its damaged geometry to every later test on the same molecule.
    """
    from formulate.experts import electronic

    electronic._ENSEMBLE_CACHE.clear()
    yield
    electronic._ENSEMBLE_CACHE.clear()


# ---------------------------------------------------------------------------
# Contract and metadata
# ---------------------------------------------------------------------------


def test_identity_and_family():
    assert EXPERT.id == "quantum_electronic"
    assert EXPERT.family is PropertyFamily.ELECTRICAL
    assert EXPERT.supported_classes == frozenset({MaterialClass.MOLECULE})
    assert EXPERT.dependencies == frozenset()


def test_method_names_the_level_of_theory_and_its_two_caveats():
    method = EXPERT.method
    assert "B3LYP/6-31G(d,p)" in method
    assert "GFN2-xTB" in method
    # The two ways this expert is most likely to mislead someone have to be in
    # the method string itself, not only in a note a reader may not reach.
    assert "not a fundamental or optical gap" in method
    assert "no zero-point correction" in method


def test_electronic_energy_is_deliberately_not_covered():
    """Registered in the property registry, refused here by omission.

    An absolute electronic energy is monotone in electron count, so ranking on
    it sorts candidates by size.  Listing it would make the registry report the
    electrical family covered when it is not.
    """
    get_property("electronic_energy")  # it is a real registered property
    assert "electronic_energy" not in EXPERT.supported_properties
    assert EXPERT.supported_properties == frozenset(
        {"dipole_moment", "homo_lumo_gap", "atomization_energy"}
    )


def test_the_quoted_dipole_constant_is_the_held_out_one():
    """The rule of this repository: quote the held-out number, not the fit.

    The constant is one fitted parameter, so it has to be the figure from the
    molecules that did not set it.  In sample the rigid set gives 0.127 D.
    """
    held_out = [row for row in VALIDATION if "held out" in row[0]]
    assert len(held_out) == 1
    label, n, statistic = held_out[0]
    assert n == 7
    assert f"{DIPOLE_METHOD_STD_DEBYE:.3f} D" in statistic
    in_sample = [row for row in VALIDATION if "in sample" in row[0]][0]
    assert "0.127 D" in in_sample[2]
    assert DIPOLE_METHOD_STD_DEBYE > 0.127


def test_is_available_reports_a_reason_or_nothing():
    if EXPERT.is_available():
        assert EXPERT.unavailable_reason() == ""
    else:  # pragma: no cover - this environment has both backends
        assert EXPERT.unavailable_reason()


# ---------------------------------------------------------------------------
# Refusals: material class
# ---------------------------------------------------------------------------


def test_polymer_is_not_its_monomer():
    """A chain's dipole depends on tacticity, length and conformation."""
    assert not EXPERT.covers("dipole_moment", MaterialClass.POLYMER)
    candidate = polymer_candidate("[*]CC([*])c1ccccc1")
    assert EXPERT.predict(
        PredictionRequest(
            candidate=candidate,
            properties=frozenset({"dipole_moment", "atomization_energy"}),
            conditions=Conditions.standard(),
        )
    ) == []


def test_mixture_is_not_its_major_component():
    assert not EXPERT.covers("dipole_moment", MaterialClass.MIXTURE)
    mixture = Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=(
                MixtureComponent(fraction=0.6, molecule=MoleculeSpec(smiles="O")),
                MixtureComponent(fraction=0.4, molecule=MoleculeSpec(smiles="CCO")),
            )
        ),
    )
    assert EXPERT.predict(
        PredictionRequest(
            candidate=mixture,
            properties=frozenset({"dipole_moment"}),
            conditions=Conditions.standard(),
        )
    ) == []


# ---------------------------------------------------------------------------
# Refusals: chemistry.  None of these may spend an SCF cycle.
# ---------------------------------------------------------------------------


ALL_THREE = ("dipole_moment", "homo_lumo_gap", "atomization_energy")


def test_charged_species_refused_for_every_property():
    out = predict("CC(=O)[O-]", *ALL_THREE)
    assert set(out) == set(ALL_THREE)
    for prop, pred in out.items():
        assert pred.status is PredictionStatus.UNSUPPORTED, prop
        assert pred.quantity is None
        reason = " ".join(pred.notes)
        assert "formal charge" in reason
        # The argument has to be chemical, not a stack trace.
        assert "origin-independent dipole" in reason
        assert "diffuse" in reason


def test_radical_refused_for_every_property():
    out = predict("[CH3]", *ALL_THREE)
    for prop, pred in out.items():
        assert pred.status is PredictionStatus.UNSUPPORTED, prop
        reason = " ".join(pred.notes)
        assert "open-shell" in reason
        assert "closed-shell SCF" in reason


def test_odd_electron_count_refused_even_without_a_radical_flag():
    """Nitric oxide: RDKit's Lewis structure carries the radical, and the
    electron-count check is the backstop when it does not."""
    out = predict("[N]=O", "dipole_moment")
    assert out["dipole_moment"].status is PredictionStatus.UNSUPPORTED


def test_oversized_molecule_refused_with_the_measured_cost():
    out = predict("CC(=O)Oc1ccccc1C(=O)O", *ALL_THREE)  # aspirin, 13 heavy atoms
    for pred in out.values():
        assert pred.status is PredictionStatus.UNSUPPORTED
        reason = " ".join(pred.notes)
        assert "13 heavy atoms" in reason
        assert "38.5 s" in reason  # the measurement, not a hand-wave


def test_element_outside_the_basis_refused():
    out = predict("Ic1ccccc1", *ALL_THREE)
    for pred in out.values():
        assert pred.status is PredictionStatus.UNSUPPORTED
        assert "6-31G(d,p) is not defined for I" in " ".join(pred.notes)


def test_bromine_is_outside_this_basis_although_it_is_a_common_element():
    """Probed, not assumed: PySCF's 6-31G(d,p) stops at zinc."""
    assert "Br" not in BASIS_ELEMENTS
    assert "Se" not in BASIS_ELEMENTS
    assert "Zn" in BASIS_ELEMENTS and "Cl" in BASIS_ELEMENTS
    out = predict("Brc1ccccc1", "dipole_moment")
    assert out["dipole_moment"].status is PredictionStatus.UNSUPPORTED
    assert "not defined for Br" in " ".join(out["dipole_moment"].notes)


def test_unparseable_smiles_fails_with_the_toolkit_reason():
    out = predict("C1CC", "dipole_moment")
    pred = out["dipole_moment"]
    assert pred.status is PredictionStatus.FAILED
    assert "cannot parse" in " ".join(pred.notes)


# ---------------------------------------------------------------------------
# Refusals scoped to one property, which is the harder case to get right
# ---------------------------------------------------------------------------


def test_saturated_molecule_refuses_the_gap_only():
    """Ethanol's lowest virtual orbital is a discretised continuum state.

    This must cost nothing: the refusal has to come before the SCF.
    """
    out = predict("CCO", "homo_lumo_gap")
    pred = out["homo_lumo_gap"]
    assert pred.status is PredictionStatus.UNSUPPORTED
    reason = " ".join(pred.notes)
    assert "no valence acceptor orbital" in reason
    assert "property of the basis set" in reason


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CCO", False),
        ("CCCC", False),
        ("CC(C)=O", True),
        ("O=C=O", True),
        ("c1ccccc1", True),
        ("CC#N", True),
        ("O", False),
    ],
)
def test_valence_acceptor_detection(smiles, expected):
    assert has_valence_acceptor(smiles) is expected


def test_element_without_an_atomic_reference_refuses_atomization_only(monkeypatch):
    """The free-atom table is the last gate before an atomization energy.

    Zinc is the natural example - inside 6-31G(d,p), absent from the
    multiplicity table - but it is now refused earlier, as a d-block element,
    so the guard is exercised by removing an element that does get this far.
    """
    assert EXPERT._missing_atomic_references("[Zn]") == ["Zn"]

    from formulate.physics.qm import thermochemistry

    monkeypatch.delitem(thermochemistry.ATOMIC_MULTIPLICITY, "O")
    out = predict("O", "atomization_energy")
    pred = out["atomization_energy"]
    assert pred.status is PredictionStatus.UNSUPPORTED
    reason = " ".join(pred.notes)
    assert "free-atom ground-state multiplicity for O" in reason
    assert "400 kJ/mol per carbon" in reason


def test_d_block_is_refused_although_the_basis_reaches_it():
    """6-31G(d,p) is defined through zinc; that is not the same as usable there.

    Chromyl chloride is the case that earned this: before the check it came
    back with a 0.68 D dipole wearing the 0.144 D bar fitted to small organics.
    """
    assert D_BLOCK_ELEMENTS < BASIS_ELEMENTS
    for smiles in ("O=[Cr](=O)(Cl)Cl", "[Zn](C)C"):
        out = predict(smiles, *ALL_THREE)
        for prop, pred in out.items():
            assert pred.status is PredictionStatus.UNSUPPORTED, (smiles, prop)
            assert pred.quantity is None
            reason = " ".join(pred.notes)
            assert "d-block element" in reason
            # It has to say what to do instead, not only that it declines.
            assert "def2" in reason
        assert not EXPERT.assess_domain(molecule_candidate(smiles)).in_domain


def test_atomization_and_gap_refused_without_xtb(monkeypatch):
    """An MMFF geometry costs 369 kJ/mol and 4.16 eV on carbon dioxide."""
    from formulate.physics.qm import xtb_backend

    monkeypatch.setattr(xtb_backend.XTBBackend, "is_available", lambda self: False)
    out = predict("O=C=O", "atomization_energy", "homo_lumo_gap")
    assert set(out) == {"atomization_energy", "homo_lumo_gap"}
    for pred in out.values():
        assert pred.status is PredictionStatus.UNSUPPORTED
        reason = " ".join(pred.notes)
        assert "GFN2-xTB is not importable" in reason
    assert "369 kJ/mol" in " ".join(out["atomization_energy"].notes)
    assert "4.16 eV" in " ".join(out["homo_lumo_gap"].notes)


@pytest.mark.slow
def test_an_xtb_that_runs_and_fails_is_not_an_xtb_that_ran(clean_ensemble_cache):
    """The refusal has to be on the relaxation, not on the import.

    ``XTBBackend.is_available()`` only says the package imports.  A GFN2-xTB
    run that raises or fails to converge leaves exactly the MMFF94 geometry
    behind, and on carbon dioxide that geometry is worth -369 kJ/mol: 1259
    against a true De of 1628, which is the number this expert would otherwise
    report inside a 31 kJ/mol bar.
    """
    from formulate.physics.qm import xtb_backend
    from formulate.physics.qm.base import ConvergenceStatus, QMResult

    def crashed(self, request):
        return QMResult(
            status=ConvergenceStatus.FAILED,
            method_signature="GFN2-xTB",
            backend="xtb",
            diagnostics=("simulated SCC failure",),
        )

    original = xtb_backend.XTBBackend.run
    xtb_backend.XTBBackend.run = crashed
    try:
        assert xtb_backend.XTBBackend().is_available()  # the import still works
        out = predict("O=C=O", *ALL_THREE)
        for prop in ("atomization_energy", "homo_lumo_gap"):
            pred = out[prop]
            assert pred.status is PredictionStatus.UNSUPPORTED, prop
            assert pred.quantity is None, prop
            reason = " ".join(pred.notes)
            assert "relaxation did not complete" in reason
            assert "Install or repair xtb" in reason
        # The dipole survives the same substitution, widened rather than refused.
        dipole = out["dipole_moment"]
        assert dipole.status is PredictionStatus.OK
        assert dipole.uncertainty.std == pytest.approx(
            math.hypot(DIPOLE_METHOD_STD_DEBYE, MMFF_GEOMETRY_DIPOLE_STD_DEBYE), abs=1e-9
        )
        assert "MMFF94" in " ".join(dipole.notes)
    finally:
        xtb_backend.XTBBackend.run = original


@pytest.mark.slow
def test_a_triplet_ground_state_is_refused_rather_than_solved_as_a_singlet(
    clean_ensemble_cache,
):
    """Dioxygen: RDKit calls it closed shell and the electron count is even.

    Neither the radical check nor the parity check sees it, and the restricted
    SCF converges happily to a singlet whose atomization energy is 355 kJ/mol
    against a true De of 505 - inside a 20 kJ/mol bar.  The 1.93 eV gap is the
    only signal, and it buys a triplet single point rather than a flag.
    """
    from formulate import chem

    assert sum(a.GetNumRadicalElectrons() for a in chem.mol_from_smiles("O=O").GetAtoms()) == 0

    out = predict("O=O", *ALL_THREE)
    for prop, pred in out.items():
        assert pred.status is PredictionStatus.UNSUPPORTED, prop
        assert pred.quantity is None, prop
        reason = " ".join(pred.notes)
        assert "not the ground state" in reason
        assert "UKS triplet" in reason
        assert "open-shell triplet" in reason  # what to do next

    ensemble = cached_ensemble("O=O", 298.15, EXPERT.max_conformers)
    assert ensemble.wrong_spin_state
    # Measured here: the triplet is about 164 kJ/mol below the singlet.
    assert ensemble.triplet_below_singlet_j / 1000.0 > 100.0


@pytest.mark.slow
def test_the_triplet_probe_does_not_fire_on_an_ordinary_closed_shell_molecule():
    """It is a screen, not a tax: above the threshold nobody pays for it."""
    for smiles in ("O", "c1ccccc1"):
        ensemble = cached_ensemble(smiles, 298.15, EXPERT.max_conformers)
        assert min(g for g in ensemble.gaps if g is not None) > TRIPLET_PROBE_GAP_EV
        assert ensemble.triplet_below_singlet_j is None
        assert not ensemble.wrong_spin_state


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------


def test_domain_basis_is_a_real_statement():
    domain = EXPERT.assess_domain(molecule_candidate("O"))
    assert domain.in_domain
    assert "6-31G(d,p)" in domain.basis
    assert "heavy atoms" in domain.basis


def test_domain_marks_the_undersampled_flexible_case():
    domain = EXPERT.assess_domain(molecule_candidate("CCCCOCCCC"))
    assert domain.score < 1.0
    assert any("rotatable bonds" in w for w in domain.warnings)


@pytest.mark.parametrize(
    "smiles", ["CC(=O)[O-]", "[CH3]", "CC(=O)Oc1ccccc1C(=O)O", "Ic1ccccc1"]
)
def test_domain_agrees_with_the_refusals(smiles):
    assert not EXPERT.assess_domain(molecule_candidate(smiles)).in_domain


def _synthetic(**overrides) -> Ensemble:
    fields = dict(
        symbols=("C",), energies=(-1.0,), dipoles=(0.0,), gaps=(None,), weights=(1.0,),
        temperature_k=298.15, geometry_source="synthetic", xtb_used=True, mmff_used=True,
        populated=1, computed=1, max_bond_shift=None,
    )
    fields.update(overrides)
    return Ensemble(**fields)


def test_small_gap_marks_every_property_out_of_domain_not_just_the_gap():
    """One wavefunction produced all three numbers, so one fault taints all three."""
    marked = EXPERT._with_gap_diagnosis(
        ApplicabilityDomain(basis="x"), _synthetic(gaps=(SMALL_GAP_EV - 1.0,))
    )
    assert not marked.in_domain
    assert any("static correlation" in w for w in marked.warnings)

    healthy = EXPERT._with_gap_diagnosis(
        ApplicabilityDomain(basis="x"), _synthetic(gaps=(SMALL_GAP_EV + 1.0,))
    )
    assert healthy.in_domain


# ---------------------------------------------------------------------------
# Ensemble arithmetic, without any SCF
# ---------------------------------------------------------------------------


def test_boltzmann_weights_sum_to_one_and_favour_the_lowest():
    weights = _boltzmann((0.0, 5000.0, 20000.0), 298.15)
    assert math.isclose(sum(weights), 1.0)
    assert weights[0] > weights[1] > weights[2]


def test_dipole_is_averaged_as_a_root_mean_square():
    """sqrt(sum w mu^2) is what a molar-polarisation measurement returns.

    1,2-dichloroethane is the case that makes the choice visible: 0 D anti,
    about 3.1 D gauche.  An arithmetic mean of the two at equal weight is
    1.55 D; the r.m.s. is 2.19 D, and it is the r.m.s. the experiment reports.
    """
    ensemble = _synthetic(
        energies=(0.0, 0.0), dipoles=(0.0, 3.1), gaps=(None, None),
        weights=(0.5, 0.5), populated=2, computed=2,
    )
    assert ensemble.mean_dipole() == pytest.approx(math.sqrt(0.5 * 3.1**2), abs=1e-9)
    assert ensemble.mean_dipole() > 0.5 * (0.0 + 3.1)
    assert ensemble.dipole_spread() == pytest.approx(1.55, abs=1e-9)


def test_under_sampled_is_reported_not_hidden():
    assert _synthetic(populated=9, computed=3).under_sampled
    assert not _synthetic(populated=3, computed=3).under_sampled


# ---------------------------------------------------------------------------
# Real predictions.  Small molecules only.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_water_dipole_matches_experiment_within_its_own_bar():
    pred = predict("O", "dipole_moment")["dipole_moment"]
    assert pred.status is PredictionStatus.OK
    assert pred.quantity is not None and pred.quantity.unit == "debye"
    assert abs(pred.quantity.value - WATER_DIPOLE_EXPERIMENTAL) < pred.uncertainty.std
    assert pred.uncertainty.std >= DIPOLE_METHOD_STD_DEBYE
    assert pred.uncertainty.kind is UncertaintyKind.COMBINED
    assert "held-out RMSE" in pred.uncertainty.basis


@pytest.mark.slow
def test_benzene_dipole_is_zero_by_symmetry():
    pred = predict("c1ccccc1", "dipole_moment")["dipole_moment"]
    assert pred.status is PredictionStatus.OK
    assert pred.quantity is not None
    assert pred.quantity.value < 0.05


@pytest.mark.slow
def test_water_atomization_matches_the_bottom_of_well_reference():
    pred = predict("O", "atomization_energy")["atomization_energy"]
    assert pred.status is PredictionStatus.OK
    assert pred.quantity is not None
    assert abs(pred.quantity.value - WATER_ATOMIZATION_DE_KJ_MOL) < 2.0 * pred.uncertainty.std


@pytest.mark.slow
def test_atomization_unit_conversion_is_pinned():
    """kJ/mol in, J/mol canonical, and the spread converts with it."""
    pred = predict("O", "atomization_energy")["atomization_energy"]
    assert pred.quantity is not None
    assert pred.quantity.unit == "kilojoule / mole"
    canonical = pred.canonical
    assert canonical is not None
    assert canonical.unit == get_property("atomization_energy").canonical_unit
    assert canonical.value == pytest.approx(pred.quantity.value * 1000.0)
    assert pred.canonical_uncertainty.std == pytest.approx(pred.uncertainty.std * 1000.0)
    # And the std is the floor-plus-fraction form, not a convergence threshold.
    assert pred.uncertainty.std == pytest.approx(
        max(ATOMIZATION_FLOOR_KJ_MOL, ATOMIZATION_RELATIVE_STD * pred.quantity.value)
    )


@pytest.mark.slow
def test_every_prediction_carries_a_defensible_record():
    out = predict("O", "dipole_moment", "atomization_energy")
    for prop, pred in out.items():
        assert pred.status is PredictionStatus.OK, prop
        assert pred.uncertainty.std is not None, prop
        assert pred.uncertainty.basis, prop
        assert pred.provenance is not None
        params = pred.provenance.parameters
        assert params["functional"] == "b3lyp"
        assert params["basis_set"] == "6-31G(d,p)"
        assert "GFN2-xTB" in params["geometry"]
        # Vacuum, not the requested 1 atm of air: say where the number lives.
        assert pred.conditions.environment == "vacuum"


@pytest.mark.slow
def test_atomization_states_the_zero_point_offset_rather_than_burying_it():
    pred = predict("O", "atomization_energy")["atomization_energy"]
    joined = " ".join(pred.notes) + pred.uncertainty.basis
    assert "bottom-of-well" in joined
    assert "zero-point" in joined
    assert "NOT in this bar" in pred.uncertainty.basis


@pytest.mark.slow
def test_gap_uncertainty_states_the_sign_of_its_bias():
    pred = predict("c1ccccc1", "homo_lumo_gap")["homo_lumo_gap"]
    assert pred.status is PredictionStatus.OK
    assert pred.quantity is not None and pred.quantity.unit == "electron_volt"
    assert pred.uncertainty.std >= GAP_RMSE_EV
    assert "LARGER" in pred.uncertainty.basis
    assert f"{GAP_MEAN_SIGNED_EV:+.2f} eV" in pred.uncertainty.basis
    # An honest wide bar: the computed gap is several eV below the measured
    # fundamental gap of benzene (9.24 - (-1.12) = 10.36 eV).
    assert pred.quantity.value + pred.uncertainty.std > 10.36


@pytest.mark.slow
def test_flexible_molecule_carries_a_conformer_term():
    """1,2-dichloroethane: 0 D anti, about 3.1 D gauche.

    This is the test that fails if the ensemble is ever reduced to the single
    lowest-MMFF conformer - which for this molecule is the one with no dipole
    at all.
    """
    pred = predict("ClCCCl", "dipole_moment")["dipole_moment"]
    assert pred.status is PredictionStatus.OK
    assert pred.quantity is not None
    ensemble = cached_ensemble("ClCCCl", 298.15, EXPERT.max_conformers)
    assert ensemble.computed > 1
    assert ensemble.dipole_spread() > 0.3
    assert pred.quantity.value > min(ensemble.dipoles) + 0.3
    assert pred.uncertainty.std > DIPOLE_METHOD_STD_DEBYE
    assert "conformer spread" in pred.uncertainty.basis


@pytest.mark.slow
def test_free_atom_references_are_cached_across_calls():
    """Without the cache the free-atom SCFs run again for every molecule."""
    import time

    free_atom_energy("O")
    started = time.perf_counter()
    value = free_atom_energy("O")
    assert value is not None
    assert time.perf_counter() - started < 0.05


@pytest.mark.slow
def test_the_ensemble_is_computed_once_per_candidate_not_once_per_property():
    """Expert.predict loops per property; three SCF runs for one candidate
    would triple the cost silently."""
    import time

    predict("O", "dipole_moment", "atomization_energy")  # warm ensemble and free atoms
    started = time.perf_counter()
    out = predict("O", "dipole_moment", "atomization_energy")
    elapsed = time.perf_counter() - started
    assert len(out) == 2
    assert all(p.status is PredictionStatus.OK for p in out.values())
    assert elapsed < 1.0
