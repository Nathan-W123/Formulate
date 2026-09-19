"""The GFN2-xTB electronic expert: accuracy, units and - hardest - refusals.

In this repository a refusal is a feature.  Most of what follows tests that
this expert declines to answer, and several tests also check the *premise* of a
refusal against the backend, so that a refusal retires itself if a future xtb
fixes the thing it guards against.

Reference dipole moments are experimental gas-phase values from the CRC
Handbook "Dipole Moments" table / NIST CCCBDB.  Predicted values pinned here
were measured by running the shipped protocol.
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
    MonomerUnit,
    PolymerSpec,
    molecule_candidate,
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import PredictionStatus, prefer
from formulate.core.properties import get_property
from formulate.core.quantity import Quantity, UncertaintyKind
from formulate.experts.base import PredictionRequest
from formulate.experts.semiempirical import (
    ANGSTROM_PER_BOHR,
    DEBYE_PER_AU,
    DIPOLE_STD_FLOOR_DEBYE,
    DIPOLE_STD_RELATIVE,
    MAX_ATOMIC_NUMBER,
    MAX_GRADIENT_EV_PER_ANGSTROM,
    SOFT_ELEMENT_FACTOR,
    SemiempiricalElectronicExpert,
    conformer_budget,
    run_ensemble,
    screen,
    single_threaded,
    torsion_count,
)

EXPERT = SemiempiricalElectronicExpert()

available = pytest.mark.skipif(
    not EXPERT.is_available(), reason="xtb or RDKit is not installed"
)

#: Experimental gas-phase dipole moments, debye (CRC / NIST CCCBDB).
EXPERIMENTAL = {
    "O": 1.855,
    "CC#N": 3.92,
    "N#Cc1ccccc1": 4.18,
    "Clc1ccccc1": 1.69,
    "CC(C)=O": 2.88,
    "c1ccncc1": 2.215,
    "CO": 1.70,
    "CCO": 1.69,
    "CCOCC": 1.15,
    "ClCCl": 1.60,
    "ClC(Cl)Cl": 1.04,
    "Nc1ccccc1": 1.53,
    "COC(C)=O": 1.72,
    "c1ccoc1": 0.66,
}


def predict(smiles: str, conditions: Conditions | None = None):
    """The single dipole prediction this expert makes for ``smiles``."""
    candidate = molecule_candidate(smiles)
    request = PredictionRequest(
        candidate=candidate,
        properties=frozenset({"dipole_moment"}),
        conditions=conditions or Conditions.standard(),
    )
    predictions = EXPERT.predict(request)
    assert len(predictions) == 1
    return predictions[0]


def debye(prediction) -> float:
    return prediction.canonical.value


# --------------------------------------------------------------------------
# Capability declaration
# --------------------------------------------------------------------------


def test_declares_only_the_dipole():
    assert EXPERT.supported_properties == frozenset({"dipole_moment"})
    assert EXPERT.supported_classes == frozenset({MaterialClass.MOLECULE})
    assert EXPERT.dependencies == frozenset()
    assert EXPERT.id == "xtb_electronic"


def test_homo_lumo_gap_is_refused_by_absence_not_by_a_bad_number():
    """A universal refusal belongs in the capability declaration.

    ``registry.uncovered()`` keys off ``covers()``, so declaring a property this
    expert never delivers would make a run report claim the gap is covered when
    nothing computes it.  Guards against a well-meaning re-addition.
    """
    assert "homo_lumo_gap" not in EXPERT.supported_properties
    assert not EXPERT.covers("homo_lumo_gap", MaterialClass.MOLECULE)
    assert EXPERT.applicable_properties(
        {"homo_lumo_gap", "dipole_moment"}, MaterialClass.MOLECULE
    ) == frozenset({"dipole_moment"})

    import formulate.experts.semiempirical as module

    doc = module.__doc__ or ""
    assert "homo_lumo_gap" in doc
    assert "B3LYP" in doc, "the refusal must carry the measurement that justifies it"


# --------------------------------------------------------------------------
# Units and conventions - the errors that produce a plausible wrong number
# --------------------------------------------------------------------------


@available
def test_positions_are_passed_in_bohr():
    """The published GFN2 energy of water only comes out on the bohr convention.

    Feeding angstrom does not raise.  It returns -3.5007 Eh and a
    plausible-looking dipole for a molecule stretched by a factor of 1.89.
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from xtb.interface import Calculator
    from xtb.libxtb import VERBOSITY_MUTED
    from xtb.utils import get_method

    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    positions = np.asarray(mol.GetConformer().GetPositions())

    with single_threaded():
        calc = Calculator(get_method("GFN2-xTB"), numbers, positions / ANGSTROM_PER_BOHR)
        calc.set_verbosity(VERBOSITY_MUTED)
        energy = float(calc.singlepoint().get_energy())

        wrong = Calculator(get_method("GFN2-xTB"), numbers, positions)
        wrong.set_verbosity(VERBOSITY_MUTED)
        wrong_energy = float(wrong.singlepoint().get_energy())

    assert energy == pytest.approx(-5.0702, abs=2e-3)
    assert not math.isclose(wrong_energy, energy, abs_tol=0.1)


def test_dipole_conversion_constant():
    """1 a.u. of dipole = 2.541746473 debye, and the constant is used as such."""
    assert DEBYE_PER_AU == pytest.approx(2.541746473, abs=1e-9)
    # e * bohr expressed in debye: e in coulomb, bohr in metre, 1 D = 1e-21/c C m.
    elementary_charge = 1.602176634e-19
    speed_of_light = 2.99792458e8
    expected = elementary_charge * ANGSTROM_PER_BOHR * 1e-10 * speed_of_light / 1e-21
    assert DEBYE_PER_AU == pytest.approx(expected, rel=1e-6)


@available
def test_quantity_is_in_debye_and_round_trips():
    prediction = predict("CC#N")
    assert prediction.quantity.unit == "debye"
    assert get_property("dipole_moment").canonical_unit == "debye"
    assert prediction.canonical.value == pytest.approx(prediction.quantity.value)
    # Uncertainty converts with difference semantics and stays a spread.
    converted = prediction.canonical_uncertainty
    assert converted.std == pytest.approx(prediction.uncertainty.std)
    assert converted.kind is UncertaintyKind.COMBINED


# --------------------------------------------------------------------------
# Accuracy - the point of the expert
# --------------------------------------------------------------------------


@available
@pytest.mark.parametrize(
    "smiles", ["O", "CC#N", "N#Cc1ccccc1", "Clc1ccccc1", "CC(C)=O", "c1ccncc1"]
)
def test_well_characterised_molecules_land_inside_their_own_two_sigma(smiles):
    """Tests the error bar, not just the value."""
    prediction = predict(smiles)
    assert prediction.is_usable
    std = prediction.uncertainty.std
    assert std is not None and std > 0
    assert abs(debye(prediction) - EXPERIMENTAL[smiles]) <= 2 * std


@available
@pytest.mark.parametrize("smiles", ["c1ccccc1", "C1CCCCC1", "ClC(Cl)(Cl)Cl"])
def test_centrosymmetric_molecules_have_no_dipole(smiles):
    """A method that gives a centrosymmetric molecule a dipole is broken."""
    assert debye(predict(smiles)) < 0.05


@available
def test_rank_order_the_engine_would_have_to_separate():
    benzene = debye(predict("c1ccccc1"))
    chlorobenzene = debye(predict("Clc1ccccc1"))
    nitrobenzene = debye(predict("O=[N+]([O-])c1ccccc1"))
    assert benzene < chlorobenzene < nitrobenzene


@available
def test_pinned_measured_values():
    """Two values measured during validation, pinned as a regression guard.

    A broken conversion factor or a broken Boltzmann weighting changes these
    long before it changes a coverage fraction.
    """
    assert debye(predict("O")) == pytest.approx(2.287, abs=0.01)
    assert debye(predict("Clc1ccccc1")) == pytest.approx(1.770, abs=0.01)


@available
def test_calibration_over_a_subset():
    """One-sigma coverage and mean absolute error over fourteen compounds.

    Bracketed rather than pinned: pinning a coverage fraction on fourteen
    points would be pinning noise.  The whole 79-compound validation gives
    MAE 0.430 D and 69.6% one-sigma coverage.
    """
    errors = []
    inside = 0
    for smiles, experimental in EXPERIMENTAL.items():
        prediction = predict(smiles)
        assert prediction.is_usable, smiles
        error = debye(prediction) - experimental
        errors.append(abs(error))
        if abs(error) <= prediction.uncertainty.std:
            inside += 1
    mae = sum(errors) / len(errors)
    coverage = inside / len(errors)
    assert mae < 0.7, f"mean absolute error {mae:.3f} D"
    assert 0.5 <= coverage <= 0.95, f"one-sigma coverage {coverage:.2f}"


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


@available
def test_heavy_element_is_screened_before_any_xtb_object_exists(monkeypatch):
    """The important assertion is that xtb is never reached.

    Z >= 87 segfaults or hits a Fortran ERROR STOP inside the backend; both
    kill the interpreter, and ``Expert.predict``'s try/except cannot save the
    run.  Constructing a Calculator here would take the pytest process with it,
    so the test makes the Calculator explode loudly instead.
    """
    import xtb.interface

    def forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("an xtb Calculator was constructed for a screened element")

    monkeypatch.setattr(xtb.interface, "Calculator", forbidden)

    for smiles, symbol in (("[U]", "U"), ("[Pu]", "Pu")):
        prediction = predict(smiles)
        assert prediction.status is PredictionStatus.UNSUPPORTED
        assert prediction.quantity is None
        assert symbol in prediction.notes[0]


def test_element_boundary_is_at_radon():
    """Z = 86 is supported, Z = 87 is not.  Pins the boundary itself."""
    assert MAX_ATOMIC_NUMBER == 86
    assert screen("[Rn]") is None
    assert "Fr" in (screen("[Fr]") or "")


@available
def test_charged_species_is_refused_for_origin_dependence():
    prediction = predict("CC(=O)[O-]")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "origin-dependent" in prediction.notes[0]


@available
def test_the_charge_refusal_premise_holds():
    """The acetate 'dipole' really does depend on where the molecule sits.

    Without this, the refusal above is an assertion rather than a measurement.
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from xtb.interface import Calculator
    from xtb.libxtb import VERBOSITY_MUTED
    from xtb.utils import get_method

    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    base = np.asarray(mol.GetConformer().GetPositions())

    values = []
    with single_threaded():
        for shift in (0.0, 5.0, 20.0):
            moved = base + np.array([shift, 0.0, 0.0])
            calc = Calculator(
                get_method("GFN2-xTB"), numbers, moved / ANGSTROM_PER_BOHR, charge=-1.0
            )
            calc.set_verbosity(VERBOSITY_MUTED)
            result = calc.singlepoint()
            values.append(
                float(np.linalg.norm(np.asarray(result.get_dipole()))) * DEBYE_PER_AU
            )

    assert values[2] > 10 * values[0], f"expected origin dependence, got {values}"


@available
def test_a_neutral_molecule_is_translation_invariant():
    """The control that makes the charge refusal meaningful."""
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from xtb.interface import Calculator
    from xtb.libxtb import VERBOSITY_MUTED
    from xtb.utils import get_method

    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)O"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    base = np.asarray(mol.GetConformer().GetPositions())

    values = []
    with single_threaded():
        for shift in (0.0, 20.0):
            calc = Calculator(
                get_method("GFN2-xTB"),
                numbers,
                (base + np.array([shift, 0.0, 0.0])) / ANGSTROM_PER_BOHR,
            )
            calc.set_verbosity(VERBOSITY_MUTED)
            values.append(
                float(np.linalg.norm(np.asarray(calc.singlepoint().get_dipole())))
                * DEBYE_PER_AU
            )

    assert values[0] == pytest.approx(values[1], abs=1e-6)


@available
@pytest.mark.parametrize("smiles", ["[CH3]", "c1ccc(cc1)[O]"])
def test_radicals_are_refused(smiles):
    prediction = predict(smiles)
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "open-shell" in prediction.notes[0]


@available
def test_the_radical_refusal_premise_holds():
    """xtb's ``uhf`` argument is ignored for the methyl radical.

    If a future xtb starts treating it properly this test fails, which is the
    signal to retire the refusal rather than to loosen the test.
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from xtb.interface import Calculator
    from xtb.libxtb import VERBOSITY_MUTED
    from xtb.utils import get_method

    mol = Chem.AddHs(Chem.MolFromSmiles("[CH3]"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    AllChem.EmbedMolecule(mol, params)
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    positions = np.asarray(mol.GetConformer().GetPositions()) / ANGSTROM_PER_BOHR

    energies = []
    with single_threaded():
        for uhf in (0, 1):
            calc = Calculator(get_method("GFN2-xTB"), numbers, positions, uhf=uhf)
            calc.set_verbosity(VERBOSITY_MUTED)
            energies.append(float(calc.singlepoint().get_energy()))

    assert energies[0] == pytest.approx(energies[1], abs=1e-9)


@available
def test_a_charge_declared_only_on_the_candidate_record_is_refused():
    """``MoleculeSpec.charge`` is part of the candidate, not decoration.

    A candidate may declare an ion whose SMILES does not spell one.  Screening
    the structure alone then hands back the neutral molecule's dipole under the
    ion's name, in domain and at full applicability, and nothing downstream can
    see it happen - while ``joback`` and ``quantum_electronic``, which both read
    the same field, refuse the identical candidate.
    """
    candidate = Candidate(
        material_class=MaterialClass.MOLECULE,
        molecule=MoleculeSpec(smiles="CC(=O)O", charge=-1),
    )
    prediction = EXPERT.predict(
        PredictionRequest(
            candidate=candidate,
            properties=frozenset({"dipole_moment"}),
            conditions=Conditions.standard(),
        )
    )[0]
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    assert "-1" in prediction.notes[0]
    # The neutral structure on its own would have answered, which is the point.
    assert predict("CC(=O)O").is_usable

    # A charge declared consistently with the SMILES is refused too, by the
    # origin-dependence argument rather than as a contradiction.
    consistent = Candidate(
        material_class=MaterialClass.MOLECULE,
        molecule=MoleculeSpec(smiles="CC(=O)[O-]", charge=-1),
    )
    other = EXPERT.predict(
        PredictionRequest(
            candidate=consistent,
            properties=frozenset({"dipole_moment"}),
            conditions=Conditions.standard(),
        )
    )[0]
    assert other.status is PredictionStatus.UNSUPPORTED
    assert "origin-dependent" in other.notes[0]


@available
def test_an_unassigned_double_bond_configuration_is_refused():
    """E and Z are different compounds, and ETKDG picks one without saying so."""
    prediction = predict("ClC=CCl")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    assert "stereogenic double bond" in prediction.notes[0]
    # The refusal has to tell a chemist what to do next.
    assert "/C=C/" in prediction.notes[0] or "Cl/C=C/Cl" in prediction.notes[0]

    for smiles in (r"Cl/C=C\Cl", "Cl/C=C/Cl", r"C/C=C\C#N"):
        assigned = predict(smiles)
        assert assigned.is_usable, smiles


@available
def test_the_double_bond_refusal_premise_holds():
    """Without this, the refusal above is an opinion.

    The two configurations differ by more than any bar this expert states, and
    the bare SMILES silently returns one of them with a zero conformer spread -
    an error bar asserting the answer is settled.
    """
    cis = debye(predict(r"Cl/C=C\Cl"))
    trans = debye(predict("Cl/C=C/Cl"))
    assert cis == pytest.approx(1.85, abs=0.05)
    assert trans == pytest.approx(0.0, abs=0.01)
    assert cis - trans > 2 * predict(r"Cl/C=C\Cl").uncertainty.std

    # ETKDG does not refuse the ambiguity, it resolves it - and for more than
    # one conformer it resolves it both ways in one ensemble.
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from formulate.experts.semiempirical import EMBED_SEED, PRUNE_RMS_THRESHOLD

    mol = Chem.AddHs(Chem.MolFromSmiles("ClC=CCl"))
    params = AllChem.ETKDGv3()
    params.randomSeed = EMBED_SEED
    params.pruneRmsThresh = PRUNE_RMS_THRESHOLD
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=8, params=params))
    chlorines = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 17]
    separations = [
        float(
            np.linalg.norm(
                mol.GetConformer(i).GetPositions()[chlorines[0]]
                - mol.GetConformer(i).GetPositions()[chlorines[1]]
            )
        )
        for i in ids
    ]
    assert min(separations) < 3.5 < max(separations), separations


@available
def test_an_allene_is_not_refused_over_an_axial_form():
    """RDKit flags a cumulated bond as potential stereo; |mu| cannot see it.

    An allene's two forms are enantiomers rather than E and Z, and RDKit flags
    methylallene, which has no stereoisomers at all.  Refusing on that would
    cost candidates over a difference of 0.0001 D.
    """
    assert screen("C=C=CC") is None
    assert predict("CC=C=CC").is_usable
    assert debye(predict(r"C/C=C=C/C")) == pytest.approx(
        debye(predict(r"C/C=C=C\C")), abs=1e-3
    )
    # A plain C=C in the same molecule is still caught.
    assert screen("C=C=CC=CC") is not None


@available
def test_unassigned_tetrahedral_centres_are_not_refused():
    """The line is drawn on a measurement, not on tidiness.

    |mu| is the same for two enantiomers, and where two centres make genuine
    diastereomers the difference is far inside the spread the ensemble already
    reports.  Refusing them would cost real candidates - three of the fifty
    bundled reference compounds carry one unassigned centre - for nothing.
    """
    assert predict("CC(O)CC").is_usable
    assert debye(predict(r"C[C@H](O)CC")) == pytest.approx(
        debye(predict(r"C[C@@H](O)CC")), abs=0.1
    )

    diol = predict("CC(O)C(C)O")
    assert diol.is_usable
    meso = debye(predict(r"C[C@H](O)[C@@H](C)O"))
    chiral = debye(predict(r"C[C@H](O)[C@H](C)O"))
    assert abs(meso - chiral) < diol.provenance.parameters["conformer_spread_debye"]


@available
def test_ion_pair_is_refused_although_its_net_charge_is_zero():
    prediction = predict("[Na+].[Cl-]")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "fragments" in prediction.notes[0]


@available
def test_unparseable_smiles_is_refused():
    prediction = predict("not a molecule at all @@@")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "parse" in prediction.notes[0]


def test_polymer_and_mixture_get_no_prediction_at_all():
    """No fallback to the repeat unit or the major component (rule 7)."""
    polymer = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(monomers=(MonomerUnit(smiles="[*]CC[*]"),)),
    )
    mixture = Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=(
                MixtureComponent(fraction=0.5, molecule=MoleculeSpec(smiles="O")),
                MixtureComponent(fraction=0.5, molecule=MoleculeSpec(smiles="CCO")),
            )
        ),
    )
    for candidate in (polymer, mixture):
        assert EXPERT.applicable_properties(
            {"dipole_moment"}, candidate.material_class
        ) == frozenset()
        request = PredictionRequest(
            candidate=candidate,
            properties=frozenset({"dipole_moment"}),
            conditions=Conditions.standard(),
        )
        assert EXPERT.predict(request) == []


@available
def test_a_backend_that_raises_becomes_a_failure_not_a_value(monkeypatch):
    import formulate.experts.semiempirical as module

    def broken(*args, **kwargs):
        raise RuntimeError("SCF blew up")

    monkeypatch.setattr(module, "run_ensemble", broken)
    prediction = predict("CCO")
    assert prediction.status is PredictionStatus.FAILED
    assert prediction.quantity is None
    assert "SCF blew up" in prediction.notes[0]


@available
def test_a_nonsense_geometry_is_a_failure_rather_than_a_number():
    """xtb does not raise for two overlapping carbons; it returns E = +939 Eh.

    ``run_ensemble`` rejects any non-negative total electronic energy, so the
    ensemble ends up empty and the caller reports a failure.
    """
    import numpy as np
    from xtb.interface import Calculator
    from xtb.libxtb import VERBOSITY_MUTED
    from xtb.utils import get_method

    with single_threaded():
        calc = Calculator(
            get_method("GFN2-xTB"),
            np.array([6, 6]),
            np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]) / ANGSTROM_PER_BOHR,
        )
        calc.set_verbosity(VERBOSITY_MUTED)
        energy = float(calc.singlepoint().get_energy())

    assert energy > 0.0, "the premise of the positive-energy guard no longer holds"


@available
def test_expert_is_unavailable_without_a_backend(monkeypatch):
    monkeypatch.setattr(SemiempiricalElectronicExpert, "is_available", lambda self: False)
    monkeypatch.setattr(
        SemiempiricalElectronicExpert, "unavailable_reason", lambda self: "no xtb here"
    )
    prediction = predict("CCO")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.notes[0] == "no xtb here"


# --------------------------------------------------------------------------
# Uncertainty behaviour
# --------------------------------------------------------------------------


@available
def test_every_prediction_carries_a_positive_spread_with_a_basis():
    for smiles in ("O", "c1ccccc1", "CC(C)=O", "ClCCCl"):
        prediction = predict(smiles)
        assert prediction.uncertainty.std is not None
        assert prediction.uncertainty.std >= DIPOLE_STD_FLOOR_DEBYE
        assert prediction.uncertainty.kind is UncertaintyKind.COMBINED
        assert "held-out" in prediction.uncertainty.basis


@available
def test_a_flexible_molecule_gets_a_wider_bar_than_a_rigid_one():
    """The conformer spread is earned per molecule, not assumed."""
    flexible = predict("ClCCCl")  # two gauche/anti conformers, dipoles cancel
    rigid = predict("Clc1ccccc1")  # same atoms' worth of chlorine, no torsion
    assert flexible.uncertainty.std > 2 * rigid.uncertainty.std


@available
@pytest.mark.parametrize("smiles,element", [("CSC", "sulfur"), ("CI", "iodine")])
def test_soft_elements_widen_the_bar_and_flag_the_domain(smiles, element):
    prediction = predict(smiles)
    spread = prediction.provenance.parameters["conformer_spread_debye"]
    plain = math.sqrt(
        DIPOLE_STD_FLOOR_DEBYE**2
        + (DIPOLE_STD_RELATIVE * debye(prediction)) ** 2
        + spread**2
    )
    assert prediction.uncertainty.std == pytest.approx(SOFT_ELEMENT_FACTOR * plain, rel=1e-6)
    warnings = " ".join(prediction.applicability.warnings)
    assert element in warnings, warnings


@available
def test_the_bias_direction_is_stated_not_corrected():
    """A ranker reading a pessimistic bound has to know which way the method leans."""
    prediction = predict("CC(C)=O")
    text = prediction.uncertainty.basis + " " + " ".join(prediction.notes)
    assert "overestimate" in text
    assert "+0.341" in text


# --------------------------------------------------------------------------
# Domain
# --------------------------------------------------------------------------


@available
def test_domain_is_real():
    inside = EXPERT.assess_domain(molecule_candidate("CCOCC"))
    assert inside.in_domain and inside.score == 1.0
    assert "gas-phase" in inside.basis

    nowhere = EXPERT.assess_domain(
        Candidate(
            material_class=MaterialClass.POLYMER,
            polymer=PolymerSpec(monomers=(MonomerUnit(smiles="[*]CC[*]"),)),
        )
    )
    assert not nowhere.in_domain

    unparseable = EXPERT.assess_domain(molecule_candidate("@@@nonsense"))
    assert not unparseable.in_domain


@available
@pytest.mark.parametrize(
    "smiles,fragment",
    [
        ("O=[N+]([O-])c1ccccc1", "nitro"),
        ("FC(F)(F)c1ccccc1", "perfluoroalkyl"),
        ("[W]", "outside the neutral-organic set"),
    ],
)
def test_the_known_failure_classes_are_flagged(smiles, fragment):
    domain = EXPERT.assess_domain(molecule_candidate(smiles))
    assert not domain.in_domain
    assert any(fragment in w for w in domain.warnings)


def test_the_overpolarised_patterns_compile():
    """Every flag SMARTS must parse, or the flag silently stops firing.

    ``chem.has_substructure`` returns False for a pattern RDKit cannot compile,
    so a broken pattern is indistinguishable from a molecule that does not
    contain the group.  Patterns are held as a tuple for the same reason: they
    were once one comma-joined string, and any future pattern with a comma
    inside brackets would have been split into two unparseable halves.
    """
    from rdkit import Chem

    from formulate.experts.semiempirical import _OVERPOLARISED_GROUPS

    for name, patterns, evidence in _OVERPOLARISED_GROUPS:
        assert isinstance(patterns, tuple), name
        for smarts in patterns:
            assert Chem.MolFromSmarts(smarts) is not None, (name, smarts)
        assert evidence


@available
def test_the_gradient_threshold_headroom_is_thin_on_the_organic_side():
    """Pins both ends of the 3 eV/A threshold, because one end is close.

    Formaldehyde at 2.43 and cyanogen at 2.64 are ordinary molecules that must
    pass; carbon dioxide at 13.5 is a force-field failure that must not.
    """
    for smiles, ceiling in (("C=O", 3.0), ("N#CC#N", 3.0), ("CC#N", 3.0)):
        assert run_ensemble(smiles).max_gradient < ceiling, smiles
    assert run_ensemble("C=O").max_gradient > 2.0
    assert run_ensemble("O=C=O").max_gradient > 2 * MAX_GRADIENT_EV_PER_ANGSTROM


@available
def test_the_geometry_guard_catches_a_bad_mmff_geometry_without_false_positives():
    """MMFF gives CO2 a 1.405 A C=O bond against a true 1.16 A, and says nothing.

    RDKit's own ``MMFFHasAllMoleculeParams`` returns True for it.  The GFN2
    gradient is what actually catches it.
    """
    bad = predict("O=C=O")
    assert not bad.applicability.in_domain
    assert any("gradient" in w for w in bad.applicability.warnings)

    for smiles in ("CCO", "c1ccccc1", "CC(C)=O", "Clc1ccccc1"):
        good = run_ensemble(smiles)
        assert good.max_gradient < MAX_GRADIENT_EV_PER_ANGSTROM, smiles


@available
def test_a_low_domain_score_loses_to_a_tight_in_domain_prediction():
    """This expert must not be able to out-rank a better one by inflated domain.

    ``prefer`` compares in_domain, then the applicability score, then the
    spread.  An out-of-domain tight-binding dipole loses to an in-domain one
    whatever its error bar says.
    """
    iodomethane = predict("CI")
    chlorobenzene = predict("Clc1ccccc1")
    assert not iodomethane.applicability.in_domain
    assert chlorobenzene.applicability.in_domain
    assert prefer(chlorobenzene, iodomethane)
    assert not prefer(iodomethane, chlorobenzene)


# --------------------------------------------------------------------------
# Protocol mechanics
# --------------------------------------------------------------------------


def test_conformer_budget_follows_flexibility():
    from rdkit import Chem

    assert torsion_count(Chem.MolFromSmiles("c1ccccc1")) == 0
    assert conformer_budget(Chem.MolFromSmiles("c1ccccc1")) == 1
    # RDKit's rotatable-bond count cannot see the two N-H rotors of
    # ethylenediamine, which is exactly where its dipole lives.
    assert torsion_count(Chem.MolFromSmiles("NCCN")) > 1
    assert conformer_budget(Chem.MolFromSmiles("NCCN")) > 8
    assert conformer_budget(Chem.MolFromSmiles("CCOCCOCCOCCOCCOCC")) <= 32


@available
def test_the_same_candidate_twice_gives_the_same_answer():
    """A content-addressed cache stores whatever comes back first.

    1,2-dichloroethane is the case a single-conformer protocol gets wrong: its
    answer there flips between 0.00 and 2.80 D with the embedding seed.
    """
    for smiles in ("ClCCCl", "CCO"):
        first = debye(predict(smiles))
        second = debye(predict(smiles))
        assert first == pytest.approx(second, abs=1e-9)


def _thread_counts() -> list[int]:
    from formulate.experts.semiempirical import _thread_handles

    out = []
    for handle in _thread_handles():
        for getter in ("omp_get_max_threads", "openblas_get_num_threads"):
            if hasattr(handle, getter):
                out.append(int(getattr(handle, getter)()))
    return out


@available
def test_thread_pinning_restores_the_prior_counts():
    """A real calculation runs inside the pin, and the prior is set by hand.

    Both halves are the test.  ``dlsym`` searches a library's dependencies, so
    once ``xtb.interface`` is imported the libxtb handle resolves the *same*
    ``omp_set_num_threads`` and ``openblas_set_num_threads`` as libgomp's and
    libopenblas's own handles.  A pin that read each prior just before writing
    it recorded 4, 4, 1, 1 and restored them in that order, leaving the whole
    interpreter at one thread for good - and a test that trusted whatever the
    counts happened to be on entry could not see it, because by then an earlier
    test had already leaked the pin and 1 restored to 1.
    """
    import ctypes

    from formulate.experts.semiempirical import _THREAD_ENTRY_POINTS, _thread_handles

    handles = _thread_handles()
    if not handles:
        pytest.skip("xtb's bundled libraries expose no thread-count entry points")
    assert isinstance(handles[0], ctypes.CDLL)

    original = _thread_counts()
    try:
        for handle in handles:
            for setter, getter in _THREAD_ENTRY_POINTS:
                if hasattr(handle, setter) and hasattr(handle, getter):
                    getattr(handle, setter)(2)
        known = _thread_counts()
        assert known and all(c == 2 for c in known), known

        with single_threaded() as pinned:
            assert pinned
            assert all(c == 1 for c in _thread_counts())
            run_ensemble("CCO")  # the leak only appeared once xtb did work
        assert _thread_counts() == known
    finally:
        for handle in handles:
            for setter, getter in _THREAD_ENTRY_POINTS:
                if hasattr(handle, setter) and hasattr(handle, getter):
                    getattr(handle, setter)(max(original) if original else 1)


def test_the_pin_skips_a_library_it_could_not_put_back(monkeypatch):
    """A setter without a getter has no prior, so it is left alone.

    Pinning one of those would be permanent: there is nothing to restore it to,
    and the cost lands on whatever else in the process uses those threads.
    """
    import formulate.experts.semiempirical as module

    class SetterOnly:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def omp_set_num_threads(self, value: int) -> None:  # no getter
            self.calls.append(value)

    stub = SetterOnly()
    monkeypatch.setattr(module, "_thread_handles", lambda: [stub])
    with single_threaded() as pinned:
        assert not pinned
    assert stub.calls == []


@available
def test_the_expert_still_answers_when_the_pin_cannot_be_applied(monkeypatch):
    """Pinning is a cost measure; losing it must not change the number."""
    import formulate.experts.semiempirical as module

    pinned = debye(predict("CCO"))
    monkeypatch.setattr(module, "_thread_handles", list)
    unpinned_prediction = predict("CCO")
    assert debye(unpinned_prediction) == pytest.approx(pinned, abs=1e-9)
    assert any("pinned to one thread" in n for n in unpinned_prediction.notes)


@available
def test_the_boltzmann_temperature_comes_from_the_request():
    hot = Conditions(temperature=Quantity(value=600.0, unit="K"))
    prediction = predict("ClCCCl", conditions=hot)
    assert prediction.provenance.parameters["boltzmann_temperature_K"] == pytest.approx(600.0)
    cold = predict("ClCCCl")
    assert cold.provenance.parameters["boltzmann_temperature_K"] == pytest.approx(298.15)
    # Flexible molecule: the ensemble average genuinely moves with temperature,
    # although the registry marks dipole_moment condition-independent.
    assert debye(prediction) != pytest.approx(debye(cold), abs=1e-6)


@available
def test_provenance_records_what_would_reproduce_the_number():
    prediction = predict("CCO")
    params = prediction.provenance.parameters
    assert params["embed_seed"] == 0xF00D
    assert params["conformers"] >= 1
    assert params["conformer_spread_debye"] >= 0.0
    assert "GFN2-xTB" in params["method"]
    assert prediction.provenance.software.backends["xtb"]
