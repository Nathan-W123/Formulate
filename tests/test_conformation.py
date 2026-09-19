"""The conformational-size expert.

Two methods under one property, and the tests are weighted accordingly: a
refusal here is a feature, so each one is pinned by name and by reason.  The
measured anchors are benzene and methane against exact gas-phase geometries,
and polystyrene and polyethylene against the SANS ``Rg/sqrt(M)`` coefficients.
"""

from __future__ import annotations

import math

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MixtureComponent,
    MixtureSpec,
    MoleculeSpec,
    MonomerRole,
    MonomerUnit,
    PolymerSpec,
    PolymerTopology,
    molecule_candidate,
)
from formulate.core.conditions import Conditions, Phase
from formulate.core.prediction import PredictionStatus
from formulate.core.properties import get_property
from formulate.core.quantity import Quantity
from formulate.experts.base import PredictionRequest
from formulate.experts.conformation import (
    ConformationExpert,
    EnsembleRefusal,
    build_ensemble,
    chain_dimension_spread,
    ensemble_size,
    gaussian_rg_angstrom,
    geometry_refusal,
    source_temperature,
    wormlike_ratio,
)

PROP = "radius_of_gyration"
PS_UNIT = "[*]CC(c1ccccc1)[*]"
PE_UNIT = "[*]CC[*]"
PBD_UNIT = "[*]CC=CC[*]"

ROOM = Conditions(temperature=Quantity(value=298.15, unit="K"))


@pytest.fixture(scope="module")
def expert() -> ConformationExpert:
    return ConformationExpert()


def predict(expert: ConformationExpert, candidate: Candidate, conditions: Conditions = ROOM):
    request = PredictionRequest(
        candidate=candidate, properties=frozenset({PROP}), conditions=conditions
    )
    out = expert.predict(request)
    return out[0] if out else None


def polymer(
    unit: str = PS_UNIT,
    mn: float | None = 100_000.0,
    *,
    topology: PolymerTopology = PolymerTopology.LINEAR,
    extra: tuple[MonomerUnit, ...] = (),
) -> Candidate:
    monomers = (MonomerUnit(smiles=unit, mole_fraction=1.0 if not extra else 0.5),) + extra
    return Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(
            monomers=monomers,
            topology=topology,
            number_average_molar_mass=None if mn is None else Quantity(value=mn, unit="g/mol"),
        ),
    )


# -- the contract ---------------------------------------------------------


def test_the_expert_declares_what_it_covers(expert):
    assert expert.id == "conformation"
    assert expert.supported_properties == frozenset({PROP})
    assert expert.supported_classes == frozenset(
        {MaterialClass.MOLECULE, MaterialClass.POLYMER}
    )
    assert expert.dependencies == frozenset()


@requires_rdkit
def test_a_mixture_is_not_answered_at_all(expert):
    """No cross-class answer: a mixture is not its major component."""
    mixture = Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=(MixtureComponent(fraction=1.0, molecule=MoleculeSpec(smiles="CCO")),)
        ),
    )
    request = PredictionRequest(
        candidate=mixture, properties=frozenset({PROP}), conditions=ROOM
    )
    assert expert.predict(request) == []


@requires_rdkit
def test_the_value_converts_to_the_registry_canonical_unit(expert):
    """The registry's canonical unit is the metre; the expert reports angstrom.

    This is the one test that would catch a factor of 1e10, which is exactly
    the error a structural property reported in angstrom is prone to.
    """
    definition = get_property(PROP)
    assert definition.canonical_unit == "meter"

    prediction = predict(expert, molecule_candidate("c1ccccc1"))
    assert prediction.quantity.unit == "angstrom"
    canonical = prediction.quantity.to_canonical()
    assert canonical.unit == "meter"
    assert canonical.value == pytest.approx(prediction.quantity.value * 1e-10, rel=1e-12)
    # benzene is one and a half angstrom, not one and a half metres
    assert 1e-10 < canonical.value < 2e-10

    low, high = definition.bounds
    assert low is not None and canonical.value > low
    assert high is None


@requires_rdkit
def test_the_prediction_carries_the_conditions_it_was_made_at(expert):
    hot = Conditions(temperature=Quantity(value=450.0, unit="K"))
    prediction = predict(expert, molecule_candidate("CCCCCCCC"), hot)
    assert prediction.conditions.temperature_k == pytest.approx(450.0)
    assert prediction.provenance.parameters["temperature_k"] == pytest.approx(450.0)


# -- the measured molecular anchors ---------------------------------------


@requires_rdkit
def test_benzene_reproduces_its_exact_gas_phase_radius_of_gyration(expert):
    """1.5026 A from r(C-C) = 1.3902 and r(C-H) = 1.0862 (NIST CCCBDB).

    Measured here: MMFF94 gives 1.5072, which is 0.31% high and inside the
    2.2% geometry floor the expert claims.
    """
    reference = 1.50265
    prediction = predict(expert, molecule_candidate("c1ccccc1"))
    assert prediction.status is PredictionStatus.OK
    assert prediction.quantity.value == pytest.approx(1.5072, abs=5e-4)
    assert abs(prediction.quantity.value - reference) / reference < 0.022


@requires_rdkit
def test_methane_reproduces_its_exact_gas_phase_radius_of_gyration(expert):
    """0.5449 A from r(C-H) = 1.087 A; MMFF94 measures 0.5475, 0.48% high."""
    reference = 0.54494
    prediction = predict(expert, molecule_candidate("C"))
    assert prediction.quantity.value == pytest.approx(0.5475, abs=5e-4)
    assert abs(prediction.quantity.value - reference) / reference < 0.022


@requires_rdkit
def test_the_answer_moves_with_temperature_because_the_registry_says_it_does(expert):
    """Measured: octane's Boltzmann average spans 3.0% between 200 and 600 K."""
    cold = predict(expert, molecule_candidate("CCCCCCCC"),
                   Conditions(temperature=Quantity(value=200.0, unit="K")))
    hot = predict(expert, molecule_candidate("CCCCCCCC"),
                  Conditions(temperature=Quantity(value=600.0, unit="K")))
    assert cold.quantity.value > hot.quantity.value
    span = (cold.quantity.value - hot.quantity.value) / cold.quantity.value
    assert span == pytest.approx(0.0295, abs=0.006)


@requires_rdkit
def test_the_boltzmann_average_is_not_the_ensemble_mean(expert):
    """Measured: octane's weighted average is 4.3% above the unweighted one.

    If this ever passes trivially, the weights have stopped being applied.
    """
    ens = build_ensemble("CCCCCCCC", ensemble_size(5))
    weighted = ens.boltzmann(298.15)[0]
    unweighted = sum(ens.radii) / len(ens.radii)
    assert weighted > unweighted
    assert (weighted - unweighted) / unweighted == pytest.approx(0.0435, abs=0.01)


@requires_rdkit
def test_two_identical_requests_agree_exactly(expert):
    """A fixed ETKDG seed, or the ranking would reorder between rounds."""
    first = predict(expert, molecule_candidate("OCC(O)CO"))
    second = predict(expert, molecule_candidate("OCC(O)CO"))
    assert first.quantity.value == second.quantity.value
    assert first.uncertainty.std == second.uncertainty.std


# -- the molecular refusals -----------------------------------------------


@requires_rdkit
@pytest.mark.parametrize("smiles", ["O=C=O", "S=C=S", "CN=C=O", "N=C=N", "C=C=C"])
def test_a_cumulated_double_bond_is_refused_by_name(expert, smiles):
    """MMFF94 answers for these and is 21% wrong on CO2 against experiment.

    ``MMFFHasAllMoleculeParams`` returns True for every one of them, so this
    guard is the only thing standing between the ranking and a confident
    twenty-per-cent error that no amount of sampling would reduce.
    """
    prediction = predict(expert, molecule_candidate(smiles))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    assert "cumulated double bond" in prediction.notes[0]


@requires_rdkit
@pytest.mark.parametrize(
    "smiles", ["CCCCCC", "c1ccccc1", "CC#N", "C=CC=C", "O=S(=O)(C)C", "CC1COC(=O)O1"]
)
def test_the_cumulene_guard_does_not_fire_on_ordinary_molecules(expert, smiles):
    prediction = predict(expert, molecule_candidate(smiles))
    assert prediction.status is PredictionStatus.OK


@requires_rdkit
def test_a_molecule_mmff_cannot_type_is_refused_rather_than_guessed(expert):
    """Trimethyl borate: no MMFF94 boron parameters, and no UFF fallback.

    UFF is a different method with a different error; substituting it under
    this expert's `method` string would be exactly the silent fallback the
    repository forbids.
    """
    prediction = predict(expert, molecule_candidate("B(OC)(OC)OC"))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    assert "MMFF94 has no parameters" in prediction.notes[0]


@requires_rdkit
def test_the_mmff_failure_sentinel_is_never_consumed_as_an_energy():
    """``MMFFOptimizeMoleculeConfs`` returns (-1, -1.0) instead of raising.

    An energy of -1.0 for every conformer makes every Boltzmann weight equal
    and yields a plausible *unweighted* number from a calculation that did not
    happen.  Whichever guard fires first, the answer must be a refusal.
    """
    for smiles in ("B(OC)(OC)OC", "C[Sn](C)(C)C", "[Fe]"):
        with pytest.raises(EnsembleRefusal):
            build_ensemble(smiles, 10)


@requires_rdkit
def test_a_disconnected_smiles_is_refused(expert):
    """Measured: ETKDG embeds '[Na+].[Cl-]' with both ions at the origin.

    The mass-weighted radius comes back as exactly 0.0, and 'O.O' returns the
    radius of one water.  Both sit inside the registry's (0, None) bounds, so
    nothing downstream would catch them.
    """
    for smiles in ("[Na+].[Cl-]", "O.O", "CCO.O"):
        prediction = predict(expert, molecule_candidate(smiles))
        assert prediction.status is PredictionStatus.UNSUPPORTED
        assert "disconnected fragment" in prediction.notes[0]


@requires_rdkit
def test_too_many_rotatable_bonds_is_refused_rather_than_answered_widely(expert):
    """Convergence was measured to seventeen; beyond twenty there is no basis."""
    prediction = predict(expert, molecule_candidate("C" * 24))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "rotatable bonds" in prediction.notes[0]


@requires_rdkit
def test_an_ion_is_marked_out_of_domain_rather_than_valued_as_a_neutral(expert):
    prediction = predict(expert, molecule_candidate("CC[N+](C)(C)C"))
    assert prediction.status is PredictionStatus.OUT_OF_DOMAIN
    assert not prediction.applicability.in_domain
    assert any("formal charge" in w for w in prediction.applicability.warnings)


@requires_rdkit
def test_no_temperature_is_a_refusal_for_both_classes(expert):
    """The Boltzmann weights need one, and dodecane moves 5.2% over 400 K."""
    for candidate in (molecule_candidate("CCO"), polymer()):
        prediction = predict(expert, candidate, Conditions())
        assert prediction.status is PredictionStatus.UNSUPPORTED
        assert "no temperature" in prediction.notes[0]


# -- the uncertainty claims -----------------------------------------------


@requires_rdkit
@pytest.mark.parametrize("smiles", ["c1ccccc1", "C", "CCCCCCCC", "OCC(O)CO", "CC(C)=O"])
def test_no_prediction_ever_carries_a_zero_or_absent_error_bar(expert, smiles):
    prediction = predict(expert, molecule_candidate(smiles))
    assert prediction.uncertainty.std is not None
    assert prediction.uncertainty.std > 0.0
    assert prediction.uncertainty.basis


@requires_rdkit
def test_a_rigid_molecule_falls_back_to_the_measured_geometry_and_energy_floors(expert):
    """Benzene's ensemble collapses to one minimum, so sampling is exactly zero.

    What is left is the quadrature of the 2.2% geometry floor and the 2.6%
    donor-free energy term: 3.4%.  A zero bar here would be the model claiming
    MMFF94's geometry is exact.
    """
    prediction = predict(expert, molecule_candidate("c1ccccc1"))
    relative = prediction.uncertainty.std / prediction.quantity.value
    assert relative == pytest.approx(math.hypot(0.022, 0.026), rel=1e-6)


@requires_rdkit
def test_the_sampling_term_is_computed_on_distinct_minima_not_embeddings(expert):
    """Measured: 300 embeddings of octane collapse onto 48 distinct minima.

    Counting the duplicates would claim 0.5% where the deduplicated standard
    error says 2.4%.  A reported relative bar under 1% is what that mistake
    looks like from the outside.
    """
    ens = build_ensemble("CCCCCCCC", ensemble_size(5))
    assert len(ens.radii) < ens.embedded / 2

    prediction = predict(expert, molecule_candidate("CCCCCCCC"))
    relative = prediction.uncertainty.std / prediction.quantity.value
    assert relative > 0.01


@requires_rdkit
def test_a_polyol_carries_a_wider_bar_than_an_alkane(expert):
    """The donor-keyed energy term, measured by GFN2-xTB reweighting.

    Glycerol's bar must exceed octane's even though octane has the larger
    conformational spread, because MMFF94's error on a polyol is in the
    *weights* - 15.7% worst measured at two or more donors - not in the
    geometries.  If the table is ever dropped this test fails.
    """
    alkane = predict(expert, molecule_candidate("CCCCCCCC"))
    polyol = predict(expert, molecule_candidate("OCC(O)CO"))
    alkane_relative = alkane.uncertainty.std / alkane.quantity.value
    polyol_relative = polyol.uncertainty.std / polyol.quantity.value
    assert polyol_relative > alkane_relative
    assert polyol_relative > 0.15


@requires_rdkit
def test_a_very_flexible_molecule_says_its_sampling_term_was_widened(expert):
    """Hexadecane: a 2.0% computed bar against a 2.5% five-seed scatter.

    The split-half term can only see minima that were found, so above twelve
    rotatable bonds it is widened by the measured ratio and the prediction says
    so rather than quietly carrying the narrow number.
    """
    prediction = predict(expert, molecule_candidate("C" * 16))
    assert prediction.status is PredictionStatus.OK
    assert any("rotatable bonds" in note for note in prediction.notes)
    assert "widened" in prediction.uncertainty.basis
    relative = prediction.uncertainty.std / prediction.quantity.value
    assert relative > 0.03


@requires_rdkit
def test_the_gas_phase_assumption_is_on_every_molecular_prediction(expert):
    prediction = predict(expert, molecule_candidate("OCC(O)CO"))
    assert any("gas-phase ensemble" in note for note in prediction.notes)
    assert any("free-energy weights" in note for note in prediction.notes)


@requires_rdkit
def test_a_condensed_phase_request_is_told_the_ensemble_is_in_vacuum(expert):
    liquid = Conditions(
        temperature=Quantity(value=298.15, unit="K"), phase=Phase.LIQUID
    )
    prediction = predict(expert, molecule_candidate("CCCCCCCC"), liquid)
    assert any("in vacuum" in note for note in prediction.notes)


# -- the ideal chain ------------------------------------------------------


def test_the_ideal_chain_reproduces_the_sans_coefficients():
    """``Rg/sqrt(M)`` from the tabulated ``<R^2>/M`` against scattering.

    Semi-circular by construction: the tabulated ``<R^2>/M`` and the quoted
    coefficients both come from the Fetters compilation, so this checks the
    algebra rather than the physics.  The non-circular number is the
    correlation spread pinned below.
    """
    from formulate.experts.mechanical import CHAIN_DIMENSIONS

    sans = {PE_UNIT: 0.460, PS_UNIT: 0.275, PBD_UNIT: 0.380,
            "[*]CC(C)(C(=O)OC)[*]": 0.270, "[*][Si](C)(C)O[*]": 0.270}
    for unit, coefficient in sans.items():
        model = gaussian_rg_angstrom(CHAIN_DIMENSIONS[unit].r2_per_mass, 1.0)
        assert model == pytest.approx(coefficient, rel=0.02)


@requires_rdkit
def test_polystyrene_and_polyethylene_land_on_their_scattering_sizes(expert):
    """Measured: 85.3 A against 0.275 sqrt(M) = 87.0, and 144.3 against 145.5."""
    ps = predict(expert, polymer(PS_UNIT, 100_000.0))
    pe = predict(expert, polymer(PE_UNIT, 100_000.0))
    assert ps.quantity.value == pytest.approx(85.34, abs=0.05)
    assert pe.quantity.value == pytest.approx(144.34, abs=0.05)
    assert abs(ps.quantity.value - 0.275 * math.sqrt(1e5)) / (0.275 * math.sqrt(1e5)) < 0.02
    assert abs(pe.quantity.value - 0.460 * math.sqrt(1e5)) / (0.460 * math.sqrt(1e5)) < 0.02


@requires_rdkit
def test_the_chain_radius_scales_as_the_square_root_of_chain_length(expert):
    small = predict(expert, polymer(PS_UNIT, 25_000.0))
    large = predict(expert, polymer(PS_UNIT, 100_000.0))
    assert large.quantity.value / small.quantity.value == pytest.approx(2.0, rel=1e-9)


def test_the_correlation_spread_is_computed_from_the_table_it_describes():
    """1.054 held out over four polymers, 1.170 over all ten.

    The expert reports the second, because the worst miss is polyethylene -
    one of the correlation's own *fit* polymers - and four held-out points
    cannot exclude a miss the method demonstrably makes.  Pinning both here
    stops the constant in the uncertainty basis drifting away from the table.
    """
    held_out, overall = chain_dimension_spread()
    assert held_out == pytest.approx(1.054, abs=0.002)
    assert overall == pytest.approx(1.170, abs=0.002)
    assert math.sqrt(overall) - 1.0 == pytest.approx(0.082, abs=0.002)


def test_the_wormlike_correction_is_negligible_for_a_chain_and_decisive_for_an_oligomer():
    """Benoit & Doty (1953), at the Kuhn counts polystyrene actually reaches."""
    assert 1.0 - wormlike_ratio(131.8) == pytest.approx(0.0057, abs=0.0005)
    assert 1.0 - wormlike_ratio(13.2) == pytest.approx(0.054, abs=0.002)
    assert 1.0 - wormlike_ratio(2.64) == pytest.approx(0.221, abs=0.005)


def test_the_reference_temperature_is_read_from_the_table_prose():
    assert source_temperature("polyethylene, melt at 413 K") == 413.0
    assert source_temperature("PDMS at 298 K") == 298.0
    assert source_temperature("atactic polypropylene") is None


@requires_rdkit
def test_an_oligomer_is_out_of_domain_and_says_how_far_from_gaussian_it_is(expert):
    """Polystyrene at M_n 2000 is 2.6 Kuhn segments; the formula overstates it 22%."""
    prediction = predict(expert, polymer(PS_UNIT, 2_000.0))
    assert prediction.status is PredictionStatus.OUT_OF_DOMAIN
    assert not prediction.applicability.in_domain
    assert any("Kuhn segments" in w for w in prediction.applicability.warnings)
    assert "Kuhn segments" in prediction.uncertainty.basis
    assert prediction.uncertainty.std / prediction.quantity.value > 0.2


@requires_rdkit
def test_a_predicted_chain_dimension_carries_the_wider_bar_and_says_so(expert):
    """PVC is not tabulated, so it goes through the side-group correlation."""
    tabulated = predict(expert, polymer(PS_UNIT, 50_000.0))
    predicted = predict(expert, polymer("[*]CC(Cl)[*]", 50_000.0))
    assert predicted.provenance.parameters["chain_dimension_split"] == "predicted"
    assert tabulated.provenance.parameters["chain_dimension_split"] == "fit"
    relative = predicted.uncertainty.std / predicted.quantity.value
    assert relative > tabulated.uncertainty.std / tabulated.quantity.value
    assert relative > 0.08
    assert any("predicted from side-group bulk" in w for w in predicted.applicability.warnings)


@requires_rdkit
def test_the_polymer_notes_say_what_this_dimension_is_not(expert):
    prediction = predict(expert, polymer(PS_UNIT, 100_000.0))
    joined = " ".join(prediction.notes)
    assert "unperturbed" in joined
    assert "good solvent" in joined
    assert "z-average" in joined
    assert "tacticity" in joined.lower()


@requires_rdkit
def test_the_temperature_term_vanishes_at_the_temperature_the_table_was_measured_at(expert):
    """Polystyrene's chain dimension is a 413 K melt value."""
    at_reference = predict(
        expert, polymer(PS_UNIT, 100_000.0),
        Conditions(temperature=Quantity(value=413.0, unit="K")),
    )
    extrapolated = predict(expert, polymer(PS_UNIT, 100_000.0))
    assert "Temperature 0.0%" in at_reference.uncertainty.basis
    assert at_reference.uncertainty.std < extrapolated.uncertainty.std


# -- the polymer refusals -------------------------------------------------


@requires_rdkit
def test_a_chain_of_unstated_length_is_refused(expert):
    """The same repeat unit spans 12 A to 85 A across the useful range."""
    prediction = predict(expert, polymer(PS_UNIT, None))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "number-average molar mass" in prediction.notes[0]


@requires_rdkit
@pytest.mark.parametrize(
    "topology",
    [PolymerTopology.NETWORK, PolymerTopology.STAR, PolymerTopology.BRANCHED,
     PolymerTopology.GRAFT, PolymerTopology.DENDRITIC],
)
def test_a_non_linear_architecture_is_refused_with_its_own_reason(expert, topology):
    prediction = predict(expert, polymer(PS_UNIT, 100_000.0, topology=topology))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert topology.value in prediction.notes[0]
    assert "branching factor" in prediction.notes[0] or "no finite chain" in prediction.notes[0]


@requires_rdkit
def test_a_copolymer_is_refused_rather_than_mole_averaged(expert):
    """``<R^2>/M`` of a random copolymer is not the mean of its homopolymers'."""
    prediction = predict(
        expert,
        polymer(PS_UNIT, 100_000.0, extra=(MonomerUnit(smiles=PE_UNIT, mole_fraction=0.5),)),
    )
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "copolymer" in prediction.notes[0]


@requires_rdkit
def test_an_end_group_does_not_count_as_a_second_backbone_unit(expert):
    """Only backbone monomers decide whether this is a homopolymer."""
    candidate = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(
            monomers=(
                MonomerUnit(smiles=PS_UNIT, mole_fraction=1.0),
                MonomerUnit(smiles="C", mole_fraction=0.0, role=MonomerRole.END_GROUP),
            ),
            number_average_molar_mass=Quantity(value=100_000.0, unit="g/mol"),
        ),
    )
    prediction = predict(expert, candidate)
    assert prediction.status is PredictionStatus.OK


@requires_rdkit
def test_a_solution_request_is_refused_because_this_is_the_unperturbed_coil(expert):
    solution = Conditions(
        temperature=Quantity(value=298.15, unit="K"), phase=Phase.SOLUTION
    )
    prediction = predict(expert, polymer(PS_UNIT, 100_000.0), solution)
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "unperturbed" in prediction.notes[0]


@requires_rdkit
def test_a_repeat_unit_with_no_attachment_points_is_refused(expert):
    """No backbone path, so no chain dimension - not a conformer search instead."""
    prediction = predict(expert, polymer("CCc1ccccc1", 100_000.0))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "chain dimension" in prediction.notes[0]


# -- no silent crossing between the two methods ---------------------------


@requires_rdkit
def test_a_polymer_is_never_answered_by_embedding_its_repeat_unit(expert):
    """The two branches leave different fingerprints in provenance.

    Polystyrene at M_n 10000 is a 27 A coil; ethylbenzene - the same repeat
    unit capped with hydrogen - is a 2.8 A molecule.  An order of magnitude
    apart, and neither prediction carries the other's parameters.
    """
    chain = predict(expert, polymer(PS_UNIT, 10_000.0))
    monomer = predict(expert, molecule_candidate("CCc1ccccc1"))

    assert chain.quantity.value / monomer.quantity.value > 8.0

    chain_params = chain.provenance.parameters
    monomer_params = monomer.provenance.parameters
    assert "r2_per_mass" in chain_params and "ensemble_size" not in chain_params
    assert "ensemble_size" in monomer_params and "r2_per_mass" not in monomer_params


@requires_rdkit
def test_a_repeat_unit_offered_as_a_molecule_is_not_quietly_uncapped(expert):
    """'[*]CC(c1ccccc1)[*]' as a MOLECULE is a molecule with two dummy atoms.

    It is refused, rather than having its attachment points stripped and being
    answered as ethylbenzene, or having them read as a chain.
    """
    prediction = predict(expert, molecule_candidate(PS_UNIT))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None


# -- small helpers --------------------------------------------------------


def test_the_ensemble_size_rule_is_the_published_one():
    """Ebejer, Morris & Deane (2012), keyed on rotatable bond count."""
    assert [ensemble_size(n) for n in (0, 7, 8, 12, 13, 20)] == [50, 50, 200, 200, 300, 300]


def test_the_ideal_chain_formula_is_the_square_root_of_r2_over_six():
    assert gaussian_rg_angstrom(6.0, 1.0) == pytest.approx(1.0)
    assert gaussian_rg_angstrom(0.437, 100_000.0) == pytest.approx(
        math.sqrt(0.437 * 100_000.0 / 6.0)
    )


# -- geometry classes MMFF94 answers for and gets badly wrong --------------
#
# Added in review.  Each one was measured against GFN2-xTB, and where an exact
# gas-phase geometry exists, against that too.  ``MMFFHasAllMoleculeParams``
# returns True for every molecule here, so the structural guard is the only
# thing between the ranking and a confidently wrong number.


@requires_rdkit
def test_dinitrogen_is_refused_rather_than_answered_thirty_per_cent_high(expert):
    """N#N is the worst geometry in this expert: MMFF94 puts the bond at 1.46 A.

    The experimental bond length is 1.0977 A (NIST CCCBDB), so the exact
    mass-weighted radius is 0.5489 A.  MMFF94 returns 0.7300 - 33% high, and
    before this guard it came back in-domain with a 3.4% error bar.
    """
    exact = 1.0977 / 2.0
    prediction = predict(expert, molecule_candidate("N#N"))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    assert "triple bond" in prediction.notes[0]
    assert "1.0977" in prediction.notes[0]
    # what it would have said, so the test fails if the guard is loosened
    raw = build_ensemble("N#N", 10).boltzmann(298.15)[0]
    assert abs(raw - exact) / exact > 0.3


@requires_rdkit
def test_sulfur_dioxide_is_refused_like_the_carbon_cumulenes(expert):
    """The cumulene guard was keyed on carbon; SO2 is the same motif at sulfur.

    r(S-O) = 1.4308 A at 119.3 degrees gives an exact 0.9442 A.  MMFF94 gives
    1.0788, 14.3% high, and GFN2-xTB agrees with experiment to 0.4%.
    """
    prediction = predict(expert, molecule_candidate("O=S=O"))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "cumulated double bond" in prediction.notes[0]
    assert abs(build_ensemble("O=S=O", 10).boltzmann(298.15)[0] - 0.9442) / 0.9442 > 0.1


@requires_rdkit
@pytest.mark.parametrize("smiles", ["N#CC#N", "C#CC#C", "N#CC#CC#N", "C#CC#N"])
def test_two_triple_bonded_atoms_joined_to_each_other_are_refused(expert, smiles):
    """Measured 5.3% to 6.4% high against GFN2 across the whole class."""
    prediction = predict(expert, molecule_candidate(smiles))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "triple-bonded" in prediction.notes[0]


@requires_rdkit
@pytest.mark.parametrize("smiles", ["[CH3]", "O=[N+][O-]", "c1ccccc1[CH2]"])
def test_an_open_shell_species_is_refused(expert, smiles):
    """MMFF94 is a closed-shell parameterisation and RDKit types radicals anyway.

    The methyl radical came back OK at 0.4855 A before this guard; NO2 came
    back OK and 10.3% above GFN2.
    """
    prediction = predict(expert, molecule_candidate(smiles))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "open-shell" in prediction.notes[0]


@requires_rdkit
@pytest.mark.parametrize(
    "smiles",
    ["CS(C)=O", "CS(=O)(=O)C", "C[N+](=O)[O-]", "OS(=O)(=O)O", "COP(=O)(OC)OC",
     "CC#N", "C#Cc1ccccc1", "CC#CC", "OCC#C", "N#CCC#N", "CCOCC"],
)
def test_the_new_guards_do_not_fire_on_ordinary_chemistry(expert, smiles):
    """Each of these was measured within 2.8% of GFN2 and must still be answered.

    A guard that refuses dimethyl sulfoxide or acetonitrile would cost more
    coverage than the classes above cost accuracy.
    """
    assert geometry_refusal(smiles) is None
    prediction = predict(expert, molecule_candidate(smiles))
    assert prediction.quantity is not None


@requires_rdkit
def test_no_bundled_reference_compound_is_caught_by_a_geometry_guard():
    """The guards are structural, so this is the cheapest test of over-reach."""
    import json
    import pathlib

    import formulate

    path = pathlib.Path(formulate.__file__).parent / "data" / "reference_compounds.json"
    compounds = json.loads(path.read_text())["compounds"]
    assert len(compounds) == 50
    refused = [c["name"] for c in compounds if geometry_refusal(c["smiles"]) is not None]
    assert refused == []


# -- the sampling bar against a converged ensemble -------------------------


@requires_rdkit
def test_the_sampling_bar_covers_the_distance_to_a_converged_ensemble(expert):
    """Octane at its production 50 conformers is 2.4% from the same answer at 1000.

    The term the ensemble can compute said 1.25% before the widening, which is
    the failure this repository calls critical: the ranker ranks on that bar.
    The widened term must cover the distance, and the total bar must cover it
    comfortably.
    """
    prediction = predict(expert, molecule_candidate("CCCCCCCC"))
    converged = build_ensemble("CCCCCCCC", 1000).boltzmann(298.15)[0]
    distance = abs(prediction.quantity.value - converged) / converged
    assert distance == pytest.approx(0.024, abs=0.004)

    sampling = float(
        prediction.uncertainty.basis.split("Sampling ")[1].split("%")[0]
    ) / 100.0
    assert sampling > distance
    assert prediction.uncertainty.std / prediction.quantity.value > distance


@requires_rdkit
def test_a_molecule_with_no_rotatable_bond_is_not_widened(expert):
    """Measured: every rigid molecule sits at 0.00% from a 1000-conformer answer.

    Cyclohexane, cyclooctane, decalin and propylene carbonate included - the
    ensemble does sample ring pucker and does converge on it - so doubling a
    zero term changes nothing, and the rigid bar stays where it was.
    """
    for smiles in ("c1ccccc1", "C1CCCCC1", "CC1COC(=O)O1"):
        prediction = predict(expert, molecule_candidate(smiles))
        # the claim being tested: a rigid molecule is already converged, so
        # whatever the widening does to its term it cannot be hiding an error
        assert build_ensemble(smiles, 1000).boltzmann(298.15)[0] == pytest.approx(
            prediction.quantity.value, rel=2e-4
        )
        assert prediction.uncertainty.std / prediction.quantity.value >= math.hypot(
            0.022, 0.026
        )
    # benzene and cyclohexane collapse to one weighted minimum, so the bar is
    # exactly the two measured floors.  Propylene carbonate does not: its ring
    # pucker gives several minima at comparable energies and a real sampling
    # term, which is conservative rather than wrong, because it converged.
    for smiles in ("c1ccccc1", "C1CCCCC1"):
        prediction = predict(expert, molecule_candidate(smiles))
        relative = prediction.uncertainty.std / prediction.quantity.value
        assert relative == pytest.approx(math.hypot(0.022, 0.026), rel=1e-3)


# -- refusals added in review ---------------------------------------------


@requires_rdkit
def test_a_non_positive_temperature_is_refused_rather_than_returning_a_nan(expert):
    """At 0 K the Boltzmann exponent divides by zero and numpy returns nan.

    That used to surface as "the ensemble produced a non-positive radius of
    gyration", which describes a different failure.
    """
    for candidate in (molecule_candidate("CCO"), polymer()):
        prediction = predict(
            expert, candidate, Conditions(temperature=Quantity(value=0.0, unit="K"))
        )
        assert prediction.status is PredictionStatus.UNSUPPORTED
        assert "0 K was requested" in prediction.notes[0]


@requires_rdkit
def test_a_chain_lighter_than_one_repeat_unit_is_refused(expert):
    """M_n 50 for polystyrene is half a monomer, and used to return 1.9 A.

    The ideal-chain formula has no lower limit of its own: it returned a
    number with an 82% error bar where the honest answer is that there is no
    chain.
    """
    prediction = predict(expert, polymer(PS_UNIT, 50.0))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "repeat unit" in prediction.notes[0]
    # one repeat unit exactly is a monomer, two is answered with a wide bar
    assert predict(expert, polymer(PS_UNIT, 104.15)).status is PredictionStatus.UNSUPPORTED
    assert predict(expert, polymer(PS_UNIT, 2_000.0)).quantity is not None


@requires_rdkit
def test_the_temperature_term_uses_the_top_of_the_range_its_basis_quotes(expert):
    """0.5 x 1.1e-3 per K over the 114.85 K from polystyrene's 413 K melt entry."""
    prediction = predict(expert, polymer(PS_UNIT, 100_000.0))
    expected = 0.5 * 1.1e-3 * abs(298.15 - 413.0)
    assert f"Temperature {expected:.1%}" in prediction.uncertainty.basis
    assert "1.1e-03 per K" in prediction.uncertainty.basis


def test_the_chain_dimension_is_read_through_the_mechanical_expert():
    """The coupling is deliberate; this is what makes a rename fail loudly.

    Both experts must answer with the same ``<R^2>/M`` or the coil size and the
    entanglement mass would describe different chains.
    """
    from formulate.experts.mechanical import PolymerMechanicalExpert

    assert hasattr(PolymerMechanicalExpert, "_chain")
    candidate = polymer(PS_UNIT, 100_000.0)
    assert (
        ConformationExpert()._chain_dimension(candidate)
        is PolymerMechanicalExpert()._chain(candidate)
    )


@requires_rdkit
def test_a_glyme_carries_the_gauche_term_although_it_has_no_donor(expert):
    """MMFF94 has no hyperconjugative term, and a glyme has no donor to key on.

    Measured by reweighting the same ensemble with GFN2-xTB single points:
    1,2-dimethoxyethane moves 8.1%, 1,2-diethoxyethane 8.4%, diglyme 5.3%,
    triglyme 12.0%, tetraglyme 10.7%, 1,2-difluoroethane 5.1% - all against
    the 2.6% the donor-free entry claims.  Getting this wrong means a glyme,
    which is an ordinary solvent, is ranked on a bar four times too narrow.
    """
    glyme = predict(expert, molecule_candidate("COCCOC"))
    ether = predict(expert, molecule_candidate("CCOCC"))
    assert glyme.uncertainty.std / glyme.quantity.value > 0.12
    assert ether.uncertainty.std / ether.quantity.value < 0.05
    assert "gauche" in glyme.uncertainty.basis
    assert any("gauche" in note for note in glyme.notes)
    assert "gauche" not in ether.uncertainty.basis


@requires_rdkit
def test_the_gauche_term_needs_the_bond_it_acts_on_to_be_able_to_rotate(expert):
    """Ethylene carbonate has the same O-C-C-O locked in a ring: 0.00% measured.

    1,3-difluoropropane (0.99%) and dimethoxymethane (0.17%) are not vicinal.
    A term that fired on those would be a wide bar with nothing behind it,
    which this repository treats as its own kind of dishonesty.
    """
    for smiles in ("O=C1OCCO1", "FCCCF", "COCOC", "OCCCCO"):
        prediction = predict(expert, molecule_candidate(smiles))
        assert "gauche" not in prediction.uncertainty.basis
