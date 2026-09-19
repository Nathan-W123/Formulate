"""Tests for the chain-scoped structural descriptor expert.

The method is exact rather than fitted, so validation here is exactness against
independently computed references, not error statistics.  The centrepiece is
that increments measured on DP 3-4-5 oligomers reproduce RDKit evaluated
directly on a built 50-mer.

Refusals are tested as hard as the predictions: in this repository a refusal is
a feature, and a silent mid-range answer is the failure mode that matters.
"""

from __future__ import annotations

import math

import pytest

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MonomerRole,
    MonomerUnit,
    PolymerSpec,
    PolymerTopology,
    Tacticity,
    polymer_candidate,
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import PredictionStatus
from formulate.core.quantity import Quantity
from formulate.experts.base import PredictionRequest

rdkit = pytest.importorskip("rdkit")

from formulate.experts.polymer import link_repeat_units, repeat_unit_mass  # noqa: E402
from formulate.experts.polymer_structural import (  # noqa: E402
    _AROMATIC,
    _END_GROUP_PANEL,
    _HEAVY,
    _MASS,
    _ROTATABLE,
    _TPSA,
    _UNSTATED_END_GROUP_MASS_SPAN,
    PolymerStructuralExpert,
    analyse_chain,
    end_group_alternatives,
    junction_mismatch,
    link_sequence,
    measure,
    unit_increment,
)

# -- reference repeat units ------------------------------------------------

PE = "[*]CC[*]"
PP = "[*]CC([*])C"
PS = "[*]CC([*])c1ccccc1"
PEO = "[*]CCO[*]"
PET = "[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]"
NYLON6 = "[*]CCCCCC(=O)N[*]"
PMMA = "[*]CC([*])(C)C(=O)OC"
PC = "[*]Oc1ccc(cc1)C(C)(C)c1ccc(cc1)OC(=O)[*]"
PDMS = "[*][Si](C)(C)O[*]"
PIB = "[*]CC([*])(C)C"
GEM = "[*]C([*])(C)C"  # both attachment points on one backbone atom

REFERENCE = (PE, PP, PS, PEO, PET, NYLON6, PMMA, PC, PDMS, PIB)

#: The 21 units the module docstring quotes its 105/105 over.  They live here,
#: not in a scratch script, so that claim is re-runnable rather than asserted.
REFERENCE_UNITS: dict[str, str] = {
    "PE": PE,
    "PP": PP,
    "PS": PS,
    "PEO": PEO,
    "PET": PET,
    "PBT": "[*]OCCCCOC(=O)c1ccc(cc1)C(=O)[*]",
    "nylon-6": NYLON6,
    "PMMA": PMMA,
    "polycarbonate": PC,
    "PDMS": PDMS,
    "PIB": PIB,
    "PVC": "[*]CC([*])Cl",
    "PAN": "[*]CC([*])C#N",
    "PVA": "[*]CC([*])O",
    "POM": "[*]CO[*]",
    "PTFE": "[*]C(F)(F)C(F)(F)[*]",
    "PEEK": "[*]Oc1ccc(cc1)Oc1ccc(cc1)C(=O)c1ccc(cc1)[*]",
    "polysulfone": "[*]Oc1ccc(cc1)S(=O)(=O)c1ccc(cc1)[*]",
    "polyurethane": "[*]OCCOC(=O)Nc1ccc(cc1)CNC(=O)[*]",
    "PLA": "[*]OC(C)C(=O)[*]",
    "PCL": "[*]OCCCCCC(=O)[*]",
}

#: Real chain terminations deliberately kept OUT of ``_END_GROUP_PANEL``, so the
#: uncertainty can be scored on chemistries it was not built from.
HELD_OUT_END_GROUPS: dict[str, str] = {
    "acetate": "[*]OC(C)=O",
    "chloride": "[*]Cl",
    "methoxy": "[*]OC",
    "2-hydroxyethoxy": "[*]OCCO",
    "octyl": "[*]CCCCCCCC",
    "ATRP ester": "[*]C(C)(C)C(=O)OCC",
    "benzoate": "[*]OC(=O)c1ccccc1",
    "amide": "[*]C(=O)N",
    "thiophenyl": "[*]Sc1ccccc1",
    "sulfonate": "[*]S(=O)(=O)O",
    "morpholino": "[*]N1CCOCC1",
    "tert-butyl": "[*]C(C)(C)C",
    "phthalimide": "[*]N1C(=O)c2ccccc2C1=O",
    "polyol": "[*]OCC(O)C(O)CO",
}

ROOM = Conditions()


def _chain_under(unit: str, end_group: str, mn: float, key: str) -> float:
    """The exact chain descriptor if the ends really were ``end_group``.

    Ground truth for the uncertainty tests, and legitimate as such: the increment
    model is verified exact against a directly built 50-mer below, so evaluating
    it under a *stated* termination is arithmetic, not a second model.
    """
    measured = unit_increment(unit, (end_group, end_group))
    n = max((mn - measured.end[_MASS]) / measured.increment[_MASS], 0.0)
    return n * measured.increment[key] + measured.end[key]


# -- helpers ---------------------------------------------------------------


def _polymer(
    units,
    *,
    mn: Quantity | None = None,
    end_groups: tuple[str, ...] = (),
    extra_monomers: tuple[MonomerUnit, ...] = (),
    **kwargs,
) -> Candidate:
    if isinstance(units, str):
        pairs = [(units, 1.0)]
    else:
        pairs = list(units)
    monomers = tuple(
        MonomerUnit(smiles=s, mole_fraction=f) for s, f in pairs
    ) + extra_monomers
    return Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(
            monomers=monomers,
            number_average_molar_mass=mn,
            end_groups=end_groups,
            **kwargs,
        ),
    )


def _predict(candidate: Candidate, properties=None):
    expert = PolymerStructuralExpert()
    props = frozenset(properties or expert.supported_properties)
    request = PredictionRequest(
        candidate=candidate, properties=props, conditions=ROOM
    )
    return {p.property: p for p in expert.predict(request)}


def _one(candidate: Candidate, prop: str):
    return _predict(candidate, [prop])[prop]


# ==========================================================================
# 1. The joiner must not drift from the one already in the repository
# ==========================================================================


@pytest.mark.parametrize("name,unit", sorted(REFERENCE_UNITS.items()))
def test_mixed_sequence_joiner_reproduces_link_repeat_units(name, unit):
    """A second joiner exists only because a copolymer needs mixed sequences.

    Two implementations can drift, so identity is asserted rather than assumed,
    over all 21 units the docstring's 105/105 counts.
    """
    from rdkit import Chem

    for n in range(1, 6):
        mine = Chem.MolToSmiles(link_sequence((unit,) * n))
        theirs = Chem.MolToSmiles(link_repeat_units(unit, n))
        assert mine == theirs, f"{name} at n={n}"


@pytest.mark.parametrize("name,unit", sorted(REFERENCE_UNITS.items()))
def test_repeat_unit_mass_matches_the_repository_helper(name, unit):
    assert unit_increment(unit).increment[_MASS] == pytest.approx(
        repeat_unit_mass(unit), abs=1e-9
    ), name


def test_the_junction_census_the_docstring_quotes_is_reproducible():
    """132 of 210 comonomer pairs are sequence-independent in every descriptor.

    The docstring leads on that number, so it is measured here rather than
    remembered.  If a future RDKit changes a rotatable-bond convention the count
    will move, and this will say so instead of the prose quietly going stale.
    """
    import itertools

    units = sorted(REFERENCE_UNITS.values())
    zero = [
        (a, b)
        for a, b in itertools.combinations(units, 2)
        if all(abs(v) <= 1e-6 for v in junction_mismatch(a, b).values())
    ]
    assert len(units) == 21
    assert len(list(itertools.combinations(units, 2))) == 210
    assert len(zero) == 132


# ==========================================================================
# 2. The centrepiece: increments from short oligomers are exact at DP 50
# ==========================================================================


@pytest.mark.parametrize("name,unit", sorted(REFERENCE_UNITS.items()))
def test_increment_model_is_exact_against_a_directly_built_50_mer(name, unit):
    """D(n) = n * inc + c, with inc measured at DP 3-5, checked at DP 50.

    This is the module docstring's headline claim - 105 of 105 over 21 repeat
    units and 5 descriptors - and all 21 are checked here rather than a
    hand-picked seven, so the number in the docstring is the number this suite
    measures.

    The 50-mer is built by ``polymer.link_repeat_units``, the joiner that was
    already in the repository, *not* by this module's own ``link_sequence``.
    Grading the increments against oligomers from the same joiner that produced
    them would pass even if that joiner were systematically wrong.
    """
    measured = unit_increment(unit)
    direct = measure(link_repeat_units(unit, 50))
    for key in (_HEAVY, _ROTATABLE, _TPSA, _AROMATIC, _MASS):
        predicted = 50 * measured.increment[key] + measured.end[key]
        assert predicted == pytest.approx(direct[key], abs=1e-6), f"{name} {key}"
        assert key not in measured.nonlinear


def test_the_gem_substituted_unit_is_why_the_increment_is_taken_at_dp_4_to_5():
    """``[*]C([*])(C)C`` has a rotatable-bond count of 0, 0, 1, 2 at DP 2-5.

    The 3-2 difference is 0 and the 4-3 and 5-4 differences are both 1, so an
    increment taken at the shortest oligomers would be wrong by one bond per
    unit.  This pins the reason the constants are 3, 4, 5.
    """
    counts = [measure(link_sequence((GEM,) * n))[_ROTATABLE] for n in (2, 3, 4, 5)]
    assert counts == [0.0, 0.0, 1.0, 2.0]

    measured = unit_increment(GEM)
    assert _ROTATABLE not in measured.nonlinear
    assert measured.increment[_ROTATABLE] == pytest.approx(1.0)
    direct = measure(link_sequence((GEM,) * 50))
    assert 50 * measured.increment[_ROTATABLE] + measured.end[_ROTATABLE] == pytest.approx(
        direct[_ROTATABLE]
    )


# ==========================================================================
# 3. Hand-computable increments, with the arithmetic in the comment
# ==========================================================================


def test_polyethylene_increment_is_hand_computable():
    inc = unit_increment(PE).increment
    assert inc[_HEAVY] == 2.0  # -CH2-CH2-
    assert inc[_ROTATABLE] == 2.0  # both backbone C-C bonds, only visible in an oligomer
    assert inc[_TPSA] == 0.0
    assert inc[_MASS] == pytest.approx(28.054, abs=5e-3)  # C2H4


def test_polystyrene_aromatic_fraction_is_exactly_three_quarters():
    inc = unit_increment(PS).increment
    assert inc[_HEAVY] == 8.0  # 2 backbone C + 6 ring C
    assert inc[_AROMATIC] == 6.0
    assert inc[_AROMATIC] / inc[_HEAVY] == pytest.approx(0.75)


def test_ether_and_amide_and_ester_tpsa_increments_match_ertls_table():
    # Ertl: ether [O](-*)(-*) = 9.23 A^2, carbonyl [O]=* = 17.07, amide NH = 12.03.
    assert unit_increment(PEO).increment[_TPSA] == pytest.approx(9.23, abs=1e-2)
    assert unit_increment(NYLON6).increment[_TPSA] == pytest.approx(
        12.03 + 17.07, abs=1e-2
    )
    assert unit_increment(PMMA).increment[_TPSA] == pytest.approx(9.23 + 17.07, abs=1e-2)
    # PET carries two esters: 2 x (9.23 + 17.07).
    assert unit_increment(PET).increment[_TPSA] == pytest.approx(2 * (9.23 + 17.07), abs=1e-2)


def test_rotatable_bond_increments_exclude_amide_and_ester_bonds():
    # Nylon-6: seven backbone bonds per unit, less the excluded amide C-N.
    assert unit_increment(NYLON6).increment[_ROTATABLE] == 6.0
    # PET: the two ester C-O bonds are excluded by RDKit's strict convention.
    assert unit_increment(PET).increment[_ROTATABLE] == 5.0
    # Poly(ethylene oxide): all three backbone bonds count.
    assert unit_increment(PEO).increment[_ROTATABLE] == 3.0


# ==========================================================================
# 4. Chain arithmetic end to end, and the units
# ==========================================================================


def test_molar_mass_is_the_chain_not_the_repeat_unit():
    candidate = _polymer(PE, mn=Quantity(value=28.054, unit="kg/mol"))
    pred = _one(candidate, "molar_mass")
    assert pred.status is PredictionStatus.OK
    assert pred.quantity.to("g/mol").value == pytest.approx(28054.0)
    # The repeat unit's own mass would be wrong by three orders of magnitude.
    assert pred.quantity.to("g/mol").value / repeat_unit_mass(PE) > 900
    assert any("CHAIN" in n.upper() for n in pred.notes)


def test_molar_mass_pins_its_unit_conversion():
    """Reported in g/mol; a target written in kg/mol must read the same number."""
    candidate = _polymer(PE, mn=Quantity(value=120.0, unit="kg/mol"))
    pred = _one(candidate, "molar_mass")
    assert pred.quantity.unit == str(Quantity(value=1, unit="g/mol").unit)
    assert pred.quantity.to("kg/mol").value == pytest.approx(120.0)
    # The registry's canonical unit is kg/mol; g/mol converts to it exactly.
    assert pred.canonical.unit == "kilogram / mole"
    assert pred.canonical.value == pytest.approx(120.0)
    assert pred.canonical_uncertainty.std == 0.0


def test_tpsa_pins_its_unit_conversion():
    """Reported in angstrom^2, which converts to the m^2 the brief asked for."""
    candidate = _polymer(NYLON6, mn=Quantity(value=20.0, unit="kg/mol"))
    pred = _one(candidate, "topological_polar_surface_area")
    assert pred.status is PredictionStatus.OK
    angstrom = pred.quantity.to("angstrom^2").value
    assert pred.quantity.to("m^2").value == pytest.approx(angstrom * 1e-20, rel=1e-9)


def test_chain_counts_are_arithmetic_in_the_stated_chain_length():
    candidate = _polymer(PE, mn=Quantity(value=28.054, unit="kg/mol"))
    model = analyse_chain(candidate.polymer)
    dp = model.degree_of_polymerization(28054.0)
    # Mn = 28054 g/mol less two methyl caps (30.07), over 28.054 g/mol a unit.
    assert dp == pytest.approx((28054.0 - model.end_mass) / 28.054, rel=1e-9)
    assert dp == pytest.approx(998.9, abs=0.1)

    heavy = _one(candidate, "heavy_atom_count")
    assert heavy.quantity.value == pytest.approx(2.0 * dp + 2.0, rel=1e-9)
    rot = _one(candidate, "rotatable_bond_count")
    assert rot.quantity.value == pytest.approx(2.0 * dp + rot.quantity.value - 2.0 * dp)
    assert rot.quantity.value == pytest.approx(2.0 * dp - 1.0, abs=0.1)


def test_molar_mass_round_trips_through_the_degree_of_polymerisation():
    """n * M0 + M_end must give back exactly the Mn that produced n."""
    candidate = _polymer(PET, mn=Quantity(value=45.0, unit="kg/mol"))
    model = analyse_chain(candidate.polymer)
    dp = model.degree_of_polymerization(45000.0)
    assert model.value(_MASS, dp) == pytest.approx(45000.0, rel=1e-12)


def test_counts_are_non_integer_and_say_why():
    candidate = _polymer(PS, mn=Quantity(value=100.0, unit="kg/mol"))
    pred = _one(candidate, "heavy_atom_count")
    assert not float(pred.quantity.value).is_integer()
    assert any("number average" in n for n in pred.notes)


# ==========================================================================
# 5. Refusals.  Each one names the actual cause.
# ==========================================================================


EXTENSIVE = (
    "molar_mass",
    "heavy_atom_count",
    "rotatable_bond_count",
    "topological_polar_surface_area",
)


def test_without_a_stated_mn_every_extensive_descriptor_refuses():
    preds = _predict(_polymer(PS))
    for prop in EXTENSIVE:
        assert preds[prop].status is PredictionStatus.UNSUPPORTED, prop
        assert "number-average molar mass" in " ".join(preds[prop].notes)
    # ... and nothing returns the repeat unit's value instead.
    assert preds["molar_mass"].quantity is None


def test_without_a_stated_mn_the_aromatic_fraction_still_answers():
    pred = _one(_polymer(PS), "aromatic_atom_fraction")
    assert pred.status is PredictionStatus.OK
    assert pred.quantity.value == pytest.approx(0.75)
    # Wide, and honestly so: the price of not stating a chain length.
    assert pred.uncertainty.std > 0.05
    assert any("infinite-chain limit" in n for n in pred.notes)


def test_a_repeat_unit_with_one_attachment_point_is_refused():
    preds = _predict(_polymer("[*]CCC"))
    for prop in preds:
        assert preds[prop].status is PredictionStatus.UNSUPPORTED
        assert "attachment point" in " ".join(preds[prop].notes)


def test_a_repeat_unit_with_three_attachment_points_is_refused():
    """Three dummies is a branch or crosslink site, not a linear chain unit."""
    preds = _predict(_polymer("[*]CC([*])C[*]", mn=Quantity(value=50.0, unit="kg/mol")))
    for prop in preds:
        assert preds[prop].status is PredictionStatus.UNSUPPORTED
        assert "attachment point" in " ".join(preds[prop].notes)


def test_an_unparseable_repeat_unit_is_refused():
    preds = _predict(_polymer("[*]C(C[*]", mn=Quantity(value=50.0, unit="kg/mol")))
    for prop in preds:
        assert preds[prop].status is PredictionStatus.UNSUPPORTED
        assert "valid SMILES" in " ".join(preds[prop].notes)


def test_a_network_has_no_finite_chain_so_counts_refuse_but_composition_answers():
    candidate = _polymer(
        PS,
        mn=Quantity(value=50.0, unit="kg/mol"),
        topology=PolymerTopology.NETWORK,
        crosslink_density=Quantity(value=1.0e-3, unit="mol/cm^3"),
    )
    preds = _predict(candidate)
    for prop in EXTENSIVE:
        assert preds[prop].status is PredictionStatus.UNSUPPORTED, prop
        assert "gel point" in " ".join(preds[prop].notes) or "crosslink" in " ".join(
            preds[prop].notes
        )
    # A composition ratio survives a gel point; a count does not.
    aromatic = preds["aromatic_atom_fraction"]
    assert aromatic.status is PredictionStatus.OUT_OF_DOMAIN
    assert aromatic.quantity.value == pytest.approx(0.75, abs=1e-3)


def test_a_crosslinker_monomer_refuses_the_chain_counts():
    candidate = _polymer(
        [(PS, 1.0)],
        mn=Quantity(value=50.0, unit="kg/mol"),
        extra_monomers=(
            MonomerUnit(
                smiles="[*]CC([*])c1ccc(cc1)C=C",
                mole_fraction=0.0,
                role=MonomerRole.CROSSLINKER,
            ),
        ),
    )
    pred = _one(candidate, "heavy_atom_count")
    assert pred.status is PredictionStatus.UNSUPPORTED
    assert "crosslinker" in " ".join(pred.notes)


def test_a_stated_crosslink_density_alone_refuses_the_chain_counts():
    candidate = _polymer(
        PS,
        mn=Quantity(value=50.0, unit="kg/mol"),
        crosslink_density=Quantity(value=2.0e-4, unit="mol/cm^3"),
    )
    pred = _one(candidate, "rotatable_bond_count")
    assert pred.status is PredictionStatus.UNSUPPORTED
    assert "crosslink" in " ".join(pred.notes)


def test_an_mn_below_one_repeat_unit_is_not_a_chain():
    candidate = _polymer(PE, mn=Quantity(value=40.0, unit="g/mol"))
    pred = _one(candidate, "heavy_atom_count")
    assert pred.status is PredictionStatus.UNSUPPORTED
    assert "does not describe a chain" in " ".join(pred.notes)


def test_an_mn_with_the_wrong_dimension_is_named_rather_than_crashing():
    """The message names the problem rather than surfacing a bare unit error.

    Every property refuses, including the one that does not need Mn: a
    specification whose chain length is a temperature is broken, and answering
    part of it would hide the breakage.
    """
    candidate = _polymer(PE, mn=Quantity(value=350.0, unit="K"))
    preds = _predict(candidate)
    assert set(preds) == set(EXTENSIVE) | {"aromatic_atom_fraction"}
    for prop, pred in preds.items():
        assert pred.status is PredictionStatus.UNSUPPORTED, prop
        assert "not a mass per amount of substance" in " ".join(pred.notes)


def test_a_descriptor_with_no_constant_increment_is_refused_by_name(monkeypatch):
    """A non-local descriptor has no per-unit increment; refuse only that one.

    Forced here, because no counterexample was found among the 21 real repeat
    units tried - which is exactly why the check must not be dead code.
    """
    import formulate.experts.polymer_structural as mod

    def quadratic(fragments):
        n = len(fragments)
        return {
            _HEAVY: float(n * n),  # deliberately non-linear
            _ROTATABLE: float(2 * n),
            _TPSA: 0.0,
            _AROMATIC: 0.0,
            _MASS: 28.054 * n + 30.07,
        }

    unit_increment.cache_clear()
    junction_mismatch.cache_clear()
    monkeypatch.setattr(mod, "_sequence_descriptors", quadratic)
    try:
        preds = _predict(_polymer(PE, mn=Quantity(value=28.054, unit="kg/mol")))
        for prop in ("heavy_atom_count", "aromatic_atom_fraction"):
            assert preds[prop].status is PredictionStatus.UNSUPPORTED, prop
            assert "no constant per-unit increment" in " ".join(preds[prop].notes)
        # The descriptors that stayed linear still answer.
        assert preds["rotatable_bond_count"].status is PredictionStatus.OK
    finally:
        monkeypatch.undo()
        unit_increment.cache_clear()
        junction_mismatch.cache_clear()


# ==========================================================================
# 6. Copolymers: mole-weight where the junction term is zero, refuse where not
# ==========================================================================


def test_junction_mismatch_is_measured_zero_rather_than_special_cased():
    """Joining adds no atoms and cannot create a ring, so heavy and aromatic
    mismatches are zero by construction - but the code must MEASURE that."""
    for a, b in ((PE, PP), (PS, PMMA), (PET, PE), (NYLON6, PE)):
        delta = junction_mismatch(a, b)
        assert delta[_HEAVY] == pytest.approx(0.0, abs=1e-9)
        assert delta[_AROMATIC] == pytest.approx(0.0, abs=1e-9)
        assert delta[_MASS] == pytest.approx(0.0, abs=1e-6)


def test_ethylene_propylene_copolymer_mole_weights_exactly():
    model = analyse_chain(_polymer([(PE, 0.5), (PP, 0.5)]).polymer)
    assert model.failure is None
    assert model.junction == {}
    assert model.increment[_HEAVY] == pytest.approx(2.5)  # (2 + 3) / 2
    assert model.increment[_MASS] == pytest.approx(
        0.5 * repeat_unit_mass(PE) + 0.5 * repeat_unit_mass(PP)
    )

    mn = model.repeat_unit_mass * 1000.0 + model.end_mass
    candidate = _polymer([(PE, 0.5), (PP, 0.5)], mn=Quantity(value=mn, unit="g/mol"))
    heavy = _one(candidate, "heavy_atom_count")
    assert heavy.status is PredictionStatus.OK
    assert heavy.quantity.value == pytest.approx(2.5 * 1000.0 + model.end[_HEAVY])


def test_styrene_mma_copolymer_answers_every_descriptor():
    model = analyse_chain(_polymer([(PS, 0.5), (PMMA, 0.5)]).polymer)
    assert model.junction == {}
    candidate = _polymer([(PS, 0.5), (PMMA, 0.5)], mn=Quantity(value=80.0, unit="kg/mol"))
    preds = _predict(candidate)
    assert all(p.status.has_value for p in preds.values())


def test_a_sequence_dependent_descriptor_refuses_while_the_others_answer():
    """PET-co-ethylene has a measured rotatable-bond junction term of +1.

    Block, random and alternating chains of that composition then have
    different rotatable-bond counts, and PolymerSpec states no sequence
    distribution.  This is the test that proves the copolymer refusal is
    conditional on a computed quantity, not a blanket rule.
    """
    assert junction_mismatch(PET, PE)[_ROTATABLE] == pytest.approx(1.0)

    candidate = _polymer([(PET, 0.5), (PE, 0.5)], mn=Quantity(value=40.0, unit="kg/mol"))
    preds = _predict(candidate)
    rot = preds["rotatable_bond_count"]
    assert rot.status is PredictionStatus.UNSUPPORTED
    assert "depends on comonomer sequence" in " ".join(rot.notes)
    for prop in ("heavy_atom_count", "aromatic_atom_fraction",
                 "topological_polar_surface_area", "molar_mass"):
        assert preds[prop].status.has_value, prop


def test_the_comonomer_end_term_residual_is_bounded_and_is_in_the_bar():
    """Mole-weighting the END term averages over which comonomer terminates.

    The junction test cannot see that: it is a whole-chain offset, not a
    per-unit error.  PE-co-PIB is the measured case - end rotatable-bond terms
    of -1 and -2 - and the model must carry the difference as uncertainty
    rather than drop it.
    """
    assert junction_mismatch(PE, PIB)[_ROTATABLE] == pytest.approx(0.0, abs=1e-9)
    model = analyse_chain(_polymer([(PE, 0.5), (PIB, 0.5)]).polymer)
    residual = model.end_spread[_ROTATABLE]
    assert residual == pytest.approx(1.0)

    predicted = 50 * model.increment[_ROTATABLE] + model.end[_ROTATABLE]
    for sequence in ((PE, PIB) * 25, (PE,) * 25 + (PIB,) * 25):
        direct = measure(link_sequence(sequence))
        assert abs(predicted - direct[_ROTATABLE]) <= residual + 1e-9
        assert predicted != pytest.approx(direct[_ROTATABLE])  # honestly not exact

    mn = model.repeat_unit_mass * 200.0 + model.end_mass
    pred = _one(
        _polymer([(PE, 0.5), (PIB, 0.5)], mn=Quantity(value=mn, unit="g/mol")),
        "rotatable_bond_count",
    )
    assert pred.uncertainty.std >= residual
    assert any("sits at each end" in n for n in pred.notes)
    # A homopolymer carries no such term at all.
    assert analyse_chain(_polymer(PE).polymer).end_spread[_ROTATABLE] == 0.0


def test_mole_weighting_reproduces_a_real_50_unit_copolymer_sequence():
    """Where the junction term is zero, mole-weighting is exact for any sequence."""
    model = analyse_chain(_polymer([(PE, 0.5), (PP, 0.5)]).polymer)
    for sequence in (
        (PE, PP) * 25,                      # alternating
        (PE,) * 25 + (PP,) * 25,            # block
        (PE, PE, PP, PP, PE, PP) * 8 + (PE, PP),
    ):
        direct = measure(link_sequence(sequence))
        for key in (_HEAVY, _ROTATABLE, _TPSA, _AROMATIC, _MASS):
            predicted = 50 * model.increment[key] + model.end[key]
            assert predicted == pytest.approx(direct[key], abs=1e-6), key


# ==========================================================================
# 7. Uncertainty behaviour
# ==========================================================================


def test_the_end_group_bar_is_negligible_for_a_long_chain_and_enormous_for_a_short_one():
    long = _one(_polymer(PP, mn=Quantity(value=200.0, unit="kg/mol")), "heavy_atom_count")
    short = _one(_polymer(PP, mn=Quantity(value=1.0, unit="kg/mol")), "heavy_atom_count")
    assert long.uncertainty.std / long.quantity.value < 0.005
    assert short.uncertainty.std / short.quantity.value > 0.10
    # Same absolute bar in both: it is a mass, not a fraction.
    assert long.uncertainty.std == pytest.approx(short.uncertainty.std)


def test_stated_end_groups_collapse_the_bar_to_zero():
    candidate = _polymer(
        [(PE, 1.0)],
        mn=Quantity(value=5.0, unit="kg/mol"),
        extra_monomers=(
            MonomerUnit(smiles="[*]C", mole_fraction=0.0, role=MonomerRole.END_GROUP),
        ),
    )
    pred = _one(candidate, "heavy_atom_count")
    assert pred.status.has_value
    assert pred.uncertainty.std == 0.0
    assert "end groups are stated" in pred.uncertainty.basis


def test_stated_heavy_end_groups_shorten_the_chain():
    """A dodecyl end group is 168 g/mol of chain that is not repeat unit."""
    light = analyse_chain(_polymer(PE).polymer)
    heavy = analyse_chain(
        _polymer(
            [(PE, 1.0)],
            extra_monomers=(
                MonomerUnit(
                    smiles="[*]CCCCCCCCCCCC", mole_fraction=0.0, role=MonomerRole.END_GROUP
                ),
            ),
        ).polymer
    )
    assert heavy.end_mass > light.end_mass + 300.0  # two dodecyls, less two methyls
    assert heavy.degree_of_polymerization(10000.0) < light.degree_of_polymerization(10000.0)
    # The end groups' own atoms are counted too, not just their mass.
    assert heavy.end[_HEAVY] == pytest.approx(24.0)


def test_molar_mass_carries_a_zero_bar_and_puts_dispersity_in_a_note():
    candidate = _polymer(PE, mn=Quantity(value=50.0, unit="kg/mol"), dispersity=2.0)
    pred = _one(candidate, "molar_mass")
    assert pred.uncertainty.std == 0.0
    spread = " ".join(pred.notes)
    assert "dispersity" in spread
    # Flory: sigma = Mn sqrt(D - 1) = Mn for the most-probable distribution.
    assert f"{50000.0 * math.sqrt(1.0):.4g}" in spread


@pytest.mark.parametrize("name", ["PE", "PP", "PIB", "PTFE", "PVC"])
def test_a_polyolefin_tpsa_is_zero_but_its_bar_is_not(name):
    """The regression that matters most in this file.

    Pricing the unknown end mass at the *repeat unit's* polar-surface-area per
    gram gives exactly zero for a backbone that has none, so the expert used to
    report 0.00 +- 0.00 - an assertion of exactness - for a chain that, if it
    was made by persulfate emulsion polymerisation, carries 127 A^2 at its ends.
    A zero bar on a non-zero quantity is the one failure this repository ranks
    on directly, because the pessimistic bound equals the value.
    """
    unit = REFERENCE_UNITS[name]
    pred = _one(
        _polymer(unit, mn=Quantity(value=10.0, unit="kg/mol")),
        "topological_polar_surface_area",
    )
    assert pred.status.has_value
    assert pred.quantity.value == pytest.approx(0.0, abs=1e-9)  # the value is right
    truth = _chain_under(unit, "[*]OS(=O)(=O)O", 10000.0, _TPSA)
    assert truth > 100.0  # a persulfate-terminated chain really carries this
    assert pred.uncertainty.std >= truth  # ... and the bar now covers it


def test_the_unstated_end_group_bar_covers_end_groups_it_was_not_built_from():
    """Held out, not in sample: the panel is scored on 14 other terminations.

    Every case is exact ground truth - the increment model is verified exact at
    DP 50 above, so evaluating it under a stated termination is arithmetic - so
    this is a coverage measurement, not an estimate of one.  The docstring
    quotes 872 of 882 for the shipped bar against 768 for the mass-equivalence
    term alone; both are re-derived here.
    """
    assert not (
        {s for s, _ in _END_GROUP_PANEL} & set(HELD_OUT_END_GROUPS.values())
    ), "the held-out set has leaked into the panel"

    mn = 10000.0
    keys = {
        "heavy_atom_count": _HEAVY,
        "rotatable_bond_count": _ROTATABLE,
        "topological_polar_surface_area": _TPSA,
    }
    inside = mass_only_inside = total = 0
    for unit in REFERENCE_UNITS.values():
        preds = _predict(_polymer(unit, mn=Quantity(value=mn, unit="g/mol")), keys)
        base = unit_increment(unit)
        for prop, key in keys.items():
            pred = preds[prop]
            mass_bar = (
                abs(base.increment[key])
                * _UNSTATED_END_GROUP_MASS_SPAN
                / base.increment[_MASS]
            )
            for end_group in HELD_OUT_END_GROUPS.values():
                error = abs(_chain_under(unit, end_group, mn, key) - pred.quantity.value)
                total += 1
                inside += error <= pred.uncertainty.std + 1e-9
                mass_only_inside += error <= mass_bar + 1e-9
    assert total == 882
    assert inside == 872
    assert mass_only_inside == 768  # what the bar used to be, and why it changed


def test_the_end_group_bar_does_not_depend_on_the_stated_chain_length():
    """It is an end effect, so it must be an absolute offset, not a fraction.

    Shifting the termination changes the chain length by a fixed mass and the
    end term by a fixed amount; both are independent of Mn, so a bar that drifted
    with Mn would mean the arithmetic had picked up a spurious chain-length term.
    """
    bars = [
        _one(
            _polymer(PMMA, mn=Quantity(value=mn, unit="kg/mol")), "heavy_atom_count"
        ).uncertainty.std
        for mn in (5.0, 50.0, 500.0)
    ]
    assert bars[0] == pytest.approx(bars[1]) == pytest.approx(bars[2])
    assert bars[0] > 0.0


@pytest.mark.parametrize("name,unit", sorted(REFERENCE_UNITS.items()))
def test_the_interior_increment_does_not_depend_on_the_caps(name, unit):
    """The panel arithmetic assumes an interior repeat unit is cap-blind.

    It is measured at DP 4->5 with identical caps at both lengths, so it should
    be - but "should be" is how a silent error gets in, and if it ever stopped
    being true the panel would be differencing two different chains.
    """
    plain = unit_increment(unit).increment
    for end_group, _ in _END_GROUP_PANEL:
        capped = unit_increment(unit, (end_group, end_group)).increment
        for key in (_HEAVY, _ROTATABLE, _TPSA, _AROMATIC, _MASS):
            assert capped[key] == pytest.approx(plain[key], abs=1e-6), (
                f"{name} {key} under {end_group}"
            )


def test_the_panel_drops_an_end_group_it_cannot_build_rather_than_refusing():
    """One unbuildable hypothetical must not cost a real prediction."""
    alternatives = end_group_alternatives(((PE, 1.0),))
    assert 1 < len(alternatives) <= len(_END_GROUP_PANEL)
    assert all(inc[_MASS] > 0.0 for _, _, inc, _ in alternatives)


# ==========================================================================
# 7b. A monomer that is not there
# ==========================================================================


def test_a_comonomer_at_zero_mole_fraction_is_not_in_the_chain():
    """It is absent, and treating it as present is wrong in both directions.

    It drags a phantom comonomer's end term into the bar, and it puts a junction
    that never occurs into the sequence test - which then REFUSES a descriptor
    of what is chemically a pure homopolymer.  The evolutionary operator that
    shifts composition takes a step of min(fraction_step, f_i), so it lands a
    comonomer on exactly 0.0 routinely; this is a reachable state, not a
    hypothetical one.
    """
    mn = Quantity(value=40.0, unit="kg/mol")
    pure = _predict(_polymer(PET, mn=mn))
    ghost = _predict(_polymer([(PET, 1.0), (PE, 0.0)], mn=mn))

    # PET-co-ethylene has a rotatable-bond junction term of +1, so a phantom
    # ethylene used to turn a PET homopolymer's count into a refusal.
    assert junction_mismatch(PET, PE)[_ROTATABLE] == pytest.approx(1.0)
    for prop, pred in pure.items():
        assert ghost[prop].status is pred.status, prop
        if pred.quantity is not None:
            assert ghost[prop].quantity.value == pytest.approx(pred.quantity.value)
            assert ghost[prop].uncertainty.std == pytest.approx(pred.uncertainty.std)
    assert any("zero mole fraction" in n for n in ghost["heavy_atom_count"].notes)

    model = analyse_chain(_polymer([(PET, 1.0), (PE, 0.0)]).polymer)
    assert model.units == ((PET, 1.0),)
    assert model.end_spread[_ROTATABLE] == 0.0  # no phantom end-term spread


def test_a_network_reason_beats_an_unbuildable_crosslinker_message():
    """A tetrafunctional crosslinker is not a malformed repeat unit.

    Telling a chemist "this repeat unit must carry exactly two attachment
    points" about a divinylbenzene crosslink site is advice that would make the
    specification worse.  The network is the real objection.
    """
    candidate = _polymer(
        [(PS, 0.9)],
        mn=Quantity(value=50.0, unit="kg/mol"),
        extra_monomers=(
            MonomerUnit(
                smiles="[*]CC([*])c1ccc(cc1)C([*])C[*]",  # divinylbenzene, 4 arms
                mole_fraction=0.1,
                role=MonomerRole.CROSSLINKER,
            ),
        ),
    )
    pred = _one(candidate, "aromatic_atom_fraction")
    assert pred.status is PredictionStatus.UNSUPPORTED
    joined = " ".join(pred.notes)
    assert "crosslinker" in joined
    assert "attachment point" not in joined


def test_the_aromatic_fraction_bar_shrinks_once_a_chain_length_is_stated():
    unstated = _one(_polymer(PET), "aromatic_atom_fraction")
    stated = _one(
        _polymer(PET, mn=Quantity(value=30.0, unit="kg/mol")), "aromatic_atom_fraction"
    )
    assert unstated.quantity.value == pytest.approx(6.0 / 14.0)
    assert unstated.uncertainty.std > 10 * stated.uncertainty.std
    assert stated.quantity.value == pytest.approx(6.0 / 14.0, abs=1e-3)


# ==========================================================================
# 8. Domain
# ==========================================================================


def test_a_short_oligomer_is_out_of_domain_but_still_answered():
    candidate = _polymer(PS, mn=Quantity(value=520.0, unit="g/mol"))  # DP about 5
    pred = _one(candidate, "heavy_atom_count")
    assert pred.status is PredictionStatus.OUT_OF_DOMAIN
    assert pred.quantity is not None
    assert any("oligomer" in w for w in pred.applicability.warnings)


def test_tacticity_is_not_penalised_because_a_graph_descriptor_cannot_see_it():
    """Guards against someone copying polymer.py's tacticity rule to here."""
    expert = PolymerStructuralExpert()
    atactic = _polymer(PP, mn=Quantity(value=100.0, unit="kg/mol"), tacticity=Tacticity.ATACTIC)
    for tacticity in (Tacticity.ISOTACTIC, Tacticity.SYNDIOTACTIC):
        candidate = _polymer(
            PP, mn=Quantity(value=100.0, unit="kg/mol"), tacticity=tacticity
        )
        domain = expert.assess_domain(candidate)
        assert domain.in_domain
        assert domain.score == 1.0
        assert _one(candidate, "heavy_atom_count").quantity.value == pytest.approx(
            _one(atactic, "heavy_atom_count").quantity.value
        )


def test_a_branched_architecture_is_flagged_rather_than_refused():
    candidate = _polymer(
        PE, mn=Quantity(value=100.0, unit="kg/mol"), topology=PolymerTopology.BRANCHED
    )
    pred = _one(candidate, "heavy_atom_count")
    assert pred.status is PredictionStatus.OUT_OF_DOMAIN
    assert pred.quantity is not None
    assert any("branch point" in n for n in pred.notes)


def test_a_silicon_backbone_gets_an_exact_number_and_a_note_about_what_it_means():
    candidate = _polymer(PDMS, mn=Quantity(value=30.0, unit="kg/mol"))
    pred = _one(candidate, "topological_polar_surface_area")
    assert pred.status is PredictionStatus.OK
    # RDKit scores the siloxane oxygen exactly as an ether oxygen: 9.23 A^2.
    assert unit_increment(PDMS).increment[_TPSA] == pytest.approx(9.23, abs=1e-2)
    assert any("siloxane" in n for n in pred.notes)


# ==========================================================================
# 9. Wiring
# ==========================================================================


def test_the_expert_is_polymer_only():
    expert = PolymerStructuralExpert()
    assert expert.covers("heavy_atom_count", MaterialClass.POLYMER)
    assert not expert.covers("heavy_atom_count", MaterialClass.MOLECULE)
    assert not expert.covers("heavy_atom_count", MaterialClass.MIXTURE)
    assert expert.dependencies == frozenset()


def test_a_polymer_candidate_from_the_convenience_constructor_works():
    candidate = polymer_candidate(PE)
    preds = _predict(candidate)
    assert preds["aromatic_atom_fraction"].status is PredictionStatus.OK
    assert preds["molar_mass"].status is PredictionStatus.UNSUPPORTED


def test_provenance_carries_the_per_repeat_unit_numbers_the_value_hides():
    candidate = _polymer(PET, mn=Quantity(value=25.0, unit="kg/mol"))
    pred = _one(candidate, "heavy_atom_count")
    params = pred.provenance.parameters
    assert params["repeat_units"] == [PET]
    assert params["repeat_unit_increment"]["heavy_atoms"] == pytest.approx(14.0)
    assert params["degree_of_polymerization"] == pytest.approx(
        (25000.0 - params["end_mass_g_mol"]) / params["repeat_unit_mass_g_mol"], rel=1e-6
    )


def test_availability_reports_a_reason_rather_than_crashing():
    expert = PolymerStructuralExpert()
    assert expert.is_available()
    assert expert.unavailable_reason() == ""


def test_unusable_stated_end_groups_are_flagged_rather_than_ignored():
    """A bare formula is not a cappable fragment; say so instead of pretending."""
    candidate = _polymer(
        PE, mn=Quantity(value=5.0, unit="kg/mol"), end_groups=("CCO",)
    )
    pred = _one(candidate, "heavy_atom_count")
    assert pred.status.has_value
    assert pred.uncertainty.std > 0.0
    assert any("not in a form" in n for n in pred.notes)
