"""Can this polymer be made?  Monomer reconstruction, route difficulty, refusals.

Three things are asserted here, and the middle one is the point.

*What the reconstruction produces.*  A wrong monomer is a confident wrong
number, so the InChIKey of every reconstructed monomer is pinned against the
monomer the polymer is actually made from.  That test checks chemistry rather
than arithmetic and it is the one that should be hardest to keep passing.

*That the repeat unit is never scored as if it were a molecule.*  The cheapest
way to "add" this expert would have been to hand the repeat-unit SMILES to the
molecule-class ``feasibility`` expert, so the difference between the two numbers
is asserted rather than assumed, and this expert is asserted not to cover
molecules at all.

*What it refuses.*  A refusal is a feature in this repository and must not
regress.  Every refusal path - a repeat unit with the wrong number of attachment
points, a tetrasubstituted backbone, an unstrained six-ring, a network, a
crosslinker, a copolymer of incompatible mechanisms, a backbone no chemistry
places - is exercised and asserted to carry a chemically specific reason, because
the cheapest way to make the coverage number look better would be to let one of
them through with a mid-range guess.

The measured numbers this file pins were produced by the module, on this
installation, and are quoted in its docstring: 30 of 30 monomer identities, 57
of 57 reference polymers routed, 223 of 224 concordant commodity/hard pairs.
"""

from __future__ import annotations

import json
import math
import random

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MonomerRole,
    MonomerUnit,
    PolymerSpec,
    PolymerTopology,
    Tacticity,
    molecule_candidate,
    polymer_candidate,
)
from formulate.core.prediction import PredictionStatus
from formulate.core.properties import PropertyFamily, get_property
from formulate.core.quantity import Quantity
from formulate.experts.base import PredictionRequest
from formulate.experts.feasibility import SynthesisFeasibilityExpert
from formulate.experts.polymer import reference_polymers
from formulate.experts.polymer_feasibility import (
    SMALL_MOLECULE_CEILING,
    Mechanism,
    PolymerFeasibilityExpert,
    analyse_repeat_unit,
    ertl_score,
    monomer_term,
    score_route,
)

pytestmark = requires_rdkit

PROP = "synthetic_accessibility"
PROPERTIES = frozenset({PROP})
EXPERT = PolymerFeasibilityExpert()

PE = "[*]CC[*]"
PS = "[*]CC(c1ccccc1)[*]"
PVC = "[*]CC(Cl)[*]"
PVAC = "[*]CC(OC(C)=O)[*]"
PVOH = "[*]CC(O)[*]"
PET = "[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]"
PCL = "[*]CCCCCC(=O)O[*]"
PTFE = "[*]C(F)(F)C(F)(F)[*]"
PDMS = "[*][Si](C)(C)O[*]"
PPP = "[*]c1ccc([*])cc1"
PAMS = "[*]CC(C)(c1ccccc1)[*]"
POLYACETYLENE = "[*]C=C[*]"
HEAD_TO_HEAD_PVC = "[*]C(Cl)C(Cl)[*]"

#: The reference file draws poly(2,6-dimethyl-1,4-phenylene oxide) with its
#: methyls *meta* to the phenolic oxygen.  This is the same polymer drawn the way
#: the real one is, with them ortho, and the two are kept apart on purpose: the
#: expert reconstructs a different phenol from each, which is a statement about
#: the drawing rather than about the expert.
PPO_AS_DRAWN = "[*]Oc1cc(C)c([*])c(C)c1"
PPO_TRUE = "[*]Oc1c(C)cc([*])cc1C"


def predict(candidate: Candidate):
    request = PredictionRequest(
        candidate=candidate, properties=PROPERTIES, conditions=candidate.conditions
    )
    predictions = EXPERT.predict(request)
    assert len(predictions) == 1
    return predictions[0]


def score(unit: str, **spec_kwargs) -> float:
    """The number this expert reports for a homopolymer of ``unit``."""
    if spec_kwargs:
        candidate = Candidate(
            material_class=MaterialClass.POLYMER,
            polymer=PolymerSpec(monomers=(MonomerUnit(smiles=unit),), **spec_kwargs),
        )
    else:
        candidate = polymer_candidate(unit)
    prediction = predict(candidate)
    assert prediction.quantity is not None, prediction.notes
    return prediction.quantity.value


def refusal(unit_or_candidate) -> str:
    candidate = (
        polymer_candidate(unit_or_candidate)
        if isinstance(unit_or_candidate, str)
        else unit_or_candidate
    )
    prediction = predict(candidate)
    assert prediction.quantity is None, (
        f"expected a refusal, got {prediction.quantity} - an unmakeable structure "
        "scored is worse than one refused"
    )
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.notes and prediction.notes[0].strip()
    return prediction.notes[0]


def inchikey(smiles: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return Chem.MolToInchiKey(mol)


def monomers_of(unit: str) -> set[str]:
    analysis = analyse_repeat_unit(unit)
    assert analysis.failure is None, analysis.failure
    return {inchikey(m) for route in analysis.routes for m in route.monomers}


# --------------------------------------------------------------------------
# 1. Monomer identity - the test that checks chemistry rather than a number
# --------------------------------------------------------------------------

#: Repeat unit -> the monomer(s) the polymer is really made from.  Where a
#: polymer has two honest routes, both monomers are required, because dropping
#: one would let a recogniser silently disappear.
MONOMER_IDENTITY: dict[str, tuple[str, list[str]]] = {
    "polyethylene": (PE, ["C=C"]),
    "polypropylene": ("[*]CC(C)[*]", ["C=CC"]),
    "polystyrene": (PS, ["C=Cc1ccccc1"]),
    "poly(vinyl chloride)": (PVC, ["C=CCl"]),
    "poly(vinylidene chloride)": ("[*]CC(Cl)(Cl)[*]", ["C=C(Cl)Cl"]),
    "PTFE": (PTFE, ["FC(F)=C(F)F"]),
    "PCTFE": ("[*]C(F)(F)C(F)(Cl)[*]", ["FC(F)=C(F)Cl"]),
    "PMMA": ("[*]CC(C)(C(=O)OC)[*]", ["C=C(C)C(=O)OC"]),
    "polyacrylonitrile": ("[*]CC(C#N)[*]", ["C=CC#N"]),
    "poly(vinyl acetate)": (PVAC, ["CC(=O)OC=C"]),
    "polyisobutylene": ("[*]CC(C)(C)[*]", ["CC(C)=C"]),
    "polybutadiene": ("[*]CC=CC[*]", ["C=CC=C"]),
    "polyisoprene": ("[*]CC(C)=CC[*]", ["C=CC(C)=C"]),
    "polychloroprene": ("[*]CC(Cl)=CC[*]", ["C=CC(Cl)=C"]),
    "poly(ethylene oxide)": ("[*]CCO[*]", ["C1CO1"]),
    "poly(propylene oxide)": ("[*]CC(C)O[*]", ["CC1CO1"]),
    "polyoxetane": ("[*]CCCO[*]", ["C1COC1"]),
    "PTMO": ("[*]CCCCO[*]", ["C1CCOC1"]),
    "polyoxymethylene": ("[*]CO[*]", ["C=O"]),
    "polycaprolactone": (PCL, ["O=C1CCCCCO1", "OCCCCCC(=O)O"]),
    "nylon-6": ("[*]NCCCCCC(=O)[*]", ["O=C1CCCCCN1", "NCCCCCC(=O)O"]),
    "polylactide": ("[*]OC(C)C(=O)[*]", ["CC(O)C(=O)O", "CC1OC(=O)C(C)OC1=O"]),
    "polyglycolide": ("[*]OCC(=O)[*]", ["OCC(=O)O", "O=C1COC(=O)CO1"]),
    "PET": (PET, ["OC(=O)c1ccc(cc1)C(=O)O", "OCCO"]),
    "nylon-6,6": (
        "[*]NCCCCCCNC(=O)CCCCC(=O)[*]",
        ["OC(=O)CCCCC(=O)O", "NCCCCCCN"],
    ),
    "PEEK": (
        "[*]Oc1ccc(Oc2ccc(C(=O)c3ccc([*])cc3)cc2)cc1",
        ["Oc1ccc(O)cc1", "O=C(c1ccc(F)cc1)c1ccc(F)cc1"],
    ),
    "poly(phenylene oxide)": (PPO_TRUE, ["Cc1cccc(C)c1O"]),
    "poly(p-phenylene)": (PPP, ["Brc1ccc(Br)cc1"]),
    "polycarbonate": (
        "[*]Oc1ccc(cc1)C(C)(C)c1ccc(cc1)OC(=O)[*]",
        ["CC(C)(c1ccc(O)cc1)c1ccc(O)cc1"],
    ),
    "PDMS": (PDMS, ["C[Si]1(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O1"]),
}


@pytest.mark.parametrize("name", sorted(MONOMER_IDENTITY))
def test_reconstructed_monomer_is_the_real_monomer(name):
    unit, expected = MONOMER_IDENTITY[name]
    found = monomers_of(unit)
    missing = [smiles for smiles in expected if inchikey(smiles) not in found]
    assert not missing, (
        f"{name}: expected {missing} among the reconstructed monomers, got "
        f"{sorted({m for r in analyse_repeat_unit(unit).routes for m in r.monomers})}"
    )


def test_all_thirty_identities_hold_at_once():
    """The count quoted in the module docstring, pinned as a count."""
    correct = sum(
        1
        for unit, expected in MONOMER_IDENTITY.values()
        if all(inchikey(s) in monomers_of(unit) for s in expected)
    )
    assert correct == len(MONOMER_IDENTITY) == 30


def test_pdms_is_not_turned_into_a_silanone():
    """The vinyl transform is gated on both backbone termini being carbon.

    Ungated it makes the silanone ``C[Si](C)=O`` out of the PDMS repeat unit,
    which is a compound that does not sit in a bottle.  That gate is the reason
    the siloxane route exists at all, so its absence must fail a test.
    """
    from rdkit import Chem

    silanone = Chem.MolToInchiKey(Chem.MolFromSmiles("C[Si](C)=O"))
    assert silanone not in monomers_of(PDMS)


def test_polycaprolactone_is_not_cut_into_hexanoic_acid():
    """Step growth cuts a trimer, not the bare repeat unit.

    Cutting the bare unit turns polycaprolactone into hexanoic acid plus water,
    because the linkage straddles the repeat boundary and only half of it is
    present.
    """
    from rdkit import Chem

    hexanoic = Chem.MolToInchiKey(Chem.MolFromSmiles("CCCCCC(=O)O"))
    assert hexanoic not in monomers_of(PCL)


def test_aryl_ether_does_not_liberate_water():
    """Each ether oxygen may lose at most one bond.

    Ungated, both PEEK and poly(phenylene oxide) hand back free water as though
    it were a monomer.
    """
    from rdkit import Chem

    water = Chem.MolToInchiKey(Chem.MolFromSmiles("O"))
    for unit in (MONOMER_IDENTITY["PEEK"][0], PPO_TRUE):
        assert water not in monomers_of(unit)


# --------------------------------------------------------------------------
# 2. Coverage: every reference polymer was actually made
# --------------------------------------------------------------------------


def test_every_reference_polymer_is_given_a_route():
    """All 57 bundled polymers exist, so all 57 must be placeable in a chemistry."""
    missed = []
    for record in reference_polymers():
        prediction = predict(polymer_candidate(record["repeat_unit"]))
        if prediction.quantity is None:
            missed.append((record["abbreviation"], record["repeat_unit"], prediction.notes[0]))
    assert len(reference_polymers()) == 57
    assert not missed, "\n".join(f"{a}: {u} -> {why}" for a, u, why in missed)


def test_every_reference_polymer_obeys_the_contract():
    lo, hi = get_property(PROP).bounds
    for record in reference_polymers():
        prediction = predict(polymer_candidate(record["repeat_unit"]))
        assert prediction.quantity is not None
        assert lo <= prediction.quantity.value <= hi
        assert prediction.quantity.unit == get_property(PROP).canonical_unit
        # An honest wide bar beats a flattering narrow one: the Ertl score alone
        # carries a nominal 1.0, so nothing here may claim better.
        assert prediction.uncertainty.std is not None
        assert prediction.uncertainty.std >= 1.0
        assert prediction.uncertainty.basis
        assert prediction.method and "Odian" in prediction.method
        assert any("from" in note for note in prediction.notes)


# --------------------------------------------------------------------------
# 3. Direction - the checks the brief asked for
# --------------------------------------------------------------------------


def test_commodity_polyolefins_are_easy():
    assert score(PE) <= 2.5
    assert score(PS) <= 2.5


def test_ptfe_is_harder_than_polyethylene():
    """Both monomers hit the small-molecule ceiling, so the route term must carry it."""
    assert score(PTFE) - score(PE) >= 1.5


def test_poly_p_phenylene_is_hard_and_harder_than_pet():
    assert score(PPP) >= 6.0
    assert score(PPP) > score(PET)


def test_head_to_head_pvc_is_far_worse_than_pvc():
    assert score(HEAD_TO_HEAD_PVC) >= score(PVC) + 3.0


def test_poly_vinyl_alcohol_costs_more_than_poly_vinyl_acetate():
    """Vinyl alcohol is not a reagent; PVOH is PVAc hydrolysed, which is a step."""
    assert score(PVOH) > score(PVAC)
    analysis = analyse_repeat_unit(PVOH)
    assert any("hydrolysis" in route.name for route in analysis.routes)
    assert inchikey("CC(=O)OC=C") in monomers_of(PVOH)


def test_stereoregular_costs_more_than_atactic():
    assert score("[*]CC(C)[*]", tacticity=Tacticity.ISOTACTIC) > score("[*]CC(C)[*]")


def test_a_tetrasubstituted_backbone_is_refused():
    reason = refusal("[*]C(C)(C)C(C)(C)[*]")
    assert "tetrasubstituted" in reason
    assert "CC(C)=C(C)C" in reason, "the refusal must name the monomer it is about"


# --------------------------------------------------------------------------
# 4. Separation, reported rather than tuned
# --------------------------------------------------------------------------

#: The tiers live in the test file, not in the module, so they cannot be
#: mistaken for model parameters.  "Hard" means made in a laboratory or not at
#: all; poly(alpha-methylstyrene) is here because its ceiling temperature is
#: 61 C, and the head-to-head unit because its monomer does not homopolymerise.
HARD_CASES = {
    "poly(p-phenylene)": PPP,
    "polyacetylene": POLYACETYLENE,
    "poly(alpha-methylstyrene)": PAMS,
    "head-to-head PVC": HEAD_TO_HEAD_PVC,
}


def test_commodity_polymers_separate_from_hard_ones():
    """Reported as measured.  The one discordant pair is named, not hidden."""
    commodity = {
        record["abbreviation"]: score(record["repeat_unit"])
        for record in reference_polymers()
        if record["repeat_unit"] != PAMS
    }
    hard = {name: score(unit) for name, unit in HARD_CASES.items()}
    assert len(commodity) == 56 and len(hard) == 4

    discordant = [
        (c_name, h_name)
        for c_name, c in commodity.items()
        for h_name, h in hard.items()
        if c >= h
    ]
    pairs = len(commodity) * len(hard)
    assert pairs == 224
    assert len(discordant) <= 1, discordant
    # The only pair allowed to cross is the one the module docstring explains:
    # the reference file's poly(phenylene oxide) is drawn with its methyls meta
    # to the oxygen, so the phenol it reconstructs has free ortho positions.
    assert all(c == "PPE" for c, _ in discordant), discordant


def test_the_phenylene_oxide_exception_is_about_the_drawing():
    as_drawn = score(PPO_AS_DRAWN)
    as_made = score(PPO_TRUE)
    assert as_made < as_drawn
    assert as_made < min(score(unit) for unit in HARD_CASES.values())
    assert inchikey("Cc1cccc(C)c1O") in monomers_of(PPO_TRUE)  # 2,6-dimethylphenol
    assert inchikey("Cc1cc(C)cc(O)c1") in monomers_of(PPO_AS_DRAWN)  # the 3,5 isomer


# --------------------------------------------------------------------------
# 5. Refusals, tested as hard as the predictions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit, expected",
    [
        ("[*]CC", "attachment point"),
        ("CC", "attachment point"),
        ("[*]CC([*])C[*]", "attachment point"),
        ("[*]C([*])C", "same atom"),
        ("[*][Si](C)(C)[*]", "one-atom backbone"),
        ("[*]C(C[*]", "does not parse"),
    ],
)
def test_malformed_repeat_units_are_refused_with_a_reason(unit, expected):
    """Evolution mutates repeat-unit SMILES with generic operators, so these occur."""
    assert expected in refusal(unit)


def test_a_backbone_no_chemistry_places_is_refused():
    reason = refusal("[*]CCCCCC[*]")
    assert "no polymerisation chemistry matched" in reason
    assert "nobody can make this" in reason


def test_an_unstrained_six_ring_is_refused():
    """Tetrahydropyran does not polymerise; a six-ring needs a carbonyl to pay."""
    reason = refusal("[*]CCCCCO[*]")
    assert "six-membered" in reason and "Odian" in reason


@pytest.mark.parametrize(
    "name, unit",
    [
        ("polyimide (no dianhydride route)", "[*]c1ccc(cc1)N2C(=O)c3ccc([*])cc3C2=O"),
        ("poly(phenylene sulfide) (no sulfide SNAr)", "[*]c1ccc(S[*])cc1"),
        ("polyoctenamer (no ROMP or ADMET)", "[*]CCCCCCC=C[*]"),
        ("polyethylene drawn on four carbons", "[*]CCCC[*]"),
        ("poly(ethylene oxide) drawn doubled", "[*]CCOCCO[*]"),
    ],
)
def test_the_known_gaps_refuse_rather_than_guess(name, unit):
    """These are real polymers this expert cannot place, and it says so.

    A false refusal costs a candidate its whole utility on this objective, which
    is the right direction to fail in but is not free.  Each is listed in the
    module docstring; pinning them here means a later recogniser that closes one
    of these gaps has to update the claim as well as the code.
    """
    reason = refusal(unit)
    assert len(reason) > 40, name


def test_a_network_is_refused():
    candidate = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(
            monomers=(MonomerUnit(smiles=PE),),
            topology=PolymerTopology.NETWORK,
            crosslink_density=Quantity(value=100.0, unit="mol/m^3"),
        ),
    )
    assert "crosslinker and the cure chemistry" in refusal(candidate)


def test_a_declared_crosslinker_is_refused():
    candidate = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(
            monomers=(
                MonomerUnit(smiles=PE, mole_fraction=1.0),
                MonomerUnit(
                    smiles="[*]CC(C)[*]", mole_fraction=0.0, role=MonomerRole.CROSSLINKER
                ),
            )
        ),
    )
    assert "cure chemistry" in refusal(candidate)


def test_a_copolymer_of_incompatible_mechanisms_is_refused():
    candidate = polymer_candidate([(PVC, 0.5), (PCL, 0.5)])
    reason = refusal(candidate)
    assert "incompatible polymerisation mechanisms" in reason
    assert "chain growth" in reason


def test_an_unactivated_aryl_ether_that_is_not_a_phenylene_oxide_is_refused():
    """No electron-withdrawing group means no SNAr, and this is no phenol either."""
    reason = refusal("[*]Oc1ccc(cc1)C(C)(C)c1ccc([*])cc1")
    assert "electron-withdrawing" in reason


def test_a_compatible_copolymer_is_scored_and_flagged():
    candidate = polymer_candidate([(PS, 0.5), (PVC, 0.5)])
    prediction = predict(candidate)
    assert prediction.quantity is not None
    assert prediction.quantity.value > max(score(PS), score(PVC))
    assert any("reactivity-ratio" in note for note in prediction.notes)
    assert prediction.applicability.score <= 0.4


# --------------------------------------------------------------------------
# 6. The Ertl artefact this expert corrects, documented as a test
# --------------------------------------------------------------------------


def test_the_small_molecule_bias_is_real_and_measured():
    """If RDKit's fragment table changes under us, the ceiling's basis changes too."""
    assert ertl_score("CC") - ertl_score("CCCCCC") > 1.0
    assert ertl_score("C=C") > ertl_score("C=CCCCCCCCC")
    assert ertl_score("C=C") > 4.0 > ertl_score("C=Cc1ccccc1")


def test_the_ceiling_only_lowers_a_score():
    capped, was_capped, note = monomer_term("C=C")
    assert was_capped and capped == SMALL_MOLECULE_CEILING and note
    raw, was_capped, _ = monomer_term("C=Cc1ccccc1")
    assert not was_capped and raw == pytest.approx(ertl_score("C=Cc1ccccc1"))
    assert raw < SMALL_MOLECULE_CEILING


def test_the_ceiling_does_not_rescue_an_unstable_small_molecule():
    """A three-ring carrying a carbonyl is an alpha-lactone, not a reagent."""
    value, was_capped, _ = monomer_term("CC1OC1=O")
    assert not was_capped
    assert value > SMALL_MOLECULE_CEILING


def test_a_ceiling_widens_the_error_bar():
    plain = predict(polymer_candidate(PS))
    capped = predict(polymer_candidate(PE))
    assert capped.uncertainty.std > plain.uncertainty.std


# --------------------------------------------------------------------------
# 7. Not the molecule expert, and not a silent substitution
# --------------------------------------------------------------------------


def test_this_expert_covers_polymers_and_polymer_blends_but_not_molecules():
    """The mixture case was added deliberately and is not a widening of scope.

    A polymer BLEND is polymers, and nothing else could answer whether one can
    be made: mixture_thermal covers the property for a formulation but
    delegates to the molecular panel, which correctly refuses a polymer
    component. A blend whose components are not all polymers is still refused
    here - see the solution test below - so this is not a general mixture
    expert, and the molecule case stays where it belongs.
    """
    assert EXPERT.covers(PROP, MaterialClass.POLYMER)
    assert EXPERT.covers(PROP, MaterialClass.MIXTURE)
    assert not EXPERT.covers(PROP, MaterialClass.MOLECULE)
    assert EXPERT.family is PropertyFamily.FEASIBILITY
    assert EXPERT.dependencies == frozenset()


def test_the_repeat_unit_is_not_scored_as_a_molecule():
    """The forbidden shortcut, asserted to give a different answer.

    Handing the repeat-unit SMILES to the molecule expert is the substitution
    this module exists to avoid; if the two ever agreed by construction, this
    expert would be decoration.
    """
    molecule_expert = SynthesisFeasibilityExpert()
    if not molecule_expert.is_available():  # pragma: no cover
        pytest.skip("SA_Score unavailable")
    for unit in (PE, PET, PPP):
        candidate = molecule_candidate(unit)
        naive = molecule_expert.predict(
            PredictionRequest(
                candidate=candidate, properties=PROPERTIES, conditions=candidate.conditions
            )
        )[0]
        assert naive.quantity is not None
        assert abs(naive.quantity.value - score(unit)) > 0.5


def test_the_minimum_over_routes_is_taken():
    """A polymer needs one route, so the easiest matched route sets the score."""
    analysis = analyse_repeat_unit(PCL)
    mechanisms = {route.mechanism for route in analysis.routes}
    assert {Mechanism.RING_OPENING, Mechanism.STEP_GROWTH} <= mechanisms
    best = min(score_route(route)[0] for route in analysis.routes)
    assert score(PCL) == pytest.approx(best)


# --------------------------------------------------------------------------
# 8. Units, domain and the pinned measurements
# --------------------------------------------------------------------------


def test_the_carothers_charge_reads_a_molar_mass_in_any_unit():
    """The one real unit conversion in this expert, pinned both ways.

    A degree of polymerisation is a molar mass divided by a repeat-unit mass, so
    the charge has to be identical whether the target is stated in kg/mol or in
    g/mol, and a target of 120 *g/mol* - less than one repeat unit - must not be
    charged at all.
    """
    in_kg = score(PET, number_average_molar_mass=Quantity(value=120.0, unit="kg/mol"))
    in_g = score(PET, number_average_molar_mass=Quantity(value=120000.0, unit="g/mol"))
    assert in_kg == pytest.approx(in_g)

    unstated = score(PET)
    tiny = score(PET, number_average_molar_mass=Quantity(value=120.0, unit="g/mol"))
    assert tiny == pytest.approx(unstated)
    assert in_kg > unstated

    # Chain growth sets molar mass by the initiator ratio, not by conversion,
    # so the same demanding target costs polyethylene nothing.
    assert score(PE, number_average_molar_mass=Quantity(value=400.0, unit="kg/mol")) == (
        pytest.approx(score(PE))
    )


def test_silicon_is_flagged_out_of_domain_rather_than_corrected():
    prediction = predict(polymer_candidate(PDMS))
    assert prediction.status is PredictionStatus.OUT_OF_DOMAIN
    assert not prediction.applicability.in_domain
    assert any("Si" in warning for warning in prediction.applicability.warnings)
    # Widened, not fixed: PDMS is a bulk commodity scoring like a research
    # compound, and the honest response is a wide bar plus the flag.
    assert prediction.uncertainty.std > 2.0


def test_a_non_linear_topology_is_flagged_out_of_domain():
    candidate = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(
            monomers=(MonomerUnit(smiles=PE),), topology=PolymerTopology.STAR
        ),
    )
    prediction = predict(candidate)
    assert prediction.quantity is not None
    assert not prediction.applicability.in_domain
    assert prediction.uncertainty.std > predict(polymer_candidate(PE)).uncertainty.std


@pytest.mark.parametrize(
    "unit, expected",
    [
        (PE, 2.00),
        (PS, 1.55),
        (PET, 2.50),
        (PTFE, 3.60),
        (PPP, 6.42),
        (PDMS, 5.08),
        (POLYACETYLENE, 5.00),
        (HEAD_TO_HEAD_PVC, 5.50),
    ],
)
def test_measured_values_are_pinned(unit, expected):
    """The numbers quoted in the module docstring, as this installation produces them."""
    assert score(unit) == pytest.approx(expected, abs=0.01)


def test_the_upper_clamp_is_recorded_rather_than_silent():
    """A value above ten is reported as clamped, not quietly squashed."""
    hard = "[*]c1sc2c(c1)[nH]c1c2c(c([*])s1)C(F)(F)F"
    other = "[*]c1cc2c(cc1[*])[nH]c1c2sc2c1cc1c(c2)OCO1"
    candidate = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(
            monomers=(
                MonomerUnit(smiles=hard, mole_fraction=0.5),
                MonomerUnit(smiles=other, mole_fraction=0.5),
            ),
            tacticity=Tacticity.ISOTACTIC,
        ),
    )
    prediction = predict(candidate)
    assert prediction.quantity is not None
    assert prediction.quantity.value == pytest.approx(10.0)
    assert any("clamped" in note for note in prediction.notes)


def test_unavailable_backend_is_reported_rather_than_guessed(monkeypatch):
    monkeypatch.setattr(
        "formulate.experts.polymer_feasibility._sascorer", lambda: None, raising=True
    )
    assert not EXPERT.is_available()
    assert "SA_Score" in EXPERT.unavailable_reason()
    assert "SA_Score" in refusal(PE)


# --------------------------------------------------------------------------
# 9. Fuzz: a mutated repeat unit gets a number or a reason, never an exception
# --------------------------------------------------------------------------


def test_mutated_repeat_units_never_crash_and_never_guess():
    random.seed(20260919)
    alphabet = list("CCCCNOSF()=#[*]1234ClBr")
    seeds = [unit for unit, _ in MONOMER_IDENTITY.values()]
    lo, hi = get_property(PROP).bounds
    scored = refused = 0
    problems: list[str] = []

    for _ in range(200):
        text = list(random.choice(seeds))
        for _ in range(random.randint(1, 4)):
            choice = random.random()
            position = random.randrange(len(text))
            if choice < 0.4:
                text[position] = random.choice(alphabet)
            elif choice < 0.7:
                text.insert(position, random.choice(alphabet))
            elif len(text) > 2:
                text.pop(position)
        unit = "".join(text)
        try:
            prediction = predict(polymer_candidate(unit))
        except Exception as exc:  # pragma: no cover - this is what the test forbids
            problems.append(f"{unit!r} raised {type(exc).__name__}: {exc}")
            continue
        if prediction.quantity is None:
            refused += 1
            if prediction.status is PredictionStatus.FAILED:
                problems.append(f"{unit!r} failed internally: {prediction.notes}")
            if not (prediction.notes and prediction.notes[0].strip()):
                problems.append(f"{unit!r} refused with no reason")
        else:
            scored += 1
            if not lo <= prediction.quantity.value <= hi:
                problems.append(f"{unit!r} scored {prediction.quantity.value}")
            if prediction.uncertainty.std is None or prediction.uncertainty.std < 1.0:
                problems.append(f"{unit!r} claimed std {prediction.uncertainty.std}")

    assert not problems, "\n".join(problems[:10])
    assert scored + refused == 200
    assert refused > 0, "the fuzzer produced nothing malformed, so it tested nothing"


def test_the_json_reference_set_is_the_one_this_was_measured_on():
    """The measured numbers above are against this file; a swap invalidates them."""
    from importlib import resources

    payload = json.loads(
        resources.files("formulate.data").joinpath("reference_polymers.json").read_text()
    )
    assert len(payload["polymers"]) == 57
    units = {record["repeat_unit"] for record in payload["polymers"]}
    assert PPO_AS_DRAWN in units
    assert PPO_TRUE not in units


# --------------------------------------------------------------------------
# 10. Gates found by adversarial review: a wrong monomer is a confident wrong
#     number, so each of these produced one before the gate existed.
# --------------------------------------------------------------------------


def routes_of(unit: str):
    analysis = analyse_repeat_unit(unit)
    assert analysis.failure is None, analysis.failure
    return analysis.routes


@pytest.mark.parametrize(
    "name, unit",
    [
        ("N-linked pyrrole", "[*]n1ccc([*])c1"),
        ("N-linked triazole", "[*]n1cc([*])nn1"),
        ("aromatic-perceived polyimide", "[*]N1C(=O)c2ccc3c(c2C1=O)C(=O)N(C3=O)c1ccc([*])cc1"),
    ],
)
def test_aryl_aryl_coupling_will_not_brominate_an_aromatic_heteroatom(name, unit):
    """A backbone joined through an aromatic nitrogen is not a Yamamoto monomer.

    Ungated, the transform wrote the attachment points as bromine wherever the
    path was aromatic, so an N-arylene came back as ``Brc1ccn(Br)c1`` - an
    N-bromo compound, which is a brominating agent rather than a feedstock - and
    the polymer was scored on it.  A carbon-nitrogen bond is made by amination,
    which this module does not recognise, so the honest answer is a refusal.
    """
    from rdkit import Chem

    for route in routes_of(unit):
        for smiles in route.monomers:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() != 35:
                    continue
                neighbour = atom.GetNeighbors()[0]
                assert neighbour.GetAtomicNum() == 6, f"{name}: {smiles} is an N-Br compound"
    assert len(refusal(unit)) > 40, name


def test_a_ring_heteroatom_off_the_attachment_points_is_still_fine():
    """The gate is on the attachment atoms, not on the whole path.

    2,5-dibromothiophene is the right monomer for polythiophene and its sulfur
    lies on the shortest path between the attachment points, so a gate written
    over the path rather than over its ends would have thrown it away.
    """
    monomers = monomers_of("[*]c1ccc([*])s1")
    assert inchikey("Brc1ccc(Br)s1") in monomers
    assert inchikey("Brc1ccc(Br)cc1") in monomers_of(PPP)


@pytest.mark.parametrize(
    "name, unit",
    [
        ("phosphorus backbone", "[*]P(C)(C)O[*]"),
        ("germanium backbone", "[*][Ge](C)(C)O[*]"),
        ("tin backbone", "[*][Sn](C)(C)O[*]"),
        ("boron-nitrogen backbone", "[*]B(C)N(C)[*]"),
    ],
)
def test_ring_opening_is_confined_to_the_rings_odian_argues_about(name, unit):
    """Ring strain versus entropy is an argument about organic rings.

    Guarded for silicon alone, this transform closed a phosphorus, germanium,
    tin or boron backbone into a four-membered inorganic ring nobody has put in
    a bottle and then charged the cheapest route in the table for it:
    ``[*]P(C)(C)O[*]`` scored 2.5 *in domain*, level with poly(ethylene
    terephthalate), which is a flattering number on a structure with no route.
    """
    assert not routes_of(unit), name
    assert len(refusal(unit)) > 40, name


def test_silicon_still_leaves_by_its_own_recogniser():
    """The element gate must not take the siloxane route down with it."""
    names = {route.name for route in routes_of(PDMS)}
    assert any("cyclosiloxane" in name for name in names), names
    assert inchikey("C[Si]1(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O1") in monomers_of(PDMS)


@pytest.mark.parametrize(
    "name, unit, bogus",
    [
        ("disulfide", "[*]CCSS[*]", "C1CSS1"),
        ("hydroxylamine ether", "[*]NOC[*]", "C1NO1"),
    ],
)
def test_no_ring_is_closed_across_a_heteroatom_heteroatom_bond(name, unit, bogus):
    """What ring-opens is a heteroatom flanked by carbon (Odian ch. 7).

    Closed across an S-S or an N-O bond instead, the transform produced
    1,2-dithietane and oxaziridine - a pair of redox reagents, not monomers -
    and both scored 2.5, easier than poly(ethylene terephthalate), because the
    small-molecule ceiling then capped their Ertl scores at 2.0.
    """
    assert inchikey(bogus) not in monomers_of(unit), name
    assert not routes_of(unit), name
    assert len(refusal(unit)) > 40, name


def test_an_aromatised_three_ring_carbonyl_is_still_an_alpha_lactone():
    """RDKit calls the oxiranedione aromatic, so the gate cannot say ``[CX3]``.

    The alpha-lactone gate is what sends polyglycolide to glycolide instead of
    to a ring that does not exist.  Written over aliphatic carbon only, it let
    the doubly-carbonylated three-rings an oxalate backbone closes to -
    ``O=c1oc1=O`` and its aza analogue - straight through, and *stopped the
    cyclic dimer being tried at all*, because the first ring that matches wins.
    """
    unit = "[*]C(=O)C(=O)O[*]"
    names = {route.name for route in routes_of(unit)}
    assert not any("3-membered" in name for name in names), names
    assert inchikey("O=c1oc1=O") not in monomers_of(unit)
    # The dimer the gate exists to fall through to is now reached.
    assert any("cyclic dimer" in name for name in names), names
    # And the case it was written for still behaves.
    assert inchikey("O=C1COC(=O)CO1") in monomers_of("[*]OCC(=O)[*]")


def test_a_urea_linkage_pays_for_the_c1_unit_it_needs():
    """Urea is discarded as a trivial fragment, so it must be charged instead.

    A polyurea needs the same diisocyanate a polyurethane does.  The carbonate
    (two oxygens on the acyl) and urethane (one of each) cases were charged for
    it; the urea case - two nitrogens - was not, so the urea cut out of the
    backbone was neither reported as a monomer nor paid for.
    """
    urea = [r for r in routes_of("[*]Nc1ccc(NC(=O)[*])cc1") if r.mechanism is Mechanism.STEP_GROWTH]
    amide = [
        r
        for r in routes_of("[*]NCCCCCCNC(=O)CCCCC(=O)[*]")
        if r.mechanism is Mechanism.STEP_GROWTH
    ]
    assert urea and amide
    assert urea[0].penalty == pytest.approx(amide[0].penalty + 1.0)
    assert any("urea" in note for note in urea[0].notes)
    # The C1 unit is paid for, not passed off as a monomer.
    assert inchikey("NC(N)=O") not in {inchikey(m) for m in urea[0].monomers}
    assert inchikey("Nc1ccc(N)cc1") in {inchikey(m) for m in urea[0].monomers}


def test_a_copolymer_never_claims_a_narrower_bar_than_one_of_its_components():
    """Pairing the hardest unit's value with only the hardest unit's bar under-claims.

    Which unit is the hardest is itself uncertain, so adding a comonomer cannot
    make the answer more certain than the comonomer's own answer was.  Measured
    before the fix: poly(vinyl alcohol) alone 3.00 +/- 1.58, PTFE alone
    3.60 +/- 1.50, the pair 4.10 +/- 1.80 - narrower than the 1.87 the wider
    component implies.
    """
    pvoh = predict(polymer_candidate(PVOH))
    ptfe = predict(polymer_candidate(PTFE))
    pair = predict(polymer_candidate([(PVOH, 0.5), (PTFE, 0.5)]))
    assert pair.quantity is not None
    widest = max(pvoh.uncertainty.std, ptfe.uncertainty.std)
    assert widest == pytest.approx(pvoh.uncertainty.std)  # the softer unit, not the harder
    assert pair.uncertainty.std == pytest.approx(math.hypot(widest, 1.0))
    # The score is still the hardest unit's, plus the copolymer charge.
    assert pair.quantity.value == pytest.approx(ptfe.quantity.value + 0.5)


def test_the_generic_refusal_says_what_to_check_next():
    """A refusal that only says "no" is not actionable.

    The commonest cause of this branch is not an exotic backbone but a repeat
    unit written on a whole multiple of its own period, which nothing here
    reduces, so the message has to name that before anything else.
    """
    reason = refusal("[*]CCCC[*]")
    assert "smallest period" in reason
    assert "ring-opening" in reason and "polycondensation" in reason
    assert "nobody can make this" in reason


# -- a blend has to be makeable too ----------------------------------------


def _blend(*units):
    """A polymer blend candidate from (repeat unit, kg/mol, mass fraction)."""
    from formulate.core.candidate import (
        Candidate,
        ComponentRole,
        FractionBasis,
        MaterialClass,
        MixtureComponent,
        MixtureSpec,
        MonomerUnit,
        PolymerSpec,
    )
    from formulate.core.conditions import Conditions
    from formulate.core.quantity import Quantity

    components = tuple(
        MixtureComponent(
            role=ComponentRole.SOLUTE,
            fraction=fraction,
            polymer=PolymerSpec(
                monomers=(MonomerUnit(smiles=smiles),),
                number_average_molar_mass=Quantity(value=mass, unit="kg/mol"),
            ),
        )
        for smiles, mass, fraction in units
    )
    return Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(components=components, basis=FractionBasis.MASS),
        conditions=Conditions.standard(),
    )


def _score(candidate):
    from formulate.core.conditions import Conditions
    from formulate.experts.base import PredictionRequest
    from formulate.experts.polymer_feasibility import PolymerFeasibilityExpert

    request = PredictionRequest(
        candidate=candidate,
        properties=frozenset({"synthetic_accessibility"}),
        conditions=Conditions.standard(),
    )
    return PolymerFeasibilityExpert().predict(request)[0]


PE, PS, PTFE = "[*]CC[*]", "[*]CC(c1ccccc1)[*]", "[*]C(F)(F)C(F)(F)[*]"


@requires_rdkit
def test_a_polymer_blend_can_be_asked_whether_it_can_be_made():
    """Nothing could, before. mixture_thermal covers synthetic_accessibility for
    a formulation but delegates to the MOLECULAR panel, which correctly refuses
    a polymer component - "a polymer has no critical point and no boiling
    point". So a hard accessibility requirement eliminated every blend in the
    pool for a value nobody could compute, which reads as "no blend can be
    made" and is not that at all."""
    prediction = _score(_blend((PE, 120.0, 0.7), (PE, 2.0, 0.3)))
    assert prediction.quantity is not None
    assert 1.0 <= prediction.quantity.value <= 10.0


@requires_rdkit
def test_blending_a_polymer_with_itself_does_not_make_it_harder_to_synthesise():
    """A bimodal blend is one chemistry at two chain lengths. Blending is a
    processing step; the monomer is the same monomer."""
    from formulate.core.candidate import Candidate, MaterialClass, MonomerUnit, PolymerSpec
    from formulate.core.conditions import Conditions
    from formulate.core.quantity import Quantity

    single = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(
            monomers=(MonomerUnit(smiles=PE),),
            number_average_molar_mass=Quantity(value=120.0, unit="kg/mol"),
        ),
        conditions=Conditions.standard(),
    )
    alone = _score(single)
    blended = _score(_blend((PE, 120.0, 0.7), (PE, 2.0, 0.3)))
    assert blended.quantity.value == pytest.approx(alone.quantity.value)


@requires_rdkit
def test_the_hardest_component_gates_the_blend():
    """An average would let an easy component hide a hard one."""
    easy = _score(_blend((PE, 50.0, 0.5), (PS, 50.0, 0.5)))
    hard = _score(_blend((PE, 50.0, 0.5), (PTFE, 50.0, 0.5)))
    assert hard.quantity.value > easy.quantity.value


@requires_rdkit
def test_a_blend_does_not_claim_a_narrower_bar_than_its_components():
    blended = _score(_blend((PE, 50.0, 0.5), (PTFE, 50.0, 0.5)))
    from formulate.core.candidate import Candidate, MaterialClass, MonomerUnit, PolymerSpec
    from formulate.core.conditions import Conditions
    from formulate.core.quantity import Quantity

    widest = 0.0
    for smiles in (PE, PTFE):
        one = _score(
            Candidate(
                material_class=MaterialClass.POLYMER,
                polymer=PolymerSpec(
                    monomers=(MonomerUnit(smiles=smiles),),
                    number_average_molar_mass=Quantity(value=50.0, unit="kg/mol"),
                ),
                conditions=Conditions.standard(),
            )
        )
        widest = max(widest, one.uncertainty.std)
    assert blended.uncertainty.std == pytest.approx(widest, rel=1e-6)


@requires_rdkit
def test_a_polymer_dissolved_in_a_solvent_is_refused_rather_than_charged_for_the_solvent():
    """What has to be MADE is the polymer; the solvent is bought. Scoring the
    formulation by the same rule would charge it for something nobody
    synthesises."""
    from formulate.core.candidate import (
        Candidate,
        ComponentRole,
        FractionBasis,
        MaterialClass,
        MixtureComponent,
        MixtureSpec,
        MoleculeSpec,
        MonomerUnit,
        PolymerSpec,
    )
    from formulate.core.conditions import Conditions
    from formulate.core.prediction import PredictionStatus
    from formulate.core.quantity import Quantity

    solution = Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=(
                MixtureComponent(
                    role=ComponentRole.SOLUTE,
                    fraction=0.2,
                    polymer=PolymerSpec(
                        monomers=(MonomerUnit(smiles=PS),),
                        number_average_molar_mass=Quantity(value=100.0, unit="kg/mol"),
                    ),
                ),
                MixtureComponent(
                    role=ComponentRole.SOLVENT,
                    fraction=0.8,
                    molecule=MoleculeSpec(smiles="CC(C)=O"),
                ),
            ),
            basis=FractionBasis.MASS,
        ),
        conditions=Conditions.standard(),
    )
    prediction = _score(solution)
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "solvent is bought" in " ".join(prediction.notes)
