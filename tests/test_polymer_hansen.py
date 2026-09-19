"""Hoftyzer-Van Krevelen solubility parameters for a polymer repeat unit.

Two things are asserted here, and the second matters as much as the first.

*What the table is worth.*  The increments are transcribed, not fitted, so the
only figure worth testing is agreement with data the transcription never
touched: the measured Hansen triples of the bundled reference liquids, where
the reference quantity is exactly the quantity the method predicts, and the
five published polymer spheres, where it is not.  Both thresholds sit just
above the measured numbers so that the tests fail on a regression rather than
on noise.

*What it refuses.*  In this repository a refusal is a feature.  A repeat unit
containing an amide, a fluorine, a fused ring or two halogens on one atom has
no honest answer here, and every one of those paths is tested as hard as the
predictions are, because the cheapest way to "improve" this expert would be to
let one of them through with a mid-range guess.
"""

from __future__ import annotations

import json
import math

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MonomerUnit,
    PolymerSpec,
    PolymerTopology,
    molecule_candidate,
    polymer_candidate,
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import PredictionStatus
from formulate.core.properties import PROPERTY_REGISTRY
from formulate.core.quantity import Quantity
from formulate.experts.base import PredictionRequest
from formulate.experts.dissolution import SOLUBILITY_SPHERES
from formulate.experts.polymer import PolymerDensityExpert, reference_polymers
from formulate.experts.polymer_hansen import (
    HVK_GROUPS,
    METHOD_STD_MPA_SQRT,
    RING_INCREMENTS,
    TEMPERATURE_WINDOW_K,
    ChainCohesion,
    PolymerHansenExpert,
    RepeatUnitRefused,
    chain_cohesion,
    repeat_unit_cohesion,
)

pytestmark = requires_rdkit

ROOM = Conditions.standard()
HOT = Conditions(temperature=Quantity(value=450.0, unit="K"))
NO_TEMPERATURE = Conditions(pressure=Quantity(value=1.0, unit="atm"))

PROPERTIES = frozenset(
    {
        "hansen_dispersion",
        "hansen_polar",
        "hansen_hydrogen_bonding",
        "hildebrand_solubility_parameter",
    }
)

#: Repeat units of the polymers whose spheres ``dissolution.py`` tabulates.
#: Polyamide 66 is here so that its refusal is tested, not assumed; cellulose
#: acetate is absent because it has no repeat unit at a fixed degree of
#: substitution, which is a statement about the material rather than a gap.
SPHERE_UNITS = {
    "polystyrene": "[*]CC(c1ccccc1)[*]",
    "poly(methyl methacrylate)": "[*]CC(C)(C(=O)OC)[*]",
    "poly(vinyl chloride)": "[*]CC(Cl)[*]",
    "poly(vinyl acetate)": "[*]CC(OC(C)=O)[*]",
    "polycarbonate": "[*]Oc1ccc(cc1)C(C)(C)c1ccc(cc1)OC(=O)[*]",
}

PS = SPHERE_UNITS["polystyrene"]
PE = "[*]CC[*]"
PVC = SPHERE_UNITS["poly(vinyl chloride)"]
NYLON66 = "[*]NCCCCCCNC(=O)CCCCC(=O)[*]"

COMPONENTS = ("hansen_dispersion", "hansen_polar", "hansen_hydrogen_bonding")


# -- plumbing ---------------------------------------------------------------


def _predict(candidate, *, conditions=ROOM, with_density=True):
    """Run the expert behind a real upstream amorphous density, as dispatch would."""
    context = {}
    if with_density:
        density_request = PredictionRequest(
            candidate=candidate,
            properties=frozenset({"amorphous_density"}),
            conditions=conditions,
        )
        context = {
            p.property: p for p in PolymerDensityExpert().predict(density_request)
        }
    request = PredictionRequest(
        candidate=candidate,
        properties=PROPERTIES,
        conditions=conditions,
        context=context,
    )
    return {p.property: p for p in PolymerHansenExpert().predict(request)}


def _mpa(prediction):
    assert prediction.quantity is not None, prediction.notes
    return prediction.quantity.to("MPa^0.5").value


def _sigma(prediction):
    std = prediction.uncertainty.converted(prediction.quantity.unit, "MPa^0.5").std
    assert std is not None
    return std


def _triple(unit_smiles, conditions=ROOM):
    preds = _predict(polymer_candidate(unit_smiles, conditions=conditions))
    return tuple(_mpa(preds[c]) for c in COMPONENTS)


# -- the group table against measured liquids -------------------------------
#
# The acceptance gate.  Nothing in the table was fitted to these compounds;
# they are what stands behind a transcription, and an increment that failed
# here was dropped rather than tuned.


def _liquid_agreement():
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    from formulate.experts.hansen import hansen_triple
    from formulate.experts.polymer_hansen import _INCREMENTS, _count_groups, _structural_refusal

    path = (
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "src/formulate/data/reference_compounds.json"
    )
    rows = []
    for compound in json.loads(path.read_text())["compounds"]:
        density = compound.get("density_25c")
        measured = hansen_triple(compound["smiles"])
        if density is None or measured is None:
            continue
        mol = Chem.MolFromSmiles(compound["smiles"])
        try:
            counts = _count_groups(mol)
        except RepeatUnitRefused:
            continue
        if _structural_refusal(mol) is not None:
            continue
        volume = Descriptors.MolWt(mol) / density
        f_d = sum(n * _INCREMENTS[k].f_d for k, n in counts.items())
        f_p2 = sum(n * _INCREMENTS[k].f_p ** 2 for k, n in counts.items())
        e_h = sum(n * _INCREMENTS[k].e_h for k, n in counts.items())
        predicted = (
            f_d / volume,
            math.sqrt(f_p2) / volume,
            math.sqrt(e_h / volume),
        )
        rows.append((compound["name"], predicted, tuple(v / 1000.0 for v in measured)))
    return rows


def test_group_table_agrees_with_measured_liquid_hansen_parameters():
    rows = _liquid_agreement()
    # 35 of the 50 bundled compounds are inside the shipped group set; the
    # other fifteen contain nitrogen, sulfur, a geminal dihalide, a bare
    # benzene or a lactone, and are refused rather than scored.
    assert len(rows) >= 33, f"coverage fell to {len(rows)}"

    def rmse(index):
        return math.sqrt(
            sum((p[index] - m[index]) ** 2 for _, p, m in rows) / len(rows)
        )

    # Measured 0.67 / 1.21 / 1.52; the thresholds sit just above.
    assert rmse(0) <= 0.80
    assert rmse(1) <= 1.40
    assert rmse(2) <= 1.75

    def total(t):
        return math.sqrt(sum(v * v for v in t))

    overall = math.sqrt(
        sum((total(p) - total(m)) ** 2 for _, p, m in rows) / len(rows)
    )
    assert overall <= 1.10, overall  # measured 0.93


def test_acetic_acid_is_not_scored_as_a_ketone_plus_a_hydroxyl():
    """Group priority is chemistry, not ordering taste.

    If -COOH did not claim its own carbonyl first, acetic acid would pick up
    the 20 kJ/mol hydroxyl term and come out near delta_h 25 instead of 13.
    """
    rows = {name: (p, m) for name, p, m in _liquid_agreement()}
    predicted, measured = rows["acetic acid"]
    assert predicted[2] == pytest.approx(13.2, abs=0.5)
    assert measured[2] == pytest.approx(13.5, abs=0.5)


# -- the polymers it exists for ---------------------------------------------


def test_polystyrene_reproduces_the_value_validated_here():
    """Pinned against the number measured during validation, not a wish.

    18.4 / 1.1 / 0.0 MPa^0.5 at the predicted amorphous density of
    1074 kg/m^3.  The handbook *sphere centre* is (21.3, 5.8, 4.3); the two
    are different quantities and this test asserts the one this expert claims
    to produce.
    """
    dispersion, polar, hydrogen = _triple(PS)
    assert dispersion == pytest.approx(18.4, abs=0.2)
    assert polar == pytest.approx(1.1, abs=0.2)
    assert hydrogen == pytest.approx(0.0, abs=0.05)


def test_polymer_triples_agree_with_the_published_spheres_within_the_stated_bar():
    errors = {c: [] for c in COMPONENTS}
    inside = 0
    for name, unit in SPHERE_UNITS.items():
        preds = _predict(polymer_candidate(unit, conditions=ROOM))
        for component, reference in zip(COMPONENTS, SOLUBILITY_SPHERES[name].centre):
            prediction = preds[component]
            deviation = _mpa(prediction) - reference
            errors[component].append(deviation)
            if abs(deviation) <= _sigma(prediction):
                inside += 1

    # Measured RMS 2.77 / 4.39 / 3.13 through the predicted density.
    limits = {"hansen_dispersion": 3.0, "hansen_polar": 4.6, "hansen_hydrogen_bonding": 3.4}
    for component, limit in limits.items():
        values = errors[component]
        rms = math.sqrt(sum(v * v for v in values) / len(values))
        assert rms <= limit, f"{component} RMS {rms:.2f}"

    # The point of the widened bar: it has to cover the disagreement it was
    # derived from.  Sized at the raw RMS this was 7 of 15.
    assert inside >= 13, f"only {inside} of 15 components inside their own one sigma"


def test_the_stated_bar_is_wider_than_the_liquid_error_and_says_why():
    prediction = _predict(polymer_candidate(PS))["hansen_polar"]
    assert _sigma(prediction) >= METHOD_STD_MPA_SQRT["hansen_polar"]
    basis = prediction.uncertainty.basis
    assert "sphere centre" in basis
    assert "propagated" in basis


def test_notes_warn_against_substituting_this_for_a_solubility_sphere():
    preds = _predict(polymer_candidate(PS))
    notes = " ".join(preds["hansen_dispersion"].notes)
    assert "sphere centre" in notes
    assert "crystallinity" in notes


# -- dimensions -------------------------------------------------------------


def test_canonical_value_is_the_megapascal_root_figure_times_a_thousand():
    """The x1000 trap ``dissolution.py`` had to document cannot recur here.

    These are square roots of a pressure, so MPa^0.5 to Pa^0.5 is a factor of
    a thousand, not a million.  Getting it wrong made every real solvent for
    polystyrene score as a non-solvent.
    """
    prediction = _predict(polymer_candidate(PS))["hansen_dispersion"]
    canonical = prediction.quantity.to_canonical()
    assert str(canonical.unit).startswith("pascal")
    assert canonical.value == pytest.approx(_mpa(prediction) * 1000.0, rel=1e-9)
    assert 1.0e4 < canonical.value < 3.0e4

    registry_unit = PROPERTY_REGISTRY["hansen_dispersion"].canonical_unit
    assert prediction.quantity.to(registry_unit).value == pytest.approx(canonical.value)


def test_no_molar_volume_is_offered_under_a_saturated_liquid_name():
    """A glassy repeat unit is not a saturated liquid, and must not share a column.

    ``properties.py`` refuses to call an amorphous polymer density
    ``liquid_density`` for this exact reason.  ``M/rho`` is still what every
    parameter here is divided by, so it travels as provenance and as a note -
    where nothing ranks on it - rather than as ``molar_volume_liquid``.
    """
    expert = PolymerHansenExpert()
    assert "molar_volume_liquid" not in expert.supported_properties

    request = PredictionRequest(
        candidate=polymer_candidate(PS, conditions=ROOM),
        properties=frozenset({"molar_volume_liquid"}),
        conditions=ROOM,
    )
    assert expert.predict(request) == []

    prediction = _predict(polymer_candidate(PS))["hansen_dispersion"]
    volume = prediction.provenance.parameters["repeat_unit_molar_volume_cm3"]
    assert volume == pytest.approx(97.0, rel=0.02)
    assert "not a saturated liquid" in " ".join(prediction.notes)


def test_a_density_without_an_error_bar_still_carries_the_group_tables_own_bar():
    """The method error does not come from the density, so it cannot vanish with it."""
    from formulate.core.quantity import Uncertainty

    candidate = polymer_candidate(PS, conditions=ROOM)
    density_request = PredictionRequest(
        candidate=candidate,
        properties=frozenset({"amorphous_density"}),
        conditions=ROOM,
    )
    density = PolymerDensityExpert().predict(density_request)[0]
    blind = density.model_copy(
        update={"uncertainty": Uncertainty.unknown("upstream stated none")}
    )
    request = PredictionRequest(
        candidate=candidate,
        properties=PROPERTIES,
        conditions=ROOM,
        context={"amorphous_density": blind},
    )
    preds = {p.property: p for p in PolymerHansenExpert().predict(request)}

    for prop in sorted(PROPERTIES):
        assert preds[prop].status is PredictionStatus.OK
        assert _sigma(preds[prop]) == pytest.approx(
            METHOD_STD_MPA_SQRT[prop], rel=1e-6
        )


def test_hildebrand_is_the_quadrature_sum_of_the_three_components():
    preds = _predict(polymer_candidate(SPHERE_UNITS["poly(methyl methacrylate)"]))
    components = [_mpa(preds[c]) for c in COMPONENTS]
    total = _mpa(preds["hildebrand_solubility_parameter"])
    assert total == pytest.approx(math.sqrt(sum(v * v for v in components)), rel=1e-9)


# -- coverage ---------------------------------------------------------------

#: The eleven repeat units in ``reference_polymers.json`` this must refuse,
#: each mapped to the word its reason has to contain.  Pinned by name so that
#: a change quietly letting an amide through fails here.
EXPECTED_REFUSALS = {
    "poly(vinylidene chloride)": "two or more halogens",
    "poly(vinyl fluoride)": "no increment covering F",
    "poly(vinylidene fluoride)": "no increment covering F",
    "poly(chlorotrifluoroethylene)": "no increment covering F",
    "polyacrylamide": "no increment covering N",
    "poly(N-isopropylacrylamide)": "no increment covering N",
    "nylon-6": "no increment covering N",
    "nylon-6,6": "no increment covering N",
    "nylon-11": "no increment covering N",
    "nylon-12": "no increment covering N",
    "poly(ethylene naphthalate)": "fused aromatic ring",
}


def test_coverage_over_the_reference_repeat_units_is_forty_six_of_fifty_seven():
    answered = []
    refused = {}
    for polymer in reference_polymers():
        try:
            repeat_unit_cohesion(polymer["repeat_unit"])
        except RepeatUnitRefused as exc:
            refused[polymer["name"]] = str(exc)
        else:
            answered.append(polymer["name"])

    assert len(answered) + len(refused) == 57
    assert len(answered) == 46, sorted(refused)
    assert set(refused) == set(EXPECTED_REFUSALS)
    for name, fragment in EXPECTED_REFUSALS.items():
        assert fragment in refused[name], f"{name}: {refused[name]}"


# -- refusals ---------------------------------------------------------------


def test_nylon_66_is_refused_rather_than_answered():
    """The composite route that would have covered it was measurably wrong.

    -CO- plus -NH- puts nylon-6,6 at a total near 20 MPa^0.5 against a
    measured Hildebrand parameter near 28, so the amide increment was left out
    and every amide is refused.
    """
    preds = _predict(polymer_candidate(NYLON66))
    for prop in sorted(PROPERTIES):
        assert preds[prop].status is PredictionStatus.UNSUPPORTED
        assert preds[prop].quantity is None
        assert "no increment covering N" in preds[prop].notes[0]


def test_missing_amorphous_density_is_refused_not_defaulted():
    preds = _predict(polymer_candidate(PS), with_density=False)
    for prop in sorted(PROPERTIES):
        assert preds[prop].status is PredictionStatus.UNSUPPORTED
        assert preds[prop].quantity is None
        assert "amorphous_density" in preds[prop].notes[0]
        assert "guessed density is a guessed solubility parameter" in preds[prop].notes[0]


def test_a_siloxane_is_refused_here_for_its_own_reason_not_the_density_experts():
    """Poly(dimethylsiloxane), and why this test is worded the way it is.

    The density expert does *not* refuse a siloxane - silicon is outside its
    fitted elements, so it answers with an out-of-domain number.  What refuses
    PDMS is this table, which has no silicon increment, and the reason on the
    prediction has to be that one rather than a borrowed dependency failure.
    """
    pdms = "[*][Si](C)(C)O[*]"
    density = PolymerDensityExpert().predict(
        PredictionRequest(
            candidate=polymer_candidate(pdms, conditions=ROOM),
            properties=frozenset({"amorphous_density"}),
            conditions=ROOM,
        )
    )[0]
    assert density.status is PredictionStatus.OUT_OF_DOMAIN
    assert density.quantity is not None

    preds = _predict(polymer_candidate(pdms))
    assert preds["hansen_polar"].status is PredictionStatus.UNSUPPORTED
    assert preds["hansen_polar"].quantity is None
    assert "no increment covering Si" in preds["hansen_polar"].notes[0]


def test_an_out_of_domain_density_makes_an_out_of_domain_solubility_parameter():
    """delta = (group sum) x rho, so a volume nobody trusts is not laundered here.

    No repeat unit reaches this path today - every element the density expert
    refuses, this table refuses first - so it is built as a guard against that
    element set widening, and exercised by handing the expert a density marked
    the way the density expert marks a siloxane.
    """
    from formulate.core.quantity import ApplicabilityDomain

    candidate = polymer_candidate(PS, conditions=ROOM)
    density = PolymerDensityExpert().predict(
        PredictionRequest(
            candidate=candidate,
            properties=frozenset({"amorphous_density"}),
            conditions=ROOM,
        )
    )[0]
    flagged = density.model_copy(
        update={
            "status": PredictionStatus.OUT_OF_DOMAIN,
            "applicability": ApplicabilityDomain.outside(
                "the packing factor was fitted only over C, Cl, F, H, N, O",
                basis="a packing factor fitted to amorphous polymer densities",
            ),
        }
    )
    preds = {
        p.property: p
        for p in PolymerHansenExpert().predict(
            PredictionRequest(
                candidate=candidate,
                properties=PROPERTIES,
                conditions=ROOM,
                context={"amorphous_density": flagged},
            )
        )
    }
    for prop in sorted(PROPERTIES):
        prediction = preds[prop]
        assert prediction.status is PredictionStatus.OUT_OF_DOMAIN
        assert prediction.quantity is not None  # marked, not withheld
        assert any(
            "upstream amorphous density is itself out of domain" in w
            for w in prediction.applicability.warnings
        )


def test_a_short_chain_is_marked_because_its_ends_are_not_in_the_sum():
    """A 400 g/mol diol is mostly end group, and an interior-unit sum cannot see it.

    Two hydroxyl ends put about 40 kJ/mol of hydrogen bonding on a chain whose
    repeat units contribute 27, which moves delta_h by roughly 5 MPa^0.5 -
    larger than the bar this expert quotes. ``polymer_tg`` draws the same line
    at the same molar mass, and drawing it in only one of the two would let a
    polyol be ranked on a parameter for a polymer it is not.
    """
    short = PolymerSpec(
        monomers=(MonomerUnit(smiles="[*]CCO[*]"),),
        number_average_molar_mass=Quantity(value=400.0, unit="g/mol"),
    )
    candidate = Candidate(
        material_class=MaterialClass.POLYMER, polymer=short, conditions=ROOM
    )
    domain = PolymerHansenExpert().assess_domain(candidate)
    assert not domain.in_domain
    assert any("Mn = 400 g/mol" in w for w in domain.warnings)
    assert any("high-polymer limit" in w for w in domain.warnings)

    preds = _predict(candidate)
    prediction = preds["hansen_hydrogen_bonding"]
    assert prediction.status is PredictionStatus.OUT_OF_DOMAIN
    assert prediction.quantity is not None


def test_a_long_chain_is_not_penalised_for_carrying_a_molar_mass():
    long = PolymerSpec(
        monomers=(MonomerUnit(smiles="[*]CCO[*]"),),
        number_average_molar_mass=Quantity(value=100_000.0, unit="g/mol"),
    )
    candidate = Candidate(
        material_class=MaterialClass.POLYMER, polymer=long, conditions=ROOM
    )
    domain = PolymerHansenExpert().assess_domain(candidate)
    assert domain.in_domain
    assert domain.score == 1.0


def test_temperature_outside_the_window_is_refused_rather_than_extrapolated():
    preds = _predict(polymer_candidate(PS), conditions=HOT)
    low, high = TEMPERATURE_WINDOW_K
    for prop in sorted(PROPERTIES):
        assert preds[prop].status is PredictionStatus.UNSUPPORTED
        assert f"{low:.0f}-{high:.0f} K" in preds[prop].notes[0]


def test_absent_temperature_is_refused_rather_than_assumed_to_be_room_temperature():
    preds = _predict(polymer_candidate(PS), conditions=NO_TEMPERATURE)
    for prop in sorted(PROPERTIES):
        assert preds[prop].status is PredictionStatus.UNSUPPORTED
        assert "no temperature was given" in preds[prop].notes[0]


@pytest.mark.parametrize(
    "spec, fragment",
    [
        (
            PolymerSpec(
                monomers=(MonomerUnit(smiles=PS),), topology=PolymerTopology.NETWORK
            ),
            "network",
        ),
        (
            PolymerSpec(
                monomers=(MonomerUnit(smiles=PS),),
                crosslink_density=Quantity(value=1.0, unit="mol/m^3"),
            ),
            "crosslink",
        ),
    ],
)
def test_crosslinked_architecture_is_refused(spec, fragment):
    candidate = Candidate(
        material_class=MaterialClass.POLYMER, polymer=spec, conditions=ROOM
    )
    preds = _predict(candidate)
    prediction = preds["hansen_dispersion"]
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert fragment in prediction.notes[0]


@pytest.mark.parametrize(
    "unit_smiles",
    ["CC(c1ccccc1)", "[*]CC(c1ccccc1)", "not a smiles at all"],
)
def test_a_repeat_unit_without_two_attachment_points_is_refused(unit_smiles):
    with pytest.raises(RepeatUnitRefused):
        chain_cohesion(PolymerSpec(monomers=(MonomerUnit(smiles=unit_smiles),)))


@pytest.mark.parametrize(
    "smiles",
    ["C1CCCC1", "C1CCCCC1", "C1CCOC1", "C1COCCO1", "CC1=CCC(CC1)C(C)=C"],
)
def test_the_ring_sizes_the_acceptance_set_actually_contains_are_kept(smiles):
    """Five- and six-membered rings are the entry; refusing them would be a bug."""
    from rdkit import Chem

    from formulate.experts.polymer_hansen import _structural_refusal

    assert _structural_refusal(Chem.MolFromSmiles(smiles)) is None


def test_a_crowded_benzene_ring_is_flagged_as_an_extrapolation():
    """Poly(2,6-dimethyl-1,4-phenylene oxide) has a tetrasubstituted ring.

    The per-atom split is determined by two published rings, at one and two
    substituents. Three and four are extrapolation, and the bundled acceptance
    set has one trisubstituted benzene and no tetrasubstituted one to check it
    against, so the number ships with the ranking told where it came from.
    """
    ppe = "[*]Oc1c(C)cc(cc1C)[*]"
    domain = PolymerHansenExpert().assess_domain(polymer_candidate(ppe))
    assert domain.in_domain
    assert any("three or four substituents" in w for w in domain.warnings)
    assert domain.score <= 0.7

    # ...and a mono- or disubstituted ring, which the published entries cover
    # outright, is not flagged, or the warning would be wallpaper.
    for covered in (PS, SPHERE_UNITS["polycarbonate"]):
        assert not any(
            "three or four substituents" in w
            for w in PolymerHansenExpert().assess_domain(
                polymer_candidate(covered)
            ).warnings
        )


@pytest.mark.parametrize(
    "smiles, fragment",
    [
        ("c1ccccc1", "substituents"),
        ("c1ccc2ccccc2c1", "fused aromatic ring"),
        ("c1ccncc1", "heteroaromatic"),
        ("O=C1CCCO1", "carbonyl inside a ring"),
        ("CC(=O)OC1COC(=O)O1", "carbonyl inside a ring"),
        ("ClC(Cl)CC", "two or more halogens"),
        # Van Krevelen tabulates one ring closure, for a five- or six-membered
        # ring. The bundled acceptance set contains nothing else, so a strained
        # ring, a macrocycle and a bicyclic skeleton are all extrapolations
        # from a single number that stands for what a normal ring does.
        ("C1CC1", "3-membered aliphatic ring"),
        ("C1CCCCCC1", "7-membered aliphatic ring"),
        ("C1CC2CCC1C2", "fused or bridged aliphatic ring"),
    ],
)
def test_structures_outside_the_table_are_named_rather_than_scored(smiles, fragment):
    from rdkit import Chem

    from formulate.experts.polymer_hansen import _structural_refusal

    reason = _structural_refusal(Chem.MolFromSmiles(smiles))
    assert reason is not None, smiles
    assert fragment in reason


@pytest.mark.parametrize(
    "unit_smiles, fragment",
    [
        # Poly(ethylene furanoate): "no increment covering O" would contradict
        # the -O- row in the same table. What is missing is the heterocycle.
        ("[*]OCCOC(=O)c1ccc(o1)C(=O)[*]", "aromatic O (a heteroaromatic ring"),
        # An ionomer: the cohesion is Coulombic and no group sum reaches it.
        ("[*]CC(C(=O)[O-])[*]", "formally charged O"),
    ],
)
def test_a_refusal_names_the_environment_and_not_just_the_element(unit_smiles, fragment):
    with pytest.raises(RepeatUnitRefused) as excinfo:
        repeat_unit_cohesion(unit_smiles)
    assert fragment in str(excinfo.value)


def test_an_unassignable_atom_is_a_refusal_not_a_zero():
    from rdkit import Chem

    from formulate.experts.polymer_hansen import _count_groups

    with pytest.raises(RepeatUnitRefused) as excinfo:
        _count_groups(Chem.MolFromSmiles("CS(=O)C"))  # dimethyl sulfoxide
    assert "S" in str(excinfo.value)
    assert "refused rather than scored with a hole in it" in str(excinfo.value)


# -- wiring -----------------------------------------------------------------


def test_expert_declares_only_registry_properties_and_one_dependency():
    expert = PolymerHansenExpert()
    assert expert.supported_properties <= set(PROPERTY_REGISTRY)
    assert expert.dependencies == frozenset({"amorphous_density"})
    assert expert.supported_classes == frozenset({MaterialClass.POLYMER})
    assert expert.supported_properties == PROPERTIES
    for prop in expert.supported_properties:
        assert PROPERTY_REGISTRY[prop].canonical_unit == "pascal ** 0.5"


def test_density_is_resolved_before_the_solubility_parameter():
    from formulate.experts.registry import ExpertRegistry

    hansen = PolymerHansenExpert()
    density = PolymerDensityExpert()
    registry = ExpertRegistry([hansen, density])
    order = [e.id for e in registry.resolution_order([hansen, density])]
    assert order.index("polymer_density") < order.index("polymer_hansen")


def test_nothing_is_produced_for_a_molecule():
    request = PredictionRequest(
        candidate=molecule_candidate("CCO"), properties=PROPERTIES, conditions=ROOM
    )
    assert PolymerHansenExpert().predict(request) == []


# -- copolymers -------------------------------------------------------------


def test_a_copolymer_averages_triples_and_not_group_counts():
    """Mole-averaging group counts before sqrt(sum F_p^2) inflates delta_p.

    For a 50/50 ethylene / vinyl chloride unit the wrong route reads about
    10.2 MPa^0.5 where the volume-fraction average of the two triples reads
    7.2, because quadrature cannot tell a mixture of a polar and a non-polar
    unit from one moderately polar unit.
    """
    candidate = polymer_candidate([(PE, 0.5), (PVC, 0.5)], conditions=ROOM)
    preds = _predict(candidate)
    polar = _mpa(preds["hansen_polar"])
    assert polar == pytest.approx(7.2, abs=0.3)

    chain = chain_cohesion(candidate.polymer)
    density = (
        preds["hansen_polar"].provenance.parameters["amorphous_density_g_cm3"]
    )
    wrong = math.sqrt(sum(0.5 * u.f_p_squared for u, _ in chain.units)) / (
        chain.mass / density
    )
    assert wrong > polar + 2.0

    assert any("random copolymer" in note for note in preds["hansen_polar"].notes)


def test_a_copolymer_is_flagged_as_wrong_for_a_block_architecture():
    candidate = polymer_candidate([(PE, 0.5), (PVC, 0.5)], conditions=ROOM)
    domain = PolymerHansenExpert().assess_domain(candidate)
    assert domain.in_domain
    assert any("block copolymer" in w for w in domain.warnings)
    assert domain.score < 1.0


# -- applicability domain ---------------------------------------------------


def test_a_conjugated_aryl_carbonyl_is_flagged_with_the_measured_direction():
    """Poly(ethylene terephthalate): the ring's single F_p cannot see the ester."""
    pet = "[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]"
    domain = PolymerHansenExpert().assess_domain(polymer_candidate(pet))
    assert domain.in_domain
    assert any("conjugated to an aromatic ring" in w for w in domain.warnings)
    assert any("3.9 MPa^0.5 low" in w for w in domain.warnings)
    assert domain.score <= 0.5


def test_the_carbonate_composite_is_declared_rather_than_passed_off_as_an_entry():
    """The table has no carbonate row; ownership makes it an ester plus an ether."""
    domain = PolymerHansenExpert().assess_domain(
        polymer_candidate(SPHERE_UNITS["polycarbonate"])
    )
    assert domain.in_domain
    assert any("open-chain carbonate" in w for w in domain.warnings)


def test_a_methyl_cap_does_not_raise_a_warning_the_repeat_unit_has_not_earned():
    """Polycaprolactone: the cap makes a methyl ester at each chain end.

    Flags are counted as a trimer-minus-dimer difference for the same reason
    the groups are, so a chain end the polymer does not have raises nothing.
    """
    domain = PolymerHansenExpert().assess_domain(polymer_candidate("[*]CCCCCC(=O)O[*]"))
    assert domain.warnings == ()
    assert domain.score == 1.0


def test_an_olefinic_backbone_is_flagged_for_its_missing_hydrogen_bonding_term():
    domain = PolymerHansenExpert().assess_domain(polymer_candidate("[*]CC=CC[*]"))
    assert domain.in_domain
    assert any("olefinic backbone" in w for w in domain.warnings)


def test_a_refused_repeat_unit_is_out_of_domain_with_the_reason():
    domain = PolymerHansenExpert().assess_domain(polymer_candidate(NYLON66))
    assert not domain.in_domain
    assert any("no increment covering N" in w for w in domain.warnings)


def test_a_declared_crosslinker_is_not_scored_as_an_uncrosslinked_chain():
    """The basis says uncrosslinked; a crosslinker monomer has to contradict it.

    A stated network or a stated crosslink density is already refused by
    ``_architecture_reason``. A monomer whose role is crosslinker with neither
    stated is the gap, and ``polymer.analyse_chain`` marks the same case.
    """
    from formulate.core.candidate import MonomerRole

    spec = PolymerSpec(
        monomers=(
            MonomerUnit(smiles=PS, mole_fraction=1.0),
            MonomerUnit(
                smiles="[*]c1ccc(cc1)[*]",
                mole_fraction=0.0,
                role=MonomerRole.CROSSLINKER,
            ),
        )
    )
    candidate = Candidate(
        material_class=MaterialClass.POLYMER, polymer=spec, conditions=ROOM
    )
    domain = PolymerHansenExpert().assess_domain(candidate)
    assert any("crosslinker is present" in w for w in domain.warnings)
    assert domain.score <= 0.5


def test_a_plain_polyolefin_is_squarely_in_domain():
    domain = PolymerHansenExpert().assess_domain(polymer_candidate(PE))
    assert domain.in_domain
    assert domain.score == 1.0
    assert domain.warnings == ()


def test_a_structural_zero_polar_term_is_a_value_and_says_so():
    """delta_p = 0 for polyethylene is an answer, not an absence.

    The registry bounds admit it and it will rank, which is worth pinning: a
    requirement minimising delta_p would otherwise score every polyolefin best
    on a number that is a structural zero rather than a measurement.
    """
    preds = _predict(polymer_candidate(PE))
    polar = preds["hansen_polar"]
    assert polar.status is PredictionStatus.OK
    assert _mpa(polar) == pytest.approx(0.0, abs=1e-12)
    assert _sigma(polar) > 0.0
    assert _mpa(preds["hildebrand_solubility_parameter"]) == pytest.approx(17.5, abs=0.3)


# -- the table itself -------------------------------------------------------


def test_the_aromatic_split_reproduces_the_published_ring_values():
    """5a + b = 1430 (phenyl) and 4a + 2b = 1270 (p-phenylene), exactly."""
    table = {g.name: g for g in HVK_GROUPS}
    ch = table["aromatic CH"].f_d
    sub = table["aromatic C(sub)"].f_d
    assert 5 * ch + sub == pytest.approx(1430.0)
    assert 4 * ch + 2 * sub == pytest.approx(1270.0)
    assert RING_INCREMENTS["aromatic ring"].f_p == pytest.approx(110.0)
    assert RING_INCREMENTS["aromatic ring"].f_d == 0.0


def test_every_increment_has_a_parseable_pattern_and_a_distinct_name():
    from rdkit import Chem

    names = [g.name for g in HVK_GROUPS]
    assert len(names) == len(set(names))
    for group in HVK_GROUPS:
        assert Chem.MolFromSmarts(group.smarts) is not None, group.name
        assert group.e_h >= 0.0


def test_a_symmetric_double_bond_assigns_both_of_its_carbons():
    """cis-1,4-polybutadiene, the case a two-atom SMARTS silently refused."""
    cohesion = repeat_unit_cohesion("[*]CC=CC[*]")
    assert cohesion.groups == {"-CH2-": 2, "=CH-": 2}
    assert cohesion.f_d == pytest.approx(2 * 270.0 + 2 * 200.0)


def test_repeat_unit_mass_and_groups_come_from_an_interior_unit():
    cohesion = repeat_unit_cohesion(PS)
    assert cohesion.mass == pytest.approx(104.15, abs=0.05)
    assert cohesion.groups == {
        "-CH2-": 1,
        ">CH-": 1,
        "aromatic CH": 5,
        "aromatic C(sub)": 1,
        "aromatic ring": 1,
    }


def test_chain_cohesion_of_a_homopolymer_is_the_unit_itself():
    chain = chain_cohesion(PolymerSpec(monomers=(MonomerUnit(smiles=PS),)))
    assert isinstance(chain, ChainCohesion)
    assert len(chain.units) == 1
    assert chain.mass == pytest.approx(repeat_unit_cohesion(PS).mass)
