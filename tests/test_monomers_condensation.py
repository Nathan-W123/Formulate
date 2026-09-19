"""The step-growth generator: what it writes, and - harder - what it refuses.

A generator is only as good as its gates.  Anyone can emit ten thousand
syntactically valid repeat units; the question is whether the ranking they fill
is made of polymers.  So the refusals get as much space here as the generation,
and three of them are asserted structurally - by building the SMILES the
grammar *would* have written and asserting it is absent - rather than by
trusting a flag.

The test that should be hardest to keep passing is :func:`test_rediscovers_the_
bundled_polyesters_and_polyamides`.  If this grammar cannot regenerate PET,
nylon-6,6 and Kevlar from its own monomer lists, then the lists are wrong and
every candidate it proposes is built on the same error.
"""

from __future__ import annotations

import json
from importlib import resources

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import polymer_candidate
from formulate.experts.base import PredictionRequest
from formulate.experts.melt import backbone_stereocentres
from formulate.experts.polymer import _FITTED_ELEMENTS
from formulate.experts.polymer_feasibility import (
    Mechanism,
    PolymerFeasibilityExpert,
    analyse_repeat_unit,
)
from formulate.exploration.monomers.condensation import (
    AB_AMIDE,
    AB_ESTER,
    AROMATIC_DIAMINES,
    Availability,
    DIACIDS,
    DIAMINES,
    DIOLS,
    DIPHENOLS,
    OXALATE_ROUTE,
    POLYAMIDE,
    POLYESTER,
    Scale,
    catalogue,
    periodic_key,
    refusals,
    units,
)

pytestmark = requires_rdkit

PROP = "synthetic_accessibility"
EXPERT = PolymerFeasibilityExpert()

#: The twelve of the 57 bundled reference polymers that this family owns.  PEEK
#: and poly(phenylene oxide) are step-growth too and are deliberately not here:
#: their linkage is an aryl ether formed by nucleophilic substitution or by
#: oxidative coupling, not an ester or an amide formed by condensation.
BUNDLED_IN_FAMILY = (
    "poly(ethylene terephthalate)",
    "poly(butylene terephthalate)",
    "poly(ethylene naphthalate)",
    "poly(ethylene adipate)",
    "polycaprolactone",
    "polylactide",
    "polyglycolide",
    "poly(3-hydroxybutyrate)",
    "nylon-6",
    "nylon-6,6",
    "nylon-11",
    "nylon-12",
)

#: The one bundled polymer this grammar writes with a different cut.  Both
#: spellings are the same periodic chain; see :func:`periodic_key`.
PHASE_SHIFTED = "polycaprolactone"


def canonical(smiles: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return Chem.MolToSmiles(mol)


def inchikey(smiles: str) -> str:
    from rdkit import Chem

    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def reference() -> list[dict]:
    text = (
        resources.files("formulate.data")
        .joinpath("reference_polymers.json")
        .read_text(encoding="utf-8")
    )
    return list(json.loads(text)["polymers"])


def score(unit: str) -> float | None:
    candidate = polymer_candidate(unit)
    prediction = EXPERT.predict(
        PredictionRequest(
            candidate=candidate,
            properties=frozenset({PROP}),
            conditions=candidate.conditions,
        )
    )[0]
    return None if prediction.quantity is None else prediction.quantity.value


# --------------------------------------------------------------------------
# 1. The contract every generated unit has to meet
# --------------------------------------------------------------------------


def test_units_are_pairs_and_deduplicated():
    generated = units()
    assert len(generated) > 250, "a grammar this size should not collapse to a list"
    assert len({s for _, s in generated}) == len(generated)
    assert len({n for n, _ in generated}) == len(generated)
    assert all(isinstance(n, str) and isinstance(s, str) for n, s in generated)


def test_every_unit_parses_with_exactly_two_attachment_points():
    from rdkit import Chem

    for name, smiles in units():
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"{name}: {smiles} does not parse"
        dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        assert len(dummies) == 2, f"{name}: {len(dummies)} attachment points"
        for dummy in dummies:
            assert len(dummy.GetNeighbors()) == 1, f"{name}: a dangling [*]"


def test_every_unit_parses_when_the_attachment_points_are_capped():
    """Capped with [H], not deleted - which is what PET's in-branch [*] tests."""
    from rdkit import Chem

    for name, smiles in units():
        capped = smiles.replace("[*]", "[H]")
        assert Chem.MolFromSmiles(capped) is not None, f"{name}: {capped}"


def test_every_unit_stays_inside_the_fitted_element_set():
    """Density and Tg were fitted over H, C, N, O, F, Cl; nothing here leaves it."""
    from rdkit import Chem

    for name, smiles in units():
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        elements = {a.GetSymbol() for a in mol.GetAtoms()} - {"*"}
        assert elements <= _FITTED_ELEMENTS, f"{name}: {sorted(elements - _FITTED_ELEMENTS)}"


def test_backbone_stereocentres_can_read_every_unit():
    """The explorer sweeps tacticity off this function; None would break the sweep."""
    for name, smiles in units():
        assert backbone_stereocentres(smiles) is not None, name


def test_only_the_alpha_substituted_polyesters_are_handed():
    """Stereochemistry is a claim, so the units that carry one are named.

    Exactly three repeat units here have a backbone atom with two different
    substituents, and all three are the bacterial and lactic polyesters.  A
    nylon backbone is a run of methylenes and cannot be tactic, and if this
    test ever grows an entry the grammar has started writing a side group
    somewhere it should not.
    """
    handed = {name for name, smiles in units() if backbone_stereocentres(smiles)}
    assert handed == {
        "poly(lactic acid) (PLA, commercial)",
        "poly(3-hydroxybutyric acid) (PHB, commercial, catalogue monomer)",
        "poly(3-hydroxyvaleric acid) (P3HV, reported, catalogue monomer)",
    }


# --------------------------------------------------------------------------
# 2. Rediscovery - the validation that matters most
# --------------------------------------------------------------------------


def test_rediscovers_the_bundled_polyesters_and_polyamides():
    """Every one of the 12 bundled units in this family comes back out.

    Eleven match as canonical SMILES.  Polycaprolactone matches only as a
    periodic chain, because the bundled file cuts it one bond further along -
    which is a difference in spelling, asserted to be nothing more than that by
    the next test.
    """
    generated = units()
    by_canonical = {canonical(s) for _, s in generated}
    by_period = {periodic_key(s) for _, s in generated}

    exact, periodic, missing = [], [], []
    for record in reference():
        if record["name"] not in BUNDLED_IN_FAMILY:
            continue
        unit = record["repeat_unit"]
        if canonical(unit) in by_canonical:
            exact.append(record["name"])
        elif periodic_key(unit) in by_period:
            periodic.append(record["name"])
        else:
            missing.append(record["name"])

    assert missing == [], f"the grammar cannot write its own family: {missing}"
    assert periodic == [PHASE_SHIFTED]
    assert len(exact) == 11


def test_polycaprolactones_two_spellings_are_one_polymer():
    """The single rediscovery miss is a cut point, and it changes no number."""
    bundled = "[*]CCCCCC(=O)O[*]"
    generated = "[*]OCCCCCC(=O)[*]"
    assert canonical(bundled) != canonical(generated), "the point of the test"
    assert periodic_key(bundled) == periodic_key(generated)
    assert score(bundled) == pytest.approx(score(generated))
    # And both reconstruct to caprolactone or 6-hydroxyhexanoic acid.
    for spelling in (bundled, generated):
        monomers = {m for r in analyse_repeat_unit(spelling).routes for m in r.monomers}
        assert any("O" in m for m in monomers)
        assert inchikey("O=C1CCCCCO1") in {inchikey(m) for m in monomers}


def test_periodic_key_refuses_what_it_cannot_place():
    assert periodic_key("not a smiles") is None
    assert periodic_key("[*]CC") is None  # one attachment point
    assert periodic_key("[*]C([*])C") is None  # both on the same atom


def test_the_aramids_are_reachable():
    """Kevlar and Nomex, by name, because they are why this family matters.

    The highest-performance fibre in commercial production has to be inside the
    search space or the search cannot return it.  Both are AA+BB polyamides of
    monomers already in the lists, so reaching them is a property of the
    grammar rather than of a special case.
    """
    by_canonical = {canonical(s): n for n, s in units()}
    kevlar = canonical("[*]Nc1ccc(cc1)NC(=O)c1ccc(cc1)C(=O)[*]")
    nomex = canonical("[*]Nc1cccc(NC(=O)c2cccc(C(=O)[*])c2)c1")
    assert by_canonical[kevlar] == "poly(p-phenylene terephthalamide) (Kevlar, commercial)"
    assert by_canonical[nomex] == "poly(m-phenylene isophthalamide) (Nomex, commercial)"


# --------------------------------------------------------------------------
# 3. The gates - a grammar is defined by what it refuses
# --------------------------------------------------------------------------


def test_refusals_are_specific_and_are_actually_refused():
    """Every refused pair names a reason and is absent from the output."""
    listed = refusals()
    assert len(listed) == 100
    emitted = {canonical(s) for _, s in units()}
    partners = {m.name: m for m in DIOLS + DIPHENOLS + DIAMINES + AROMATIC_DIAMINES}
    acids = {m.name: m for m in DIACIDS}
    for partner_name, acid_name, reason in listed:
        assert len(reason) > 40, f"{partner_name}/{acid_name}: a reason must be a reason"
        partner, acid = partners[partner_name], acids[acid_name]
        amide = partner in DIAMINES + AROMATIC_DIAMINES
        template = POLYAMIDE if amide else POLYESTER
        would_be = template.format(partner=partner.core, acid=acid.core)
        assert canonical(would_be) not in emitted, f"{partner_name} + {acid_name} leaked"


def test_no_alpha_amino_acid_is_emitted():
    """Glycine's dimer closes to a diketopiperazine; nylon-2 is not a polymer."""
    emitted = {canonical(s) for _, s in units()}
    for core in ("", "C(C)"):  # glycine, alanine
        assert canonical(AB_AMIDE.format(core=core)) not in emitted


def test_no_malonate_or_malonamide_is_emitted():
    """A carboxyl beta to a carbonyl decarboxylates below polycondensation."""
    assert all(m.name != "malonic acid" for m in DIACIDS)
    emitted = {canonical(s) for _, s in units()}
    assert canonical(POLYESTER.format(partner="CC", acid="C")) not in emitted
    assert canonical(POLYAMIDE.format(partner="CCCCCC", acid="C")) not in emitted


def test_no_ethylenediamine_polyamide_is_emitted():
    """The second nitrogen closes to an imidazoline and terminates the chain."""
    emitted = {canonical(s) for _, s in units()}
    for acid in DIACIDS:
        would_be = POLYAMIDE.format(partner="CC", acid=acid.core)
        assert canonical(would_be) not in emitted
    # ...and ethylene glycol, the same carbon skeleton with oxygen, is not
    # refused: poly(ethylene terephthalate) is the most made polyester there is.
    assert canonical(POLYESTER.format(partner="CC", acid="c1ccc(cc1)")) in emitted


def test_no_propanediamine_polyamide_is_emitted():
    """The second nitrogen closes to a six-membered amidine, the easier ring.

    This is the gate that has to be argued from ring size rather than from
    speed: refusing the five-membered imidazoline from 1,2-ethanediamine and
    keeping the six-membered tetrahydropyrimidine from 1,3-propanediamine would
    have had Carothers' rule backwards, since six is the ring that closes best.
    """
    emitted = {canonical(s) for _, s in units()}
    for acid in DIACIDS:
        would_be = POLYAMIDE.format(partner="CCC", acid=acid.core)
        assert canonical(would_be) not in emitted, acid.name
    # ...and 1,3-propanediol, the same skeleton with oxygen, is not refused:
    # poly(trimethylene terephthalate) is sold as Sorona.
    assert canonical(POLYESTER.format(partner="CCC", acid="c1ccc(cc1)")) in emitted


def test_no_succinamide_or_glutaramide_is_emitted():
    """A C4 or C5 diacid closes an imide onto the amide it just made.

    The asymmetry with the ester side is the point and is asserted here: the
    same two diacids are kept for the diols, because the ring an oxygen closes
    is the anhydride and an anhydride goes on acylating.  Poly(butylene
    succinate) is sold by the kilotonne; no AA+BB polyamide anyone has sold
    uses a diacid shorter than adipic.
    """
    emitted = {canonical(s) for _, s in units()}
    for amine in DIAMINES + AROMATIC_DIAMINES:
        for core in ("CC", "CCC"):  # succinic, glutaric
            would_be = POLYAMIDE.format(partner=amine.core, acid=core)
            assert canonical(would_be) not in emitted, f"{amine.name} + C{len(core) + 2}"
    assert canonical(POLYESTER.format(partner="CCCC", acid="CC")) in emitted  # PBS
    assert canonical(POLYESTER.format(partner="CC", acid="CCC")) in emitted


def test_adipic_is_where_the_imide_stops_being_a_ring():
    """The gate is a ring-size rule, so the first acid that escapes it is named.

    Adipic acid's imide would be seven-membered and does not close, which is
    why nylon-6,6 exists and nylon-6,4 does not.  If this ever fails the gate
    has become a blanket ban on short diacids instead of an argument.
    """
    emitted = {canonical(s) for _, s in units()}
    assert canonical(POLYAMIDE.format(partner="CCCCCC", acid="CCCC")) in emitted
    assert canonical(POLYAMIDE.format(partner="CCCCCC", acid="")) in emitted  # oxamide


def test_the_oxalates_name_the_bottle_they_are_made_from():
    """Every oxalate and oxamide says it starts from the diester, not the acid.

    Oxalic acid decomposes below polycondensation temperature, so the
    reconstruction downstream - which hands back the free acid, because that is
    what the structure says - would send someone to the wrong shelf.  The
    polymers are real; the route is the caveat, and it is carried on the unit.
    """
    oxalates = [u for u in catalogue() if "oxalic acid" in u.monomers]
    assert len(oxalates) == 18, len(oxalates)
    for unit in oxalates:
        assert OXALATE_ROUTE in unit.note, unit.name


def test_no_homopolymer_sold_only_as_a_copolymer_is_called_commercial():
    """COMMERCIAL has to mean the polymer is sold, not that its family is.

    Poly(3-hydroxyvalerate) is the case that motivated this: it is bought only
    inside PHBV, and a ranking that reads "commercial" off the label would be
    told a product exists that does not.  Any unit whose own note says it is
    sold only copolymerised has to be REPORTED, and the note is the place the
    contradiction shows up, so the note is what this reads.
    """
    for unit in catalogue():
        note = unit.note.lower()
        if "only" in note and "copolymer" in note:
            assert unit.availability is not Availability.COMMERCIAL, unit.name
    by_name = {u.name: u for u in catalogue()}
    assert by_name["poly(3-hydroxyvaleric acid)"].availability is Availability.REPORTED
    assert by_name["poly(6-hydroxy-2-naphthoic acid)"].availability is Availability.REPORTED
    assert by_name["poly(lactic acid)"].availability is Availability.COMMERCIAL


def test_no_phenol_is_paired_with_an_aliphatic_diacid():
    """A phenol will not melt-esterify; the routes that work are aromatic-only."""
    emitted = {canonical(s) for _, s in units()}
    for phenol in DIPHENOLS:
        for acid in DIACIDS:
            would_be = canonical(POLYESTER.format(partner=phenol.core, acid=acid.core))
            assert (would_be in emitted) is acid.aromatic, f"{phenol.name} + {acid.name}"


def test_no_arylamine_is_paired_with_an_aliphatic_diacid():
    """An arylamine will not form the nylon salt a melt polycondensation needs."""
    emitted = {canonical(s) for _, s in units()}
    for amine in AROMATIC_DIAMINES:
        for acid in DIACIDS:
            would_be = canonical(POLYAMIDE.format(partner=amine.core, acid=acid.core))
            assert (would_be in emitted) is acid.aromatic, f"{amine.name} + {acid.name}"


def test_an_aliphatic_diamine_with_an_aromatic_diacid_is_not_refused():
    """The gate is one-directional, and it has to be: PA9T and MXD6 are sold."""
    emitted = {canonical(s) for _, s in units()}
    pa9t = POLYAMIDE.format(partner="CCCCCCCCC", acid="c1ccc(cc1)")
    mxd6 = POLYAMIDE.format(partner="Cc1cc(ccc1)C", acid="CCCC")
    assert canonical(pa9t) in emitted
    assert canonical(mxd6) in emitted


# --------------------------------------------------------------------------
# 4. What the engine's own feasibility expert says about the output
# --------------------------------------------------------------------------


def test_every_generated_unit_survives_the_feasibility_expert():
    """A unit this engine calls unmakeable is a unit the grammar should not emit."""
    refused = []
    values = []
    for name, smiles in units():
        value = score(smiles)
        if value is None:
            refused.append(name)
        else:
            values.append(value)
    assert refused == [], f"{len(refused)} unmakeable units emitted: {refused[:5]}"
    assert min(values) >= 1.0 and max(values) <= 10.0
    # PET is the family's benchmark: everything here is made the same way it is.
    assert max(values) < 5.0, "nothing in a polycondensation family should be exotic"


def test_every_unit_is_routed_as_step_growth():
    """The mechanism is the family's identity, and a stray mechanism is a bug."""
    for name, smiles in units():
        analysis = analyse_repeat_unit(smiles)
        assert analysis.failure is None, f"{name}: {analysis.failure}"
        assert Mechanism.STEP_GROWTH in analysis.mechanisms, name


@pytest.mark.parametrize(
    "unit, expected",
    [
        # PET -> terephthalic acid + ethylene glycol
        ("[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]", ("O=C(O)c1ccc(C(=O)O)cc1", "OCCO")),
        # nylon-6,6 -> adipic acid + hexamethylenediamine
        ("[*]NCCCCCCNC(=O)CCCCC(=O)[*]", ("O=C(O)CCCCC(=O)O", "NCCCCCCN")),
        # Kevlar -> terephthalic acid + p-phenylenediamine
        (
            "[*]Nc1ccc(cc1)NC(=O)c1ccc(cc1)C(=O)[*]",
            ("O=C(O)c1ccc(C(=O)O)cc1", "Nc1ccc(N)cc1"),
        ),
        # poly(4-hydroxybenzoic acid) -> 4-hydroxybenzoic acid
        ("[*]Oc1ccc(cc1)C(=O)[*]", ("O=C(O)c1ccc(O)cc1",)),
        # poly(butylene succinate) -> succinic acid + 1,4-butanediol
        ("[*]OCCCCOC(=O)CCC(=O)[*]", ("O=C(O)CCC(=O)O", "OCCCCO")),
    ],
)
def test_the_grammar_writes_units_the_reconstruction_reads_correctly(unit, expected):
    """A repeat unit is written correctly when it cuts back to its own monomers.

    This is the test that catches a SMILES that parses and is still wrong: the
    engine's reconstruction is the independent reader, and it has to come back
    with the bottle the chemist would order.
    """
    emitted = {canonical(s) for _, s in units()}
    assert canonical(unit) in emitted, "the grammar does not write this unit at all"
    analysis = analyse_repeat_unit(unit)
    step = [r for r in analysis.routes if r.mechanism is Mechanism.STEP_GROWTH]
    assert step, "no polycondensation route"
    got = {inchikey(m) for m in step[0].monomers}
    assert got == {inchikey(m) for m in expected}


# --------------------------------------------------------------------------
# 5. Availability - a product and a possibility must not look alike
# --------------------------------------------------------------------------


def test_availability_is_carried_in_the_name():
    labels = {n for n, _ in units()}
    assert "poly(ethylene terephthalate) (PET, commercial)" in labels
    assert "poly(hexamethylene adipamide) (nylon-6,6, commercial)" in labels
    assert any(label.endswith("(constructible)") for label in labels)
    for label in labels:
        assert any(
            word in label
            for word in ("commercial", "reported", "constructible")
        ), label


def test_the_commercial_ones_are_a_minority_and_are_named():
    entries = catalogue()
    commercial = [u for u in entries if u.availability is Availability.COMMERCIAL]
    assert 25 <= len(commercial) <= 60, len(commercial)
    assert len(commercial) < len(entries) / 4, "a search space is mostly unexplored"
    for unit in commercial:
        assert unit.common_name, f"{unit.name} is called commercial with no name"


def test_catalogue_monomers_are_flagged():
    """Pimelic acid is a catalogue chemical; adipic acid is a commodity."""
    by_name = {u.name: u for u in catalogue()}
    assert by_name["poly(hexamethylene pimelamide)"].monomer_scale is Scale.CATALOGUE
    assert by_name["poly(hexamethylene adipamide)"].monomer_scale is Scale.BULK
    assert "catalogue monomer" in by_name["poly(hexamethylene pimelamide)"].label


def test_the_ab_polyesters_and_polyamides_are_all_present():
    """One repeat unit per AB monomer, and no pair templates leaking into them."""
    subfamilies = {u.subfamily for u in catalogue()}
    assert subfamilies == {"polyester", "polyamide", "AB polyester", "AB polyamide"}
    ab = [u for u in catalogue() if u.subfamily.startswith("AB")]
    assert len(ab) == 21
    assert all(len(u.monomers) == 1 for u in ab)
    assert all(
        u.smiles.count("C(=O)") == 1 for u in ab
    ), "an AB unit carries one linkage, a pair carries two"


def test_ab_templates_are_the_ones_the_bundled_file_uses():
    """Pinned so a reformatting of the templates cannot silently move a cut."""
    assert AB_ESTER.format(core="C") == "[*]OCC(=O)[*]"  # polyglycolide, verbatim
    assert AB_AMIDE.format(core="CCCCC") == "[*]NCCCCCC(=O)[*]"  # nylon-6, verbatim
    bundled = {r["name"]: r["repeat_unit"] for r in reference()}
    assert AB_ESTER.format(core="C") == bundled["polyglycolide"]
    assert AB_AMIDE.format(core="CCCCC") == bundled["nylon-6"]
    assert (
        POLYESTER.format(partner="CC", acid="c1ccc(cc1)")
        == bundled["poly(ethylene terephthalate)"]
    )
    assert (
        POLYAMIDE.format(partner="CCCCCC", acid="CCCC") == bundled["nylon-6,6"]
    )
