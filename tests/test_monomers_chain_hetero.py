"""The heteroatom-backbone grammar: what it writes, and harder, what it refuses.

This module is the third of three and it is the one that owed a debt: the vinyl
and condensation grammars between them regenerate 50 of the 57 bundled
reference polymers and cannot reach the other eight, and all eight are here.
:func:`test_regenerates_the_eight_the_other_two_modules_could_not_reach` is
therefore the test that should be hardest to keep passing - if the grammar
cannot write poly(ethylene oxide), PEEK and Kapton from its own monomer lists,
then the lists are wrong and everything else it proposes is built on the same
error.

The gates get as much space as the generation, because a generator is only as
good as its refusals: anything can emit ``[*]OC(=O)O{R}[*]`` for arbitrary R,
and the value is in the R it will not emit.  Two of the gates are
mutation-checked - removed at runtime, with the test asserting that a structure
then slips through that should not - so that a gate cannot rot into a no-op
while its test goes on passing for some other reason.

The measured numbers pinned here are in ``chain_hetero.MEASURED``: 171 repeat
units, 24 refusals, 8 of 8 regenerated, 144 of 171 scored by the engine's own
feasibility expert with the other 27 named.
"""

from __future__ import annotations

import collections
import json
from importlib import resources

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import polymer_candidate
from formulate.experts.base import PredictionRequest
from formulate.experts.melt import backbone_stereocentres
from formulate.experts.polymer import _FITTED_ELEMENTS
from formulate.experts.polymer_feasibility import PolymerFeasibilityExpert
from formulate.exploration.monomers import chain_hetero as hetero
from formulate.exploration.monomers.chain_hetero import (
    ARYL_ETHER,
    CARBONATE,
    CYCLIC_ETHER,
    EPOXIDE,
    LACTONE,
    MEASURED,
    NO_FEASIBILITY_ROUTE,
    OXETANE,
    PARA,
    PHENYLENE_OXIDE,
    SAME_CHAIN_TWICE,
    Availability,
    HeteroUnit,
    Melt,
    Scale,
    periodic_identity,
    records,
    refusals,
    units,
)

pytestmark = requires_rdkit

EXPERT = PolymerFeasibilityExpert()

#: The eight bundled reference polymers this module exists to reach, by the
#: abbreviation the bundled file uses.  Listed by hand rather than derived,
#: because a new reference polymer in this family should be looked at by a
#: person instead of quietly changing what the rediscovery test means.
OWED = {
    "PEO": "poly(ethylene oxide)",
    "PPO-diol": "poly(propylene oxide)",
    "PTMO": "poly(tetramethylene oxide)",
    "POM": "polyoxymethylene",
    "POTM": "poly(oxytrimethylene)",
    "PCL": "polycaprolactone",
    "PPE": "poly(2,6-dimethyl-1,4-phenylene oxide)",
    "PEEK": "poly(ether ether ketone)",
}


def canon(smiles: str) -> str:
    value = hetero.canonical(smiles)
    assert value is not None, smiles
    return value


def probe(name: str, smiles: str, **kwargs) -> HeteroUnit:
    return HeteroUnit(
        name=name,
        smiles=smiles,
        subfamily="test",
        monomers=("test",),
        availability=Availability.CONSTRUCTIBLE,
        scale=Scale.CATALOGUE,
        melt=Melt.MELT,
        **kwargs,
    )


def reference() -> list[dict]:
    text = (
        resources.files("formulate.data")
        .joinpath("reference_polymers.json")
        .read_text(encoding="utf-8")
    )
    return list(json.loads(text)["polymers"])


def score(smiles: str) -> float | None:
    candidate = polymer_candidate(smiles)
    prediction = EXPERT.predict(
        PredictionRequest(
            candidate=candidate,
            properties=("synthetic_accessibility",),
            conditions=candidate.conditions,
        )
    )[0]
    return None if prediction.quantity is None else prediction.quantity.value


# --------------------------------------------------------------------------
# 1. The contract the orchestrator relies on
# --------------------------------------------------------------------------


def test_units_obeys_the_contract():
    generated = units()
    assert isinstance(generated, list) and len(generated) > 150
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in generated)
    names = [name for name, _ in generated]
    smiles = [s for _, s in generated]
    assert len(set(names)) == len(names), "two units share a name"
    assert len({canon(s) for s in smiles}) == len(smiles), "units() must be deduplicated"


def test_every_unit_parses_with_exactly_two_attachment_points_and_caps():
    from rdkit import Chem

    for name, smiles in units():
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"{name}: {smiles} does not parse"
        dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        assert len(dummies) == 2, f"{name}: {len(dummies)} attachment points"
        for dummy in dummies:
            assert len(dummy.GetNeighbors()) == 1, f"{name}: a dangling [*]"
        # Capped with [H], not deleted: a unit that only parses with its [*]
        # removed is one whose valences do not close.  PEEK's in-branch [*] and
        # the polyimide templates, where an [*] sits on a ring nitrogen, are the
        # cases this catches.
        assert Chem.MolFromSmiles(smiles.replace("[*]", "[H]")) is not None, name


def test_the_crystallinity_test_can_read_every_unit():
    # PolymerLibraryExplorer sweeps tacticity for repeat units with backbone
    # stereocentres and leaves it unspecified otherwise.  None means "could not
    # be read", which the explorer cannot tell from "regular", so a unit that
    # returns it would be silently mis-swept.
    for name, smiles in units():
        assert backbone_stereocentres(smiles) is not None, name


def test_templates_reproduce_the_bundled_strings_verbatim():
    """Pinned so that reformatting a template cannot silently move a cut."""
    bundled = {r["abbreviation"]: r["repeat_unit"] for r in reference()}
    assert EPOXIDE.format(r="") == bundled["PEO"]
    assert EPOXIDE.format(r="(C)") == bundled["PPO-diol"]
    assert OXETANE.format(r="") == bundled["POTM"]
    assert CYCLIC_ETHER.format(core="CCCC") == bundled["PTMO"]
    assert LACTONE.format(core="CCCCC") == bundled["PCL"]
    assert PHENYLENE_OXIDE.format(r="C") == bundled["PPE"]


# --------------------------------------------------------------------------
# 2. Rediscovery - the validation that matters most
# --------------------------------------------------------------------------


def test_regenerates_the_eight_the_other_two_modules_could_not_reach():
    generated = {canon(s) for _, s in units()}
    by_abbreviation = {r["abbreviation"]: r for r in reference()}
    assert len(by_abbreviation) == 57

    missing = []
    for abbreviation, name in OWED.items():
        record = by_abbreviation[abbreviation]
        assert record["name"] == name, "the bundled set changed; re-read OWED"
        if canon(record["repeat_unit"]) not in generated:
            missing.append((abbreviation, record["repeat_unit"]))
    assert not missing, f"the grammar cannot write its own family: {missing}"
    assert MEASURED["rediscovered"] == len(OWED)


def test_the_grammar_stays_inside_its_family():
    """Nothing here collides with a polymer another module's grammar owns.

    A collision would put one polymer in the ranking twice under two names,
    which is worse than missing it: it doubles its share of the answer.
    """
    generated = {canon(s) for _, s in units()}
    strays = [
        r["abbreviation"]
        for r in reference()
        if r["abbreviation"] not in OWED and canon(r["repeat_unit"]) in generated
    ]
    assert not strays


def test_polycaprolactone_is_the_one_deliberate_overlap_with_condensation():
    """Stated rather than discovered: two modules write this one chain.

    ``condensation.py`` writes it from 6-hydroxyhexanoic acid and cuts it one
    bond further along, so the two canonical SMILES differ and the bundled
    spelling is unreachable from there.  This module writes the bundled
    spelling, from caprolactone, which is the route actually used.  Both are the
    same periodic chain, and this test is what keeps that fact from being a
    surprise.
    """
    from formulate.exploration.monomers import condensation

    bundled = next(r["repeat_unit"] for r in reference() if r["abbreviation"] == "PCL")
    mine = {canon(s) for _, s in units()}
    theirs = {canon(s) for _, s in condensation.units()}

    assert canon(bundled) in mine
    assert canon(bundled) not in theirs, "condensation.py now writes the bundled cut too"
    same_chain = [
        s for _, s in condensation.units() if condensation.periodic_key(s) ==
        condensation.periodic_key(bundled)
    ]
    assert same_chain, "condensation.py no longer writes this chain at all"
    # ...and it is the only chain the two modules share.
    shared = {
        condensation.periodic_key(s) for _, s in units()
    } & {condensation.periodic_key(s) for _, s in condensation.units()}
    shared.discard(None)
    assert shared == {condensation.periodic_key(bundled)}


def test_no_two_units_are_the_same_chain_written_at_two_cuts():
    """Canonical SMILES is an identity for a molecule, not for a chain.

    A repeat unit records where someone chose to cut a periodic chain, so
    ``-O-A-O-B-`` and ``-O-B-O-A-`` are one polymer with two canonical SMILES.
    The bisphenol x dihaloarene grid reaches both spellings of one chain, and
    emitting both would put that material in a ranking twice under two names -
    which is worse than missing it, because it doubles its share of the answer.
    """
    by_chain = collections.defaultdict(list)
    for name, smiles in units():
        by_chain[periodic_identity(smiles)].append(name)
    assert None not in by_chain, "a unit has no readable chain identity"
    twice = {chain: names for chain, names in by_chain.items() if len(names) > 1}
    assert not twice, list(twice.values())
    assert len(by_chain) == len(units())


def test_the_duplicate_cut_the_grid_actually_reaches_is_the_named_one():
    """Named rather than discovered, so that a second one fails loudly.

    4,4'-dihydroxybenzophenone with 4,4'-dichlorodiphenyl sulfone writes the same
    chain as bisphenol S with 4,4'-difluorobenzophenone: SNAr does not care which
    half of the pair arrived carrying the halide, so the grid reaches the chain
    from both corners.  Both routes are real; only one name reaches the ranking.
    """
    dropped, kept = SAME_CHAIN_TWICE
    names = {r.name for r in records()}
    assert kept in names and dropped not in names
    refusal = next(r for r in refusals() if r.name == dropped)
    assert refusal.gate == "duplicate-cut"
    assert kept in refusal.reason
    assert periodic_identity(refusal.smiles) == periodic_identity(
        next(r.smiles for r in records() if r.name == kept)
    )
    # ...and the two canonical SMILES really do differ, which is why canonical
    # deduplication could not have caught it.
    assert canon(refusal.smiles) != canon(
        next(r.smiles for r in records() if r.name == kept)
    )


def test_periodic_identity_forgets_the_cut_and_nothing_else():
    """The two polycaprolactone spellings agree; two real polymers do not."""
    assert periodic_identity("[*]CCCCCC(=O)O[*]") == periodic_identity("[*]OCCCCCC(=O)[*]")
    assert periodic_identity("[*]CCO[*]") != periodic_identity("[*]CCCO[*]")
    # A two-atom backbone has no ring to close and no cut left to choose, so the
    # canonical SMILES already is the identity.
    assert periodic_identity("[*]CO[*]") == periodic_identity("[*]OC[*]")
    assert periodic_identity("[*][Si](C)(C)O[*]") == periodic_identity("[*]O[Si](C)(C)[*]")
    assert periodic_identity("c1ccccc1") is None


def test_a_monomer_that_is_sold_is_not_a_polymer_that_is_sold():
    """Three places the two are easy to confuse, and all three say reported.

    Allyl glycidyl ether, methylvinyldichlorosilane and sodium disulfide are all
    bought in bulk, and in every case what is sold is a *copolymer* carrying a
    few per cent of the unit, or a chain of a different sulfur rank.  Calling the
    homopolymer commercial would be the claim a ranking cannot check.
    """
    by_name = {r.name: r for r in records()}
    for name, must_say in (
        ("poly(allyl glycidyl ether)", "termonomer"),
        ("poly(methylvinylsiloxane)", "VMQ"),
        ("poly(ethylene disulfide)", "tetra"),
    ):
        record = by_name[name]
        assert record.availability is Availability.REPORTED, name
        assert must_say in record.note, name
        assert "commercial" not in record.label
    # The rubber sold as Thiokol A is the tetrasulfide, so no unit may claim it.
    assert not any("Thiokol A" in label for label, _ in units())
    # LARC-TPI is BTDA with 3,3'-diaminobenzophenone, which is not a diamine in
    # this library, so no unit may claim that name either.
    assert not any("Larc" in label or "LARC" in label for label, _ in units())
    assert by_name["poly(BTDA-MDA imide)"].availability is Availability.REPORTED


def test_the_named_high_performance_polymers_are_reachable_by_name():
    """PEEK, Udel, Radel, Lexan, Kapton and Ultem, because they are the point.

    These are the materials a specification asking for 250 C service or 3 GPa
    modulus actually wants, and a search cannot return what is not in its space.
    """
    labels = {canon(s): n for n, s in units()}
    for smiles, fragment in (
        ("[*]Oc1ccc(Oc2ccc(C(=O)c3ccc([*])cc3)cc2)cc1", "PEEK"),
        (ARYL_ETHER.format(
            bisphenol=PARA + "C(C)(C)" + PARA, activated=PARA + "S(=O)(=O)" + PARA
        ), "Udel"),
        (ARYL_ETHER.format(
            bisphenol=PARA + PARA, activated=PARA + "S(=O)(=O)" + PARA
        ), "Radel"),
        (CARBONATE.format(core=PARA + "C(C)(C)" + PARA), "Lexan"),
        ("[*]N5C(=O)c6cc7c(cc6C5=O)C(=O)N(c1ccc(cc1)Oc1ccc(cc1)[*])C7=O", "Kapton"),
        ("[*][Si](C)(C)O[*]", "PDMS"),
        ("[*]S" + PARA + "[*]", "Ryton"),
    ):
        assert canon(smiles) in labels, fragment
        assert fragment in labels[canon(smiles)], labels[canon(smiles)]


# --------------------------------------------------------------------------
# 3. The gates - a grammar is defined by what it refuses
# --------------------------------------------------------------------------


def test_every_probe_is_refused_by_a_named_gate_with_a_reason():
    refused = {r.name: r for r in refusals()}
    emitted = {name for name, _ in units()}
    for candidate in hetero.PROBES:
        assert candidate.name in refused, f"{candidate.name} slipped past every gate"
        assert candidate.label not in emitted
        refusal = refused[candidate.name]
        assert refusal.gate, candidate.name
        assert len(refusal.reason) > 60, f"{candidate.name}: a reason must be a reason"


def test_every_gate_has_at_least_one_refusal_to_its_name():
    """A gate with nothing to refuse is a claim, not a check."""
    fired = collections.Counter(r.gate for r in refusals())
    assert set(fired) == {
        "unstrained-ring",
        "five-ring-lactone",
        "cyclic-carbonate",
        "aryl-carbamate",
        "aryl-ether",
        "free-hydroxyl",
        "pendant-polymerisable",
        "backbone-bond",
        "no-homopolymer",
        # Not a chemistry gate: the library-level check that one polymer does not
        # reach the ranking twice under two names.  See SAME_CHAIN_TWICE.
        "duplicate-cut",
    }
    assert sum(fired.values()) == MEASURED["refusals"]


def test_nothing_the_library_proposes_is_refused():
    """The probes and the two deliberate in-library refusals are the whole list.

    A surprise here means the tables and the gates disagree about chemistry, and
    silently shrinking the library is the worst way to resolve that.
    """
    expected = {p.name for p in hetero.PROBES} | {
        # Built inside the generation loop on purpose, so that the gate fires
        # against the real library rather than only against a hand-written probe.
        "poly(pentamethylene oxide)",
        "poly(ethylene carbonate)",
        "poly(propylene carbonate)",
        "poly(MDI-bisphenol A urethane)",
        "poly(TDI-bisphenol A urethane)",
        "poly(PPDI-bisphenol A urethane)",
        "poly(HDI-bisphenol A urethane)",
        "poly(IPDI-bisphenol A urethane)",
        "poly(H12MDI-bisphenol A urethane)",
        # Not refused for being unmakeable - refused for being the polymer two
        # rows above it, written at the other cut.
        SAME_CHAIN_TWICE[0],
    }
    surprises = [r for r in refusals() if r.name not in expected]
    assert not surprises, [(r.name, r.gate, r.reason) for r in surprises]


@pytest.mark.parametrize(
    "name, smiles, gate",
    [
        # Odian ch. 7: the unstrained six-ring has no free energy to give.
        ("tetrahydropyran", "[*]CCCCCO[*]", "unstrained-ring"),
        ("1,4-dioxane", "[*]OCCOCC[*]", "unstrained-ring"),
        ("thiane", "[*]CCCCCS[*]", "unstrained-ring"),
        # A five-ring carbonate is a battery solvent, not a monomer.
        ("ethylene carbonate", "[*]OC(=O)OCC[*]", "cyclic-carbonate"),
        ("propylene carbonate", "[*]OC(=O)OCC(C)[*]", "cyclic-carbonate"),
        # An isocyanate plus a phenol is a blocked isocyanate.
        ("MDI + bisphenol A",
         "[*]N" + PARA + "C" + PARA + "NC(=O)O" + PARA + "C(C)(C)" + PARA + "OC(=O)[*]",
         "aryl-carbamate"),
        ("HDI + hydroquinone", "[*]NCCCCCCNC(=O)O" + PARA + "OC(=O)[*]", "aryl-carbamate"),
        # SNAr needs activation; oxidative coupling needs blocked ortho positions.
        ("bisphenol A + dichlorobiphenyl",
         ARYL_ETHER.format(bisphenol=PARA + "C(C)(C)" + PARA, activated=PARA + PARA),
         "aryl-ether"),
        ("hydroquinone + dichlorobenzene",
         ARYL_ETHER.format(bisphenol=PARA, activated=PARA), "aryl-ether"),
        ("phenol", "[*]Oc1ccc(cc1)[*]", "aryl-ether"),
        ("o-cresol", "[*]Oc1ccc([*])cc1C", "aryl-ether"),
        # A spare hydroxyl is a branch point, not a substituent.
        ("glycidol", "[*]CC(CO)O[*]", "free-hydroxyl"),
        ("trimethylolpropane oxetane", "[*]CC(CC)(CO)CO[*]", "free-hydroxyl"),
        # A second polymerisable group is a crosslinker.
        ("glycidyl methacrylate", "[*]CC(COC(=O)C(C)=C)O[*]", "pendant-polymerisable"),
        ("4-vinylphenyl glycidyl ether", "[*]CC(COc1ccc(C=C)cc1)O[*]",
         "pendant-polymerisable"),
        # A peroxide is an initiator and a hydrazine is a reducing agent.
        ("ethylene peroxide", "[*]CCOO[*]", "backbone-bond"),
        ("ethylene hydrazine", "[*]CCNNCC[*]", "backbone-bond"),
        # The five-membered lactone is the gap in the series: the four-ring and
        # the seven-ring open, gamma-butyrolactone is a solvent.
        ("gamma-butyrolactone", "[*]CCCC(=O)O[*]", "five-ring-lactone"),
        ("gamma-valerolactone", "[*]C(C)CCC(=O)O[*]", "five-ring-lactone"),
        # Curated: real rings, real compounds, no homopolymer.
        ("2-methyltetrahydrofuran", "[*]C(C)CCCO[*]", "no-homopolymer"),
        ("tetrahydrothiophene", "[*]CCCCS[*]", "no-homopolymer"),
        # And the structural contract itself.
        ("one attachment point", "[*]CCO", "structure"),
    ],
)
def test_the_gate_that_fires_is_the_right_one(name, smiles, gate):
    verdict = hetero.gate(probe(name, smiles))
    assert verdict is not None, f"{name} was admitted"
    assert verdict.gate == gate, f"{name} refused by {verdict.gate}: {verdict.reason}"


@pytest.mark.parametrize(
    "name, smiles",
    [
        # The five-ring and the seven-ring open; only the six-ring does not.
        ("tetrahydrofuran", "[*]CCCCO[*]"),
        ("oxepane", "[*]CCCCCCO[*]"),
        ("1,3-dioxolane", "[*]COCCO[*]"),
        # A carbonyl pays for a six-ring: trimethylene carbonate is a real,
        # commercial, bioresorbable monomer and shares the gate's ring size.
        ("trimethylene carbonate", "[*]OC(=O)OCCC[*]"),
        ("epsilon-caprolactone", "[*]CCCCCC(=O)O[*]"),
        # The lactones either side of the five-ring, which both open: the strain
        # of the four-ring pays for it and the seven-ring is strained again.
        ("beta-propiolactone", "[*]CCC(=O)O[*]"),
        ("delta-valerolactone", "[*]CCCCC(=O)O[*]"),
        # A five-atom backbone with one oxygen and no ester: an ether, not a
        # lactone, and tetrahydrofuran is the monomer the whole family starts at.
        ("tetrahydrofuran again", "[*]CCCCO[*]"),
        # The S-S of the Thiokols and the Si-O of the silicones are the two
        # heteroatom-heteroatom bonds that are real polymer linkages.
        ("Thiokol A", "[*]CCSS[*]"),
        ("Thiokol LP", "[*]CCOCOCCSS[*]"),
        ("PDMS", "[*][Si](C)(C)O[*]"),
        # An allyl ether does not propagate, it transfers - which is why allyl
        # glycidyl ether is sold as a latent cure site.
        ("allyl glycidyl ether", "[*]CC(COCC=C)O[*]"),
        # ...and a vinyl on silicon survives siloxane equilibration.
        ("methylvinylsiloxane", "[*][Si](C)(C=C)O[*]"),
        # Both ortho positions blocked, by methyl or by phenyl.
        ("2,6-dimethylphenol", "[*]Oc1cc(C)c([*])c(C)c1"),
        ("2,6-diphenylphenol", "[*]Oc1cc(-c2ccccc2)c([*])c(-c2ccccc2)c1"),
        # Activated: the carbonyl and the sulfone hold the Meisenheimer charge.
        ("PEEK", "[*]Oc1ccc(Oc2ccc(C(=O)c3ccc([*])cc3)cc2)cc1"),
        ("PES", "[*]O" + PARA + "S(=O)(=O)" + PARA + "[*]"),
        # An aliphatic urethane, which is the whole polyurethane sub-family.
        ("MDI + 1,4-butanediol",
         "[*]N" + PARA + "C" + PARA + "NC(=O)OCCCCOC(=O)[*]"),
        # An aryl ether inside a polyimide: activated by the imide carbonyl,
        # which RDKit perceives as aromatic and an EWG query must still see.
        ("Kapton", "[*]N5C(=O)c6cc7c(cc6C5=O)C(=O)N(c1ccc(cc1)Oc1ccc(cc1)[*])C7=O"),
        # No ether oxygen at all, so the aryl-ether gate must not reach for it.
        ("poly(phenylene sulfide)", "[*]S" + PARA + "[*]"),
    ],
)
def test_the_gates_do_not_refuse_real_polymers(name, smiles):
    verdict = hetero.gate(probe(name, smiles))
    assert verdict is None, f"{name} refused by {verdict.gate}: {verdict.reason}"


def test_the_co2_route_is_what_admits_poly_propylene_carbonate():
    """The same structure, refused as a condensation and admitted from CO2.

    Without the distinction this module would be claiming that 1,2-propanediol
    and a carbonate source give a polymer, which is the thing this sub-family is
    most often wrong about: what they give is propylene carbonate, a solvent.
    """
    smiles = CARBONATE.format(core="CC(C)")
    direct = hetero.gate(probe("condensation", smiles))
    assert direct is not None and direct.gate == "cyclic-carbonate"
    assert hetero.gate(probe("from CO2", smiles, route="CO2 + epoxide")) is None

    record = next(r for r in records() if r.name == "poly(propylene carbonate)")
    assert record.route != "direct"
    assert record.monomers == ("propylene oxide", "carbon dioxide")
    # ...and the refused condensation is still reported by name.
    assert any(
        r.name == "poly(propylene carbonate)" and r.gate == "cyclic-carbonate"
        for r in refusals()
    )


def test_the_five_ring_lactone_gate_refuses_the_monomer_and_not_the_chain():
    """poly(4-hydroxybutyrate) is this chain, is sold, and is not refused anywhere.

    The gate's claim is about gamma-butyrolactone, which does not open, and not
    about the polymer it would give if it did: ``condensation.py`` writes the
    same chain from 4-hydroxybutyric acid and it is a bioresorbable product. A
    unit carrying a route other than ``direct`` is therefore exempt, which is the
    device the CO2 carbonates already use.
    """
    from formulate.exploration.monomers import condensation

    smiles = "[*]CCCC(=O)O[*]"
    direct = hetero.gate(probe("gamma-butyrolactone", smiles))
    assert direct is not None and direct.gate == "five-ring-lactone"
    assert "4-hydroxybutyrate" in direct.reason
    assert hetero.gate(probe("from the hydroxy acid", smiles, route="hydroxy acid")) is None

    # ...and the chain really is one condensation.py emits, at its own cut.
    elsewhere = [
        n
        for n, s in condensation.units()
        if condensation.periodic_key(s) == condensation.periodic_key(smiles)
    ]
    assert elsewhere, "condensation.py no longer writes poly(4-hydroxybutyrate)"
    # This module does not write it, so the two families still share one chain only.
    assert smiles not in {s for _, s in units()}


def test_the_curated_refusal_table_parses():
    for smiles in hetero._NO_HOMOPOLYMER:
        assert hetero.canonical(smiles) is not None, smiles


# --------------------------------------------------------------------------
# 4. Mutation checks: is each gate load-bearing?
# --------------------------------------------------------------------------
#
# A gate can rot into a no-op while its test goes on passing, because some other
# gate happens to catch the same probe first.  These two remove a gate at
# runtime and assert that the structure it was standing in front of then reaches
# units().  If a mutation test stops failing, the gate has stopped mattering.


def _without(gate_function):
    return tuple(g for g in hetero._GATES if g is not gate_function)


@pytest.mark.parametrize(
    "gate_function, name, smiles",
    [
        (hetero._gate_unstrained_ring, "poly(pentamethylene oxide)", "[*]CCCCCO[*]"),
        (hetero._gate_aryl_carbamate, "poly(MDI-bisphenol A urethane)", None),
    ],
)
def test_removing_a_gate_lets_its_probe_through(monkeypatch, gate_function, name, smiles):
    before = {n for n, _ in units()}
    assert not any(n.startswith(name) for n in before), f"{name} is already emitted"

    monkeypatch.setattr(hetero, "_GATES", _without(gate_function))
    hetero._built.cache_clear()
    try:
        after = {n for n, _ in units()}
        assert any(n.startswith(name) for n in after), (
            f"removing {gate_function.__name__} changed nothing, so it is not the gate "
            f"holding {name} back"
        )
        if smiles is not None:
            assert hetero.gate(probe(name, smiles)) is None
    finally:
        monkeypatch.undo()
        hetero._built.cache_clear()

    assert {n for n, _ in units()} == before, "the gate was not restored"


# --------------------------------------------------------------------------
# 5. The flags - a label has to carry what changes the meaning of a number
# --------------------------------------------------------------------------


def test_availability_is_carried_in_the_name_and_the_products_are_a_minority():
    labels = {name for name, _ in units()}
    for label in labels:
        assert any(
            word in label for word in ("commercial", "reported", "constructible")
        ), label
    counts = collections.Counter(r.availability.value for r in records())
    assert counts == MEASURED["availability"]
    assert counts["commercial"] < len(records()) / 3, "a search space is mostly unexplored"
    # Calling something commercial is a claim, so it has to come with something
    # a reader can check: a trade name, or a sentence saying what it is sold as.
    # Unlike the condensation family, many of these are known by their systematic
    # name - nobody calls poly(ethylene oxide) anything else - so a trade name
    # cannot be required outright.
    for record in records():
        if record.availability is Availability.COMMERCIAL:
            assert record.common_name or record.note, (
                f"{record.name} is called commercial with nothing to check it against"
            )


def test_every_monomer_is_one_you_can_buy():
    """No unit here needs a custom synthesis, which is the point of a library.

    ``vinyl.py`` allows a small "literature" tail; this family does not have
    one, because every monomer in every table above is a catalogue or commodity
    chemical.  If that ever stops being true it should stop quietly.
    """
    counts = collections.Counter(r.scale.value for r in records())
    assert counts == MEASURED["monomer_scale"]
    assert set(counts) == {"bulk", "catalogue"}


def test_the_element_budget_tracks_the_model_it_is_a_statement_about():
    assert hetero.ELEMENT_BUDGET == _FITTED_ELEMENTS


def test_silicon_and_sulfur_are_flagged_rather_than_refused():
    """The deliberate difference from ``vinyl.py``, asserted so it stays deliberate.

    Refusing out-of-budget elements here would delete every silicone, every
    sulfone and every sulfide, and unlike poly(vinyl bromide) none of those has
    an in-budget twin.  They are emitted, they say so in the label, and the
    density and Tg experts refuse them by their own domain check.
    """
    outside = [r for r in records() if r.outside_element_budget]
    assert len(outside) == MEASURED["outside_element_budget"]
    assert {"Si", "S"} & {e for r in outside for e in hetero.elements(r.smiles)}
    for record in outside:
        assert "outside fitted elements" in record.label
    for record in records():
        if not record.outside_element_budget:
            assert "outside fitted elements" not in record.label
            assert hetero.elements(record.smiles) <= hetero.ELEMENT_BUDGET
    # PDMS in particular, since it is the polymer the density model is on record
    # as getting wrong.
    assert any(r.name == "poly(dimethylsiloxane)" for r in outside)


def test_melt_processability_is_legible_from_the_name():
    counts = collections.Counter(r.melt.value for r in records())
    assert counts == MEASURED["melt"]
    by_name = {r.name: r for r in records()}
    # Kapton is cast and imidised in place; Ultem is injection moulded. A search
    # for a melt-processed part must be able to tell them apart by name alone.
    assert by_name["poly(PMDA-ODA imide)"].melt is Melt.NONE
    assert "not melt processable" in by_name["poly(PMDA-ODA imide)"].label
    assert by_name["poly(BPADA-MPD imide)"].melt is Melt.MELT
    assert "not melt processable" not in by_name["poly(BPADA-MPD imide)"].label
    # Every polyimide whose dianhydride has no ether or hexafluoro swivel is a
    # solid that decomposes before it flows.
    rigid = [r for r in records() if r.subfamily == "polyimide" and r.melt is Melt.NONE]
    assert len(rigid) == 21
    assert all(r.name.startswith(("poly(PMDA", "poly(BPDA", "poly(BTDA")) or
               r.name.endswith("PPD imide)") for r in rigid)


def test_every_polyurethane_says_it_is_only_the_hard_segment():
    """A silent average over a block copolymer is what this repository forbids.

    A polyurethane is a two-phase segmented copolymer and these units are one
    phase of it, so every number predicted from them describes the hard segment
    and not the material.  That has to be in the name, because ``(name, smiles)``
    is the only channel the explorer has.
    """
    urethanes = [r for r in records() if r.subfamily == "polyurethane"]
    assert len(urethanes) == MEASURED["subfamilies"]["polyurethane"]
    for record in urethanes:
        assert record.melt is Melt.HARD_SEGMENT
        assert "hard segment of a block copolymer" in record.label
        assert "HARD SEGMENT" in record.note and "soft segment" in record.note


def test_measured_matches_what_the_module_actually_generates():
    assert len(units()) == MEASURED["units"]
    assert len(refusals()) == MEASURED["refusals"]
    assert collections.Counter(r.subfamily for r in records()) == MEASURED["subfamilies"]


# --------------------------------------------------------------------------
# 6. What the engine's own feasibility expert says
# --------------------------------------------------------------------------


@pytest.mark.skipif(not EXPERT.is_available(), reason=EXPERT.unavailable_reason())
def test_the_feasibility_expert_reaches_everything_except_the_named_27():
    """Every unit is either scored or on a list that says why it is not.

    A missing synthetic-accessibility number is penalised by the default
    missing-objective policy, so a unit the expert cannot route is handicapped
    for a reason that is about the expert's route table rather than about the
    polymer.  Naming them is what stops that penalty from looking like a verdict.
    """
    scored, unroutable = [], []
    for record in records():
        value = score(record.smiles)
        (unroutable if value is None else scored).append(record.name)
        if value is not None:
            assert 1.0 <= value <= 10.0, record.name
    assert set(unroutable) == NO_FEASIBILITY_ROUTE, (
        "the expert's route table moved: "
        f"newly unroutable {sorted(set(unroutable) - NO_FEASIBILITY_ROUTE)}, "
        f"newly routable {sorted(NO_FEASIBILITY_ROUTE - set(unroutable))}"
    )
    assert len(scored) == MEASURED["feasibility_scored"]
    assert len(unroutable) == MEASURED["feasibility_refused"]


@pytest.mark.skipif(not EXPERT.is_available(), reason=EXPERT.unavailable_reason())
def test_the_polyimides_the_expert_can_route_are_exactly_the_aryl_ether_ones():
    """A sharp line, and the right one: it is how polyetherimide is really made.

    The expert has no imide chemistry, so it reaches a polyimide only when the
    dianhydride carries an aryl ether it can cut by nucleophilic aromatic
    substitution - which is the commercial route to Ultem, not a consolation.
    """
    routed = {
        r.name
        for r in records()
        if r.subfamily == "polyimide" and score(r.smiles) is not None
    }
    assert routed == {
        f"poly({code}-{amine} imide)"
        for code in ("ODPA", "BPADA")
        for amine in ("PPD", "MPD", "ODA", "MDA", "DDS", "BAPP")
    }


@pytest.mark.skipif(not EXPERT.is_available(), reason=EXPERT.unavailable_reason())
def test_the_commodity_polymers_land_near_the_commodity_band():
    # Not a calibration, a sanity check on the grammar: if a megatonne monomer
    # came back priced like a research compound, the repeat unit would be
    # written wrongly rather than the monomer being hard.
    for smiles, ceiling in (
        ("[*]CCO[*]", 2.5),                        # poly(ethylene oxide)
        ("[*]CC(C)O[*]", 3.0),                     # poly(propylene oxide)
        ("[*]CCCCO[*]", 2.6),                      # poly(tetramethylene oxide)
        ("[*]CO[*]", 3.1),                         # polyoxymethylene
        ("[*]CCCCCC(=O)O[*]", 2.3),                # polycaprolactone
        ("[*]Oc1ccc(Oc2ccc(C(=O)c3ccc([*])cc3)cc2)cc1", 2.6),   # PEEK
        (CARBONATE.format(core=PARA + "C(C)(C)" + PARA), 3.4),  # bisphenol A PC
    ):
        value = score(smiles)
        assert value is not None and value <= ceiling, (smiles, value)


@pytest.mark.skipif(not EXPERT.is_available(), reason=EXPERT.unavailable_reason())
def test_the_silicones_are_where_the_expert_says_it_is_least_trustworthy():
    """Reported rather than corrected, because correcting it would be fitting.

    ``polymer_feasibility`` records in its own source that its fragment
    statistics barely contain silicon and that PDMS therefore scores like a
    research compound.  This asserts that is still what happens, so that the
    silicone scores in a ranking are a known disagreement rather than a surprise.
    """
    silicones = [score(r.smiles) for r in records() if r.subfamily == "polysiloxane"]
    assert all(v is not None for v in silicones)
    assert min(silicones) > 3.5, "the expert stopped over-charging silicon"
    assert max(silicones) == pytest.approx(MEASURED["feasibility_max"], abs=0.05)
    # ...and it is the top of the module, which is the fact worth seeing.
    everything = [score(r.smiles) for r in records()]
    assert max(v for v in everything if v is not None) == max(silicones)
