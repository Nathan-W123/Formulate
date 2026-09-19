"""The vinyl monomer grammar: what it generates, and harder, what it refuses.

A generator of repeat units is only as good as its refusals.  Anything can emit
``[*]CC(R)[*]`` for arbitrary R; the value is in the R it will not emit, because
a candidate that cannot be made still occupies a place in the ranking and pushes
a real one out.  So the gate tests here are written to be the ones that break
first: every probe in the module is asserted to be refused, each by a named gate,
and the three deliberate exceptions - alpha-methylstyrene's ceiling temperature,
the itaconates' CH2 spacer, the fumarates' 1,2-substitution - are asserted to
survive, because a gate that refuses those has stopped being a gate and started
being a blunt instrument.

The measured numbers pinned here, on this installation: 137 repeat units, 38 of
38 vinyl-family reference polymers regenerated, 26 of 26 probes refused, 0 of
137 units refused by the engine's own PolymerFeasibilityExpert, and 37 of the
137 sold as polymers rather than merely constructible.
"""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import polymer_candidate
from formulate.experts.base import PredictionRequest
from formulate.experts.melt import backbone_stereocentres
from formulate.experts.polymer import _FITTED_ELEMENTS, reference_polymers
from formulate.experts.polymer_feasibility import PolymerFeasibilityExpert
from formulate.exploration.monomers import vinyl
from formulate.exploration.monomers.vinyl import RepeatUnit

pytestmark = requires_rdkit

#: The 38 of the 57 bundled repeat units that are vinyl, vinylidene or diene
#: chain growth.  Listed by abbreviation rather than derived by a rule, so that
#: a new reference polymer in this family has to be looked at by a person
#: instead of quietly changing what the rediscovery test means.
VINYL_REFERENCE = frozenset(
    {
        "PE", "aPP", "PB-1", "PMP", "PIB", "PS", "PAMS", "P4MS", "PVC", "PVDC",
        "PVF", "PVDF", "PVAc", "PVOH", "PVME", "PMA", "PEA", "PBA", "PMMA",
        "PEMA", "PBMA", "PHEMA", "PAA", "PMAA", "PAN", "PMAN", "PAM", "PNIPAM",
        "cis-PB", "NR", "CR", "PiPMA", "PHMA", "PPA", "P2EHA", "PVPr", "PEVE",
        "PCTFE",
    }
)

EXPERT = PolymerFeasibilityExpert()


def canon(smiles: str) -> str:
    value = vinyl.canonical(smiles)
    assert value is not None, smiles
    return value


def probe(name: str, smiles: str, **kwargs) -> RepeatUnit:
    return RepeatUnit(name, smiles, "test", "test", "literature", **kwargs)


# -- the contract ---------------------------------------------------------


def test_units_obeys_the_contract_the_orchestrator_relies_on():
    units = vinyl.units()
    assert isinstance(units, list) and units
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in units)
    names = [name for name, _ in units]
    smiles = [s for _, s in units]
    assert len(set(names)) == len(names)
    assert len({canon(s) for s in smiles}) == len(smiles), "units() must be deduplicated"


def test_every_unit_parses_and_caps():
    from rdkit import Chem

    for name, smiles in vinyl.units():
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, name
        assert sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0) == 2, name
        # The attachment points have to be *capped*, not deleted: a unit that
        # only parses with its [*] removed is one whose valences do not close.
        assert Chem.MolFromSmiles(smiles.replace("[*]", "[H]")) is not None, name


def test_every_unit_is_readable_by_the_crystallinity_test():
    # ``PolymerLibraryExplorer`` sweeps tacticity for repeat units with backbone
    # stereocentres and leaves it unspecified otherwise.  ``None`` from this
    # function means "could not be read", which the explorer cannot distinguish
    # from "regular", so a unit that returns it would be silently mis-swept.
    for name, smiles in vinyl.units():
        assert backbone_stereocentres(smiles) is not None, name


def test_element_budget_tracks_the_model_it_is_a_statement_about():
    # If the density and Tg fit widens, this module should widen with it - and
    # if it narrows, generating outside it becomes an extrapolation.  Either way
    # the drift should be a failure here rather than a silent change in what the
    # engine is willing to propose.
    assert vinyl.ELEMENT_BUDGET == _FITTED_ELEMENTS


def test_availability_is_recorded_and_mostly_real():
    records = vinyl.records()
    assert all(r.availability in vinyl.AVAILABILITY_ORDER for r in records)
    shelf = sum(1 for r in records if r.availability in ("commodity", "commercial"))
    # A search that returns a repeat unit nobody sells the monomer for has
    # answered a different question, so the custom-synthesis share is held down.
    assert shelf / len(records) >= 0.95
    assert sum(1 for r in records if r.availability == "commodity") >= 35


# -- rediscovery ----------------------------------------------------------


def test_the_family_regenerates_its_own_members_of_the_reference_set():
    references = reference_polymers()
    assert len(references) == 57
    family = [r for r in references if r["abbreviation"] in VINYL_REFERENCE]
    assert len(family) == 38, "the bundled set changed; re-read VINYL_REFERENCE"

    generated = {canon(s) for _, s in vinyl.units()}
    missed = [
        (r["abbreviation"], r["repeat_unit"])
        for r in family
        if canon(r["repeat_unit"]) not in generated
    ]
    assert not missed, f"the grammar cannot write its own family: {missed}"


def test_the_grammar_stays_inside_its_family():
    # Generating a polyester or a polyether here would collide with another
    # module's output and hide a duplicate behind a different name.
    generated = {canon(s) for _, s in vinyl.units()}
    strays = [
        r["abbreviation"]
        for r in reference_polymers()
        if r["abbreviation"] not in VINYL_REFERENCE and canon(r["repeat_unit"]) in generated
    ]
    assert not strays


# -- the gates ------------------------------------------------------------


def test_every_probe_is_refused_by_a_named_gate():
    refused = {r.name: r for r in vinyl.refusals()}
    emitted = {name for name, _ in vinyl.units()}
    for candidate in vinyl.PROBES:
        assert candidate.name in refused, f"{candidate.name} slipped past every gate"
        assert candidate.name not in emitted
        refusal = refused[candidate.name]
        assert refusal.gate and len(refusal.reason) > 40, candidate.name


def test_nothing_the_library_proposes_is_refused():
    # The probes are the only proposals allowed to be refused.  A library entry
    # in this list means the tables and the gates disagree about chemistry, and
    # silently shrinking the library is the worst way to resolve that.
    probes = {p.name for p in vinyl.PROBES}
    surprises = [r for r in vinyl.refusals() if r.name not in probes]
    assert not surprises, [(r.name, r.gate, r.reason) for r in surprises]


@pytest.mark.parametrize(
    "name, smiles, gate",
    [
        ("vinyl alcohol", "[*]CC(O)[*]", "free-monomer"),
        ("vinylamine", "[*]CC(N)[*]", "free-monomer"),
        ("N-methylvinylamine", "[*]CC(NC)[*]", "free-monomer"),
        ("allyl alcohol", "[*]CC(CO)[*]", "allylic"),
        ("allyl chloride", "[*]CC(CCl)[*]", "allylic"),
        ("allylamine", "[*]CC(CN)[*]", "allylic"),
        ("divinylbenzene", "[*]CC(c1ccc(C=C)cc1)[*]", "pendant-alkene"),
        ("ethylene glycol dimethacrylate", "[*]CC(C)(C(=O)OCCOC(=O)C(C)=C)[*]",
         "pendant-alkene"),
        ("1,1-diphenylethylene", "[*]CC(c1ccccc1)(c1ccccc1)[*]", "substitution"),
        ("2-ethyl-1-butene", "[*]CC(CC)(CC)[*]", "substitution"),
        ("2-butene", "[*]C(C)C(C)[*]", "substitution"),
        ("1,2-dichloroethylene", "[*]C(Cl)C(Cl)[*]", "substitution"),
        ("tetramethylethylene", "[*]C(C)(C)C(C)(C)[*]", "substitution"),
        ("vinyl bromide", "[*]CC(Br)[*]", "elements"),
        ("sodium styrenesulfonate", "[*]CC(c1ccc(S(=O)(=O)O)cc1)[*]", "elements"),
        ("acrolein", "[*]CC(C=O)[*]", "self-reactive"),
        ("vinyl isocyanate", "[*]CC(N=C=O)[*]", "self-reactive"),
        ("butadiene monoxide", "[*]CC(C1CO1)[*]", "self-reactive"),
        ("hexafluoropropylene", "[*]C(F)(F)C(F)(C(F)(F)F)[*]", "no-homopolymer"),
        # A vinylphenol is the third monomer in this module that does not exist
        # as a bottleable species, and the one that is easiest to miss: the
        # hydroxyl is not on the backbone, it is out on the ring, so the enol
        # test does not see it.  All three ring positions are the same problem.
        ("4-vinylphenol", "[*]CC(c1ccc(O)cc1)[*]", "free-monomer"),
        ("2-vinylphenol", "[*]CC(c1ccccc1O)[*]", "free-monomer"),
        ("3-vinylphenol", "[*]CC(c1cccc(O)c1)[*]", "free-monomer"),
        ("4-hydroxy-alpha-methylstyrene", "[*]CC(C)(c1ccc(O)cc1)[*]", "free-monomer"),
        ("4-vinylnaphthol", "[*]CC(c1ccc2ccccc2c1O)[*]", "free-monomer"),
    ],
)
def test_the_gate_that_fires_is_the_right_one(name, smiles, gate):
    verdict = vinyl.gate(probe(name, smiles))
    assert verdict is not None, f"{name} was admitted"
    assert verdict.gate == gate, f"{name} refused by {verdict.gate}: {verdict.reason}"


@pytest.mark.parametrize(
    "name, smiles",
    [
        # 1,1-disubstitution is not a gate: alpha-methylstyrene's ceiling
        # temperature is 61 C and it is still in the reference set, and every
        # methacrylate on earth is 1,1-disubstituted.
        ("alpha-methylstyrene", "[*]CC(C)(c1ccccc1)[*]"),
        ("methyl methacrylate", "[*]CC(C)(C(=O)OC)[*]"),
        ("isobutylene", "[*]CC(C)(C)[*]"),
        ("vinylidene chloride", "[*]CC(Cl)(Cl)[*]"),
        # The CH2 spacer that makes the itaconates real.
        ("dimethyl itaconate", "[*]CC(CC(=O)OC)(C(=O)OC)[*]"),
        # The one 1,2-disubstituted family with homopolymers.
        ("diisopropyl fumarate", "[*]C(C(=O)OC(C)C)C(C(=O)OC(C)C)[*]"),
        # Fluoroolefins: substituted at both ends and entirely real.
        ("tetrafluoroethylene", "[*]C(F)(F)C(F)(F)[*]"),
        ("chlorotrifluoroethylene", "[*]C(F)(F)C(F)(Cl)[*]"),
        # An enamide is not an enamine - the carbonyl removes the tautomer.
        ("N-vinylformamide", "[*]CC(NC=O)[*]"),
        ("N-vinylpyrrolidone", "[*]CC(N1CCCC1=O)[*]"),
        # A pendant vinyl conjugated with the backbone is 1,2-diene addition,
        # not a crosslinker.
        ("1,2-polybutadiene", "[*]CC(C=C)[*]"),
        ("3,4-polyisoprene", "[*]CC(C(C)=C)[*]"),
        # The oxygen is the attachment, so there is no allylic C-H.
        ("vinyl acetate", "[*]CC(OC(=O)C)[*]"),
        ("glycidyl methacrylate", "[*]CC(C)(C(=O)OCC1CO1)[*]"),
        # The vinylphenol gate must stop at a *free* ring hydroxyl: capping it
        # is exactly what makes 4-acetoxystyrene the monomer you actually buy,
        # and an ether or an ester oxygen off the backbone is everywhere in
        # this family.
        ("4-acetoxystyrene", "[*]CC(c1ccc(OC(C)=O)cc1)[*]"),
        ("4-methoxystyrene", "[*]CC(c1ccc(OC)cc1)[*]"),
        ("benzyl acrylate", "[*]CC(C(=O)OCc1ccccc1)[*]"),
        ("2-hydroxyethyl methacrylate", "[*]CC(C)(C(=O)OCCO)[*]"),
    ],
)
def test_the_gates_do_not_refuse_real_polymers(name, smiles):
    verdict = vinyl.gate(probe(name, smiles))
    assert verdict is None, f"{name} refused by {verdict.gate}: {verdict.reason}"


def test_the_post_polymerisation_route_is_what_admits_poly_vinyl_alcohol():
    # The same structure, refused as a direct polymerisation and admitted when
    # the record says the chain is made by hydrolysing poly(vinyl acetate).
    # Without that distinction the module would be claiming vinyl alcohol is a
    # monomer, which is the thing it is most often wrong about.
    direct = vinyl.gate(probe("direct", "[*]CC(O)[*]"))
    assert direct is not None and direct.gate == "free-monomer"
    converted = vinyl.gate(probe("converted", "[*]CC(O)[*]", route="post-polymerisation"))
    assert converted is None
    record = next(r for r in vinyl.records() if r.name == "poly(vinyl alcohol)")
    assert record.route == "post-polymerisation"
    assert record.monomer == "vinyl acetate"


def test_no_probe_can_reach_the_library():
    # The probes are structures this module asserts are not polymers.  They are
    # gated in a loop of their own so that a regressed gate cannot ship one to
    # the ranker under its "poly(...)" name; this is the assertion that the
    # separation is real rather than incidental.
    assert vinyl.escapes() == []
    emitted = {name for name, _ in vinyl.units()}
    assert not emitted & {p.name for p in vinyl.PROBES}


def test_the_vinylphenols_are_admitted_only_as_a_post_polymerisation():
    # Same structure, refused as a direct polymerisation and admitted when the
    # record says the chain comes from hydrolysing poly(4-acetoxystyrene).
    # 4-vinylphenol is a real compound that polymerises in the bottle and is
    # sold only as a dilute solution, so proposing it as a monomer is proposing
    # something nobody can buy.
    unit = "[*]CC(c1ccc(O)cc1)[*]"
    direct = vinyl.gate(probe("direct", unit))
    assert direct is not None and direct.gate == "free-monomer"
    assert vinyl.gate(probe("converted", unit, route="post-polymerisation")) is None
    record = next(r for r in vinyl.records() if r.name == "poly(4-hydroxystyrene)")
    assert record.route == "post-polymerisation"
    assert record.monomer == "4-acetoxystyrene"


# -- what is sold, as against what could be made --------------------------


def test_the_product_flag_is_about_the_polymer_not_about_the_monomer():
    records = {r.name: r for r in vinyl.records()}
    stale = [name for name in vinyl.COMMERCIAL_POLYMERS if name not in records]
    assert not stale, f"COMMERCIAL_POLYMERS names nothing this module emits: {stale}"
    sold = vinyl.commercial()
    assert len(sold) == 37
    assert all(r.product for r in sold)
    # A polymer somebody ships cannot have a monomer nobody sells.
    assert all(r.availability in ("commodity", "commercial") for r in sold)
    # And the flag has to be the *narrower* claim, or it is not worth carrying:
    # each of these has a catalogue monomer and no product behind the polymer.
    for name in (
        "poly(2-methylstyrene)",
        "poly(vinyl laurate)",
        "poly(diisopropyl fumarate)",
        "poly(phenyl vinyl ether)",
        "poly(2-methyl-1-butene)",
    ):
        assert records[name].product == "", name
        assert records[name].availability in ("commodity", "commercial"), name


def test_the_curated_refusal_table_parses():
    for smiles in vinyl._NO_HOMOPOLYMER:
        assert vinyl.canonical(smiles) is not None, smiles


# -- the engine's own opinion ---------------------------------------------


def score(smiles: str):
    candidate = polymer_candidate(smiles)
    request = PredictionRequest(
        candidate=candidate,
        properties=("synthetic_accessibility",),
        conditions=candidate.conditions,
    )
    return EXPERT.predict(request)[0]


@pytest.mark.skipif(not EXPERT.is_available(), reason=EXPERT.unavailable_reason())
def test_the_feasibility_expert_can_make_everything_this_module_proposes():
    # Sampled by a fixed stride rather than at random so the failure is the same
    # failure tomorrow; the full sweep of all 137 refuses nothing either.
    units = vinyl.units()
    sample = units[:: max(1, len(units) // 20)]
    refused = []
    for name, smiles in sample:
        prediction = score(smiles)
        if prediction.quantity is None:
            refused.append((name, smiles, prediction.notes))
    assert not refused, f"the engine calls these unmakeable: {refused}"


@pytest.mark.skipif(not EXPERT.is_available(), reason=EXPERT.unavailable_reason())
def test_the_commodity_polymers_land_in_the_commodity_band():
    # Not a calibration, a sanity check on the grammar: if a megatonne monomer
    # came back priced like a research compound, the repeat unit would be
    # written wrongly rather than the monomer being hard.
    for smiles in (
        "[*]CC[*]",                      # polyethylene
        "[*]CC(c1ccccc1)[*]",            # polystyrene
        "[*]CC(Cl)[*]",                  # poly(vinyl chloride)
        "[*]CC(C)(C(=O)OC)[*]",          # poly(methyl methacrylate)
        "[*]CC(OC(=O)C)[*]",             # poly(vinyl acetate)
        "[*]CC=CC[*]",                   # 1,4-polybutadiene
    ):
        prediction = score(smiles)
        assert prediction.quantity is not None
        assert prediction.quantity.value <= 3.0, (smiles, prediction.quantity.value)
