"""Melt properties of a polymer: whether it melts, how thick, how slack.

The interesting cases here are the refusals. An amorphous polymer has no
melting point, and saying so is an answer rather than a coverage gap; a melt
viscosity without a chain length is not a number at all; and WLF referenced to
Tg has a range that a hot-melt nozzle sits outside of.
"""

from __future__ import annotations

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MonomerUnit,
    PolymerSpec,
    Tacticity,
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import Prediction, PredictionStatus
from formulate.core.quantity import Quantity
from formulate.experts.base import PredictionRequest
from formulate.experts.melt import (
    MELTING_POINTS,
    SURFACE_TENSION_DGDT,
    PolymerMeltExpert,
    melt_viscosity,
)

PE, PS, PMMA = "[*]CC[*]", "[*]CC(c1ccccc1)[*]", "[*]CC(C)(C(=O)OC)[*]"
NYLON66 = "[*]NCCCCCCNC(=O)CCCCC(=O)[*]"


def _polymer(
    *repeat_units: str,
    mn_kg_mol: float | None = None,
    tacticity: Tacticity = Tacticity.UNSPECIFIED,
):
    monomers = tuple(
        MonomerUnit(smiles=s, mole_fraction=1.0 / len(repeat_units)) for s in repeat_units
    )
    kwargs = {"tacticity": tacticity}
    if mn_kg_mol is not None:
        kwargs["number_average_molar_mass"] = Quantity(value=mn_kg_mol, unit="kg/mol")
    return Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(monomers=monomers, **kwargs),
        conditions=Conditions.standard(),
    )


def _predict(candidate, prop, *, temperature_k=298.15, context=None):
    request = PredictionRequest(
        candidate=candidate,
        properties=frozenset({prop}),
        conditions=Conditions(temperature=Quantity(value=temperature_k, unit="K")),
        context=context or {},
    )
    return next(p for p in PolymerMeltExpert().predict(request) if p.property == prop)


def _context(
    tg_k: float, me_kg_mol: float, density_g_cm3: float | None = None
) -> dict[str, Prediction]:
    """Stand in for the upstream experts this one depends on."""
    extra = {}
    if density_g_cm3 is not None:
        extra["amorphous_density"] = Prediction(
            property="amorphous_density",
            quantity=Quantity(value=density_g_cm3, unit="g/cm^3"),
            expert_id="stub",
        )
    return {
        **extra,
        "glass_transition_temperature": Prediction(
            property="glass_transition_temperature",
            quantity=Quantity(value=tg_k, unit="K"),
            expert_id="stub",
        ),
        "entanglement_molar_mass": Prediction(
            property="entanglement_molar_mass",
            quantity=Quantity(value=me_kg_mol, unit="kg/mol"),
            expert_id="stub",
        ),
    }


# -- melting point ---------------------------------------------------------


@requires_rdkit
def test_a_semicrystalline_polymer_gets_its_measured_melting_point():
    prediction = _predict(_polymer(NYLON66), "melting_point")
    assert prediction.quantity is not None
    assert prediction.quantity.to_canonical().value == pytest.approx(538.0)


@requires_rdkit
def test_an_amorphous_polymer_is_refused_because_it_has_no_melting_point():
    """Not a coverage gap. Atactic polystyrene softens; it does not melt."""
    prediction = _predict(_polymer(PS, tacticity=Tacticity.ATACTIC), "melting_point")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    assert "no melting point" in " ".join(prediction.notes)


@requires_rdkit
def test_an_untabulated_polymer_is_refused_differently_from_an_amorphous_one():
    """The two refusals must not read alike: one is a gap, one is a fact."""
    gap = _predict(
        _polymer("[*]CCCCCCCC[*]", tacticity=Tacticity.ATACTIC), "melting_point"
    )
    fact = _predict(_polymer(PMMA, tacticity=Tacticity.ATACTIC), "melting_point")
    assert gap.quantity is None and fact.quantity is None
    assert "cannot be estimated" in " ".join(gap.notes)
    assert "softens through" in " ".join(fact.notes)


def test_the_regularity_test_agrees_with_the_tables_it_replaced():
    """Six polymers were listed amorphous and thirteen were given a melting
    point. The structural test reproduces nineteen of those twenty calls, and
    the exception - polyisobutylene, which is regular and still does not
    crystallise at rest - is listed as an exception with its reason."""
    from formulate.experts.melt import (
        CRYSTALLISES_ONLY_UNDER_STRAIN,
        DECOMPOSES_BEFORE_MELTING,
        backbone_stereocentres,
    )

    was_amorphous = [
        "[*]CC(c1ccccc1)[*]",          # atactic polystyrene
        "[*]CC(C)(C(=O)OC)[*]",        # atactic PMMA
        "[*]CC(Cl)[*]",                # PVC
        "[*]CC(OC(C)=O)[*]",           # poly(vinyl acetate)
    ]
    for repeat in was_amorphous:
        assert backbone_stereocentres(repeat), repeat

    # The two whose reason is not structural are named as such.
    assert "[*]CC(C)(C)[*]" in CRYSTALLISES_ONLY_UNDER_STRAIN
    assert "[*]CC(C#N)[*]" in DECOMPOSES_BEFORE_MELTING

    # And every polymer with a tabulated melting point is regular, bar the two
    # whose table entries already said "isotactic only" and "stereoregular".
    needs_stereoregularity = {"[*]CC(C)[*]", "[*]OC(C)C(=O)[*]"}
    for repeat in MELTING_POINTS:
        regular = not backbone_stereocentres(repeat)
        assert regular is (repeat not in needs_stereoregularity), repeat


# -- surface tension -------------------------------------------------------


@requires_rdkit
def test_surface_tension_is_corrected_from_room_temperature_to_the_melt():
    cold = _predict(_polymer(PE), "surface_tension", temperature_k=298.15)
    hot = _predict(_polymer(PE), "surface_tension", temperature_k=473.15)
    drop = cold.quantity.value - hot.quantity.value
    assert drop == pytest.approx(-SURFACE_TENSION_DGDT * 175.0, rel=1e-6)
    assert hot.quantity.value < cold.quantity.value


@requires_rdkit
def test_a_copolymer_is_refused_rather_than_averaged():
    prediction = _predict(_polymer(PS, PE), "surface_tension")
    assert prediction.status is PredictionStatus.UNSUPPORTED


@requires_rdkit
def test_an_untabulated_surface_energy_falls_back_to_the_parachor():
    """Nine polymers had a measured surface energy; every other repeat unit
    used to get nothing, which took filament stability down with it."""
    prediction = _predict(
        _polymer("[*]CC(CC)[*]"), "surface_tension",
        context=_context(200.0, 2.0, density_g_cm3=0.92),
    )
    assert prediction.quantity is not None
    # Poly(1-butene) measures near 33 mN/m; this is the right neighbourhood.
    assert 0.020 < prediction.quantity.to("N/m").value < 0.050
    assert "parachor" in " ".join(prediction.notes)


@requires_rdkit
def test_the_parachor_is_doubted_more_than_a_measurement():
    measured = _predict(
        _polymer(PE), "surface_tension", context=_context(195.0, 1.15, density_g_cm3=0.855)
    )
    predicted = _predict(
        _polymer("[*]CC(CC)[*]"), "surface_tension",
        context=_context(200.0, 2.0, density_g_cm3=0.92),
    )
    assert predicted.uncertainty.std > measured.uncertainty.std


@requires_rdkit
def test_the_parachor_needs_a_density_and_says_so_rather_than_guessing():
    prediction = _predict(_polymer("[*]CC(CC)[*]"), "surface_tension")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "amorphous density" in " ".join(prediction.notes)


@requires_rdkit
def test_the_parachor_reproduces_the_measured_polymer_surface_energies():
    """Within a third, over every polymer that has both a measurement and a
    tabulated density. Looser than an experiment, and it says so."""
    import json
    from importlib import resources

    from formulate.experts.adhesion import POLYMER_SURFACES, SUBSTRATES
    from formulate.experts.melt import PARACHOR_SPREAD, parachor_surface_tension

    text = (
        resources.files("formulate.data")
        .joinpath("reference_polymers.json")
        .read_text(encoding="utf-8")
    )
    density = {
        p["repeat_unit"]: p["amorphous_density_g_cm3"]
        for p in json.loads(text)["polymers"]
        if p.get("amorphous_density_g_cm3") is not None
    }
    checked = 0
    for repeat, key in POLYMER_SURFACES.items():
        if repeat not in density:
            continue
        predicted = parachor_surface_tension(repeat, density[repeat])
        assert predicted is not None, repeat
        ratio = predicted / SUBSTRATES[key].total
        assert 1.0 / PARACHOR_SPREAD <= ratio <= PARACHOR_SPREAD, repeat
        checked += 1
    assert checked >= 8


@requires_rdkit
def test_an_attachment_point_inside_a_branch_still_parses():
    """PEEK's second [*] sits inside a parenthesis. Deleting it left an empty
    "()" that RDKit refused, so every aromatic backbone lost its parachor."""
    from formulate.experts.melt import parachor_surface_tension

    peek = "[*]Oc1ccc(Oc2ccc(C(=O)c3ccc([*])cc3)cc2)cc1"
    gamma = parachor_surface_tension(peek, 1.263)
    assert gamma is not None
    assert 30.0 < gamma < 70.0


# -- melt viscosity --------------------------------------------------------


@requires_rdkit
def test_viscosity_without_a_chain_length_is_refused_not_guessed():
    """The same repeat unit spans six orders between an oligomer and a polymer."""
    prediction = _predict(
        _polymer(PS), "shear_viscosity", temperature_k=423.15,
        context=_context(373.0, 18.1),
    )
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "molar mass" in " ".join(prediction.notes)


@requires_rdkit
def test_an_unentangled_chain_gets_rouse_not_the_reptation_power():
    """A wax lives below the entanglement threshold, so a blend cannot be
    scored without this branch - but it is the other law, not an
    extrapolation of the 3.4 power into a regime it does not hold in."""
    prediction = _predict(
        _polymer(PS, mn_kg_mol=12.0), "shear_viscosity", temperature_k=423.15,
        context=_context(373.0, 18.1),
    )
    assert prediction.quantity is not None
    joined = " ".join(prediction.notes)
    assert "unentangled" in joined
    assert "carries no load" in joined


def test_the_two_chain_length_branches_join_at_the_threshold():
    """Rouse and reptation agree at the critical mass, which is where both hold."""
    from formulate.experts.melt import CRITICAL_OVER_ENTANGLEMENT, melt_viscosity

    me = 1.15
    critical = CRITICAL_OVER_ENTANGLEMENT * me
    below = melt_viscosity(critical * 0.999, me, 198.0, 250.0)
    above = melt_viscosity(critical * 1.001, me, 198.0, 250.0)
    assert below == pytest.approx(above, rel=5e-3)


def test_a_wax_is_orders_of_magnitude_thinner_than_the_polymer():
    """Which is the whole reason a blend can carry a longer backbone."""
    from formulate.experts.melt import melt_viscosity

    wax = melt_viscosity(0.8, 1.15, 198.0, 473.15, 27.0e3)
    polymer = melt_viscosity(20.0, 1.15, 198.0, 473.15, 27.0e3)
    assert polymer / wax > 1000.0


@requires_rdkit
def test_above_the_wlf_range_arrhenius_carries_it_and_says_so():
    """A hot-melt nozzle sits past where WLF is referenced, so a tabulated
    flow activation energy carries the curve the rest of the way."""
    prediction = _predict(
        _polymer(PE, mn_kg_mol=50.0), "shear_viscosity", temperature_k=473.15,
        context=_context(198.0, 1.15),
    )
    assert prediction.quantity is not None
    # Measured HDPE at this chain length and 200 C is a few thousand Pa.s.
    assert 500.0 < prediction.quantity.value < 10000.0
    assert "Arrhenius" in " ".join(prediction.notes)


@requires_rdkit
def test_without_a_tabulated_activation_energy_the_barrier_is_predicted():
    """A polymer outside the table used to be refused the Arrhenius branch.

    It is now carried on a barrier predicted from its own glass transition,
    which says so and which doubles its uncertainty to pay for the guess. The
    alternative was that any repeat unit not in a fourteen-entry table had no
    melt viscosity at processing temperature at all.
    """
    prediction = _predict(
        _polymer("[*]CC(CC)[*]", mn_kg_mol=50.0), "shear_viscosity",
        temperature_k=473.15, context=_context(200.0, 2.0),
    )
    assert prediction.status is PredictionStatus.OK
    assert prediction.quantity is not None
    notes = " ".join(prediction.notes)
    assert "predicted from its glass transition" in notes


def test_the_predicted_barrier_is_doubted_more_than_a_tabulated_one():
    """Same polymer, same temperature; only the provenance of Ea differs."""
    from formulate.experts.melt import (
        ACTIVATION_ENERGY_RTOL,
        PREDICTED_ACTIVATION_RTOL,
    )

    assert PREDICTED_ACTIVATION_RTOL > ACTIVATION_ENERGY_RTOL

    tabulated = _predict(
        _polymer("[*]CC[*]", mn_kg_mol=50.0), "shear_viscosity",
        temperature_k=473.15, context=_context(195.0, 1.15),
    )
    predicted = _predict(
        _polymer("[*]CC(CC)[*]", mn_kg_mol=50.0), "shear_viscosity",
        temperature_k=473.15, context=_context(195.0, 1.15),
    )
    assert tabulated.status is PredictionStatus.OK
    assert predicted.status is PredictionStatus.OK
    relative = lambda p: p.uncertainty.std / p.quantity.to_canonical().value
    assert relative(predicted) > relative(tabulated)


def test_the_predicted_barrier_reproduces_the_tabulated_ones():
    """Held out one at a time the correlation misses by at most 1.66x."""
    from formulate.experts.melt import (
        FLOW_ACTIVATION_ENERGY,
        PREDICTED_ACTIVATION_RTOL,
        predicted_activation_energy,
    )
    from formulate.experts.polymer import reference_polymers

    tg = {p["repeat_unit"]: p["glass_transition_k"] for p in reference_polymers()}
    worst = max(
        max(predicted_activation_energy(tg[smi]) / ea, ea / predicted_activation_energy(tg[smi]))
        for smi, ea in FLOW_ACTIVATION_ENERGY.items()
        if smi in tg
    )
    # In sample, so this is the optimistic figure; the stated one-sigma is the
    # held-out worst case, which is wider still.
    assert worst <= 1.0 + PREDICTED_ACTIVATION_RTOL


def test_the_predicted_barrier_has_a_floor():
    """The straight line goes negative below about 117 K, which is not physics."""
    from formulate.experts.melt import ACTIVATION_FLOOR, predicted_activation_energy

    assert predicted_activation_energy(80.0) == ACTIVATION_FLOOR
    assert predicted_activation_energy(400.0) > ACTIVATION_FLOOR


@requires_rdkit
def test_the_two_viscosity_branches_join_without_a_step():
    """WLF hands over to Arrhenius at Tg + WLF_RANGE_K.

    The value is continuous by construction - Arrhenius starts from whatever
    WLF says at the crossover - which is what is asserted here. The *slope* is
    not continuous, and is not claimed to be: the two forms have different
    temperature dependences and joining them at a point is the whole idea.
    """
    from formulate.experts.melt import WLF_RANGE_K, melt_viscosity

    tg, crossover = 198.0, 198.0 + WLF_RANGE_K
    at = melt_viscosity(50.0, 1.15, tg, crossover, 27.0e3)
    just_above = melt_viscosity(50.0, 1.15, tg, crossover + 1e-9, 27.0e3)
    assert just_above == pytest.approx(at, rel=1e-9)


@requires_rdkit
def test_atactic_polypropylene_is_not_given_the_isotactic_melting_point():
    """Keyed on the repeat unit the two are identical, and one does not melt."""
    from formulate.core.candidate import Tacticity

    atactic = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(
            monomers=(MonomerUnit(smiles="[*]CC(C)[*]"),), tacticity=Tacticity.ATACTIC
        ),
        conditions=Conditions.standard(),
    )
    isotactic = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(
            monomers=(MonomerUnit(smiles="[*]CC(C)[*]"),), tacticity=Tacticity.ISOTACTIC
        ),
        conditions=Conditions.standard(),
    )
    assert _predict(atactic, "melting_point").quantity is None
    assert _predict(isotactic, "melting_point").quantity is not None


@requires_rdkit
def test_unstated_tacticity_is_refused_rather_than_assumed():
    prediction = _predict(_polymer("[*]CC(C)[*]"), "melting_point")
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "does not state" in " ".join(prediction.notes)


@requires_rdkit
def test_below_the_glass_transition_it_is_a_solid_not_a_melt():
    prediction = _predict(
        _polymer(PS, mn_kg_mol=100.0), "shear_viscosity", temperature_k=300.0,
        context=_context(373.0, 18.1),
    )
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "solid, not a melt" in " ".join(prediction.notes)


@requires_rdkit
def test_viscosity_in_range_carries_an_order_of_magnitude_of_uncertainty():
    """Universal WLF constants are worth about a decade, and must say so."""
    prediction = _predict(
        _polymer(PS, mn_kg_mol=100.0), "shear_viscosity", temperature_k=443.15,
        context=_context(373.0, 18.1),
    )
    assert prediction.quantity is not None
    ratio = prediction.uncertainty.std / prediction.quantity.value
    assert 4.0 < ratio < 5.0     # (10^1 - 1)/2


def test_the_viscosity_form_is_anchored_at_the_definition_of_tg():
    """At Tg and the critical mass it must return the value that defines Tg."""
    assert melt_viscosity(2.0, 1.0, 373.0, 373.0) == pytest.approx(1e12)


def test_viscosity_falls_with_temperature_and_rises_with_chain_length():
    hot = melt_viscosity(50.0, 10.0, 373.0, 443.0)
    cold = melt_viscosity(50.0, 10.0, 373.0, 403.0)
    longer = melt_viscosity(100.0, 10.0, 373.0, 443.0)
    assert hot < cold
    assert longer > hot
    assert longer / hot == pytest.approx(2.0**3.4, rel=1e-6)


@requires_rdkit
def test_polycaprolactone_melts_low_enough_to_handle():
    """The safety-relevant entry: 60 C against polyethylene's 135 and the
    200 C a hot-melt nozzle runs at. Molten polymer sticks to skin, so the
    melting point of the material is a burn risk, not just a process setting."""
    prediction = _predict(_polymer("[*]CCCCCC(=O)O[*]"), "melting_point")
    assert prediction.quantity.to_canonical().value == pytest.approx(333.0)


@requires_rdkit
def test_a_measured_modulus_does_not_need_a_chain_dimension():
    """Gating it there refused polycaprolactone a stiffness sitting in a table,
    and with it the whole low-melting branch of the search. Polycaprolactone
    has a chain dimension now, so nylon-6,6 stands in: same situation, a
    measured modulus and no tabulated chain."""
    from formulate.core.candidate import Candidate, MaterialClass
    from formulate.experts import polymer_registry
    from formulate.experts.base import PredictionRequest as Req
    from formulate.experts.mechanical import CHAIN_DIMENSIONS

    assert NYLON66 not in CHAIN_DIMENSIONS, "the point of the test"

    candidate = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(monomers=(MonomerUnit(smiles=NYLON66),)),
        conditions=Conditions.standard(),
    )
    registry = polymer_registry()
    wanted = frozenset({"youngs_modulus", "glass_transition_temperature", "amorphous_density"})
    context = {}
    for expert in registry.resolution_order(
        registry.experts_for(wanted, MaterialClass.POLYMER)
    ):
        for prediction in expert.predict(
            Req(candidate=candidate, properties=wanted,
                conditions=candidate.conditions, context=dict(context))
        ):
            if prediction.is_usable:
                context.setdefault(prediction.property, prediction)
    assert context["youngs_modulus"].quantity.to_canonical().value == pytest.approx(2.8e9)
