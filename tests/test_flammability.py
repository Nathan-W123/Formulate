"""Flash point: the refusals are tested as hard as the numbers.

A flash point is a safety number, so the three things pinned hardest here are
the three that would hurt someone rather than merely rank a candidate badly.

The **refusals**, because in this repository a refusal is a feature and this
expert has more of them than any other in the thermal family.  Every screen
below exists because a measurement said so, and the measurement is in the test:
if a future edit lets carbon tetrachloride or an ethanol/water blend come back
with a number, these fail.

The **reference state**, because getting it wrong is silent.  The registry
defines ``enthalpy_vaporization`` at the normal boiling point and the
correlation wants it at 298.15 K.  Feed the registry value in unconverted and
every flash point comes out about eight kelvin low, looking entirely plausible.
One test asserts the conversion moves ethanol by more than four kelvin, so
dropping it cannot pass quietly.

The **unit**, because an uncertainty in kelvin that converts to degrees Celsius
by absolute rather than difference semantics turns a 12 K error bar into -261.
"""

from __future__ import annotations

import math

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import (
    Candidate,
    ComponentRole,
    MaterialClass,
    MixtureComponent,
    MixtureSpec,
    MoleculeSpec,
    molecule_candidate,
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import Prediction, PredictionStatus
from formulate.core.properties import PropertyFamily, get_property
from formulate.core.quantity import Quantity, Uncertainty, UncertaintyKind
from formulate.experts.base import PredictionRequest
from formulate.experts.flammability import (
    CARBOXYLIC_ACID_SMARTS,
    DEPENDENCY_UNITS,
    FLAMMABLE_LIQUID_TB_FLOOR_K,
    METHOD_SPREAD_K,
    ORGANOSULFUR_SPREAD_K,
    REFERENCE_TEMPERATURE_K,
    SUPPORTED_ELEMENTS,
    WATSON_MAX_REDUCED_TEMPERATURE,
    FlashPointExpert,
    c_nitro_count,
    carbon_count,
    carbon_hydrogen_count,
    catoire_naudin_flash_point,
    halogen_count,
    hydroxyl_count,
    ionic_reason,
    mixture_refusal_reason,
    non_combustible_reason,
)

FLASH = frozenset({"flash_point"})

ROOM = Conditions(
    temperature=Quantity(value=298.15, unit="K"), pressure=Quantity(value=1.0, unit="atm")
)

#: Measured inputs and the number this expert returns for them.
#:
#: (SMILES, name, Tb/K, Tc/K, dHvap(Tb) / (J/mol), predicted flash point / K).
#: The inputs are compiled values from ``chemicals``/``thermo``; they are
#: written out here rather than looked up so that the accuracy of the
#: correlation is tested even in an environment without those packages.
MEASURED = [
    ("CCO", "ethanol", 351.5704, 514.71, 38548.3, 285.32),
    ("Cc1ccccc1", "toluene", 383.7458, 591.75, 33182.1, 277.98),
    ("CC(C)=O", "acetone", 329.2249, 508.10, 29098.5, 250.11),
    ("CCCCCC", "hexane", 341.8656, 507.82, 28851.0, 248.14),
    ("CO", "methanol", 337.6324, 513.38, 35219.0, 282.09),
    ("CCOC(C)=O", "ethyl acetate", 350.2500, 523.30, 31940.7, 264.12),
    ("CCCCO", "1-butanol", 390.7500, 563.00, 43302.4, 306.61),
    ("CCOCC", "diethyl ether", 307.6044, 466.70, 26522.9, 227.81),
    ("Cc1ccccc1C", "o-xylene", 417.5210, 630.26, 36248.4, 301.32),
    ("CCCO", "1-propanol", 370.1900, 536.80, 41455.1, 295.22),
    ("CS(C)=O", "dimethyl sulfoxide", 465.0500, 707.00, 42905.3, 368.71),
    ("CCCCCCCCCCCC", "dodecane", 489.4416, 658.10, 44092.8, 352.03),
    ("c1ccccc1", "benzene", 353.2188, 562.02, 30721.2, 257.39),
    ("CC#N", "acetonitrile", 354.7500, 545.50, 29753.0, 274.69),
]

INPUTS = {row[0]: row[2:5] for row in MEASURED}

#: The ten flash points this module was briefed with, in kelvin. They are not
#: the compilation's values - the two sources disagree by 2.47 K RMSE and by
#: 6.85 K on 1-propanol - which is the point of the spread the expert declares.
BRIEFED = {
    "CCO": 286.0,
    "Cc1ccccc1": 277.0,
    "CC(C)=O": 253.0,
    "CCCCCC": 251.0,
    "CO": 284.0,
    "CCOC(C)=O": 269.0,
    "CCCCO": 308.0,
    "CCOCC": 228.0,
    "Cc1ccccc1C": 300.0,
    "CCCO": 295.0,
}


# -- scaffolding -----------------------------------------------------------


def _upstream(
    prop: str, value: float, unit: str, std: float | None, conditions: Conditions | None = None
) -> Prediction:
    """A usable upstream prediction, as the dispatcher would hand one over."""
    return Prediction(
        property=prop,
        quantity=Quantity(value=value, unit=unit),
        uncertainty=Uncertainty(std=std, kind=UncertaintyKind.EPISTEMIC, basis="test fixture"),
        status=PredictionStatus.OK,
        expert_id="test",
        conditions=conditions or Conditions(),
    )


def _context(
    smiles: str,
    *,
    tb_std: float = 0.5,
    tc_std: float = 2.0,
    hvap_std: float = 500.0,
    hvap_at: float | None = None,
    drop: str = "",
) -> dict[str, Prediction]:
    """Measured boiling point, critical temperature and enthalpy for ``smiles``.

    ``hvap_at`` is the temperature the enthalpy is stamped as being defined at;
    the default is the normal boiling point, which is what the registry says
    and what Joback stamps.
    """
    tb, tc, hvap = INPUTS[smiles]
    stamp = Conditions(temperature=Quantity(value=hvap_at or tb, unit="K"))
    context = {
        "normal_boiling_point": _upstream("normal_boiling_point", tb, "K", tb_std),
        "critical_temperature": _upstream("critical_temperature", tc, "K", tc_std),
        "enthalpy_vaporization": _upstream(
            "enthalpy_vaporization", hvap, "J/mol", hvap_std, stamp
        ),
    }
    context.pop(drop, None)
    return context


def _predict(candidate: Candidate, context=None, conditions: Conditions = ROOM) -> Prediction:
    """The single flash-point prediction for one candidate."""
    out = FlashPointExpert().predict(
        PredictionRequest(
            candidate=candidate,
            properties=FLASH,
            conditions=conditions,
            context=context or {},
        )
    )
    assert len(out) == 1
    return out[0]


def _for(smiles: str, **kwargs) -> Prediction:
    return _predict(molecule_candidate(smiles, conditions=ROOM), _context(smiles, **kwargs))


def _kelvin(prediction: Prediction) -> float:
    assert prediction.quantity is not None
    return prediction.quantity.to("K").value


def _blend(*parts: tuple[str, float]) -> Candidate:
    return Candidate(
        material_class=MaterialClass.MIXTURE,
        mixture=MixtureSpec(
            components=tuple(
                MixtureComponent(
                    molecule=MoleculeSpec(smiles=smiles),
                    fraction=fraction,
                    role=ComponentRole.SOLVENT,
                )
                for smiles, fraction in parts
            )
        ),
        conditions=ROOM,
    )


# ==========================================================================
# The correlation
# ==========================================================================


def test_the_correlation_is_the_published_one():
    """Catoire & Naudin (2004) on ethanol, by hand.

    1.477 * 351.57^0.79686 * 42.93^0.16845 * 2^-0.05948, with the enthalpy
    already moved to 298.15 K, is 285.3 K against a measured 285.15.
    """
    by_hand = 1.477 * 351.5704**0.79686 * 42.93**0.16845 * 2**-0.05948
    assert catoire_naudin_flash_point(351.5704, 42930.0, 2) == pytest.approx(by_hand)
    assert by_hand == pytest.approx(285.3, abs=0.2)


def test_the_correlation_takes_joules_and_returns_kelvin():
    """The registry's canonical unit in, kelvin out - not kJ/mol in."""
    in_joules = catoire_naudin_flash_point(351.57, 42930.0, 2)
    in_kilojoules = catoire_naudin_flash_point(351.57, 42.93, 2)
    assert in_joules == pytest.approx(285.0, abs=1.0)
    # A factor of a thousand under an exponent of 0.169 is a factor of 3.2, so
    # the mistake is large but not obviously absurd, which is why it is pinned.
    assert in_joules / in_kilojoules == pytest.approx(1000.0**0.16845, rel=1e-9)


@pytest.mark.parametrize(
    "tb,hvap,nc", [(0.0, 42930.0, 2), (351.57, 0.0, 2), (351.57, 42930.0, 0)]
)
def test_the_correlation_refuses_impossible_arguments(tb, hvap, nc):
    with pytest.raises(ValueError):
        catoire_naudin_flash_point(tb, hvap, nc)


# ==========================================================================
# The numbers, pinned
# ==========================================================================


@requires_rdkit
@pytest.mark.parametrize("smiles,name,tb,tc,hvap,expected", MEASURED)
def test_each_compound_returns_the_value_it_was_validated_at(
    smiles, name, tb, tc, hvap, expected
):
    """Pinned to a tenth of a kelvin, so any change to the path is visible."""
    prediction = _for(smiles)
    assert prediction.is_usable, prediction.notes
    assert _kelvin(prediction) == pytest.approx(expected, abs=0.1)


@requires_rdkit
def test_the_ten_briefed_solvents_agree_to_better_than_three_kelvin():
    """Measured, not asserted: RMSE 2.22 K over the ten solvents in the brief."""
    errors = []
    for smiles, reference in BRIEFED.items():
        prediction = _for(smiles)
        assert prediction.is_usable, prediction.notes
        error = _kelvin(prediction) - reference
        assert abs(error) < 6.0, f"{smiles}: {error:+.2f} K"
        errors.append(error)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    assert rmse == pytest.approx(2.22, abs=0.1), f"measured RMSE {rmse:.3f} K"


@requires_rdkit
def test_1_propanol_sits_inside_the_disagreement_between_the_two_references():
    """The brief says 295 K, the compilation 288.15, and the prediction is 295.2.

    This is the compound that justifies the 5.2 K correlation spread: an error
    bar narrower than the gap between two references is a claim the world does
    not support.
    """
    value = _kelvin(_for("CCCO"))
    assert 288.15 <= value <= 296.0
    assert abs(value - 295.0) < abs(value - 288.15)


# ==========================================================================
# Dimensions
# ==========================================================================


@requires_rdkit
def test_the_prediction_is_in_the_property_s_canonical_unit():
    prediction = _for("CCO")
    assert get_property("flash_point").canonical_unit == "kelvin"
    assert prediction.quantity is not None
    assert prediction.quantity.dimensionality == get_property("flash_point").dimensionality
    assert prediction.canonical is not None
    assert prediction.canonical.value == pytest.approx(285.32, abs=0.1)


@requires_rdkit
def test_the_error_bar_converts_with_difference_semantics():
    """A 5.2 K bar stays 5.2 in degrees Celsius rather than becoming -268."""
    prediction = _for("CCO")
    in_kelvin = prediction.uncertainty.std
    assert in_kelvin is not None
    in_celsius = prediction.uncertainty.converted("K", "degC").std
    assert in_celsius == pytest.approx(in_kelvin)
    assert prediction.quantity is not None
    assert prediction.quantity.to("degC").value == pytest.approx(_kelvin(prediction) - 273.15)


@requires_rdkit
def test_a_twelve_kelvin_bar_is_still_twelve_in_celsius():
    """The README's own example, on this expert's output."""
    prediction = _for("CCO", tb_std=17.9)
    std = prediction.uncertainty.std
    assert std is not None and std > 12.0
    assert prediction.uncertainty.converted("K", "degC").std == pytest.approx(std)


# ==========================================================================
# The reference state - the silent failure
# ==========================================================================


@requires_rdkit
def test_the_watson_correction_is_load_bearing():
    """Skip the reference-state conversion and ethanol drops five kelvin.

    Over the whole panel that is RMSE 9.75 K with a -7.36 K bias, against
    6.33 K corrected. Nothing in the output would reveal it, so it is pinned
    here: an enthalpy already stamped at 298.15 K is used as-is, and one
    stamped at the boiling point is converted.
    """
    at_boiling_point = _kelvin(_for("CCO"))
    already_at_reference = _kelvin(_for("CCO", hvap_at=REFERENCE_TEMPERATURE_K))
    assert at_boiling_point - already_at_reference > 4.0
    assert already_at_reference == pytest.approx(280.19, abs=0.1)
    assert at_boiling_point == pytest.approx(285.32, abs=0.1)


@requires_rdkit
def test_an_unstamped_enthalpy_is_read_at_the_boiling_point():
    """The registry's definition is the fallback, not 298.15 K.

    An upstream expert that states no temperature is taken at its word about
    the property's registered meaning rather than assumed to be helpful.
    """
    stamped = _for("CCO")
    context = _context("CCO")
    hvap = context["enthalpy_vaporization"]
    context["enthalpy_vaporization"] = _upstream(
        "enthalpy_vaporization", hvap.quantity.value, "J/mol", 500.0, None
    )
    unstamped = _predict(molecule_candidate("CCO", conditions=ROOM), context)
    assert _kelvin(unstamped) == pytest.approx(_kelvin(stamped), abs=1e-6)


@requires_rdkit
def test_the_reference_temperature_is_recorded_in_provenance():
    """So that a reader can tell which state the enthalpy came in at."""
    prediction = _for("CCO")
    assert prediction.provenance is not None
    recorded = prediction.provenance.parameters["enthalpy_reference_temperature_k"]
    assert float(recorded) == pytest.approx(351.5704, abs=1e-3)


# ==========================================================================
# Dependency discipline
# ==========================================================================


@requires_rdkit
@pytest.mark.parametrize("missing", sorted(DEPENDENCY_UNITS))
def test_a_missing_dependency_is_refused_by_name(missing):
    prediction = _for("CCO", drop=missing)
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    assert missing in prediction.notes[0]


@requires_rdkit
def test_nothing_at_all_comes_out_of_an_empty_context():
    prediction = _predict(molecule_candidate("CCO", conditions=ROOM), {})
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None


@requires_rdkit
def test_an_unusable_upstream_prediction_is_not_a_usable_one():
    """A refusal upstream must not be read through as a missing key would be."""
    context = _context("CCO")
    context["critical_temperature"] = Prediction.unsupported(
        "critical_temperature", "test", "no groups"
    )
    prediction = _predict(molecule_candidate("CCO", conditions=ROOM), context)
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "critical_temperature" in prediction.notes[0]


@requires_rdkit
def test_the_declared_dependencies_are_exactly_what_is_read():
    expert = FlashPointExpert()
    assert expert.dependencies == frozenset(DEPENDENCY_UNITS)
    assert expert.dependencies == {
        "normal_boiling_point",
        "enthalpy_vaporization",
        "critical_temperature",
    }


# ==========================================================================
# Refusals: things that do not burn, or do something worse than burning
# ==========================================================================


@requires_rdkit
@pytest.mark.parametrize(
    "smiles,name,expected_phrase",
    [
        ("O", "water", "no carbon"),
        ("N", "ammonia", "no carbon"),
        ("ClC(Cl)(Cl)Cl", "carbon tetrachloride", "carbon-bound hydrogen"),
        ("ClC(Cl)(Cl)C(Cl)(Cl)Cl", "hexachloroethane", "carbon-bound hydrogen"),
        (
            "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
            "perfluorohexane",
            "carbon-bound hydrogen",
        ),
        ("S=C=S", "carbon disulfide", "carbon-bound hydrogen"),
        ("ClC(Cl)Cl", "chloroform", "halogen"),
        ("ClCCl", "dichloromethane", "halogen"),
        ("CC(Cl)(Cl)Cl", "1,1,1-trichloroethane", "halogen"),
        ("ClC=C(Cl)Cl", "trichloroethylene", "halogen"),
        ("CC(O)=O", "acetic acid", "carboxylic acid"),
        ("OC=O", "formic acid", "carboxylic acid"),
        ("OCCO", "ethylene glycol", "hydroxyl"),
        ("OCC(O)CO", "glycerol", "hydroxyl"),
        ("O=C(OOC(=O)c1ccccc1)c1ccccc1", "benzoyl peroxide", "peroxide"),
        ("CC(C)(C)OO", "tert-butyl hydroperoxide", "peroxide"),
        ("CCC(C)(OO)OOC(C)(CC)OO", "MEKP", "peroxide"),
        (
            "[O-][N+](=O)OCC(O[N+]([O-])=O)CO[N+]([O-])=O",
            "nitroglycerin",
            "nitrate ester",
        ),
        ("CCCN=[N+]=[N-]", "propyl azide", "azide"),
        (
            "O=[N+]([O-])N1CN(CN(C1)[N+]([O-])=O)[N+]([O-])=O",
            "RDX",
            "nitramine",
        ),
        ("CCO[Si](OCC)(OCC)OCC", "tetraethoxysilane", "Si"),
        ("CCCCP(CCCC)CCCC", "tributylphosphine", "P"),
    ],
)
def test_the_screens_refuse_what_does_not_have_a_flash_point(smiles, name, expected_phrase):
    """One test per refusal, each asserting the reason a chemist would want."""
    reason = non_combustible_reason(smiles)
    assert reason is not None, f"{name} was not refused"
    assert expected_phrase in reason, f"{name}: {reason}"

    # And end to end, with every dependency present, so that the refusal is not
    # an artefact of a missing input.
    context = {
        "normal_boiling_point": _upstream("normal_boiling_point", 400.0, "K", 0.5),
        "critical_temperature": _upstream("critical_temperature", 600.0, "K", 2.0),
        "enthalpy_vaporization": _upstream(
            "enthalpy_vaporization",
            35000.0,
            "J/mol",
            500.0,
            Conditions(temperature=Quantity(value=400.0, unit="K")),
        ),
    }
    prediction = _predict(molecule_candidate(smiles, conditions=ROOM), context)
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    assert prediction.notes and prediction.notes[0]


@requires_rdkit
@pytest.mark.parametrize(
    "smiles,name",
    [
        ("CCO", "ethanol"),
        ("CCl", "methyl chloride"),
        ("Clc1ccccc1", "chlorobenzene"),
        ("ClCCCl", "1,2-dichloroethane"),
        ("C[N+](=O)[O-]", "nitromethane"),
        ("[O-][N+](=O)c1ccccc1", "nitrobenzene"),
        ("CCOC(C)=O", "ethyl acetate"),
        ("CCCCC(O)=O", "valeric acid is refused, but pentanol is not"),
        ("CCCCCO", "1-pentanol"),
    ],
)
def test_the_screens_do_not_refuse_ordinary_flammable_liquids(smiles, name):
    """The negative controls. A screen that refuses everything is not a screen.

    Simple C-nitro compounds are the load-bearing case: nitromethane and
    nitrobenzene predict to within 1.8 K here and are ordinary flammable
    liquids, so the self-oxidising screen must not reach them.
    """
    if "refused" in name:  # valeric acid: refused, and that is the acid screen
        assert non_combustible_reason(smiles) is not None
        return
    assert non_combustible_reason(smiles) is None, name


@requires_rdkit
def test_a_gas_is_refused_at_the_ghs_boundary():
    """Propane boils at 231 K; below 293.15 the closed-cup test is not done."""
    context = {
        "normal_boiling_point": _upstream("normal_boiling_point", 231.04, "K", 0.5),
        "critical_temperature": _upstream("critical_temperature", 369.83, "K", 2.0),
        "enthalpy_vaporization": _upstream(
            "enthalpy_vaporization",
            19040.0,
            "J/mol",
            500.0,
            Conditions(temperature=Quantity(value=231.04, unit="K")),
        ),
    }
    prediction = _predict(molecule_candidate("CCC", conditions=ROOM), context)
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "flammable gas" in prediction.notes[0]
    assert str(FLAMMABLE_LIQUID_TB_FLOOR_K) in prediction.notes[0]


@requires_rdkit
def test_an_ion_is_refused_before_anything_else_is_tried():
    """An ionic liquid's vapour pressure is what makes a flash point meaningless."""
    prediction = _predict(
        molecule_candidate("CCCC[N+](C)(C)C", conditions=ROOM),
        {
            "normal_boiling_point": _upstream("normal_boiling_point", 500.0, "K", 0.5),
            "critical_temperature": _upstream("critical_temperature", 700.0, "K", 2.0),
            "enthalpy_vaporization": _upstream(
                "enthalpy_vaporization",
                45000.0,
                "J/mol",
                500.0,
                Conditions(temperature=Quantity(value=500.0, unit="K")),
            ),
        },
    )
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "charge" in prediction.notes[0]


@requires_rdkit
@pytest.mark.parametrize(
    "smiles,name",
    [
        ("[NH3+]CC(=O)[O-]", "glycine"),
        ("C[N+](C)(C)CC(=O)[O-]", "betaine"),
        ("C[N+](C)(C)CCCS(=O)(=O)[O-]", "a sulfobetaine"),
    ],
)
def test_a_zwitterion_is_refused_although_its_net_charge_is_zero(smiles, name):
    """The case a net formal charge misses. An inner salt does not evaporate.

    Glycine totals zero and is no more volatile than sodium acetate; before
    this screen it was handed a flash point of 418 K.
    """
    reason = ionic_reason(smiles)
    assert reason is not None, name
    assert "zwitterion" in reason
    prediction = _predict(molecule_candidate(smiles, conditions=ROOM), _context("CCO"))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None


@requires_rdkit
@pytest.mark.parametrize(
    "smiles,name",
    [
        ("C[N+](=O)[O-]", "nitromethane, drawn charge-separated"),
        ("CN(=O)=O", "nitromethane, drawn pentavalent"),
        ("[O-][N+](=O)c1ccccc1", "nitrobenzene"),
        ("C[N+](C)(C)[O-]", "trimethylamine N-oxide"),
        ("[O-][n+]1ccccc1", "pyridine N-oxide"),
        ("CCCN=[N+]=[N-]", "propyl azide"),
        ("CS(C)=O", "dimethyl sulfoxide"),
        ("CCO", "ethanol"),
    ],
)
def test_the_zwitterion_screen_does_not_reach_a_charge_separated_group(smiles, name):
    """A plus and a minus on bonded atoms is one neutral group, not a salt.

    This is the screen's whole difficulty: nitromethane is written with both
    charges and is an ordinary flammable liquid, so the test has to be where
    the charges sit rather than whether they are there.
    """
    assert ionic_reason(smiles) is None, name


@requires_rdkit
@pytest.mark.parametrize(
    "smiles,name,groups",
    [
        ("Cc1ccc(cc1[N+]([O-])=O)[N+]([O-])=O", "2,4-dinitrotoluene", 2),
        ("Cc1c(cc(cc1[N+]([O-])=O)[N+]([O-])=O)[N+]([O-])=O", "TNT", 3),
        ("Oc1c(cc(cc1[N+]([O-])=O)[N+]([O-])=O)[N+]([O-])=O", "picric acid", 3),
        ("[O-][N+](=O)c1ccccc1[N+]([O-])=O", "1,2-dinitrobenzene", 2),
    ],
)
def test_a_polynitro_compound_is_refused_like_any_other_self_oxidiser(smiles, name, groups):
    """One nitro group is a solvent; two is a secondary explosive.

    The same argument the nitrate-ester and nitramine patterns make. Nothing
    in the 426-compound compilation carries two, so the screen costs no
    measured coverage and closes the gap TNT walked through.
    """
    assert c_nitro_count(smiles) == groups
    reason = non_combustible_reason(smiles)
    assert reason is not None, name
    assert "nitro groups" in reason
    prediction = _predict(molecule_candidate(smiles, conditions=ROOM), _context("CCO"))
    assert prediction.status is PredictionStatus.UNSUPPORTED


@requires_rdkit
@pytest.mark.parametrize("smiles", ["C[N+](=O)[O-]", "CN(=O)=O", "[O-][N+](=O)c1ccccc1"])
def test_one_nitro_group_is_still_an_ordinary_flammable_liquid(smiles):
    """The negative control that decides where the screen is drawn.

    Nitromethane and nitrobenzene predict to within 1.7 K, and the eight
    mononitro compounds in the compilation to 3.6 K mean absolute error.
    """
    assert c_nitro_count(smiles) == 1
    assert non_combustible_reason(smiles) is None


@requires_rdkit
def test_a_critical_temperature_beside_the_boiling_point_is_refused():
    """The Watson correction divides by 1 - Tb/Tc, and says nothing as it blows up.

    Ethanol with a critical temperature half a kelvin above its boiling point
    returned 381.7 K before this guard - thirty kelvin above where ethanol
    boils - with a 39 K bar and an OK status. Nothing in the compilation
    exceeds Tb/Tc = 0.816.
    """
    tb, _tc, hvap = INPUTS["CCO"]
    context = _context("CCO")
    context["critical_temperature"] = _upstream("critical_temperature", tb + 0.5, "K", 2.0)
    prediction = _predict(molecule_candidate("CCO", conditions=ROOM), context)
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    assert "Watson" in prediction.notes[0]
    assert "critical temperature" in prediction.notes[0]

    # And the threshold is where it says it is: 0.85 of the critical
    # temperature still answers, 0.90 does not.
    for ratio, usable in ((0.85, True), (WATSON_MAX_REDUCED_TEMPERATURE, False)):
        context = _context("CCO")
        context["critical_temperature"] = _upstream(
            "critical_temperature", tb / ratio, "K", 2.0
        )
        assert _predict(molecule_candidate("CCO", conditions=ROOM), context).is_usable is usable
    assert hvap > 0  # the enthalpy is untouched by any of this


@requires_rdkit
def test_a_flash_point_above_the_boiling_point_is_refused():
    """A liquid whose vapour is already at one atmosphere flashed long ago.

    The backstop for an input that is wrong in a way the ratio guard does not
    see: an enthalpy of 200 kJ/mol on ethanol's boiling point puts the
    correlation at 376 K, which is 25 K above where ethanol boils.
    """
    tb, tc, _hvap = INPUTS["CCO"]
    context = _context("CCO")
    context["enthalpy_vaporization"] = _upstream(
        "enthalpy_vaporization",
        200000.0,
        "J/mol",
        500.0,
        Conditions(temperature=Quantity(value=tb, unit="K")),
    )
    prediction = _predict(molecule_candidate("CCO", conditions=ROOM), context)
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    assert "boiling point" in prediction.notes[0]
    assert tc > tb  # the critical temperature here is perfectly ordinary


@requires_rdkit
def test_an_enthalpy_stamped_above_the_critical_temperature_is_refused_not_failed():
    """A wrong input is a refusal with a reason, not an exception with a traceback."""
    prediction = _for("CCO", hvap_at=600.0)
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "600.0 K the enthalpy of vaporisation is stated at" in prediction.notes[0]


@requires_rdkit
def test_an_unparseable_molecule_is_refused_rather_than_crashing():
    prediction = _predict(molecule_candidate("not a molecule", conditions=ROOM), _context("CCO"))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert "parsed" in prediction.notes[0]


# ==========================================================================
# Refusals: mixtures
# ==========================================================================


@requires_rdkit
def test_a_blend_is_refused_with_the_chemistry_as_the_reason():
    """Not an average. Le Chatelier over partial pressures, and no data to check it."""
    prediction = _predict(_blend(("CCO", 0.5), ("O", 0.5)))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None
    reason = prediction.notes[0]
    assert "lower flammable limit" in reason
    assert "average" in reason


@requires_rdkit
def test_the_blend_refusal_names_the_component_that_would_decide_it():
    reason = mixture_refusal_reason(_blend(("CCCCCCCC", 0.97), ("CCOCC", 0.03)).mixture)
    assert "CCOCC" in reason, reason


@requires_rdkit
def test_a_blend_of_two_things_that_both_have_flash_points_is_still_refused():
    """The case where an average looks most tempting, and is most wrong.

    Toluene flashes at 277 K and dodecane at 347; three per cent of toluene
    does not give a blend a flash point of 345.
    """
    prediction = _predict(_blend(("Cc1ccccc1", 0.03), ("CCCCCCCCCCCC", 0.97)))
    assert prediction.status is PredictionStatus.UNSUPPORTED


@requires_rdkit
def test_a_mixture_is_declared_rather_than_silently_uncovered():
    """Listing MIXTURE is deliberate: the refusal carries the chemistry."""
    expert = FlashPointExpert()
    assert expert.covers("flash_point", MaterialClass.MIXTURE)
    assert expert.covers("flash_point", MaterialClass.MOLECULE)
    assert not expert.covers("flash_point", MaterialClass.POLYMER)


@requires_rdkit
def test_a_polymer_gets_nothing_at_all():
    """No silent fallback between material classes: a polymer is not its monomer."""
    from formulate.core.candidate import polymer_candidate

    out = FlashPointExpert().predict(
        PredictionRequest(
            candidate=polymer_candidate("[*]CC(c1ccccc1)[*]", conditions=ROOM),
            properties=FLASH,
            conditions=ROOM,
            context=_context("CCO"),
        )
    )
    assert out == []


# ==========================================================================
# Uncertainty
# ==========================================================================


@requires_rdkit
def test_on_measured_inputs_the_correlation_error_dominates():
    prediction = _for("CCO")
    std = prediction.uncertainty.std
    assert std is not None
    assert prediction.uncertainty.kind is UncertaintyKind.COMBINED
    assert METHOD_SPREAD_K <= std < METHOD_SPREAD_K + 0.5
    assert "correlation error dominates" in prediction.uncertainty.basis


@requires_rdkit
def test_an_estimated_boiling_point_widens_the_bar_by_two_thirds_of_its_own():
    """Joback declares 12.9 K on a boiling point; about 0.66 of it arrives here."""
    tight = _for("CCO")
    loose = _for("CCO", tb_std=12.9, tc_std=29.0, hvap_std=4960.0)
    assert tight.uncertainty.std is not None and loose.uncertainty.std is not None
    assert loose.uncertainty.std > tight.uncertainty.std

    # The sensitivity itself, measured rather than assumed: shift the boiling
    # point by ten kelvin and see how much of it arrives.
    shifted = _context("CCO")
    tb = shifted["normal_boiling_point"].quantity.value
    shifted["normal_boiling_point"] = _upstream("normal_boiling_point", tb + 10.0, "K", 0.5)
    moved = _predict(molecule_candidate("CCO", conditions=ROOM), shifted)
    slope = (_kelvin(moved) - _kelvin(tight)) / 10.0
    assert slope == pytest.approx(0.66, abs=0.05)

    expected = math.hypot(METHOD_SPREAD_K, slope * 12.9)
    assert loose.uncertainty.std == pytest.approx(expected, rel=0.25)
    assert "propagated input error dominates" in loose.uncertainty.basis


@requires_rdkit
def test_an_estimated_boiling_point_earns_an_explicit_warning():
    loose = _for("CCO", tb_std=12.9)
    assert any("GHS category" in note for note in loose.notes)
    assert not any("GHS category" in note for note in _for("CCO").notes)


@requires_rdkit
def test_organosulfur_gets_a_wider_bar_and_is_marked_out_of_domain():
    """Four of four over-predict, by +11.9 K on average. That is a bias, not scatter."""
    dmso = _for("CS(C)=O")
    assert dmso.is_usable
    assert dmso.status is PredictionStatus.OUT_OF_DOMAIN
    assert dmso.uncertainty.std is not None
    assert dmso.uncertainty.std >= ORGANOSULFUR_SPREAD_K
    assert any("organosulfur" in w for w in dmso.applicability.warnings)
    # The measured error, in the direction the warning claims.
    assert _kelvin(dmso) - 359.86 == pytest.approx(8.85, abs=0.2)


@requires_rdkit
def test_a_bromide_carries_the_warning_the_one_measurement_supports():
    prediction = _predict(
        molecule_candidate("BrCC", conditions=ROOM),
        {
            "normal_boiling_point": _upstream("normal_boiling_point", 311.35, "K", 0.5),
            "critical_temperature": _upstream("critical_temperature", 504.0, "K", 2.0),
            "enthalpy_vaporization": _upstream(
                "enthalpy_vaporization",
                27040.0,
                "J/mol",
                500.0,
                Conditions(temperature=Quantity(value=311.35, unit="K")),
            ),
        },
    )
    assert prediction.is_usable
    assert any("bromoethane" in w for w in prediction.applicability.warnings)


@requires_rdkit
def test_every_prediction_says_it_is_a_one_atmosphere_closed_cup_number():
    prediction = _for("Cc1ccccc1")
    assert any("closed-cup" in note for note in prediction.notes)
    assert any("altitude" in note for note in prediction.notes)


# ==========================================================================
# The structural helpers, which the screens are built out of
# ==========================================================================


@requires_rdkit
@pytest.mark.parametrize(
    "smiles,carbons,ch_hydrogens,halogens,hydroxyls",
    [
        ("CCO", 2, 5, 0, 1),  # the hydroxyl hydrogen is not on carbon
        ("O", 0, 0, 0, 0),  # water is not a hydroxyl: [OX2H1] wants one H, not two
        ("ClC(Cl)Cl", 1, 1, 3, 0),
        ("ClCCl", 1, 2, 2, 0),
        ("OCCO", 2, 4, 0, 2),
        ("ClC(Cl)(Cl)Cl", 1, 0, 4, 0),
        ("BrCCBr", 2, 4, 2, 0),  # passes the screen, and is in fact non-flammable
    ],
)
def test_the_atom_counts_are_what_the_screens_think_they_are(
    smiles, carbons, ch_hydrogens, halogens, hydroxyls
):
    assert carbon_count(smiles) == carbons
    assert carbon_hydrogen_count(smiles) == ch_hydrogens
    assert halogen_count(smiles) == halogens
    assert hydroxyl_count(smiles) == hydroxyls


@requires_rdkit
def test_the_halogen_screen_lets_through_a_liquid_that_does_not_burn():
    """A known and recorded cost, pinned so it is not discovered by surprise.

    1,2-dibromoethane has four carbon-bound hydrogens against two bromines, so
    the atom-counting screen passes it, and it is in fact non-flammable:
    bromine inhibits far more strongly per atom than chlorine. The error is
    conservative for safety and expensive for a search targeting a high flash
    point. The bromine warning in the applicability domain is the only signal.
    """
    assert non_combustible_reason("BrCCBr") is None
    domain = FlashPointExpert().assess_domain(molecule_candidate("BrCCBr", conditions=ROOM))
    assert any("bromine" in w for w in domain.warnings)


@requires_rdkit
def test_the_element_screen_matches_the_declared_set():
    assert SUPPORTED_ELEMENTS == {"C", "H", "N", "O", "S", "F", "Cl", "Br", "I"}
    assert non_combustible_reason("CCO[Si](OCC)(OCC)OCC") is not None
    assert non_combustible_reason("CC#N") is None  # nitrogen is in the set


@requires_rdkit
def test_the_acid_pattern_does_not_catch_an_ester():
    from formulate import chem

    assert chem.has_substructure("CC(O)=O", CARBOXYLIC_ACID_SMARTS)
    assert not chem.has_substructure("CCOC(C)=O", CARBOXYLIC_ACID_SMARTS)


# ==========================================================================
# The expert contract
# ==========================================================================


def test_the_expert_declares_itself_correctly():
    expert = FlashPointExpert()
    assert expert.id == "flash_point"
    assert expert.family is PropertyFamily.THERMAL
    assert expert.supported_properties == {"flash_point"}
    assert get_property("flash_point").family is PropertyFamily.THERMAL
    assert "Catoire" in expert.method


@requires_rdkit
def test_the_domain_is_real_and_says_what_it_was_fitted_to():
    expert = FlashPointExpert()
    domain = expert.assess_domain(molecule_candidate("CCO", conditions=ROOM))
    assert domain.in_domain and domain.score == 1.0
    assert "91 compounds" in domain.basis

    outside = expert.assess_domain(molecule_candidate("CCCC[N+](C)(C)C", conditions=ROOM))
    assert not outside.in_domain

    blend = expert.assess_domain(_blend(("CCO", 0.5), ("O", 0.5)))
    assert not blend.in_domain


# ==========================================================================
# The held-out panel: the number this expert actually stands behind
# ==========================================================================


def _reference_panel():
    """(smiles, T_flash, Tb, Tc, Hvap(Tb)) for everything compiled here."""
    from chemicals import Tb as tb_lookup, Tc as tc_lookup
    from chemicals.safety import T_flash
    from thermo import EnthalpyVaporization

    from formulate.experts.hansen import resolve_cas

    smiles_list = """
    CCCCC CCCCCC CCCCCCC CCCCCCCC CCCCCCCCC CCCCCCCCCC CCCCCCCCCCCC CC(C)CC(C)(C)C
    C1CCCCC1 CC1CCCCC1 c1ccccc1 Cc1ccccc1 Cc1ccccc1C Cc1cccc(C)c1 Cc1ccc(C)cc1
    CCc1ccccc1 CC(C)c1ccccc1 Cc1ccc(C)c(C)c1 c1ccc2ccccc2c1 c1ccc(cc1)c1ccccc1
    Clc1ccccc1 [O-][N+](=O)c1ccccc1 c1ccc(cc1)C=O Oc1ccccc1 Cc1ccccn1 c1ccncc1
    COc1ccccc1 Nc1ccccc1 c1ccc(cc1)C#N C=Cc1ccccc1 CO CCO CCCO CC(C)O CCCCO CC(C)CO
    CCC(C)O CC(C)(C)O CCCCCO CCCCCCO CCCCCCCCO OCC=C OCCOC OCCOCC CC(C)=O CCC(C)=O
    CCCCC(C)=O CC(=O)CC(C)C O=C1CCCCC1 CCC=O CCCC=O COC=O CCOC=O COC(C)=O CCOC(C)=O
    CCCOC(C)=O CCCCOC(C)=O CC(C)COC(C)=O CCOC(=O)CC COC(=O)C(C)=C CCOC(=O)C=C CCOCC
    CCCCOCCCC C1CCOC1 C1COCCO1 CC(C)OC(C)C COCCOC CCCN CCCCN CCN(CC)CC CCNCC C1CCNCC1
    NCCO C1COCCN1 CC#N CCC#N CN(C)C=O C[N+](=O)[O-] CCC[N+](=O)[O-] CS(C)=O CSC CCS
    CSSC ClCCCl BrCC ClCC=C c1ccoc1 o1cccc1C=O OCc1ccco1 CC=O CC1CO1
    """.split()

    panel = []
    for smiles in smiles_list:
        cas = resolve_cas(smiles)
        if cas is None:
            continue
        try:
            flash = T_flash(cas)
        except Exception:
            flash = None
        boiling, critical = tb_lookup(cas), tc_lookup(cas)
        if flash is None or boiling is None or critical is None:
            continue
        try:
            model = EnthalpyVaporization(CASRN=cas, Tb=boiling, Tc=critical)
        except Exception:
            continue
        hvap = None
        for method in ("CRC_HVAP_TB", "DIPPR_PERRY_8E", "VDI_PPDS", "HEOS_FIT"):
            if method not in (model.all_methods or ()):
                continue
            try:
                hvap = model.calculate(boiling, method)
            except Exception:
                continue
            if hvap and hvap == hvap and hvap > 0:
                break
            hvap = None
        if hvap is None:
            continue
        panel.append((smiles, float(flash), float(boiling), float(critical), float(hvap)))
    return panel


@requires_rdkit
def test_the_held_out_panel_is_the_accuracy_that_is_claimed():
    """91 compounds against compiled closed-cup flash points: RMSE 6.33 K.

    This is the number in the module docstring, measured rather than quoted,
    and the test fails if a future edit degrades it. Skipped rather than failed
    where the compilation is not installed, because it is an optional extra.
    """
    pytest.importorskip("chemicals")
    pytest.importorskip("thermo")
    panel = _reference_panel()
    assert len(panel) >= 80, f"only {len(panel)} reference compounds resolved"

    errors, inside_one, inside_two, predicted = [], 0, 0, 0
    for smiles, flash, boiling, critical, hvap in panel:
        context = {
            "normal_boiling_point": _upstream("normal_boiling_point", boiling, "K", 0.5),
            "critical_temperature": _upstream("critical_temperature", critical, "K", 2.0),
            "enthalpy_vaporization": _upstream(
                "enthalpy_vaporization",
                hvap,
                "J/mol",
                500.0,
                Conditions(temperature=Quantity(value=boiling, unit="K")),
            ),
        }
        prediction = _predict(molecule_candidate(smiles, conditions=ROOM), context)
        if not prediction.is_usable:
            continue
        predicted += 1
        error = _kelvin(prediction) - flash
        errors.append(error)
        std = prediction.uncertainty.std
        assert std is not None
        inside_one += abs(error) <= std
        inside_two += abs(error) <= 2 * std

    n = len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    mae = sum(abs(e) for e in errors) / n
    worst = max(abs(e) for e in errors)
    assert n >= 80, f"only {n} compounds predicted"
    assert rmse < 8.0, f"RMSE {rmse:.2f} K"
    assert mae < 6.0, f"MAE {mae:.2f} K"
    assert worst < 30.0, f"worst {worst:.2f} K"

    # Calibration: the bar must be neither flattering nor useless. A correct
    # one-sigma estimate catches about 68 per cent; this measures 74.7, wide
    # rather than narrow, which is the posture the README asks for.
    coverage = inside_one / n
    assert 0.6 <= coverage <= 0.9, f"one-sigma coverage {coverage:.1%}"
    assert inside_two / n >= 0.88, f"two-sigma coverage {inside_two / n:.1%}"


# ==========================================================================
# The error bar's own held-out test
# ==========================================================================
#
# The panel above measures the correlation out of sample - it is published and
# fitted to somebody else's compounds - but it does not measure the error bar
# out of sample, because METHOD_SPREAD_K is 1.253 times that panel's own mean
# absolute error.  Asking those ninety-one compounds whether the bar covers
# them is asking the bar about the data it was set from.
#
# So the whole compilation is driven through instead: every compound in
# chemicals' DIPPR-Serat, IEC 60079-20-1 and NFPA 497 flash-point tables with
# a resolvable structure and a compiled boiling point, critical temperature and
# enthalpy of vaporisation.  Four hundred and twenty-six resolve, the screens
# refuse fifty-nine, and two hundred and seventy-nine of the rest are compounds
# the panel above has never seen.


def _compilation_panel():
    """(smiles, T_flash, Tb, Tc, Hvap(Tb)) for every compound that resolves."""
    from chemicals import Tb as tb_lookup, Tc as tc_lookup, safety
    from chemicals.identifiers import search_chemical
    from thermo import EnthalpyVaporization

    from formulate import chem

    registry = []
    for table in (safety.DIPPR_SERAT_data, safety.IEC_2010_data, safety.NFPA_2008_data):
        registry.extend(list(table.index))

    panel = []
    for cas in sorted(set(registry)):
        try:
            flash = safety.T_flash(cas)
        except Exception:
            continue
        if flash is None:
            continue
        try:
            smiles = search_chemical(cas).smiles
        except Exception:
            continue
        if not smiles or chem.mol_from_smiles(smiles) is None:
            continue
        boiling, critical = tb_lookup(cas), tc_lookup(cas)
        if boiling is None or critical is None:
            continue
        try:
            model = EnthalpyVaporization(CASRN=cas, Tb=boiling, Tc=critical)
        except Exception:
            continue
        hvap = None
        for method in ("CRC_HVAP_TB", "DIPPR_PERRY_8E", "VDI_PPDS", "HEOS_FIT"):
            if method not in (model.all_methods or ()):
                continue
            try:
                hvap = model.calculate(boiling, method)
            except Exception:
                continue
            if hvap and hvap == hvap and hvap > 0:
                break
            hvap = None
        if hvap is None:
            continue
        panel.append((smiles, float(flash), float(boiling), float(critical), float(hvap)))
    return panel


@requires_rdkit
def test_the_error_bar_is_calibrated_on_compounds_it_was_not_set_from():
    """279 compounds the 5.2 K spread has never seen: one-sigma coverage 71.0%.

    The number that matters in this repository, because the ranker scores at a
    bound one sigma into the unfavourable direction. Non-sulfur only - which
    is what the 5.2 K is for - it measures 67.7 per cent against the 68 a
    correct estimate implies.

    Two sigma is the one figure that does not hold up: 90.7 per cent against
    95. The tails are fatter than normal, which the module docstring says.
    """
    pytest.importorskip("chemicals")
    pytest.importorskip("thermo")
    from rdkit import Chem

    already_seen = {Chem.CanonSmiles(row[0]) for row in _reference_panel()}
    panel = _compilation_panel()
    assert len(panel) >= 380, f"only {len(panel)} compounds resolved"

    errors, stds, sulfur = [], [], []
    refused = 0
    for smiles, flash, boiling, critical, hvap in panel:
        if Chem.CanonSmiles(smiles) in already_seen:
            continue
        context = {
            "normal_boiling_point": _upstream("normal_boiling_point", boiling, "K", 0.5),
            "critical_temperature": _upstream("critical_temperature", critical, "K", 2.0),
            "enthalpy_vaporization": _upstream(
                "enthalpy_vaporization",
                hvap,
                "J/mol",
                500.0,
                Conditions(temperature=Quantity(value=boiling, unit="K")),
            ),
        }
        prediction = _predict(molecule_candidate(smiles, conditions=ROOM), context)
        if not prediction.is_usable:
            refused += 1
            continue
        assert prediction.uncertainty.std is not None
        errors.append(_kelvin(prediction) - flash)
        stds.append(prediction.uncertainty.std)
        sulfur.append("S" in {a.GetSymbol() for a in Chem.MolFromSmiles(smiles).GetAtoms()})

    n = len(errors)
    assert n >= 250, f"only {n} held-out compounds predicted"
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    mae = sum(abs(e) for e in errors) / n
    assert rmse < 9.0, f"held-out RMSE {rmse:.2f} K"
    assert mae < 6.0, f"held-out MAE {mae:.2f} K"

    one = sum(abs(e) <= s for e, s in zip(errors, stds)) / n
    two = sum(abs(e) <= 2 * s for e, s in zip(errors, stds)) / n
    assert 0.6 <= one <= 0.85, f"held-out one-sigma coverage {one:.1%}"
    assert two >= 0.85, f"held-out two-sigma coverage {two:.1%}"

    # The 5.2 K bar on its own, without the organosulfur compounds it is not
    # claimed for and whose 14.9 K would flatter the figure.
    plain = [(e, s) for e, s, is_s in zip(errors, stds, sulfur) if not is_s]
    plain_one = sum(abs(e) <= s for e, s in plain) / len(plain)
    assert 0.6 <= plain_one <= 0.8, f"non-sulfur one-sigma coverage {plain_one:.1%}"

    # The screens are load-bearing on a set this size rather than incidental.
    assert refused >= 20, f"only {refused} of the held-out compounds were refused"
