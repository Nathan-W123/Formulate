"""Liquid transport: how thick a solvent is, and how fast a molecule moves.

The refusals are tested as hard as the predictions.  Orrick-Erbar's failure
mode is that a group the table does not contain contributes zero and the
arithmetic still completes, so every gate here stands between the engine and a
confident wrong number, and each test records the measured error it prevents.

The reference table below is frozen into this file rather than read from
``thermo`` at test time, so that a future change to that package moves no
number here.  Densities, molar masses and boiling points are the DIPPR,
REFPROP, VDI and Viswanath-Natarajan correlations at 298.15 K and 1 atm;
viscosities are from the same source and are what the predictions are scored
against.
"""

from __future__ import annotations

import math

import pytest

from conftest import requires_rdkit

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MonomerUnit,
    PolymerSpec,
    molecule_candidate,
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import Prediction, PredictionStatus
from formulate.core.properties import get_property
from formulate.core.quantity import Quantity
from formulate.experts import default_registry
from formulate.experts.base import PredictionRequest
from formulate.experts.registry import ExpertRegistry
from formulate.experts.transport import (
    LiquidTransportExpert,
    group_counts,
    hydrodynamic_radius,
    orrick_erbar_viscosity,
    stokes_einstein_diffusion,
    structural_refusal,
    uncertainty_bucket,
)

BOTH = ("shear_viscosity", "self_diffusion_coefficient")


# -- harness ----------------------------------------------------------------


def _context(
    density: float | None,
    molar_mass: float | None,
    boiling: float | None,
    molar_volume: float | None = None,
    *,
    density_std: float | None = None,
) -> dict[str, Prediction]:
    """Stand in for the upstream experts this one depends on."""
    out: dict[str, Prediction] = {}
    if density is not None:
        out["liquid_density"] = _stub("liquid_density", density, "kg/m^3", density_std)
    if molar_mass is not None:
        out["molar_mass"] = _stub("molar_mass", molar_mass, "g/mol", None)
    if boiling is not None:
        out["normal_boiling_point"] = _stub("normal_boiling_point", boiling, "K", None)
    if molar_volume is not None:
        out["molar_volume_liquid"] = _stub(
            "molar_volume_liquid", molar_volume, "m^3/mol", None
        )
    return out


def _stub(prop: str, value: float, unit: str, std: float | None) -> Prediction:
    from formulate.core.quantity import Uncertainty

    return Prediction(
        property=prop,
        quantity=Quantity(value=value, unit=unit),
        expert_id="stub",
        uncertainty=(
            Uncertainty(std=std) if std is not None else Uncertainty.unknown("stub")
        ),
    )


def _predict(
    smiles: str,
    *,
    density: float | None = None,
    molar_mass: float | None = None,
    boiling: float | None = None,
    molar_volume: float | None = "auto",  # type: ignore[assignment]
    temperature_k: float | None = 298.15,
    density_std: float | None = None,
    properties: tuple[str, ...] = BOTH,
) -> dict[str, Prediction]:
    if molar_volume == "auto":
        molar_volume = (
            None
            if (density is None or molar_mass is None)
            else molar_mass / density * 1e-3
        )
    conditions = (
        Conditions()
        if temperature_k is None
        else Conditions(temperature=Quantity(value=temperature_k, unit="K"))
    )
    request = PredictionRequest(
        candidate=molecule_candidate(smiles),
        properties=frozenset(properties),
        conditions=conditions,
        context=_context(
            density, molar_mass, boiling, molar_volume, density_std=density_std
        ),
    )
    return {p.property: p for p in LiquidTransportExpert().predict(request)}


def _viscosity_mpa_s(row: tuple) -> tuple[float, float]:
    """Predicted viscosity and its one-sigma bar, both in mPa*s."""
    _, smiles, density, molar_mass, boiling, _ = row
    pred = _predict(
        smiles,
        density=density,
        molar_mass=molar_mass,
        boiling=boiling,
        properties=("shear_viscosity",),
    )["shear_viscosity"]
    assert pred.is_usable, f"{row[0]}: {pred.notes}"
    value = pred.quantity.to("mPa*s").value
    std = pred.uncertainty.converted(pred.quantity.unit, "mPa*s").std
    return value, std


# -- reference data ---------------------------------------------------------

#: name, SMILES, liquid density at 25 C (kg/m^3), molar mass (g/mol), normal
#: boiling point (K), measured shear viscosity at 25 C (mPa*s).
REFERENCE = [
    # -- acyclic, 0-1 oxygen-bearing group --------------------------------
    ("1,1,1-trichloroethane", "CC(Cl)(Cl)Cl", 1330.27, 133.404, 347.17, 0.7654),
    ("1,2-dibromoethane", "BrCCBr", 2169.25, 187.861, 404.45, 1.6012),
    ("1,2-dichloroethane", "ClCCCl", 1245.57, 98.959, 356.65, 0.7707),
    ("1-bromopropane", "CCCBr", 1282.59, 122.992, 343.95, 0.3652),
    ("1-butanol", "CCCCO", 804.08, 74.122, 390.75, 2.5478),
    ("1-chlorobutane", "CCCCCl", 881.41, 92.567, 351.55, 0.4255),
    ("1-chloropropane", "CCCCl", 884.77, 78.541, 319.35, 0.3386),
    ("1-decanol", "CCCCCCCCCCO", 820.85, 158.281, 502.15, 11.3386),
    ("1-decene", "C=CCCCCCCCC", 738.31, 140.266, 444.15, 0.765),
    ("1-heptanol", "CCCCCCCO", 821.04, 116.201, 451.15, 5.8153),
    ("1-hexanol", "CCCCCCO", 816.01, 102.175, 430.05, 4.5279),
    ("1-hexene", "C=CCCCC", 669.3, 84.159, 336.55, 0.2677),
    ("1-nonanol", "CCCCCCCCCO", 824.48, 144.255, 486.85, 9.0343),
    ("1-octanol", "CCCCCCCCO", 826.01, 130.228, 467.85, 7.2892),
    ("1-octene", "C=CCCCCCC", 713.1, 112.213, 394.45, 0.48),
    ("1-pentanol", "CCCCCO", 812.19, 88.148, 410.75, 3.4474),
    ("1-propanol", "CCCO", 799.53, 60.095, 370.19, 1.9514),
    ("1-undecanol", "CCCCCCCCCCCO", 830.62, 172.308, 519.15, 14.0084),
    ("2,2,4-trimethylpentane", "CC(C)CC(C)(C)C", 688.03, 114.229, 372.36, 0.4765),
    ("2,3-dimethylbutane", "CC(C)C(C)C", 657.13, 86.175, 331.18, 0.3241),
    ("2-butanol", "CCC(C)O", 802.35, 74.122, 372.55, 3.0739),
    ("2-butanone", "CCC(C)=O", 799.54, 72.106, 352.75, 0.3951),
    ("2-ethylhexanol", "CCCCC(CC)CO", 787.29, 130.228, 459.35, 6.2784),
    ("2-heptanone", "CCCCCC(C)=O", 811.79, 114.185, 424.15, 0.7517),
    ("2-hexanone", "CCCCC(C)=O", 807.0, 100.159, 400.75, 0.5842),
    ("2-methyl-1-propanol", "CC(C)CO", 798.38, 74.122, 380.99, 3.3413),
    ("2-methyl-2-butanol", "CCC(C)(C)O", 788.16, 88.148, 375.55, 3.7221),
    ("2-methylpentane", "CCCC(C)C", 648.71, 86.175, 333.36, 0.2765),
    ("2-pentanol", "CCCC(C)O", 806.56, 88.148, 392.25, 3.3558),
    ("2-propanol", "CC(C)O", 781.87, 60.095, 355.36, 2.05),
    ("3-methyl-1-butanol", "CC(C)CCO", 807.13, 88.148, 403.95, 3.6288),
    ("3-methylpentane", "CCC(C)CC", 660.04, 86.175, 336.38, 0.2918),
    ("3-pentanol", "CCC(O)CC", 803.71, 88.148, 396.15, 4.1525),
    ("3-pentanone", "CCC(=O)CC", 809.53, 86.132, 375.05, 0.4441),
    ("4-methyl-2-pentanone", "CC(C)CC(C)=O", 796.34, 100.159, 388.85, 0.5444),
    ("acetic acid", "CC(=O)O", 1042.06, 60.052, 391.05, 1.1162),
    ("acetone", "CC(C)=O", 784.77, 58.079, 329.22, 0.3159),
    ("bromoethane", "CCBr", 1448.06, 108.965, 311.35, 0.3758),
    ("butyl acetate", "CCCCOC(C)=O", 876.02, 116.158, 399.15, 0.6777),
    ("butyraldehyde", "CCCC=O", 797.42, 72.106, 347.95, 0.4196),
    ("butyric acid", "CCCC(=O)O", 951.65, 88.105, 436.85, 1.4783),
    ("carbon tetrachloride", "ClC(Cl)(Cl)Cl", 1583.72, 153.823, 349.85, 0.9072),
    ("chloroform", "ClC(Cl)Cl", 1483.0, 119.378, 334.35, 0.5391),
    ("decane", "CCCCCCCCCC", 726.62, 142.282, 447.27, 0.8492),
    ("dibutyl ether", "CCCCOCCCC", 764.3, 130.228, 414.75, 0.6829),
    ("dichloromethane", "ClCCl", 1318.32, 84.933, 312.95, 0.4135),
    ("diethyl ether", "CCOCC", 707.88, 74.122, 307.6, 0.2245),
    ("diisopropyl ether", "CC(C)OC(C)C", 720.8, 102.175, 341.55, 0.3215),
    ("dipropyl ether", "CCCOCCC", 742.81, 102.175, 363.25, 0.3974),
    ("dodecane", "CCCCCCCCCCCC", 745.82, 170.335, 489.44, 1.3612),
    ("ethanol", "CCO", 785.16, 46.068, 351.57, 1.0829),
    ("ethyl acetate", "CCOC(C)=O", 893.72, 88.105, 350.25, 0.4308),
    ("ethyl formate", "CCOC=O", 915.69, 74.079, 327.24, 0.3803),
    ("formic acid", "OC=O", 1213.72, 46.025, 374.15, 1.6116),
    ("heptane", "CCCCCCC", 679.72, 100.202, 371.55, 0.3909),
    ("hexadecane", "CCCCCCCCCCCCCCCC", 770.3, 226.441, 559.9, 3.0815),
    ("hexanal", "CCCCCC=O", 809.73, 100.159, 402.75, 0.6633),
    ("hexane", "CCCCCC", 654.96, 86.175, 341.87, 0.2983),
    ("hexanoic acid", "CCCCCC(=O)O", 919.16, 116.158, 478.05, 2.843),
    ("iodomethane", "CI", 2062.73, 141.939, 315.55, 0.3584),
    ("isobutyl acetate", "CC(C)COC(C)=O", 849.51, 116.158, 390.05, 0.6762),
    ("isobutyric acid", "CC(C)C(=O)O", 944.04, 88.105, 427.55, 1.2248),
    ("methanol", "CO", 786.34, 32.042, 337.63, 0.5439),
    ("methyl acetate", "COC(C)=O", 928.12, 74.079, 329.85, 0.3638),
    ("methyl formate", "COC=O", 966.98, 60.052, 304.75, 0.333),
    ("methyl methacrylate", "C=C(C)C(=O)OC", 938.07, 100.116, 373.75, 0.5445),
    ("nonane", "CCCCCCCCC", 714.19, 128.255, 423.91, 0.6554),
    ("octane", "CCCCCCCC", 698.69, 114.229, 398.79, 0.5123),
    ("octanoic acid", "CCCCCCCC(=O)O", 903.96, 144.211, 513.15, 5.177),
    ("pentadecane", "CCCCCCCCCCCCCCC", 763.97, 212.415, 543.75, 2.5462),
    ("pentane", "CCCCC", 621.26, 72.149, 309.21, 0.2202),
    ("pentanoic acid", "CCCCC(=O)O", 933.1, 102.132, 459.25, 2.0139),
    ("propionic acid", "CCC(=O)O", 988.26, 74.079, 414.65, 1.0273),
    ("propyl acetate", "CCCOC(C)=O", 882.36, 102.132, 374.15, 0.5549),
    ("tetrachloroethylene", "ClC(Cl)=C(Cl)Cl", 1613.52, 165.833, 394.35, 0.8467),
    ("tetradecane", "CCCCCCCCCCCCCC", 758.96, 198.388, 526.65, 2.0823),
    ("trichloroethylene", "ClC=C(Cl)Cl", 1455.77, 131.388, 359.95, 0.5455),
    ("tridecane", "CCCCCCCCCCCCC", 750.64, 184.361, 508.55, 1.693),
    ("undecane", "CCCCCCCCCCC", 736.57, 156.308, 468.93, 1.0798),
    ("vinyl acetate", "C=COC(C)=O", 926.3, 86.089, 345.75, 0.4055),
    # -- aromatic six-ring --------------------------------------------------
    ("1,2,4-trimethylbenzene", "Cc1ccc(C)c(C)c1", 872.28, 120.192, 442.55, 0.8875),
    ("1,2-dichlorobenzene", "Clc1ccccc1Cl", 1301.38, 147.002, 453.35, 1.3046),
    ("2-chlorotoluene", "Cc1ccccc1Cl", 1040.09, 126.583, 431.95, 0.9654),
    ("acetophenone", "CC(=O)c1ccccc1", 1023.3, 120.149, 475.25, 1.6471),
    ("anisole", "COc1ccccc1", 990.66, 108.138, 426.75, 1.0242),
    ("benzaldehyde", "O=Cc1ccccc1", 1041.26, 106.122, 451.85, 1.3762),
    ("benzene", "c1ccccc1", 873.76, 78.112, 353.22, 0.6034),
    ("benzyl alcohol", "OCc1ccccc1", 1041.32, 108.138, 478.45, 5.2151),
    ("bromobenzene", "Brc1ccccc1", 1487.43, 157.008, 429.05, 1.0692),
    ("chlorobenzene", "Clc1ccccc1", 1101.47, 112.557, 405.21, 0.7496),
    ("cumene", "CC(C)c1ccccc1", 859.2, 120.192, 425.55, 0.7385),
    ("ethylbenzene", "CCc1ccccc1", 862.64, 106.165, 409.31, 0.6314),
    ("iodobenzene", "Ic1ccccc1", 1823.06, 204.008, 461.65, 1.5619),
    ("m-cresol", "Cc1cccc(O)c1", 1029.98, 108.138, 475.35, 13.1421),
    ("m-xylene", "Cc1cccc(C)c1", 860.03, 106.165, 412.21, 0.583),
    ("methyl benzoate", "COC(=O)c1ccccc1", 1083.89, 136.148, 472.15, 1.855),
    ("n-butylbenzene", "CCCCc1ccccc1", 857.12, 134.218, 456.45, 0.9661),
    ("o-xylene", "Cc1ccccc1C", 876.12, 106.165, 417.52, 0.7577),
    ("p-xylene", "Cc1ccc(C)cc1", 856.84, 106.165, 411.47, 0.6077),
    ("phenetole", "CCOc1ccccc1", 912.48, 122.164, 442.95, 1.1989),
    ("styrene", "C=Cc1ccccc1", 900.52, 104.149, 418.45, 0.7018),
    ("toluene", "Cc1ccccc1", 862.34, 92.138, 383.75, 0.5533),
    # -- acyclic, two or more oxygen-bearing groups -------------------------
    ("1,2-butanediol", "CCC(O)CO", 999.24, 90.121, 469.1, 49.0943),
    ("1,2-propanediol", "CC(O)CO", 1032.61, 76.094, 460.45, 42.5055),
    ("1,3-propanediol", "OCCCO", 1049.87, 76.094, 487.85, 42.7117),
    ("diethylene glycol", "OCCOCCO", 1114.24, 106.12, 518.65, 28.8063),
    ("ethyl lactate", "CCOC(=O)C(C)O", 1030.99, 118.131, 427.15, 2.3808),
    ("ethylene glycol", "OCCO", 1109.88, 62.068, 470.31, 16.8359),
    ("glycerol", "OCC(O)CO", 1258.1, 92.094, 562.15, 1013.4325),
    ("triethylene glycol", "OCCOCCOCCO", 1120.79, 150.173, 561.75, 37.2866),
]

BY_NAME = {row[0]: row for row in REFERENCE}

#: Measured NMR self-diffusion coefficients at 25 C, m^2/s.
SELF_DIFFUSION = {
    "pentane": 5.45e-9,
    "hexane": 4.21e-9,
    "heptane": 3.12e-9,
    "octane": 2.04e-9,
    "decane": 1.31e-9,
    "benzene": 2.21e-9,
    "toluene": 2.27e-9,
    "methanol": 2.42e-9,
    "ethanol": 1.05e-9,
    "1-propanol": 0.65e-9,
    "2-propanol": 0.65e-9,
    "1-butanol": 0.46e-9,
    "1-octanol": 0.156e-9,
    "acetone": 4.77e-9,
    "carbon tetrachloride": 1.41e-9,
    "chloroform": 2.59e-9,
    "diethyl ether": 4.30e-9,
}

#: The upper bound on each bucket's RMS ln-ratio, a shade above the measured
#: 0.195 / 0.242 / 0.734 so that noise does not fail the suite but a real
#: regression does.
RMS_CEILING = {"acyclic": 0.21, "aromatic": 0.26, "polyoxygenated": 0.76}


# -- what the correlation returns -------------------------------------------


@requires_rdkit
def test_viscosity_reproduces_the_named_measurements():
    """The six of the brief's seven that this expert agrees to answer.

    Stated measurements at 25 degrees Celsius, against which every prediction
    must sit inside its own stated bound - with one exception, recorded rather
    than papered over.
    """
    stated = {
        "ethanol": 1.07,
        "toluene": 0.56,
        "hexane": 0.30,
        "acetone": 0.31,
        "ethylene glycol": 16.1,
    }
    for name, measured in stated.items():
        value, std = _viscosity_mpa_s(BY_NAME[name])
        assert abs(value - measured) <= std, (
            f"{name}: predicted {value:.3f} +/- {std:.3f} mPa*s against a measured "
            f"{measured}"
        )

    # Glycerol is the one miss, and it is the reason the polyoxygenated bucket
    # reports 88 per cent one-sigma coverage rather than 100 per cent. Predicted
    # 472 mPa*s against a stated 950: a factor of two low, inside two sigma and
    # outside one. Asserting the miss keeps it from being quietly "fixed" by
    # widening the bar until everything fits.
    value, std = _viscosity_mpa_s(BY_NAME["glycerol"])
    assert value == pytest.approx(472.3, rel=0.01)
    assert abs(value - 950.0) > std
    assert abs(value - 950.0) <= 2 * std


@requires_rdkit
def test_water_is_refused_rather_than_answered():
    """The seventh of the brief's seven.

    Water has no carbon for the correlation's backbone term and no matchable
    group; unrefused it predicts 0.04 mPa*s against a measured 0.89, a factor
    of twenty. Two independent gates catch it, and the carbon one fires first.
    """
    predictions = _predict("O", density=997.06, molar_mass=18.015, boiling=373.12)
    for prop in BOTH:
        assert predictions[prop].status is PredictionStatus.UNSUPPORTED
        assert predictions[prop].quantity is None
        assert "no carbon" in predictions[prop].notes[0]
    # The oxygen mass balance would have caught it too, from the other side.
    assert group_counts("O")["unaccounted_oxygen"] == 1


@requires_rdkit
@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_every_reference_compound_predicts_with_a_defensible_bar(name):
    value, std = _viscosity_mpa_s(BY_NAME[name])
    assert value > 0.0
    assert std > 0.0
    # No bar so wide it says nothing, and none so narrow it cannot be wrong.
    assert 0.1 < std / value < 1.5


@requires_rdkit
@pytest.mark.parametrize("bucket", sorted(RMS_CEILING))
def test_bucket_accuracy_and_honesty_do_not_regress(bucket):
    """Accuracy *and* calibration, which is what ``formulate calibrate`` measures.

    A model with a 30 per cent error claiming 5 per cent is more dangerous to a
    ranking than one with a 30 per cent error claiming 30, so this fails both
    when the correlation degrades and when the error bar becomes dishonest in
    either direction.
    """
    residuals: list[float] = []
    covered = 0
    for row in REFERENCE:
        if uncertainty_bucket(row[1]) != bucket:
            continue
        measured = row[5]
        value, std = _viscosity_mpa_s(row)
        residuals.append(math.log(value / measured))
        covered += abs(value - measured) <= std

    n = len(residuals)
    assert n >= 8
    rms = math.sqrt(sum(r * r for r in residuals) / n)
    assert rms <= RMS_CEILING[bucket], f"{bucket} RMS ln-ratio regressed to {rms:.3f}"
    fraction = covered / n
    assert 0.60 <= fraction <= 0.90, (
        f"{bucket} one-sigma coverage is {fraction:.0%} over {n} compounds, against "
        "the ~68 per cent a correct estimate implies"
    )


@requires_rdkit
def test_homologous_series_reproduce_the_transcribed_group_table():
    """The check that the group table is right rather than merely plausible.

    A misremembered coefficient does not reproduce two homologous series
    across ten carbons each to within five per cent.
    """
    for name in ("heptane", "octane", "decane", "dodecane", "tetradecane"):
        value, _ = _viscosity_mpa_s(BY_NAME[name])
        assert value / BY_NAME[name][5] == pytest.approx(1.0, abs=0.05)
    for name in ("1-butanol", "1-hexanol", "1-octanol", "1-decanol"):
        value, _ = _viscosity_mpa_s(BY_NAME[name])
        assert value / BY_NAME[name][5] == pytest.approx(1.0, abs=0.05)


@requires_rdkit
def test_carboxylic_acid_is_not_charged_for_its_hydroxyl_twice():
    """The trap that put acetic acid ten times too thick before it was fixed.

    -COOH contains a hydroxyl and a carbonyl. Claiming the composite group
    first and excluding its atoms is what keeps the -OH contribution, which is
    the single largest B term in the table at +1600, from being added again.
    """
    counts = group_counts("CC(=O)O")
    assert counts["carboxyl"] == 1
    assert counts["hydroxyl"] == 0
    assert counts["carbonyl"] == 0
    value, _ = _viscosity_mpa_s(BY_NAME["acetic acid"])
    assert value / BY_NAME["acetic acid"][5] == pytest.approx(1.0, abs=0.1)


@requires_rdkit
def test_disubstituted_benzene_positions_are_counted():
    """Without the ortho/meta/para terms o-xylene came out at 0.84x."""
    assert group_counts("Cc1ccccc1C")["ortho"] == 1
    assert group_counts("Cc1cccc(C)c1")["meta"] == 1
    assert group_counts("Cc1ccc(C)cc1")["para"] == 1
    for name in ("o-xylene", "m-xylene", "p-xylene"):
        value, _ = _viscosity_mpa_s(BY_NAME[name])
        assert value / BY_NAME[name][5] == pytest.approx(1.0, abs=0.06)


# -- units and contract ------------------------------------------------------


@requires_rdkit
def test_units_convert_to_the_canonical_property_units():
    """The correlation's own unit is returned; the registry does the scaling.

    Hand-scaling centipoise to pascal-seconds inside the expert would be one
    more place for a factor of a thousand to hide.
    """
    predictions = _predict(
        "Cc1ccccc1", density=862.34, molar_mass=92.138, boiling=383.75
    )
    viscosity = predictions["shear_viscosity"]
    assert viscosity.quantity.unit == "centipoise"
    assert viscosity.quantity.dimensionality == get_property(
        "shear_viscosity"
    ).dimensionality
    assert viscosity.quantity.to("Pa*s").value == pytest.approx(0.4814e-3, rel=1e-3)
    assert viscosity.quantity.to("mPa*s").value == pytest.approx(0.4814, rel=1e-3)

    diffusion = predictions["self_diffusion_coefficient"]
    assert diffusion.quantity.dimensionality == get_property(
        "self_diffusion_coefficient"
    ).dimensionality
    assert diffusion.quantity.to_canonical().value == pytest.approx(1.95e-9, rel=0.02)


@requires_rdkit
def test_every_prediction_carries_an_uncertainty_and_a_method():
    predictions = _predict("CCCCCC", density=654.96, molar_mass=86.175, boiling=341.87)
    for prop in BOTH:
        pred = predictions[prop]
        assert pred.is_usable
        assert pred.uncertainty.std is not None and pred.uncertainty.std > 0.0
        assert pred.uncertainty.basis
        assert pred.method
        assert pred.provenance is not None
        assert pred.applicability.basis


def test_supported_classes_are_molecules_only():
    """A molecule correlation must not leak into the polymer experts' territory.

    ``polymer_melt`` and ``polymer_blend_melt`` already own shear_viscosity for
    those classes, and a melt viscosity is a different physical object from a
    solvent's.
    """
    expert = LiquidTransportExpert()
    assert expert.supported_classes == frozenset({MaterialClass.MOLECULE})
    for material_class in (MaterialClass.POLYMER, MaterialClass.MIXTURE):
        assert not expert.applicable_properties(BOTH, material_class)

    polymer = Candidate(
        material_class=MaterialClass.POLYMER,
        polymer=PolymerSpec(monomers=(MonomerUnit(smiles="[*]CC[*]"),)),
        conditions=Conditions.standard(),
    )
    request = PredictionRequest(
        candidate=polymer,
        properties=frozenset(BOTH),
        conditions=Conditions.standard(),
    )
    assert expert.predict(request) == []


def test_expert_does_not_depend_on_a_property_it_supplies():
    """The guard against a cycle that would kill the whole run.

    An expert that both supplies and depends on shear_viscosity makes
    ``resolution_order`` raise ``ValueError('Cyclic expert dependencies')``,
    because the external-dependency escape hatch checks whether anything still
    pending supplies the property - and this expert does. Self-diffusion
    therefore recomputes the viscosity internally.
    """
    expert = LiquidTransportExpert()
    assert not (expert.dependencies & expert.supported_properties)

    registry = ExpertRegistry(list(default_registry()))
    registry.register(expert)
    order = [e.id for e in registry.resolution_order(list(registry))]
    assert order.index("interfacial") < order.index("liquid_transport")
    assert order.index("joback") < order.index("liquid_transport")


# -- refusals ----------------------------------------------------------------


@requires_rdkit
@pytest.mark.parametrize(
    "name, smiles, fragment",
    [
        # Nitrogen, sulfur and fluorine have no contribution at all, so they
        # would cost nothing and the arithmetic would still complete.
        # Measured: aniline 0.13x, DMF 0.30x, DMSO 0.13x, perfluorohexane 9.9x.
        ("aniline", "Nc1ccccc1", "no group contribution for N"),
        ("dimethylformamide", "CN(C)C=O", "no group contribution for N"),
        ("dimethyl sulfoxide", "CS(C)=O", "no group contribution for S"),
        (
            "perfluorohexane",
            "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
            "no group contribution for F",
        ),
        # Saturated rings are biased 0.59x over ten compounds, worst
        # cyclohexanol at 0.13x - a wrong model, not an imprecise one.
        ("cyclohexane", "C1CCCCC1", "saturated-ring contributions"),
        ("tetrahydrofuran", "C1CCOC1", "saturated-ring contributions"),
        ("cyclohexanol", "OC1CCCCC1", "saturated-ring contributions"),
        ("1,4-dioxane", "C1COCCO1", "saturated-ring contributions"),
        # A ring the table has no term for at all.
        ("cyclooctane", "C1CCCCCCC1", "neither aromatic nor a five-"),
        ("propylene oxide", "CC1CO1", "neither aromatic nor a five-"),
        # The aromatic term was fitted to benzene rings.
        ("furan", "c1ccoc1", "matched no group"),
        # An oxygen the patterns cannot cost.
        ("ketene", "C=C=O", "matched no group"),
        # An oxygen costed twice: the ester pattern matches from both sides of
        # a shared bridging oxygen. Measured cost of allowing it through:
        # acetic anhydride 1.15x, dimethyl carbonate 1.30x.
        ("acetic anhydride", "CC(=O)OC(C)=O", "more oxygen than the molecule contains"),
        ("dimethyl carbonate", "COC(=O)OC", "more oxygen than the molecule contains"),
        ("propylene carbonate", "CC1COC(=O)O1", "more oxygen than the molecule contains"),
        # A peroxide balances its oxygens but is not two ethers.
        ("diethyl peroxide", "CCOOCC", "oxygen-oxygen bond"),
        ("tert-butyl hydroperoxide", "CC(C)(C)OO", "oxygen-oxygen bond"),
        # No triple-bond contribution exists.
        ("1-hexyne", "C#CCCCC", "no triple-bond contribution"),
        # Fitted to neutral liquids.
        ("acetate anion", "CC(=O)[O-]", "fitted to neutral liquids"),
        # No carbon to sum over.
        ("water", "O", "no carbon"),
        # A dotted SMILES is not one molecule, and the group counter cannot
        # tell: two hexanes sum to twelve carbons and read as dodecane.
        ("two hexanes in one SMILES", "CCCCCC.CCCCCC", "more than one disconnected"),
        ("two ethanols in one SMILES", "CCO.CCO", "more than one disconnected"),
        ("sodium acetate", "CC(=O)[O-].[Na+]", "no group contribution for Na"),
        # Net charge is zero and it is still an inner salt.
        ("a C/H/O inner salt", "C[O+](C)CC(=O)[O-]", "zwitterion"),
        # An unfilled valence the table happily counts as ethane.
        ("ethyl radical", "C[CH2]", "unpaired electron"),
    ],
)
def test_structures_outside_the_table_are_refused_not_guessed(name, smiles, fragment):
    reason = structural_refusal(smiles)
    assert reason is not None, f"{name} should be refused"
    assert fragment in reason

    predictions = _predict(smiles, density=900.0, molar_mass=100.0, boiling=450.0)
    for prop in BOTH:
        assert predictions[prop].status is PredictionStatus.UNSUPPORTED
        assert predictions[prop].quantity is None
        assert fragment in predictions[prop].notes[0]


@requires_rdkit
def test_a_mixture_written_as_one_smiles_is_refused_not_averaged():
    """Rule seven, enforced where it would otherwise fail silently.

    ``group_counts`` sums over atoms and has no idea a SMILES is disconnected,
    so two hexanes present as twelve carbons. Unrefused the arithmetic reads
    1.18 mPa*s - dodecane, near enough - against hexane's measured 0.298, and
    two ethanols read 32 mPa*s against 1.08. Nothing about either number looks
    wrong on the way past.
    """
    for smiles in ("CCCCCC.CCCCCC", "CCO.CCO"):
        assert structural_refusal(smiles) is not None
        assert not LiquidTransportExpert().assess_domain(
            molecule_candidate(smiles)
        ).in_domain

    # What the gate is standing in front of, computed directly so the number
    # is measured rather than asserted from memory.
    counts = group_counts("CCCCCC.CCCCCC")
    assert counts["carbon"] == 12
    summed = orrick_erbar_viscosity(counts, 0.65496, 2 * 86.175, 298.15)
    assert summed == pytest.approx(1.18, abs=0.02)
    assert summed / BY_NAME["hexane"][5] > 3.9


@requires_rdkit
def test_an_open_shell_or_charged_structure_is_refused():
    """Neutral means every atom, and closed-shell means an unbroken valence.

    A net-charge test passes an inner salt, which is ionic; an unfilled
    valence is a structure the table is counting atoms that are not bonded the
    way it assumes. The ethyl radical reads 0.054 mPa*s, ethane's number, for
    something with no bulk liquid phase at all.
    """
    inner_salt = "C[O+](C)CC(=O)[O-]"
    from rdkit import Chem

    assert Chem.GetFormalCharge(Chem.MolFromSmiles(inner_salt)) == 0
    assert "zwitterion" in structural_refusal(inner_salt)
    assert "unpaired electron" in structural_refusal("C[CH2]")


@requires_rdkit
def test_the_domain_basis_quotes_the_set_it_was_actually_validated_on():
    """A stale validation number is the defect this file exists to catch.

    The basis string is attached to every prediction and every refusal this
    expert makes, so the counts in it have to be the counts in this file. It
    said 113 and 90 while the validation set was 134 and 110.
    """
    basis = LiquidTransportExpert().assess_domain(
        molecule_candidate("Cc1ccccc1")
    ).basis
    assert str(len(REFERENCE)) in basis
    assert len(REFERENCE) == 110
    assert "134" in basis and "24" in basis


@requires_rdkit
def test_a_refused_structure_is_also_out_of_domain():
    """assess_domain and the refusal must not disagree about the same molecule."""
    expert = LiquidTransportExpert()
    assert not expert.assess_domain(molecule_candidate("Nc1ccccc1")).in_domain
    assert expert.assess_domain(molecule_candidate("Cc1ccccc1")).in_domain


@requires_rdkit
def test_glycol_ethers_are_predicted_but_flagged():
    """The bucket with the least evidence behind its bar, and it says so."""
    domain = LiquidTransportExpert().assess_domain(molecule_candidate("OCCOCCO"))
    assert domain.in_domain
    assert domain.score < 0.5
    assert any("oxygen-bearing groups" in w for w in domain.warnings)


# -- dependencies ------------------------------------------------------------


@requires_rdkit
def test_a_missing_density_refuses_both_properties():
    predictions = _predict(
        "Cc1ccccc1", density=None, molar_mass=92.138, boiling=383.75, molar_volume=1.07e-4
    )
    for prop in BOTH:
        assert predictions[prop].status is PredictionStatus.UNSUPPORTED
        assert "liquid_density" in predictions[prop].notes[0]
        assert predictions[prop].quantity is None


@requires_rdkit
def test_a_missing_boiling_point_refuses_both_properties():
    predictions = _predict(
        "Cc1ccccc1", density=862.34, molar_mass=92.138, boiling=None
    )
    for prop in BOTH:
        assert predictions[prop].status is PredictionStatus.UNSUPPORTED
        assert "normal_boiling_point" in predictions[prop].notes[0]


@requires_rdkit
def test_a_missing_molar_volume_refuses_only_self_diffusion():
    """Refusal is per property, not for the whole expert.

    A cyclic-carbonate candidate gets no liquid density from the current panel
    at all; a candidate missing only the molar volume can still be told how
    thick it is.
    """
    predictions = _predict(
        "Cc1ccccc1",
        density=862.34,
        molar_mass=92.138,
        boiling=383.75,
        molar_volume=None,
    )
    assert predictions["shear_viscosity"].is_usable
    assert (
        predictions["self_diffusion_coefficient"].status is PredictionStatus.UNSUPPORTED
    )
    assert "molar_volume_liquid" in predictions["self_diffusion_coefficient"].notes[0]


@requires_rdkit
def test_no_temperature_refuses_both_properties():
    predictions = _predict(
        "Cc1ccccc1",
        density=862.34,
        molar_mass=92.138,
        boiling=383.75,
        temperature_k=None,
    )
    for prop in BOTH:
        assert predictions[prop].status is PredictionStatus.UNSUPPORTED
        assert "temperature" in predictions[prop].notes[0]


@requires_rdkit
def test_upstream_density_uncertainty_widens_the_bar():
    """Uncertainty is propagated, not just the correlation's own error quoted.

    A novel molecule's density arrives from Rackett at roughly ten per cent; a
    tabulated one at a fraction of a per cent. The viscosity is linear in
    density, so the difference has to show.
    """
    tight = _predict(
        "Cc1ccccc1", density=862.34, molar_mass=92.138, boiling=383.75, density_std=2.0
    )["shear_viscosity"]
    loose = _predict(
        "Cc1ccccc1", density=862.34, molar_mass=92.138, boiling=383.75, density_std=95.0
    )["shear_viscosity"]
    value = tight.quantity.value
    assert loose.quantity.value == pytest.approx(value, rel=1e-9)
    assert loose.uncertainty.std > tight.uncertainty.std

    # The widening is exactly the density's own relative error, because the
    # correlation is linear in density: 95 of 862 kg/m^3 is 11 per cent, and
    # what it adds in quadrature recovers that.
    added = math.sqrt(loose.uncertainty.std**2 - tight.uncertainty.std**2)
    assert added / value == pytest.approx(95.0 / 862.34, rel=0.05)


# -- phase -------------------------------------------------------------------


@requires_rdkit
def test_above_the_boiling_point_there_is_no_liquid():
    predictions = _predict(
        "Cc1ccccc1",
        density=862.34,
        molar_mass=92.138,
        boiling=383.75,
        temperature_k=473.15,
    )
    for prop in BOTH:
        assert predictions[prop].status is PredictionStatus.UNSUPPORTED
        assert "boiling point" in predictions[prop].notes[0]


@requires_rdkit
def test_below_the_melting_point_the_extrapolation_is_refused():
    """An Arrhenius form extrapolated into a solid is a supercooled fiction."""
    from formulate.experts.measured import not_liquid_at

    if not_liquid_at("CCCCCCCCCCCCCCCC", 273.15) is None:
        pytest.skip("no compiled melting point available for hexadecane")
    predictions = _predict(
        "CCCCCCCCCCCCCCCC",
        density=770.3,
        molar_mass=226.441,
        boiling=559.9,
        temperature_k=273.15,
    )
    for prop in BOTH:
        assert predictions[prop].status is PredictionStatus.UNSUPPORTED
        assert "melting" in predictions[prop].notes[0]


@requires_rdkit
def test_the_bar_widens_away_from_the_calibration_temperature():
    """The B/T form is Arrhenius and drifts one-directionally as T rises."""
    at_25 = _predict(
        "CCCCCCCCCCCCCCCC",
        density=770.3,
        molar_mass=226.441,
        boiling=559.9,
        temperature_k=298.15,
    )["shear_viscosity"]
    # 682.5 kg/m^3 is hexadecane at 150 degrees Celsius. The number here used
    # to be 735, which is its density at about 75 - a seven per cent error in
    # the linear prefactor, in the flattering direction, inside a test whose
    # whole point is the extrapolation. It also meant the test did not
    # reproduce the 0.416 mPa*s the module docstring quotes; it produced 0.448.
    at_150 = _predict(
        "CCCCCCCCCCCCCCCC",
        density=682.5,
        molar_mass=226.441,
        boiling=559.9,
        temperature_k=423.15,
    )["shear_viscosity"]
    relative_25 = at_25.uncertainty.std / at_25.quantity.value
    relative_150 = at_150.uncertainty.std / at_150.quantity.value
    assert relative_150 > 1.4 * relative_25
    # And it has to still be right: hexadecane measures 0.537 mPa*s at 150 C.
    # This is the docstring's 0.416, and the margin is thin - 0.12 of a 0.137
    # bar, nine tenths of one sigma - which is what the wrong density was
    # hiding.
    assert at_150.quantity.to("mPa*s").value == pytest.approx(0.416, abs=0.002)
    assert abs(at_150.quantity.to("mPa*s").value - 0.537) <= at_150.uncertainty.converted(
        at_150.quantity.unit, "mPa*s"
    ).std

    # The extrapolation is said out loud rather than only priced in.
    assert not any("extrapolation and not a calibrated point" in n for n in at_25.notes)
    assert any("extrapolation and not a calibrated point" in n for n in at_150.notes)


# -- self-diffusion -----------------------------------------------------------


@requires_rdkit
def test_self_diffusion_reproduces_measured_nmr_coefficients():
    """Seventeen liquids, each inside its own stated bound, and the spread reported."""
    residuals: list[float] = []
    covered = 0
    for name, measured in SELF_DIFFUSION.items():
        row = BY_NAME[name]
        pred = _predict(row[1], density=row[2], molar_mass=row[3], boiling=row[4])[
            "self_diffusion_coefficient"
        ]
        assert pred.is_usable, f"{name}: {pred.notes}"
        value = pred.quantity.to("m^2/s").value
        residuals.append(math.log(value / measured))
        covered += abs(value - measured) <= pred.uncertainty.std

    n = len(residuals)
    rms = math.sqrt(sum(r * r for r in residuals) / n)
    bias = math.exp(sum(residuals) / n)
    assert rms <= 0.33, f"self-diffusion RMS ln-ratio regressed to {rms:.3f}"
    assert 0.70 <= bias <= 0.90, f"bias moved to x{bias:.2f}"
    assert 0.60 <= covered / n <= 0.90, f"coverage is {covered / n:.0%} over {n} liquids"


@requires_rdkit
def test_the_slip_prefactor_is_pinned_against_the_stick_one():
    """4*pi rather than 6*pi, asserted rather than asserted in a comment.

    The stick form is systematically a factor of two low - bias x0.52 against
    x0.78 - which is exactly the "factor of 2-3 out" a small molecule diffusing
    in a liquid of its own size is expected to produce.
    """
    slip: list[float] = []
    stick: list[float] = []
    slip_covered = stick_covered = 0
    for name, measured in SELF_DIFFUSION.items():
        row = BY_NAME[name]
        pred = _predict(row[1], density=row[2], molar_mass=row[3], boiling=row[4])[
            "self_diffusion_coefficient"
        ]
        value = pred.quantity.to("m^2/s").value
        slip.append(math.log(value / measured))
        stick.append(math.log(value * 2.0 / 3.0 / measured))
        slip_covered += abs(value - measured) <= pred.uncertainty.std
        stick_covered += abs(value * 2.0 / 3.0 - measured) <= pred.uncertainty.std

    n = len(slip)
    assert math.exp(sum(slip) / n) == pytest.approx(0.79, abs=0.03)
    assert math.exp(sum(stick) / n) == pytest.approx(0.53, abs=0.03)
    # The stick form is not merely worse on average; against the bar this
    # expert actually reports it lands inside for four of seventeen liquids
    # where the slip form lands inside for thirteen.
    assert slip_covered >= 12
    assert stick_covered <= 5
    assert stick_covered * 2 < slip_covered


@requires_rdkit
def test_a_thick_liquid_refuses_self_diffusion_but_still_reports_viscosity():
    """Diffusion decouples from viscosity near the glass transition.

    Glycerol reads x0.42 against the x0.66-x1.01 everything from 0.2 to
    17 mPa*s reads. That is a different physical regime, not a wider bar.
    """
    row = BY_NAME["glycerol"]
    predictions = _predict(row[1], density=row[2], molar_mass=row[3], boiling=row[4])
    assert predictions["shear_viscosity"].is_usable
    diffusion = predictions["self_diffusion_coefficient"]
    assert diffusion.status is PredictionStatus.UNSUPPORTED
    assert diffusion.quantity is None
    assert "decouples from viscosity" in diffusion.notes[0]
    assert "glass transition" in diffusion.notes[0]


@requires_rdkit
def test_a_heavy_molecule_is_told_it_is_being_extrapolated():
    """The residual trends with molar mass and the set tops out at 154 g/mol."""
    row = BY_NAME["hexadecane"]
    pred = _predict(row[1], density=row[2], molar_mass=row[3], boiling=row[4])[
        "self_diffusion_coefficient"
    ]
    assert pred.is_usable
    assert any("155 g/mol" in note for note in pred.notes)


@requires_rdkit
def test_an_inconsistent_molar_volume_is_flagged_not_absorbed():
    """The engine resolves density and molar volume independently.

    It can hand over a measured density beside an estimated molar volume, and
    then the radius is not describing the same liquid as the viscosity it
    divides.
    """
    row = BY_NAME["toluene"]
    consistent = _predict(row[1], density=row[2], molar_mass=row[3], boiling=row[4])[
        "self_diffusion_coefficient"
    ]
    assert not any("disagrees" in note for note in consistent.notes)

    inconsistent = _predict(
        row[1],
        density=row[2],
        molar_mass=row[3],
        boiling=row[4],
        molar_volume=row[3] / row[2] * 1e-3 * 1.3,
    )["self_diffusion_coefficient"]
    assert any("disagrees" in note for note in inconsistent.notes)


# -- the correlations in isolation -------------------------------------------


def test_orrick_erbar_is_the_published_form():
    """eta = rho * M * exp(A + B/T), evaluated on hexane's own group count.

    Hexane matches nothing but its six carbons, so A and B are the backbone
    terms alone: A = -(6.95 + 0.21*6) = -8.21 and B = 275 + 99*6 = 869.
    """
    counts = {"carbon": 6}
    expected = 0.65496 * 86.175 * math.exp(-8.21 + 869.0 / 298.15)
    assert orrick_erbar_viscosity(counts, 0.65496, 86.175, 298.15) == pytest.approx(
        expected
    )
    assert expected == pytest.approx(0.283, abs=0.002)


def test_stokes_einstein_radius_and_relation():
    """A sphere of the per-molecule liquid volume, moving with slip."""
    molar_volume = 92.138 / 862.34 * 1e-3  # toluene, m^3/mol
    radius = hydrodynamic_radius(molar_volume)
    assert radius == pytest.approx(3.49e-10, rel=0.01)
    diffusion = stokes_einstein_diffusion(298.15, 0.5533e-3, radius)
    assert diffusion == pytest.approx(1.70e-9, rel=0.02)
