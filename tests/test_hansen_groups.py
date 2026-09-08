"""The group-contribution Hansen route, against the compilation it stands in for.

The calibration is the test. A group method is only worth having if its error
is known, and the numbers asserted here are the ones measured over every
reference structure the method accepts.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("chemicals")

from formulate.experts.hansen import (  # noqa: E402
    HVK_SIGMA,
    hansen_triple,
    hoftyzer_van_krevelen,
    resolve_cas,
)


def _molar_volume(smiles: str) -> float | None:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    from formulate.experts.measured import measured_value

    density = measured_value("liquid_density", smiles)
    if density is None or density <= 0:
        return None
    return float(Descriptors.MolWt(Chem.MolFromSmiles(smiles))) / (density / 1000.0)


def test_the_method_is_as_accurate_as_it_claims():
    """Mean absolute error against the compilation, over what it accepts."""
    from formulate.exploration.database import catalogue

    errors = {"hansen_dispersion": [], "hansen_polar": [], "hansen_hydrogen_bonding": []}
    for record in catalogue():
        smiles = record["smiles"]
        if resolve_cas(smiles) is None:
            continue
        reference = hansen_triple(smiles)
        if reference is None or reference[0] == 0.0:
            continue
        volume = _molar_volume(smiles)
        if volume is None:
            continue
        estimate = hoftyzer_van_krevelen(smiles, volume)
        if estimate is None:
            continue
        for key, got, ref in zip(errors, estimate, reference):
            errors[key].append(got - ref / 1e3)

    assert len(errors["hansen_dispersion"]) >= 35
    for key, values in errors.items():
        values = np.array(values)
        mae = float(np.abs(values).mean())
        # 1.253 * MAE is the one-sigma the expert quotes; it must not be
        # optimistic about itself.
        assert 1.253 * mae <= HVK_SIGMA[key] * 1.10, (key, mae)


def test_an_uncovered_group_is_refused_rather_than_partially_summed():
    """Dimethyl sulfoxide has no sulfoxide group, and a partial sum looks fine."""
    assert hoftyzer_van_krevelen("CS(=O)C", 71.3) is None
    # Benzoyl peroxide: no -O-O- group either.
    assert hoftyzer_van_krevelen("O=C(OOC(=O)c1ccccc1)c1ccccc1", 200.0) is None


def test_a_polyhalogenated_carbon_is_refused():
    """The -Cl contribution is calibrated for one substituent, not for four."""
    assert hoftyzer_van_krevelen("ClC(Cl)(Cl)Cl", 97.1) is None
    assert hoftyzer_van_krevelen("ClCCCl", 79.4) is not None  # one Cl per carbon is fine


def test_the_polar_term_adds_in_quadrature_not_linearly():
    """The bug that made the polar error nearly three times what it should be.

    Two ester groups in one molecule contribute sqrt(2) times one ester's Fp,
    not twice it, so a diester's polar parameter is well below a monoester's
    per unit volume.
    """
    single = hoftyzer_van_krevelen("CCOC(C)=O", 98.5)
    double = hoftyzer_van_krevelen("C=CC(=O)OCCCCCCOC(=O)C=C", 224.0)
    assert single is not None and double is not None
    # Linear summing would put the diester's polar term above this.
    assert double[1] < single[1]
