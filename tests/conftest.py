"""Shared fixtures."""

from __future__ import annotations

import pytest

from formulate import chem
from formulate.core.candidate import molecule_candidate
from formulate.core.conditions import Conditions
from formulate.experts import default_registry
from formulate.targets.spec import TargetSpec

requires_rdkit = pytest.mark.skipif(
    not chem.rdkit_available(), reason="RDKit is not installed"
)


@pytest.fixture
def registry():
    return default_registry()


@pytest.fixture
def standard_conditions():
    return Conditions.standard()


@pytest.fixture
def solvent_spec() -> TargetSpec:
    """A realistic multi-objective request with one hard constraint."""
    return TargetSpec.from_dict(
        {
            "name": "test solvent",
            "conditions": {"temperature": "25 degC", "pressure": "1 atm"},
            "requirements": [
                {
                    "property": "normal_boiling_point",
                    "direction": "in_range",
                    "lower": "60 degC",
                    "upper": "160 degC",
                    "hard": True,
                },
                {
                    "property": "surface_tension",
                    "direction": "minimize",
                    "lower": "0.015 N/m",
                    "upper": "0.040 N/m",
                    "weight": 2.0,
                },
                {
                    "property": "synthetic_accessibility",
                    "direction": "minimize",
                    "lower": 1,
                    "upper": 5,
                },
            ],
        }
    )


@pytest.fixture
def small_pool():
    return [
        molecule_candidate(smiles, label=name)
        for smiles, name in [
            ("CCCCCC", "hexane"),
            ("c1ccccc1", "benzene"),
            ("CCO", "ethanol"),
            ("CC(C)=O", "acetone"),
            ("Cc1ccccc1", "toluene"),
        ]
    ]
