"""Polymer repeat-unit experts: glass transition and amorphous density.

The point of most of these tests is not that the numbers are pretty.  Both
models are fitted in this repository against ``reference_polymers.json``, so
reproducing the polymers they were fitted to proves nothing at all.  What is
asserted here is the error on the validation split - polymers no coefficient,
and no descriptor choice, ever saw - and the refusals: the cases where the
right answer is to decline rather than to extrapolate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import requires_rdkit

from formulate.core.candidate import (
    MaterialClass,
    MonomerUnit,
    PolymerSpec,
    PolymerTopology,
    Tacticity,
    molecule_candidate,
    polymer_candidate,
)
from formulate.core.conditions import Conditions
from formulate.core.prediction import PredictionStatus
from formulate.core.properties import PROPERTY_REGISTRY
from formulate.core.quantity import Quantity, UncertaintyKind
from formulate.experts.base import PredictionRequest
from formulate.experts.polymer import (
    analyse_chain,
    cached_vdw_volume,
    glass_transition_model,
    link_repeat_units,
    packing_model,
    predict_glass_transition,
    reference_split,
    repeat_unit_groups,
    repeat_unit_mass,
    repeat_unit_vdw_volume,
    symmetric_gem_substitution,
    PolymerDensityExpert,
    PolymerGlassTransitionExpert,
)

pytestmark = requires_rdkit

ROOM = Conditions(temperature=Quantity(value=298.15, unit="K"))

PE = "[*]CC[*]"
PP = "[*]CC(C)[*]"
PS = "[*]CC(c1ccccc1)[*]"
PMMA = "[*]CC(C)(C(=O)OC)[*]"
PIB = "[*]CC(C)(C)[*]"
PDMS = "[*][Si](C)(C)O[*]"
PC = "[*]OC(=O)Oc1ccc(cc1)C(C)(C)c1ccc(cc1)[*]"


# -- repeat-unit machinery -------------------------------------------------


def test_repeat_unit_mass_is_the_monomer_not_the_capped_oligomer():
    # Ethylene, not ethane: the caps must cancel out of the difference.
    assert repeat_unit_mass(PE) == pytest.approx(28.05, abs=0.02)
    assert repeat_unit_mass(PS) == pytest.approx(104.15, abs=0.02)
    assert repeat_unit_mass(PMMA) == pytest.approx(100.12, abs=0.02)


def test_linking_grows_the_chain_by_one_unit_at_a_time():
    sizes = [link_repeat_units(PP, n).GetNumAtoms() for n in (2, 3, 4)]
    assert sizes[1] - sizes[0] == sizes[2] - sizes[1] == 3


def test_decomposition_isolates_one_interior_repeat_unit():
    # Polyethylene is two methylenes and nothing else; the methyl caps and the
    # terminal groups they create must not leak into the count.
    groups = repeat_unit_groups(PE)
    assert sum(groups.values()) == 2
    assert set(groups.values()) == {2}

    # Polystyrene: one CH2, one CH, and a whole aromatic ring.
    assert sum(repeat_unit_groups(PS).values()) == 8


def test_decomposition_refuses_backbones_outside_the_group_set():
    # No siloxane and no carbonate group exists in Joback's set. Returning a
    # partial decomposition would silently drop atoms and under-predict Tg.
    assert repeat_unit_groups(PDMS) is None
    assert repeat_unit_groups(PC) is None


def test_a_repeat_unit_needs_exactly_two_attachment_points():
    with pytest.raises(ValueError, match="attachment points"):
        repeat_unit_mass("CCO")


def test_symmetric_geminal_substitution_is_detected_only_when_symmetric():
    assert symmetric_gem_substitution(PIB) == 1                 # two methyls
    assert symmetric_gem_substitution("[*]CC(Cl)(Cl)[*]") == 1  # two chlorines
    assert symmetric_gem_substitution(PMMA) == 0                # methyl and ester
    assert symmetric_gem_substitution(PS) == 0
    assert symmetric_gem_substitution("[*]C(F)(F)C(F)(Cl)[*]") == 1


# -- the glass-transition model -------------------------------------------


def test_validation_split_error_is_within_the_declared_uncertainty():
    """The only honest number: polymers the fit and the descriptors never saw."""
    model = glass_transition_model()
    errors = []
    for record in reference_split("validation"):
        predicted = predict_glass_transition(record["repeat_unit"], model)
        assert predicted is not None, record["name"]
        errors.append(predicted - float(record["glass_transition_k"]))
    errors = np.array(errors)

    assert len(errors) >= 8
    assert np.sqrt(np.mean(errors**2)) < 30.0
    assert np.max(np.abs(errors)) < 3.0 * model.sigma
    # The validation split does carry a systematic offset - the model runs
    # about 13 K high on polymers it has not seen, against a 22 K scatter,
    # so the offset is real but sits well inside the quoted uncertainty.
    # Correcting for it would be fitting to the validation set, which is the
    # one thing that set exists not to be used for.
    assert 0.0 < errors.mean() < model.sigma


def test_quoted_uncertainty_is_not_the_in_sample_residual():
    model = glass_transition_model()
    fitted = []
    for record in reference_split("fit"):
        predicted = predict_glass_transition(record["repeat_unit"], model)
        if predicted is not None:
            fitted.append(predicted - float(record["glass_transition_k"]))
    in_sample = float(np.sqrt(np.mean(np.square(fitted))))
    assert model.sigma > in_sample


def test_the_symmetry_descriptor_earns_its_place_out_of_sample():
    """Held-out error must be worse without it, or it is decoration."""
    model = glass_transition_model()
    records = [
        r for r in reference_split("fit") if repeat_unit_groups(r["repeat_unit"]) is not None
    ]
    counts = np.array(
        [[repeat_unit_groups(r["repeat_unit"]).get(k, 0) for k in model.group_keys] for r in records],
        dtype=float,
    )
    gem = np.array([[symmetric_gem_substitution(r["repeat_unit"])] for r in records], dtype=float)
    mass = np.array([repeat_unit_mass(r["repeat_unit"]) for r in records])
    tg = np.array([float(r["glass_transition_k"]) for r in records])

    def held_out(features):
        design = features / mass[:, None]
        residuals = []
        for i in range(len(tg)):
            keep = np.ones(len(tg), dtype=bool)
            keep[i] = False
            present = counts[i] > 0
            if (counts[keep][:, present] > 0).sum(axis=0).min() < 3:
                continue
            coefficients = np.linalg.solve(
                design[keep].T @ design[keep] + 1e-5 * np.eye(design.shape[1]),
                design[keep].T @ tg[keep],
            )
            residuals.append(tg[i] - design[i] @ coefficients)
        return float(np.sqrt(np.mean(np.square(residuals))))

    assert held_out(np.hstack([counts, gem])) < held_out(counts) - 5.0


def test_polyisobutylene_sits_below_polypropylene():
    """The effect the symmetry descriptor exists for, on the sign that matters."""
    model = glass_transition_model()
    assert predict_glass_transition(PIB, model) < predict_glass_transition(PP, model)


def test_prediction_refuses_rather_than_approximating_an_undecomposable_unit():
    assert predict_glass_transition(PDMS) is None
    assert predict_glass_transition(PC) is None


# -- the packing model ----------------------------------------------------


def test_van_der_waals_volume_reproduces_bondi_group_volumes():
    # Bondi's increments: CH2 10.23, CH 6.78, CH3 13.67, C6H5 45.84 cm^3/mol,
    # divided by 0.6022 to reach cubic angstroms per repeat unit.
    assert repeat_unit_vdw_volume(PE) == pytest.approx(2 * 10.23 / 0.6022, rel=0.03)
    assert repeat_unit_vdw_volume(PP) == pytest.approx((10.23 + 6.78 + 13.67) / 0.6022, rel=0.03)
    assert repeat_unit_vdw_volume(PS) == pytest.approx((10.23 + 6.78 + 45.84) / 0.6022, rel=0.03)


def test_packing_factor_lands_near_the_published_value():
    # van Krevelen puts the bulk molar volume of an amorphous polymer at
    # roughly 1.6 times its van der Waals volume. Nothing here was fitted to
    # that number, so recovering it is independent corroboration that the
    # volume route is right rather than merely self-consistent.
    model = packing_model()
    assert 1.4 < model.packing_factor < 1.7


def test_density_validation_error_is_small_and_honest():
    model = packing_model()
    assert model.validation_n >= 4
    assert model.validation_rms_percent < 6.0
    assert model.sigma_percent >= model.validation_rms_percent


def test_the_glassy_rubbery_refinement_was_rejected_for_a_reason():
    """Two packing factors fit the fit split better and transfer worse."""
    records = [
        r for r in reference_split("fit")
        if r.get("van_der_waals_volume_a3") and r.get("amorphous_density_g_cm3")
    ]
    ratio = np.array([
        repeat_unit_mass(r["repeat_unit"])
        / (r["amorphous_density_g_cm3"] * r["van_der_waals_volume_a3"] * 0.6022140761)
        for r in records
    ])
    glassy = np.array([298.15 < float(r["glass_transition_k"]) for r in records])
    # The two populations do differ - a rubber at room temperature has expanded
    # past its transition - which is why the refinement was worth trying.
    assert ratio[~glassy].mean() > ratio[glassy].mean()
    # But not by enough to beat the scatter within either population.
    spread = ratio.std()
    assert abs(ratio[~glassy].mean() - ratio[glassy].mean()) < spread


# -- reading a PolymerSpec ------------------------------------------------


def test_copolymer_is_averaged_over_its_backbone_monomers():
    spec = PolymerSpec(
        monomers=(
            MonomerUnit(smiles=PS, mole_fraction=0.5),
            MonomerUnit(smiles=PE, mole_fraction=0.5),
        )
    )
    analysis = analyse_chain(spec)
    assert analysis.mass == pytest.approx(0.5 * (104.15 + 28.05), abs=0.05)
    assert any("random copolymer" in note for note in analysis.notes)


def test_copolymer_glass_transition_lies_between_its_homopolymers():
    expert = PolymerGlassTransitionExpert()
    styrene = _value(expert, polymer_candidate(PS))
    ethylene = _value(expert, polymer_candidate(PE))
    blend = _value(expert, polymer_candidate([(PS, 0.5), (PE, 0.5)]))
    assert ethylene < blend < styrene


def test_end_groups_are_not_repeat_units():
    from formulate.core.candidate import MonomerRole

    spec = PolymerSpec(
        monomers=(
            MonomerUnit(smiles=PS, mole_fraction=1.0),
            MonomerUnit(smiles="[*]C", role=MonomerRole.END_GROUP, mole_fraction=0.0),
        )
    )
    assert analyse_chain(spec).mass == pytest.approx(repeat_unit_mass(PS), abs=0.05)


# -- expert behaviour -----------------------------------------------------


def _predict(expert, candidate, conditions=ROOM):
    request = PredictionRequest(
        candidate=candidate,
        properties=frozenset(expert.supported_properties),
        conditions=conditions,
    )
    return expert.predict(request)[0]


def _value(expert, candidate, unit="K", conditions=ROOM):
    prediction = _predict(expert, candidate, conditions)
    assert prediction.quantity is not None, prediction.notes
    return prediction.quantity.to(unit).value


def test_glass_transition_expert_reproduces_two_polymers_it_was_fitted_to():
    expert = PolymerGlassTransitionExpert()
    assert _value(expert, polymer_candidate(PS)) == pytest.approx(373, abs=40)
    assert _value(expert, polymer_candidate(PE)) == pytest.approx(195, abs=40)


def test_experts_ignore_molecules():
    for expert in (PolymerGlassTransitionExpert(), PolymerDensityExpert()):
        request = PredictionRequest(
            candidate=molecule_candidate("CCO"),
            properties=frozenset(expert.supported_properties),
            conditions=ROOM,
        )
        assert expert.predict(request) == []


def test_stereoregular_polymer_is_flagged_not_answered_as_atactic():
    expert = PolymerGlassTransitionExpert()
    spec = PolymerSpec(monomers=(MonomerUnit(smiles=PMMA),), tacticity=Tacticity.ISOTACTIC)
    from formulate.core.candidate import Candidate

    prediction = _predict(
        expert, Candidate(material_class=MaterialClass.POLYMER, polymer=spec)
    )
    assert prediction.status is PredictionStatus.OUT_OF_DOMAIN
    assert prediction.is_usable  # penalised, not discarded
    assert any("atactic" in w for w in prediction.applicability.warnings)


def test_network_polymer_is_out_of_domain_for_both_experts():
    from formulate.core.candidate import Candidate

    spec = PolymerSpec(monomers=(MonomerUnit(smiles=PS),), topology=PolymerTopology.NETWORK)
    candidate = Candidate(material_class=MaterialClass.POLYMER, polymer=spec)
    for expert in (PolymerGlassTransitionExpert(), PolymerDensityExpert()):
        assert _predict(expert, candidate).status is PredictionStatus.OUT_OF_DOMAIN


def test_short_chains_widen_the_uncertainty_then_leave_the_domain():
    from formulate.core.candidate import Candidate

    expert = PolymerGlassTransitionExpert()

    def predict(mn):
        spec = PolymerSpec(
            monomers=(MonomerUnit(smiles=PS),),
            number_average_molar_mass=Quantity(value=mn, unit="g/mol"),
        )
        return _predict(expert, Candidate(material_class=MaterialClass.POLYMER, polymer=spec))

    high = _predict(expert, polymer_candidate(PS))
    medium = predict(4000.0)
    oligomer = predict(800.0)

    assert medium.uncertainty.std > high.uncertainty.std
    assert medium.status is PredictionStatus.OK
    assert oligomer.status is PredictionStatus.OUT_OF_DOMAIN
    # The value itself is unchanged: the chain-end constant is polymer-specific
    # and unknown here, so the depression sizes the error bar and nothing else.
    assert medium.quantity.value == pytest.approx(high.quantity.value)


def test_thinly_supported_groups_leave_the_domain():
    # Nitrile appears in two reference polymers, below the support floor.
    expert = PolymerGlassTransitionExpert()
    prediction = _predict(expert, polymer_candidate("[*]CC(C#N)[*]"))
    assert prediction.status is PredictionStatus.OUT_OF_DOMAIN
    assert any("fewer than" in w for w in prediction.applicability.warnings)


def test_glass_transition_expert_declines_an_undecomposable_backbone():
    expert = PolymerGlassTransitionExpert()
    prediction = _predict(expert, polymer_candidate(PDMS))
    assert prediction.status is PredictionStatus.UNSUPPORTED
    assert prediction.quantity is None


def test_density_expert_answers_where_the_group_table_cannot():
    """A siloxane or a carbonate has no Joback group but it still has a volume."""
    expert = PolymerDensityExpert()
    carbonate = _predict(expert, polymer_candidate(PC))
    assert carbonate.status is PredictionStatus.OK
    assert carbonate.quantity.to("g/cm^3").value == pytest.approx(1.20, rel=0.06)


def test_density_expert_refuses_to_extrapolate_off_its_elements():
    # Poly(dimethylsiloxane) measures 0.97 g/cm^3 where a carbon-backbone
    # packing factor says about 1.13. The expert must not present that as OK.
    expert = PolymerDensityExpert()
    prediction = _predict(expert, polymer_candidate(PDMS))
    assert prediction.status is PredictionStatus.OUT_OF_DOMAIN
    assert any("Si" in w for w in prediction.applicability.warnings)


def test_density_needs_a_temperature_and_refuses_outside_its_window():
    expert = PolymerDensityExpert()
    assert _predict(expert, polymer_candidate(PS), Conditions()).status is (
        PredictionStatus.UNSUPPORTED
    )
    hot = _predict(
        expert, polymer_candidate(PS), Conditions(temperature=Quantity(value=450.0, unit="K"))
    )
    assert hot.status is PredictionStatus.UNSUPPORTED


def test_every_prediction_carries_a_real_uncertainty():
    for expert, unit in ((PolymerGlassTransitionExpert(), "K"), (PolymerDensityExpert(), "kg/m^3")):
        prediction = _predict(expert, polymer_candidate(PS))
        assert prediction.uncertainty.std is not None
        assert prediction.uncertainty.std > 0.0
        assert prediction.uncertainty.kind is not UncertaintyKind.UNKNOWN
        assert prediction.uncertainty.basis
        assert math.isfinite(prediction.uncertainty.std)


def test_registry_now_covers_polymers(registry):
    coverage = registry.coverage(
        ["glass_transition_temperature", "amorphous_density"], MaterialClass.POLYMER
    )
    assert coverage["glass_transition_temperature"] == ["polymer_tg"]
    assert coverage["amorphous_density"] == ["polymer_density"]
    # Young's modulus is covered now too, by the mechanical expert, which reads
    # the transition this expert predicts to decide whether the polymer is a
    # glass or a rubber at the stated temperature.
    assert registry.coverage(["youngs_modulus"], MaterialClass.POLYMER)["youngs_modulus"] == [
        "polymer_mechanical"
    ]
    # What stays uncovered is what cannot be predicted from a structure: a
    # tensile strength is set by the largest flaw in the specimen.
    assert "tensile_strength" not in PROPERTY_REGISTRY


def test_cached_volume_matches_a_live_computation():
    stored = cached_vdw_volume(PE)
    # The stored values are rounded for legibility, not recomputed at import.
    assert stored == pytest.approx(repeat_unit_vdw_volume(PE), abs=0.01)
