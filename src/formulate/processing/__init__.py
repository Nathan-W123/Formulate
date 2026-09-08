"""Process calculations: what happens to a material when it is made into something.

Distinct from :mod:`formulate.experts`, which predicts properties of a material,
and from :mod:`formulate.physics`, which computes them from first principles.
Nothing here is a property: a jet's fate depends on the nozzle and the velocity
as much as on the liquid, so these are not things a candidate *has* and they do
not enter the property registry or the ranking.
"""

from .curing import (
    POLYMERISATION,
    CureAssessment,
    CureConditions,
    CureVerdict,
    adiabatic_temperature_rise,
    assess_cure,
    impact_energy,
    jet_thrust,
    thermal_time,
)
from .drying import (
    DryingAssessment,
    DryingConditions,
    DryingRegime,
    assess_drying,
    fuller_diffusion_volume,
    fuller_diffusivity,
)
from .spinning import (
    MARK_HOUWINK,
    JetRegime,
    SpinningAssessment,
    SpinningConditions,
    assess_jet,
    breakup_length,
    deborah_number,
    extrusion_pressure,
    reynolds_number,
    weber_number,
    ohnesorge_number,
    overlap_concentration,
    rayleigh_time,
    zimm_relaxation_time,
)

__all__ = [
    "POLYMERISATION", "CureAssessment", "CureConditions", "CureVerdict",
    "adiabatic_temperature_rise", "assess_cure", "impact_energy", "jet_thrust",
    "thermal_time",
    "DryingAssessment", "DryingConditions", "DryingRegime", "MARK_HOUWINK", "JetRegime", "SpinningAssessment", "SpinningConditions",
    "assess_drying", "assess_jet", "breakup_length",
    "extrusion_pressure", "reynolds_number", "weber_number", "deborah_number", "ohnesorge_number", "overlap_concentration",
    "fuller_diffusion_volume", "fuller_diffusivity", "rayleigh_time",
    "zimm_relaxation_time",
]
