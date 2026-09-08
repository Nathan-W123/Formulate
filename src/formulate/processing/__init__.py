"""Process calculations: what happens to a material when it is made into something.

Distinct from :mod:`formulate.experts`, which predicts properties of a material,
and from :mod:`formulate.physics`, which computes them from first principles.
Nothing here is a property: a jet's fate depends on the nozzle and the velocity
as much as on the liquid, so these are not things a candidate *has* and they do
not enter the property registry or the ranking.
"""

from .spinning import (
    MARK_HOUWINK,
    JetRegime,
    SpinningAssessment,
    SpinningConditions,
    assess_jet,
    deborah_number,
    ohnesorge_number,
    overlap_concentration,
    rayleigh_time,
    zimm_relaxation_time,
)

__all__ = [
    "MARK_HOUWINK", "JetRegime", "SpinningAssessment", "SpinningConditions",
    "assess_jet", "deborah_number", "ohnesorge_number", "overlap_concentration",
    "rayleigh_time", "zimm_relaxation_time",
]
