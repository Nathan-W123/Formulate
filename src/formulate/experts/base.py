"""The expert interface.

Specification section 4: "Experts do not independently choose their favorite
materials.  All applicable experts score the same candidates so trade-offs are
comparable."

The interface is model-agnostic by design.  A subclass may wrap a graph neural
network, a 3D equivariant model, a descriptor regression, an empirical
correlation, or a deterministic physics calculation; the coordinator cannot
tell and must not need to.

Experts may depend on properties produced by other experts (a surface-tension
correlation needs a critical temperature).  Those dependencies are declared,
not resolved by reaching into another expert, so the engine can order the
dispatch and so uncertainty propagates through a visible path.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from formulate.core.candidate import Candidate, MaterialClass
from formulate.core.conditions import Conditions
from formulate.core.prediction import Prediction, PredictionStatus
from formulate.core.properties import PropertyFamily, get_property
from formulate.core.provenance import ProvenanceKind, ProvenanceRecord, SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, Quantity, Uncertainty, UncertaintyKind


@dataclass(frozen=True, slots=True)
class PredictionRequest:
    """What an expert is asked to do.

    ``context`` holds predictions already produced for this candidate by
    upstream experts, keyed by property name.  An expert must treat a missing
    dependency as a reason to return UNSUPPORTED, never as a reason to guess.
    """

    candidate: Candidate
    properties: frozenset[str]
    conditions: Conditions
    context: Mapping[str, Prediction] = field(default_factory=lambda: MappingProxyType({}))

    def dependency(self, prop: str) -> Prediction | None:
        """A usable upstream prediction for ``prop``, if one exists."""
        pred = self.context.get(prop)
        return pred if pred is not None and pred.is_usable else None

    def dependency_value(self, prop: str, unit: str) -> float | None:
        """Value of an upstream prediction converted to ``unit``."""
        pred = self.dependency(prop)
        if pred is None or pred.quantity is None:
            return None
        return pred.quantity.to(unit).value


class Expert(abc.ABC):
    """Base class for every property predictor.

    Subclasses declare what they cover and implement :meth:`_predict_one` or
    override :meth:`predict` wholesale.
    """

    #: Stable identifier used in provenance and cache keys.
    id: str = "expert"
    #: Bump whenever the numerical behaviour changes; it is part of the cache key.
    version: str = "0"
    #: Human-readable description of the underlying method and its citation.
    method: str = ""
    family: PropertyFamily = PropertyFamily.CHEMICAL
    #: Properties this expert can produce.
    supported_properties: frozenset[str] = frozenset()
    #: Material classes this expert is valid for.
    supported_classes: frozenset[MaterialClass] = frozenset({MaterialClass.MOLECULE})
    #: Properties this expert needs from other experts before it can run.
    dependencies: frozenset[str] = frozenset()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        for prop in cls.supported_properties:
            get_property(prop)  # fail loudly at import time on a typo

    # -- capability --------------------------------------------------------

    def covers(self, prop: str, material_class: MaterialClass) -> bool:
        return prop in self.supported_properties and material_class in self.supported_classes

    def applicable_properties(
        self, requested: Sequence[str] | frozenset[str], material_class: MaterialClass
    ) -> frozenset[str]:
        """Intersection of what was asked for and what this expert covers."""
        if material_class not in self.supported_classes:
            return frozenset()
        return frozenset(requested) & self.supported_properties

    def is_available(self) -> bool:
        """False when a required backend is missing.

        An unavailable expert is skipped with a recorded reason rather than
        crashing the run.
        """
        return True

    def unavailable_reason(self) -> str:
        return ""

    # -- domain ------------------------------------------------------------

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        """How well this candidate sits inside the model's training/validity region.

        The default is fully in-domain; every real expert should override this,
        because section 12 measures the system on its ability to identify
        out-of-domain predictions.
        """
        return ApplicabilityDomain(basis=f"{self.id} declares no domain model")

    # -- prediction --------------------------------------------------------

    def predict(self, request: PredictionRequest) -> list[Prediction]:
        """Predict every requested property this expert covers."""
        props = self.applicable_properties(request.properties, request.candidate.material_class)
        if not props:
            return []
        if not self.is_available():
            reason = self.unavailable_reason() or f"{self.id} backend unavailable"
            return [Prediction.unsupported(p, self.id, reason) for p in sorted(props)]

        domain = self.assess_domain(request.candidate)
        out: list[Prediction] = []
        for prop in sorted(props):
            try:
                pred = self._predict_one(prop, request, domain)
            except Exception as exc:  # an expert failing must not kill the run
                pred = Prediction.failed(prop, self.id, f"{type(exc).__name__}: {exc}")
            if pred is not None:
                out.append(pred)
        return out

    @abc.abstractmethod
    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        """Produce one prediction, or ``None`` to omit it entirely."""

    # -- helpers -----------------------------------------------------------

    def _software(self) -> SoftwareEnvironment:
        return SoftwareEnvironment.capture()

    def _provenance(self, request: PredictionRequest, prop: str, **params: object) -> ProvenanceRecord:
        inputs = [request.candidate.candidate_id]
        inputs.extend(
            sorted(
                p.provenance.record_id
                for name, p in request.context.items()
                if name in self.dependencies and p.provenance is not None
            )
        )
        return ProvenanceRecord(
            kind=ProvenanceKind.PREDICTION,
            producer=self.id,
            producer_version=self.version,
            input_ids=tuple(inputs),
            parameters={"property": prop, "method": self.method, **params},
            software=self._software(),
        )

    def _make(
        self,
        prop: str,
        value: float,
        unit: str,
        request: PredictionRequest,
        domain: ApplicabilityDomain,
        *,
        std: float | None = None,
        kind: UncertaintyKind = UncertaintyKind.EPISTEMIC,
        basis: str = "",
        conditions: Conditions | None = None,
        notes: Sequence[str] = (),
        **params: object,
    ) -> Prediction:
        """Assemble a complete prediction record."""
        prop_def = get_property(prop)
        lo, hi = prop_def.bounds
        note_list = list(notes)
        status = PredictionStatus.OK if domain.in_domain else PredictionStatus.OUT_OF_DOMAIN

        # A value outside the physically admissible range is a model failure,
        # not a low score: report it rather than ranking on nonsense.
        canonical_value = Quantity(value=value, unit=unit).to_canonical().value
        if (lo is not None and canonical_value < lo) or (hi is not None and canonical_value > hi):
            return Prediction.failed(
                prop,
                self.id,
                f"predicted {canonical_value:.6g} {prop_def.canonical_unit} is outside the "
                f"physically admissible range {lo} to {hi}",
            )

        return Prediction(
            property=prop,
            quantity=Quantity(value=value, unit=unit),
            uncertainty=Uncertainty(std=std, kind=kind, basis=basis)
            if std is not None
            else Uncertainty.unknown(basis or f"{self.id} reports no uncertainty for {prop}"),
            applicability=domain,
            status=status,
            expert_id=self.id,
            expert_version=self.version,
            method=self.method,
            conditions=conditions if conditions is not None else request.conditions,
            provenance=self._provenance(request, prop, **params),
            notes=tuple(note_list),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} id={self.id!r} v{self.version}>"
