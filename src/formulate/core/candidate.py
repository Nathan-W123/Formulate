"""The canonical candidate object.

Specification section 2: "Every module reads/writes one canonical candidate
object.  Representation is class-dependent but exposes a common interface."

The class-dependent part is the payload (:class:`MoleculeSpec`,
:class:`PolymerSpec`, :class:`MixtureSpec`); the common interface is
:class:`Candidate` itself - identity, provenance, conditions and results.

Identity is content-addressed over *structure and conditions only*.  Results
are excluded deliberately: evaluating a candidate must not change its id, or
the cache described in section 11 could never hit.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conditions import Conditions
from .errors import CandidateError
from .hashing import content_hash
from .prediction import Prediction
from .provenance import ProvenanceRecord
from .quantity import Quantity

#: Tolerance on composition fractions summing to unity.
FRACTION_TOLERANCE = 1e-6


class MaterialClass(str, Enum):
    """Material classes the pipeline maintains (specification section 3)."""

    MOLECULE = "molecule"
    POLYMER = "polymer"
    MIXTURE = "mixture"
    COMPOSITE = "composite"


# --------------------------------------------------------------------------
# Class-dependent payloads
# --------------------------------------------------------------------------


class MoleculeSpec(BaseModel):
    """A single small molecule."""

    model_config = ConfigDict(frozen=True)

    smiles: str
    #: Net formal charge; kept explicit because many predictors are neutral-only.
    charge: int = 0
    #: True when stereochemistry is fully specified in ``smiles``.
    stereo_defined: bool = False
    #: Number of 3D conformers generated for this molecule, if any.
    conformer_count: int = 0

    @model_validator(mode="after")
    def _non_empty(self) -> "MoleculeSpec":
        if not self.smiles.strip():
            raise ValueError("MoleculeSpec.smiles must not be empty.")
        return self

    def canonical_key(self) -> str:
        """Toolkit-canonical structure string used for content addressing.

        Falls back to the raw SMILES when RDKit is unavailable, so identity
        stays defined (if coarser) in a bare environment.
        """
        from formulate import chem

        if chem.rdkit_available():
            canon = chem.canonical_smiles(self.smiles)
            if canon is not None:
                return canon
        return self.smiles.strip()

    def identity_payload(self) -> dict[str, Any]:
        return {"smiles": self.canonical_key(), "charge": self.charge}

    def all_smiles(self) -> list[str]:
        return [self.smiles]


class PolymerTopology(str, Enum):
    LINEAR = "linear"
    BRANCHED = "branched"
    STAR = "star"
    GRAFT = "graft"
    NETWORK = "network"
    DENDRITIC = "dendritic"


class Tacticity(str, Enum):
    ATACTIC = "atactic"
    ISOTACTIC = "isotactic"
    SYNDIOTACTIC = "syndiotactic"
    UNSPECIFIED = "unspecified"


class MonomerRole(str, Enum):
    BACKBONE = "backbone"
    COMONOMER = "comonomer"
    CROSSLINKER = "crosslinker"
    END_GROUP = "end_group"


class MonomerUnit(BaseModel):
    """One repeat unit in a polymer.

    ``smiles`` should carry attachment points (``[*]``) marking where the unit
    joins the chain.
    """

    model_config = ConfigDict(frozen=True)

    smiles: str
    mole_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    role: MonomerRole = MonomerRole.BACKBONE

    def canonical_key(self) -> str:
        from formulate import chem

        if chem.rdkit_available():
            canon = chem.canonical_smiles(self.smiles)
            if canon is not None:
                return canon
        return self.smiles.strip()


class PolymerSpec(BaseModel):
    """A polymer: repeat units, architecture and chain statistics.

    Section 13 warns against validating bulk polymer behaviour from an isolated
    molecule, so chain-length and architecture fields are part of identity
    rather than optional metadata.
    """

    model_config = ConfigDict(frozen=True)

    monomers: tuple[MonomerUnit, ...]
    topology: PolymerTopology = PolymerTopology.LINEAR
    tacticity: Tacticity = Tacticity.UNSPECIFIED
    #: Target number-average molar mass.
    number_average_molar_mass: Quantity | None = None
    #: Mw/Mn.
    dispersity: float | None = Field(default=None, ge=1.0)
    #: Crosslink density, only meaningful for networks.
    crosslink_density: Quantity | None = None
    end_groups: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> "PolymerSpec":
        if not self.monomers:
            raise ValueError("PolymerSpec requires at least one monomer.")
        chain = [m for m in self.monomers if m.role is not MonomerRole.END_GROUP]
        if not chain:
            raise ValueError("PolymerSpec requires at least one non-end-group monomer.")
        total = sum(m.mole_fraction for m in chain)
        if not math.isclose(total, 1.0, abs_tol=FRACTION_TOLERANCE):
            raise ValueError(
                f"Polymer monomer mole fractions must sum to 1.0, got {total:.9g}."
            )
        if self.topology is PolymerTopology.NETWORK and self.crosslink_density is None:
            # Not fatal, but a network without a crosslink density is under-specified
            # for any bulk mechanical prediction.
            pass
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "monomers": sorted(
                (
                    {"smiles": m.canonical_key(), "f": m.mole_fraction, "role": m.role.value}
                    for m in self.monomers
                ),
                key=lambda d: (d["role"], d["smiles"]),
            ),
            "topology": self.topology.value,
            "tacticity": self.tacticity.value,
            "mn": None
            if self.number_average_molar_mass is None
            else self.number_average_molar_mass.to("g/mol").value,
            "dispersity": self.dispersity,
            "crosslink_density": None
            if self.crosslink_density is None
            else self.crosslink_density.to_canonical().value,
            "end_groups": sorted(self.end_groups),
        }

    def all_smiles(self) -> list[str]:
        return [m.smiles for m in self.monomers]


class ComponentRole(str, Enum):
    """Chemically meaningful roles in a formulation (specification section 3)."""

    SOLVENT = "solvent"
    CO_SOLVENT = "co_solvent"
    SOLUTE = "solute"
    BINDER = "binder"
    MATRIX = "matrix"
    FILLER = "filler"
    PLASTICIZER = "plasticizer"
    SURFACTANT = "surfactant"
    CROSSLINKER = "crosslinker"
    CATALYST = "catalyst"
    PIGMENT = "pigment"
    STABILIZER = "stabilizer"
    ADDITIVE = "additive"


class FractionBasis(str, Enum):
    MOLE = "mole"
    MASS = "mass"
    VOLUME = "volume"


class PhaseAssumption(str, Enum):
    SINGLE_PHASE = "single_phase"
    EMULSION = "emulsion"
    DISPERSION = "dispersion"
    SUSPENSION = "suspension"
    UNKNOWN = "unknown"


class MixtureComponent(BaseModel):
    """One component of a formulation, itself a molecule or a polymer."""

    model_config = ConfigDict(frozen=True)

    role: ComponentRole = ComponentRole.ADDITIVE
    fraction: float = Field(ge=0.0, le=1.0)
    molecule: MoleculeSpec | None = None
    polymer: PolymerSpec | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "MixtureComponent":
        if (self.molecule is None) == (self.polymer is None):
            raise ValueError(
                "MixtureComponent must carry exactly one of 'molecule' or 'polymer'."
            )
        return self

    @property
    def payload(self) -> MoleculeSpec | PolymerSpec:
        return self.molecule if self.molecule is not None else self.polymer  # type: ignore[return-value]

    def canonical_key(self) -> str:
        return content_hash(self.payload.identity_payload())

    def all_smiles(self) -> list[str]:
        return self.payload.all_smiles()


class MixtureSpec(BaseModel):
    """A multicomponent formulation."""

    model_config = ConfigDict(frozen=True)

    components: tuple[MixtureComponent, ...]
    basis: FractionBasis = FractionBasis.MASS
    phase_assumption: PhaseAssumption = PhaseAssumption.UNKNOWN

    @model_validator(mode="after")
    def _validate(self) -> "MixtureSpec":
        if not self.components:
            raise ValueError("MixtureSpec requires at least one component.")
        total = sum(c.fraction for c in self.components)
        if not math.isclose(total, 1.0, abs_tol=FRACTION_TOLERANCE):
            raise ValueError(
                f"Mixture component fractions must sum to 1.0 on a {self.basis.value} "
                f"basis, got {total:.9g}."
            )
        keys = [c.canonical_key() for c in self.components]
        if len(set(keys)) != len(keys):
            raise ValueError("Mixture contains duplicate components; merge their fractions.")
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "basis": self.basis.value,
            "phase": self.phase_assumption.value,
            "components": sorted(
                (
                    {
                        "key": c.canonical_key(),
                        "role": c.role.value,
                        "f": c.fraction,
                    }
                    for c in self.components
                ),
                key=lambda d: (d["key"], d["role"]),
            ),
        }

    def all_smiles(self) -> list[str]:
        out: list[str] = []
        for c in self.components:
            out.extend(c.all_smiles())
        return out

    def component_of_role(self, role: ComponentRole) -> list[MixtureComponent]:
        return [c for c in self.components if c.role is role]


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


class ConstraintViolation(BaseModel):
    """A breached requirement, recorded rather than folded into a score."""

    model_config = ConfigDict(frozen=True)

    property: str
    #: Human-readable statement of what was required.
    requirement: str
    #: Signed magnitude of the breach in the property's canonical unit.
    margin: float | None = None
    hard: bool = True
    reason: str = ""

    def __str__(self) -> str:
        kind = "hard" if self.hard else "soft"
        margin = "" if self.margin is None else f" by {abs(self.margin):.4g}"
        return f"[{kind}] {self.property}: {self.requirement}{margin}. {self.reason}".strip()


class CandidateResults(BaseModel):
    """Everything computed about a candidate.

    Specification section 2: "expert predictions, uncertainty,
    applicability-domain flags, simulation results, objective vector,
    constraint violations, aggregate rank".
    """

    model_config = ConfigDict(frozen=True)

    predictions: tuple[Prediction, ...] = ()
    #: Dimensionless utilities keyed by property name, in [0, 1].
    objective_vector: dict[str, float] = Field(default_factory=dict)
    #: Properties a requirement asked for but no expert could supply.
    missing_properties: tuple[str, ...] = ()
    constraint_violations: tuple[ConstraintViolation, ...] = ()
    feasible: bool = True
    #: Weighted-sum baseline only; never the sole basis for a recommendation.
    scalar_score: float | None = None
    #: 0 for the non-dominated front, 1 for the next, and so on.
    pareto_front: int | None = None
    aggregate_rank: int | None = None
    #: Present once physics validation has run (Phase 3).
    simulation_ids: tuple[str, ...] = ()

    def prediction_for(self, prop: str) -> Prediction | None:
        """The usable prediction for ``prop``, preferring in-domain results."""
        matches = [p for p in self.predictions if p.property == prop and p.is_usable]
        if not matches:
            return None
        matches.sort(key=lambda p: (not p.applicability.in_domain, -p.applicability.score))
        return matches[0]

    def predictions_for(self, prop: str) -> list[Prediction]:
        return [p for p in self.predictions if p.property == prop]

    @property
    def hard_violations(self) -> tuple[ConstraintViolation, ...]:
        return tuple(v for v in self.constraint_violations if v.hard)

    @property
    def out_of_domain_properties(self) -> tuple[str, ...]:
        return tuple(
            sorted({p.property for p in self.predictions
                    if p.is_usable and not p.applicability.in_domain})
        )


# --------------------------------------------------------------------------
# The candidate
# --------------------------------------------------------------------------


class Candidate(BaseModel):
    """A proposed material, in any supported class."""

    model_config = ConfigDict(frozen=True)

    material_class: MaterialClass
    molecule: MoleculeSpec | None = None
    polymer: PolymerSpec | None = None
    mixture: MixtureSpec | None = None

    #: Conditions this candidate is intended to operate at.
    conditions: Conditions = Field(default_factory=Conditions)
    #: How this candidate was produced, e.g. "database:pubchem", "evolutionary:mutate".
    generation_strategy: str = "unspecified"
    #: Candidate ids this was derived from.
    parent_ids: tuple[str, ...] = ()
    provenance: ProvenanceRecord | None = None
    label: str = ""
    results: CandidateResults | None = None

    @model_validator(mode="after")
    def _payload_matches_class(self) -> "Candidate":
        expected = {
            MaterialClass.MOLECULE: "molecule",
            MaterialClass.POLYMER: "polymer",
            MaterialClass.MIXTURE: "mixture",
            MaterialClass.COMPOSITE: "mixture",
        }[self.material_class]
        present = [n for n in ("molecule", "polymer", "mixture") if getattr(self, n) is not None]
        if present != [expected]:
            raise CandidateError(
                f"Candidate of class {self.material_class.value} must carry exactly the "
                f"{expected!r} payload, but carries {present or ['nothing']}."
            )
        return self

    # -- identity ----------------------------------------------------------

    @property
    def payload(self) -> MoleculeSpec | PolymerSpec | MixtureSpec:
        for name in ("molecule", "polymer", "mixture"):
            value = getattr(self, name)
            if value is not None:
                return value
        raise CandidateError("Candidate has no payload.")  # pragma: no cover

    def identity_payload(self) -> dict[str, Any]:
        """Structure and conditions - the fields identity is defined over."""
        return {
            "class": self.material_class.value,
            "payload": self.payload.identity_payload(),
            "conditions": self.conditions.model_dump(mode="json"),
        }

    @property
    def candidate_id(self) -> str:
        """Content-addressed identity.

        Excludes results, provenance and label so that the same structure at
        the same conditions always has the same id, whoever proposed it.
        """
        return content_hash(self.identity_payload(), prefix="cand")

    @property
    def structure_id(self) -> str:
        """Identity of the structure alone, ignoring conditions."""
        return content_hash(
            {"class": self.material_class.value, "payload": self.payload.identity_payload()},
            prefix="struct",
        )

    # -- convenience -------------------------------------------------------

    def all_smiles(self) -> list[str]:
        """Every SMILES string in this candidate, for filters and similarity."""
        return self.payload.all_smiles()

    @property
    def primary_smiles(self) -> str | None:
        """The single structure most representative of this candidate.

        For a mixture this is the highest-fraction component, which is a
        convenience for similarity and display only - section 13 forbids
        treating it as a stand-in for bulk behaviour.
        """
        if self.molecule is not None:
            return self.molecule.smiles
        if self.polymer is not None:
            chain = [m for m in self.polymer.monomers if m.role is not MonomerRole.END_GROUP]
            return max(chain, key=lambda m: m.mole_fraction).smiles if chain else None
        if self.mixture is not None and self.mixture.components:
            top = max(self.mixture.components, key=lambda c: c.fraction)
            smis = top.all_smiles()
            return smis[0] if smis else None
        return None

    def with_results(self, results: CandidateResults) -> "Candidate":
        """Return a copy carrying ``results``; identity is unchanged."""
        return self.model_copy(update={"results": results})

    def with_conditions(self, conditions: Conditions) -> "Candidate":
        """Return a copy at different conditions.

        This *does* change identity, because a property predicted at 298 K is
        not a property predicted at 423 K.
        """
        return self.model_copy(update={"conditions": conditions, "results": None})

    def describe(self) -> str:
        name = self.label or self.primary_smiles or self.material_class.value
        return f"{name} [{self.material_class.value}] ({self.conditions.describe()})"

    def __str__(self) -> str:
        return self.describe()


def molecule_candidate(smiles: str, **kwargs: Any) -> Candidate:
    """Convenience constructor for a single-molecule candidate."""
    return Candidate(
        material_class=MaterialClass.MOLECULE, molecule=MoleculeSpec(smiles=smiles), **kwargs
    )


def dedupe(candidates: Iterator[Candidate] | list[Candidate]) -> list[Candidate]:
    """Collapse candidates sharing a content id, preserving first-seen order.

    Section 3 requires the parallel search strategies to "merge into one
    deduplicated candidate pool".
    """
    seen: dict[str, Candidate] = {}
    for cand in candidates:
        seen.setdefault(cand.candidate_id, cand)
    return list(seen.values())
