"""Whether a polymer draws or snaps, and the architecture that decides it.

A cured 1,6-hexanediol diacrylate network was the correct answer to five
successive specifications and the wrong material to build.  Nothing in those
specifications was wrong; what was missing was that the properties separating a
tough thermoplastic from a brittle thermoset were not in the registry, so a
brittle thermoset could not lose a single point for being one.  This module
supplies two of the three that were missing - an elongation at break and a
crosslink density - and reads the measured column of the polymer catalogue that
nothing else was reading.

**Elongation at break is the toughness proxy, and it is a proxy.**  Work to
fracture is the integral of stress over strain, and the strength half of that
integral varies between materials by a factor of five while the strain half
varies by a factor of a thousand.  Polystyrene and polyamide 6 differ by two in
tensile strength and by fifty in elongation; a bar of each absorbs energies that
differ by a factor of twenty-five, and the whole of that factor is in the
elongation.  So a single number that separates the two is worth having even
though it is not the work to fracture itself.

Three experts, and the interesting one is the one that mostly fails.

``MeasuredPolymerExpert``
    The measured column of ``data/polymers.json``: glass transition, amorphous
    density and elongation at break where a compilation gives them, matched on
    the repeat units and the architecture rather than on a name.  Elongations
    are stored as ``[low, high]`` ranges because that is how every source
    quotes them, and the range is not a formatting choice - it is the
    measurement.  Over the sixteen ranges in the catalogue the high end is a
    geometric mean of 3.89 times the low end, so a quoted elongation is worth
    about a factor of two either way, and the prediction says so.

``ChainToughnessExpert``
    The structural proxy, for the polymers the catalogue has no elongation for.
    A covalent network is brittle; a chain well below its glass transition at
    the stated temperature is a rubber; one above it is a glass.  Calibrated
    leave-one-out against the measured column, and the result is that it
    separates rubbers well and glasses hardly at all - a factor of 1.40 within
    the rubbery class and 6.64 within the glassy one, on an overall hit rate of
    seven in eleven.  The reason is stated where the constants are, and the
    consequence is that ``prefer`` chooses the measurement wherever one exists,
    which is the correct outcome and is arrived at by comparing spreads rather
    than by privileging anything.

``PolymerArchitectureExpert``
    The crosslink density, read off the specification exactly.  Not a model: a
    linear or branched chain has no crosslinks by definition, a stated network
    has whatever it states, and a network that states nothing is refused rather
    than assumed.  It is here because "must be remeltable" is a requirement a
    hot-melt specification has to be able to state, and until this existed it
    could only be stated as prose in a ``notes`` block that nothing reads.

One thing this module deliberately does not do.  It does not turn the
catalogue's tensile strengths into a ``tensile_strength`` prediction, because
:mod:`formulate.experts.mechanical` argues at length that there should be no
such property - a real strength is set by the largest flaw in the specimen,
which is a property of how it was made.  The tensile column is used instead to
measure what the ``theoretical_strength`` bound is worth, in
:func:`strength_utilisation`, which is the question that bound was introduced
to answer and had until now no data to answer it with.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Mapping, Sequence

from formulate.core.candidate import (
    Candidate,
    MaterialClass,
    MonomerRole,
    PolymerSpec,
    PolymerTopology,
)
from formulate.core.prediction import Prediction
from formulate.core.properties import PropertyFamily
from formulate.core.quantity import ApplicabilityDomain, UncertaintyKind

from .base import Expert, PredictionRequest

#: How far above the stated temperature a glass transition has to sit before
#: the material is a glass rather than a leathery solid.  Fifteen kelvin is the
#: width of the transition itself in a conventional scan, so inside it the
#: material is neither.
_GLASSY_MARGIN_K = 15.0

#: How far above the stated temperature the proxy stops claiming to know
#: anything.  A polymer a hundred kelvin above its transition and a polymer
#: three hundred kelvin above it are both glasses at room temperature and the
#: catalogue contains none of the second kind, so the class has nothing behind
#: it there and the expert refuses rather than extrapolating a mean over
#: materials it has never seen.
_GLASSY_CEILING_K = 100.0


# --------------------------------------------------------------------------
# The measured column
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PolymerRecord:
    """One row of the polymer catalogue, with its measured columns."""

    name: str
    abbreviation: str
    #: Canonical repeat units with their mole fractions, sorted for matching.
    signature: tuple[tuple[str, float], ...]
    topology: str
    tacticity: str
    glass_transition_k: float | None
    amorphous_density_g_cm3: float | None
    #: ``(low, high)`` as the source quotes it; the reported value is the
    #: geometric mean, which is the right centre for a quantity whose error is
    #: multiplicative.
    elongation_at_break: tuple[float, float] | None
    tensile_strength_mpa: tuple[float, float] | None
    crosslink_density_mol_m3: float | None
    source: str
    note: str

    @property
    def is_network(self) -> bool:
        return self.topology in ("network", "dendritic") or bool(self.crosslink_density_mol_m3)

    @property
    def elongation(self) -> float | None:
        if self.elongation_at_break is None:
            return None
        low, high = self.elongation_at_break
        return math.sqrt(low * high)

    @property
    def tensile_pa(self) -> float | None:
        if self.tensile_strength_mpa is None:
            return None
        low, high = self.tensile_strength_mpa
        return math.sqrt(low * high) * 1.0e6


def _canonical(smiles: str) -> str:
    from formulate import chem

    if not chem.rdkit_available():
        return smiles.strip()
    return chem.canonical_smiles(smiles) or smiles.strip()


def _signature(
    units: Sequence[tuple[str, float]], topology: str, tacticity: str
) -> tuple[tuple[tuple[str, float], ...], str, str]:
    """The key a catalogue row and a candidate are matched on.

    Structure alone is not enough and the catalogue proves it: the two
    polyethylenes share a repeat unit and differ by a factor of two in
    strength, which is chain branching and is carried in the topology.  Nor is
    the name enough, because a candidate reaching this expert from anywhere but
    the bundled explorer has no name at all.  So the key is the structure, the
    proportions, the architecture and the tacticity - everything identity is
    defined over that a bulk property depends on.
    """
    canonical = tuple(sorted((_canonical(s), round(f, 4)) for s, f in units))
    return canonical, topology, tacticity


@lru_cache(maxsize=1)
def polymer_records() -> tuple[PolymerRecord, ...]:
    """The catalogue rows, parsed once."""
    text = (
        resources.files("formulate.data").joinpath("polymers.json").read_text(encoding="utf-8")
    )
    raw: list[dict[str, Any]] = json.loads(text)["polymers"]
    out: list[PolymerRecord] = []
    for record in raw:
        units = tuple(
            (unit["smiles"], float(unit["mole_fraction"])) for unit in record["repeat_units"]
        )
        signature, _, _ = _signature(units, "", "")
        elongation = record.get("elongation_at_break")
        tensile = record.get("tensile_strength_mpa")
        out.append(
            PolymerRecord(
                name=record["name"],
                abbreviation=record["abbreviation"],
                signature=signature,
                topology=record.get("topology", "linear"),
                tacticity=record.get("tacticity", "unspecified"),
                glass_transition_k=record.get("glass_transition_k"),
                amorphous_density_g_cm3=record.get("amorphous_density_g_cm3"),
                elongation_at_break=None if elongation is None else tuple(elongation),
                tensile_strength_mpa=None if tensile is None else tuple(tensile),
                crosslink_density_mol_m3=record.get("crosslink_density_mol_m3"),
                source=record.get("source", ""),
                note=record.get("note", ""),
            )
        )
    return tuple(out)


@lru_cache(maxsize=1)
def _by_signature() -> Mapping[tuple[tuple[tuple[str, float], ...], str, str], PolymerRecord]:
    return {
        _signature(
            [(s, f) for s, f in record.signature], record.topology, record.tacticity
        ): record
        for record in polymer_records()
    }


def catalogue_record(spec: PolymerSpec) -> PolymerRecord | None:
    """The catalogue row this specification is, or ``None`` if it is not one."""
    backbone = [m for m in spec.monomers if m.role is not MonomerRole.END_GROUP]
    if not backbone:
        return None
    units = [(m.smiles, m.mole_fraction) for m in backbone]
    key = _signature(units, spec.topology.value, spec.tacticity.value)
    return _by_signature().get(key)


#: Spread on a tabulated glass transition, K.
#:
#: Not measured from the two tables in this repository, and the reason is worth
#: recording.  ``polymers.json`` and ``reference_polymers.json`` agree to 1.4 K
#: on average over the twelve polymers they share, and eleven of those twelve
#: agree exactly.  That measures that both were compiled from the same handbook
#: tradition, not that a glass transition is known to a kelvin.  The honest
#: figure is the one ``reference_polymers.json`` states in its own caveats: a
#: reported transition is a kinetic event that depends on cooling rate and on
#: whether it was measured by DSC, DMA or dilatometry, and compilations differ
#: by 5-10 K on the same polymer.
#:
#: The one disagreement above that in the shared set is polypropylene, 267 K
#: here against 253 K there, and it is not a compilation difference at all -
#: it is isotactic against atactic, which is a different material.
_GLASS_TRANSITION_SPREAD_K = 7.0

#: Spread on a tabulated amorphous density, g/cm^3.  The largest disagreement
#: between the two tables over their shared polymers, which is 0.02 on
#: polyamide 12 and on poly(butylene terephthalate); the mean is 0.0035.
_AMORPHOUS_DENSITY_SPREAD = 0.02


@lru_cache(maxsize=1)
def elongation_source_spread() -> float:
    """What a tabulated elongation is worth, as a factor.

    Every source quotes an elongation at break as a range, and the range is the
    measurement rather than a rounding of it: an elongation is a property of a
    specimen, and two bars of one polymer differ on mould temperature, cooling
    rate and how the gate was placed.  The geometric mean of ``high / low``
    across the catalogue is the width of that range, and half of it in the log
    is the one-sigma spread on the geometric centre.
    """
    widths = [
        math.log(record.elongation_at_break[1] / record.elongation_at_break[0])
        for record in polymer_records()
        if record.elongation_at_break is not None
    ]
    return math.exp(statistics.fmean(widths) / 2.0)


# --------------------------------------------------------------------------
# The structural proxy
# --------------------------------------------------------------------------


def classify(
    *,
    is_network: bool,
    glass_transition_k: float | None,
    temperature_k: float,
) -> str | None:
    """Which class a chain falls in at a stated temperature, or ``None``.

    Three classes and a refusal.  A covalent network is a network whatever its
    transition is; below that, everything turns on how far the temperature sits
    from the glass transition.
    """
    if is_network:
        return "network"
    if glass_transition_k is None:
        return None
    if glass_transition_k < temperature_k + _GLASSY_MARGIN_K:
        return "rubbery"
    if glass_transition_k <= temperature_k + _GLASSY_CEILING_K:
        return "glassy"
    return None


@dataclass(frozen=True, slots=True)
class ToughnessProxy:
    """A class-mean elongation and the leave-one-out error it earns."""

    #: Class -> (geometric-mean elongation, one-sigma spread in ln, members).
    #: The spread is kept in the log because that is the scale the property's
    #: error lives on; ``exp`` of it is the factor a prediction is worth.
    classes: Mapping[str, tuple[float, float, int]]
    #: Leave-one-out hits over the classes with more than one member.
    hits: int
    trials: int
    #: Geometric mean of predicted/measured over those trials.
    typical_ratio: float
    #: The worst leave-one-out ratio, and which polymer produced it.
    worst_ratio: float
    worst_polymer: str

    @property
    def hit_rate(self) -> float:
        return self.hits / self.trials if self.trials else 0.0


@lru_cache(maxsize=1)
def toughness_proxy(temperature_k: float = 298.15) -> ToughnessProxy:
    """Fit the class means against the catalogue and measure what they are worth.

    Fitted at the temperature the catalogue's own mechanical figures were
    measured at, which is ambient: an elongation at break quoted for a moulded
    bar is quoted at room temperature, so a class boundary drawn against any
    other temperature would be drawn against data that does not exist.  The
    expert calls this once and then applies the classes at whatever temperature
    it was asked about, which is an extrapolation and is stated as one.

    The result, and it is not flattering.  The rubbery class holds together:
    five polymers whose elongations span two to five, reproduced leave-one-out
    to a factor of 1.40.  The glassy class does not: six polymers spanning
    0.017 to 1.22, a seventy-fold range, reproduced to a factor of 6.64.

    That failure is informative rather than a defect to be tuned away.  What
    separates polystyrene at two per cent from polyamide 6 at seventy-seven is
    not how far either sits above its transition - they are 75 and 25 K above
    it and the ductile one is the closer - it is whether the glass shear-yields
    before it crazes, which is set by the entanglement density of the chains in
    the glass.  :mod:`formulate.experts.mechanical` computes exactly that
    quantity from the packing length, and holds the chain dimension it needs
    for nine repeat units, none of them a polyamide or a polyester.  So the
    better proxy is one measurement per polymer away and is not reachable from
    the structure, which is why this one refuses to claim more than a factor of
    six and why every measured elongation outranks it.
    """
    members: dict[str, list[tuple[str, float]]] = {}
    for record in polymer_records():
        value = record.elongation
        if value is None:
            continue
        name = classify(
            is_network=record.is_network,
            glass_transition_k=record.glass_transition_k,
            temperature_k=temperature_k,
        )
        if name is None:
            continue
        members.setdefault(name, []).append((record.abbreviation, value))

    classes: dict[str, tuple[float, float, int]] = {}
    hits = 0
    trials = 0
    ratios: list[float] = []
    worst_ratio = 1.0
    worst_polymer = ""

    for name, entries in sorted(members.items()):
        logs = [math.log(v) for _, v in entries]
        centre = math.exp(statistics.fmean(logs))
        if len(logs) > 1:
            spread = statistics.stdev(logs)
            for index, (label, value) in enumerate(entries):
                others = [logs[j] for j in range(len(logs)) if j != index]
                predicted = statistics.fmean(others)
                held_out_spread = statistics.stdev(others) if len(others) > 1 else 0.0
                deviation = abs(math.log(value) - predicted)
                trials += 1
                hits += deviation <= held_out_spread
                ratio = math.exp(deviation)
                ratios.append(ratio)
                if ratio > worst_ratio:
                    worst_ratio, worst_polymer = ratio, label
        else:
            # A class of one cannot state a spread of its own.  It borrows the
            # widest spread any class has measured, which is an admission that
            # the number rests on a single material rather than a claim that it
            # is as good as the classes that do not.
            spread = 0.0
        classes[name] = (centre, spread, len(entries))

    widest = max((s for _, s, _ in classes.values()), default=1.0)
    classes = {
        name: (centre, spread if spread > 0.0 else widest, count)
        for name, (centre, spread, count) in classes.items()
    }
    return ToughnessProxy(
        classes=classes,
        hits=hits,
        trials=trials,
        typical_ratio=math.exp(statistics.fmean([math.log(r) for r in ratios])) if ratios else 1.0,
        worst_ratio=worst_ratio,
        worst_polymer=worst_polymer,
    )


def strength_utilisation(theoretical_pa: Mapping[str, float]) -> dict[str, float]:
    """Measured tensile strength over a predicted flaw-free bound, per polymer.

    :mod:`formulate.experts.mechanical` introduces ``theoretical_strength`` as
    "how much of the possible a given material is actually delivering" and had
    nothing measured to compare it against.  The catalogue's tensile column is
    that comparison, and it is the only thing the column is used for: it is
    deliberately not turned into a ``tensile_strength`` prediction, because a
    real strength is set by the largest flaw in the specimen and is therefore
    not a property of the material at all.

    ``theoretical_pa`` is keyed by abbreviation so the caller supplies the
    predictions rather than this module reaching into the panel.
    """
    out: dict[str, float] = {}
    for record in polymer_records():
        bound = theoretical_pa.get(record.abbreviation)
        measured = record.tensile_pa
        if bound is None or measured is None or bound <= 0.0:
            continue
        out[record.abbreviation] = measured / bound
    return out


# --------------------------------------------------------------------------
# Experts
# --------------------------------------------------------------------------


class _CataloguePolymerExpert(Expert):
    """Shared plumbing for the two experts that read a repeat unit."""

    supported_classes = frozenset({MaterialClass.POLYMER})

    def is_available(self) -> bool:
        from formulate import chem

        return chem.rdkit_available()

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "RDKit is required to match a repeat unit"


class MeasuredPolymerExpert(_CataloguePolymerExpert):
    """Tabulated bulk properties for a catalogue polymer, and nothing else.

    The same argument :mod:`formulate.experts.measured` makes for molecules:
    where a property has been measured, estimating it instead is a choice to be
    less accurate.  It matters more here than there.  The group-contribution
    transition is worth 22 K on polymers withheld from its fit, and a
    specification asking for a transition inside an eighty-kelvin window is
    asking a question that 22 K cannot answer and 7 K can.

    Where the catalogue has no value this expert says so and the estimating
    experts answer.  It never fills a gap.
    """

    id = "polymer_measured"
    version = "1"
    method = "tabulated bulk properties from the bundled polymer catalogue"
    family = PropertyFamily.MECHANICAL
    supported_properties = frozenset(
        {"glass_transition_temperature", "amorphous_density", "elongation_at_break"}
    )

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        basis = f"{len(polymer_records())} polymers with tabulated bulk properties"
        spec = candidate.polymer
        if spec is None:
            return ApplicabilityDomain.outside("candidate carries no polymer", basis=basis)
        if catalogue_record(spec) is None:
            return ApplicabilityDomain.outside(
                "this repeat unit, at this architecture, is not in the bundled polymer "
                "catalogue; there is no measurement here to supply",
                basis=basis,
            )
        return ApplicabilityDomain(
            basis="a tabulated value for this exact polymer, not an estimate"
        )

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        spec = request.candidate.polymer
        if spec is None:
            return Prediction.unsupported(prop, self.id, "candidate carries no polymer")
        record = catalogue_record(spec)
        if record is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "this repeat unit, at this architecture, is not in the bundled polymer "
                "catalogue",
            )

        if prop == "glass_transition_temperature":
            if record.glass_transition_k is None:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"the catalogue carries no measured transition for {record.name}"
                    + (f": {record.note}" if record.note else ""),
                )
            return self._make(
                prop,
                record.glass_transition_k,
                "K",
                request,
                domain,
                std=_GLASS_TRANSITION_SPREAD_K,
                kind=UncertaintyKind.EPISTEMIC,
                basis=(
                    "spread between compilations for a measured glass transition, 5-10 K; "
                    "it is a kinetic event and its value depends on cooling rate and on "
                    "the technique that saw it"
                ),
                notes=self._notes(record),
                polymer=record.name,
            )

        if prop == "amorphous_density":
            if record.amorphous_density_g_cm3 is None:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    f"the catalogue carries no measured amorphous density for {record.name}"
                    + (f": {record.note}" if record.note else ""),
                )
            return self._make(
                prop,
                record.amorphous_density_g_cm3,
                "g/cm^3",
                request,
                domain,
                std=_AMORPHOUS_DENSITY_SPREAD,
                kind=UncertaintyKind.EPISTEMIC,
                basis=(
                    "largest disagreement between the two compilations bundled here over "
                    "the twelve polymers they share"
                ),
                notes=self._notes(record)
                + (
                    "the amorphous phase, not the sample: every semicrystalline polymer "
                    "in the catalogue is denser than this in service by an amount set by "
                    "processing",
                ),
                polymer=record.name,
            )

        if record.elongation_at_break is None:
            return Prediction.unsupported(
                prop,
                self.id,
                f"the catalogue carries no measured elongation for {record.name}"
                + (f": {record.note}" if record.note else ""),
            )
        low, high = record.elongation_at_break
        value = math.sqrt(low * high)
        factor = elongation_source_spread()
        return self._make(
            prop,
            value,
            "",
            request,
            domain,
            # Multiplicative: the property is registered as one, so the spread
            # has to be a factor turned into a standard deviation at this value
            # rather than a width that means something different at 0.02 and
            # at 10.
            std=value * (factor - 1.0),
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                f"half the geometric width of the quoted ranges across the catalogue, a "
                f"factor of {factor:.2f}. An elongation at break is a property of a "
                "specimen rather than of a polymer, and the range is the measurement"
            ),
            notes=self._notes(record)
            + (
                f"quoted as {low:.3g} to {high:.3g}; the geometric centre is reported",
                "for a moulded bar at ambient temperature and a conventional strain "
                "rate. It falls by an order of magnitude at high rate or low "
                "temperature, and neither of those is a laboratory condition",
            ),
            polymer=record.name,
            quoted_range=[low, high],
        )

    @staticmethod
    def _notes(record: PolymerRecord) -> tuple[str, ...]:
        notes = (f"{record.name} ({record.abbreviation}), source: {record.source}",)
        return notes + ((record.note,) if record.note else ())


class ChainToughnessExpert(_CataloguePolymerExpert):
    """Elongation at break from architecture and distance above the transition.

    For the polymers the catalogue has no elongation for, which is every
    polymer a search generates rather than retrieves.  Three classes, fitted
    leave-one-out against the measured column, and honest about the fact that
    one of the three barely works.
    """

    id = "toughness_proxy"
    version = "1"
    method = (
        "class mean elongation by network/glassy/rubbery classification, fitted "
        "leave-one-out against the bundled polymer catalogue"
    )
    family = PropertyFamily.MECHANICAL
    supported_properties = frozenset({"elongation_at_break"})
    dependencies = frozenset({"glass_transition_temperature"})

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        proxy = toughness_proxy()
        basis = (
            "class means over "
            + ", ".join(f"{count} {name}" for name, (_, _, count) in sorted(proxy.classes.items()))
            + f" polymers, leave-one-out hit rate {proxy.hits}/{proxy.trials}"
        )
        spec = candidate.polymer
        if spec is None:
            return ApplicabilityDomain.outside("candidate carries no polymer", basis=basis)
        if not _is_network(spec) and _attachment_points(spec) != 2:
            return ApplicabilityDomain.outside(
                "a repeat unit with other than two attachment points is a branch point or "
                "a chain end, and this classification is about linear chains and stated "
                "networks; it has nothing to say about anything between them",
                basis=basis,
            )
        return ApplicabilityDomain(basis=basis)

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        spec = request.candidate.polymer
        if spec is None:
            return Prediction.unsupported(prop, self.id, "candidate carries no polymer")

        temperature = request.conditions.temperature_k
        if temperature is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "a polymer is a glass or a rubber depending on where the temperature "
                "sits relative to its transition, and the request states no temperature",
            )

        network = _is_network(spec)
        points = _attachment_points(spec)
        if not network and points != 2:
            return Prediction.unsupported(
                prop,
                self.id,
                f"the repeat unit carries {points} attachment points rather than two, so "
                "it is neither a linear chain nor a stated network and this "
                "classification does not cover it",
            )

        glass_transition: float | None = None
        transition_source = ""
        if not network:
            upstream = request.dependency("glass_transition_temperature")
            if upstream is None or upstream.quantity is None:
                return Prediction.unsupported(
                    prop,
                    self.id,
                    "the class depends on how far the temperature sits from the glass "
                    "transition, and no upstream expert supplied one",
                )
            glass_transition = upstream.quantity.to("K").value
            transition_source = f"{upstream.expert_id} put the transition at {glass_transition:.0f} K"

        name = classify(
            is_network=network,
            glass_transition_k=glass_transition,
            temperature_k=temperature,
        )
        if name is None:
            shift = 0.0 if glass_transition is None else glass_transition - temperature
            return Prediction.unsupported(
                prop,
                self.id,
                f"the transition is {shift:.0f} K above the stated temperature, beyond "
                f"the {_GLASSY_CEILING_K:.0f} K the catalogue "
                "covers. Every polymer behind the glassy class sits inside that window "
                "and a rigid engineering plastic well outside it is a different material",
            )

        proxy = toughness_proxy()
        if name not in proxy.classes:  # pragma: no cover - class table invariant
            return Prediction.unsupported(
                prop, self.id, f"no catalogue polymer falls in the {name} class"
            )
        centre, spread, count = proxy.classes[name]

        notes = [
            f"classified {name}: " + _class_reason(name, glass_transition, temperature, spec),
            f"the class mean over {count} catalogue polymer(s); this is a classification "
            "reported as a number, not a structure-property correlation",
        ]
        if transition_source:
            notes.append(transition_source)
        if count == 1:
            notes.append(
                f"the {name} class has one member, so its spread is borrowed from the "
                "widest class rather than measured; the number is a single material's"
            )
        if name == "glassy":
            notes.append(
                "the glassy class is the weak one: polystyrene at two per cent and "
                "polyamide 6 at seventy-seven are both glasses at ambient and are 75 and "
                "25 K above their transitions, so distance above Tg does not separate "
                "them. What does is whether the glass shear-yields before it crazes, "
                "which needs an entanglement density the panel can only compute for "
                "repeat units it has a measured chain dimension for"
            )
        if abs(temperature - 298.15) > 5.0:
            notes.append(
                f"the classes were fitted against elongations measured at ambient and "
                f"this request is at {temperature:.0f} K; the boundaries move with "
                "temperature but the class means were not refitted at it"
            )

        # Two independent errors, added in the log because the property's error
        # is multiplicative. The first is how far the class mean sits from a
        # polymer's tabulated centre, measured leave-one-out. The second is how
        # far a specimen sits from that centre, which the measured route also
        # carries - and leaving it out here would have been a quiet cheat: the
        # class mean over five rubbers is a tighter number (1.40) than one
        # handbook range's half-width (1.97), so a proxy quoting only its own
        # fit error outranks the measurement it was fitted to. It was doing
        # exactly that until this line existed. Both routes predict the same
        # thing - what a bar of this polymer will do - so both have to be
        # quoted against it.
        total = math.hypot(spread, math.log(elongation_source_spread()))
        return self._make(
            prop,
            centre,
            "",
            request,
            domain,
            std=centre * (math.exp(total) - 1.0),
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                f"leave-one-out over the catalogue's measured column, a factor of "
                f"{math.exp(spread):.2f} within the {name} class, compounded with the "
                f"factor of {elongation_source_spread():.2f} between two specimens of "
                f"one polymer: {math.exp(total):.2f} in total, on an overall hit rate "
                f"of {proxy.hits}/{proxy.trials} and a typical miss of "
                f"{proxy.typical_ratio:.2f} times"
            ),
            notes=tuple(notes),
            classification=name,
        )


class PolymerArchitectureExpert(_CataloguePolymerExpert):
    """Crosslink density, read off the specification rather than modelled.

    Exact where it is defined and a refusal where it is not, in the same way
    :mod:`formulate.experts.structural` is exact about a molar mass.  A linear
    or branched chain has no crosslinks: that is what the word means, and
    reporting zero for it is a statement about the definition rather than an
    estimate.  A network that states no density is refused, because "it is
    crosslinked, by an unknown amount" and "it is not crosslinked" are opposite
    answers to the question a hot-melt specification is asking.
    """

    id = "polymer_architecture"
    version = "1"
    method = "crosslink density read from the polymer specification"
    family = PropertyFamily.STRUCTURAL
    supported_properties = frozenset({"crosslink_density"})

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return ""

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        basis = "read from the candidate's own architecture; exact where it is defined"
        if candidate.polymer is None:
            return ApplicabilityDomain.outside("candidate carries no polymer", basis=basis)
        return ApplicabilityDomain(basis=basis)

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        spec = request.candidate.polymer
        if spec is None:
            return Prediction.unsupported(prop, self.id, "candidate carries no polymer")

        stated = spec.crosslink_density
        if stated is not None:
            return self._make(
                prop,
                stated.to("mol/m^3").value,
                "mol/m^3",
                request,
                domain,
                std=0.0,
                kind=UncertaintyKind.EPISTEMIC,
                basis="the value the candidate states; nothing is inferred from it",
                notes=(
                    f"topology {spec.topology.value}",
                    "a covalent network cannot be remelted, redissolved or reprocessed, "
                    "whatever the rest of the panel says about it",
                ),
                topology=spec.topology.value,
            )

        if spec.topology in (PolymerTopology.NETWORK, PolymerTopology.DENDRITIC):
            return Prediction.unsupported(
                prop,
                self.id,
                f"a {spec.topology.value} polymer that states no crosslink density is "
                "under-specified for this question. Reporting zero would say it is a "
                "thermoplastic, which is the opposite of what its topology says, and "
                "guessing a value would invent the number a specification is asking about",
            )

        return self._make(
            prop,
            0.0,
            "mol/m^3",
            request,
            domain,
            std=0.0,
            kind=UncertaintyKind.EPISTEMIC,
            basis=(
                f"a {spec.topology.value} chain has no crosslinks by definition; this is "
                "the definition rather than a measurement"
            ),
            notes=(
                "zero here means remeltable: the chains are held together by "
                "entanglement and secondary forces, both of which heat undoes",
            ),
            topology=spec.topology.value,
        )


def _is_network(spec: PolymerSpec) -> bool:
    if spec.topology in (PolymerTopology.NETWORK, PolymerTopology.DENDRITIC):
        return True
    if spec.crosslink_density is not None and spec.crosslink_density.to("mol/m^3").value > 0.0:
        return True
    return any(m.role is MonomerRole.CROSSLINKER for m in spec.monomers)


def _attachment_points(spec: PolymerSpec) -> int:
    """Attachment points on the highest-weighted backbone repeat unit."""
    backbone = [m for m in spec.monomers if m.role is not MonomerRole.END_GROUP]
    if not backbone:
        return 0
    unit = max(backbone, key=lambda m: m.mole_fraction)
    from formulate import chem

    molecule = chem.mol_from_smiles(unit.smiles)
    if molecule is None:
        return unit.smiles.count("[*]")
    return sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0)


def _class_reason(
    name: str, glass_transition: float | None, temperature: float, spec: PolymerSpec
) -> str:
    if name == "network" or glass_transition is None:
        return (
            f"a {spec.topology.value} polymer is a single molecule, and a single molecule "
            "cannot draw - the strands between junctions have nowhere to go"
        )
    shift = glass_transition - temperature
    if name == "rubbery":
        return (
            f"the transition is {shift:+.0f} K from the stated temperature, so the chains "
            "are mobile and the material draws before it breaks"
        )
    return (
        f"the transition is {shift:+.0f} K above the stated temperature, so the chains are "
        "frozen and the material carries load elastically until something gives"
    )


def toughness_experts() -> list[Expert]:
    return [MeasuredPolymerExpert(), ChainToughnessExpert(), PolymerArchitectureExpert()]
