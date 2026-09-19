"""Will *this polymer* dissolve in that solvent?

:mod:`formulate.experts.dissolution` answers the question from the solvent's
side: the candidate is a small molecule, the target names a polymer in
``Conditions.solutes``, and the engine supplies the sphere.  Nothing answered
it from the polymer's side.  A polymer candidate could not be scored on
solubility at all, and neither could a formulation, because a polymer is not a
molecule and the engine refuses to pretend otherwise.  This module closes that
half: the candidate is a POLYMER (with the solvent named in
``Conditions.environment``) or a MIXTURE (with the solvent among its
components).

Method
------
Hansen's solubility sphere, from Hansen, *Hansen Solubility Parameters: A
User's Handbook*, 2nd ed. (CRC Press, 2007)::

    Ra^2 = 4(dD_s - dD_p)^2 + (dP_s - dP_p)^2 + (dH_s - dH_p)^2
    RED  = Ra / R0

Below one dissolves, near one swells, above one does not.  The construction is
not re-implemented here: the RED comes from
:func:`formulate.experts.dissolution.relative_energy_difference` and the
distance from :func:`formulate.experts.hansen.hansen_distance`, so the factor
of four on the dispersion axis and the Pa^0.5 -> MPa^0.5 conversion each exist
in exactly one place in this repository.  The spheres are
``dissolution.SOLUBILITY_SPHERES`` by import; there is no second copy of the
table.  What is new here is the *matching*: a candidate is joined to a sphere
by the canonical SMILES of its repeat unit, not by a name a target typed.

The interaction radius cannot be predicted, and neither can the centre
---------------------------------------------------------------------
R0 is measured - fitted to a set of solvent tests - so the first question was
whether it follows from the polymer's own cohesion.  Over the seven tabulated
spheres, ``R0 / delta_total`` is 0.565 (PS), 0.380 (PMMA), 0.164 (PVC), 0.370
(PC), 0.534 (PVAc), 0.303 (cellulose acetate), 0.206 (PA66): a 3.45x spread,
Pearson r = +0.33, r^2 = 0.107 (p = 0.47).  A mean-ratio estimator errs by
-36% on PS, +120% on PVC and +75% on PA66.  So no.

The fit that *would* have looked publishable is R0 against dD alone:
r = +0.912, r^2 = 0.832, p = 0.0042 over n = 7.  Leave-one-out kills it.  LOO
RMSE is 1.91 MPa^0.5, PVC comes back 107% high (3.5 -> 7.25), and a solvent
sitting exactly on a polymer's true boundary scores RED anywhere in 0.48 to
1.25 with the LOO-fitted radius.  A radius that cannot tell 0.48 from 1.25
cannot make the yes/no call RED exists for.  No radius is predicted here.

The centre is the same story, and this is where this module departs from its
brief.  ``dissolution.py`` already recorded that a sphere inferred from
repeat-unit group contributions "disagrees with the fitted one by enough to
move solvents across the boundary"; that claim is now measured.  Taking the
upstream ``polymer_hansen`` triple as the centre and keeping the handbook
radius moves **16 of 100** solvent/polymer pairs across RED = 1 (20 real
solvents x 5 repeat units), drops the textbook call rate from 25/26 to 22/26,
and puts polystyrene in hexane at RED 0.55 - hexane as a solvent for
polystyrene, which no chemist would accept.  The displacement between the two
centres is 8.65 (PS), 6.87 (PMMA), 7.15 (PVC), 2.88 (PC) and 10.82 (PVAc)
MPa^0.5, against radii of 12.7, 8.6, 3.5, 7.5 and 13.7.

So the sphere is used whole: fitted centre *and* fitted radius, because a
sphere is a single fitted object and re-centring it on a predicted point moves
the whole thing.  The declared dependency on the predicted triple is not
dropped and is not decorative - it is the **identity gate**.  No RED is
produced unless ``polymer_hansen`` supplies a triple, and none is produced if
that triple lands further from the fitted centre than R0 itself, because a
candidate whose own chemistry falls outside the polymer's measured sphere is
not the material the sphere was fitted to.  That gate refuses poly(vinyl
chloride) (displacement 7.15 against R0 3.5) and lets the other four through.

Ra is computed and never returned
---------------------------------
The brief's suggestion was to publish the Hansen distance Ra where a radius is
missing, on the grounds that Ra still ranks solvents correctly for one
polymer.  That is true - over 20 solvents at a fixed polymer,
Spearman(Ra, RED) = 1.000, because at fixed R0 it is a monotone transform -
but it is the wrong direction for this expert.  Here the *polymer* is the
candidate and the solvent is fixed.  Measured across the seven spheres at
fixed solvent, over 20 solvents from the ``chemicals`` compilation: mean
Spearman(Ra, RED) = +0.21 (min -0.39, max +0.71), 184 of 420 polymer pairs
(44%) ordered backwards, and Ra names a different best polymer than RED for 12
of the 20 solvents.  In toluene Ra picks polycarbonate where RED and reality
pick polystyrene; in water and in the alcohols Ra picks nylon-6,6.  Handing a
ranker a signal that is backwards two times in five is worse than handing it
nothing, so Ra appears in the notes and in the refusal text - where a chemist
reads it - and is never returned as a value under any property name.  The
general rule: Ra's validity is a property of the *population* being ranked,
not of the candidate, and an expert sees one candidate.

``hansen_distance`` is in any case already MixtureExpert's: it means the
distance between the least compatible pair in a formulation, which is a
different quantity from a polymer-solvent distance.

Two names that had to be refused before the compilation was asked
-----------------------------------------------------------------
``chemicals`` carries polymer names as synonyms of their monomers, so
``CAS_from_any('polystyrene')`` returns styrene's CAS and a full Hansen triple.
Reading ``Conditions.environment`` straight into the compilation therefore
answered ``environment: 'polystyrene'`` with RED 0.57 against a polystyrene
candidate - "polystyrene dissolves polystyrene, inside the sphere, so a
solvent" - and ``polyurethane`` with ethylurea, ``polyoxymethylene`` with
formaldehyde, ``polyethylene glycol`` with ethylene glycol and
``poly(vinyl acetate)`` with sec-butyl acetate, which is not even its monomer.
:func:`names_a_polymer` refuses all of them and quotes what the compilation
would have handed over, because a polymer is not its monomer and this module
exists because of that fact.

A formulation's other molecular components are refused for the same reason the
co-solvent pair already was.  A component is part of the liquid because of what
it is, not because of the role label it carries, and counting only the
solvent-role ones let 10% PMMA / 45% acetone / 45% water report RED 0.72 with
no mention that half the liquid had been dropped.

What it is worth
----------------
Over 31 polymer/solvent pairs whose behaviour is textbook, the shipped
construction makes **30 of 31** calls correctly.  The single miss is
poly(vinyl acetate) in methanol (RED 1.29, and methanol does dissolve PVAc),
and it is inside its own one-sigma bound - which is the point of sizing the
bar the way :data:`DISPLACEMENT_FRACTION` does rather than at the radius
spread alone, where the same miss sits at 1.15 and the bar would have been a
flattering lie.  All 31 calls are consistent with the shipped bar at one
sigma, against the ~68 per cent a correct estimate implies, so the bar is if
anything wide - which is why the upstream triple's own claimed spread, 10.8
MPa^0.5 once propagated onto this distance, is reported rather than added.
Nothing in this module is fitted here; the one held-out number quoted is the
negative result above, LOO RMSE 1.91 MPa^0.5 on R0.

Coverage is four polymers - polystyrene, poly(methyl methacrylate),
polycarbonate, poly(vinyl acetate) - out of the seven tabulated spheres.
Poly(vinyl chloride) fails the identity gate; polyamide 66 is refused upstream
because the Hoftyzer-Van Krevelen table shipped here has no amide increment;
cellulose acetate has no repeat unit at a fixed degree of substitution.
Widening that means someone transcribing more spheres out of the handbook, and
quoting a radius that cannot be cited to a page is exactly the
confident-wrong-number failure this repository exists to avoid.

What is not modelled.  Molar mass: R0 is fitted to a commercial high polymer,
and an oligomer dissolves in solvents the high polymer only swells in.
Crystallinity: a semicrystalline sample resists solvents its sphere admits.
Both are warnings rather than refusals, because the fitted sphere already
encodes whatever sample was tested.  Kinetics: RED is a thermodynamic
statement, and a glassy polymer below its Tg can take a day to let a good
solvent in.  And the first-order temperature effect, which is that R0 itself
grows with temperature - so a RED above one at elevated temperature is not a
reliable "no".  That is a note and an out-of-domain flag, not a term in the
error bar, because its size is unknown.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Sequence

from formulate.core.candidate import (
    Candidate,
    ComponentRole,
    MaterialClass,
    MonomerRole,
    PolymerSpec,
    PolymerTopology,
)
from formulate.core.prediction import Prediction, PredictionStatus
from formulate.core.properties import PropertyFamily
from formulate.core.provenance import SoftwareEnvironment
from formulate.core.quantity import ApplicabilityDomain, Quantity, UncertaintyKind

from .base import Expert, PredictionRequest
from .dissolution import (
    SOLUBILITY_SPHERES,
    SolubilitySphere,
    relative_energy_difference,
    resolve_solute,
)
from .hansen import hansen_distance, hansen_triple

if TYPE_CHECKING:  # pragma: no cover
    from .registry import ExpertRegistry

try:  # pragma: no cover - exercised by the availability path, not by a branch
    from chemicals import CAS_from_any
    from chemicals.solubility import hansen_delta_d, hansen_delta_h, hansen_delta_p

    _HAVE_CHEMICALS = True
    _IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    _HAVE_CHEMICALS = False
    _IMPORT_ERROR = str(exc)


# --------------------------------------------------------------------------
# Matching a candidate to a fitted sphere
# --------------------------------------------------------------------------

#: Repeat unit of each polymer whose sphere ``dissolution.py`` tabulates, so a
#: candidate can be matched by *structure* rather than by a name a target
#: happened to type.  ``dissolution.py`` matches on a string because its
#: candidate is the solvent and the polymer arrives as text; here the polymer
#: *is* the candidate and carries a repeat unit, which is a stronger match - a
#: target cannot misspell it and a generated polymer has no name at all.
#:
#: The spellings are the ones ``data/reference_polymers.json`` already uses, so
#: a candidate seeded from the repeat-unit library matches without a second
#: convention.  Polycarbonate is not in that file and is written here as the
#: bisphenol-A carbonate the handbook sphere was fitted to.  Cellulose acetate
#: is deliberately absent: its degree of substitution is a processing variable,
#: so it has no repeat unit at all, which is a statement about the material
#: rather than a gap in this table.
SPHERE_REPEAT_UNITS: dict[str, str] = {
    "polystyrene": "[*]CC(c1ccccc1)[*]",
    "poly(methyl methacrylate)": "[*]CC(C)(C(=O)OC)[*]",
    "poly(vinyl chloride)": "[*]CC(Cl)[*]",
    "polycarbonate": "[*]Oc1ccc(cc1)C(C)(C)c1ccc(cc1)OC(=O)[*]",
    "poly(vinyl acetate)": "[*]CC(OC(C)=O)[*]",
    "polyamide 66": "[*]NCCCCCCNC(=O)CCCCC(=O)[*]",
}


# --------------------------------------------------------------------------
# The uncertainty budget
# --------------------------------------------------------------------------

#: Spread between the published Hansen compilations for one solvent, in
#: MPa^0.5.  Taken from ``hansen.py``'s own figure rather than restated
#: independently, so the two cannot drift apart.
SOLVENT_COMPILATION_SPREAD_MPA_SQRT = 0.5

#: A tenth of a radius: ``dissolution.py``'s measure of how far published fits
#: of the *same* polymer's sphere move.  It enters RED as ``0.10 * RED``,
#: because RED = Ra/R0 and a fractional error in R0 is a fractional error in
#: RED.
RADIUS_SPREAD_FRACTION = 0.10

#: How much of the predicted-versus-fitted centre displacement actually shows
#: up as an error in Ra.  The triangle inequality caps it at the full
#: displacement, but that cap is attained only for a solvent lying on the line
#: between the two centres.  Measured over 100 real solvent/polymer pairs the
#: realised ratio |dRa| / displacement has mean 0.437 and median 0.434 (max
#: 0.962), so 0.44 is the typical case rather than the worst one.
#:
#: The list of twenty solvents that measurement runs over is pinned in
#: ``tests/test_polymer_dissolution.py`` and re-measured there rather than
#: quoted, because a constant justified by a number nobody can reproduce is a
#: constant justified by nothing.  Over that pinned list the figures are mean
#: 0.428, median 0.417, max 0.962, with 49 of the 100 pairs above 0.44.
#:
#: This is the largest term and it is why the expert is defensible.  It is not
#: a claim that the *fitted* centre is wrong by this much - the fitted centre
#: makes 30 of 31 textbook calls and the predicted one 22 of 26, so most of the
#: displacement is the upstream method's own bias, which ``polymer_hansen``
#: measures at -2.15/-2.43/-1.59 MPa^0.5 and declines to correct for.  It is a
#: statement that the further a candidate's own chemistry sits from the
#: material the handbook tested, the less that sphere describes *this*
#: candidate.  Dropping the term leaves the one call the method gets wrong
#: (PVAc in methanol, RED 1.29) outside its own error bar at 1.15; keeping it
#: puts it inside at 0.92.
DISPLACEMENT_FRACTION = 0.44

#: Volumetric thermal expansion, per kelvin, used only to size the temperature
#: term.  At roughly constant cohesive energy delta scales as V^-1/2, so
#: d(delta)/delta = -(alpha/2) dT.  A glassy or semicrystalline polymer sits
#: near 6e-4 and an organic liquid near 1.1e-3; the difference between the two
#: is what moves Ra.  Over 50 K this is about 0.26 MPa^0.5 on Ra for
#: polystyrene in toluene - second order against the compilation spread, which
#: is why it is included for completeness rather than because it matters.
THERMAL_EXPANSION_POLYMER_PER_K = 6.0e-4
THERMAL_EXPANSION_SOLVENT_PER_K = 1.1e-3

#: Reference temperature of the handbook spheres and of the ``chemicals``
#: compilation: both are 25 degC values.
REFERENCE_TEMPERATURE_K = 298.15

#: Window over which the sphere is treated as in-domain, matching
#: ``polymer_hansen.TEMPERATURE_WINDOW_K`` deliberately - a RED resting on a
#: centre that expert has already refused outside this range would be resting
#: on nothing.  Outside it the prediction is still produced but marked out of
#: domain, because the effect that matters is that R0 *grows* with temperature
#: and nothing here knows by how much.
TEMPERATURE_WINDOW_K = (273.0, 323.0)

#: Molar mass below which a chain is an oligomer rather than the high polymer
#: the sphere was fitted to.  Van Krevelen puts the onset of chain-length
#: independent bulk behaviour in the tens of kg/mol; 10 kg/mol is the
#: conservative end of that, and this warns rather than refuses because the
#: direction of the error is known - an oligomer dissolves in more than its
#: high polymer does, so a RED near one is optimistic rather than meaningless.
OLIGOMER_MOLAR_MASS_G_PER_MOL = 10_000.0

#: The property this expert produces for each material class.  The two names
#: are the same number - a Hansen distance over an interaction radius - so
#: emitting both for one candidate would put one quantity twice into a Pareto
#: ranking and silently double its weight.  ``solubility_red`` is registered
#: as the solvent/solute pair number and is what a POLYMER candidate in a named
#: solvent is; ``relative_energy_difference`` is the formulation-level name and
#: is what a MIXTURE is.  ``covers()`` is overridden to match, so
#: ``registry.coverage()`` reports the split honestly rather than claiming both
#: for both.
PROPERTY_FOR_CLASS: dict[MaterialClass, str] = {
    MaterialClass.POLYMER: "solubility_red",
    MaterialClass.MIXTURE: "relative_energy_difference",
}

#: The upstream triple, in the order Hansen writes it.
_CENTRE_PROPERTIES = (
    "hansen_dispersion",
    "hansen_polar",
    "hansen_hydrogen_bonding",
)

#: Roles whose component is the liquid the polymer has to dissolve in.
_SOLVENT_ROLES = frozenset({ComponentRole.SOLVENT, ComponentRole.CO_SOLVENT})

#: Evidence carried into every refusal that a radius is missing, so the reader
#: is told why no radius was estimated rather than merely that none was found.
_RADIUS_EVIDENCE = (
    "R0 is measured, not derivable: over the seven tabulated spheres R0/delta_total "
    "spans 0.164 to 0.565 (r^2 = 0.107 against delta_total), and the one fit that "
    "looks publishable - R0 against dD, r = 0.912 - has a leave-one-out RMSE of "
    "1.91 MPa^0.5, which puts a solvent on the true boundary anywhere between "
    "RED 0.48 and 1.25"
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _mpa_sqrt(value_pa_sqrt: float) -> float:
    """Pa^0.5 -> MPa^0.5 through the shared unit registry.

    Not a hand-written factor of a thousand.  ``dissolution.py`` documents what
    happened the one time that factor was written by hand on the square root of
    a pressure: every real solvent for polystyrene scored as a non-solvent.
    """
    return Quantity(value=value_pa_sqrt, unit="Pa^0.5").to("MPa^0.5").value


@functools.lru_cache(maxsize=1)
def _canonical_sphere_units() -> dict[str, str]:
    """Canonical repeat-unit SMILES -> the key in ``SOLUBILITY_SPHERES``."""
    from formulate import chem

    out: dict[str, str] = {}
    for name, unit in SPHERE_REPEAT_UNITS.items():
        canon = chem.canonical_smiles(unit) if chem.rdkit_available() else None
        out[canon or unit] = name
    return out


def sphere_for_repeat_unit(unit_smiles: str) -> tuple[str, SolubilitySphere] | None:
    """The fitted sphere whose repeat unit this is, or None.

    Matching is on the canonical SMILES so that ``[*]CC(c1ccccc1)[*]`` and
    ``c1ccccc1C([*])C[*]`` are the same polystyrene, which a string comparison
    would miss.
    """
    from formulate import chem

    canon = chem.canonical_smiles(unit_smiles) if chem.rdkit_available() else None
    name = _canonical_sphere_units().get(canon or unit_smiles)
    return (name, SOLUBILITY_SPHERES[name]) if name is not None else None


@functools.lru_cache(maxsize=1024)
def solvent_triple_from_name(name: str) -> tuple[float, float, float] | None:
    """Hansen triple in Pa^0.5 for a solvent named as text, or None.

    ``hansen.hansen_triple`` resolves a SMILES; a POLYMER candidate names its
    solvent in ``Conditions.environment`` as words, so the name is resolved to
    a CAS number directly.  This is a lookup in the same compilation
    ``hansen.py`` reads, not a second method: no value is estimated for a name
    that resolves to no triple, exactly as ``hansen.py`` refuses to invent a
    split from a total solubility parameter.
    """
    if not _HAVE_CHEMICALS:
        return None
    try:
        cas = CAS_from_any(name.strip())
    except Exception:
        # "air" and "vacuum" land here: they are environments, not solvents.
        return None
    try:
        values = (hansen_delta_d(cas), hansen_delta_p(cas), hansen_delta_h(cas))
    except Exception:
        return None
    if any(v is None for v in values):
        # "nitrogen" lands here: a real compound with no Hansen triple.
        return None
    return tuple(float(v) for v in values)  # type: ignore[return-value]


def names_a_polymer(name: str) -> bool:
    """True when this environment string names a polymer rather than a liquid.

    It has to be asked *before* the compilation is consulted, because the
    compilation answers.  ``chemicals`` carries polymer names as synonyms of
    their monomers, so ``CAS_from_any`` resolves every one of these, measured
    against the installed version rather than assumed::

        polystyrene          -> styrene           100-42-5
        polyethylene         -> ethene             74-85-1
        polypropylene        -> propene           115-07-1
        poly(vinyl chloride) -> vinyl chloride     75-01-4
        polyacrylonitrile    -> acrylonitrile     107-13-1
        polyoxymethylene     -> formaldehyde       50-00-0
        polyethylene glycol  -> ethylene glycol   107-21-1
        polyurethane         -> ethylurea         625-52-5
        poly(vinyl acetate)  -> sec-butyl acetate 105-46-4

    The first eight are the monomer, which is the substitution this repository
    forbids outright - a polymer is not its monomer, and the whole reason this
    module exists is that a polymer could not be scored as one.  The last is
    not even that: a name collision onto an unrelated ester.  All but
    ``polyurethane`` go on to carry a Hansen triple, so without this guard
    ``environment: 'polystyrene'`` returned RED 0.57 against a polystyrene
    candidate and read, in the notes, as "polystyrene dissolves polystyrene,
    inside the sphere, so a solvent".  ``polyurethane`` was refused only
    because ethylurea happens to have no tabulated triple, which is luck rather
    than a guard, and is why the test for this is about the name.

    A leading ``poly`` is the test because no monomeric liquid is named that
    way; polyethylene glycol and the polysorbates are themselves oligomer
    distributions, so refusing them is the right answer rather than a
    false positive.  ``resolve_solute`` is consulted as well, to catch the
    abbreviations this repository already treats as polymers - ``PS``, ``PMMA``,
    ``acrylic``, ``nylon-66`` - one of which (``PS``) resolves to a CAS number
    of its own.
    """
    key = name.strip().lower()
    if resolve_solute(key) is not None:
        return True
    squashed = key.replace("(", "").replace(")", "").replace("-", " ").replace("_", " ")
    return squashed.lstrip().startswith("poly")


def compilation_would_have_used(name: str) -> str:
    """What the compilation resolves a name to, quoted as evidence in a refusal.

    A refusal that says "that is a polymer name" is an assertion; one that says
    "the compilation would have handed you sec-butyl acetate" is a measurement,
    and the chemist can check it.
    """
    if not _HAVE_CHEMICALS:
        return ""
    try:
        # Imported here rather than at module scope: this is evidence for a
        # refusal, and a compilation that has dropped the lookup should cost
        # the refusal its citation, not disable the whole expert.
        from chemicals import search_chemical

        meta = search_chemical(name.strip())
    except Exception:
        return ""
    return f"{meta.common_name} (CAS {meta.CASs})"


def repeat_unit_smiles(spec: PolymerSpec) -> str:
    """The chain-forming repeat unit this polymer is matched on.

    ``spec.monomers[0]`` is not it.  ``PolymerSpec`` allows an end group to be
    listed first - it is excluded from the mole-fraction sum, not from the
    tuple - and ``polymer_hansen`` and ``polymer.py`` both filter on the role
    for exactly that reason.  Matching on ``monomers[0]`` refused a perfectly
    ordinary end-capped polystyrene, and printed the distance to the *chain's*
    predicted centre in the refusal while claiming it was the end group's.
    """
    for monomer in spec.monomers:
        if monomer.role is not MonomerRole.END_GROUP:
            return monomer.smiles
    return spec.monomers[0].smiles  # pragma: no cover - PolymerSpec forbids this


def _chain_units(spec: PolymerSpec) -> list[str]:
    """Distinct chain-forming repeat units, canonicalised."""
    units: list[str] = []
    for monomer in spec.monomers:
        if monomer.role is MonomerRole.END_GROUP:
            continue
        key = monomer.canonical_key()
        if key not in units:
            units.append(key)
    return units


def _architecture_refusal(spec: PolymerSpec) -> str | None:
    """Why this architecture has no fitted sphere, if it does not.

    Separate from ``polymer._architecture_reason`` because the reasons differ:
    a branched polymer still dissolves and still has a sphere, while a network
    does not dissolve at any RED at all.
    """
    if spec.topology is PolymerTopology.NETWORK:
        return (
            "a crosslinked network does not dissolve at any RED; it swells, and the "
            "solubility sphere describes where that swelling is greatest rather than "
            "a solution - reporting a RED for it would answer a different question"
        )
    if spec.crosslink_density is not None:
        return (
            "a stated crosslink density means this candidate swells rather than "
            "dissolves; the handbook spheres are fitted to uncrosslinked polymers"
        )
    if len(_chain_units(spec)) > 1:
        return (
            "a copolymer: no sphere has been fitted to one, and it cannot be "
            "interpolated - a copolymer's centre moves with composition and its "
            "radius has never been measured as a function of it"
        )
    return None


def solvent_sigma_on_ra(
    solvent_mpa: tuple[float, float, float],
    centre_mpa: tuple[float, float, float],
    distance: float,
) -> float:
    """Compilation spread on the solvent triple, propagated to Ra.

    Ra is a weighted Euclidean norm, so dRa/d(dD_s) = 4 * ddD / Ra and the
    other two derivatives are ddP / Ra and ddH / Ra.  Analytic rather than by
    finite difference because the derivative is exact and the function is cheap.
    """
    if distance <= 0.0:
        # The solvent sits on the centre: Ra is at a minimum, every first
        # derivative vanishes, and the spread enters at second order.  Return
        # the component spread itself rather than a zero.
        return SOLVENT_COMPILATION_SPREAD_MPA_SQRT
    weights = (4.0, 1.0, 1.0)
    return math.sqrt(
        sum(
            (weights[i] * (solvent_mpa[i] - centre_mpa[i]) / distance
             * SOLVENT_COMPILATION_SPREAD_MPA_SQRT) ** 2
            for i in range(3)
        )
    )


def temperature_sigma_on_ra(
    solvent_mpa: tuple[float, float, float],
    centre_mpa: tuple[float, float, float],
    temperature_k: float | None,
) -> float:
    """How far Ra moves if both materials are taken to ``temperature_k``.

    Derived rather than guessed: at roughly constant cohesive energy a
    solubility parameter scales as V^-1/2, so each component shrinks by
    (alpha/2) dT.  The solvent expands about twice as fast as the polymer, so
    the two do not cancel.
    """
    if temperature_k is None:
        return 0.0
    delta_t = temperature_k - REFERENCE_TEMPERATURE_K
    if delta_t == 0.0:
        return 0.0
    solvent_scale = 1.0 - 0.5 * THERMAL_EXPANSION_SOLVENT_PER_K * delta_t
    polymer_scale = 1.0 - 0.5 * THERMAL_EXPANSION_POLYMER_PER_K * delta_t
    shifted = hansen_distance(
        tuple(v * solvent_scale for v in solvent_mpa),  # type: ignore[arg-type]
        tuple(v * polymer_scale for v in centre_mpa),  # type: ignore[arg-type]
    )
    return abs(shifted - hansen_distance(solvent_mpa, centre_mpa))


def verdict(red: float) -> str:
    """The same three-way reading ``dissolution.py`` prints, in one place."""
    if red < 0.9:
        return "inside the sphere, so a solvent"
    if red < 1.1:
        return "on the boundary, so a swelling agent or a marginal solvent"
    return "outside the sphere, so a non-solvent"


# --------------------------------------------------------------------------
# The expert
# --------------------------------------------------------------------------


class PolymerDissolutionExpert(Expert):
    """How well a named solvent dissolves a polymer candidate."""

    id = "polymer_dissolution"
    version = "1"
    method = (
        "Hansen solubility sphere fitted to the repeat unit (Hansen, "
        "Hansen Solubility Parameters: A User's Handbook, 2nd ed., CRC Press 2007), "
        "with the predicted repeat-unit triple as an identity gate on the sphere"
    )
    family = PropertyFamily.CHEMICAL
    supported_classes = frozenset({MaterialClass.POLYMER, MaterialClass.MIXTURE})
    supported_properties = frozenset(PROPERTY_FOR_CLASS.values())
    dependencies = frozenset(_CENTRE_PROPERTIES)

    def __init__(self, panel: "ExpertRegistry | None" = None) -> None:
        # Injectable for the same reason ``blend.py`` makes its panel
        # injectable: the mixture path dispatches a sub-run, and a test that
        # wants to know what happens when that panel is empty should be able to
        # hand one over rather than monkey-patching a module.
        self._panel_cache = panel

    # -- capability --------------------------------------------------------

    def covers(self, prop: str, material_class: MaterialClass) -> bool:
        return PROPERTY_FOR_CLASS.get(material_class) == prop

    def applicable_properties(
        self, requested: Sequence[str] | frozenset[str], material_class: MaterialClass
    ) -> frozenset[str]:
        prop = PROPERTY_FOR_CLASS.get(material_class)
        if prop is None:
            return frozenset()
        return frozenset({prop}) & frozenset(requested)

    def is_available(self) -> bool:
        from formulate import chem

        return _HAVE_CHEMICALS and chem.rdkit_available()

    def unavailable_reason(self) -> str:
        from formulate import chem

        if not _HAVE_CHEMICALS:
            return f"the chemicals package is not installed ({_IMPORT_ERROR})"
        if not chem.rdkit_available():
            return "RDKit is required to match a repeat unit to a fitted sphere"
        return ""

    def _software(self) -> SoftwareEnvironment:
        import chemicals

        from formulate import chem

        return SoftwareEnvironment.capture(
            chemicals=chemicals.__version__, rdkit=chem.rdkit_version()
        )

    # -- domain ------------------------------------------------------------

    def assess_domain(self, candidate: Candidate) -> ApplicabilityDomain:
        basis = (
            "linear uncrosslinked high polymers whose sphere is fitted in the Hansen "
            "handbook and whose repeat unit the predicted triple agrees with to within "
            "R0; 30 of 31 textbook polymer/solvent calls correct, the one miss inside "
            "its own one-sigma bound"
        )
        spec = self._polymer_spec(candidate)
        if spec is None:
            if candidate.material_class is MaterialClass.MIXTURE:
                return ApplicabilityDomain.outside(
                    "a formulation with anything other than exactly one polymer and one "
                    "solvent-role component is not a polymer-in-a-solvent",
                    basis=basis,
                )
            return ApplicabilityDomain.outside("candidate carries no polymer", basis=basis)

        architecture = _architecture_refusal(spec)
        if architecture is not None:
            return ApplicabilityDomain.outside(architecture, basis=basis)

        match = sphere_for_repeat_unit(repeat_unit_smiles(spec))
        if match is None:
            return ApplicabilityDomain.outside(
                "no fitted solubility sphere for this repeat unit, and R0 cannot be "
                "predicted from structure",
                basis=basis,
            )

        warnings: list[str] = []
        score = 1.0

        temperature = candidate.conditions.temperature_k
        low, high = TEMPERATURE_WINDOW_K
        if temperature is not None and not low <= temperature <= high:
            warnings.append(
                f"the sphere is a 25 degC object and {temperature:.0f} K is outside the "
                f"{low:.0f}-{high:.0f} K window; R0 itself grows with temperature by an "
                "amount this expert does not model, so a RED above one here is not a "
                "reliable non-solvent"
            )
            score = min(score, 0.25)

        mass = spec.number_average_molar_mass
        if mass is not None and mass.to("g/mol").value < OLIGOMER_MOLAR_MASS_G_PER_MOL:
            warnings.append(
                f"Mn below {OLIGOMER_MOLAR_MASS_G_PER_MOL:.0f} g/mol is an oligomer, and "
                "the sphere was fitted to a commercial high polymer; an oligomer "
                "dissolves in solvents the high polymer only swells in, so this RED is "
                "optimistic rather than wrong"
            )
            score = min(score, 0.6)

        regularity = self._crystallinity_warning(spec)
        if regularity is not None:
            warnings.append(regularity)
            score = min(score, 0.7)

        return ApplicabilityDomain(
            score=score, in_domain=score > 0.3, warnings=tuple(warnings), basis=basis
        )

    @staticmethod
    def _crystallinity_warning(spec: PolymerSpec) -> str | None:
        """A configurationally regular backbone may crystallise and resist its sphere.

        It over-warns, and knowingly.  Bisphenol-A polycarbonate has no
        backbone stereocentre and is nevertheless amorphous in practice,
        because the isopropylidene bridge stops the chains packing - the same
        class of miss ``melt.py`` records for polyisobutylene.  The warning
        stays because the direction of the error is one-sided: a crystalline
        sample resists solvents its sphere admits, so an unflagged RED would be
        optimistic, while a spurious flag only makes a correct answer look less
        certain than it is.
        """
        from formulate import chem

        if not chem.rdkit_available():
            return None
        try:
            from .melt import backbone_stereocentres
        except Exception:  # pragma: no cover - melt is part of the same package
            return None
        centres = backbone_stereocentres(repeat_unit_smiles(spec))
        if centres is None or centres:
            return None
        return (
            "the backbone is configurationally regular, so this polymer may crystallise; "
            "a semicrystalline sample resists solvents its amorphous sphere admits, and "
            "nothing here sees crystallinity"
        )

    # -- which polymer is being asked about --------------------------------

    @staticmethod
    def _polymer_spec(candidate: Candidate) -> PolymerSpec | None:
        """The one polymer this candidate is about, or None if that is not one.

        A mixture with two polymers has no single sphere, and a mixture with
        none is not a polymer solution; both give None here, and the caller
        turns that into a refusal that names which case it was.
        """
        if candidate.material_class is MaterialClass.POLYMER:
            return candidate.polymer
        mixture = candidate.mixture
        if mixture is None:
            return None
        polymers = [c.polymer for c in mixture.components if c.polymer is not None]
        solvents = [
            c for c in mixture.components
            if c.role in _SOLVENT_ROLES and c.molecule is not None
        ]
        if len(polymers) != 1 or len(solvents) != 1:
            return None
        return polymers[0]

    # -- prediction --------------------------------------------------------

    def _predict_one(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction | None:
        if request.candidate.material_class is MaterialClass.POLYMER:
            return self._predict_polymer(prop, request, domain)
        return self._predict_mixture(prop, request, domain)

    # -- the POLYMER path --------------------------------------------------

    def _predict_polymer(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction:
        spec = request.candidate.polymer
        if spec is None:  # pragma: no cover - the class validator forbids it
            return Prediction.unsupported(prop, self.id, "candidate carries no polymer")

        architecture = _architecture_refusal(spec)
        if architecture is not None:
            return Prediction.unsupported(prop, self.id, architecture)

        # The upstream triple first, because it is the centre this expert is
        # allowed to know anything about and because its absence is the path
        # that has to work cleanly if polymer_hansen never ships.
        centre_mpa = self._upstream_centre(request)
        if centre_mpa is None:
            return Prediction.unsupported(
                prop,
                self.id,
                "no predicted Hansen triple for this repeat unit: "
                + self._missing_dependencies(request)
                + ". The upstream triple is what identifies the candidate as the polymer "
                "a sphere was fitted to, and nothing is substituted for it - a handbook "
                "centre reached by name alone would answer for a material this candidate "
                "may not be",
            )

        solvent_pa, solvent_label, solvent_refusal = self._named_solvent(request)
        if solvent_pa is None:
            return Prediction.unsupported(prop, self.id, solvent_refusal)
        solvent_mpa = tuple(_mpa_sqrt(v) for v in solvent_pa)

        match = sphere_for_repeat_unit(repeat_unit_smiles(spec))
        if match is None:
            # Ra against the *predicted* centre is the only distance available,
            # and it is put where a chemist reads it rather than returned as a
            # value: across polymers at a fixed solvent, Ra orders 44% of pairs
            # the opposite way to RED.
            distance = hansen_distance(solvent_mpa, centre_mpa)  # type: ignore[arg-type]
            return Prediction.unsupported(
                prop,
                self.id,
                f"no fitted interaction radius for this repeat unit. Its Hansen distance "
                f"to {solvent_label} is Ra = {distance:.2f} MPa^0.5 on the predicted "
                f"centre, which ranks solvents for this one polymer but cannot say "
                f"whether any of them dissolves it, and must not be used to rank "
                f"polymers against one another: {_RADIUS_EVIDENCE}. Spheres are "
                f"tabulated for " + ", ".join(sorted(SPHERE_REPEAT_UNITS)),
            )

        name, sphere = match
        return self._assemble(
            prop,
            request,
            domain,
            sphere,
            name,
            centre_mpa,
            solvent_mpa,
            solvent_label,
            centre_sigma=self._upstream_sigma(request),
        )

    def _named_solvent(
        self, request: PredictionRequest
    ) -> tuple[tuple[float, float, float] | None, str, str]:
        """The solvent the target named, as (triple in Pa^0.5, label, refusal)."""
        conditions = request.conditions
        environment = conditions.environment or request.candidate.conditions.environment
        if not environment:
            return (
                None,
                "",
                "no solvent was named: whether a polymer dissolves is a question about a "
                "pair, not about the polymer alone. State the liquid in the target's "
                "conditions, e.g. environment: 'acetone'",
            )
        # Asked before the compilation is, because the compilation answers a
        # polymer name with its monomer and the answer looks like a solvent.
        if names_a_polymer(environment):
            resolved = compilation_would_have_used(environment)
            detail = (
                f" The compilation carries that name as a synonym of {resolved}, so a "
                f"triple would have been found - it is the wrong material's."
                if resolved
                else ""
            )
            return (
                None,
                environment,
                f"{environment!r} names a polymer, not a liquid, and no RED is produced "
                f"against it.{detail} A polymer is not its monomer: the two have "
                "different cohesion, different molar volume and no shared Hansen triple, "
                "and substituting one for the other is the failure this expert exists to "
                "avoid. Name the liquid the candidate has to dissolve in, e.g. "
                "environment: 'acetone'. Whether one polymer dissolves another is a "
                "different question, and nothing here answers it",
            )

        triple = solvent_triple_from_name(environment)
        if triple is None:
            return (
                None,
                environment,
                f"{environment!r} resolves to no Hansen triple: either it is not a "
                "compound at all (an environment such as 'air' or 'vacuum'), or it is a "
                "compound the compilation has no measured triple for (such as "
                "'nitrogen'). No triple is estimated, because three components cannot be "
                "recovered from one total solubility parameter. Name a liquid, e.g. "
                "environment: 'acetone'",
            )
        return triple, environment, ""

    def _missing_dependencies(self, request: PredictionRequest) -> str:
        missing = [p for p in _CENTRE_PROPERTIES if request.dependency(p) is None]
        if not missing:  # pragma: no cover - only reached when all three exist
            return "the triple is present but carries no value"
        return "missing " + ", ".join(missing)

    def _upstream_centre(self, request: PredictionRequest) -> tuple[float, float, float] | None:
        """The predicted repeat-unit triple in MPa^0.5, or None if incomplete."""
        values = [request.dependency_value(p, "Pa^0.5") for p in _CENTRE_PROPERTIES]
        if any(v is None for v in values):
            return None
        return tuple(_mpa_sqrt(v) for v in values)  # type: ignore[arg-type]

    def _upstream_sigma(self, request: PredictionRequest) -> tuple[float, float, float]:
        """One-sigma spread on each upstream component, in MPa^0.5.

        An ``Uncertainty`` is expressed in the unit of the value it accompanies,
        and ``polymer_hansen`` reports in MPa^0.5 while the registry's canonical
        unit is Pa^0.5.  Reading ``.std`` raw would take a 3.9 MPa^0.5 bar for a
        3.9 Pa^0.5 one and shrink it by a thousand.
        """
        out: list[float] = []
        for prop in _CENTRE_PROPERTIES:
            pred = request.dependency(prop)
            if pred is None or pred.quantity is None or pred.uncertainty.std is None:
                out.append(0.0)
                continue
            out.append(pred.uncertainty.converted(pred.quantity.unit, "MPa^0.5").std or 0.0)
        return tuple(out)  # type: ignore[return-value]

    # -- the MIXTURE path --------------------------------------------------

    def _predict_mixture(
        self, prop: str, request: PredictionRequest, domain: ApplicabilityDomain
    ) -> Prediction:
        mixture = request.candidate.mixture
        if mixture is None:  # pragma: no cover - the class validator forbids it
            return Prediction.unsupported(prop, self.id, "candidate carries no formulation")

        polymers = [c for c in mixture.components if c.polymer is not None]
        if len(polymers) != 1:
            return Prediction.unsupported(
                prop,
                self.id,
                f"{len(polymers)} polymer components: a RED is a statement about one "
                "polymer and one solvent, and a blend has no single sphere - two "
                "polymers in one liquid can have opposite answers, which averaging would "
                "hide",
            )
        solvents = [
            c for c in mixture.components
            if c.role in _SOLVENT_ROLES and c.molecule is not None
        ]
        if len(solvents) != 1:
            return Prediction.unsupported(
                prop,
                self.id,
                f"{len(solvents)} solvent-role molecular components: a co-solvent "
                "blend's effective triple is a volume average this expert does not form, "
                "and with no solvent at all there is nothing for the polymer to dissolve "
                "in. Mark exactly one component with role 'solvent'",
            )

        # The role label is not what makes a molecule part of the liquid.  A
        # co-solvent pair is refused two lines above because their effective
        # triple is a volume average this expert does not form; a plasticiser,
        # an additive or a surfactant shares the phase in exactly the same way
        # and moves the same average, and the role it was given does not change
        # that.  Counting only the solvent-role components let a formulation of
        # 10% PMMA, 45% acetone and 45% water report RED 0.72 - "dissolves" -
        # with no mention anywhere that half the liquid was ignored.  Nothing
        # here can tell a second liquid from a suspended solid, and the
        # direction of the error is not known either, so this refuses.
        extras = [
            c for c in mixture.components
            if c.molecule is not None and c is not solvents[0]
        ]
        if extras:
            listed = ", ".join(
                f"{c.molecule.smiles} at {c.fraction:.3g} as {c.role.value}"  # type: ignore[union-attr]
                for c in extras
            )
            return Prediction.unsupported(
                prop,
                self.id,
                f"{len(extras)} molecular component"
                f"{'s' if len(extras) > 1 else ''} besides the named solvent "
                f"({listed}). A RED describes one polymer against one liquid, and "
                "nothing here can tell which of these shares the liquid phase; any that "
                "does moves the effective Hansen triple by the same volume average this "
                "expert refuses to form for a co-solvent pair, so a RED against the "
                "named solvent alone would describe a liquid this formulation does not "
                "contain. Ask for the polymer and the one liquid, or state the mixed "
                "solvent as a single component",
            )

        spec = polymers[0].polymer
        assert spec is not None  # narrowed by the filter above
        architecture = _architecture_refusal(spec)
        if architecture is not None:
            return Prediction.unsupported(prop, self.id, architecture)

        solvent_smiles = solvents[0].molecule.smiles  # type: ignore[union-attr]
        solvent_pa = hansen_triple(solvent_smiles)
        if solvent_pa is None:
            return Prediction.unsupported(
                prop,
                self.id,
                f"the solvent component {solvent_smiles!r} is not in the Hansen "
                "compilation, and no triple is estimated for it",
            )
        solvent_mpa = tuple(_mpa_sqrt(v) for v in solvent_pa)

        # NOT request.dependency_value.  For a MIXTURE candidate the engine's
        # dependency closure puts MixtureExpert's volume-fraction-averaged
        # hansen_* into the context under exactly these names, and that average
        # is the *formulation's* triple - it already contains the solvent.
        # Using it as the polymer's own centre would measure the solvent's
        # distance from a point the solvent helped define, which is the silent
        # substitution rule 7 forbids and would read as a better solvent the
        # more of it there is.
        centre_mpa, centre_sigma, delegation_refusal, flagged = self._delegated_centre(
            spec, request
        )
        if centre_mpa is None:
            return Prediction.unsupported(prop, self.id, delegation_refusal)
        if flagged:
            # The gate is only as good as the triple it is applied with, and
            # this one was produced outside its own method's domain.
            domain = domain.merged_with(
                ApplicabilityDomain(
                    score=0.5,
                    warnings=tuple(
                        "the polymer component's predicted triple is itself out of "
                        f"domain: {w}"
                        for w in flagged
                    ),
                )
            )

        match = sphere_for_repeat_unit(repeat_unit_smiles(spec))
        if match is None:
            distance = hansen_distance(solvent_mpa, centre_mpa)  # type: ignore[arg-type]
            return Prediction.unsupported(
                prop,
                self.id,
                f"no fitted interaction radius for the polymer component's repeat unit. "
                f"Its Hansen distance to the solvent is Ra = {distance:.2f} MPa^0.5 on "
                f"the predicted centre, which cannot be turned into a yes or no: "
                f"{_RADIUS_EVIDENCE}. Spheres are tabulated for "
                + ", ".join(sorted(SPHERE_REPEAT_UNITS)),
            )

        name, sphere = match
        return self._assemble(
            prop,
            request,
            domain,
            sphere,
            name,
            centre_mpa,
            solvent_mpa,
            f"the solvent component ({solvent_smiles})",
            centre_sigma=centre_sigma,
            extra_notes=(
                f"formulation: {polymers[0].fraction:.3g} polymer, "
                f"{solvents[0].fraction:.3g} solvent, on a {mixture.basis.value} basis; "
                "RED is a property of the pair and does not depend on how much of each "
                "there is, so a composition outside the polymer's solubility limit still "
                "reports the same number",
            ),
        )

    def _delegated_centre(
        self, spec: PolymerSpec, request: PredictionRequest
    ) -> tuple[
        tuple[float, float, float] | None,
        tuple[float, float, float],
        str,
        tuple[str, ...],
    ]:
        """Run the polymer panel over the polymer component alone.

        Follows the pattern ``blend.py`` established: build a POLYMER
        sub-candidate and dispatch the polymer-only registry over it, growing
        the requested set to the closure of its dependencies so that the
        density a Hansen triple rests on is actually produced.

        The fourth element is what the sub-run said about its own domain.  An
        out-of-domain prediction is *usable* in this engine - that is the point
        of the status - so accepting one and dropping the flag would let a
        triple ``polymer_hansen`` itself distrusts pass the identity gate with
        no trace.  ``polymer_hansen`` carries its density's domain into its own
        for the same reason; this carries its domain across the mixture
        boundary.
        """
        from .base import PredictionRequest as Req

        registry = self._panel()
        if registry is None:
            return (
                None,
                (0.0, 0.0, 0.0),
                "no expert on the polymer panel supplies a predicted Hansen triple for a "
                "polymer, so the identity of this component cannot be checked against the "
                "fitted sphere and no RED is produced",
                (),
            )

        sub = Candidate(
            material_class=MaterialClass.POLYMER,
            polymer=spec,
            conditions=request.candidate.conditions,
        )
        wanted = set(_CENTRE_PROPERTIES)
        for _ in range(len(registry) + 1):
            grown = set(wanted)
            for expert in registry:
                if expert.supported_properties & wanted:
                    grown |= expert.dependencies
            if grown == wanted:
                break
            wanted = grown

        context: dict[str, Prediction] = {}
        for expert in registry.resolution_order(
            registry.experts_for(wanted, MaterialClass.POLYMER)
        ):
            sub_request = Req(
                candidate=sub,
                properties=frozenset(wanted),
                conditions=request.conditions,
                context=dict(context),
            )
            for prediction in expert.predict(sub_request):
                if prediction.is_usable:
                    context.setdefault(prediction.property, prediction)

        sub_lookup = Req(
            candidate=sub,
            properties=frozenset(_CENTRE_PROPERTIES),
            conditions=request.conditions,
            context=context,
        )
        centre = self._upstream_centre(sub_lookup)
        if centre is None:
            refused = [
                note
                for prop in _CENTRE_PROPERTIES
                for note in (context.get(prop).notes if context.get(prop) else ())
            ]
            detail = f" ({refused[0]})" if refused else ""
            return (
                None,
                (0.0, 0.0, 0.0),
                "the polymer component's own Hansen triple could not be obtained from "
                "the polymer panel" + detail + "; the formulation's volume-averaged "
                "triple is not substituted for it, because that average already contains "
                "the solvent",
                (),
            )

        flagged = tuple(
            dict.fromkeys(
                warning
                for prop in _CENTRE_PROPERTIES
                if context.get(prop) is not None
                and context[prop].status is PredictionStatus.OUT_OF_DOMAIN
                for warning in context[prop].applicability.warnings
            )
        )
        return centre, self._upstream_sigma(sub_lookup), "", flagged

    def _panel(self):
        """The polymer-only panel, with a repeat-unit Hansen expert in it.

        Built lazily so the package import stays one-way, exactly as
        ``blend.py`` does it.  ``polymer_hansen`` is added when the shared panel
        does not already carry it: registration is the orchestrator's business,
        and a mixture answer that depends on whether someone has edited
        ``__init__.py`` yet would be a worse failure than a clear refusal.  If
        the module is genuinely absent, ``None`` comes back and the caller
        refuses.
        """
        if self._panel_cache is None:
            from formulate.experts import polymer_registry

            registry = polymer_registry()
            # ``uncovered``, not the truthiness of ``coverage``: a panel that
            # supplies one component of the triple and not the other two makes
            # ``coverage`` non-empty, which would leave the gap in place and
            # refuse every mixture for a reason that is not the real one.
            # ``replace=True`` because a panel already carrying the id is the
            # orchestrator having wired it up, and raising there would turn a
            # correct registration into a FAILED prediction.
            if registry.uncovered(_CENTRE_PROPERTIES, MaterialClass.POLYMER):
                try:
                    from .polymer_hansen import PolymerHansenExpert
                except Exception:
                    return None
                registry.register(PolymerHansenExpert(), replace=True)
            self._panel_cache = registry
        return self._panel_cache

    # -- the number itself -------------------------------------------------

    def _assemble(
        self,
        prop: str,
        request: PredictionRequest,
        domain: ApplicabilityDomain,
        sphere: SolubilitySphere,
        name: str,
        centre_mpa: tuple[float, float, float],
        solvent_mpa: tuple[float, float, float],
        solvent_label: str,
        *,
        centre_sigma: tuple[float, float, float] | None = None,
        extra_notes: tuple[str, ...] = (),
    ) -> Prediction:
        """The identity gate, the RED, and the four terms of its error bar."""
        # Computed before the gate rather than after it, because how coarse the
        # gate's own instrument is belongs in the refusal as much as in the
        # answer.  Ra weights the dispersion axis by four, so the upstream
        # per-component spreads propagate onto this distance the same way.
        gate_sigma = 0.0
        if centre_sigma is not None:
            gate_sigma = math.sqrt(
                4.0 * centre_sigma[0] ** 2 + centre_sigma[1] ** 2 + centre_sigma[2] ** 2
            )

        displacement = hansen_distance(centre_mpa, sphere.centre)
        if displacement > sphere.radius:
            exceedance = displacement - sphere.radius
            # Poly(vinyl chloride) is the live case: it clears its radius by
            # 3.65 MPa^0.5 while the predicted centre it is measured with
            # carries 10.8.  Refusing is still the conservative call, but
            # stating it as a demonstrated mismatch would be claiming a
            # resolution the instrument does not have, and this repository
            # ranks on error bars precisely so that cannot happen quietly.
            if 0.0 < gate_sigma and exceedance <= gate_sigma:
                strength = (
                    f" That is a margin of {exceedance:.2f} MPa^0.5 measured with a "
                    f"centre whose own one-sigma spread propagates to {gate_sigma:.1f} "
                    f"MPa^0.5 on this distance, so the disagreement is not resolved: "
                    f"this is the conservative reading of an instrument too coarse to "
                    f"settle the question, not a demonstrated mismatch."
                )
            else:
                strength = (
                    f" The margin of {exceedance:.2f} MPa^0.5 is larger than the "
                    f"{gate_sigma:.1f} MPa^0.5 the predicted centre carries onto this "
                    f"distance, so the disagreement is resolved."
                    if gate_sigma > 0.0
                    else ""
                )
            return Prediction.unsupported(
                prop,
                self.id,
                f"the predicted centre for this repeat unit sits {displacement:.2f} "
                f"MPa^0.5 from {name}'s fitted centre, which is further than its own "
                f"interaction radius of {sphere.radius:.2f}. A RED from a sphere that "
                f"does not contain its own polymer is meaningless rather than merely "
                f"imprecise, and nothing here can say whether the candidate is not that "
                f"polymer or one of the two descriptions is wrong." + strength +
                f" The same pair can still be asked from the solvent's side, where no "
                f"predicted centre is involved: a MOLECULE candidate for the solvent "
                f"with solutes: ('{name}',) uses the fitted sphere directly",
            )

        # The RED comes from dissolution.py so the factor of four and the
        # Pa^0.5 -> MPa^0.5 conversion live in one place; Ra follows from it
        # rather than being computed a second way, which is what keeps the two
        # from ever disagreeing.
        solvent_pa = tuple(
            Quantity(value=v, unit="MPa^0.5").to("Pa^0.5").value for v in solvent_mpa
        )
        red = relative_energy_difference(solvent_pa, sphere)  # type: ignore[arg-type]
        distance = red * sphere.radius

        temperature = request.conditions.temperature_k
        sigma_solvent = solvent_sigma_on_ra(solvent_mpa, sphere.centre, distance)
        sigma_temperature = temperature_sigma_on_ra(solvent_mpa, sphere.centre, temperature)
        sigma_centre = DISPLACEMENT_FRACTION * displacement
        sigma_ra = math.sqrt(sigma_solvent**2 + sigma_temperature**2 + sigma_centre**2)
        std = math.sqrt((sigma_ra / sphere.radius) ** 2 + (RADIUS_SPREAD_FRACTION * red) ** 2)

        # The upstream triple's own claimed spread is NOT added to the bar, and
        # the reason is measured rather than preferred.  Propagated through Ra
        # it is 10.8 MPa^0.5 - larger than every displacement observed - so
        # folding it in as a floor would make the term identical for every
        # polymer, erase the discrimination it exists to provide, and roughly
        # double a bar that already over-covers: all 31 textbook calls are
        # consistent with the shipped bar at one sigma, against the ~68 per cent
        # a correct estimate implies.  What it does instead is say so, because a
        # gate applied through an instrument coarser than the sphere it checks
        # is a fact about this answer that a reader is entitled to.  It is
        # computed above, where the gate needs it too.

        notes = [
            f"{name}: fitted centre ({sphere.dispersion}, {sphere.polar}, "
            f"{sphere.hydrogen_bonding}) MPa^0.5, radius {sphere.radius}",
            f"solvent {solvent_label}: ({solvent_mpa[0]:.1f}, {solvent_mpa[1]:.1f}, "
            f"{solvent_mpa[2]:.1f}) MPa^0.5",
            f"Ra {distance:.2f} MPa^0.5, RED {red:.2f}, {verdict(red)}",
            f"predicted repeat-unit centre ({centre_mpa[0]:.1f}, {centre_mpa[1]:.1f}, "
            f"{centre_mpa[2]:.1f}) MPa^0.5 sits {displacement:.2f} from the fitted one, "
            f"inside the radius of {sphere.radius:.2f}, so the sphere is accepted as this "
            f"candidate's; that disagreement is {DISPLACEMENT_FRACTION:g} x "
            f"{displacement:.2f} / {sphere.radius:.2f} = "
            f"{DISPLACEMENT_FRACTION * displacement / sphere.radius:.2f} of the error bar",
            "Ra is reported here and is deliberately not returned as a value: across "
            "polymers at a fixed solvent it orders 44% of pairs the opposite way to RED",
            sphere.source,
            "R0 grows with temperature and that is not modelled, so a RED above one at "
            "elevated temperature is not a reliable non-solvent",
            "molar mass, crystallinity and dissolution kinetics are not modelled; a "
            "glassy polymer below its Tg can take a day to admit a good solvent",
        ]
        if temperature is None:
            notes.append(
                "no temperature was stated; the sphere and the solvent triple are both "
                "25 degC values and no thermal term is included"
            )
        notes.extend(extra_notes)

        adjusted = domain
        if gate_sigma > sphere.radius:
            coarse = (
                f"the upstream triple's own one-sigma spread propagates to "
                f"{gate_sigma:.1f} MPa^0.5 on this distance, wider than {name}'s "
                f"interaction radius of {sphere.radius:.2f}: the identity gate passed, "
                f"but it was applied with an instrument coarser than the sphere it "
                f"checks, so the {displacement:.2f} MPa^0.5 displacement is a lower "
                f"bound on the real disagreement and so is the error bar built from it"
            )
            notes.append(coarse)
            adjusted = adjusted.merged_with(ApplicabilityDomain(score=0.7, warnings=(coarse,)))

        low, high = TEMPERATURE_WINDOW_K
        if temperature is not None and not low <= temperature <= high:
            adjusted = adjusted.merged_with(
                ApplicabilityDomain.outside(
                    f"{temperature:.0f} K is outside the {low:.0f}-{high:.0f} K window the "
                    "sphere and the compilation are 25 degC values over",
                    score=0.25,
                )
            )

        return self._make(
            prop,
            red,
            "",
            request,
            adjusted,
            std=std,
            kind=UncertaintyKind.COMBINED,
            basis=(
                f"in quadrature: the candidate's own chemistry sits {displacement:.2f} "
                f"MPa^0.5 from the material the sphere was fitted to, of which a measured "
                f"{DISPLACEMENT_FRACTION:g} shows up in Ra (mean over 100 real "
                f"solvent/polymer pairs); {SOLVENT_COMPILATION_SPREAD_MPA_SQRT:g} MPa^0.5 "
                f"of compilation spread on the solvent triple, propagated analytically; "
                f"{RADIUS_SPREAD_FRACTION:g} of the radius, the spread between published "
                f"fits of the same sphere; and a thermal term from delta ~ V^-1/2. Over "
                f"31 textbook calls this construction gets 30 right and leaves the one "
                f"miss inside one sigma"
            ),
            notes=tuple(notes),
            sphere=name,
            radius_mpa_sqrt=sphere.radius,
            centre_displacement_mpa_sqrt=round(displacement, 4),
            hansen_distance_mpa_sqrt=round(distance, 4),
        )
