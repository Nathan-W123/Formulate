"""The molecular-dynamics engine.

Specification section 6, "Core integration: compute forces -> integrate
nuclear motion -> apply ensemble controls/boundaries -> sample observables."
ASE supplies the integrators, so this module does the parts ASE does not:
build the system, decide whether the run can mean anything, discard the
equilibration transient, and report sampling error that accounts for the
correlation between frames.

The feasibility gate is not a convenience.  Every bulk protocol in section 6
needs a periodic condensed phase, and on this installation the only
periodic-capable potential fails above roughly seventy atoms.  A density from
eight molecules is not a rough density; the error is systematic, so a wide
error bar would not rescue it.  Such requests are refused with a cost estimate
for what an adequate run would take.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from formulate.core.provenance import ProvenanceKind, ProvenanceRecord, SoftwareEnvironment
from formulate.core.quantity import Quantity

from ..geometry import EV_TO_J_PER_MOL, MolecularGeometry
from ..quiet import suppress_native_output
from .base import (
    REQUIREMENTS,
    Ensemble,
    FeasibilityVerdict,
    MDProtocol,
    MDRequest,
    MDResult,
)
from .calculators import CalculatorChoice, attach_calculator, choose_calculator
from .statistics import BlockAverage, block_average, detect_equilibration

_COMMON_LIMITATIONS = (
    "a classical trajectory with fixed topology cannot describe bond breaking or "
    "forming, and carries no electronic information (specification section 6)",
    "sampling error is reported from block averaging; finite-size error is separate, "
    "systematic, and does not shrink with longer sampling",
)


class MDEngine:
    """Runs section 6's property protocols, or explains why it will not."""

    id = "ase-md"
    version = "1"

    def __init__(self, calculator: str | None = None) -> None:
        self.default_calculator = calculator

    # -- feasibility -------------------------------------------------------

    def assess(self, request: MDRequest) -> FeasibilityVerdict:
        """Decide whether this protocol can produce a meaningful number here."""
        requirement = REQUIREMENTS[request.protocol]
        choice = self._calculator_for(request, periodic=requirement.needs_periodic)

        if choice is None:
            adequate = self._adequate_cost(request, requirement, None)
            return FeasibilityVerdict(
                feasible=False,
                reason=(
                    f"{request.protocol.value} needs a periodic condensed phase, and no "
                    "installed potential supports a periodic cell for this chemistry. "
                    f"{requirement.rationale}"
                    if requirement.needs_periodic
                    else "no force provider is available at all"
                ),
                adequate_cost_seconds=adequate,
            )

        if requirement.needs_periodic:
            # The periodic-capable potential here fails well below the system
            # size a bulk observable needs, so the shortfall is in the system,
            # not merely the run length.
            adequate = self._adequate_cost(request, requirement, choice)
            return FeasibilityVerdict(
                feasible=False,
                reason=(
                    f"{request.protocol.value} needs at least {requirement.min_molecules} "
                    f"molecules in a periodic cell sampled for {requirement.min_production_ps:.0f} ps. "
                    f"The only periodic-capable potential available ({choice.label}) costs about "
                    f"{choice.reference_ms_per_evaluation:.0f} ms per step at a few tens of atoms "
                    "and does not run at that system size at all. "
                    f"{requirement.rationale}"
                ),
                adequate_cost_seconds=adequate,
                affordable_description=(
                    "what is affordable here is a single-molecule or small-cluster run, "
                    "which answers a different question"
                ),
            )

        shortfall = request.production_ps < requirement.min_production_ps
        return FeasibilityVerdict(
            feasible=True,
            reason=(
                f"a non-periodic {request.protocol.value} run is supported by {choice.label}"
                + (
                    f", though {request.production_ps:.1f} ps is short of the "
                    f"{requirement.min_production_ps:.0f} ps this observable normally needs"
                    if shortfall
                    else ""
                )
            ),
            adequate_cost_seconds=self._adequate_cost(request, requirement, choice),
        )

    def _adequate_cost(
        self, request: MDRequest, requirement: Any, choice: CalculatorChoice | None
    ) -> float | None:
        """Wall time a scientifically adequate run would need.

        Deliberately reported even when the run is refused: "not feasible" is
        far more useful with a number attached, and it tells a reader what
        hardware would change the answer.
        """
        if choice is None:
            return None
        atoms_per_molecule = max(1, request.geometry.n_atoms)
        atoms = atoms_per_molecule * requirement.min_molecules
        steps = requirement.min_production_ps * 1000.0 / request.timestep_fs
        # Cost per step grows at least linearly with system size; the reference
        # figure is for a molecule of roughly ten to twenty atoms.
        scale = atoms / 15.0
        return steps * (choice.reference_ms_per_evaluation / 1000.0) * scale

    def _calculator_for(
        self, request: MDRequest, periodic: bool
    ) -> CalculatorChoice | None:
        return choose_calculator(
            periodic=periodic, prefer=request.calculator or self.default_calculator
        )

    # -- execution ---------------------------------------------------------

    def run(self, request: MDRequest, rdkit_mol: Any = None) -> MDResult:
        verdict = self.assess(request)
        requirement = REQUIREMENTS[request.protocol]

        if not verdict.feasible and not request.force_run:
            return MDResult(
                protocol=request.protocol,
                calculator="none",
                feasible=False,
                feasibility_reason=verdict.describe(),
                limitations=_COMMON_LIMITATIONS,
                provenance=self._provenance(request, "refused"),
            )

        choice = self._calculator_for(request, periodic=requirement.needs_periodic)
        if choice is None:
            return MDResult(
                protocol=request.protocol,
                calculator="none",
                feasible=False,
                feasibility_reason="no force provider is available",
                limitations=_COMMON_LIMITATIONS,
                provenance=self._provenance(request, "no-calculator"),
            )

        started = time.perf_counter()
        try:
            if request.protocol is MDProtocol.COHESIVE_ENERGY:
                result = self._cohesive_energy(request, choice, rdkit_mol)
            else:
                result = self._conformational_ensemble(request, choice, rdkit_mol)
        except Exception as exc:
            return MDResult(
                protocol=request.protocol,
                calculator=choice.label,
                feasible=True,
                feasibility_reason=verdict.describe(),
                diagnostics=(f"{type(exc).__name__}: {exc}",),
                limitations=_COMMON_LIMITATIONS,
                wall_time_seconds=time.perf_counter() - started,
                provenance=self._provenance(request, choice.label),
            )

        return result.model_copy(
            update={
                "feasibility_reason": verdict.describe(),
                "wall_time_seconds": time.perf_counter() - started,
                "provenance": self._provenance(request, choice.label),
            }
        )

    # -- protocols ---------------------------------------------------------

    def _conformational_ensemble(
        self, request: MDRequest, choice: CalculatorChoice, rdkit_mol: Any
    ) -> MDResult:
        """Sample one molecule's conformational space at temperature.

        The reported observable is the radius of gyration, which is the
        cheapest honest summary of how extended the ensemble is. Whether the
        run actually sampled the ensemble is a separate question, answered by
        the sampling diagnostics rather than assumed from the run length.
        """
        atoms, trajectory = self._simulate(request, choice, request.geometry, rdkit_mol)
        radii = np.array([_radius_of_gyration(frame, request.geometry) for frame in trajectory])
        energies = np.array([e for _, e in trajectory]) if trajectory else np.array([])

        discard = detect_equilibration(radii) if radii.size > 10 else 0
        sampled = radii[discard:]
        statistics = block_average(sampled)

        diagnostics: list[str] = []
        requirement = REQUIREMENTS[request.protocol]
        if request.production_ps < requirement.min_production_ps:
            diagnostics.append(
                f"{request.production_ps:.1f} ps of production sampling against the "
                f"{requirement.min_production_ps:.0f} ps this observable normally needs: "
                "the trajectory probably explored one conformational well rather than "
                "the ensemble, and the mean describes that well"
            )
        diagnostics.extend(statistics.notes)

        return MDResult(
            protocol=request.protocol,
            calculator=choice.label,
            feasible=True,
            feasibility_reason="",
            value=Quantity(value=float(statistics.mean), unit="angstrom"),
            sampling=statistics,
            observables={
                "radius_of_gyration": statistics,
                "potential_energy_ev": block_average(energies[discard:])
                if energies.size > discard + 2
                else None,
            },
            discarded_frames=discard,
            production_ps=request.production_ps,
            n_molecules=1,
            diagnostics=tuple(diagnostics),
            limitations=_COMMON_LIMITATIONS
            + (
                "a radius of gyration averaged over an unconverged trajectory describes "
                "the sampled region, not the molecule",
            ),
        )

    def _cohesive_energy(
        self, request: MDRequest, choice: CalculatorChoice, rdkit_mol: Any
    ) -> MDResult:
        """Binding energy per molecule of a finite cluster.

        E_coh = (n * E_isolated - E_cluster) / n, positive for a bound cluster.
        A cluster is not a bulk phase: its molecules are disproportionately at
        the surface, so this underestimates the bulk cohesive energy by an
        amount set by the cluster size and not by the sampling.
        """
        from ase import Atoms

        n = max(2, request.n_molecules)
        cluster_geometry = build_cluster(request.geometry, n, seed=request.seed)
        cluster_geometry = self._relax(cluster_geometry, choice)

        _, cluster_traj = self._simulate(request, choice, cluster_geometry, None)
        cluster_energies = np.array([e for _, e in cluster_traj])

        # The reference must be sampled at the SAME temperature as the cluster.
        # Comparing a minimised isolated molecule against a thermally excited
        # cluster charges the whole intramolecular vibrational energy of every
        # molecule against the binding energy: for ethanol at 250 K that turned
        # a roughly +40 kJ/mol cohesion into -10 kJ/mol, reversing the sign.
        isolated_request = request.model_copy(
            update={"protocol": MDProtocol.CONFORMATIONAL_ENSEMBLE, "n_molecules": 1}
        )
        _, isolated_traj = self._simulate(
            isolated_request, choice, request.geometry, rdkit_mol
        )
        isolated_energies = np.array([e for _, e in isolated_traj])
        isolated_discard = (
            detect_equilibration(isolated_energies) if isolated_energies.size > 10 else 0
        )
        isolated_statistics = block_average(isolated_energies[isolated_discard:])
        isolated_energy = isolated_statistics.mean

        discard = detect_equilibration(cluster_energies) if cluster_energies.size > 10 else 0
        sampled = cluster_energies[discard:]
        cluster_statistics = block_average(sampled)

        cohesive_ev = (n * isolated_energy - cluster_statistics.mean) / n
        # Both averages carry sampling error; they are independent runs, so the
        # errors add in quadrature rather than the cluster's alone being used.
        error_ev = math.sqrt(
            (cluster_statistics.standard_error / n) ** 2
            + (isolated_statistics.standard_error or 0.0) ** 2
        )

        surface_fraction = _surface_fraction(n)
        diagnostics = [
            f"a {n}-molecule cluster has roughly {surface_fraction:.0%} of its molecules "
            "at the surface, so this underestimates the bulk cohesive energy by a "
            "systematic amount that longer sampling cannot reduce",
            f"the isolated-molecule reference was sampled at the same {request.temperature_k:.0f} K "
            "so that intramolecular thermal energy cancels between the two averages",
        ]
        diagnostics.extend(cluster_statistics.notes)
        diagnostics.extend(f"isolated reference: {note}" for note in isolated_statistics.notes)

        return MDResult(
            protocol=request.protocol,
            calculator=choice.label,
            feasible=True,
            feasibility_reason="",
            value=Quantity(value=cohesive_ev * EV_TO_J_PER_MOL, unit="J/mol"),
            sampling=BlockAverage(
                mean=cohesive_ev * EV_TO_J_PER_MOL,
                standard_error=error_ev * EV_TO_J_PER_MOL,
                naive_standard_error=cluster_statistics.naive_standard_error / n * EV_TO_J_PER_MOL,
                n_samples=cluster_statistics.n_samples,
                n_blocks=cluster_statistics.n_blocks,
                block_size=cluster_statistics.block_size,
                inflation=cluster_statistics.inflation,
                converged=cluster_statistics.converged,
            ),
            observables={
                "cluster_energy_ev": cluster_statistics,
                "isolated_energy_ev": isolated_statistics,
            },
            discarded_frames=discard,
            production_ps=request.production_ps,
            n_molecules=n,
            diagnostics=tuple(diagnostics),
            limitations=_COMMON_LIMITATIONS
            + (
                "a finite cluster is not a condensed phase; specification section 13 "
                "forbids presenting this as a validated bulk property",
            ),
        )

    # -- integration -------------------------------------------------------

    def _relax(
        self, geometry: MolecularGeometry, choice: CalculatorChoice, steps: int = 200
    ) -> MolecularGeometry:
        """Minimise a freshly built lattice before dynamics.

        A lattice starts with atoms closer than any equilibrium contact and
        with an artificial orientational order. Starting dynamics from it
        converts that strain into kinetic energy and heats the cluster far
        above the requested temperature, so it is minimised first.
        """
        from ase.optimize import FIRE

        atoms = geometry.to_ase()
        with suppress_native_output():
            attach_calculator(atoms, choice, None)
            try:
                FIRE(atoms, logfile=None).run(fmax=0.5, steps=steps)
            except Exception:
                # A failed relaxation is not fatal; the equilibration phase
                # still runs, and the diagnostics will show if it did not settle.
                pass
            positions = np.asarray(atoms.get_positions(), dtype=float)
        return geometry.with_positions(positions, source=geometry.source + ", relaxed")

    def _simulate(
        self,
        request: MDRequest,
        choice: CalculatorChoice,
        geometry: MolecularGeometry,
        rdkit_mol: Any,
    ) -> tuple[Any, list[tuple[np.ndarray, float]]]:
        """Equilibrate, then run production while sampling frames."""
        from ase import units
        from ase.md.langevin import Langevin
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.verlet import VelocityVerlet

        atoms = geometry.to_ase()
        trajectory: list[tuple[np.ndarray, float]] = []

        with suppress_native_output():
            attach_calculator(atoms, choice, rdkit_mol)
            MaxwellBoltzmannDistribution(
                atoms, temperature_K=request.temperature_k, rng=np.random.default_rng(request.seed)
            )

            if request.ensemble is Ensemble.NVE:
                dynamics = VelocityVerlet(atoms, request.timestep_fs * units.fs)
            else:
                # Langevin gives canonical sampling and, unlike a rescaling
                # thermostat, does not distort the fluctuations that the block
                # averaging then has to interpret.
                dynamics = Langevin(
                    atoms,
                    request.timestep_fs * units.fs,
                    temperature_K=request.temperature_k,
                    friction=request.friction,
                    rng=np.random.default_rng(request.seed + 1),
                )

            if request.equilibration_steps > 0:
                dynamics.run(request.equilibration_steps)

            def sample() -> None:
                trajectory.append(
                    (
                        np.asarray(atoms.get_positions(), dtype=float).copy(),
                        float(atoms.get_potential_energy()),
                    )
                )

            dynamics.attach(sample, interval=max(1, request.sample_interval))
            dynamics.run(request.production_steps)

        return atoms, trajectory

    def _provenance(self, request: MDRequest, calculator: str) -> ProvenanceRecord:
        return ProvenanceRecord(
            kind=ProvenanceKind.SIMULATION,
            producer=self.id,
            producer_version=self.version,
            parameters={
                "protocol": request.protocol.value,
                "calculator": calculator,
                "ensemble": request.ensemble.value,
                "temperature_k": request.temperature_k,
                "timestep_fs": request.timestep_fs,
                "equilibration_steps": request.equilibration_steps,
                "production_steps": request.production_steps,
                "n_molecules": request.n_molecules,
                "seed": request.seed,
                "geometry": request.geometry.identity_payload(),
            },
            software=SoftwareEnvironment.capture(),
        )


def build_cluster(
    geometry: MolecularGeometry,
    n_molecules: int,
    *,
    seed: int = 0,
    target_density_g_cm3: float = 0.85,
) -> MolecularGeometry:
    """Place copies of a molecule on a lattice at a target density.

    There is no packmol here, so this is a cubic lattice. What matters is the
    lattice spacing, and it must come from a density rather than from the
    molecule's own extent times a margin: sizing it from the extent produced
    209 cubic Angstrom per ethanol molecule against 97 for the liquid, which
    is a gas. Molecules that far apart barely interact, and the cohesive
    energy measured from such a cluster came out thirty times too small.

    The spacing is therefore derived from the molar mass and the requested
    density, defaulting slightly below a typical organic liquid so the
    starting lattice has room to relax inward rather than exploding out of an
    overlap.
    """
    positions = np.asarray(geometry.positions, dtype=float)
    centre = positions.mean(axis=0)
    centred = positions - centre

    molar_mass = sum(_MASSES.get(symbol, 12.0) for symbol in geometry.symbols)
    avogadro = 6.02214076e23
    # g/mol / (g/cm^3 * /mol) -> cm^3, then to cubic Angstrom.
    volume_per_molecule = molar_mass / (target_density_g_cm3 * avogadro) * 1e24
    step = float(volume_per_molecule ** (1.0 / 3.0))

    # A lattice cell smaller than the molecule guarantees overlapping atoms and
    # a force blow-up, so never go below the molecule's own footprint.
    footprint = float(np.abs(centred).max()) * 2.0 * 0.75
    step = max(step, footprint)

    side = int(math.ceil(n_molecules ** (1 / 3)))
    rng = np.random.default_rng(seed)

    symbols: list[str] = []
    coordinates: list[np.ndarray] = []
    placed = 0
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if placed >= n_molecules:
                    break
                offset = np.array([i, j, k], dtype=float) * step
                # A small jitter stops every copy sharing an orientation, which
                # would give the cluster an artificial crystalline order.
                jitter = rng.normal(scale=step * 0.05, size=3)
                symbols.extend(geometry.symbols)
                coordinates.extend(centred + offset + jitter)
                placed += 1

    return MolecularGeometry(
        symbols=tuple(symbols),
        positions=np.array(coordinates, dtype=float),
        charge=geometry.charge * n_molecules,
        spin_multiplicity=1,
        source=(
            f"{n_molecules}-molecule cubic cluster at {target_density_g_cm3:g} g/cm^3 "
            f"from {geometry.source or 'input geometry'}"
        ),
        notes=(
            f"lattice spacing {step:.2f} Angstrom, sized from the target density; "
            "equilibration is what makes it a liquid-like cluster rather than a crystal",
        ),
    )


def _radius_of_gyration(frame: tuple[np.ndarray, float], geometry: MolecularGeometry) -> float:
    positions = frame[0]
    masses = np.array([_MASSES.get(s, 12.0) for s in geometry.symbols], dtype=float)
    if positions.shape[0] != masses.size:
        masses = np.ones(positions.shape[0])
    total = masses.sum()
    centre = (positions * masses[:, None]).sum(axis=0) / total
    squared = (((positions - centre) ** 2).sum(axis=1) * masses).sum() / total
    return float(np.sqrt(squared))


def _surface_fraction(n_molecules: int) -> float:
    """Rough fraction of molecules on the surface of a compact cluster."""
    radius = (3.0 * n_molecules / (4.0 * math.pi)) ** (1 / 3)
    if radius <= 1.0:
        return 1.0
    interior = max(0.0, (radius - 0.5) ** 3 / radius**3)
    return 1.0 - interior


_MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Br": 79.904, "I": 126.904,
}
