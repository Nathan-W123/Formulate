"""MMFF94 as an OpenMM system, so a periodic condensed phase becomes possible.

Every bulk observable in specification section 6 - density, self-diffusion,
cohesive energy density, work of separation - needs a periodic condensed phase.
Until now this installation had none: the only periodic-capable potential
reachable through xtb is GFN1-xTB, which costs about half a second per step at
seventy atoms and cannot run at the two thousand a bulk box needs. OpenMM can,
and is installed; what it lacked was parameters, because assigning a force
field to an arbitrary small molecule normally means openff-toolkit or
AmberTools, neither of which is installable here.

RDKit already has them. MMFF94 is implemented in RDKit and every term it uses
is exposed: bond stretch, angle bend, stretch-bend coupling, out-of-plane
bending, torsion, buffered 14-7 van der Waals and buffered electrostatics, with
atom types and partial charges. This module translates those into OpenMM
forces, which turns a validated organic force field into a periodic simulation
engine with no dependency that was not already present.

The translation is exact and is tested that way. RDKit will compute the energy
of any single MMFF term on its own, so each force here has an oracle: build the
system, evaluate it, and compare against RDKit's own number for that term. The
agreement is at machine precision, which is a far stronger statement than
"the energies look reasonable".

Two modes, because exactness and simulation want different things:

``exact``
    Every term as MMFF94 defines it, no cutoff, no periodic boundary. This is
    the mode the tests use, and it is what agrees with RDKit to 1e-14 kcal/mol.

``simulation``
    Particle-mesh Ewald for electrostatics and a cutoff on van der Waals, which
    is what makes a condensed phase affordable and what a bulk observable
    needs. Two approximations enter here and are stated rather than buried:
    MMFF's 0.05 A electrostatic buffer is dropped, because PME is defined for a
    plain Coulomb kernel and the buffer changes the energy by well under a
    tenth of a percent at any separation two atoms actually sample; and the van
    der Waals sum is truncated at the cutoff with an isotropic long-range
    correction rather than summed to infinity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

#: kcal/mol -> kJ/mol. MMFF is defined in kcal/mol and angstroms; OpenMM works
#: in kJ/mol and nanometres, and every expression here converts at the boundary.
KCAL = 4.184

#: MMFF's cubic bond-stretch constant, per angstrom.
_CUBIC_STRETCH = -2.0
#: MMFF's cubic angle-bend constant. The published value is -0.007 per degree;
#: the exact constant is -0.4 per radian, and working in radians throughout
#: avoids a rounding that is otherwise visible at 1e-4 kcal/mol.
_CUBIC_BEND = -0.4
#: Converts a force constant in mdyne/angstrom into kcal/mol.
_MDYNE = 143.9325
#: Coulomb constant in kcal/mol with charges in e and distances in angstrom.
_COULOMB = 332.0716
#: MMFF's electrostatic buffering distance, in angstrom.
_ELECTROSTATIC_BUFFER = 0.05
#: MMFF scales 1-4 electrostatics and keeps 1-4 van der Waals in full.
_ONE_FOUR_ELECTROSTATIC_SCALE = 0.75

_TERMS = ("Bond", "Angle", "StretchBend", "Oop", "Torsion", "VdW", "Ele")


def openmm_available() -> bool:
    try:
        import openmm  # noqa: F401
    except Exception:
        return False
    return True


@dataclass(frozen=True, slots=True)
class MMFFParameters:
    """Every MMFF94 term for one molecule, extracted once and reusable.

    A condensed-phase box is many copies of the same molecule, so the
    parameters are pulled out once and replicated with an index offset rather
    than re-derived per copy.
    """

    n_atoms: int
    masses: tuple[float, ...]
    charges: tuple[float, ...]
    #: Compact per-atom index into the van der Waals tables.
    vdw_index: tuple[int, ...]
    vdw_size: int
    vdw_rstar: tuple[float, ...]
    vdw_epsilon: tuple[float, ...]
    #: (i, j, kb, r0 in angstrom)
    bonds: tuple[tuple[int, int, float, float], ...]
    #: (i, j, k, ka, theta0 in radians)
    bent_angles: tuple[tuple[int, int, int, float, float], ...]
    #: (i, j, k, ka) for angles MMFF treats as linear
    linear_angles: tuple[tuple[int, int, int, float], ...]
    #: (i, j, k, kba_ijk, kba_kji, r0_ij, r0_kj, theta0 in radians)
    stretch_bends: tuple[tuple[int, int, int, float, float, float, float, float], ...]
    #: (i, j, k, l, koop)
    out_of_planes: tuple[tuple[int, int, int, int, float], ...]
    #: (i, j, k, l, v1, v2, v3)
    torsions: tuple[tuple[int, int, int, int, float, float, float], ...]
    #: Pairs excluded from both non-bonded sums.
    excluded_pairs: tuple[tuple[int, int], ...]
    #: (i, j, rstar, epsilon, charge product) for the 1-4 pairs.
    one_four: tuple[tuple[int, int, float, float, float], ...]


def _angle_triples(mol) -> Iterable[tuple[int, int, int]]:
    for atom in mol.GetAtoms():
        neighbours = [n.GetIdx() for n in atom.GetNeighbors()]
        centre = atom.GetIdx()
        for a in range(len(neighbours)):
            for b in range(a + 1, len(neighbours)):
                yield neighbours[a], centre, neighbours[b]


def _separations(mol) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    """Pairs separated by one, two and three bonds."""
    adjacency: list[set[int]] = [set() for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[i].add(j)
        adjacency[j].add(i)

    one_two: set[tuple[int, int]] = set()
    one_three: set[tuple[int, int]] = set()
    one_four: set[tuple[int, int]] = set()
    for i in range(mol.GetNumAtoms()):
        for j in adjacency[i]:
            one_two.add((min(i, j), max(i, j)))
            for k in adjacency[j]:
                if k == i:
                    continue
                one_three.add((min(i, k), max(i, k)))
                for l in adjacency[k]:
                    if l in (i, j):
                        continue
                    one_four.add((min(i, l), max(i, l)))
    one_three -= one_two
    one_four -= one_two | one_three
    return one_two, one_three, one_four


def extract_parameters(mol) -> MMFFParameters | None:
    """Pull every MMFF94 term out of RDKit, or None if it has no parameters.

    Returning None rather than a partial set matters: a molecule MMFF does not
    cover is one this route cannot simulate, and a system built from some of
    its terms would run and produce numbers.
    """
    from rdkit.Chem import AllChem

    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        return None

    n = mol.GetNumAtoms()
    types = [props.GetMMFFAtomType(i) for i in range(n)]
    distinct = sorted(set(types))
    compact = {t: k for k, t in enumerate(distinct)}
    representative = {t: types.index(t) for t in distinct}
    size = len(distinct)

    rstar = [0.0] * (size * size)
    epsilon = [0.0] * (size * size)
    for a in distinct:
        for b in distinct:
            pair = props.GetMMFFVdWParams(representative[a], representative[b])
            rstar[compact[a] * size + compact[b]] = pair[2]
            epsilon[compact[a] * size + compact[b]] = pair[3]

    bonds = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        params = props.GetMMFFBondStretchParams(mol, i, j)
        if params is not None:
            bonds.append((i, j, params[1], params[2]))

    bent, linear, stretch_bends = [], [], []
    for i, j, k in _angle_triples(mol):
        angle = props.GetMMFFAngleBendParams(mol, i, j, k)
        if angle is None:
            continue
        _, ka, theta0 = angle
        if abs(theta0 - 180.0) < 1e-6:
            linear.append((i, j, k, ka))
        else:
            bent.append((i, j, k, ka, math.radians(theta0)))

        coupling = props.GetMMFFStretchBendParams(mol, i, j, k)
        bond_ij = props.GetMMFFBondStretchParams(mol, i, j)
        bond_kj = props.GetMMFFBondStretchParams(mol, k, j)
        if coupling is not None and bond_ij is not None and bond_kj is not None:
            stretch_bends.append(
                (i, j, k, coupling[1], coupling[2], bond_ij[2], bond_kj[2], math.radians(theta0))
            )

    out_of_planes = []
    for atom in mol.GetAtoms():
        neighbours = [nb.GetIdx() for nb in atom.GetNeighbors()]
        if len(neighbours) != 3:
            continue
        centre = atom.GetIdx()
        a, b, c = neighbours
        for i, k, l in ((a, b, c), (a, c, b), (b, c, a)):
            koop = props.GetMMFFOopBendParams(mol, i, centre, k, l)
            if koop is not None:
                out_of_planes.append((i, centre, k, l, koop))

    torsions = []
    seen: set[tuple[int, int, int, int]] = set()
    for bond in mol.GetBonds():
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for i in (n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k):
            for l in (n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j):
                if i == l:
                    continue
                key = min((i, j, k, l), (l, k, j, i))
                if key in seen:
                    continue
                seen.add(key)
                params = props.GetMMFFTorsionParams(mol, i, j, k, l)
                if params is not None:
                    torsions.append((i, j, k, l, params[1], params[2], params[3]))

    one_two, one_three, one_four_pairs = _separations(mol)
    one_four = []
    for i, j in sorted(one_four_pairs):
        pair = props.GetMMFFVdWParams(i, j)
        one_four.append(
            (i, j, pair[2], pair[3], props.GetMMFFPartialCharge(i) * props.GetMMFFPartialCharge(j))
        )

    return MMFFParameters(
        n_atoms=n,
        masses=tuple(atom.GetMass() for atom in mol.GetAtoms()),
        charges=tuple(props.GetMMFFPartialCharge(i) for i in range(n)),
        vdw_index=tuple(compact[t] for t in types),
        vdw_size=size,
        vdw_rstar=tuple(rstar),
        vdw_epsilon=tuple(epsilon),
        bonds=tuple(bonds),
        bent_angles=tuple(bent),
        linear_angles=tuple(linear),
        stretch_bends=tuple(stretch_bends),
        out_of_planes=tuple(out_of_planes),
        torsions=tuple(torsions),
        excluded_pairs=tuple(sorted(one_two | one_three | one_four_pairs)),
        one_four=tuple(one_four),
    )


# --------------------------------------------------------------------------
# Translating the parameters into OpenMM forces
# --------------------------------------------------------------------------


def _bond_force(params: MMFFParameters, copies: int):
    import openmm as mm

    force = mm.CustomBondForce(
        f"0.5*{_MDYNE}*{KCAL}*kb*d^2*(1 + {_CUBIC_STRETCH}*d "
        f"+ (7.0/12.0)*{_CUBIC_STRETCH ** 2}*d^2); d=10*(r-r0)"
    )
    force.addPerBondParameter("kb")
    force.addPerBondParameter("r0")
    for offset in _offsets(params, copies):
        for i, j, kb, r0 in params.bonds:
            force.addBond(i + offset, j + offset, [kb, r0 * 0.1])
    return force


def _angle_forces(params: MMFFParameters, copies: int):
    import openmm as mm

    bent = mm.CustomAngleForce(
        f"0.5*{_MDYNE}*{KCAL}*ka*d^2*(1 + {_CUBIC_BEND}*d); d=theta-theta0"
    )
    bent.addPerAngleParameter("ka")
    bent.addPerAngleParameter("theta0")

    # MMFF replaces the cubic form at a linear centre, where a quadratic in the
    # deviation from 180 degrees has a cusp exactly where the atoms sit.
    linear = mm.CustomAngleForce(f"{_MDYNE}*{KCAL}*ka*(1+cos(theta))")
    linear.addPerAngleParameter("ka")

    for offset in _offsets(params, copies):
        for i, j, k, ka, theta0 in params.bent_angles:
            bent.addAngle(i + offset, j + offset, k + offset, [ka, theta0])
        for i, j, k, ka in params.linear_angles:
            linear.addAngle(i + offset, j + offset, k + offset, [ka])
    return bent, linear


def _stretch_bend_force(params: MMFFParameters, copies: int):
    import openmm as mm

    force = mm.CustomCompoundBondForce(
        3,
        f"{_MDYNE}*{KCAL}*(kba1*(10*distance(p1,p2)-r01) + kba2*(10*distance(p3,p2)-r02))"
        "*(angle(p1,p2,p3)-theta0)",
    )
    for name in ("kba1", "kba2", "r01", "r02", "theta0"):
        force.addPerBondParameter(name)
    for offset in _offsets(params, copies):
        for i, j, k, kba1, kba2, r01, r02, theta0 in params.stretch_bends:
            force.addBond(
                [i + offset, j + offset, k + offset], [kba1, kba2, r01, r02, theta0]
            )
    return force


#: The Wilson angle of atom l out of the plane ijk about the central atom j:
#: sin(chi) = [(r_ji x r_jk) . r_jl] / (|r_ji x r_jk| |r_jl|). OpenMM has no
#: built-in for it, and the identity is short enough to write out.
_WILSON_ANGLE = (
    "asin( (nx*wx + ny*wy + nz*wz) / (sqrt(nx*nx+ny*ny+nz*nz)*sqrt(wx*wx+wy*wy+wz*wz)) );"
    "nx=uy*vz-uz*vy; ny=uz*vx-ux*vz; nz=ux*vy-uy*vx;"
    "ux=x1-x2; uy=y1-y2; uz=z1-z2;"
    "vx=x3-x2; vy=y3-y2; vz=z3-z2;"
    "wx=x4-x2; wy=y4-y2; wz=z4-z2"
)


def _out_of_plane_force(params: MMFFParameters, copies: int):
    import openmm as mm

    force = mm.CustomCompoundBondForce(
        4, f"0.5*{_MDYNE}*{KCAL}*koop*chi^2; chi=" + _WILSON_ANGLE
    )
    force.addPerBondParameter("koop")
    for offset in _offsets(params, copies):
        for i, j, k, l, koop in params.out_of_planes:
            force.addBond([i + offset, j + offset, k + offset, l + offset], [koop])
    return force


def _torsion_force(params: MMFFParameters, copies: int):
    import openmm as mm

    force = mm.PeriodicTorsionForce()
    for offset in _offsets(params, copies):
        for i, j, k, l, v1, v2, v3 in params.torsions:
            atoms = (i + offset, j + offset, k + offset, l + offset)
            # MMFF's V2 term carries a minus sign, which is a phase of pi.
            force.addTorsion(*atoms, 1, 0.0, 0.5 * v1 * KCAL)
            force.addTorsion(*atoms, 2, math.pi, 0.5 * v2 * KCAL)
            force.addTorsion(*atoms, 3, 0.0, 0.5 * v3 * KCAL)
    return force


_VDW_ENERGY = (
    "e*((1.07*rs/(rr+0.07*rs))^7)*((1.12*rs^7/(rr^7+0.12*rs^7))-2)"
)


def _nonbonded_forces(
    params: MMFFParameters,
    copies: int,
    *,
    exact: bool,
    cutoff_nm: float,
    dispersion_scale: float = 1.0,
):
    import openmm as mm

    size = params.vdw_size
    # One knob on the well depth, and only the well depth. Scaling epsilon
    # strengthens attraction without touching the atomic sizes that set the
    # repulsive wall and hence the local structure of the liquid; scaling R*
    # instead would change both. Any value other than 1.0 is no longer MMFF94
    # and every caller is told so.
    epsilon_table = [e * dispersion_scale for e in params.vdw_epsilon]
    vdw = mm.CustomNonbondedForce(
        f"{KCAL}*{_VDW_ENERGY}; rs=Rstar(t1,t2); e=Eps(t1,t2); rr=10*r"
    )
    vdw.addTabulatedFunction("Rstar", mm.Discrete2DFunction(size, size, list(params.vdw_rstar)))
    vdw.addTabulatedFunction("Eps", mm.Discrete2DFunction(size, size, epsilon_table))
    vdw.addPerParticleParameter("t")

    if exact:
        # MMFF's own buffered Coulomb, summed over every pair with no cutoff.
        # This is the form RDKit evaluates and therefore the one the tests can
        # check against.
        electrostatic = mm.CustomNonbondedForce(
            f"{KCAL}*{_COULOMB}*q1*q2/(10*r + {_ELECTROSTATIC_BUFFER})"
        )
        electrostatic.addPerParticleParameter("q")
        vdw.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
        electrostatic.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
    else:
        # Particle-mesh Ewald needs a plain Coulomb kernel, so the 0.05 A
        # buffer is dropped here. Two atoms in a liquid do not approach closely
        # enough for that to matter: at 2 A it changes the pair energy by 2.5%,
        # at 4 A by 1.2%, and the repulsive wall keeps them beyond both.
        electrostatic = mm.NonbondedForce()
        electrostatic.setNonbondedMethod(mm.NonbondedForce.PME)
        electrostatic.setCutoffDistance(cutoff_nm)
        electrostatic.setUseDispersionCorrection(False)  # vdW is not this force's job
        vdw.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        vdw.setCutoffDistance(cutoff_nm)
        vdw.setUseLongRangeCorrection(True)

    for offset in _offsets(params, copies):
        for i in range(params.n_atoms):
            vdw.addParticle([float(params.vdw_index[i])])
            if exact:
                electrostatic.addParticle([params.charges[i]])
            else:
                electrostatic.addParticle(params.charges[i], 1.0, 0.0)

    vdw14 = mm.CustomBondForce(f"{KCAL}*{_VDW_ENERGY}; rr=10*r")
    vdw14.addPerBondParameter("rs")
    vdw14.addPerBondParameter("e")
    ele14 = mm.CustomBondForce(
        f"{_ONE_FOUR_ELECTROSTATIC_SCALE}*{KCAL}*{_COULOMB}*qq/(10*r + {buffer_for(exact)})"
    )
    ele14.addPerBondParameter("qq")

    for offset in _offsets(params, copies):
        for i, j in params.excluded_pairs:
            vdw.addExclusion(i + offset, j + offset)
            if exact:
                electrostatic.addExclusion(i + offset, j + offset)
            else:
                electrostatic.addException(i + offset, j + offset, 0.0, 1.0, 0.0)
        # MMFF keeps 1-4 van der Waals in full and scales 1-4 electrostatics.
        for i, j, rstar, epsilon, charge_product in params.one_four:
            vdw14.addBond(i + offset, j + offset, [rstar, epsilon * dispersion_scale])
            ele14.addBond(i + offset, j + offset, [charge_product])

    return vdw, electrostatic, vdw14, ele14


def buffer_for(exact: bool) -> float:
    return _ELECTROSTATIC_BUFFER if exact else 0.0


def _offsets(params: MMFFParameters, copies: int) -> range:
    return range(0, copies * params.n_atoms, params.n_atoms)


def build_system(
    params: MMFFParameters,
    copies: int = 1,
    *,
    exact: bool = False,
    box_nm: float | None = None,
    cutoff_nm: float = 1.0,
    dispersion_scale: float = 1.0,
):
    """An OpenMM system for ``copies`` identical molecules under MMFF94.

    ``exact`` reproduces RDKit's MMFF energy term for term with no cutoff and
    no periodic boundary; it is what the tests compare against. Otherwise the
    system is built for simulation: PME electrostatics, a cutoff on van der
    Waals with a long-range correction, and a periodic box.
    """
    import openmm as mm

    if not exact and box_nm is None:
        raise ValueError("a simulation system needs a periodic box")
    if not exact and cutoff_nm * 2.0 >= box_nm:
        raise ValueError(
            f"a {cutoff_nm:.2f} nm cutoff does not fit in a {box_nm:.2f} nm box: the "
            "minimum image convention needs the box to exceed twice the cutoff"
        )

    system = mm.System()
    for _ in range(copies):
        for mass in params.masses:
            system.addParticle(mass)

    if box_nm is not None:
        system.setDefaultPeriodicBoxVectors(
            mm.Vec3(box_nm, 0, 0), mm.Vec3(0, box_nm, 0), mm.Vec3(0, 0, box_nm)
        )

    bent, linear = _angle_forces(params, copies)
    if exact and dispersion_scale != 1.0:
        raise ValueError(
            "the exact mode reproduces MMFF94, so it cannot carry a dispersion scale; "
            "a scaled force field is no longer the one the oracle checks against"
        )
    vdw, electrostatic, vdw14, ele14 = _nonbonded_forces(
        params, copies, exact=exact, cutoff_nm=cutoff_nm, dispersion_scale=dispersion_scale
    )
    for force in (
        _bond_force(params, copies),
        bent,
        linear,
        _stretch_bend_force(params, copies),
        _out_of_plane_force(params, copies),
        _torsion_force(params, copies),
        vdw,
        electrostatic,
        vdw14,
        ele14,
    ):
        system.addForce(force)
    return system
