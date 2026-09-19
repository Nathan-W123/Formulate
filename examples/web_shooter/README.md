# Web shooter: a stress test of the engine

A deliberately hard target, chosen because it lands on the parts of the engine
that are least well covered: a fluid that leaves a nozzle, becomes a solid
strand in flight, sticks hard to whatever it hits, and holds 30-50 lb.

The interesting output is not the ranking. It is the three places the engine
declined to answer, and the one place it caught a mistake in the specification.

Reproduce with:

```bash
formulate run examples/web_shooter/carrier.yaml --top 5
formulate run examples/web_shooter/strand.yaml  --top 5   # proposes nothing; see below
python examples/web_shooter/score_strand.py examples/web_shooter/strand.yaml 5
```

## Turning the ask into numbers

The engine does not do this part - section 5 puts it in the reasoning layer -
so it is written down here instead.

50 lb is 222 N. On a round strand that is:

| strand diameter | working stress | with a 2x snatch load |
|---|---|---|
| 1.0 mm | 283 MPa | 566 MPa |
| 1.5 mm | 126 MPa | 252 MPa |
| 2.0 mm | 71 MPa | 142 MPa |
| 3.0 mm | 32 MPa | 63 MPa |

A cast bar of any common glassy polymer breaks between 45 and 80 MPa. So the
1 mm strand of the films is not available from an undrawn polymer, and **the
small radius is the first thing that has to go.**

The device is a dry-spinning line: a polymer dissolved in a solvent, extruded,
the solvent flashing off in flight. That is how cellulose acetate and acrylic
fibres have been spun since the 1920s, and it is the only route that stores as
a pumpable fluid and arrives as a solid. It splits the problem in two, which is
what the two specifications here are.

## What the engine answered

**Carrier solvent** (`carrier.yaml`, molecule class, 50 reference compounds).
Three requirements, one per job the solvent does: dissolve the strand polymer
(`solubility_red`, from `conditions.solutes`), wet the target
(`work_of_separation`, from `conditions.surfaces`), and leave before the strand
lands (`normal_boiling_point`).

4 of 44 candidates feasible. Ranked: **acetone**, ethyl acetate, 2-butanone,
THF. Acetone is the solvent the acetate-fibre industry actually dry-spins from,
which is a reasonable sign the pipeline is not lost.

**Strand polymer** (`strand.yaml`, polymer class, 57 reference polymers).
2 of 57 feasible: **polystyrene** and **PMMA** - which are the only two
polymers in the bundled set that are both glassy at 25 degC and have tabulated
chain dimensions, so the result is as much a statement about coverage as about
polymers. Polystyrene leads; PMMA grips steel harder but is denser and closer
to its glass transition.

## The three refusals

**1. Adhesion cannot be scored for any solvent that works** - and asking the
solvent was the wrong question. Resolved; the history is kept because the
diagnosis is the useful part.

`work_of_separation` comes from the Owens-Wendt expert, which needs a measured
dispersive/polar split for the liquid. Ten liquids have one. Seven polymers
have a tabulated Hansen sphere. The overlap - liquids that both have an
Owens-Wendt split and dissolve one of those polymers - is **empty**:

```
polystyrene                    NONE
poly(methyl methacrylate)      NONE
poly(vinyl chloride)           NONE
polycarbonate                  NONE
poly(vinyl acetate)            NONE
cellulose acetate              NONE
polyamide 66                   NONE
```

So every feasible carrier reports `work_of_separation` as not available, and
the adhesion requirement is inert. Sweeping the substrate over steel, glass and
polyethylene changes the ranking not at all - the engine cannot see the
substrate, because it cannot see the liquid.

Making the requirement hard turns this into a proper refusal, and the failure
diagnosis names the gap without being told about it:

```
Constraints never satisfied together by any candidate in this pool:
  - solubility_red [hard]: in range [-inf, 1] and
    work_of_separation [hard]: in range [0.06, +inf] newton / meter
```

It also distinguishes the two ways a candidate can fail - "eliminated 39
candidate(s), 37 of them because the value could not be predicted" - which is
the difference between a bad solvent and an unknown one.

**The fix, and why it needed no new measurements.** The solvent is not what
holds. It wets the target for the instant before it evaporates; what is left
gripping the wall is the solid strand. So the adhesion question belongs to the
polymer, and eight polymer surface energies were already in the repository -
sitting in the substrate table, usable only as the thing being stuck *to*.

The expert now accepts a polymer candidate and reads that same table from the
other side: a clean polystyrene surface has one surface energy whether the
polystyrene is the wall or the web. Same Owens-Wendt physics, same measured
data, no new correlation. It still refuses an untabulated polymer, and it
refuses copolymers outright, because the lower-energy unit enriches at the
surface and a mole-weighted average would be biased in a known direction.

With `surfaces: [steel]` in `strand.yaml`, the strand ranking now scores it:

| polymer | steel | glass | polyethylene |
|---|---|---|---|
| polystyrene | 76.5 | 84.3 | 73.9 |
| PMMA | 81.7 | 96.7 | 68.8 |

mN/m. PMMA grips steel and glass better; polystyrene wins on plastics. The
substrate finally changes the answer.

This still predicts no pounds. It is reversible work, and a real joint
dissipates one to three orders of magnitude more, so it ranks polymers against
each other and says nothing about whether the web holds.

**2. The polymer branch is unreachable from the CLI.** `formulate run
strand.yaml` proposes zero candidates. There are two polymer experts and 57
reference polymers with measured Tg and density, but no explorer that puts a
polymer in the pool: `ReferenceDatabaseExplorer` returns early unless the spec
asks for `MaterialClass.MOLECULE`, and `EvolutionaryExplorer` can only mutate a
polymer that is already there.

`score_strand.py` fills the gap with a twenty-line explorer and runs the real
pipeline behind it. Worth noting that this failure is quieter than the one
above: an empty pool prints "No candidates were evaluated" and does not say
that no explorer serves the requested material class.

**3. There is no strength, and no cure.** `theoretical_strength` is a flaw-free
bound, and in this engine it is `0.1 x E`, where `E` in the glass is a constant
2.86 GPa for every polymer. So it is two-valued - 286 MPa if glassy, ~0.1 MPa
if rubbery - and it is a glassiness switch, not a strength ranking. It cannot
choose between two glassy polymers, and nothing here predicts the load at which
a strand breaks. That is the right call rather than a gap: a specimen fails at
its largest flaw and the flaw belongs to the process.

Separately, the engine has no reaction or cure model at all, which rules out
the entire class of reactive one-part systems - cyanoacrylate above all - that
are what a real web would most likely be made of.

## The mistake it caught

The first draft of `strand.yaml` asked for `glass_transition_temperature` with
`direction: minimize` and `lower: 60 degC`. A minimize with a lower anchor
gives a perfect score to *anything below it*, so polyethylene came top of the
ranking with a Tg of -75 degC, scoring 1.000 on a requirement written to demand
glassiness. Glassiness is an `in_range`, and the printed per-requirement
utilities are what made this visible.

The same draft asked for `theoretical_strength >= 710 MPa`, a 10x margin on a
2 mm strand. Unsatisfiable by construction against a 286 MPa ceiling - and the
failure report said so precisely: `nearest miss passes at lower bound 2.86e+08
(currently 7.1e+08)`.

## Asking for a web that deforms: no answer, and a real reason

`strand_tough.yaml` is the same spec with the toughness axis promoted from a
soft objective to a hard cap - entanglement molar mass at or below 5 kg/mol, so
a chain tangles often enough to carry load through a network instead of
snapping. Nothing passes, and the failure names the pair:

```
Constraints never satisfied together by any candidate in this pool:
  - glass_transition_temperature [hard]: in range [333.15, 453.15] kelvin and
    entanglement_molar_mass [hard]: in range [-inf, 5] kilogram / mole
```

Glassy and tough are mutually exclusive across the whole bundled set. That is
not an artifact of the fitted `M_e` model: against the *measured* entanglement
values for the six polymers that have both, Tg correlates with `ln(M_e)` at
r = +0.75.

| polymer | Tg (K) | measured M_e (g/mol) | glassy at 298 K |
|---|---|---|---|
| cis-1,4-polybutadiene | 171 | 1850 | no |
| polyethylene | 195 | 1150 | no |
| cis-1,4-polyisoprene | 200 | 6200 | no |
| polyisobutylene | 200 | 6700 | no |
| polystyrene | 373 | 18100 | yes |
| PMMA | 378 | 9200 | yes |

There is a mechanism behind it: `M_e` scales as the cube of the packing length,
and a bulky side group both fattens the chain - raising the packing length -
and hinders its rotation, raising Tg. One cause, both effects. Six points is
not a law, but the direction is physical rather than fitted.

So no single homopolymer that is glassy at room temperature is also tough, and
the engine is right to return nothing. What actually resolves it in the real
world is two phases, not one: crystallites acting as crosslinks in a drawn
semicrystalline fibre (nylon, PET, UHMWPE - and spider silk, which is
beta-sheet crystallites in an amorphous matrix), or rubber particles dispersed
in a glassy matrix (HIPS, ABS).

Both are outside this engine entirely. There is no crystallinity model - the
modulus is a switch on Tg alone - and while `MaterialClass.COMPOSITE` exists,
no expert supports it. The registry also has no elongation at break, no yield
stress and no fracture toughness, so `entanglement_molar_mass` is the only
toughness axis available and it is an indirect one.

## Dropping glassiness, which was never a requirement

The deadlock above dissolves once you ask why a glass was demanded at all. It
was not demanded by the problem. It was inherited from this engine's strength
model: `theoretical_strength` is a tenth of Young's modulus, so anything soft
scores as weak by construction, and an elastomer comes out at 0.1 MPa.

That is an artifact. Strength and stiffness are different quantities. Natural
rubber breaks near 25 MPa - its strength comes from strain-induced
crystallisation, not from its modulus - which holds 50 lb on a 3.4 mm strand
with room to spare, while stretching several hundred percent to absorb a catch.
A load-bearing line does not have to be rigid; a bungee is not, and neither is
silk.

`strand_extensible.yaml` drops the glass transition and the strength bound and
keeps the toughness cap. 4 of 57 feasible: **polyethylene**, **cis-1,4-
polyisoprene** (natural rubber), **cis-1,4-polybutadiene**, poly(ethylene
oxide). The two rubbers are what the problem was asking for all along, and
natural rubber dissolved in a light alkane is rubber cement - a fluid that
squirts, flashes off its solvent, and leaves a stretchy strand that grips.

The honest cost of this run is that it has **no strength axis at all**. It
selects on toughness and adhesion and takes on trust that an elastomer carries
the load, because the only strength number here would score it a thousand times
too low. The engine cannot confirm the thing the design now rests on.

Two further gaps this exposes: `solubility_red` has no Hansen sphere for
polyisoprene or polybutadiene, so the carrier spec cannot check that a solvent
dissolves them either, and neither rubber has a tabulated surface energy, so
the adhesion axis goes unavailable again for the top rubber candidates.

## What the answers are worth

Every axis on both winners scores 0.000 at the pessimistic bound and is flagged
`meets its bound only within uncertainty`. Tg carries +/- 34 K, so PMMA's
350 K is 316 K at one sigma - below the 333 K the spec demands. Entanglement
molar mass carries a factor of 1.74. Neither polymer survives its own error
bars, and the engine says so on the same line as the number.

## The four questions the engine cannot answer at all

Whether the thing works is decided by transport and kinetics, and there is no
expert for either. Worked outside the engine, in
`scratchpad/web_physics.py` terms:

**Strength.** Cross-section for 50 lb at a safety factor of 2, and what 3 m of
it weighs:

| material | UTS | area | single strand | 3 m dry | dope at 25% solids |
|---|---|---|---|---|---|
| unvulcanised natural rubber (rubber cement) | ~2 MPa | 297 mm2 | 19.4 mm | 836 g | 3.3 kg |
| SBS block copolymer (contact cement) | 20 MPa | 22 mm2 | 5.3 mm | 63 g | 251 g |
| cast PMMA | 70 MPa | 6.4 mm2 | 2.8 mm | 22 g | 90 g |
| nylon-6,6 drawn fibre | 700 MPa | 0.6 mm2 | 0.9 mm | 2 g | 9 g |

This kills rubber cement outright. Unvulcanised rubber has no permanent
network - the chains slide - so it creeps under sustained load, and its useful
strength is a couple of MPa rather than the 25 MPa a vulcanised rubber reaches
by strain-induced crystallisation. A 19 mm rope needing 3.3 kg of dope per shot
is not a wrist device. SBS is the one that survives, and it survives precisely
because its glassy polystyrene domains act as physical crosslinks - the
two-phase structure the engine pointed at and cannot represent.

**Drying.** No, and not by a small margin. Solvent leaving a filament is
diffusion-limited at `t ~ R^2/D`, with `D ~ 1e-10 m^2/s` in a concentrated
solution:

| filament | evaporation-limited | diffusion-limited | fits 0.6 s flight |
|---|---|---|---|
| 50 um | 0.02 s | 6 s | no |
| 300 um | 0.37 s | 225 s | no |
| 3000 um | 12 s | 22500 s | no |

The surface skins over in milliseconds and the core stays liquid for minutes.
Drying in flight would need filaments around 15 um - which is what a real
textile spinneret makes, and why industrial dry-spinning columns are metres
tall - and 125,000 of them to reach 22 mm2.

**Sticking.** Yes, ironically *because* it does not dry. It arrives wet and
tacky, which is a good adhesive contact and a terrible structural one.

**Clogging.** Yes, in seconds. A stagnant meniscus in still air skins to a
tenth of the orifice radius in 1.4 s at 300 um, 3.3 s at 700 um.

### What the numbers point at instead

Drop the solvent. Heat diffuses about a thousand times faster than solvent
does, so a melt freezes on the timescale a solution cannot dry on:

| filament | freezes | would dry in |
|---|---|---|
| 100 um | 0.02 s | 25 s |
| 300 um | 0.22 s | 225 s |
| 1000 um | 2.5 s | 2500 s |

A 300 um hot-melt filament solidifies in 0.22 s, inside the flight time, and
314 of them give the 22 mm2 SBS needs. No solvent to evaporate, no skinning in
the nozzle, and it bonds on contact because it lands molten - which is how a
glue gun works. The nozzle now freezes rather than skins, which is a heater
problem rather than a chemistry one.

None of this is in the engine's reach: no evaporation kinetics, no heat
transfer, no viscosity for a polymer solution, no creep.

## Hardware, which the engine says nothing about

No cost property exists in the registry; `synthetic_accessibility` stands in for
price in `carrier.yaml` and is a poor proxy - it scores how hard a molecule is
to make from scratch, not what a drum of commodity solvent costs. Everything
below is ordinary fluid mechanics done outside the engine.

**A single fat strand is hydraulically impossible.** Dry spinning shrinks the
filament as solvent leaves: at 30% solids a 3 mm dry strand needs a 5.8 mm
orifice, which passes 132 mL/s at 5 m/s and empties a wrist cartridge in a
quarter of a second. Shrink the orifice to 1 mm and a 10 Pa.s dope needs
160 bar to push through it.

**A bundle is the only way**, which is also what a spinneret and a spider's
dragline both are. Load is carried by total cross-section, but pressure drop
goes as `1/R^2` per hole, so many small holes beat one big one:

| filaments | dry filament | solids | viscosity | pressure | flow | holds at 100 MPa |
|---|---|---|---|---|---|---|
| 1 | 3000 um | 30% | 10 Pa.s | 2 bar | 132 mL/s | 159 lb |
| 40 | 300 um | 30% | 10 Pa.s | 191 bar | 53 mL/s | 64 lb |
| **40** | **300 um** | **20%** | **1 Pa.s** | **13 bar** | **80 mL/s** | **64 lb** |
| 200 | 150 um | 20% | 1 Pa.s | 50 bar | 100 mL/s | 79 lb |

The 13 bar design point is the cheap one, and it is cheap because of the
viscosity, not the hardware: 20% solids in acetone is a thin dope, and 13 bar
is a bicycle pump into a PET bottle.

- **Pressure**: a 2 L PET bottle pumped to 13 bar with a bike pump, or a 12 g
  CO2 cartridge and a piercing valve (~36 shots at 5 bar, ~18 at 10 bar).
  PET soda bottles are rated well above this and fail by splitting, not
  shattering.
- **Reservoir**: a 60 mL luer-lock syringe barrel. At 80 mL/s a 0.6 s shot uses
  ~48 mL, so this is **one shot per barrel** - the real cost of the bundle is
  that a web shooter carries about four shots, not forty.
- **Spinneret**: 40 holes of 0.7 mm. A drilled aluminium disc, or the cheapest
  version, 40 blunt 22-gauge dispensing needles in a manifold.
- **Valve**: a trigger-operated ball valve. The nozzle must be sealed or
  solvent-flooded between shots or the dope skins over in it, which is the
  failure mode that will actually end the project.

Total, well under $50, and none of it is the hard part.

## Verdict

The engine handled the parts it covers and refused the rest legibly rather than
guessing - the infeasibility diagnosis found the solvency/adhesion gap on its
own, and the uncertainty reporting made clear that neither surviving polymer
clears its bounds robustly.

The adhesion gap turned out not to be a data gap at all: the question had been
put to the wrong phase, and asking the polymer instead needed only an existing
table read from the other side. The unreachable polymer class remains a
plumbing gap worth closing. What no amount of data fixes is that reversible
work is not peel strength, so the engine still cannot say whether the web
holds - only which polymer holds better.
